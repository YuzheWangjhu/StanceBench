#!/usr/bin/env python3
"""
run_turnover_gemini.py

Gemini (Developer API) model swap for the existing local judge pipeline.

Design goal:
- Preserve the exact same *evaluation pipeline* used by run_turnover_qwen_QA.py / run_turnover_KIMI.py / run_turnover_gpt_QA.py:
  - filter_roles.py -> build_eval_inputs.py -> manifest JSONL
  - Per-speaker scoring loop with Balanced Position (two variants swapping P/N definition order)
  - Same forced-choice JSON output schema: {choice, probability, evidence}
  - Same deterministic mapping: (choice, probability) -> {-2,-1,0,1,2}
  - Same CSV outputs: evidence_{a,b}, avg_score_{a,b}, flip_rate_{a,b}, fail_note_{a,b}

Main difference:
- The judge call is performed via the Gemini Developer API (google-genai Python SDK),
  including audio inputs and structured outputs with JSON Schema.

Requirements:
- Install SDK: pip install google-genai
- Provide API key via env GOOGLE_API_KEY or GEMINI_API_KEY, or pass --gemini-api-key.

Notes:
- This script uses inline audio bytes by default (fewer API calls). If an audio file is
  larger than --gemini-inline-max-bytes, it falls back to the Files API upload path.
  You can force Files API via --gemini-use-files-api.

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# Gemini SDK (google-genai). Imported lazily in main() to give a clean error message if missing.


# ----------------------------
# Repo-relative helper scripts
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FILTER_SCRIPT = REPO_ROOT / "scripts" / "filter_roles.py"
BUILD_INPUT_SCRIPT = REPO_ROOT / "scripts" / "build_eval_inputs.py"

# Environment-file parser (for .env / .env/api_keys.env style files)
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_matching_quotes(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
        return v[1:-1]
    return v


def _load_env_file_if_present(path: Path) -> None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return

    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_RE.match(key):
            continue

        value = _strip_matching_quotes(value.strip())
        # Keep process-level env as highest precedence.
        os.environ.setdefault(key, value)


def _bootstrap_env_from_repo() -> None:
    # Optional override for custom env file locations.
    env_override = os.environ.get("SPEAR_ENV_FILE", "").strip()
    if env_override:
        _load_env_file_if_present(Path(env_override))

    repo_root = SCRIPT_DIR.parent
    candidates = [
        repo_root / ".env",
        repo_root / ".env" / "api_keys.env",
        SCRIPT_DIR / ".env",
        SCRIPT_DIR / ".env" / "api_keys.env",
        Path.cwd() / ".env",
        Path.cwd() / ".env" / "api_keys.env",
    ]
    for p in candidates:
        _load_env_file_if_present(p)


# ----------------------------
# JSON extraction helpers
# ----------------------------
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from a string (best-effort robustness)."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None

    # First try strict parse.
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Fallback: find a {...} block.
    m = _JSON_BLOCK_RE.search(s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _coerce_evidence(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: List[str] = []
        for x in v:
            if isinstance(x, str):
                t = x.strip()
                if t:
                    out.append(t)
        return out
    if isinstance(v, str):
        t = v.strip()
        return [t] if t else []
    return []


def _as_dict_best_effort(v: Any) -> Optional[Dict[str, Any]]:
    """Best-effort conversion for SDK response objects to a plain dict."""
    if isinstance(v, dict):
        return v

    for meth in ("model_dump", "to_json_dict", "dict"):
        fn = getattr(v, meth, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    return None


def _extract_texts_from_tree(root: Any, limit: int = 64) -> List[str]:
    """Collect likely text fields from a nested dict/list structure."""
    out: List[str] = []
    seen: set[str] = set()

    stack: List[Any] = [root]
    while stack and len(out) < limit:
        cur = stack.pop()
        if isinstance(cur, str):
            t = cur.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
            continue

        if isinstance(cur, dict):
            for k, v in cur.items():
                kl = str(k).strip().lower()
                if isinstance(v, str) and kl in {
                    "text",
                    "output_text",
                    "content",
                    "response",
                    "message",
                    "raw_text",
                }:
                    t = v.strip()
                    if t and t not in seen:
                        seen.add(t)
                        out.append(t)
                if isinstance(v, (dict, list)):
                    stack.append(v)
            continue

        if isinstance(cur, list):
            for it in cur:
                if isinstance(it, (dict, list, str)):
                    stack.append(it)

    return out


def _extract_json_object_from_any(v: Any) -> Optional[Dict[str, Any]]:
    """Try multiple object/text paths to recover a JSON object."""
    if v is None:
        return None

    if isinstance(v, dict):
        # Fast path for already-materialized structured outputs.
        if {"choice", "probability", "evidence"}.issubset(set(v.keys())):
            return v

        # Some SDK payloads may nest the useful object.
        for key in ("parsed", "json", "response", "content", "output"):
            if key in v:
                nested = _extract_json_object_from_any(v.get(key))
                if nested:
                    return nested

        # Also scan textual fields.
        for t in _extract_texts_from_tree(v):
            obj = _extract_json_object(t)
            if obj:
                return obj
        return None

    if isinstance(v, str):
        return _extract_json_object(v)

    # Handle simple typed objects that expose fields directly.
    has_choice = hasattr(v, "choice")
    has_prob = hasattr(v, "probability")
    has_ev = hasattr(v, "evidence")
    if has_choice and has_prob and has_ev:
        try:
            return {
                "choice": getattr(v, "choice"),
                "probability": getattr(v, "probability"),
                "evidence": getattr(v, "evidence"),
            }
        except Exception:
            pass

    d = _as_dict_best_effort(v)
    if d is not None:
        return _extract_json_object_from_any(d)

    return None


def _collect_gemini_response_texts(resp: Any, limit: int = 64) -> List[str]:
    """Collect text candidates from Gemini response object beyond resp.text."""
    out: List[str] = []
    seen: set[str] = set()

    def _add(t: Any) -> None:
        if not isinstance(t, str):
            return
        s = t.strip()
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    _add(getattr(resp, "text", None))

    cands = getattr(resp, "candidates", None)
    if isinstance(cands, list):
        for cand in cands:
            _add(getattr(cand, "text", None))
            _add(getattr(cand, "output_text", None))

            content = getattr(cand, "content", None)
            if content is not None:
                _add(getattr(content, "text", None))
                parts = getattr(content, "parts", None)
                if isinstance(parts, list):
                    for part in parts:
                        _add(getattr(part, "text", None))
                        if isinstance(part, dict):
                            _add(part.get("text", None))

            if len(out) >= limit:
                return out[:limit]

    # Final fallback: scan dict-converted payload for text fields.
    d = _as_dict_best_effort(resp)
    if d is not None:
        for t in _extract_texts_from_tree(d, limit=limit):
            _add(t)
            if len(out) >= limit:
                break

    return out[:limit]


def _gemini_response_debug_summary(resp: Any) -> str:
    """Small diagnostic string when no JSON/text payload is recoverable."""
    bits: List[str] = []
    try:
        cands = getattr(resp, "candidates", None)
        if isinstance(cands, list) and cands:
            fr = getattr(cands[0], "finish_reason", None)
            if fr is not None:
                bits.append(f"finish_reason={fr}")
    except Exception:
        pass

    try:
        pf = getattr(resp, "prompt_feedback", None)
        if pf is not None:
            s = str(pf).strip()
            if s:
                if len(s) > 300:
                    s = s[:300] + "..."
                bits.append(f"prompt_feedback={s}")
    except Exception:
        pass

    return "; ".join(bits).strip()


# ----------------------------
# Prompt builders (copied from the other scripts; audio paths are converted later)
# ----------------------------


def build_judge_conv_prefix_single(
    question_text: str,
    def_p: str,
    def_n: str,
    audio_path: str,
    variant: int = 0,
) -> List[Dict[str, Any]]:
    """Single-speaker prompt: one audio (TARGET)."""
    system_text = (
        "You are a careful dialogue analyst. "
        "You will hear ONE audio segment spoken by the TARGET speaker. "
        "Your task is to judge the TARGET speaker's stance/tone/style relative to the question.\n\n"
        "IMPORTANT OUTPUT FORMAT:\n"
        "- Output ONLY one JSON object with keys 'choice', 'probability', and 'evidence'.\n"
        "- 'choice' must be either \"P\" or \"N\". Always choose one.\n"
        "- 'probability' must be a number from 0.0 to 1.0 representing probability that your choice is correct (0.0 to 1.0).\n"
        "- If unsure, choose the closer option but set probability low.\n"
        "- 'evidence' must be a JSON array of 2 to 4 short strings describing observable cues "
        "(tone, prosody, wording, engagement).\n"
        "- No extra text.\n"
        "Example JSONs (format only):\n"
        "Example 1: {\"choice\":\"P\",\"probability\":0.60,\"evidence\":[\"Cue about tone/prosody\",\"Cue about wording\"]}\n"
        "Example 2: {\"choice\":\"N\",\"probability\":0.60,\"evidence\":[\"Cue about tone/prosody\",\"Cue about wording\"]}"
    )

    p_block = "Definition P:\n" + str(def_p).strip() + "\n"
    n_block = "Definition N:\n" + str(def_n).strip() + "\n"
    defs_text = (p_block + "\n" + n_block) if (variant % 2 == 0) else (n_block + "\n" + p_block)

    user_text = (
        "TARGET speaker audio follows.\n"
        f"Question: {question_text}\n\n"
        "Definitions:\n"
        f"{defs_text}\n"
        "Return ONLY JSON with keys: choice, probability, evidence."
    )

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "audio", "path": audio_path},
            ],
        },
    ]


def build_judge_conv_prefix_interaction(
    question_text: str,
    def_p: str,
    def_n: str,
    context_audio_path: str,
    target_audio_path: str,
    variant: int = 0,
) -> List[Dict[str, Any]]:
    """Interaction prompt: audio1=context, audio2=TARGET."""
    system_text = (
        "You are a careful dialogue analyst. "
        "You will hear TWO audio segments. "
        "The first audio is CONTEXT from the other speaker. "
        "The second audio is the TARGET speaker's response and is the only segment you should evaluate.\n\n"
        "IMPORTANT OUTPUT FORMAT:\n"
        "- Output ONLY one JSON object with keys 'choice', 'probability', and 'evidence'.\n"
        "- 'choice' must be either \"P\" or \"N\". Always choose one.\n"
        "- 'probability' must be a number from 0.0 to 1.0 representing probability that your choice is correct (0.0 to 1.0).\n"
        "- If unsure, choose the closer option but set probability low.\n"
        "- 'evidence' must be a JSON array of 2 to 4 short strings describing observable cues "
        "in the TARGET speaker's response (tone, prosody, wording, engagement).\n"
        "- Use the context only to interpret the TARGET response; do not judge the context speaker.\n"
        "- No extra text.\n"
        "Example JSONs (format only):\n"
        "Example 1: {\"choice\":\"P\",\"probability\":0.60,\"evidence\":[\"Cue about tone/prosody\",\"Cue about wording\"]}\n"
        "Example 2: {\"choice\":\"N\",\"probability\":0.60,\"evidence\":[\"Cue about tone/prosody\",\"Cue about wording\"]}"
    )

    p_block = "Definition P:\n" + str(def_p).strip() + "\n"
    n_block = "Definition N:\n" + str(def_n).strip() + "\n"
    defs_text = (p_block + "\n" + n_block) if (variant % 2 == 0) else (n_block + "\n" + p_block)

    user_text = (
        "Audio 1 is context (other speaker). Audio 2 is the TARGET speaker response.\n"
        f"Question: {question_text}\n\n"
        "Definitions:\n"
        f"{defs_text}\n"
        "Always choose either P or N. If unsure, choose the closer one but set probability low.\n"
        "Return ONLY JSON with keys: choice, probability, evidence."
    )

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "audio", "path": context_audio_path},
                {"type": "audio", "path": target_audio_path},
            ],
        },
    ]


def build_judge_conv_prefix_interaction_target_first(
    question_text: str,
    def_p: str,
    def_n: str,
    target_audio_path: str,
    context_audio_path: str,
    variant: int = 0,
) -> List[Dict[str, Any]]:
    """Interaction prompt: audio1=TARGET, audio2=context."""
    system_text = (
        "You are a careful dialogue analyst. "
        "You will hear TWO audio segments. "
        "The first audio is the TARGET segment and is the only segment you should evaluate. "
        "The second audio is CONTEXT from the other speaker.\n\n"
        "IMPORTANT OUTPUT FORMAT:\n"
        "- Output ONLY one JSON object with keys 'choice', 'probability', and 'evidence'.\n"
        "- 'choice' must be either \"P\" or \"N\". Always choose one.\n"
        "- 'probability' must be a number from 0.0 to 1.0 representing probability that your choice is correct (0.0 to 1.0).\n"
        "- If unsure, choose the closer option but set probability low.\n"
        "- 'evidence' must be a JSON array of 2 to 4 short strings describing observable cues "
        "in the TARGET segment (tone, prosody, wording, engagement).\n"
        "- Use the context only to interpret the TARGET segment; do not judge the context speaker.\n"
        "- No extra text.\n"
        "Example JSONs (format only):\n"
        "Example 1: {\"choice\":\"P\",\"probability\":0.60,\"evidence\":[\"Cue about tone/prosody\",\"Cue about wording\"]}\n"
        "Example 2: {\"choice\":\"N\",\"probability\":0.60,\"evidence\":[\"Cue about tone/prosody\",\"Cue about wording\"]}"
    )

    p_block = "Definition P:\n" + str(def_p).strip() + "\n"
    n_block = "Definition N:\n" + str(def_n).strip() + "\n"
    defs_text = (p_block + "\n" + n_block) if (variant % 2 == 0) else (n_block + "\n" + p_block)

    user_text = (
        "Audio 1 is the TARGET segment. Audio 2 is context (other speaker).\n"
        f"Question: {question_text}\n\n"
        "Definitions:\n"
        f"{defs_text}\n"
        "Always choose either P or N. If unsure, choose the closer one but set probability low.\n"
        "Return ONLY JSON with keys: choice, probability, evidence."
    )

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "audio", "path": target_audio_path},
                {"type": "audio", "path": context_audio_path},
            ],
        },
    ]


# ----------------------------
# Deterministic mapping (same as other scripts)
# ----------------------------

TAU0_DEFAULT = 0.45
TAU2_DEFAULT = 0.75


def map_choice_probability_to_score(
    choice: str,
    probability: float,
    tau0: float = TAU0_DEFAULT,
    tau2: float = TAU2_DEFAULT,
) -> int:
    c = str(choice).strip().upper()
    if c not in {"P", "N"}:
        raise ValueError(f"Invalid choice: {choice!r}")
    try:
        prob = float(probability)
    except Exception as e:
        raise ValueError(f"Invalid probability: {probability!r}") from e

    prob = max(0.0, min(1.0, prob))
    s = 1 if c == "P" else -1
    if prob < float(tau0):
        return 0
    if prob >= float(tau2):
        return 2 * s
    return 1 * s


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# ----------------------------
# Audio helpers (same as other scripts)
# ----------------------------


def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    m = float(np.max(np.abs(x)))
    if m > 0:
        x = x / m
    return x.astype(np.float32)


def _load_prepared_audio(audio_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(audio_path), always_2d=True)
    if data.ndim != 2:
        raise RuntimeError(f"Expected 2-D audio array for {audio_path}, got shape {data.shape}.")
    mono = np.mean(data, axis=1).astype(np.float32)
    if int(sr) != int(target_sr):
        mono = librosa.resample(mono, orig_sr=int(sr), target_sr=int(target_sr))
        sr = int(target_sr)
    mono = normalize_audio(mono)
    return mono, int(sr)


def _guess_audio_mime_type(path: str) -> str:
    p = str(path).lower()
    if p.endswith(".wav"):
        return "audio/wav"
    if p.endswith(".mp3"):
        return "audio/mp3"
    if p.endswith(".flac"):
        return "audio/flac"
    if p.endswith(".ogg"):
        return "audio/ogg"
    if p.endswith(".aac"):
        return "audio/aac"
    if p.endswith(".aiff") or p.endswith(".aif"):
        return "audio/aiff"
    # Default: WAV (our prepared audio is wav)
    return "audio/wav"


# ----------------------------
# Gemini client wrapper + per-item audio cache
# ----------------------------


class GeminiJudgeClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
    ):
        # Lazy import for clearer error messages in environments without google-genai.
        try:
            from google import genai  # type: ignore
        except Exception as e:
            raise SystemExit(
                "Missing dependency 'google-genai'. Install with: pip install google-genai"
            ) from e

        self.genai = genai
        self.api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)

        # Under the hood, the SDK reads GOOGLE_API_KEY too, but we pass explicitly for clarity.
        self.client = genai.Client(api_key=self.api_key)

        # Import types lazily (SDK-provided)
        from google.genai import types  # type: ignore

        self.types = types

    def default_response_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "choice": {"type": "string", "enum": ["P", "N"]},
                "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 4,
                },
            },
            "required": ["choice", "probability", "evidence"],
        }

    def default_response_schema(self) -> Any:
        return self.types.Schema(
            type=self.types.Type.OBJECT,
            properties={
                "choice": self.types.Schema(type=self.types.Type.STRING, enum=["P", "N"]),
                "probability": self.types.Schema(type=self.types.Type.NUMBER),
                "evidence": self.types.Schema(
                    type=self.types.Type.ARRAY,
                    items=self.types.Schema(type=self.types.Type.STRING),
                ),
            },
            required=["choice", "probability", "evidence"],
        )

    def generate_json(
        self,
        *,
        system_instruction: str,
        contents: List[Any],
        response_json_schema: Dict[str, Any],
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        seed: Optional[int] = None,
    ) -> Tuple[str, Any]:
        """Return (response_text, raw_response_obj)."""

        model_name = str(self.model).strip().lower()
        thinking_cfg: Any = self.types.ThinkingConfig(thinking_budget=128)
        schema_key = "response_json_schema"
        schema_value: Any = response_json_schema

        if "gemini-3-flash-preview" in model_name:
            thinking_cfg = self.types.ThinkingConfig(thinking_level="minimal")
            schema_key = "response_schema"
            schema_value = self.default_response_schema()
        elif "gemini-2.5-flash" in model_name:
            thinking_cfg = self.types.ThinkingConfig(thinking_budget=128)
            schema_key = "response_schema"
            schema_value = self.default_response_schema()
        elif "gemini-2.5-pro" in model_name:
            thinking_cfg = self.types.ThinkingConfig(thinking_budget=128)

        cfg_kwargs: Dict[str, Any] = {
            "system_instruction": str(system_instruction).strip() if system_instruction else None,
            "response_mime_type": "application/json",
            "thinking_config": thinking_cfg,
            "max_output_tokens": int(max_output_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": int(seed) if seed is not None else None,
        }
        cfg_kwargs[schema_key] = schema_value
        cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if v is not None}
        cfg = self.types.GenerateContentConfig(**cfg_kwargs)

        try:
            # Note: google-genai handles auth headers internally.
            resp = self.client.models.generate_content(
                model=str(self.model),
                contents=contents,
                config=cfg,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e

        # 1) Prefer structured payload when available.
        obj = _extract_json_object_from_any(getattr(resp, "parsed", None))
        if obj:
            return json.dumps(obj, ensure_ascii=False), resp

        # 2) Fall back to response.text.
        txt = str(getattr(resp, "text", "") or "").strip()
        if txt:
            obj = _extract_json_object(txt)
            if obj:
                return json.dumps(obj, ensure_ascii=False), resp

        # 3) Scan candidate parts / dict dump for additional text.
        alt_texts = _collect_gemini_response_texts(resp)
        for t in alt_texts:
            obj = _extract_json_object(t)
            if obj:
                return json.dumps(obj, ensure_ascii=False), resp

        # 4) Return best available textual payload for fail-note diagnostics.
        if txt:
            return txt, resp
        if alt_texts:
            return alt_texts[0], resp
        dbg = _gemini_response_debug_summary(resp)
        return dbg or "", resp


class GeminiAudioCache:
    """Per-item cache for audio parts (inline bytes or Files API uploads)."""

    def __init__(
        self,
        *,
        client: Any,
        types_mod: Any,
        inline_max_bytes: int,
        force_files_api: bool,
        delete_uploaded_files: bool,
    ):
        self._client = client
        self._types = types_mod
        self.inline_max_bytes = int(inline_max_bytes)
        self.force_files_api = bool(force_files_api)
        self.delete_uploaded_files = bool(delete_uploaded_files)

        self._cache: Dict[str, Any] = {}
        self._uploaded_names: List[str] = []

    def get(self, audio_path: str) -> Any:
        p = str(Path(audio_path).resolve())
        if p in self._cache:
            return self._cache[p]

        if not Path(p).exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        if self.force_files_api:
            fobj = self._client.files.upload(file=p)
            self._cache[p] = fobj
            if self.delete_uploaded_files and getattr(fobj, "name", None):
                self._uploaded_names.append(str(fobj.name))
            return fobj

        size_b = int(os.path.getsize(p))
        if size_b <= self.inline_max_bytes:
            with open(p, "rb") as f:
                data = f.read()
            part = self._types.Part.from_bytes(data=data, mime_type=_guess_audio_mime_type(p))
            self._cache[p] = part
            return part

        # Fallback to Files API for large audio.
        fobj = self._client.files.upload(file=p)
        self._cache[p] = fobj
        if self.delete_uploaded_files and getattr(fobj, "name", None):
            self._uploaded_names.append(str(fobj.name))
        return fobj

    def cleanup(self) -> None:
        if not self.delete_uploaded_files:
            return
        for name in self._uploaded_names:
            try:
                self._client.files.delete(name=name)
            except Exception:
                pass


def _conv_prefix_to_gemini_request(
    conv_prefix: List[Dict[str, Any]],
    audio_cache: GeminiAudioCache,
) -> Tuple[str, List[Any]]:
    """Convert the Qwen-style conv_prefix (with audio paths) to Gemini API (system_instruction, contents)."""
    system_instruction = ""
    contents: List[Any] = []

    for msg in conv_prefix:
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content", "")

        # System message: pull text parts into system instruction.
        if role == "system":
            if isinstance(content, str):
                system_instruction = (system_instruction + "\n" + content).strip()
            elif isinstance(content, list):
                texts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and str(part.get("type", "")).lower() == "text":
                        texts.append(str(part.get("text", "")))
                system_instruction = (system_instruction + "\n" + "\n".join(texts)).strip()
            else:
                system_instruction = (system_instruction + "\n" + str(content)).strip()
            continue

        # User message: append text and audio parts to contents.
        if role == "user":
            if isinstance(content, str):
                t = content.strip()
                if t:
                    contents.append(t)
                continue

            if not isinstance(content, list):
                t = str(content).strip()
                if t:
                    contents.append(t)
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = str(part.get("type", "")).strip().lower()
                if ptype == "text":
                    t = str(part.get("text", "")).strip()
                    if t:
                        contents.append(t)
                elif ptype == "audio":
                    apath = str(part.get("path", "")).strip()
                    if apath:
                        contents.append(audio_cache.get(apath))
                else:
                    continue

    system_instruction = system_instruction.strip()
    return system_instruction, contents


def judge_item_choice_probability(
    client: GeminiJudgeClient,
    conv_prefix: List[Dict[str, Any]],
    audio_cache: GeminiAudioCache,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: Optional[int] = None,
) -> Tuple[Optional[str], Optional[float], List[str], str]:
    """Gemini judge; mirrors the other scripts' function signature/output."""
    def _parse_obj(raw_text_in: str, raw_resp_in: Any) -> Optional[Dict[str, Any]]:
        obj_local = _extract_json_object(raw_text_in)
        if not obj_local:
            obj_local = _extract_json_object_from_any(getattr(raw_resp_in, "parsed", None))
        if not obj_local:
            obj_local = _extract_json_object_from_any(raw_resp_in)
        return obj_local

    if do_sample:
        temp = float(temperature)
        tp = float(top_p)
    else:
        temp = 0.0
        tp = 1.0

    system_instruction, contents = _conv_prefix_to_gemini_request(conv_prefix, audio_cache)
    raw_text, _raw_resp = client.generate_json(
        system_instruction=system_instruction,
        contents=contents,
        response_json_schema=client.default_response_json_schema(),
        max_output_tokens=int(max_new_tokens),
        temperature=temp,
        top_p=tp,
        seed=seed,
    )
    raw_text = str(raw_text).strip()
    obj = _parse_obj(raw_text, _raw_resp)
    if not obj:
        return None, None, [], raw_text

    choice_val = obj.get("choice", None)
    if not isinstance(choice_val, str):
        choice = None
    else:
        c = choice_val.strip().upper()
        if c in {"P", "N"}:
            choice = c
        elif c.startswith("P"):
            choice = "P"
        elif c.startswith("N"):
            choice = "N"
        else:
            choice = None

    prob_val = obj.get("probability", None)
    probability: Optional[float]
    if prob_val is None:
        probability = None
    else:
        try:
            probability = float(prob_val)
        except Exception:
            probability = None
    if probability is not None:
        probability = max(0.0, min(1.0, probability))

    evidence = _coerce_evidence(obj.get("evidence", None))
    if len(evidence) > 4:
        evidence = evidence[:4]

    if choice is None or probability is None:
        return None, None, evidence, raw_text

    return choice, probability, evidence, raw_text


# ----------------------------
# Question config + manifest parsing (identical behavior)
# ----------------------------


def load_question_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit("question-config must be a JSON object with key 'outside_judge'.")

    sec = cfg.get("outside_judge", None)
    if not isinstance(sec, list) or not sec:
        raise SystemExit("question-config must contain a non-empty 'outside_judge' list.")

    if len(sec) != 1:
        raise SystemExit(
            "This pipeline runs ONE question at a time. "
            "Please provide a single-question config JSON (outside_judge list length must be 1)."
        )

    q = sec[0]
    if not isinstance(q, dict):
        raise SystemExit("question-config['outside_judge'][0] must be an object.")

    for key in (
        "question",
        "related_categories",
        "positive_followups",
        "negative_followups",
        "positive_defination",
        "negative_defination",
    ):
        if key not in q:
            raise SystemExit(f"question-config['outside_judge'][0] missing key '{key}'.")

    if not isinstance(q["related_categories"], list) or len(q["related_categories"]) < 1:
        raise SystemExit("related_categories must be a non-empty list.")

    if not q["positive_followups"] or not q["negative_followups"]:
        raise SystemExit("positive_followups and negative_followups must be non-empty lists.")

    if not isinstance(q["positive_defination"], str) or not q["positive_defination"].strip():
        raise SystemExit("positive_defination must be a non-empty string.")
    if not isinstance(q["negative_defination"], str) or not q["negative_defination"].strip():
        raise SystemExit("negative_defination must be a non-empty string.")

    return q


def build_category_to_question_map(single_question: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cat2q: Dict[str, Dict[str, Any]] = {}
    for cat in single_question["related_categories"]:
        if cat in cat2q:
            raise SystemExit(f"Duplicate category '{cat}' inside related_categories.")
        cat2q[cat] = single_question
    return cat2q


def _normalize_target_spk_label(v: Any) -> Optional[str]:
    s = str(v).strip()
    low = s.lower()
    if low in {"a", "spka", "speakera", "speaker_a"}:
        return "A"
    if low in {"b", "spkb", "speakerb", "speaker_b"}:
        return "B"
    return None


def load_eval_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not manifest_path.exists():
        return items

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise SystemExit(f"Invalid JSON in manifest {manifest_path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"Manifest entry must be a JSON object at {manifest_path}:{line_no}.")

            item_type = str(obj.get("type", "")).strip().lower()
            if item_type not in {"single", "interaction"}:
                raise SystemExit(
                    f"Unsupported manifest type '{obj.get('type')}' at {manifest_path}:{line_no}."
                )
            obj["type"] = item_type

            tgt = _normalize_target_spk_label(obj.get("target_spk", ""))
            if tgt is None:
                raise SystemExit(
                    f"Invalid target_spk at {manifest_path}:{line_no}: {obj.get('target_spk')!r}"
                )
            obj["target_spk"] = tgt

            if item_type == "interaction":
                try:
                    pos = int(obj.get("target_position", 2))
                except Exception:
                    pos = 2
                if pos not in {1, 2}:
                    pos = 2
                obj["target_position"] = int(pos)

            for key in ("audio", "context_audio", "target_audio"):
                if key in obj and str(obj[key]).strip():
                    p = Path(str(obj[key]))
                    if not p.is_absolute():
                        p = (manifest_path.parent / p).resolve()
                    obj[key] = str(p)

            items.append(obj)

    return items


# ----------------------------
# CSV utilities (same behavior)
# ----------------------------


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp.csv", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if s == "":
        return True
    if s.lower() in {"nan", "none", "null"}:
        return True
    return False


def _str2bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y"}


def _is_gemini_internal_500_runtime_error(err: BaseException) -> bool:
    s = str(err).lower()
    return (
        isinstance(err, RuntimeError)
        and "gemini api error" in s
        and "500" in s
        and "internal" in s
    )


# ----------------------------
# Core per-conversation scoring (same logic; judge call differs)
# ----------------------------


def score_conversation_speaker(
    role: str,
    category: str,
    roles_of_interest: List[str],
    questions_by_category: Dict[str, Dict[str, Any]],
    manifest_items: List[Dict[str, Any]],
    target_spk_label: str,
    client: GeminiJudgeClient,
    target_sr: int,
    tmpdir: Path,
    judge_max_new_tokens: int,
    judge_do_sample: bool,
    judge_temperature: float,
    judge_top_p: float,
    judge_seed: Optional[int],
    gemini_inline_max_bytes: int,
    gemini_use_files_api: bool,
    gemini_delete_uploaded_files: bool,
) -> Tuple[str, float, float, str]:
    if role not in roles_of_interest:
        return "", float("nan"), float("nan"), ""

    if category not in questions_by_category:
        raise SystemExit(f"Category '{category}' for role '{role}' has no question config.")

    qcfg = questions_by_category[category]
    q_text = str(qcfg["question"])
    def_p = str(qcfg["positive_defination"])
    def_n = str(qcfg["negative_defination"])

    relevant_items: List[Tuple[int, Dict[str, Any]]] = []
    for item_idx, item in enumerate(manifest_items):
        item_target = _normalize_target_spk_label(item.get("target_spk", ""))
        if item_target != target_spk_label:
            continue
        item_type = str(item.get("type", "")).strip().lower()
        if item_type not in {"single", "interaction"}:
            continue
        relevant_items.append((item_idx, item))

    if not relevant_items:
        return "", float("nan"), float("nan"), ""

    score_sum = 0.0
    flip_count = 0
    n_item = 0

    evidence_items: List[Dict[str, Any]] = []
    fail_notes: List[Dict[str, Any]] = []

    for item_idx, item in relevant_items:
        item_type = str(item.get("type", "")).strip().lower()
        item_label = item.get("target_ipu_id") or item.get("audio") or f"item_{item_idx:05d}"

        try:
            sr = int(target_sr)
            base = f"manifest_{item_idx:05d}"

            if item_type == "interaction":
                ctx_path = Path(str(item.get("context_audio", "")))
                tgt_path = Path(str(item.get("target_audio", "")))
                try:
                    target_position = int(item.get("target_position", 2))
                except Exception:
                    target_position = 2
                if target_position not in {1, 2}:
                    target_position = 2
                if not ctx_path.exists() or not tgt_path.exists():
                    raise RuntimeError(
                        f"Missing interaction audio(s): context={ctx_path.exists()} target={tgt_path.exists()}"
                    )

                ctx_audio, sr1 = _load_prepared_audio(ctx_path, target_sr)
                tgt_audio, sr2 = _load_prepared_audio(tgt_path, target_sr)
                if sr1 != sr2:
                    raise RuntimeError(
                        f"Prepared interaction audio sample rates differ ({sr1} vs {sr2}) for item {item_idx}."
                    )

                ctx_tmp = tmpdir / f"{base}_ctx.wav"
                tgt_tmp = tmpdir / f"{base}_tgt.wav"
                sf.write(str(ctx_tmp), ctx_audio, sr)
                sf.write(str(tgt_tmp), tgt_audio, sr)

                def _make_conv(variant: int) -> List[Dict[str, Any]]:
                    if int(target_position) == 1:
                        return build_judge_conv_prefix_interaction_target_first(
                            question_text=q_text,
                            def_p=def_p,
                            def_n=def_n,
                            target_audio_path=str(tgt_tmp),
                            context_audio_path=str(ctx_tmp),
                            variant=int(variant),
                        )
                    return build_judge_conv_prefix_interaction(
                        question_text=q_text,
                        def_p=def_p,
                        def_n=def_n,
                        context_audio_path=str(ctx_tmp),
                        target_audio_path=str(tgt_tmp),
                        variant=int(variant),
                    )

            elif item_type == "single":
                tgt_path = Path(str(item.get("audio", "")))
                if not tgt_path.exists():
                    raise RuntimeError(f"Missing single audio: {tgt_path}")

                tgt_audio, _sr2 = _load_prepared_audio(tgt_path, target_sr)
                tgt_tmp = tmpdir / f"{base}_tgt.wav"
                sf.write(str(tgt_tmp), tgt_audio, sr)

                def _make_conv(variant: int) -> List[Dict[str, Any]]:
                    return build_judge_conv_prefix_single(
                        question_text=q_text,
                        def_p=def_p,
                        def_n=def_n,
                        audio_path=str(tgt_tmp),
                        variant=int(variant),
                    )

            else:
                continue

        except Exception as e:
            fail_notes.append(
                {
                    "item_index": int(item_idx),
                    "item_type": item_type,
                    "item_label": str(item_label),
                    "target_spk_label": target_spk_label,
                    "variant": None,
                    "error": str(e),
                }
            )
            continue

        # Per-item audio cache ensures we do not re-read or re-upload audio between v0 and v1.
        audio_cache = GeminiAudioCache(
            client=client.client,
            types_mod=client.types,
            inline_max_bytes=int(gemini_inline_max_bytes),
            force_files_api=bool(gemini_use_files_api),
            delete_uploaded_files=bool(gemini_delete_uploaded_files),
        )

        choice_v0: Optional[str] = None
        prob_v0: Optional[float] = None
        ev_v0: List[str] = []
        raw0 = ""
        choice_v1: Optional[str] = None
        prob_v1: Optional[float] = None
        ev_v1: List[str] = []
        raw1 = ""
        item_runtime_err: Optional[RuntimeError] = None
        try:
            for attempt in range(2):  # one retry for Gemini 500 INTERNAL
                try:
                    conv0 = _make_conv(0)
                    choice_v0, prob_v0, ev_v0, raw0 = judge_item_choice_probability(
                        client=client,
                        conv_prefix=conv0,
                        audio_cache=audio_cache,
                        max_new_tokens=judge_max_new_tokens,
                        do_sample=judge_do_sample,
                        temperature=judge_temperature,
                        top_p=judge_top_p,
                        seed=judge_seed,
                    )

                    conv1 = _make_conv(1)
                    choice_v1, prob_v1, ev_v1, raw1 = judge_item_choice_probability(
                        client=client,
                        conv_prefix=conv1,
                        audio_cache=audio_cache,
                        max_new_tokens=judge_max_new_tokens,
                        do_sample=judge_do_sample,
                        temperature=judge_temperature,
                        top_p=judge_top_p,
                        seed=judge_seed,
                    )
                    item_runtime_err = None
                    break
                except RuntimeError as e:
                    if _is_gemini_internal_500_runtime_error(e) and attempt == 0:
                        print(
                            f"[warning] Gemini 500 INTERNAL at item_index={item_idx}; retrying once.",
                            flush=True,
                        )
                        continue
                    item_runtime_err = e
                    break
        finally:
            audio_cache.cleanup()

        if item_runtime_err is not None:
            if _is_gemini_internal_500_runtime_error(item_runtime_err):
                fail_notes.append(
                    {
                        "item_index": int(item_idx),
                        "item_type": item_type,
                        "item_label": str(item_label),
                        "target_spk_label": target_spk_label,
                        "variant": "both",
                        "error": f"{item_runtime_err} (retried_once_then_skipped)",
                    }
                )
                continue
            raise item_runtime_err

        if choice_v0 is None or prob_v0 is None:
            fail_notes.append(
                {
                    "item_index": int(item_idx),
                    "item_type": item_type,
                    "item_label": str(item_label),
                    "target_spk_label": target_spk_label,
                    "variant": 0,
                    "raw_output": raw0,
                }
            )
        if choice_v1 is None or prob_v1 is None:
            fail_notes.append(
                {
                    "item_index": int(item_idx),
                    "item_type": item_type,
                    "item_label": str(item_label),
                    "target_spk_label": target_spk_label,
                    "variant": 1,
                    "raw_output": raw1,
                }
            )

        if choice_v0 is None or prob_v0 is None or choice_v1 is None or prob_v1 is None:
            continue

        try:
            score_v0_m = int(map_choice_probability_to_score(choice_v0, float(prob_v0)))
            score_v1_m = int(map_choice_probability_to_score(choice_v1, float(prob_v1)))
        except Exception as e:
            fail_notes.append(
                {
                    "item_index": int(item_idx),
                    "item_type": item_type,
                    "item_label": str(item_label),
                    "target_spk_label": target_spk_label,
                    "variant": "map",
                    "error": str(e),
                    "choice_v0": choice_v0,
                    "prob_v0": prob_v0,
                    "choice_v1": choice_v1,
                    "prob_v1": prob_v1,
                }
            )
            continue

        score_avg = (float(score_v0_m) + float(score_v1_m)) / 2.0

        s0 = _sign(score_v0_m)
        s1 = _sign(score_v1_m)
        if s0 != 0 and s1 != 0 and s0 != s1:
            flip_count += 1

        score_sum += score_avg
        n_item += 1

        rec: Dict[str, Any] = {
            "item_index": int(item_idx),
            "item_type": item_type,
            "item_label": str(item_label),
            "target_spk_label": target_spk_label,
            "tau0": float(TAU0_DEFAULT),
            "tau2": float(TAU2_DEFAULT),
            "choice_v0": str(choice_v0),
            "prob_v0": float(prob_v0),
            "score_v0_mapped": int(score_v0_m),
            "evidence_v0": ev_v0,
            "choice_v1": str(choice_v1),
            "prob_v1": float(prob_v1),
            "score_v1_mapped": int(score_v1_m),
            "evidence_v1": ev_v1,
            "score_avg": float(score_avg),
        }

        if item_type == "interaction":
            rec["context_ipu_id"] = item.get("context_ipu_id", "")
            rec["target_ipu_id"] = item.get("target_ipu_id", "")
            rec["context_active_s"] = item.get("context_active_s", None)
            rec["target_active_s"] = item.get("target_active_s", None)
            rec["target_position"] = int(item.get("target_position", 2))
        elif item_type == "single":
            rec["source_ipus"] = item.get("source_ipus", [])
            rec["active_speech_s"] = item.get("active_speech_s", None)
            rec["segment_duration_s"] = item.get("segment_duration_s", None)

        evidence_items.append(rec)

    if n_item == 0:
        evidence_json = json.dumps(evidence_items, ensure_ascii=False)
        fail_note_json = json.dumps(fail_notes, ensure_ascii=False)
        return evidence_json, float("nan"), float("nan"), fail_note_json

    avg_score = float(score_sum / n_item)
    flip_rate = float(flip_count / n_item)

    evidence_json = json.dumps(evidence_items, ensure_ascii=False)
    fail_note_json = json.dumps(fail_notes, ensure_ascii=False)
    return evidence_json, avg_score, flip_rate, fail_note_json


# ----------------------------
# Pipeline steps: filter_roles + build_eval_inputs
# ----------------------------


def run_filter_roles(
    filter_script: Path,
    roles_of_interest: List[str],
    selection_portion: float,
    seed: Optional[int],
    filtered_csv: Path,
):
    cmd = [
        sys.executable,
        str(filter_script),
        "--names",
        *roles_of_interest,
        "--selection-portion",
        str(selection_portion),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    cmd.extend(["--output-csv", str(filtered_csv)])
    print("Running filter_roles.py:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_build_eval_inputs_for_row(
    build_input_script: Path,
    prompt_id_unique: str,
    a_id: str,
    b_id: str,
    input_mode: str,
):
    cmd = [
        sys.executable,
        str(build_input_script),
        "--prompt-id-unique",
        prompt_id_unique,
        "--a-id",
        a_id,
        "--b-id",
        b_id,
        "--input-mode",
        str(input_mode),
    ]
    print("Running build_eval_inputs.py:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _fmt_hms(seconds: float) -> str:
    try:
        s = float(seconds)
    except Exception:
        return "??:??:??"
    if not np.isfinite(s) or s < 0:
        return "??:??:??"
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ----------------------------
# Args
# ----------------------------


def parse_args():
    _bootstrap_env_from_repo()

    parser = argparse.ArgumentParser(
        description=(
            "Run filter_roles -> build_eval_inputs -> Gemini scoring "
            "(Evidence-First + forced-choice + probability + Balanced Position)."
        )
    )
    parser.add_argument(
        "--roles-of-interest",
        nargs="+",
        required=True,
        help="List of role names to evaluate (same as passed to filter_roles.py).",
    )
    parser.add_argument(
        "--selection-portion",
        type=float,
        default=1.0,
        help="Fraction of matching rows to keep in filtered_subset.csv (0,1], passed to filter_roles.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed forwarded to filter_roles.py for deterministic subset selection. "
            "Default: None (non-deterministic sampling)."
        ),
    )
    parser.add_argument(
        "--start-over",
        type=_str2bool,
        default=True,
        help=(
            "If true, overwrite the filtered CSV and re-run from scratch. "
            "If false, append only new rows and resume unfinished evaluation. Default: true."
        ),
    )
    parser.add_argument(
        "--question-config",
        type=Path,
        required=True,
        help="Path to JSON file with question config (outside_judge only; must contain exactly one question).",
    )
    parser.add_argument(
        "--filtered-csv",
        type=Path,
        default=Path("filtered_subset.csv"),
        help="Path to write/read filtered subset CSV (default: ./filtered_subset.csv).",
    )
    parser.add_argument(
        "--input-mode",
        choices=("single", "interaction"),
        default="interaction",
        help="Input builder mode: 'single' (merged segments) or 'interaction' (context/target pairs).",
    )
    parser.add_argument(
        "--keep-turnovers",
        action="store_true",
        help="If set, keep built input audio files after scoring. Otherwise delete them per conversation.",
    )

    # Gemini model / API config
    parser.add_argument(
        "--gemini-model",
        "--model",
        "--gpt-model",
        "--openai-model",
        "--qwen-model",  # alias for drop-in compatibility with existing job scripts
        dest="model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        help=(
            "Gemini model ID (must support audio input). "
            "Default: env GEMINI_MODEL or 'gemini-2.5-flash'."
        ),
    )
    parser.add_argument(
        "--gemini-api-key",
        default=os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", ""),
        help=(
            "Gemini API key (default: env GEMINI_API_KEY or GOOGLE_API_KEY; "
            "the script also preloads .env/.env/api_keys.env when present)."
        ),
    )
    parser.add_argument(
        "--gemini-timeout-s",
        type=float,
        default=float(os.environ.get("GEMINI_TIMEOUT_S", 120.0)),
        help="Best-effort request timeout in seconds (SDK-dependent; default: 120).",
    )
    parser.add_argument(
        "--gemini-inline-max-bytes",
        type=int,
        default=int(os.environ.get("GEMINI_INLINE_MAX_BYTES", 8 * 1024 * 1024)),
        help=(
            "Inline audio byte limit per audio file. If the prepared WAV exceeds this, "
            "the script uses the Files API upload path for that audio. Default: 8 MiB."
        ),
    )
    parser.add_argument(
        "--gemini-use-files-api",
        type=_str2bool,
        default=_str2bool(os.environ.get("GEMINI_USE_FILES_API", "false")),
        help="If true, always use Files API uploads instead of inline audio bytes. Default: false.",
    )
    parser.add_argument(
        "--gemini-delete-uploaded-files",
        type=_str2bool,
        default=_str2bool(os.environ.get("GEMINI_DELETE_UPLOADED_FILES", "true")),
        help=(
            "If true, delete any Files API uploads created for oversized audio after each item. "
            "Default: true."
        ),
    )

    # Judge decoding controls
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=512,
        help="Max output tokens for judge JSON output (default: 512).",
    )
    parser.add_argument(
        "--judge-do-sample",
        action="store_true",
        help="If set, use sampling (stochastic decoding). Otherwise deterministic decoding.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (used only if --judge-do-sample).",
    )
    parser.add_argument(
        "--judge-top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (used only if --judge-do-sample).",
    )
    parser.add_argument(
        "--judge-seed",
        type=int,
        default=None,
        help="Optional Gemini seed for best-effort determinism.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.selection_portion <= 1.0):
        raise SystemExit("selection_portion must be in the interval (0, 1].")

    if not args.gemini_api_key:
        raise SystemExit(
            "Missing Gemini API key. Set env GOOGLE_API_KEY or GEMINI_API_KEY "
            "(for example in .env/.env/api_keys.env), or pass --gemini-api-key."
        )

    roles_of_interest = args.roles_of_interest

    # Step 1: filter_roles
    if args.start_over:
        run_filter_roles(
            filter_script=FILTER_SCRIPT,
            roles_of_interest=roles_of_interest,
            selection_portion=args.selection_portion,
            seed=args.seed,
            filtered_csv=args.filtered_csv,
        )
    else:
        # Resume mode: merge a newly sampled subset into the existing CSV
        args.filtered_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{args.filtered_csv.stem}_subset_",
            suffix=".csv",
            dir=str(args.filtered_csv.parent),
        )
        os.close(tmp_fd)
        tmp_subset = Path(tmp_name)

        try:
            run_filter_roles(
                filter_script=FILTER_SCRIPT,
                roles_of_interest=roles_of_interest,
                selection_portion=args.selection_portion,
                seed=args.seed,
                filtered_csv=tmp_subset,
            )

            if not args.filtered_csv.exists():
                os.replace(tmp_subset, args.filtered_csv)
                try:
                    dir_fd = os.open(str(args.filtered_csv.parent), os.O_DIRECTORY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except Exception:
                    pass
            else:
                df_existing = pd.read_csv(args.filtered_csv, dtype=str, low_memory=False)
                df_new = pd.read_csv(tmp_subset, dtype=str, low_memory=False)

                key_cols = ["prompt_id_unique", "a_id", "b_id", "role_a", "role_b"]
                for c in key_cols:
                    if c not in df_existing.columns or c not in df_new.columns:
                        raise SystemExit(
                            f"Cannot resume: missing key column '{c}' in existing or new subset CSV."
                        )

                existing_keys = set(df_existing[key_cols].astype(str).agg("||".join, axis=1))
                new_keys = df_new[key_cols].astype(str).agg("||".join, axis=1)
                df_add = df_new.loc[~new_keys.isin(existing_keys)].copy()

                if not df_add.empty:
                    for col in df_add.columns:
                        if col not in df_existing.columns:
                            df_existing[col] = ""
                    for col in df_existing.columns:
                        if col not in df_add.columns:
                            df_add[col] = ""

                    df_add = df_add[df_existing.columns]
                    df_existing = pd.concat([df_existing, df_add], ignore_index=True)
                    atomic_write_csv(df_existing, args.filtered_csv)

        finally:
            try:
                if tmp_subset.exists():
                    tmp_subset.unlink()
            except OSError:
                pass

    # Step 2: load filtered subset
    df = pd.read_csv(args.filtered_csv, dtype=str, low_memory=False)

    required_cols = [
        "prompt_id_unique",
        "a_id",
        "b_id",
        "role_a",
        "role_b",
        "category_a",
        "category_b",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise SystemExit(f"filtered_subset.csv is missing required column '{c}'.")

    n_total = len(df)

    out_cols = [
        "evidence_a",
        "avg_score_a",
        "flip_rate_a",
        "fail_note_a",
        "evidence_b",
        "avg_score_b",
        "flip_rate_b",
        "fail_note_b",
    ]
    for col in out_cols:
        if col not in df.columns:
            df[col] = ""

    atomic_write_csv(df, args.filtered_csv)

    # Step 3: load single-question config
    single_q = load_question_config(args.question_config)
    cat2q = build_category_to_question_map(single_q)

    # Step 4: init Gemini client
    print(f"Using Gemini model: {args.model}")
    judge_client = GeminiJudgeClient(
        api_key=str(args.gemini_api_key),
        model=str(args.model),
        timeout_s=float(args.gemini_timeout_s),
    )

    # Keep consistent with builder (and with Gemini audio preprocessing behavior).
    target_sr = 16000
    tmpdir = Path(tempfile.mkdtemp(prefix="gemini_turnovers_"))

    run_t0 = time.perf_counter()

    def _row_results_all_empty(r: pd.Series) -> bool:
        return all(_is_missing_value(r.get(col, "")) for col in out_cols)

    def _row_needs_eval(r: pd.Series) -> bool:
        if (not args.start_over) and (not _row_results_all_empty(r)):
            return False
        ra = r.get("role_a", "")
        rb = r.get("role_b", "")
        need_a = (ra in roles_of_interest) and _is_missing_value(r.get("avg_score_a", ""))
        need_b = (rb in roles_of_interest) and _is_missing_value(r.get("avg_score_b", ""))
        return bool(need_a or need_b)

    n_eval_total = sum(_row_needs_eval(r) for _, r in df.iterrows())
    n_eval_done = 0

    try:
        for i, (idx, row) in enumerate(df.iterrows(), start=1):
            prompt_id = row["prompt_id_unique"]
            a_id = row["a_id"]
            b_id = row["b_id"]
            role_a = row["role_a"]
            role_b = row["role_b"]
            cat_a = row["category_a"]
            cat_b = row["category_b"]

            if (not args.start_over) and (not _row_results_all_empty(row)):
                continue

            need_eval_a = (role_a in roles_of_interest) and _is_missing_value(row.get("avg_score_a", ""))
            need_eval_b = (role_b in roles_of_interest) and _is_missing_value(row.get("avg_score_b", ""))

            if not need_eval_a and not need_eval_b:
                continue

            run_build_eval_inputs_for_row(
                build_input_script=BUILD_INPUT_SCRIPT,
                prompt_id_unique=prompt_id,
                a_id=a_id,
                b_id=b_id,
                input_mode=args.input_mode,
            )

            out_dir = Path.cwd() / f"{prompt_id}_full_turns"
            base_prefix = f"{prompt_id}_A-{a_id}_B-{b_id}"
            if not out_dir.exists():
                print(f"[warning] Out dir {out_dir} does not exist; skipping.")
                continue

            manifest_path = out_dir / f"{base_prefix}__manifest.jsonl"
            manifest_items = load_eval_manifest(manifest_path)
            if not manifest_items:
                print(f"[warning] No manifest items found for base prefix {base_prefix}.")
                if not args.keep_turnovers:
                    try:
                        if manifest_path.exists():
                            manifest_path.unlink()
                    except OSError:
                        pass
                    try:
                        out_dir.rmdir()
                    except OSError:
                        pass
                continue

            n_eval_done += 1
            elapsed_s = time.perf_counter() - run_t0
            est_total_s = (elapsed_s / n_eval_done) * n_eval_total if n_eval_done > 0 else float("nan")

            print(
                f"\n[progress] Conversation {i}/{n_total} "
                f"[eval {n_eval_done}/{n_eval_total}] "
                f"{_fmt_hms(elapsed_s)}/{_fmt_hms(est_total_s)} "
                f"(idx={idx}, prompt_id={prompt_id}, A_id={a_id}, B_id={b_id})",
                flush=True,
            )

            updated_row = False

            if need_eval_a:
                evidence_a, avg_score_a, flip_rate_a, fail_note_a = score_conversation_speaker(
                    role=role_a,
                    category=cat_a,
                    roles_of_interest=roles_of_interest,
                    questions_by_category=cat2q,
                    manifest_items=manifest_items,
                    target_spk_label="A",
                    client=judge_client,
                    target_sr=target_sr,
                    tmpdir=tmpdir,
                    judge_max_new_tokens=args.judge_max_new_tokens,
                    judge_do_sample=args.judge_do_sample,
                    judge_temperature=args.judge_temperature,
                    judge_top_p=args.judge_top_p,
                    judge_seed=args.judge_seed,
                    gemini_inline_max_bytes=args.gemini_inline_max_bytes,
                    gemini_use_files_api=args.gemini_use_files_api,
                    gemini_delete_uploaded_files=args.gemini_delete_uploaded_files,
                )

                df.at[idx, "evidence_a"] = evidence_a
                df.at[idx, "avg_score_a"] = avg_score_a
                df.at[idx, "flip_rate_a"] = flip_rate_a
                df.at[idx, "fail_note_a"] = fail_note_a
                updated_row = True

            if need_eval_b:
                evidence_b, avg_score_b, flip_rate_b, fail_note_b = score_conversation_speaker(
                    role=role_b,
                    category=cat_b,
                    roles_of_interest=roles_of_interest,
                    questions_by_category=cat2q,
                    manifest_items=manifest_items,
                    target_spk_label="B",
                    client=judge_client,
                    target_sr=target_sr,
                    tmpdir=tmpdir,
                    judge_max_new_tokens=args.judge_max_new_tokens,
                    judge_do_sample=args.judge_do_sample,
                    judge_temperature=args.judge_temperature,
                    judge_top_p=args.judge_top_p,
                    judge_seed=args.judge_seed,
                    gemini_inline_max_bytes=args.gemini_inline_max_bytes,
                    gemini_use_files_api=args.gemini_use_files_api,
                    gemini_delete_uploaded_files=args.gemini_delete_uploaded_files,
                )

                df.at[idx, "evidence_b"] = evidence_b
                df.at[idx, "avg_score_b"] = avg_score_b
                df.at[idx, "flip_rate_b"] = flip_rate_b
                df.at[idx, "fail_note_b"] = fail_note_b
                updated_row = True

            if updated_row:
                atomic_write_csv(df, args.filtered_csv)

            if not args.keep_turnovers:
                cleanup_paths: set[Path] = set()
                cleanup_paths.add(manifest_path)
                for item in manifest_items:
                    for key in ("audio", "context_audio", "target_audio"):
                        pv = item.get(key, "")
                        if not pv:
                            continue
                        cleanup_paths.add(Path(str(pv)))
                for p in cleanup_paths:
                    try:
                        if p.exists():
                            p.unlink()
                    except OSError:
                        pass
                try:
                    out_dir.rmdir()
                except OSError:
                    pass

        print(f"\nUpdated filtered CSV in-place: {args.filtered_csv}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
