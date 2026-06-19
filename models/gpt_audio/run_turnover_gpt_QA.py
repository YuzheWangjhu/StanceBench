#!/usr/bin/env python3
"""run_turnover_gpt_QA.py

Best-effort OpenAI/ChatGPT (API) "model swap" for the local Qwen judge pipeline.

Goals (kept identical to run_turnover_qwen_QA_2.py):
  - Same input construction: filter_roles.py -> build_eval_inputs.py -> manifest JSONL
  - Same per-speaker scoring loop (Balanced Position v0/v1)
  - Same forced-choice JSON output schema: {choice, probability, evidence}
  - Same deterministic mapping: (choice, probability) -> {-2,-1,0,1,2}
  - Same CSV outputs: evidence_{a,b}, avg_score_{a,b}, flip_rate_{a,b}, fail_note_{a,b}

Main difference:
  - The judge call is performed via OpenAI Chat Completions API with audio inputs
    (base64-encoded WAV) and Structured Outputs (json_schema) when available.

Notes:
  - Requires OPENAI_API_KEY (env var) or --openai-api-key.
  - Default model is an audio-capable chat model (see --gpt-model).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
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
import requests
import soundfile as sf


# ----------------------------
# Repo-relative helper scripts
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FILTER_SCRIPT = REPO_ROOT / "scripts" / "filter_roles.py"
BUILD_INPUT_SCRIPT = REPO_ROOT / "scripts" / "build_eval_inputs.py"


# ----------------------------
# JSON extraction helpers
# ----------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from a string.

    API JSON mode should already return valid JSON, but we keep this for
    best-effort robustness.
    """
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


# ----------------------------
# Prompt builders (copied from Qwen script; audio parts are converted later)
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
# Deterministic mapping (same as Qwen script)
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
# Audio helpers (same as Qwen script)
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


# ----------------------------
# OpenAI chat client (requests)
# ----------------------------


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 120.0,
        max_retries: int = 6,
        min_retry_sleep_s: float = 1.0,
        max_retry_sleep_s: float = 30.0,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self._chat_url = self.base_url + "/chat/completions"
        else:
            # Allow passing base_url without /v1
            self._chat_url = self.base_url + "/v1/chat/completions"
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.min_retry_sleep_s = float(min_retry_sleep_s)
        self.max_retry_sleep_s = float(max_retry_sleep_s)
        self.organization = organization
        self.project = project

        # Best-effort per-model capability cache to avoid repeated 400s.
        self._supports_json_schema: Optional[bool] = None
        self._supports_response_format: Optional[bool] = None
        self._supports_max_completion_tokens: Optional[bool] = None
        self._supports_modalities: Optional[bool] = None

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            h["OpenAI-Organization"] = self.organization
        if self.project:
            h["OpenAI-Project"] = self.project
        return h

    def _default_response_format(self) -> Dict[str, Any]:
        # Structured Outputs: enforce {choice, probability, evidence}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_output",
                "description": "Forced-choice (P/N) with probability and evidence.",
                "schema": {
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
                },
                "strict": True,
            },
        }

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        seed: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Return (assistant_content, full_response_json)."""

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "n": 1,
        }
        if self._supports_max_completion_tokens is not False:
            payload["max_completion_tokens"] = int(max_completion_tokens)
        else:
            payload["max_tokens"] = int(max_completion_tokens)
        if self._supports_modalities is not False:
            payload["modalities"] = ["text"]

        if seed is not None:
            payload["seed"] = int(seed)

        # Prefer json_schema; fall back to json_object when schema is unsupported.
        if self._supports_response_format is not False:
            if self._supports_json_schema is not False:
                payload["response_format"] = self._default_response_format()
            else:
                payload["response_format"] = {"type": "json_object"}

        last_err: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    self._chat_url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )

                if r.status_code == 200:
                    j = r.json()
                    if "response_format" in payload:
                        self._supports_response_format = True
                    # Mark json_schema support if we used it successfully.
                    if payload.get("response_format", {}).get("type") == "json_schema":
                        self._supports_json_schema = True
                    if "max_completion_tokens" in payload:
                        self._supports_max_completion_tokens = True
                    if "modalities" in payload:
                        self._supports_modalities = True

                    content = (
                        j.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if not isinstance(content, str):
                        content = str(content)
                    return content, j

                # Try to parse error
                try:
                    err = r.json().get("error", {})
                    err_msg = err.get("message", r.text)
                    err_type = err.get("type", "")
                except Exception:
                    err_msg = r.text
                    err_type = ""

                msg_l = str(err_msg).lower()
                invalid_param = (
                    "invalid parameter" in msg_l
                    or "unknown parameter" in msg_l
                    or "not supported" in msg_l
                    or "unsupported" in msg_l
                )

                if r.status_code == 400:
                    if (
                        invalid_param
                        and "max_completion_tokens" in msg_l
                        and "max_completion_tokens" in payload
                    ):
                        payload["max_tokens"] = payload.pop("max_completion_tokens")
                        self._supports_max_completion_tokens = False
                        last_err = f"400 max_completion_tokens unsupported: {err_msg}"
                        continue

                    if invalid_param and "modalities" in msg_l and "modalities" in payload:
                        payload.pop("modalities", None)
                        self._supports_modalities = False
                        last_err = f"400 modalities unsupported: {err_msg}"
                        continue

                    rf = payload.get("response_format")
                    rf_type = rf.get("type") if isinstance(rf, dict) else ""
                    mentions_response_format = "response_format" in msg_l
                    mentions_json_schema = "json_schema" in msg_l

                    if (
                        rf_type == "json_schema"
                        and (mentions_json_schema or (mentions_response_format and not invalid_param))
                    ):
                        self._supports_json_schema = False
                        payload["response_format"] = {"type": "json_object"}
                        last_err = f"400 json_schema unsupported: {err_msg}"
                        continue

                    if (
                        "response_format" in payload
                        and invalid_param
                        and (mentions_response_format or mentions_json_schema)
                    ):
                        payload.pop("response_format", None)
                        self._supports_response_format = False
                        self._supports_json_schema = False
                        last_err = f"400 response_format unsupported: {err_msg}"
                        continue

                # Retry on common transient statuses.
                if r.status_code in {408, 409, 429, 500, 502, 503, 504}:
                    last_err = f"HTTP {r.status_code}: {err_type} {err_msg}"
                else:
                    raise RuntimeError(f"OpenAI API error {r.status_code}: {err_type} {err_msg}")

            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = f"network error: {e}"
            except Exception as e:
                # Non-retryable error
                raise

            # Backoff
            if attempt < self.max_retries:
                sleep_s = min(self.max_retry_sleep_s, self.min_retry_sleep_s * (2**attempt))
                sleep_s = sleep_s * (0.75 + 0.5 * random.random())  # jitter
                time.sleep(sleep_s)

        raise RuntimeError(f"OpenAI API request failed after retries: {last_err}")


# Cache for base64-encoded audio files
_AUDIO_B64_CACHE: Dict[str, str] = {}


def _audio_path_to_b64(path: str) -> str:
    p = str(Path(path).resolve())
    if p in _AUDIO_B64_CACHE:
        return _AUDIO_B64_CACHE[p]
    with open(p, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("ascii")
    _AUDIO_B64_CACHE[p] = b64
    return b64


def _conv_prefix_to_openai_messages(conv_prefix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert the Qwen-style conv_prefix (with audio paths) to OpenAI messages."""
    out: List[Dict[str, Any]] = []
    for msg in conv_prefix:
        role = str(msg.get("role", "")).strip() or "user"
        content = msg.get("content", "")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue

        parts_out: List[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type", "")).strip().lower()
            if ptype == "text":
                parts_out.append({"type": "text", "text": str(part.get("text", ""))})
            elif ptype == "audio":
                apath = str(part.get("path", ""))
                if not apath:
                    continue
                # Assume wav unless file extension suggests mp3
                fmt = "mp3" if apath.lower().endswith(".mp3") else "wav"
                parts_out.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": _audio_path_to_b64(apath),
                            "format": fmt,
                        },
                    }
                )
            else:
                # Ignore unknown part types
                continue

        out.append({"role": role, "content": parts_out})

    return out


def judge_item_choice_probability(
    client: OpenAIChatClient,
    conv_prefix: List[Dict[str, Any]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: Optional[int] = None,
) -> Tuple[Optional[str], Optional[float], List[str], str]:
    """OpenAI API judge; mirrors the Qwen function signature/output."""

    messages = _conv_prefix_to_openai_messages(conv_prefix)

    if do_sample:
        temp = float(temperature)
        tp = float(top_p)
    else:
        # Greedy-ish
        temp = 0.0
        tp = 1.0

    raw_text, _full = client.chat_completion(
        messages=messages,
        max_completion_tokens=int(max_new_tokens),
        temperature=temp,
        top_p=tp,
        seed=seed,
    )
    raw_text = str(raw_text).strip()

    obj = _extract_json_object(raw_text)
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
# Question config + manifest parsing (identical to Qwen script)
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
# CSV utilities (same behavior as Qwen script)
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


# ----------------------------
# Core per-conversation scoring (same logic; only judge call differs)
# ----------------------------


def score_conversation_speaker(
    role: str,
    category: str,
    roles_of_interest: List[str],
    questions_by_category: Dict[str, Dict[str, Any]],
    manifest_items: List[Dict[str, Any]],
    target_spk_label: str,
    client: OpenAIChatClient,
    target_sr: int,
    tmpdir: Path,
    judge_max_new_tokens: int,
    judge_do_sample: bool,
    judge_temperature: float,
    judge_top_p: float,
    judge_seed: Optional[int] = None,
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
                _AUDIO_B64_CACHE.pop(str(ctx_tmp.resolve()), None)
                sf.write(str(tgt_tmp), tgt_audio, sr)
                _AUDIO_B64_CACHE.pop(str(tgt_tmp.resolve()), None)

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
                _AUDIO_B64_CACHE.pop(str(tgt_tmp.resolve()), None)

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

        conv0 = _make_conv(0)
        choice_v0, prob_v0, ev_v0, raw0 = judge_item_choice_probability(
            client=client,
            conv_prefix=conv0,
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
            max_new_tokens=judge_max_new_tokens,
            do_sample=judge_do_sample,
            temperature=judge_temperature,
            top_p=judge_top_p,
            seed=judge_seed,
        )

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
    parser = argparse.ArgumentParser(
        description=(
            "Run filter_roles -> build_eval_inputs -> OpenAI (ChatGPT) scoring "
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

    # Model / API config
    parser.add_argument(
        "--gpt-model",
        "--openai-model",
        "--qwen-model",  # alias for drop-in compatibility with existing command lines
        dest="model",
        default=os.environ.get("OPENAI_MODEL", "gpt-audio-2025-08-28"),
        help=(
            "OpenAI chat model ID (must support audio input). "
            "Default: env OPENAI_MODEL or 'gpt-audio-2025-08-28'."
        ),
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (default: env OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI API base URL (default: https://api.openai.com/v1).",
    )
    parser.add_argument(
        "--openai-org",
        default=os.environ.get("OPENAI_ORG", ""),
        help="Optional OpenAI organization (header OpenAI-Organization).",
    )
    parser.add_argument(
        "--openai-project",
        default=os.environ.get("OPENAI_PROJECT", ""),
        help="Optional OpenAI project (header OpenAI-Project).",
    )
    parser.add_argument(
        "--openai-timeout-s",
        type=float,
        default=float(os.environ.get("OPENAI_TIMEOUT_S", 120.0)),
        help="HTTP timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=int(os.environ.get("OPENAI_MAX_RETRIES", 6)),
        help="Max retries for transient API errors (default: 6).",
    )

    # Judge decoding controls
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=128,
        help="Max completion tokens for judge JSON output (default: 128).",
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
        help="Optional OpenAI seed for best-effort determinism.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.selection_portion <= 1.0):
        raise SystemExit("selection_portion must be in the interval (0, 1].")

    if not args.openai_api_key:
        raise SystemExit(
            "Missing OpenAI API key. Set env OPENAI_API_KEY or pass --openai-api-key."
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

    # Step 4: init OpenAI client
    print(f"Using OpenAI model: {args.model}")
    client = OpenAIChatClient(
        api_key=str(args.openai_api_key),
        model=str(args.model),
        base_url=str(args.openai_base_url),
        timeout_s=float(args.openai_timeout_s),
        max_retries=int(args.openai_max_retries),
        organization=str(args.openai_org).strip() or None,
        project=str(args.openai_project).strip() or None,
    )

    # OpenAI audio models typically accept 16k; keep consistent with builder.
    target_sr = 16000
    tmpdir = Path(tempfile.mkdtemp(prefix="gpt_turnovers_"))

    run_t0 = time.perf_counter()

    def _row_needs_eval(r: pd.Series) -> bool:
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
                    client=client,
                    target_sr=target_sr,
                    tmpdir=tmpdir,
                    judge_max_new_tokens=args.judge_max_new_tokens,
                    judge_do_sample=args.judge_do_sample,
                    judge_temperature=args.judge_temperature,
                    judge_top_p=args.judge_top_p,
                    judge_seed=args.judge_seed,
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
                    client=client,
                    target_sr=target_sr,
                    tmpdir=tmpdir,
                    judge_max_new_tokens=args.judge_max_new_tokens,
                    judge_do_sample=args.judge_do_sample,
                    judge_temperature=args.judge_temperature,
                    judge_top_p=args.judge_top_p,
                    judge_seed=args.judge_seed,
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
