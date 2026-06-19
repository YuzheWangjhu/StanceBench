#!/usr/bin/env python3
"""\
Pipeline script (outside-judge only; Evidence-First; Balanced Position with forced-choice + probability):

1) Call filter_roles.py to create a filtered subset CSV from interactions_role_ABmapped.csv,
   restricted to a set of role labels of interest.

2) For each conversation (row) in the filtered subset:
   - Call build_eval_inputs.py to build evaluation inputs and a manifest.
   - Read the manifest and evaluate each target-speaker item.
   - Look up the selected OUTSIDE-JUDGE question config by category_a/category_b.
   - For each target-speaker evaluation item, run the judge model twice (Balanced Position):
         * Variant 0: Definition P first, then Definition N
         * Variant 1: Definition N first, then Definition P
     Each run must return JSON only:
       {"choice":"P"|"N", "probability": 0.0 to 1.0, "evidence": [...]}

     Deterministic mapping from (choice, probability) to score in {-2,-1,0,1,2}:
       - probability < tau0  -> 0
       - tau0 <= probability < tau2 -> +/-1
       - probability >= tau2 -> +/-2

     Item score is averaged:
       score_avg = (score_v0_mapped + score_v1_mapped) / 2

     flip_rate counts sign disagreements between v0 and v1, excluding zeros.

3) Write results back to the same --filtered-csv file incrementally (after each conversation).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

import logging

class _SuppressQwenSystemPromptWarning(logging.Filter):
    _needle = "System prompt modified, audio output may not work as expected"
    def filter(self, record):
        return self._needle not in record.getMessage()

logging.getLogger().addFilter(_SuppressQwenSystemPromptWarning())

# Fixed script paths (set these at the top as needed)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FILTER_SCRIPT = REPO_ROOT / "scripts" / "filter_roles.py"
BUILD_INPUT_SCRIPT = REPO_ROOT / "scripts" / "build_eval_inputs.py"


def _is_missing_value(v: Any) -> bool:
    """Return True if a value should be treated as missing/unevaluated."""
    if v is None:
        return True
    s = str(v).strip()
    return (s == "") or (s.lower() in {"nan", "none", "null"})


def atomic_write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Write CSV via a temp file + atomic rename (os.replace) for robustness."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=csv_path.name + ".",
        suffix=".tmp",
        dir=str(csv_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(str(tmp_path), str(csv_path))

        # Ensure the directory entry is durable on POSIX filesystems.
        try:
            dir_fd = os.open(str(csv_path.parent), os.O_DIRECTORY)
        except Exception:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        # If something went wrong before os.replace, clean up the temp file.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _str2bool(v: Any) -> bool:
    """Argparse helper: parse common true/false strings into bool."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


# ------------ Judge helpers ------------

def _decode_ids(processor: Qwen2_5OmniProcessor, token_ids: torch.Tensor) -> str:
    """
    Robustly decode generated token ids across processor/tokenizer variants.
    """
    # Prefer processor.batch_decode if available
    if hasattr(processor, "batch_decode"):
        try:
            return processor.batch_decode([token_ids], skip_special_tokens=True)[0]
        except Exception:
            pass

    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "decode"):
        try:
            return tok.decode(token_ids.tolist(), skip_special_tokens=True)
        except Exception:
            try:
                return tok.decode(token_ids, skip_special_tokens=True)
            except Exception:
                pass

    # Fallback
    return str(token_ids)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of a single JSON object from model text.
    Uses a JSONDecoder scan so it can recover a JSON object even if the model
    outputs extra content before/after the JSON.

    Returns:
        dict if a JSON object is found and parsed; else None.
    """
    if not text:
        return None

    s = text.strip()

    # If model outputs just an integer score, accept it as {"score": int}
    if re.fullmatch(r"-?\d+", s):
        try:
            return {"score": int(s)}
        except Exception:
            return None

    # Fast path: whole string is valid JSON dict
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()

    # Scan from each '{' position and try raw_decode
    start = 0
    while True:
        i = s.find("{", start)
        if i == -1:
            break
        try:
            obj, end = decoder.raw_decode(s, i)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = i + 1

    # Also handle the case where the JSON is an object inside code fences
    # (e.g., ```json {...} ```), by scanning the unfenced text.
    # This is still a scan, just with common fence markers removed.
    unfenced = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).replace("```", "")
    if unfenced != s:
        start = 0
        while True:
            i = unfenced.find("{", start)
            if i == -1:
                break
            try:
                obj, end = decoder.raw_decode(unfenced, i)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            start = i + 1

    # Regex fallback: if there's a "score": <int> somewhere but JSON is malformed
    m_score = re.search(r'"?score"?\s*[:=]\s*(-?\d+)', s)
    if m_score:
        try:
            return {"score": int(m_score.group(1))}
        except Exception:
            return None

    return None

def _coerce_evidence(evidence_val: Any) -> List[str]:
    """
    Convert various evidence formats into a list of strings.
    """
    if evidence_val is None:
        return []
    if isinstance(evidence_val, list):
        out = []
        for x in evidence_val:
            if isinstance(x, str):
                t = x.strip()
                if t:
                    out.append(t)
            else:
                t = str(x).strip()
                if t:
                    out.append(t)
        return out
    if isinstance(evidence_val, str):
        # Split on newlines / bullets
        lines = [ln.strip(" \t-•") for ln in evidence_val.splitlines()]
        return [ln for ln in lines if ln]
    # Fallback: string cast
    t = str(evidence_val).strip()
    return [t] if t else []

def build_judge_conv_prefix_single(
    question_text: str,
    def_p: str,
    def_n: str,
    audio_path: str,
    variant: int,
) -> List[Dict[str, Any]]:
    """
    Build a chat prompt for Category 1 (single-speaker, no interlocutor context).

    variant controls ordering to reduce position bias:
      variant=0: Definition P first, then Definition N
      variant=1: Definition N first, then Definition P

    The judge must output JSON only:
      {"choice":"P"|"N", "probability":0.0 to 1.0, "evidence":[...]}
    """
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
    if variant % 2 == 0:
        defs_text = p_block + "\n" + n_block
    else:
        defs_text = n_block + "\n" + p_block

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
    variant: int,
) -> List[Dict[str, Any]]:
    """
    Build a chat prompt for Category 2 (interaction evaluation).

    Audio order:
      - First audio: other speaker context (immediately preceding partner turn)
      - Second audio: TARGET speaker response (to be evaluated)

    variant controls ordering to reduce position bias:
      variant=0: Definition P first, then Definition N
      variant=1: Definition N first, then Definition P

    The judge must output JSON only:
      {"choice":"P"|"N", "probability":0.0 to 1.0, "evidence":[...]}
    """
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
    if variant % 2 == 0:
        defs_text = p_block + "\n" + n_block
    else:
        defs_text = n_block + "\n" + p_block

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
    variant: int,
) -> List[Dict[str, Any]]:
    """
    Build a chat prompt for Category 2 where Audio 1 is TARGET and Audio 2 is context.
    """
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
    if variant % 2 == 0:
        defs_text = p_block + "\n" + n_block
    else:
        defs_text = n_block + "\n" + p_block

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


def judge_item_choice_probability(
    processor: Qwen2_5OmniProcessor,
    model: Qwen2_5OmniThinkerForConditionalGeneration,
    conv_prefix: List[Dict[str, Any]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[Optional[str], Optional[float], List[str], str]:
    """
    Run judge once on a single evaluation item, returning:
      choice: "P" or "N" (or None if parse failed)
      probability: float in [0.0, 1.0] (or None if parse failed)
      evidence: list of strings (may be empty if not provided)
      raw_text: decoded model output (for logging/debugging)
    """
    inputs = processor.apply_chat_template(
        conv_prefix,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        pad_id = getattr(tok, "pad_token_id", None)
        eos_id = getattr(tok, "eos_token_id", None)
        if pad_id is None and eos_id is not None:
            gen_kwargs["pad_token_id"] = eos_id

    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0, prompt_len:]
    raw_text = _decode_ids(processor, gen_ids).strip()

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
        if probability < 0.0:
            probability = 0.0
        if probability > 1.0:
            probability = 1.0

    evidence = _coerce_evidence(obj.get("evidence", None))
    if len(evidence) > 4:
        evidence = evidence[:4]

    if choice is None or probability is None:
        return None, None, evidence, raw_text

    return choice, probability, evidence, raw_text


TAU0_DEFAULT = 0.45
TAU2_DEFAULT = 0.75


def map_choice_probability_to_score(
    choice: str,
    probability: float,
    tau0: float = TAU0_DEFAULT,
    tau2: float = TAU2_DEFAULT,
) -> int:
    """
    Deterministic mapping from (choice, probability) to score in {-2,-1,0,1,2}.

    Let s = +1 if choice == "P" else -1.

      - if probability < tau0: score = 0
      - else if probability >= tau2: score = 2*s
      - else: score = 1*s
    """
    c = str(choice).strip().upper()
    if c not in {"P", "N"}:
        raise ValueError(f"Invalid choice: {choice!r}")
    try:
        prob = float(probability)
    except Exception as e:
        raise ValueError(f"Invalid probability: {probability!r}") from e

    if prob < 0.0:
        prob = 0.0
    if prob > 1.0:
        prob = 1.0

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


# ------------ Audio helpers ------------

def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    m = float(np.max(np.abs(x)))
    if m > 0:
        x = x / m
    return x.astype(np.float32)


def extract_speaker_segments_from_turnover(
    wav_path: Path,
    target_sr: int,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Read a turnover .wav (stereo from turnover_extractor.py) and return:

        seg1: mono audio for Speaker 1 in this turnover
        seg2: mono audio for Speaker 2 in this turnover
        sr:   sampling rate (target_sr)
        tag:  "AtoB" or "BtoA"

    Mapping:

    - File name contains "_AtoB_" => A speaks first, B second
        seg1 = left channel (A), seg2 = right channel (B)
    - File name contains "_BtoA_" => B speaks first, A second
        seg1 = right channel (B), seg2 = left channel (A)
    """
    name = wav_path.name
    if "_AtoB_" in name:
        tag = "AtoB"
    elif "_BtoA_" in name:
        tag = "BtoA"
    else:
        raise RuntimeError(f"Turnover file {name} does not contain AtoB/BtoA tag.")

    data, sr = sf.read(str(wav_path), always_2d=True)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Expected stereo turnover, got shape {data.shape} for {wav_path}.")

    L = data[:, 0]
    R = data[:, 1]

    if tag == "AtoB":
        spk1 = L
        spk2 = R
    else:  # BtoA
        spk1 = R
        spk2 = L

    if sr != target_sr:
        spk1 = librosa.resample(spk1, orig_sr=sr, target_sr=target_sr)
        spk2 = librosa.resample(spk2, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    spk1 = normalize_audio(spk1)
    spk2 = normalize_audio(spk2)
    return spk1, spk2, sr, tag


# ------------ Question config helpers ------------

def load_question_config(path: Path) -> Dict[str, Any]:
    """
    Load question config JSON.

    Expected structure:
    {
      "outside_judge": [
        {
          "question": "...",
          "related_categories": [...],
          "positive_followups": [...],
          "negative_followups": [...],
          "positive_defination": "...",
          "negative_defination": "..."
        }
      ]
    }

    This pipeline expects exactly ONE question entry in outside_judge.
    Returns that single question dict.
    """
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit("question-config must be a JSON object with key 'outside_judge'.")

    sec = cfg.get("outside_judge", None)
    if not isinstance(sec, list) or not sec:
        raise SystemExit("question-config must contain a non-empty 'outside_judge' list.")

    if len(sec) != 1:
        raise SystemExit(
            "This pipeline runs ONE question at a time. "
            "Please provide a single-question config JSON (outside_judge list length must be 1). "
            "Tip: use select_one_question.py to extract one question."
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
    """
    Map each category key in the selected single question to the single question config.
    """
    cat2q: Dict[str, Dict[str, Any]] = {}
    for cat in single_question["related_categories"]:
        if cat in cat2q:
            raise SystemExit(f"Duplicate category '{cat}' inside related_categories.")
        cat2q[cat] = single_question
    return cat2q


# ------------ Core per-conversation scoring ------------

def _normalize_target_spk_label(v: Any) -> Optional[str]:
    s = str(v).strip()
    low = s.lower()
    if low in {"a", "spka", "speakera", "speaker_a"}:
        return "A"
    if low in {"b", "spkb", "speakerb", "speaker_b"}:
        return "B"
    return None


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


def score_conversation_speaker(
    role: str,
    category: str,
    roles_of_interest: List[str],
    questions_by_category: Dict[str, Dict[str, Any]],
    manifest_items: List[Dict[str, Any]],
    target_spk_label: str,
    processor: Qwen2_5OmniProcessor,
    model: Qwen2_5OmniThinkerForConditionalGeneration,
    target_sr: int,
    tmpdir: Path,
    judge_max_new_tokens: int,
    judge_do_sample: bool,
    judge_temperature: float,
    judge_top_p: float,
) -> Tuple[str, float, float, str]:
    """
    Score one speaker (A or B) for a single conversation using manifest entries.

    Uses Balanced Position (two definition-order variants). Each variant yields:
      - choice in {"P","N"}
      - probability in [0.0,1.0]
      - evidence list

    Each variant is mapped deterministically to a score in {-2,-1,0,1,2}.
    Item score is averaged: score_avg = (score_v0_mapped + score_v1_mapped) / 2.

    flip_rate counts sign disagreements between v0 and v1, excluding zeros.
    """
    if role not in roles_of_interest:
        return "", float("nan"), float("nan"), ""

    if category not in questions_by_category:
        raise SystemExit(f"Category '{category}' for role '{role}' has no question config.")

    qcfg = questions_by_category[category]
    q_text = str(qcfg["question"])
    def_p = str(qcfg["positive_defination"])
    def_n = str(qcfg["negative_defination"])

    # Select manifest items for this target speaker.
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

        # Prepare audio for this item and build prompts.
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

        # Balanced Position: exactly two definition-order variants
        conv0 = _make_conv(0)
        choice_v0, prob_v0, ev_v0, raw0 = judge_item_choice_probability(
            processor=processor,
            model=model,
            conv_prefix=conv0,
            max_new_tokens=judge_max_new_tokens,
            do_sample=judge_do_sample,
            temperature=judge_temperature,
            top_p=judge_top_p,
        )

        conv1 = _make_conv(1)
        choice_v1, prob_v1, ev_v1, raw1 = judge_item_choice_probability(
            processor=processor,
            model=model,
            conv_prefix=conv1,
            max_new_tokens=judge_max_new_tokens,
            do_sample=judge_do_sample,
            temperature=judge_temperature,
            top_p=judge_top_p,
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
            # Skip item if we cannot apply Balanced Position properly
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

    cmd.extend([
        "--output-csv",
        str(filtered_csv),
    ])
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
                raise SystemExit(f"Invalid target_spk at {manifest_path}:{line_no}: {obj.get('target_spk')!r}")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run filter_roles -> build_eval_inputs -> Qwen scoring (Evidence-First + forced-choice + probability + Balanced Position)."
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
            "If true, overwrite the filtered CSV and re-run from scratch (current behavior). "
            "If false, append only new rows from the newly-sampled subset and resume unfinished evaluation. "
            "Default: true."
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
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen2.5-Omni-7B",
        help="Qwen2.5-Omni Thinker checkpoint ID (default: Qwen/Qwen2.5-Omni-7B).",
    )
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens to generate for judge JSON output (default: 128).",
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
    return parser.parse_args()

def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS; return ?? if unknown."""
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

def main():
    args = parse_args()

    if not (0.0 < args.selection_portion <= 1.0):
        raise SystemExit("selection_portion must be in the interval (0, 1].")

    roles_of_interest = args.roles_of_interest

    # Step 1: run filter_roles.py to produce a candidate subset.
    # start_over=True preserves the previous behavior (overwrite from scratch).
    if args.start_over:
        run_filter_roles(
            filter_script=FILTER_SCRIPT,
            roles_of_interest=roles_of_interest,
            selection_portion=args.selection_portion,
            seed=args.seed,
            filtered_csv=args.filtered_csv,
        )
    else:
        # Always generate a fresh candidate subset to a temporary file, then
        # merge it into the existing filtered_csv by appending only new rows.
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
                    # Align schemas (preserve existing columns/order; new rows get empty output fields).
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

    # Step 2: load filtered subset CSV
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

    # Output columns (in-place update of args.filtered_csv)
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

    # Persist the new columns immediately so the CSV schema updates before scoring begins.
    atomic_write_csv(df, args.filtered_csv)

    # Step 3: load single-question config and build category->question map
    single_q = load_question_config(args.question_config)
    cat2q = build_category_to_question_map(single_q)

    # Step 4: load Qwen model
    print(f"Loading Qwen model: {args.qwen_model}")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.qwen_model)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.qwen_model,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    audio_fe = getattr(processor, "feature_extractor", None) or getattr(processor, "audio_processor", None)
    target_sr = getattr(audio_fe, "sampling_rate", 16000)

    tmpdir = Path(tempfile.mkdtemp(prefix="qwen_turnovers_"))

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

            # Resume logic: only evaluate speakers that (a) have a role of interest and
            # (b) have not been evaluated yet.
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

            # --- timing tracker: elapsed / estimated total for remaining evaluations ---
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
                # Speaker A: evaluate entries in the manifest targeting A.
                evidence_a, avg_score_a, flip_rate_a, fail_note_a = score_conversation_speaker(
                    role=role_a,
                    category=cat_a,
                    roles_of_interest=roles_of_interest,
                    questions_by_category=cat2q,
                    manifest_items=manifest_items,
                    target_spk_label="A",
                    processor=processor,
                    model=model,
                    target_sr=target_sr,
                    tmpdir=tmpdir,
                    judge_max_new_tokens=args.judge_max_new_tokens,
                    judge_do_sample=args.judge_do_sample,
                    judge_temperature=args.judge_temperature,
                    judge_top_p=args.judge_top_p,
                )

                df.at[idx, "evidence_a"] = evidence_a
                df.at[idx, "avg_score_a"] = avg_score_a
                df.at[idx, "flip_rate_a"] = flip_rate_a
                df.at[idx, "fail_note_a"] = fail_note_a
                updated_row = True

            if need_eval_b:
                # Speaker B: evaluate entries in the manifest targeting B.
                evidence_b, avg_score_b, flip_rate_b, fail_note_b = score_conversation_speaker(
                    role=role_b,
                    category=cat_b,
                    roles_of_interest=roles_of_interest,
                    questions_by_category=cat2q,
                    manifest_items=manifest_items,
                    target_spk_label="B",
                    processor=processor,
                    model=model,
                    target_sr=target_sr,
                    tmpdir=tmpdir,
                    judge_max_new_tokens=args.judge_max_new_tokens,
                    judge_do_sample=args.judge_do_sample,
                    judge_temperature=args.judge_temperature,
                    judge_top_p=args.judge_top_p,
                )

                df.at[idx, "evidence_b"] = evidence_b
                df.at[idx, "avg_score_b"] = avg_score_b
                df.at[idx, "flip_rate_b"] = flip_rate_b
                df.at[idx, "fail_note_b"] = fail_note_b
                updated_row = True

            # Persist progress after each evaluated conversation row (in-place update, atomic).
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
