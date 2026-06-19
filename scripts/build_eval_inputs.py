#!/usr/bin/env python3
"""
Build evaluation inputs for one conversation from IPU extraction.

Workflow:
1) Run turnover_extractor_IPU.py in a temporary working directory.
2) Read extracted *.meta.json sidecars for one (prompt_id_unique, a_id, b_id).
3) Build either:
   - single: merged target-speaker segments, or
   - interaction: (context, target) audio pairs.
4) Write only final evaluation audios + a JSONL manifest to:
   <cwd>/<prompt_id_unique>_full_turns/
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
IPU_EXTRACTOR_SCRIPT = SCRIPT_DIR / "turnover_extractor_IPU.py"
SILENCE_BETWEEN_IPUS_S = 0.25

# Category 1 (single-speaker merged segments)
SINGLE_TARGET_ACTIVE_MIN_S = 30.0
SINGLE_TARGET_ACTIVE_MAX_S = 45.0
SINGLE_TAIL_MIN_KEEP_S = 15.0
SINGLE_MAX_BOUNDARY_FRACTION = 0.25

# Category 2 (interaction context/target pairs; switch-event v2)
BC_MIN_ACTIVE_S = 1.0
GAP_FILL_MAX_S = 0.7
TARGET_ACTIVE_MIN_S = 2.0
TARGET_ACTIVE_MAX_S = 20.0
CONTEXT_MAX_ACTIVE_S = 15.0
CONTEXT_MAX_LOOKBACK_WALL_S = 30.0
# TARGET_TURN_SLICE = "first15"

SWITCH_HOP_MS = 10.0
SWITCH_WIN_MS = 25.0
SWITCH_BOUNDARY_EPS_S = 0.35
SWITCH_GAP_MAX_S = 2.0
SWITCH_OVERLAP_MAX_S = 0.5

BC_ACTIVE_MAX_S = 0.8
TARGET_MIN_S = 2.0
TARGET_MAX_S = 15.0
CTX_MIN_S = 1.0
CTX_MAX_S = 10.0

MAX_EXTEND_IPUS = 6
MAX_LOOKBACK_WALL_S = 30.0
SWITCH_DEDUP_WINDOW_S = 2.0
MAX_SWITCH_EVENTS_PER_CONV = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run turnover_extractor_IPU.py in a temp directory and build eval inputs "
            "(single or interaction) + manifest for one conversation."
        )
    )
    parser.add_argument("--prompt-id-unique", required=True, help="Conversation prompt_id_unique.")
    parser.add_argument("--a-id", required=True, help="A-side recording ID (a_id).")
    parser.add_argument("--b-id", required=True, help="B-side recording ID (b_id).")
    parser.add_argument(
        "--input-mode",
        choices=("single", "interaction"),
        required=True,
        help="single: merged target-speaker clips; interaction: context/target pairs.",
    )
    parser.add_argument(
        "--extractor-script",
        type=Path,
        default=IPU_EXTRACTOR_SCRIPT,
        help="Path to turnover_extractor_IPU.py (default: scripts/turnover_extractor_IPU.py).",
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Path to downloaded Seamless Interaction improvised audio root. Defaults to extractor env.",
    )
    parser.add_argument(
        "--dyad-lookup-csv",
        default="",
        help="Path to Seamless Interaction dyad_lookup.csv. Defaults to extractor env.",
    )
    parser.add_argument(
        "--interactions-role-abmapped-csv",
        default="",
        help="Path to StanceBench AB-mapped interaction metadata. Defaults to metadata/interactions_role_ABmapped.csv.",
    )
    return parser.parse_args()


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    ratio = float(sr_out) / float(sr_in)
    n_out = int(round(len(x) * ratio))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def run_ipu_extractor_temp(
    extractor_script: Path,
    temp_workdir: Path,
    prompt_id_unique: str,
    a_id: str,
    b_id: str,
    dataset_root: str = "",
    dyad_lookup_csv: str = "",
    interactions_role_abmapped_csv: str = "",
) -> Path:
    extractor_script = Path(extractor_script).expanduser()
    if extractor_script.is_absolute():
        extractor_script = extractor_script.resolve()
    else:
        # Prefer CWD-relative for caller overrides; fall back to this script's directory.
        cand_cwd = (Path.cwd() / extractor_script).resolve()
        cand_local = (SCRIPT_DIR / extractor_script).resolve()
        extractor_script = cand_cwd if cand_cwd.exists() else cand_local

    if not extractor_script.exists():
        raise SystemExit(f"Extractor script not found: {extractor_script}")

    cmd = [
        sys.executable,
        str(extractor_script),
        "--prompt-id-unique",
        prompt_id_unique,
        "--a-id",
        a_id,
        "--b-id",
        b_id,
    ]
    if dataset_root:
        cmd.extend(["--dataset-root", dataset_root])
    if dyad_lookup_csv:
        cmd.extend(["--dyad-lookup-csv", dyad_lookup_csv])
    if interactions_role_abmapped_csv:
        cmd.extend(["--interactions-role-abmapped-csv", interactions_role_abmapped_csv])
    print("Running turnover_extractor_IPU.py:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(temp_workdir))
    return temp_workdir / f"{prompt_id_unique}_full_turns"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return float(default)


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def load_ipu_metas(temp_out_dir: Path, base_prefix: str) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for meta_path in sorted(temp_out_dir.glob(f"{base_prefix}_spk*.meta.json")):
        try:
            obj = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warning] Failed to parse {meta_path}: {e}", file=sys.stderr)
            continue

        ipu_wav = obj.get("ipu_wav", "")
        if not ipu_wav:
            continue
        ipu_wav_path = Path(str(ipu_wav))
        if not ipu_wav_path.is_absolute():
            ipu_wav_path = (meta_path.parent / ipu_wav_path).resolve()

        speaker = str(obj.get("speaker", "")).strip().upper()
        if speaker not in {"A", "B"}:
            continue

        item = {
            "meta_path": str(meta_path.resolve()),
            "ipu_wav": str(ipu_wav_path),
            "ipu_id": Path(str(ipu_wav_path)).stem,
            "speaker": speaker,
            "clip_start_s": _safe_float(obj.get("clip_start_s", 0.0), 0.0),
            "clip_end_s": _safe_float(obj.get("clip_end_s", 0.0), 0.0),
            "duration_s": _safe_float(obj.get("duration_s", 0.0), 0.0),
            "is_backchannel": _safe_bool(obj.get("is_backchannel", False)),
        }
        metas.append(item)

    metas.sort(key=lambda d: (float(d["clip_start_s"]), str(d["speaker"]), str(d["ipu_id"])))
    return metas


def read_target_mono_from_ipu(ipu_wav: Path, speaker: str) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(ipu_wav), always_2d=True)
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), int(sr)

    if x.shape[1] == 1:
        mono = x[:, 0]
    else:
        if speaker == "A":
            mono = x[:, 0]
        elif speaker == "B":
            mono = x[:, 1]
        else:
            mono = np.mean(x, axis=1)

    mono = np.asarray(mono, dtype=np.float32)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak
    return mono, int(sr)


def _frame_rms(x: np.ndarray, sr: int, win_ms: float = 25.0, hop_ms: float = 10.0) -> Tuple[np.ndarray, int]:
    w = max(1, int(round(sr * win_ms / 1000.0)))
    h = max(1, int(round(sr * hop_ms / 1000.0)))
    if x.size == 0:
        return np.zeros(0, dtype=np.float32), h

    if len(x) < w:
        rms = np.array([np.sqrt(float(np.mean(x * x)) + 1e-12)], dtype=np.float32)
        return rms, h

    n = 1 + max(0, (len(x) - w) // h)
    rms = np.empty(n, dtype=np.float32)
    off = 0
    for i in range(n):
        seg = x[off : off + w]
        rms[i] = np.sqrt(float(np.mean(seg * seg)) + 1e-12)
        off += h
    return rms, h


def _detect_active_frames(x: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    rms, hop = _frame_rms(x, sr)
    if rms.size == 0:
        return np.zeros(0, dtype=bool), hop

    noise = float(np.percentile(rms, 20))
    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med)) + 1e-12)
    thr_hi = max(noise * 3.0, med + 2.5 * mad)
    thr_lo = 0.6 * thr_hi

    active = np.zeros(len(rms), dtype=bool)
    i = 0
    while i < len(rms):
        if rms[i] >= thr_hi:
            j = i + 1
            while j < len(rms) and rms[j] >= thr_lo:
                j += 1
            active[i:j] = True
            i = j
        else:
            i += 1
    return active, hop


def compute_active_speech_seconds(x: np.ndarray, sr: int) -> float:
    if x.size == 0:
        return 0.0

    active, hop = _detect_active_frames(x, sr)
    if active.size == 0:
        return 0.0

    return float(active.sum() * (hop / float(sr)))


def trim_to_last_active_seconds(x: np.ndarray, sr: int, max_active_s: float) -> Tuple[np.ndarray, float]:
    if x.size == 0:
        return x, 0.0
    max_active_s = float(max_active_s)
    if not np.isfinite(max_active_s) or max_active_s <= 0.0:
        return np.zeros(0, dtype=np.float32), 0.0

    active, hop = _detect_active_frames(x, sr)
    if active.size == 0:
        return x, 0.0

    frame_s = hop / float(sr)
    total_active_s = float(active.sum() * frame_s)
    if total_active_s <= max_active_s:
        return x, total_active_s

    need_frames = int(np.ceil(max_active_s / frame_s))
    if need_frames <= 0:
        return np.zeros(0, dtype=np.float32), 0.0

    got = 0
    start_frame = 0
    for i in range(len(active) - 1, -1, -1):
        if active[i]:
            got += 1
            if got >= need_frames:
                start_frame = i
                break

    start_sample = max(0, int(start_frame * hop))
    x_out = x[start_sample:].astype(np.float32)
    active_out_s = float(compute_active_speech_seconds(x_out, sr))
    return x_out, active_out_s


def trim_to_first_active_seconds(x: np.ndarray, sr: int, max_active_s: float) -> Tuple[np.ndarray, float]:
    if x.size == 0:
        return x, 0.0
    max_active_s = float(max_active_s)
    if not np.isfinite(max_active_s) or max_active_s <= 0.0:
        return np.zeros(0, dtype=np.float32), 0.0

    active, hop = _detect_active_frames(x, sr)
    if active.size == 0:
        return x, 0.0

    frame_s = hop / float(sr)
    total_active_s = float(active.sum() * frame_s)
    if total_active_s <= max_active_s:
        return x, total_active_s

    need_frames = int(np.ceil(max_active_s / frame_s))
    if need_frames <= 0:
        return np.zeros(0, dtype=np.float32), 0.0

    got = 0
    end_frame = len(active) - 1
    for i in range(len(active)):
        if active[i]:
            got += 1
            if got >= need_frames:
                end_frame = i
                break

    end_sample = min(len(x), int((end_frame + 1) * hop))
    x_out = x[:end_sample].astype(np.float32)
    active_out_s = float(compute_active_speech_seconds(x_out, sr))
    return x_out, active_out_s


def _load_ipu_assets(ipu_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for m in ipu_rows:
        wav_path = Path(str(m.get("ipu_wav", "")))
        if not wav_path.exists():
            continue
        spk = str(m.get("speaker", "")).strip().upper()
        if spk not in {"A", "B"}:
            continue

        x, sr = read_target_mono_from_ipu(wav_path, spk)
        if x.size == 0:
            continue

        assets.append(
            {
                **m,
                "speaker": spk,
                "waveform": x.astype(np.float32),
                "sr": int(sr),
                "duration_s_audio": float(len(x) / float(sr)),
                "active_speech_s": float(compute_active_speech_seconds(x, int(sr))),
            }
        )

    assets.sort(key=lambda d: float(d.get("clip_start_s", 0.0)))
    return assets


def _segment_stats(assets: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    active_s = float(sum(float(a.get("active_speech_s", 0.0)) for a in assets))
    raw_s = float(sum(float(a.get("duration_s_audio", 0.0)) for a in assets))
    boundary_s = float(max(0, len(assets) - 1) * float(SILENCE_BETWEEN_IPUS_S))
    total_s = raw_s + boundary_s
    return active_s, raw_s, boundary_s, total_s


def _merge_ipu_waveforms(ipus: List[Dict[str, Any]]) -> Tuple[np.ndarray, int, List[str], float]:
    if not ipus:
        return np.zeros(0, dtype=np.float32), 16000, [], 0.0

    sr_ref = int(ipus[0]["sr"])
    gap_n = max(0, int(round(float(SILENCE_BETWEEN_IPUS_S) * float(sr_ref))))

    pieces: List[np.ndarray] = []
    source_ipus: List[str] = []
    active_sum = 0.0
    for i, ipu in enumerate(ipus):
        x = np.asarray(ipu["waveform"], dtype=np.float32)
        ipu_sr = int(ipu["sr"])
        if ipu_sr != sr_ref:
            x = resample_linear(x, ipu_sr, sr_ref)
        if x.size == 0:
            continue
        pieces.append(x)
        source_ipus.append(str(ipu["ipu_id"]))
        active_sum += float(ipu.get("active_speech_s", 0.0))
        if i + 1 < len(ipus) and gap_n > 0:
            pieces.append(np.zeros(gap_n, dtype=np.float32))

    if not pieces:
        return np.zeros(0, dtype=np.float32), sr_ref, source_ipus, float(active_sum)

    merged = np.concatenate(pieces, axis=0).astype(np.float32)
    return merged, sr_ref, source_ipus, float(active_sum)


def _detect_active_frames_with_params(
    x: np.ndarray,
    sr: int,
    win_ms: float,
    hop_ms: float,
) -> Tuple[np.ndarray, int]:
    rms, hop = _frame_rms(x, sr, win_ms=win_ms, hop_ms=hop_ms)
    if rms.size == 0:
        return np.zeros(0, dtype=bool), hop

    noise = float(np.percentile(rms, 20))
    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med)) + 1e-12)
    thr_hi = max(noise * 3.0, med + 2.5 * mad)
    thr_lo = 0.6 * thr_hi

    active = np.zeros(len(rms), dtype=bool)
    i = 0
    while i < len(rms):
        if rms[i] >= thr_hi:
            j = i + 1
            while j < len(rms) and rms[j] >= thr_lo:
                j += 1
            active[i:j] = True
            i = j
        else:
            i += 1
    return active, hop


def _build_joint_vad_masks_from_assets(
    assets: List[Dict[str, Any]],
    win_ms: float = SWITCH_WIN_MS,
    hop_ms: float = SWITCH_HOP_MS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    hop_s = float(hop_ms) / 1000.0
    if not assets or hop_s <= 0.0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), hop_s

    max_end_s = max(float(a.get("clip_end_s", 0.0)) for a in assets)
    n_frames = int(np.ceil(max_end_s / hop_s)) + 2
    if n_frames <= 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), hop_s

    mask_a = np.zeros(n_frames, dtype=bool)
    mask_b = np.zeros(n_frames, dtype=bool)

    for a in assets:
        spk = str(a.get("speaker", "")).strip().upper()
        if spk not in {"A", "B"}:
            continue

        x = np.asarray(a.get("waveform", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        sr = int(a.get("sr", 0))
        start_s = float(a.get("clip_start_s", 0.0))
        end_s = float(a.get("clip_end_s", start_s))
        if sr <= 0 or end_s <= start_s:
            continue

        local_active, local_hop = _detect_active_frames_with_params(x, sr, win_ms=win_ms, hop_ms=hop_ms)
        local_hop_s = float(local_hop) / float(sr)
        dest = mask_a if spk == "A" else mask_b

        if local_active.size == 0 or (not bool(np.any(local_active))):
            g0 = max(0, int(np.floor(start_s / hop_s)))
            g1 = min(len(dest), int(np.ceil(end_s / hop_s)))
            if g1 > g0:
                dest[g0:g1] = True
            continue

        for fi in np.flatnonzero(local_active):
            t_s = start_s + float(fi) * local_hop_s
            gi = int(round(t_s / hop_s))
            if 0 <= gi < len(dest):
                dest[gi] = True

    return mask_a, mask_b, hop_s


def _build_joint_state(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    n = int(min(len(mask_a), len(mask_b)))
    if n <= 0:
        return np.zeros(0, dtype=np.uint8)

    m_a = np.asarray(mask_a[:n], dtype=bool)
    m_b = np.asarray(mask_b[:n], dtype=bool)

    state = np.zeros(n, dtype=np.uint8)
    state[(m_a) & (~m_b)] = 1
    state[(~m_a) & (m_b)] = 2
    state[(m_a) & (m_b)] = 3

    prev = 0
    for i in range(n):
        if state[i] == 3:
            state[i] = prev if prev in (1, 2) else 3
        elif state[i] in (1, 2):
            prev = int(state[i])
    return state


def _state_runs_nonzero(state: np.ndarray) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    i = 0
    n = len(state)
    while i < n:
        if state[i] in (1, 2):
            lab = int(state[i])
            j = i + 1
            while j < n and int(state[j]) == lab:
                j += 1
            out.append((lab, i, j))
            i = j
        else:
            i += 1
    return out


def _label_to_speaker(lab: int) -> Optional[str]:
    if int(lab) == 1:
        return "A"
    if int(lab) == 2:
        return "B"
    return None


def _detect_switch_events_from_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    hop_s: float,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if hop_s <= 0.0:
        return events

    state = _build_joint_state(mask_a, mask_b)
    runs = _state_runs_nonzero(state)
    if len(runs) < 2:
        return events

    overlap = np.logical_and(mask_a[: len(state)], mask_b[: len(state)])
    half_window_frames = max(1, int(round(0.5 / hop_s)))

    for i in range(len(runs) - 1):
        pre_lab, pre_s, pre_e = runs[i]
        post_lab, post_s, post_e = runs[i + 1]
        if pre_lab == post_lab:
            continue

        pre_spk = _label_to_speaker(pre_lab)
        post_spk = _label_to_speaker(post_lab)
        if pre_spk is None or post_spk is None:
            continue

        gap_s = max(0.0, float(post_s - pre_e) * hop_s)
        if gap_s > float(SWITCH_GAP_MAX_S):
            continue

        lo = max(0, min(pre_e, post_s) - half_window_frames)
        hi = min(len(overlap), max(pre_e, post_s) + half_window_frames)
        overlap_s = float(overlap[lo:hi].sum()) * hop_s
        if overlap_s > float(SWITCH_OVERLAP_MAX_S):
            continue

        tb = 0.5 * float(pre_e + post_s) * hop_s
        events.append(
            {
                "pre_spk": pre_spk,
                "post_spk": post_spk,
                "boundary_s": float(tb),
                "pre_run_start_s": float(pre_s) * hop_s,
                "pre_run_end_s": float(pre_e) * hop_s,
                "post_run_start_s": float(post_s) * hop_s,
                "post_run_end_s": float(post_e) * hop_s,
                "boundary_gap_s": float(gap_s),
                "boundary_overlap_s": float(overlap_s),
            }
        )
    return events


def _index_assets_by_speaker(assets: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_spk: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}
    for a in assets:
        spk = str(a.get("speaker", "")).strip().upper()
        if spk in by_spk:
            by_spk[spk].append(a)
    for spk in ("A", "B"):
        by_spk[spk].sort(
            key=lambda d: (
                float(d.get("clip_start_s", 0.0)),
                float(d.get("clip_end_s", 0.0)),
                str(d.get("ipu_id", "")),
            )
        )
    return by_spk


def _find_boundary_anchor_ipus(
    boundary_s: float,
    pre_spk: str,
    post_spk: str,
    by_spk: Dict[str, List[Dict[str, Any]]],
    eps_s: float = SWITCH_BOUNDARY_EPS_S,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    pre_rows = list(by_spk.get(pre_spk, []))
    post_rows = list(by_spk.get(post_spk, []))
    if not pre_rows or not post_rows:
        return None, None

    pre_eligible = [r for r in pre_rows if float(r.get("clip_end_s", 0.0)) <= float(boundary_s) + float(eps_s)]
    if pre_eligible:
        pre_ipu = max(pre_eligible, key=lambda r: float(r.get("clip_end_s", 0.0)))
    else:
        pre_ipu = min(pre_rows, key=lambda r: abs(float(r.get("clip_end_s", 0.0)) - float(boundary_s)))

    post_eligible = [r for r in post_rows if float(r.get("clip_start_s", 0.0)) >= float(boundary_s) - float(eps_s)]
    if post_eligible:
        post_ipu = min(post_eligible, key=lambda r: float(r.get("clip_start_s", 0.0)))
    else:
        post_ipu = min(post_rows, key=lambda r: abs(float(r.get("clip_start_s", 0.0)) - float(boundary_s)))

    return pre_ipu, post_ipu


def _map_switch_events_to_ipus(
    raw_events: List[Dict[str, Any]],
    by_spk: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    for ev in raw_events:
        pre_spk = str(ev.get("pre_spk", "")).strip().upper()
        post_spk = str(ev.get("post_spk", "")).strip().upper()
        if pre_spk not in {"A", "B"} or post_spk not in {"A", "B"}:
            continue
        if pre_spk == post_spk:
            continue

        tb = float(ev.get("boundary_s", 0.0))
        pre_ipu, post_ipu = _find_boundary_anchor_ipus(
            boundary_s=tb,
            pre_spk=pre_spk,
            post_spk=post_spk,
            by_spk=by_spk,
            eps_s=float(SWITCH_BOUNDARY_EPS_S),
        )
        if pre_ipu is None or post_ipu is None:
            continue

        pre_end_s = float(pre_ipu.get("clip_end_s", 0.0))
        post_start_s = float(post_ipu.get("clip_start_s", 0.0))
        ipu_gap_s = max(0.0, post_start_s - pre_end_s)
        ipu_overlap_s = max(0.0, pre_end_s - post_start_s)
        if ipu_gap_s > float(SWITCH_GAP_MAX_S):
            continue
        if ipu_overlap_s > float(SWITCH_OVERLAP_MAX_S):
            continue

        mapped.append(
            {
                **ev,
                "pre_spk": pre_spk,
                "post_spk": post_spk,
                "boundary_s": tb,
                "pre_ipu_id": str(pre_ipu.get("ipu_id", "")),
                "post_ipu_id": str(post_ipu.get("ipu_id", "")),
                "pre_ipu_start_s": float(pre_ipu.get("clip_start_s", 0.0)),
                "pre_ipu_end_s": pre_end_s,
                "post_ipu_start_s": post_start_s,
                "post_ipu_end_s": float(post_ipu.get("clip_end_s", post_start_s)),
                "ipu_gap_s": float(ipu_gap_s),
                "ipu_overlap_s": float(ipu_overlap_s),
            }
        )
    return mapped


def _dedupe_switch_events(
    events: List[Dict[str, Any]],
    dedup_window_s: float = SWITCH_DEDUP_WINDOW_S,
    max_events: int = MAX_SWITCH_EVENTS_PER_CONV,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    last_seen: Dict[Tuple[str, str, str, str], float] = {}
    for ev in sorted(events, key=lambda d: float(d.get("boundary_s", 0.0))):
        key = (
            str(ev.get("pre_spk", "")),
            str(ev.get("post_spk", "")),
            str(ev.get("pre_ipu_id", "")),
            str(ev.get("post_ipu_id", "")),
        )
        tb = float(ev.get("boundary_s", 0.0))
        prev_tb = last_seen.get(key, None)
        if prev_tb is not None and abs(tb - prev_tb) < float(dedup_window_s):
            continue
        kept.append(ev)
        last_seen[key] = tb
        if int(max_events) > 0 and len(kept) >= int(max_events):
            break
    return kept


def _ipu_id_to_index(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, r in enumerate(rows):
        out[str(r.get("ipu_id", ""))] = i
    return out


def _has_contentful_partner_between(
    partner_rows: List[Dict[str, Any]],
    start_s: float,
    end_s: float,
) -> bool:
    if end_s <= start_s:
        return False
    for p in partner_rows:
        p_active = float(p.get("active_speech_s", 0.0))
        if p_active < float(BC_ACTIVE_MAX_S):
            continue
        p_start = float(p.get("clip_start_s", 0.0))
        p_end = float(p.get("clip_end_s", p_start))
        if p_end <= start_s:
            continue
        if p_start >= end_s:
            break
        return True
    return False


def _sum_active(ipus: List[Dict[str, Any]]) -> float:
    return float(sum(float(x.get("active_speech_s", 0.0)) for x in ipus))


def _extend_pre_side(
    pre_rows: List[Dict[str, Any]],
    partner_rows: List[Dict[str, Any]],
    start_idx: int,
    boundary_s: float,
) -> List[Dict[str, Any]]:
    seg = [pre_rows[start_idx]]
    idx = int(start_idx)
    n_ext = 0

    while _sum_active(seg) < float(CTX_MIN_S):
        if n_ext >= int(MAX_EXTEND_IPUS):
            break
        cand_idx = idx - 1
        if cand_idx < 0:
            break
        cand = pre_rows[cand_idx]
        cand_start_s = float(cand.get("clip_start_s", 0.0))
        cand_end_s = float(cand.get("clip_end_s", cand_start_s))
        if float(boundary_s) - cand_start_s > float(MAX_LOOKBACK_WALL_S):
            break
        if _has_contentful_partner_between(
            partner_rows=partner_rows,
            start_s=float(cand_end_s),
            end_s=float(boundary_s),
        ):
            break
        seg.insert(0, cand)
        idx = cand_idx
        n_ext += 1

    return seg


def _extend_post_side(
    post_rows: List[Dict[str, Any]],
    partner_rows: List[Dict[str, Any]],
    start_idx: int,
    boundary_s: float,
) -> List[Dict[str, Any]]:
    seg = [post_rows[start_idx]]
    idx = int(start_idx)
    n_ext = 0

    while _sum_active(seg) < float(TARGET_MIN_S):
        if n_ext >= int(MAX_EXTEND_IPUS):
            break
        cand_idx = idx + 1
        if cand_idx >= len(post_rows):
            break
        cand = post_rows[cand_idx]
        cand_start_s = float(cand.get("clip_start_s", 0.0))
        cand_end_s = float(cand.get("clip_end_s", cand_start_s))
        if cand_end_s - float(boundary_s) > float(MAX_LOOKBACK_WALL_S):
            break
        if _has_contentful_partner_between(
            partner_rows=partner_rows,
            start_s=float(boundary_s),
            end_s=float(cand_start_s),
        ):
            break
        seg.append(cand)
        idx = cand_idx
        n_ext += 1

    return seg


def _build_side_segments_for_switch(
    ev: Dict[str, Any],
    by_spk: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    pre_spk = str(ev.get("pre_spk", "")).strip().upper()
    post_spk = str(ev.get("post_spk", "")).strip().upper()
    if pre_spk not in {"A", "B"} or post_spk not in {"A", "B"}:
        return None

    pre_rows = list(by_spk.get(pre_spk, []))
    post_rows = list(by_spk.get(post_spk, []))
    if not pre_rows or not post_rows:
        return None

    pre_idx_map = _ipu_id_to_index(pre_rows)
    post_idx_map = _ipu_id_to_index(post_rows)

    pre_anchor_id = str(ev.get("pre_ipu_id", ""))
    post_anchor_id = str(ev.get("post_ipu_id", ""))
    if pre_anchor_id not in pre_idx_map or post_anchor_id not in post_idx_map:
        return None

    boundary_s = float(ev.get("boundary_s", 0.0))
    partner_for_pre = by_spk.get(post_spk, [])
    partner_for_post = by_spk.get(pre_spk, [])

    pre_seg = _extend_pre_side(
        pre_rows=pre_rows,
        partner_rows=partner_for_pre,
        start_idx=int(pre_idx_map[pre_anchor_id]),
        boundary_s=boundary_s,
    )
    post_seg = _extend_post_side(
        post_rows=post_rows,
        partner_rows=partner_for_post,
        start_idx=int(post_idx_map[post_anchor_id]),
        boundary_s=boundary_s,
    )
    if not pre_seg or not post_seg:
        return None

    return {
        "pre_spk": pre_spk,
        "post_spk": post_spk,
        "boundary_s": boundary_s,
        "pre_ipus": pre_seg,
        "post_ipus": post_seg,
        "pre_active_s": float(_sum_active(pre_seg)),
        "post_active_s": float(_sum_active(post_seg)),
    }


def _prepare_context_audio_from_ipus(
    ipus: List[Dict[str, Any]],
    trim_mode: str = "last",
) -> Optional[Dict[str, Any]]:
    x, sr, source_ipus, _active_sum = _merge_ipu_waveforms(ipus)
    if x.size == 0:
        return None

    mode = str(trim_mode).strip().lower()
    if mode == "first":
        x_trim, active_trim_s = trim_to_first_active_seconds(x, sr, float(CTX_MAX_S))
    else:
        x_trim, active_trim_s = trim_to_last_active_seconds(x, sr, float(CTX_MAX_S))
    if x_trim.size == 0:
        return None

    active_trim_s = float(active_trim_s)
    if active_trim_s < float(CTX_MIN_S):
        return None

    return {
        "waveform": x_trim.astype(np.float32),
        "sr": int(sr),
        "source_ipus": list(source_ipus),
        "active_s": float(active_trim_s),
    }


def _prepare_target_audio_from_ipus(
    ipus: List[Dict[str, Any]],
    side_role: str,
) -> Optional[Dict[str, Any]]:
    x, sr, source_ipus, _active_sum = _merge_ipu_waveforms(ipus)
    if x.size == 0:
        return None

    active_total_s = float(compute_active_speech_seconds(x, sr))
    x_out = x.astype(np.float32)
    active_out_s = float(active_total_s)
    sliced = False
    slice_mode: Optional[str] = None

    if active_total_s > float(TARGET_MAX_S):
        sliced = True
        role = str(side_role).strip().lower()
        if role == "pre":
            x_out, active_out_s = trim_to_last_active_seconds(x_out, sr, float(TARGET_MAX_S))
            slice_mode = "last15"
        else:
            x_out, active_out_s = trim_to_first_active_seconds(x_out, sr, float(TARGET_MAX_S))
            slice_mode = "first15"

    if x_out.size == 0:
        return None

    active_out_s = float(active_out_s)
    if active_out_s < float(TARGET_MIN_S):
        return None
    if active_out_s > float(TARGET_MAX_S):
        role = str(side_role).strip().lower()
        if role == "pre":
            x_out, active_out_s = trim_to_last_active_seconds(x_out, sr, float(TARGET_MAX_S))
            slice_mode = "last15"
        else:
            x_out, active_out_s = trim_to_first_active_seconds(x_out, sr, float(TARGET_MAX_S))
            slice_mode = "first15"
        sliced = True
        if x_out.size == 0:
            return None
        active_out_s = float(active_out_s)
        if active_out_s > float(TARGET_MAX_S):
            return None

    return {
        "waveform": x_out.astype(np.float32),
        "sr": int(sr),
        "source_ipus": list(source_ipus),
        "active_s": float(active_out_s),
        "active_total_s": float(active_total_s),
        "sliced": bool(sliced),
        "slice_mode": slice_mode,
    }


def _has_contentful_partner_in_gap(
    assets: List[Dict[str, Any]],
    current_idx: int,
    cur_spk: str,
    gap_start_s: float,
    gap_end_s: float,
) -> bool:
    if gap_end_s <= gap_start_s:
        return False
    partner = "B" if cur_spk == "A" else "A"

    for cand in assets[:current_idx]:
        if str(cand.get("speaker", "")) != partner:
            continue
        if float(cand.get("active_speech_s", 0.0)) < float(BC_MIN_ACTIVE_S):
            continue
        c_start = float(cand.get("clip_start_s", 0.0))
        c_end = float(cand.get("clip_end_s", c_start))
        # Any overlap with the open gap counts as partner content inside the gap.
        if c_end <= gap_start_s:
            continue
        if c_start >= gap_end_s:
            continue
        return True
    return False


def _build_turn_runs(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    if not assets:
        return turns

    cur_turn: Optional[Dict[str, Any]] = None

    def _start_turn(ipu: Dict[str, Any]) -> Dict[str, Any]:
        start_s = float(ipu.get("clip_start_s", 0.0))
        end_s = float(ipu.get("clip_end_s", start_s))
        return {
            "speaker": str(ipu["speaker"]),
            "ipus": [ipu],
            "start_s": start_s,
            "end_s": end_s,
            "last_end_s": end_s,
            "active_speech_s": float(ipu.get("active_speech_s", 0.0)),
            "backchannels_between": [],
        }

    def _append_turn(turn: Dict[str, Any], ipu: Dict[str, Any]) -> None:
        start_s = float(ipu.get("clip_start_s", 0.0))
        end_s = float(ipu.get("clip_end_s", start_s))
        turn["ipus"].append(ipu)
        turn["end_s"] = max(float(turn.get("end_s", start_s)), end_s)
        turn["last_end_s"] = max(float(turn.get("last_end_s", start_s)), end_s)
        turn["active_speech_s"] = float(turn.get("active_speech_s", 0.0)) + float(
            ipu.get("active_speech_s", 0.0)
        )

    for idx, ipu in enumerate(assets):
        spk = str(ipu.get("speaker", ""))
        if spk not in {"A", "B"}:
            continue

        if cur_turn is None:
            cur_turn = _start_turn(ipu)
            continue

        ipu_active = float(ipu.get("active_speech_s", 0.0))
        ipu_contentful = ipu_active >= float(BC_MIN_ACTIVE_S)
        ipu_start = float(ipu.get("clip_start_s", 0.0))

        if spk == str(cur_turn["speaker"]):
            gap_s = ipu_start - float(cur_turn.get("last_end_s", ipu_start))
            if gap_s <= float(GAP_FILL_MAX_S):
                _append_turn(cur_turn, ipu)
                continue

            partner_inside_gap = _has_contentful_partner_in_gap(
                assets=assets,
                current_idx=idx,
                cur_spk=spk,
                gap_start_s=float(cur_turn.get("last_end_s", ipu_start)),
                gap_end_s=ipu_start,
            )
            if partner_inside_gap:
                turns.append(cur_turn)
                cur_turn = _start_turn(ipu)
            else:
                _append_turn(cur_turn, ipu)
            continue

        # Different speaker: only contentful partner speech forces a turn boundary.
        if ipu_contentful:
            turns.append(cur_turn)
            cur_turn = _start_turn(ipu)
        else:
            cur_turn["backchannels_between"].append(
                {
                    "ipu_id": str(ipu.get("ipu_id", "")),
                    "speaker": spk,
                    "active_speech_s": ipu_active,
                    "clip_start_s": float(ipu.get("clip_start_s", 0.0)),
                    "clip_end_s": float(ipu.get("clip_end_s", 0.0)),
                }
            )

    if cur_turn is not None:
        turns.append(cur_turn)

    for turn_idx, t in enumerate(turns, start=1):
        t["turn_id"] = f"turn_{turn_idx:05d}_spk{t['speaker']}"
    return turns


def _select_context_turns(turns: List[Dict[str, Any]], target_turn_idx: int) -> Optional[List[Dict[str, Any]]]:
    if target_turn_idx <= 0:
        return None
    partner_turn = turns[target_turn_idx - 1]
    context_turns = [partner_turn]

    partner_active = float(partner_turn.get("active_speech_s", 0.0))
    if partner_active >= float(BC_MIN_ACTIVE_S):
        return context_turns

    target_turn = turns[target_turn_idx]
    target_start_s = float(target_turn.get("start_s", 0.0))
    partner_spk = str(partner_turn.get("speaker", ""))

    # If immediate partner turn is too short, prepend one earlier partner turn when local.
    for j in range(target_turn_idx - 2, -1, -1):
        cand = turns[j]
        if str(cand.get("speaker", "")) != partner_spk:
            continue
        cand_end_s = float(cand.get("end_s", 0.0))
        if target_start_s - cand_end_s > float(CONTEXT_MAX_LOOKBACK_WALL_S):
            break
        context_turns.insert(0, cand)
        break

    return context_turns


def cleanup_previous_outputs(out_dir: Path, base_prefix: str) -> None:
    if not out_dir.exists():
        return
    for p in out_dir.glob(f"{base_prefix}*"):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


def write_single_inputs(
    ipu_meta: List[Dict[str, Any]],
    out_dir: Path,
    base_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Category 1 input construction (single-speaker, no-interlocutor context).

    Rules:
      - Exclude backchannels by default (is_backchannel=True).
      - Exclude any single IPU whose active speech exceeds SINGLE_TARGET_ACTIVE_MAX_S.
      - Merge remaining IPUs in chronological order into multiple segments.
      - Insert a fixed SILENCE_BETWEEN_IPUS_S between adjacent IPUs inside a segment.
      - Enforce:
          * For every segment except the final segment: active speech must be in
            [SINGLE_TARGET_ACTIVE_MIN_S, SINGLE_TARGET_ACTIVE_MAX_S].
          * The final segment may be shorter than SINGLE_TARGET_ACTIVE_MIN_S, but will be
            dropped if it is extremely short (active speech < SINGLE_TAIL_MIN_KEEP_S),
            unless it is the only segment kept for that speaker.
      - Stop growing a segment when adding the next IPU would exceed:
          * SINGLE_TARGET_ACTIVE_MAX_S (active speech), or
          * SINGLE_MAX_BOUNDARY_FRACTION (boundary time / total duration).
    """
    entries: List[Dict[str, Any]] = []

    for spk in ("A", "B"):
        spk_tag = f"spk{spk}"

        # Exclude backchannels at the meta level for Category 1.
        rows = [m for m in ipu_meta if m["speaker"] == spk and not bool(m["is_backchannel"])]
        assets_all = _load_ipu_assets(rows)
        if not assets_all:
            continue

        # Exclude overly-long single IPUs (active speech > max).
        assets = [
            a
            for a in assets_all
            if float(a.get("active_speech_s", 0.0)) <= float(SINGLE_TARGET_ACTIVE_MAX_S)
        ]
        if not assets:
            continue

        # First pass: greedy segmentation by max-active and max-boundary-fraction.
        segments: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []

        for asset in assets:
            if not cur:
                cur = [asset]
                continue

            cand = cur + [asset]
            cand_active_s, _cand_raw_s, cand_boundary_s, cand_total_s = _segment_stats(cand)
            cand_boundary_frac = (cand_boundary_s / cand_total_s) if cand_total_s > 0.0 else 0.0

            exceeds_active = cand_active_s > float(SINGLE_TARGET_ACTIVE_MAX_S)
            exceeds_boundary = cand_boundary_frac > float(SINGLE_MAX_BOUNDARY_FRACTION)

            if exceeds_active or exceeds_boundary:
                segments.append(cur)
                cur = [asset]
            else:
                cur = cand

        if cur:
            segments.append(cur)

        if not segments:
            continue

        # Second pass: enforce the min-active requirement for all but the final segment.
        kept_segments: List[List[Dict[str, Any]]] = []
        for seg_i, seg in enumerate(segments):
            if not seg:
                continue
            seg_active_s, _seg_raw_s, _seg_boundary_s, _seg_total_s = _segment_stats(seg)
            is_last = (seg_i == len(segments) - 1)

            if not is_last:
                # Strict: internal segments must hit the target min.
                if seg_active_s < float(SINGLE_TARGET_ACTIVE_MIN_S):
                    continue
                kept_segments.append(seg)
            else:
                # Tail segment: allow shorter, but avoid extremely small tails unless nothing else exists.
                if seg_active_s <= 0.0:
                    continue
                if (seg_active_s < float(SINGLE_TAIL_MIN_KEEP_S)) and kept_segments:
                    continue
                kept_segments.append(seg)

        if not kept_segments:
            # Nothing met the constraints; as a fallback, keep the final original segment if it has any speech.
            last_seg = segments[-1]
            last_active_s, *_ = _segment_stats(last_seg)
            if last_active_s > 0.0:
                kept_segments = [last_seg]
            else:
                continue

        # Materialize kept segments and write manifest entries.
        kept_idx = 0
        for seg in kept_segments:
            kept_idx += 1

            sr_ref = int(seg[0]["sr"])
            gap_n = max(0, int(round(float(SILENCE_BETWEEN_IPUS_S) * float(sr_ref))))

            valid_clips: List[np.ndarray] = []
            source_ipus: List[str] = []
            seg_active_sum = 0.0

            for a in seg:
                x = np.asarray(a["waveform"], dtype=np.float32)
                a_sr = int(a["sr"])
                if a_sr != sr_ref:
                    x = resample_linear(x, a_sr, sr_ref)
                if x.size == 0:
                    continue
                valid_clips.append(x)
                source_ipus.append(str(a["ipu_id"]))
                seg_active_sum += float(a.get("active_speech_s", 0.0))

            if not valid_clips:
                continue

            pieces: List[np.ndarray] = []
            for i, x in enumerate(valid_clips):
                pieces.append(x)
                if i + 1 < len(valid_clips) and gap_n > 0:
                    pieces.append(np.zeros(gap_n, dtype=np.float32))

            merged = np.concatenate(pieces, axis=0).astype(np.float32)
            total_s = float(len(merged) / float(sr_ref))
            boundary_s = float(max(0, len(source_ipus) - 1) * float(SILENCE_BETWEEN_IPUS_S))
            boundary_frac = (boundary_s / total_s) if total_s > 0.0 else 0.0

            # Use the segment's summed active speech (consistent with segmentation decisions).
            active_s = float(seg_active_sum)

            out_name = f"{base_prefix}_{spk_tag}_single_{kept_idx:03d}.wav"
            out_path = out_dir / out_name
            sf.write(str(out_path), merged, sr_ref, subtype="PCM_16")

            entries.append(
                {
                    "type": "single",
                    "target_spk": spk_tag,
                    "audio": out_name,
                    "source_ipus": source_ipus,
                    "active_speech_s": active_s,
                    "segment_duration_s": total_s,
                    "boundary_s": boundary_s,
                    "boundary_fraction": boundary_frac,
                    "under_target_min_30s": bool(active_s < float(SINGLE_TARGET_ACTIVE_MIN_S)),
                    "under_keep_threshold_15s": bool(active_s < float(SINGLE_TAIL_MIN_KEEP_S)),
                }
            )

    return entries


def write_interaction_inputs(
    ipu_meta: List[Dict[str, Any]],
    out_dir: Path,
    base_prefix: str,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    assets = _load_ipu_assets(ipu_meta)
    if not assets:
        return entries

    mask_a, mask_b, hop_s = _build_joint_vad_masks_from_assets(
        assets,
        win_ms=float(SWITCH_WIN_MS),
        hop_ms=float(SWITCH_HOP_MS),
    )
    raw_switches = _detect_switch_events_from_masks(mask_a, mask_b, hop_s)
    if not raw_switches:
        return entries

    by_spk = _index_assets_by_speaker(assets)
    mapped_switches = _map_switch_events_to_ipus(raw_switches, by_spk)
    switches = _dedupe_switch_events(
        mapped_switches,
        dedup_window_s=float(SWITCH_DEDUP_WINDOW_S),
        max_events=int(MAX_SWITCH_EVENTS_PER_CONV),
    )
    if not switches:
        return entries

    pair_idx = 0
    for switch_idx, ev in enumerate(switches, start=1):
        side_pack = _build_side_segments_for_switch(ev, by_spk)
        if side_pack is None:
            continue

        pre_ipus = list(side_pack["pre_ipus"])
        post_ipus = list(side_pack["post_ipus"])
        boundary_s = float(side_pack["boundary_s"])

        pre_target = _prepare_target_audio_from_ipus(pre_ipus, side_role="pre")
        post_target = _prepare_target_audio_from_ipus(post_ipus, side_role="post")

        # Emit both directions only when both sides are valid as targets.
        emit_post_target = post_target is not None
        emit_pre_target = pre_target is not None
        if not emit_post_target and not emit_pre_target:
            continue

        def _emit_item(
            *,
            direction: str,
            target_spk: str,
            context_side_ipus: List[Dict[str, Any]],
            target_side_ipus: List[Dict[str, Any]],
            target_pack: Dict[str, Any],
        ) -> None:
            nonlocal pair_idx

            context_trim_mode = "first" if direction == "pre_target" else "last"
            context_pack = _prepare_context_audio_from_ipus(
                context_side_ipus,
                trim_mode=context_trim_mode,
            )
            if context_pack is None:
                return

            ctx_x = np.asarray(context_pack["waveform"], dtype=np.float32)
            ctx_sr = int(context_pack["sr"])
            ctx_active_s = float(context_pack["active_s"])
            ctx_source_ipus = list(context_pack["source_ipus"])

            tgt_x = np.asarray(target_pack["waveform"], dtype=np.float32)
            tgt_sr = int(target_pack["sr"])
            tgt_active_s = float(target_pack["active_s"])
            tgt_source_ipus = list(target_pack["source_ipus"])

            if tgt_sr != ctx_sr:
                tgt_x = resample_linear(tgt_x, tgt_sr, ctx_sr)
                tgt_sr = ctx_sr
                tgt_active_s = float(compute_active_speech_seconds(tgt_x, tgt_sr))
                if tgt_active_s < float(TARGET_MIN_S) or tgt_active_s > float(TARGET_MAX_S):
                    return

            context_start_s = min(float(x.get("clip_start_s", 0.0)) for x in context_side_ipus)
            context_end_s = max(float(x.get("clip_end_s", 0.0)) for x in context_side_ipus)
            target_start_s = min(float(x.get("clip_start_s", 0.0)) for x in target_side_ipus)
            target_end_s = max(float(x.get("clip_end_s", 0.0)) for x in target_side_ipus)

            pair_idx += 1
            spk_tag = f"spk{target_spk}"
            ctx_name = f"{base_prefix}_{spk_tag}_interaction_{pair_idx:03d}_ctx.wav"
            tgt_name = f"{base_prefix}_{spk_tag}_interaction_{pair_idx:03d}_tgt.wav"

            sf.write(str(out_dir / ctx_name), ctx_x, int(ctx_sr), subtype="PCM_16")
            sf.write(str(out_dir / tgt_name), tgt_x, int(tgt_sr), subtype="PCM_16")

            context_id = "+".join(ctx_source_ipus)
            target_id = "+".join(tgt_source_ipus)
            if direction == "pre_target":
                gap_wall_s = max(0.0, float(context_start_s - target_end_s))
            else:
                gap_wall_s = max(0.0, float(target_start_s - context_end_s))

            entries.append(
                {
                    "type": "interaction",
                    "target_spk": spk_tag,
                    "context_audio": ctx_name,
                    "target_audio": tgt_name,
                    "target_position": 1 if direction == "pre_target" else 2,
                    "context_turn_id": context_id,
                    "target_turn_id": target_id,
                    "context_active_speech_s": float(ctx_active_s),
                    "target_active_speech_s": float(tgt_active_s),
                    "gap_wall_s": float(gap_wall_s),
                    "target_turn_active_total": float(target_pack["active_total_s"]),
                    "sliced": bool(target_pack["sliced"]),
                    "target_turn_slice": target_pack["slice_mode"],
                    "context_source_ipus": ctx_source_ipus,
                    "target_source_ipus": tgt_source_ipus,
                    "direction": direction,
                    "switch_index": int(switch_idx),
                    "switch_boundary_s": float(boundary_s),
                    "switch_pre_spk": str(ev.get("pre_spk", "")),
                    "switch_post_spk": str(ev.get("post_spk", "")),
                    "switch_pre_ipu_id": str(ev.get("pre_ipu_id", "")),
                    "switch_post_ipu_id": str(ev.get("post_ipu_id", "")),
                    "switch_ipu_gap_s": float(ev.get("ipu_gap_s", 0.0)),
                    "switch_ipu_overlap_s": float(ev.get("ipu_overlap_s", 0.0)),
                    "context_span_start_s": float(context_start_s),
                    "context_span_end_s": float(context_end_s),
                    "target_span_start_s": float(target_start_s),
                    "target_span_end_s": float(target_end_s),
                    "pre_side_active_s": float(side_pack["pre_active_s"]),
                    "post_side_active_s": float(side_pack["post_active_s"]),
                    # Legacy keys retained for downstream compatibility.
                    "context_ipu_id": context_id,
                    "target_ipu_id": target_id,
                    "context_active_s": float(ctx_active_s),
                    "target_active_s": float(tgt_active_s),
                }
            )

        if emit_post_target:
            _emit_item(
                direction="post_target",
                target_spk=str(side_pack["post_spk"]),
                context_side_ipus=pre_ipus,
                target_side_ipus=post_ipus,
                target_pack=post_target,
            )

        if emit_pre_target:
            _emit_item(
                direction="pre_target",
                target_spk=str(side_pack["pre_spk"]),
                context_side_ipus=post_ipus,
                target_side_ipus=pre_ipus,
                target_pack=pre_target,
            )

    return entries


def write_manifest(manifest_path: Path, entries: List[Dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    prompt_id_unique = str(args.prompt_id_unique)
    a_id = str(args.a_id)
    b_id = str(args.b_id)
    input_mode = str(args.input_mode)

    base_prefix = f"{prompt_id_unique}_A-{a_id}_B-{b_id}"
    run_out_dir = Path.cwd() / f"{prompt_id_unique}_full_turns"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    cleanup_previous_outputs(run_out_dir, base_prefix)

    with tempfile.TemporaryDirectory(prefix="ipu_eval_inputs_") as td:
        temp_root = Path(td)
        temp_out_dir = run_ipu_extractor_temp(
            extractor_script=args.extractor_script,
            temp_workdir=temp_root,
            prompt_id_unique=prompt_id_unique,
            a_id=a_id,
            b_id=b_id,
            dataset_root=str(args.dataset_root),
            dyad_lookup_csv=str(args.dyad_lookup_csv),
            interactions_role_abmapped_csv=str(args.interactions_role_abmapped_csv),
        )

        if not temp_out_dir.exists():
            raise SystemExit(f"IPU extractor output directory not found: {temp_out_dir}")

        ipu_meta = load_ipu_metas(temp_out_dir, base_prefix)
        if not ipu_meta:
            print(f"[warning] No IPU metadata found for base prefix: {base_prefix}", file=sys.stderr)

        if input_mode == "single":
            entries = write_single_inputs(ipu_meta, run_out_dir, base_prefix)
        else:
            entries = write_interaction_inputs(ipu_meta, run_out_dir, base_prefix)

    manifest_path = run_out_dir / f"{base_prefix}__manifest.jsonl"
    write_manifest(manifest_path, entries)

    print(f"[info] input_mode={input_mode}")
    print(f"[info] wrote {len(entries)} manifest item(s)")
    print(f"[info] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
