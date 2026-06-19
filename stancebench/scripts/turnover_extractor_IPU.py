#!/usr/bin/env python3
"""\
IPU extractor for a single Seamless Interaction dyad.

Edits from the original turnover extractor:

- Extract per-speaker Inter-Pausal Units (IPUs) from each channel independently.
  An IPU boundary occurs when silence gap >= IPU_PAUSE_S. Gaps < IPU_PAUSE_S are merged.
- Save audio segments in the same directory structure as before:
    <cwd>/<PROMPT_ID_UNIQUE>_full_turns/
  (Option A: choose your CWD externally; keep OUTPUT_DIR=None.)
- Naming: replace AtoB/BtoA with per-speaker tags: _spkA_ / _spkB_.
- Backchannels: keep very short IPUs, but mark them in the filename with
  suffix _is_backchannel if duration_s < IPU_MIN_S.

Only the two WAVs referenced by the selected dyad_lookup row are used.
"""

from pathlib import Path
import os

# ----------------------------
# CONFIG
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
METADATA_DIR = REPO_ROOT / "metadata"

PROMPT_ID_UNIQUE = "P3355_v3.2.3.RP_AMCN_ANCM"
DATASET_ROOT = os.environ.get("SEAMLESS_DATASET_ROOT", "")
DYAD_LOOKUP_CSV = os.environ.get("SEAMLESS_DYAD_LOOKUP_CSV", "")

# AB-mapped interactions file (one row per dyad conversation)
INTERACTIONS_ROLE_ABMAPPED_CSV = os.environ.get(
    "STANCEBENCH_INTERACTIONS_ROLE_ABMAPPED_CSV",
    str(METADATA_DIR / "interactions_role_ABmapped.csv"),
)

ROW_PICK         = "longest"   # or "p=1114A" (unused when using AB-mapped mapping)

# Optional filters to target a single conversation under PROMPT_ID_UNIQUE
# If both are None: process all conversations for this prompt.
# If both are non-None: process only the row with this (a_id, b_id).
A_ID             = None        # e.g. "V00_S1047_I00000634_P0092A"
B_ID             = None        # e.g. "V00_S1047_I00000634_P0799A"

OUTPUT_DIR       = None        # None -> "<cwd>/<PROMPT_ID_UNIQUE>_full_turns"

# VAD
WIN_MS           = 25.0
HOP_MS           = 10.0

# IPU segmentation
IPU_PAUSE_S      = 0.30  # silence gap >= this splits IPUs; gaps < this are merged
IPU_MIN_S        = 0.20  # duration_s < this => mark filename with _is_backchannel
# ----------------------------

import pandas as pd
import numpy as np
import re, struct, math, json
import argparse  # NEW

P_PAT = re.compile(r"_P([\w\-]+)\.wav$", re.IGNORECASE)

# ---------- WAV I/O ----------
def _have_soundfile():
    try:
        import soundfile as sf  # noqa: F401
        return True
    except Exception:
        return False

def _pcm24_bytes_to_float(x_bytes, nchan):
    b = np.frombuffer(x_bytes, dtype=np.uint8)
    if len(b) % (3*nchan) != 0:
        nframes = len(b) // (3*nchan)
        b = b[: nframes * 3 * nchan]
    arr = b.reshape(-1, nchan, 3).astype(np.uint32)
    val = (arr[...,0] | (arr[...,1] << 8) | (arr[...,2] << 16)).astype(np.int32)
    neg = (val & 0x800000) != 0
    val[neg] -= 1 << 24
    return (val / 8388608.0).astype(np.float32)

def _read_wav_native(path):
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[0:4] not in (b"RIFF", b"RIFX") or header[8:12] != b"WAVE":
            raise RuntimeError(f"{path} is not a RIFF/WAVE file")
        if header[0:4] == b"RIFX":
            raise RuntimeError("Big-endian RIFX not supported")

        fmt = None; data_bytes = None; nchan = sr = bps = None; wformat = None
        while True:
            h = f.read(8)
            if len(h) < 8: break
            cid, csz = h[:4], int.from_bytes(h[4:8], "little")
            cdata = f.read(csz)
            if csz % 2 == 1: f.seek(1, 1)
            if cid == b"fmt ":
                fmt = cdata
                if len(fmt) < 16: raise RuntimeError("fmt chunk too small")
                wformat = int.from_bytes(fmt[0:2], "little")
                nchan   = int.from_bytes(fmt[2:4], "little")
                sr      = int.from_bytes(fmt[4:8], "little")
                bps     = int.from_bytes(fmt[14:16], "little")
                if wformat == 0xFFFE and len(fmt) >= 40:
                    subformat_tag = int.from_bytes(fmt[24:28], "little")
                    if subformat_tag in (1, 3): wformat = subformat_tag
            elif cid == b"data":
                data_bytes = cdata

        if fmt is None or data_bytes is None:
            raise RuntimeError("Missing fmt or data chunk")

        if wformat == 1:  # PCM
            if bps == 8:
                x = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float32); x = (x - 128.0) / 128.0
            elif bps == 16:
                x = np.frombuffer(data_bytes, dtype="<i2").astype(np.float32) / 32768.0
            elif bps == 24:
                x = _pcm24_bytes_to_float(data_bytes, nchan).reshape(-1, nchan)
            elif bps == 32:
                x = np.frombuffer(data_bytes, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                raise RuntimeError(f"Unsupported PCM bits-per-sample: {bps}")
            if x.ndim == 1:
                frame_size = nchan * (bps // 8 if bps != 24 else 3)
                nframes = len(data_bytes) // frame_size
                x = x.reshape(nframes, nchan)
        elif wformat == 3:  # IEEE float
            if bps == 32:
                x = np.frombuffer(data_bytes, dtype="<f4").reshape(-1, nchan)
            elif bps == 64:
                x = np.frombuffer(data_bytes, dtype="<f8").astype(np.float32).reshape(-1, nchan)
            else:
                raise RuntimeError(f"Unsupported IEEE float bits-per-sample: {bps}")
        else:
            raise RuntimeError(f"Unsupported format tag: {wformat}")

        if x.ndim == 2 and x.shape[1] > 1: x = x.mean(axis=1)
        return x.astype(np.float32), int(sr)

def read_wav_mono(path):
    if _have_soundfile():
        import soundfile as sf
        x, sr = sf.read(str(path), always_2d=True)
        x = x.mean(axis=1).astype(np.float32)
        m = float(np.max(np.abs(x))) if x.size else 1.0
        if m > 1.0: x = x / m
        return x, int(sr)
    x, sr = _read_wav_native(path)
    if x.ndim == 2: x = x.mean(axis=1)
    x = x.astype(np.float32)
    m = float(np.max(np.abs(x))) if x.size else 1.0
    if m > 1.0: x = x / m
    return x, int(sr)

def write_stereo(path, L, R, sr):
    L = np.asarray(L, dtype=np.float64); R = np.asarray(R, dtype=np.float64)
    peak = max(1e-9, float(np.max(np.abs([L, R]))))
    if peak > 1.0: L /= peak * 1.05; R /= peak * 1.05
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    if _have_soundfile():
        import soundfile as sf
        stereo = np.column_stack([L, R]).astype(np.float32)
        sf.write(str(out), stereo, int(sr), subtype="PCM_16"); return
    import wave
    stereo = np.column_stack([L, R]).astype(np.float64); stereo = np.clip(stereo, -1.0, 1.0)
    stereo_i16 = (stereo * 32767.0).astype("<i2").ravel()
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(int(sr)); w.writeframes(stereo_i16.tobytes())

def resample_linear(x, sr_in, sr_out):
    if sr_in == sr_out or x.size == 0: return x
    ratio = float(sr_out) / float(sr_in)
    n_out = int(round(len(x) * ratio))
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


# ---------- Transcript helpers (dataset-provided JSON) ----------
# Cache transcript JSON parses to avoid re-reading for every clip.
_TRANSCRIPT_CACHE = {}

def _wav_path_to_transcript_json(wav_path):
    """
    Convert a dataset .wav path to the corresponding transcript .json path.

    Convention:
      /path/to/FILE.wav  ->  /path/to/FILE.json
    """
    return Path(str(wav_path)).with_suffix(".json")

def _load_dataset_transcript_segments(json_path):
    """
    Load dataset transcript JSON and return the segment list.

    Expected (per dataset convention):
      {
        "metadata:transcript": [
            {"start": <float>, "end": <float>, "transcript": <str>, "words": [...]},
            ...
        ],
        ...
      }
    """
    key = str(json_path)
    if key in _TRANSCRIPT_CACHE:
        return _TRANSCRIPT_CACHE[key]

    jp = Path(str(json_path))
    if not jp.exists():
        raise FileNotFoundError(f"Transcript JSON not found: {jp}")

    obj = json.loads(jp.read_text(encoding="utf-8"))
    segs = obj.get("metadata:transcript") if isinstance(obj, dict) and "metadata:transcript" in obj else obj
    if not isinstance(segs, list):
        raise ValueError(f"Unexpected transcript JSON structure at {jp}")

    # Best-effort: ensure segments are sorted by start time if present.
    try:
        segs = sorted(segs, key=lambda d: float(d.get("start", 0.0)))
    except Exception:
        pass

    _TRANSCRIPT_CACHE[key] = segs
    return segs

def _extract_words_in_window(segments, start_s, end_s):
    """
    Extract word-level tokens overlapping [start_s, end_s) and join into a transcript string.
    """
    try:
        s0 = float(start_s)
        s1 = float(end_s)
    except Exception:
        return ""

    if not np.isfinite(s0) or not np.isfinite(s1) or s1 <= s0:
        return ""

    words = []
    for seg in segments:
        # Segment-level pruning (if available)
        try:
            seg_start = float(seg.get("start", -1e9))
            seg_end = float(seg.get("end", 1e9))
            if seg_end <= s0 or seg_start >= s1:
                continue
        except Exception:
            pass

        wlist = seg.get("words", None)
        if isinstance(wlist, list) and wlist:
            for w in wlist:
                try:
                    ws = float(w.get("start", -1e9))
                    we = float(w.get("end", -1e9))
                except Exception:
                    continue
                if we <= s0 or ws >= s1:
                    continue
                token = str(w.get("word", "")).strip()
                if token:
                    words.append(token)
        else:
            # Fallback: if no word list, include whole segment transcript if overlapping.
            t = str(seg.get("transcript", "")).strip()
            if t:
                words.append(t)

    out = " ".join(words)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# ---------- CSV ----------
def load_interaction_row(csv_path, prompt_value):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    for c in ("prompt_ad_unique","prompt_id_unique","prompt_id"):
        if c in df.columns:
            hit = df[df[c].eq(prompt_value)]
            if len(hit) == 1: return hit.iloc[0]
    mask = df.apply(lambda col: col.eq(prompt_value) if col.dtype == "object" else False).any(axis=1)
    hits = df[mask]
    if len(hits) != 1: raise SystemExit(f"Expected one row for '{prompt_value}', found {len(hits)} in {csv_path}")
    return hits.iloc[0]

def extract_I_from_row(row):
    if "prompt_hash" in row.index and isinstance(row["prompt_hash"], str) and re.fullmatch(r"\d{8}", row["prompt_hash"]):
        return f"I{row['prompt_hash']}"
    for v in row.values:
        if isinstance(v, str) and re.fullmatch(r"\d{8}", v): return f"I{v}"
    raise SystemExit("Could not find 8-digit prompt_hash in interactions.csv row.")

def parse_p(pathlike):
    m = P_PAT.search(Path(pathlike).name)
    return m.group(1) if m else None

def choose_pair_from_lookup(dyad_csv, I_tag, pick, root):
    df = pd.read_csv(dyad_csv, dtype=str, low_memory=False)
    need = {"participant1_relpath","participant2_relpath"}
    if not need.issubset(df.columns): raise SystemExit(f"{dyad_csv} must contain columns {sorted(need)}")
    m = df["participant1_relpath"].astype(str).str.contains(f"_{I_tag}_") | \
        df["participant2_relpath"].astype(str).str.contains(f"_{I_tag}_")
    rows = df[m].copy()
    if rows.empty: raise SystemExit(f"No dyad rows reference {I_tag} in {dyad_csv}")

    rows["P1"] = rows["participant1_relpath"].map(parse_p)
    rows["P2"] = rows["participant2_relpath"].map(parse_p)

    if pick.startswith("p="):
        target = pick.split("=",1)[1]
        sub = rows[(rows["P1"] == target) | (rows["P2"] == target)]
        if sub.empty: raise SystemExit(f"No dyad row with P{target} at {I_tag}.")
        r = sub.iloc[0]
        return (Path(root)/r["participant1_relpath"]).resolve(), (Path(root)/r["participant2_relpath"]).resolve()

    if pick != "longest": raise SystemExit("ROW_PICK must be 'longest' or 'p=<id>'")

    best = None; best_bytes = -1
    for _, r in rows.iterrows():
        p1 = (Path(root)/r["participant1_relpath"]).resolve()
        p2 = (Path(root)/r["participant2_relpath"]).resolve()
        if not (p1.exists() and p2.exists()): continue
        sz = p1.stat().st_size + p2.stat().st_size
        if sz > best_bytes: best = (p1, p2); best_bytes = sz
    if best is None: raise SystemExit("No existing pair found on disk for this I. Check DATASET_ROOT or relpaths.")
    return best

# Load all conversations for a prompt (and optionally a single a_id/b_id)
def load_abmapped_rows(csv_path, prompt_id_unique, a_id=None, b_id=None):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    if "prompt_id_unique" not in df.columns:
        raise SystemExit(f"{csv_path} must contain column 'prompt_id_unique'")
    base = df[df["prompt_id_unique"] == prompt_id_unique]
    if base.empty:
        raise SystemExit(
            f"No rows for prompt_id_unique='{prompt_id_unique}' in {csv_path}"
        )
    if a_id is not None and b_id is not None:
        if "a_id" not in base.columns or "b_id" not in base.columns:
            raise SystemExit(f"{csv_path} must contain columns 'a_id' and 'b_id'")
        sub = base[(base["a_id"] == a_id) & (base["b_id"] == b_id)]
        if sub.empty:
            raise SystemExit(
                f"No rows for prompt_id_unique='{prompt_id_unique}' with "
                f"a_id='{a_id}' and b_id='{b_id}' in {csv_path}"
            )
        return sub
    return base

# Map (a_id, b_id) to WAV paths, ensuring A->left, B->right
def choose_pair_from_abmapped(dyad_df, a_id, b_id, root):
    need = {"participant1_relpath", "participant2_relpath"}
    if not need.issubset(dyad_df.columns):
        raise SystemExit(f"dyad_lookup must contain columns {sorted(need)}")

    p1 = dyad_df["participant1_relpath"].astype(str)
    p2 = dyad_df["participant2_relpath"].astype(str)

    mask = (
        (p1.str.contains(a_id, regex=False) & p2.str.contains(b_id, regex=False)) |
        (p1.str.contains(b_id, regex=False) & p2.str.contains(a_id, regex=False))
    )
    rows = dyad_df[mask]
    if rows.empty:
        raise SystemExit(
            f"No dyad row found for a_id='{a_id}' and b_id='{b_id}' in dyad_lookup."
        )

    r = rows.iloc[0]
    rel1 = str(r["participant1_relpath"])
    rel2 = str(r["participant2_relpath"])

    # Ensure left channel is Speaker A and right is Speaker B
    if a_id in rel1 and b_id in rel2:
        pA = (Path(root) / rel1).resolve()
        pB = (Path(root) / rel2).resolve()
    elif a_id in rel2 and b_id in rel1:
        pA = (Path(root) / rel2).resolve()
        pB = (Path(root) / rel1).resolve()
    else:
        raise SystemExit(
            "Found dyad row for given a_id/b_id but could not assign channels "
            f"cleanly (relpaths: '{rel1}', '{rel2}')."
        )

    return pA, pB

# ---------- VAD and helpers ----------
def frame_rms(x, sr, win_ms, hop_ms):
    w = int(round(sr * win_ms/1000.0)); h = int(round(sr * hop_ms/1000.0))
    if w <= 0: w = 1
    if h <= 0: h = 1
    n = 1 + max(0, (len(x) - w) // h)
    rms = np.empty(n, dtype=np.float32); off = 0
    for i in range(n):
        seg = x[off:off+w]; rms[i] = math.sqrt(float(np.mean(seg*seg)) + 1e-12); off += h
    t = (np.arange(n)*h + w/2.0)/sr
    return rms, t, h, w

def hysteresis_vad(x, sr, win_ms=25.0, hop_ms=10.0):
    rms, t, hop, win = frame_rms(x, sr, win_ms, hop_ms)
    n = len(rms)
    if n == 0: return np.zeros(0, dtype=bool), t, hop, win
    noise = np.percentile(rms, 20)
    med = np.median(rms); mad = np.median(np.abs(rms - med)) + 1e-12
    thr_hi = max(noise*3.0, med + 2.5*mad); thr_lo = 0.6 * thr_hi
    active = np.zeros(n, dtype=bool); i = 0
    while i < n:
        if rms[i] >= thr_hi:
            j = i + 1
            while j < n and rms[j] >= thr_lo: j += 1
            active[i:j] = True; i = j
        else:
            i += 1
    return active, t, hop, win

def runs(mask):
    on = np.flatnonzero(mask)
    if on.size == 0: return []
    spans = []; s = on[0]; p = on[0]
    for k in on[1:]:
        if k == p + 1: p = k
        else: spans.append((s, p+1)); s = p = k
    spans.append((s, p+1))
    return spans

def drop_short_true(mask, min_frames):
    if min_frames <= 1: return mask.copy()
    out = mask.copy()
    for s,e in runs(mask):
        if (e - s) < min_frames: out[s:e] = False
    return out

def b_has_long_run_in(span, maskB, min_frames):
    s,e = span
    for sb, eb in runs(maskB):
        if eb <= s: continue
        if sb >= e: break
        inter_s = max(s, sb); inter_e = min(e, eb)
        if inter_e > inter_s and (inter_e - inter_s) >= min_frames: return True
    return False

def fill_gaps_same_speaker(maskA, maskB, backchan_frames):
    out = maskA.copy(); rA = runs(out)
    for i in range(len(rA)-1):
        s1, e1 = rA[i]; s2, e2 = rA[i+1]; gap = (e1, s2)
        if gap[1] <= gap[0]: continue
        if not b_has_long_run_in(gap, maskB, backchan_frames):
            out[e1:s2] = True
    return out

def count_true(mask, s, e): return int(mask[int(s):int(e)].sum())
def sec_from_frames(frames, hop, sr): return frames * (hop / sr)
def frames_from_sec(sec, hop, sr): return int(round(sec * sr / hop))

def frames_from_sec_floor(sec, hop, sr):
    """Convert seconds to frame count using floor, for threshold comparisons.

    We use floor here so the comparison "gap >= IPU_PAUSE_S" is implemented as
    "gap_frames >= floor(IPU_PAUSE_S * sr / hop)".
    """
    try:
        s = float(sec)
    except Exception:
        return 0
    if not np.isfinite(s) or s <= 0.0:
        return 0
    return int(math.floor(s * float(sr) / float(hop) + 1e-9))


def fill_gaps_max(mask, max_gap_frames):
    """Merge runs separated by a gap shorter than max_gap_frames.

    For IPU extraction:
      - silence gap >= IPU_PAUSE_S splits (do NOT fill)
      - silence gap <  IPU_PAUSE_S merges (fill)
    """
    out = mask.copy()
    if max_gap_frames <= 0:
        return out
    r = runs(out)
    for i in range(len(r) - 1):
        _s1, e1 = r[i]
        s2, _e2 = r[i + 1]
        gap = int(s2 - e1)
        if gap > 0 and gap < int(max_gap_frames):
            out[e1:s2] = True
    return out

def build_state(maskA, maskB):
    """Legacy helper (unused in IPU mode)."""
    mA, mB = maskA, maskB
    state = np.zeros(len(mA), dtype=np.uint8)
    state[( mA) & (~mB)] = 1
    state[(~mA) & ( mB)] = 2
    state[( mA) & ( mB)] = 3
    prev = 0
    for i in range(len(state)):
        if state[i] == 3:
            state[i] = prev if prev in (1, 2) else 3
        elif state[i] in (1, 2):
            prev = state[i]
    return state


def state_runs(state):
    """Legacy helper (unused in IPU mode)."""
    out = []
    i = 0
    n = len(state)
    while i < n:
        if state[i] in (1, 2):
            lab = state[i]
            j = i + 1
            while j < n and state[j] == lab:
                j += 1
            out.append((lab, i, j))
            i = j
        else:
            i += 1
    return out


# ---------- Core extraction (per-speaker IPUs) ----------
def extract_ipus(
    xL,
    xR,
    sr,
    out_dir,
    base,
    *,
    wav_A_path=None,
    wav_B_path=None,
    prompt_id_unique=None,
    a_id=None,
    b_id=None,
):
    """Extract per-speaker IPUs and write them as stereo WAVs.

    - For spkA clips: left channel keeps A audio, right channel is zeroed.
    - For spkB clips: right channel keeps B audio, left channel is zeroed.
    - If duration_s < IPU_MIN_S, append suffix '_is_backchannel' to the filename.

    Returns:
        (written_total, n_ipu_A, n_ipu_B)
    """

    mL_raw, _t, hop, win = hysteresis_vad(xL, sr, WIN_MS, HOP_MS)
    mR_raw, _t2, _hop2, _win2 = hysteresis_vad(xR, sr, WIN_MS, HOP_MS)

    # Merge gaps shorter than IPU_PAUSE_S within each speaker mask.
    ipu_gap_frames = frames_from_sec_floor(IPU_PAUSE_S, hop, sr)
    mL = fill_gaps_max(mL_raw, ipu_gap_frames)
    mR = fill_gaps_max(mR_raw, ipu_gap_frames)

    ipus_A = runs(mL)
    ipus_B = runs(mR)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0

    def _write_one(spk_label: str, span_f, k: int):
        nonlocal written
        s_f, e_f = span_f
        if e_f <= s_f:
            return

        # Convert frame span -> sample span.
        # Frame i corresponds to window [i*hop, i*hop+win). Cover full last window.
        sS = int(s_f * hop)
        eS = int((e_f - 1) * hop + win)
        sS = max(0, sS)
        eS = min(len(xL), eS)
        if eS <= sS:
            return

        L = xL[sS:eS].copy()
        R = xR[sS:eS].copy()

        # Mute the non-target speaker for the full segment.
        if spk_label == "A":
            R[:] = 0.0
            spk_tag = "spkA"
            wav_src = wav_A_path
        else:
            L[:] = 0.0
            spk_tag = "spkB"
            wav_src = wav_B_path

        start_ms = int((float(sS) / float(sr)) * 1000.0)
        end_ms = int((float(eS) / float(sr)) * 1000.0)
        duration_s = float(eS - sS) / float(sr)
        is_backchannel = bool(duration_s < float(IPU_MIN_S))
        bc_suffix = "_is_backchannel" if is_backchannel else ""

        fname = f"{base}_{spk_tag}_{k:03d}_{start_ms}_{end_ms}{bc_suffix}.wav"
        wav_out = out_dir / fname
        write_stereo(wav_out, L, R, sr)

        # --- write metadata sidecar for transcript alignment ---
        transcript = ""
        try:
            if wav_src:
                segs = _load_dataset_transcript_segments(_wav_path_to_transcript_json(wav_src))
                transcript = _extract_words_in_window(segs, float(sS) / float(sr), float(eS) / float(sr))
        except Exception as _te:
            transcript = ""
            transcript_error = str(_te)
        else:
            transcript_error = ""

        meta = {
            "prompt_id_unique": str(prompt_id_unique) if prompt_id_unique is not None else "",
            "a_id": str(a_id) if a_id is not None else "",
            "b_id": str(b_id) if b_id is not None else "",
            "wav_A": str(wav_A_path) if wav_A_path is not None else "",
            "wav_B": str(wav_B_path) if wav_B_path is not None else "",
            "ipu_wav": str(wav_out.resolve()),
            "speaker": spk_label,
            "spk_tag": spk_tag,
            "k": int(k),
            "sr": int(sr),
            "hop_samples": int(hop),
            "win_samples": int(win),
            "clip_start_sample": int(sS),
            "clip_end_sample": int(eS),
            "clip_start_s": float(sS) / float(sr),
            "clip_end_s": float(eS) / float(sr),
            "clip_start_ms": int(start_ms),
            "clip_end_ms": int(end_ms),
            "duration_s": float(duration_s),
            "is_backchannel": bool(is_backchannel),
            "speaker_transcript": str(transcript),
        }
        if transcript_error:
            meta["transcript_error"] = transcript_error

        meta_path = wav_out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        written += 1

    for k, span in enumerate(ipus_A, start=1):
        _write_one("A", span, k)

    for k, span in enumerate(ipus_B, start=1):
        _write_one("B", span, k)

    return written, len(ipus_A), len(ipus_B)

# ---------- CLI parsing ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-speaker IPUs (spkA/spkB) for Seamless Interaction dyads. "
            "Writes clips to <cwd>/<prompt_id_unique>_full_turns by default."
        )
    )
    parser.add_argument(
        "--prompt-id-unique",
        default=PROMPT_ID_UNIQUE,
        help="prompt_id_unique to process (default: value in script)",
    )
    parser.add_argument(
        "--a-id",
        default=A_ID,
        help="A-side recording ID (a_id). If provided with --b-id, process only that conversation.",
    )
    parser.add_argument(
        "--b-id",
        default=B_ID,
        help="B-side recording ID (b_id). If provided with --a-id, process only that conversation.",
    )
    parser.add_argument(
        "--dataset-root",
        default=DATASET_ROOT,
        help="Path to downloaded Seamless Interaction improvised audio root. Defaults to env SEAMLESS_DATASET_ROOT.",
    )
    parser.add_argument(
        "--dyad-lookup-csv",
        default=DYAD_LOOKUP_CSV,
        help="Path to Seamless Interaction dyad_lookup.csv. Defaults to env SEAMLESS_DYAD_LOOKUP_CSV.",
    )
    parser.add_argument(
        "--interactions-role-abmapped-csv",
        default=INTERACTIONS_ROLE_ABMAPPED_CSV,
        help=(
            "Path to StanceBench AB-mapped interaction metadata. Defaults to env "
            "STANCEBENCH_INTERACTIONS_ROLE_ABMAPPED_CSV or stancebench/metadata/interactions_role_ABmapped.csv."
        ),
    )
    return parser.parse_args()

# ---------- Main ----------
def main():
    args = parse_args()

    prompt_id_unique = args.prompt_id_unique
    a_id_filter = args.a_id if args.a_id not in ("", None) else None
    b_id_filter = args.b_id if args.b_id not in ("", None) else None
    dataset_root = args.dataset_root
    dyad_lookup_csv = args.dyad_lookup_csv
    abmapped_csv = args.interactions_role_abmapped_csv

    if not dataset_root:
        raise SystemExit("Missing dataset root. Set SEAMLESS_DATASET_ROOT or pass --dataset-root.")
    if not dyad_lookup_csv:
        raise SystemExit("Missing dyad lookup CSV. Set SEAMLESS_DYAD_LOOKUP_CSV or pass --dyad-lookup-csv.")

    # Load AB-mapped conversations for this prompt and optional a_id/b_id
    conv_df = load_abmapped_rows(
        abmapped_csv,
        prompt_id_unique,
        a_id=a_id_filter,
        b_id=b_id_filter,
    )

    # Load dyad_lookup once
    dyad_df = pd.read_csv(dyad_lookup_csv, dtype=str, low_memory=False)

    out_dir = OUTPUT_DIR or (Path.cwd()/f"{prompt_id_unique}_full_turns")

    for _, row in conv_df.iterrows():
        a_id = row["a_id"]
        b_id = row["b_id"]

        # Map this conversation's A/B IDs to WAV paths, ensuring A=left, B=right
        p_left, p_right = choose_pair_from_abmapped(dyad_df, a_id, b_id, dataset_root)

        print(f"prompt_id_unique: {prompt_id_unique}")
        print(f"A_id: {a_id}")
        print(f"B_id: {b_id}")
        print(f"Left  (Speaker A): {p_left}")
        print(f"Right (Speaker B): {p_right}")

        xL, srL = read_wav_mono(p_left)
        xR, srR = read_wav_mono(p_right)
        if srL != srR:
            xR = resample_linear(xR, srR, srL); sr = srL
        else:
            sr = srL
        n = max(len(xL), len(xR))
        if len(xL) < n: xL = np.pad(xL, (0, n-len(xL)))
        if len(xR) < n: xR = np.pad(xR, (0, n-len(xR)))

        # Include a_id/b_id in base to keep filenames unique across conversations
        base = f"{prompt_id_unique}_A-{a_id}_B-{b_id}"

        wrote, n_ipu_a, n_ipu_b = extract_ipus(
            xL, xR, sr, out_dir, base,
            wav_A_path=str(p_left),
            wav_B_path=str(p_right),
            prompt_id_unique=prompt_id_unique,
            a_id=a_id,
            b_id=b_id,
        )
        print(f"[debug] IPU candidates: A={n_ipu_a} B={n_ipu_b}")
        print(f"Wrote {wrote} clip(s) to: {Path(out_dir).resolve()}")

if __name__ == "__main__":
    main()
