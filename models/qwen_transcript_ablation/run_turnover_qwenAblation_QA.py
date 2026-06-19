#!/usr/bin/env python3
"""Compatibility wrapper for stancebench/models/qwen_transcript_ablation/run_turnover_qwenAblation_QA.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "stancebench" / "models" / "qwen_transcript_ablation" / "run_turnover_qwenAblation_QA.py"),
        run_name="__main__",
    )

