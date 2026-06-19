#!/usr/bin/env python3
"""Compatibility wrapper for stancebench/models/kimi_audio/run_turnover_KIMI.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "stancebench" / "models" / "kimi_audio" / "run_turnover_KIMI.py"),
        run_name="__main__",
    )

