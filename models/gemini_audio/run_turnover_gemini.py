#!/usr/bin/env python3
"""Compatibility wrapper for stancebench/models/gemini_audio/run_turnover_gemini.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "stancebench" / "models" / "gemini_audio" / "run_turnover_gemini.py"),
        run_name="__main__",
    )

