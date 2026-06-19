#!/usr/bin/env python3
"""Compatibility wrapper for stancebench/scripts/select_one_question.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "stancebench" / "scripts" / "select_one_question.py"),
        run_name="__main__",
    )

