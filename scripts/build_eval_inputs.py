#!/usr/bin/env python3
"""Compatibility wrapper for stancebench/scripts/build_eval_inputs.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "stancebench" / "scripts" / "build_eval_inputs.py"),
        run_name="__main__",
    )

