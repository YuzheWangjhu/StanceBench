#!/usr/bin/env python3
"""Compatibility wrapper for stancebench/scripts/turnover_extractor_IPU.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "stancebench" / "scripts" / "turnover_extractor_IPU.py"),
        run_name="__main__",
    )

