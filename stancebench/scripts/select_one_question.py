#!/usr/bin/env python3
"""
Select one question entry from a multi-question config JSON and write a single-question config.

Example:
  python select_one_question.py --input questions_main_with_definitions.json --index 0 --output question_warmth_only.json
"""
import argparse, json
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--index", type=int, required=True, help="0-based index into config['outside_judge']")
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = json.loads(args.input.read_text(encoding="utf-8"))
    if "outside_judge" not in cfg or not isinstance(cfg["outside_judge"], list):
        raise SystemExit("Input JSON must have key 'outside_judge' with a list value.")
    items = cfg["outside_judge"]
    if not items:
        raise SystemExit("Input JSON has empty 'outside_judge' list.")
    if args.index < 0 or args.index >= len(items):
        raise SystemExit(f"--index out of range. Must be in [0, {len(items)-1}]")
    out_cfg = {"outside_judge": [items[args.index]]}
    args.output.write_text(json.dumps(out_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote single-question config to: {args.output}")

if __name__ == "__main__":
    main()
