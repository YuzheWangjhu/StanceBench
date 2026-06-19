from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

from .config import load_config, merge_config
from .dimensions import DIMENSIONS, get_dimension, question_for_dimension, roles_for_dimension
from .metrics import write_metrics
from .pipeline import MODEL_RUNNERS, build_inputs, filter_roles, run_model, select_question
from .validation import validate_metadata, validate_seamless_layout


def _dimension_choices() -> list[str]:
    return sorted(DIMENSIONS)


def _model_choices() -> list[str]:
    return sorted(MODEL_RUNNERS)


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML or JSON config file. CLI flags override config fields.")


def _str_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def cmd_validate_data(args: argparse.Namespace) -> int:
    errors = validate_metadata(args.metadata_dir)
    errors.extend(validate_seamless_layout(args.dataset_root, args.dyad_lookup_csv))
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print("[OK] StanceBench metadata and requested data paths are valid.")
    return 0


def cmd_select_question(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    merged = merge_config(cfg, {"dimension": args.dimension, "output": args.output})
    dimension = get_dimension(merged["dimension"])
    output = Path(merged["output"])
    select_question(dimension, output)
    return 0


def cmd_filter_roles(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    merged = merge_config(
        cfg,
        {
            "dimension": args.dimension,
            "output_csv": args.output_csv,
            "selection_portion": args.selection_portion,
            "seed": args.seed,
            "interactions_csv": args.interactions_csv,
            "category_roles_csv": args.category_roles_csv,
        },
    )
    dimension = get_dimension(merged["dimension"])
    filter_roles(
        dimension=dimension,
        output_csv=Path(merged["output_csv"]),
        selection_portion=float(merged.get("selection_portion", 1.0)),
        seed=merged.get("seed", 666),
        interactions_csv=Path(merged["interactions_csv"]) if merged.get("interactions_csv") else None,
        category_roles_csv=Path(merged["category_roles_csv"]) if merged.get("category_roles_csv") else None,
    )
    return 0


def cmd_build_inputs(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    merged = merge_config(
        cfg,
        {
            "dimension": args.dimension,
            "prompt_id_unique": args.prompt_id_unique,
            "a_id": args.a_id,
            "b_id": args.b_id,
            "output_dir": args.output_dir,
            "dataset_root": args.dataset_root,
            "dyad_lookup_csv": args.dyad_lookup_csv,
            "interactions_csv": args.interactions_csv,
        },
    )
    dimension = get_dimension(merged["dimension"])
    build_inputs(
        dimension=dimension,
        prompt_id_unique=merged["prompt_id_unique"],
        a_id=merged["a_id"],
        b_id=merged["b_id"],
        cwd=Path(merged["output_dir"]) if merged.get("output_dir") else None,
        dataset_root=merged.get("dataset_root"),
        dyad_lookup_csv=merged.get("dyad_lookup_csv"),
        interactions_csv=merged.get("interactions_csv"),
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    merged = merge_config(
        cfg,
        {
            "model": args.model,
            "dimension": args.dimension,
            "output_root": args.output_root,
            "run_id": args.run_id,
            "selection_portion": args.selection_portion,
            "seed": args.seed,
            "dataset_root": args.dataset_root,
            "dyad_lookup_csv": args.dyad_lookup_csv,
            "interactions_csv": args.interactions_csv,
            "model_id": args.model_id,
            "keep_turnovers": args.keep_turnovers,
            "runner_extra": args.runner_extra,
        },
    )
    dimension = get_dimension(merged["dimension"])
    paths = run_model(
        model=merged["model"],
        dimension=dimension,
        output_root=Path(merged.get("output_root", "runs")),
        run_id=merged.get("run_id"),
        selection_portion=float(merged.get("selection_portion", 1.0)),
        seed=merged.get("seed", 666),
        dataset_root=merged.get("dataset_root"),
        dyad_lookup_csv=merged.get("dyad_lookup_csv"),
        interactions_csv=merged.get("interactions_csv"),
        model_id=merged.get("model_id"),
        keep_turnovers=bool(merged.get("keep_turnovers", True)),
        runner_extra=list(merged.get("runner_extra") or []),
        resolved_config=merged,
    )
    print(f"[OK] run_dir: {paths.run_dir}")
    print(f"[OK] legacy_csv: {paths.legacy_csv}")
    print(f"[OK] metrics: {paths.metrics_json}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    merged = merge_config(
        cfg,
        {
            "dimension": args.dimension,
            "run_dir": args.run_dir,
            "csv": args.csv,
            "output": args.output,
        },
    )
    run_dir = Path(merged["run_dir"]) if merged.get("run_dir") else None
    if merged.get("csv"):
        csv_path = Path(merged["csv"])
    elif run_dir:
        csv_path = run_dir / "legacy_filtered_subset.csv"
    else:
        raise SystemExit("stancebench metrics requires --run-dir or --csv")
    output = Path(merged["output"]) if merged.get("output") else (run_dir / "metrics.json" if run_dir else csv_path.parent / "metrics.json")
    dimension_id = merged.get("dimension")
    if not dimension_id and run_dir:
        resolved_cfg = run_dir / "config.resolved.yaml"
        if resolved_cfg.exists():
            dimension_id = load_config(resolved_cfg).get("dimension")
    if not dimension_id:
        raise SystemExit("--dimension is required when it cannot be inferred from run_dir/config.resolved.yaml")
    metrics = write_metrics(csv_path, get_dimension(dimension_id), output)
    print(json.dumps(metrics, indent=2))
    return 0


def _analysis_notebook_path() -> Path:
    package_root = Path(__file__).resolve().parent
    candidates = [
        package_root.parent / "examples/notebooks/analyze_all_paper.ipynb",
        package_root / "examples/notebooks/analyze_all_paper.ipynb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("Analysis notebook not found in source checkout or installed package data.")


def cmd_analyze(args: argparse.Namespace) -> int:
    notebook = _analysis_notebook_path()
    if args.runs:
        runs = args.runs
        if not runs.exists():
            raise SystemExit(f"--runs does not exist: {runs}")
        csvs = sorted(runs.glob("*/S*/*/legacy_filtered_subset.csv"))
        print(f"[OK] discovered {len(csvs)} CLI run CSV(s) under: {runs}")
        for csv in csvs[:20]:
            print(f"  {csv}")
        if len(csvs) > 20:
            print(f"  ... {len(csvs) - 20} more")
    print(f"[OK] analysis notebook: {notebook}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stancebench", description="StanceBench command line interface.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-data", help="Validate StanceBench metadata and optional Seamless data paths.")
    p.add_argument("--metadata-dir", type=Path, default=None)
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--dyad-lookup-csv", type=Path, default=None)
    p.set_defaults(func=cmd_validate_data)

    p = sub.add_parser("select-question", help="Write a one-question config for a stance dimension.")
    _add_config_arg(p)
    p.add_argument("--dimension", choices=_dimension_choices(), required=False)
    p.add_argument("--output", type=Path, required=False)
    p.set_defaults(func=cmd_select_question)

    p = sub.add_parser("filter-roles", help="Filter benchmark rows for a stance dimension.")
    _add_config_arg(p)
    p.add_argument("--dimension", choices=_dimension_choices(), required=False)
    p.add_argument("--output-csv", type=Path, required=False)
    p.add_argument("--selection-portion", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--interactions-csv", type=Path, default=None)
    p.add_argument("--category-roles-csv", type=Path, default=None)
    p.set_defaults(func=cmd_filter_roles)

    p = sub.add_parser("build-inputs", help="Build evaluation audio inputs for one dyad conversation.")
    _add_config_arg(p)
    p.add_argument("--dimension", choices=_dimension_choices(), required=False)
    p.add_argument("--prompt-id-unique")
    p.add_argument("--a-id")
    p.add_argument("--b-id")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dataset-root")
    p.add_argument("--dyad-lookup-csv")
    p.add_argument("--interactions-csv")
    p.set_defaults(func=cmd_build_inputs)

    p = sub.add_parser("run", help="Run one paper model on one stance dimension.")
    _add_config_arg(p)
    p.add_argument("--model", choices=_model_choices(), required=False)
    p.add_argument("--dimension", choices=_dimension_choices(), required=False)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--run-id")
    p.add_argument("--selection-portion", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset-root")
    p.add_argument("--dyad-lookup-csv")
    p.add_argument("--interactions-csv")
    p.add_argument("--model-id", help="Checkpoint or API model ID for the selected model.")
    p.add_argument("--keep-turnovers", type=_str_bool, default=None)
    p.add_argument("--runner-extra", action="append", default=None, help="Extra argument forwarded to the model runner. Repeat for multiple args.")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("metrics", help="Compute StanceBench summary metrics for a run CSV.")
    _add_config_arg(p)
    p.add_argument("--dimension", choices=_dimension_choices(), required=False)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--csv", type=Path)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("analyze", help="Locate the primary analysis notebook and summarize CLI run outputs.")
    p.add_argument("--runs", type=Path, help="Optional runs root to scan for CLI legacy_filtered_subset.csv outputs.")
    p.set_defaults(func=cmd_analyze)

    return parser


def _require_fields(command: str, values: dict[str, object], fields: list[str]) -> None:
    missing = [field for field in fields if values.get(field) in (None, "")]
    if missing:
        raise SystemExit(f"stancebench {command} missing required field(s): {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "select-question":
        _require_fields(args.command, vars(args) | load_config(args.config), ["dimension", "output"])
    elif args.command == "filter-roles":
        _require_fields(args.command, vars(args) | load_config(args.config), ["dimension", "output_csv"])
    elif args.command == "build-inputs":
        _require_fields(args.command, vars(args) | load_config(args.config), ["dimension", "prompt_id_unique", "a_id", "b_id"])
    elif args.command == "run":
        _require_fields(args.command, vars(args) | load_config(args.config), ["model", "dimension"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
