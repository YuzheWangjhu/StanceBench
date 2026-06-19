from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from typing import Any

from .config import write_config
from .dimensions import Dimension, METADATA_DIR, roles_for_dimension
from .metrics import write_metrics


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SCRIPT_DIR = PACKAGE_ROOT / "scripts"
MODEL_RUNNERS = {
    "qwen-omni": {
        "path": PACKAGE_ROOT / "models/qwen_omni/run_turnover_qwen_QA.py",
        "model_arg": "--qwen-model",
        "default_model": "Qwen/Qwen2.5-Omni-7B",
    },
    "kimi-audio": {
        "path": PACKAGE_ROOT / "models/kimi_audio/run_turnover_KIMI.py",
        "model_arg": "--kimi-model",
        "default_model": "moonshotai/Kimi-Audio-7B-Instruct",
    },
    "granite-speech": {
        "path": PACKAGE_ROOT / "models/granite_speech/run_turnover_granite_QA.py",
        "model_arg": "--granite-model",
        "default_model": "ibm-granite/granite-speech-3.3-8b",
    },
    "gpt-audio": {
        "path": PACKAGE_ROOT / "models/gpt_audio/run_turnover_gpt_QA.py",
        "model_arg": "--gpt-model",
        "default_model": os.environ.get("OPENAI_MODEL", "gpt-audio-2025-08-28"),
    },
    "gemini-audio": {
        "path": PACKAGE_ROOT / "models/gemini_audio/run_turnover_gemini.py",
        "model_arg": "--gemini-model",
        "default_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    },
    "qwen-transcript-ablation": {
        "path": PACKAGE_ROOT / "models/qwen_transcript_ablation/run_turnover_qwenAblation_QA.py",
        "model_arg": "--qwen-model",
        "default_model": "Qwen/Qwen2.5-Omni-7B",
    },
}


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    question_json: Path
    legacy_csv: Path
    filtered_rows_csv: Path
    scores_csv: Path
    metrics_json: Path
    config_yaml: Path
    log_file: Path


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_paths(output_root: Path, model: str, dimension: Dimension, run_id: str | None = None) -> RunPaths:
    run_dir = output_root / model / dimension.id / (run_id or make_run_id())
    return RunPaths(
        run_dir=run_dir,
        question_json=run_dir / "question.json",
        legacy_csv=run_dir / "legacy_filtered_subset.csv",
        filtered_rows_csv=run_dir / "filtered_rows.csv",
        scores_csv=run_dir / "scores.csv",
        metrics_json=run_dir / "metrics.json",
        config_yaml=run_dir / "config.resolved.yaml",
        log_file=run_dir / "logs/run.log",
    )


def run_command(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, log_file: Path | None = None) -> None:
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            result = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def select_question(dimension: Dimension, output: Path, questions_path: Path | None = None) -> None:
    input_path = questions_path or METADATA_DIR / "questions_main.json"
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "select_one_question.py"),
            "--input",
            str(input_path),
            "--index",
            str(dimension.public_index),
            "--output",
            str(output),
        ]
    )


def filter_roles(
    dimension: Dimension,
    output_csv: Path,
    selection_portion: float = 1.0,
    seed: int | None = 666,
    interactions_csv: Path | None = None,
    category_roles_csv: Path | None = None,
) -> None:
    env = os.environ.copy()
    if interactions_csv:
        env["STANCEBENCH_INTERACTIONS_ROLE_ABMAPPED_CSV"] = str(interactions_csv)
    if category_roles_csv:
        env["STANCEBENCH_CATEGORY_ROLES_CSV"] = str(category_roles_csv)
    roles = roles_for_dimension(dimension, category_roles_csv)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "filter_roles.py"),
        "--names",
        *roles,
        "--selection-portion",
        str(selection_portion),
        "--output-csv",
        str(output_csv),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    run_command(cmd, env=env)


def build_inputs(
    dimension: Dimension,
    prompt_id_unique: str,
    a_id: str,
    b_id: str,
    cwd: Path | None = None,
    dataset_root: str | None = None,
    dyad_lookup_csv: str | None = None,
    interactions_csv: str | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_eval_inputs.py"),
        "--prompt-id-unique",
        prompt_id_unique,
        "--a-id",
        a_id,
        "--b-id",
        b_id,
        "--input-mode",
        dimension.input_mode,
    ]
    if dataset_root:
        cmd.extend(["--dataset-root", dataset_root])
    if dyad_lookup_csv:
        cmd.extend(["--dyad-lookup-csv", dyad_lookup_csv])
    if interactions_csv:
        cmd.extend(["--interactions-role-abmapped-csv", interactions_csv])
    run_command(cmd, cwd=cwd)


def discover_manifests(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*_full_turns/*__manifest.jsonl"))


def run_model(
    model: str,
    dimension: Dimension,
    output_root: Path,
    run_id: str | None = None,
    selection_portion: float = 1.0,
    seed: int | None = 666,
    dataset_root: str | None = None,
    dyad_lookup_csv: str | None = None,
    interactions_csv: str | None = None,
    model_id: str | None = None,
    keep_turnovers: bool = True,
    runner_extra: list[str] | None = None,
    resolved_config: dict[str, Any] | None = None,
) -> RunPaths:
    if model not in MODEL_RUNNERS:
        raise SystemExit(f"Unknown model '{model}'. Expected one of: {', '.join(sorted(MODEL_RUNNERS))}")
    runner = MODEL_RUNNERS[model]
    paths = build_run_paths(output_root, model, dimension, run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    select_question(dimension, paths.question_json)
    roles = roles_for_dimension(dimension)

    env = os.environ.copy()
    if dataset_root:
        env["SEAMLESS_DATASET_ROOT"] = dataset_root
    if dyad_lookup_csv:
        env["SEAMLESS_DYAD_LOOKUP_CSV"] = dyad_lookup_csv
    if interactions_csv:
        env["STANCEBENCH_INTERACTIONS_ROLE_ABMAPPED_CSV"] = interactions_csv

    cmd = [
        sys.executable,
        str(runner["path"]),
        "--roles-of-interest",
        *roles,
        "--selection-portion",
        str(selection_portion),
        "--start-over",
        "True",
        "--question-config",
        str(paths.question_json),
        "--filtered-csv",
        str(paths.legacy_csv),
        "--input-mode",
        dimension.input_mode,
        str(runner["model_arg"]),
        str(model_id or runner["default_model"]),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if keep_turnovers:
        cmd.append("--keep-turnovers")
    if runner_extra:
        cmd.extend(runner_extra)

    config_out = {
        "model": model,
        "dimension": dimension.id,
        "source_index": dimension.source_index,
        "input_mode": dimension.input_mode,
        "roles_of_interest": roles,
        "selection_portion": selection_portion,
        "seed": seed,
        "dataset_root": dataset_root,
        "dyad_lookup_csv": dyad_lookup_csv,
        "interactions_csv": interactions_csv,
        "model_id": model_id or runner["default_model"],
        "keep_turnovers": keep_turnovers,
        "runner_extra": runner_extra or [],
        "legacy_csv": str(paths.legacy_csv),
    }
    if resolved_config:
        config_out.update({k: v for k, v in resolved_config.items() if v is not None})
    write_config(paths.config_yaml, config_out)

    run_command(cmd, cwd=paths.run_dir, env=env, log_file=paths.log_file)

    if paths.legacy_csv.exists():
        shutil.copy2(paths.legacy_csv, paths.filtered_rows_csv)
        shutil.copy2(paths.legacy_csv, paths.scores_csv)
        write_metrics(paths.legacy_csv, dimension, paths.metrics_json)
    return paths
