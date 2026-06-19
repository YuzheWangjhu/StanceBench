from __future__ import annotations

from pathlib import Path
import csv
import json
import math

from .dimensions import Dimension


def _safe_float(value: object) -> float:
    try:
        x = float(str(value).strip())
        return x if math.isfinite(x) else math.nan
    except Exception:
        return math.nan


def _sign(value: object) -> int:
    x = _safe_float(value)
    if not math.isfinite(x):
        return 0
    return 1 if x > 0 else -1 if x < 0 else 0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_summary(rows: list[dict[str, str]], dimension: Dimension) -> dict[str, float | int]:
    positive = set(dimension.positive_categories)
    negative = set(dimension.negative_categories)
    evaluated = 0
    attempts = 0
    categorical_hits = 0
    categorical_total = 0
    flip_rates: list[float] = []

    for row in rows:
        for side in ("a", "b"):
            category = row.get(f"category_{side}", "")
            if category not in positive and category not in negative:
                continue
            attempts += 1
            avg = _safe_float(row.get(f"avg_score_{side}", ""))
            if not math.isfinite(avg):
                continue
            evaluated += 1
            expected = 1 if category in positive else -1
            predicted = _sign(avg)
            if predicted != 0:
                categorical_total += 1
                categorical_hits += int(predicted == expected)
            flip_rate = _safe_float(row.get(f"flip_rate_{side}", ""))
            if math.isfinite(flip_rate):
                flip_rates.append(flip_rate)

    return {
        "attempts": attempts,
        "evaluated": evaluated,
        "failure_rate": (attempts - evaluated) / attempts if attempts else math.nan,
        "categorical_pole_consistency": categorical_hits / categorical_total if categorical_total else math.nan,
        "mean_flip_rate": sum(flip_rates) / len(flip_rates) if flip_rates else math.nan,
    }


def write_metrics(csv_path: Path, dimension: Dimension, output_path: Path) -> dict[str, float | int]:
    metrics = score_summary(read_csv_rows(csv_path), dimension)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
