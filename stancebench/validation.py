from __future__ import annotations

from pathlib import Path
import csv

from .dimensions import METADATA_DIR, validate_dimension_mapping, load_category_roles


def validate_metadata(metadata_dir: Path | None = None) -> list[str]:
    metadata_dir = metadata_dir or METADATA_DIR
    errors: list[str] = []
    questions_path = metadata_dir / "questions_main.json"
    category_roles_path = metadata_dir / "category_roles.csv"
    interactions_path = metadata_dir / "interactions_role_ABmapped.csv"

    for path in [questions_path, category_roles_path, interactions_path]:
        if not path.exists():
            errors.append(f"Missing metadata file: {path}")

    if questions_path.exists():
        errors.extend(validate_dimension_mapping(questions_path))

    if category_roles_path.exists():
        try:
            category_roles = load_category_roles(category_roles_path)
            if len(category_roles) < 2:
                errors.append(f"category_roles.csv has too few categories: {len(category_roles)}")
        except Exception as exc:
            errors.append(f"Could not parse category_roles.csv: {exc}")

    if interactions_path.exists():
        with interactions_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"prompt_id_unique", "a_id", "b_id", "role_a", "role_b"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                errors.append(f"interactions_role_ABmapped.csv missing columns: {sorted(missing)}")
            if not any(True for _ in reader):
                errors.append("interactions_role_ABmapped.csv has no rows")

    return errors


def validate_seamless_layout(dataset_root: Path | None, dyad_lookup_csv: Path | None) -> list[str]:
    errors: list[str] = []
    if dataset_root is not None:
        if not dataset_root.exists():
            errors.append(f"Dataset root does not exist: {dataset_root}")
        elif not dataset_root.is_dir():
            errors.append(f"Dataset root is not a directory: {dataset_root}")
    if dyad_lookup_csv is not None:
        if not dyad_lookup_csv.exists():
            errors.append(f"Dyad lookup CSV does not exist: {dyad_lookup_csv}")
        elif not dyad_lookup_csv.is_file():
            errors.append(f"Dyad lookup CSV is not a file: {dyad_lookup_csv}")
    return errors
