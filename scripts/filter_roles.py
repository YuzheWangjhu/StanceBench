#!/usr/bin/env python3
"""
Filter interactions_role_ABmapped.csv by role/category and add category_a/category_b.

- Reads roles/categories from category_roles.csv.
- Takes a list of names (roles from category_roles.csv).
- A row is kept if either role_a or role_b is in the selected roles set.
- selection_portion in (0, 1]:
    * 1.0 -> keep all matching rows
    * <1.0 -> per-role sampling:
        - For each requested role, compute selection_portion * N_role.
        - If this is >= 10, sample that many rows for that role.
        - If this is < 10, sample up to 10 rows or all available rows
          for that role if fewer than 10 exist.
- Output CSV keeps all original columns plus:
    * category_a
    * category_b
  inserted between b_id and role_a.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
METADATA_DIR = REPO_ROOT / "metadata"
INTERACTIONS_CSV = Path(
    os.environ.get(
        "STANCEBENCH_INTERACTIONS_ROLE_ABMAPPED_CSV",
        str(METADATA_DIR / "interactions_role_ABmapped.csv"),
    )
)
ROLE_CAT_CSV = Path(
    os.environ.get(
        "STANCEBENCH_CATEGORY_ROLES_CSV",
        str(METADATA_DIR / "category_roles.csv"),
    )
)


def parse_roles_categories(path: Path):
    """
    Parse category_roles.csv into:
      - role_to_category: mapping from role -> category
      - category_to_roles: mapping from category -> set of roles

    Expected columns:
      - "category_name": category label
      - "Adjectives assigned": comma-separated roles/adjectives

    Example (conceptual):

        category_name = "Friendly"
        Adjectives assigned = "Affable, Agreeable, Companionable, Friendly, ..."
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Required columns
    if "category_name" not in df.columns or "Adjectives assigned" not in df.columns:
        raise SystemExit(
            f"category_roles.csv must contain columns 'category_name' and 'Adjectives assigned'. "
            f"Found columns: {list(df.columns)}"
        )

    role_to_category: dict[str, str] = {}
    category_to_roles: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        current_cat = str(row["category_name"]).strip()
        if not current_cat:
            continue

        if current_cat not in category_to_roles:
            category_to_roles[current_cat] = set()

        roles_str = str(row["Adjectives assigned"]).strip()
        if not roles_str:
            continue

        # Roles are comma-separated in this column
        parts = [p.strip() for p in roles_str.split(",") if p.strip()]
        for role in parts:
            role_to_category[role] = current_cat
            category_to_roles[current_cat].add(role)

    return role_to_category, category_to_roles


def build_selected_roles(
    requested_names: list[str],
    role_to_category: dict[str, str],
    category_to_roles: dict[str, set[str]],
) -> set[str]:
    """
    From user-specified names, build the set of roles to filter by.

    Here, names are interpreted strictly as roles that appear in the
    role lists in category_roles.csv (not as category labels).
    """
    if not requested_names:
        raise SystemExit("At least one role name must be provided.")

    all_roles = set(role_to_category.keys())
    selected_roles: set[str] = set()

    for name in requested_names:
        name = name.strip()
        if not name:
            continue
        if name in all_roles:
            selected_roles.add(name)
        else:
            raise SystemExit(
                f"'{name}' not found as a role in category_roles.csv"
            )

    if not selected_roles:
        raise SystemExit("After expansion, no roles were selected.")

    return selected_roles


def insert_category_columns(df: pd.DataFrame, role_to_category: dict[str, str]) -> pd.DataFrame:
    """
    Add category_a and category_b between b_id and role_a.
    category_a/category_b contain the category labels (e.g. 'Warmth', 'Coldness').
    """
    if "role_a" not in df.columns or "role_b" not in df.columns or "b_id" not in df.columns:
        raise SystemExit("Input CSV must contain 'b_id', 'role_a', and 'role_b' columns.")

    df = df.copy()

    # Map roles to categories; unknown roles get NaN
    df["category_a"] = df["role_a"].map(role_to_category)
    df["category_b"] = df["role_b"].map(role_to_category)

    # Reinsert in the requested position: b_id, category_a, category_b, role_a, ...
    cols = list(df.columns)

    # Remove the newly added columns from the end
    cols.remove("category_a")
    cols.remove("category_b")

    new_cols = []
    for c in cols:
        new_cols.append(c)
        if c == "b_id":
            new_cols.extend(["category_a", "category_b"])

    df = df[new_cols]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Filter interactions_role_ABmapped.csv by role/category and add category columns."
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help=(
            "List of role names (as in category_roles.csv). "
            "A row is kept if role_a or role_b is in the expanded roles set."
        ),
    )
    parser.add_argument(
        "--selection-portion",
        type=float,
        default=1.0,
        help="Fraction of matching rows to keep, in (0,1]. Default: 1.0 (keep all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for deterministic subset selection (used only when selection_portion < 1.0). "
            "If omitted, sampling is non-deterministic (current behavior)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("interactions_role_ABmapped_filtered.csv"),
        help="Output CSV path (default: ./interactions_role_ABmapped_filtered.csv)",
    )

    args = parser.parse_args()

    if not (0.0 < args.selection_portion <= 1.0):
        raise SystemExit("selection_portion must be in the interval (0, 1].")

    # Load roles/categories
    role_to_category, category_to_roles = parse_roles_categories(ROLE_CAT_CSV)

    # Expand requested names into a set of roles (strictly roles, not categories)
    selected_roles = build_selected_roles(
        args.names,
        role_to_category=role_to_category,
        category_to_roles=category_to_roles,
    )

    # Load interactions CSV from this pipeline directory for reproducibility.
    interactions_csv = Path(INTERACTIONS_CSV)
    if not interactions_csv.exists():
        raise SystemExit(f"Missing interactions CSV: {interactions_csv}")
    df = pd.read_csv(interactions_csv, dtype=str, low_memory=False)

    # Filter by roles: keep rows where role_a or role_b is in the selected roles set
    if "role_a" not in df.columns or "role_b" not in df.columns:
        raise SystemExit("Input CSV must contain 'role_a' and 'role_b' columns.")

    mask = df["role_a"].isin(selected_roles) | df["role_b"].isin(selected_roles)
    df_matches = df[mask].copy()

    if df_matches.empty:
        print("No rows matched the specified roles; writing empty CSV.")
        df_matches.to_csv(args.output_csv, index=False)
        return

    # Sampling logic
    if args.selection_portion >= 1.0:
        # Keep all matching rows
        df_sel = df_matches
    else:
        # Per-role sampling with a minimum of up to 10 rows per role (if available)
        frames = []
        for role in sorted(selected_roles):
            role_mask = (df_matches["role_a"] == role) | (df_matches["role_b"] == role)
            df_r = df_matches[role_mask]
            n_r = len(df_r)
            if n_r == 0:
                continue

            base_n = int(round(n_r * args.selection_portion))
            if base_n < 10:
                n_keep = min(10, n_r)
            else:
                n_keep = base_n

            # Sample without replacement (optionally deterministic via --seed)
            frames.append(df_r.sample(n=n_keep, replace=False, random_state=args.seed))

        if not frames:
            # Fallback: no per-role frames (should not happen if df_matches non-empty)
            df_sel = df_matches
        else:
            df_sel = pd.concat(frames, axis=0).drop_duplicates()

    # Insert category_a / category_b columns
    df_sel = insert_category_columns(df_sel, role_to_category)

    # Save
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sel.to_csv(args.output_csv, index=False)
    print(
        f"Wrote {len(df_sel)} rows to {args.output_csv} "
        f"(selection_portion={args.selection_portion})"
    )


if __name__ == "__main__":
    main()


"""
python filter_roles.py \
  --names Friendly Pleasant Distant Indifferent \
  --selection-portion 0.2 \
  --output-csv filtered_subset.csv
"""
