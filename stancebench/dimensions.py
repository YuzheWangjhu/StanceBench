from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json


PACKAGE_ROOT = Path(__file__).resolve().parent
METADATA_DIR = PACKAGE_ROOT / "metadata"


@dataclass(frozen=True)
class Dimension:
    id: str
    public_index: int
    source_index: int
    name: str
    input_mode: str
    positive_categories: tuple[str, ...]
    negative_categories: tuple[str, ...]
    paper_roles: tuple[str, ...]

    @property
    def related_categories(self) -> tuple[str, ...]:
        return self.positive_categories + self.negative_categories


DIMENSIONS: dict[str, Dimension] = {
    "S0": Dimension(
        "S0",
        0,
        0,
        "Interpersonal Warmth",
        "single",
        ("Warmth",),
        ("Coldness",),
        ("Friendly", "Warm", "Approachable", "Welcoming", "Aloof", "Distant", "Impersonal", "Indifferent"),
    ),
    "S1": Dimension(
        "S1",
        1,
        1,
        "Compassion and Empathy",
        "single",
        ("Compassion",),
        ("Callousness",),
        (
            "Empathetic",
            "Considerate",
            "Understanding",
            "Concerned",
            "Insensitive",
            "Unsympathetic",
            "Inconsiderate",
            "Callous",
        ),
    ),
    "S2": Dimension(
        "S2",
        2,
        2,
        "Politeness and Respect",
        "single",
        ("Politeness",),
        ("Rudeness",),
        ("Polite", "Respectful", "Courteous", "Disrespectful", "Impolite", "Uncivil"),
    ),
    "S3": Dimension(
        "S3",
        3,
        3,
        "Assertiveness",
        "single",
        ("Assertiveness",),
        ("Inhibition",),
        ("Assertive", "Decisive", "Self-assured", "Firm", "Indecisive", "Self-doubting", "Unassertive", "Timid"),
    ),
    "S4": Dimension(
        "S4",
        4,
        4,
        "Sincerity and Honesty",
        "single",
        ("Honesty",),
        ("Deception",),
        ("Honest", "Ingenuous", "Uncalculating", "Manipulative", "Calculating", "Devious"),
    ),
    "S5": Dimension(
        "S5",
        5,
        5,
        "Cognitive Attentiveness",
        "single",
        ("Focus",),
        ("Distraction",),
        ("Alert", "Attentive", "Concentrating", "Engaged", "Bewildered", "Distracted", "Drowsy", "Unfocused"),
    ),
    "S6": Dimension(
        "S6",
        6,
        7,
        "Social Engagement",
        "interaction",
        ("Sociability",),
        ("Withdrawal",),
        ("Engaging", "Sociable", "Gregarious", "Outgoing", "Withdrawn", "Disengaged", "Reticent", "Taciturn"),
    ),
    "S7": Dimension(
        "S7",
        7,
        8,
        "Power Orientation",
        "interaction",
        ("Deference",),
        ("Dominance",),
        ("Submissive", "Meek", "Yielding", "Undemanding", "Forceful", "Overbearing", "Dominant", "Domineering"),
    ),
    "S8": Dimension(
        "S8",
        8,
        9,
        "Conflict Regulation",
        "interaction",
        ("Calmness", "Avoidance"),
        ("Aggression",),
        ("Stable", "Steady", "Unaggressive", "Unargumentative", "Aggressive", "Cruel", "Ruthless", "Vindictive"),
    ),
}


def normalize_dimension_id(value: str) -> str:
    dim = value.strip().upper()
    if dim not in DIMENSIONS:
        valid = ", ".join(sorted(DIMENSIONS))
        raise ValueError(f"Unknown dimension '{value}'. Expected one of: {valid}")
    return dim


def get_dimension(value: str) -> Dimension:
    return DIMENSIONS[normalize_dimension_id(value)]


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or METADATA_DIR / "questions_main.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    questions = cfg.get("outside_judge")
    if not isinstance(questions, list):
        raise ValueError(f"{path} must contain an outside_judge list")
    return questions


def load_category_roles(path: Path | None = None) -> dict[str, list[str]]:
    path = path or METADATA_DIR / "category_roles.csv"
    category_to_roles: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("category_name", "").strip()
            roles_raw = row.get("Adjectives assigned", "")
            roles = [role.strip() for role in roles_raw.split(",") if role.strip()]
            if category:
                category_to_roles[category] = roles
    return category_to_roles


def roles_for_dimension(dimension: Dimension, category_roles_path: Path | None = None) -> list[str]:
    if category_roles_path is not None:
        category_to_roles = load_category_roles(category_roles_path)
        missing = [category for category in dimension.related_categories if category not in category_to_roles]
        if missing:
            raise ValueError(f"Missing categories in category_roles.csv: {', '.join(missing)}")
    return list(dimension.paper_roles)


def question_for_dimension(dimension: Dimension, questions_path: Path | None = None) -> dict:
    questions = load_questions(questions_path)
    if dimension.public_index >= len(questions):
        raise ValueError(f"{dimension.id} index {dimension.public_index} is out of range for questions")
    return questions[dimension.public_index]


def validate_dimension_mapping(questions_path: Path | None = None) -> list[str]:
    questions = load_questions(questions_path)
    errors: list[str] = []
    if len(questions) != len(DIMENSIONS):
        errors.append(f"Expected {len(DIMENSIONS)} questions, found {len(questions)}")
    for dim in DIMENSIONS.values():
        if dim.public_index >= len(questions):
            errors.append(f"{dim.id} public index out of range")
            continue
        question = questions[dim.public_index]
        if tuple(question.get("related_categories", ())) != dim.related_categories:
            errors.append(
                f"{dim.id} related_categories mismatch: "
                f"expected {list(dim.related_categories)}, found {question.get('related_categories')}"
            )
        if question.get("source_index") != dim.source_index:
            errors.append(
                f"{dim.id} source_index mismatch: expected {dim.source_index}, found {question.get('source_index')}"
            )
    return errors
