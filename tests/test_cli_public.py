import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from stancebench.dimensions import DIMENSIONS, get_dimension, question_for_dimension, roles_for_dimension
from stancebench.metrics import score_summary
from stancebench.pipeline import discover_manifests
from stancebench.cli import _analysis_notebook_path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "stancebench.cli", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


class PublicCliTests(unittest.TestCase):
    def test_dimension_mapping_and_roles(self):
        self.assertEqual(list(DIMENSIONS), [f"S{i}" for i in range(9)])
        self.assertEqual(get_dimension("S6").source_index, 7)
        self.assertEqual(get_dimension("S6").input_mode, "interaction")
        self.assertEqual(question_for_dimension(get_dimension("S0"))["source_index"], 0)
        self.assertEqual(question_for_dimension(get_dimension("S6"))["source_index"], 7)
        self.assertIn("Friendly", roles_for_dimension(get_dimension("S0")))
        self.assertIn("Engaging", roles_for_dimension(get_dimension("S6")))

    def test_cli_help_and_validate_data(self):
        for args in [
            ("--help",),
            ("validate-data", "--help"),
            ("select-question", "--help"),
            ("filter-roles", "--help"),
            ("build-inputs", "--help"),
            ("run", "--help"),
            ("metrics", "--help"),
            ("analyze", "--help"),
        ]:
            result = run_cli(*args)
            self.assertIn("usage:", result.stdout)
        result = run_cli("validate-data")
        self.assertIn("[OK]", result.stdout)

    def test_select_question_cli_and_wrapper_match(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cli_out = tmp / "cli_question.json"
            wrapper_out = tmp / "wrapper_question.json"
            run_cli("select-question", "--dimension", "S0", "--output", str(cli_out))
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/select_one_question.py"),
                    "--input",
                    str(REPO_ROOT / "metadata/questions_main.json"),
                    "--index",
                    "0",
                    "--output",
                    str(wrapper_out),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            self.assertEqual(json.loads(cli_out.read_text()), json.loads(wrapper_out.read_text()))

    def test_filter_roles_cli_on_tiny_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fixture = tmp / "interactions.csv"
            output = tmp / "filtered.csv"
            with (REPO_ROOT / "stancebench/metadata/interactions_role_ABmapped.csv").open(newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    if row["role_a"] in {"Friendly", "Aloof"} or row["role_b"] in {"Friendly", "Aloof"}:
                        rows.append(row)
                    if len(rows) == 4:
                        break
                fieldnames = reader.fieldnames
            with fixture.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            run_cli(
                "filter-roles",
                "--dimension",
                "S0",
                "--interactions-csv",
                str(fixture),
                "--selection-portion",
                "1.0",
                "--output-csv",
                str(output),
            )
            with output.open(newline="") as f:
                filtered = list(csv.DictReader(f))
            self.assertEqual(len(filtered), 4)
            self.assertIn("category_a", filtered[0])
            self.assertTrue(any(row["role_a"] in {"Friendly", "Aloof"} for row in filtered))

    def test_metrics_computation(self):
        rows = [
            {"category_a": "Warmth", "avg_score_a": "2", "flip_rate_a": "0", "category_b": "", "avg_score_b": "", "flip_rate_b": ""},
            {"category_a": "Coldness", "avg_score_a": "-1", "flip_rate_a": "0", "category_b": "", "avg_score_b": "", "flip_rate_b": ""},
        ]
        metrics = score_summary(rows, get_dimension("S0"))
        self.assertEqual(metrics["attempts"], 2)
        self.assertEqual(metrics["evaluated"], 2)
        self.assertEqual(metrics["failure_rate"], 0.0)
        self.assertEqual(metrics["categorical_pole_consistency"], 1.0)

    def test_metrics_cli_with_csv_defaults_output_next_to_csv(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            csv_path = tmp / "scores.csv"
            csv_path.write_text(
                "category_a,avg_score_a,flip_rate_a,category_b,avg_score_b,flip_rate_b\n"
                "Warmth,2,0,,,\n",
                encoding="utf-8",
            )
            run_cli("metrics", "--csv", str(csv_path), "--dimension", "S0")
            self.assertTrue((tmp / "metrics.json").exists())

    def test_analyze_scans_cli_runs(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            csv_path = runs / "qwen-omni" / "S0" / "test-run" / "legacy_filtered_subset.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("", encoding="utf-8")
            result = run_cli("analyze", "--runs", str(runs))
            self.assertIn("discovered 1 CLI run CSV", result.stdout)
            self.assertIn("analyze_all_paper.ipynb", result.stdout)

    def test_analysis_notebook_packaged_path_exists(self):
        self.assertTrue(_analysis_notebook_path().exists())

    def test_manifest_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "example_full_turns/example__manifest.jsonl"
            manifest.parent.mkdir()
            manifest.write_text("{}\n")
            self.assertEqual(discover_manifests(root), [manifest])


if __name__ == "__main__":
    unittest.main()
