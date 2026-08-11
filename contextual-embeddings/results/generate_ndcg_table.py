#!/usr/bin/env python3
"""Generate a Markdown table of nDCG@10 results from JSON result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = (
    ("MLDR", "mldr-conteb-eval"),
    ("SQuAD", "squad-conteb-eval"),
    ("NarrativeQA", "narrative-qa"),
    ("COVID-QA", "covid-qa"),
    ("ESG Reports", "esg-reports"),
    ("Football", "football"),
    ("Geography", "geography"),
    ("Insurance", "insurance"),
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Collect nDCG@10 values from result JSON files into Markdown."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir,
        help="directory containing result JSON files (default: script directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "ndcg_at_10_results.md",
        help="output Markdown file",
    )
    return parser.parse_args()


def format_value(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    return str(value)


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|")


def generate_table(input_dir: Path) -> str:
    headers = ["JSON file", *(name for name, _ in DATASETS)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(DATASETS)) + " |",
    ]

    for json_path in sorted(input_dir.glob("*.json"), key=lambda path: path.name.lower()):
        try:
            with json_path.open(encoding="utf-8") as file:
                result = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to read {json_path}: {exc}") from exc

        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        values = []
        for _, dataset_key in DATASETS:
            dataset_metrics = metrics.get(dataset_key, {})
            value = (
                dataset_metrics.get("ndcg_at_10")
                if isinstance(dataset_metrics, dict)
                else None
            )
            values.append(format_value(value))

        row = [markdown_escape(json_path.name), *values]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_table(input_dir), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
