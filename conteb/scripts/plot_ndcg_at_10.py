#!/usr/bin/env python3
"""Plot the hard-coded ModernBERT ConTEB nDCG@10 results.

Example:
    uv run python3 scripts/plot_ndcg_at_10.py
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

DATASETS = (
    "covid-qa",
    "geography",
    "insurance",
    "football",
    "squad-conteb-eval",
    "narrative-qa",
    "mldr-conteb-eval",
)
LABELS = ("chunk-only", "LC", "LC-left", "LC-right")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_FILES = (
    PROJECT_ROOT / "results/metrics_use_prefix_modernbert_sentence_transformer.json",
    PROJECT_ROOT / "results/metrics_use_prefix_modernbert_contextual-naive.json",
    PROJECT_ROOT / "results/metrics_use_prefix_modernbert_contextual-left.json",
    PROJECT_ROOT / "results/metrics_use_prefix_modernbert-right_contextual.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/modernbert.svg"

DEFAULT_RESULT_FILES = (
    PROJECT_ROOT / "results/metrics_use_prefix_modernbert-insent_sentence_transformer.json",
    PROJECT_ROOT / "results/metrics_use_prefix_modernbert-insent-reproduce_contextual.json",
    PROJECT_ROOT / "results/metrics_use_prefix_modernbert-insent-left_contextual.json",
    PROJECT_ROOT / "results/metrics_use_prefix_modernbert-insent-right_contextual.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/modernbert-insent.svg"


def read_ndcg_at_10(path: Path) -> dict[str, float]:
    """Load and validate the seven nDCG@10 values in an evaluation JSON file."""
    try:
        with path.open(encoding="utf-8") as file:
            payload: dict[str, Any] = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Result file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} must contain a top-level 'metrics' object.")

    values = {}
    for dataset in DATASETS:
        dataset_metrics = metrics.get(dataset)
        if not isinstance(dataset_metrics, dict):
            raise ValueError(f"{path} is missing metrics.{dataset}.")
        value = next(
            (dataset_metrics[key] for key in ("ndcg_at_10", "ndcg@10") if key in dataset_metrics),
            None,
        )
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{path} has a non-numeric nDCG@10 for '{dataset}'.")
        values[dataset] = float(value)
    return values


def build_rows(paths: list[Path]) -> list[dict[str, str | float]]:
    """Return long-form rows for seaborn, including per-method averages."""
    rows = []
    for label, path in zip(LABELS, paths, strict=True):
        scores = read_ndcg_at_10(path)
        for dataset in DATASETS:
            rows.append({"Dataset": dataset, "Method": label, "nDCG@10": scores[dataset]})
        rows.append(
            {
                "Dataset": "avg",
                "Method": label,
                "nDCG@10": sum(scores.values()) / len(scores),
            }
        )
    return rows


def plot(rows: list[dict[str, str | float]], output: Path, dpi: int) -> None:
    """Create a publication-oriented grouped bar chart with seaborn."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires seaborn and matplotlib. Install them with "
            "`uv add --group dev seaborn matplotlib`."
        ) from error

    dataset_order = [*DATASETS, "avg"]
    data = pd.DataFrame.from_records(rows)
    palette = ["#4C78A8", "#59A14F", "#E15759", "#B07AA1"]
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    figure, axis = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    sns.barplot(
        data=data,
        x="Dataset",
        y="nDCG@10",
        hue="Method",
        order=dataset_order,
        hue_order=LABELS,
        palette=palette,
        edgecolor="white",
        linewidth=0.7,
        errorbar=None,
        ax=axis,
    )

    axis.set_xlabel("")
    axis.set_ylabel("nDCG@10")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(
        range(len(dataset_order)),
        labels=["COVID-QA", "Geography", "Insurance", "Football", "SQuAD", "NarrativeQA", "MLDR", "Average"],
    )
    axis.tick_params(axis="x", labelrotation=25)
    for tick in axis.get_xticklabels():
        tick.set_ha("right")
    axis.grid(axis="x", visible=False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(title="", ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the seven ConTEB nDCG@10 scores and their average for four methods."
    )
    parser.add_argument(
        "result_files",
        nargs="*",
        type=Path,
        metavar="RESULT_JSON",
        help=(
            "Optional JSON files in this order: chunk-only, LC, LC-left, LC-right. "
            "If omitted, use the hard-coded ModernBERT result files."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path (default: %(default)s).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI (default: %(default)s).")
    args = parser.parse_args()
    if not args.result_files:
        args.result_files = list(DEFAULT_RESULT_FILES)
    elif len(args.result_files) != len(LABELS):
        parser.error(f"expected either no result files or exactly {len(LABELS)} result files")
    return args


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    rows = build_rows(args.result_files)
    plot(rows, args.output, args.dpi)
    print(f"Saved nDCG@10 comparison plot to {args.output}")


if __name__ == "__main__":
    main()
