"""Plot query/gold similarity and gold-chunk rank distributions.

This experiment uses the same model and dataset interfaces as
``motivation_experiments.py``.  For one embedder it creates one six-by-four
figure: the first two rows compare gold and within-document distractor
similarities, the next two rows show within-document ranks, and the last two
rows show ranks among all chunks in the dataset.

Run from ``conteb-research/conteb`` with, for example::

    python scripts/motivation_rank_experiments.py \
        --embedder-model sentence-transformers/all-MiniLM-L6-v2 \
        --embedder-type sentence_transformer \
        --output-dir motivation-rank-results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_CONTEB_ROOT = Path(__file__).resolve().parents[1]
if str(_CONTEB_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONTEB_ROOT))

from cde_benchmark.formatters.data_formatter import DataFormatter  # noqa: E402
from scripts.evaluation.eval_conteb import DATASETS, get_model, get_split  # noqa: E402
from scripts.motivation_experiments import _cosine_matrix, _doc_id, _embed_dataset  # noqa: E402


def _dataset_statistics(
    query_embeddings: torch.Tensor,
    gold_ids: list[str],
    chunk_embeddings: torch.Tensor,
    chunk_ids: list[str],
) -> dict[str, list[float]]:
    """Return the three requested per-query distributions for one dataset."""
    scores = _cosine_matrix(query_embeddings, chunk_embeddings)
    id_to_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    by_doc: dict[str, list[int]] = {}
    for index, chunk_id in enumerate(chunk_ids):
        by_doc.setdefault(_doc_id(chunk_id), []).append(index)

    gold_similarities: list[float] = []
    distractor_similarities: list[float] = []
    within_doc_ranks: list[float] = []
    all_chunk_ranks: list[float] = []
    for query_scores, gold_id in zip(scores, gold_ids, strict=True):
        if gold_id not in id_to_index:
            raise ValueError(f"Gold chunk {gold_id!r} is absent from the document chunks")
        gold_index = id_to_index[gold_id]
        gold_score = float(query_scores[gold_index])
        gold_doc = _doc_id(gold_id)
        same_doc = by_doc[gold_doc]
        other_same_doc = [index for index in same_doc if index != gold_index]
        gold_similarities.append(gold_score)

        # A one-chunk document has no within-document distractor.  Its rank is
        # defined as the best possible normalized rank (zero).
        if other_same_doc:
            distractor_similarities.append(float(query_scores[other_same_doc].max()))
            within_ahead = sum(float(query_scores[index]) > gold_score for index in other_same_doc)
            within_doc_ranks.append(within_ahead / len(other_same_doc))
        else:
            within_doc_ranks.append(0.0)

        all_ahead = sum(float(score) > gold_score for score in query_scores)
        all_chunk_ranks.append(all_ahead / max(len(chunk_ids) - 1, 1))

    return {
        "gold_similarity": gold_similarities,
        "within_document_max_other_similarity": distractor_similarities,
        "within_document_rank": within_doc_ranks,
        "all_chunk_rank": all_chunk_ranks,
    }


def _histogram_edges(values: list[float], bins: int) -> np.ndarray:
    if not values:
        return np.linspace(0.0, 1.0, bins + 1)
    low, high = min(values), max(values)
    if np.isclose(low, high):
        padding = max(abs(low) * 0.01, 0.01)
        low -= padding
        high += padding
    return np.linspace(low, high, bins + 1)


def _plot_distributions(
    values: dict[str, dict[str, list[float]]], output: Path, similarity_bins: int, rank_bins: int
) -> None:
    """Create the requested three groups of eight dataset histograms."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(6, 4, figsize=(16, 21), squeeze=False)
    similarity_colors = ("#4C78A8", "#F58518")
    similarity_labels = ("gold chunk", "max other chunk in document")
    rank_edges = np.linspace(0.0, 1.0, rank_bins + 1)
    group_titles = (
        "Query-to-chunk cosine similarity",
        "Gold rank within its document",
        "Gold rank among all dataset chunks",
    )

    for dataset_index, dataset in enumerate(DATASETS):
        row = (dataset_index // 4)
        column = dataset_index % 4
        dataset_values = values[dataset]

        similarity_axis = axes[row, column]
        similarity_values = dataset_values["gold_similarity"] + dataset_values[
            "within_document_max_other_similarity"
        ]
        edges = _histogram_edges(similarity_values, similarity_bins)
        for key, color, label in zip(
            ("gold_similarity", "within_document_max_other_similarity"),
            similarity_colors,
            similarity_labels,
            strict=True,
        ):
            # Density makes both distributions enclose unit area despite
            # differing sample counts, as requested.
            similarity_axis.hist(
                dataset_values[key], bins=edges, density=True, alpha=0.38,
                color=color, edgecolor=color, linewidth=1.0, label=label,
            )
        similarity_axis.set_title(dataset, fontsize=12)
        similarity_axis.set_ylabel("density")
        if dataset_index == 0:
            similarity_axis.legend(frameon=False, fontsize=8)

        within_axis = axes[2 + row, column]
        within_axis.hist(dataset_values["within_document_rank"], bins=rank_edges, density=True,
                         color="#54A24B", alpha=0.75, edgecolor="white")
        within_axis.set_title(dataset, fontsize=12)
        within_axis.set_xlim(0, 1)
        within_axis.set_ylabel("density")

        all_axis = axes[4 + row, column]
        all_axis.hist(dataset_values["all_chunk_rank"], bins=rank_edges, density=True,
                      color="#B279A2", alpha=0.75, edgecolor="white")
        all_axis.set_title(dataset, fontsize=12)
        all_axis.set_xlim(0, 1)
        all_axis.set_ylabel("density")

    for axis in axes[0]:
        axis.set_xlabel("cosine similarity")
    for axis in axes[2:4].ravel():
        axis.set_xlabel("fraction of other document chunks ranked ahead")
    for axis in axes[4:6].ravel():
        axis.set_xlabel("fraction of other dataset chunks ranked ahead")
    for row, title in zip((0, 2, 4), group_titles, strict=True):
        fig.text(0.01, 1 - (row + 1) / 6, title, rotation=90, va="center", ha="center", fontsize=14)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.985, bottom=0.045, hspace=0.48, wspace=0.25)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = get_model(args.embedder_model, False, args.embedder_type, args.use_prefix, args.extend_context)
    values: dict[str, dict[str, list[float]]] = {}
    for dataset in DATASETS:
        dataset_path = f"{args.data_base_path}/{dataset}"
        formatter = DataFormatter(dataset_path, get_split(dataset_path), query_key="queries")
        queries, gold_ids, chunks, chunk_ids = _embed_dataset(formatter, embedder)
        values[dataset] = _dataset_statistics(queries, gold_ids, chunks, chunk_ids)
        (output_dir / "progress.json").write_text(
            json.dumps({"completed_datasets": list(values)}, indent=2), encoding="utf-8"
        )

    (output_dir / "statistics.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    _plot_distributions(
        values,
        output_dir / "motivation_rank_distributions.svg",
        args.similarity_bins,
        args.rank_bins,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder-model", required=True)
    parser.add_argument("--embedder-type", required=True)
    parser.add_argument("--data-base-path", default="illuin-conteb")
    parser.add_argument("--output-dir", default="motivation-rank-results")
    parser.add_argument("--similarity-bins", type=int, default=20)
    parser.add_argument("--rank-bins", type=int, default=20)
    parser.add_argument("--use-prefix", action="store_true")
    parser.add_argument("--extend-context", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
