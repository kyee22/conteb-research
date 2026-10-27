"""Plot query/gold and within-document distractor similarity distributions.

This experiment uses the same model and dataset interfaces as
``motivation_experiments.py``. For one embedder it creates one figure comparing
gold and within-document distractor similarities across the datasets.

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
from scripts.motivation_experiments import _as_matrix, _cosine_matrix, _doc_id  # noqa: E402


def _embed_rank_dataset(formatter: DataFormatter, embedder: object):
    """Embed queries and chunks while preserving each query's gold ID list."""
    query_embeddings, gold_ids = embedder.process_queries(formatter)
    chunk_embeddings, chunk_ids = embedder.process_documents(formatter)
    query_embeddings = _as_matrix(query_embeddings)
    chunk_embeddings = _as_matrix(chunk_embeddings)
    gold_ids = [
        [str(gold_id) for gold_id in item] if isinstance(item, (list, tuple)) else str(item)
        for item in gold_ids
    ]
    chunk_ids = [str(chunk_id) for chunk_id in chunk_ids]
    if query_embeddings.shape[0] != len(gold_ids):
        raise ValueError("Number of query embeddings does not match query gold IDs")
    if chunk_embeddings.shape[0] != len(chunk_ids):
        raise ValueError("Number of chunk embeddings does not match chunk IDs")
    return query_embeddings, gold_ids, chunk_embeddings, chunk_ids


def _dataset_statistics(
    query_embeddings: torch.Tensor,
    gold_ids: list[str | list[str]],
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
    similarity_margins: list[float] = []
    within_doc_ranks: list[float] = []
    all_chunk_ranks: list[float] = []
    for query_scores, query_gold_ids in zip(scores, gold_ids, strict=True):
        pair_gold_ids = query_gold_ids if isinstance(query_gold_ids, list) else [query_gold_ids]
        for gold_id in pair_gold_ids:
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
                distractor_score = float(query_scores[other_same_doc].max())
                distractor_similarities.append(distractor_score)
                similarity_margins.append(gold_score - distractor_score)
                within_ahead = sum(float(query_scores[index]) > gold_score for index in other_same_doc)
                within_doc_ranks.append(within_ahead / len(other_same_doc))
            else:
                within_doc_ranks.append(0.0)

            all_ahead = sum(float(score) > gold_score for score in query_scores)
            all_chunk_ranks.append(all_ahead / max(len(chunk_ids) - 1, 1))

    return {
        "gold_similarity": gold_similarities,
        "within_document_max_other_similarity": distractor_similarities,
        "gold_minus_distractor_similarity": similarity_margins,
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


def _kde_curve(samples: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimate a Gaussian KDE using Silverman's rule-of-thumb bandwidth."""
    bandwidth = 1.06 * samples.std() * samples.size ** (-0.2)
    if bandwidth <= 1e-12:
        return np.zeros_like(x)
    scaled = (x[:, None] - samples[None, :]) / bandwidth
    return np.exp(-0.5 * scaled**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def _plot_distributions(
    values: dict[str, dict[str, list[float]]], output: Path, similarity_bins: int
) -> None:
    """Plot similarity histograms, KDE curves, means, and variances by dataset."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
    similarity_colors = ("#4C78A8", "#F58518")
    similarity_labels = ("gold chunk", "max other chunk in document")

    for dataset_index, dataset in enumerate(DATASETS):
        row = dataset_index // 4
        column = dataset_index % 4
        dataset_values = values[dataset]
        axis = axes[row, column]
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
            samples = np.asarray(dataset_values[key], dtype=float)
            if samples.size == 0:
                continue
            mean = float(samples.mean())
            variance = float(samples.var())
            legend_label = f"{label} (mean={mean:.3f}, var={variance:.4f})"
            axis.hist(
                samples, bins=edges, density=True, alpha=0.22,
                color=color, edgecolor=color, linewidth=0.8,
            )
            curve_x = np.linspace(float(edges[0]), float(edges[-1]), 300)
            if samples.size > 1 and variance > 1e-12:
                axis.plot(
                    curve_x, _kde_curve(samples, curve_x),
                    color=color, linewidth=1.8, label=legend_label,
                )
            else:
                axis.plot([], [], color=color, linewidth=1.8, label=legend_label)
            axis.axvline(mean, color=color, linestyle="--", linewidth=1.2, label="_nolegend_")

        axis.set_title(dataset, fontsize=11)
        axis.set_xlabel("cosine similarity")
        axis.set_ylabel("density")
        axis.legend(frameon=False, fontsize=7)

    fig.suptitle("Query-to-chunk cosine similarity", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_margin_distributions(
    values: dict[str, dict[str, list[float]]], output: Path, bins: int
) -> None:
    """Plot gold-minus-distractor similarity margins across datasets."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
    color = "#54A24B"

    for dataset_index, dataset in enumerate(DATASETS):
        axis = axes[dataset_index // 4, dataset_index % 4]
        margins = np.asarray(values[dataset]["gold_minus_distractor_similarity"], dtype=float)
        axis.set_title(dataset, fontsize=11)
        axis.set_xlabel("gold similarity - max distractor similarity")
        axis.set_ylabel("density")
        axis.axvline(0.0, color="#E45756", linestyle="--", linewidth=1.2, label="equal similarity")
        if margins.size == 0:
            axis.text(0.5, 0.5, "no within-document distractors", transform=axis.transAxes,
                      ha="center", va="center", fontsize=8)
            axis.legend(frameon=False, fontsize=7)
            continue

        edges = _histogram_edges(margins.tolist(), bins)
        mean = float(margins.mean())
        variance = float(margins.var())
        win_rate = float((margins > 0).mean())
        axis.hist(margins, bins=edges, density=True, color=color, alpha=0.24,
                  edgecolor=color, linewidth=0.8)
        curve_x = np.linspace(float(edges[0]), float(edges[-1]), 300)
        if margins.size > 1 and variance > 1e-12:
            axis.plot(curve_x, _kde_curve(margins, curve_x), color=color, linewidth=1.8,
                      label=f"KDE (mean={mean:.3f}, var={variance:.4f}, win={win_rate:.1%})")
        else:
            axis.plot([], [], color=color, linewidth=1.8,
                      label=f"mean={mean:.3f}, var={variance:.4f}, win={win_rate:.1%}")
        axis.axvline(mean, color=color, linestyle=":", linewidth=1.2, label="mean")
        axis.legend(frameon=False, fontsize=7)

    fig.suptitle("Gold minus strongest within-document distractor similarity", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        queries, gold_ids, chunks, chunk_ids = _embed_rank_dataset(formatter, embedder)
        values[dataset] = _dataset_statistics(queries, gold_ids, chunks, chunk_ids)
        (output_dir / "progress.json").write_text(
            json.dumps({"completed_datasets": list(values)}, indent=2), encoding="utf-8"
        )

    (output_dir / "statistics.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    _plot_distributions(
        values,
        output_dir / "motivation_rank_distributions.svg",
        args.similarity_bins,
    )
    _plot_margin_distributions(
        values,
        output_dir / "motivation_rank_margin_distributions.svg",
        args.similarity_bins,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder-model", required=True)
    parser.add_argument("--embedder-type", required=True)
    parser.add_argument("--data-base-path", default="illuin-conteb")
    parser.add_argument("--output-dir", default="motivation-rank-results")
    parser.add_argument("--similarity-bins", type=int, default=20)
    parser.add_argument("--use-prefix", action="store_true")
    parser.add_argument("--extend-context", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
