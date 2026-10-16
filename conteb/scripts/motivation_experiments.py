"""Motivation experiments for contextual versus independent chunk encoding.

The script intentionally uses the benchmark's public Embedder/DataFormatter
interfaces.  Run it from the repository root, for example:

    python conteb/scripts/motivation_experiments.py \
        --embedder1-model sentence-transformers/all-MiniLM-L6-v2 \
        --embedder1-type sentence_transformer \
        --embedder2-model /path/to/contextual-checkpoint \
        --embedder2-type contextual \
        --use-prefix \
        --output-dir conteb/motivation-results

The model type names and model construction are the same as eval_conteb.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

_CONTEB_ROOT = Path(__file__).resolve().parents[1]
if str(_CONTEB_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONTEB_ROOT))

from cde_benchmark.formatters.data_formatter import DataFormatter
from scripts.evaluation.eval_conteb import DATASETS, get_model, get_split


# The model imports and get_model() implementation used by eval_conteb.py are
# intentionally kept here as a copyable reference.  The active implementation
# above imports them from eval_conteb.py so model loading remains centralized.
#
# from huggingface_hub import HfApi, hf_hub_download
# from sentence_transformers import SentenceTransformer
# from contextual_embeddings import HyperContextEmbeddingModel
# from cde_benchmark.embedders.base_embedder import Embedder
# from cde_benchmark.embedders.colbert_embedder import ColBERTEmbedder
# from cde_benchmark.embedders.contextual_embedder import (
#     ContextualEmbedder, LateInteractionContextualEmbedder,
#     MergedContextualEmbedder, PPLXContextualEmbedder,
# )
# from cde_benchmark.embedders.flex_contextual_embedder import FlexContextualEmbedder
# from cde_benchmark.embedders.hyper_contextual_embedder import HyperContextualEmbedder
# from cde_benchmark.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
# from cde_benchmark.embedders.sitemb_embedder import (
#     Qwen3Embedder, SitEmbChunkOnlyEmbedder, SitEmbContextualEmbedder,
# )
# from cde_benchmark.embedders.qwen3_late_chunking_embedder import Qwen3LateChunkingEmbedder
# from cde_benchmark.embedders.voyage_contextual_embedder import VoyageContextualEmbedder
#
# Copyable model loader from conteb/scripts/evaluation/eval_conteb.py. The
# corresponding imports above are also kept here for direct reuse.
#
# def get_model(
#     model_path: str, colbert_model: bool, model_type: str, use_prefix: bool, extend_context: bool
# ) -> Embedder:
#     if model_type == 'pplx_contextual':
#         return PPLXContextualEmbedder(model_path, add_prefix=use_prefix, device=DEVICE)
#     if model_type in {'sit_chunk_only', 'sitemb_chunk_only'}:
#         return SitEmbChunkOnlyEmbedder(
#             adapter_name=model_path, device=DEVICE,
#             torch_dtype=torch.bfloat16 if DEVICE.startswith('cuda') else None,
#         )
#     if model_type in {'sit_contextual', 'sitemb_contextual'}:
#         return SitEmbContextualEmbedder(
#             adapter_name=model_path, device=DEVICE,
#             torch_dtype=torch.bfloat16 if DEVICE.startswith('cuda') else None,
#         )
#     if model_type == 'qwen3':
#         return Qwen3Embedder(
#             model_path=model_path, device=DEVICE,
#             torch_dtype=torch.bfloat16 if DEVICE.startswith('cuda') else None,
#         )
#     if model_type in {'qwen3_late', 'qwen3_late_chunking'}:
#         return Qwen3LateChunkingEmbedder.from_checkpoint(model_path, device=DEVICE)
#     if model_type == 'voyage':
#         return VoyageContextualEmbedder(model_name=model_path, device=DEVICE)
#     if colbert_model:
#         raise NotImplementedError('pylate is temporarily unavailable')
#     if model_type in {'hyper_contextual', 'hypernet_contextual'}:
#         model = HyperContextEmbeddingModel.from_checkpoint(model_path, add_prefix=use_prefix)
#         return HyperContextualEmbedder(model, device=DEVICE)
#     if model_type in {'flex_contextual', 'flex'}:
#         embedder = FlexContextualEmbedder.from_checkpoint(
#             model_path, add_prefix=use_prefix, device=DEVICE,
#         )
#         if extend_context:
#             embedder.base_model.max_seq_length = sys.maxsize
#             embedder.base_model.tokenizer.model_max_length = sys.maxsize
#         return embedder
#
#     base_model = SentenceTransformer(model_path, trust_remote_code=True, device=DEVICE)
#     if extend_context:
#         base_model.max_seq_length = sys.maxsize
#         base_model.tokenizer.model_max_length = sys.maxsize
#     match model_type:
#         case 'contextual':
#             return ContextualEmbedder(base_model, add_prefix=use_prefix, device=DEVICE)
#         case 'merged_contextual':
#             return MergedContextualEmbedder(base_model, add_prefix=use_prefix, device=DEVICE)
#         case 'late_interaction' | 'li':
#             return LateInteractionContextualEmbedder(base_model, add_prefix=use_prefix, device=DEVICE)
#         case 'sentence_transformer':
#             return SentenceTransformerEmbedder(
#                 base_model, add_prefix=use_prefix, batch_size=32, device=DEVICE,
#                 normalize_embeddings=True,
#             )
#         case 'colbert':
#             batch_size = 1 if extend_context else 16
#             return ColBERTEmbedder(base_model, batch_size=batch_size, device=DEVICE)
#         case _:
#             raise ValueError(f'Unknown model type: {model_type}')


def _flatten_embeddings(value: Any) -> list[Any]:
    """Flatten one or more container levels while preserving embedding items."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return [value]
        return [row for row in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return [value]
        return [row for row in value]
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)) and not _is_vector(item):
                result.extend(_flatten_embeddings(item))
            else:
                result.append(item)
        return result
    return [value]


def _is_vector(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.ndim == 1
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    if isinstance(value, (list, tuple)):
        return bool(value) and all(np.isscalar(x) for x in value)
    return False


def _as_matrix(embeddings: Any) -> torch.Tensor:
    rows = _flatten_embeddings(embeddings)
    tensors = [torch.as_tensor(row).detach().float().cpu() for row in rows]
    if not tensors:
        raise ValueError("The embedder returned no embeddings")
    try:
        return torch.stack(tensors)
    except RuntimeError as exc:
        raise ValueError("The embedder must return one fixed-size vector per item") from exc


def _cosine_matrix(queries: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
    queries = torch.nn.functional.normalize(queries.float(), dim=1)
    chunks = torch.nn.functional.normalize(chunks.float(), dim=1)
    return queries @ chunks.T


def _doc_id(chunk_id: str) -> str:
    return str(chunk_id).split("_", 1)[0]


def _embed_dataset(formatter: DataFormatter, embedder: Any) -> tuple[torch.Tensor, list[str], torch.Tensor, list[str]]:
    query_embeddings, gold_ids = embedder.process_queries(formatter)
    chunk_embeddings, chunk_ids = embedder.process_documents(formatter)
    query_embeddings = _as_matrix(query_embeddings)
    chunk_embeddings = _as_matrix(chunk_embeddings)
    gold_ids = [str(item) for item in gold_ids]
    chunk_ids = [str(item) for item in chunk_ids]
    if query_embeddings.shape[0] != len(gold_ids):
        raise ValueError("Number of query embeddings does not match gold chunk ids")
    if chunk_embeddings.shape[0] != len(chunk_ids):
        raise ValueError("Number of chunk embeddings does not match chunk ids")
    return query_embeddings, gold_ids, chunk_embeddings, chunk_ids


def _internal_similarity_values(chunk_embeddings: torch.Tensor, chunk_ids: list[str]) -> list[float]:
    by_doc: dict[str, list[int]] = {}
    for index, chunk_id in enumerate(chunk_ids):
        by_doc.setdefault(_doc_id(chunk_id), []).append(index)
    values: list[float] = []
    normalized = torch.nn.functional.normalize(chunk_embeddings.float(), dim=1)
    for indexes in by_doc.values():
        if len(indexes) < 2:
            continue
        matrix = normalized[indexes] @ normalized[indexes].T
        upper = matrix[torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)]
        values.append(float(upper.mean()))
    return values


def _ranking_errors(
    query_embeddings: torch.Tensor,
    gold_ids: list[str],
    chunk_embeddings: torch.Tensor,
    chunk_ids: list[str],
) -> list[tuple[int, int]]:
    scores = _cosine_matrix(query_embeddings, chunk_embeddings)
    id_to_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    errors: list[tuple[int, int]] = []
    for query_scores, gold_id in zip(scores, gold_ids, strict=True):
        if gold_id not in id_to_index:
            raise ValueError(f"Gold chunk {gold_id!r} is absent from the document chunks")
        gold_index = id_to_index[gold_id]
        gold_score = query_scores[gold_index]
        gold_doc = _doc_id(gold_id)
        ahead = torch.nonzero(query_scores > gold_score, as_tuple=False).flatten().tolist()
        type_i = sum(_doc_id(chunk_ids[index]) != gold_doc for index in ahead)
        type_ii = sum(_doc_id(chunk_ids[index]) == gold_doc for index in ahead)
        errors.append((type_i, type_ii))
    return errors


def _plot_similarity_histograms(dataset_values: dict[str, dict[str, list[float]]], output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    # Each dataset gets its own x/y limits so narrow distributions remain
    # visible even when another dataset has a much broader histogram.
    fig, axes = plt.subplots(
        2, 4, figsize=(14, 7.5), sharex=False, sharey=False, squeeze=False
    )
    axes = axes.ravel()
    colors = {"embedder1": "#7895B2", "embedder2": "#C58B8B"}
    labels = {"embedder1": "independent", "embedder2": "contextual"}
    for axis, dataset in zip(axes[: len(DATASETS)], DATASETS, strict=True):
        values = [
            value
            for key in ("embedder1", "embedder2")
            for value in dataset_values[dataset][key]
        ]
        bin_edges = np.histogram_bin_edges(values, bins=20)
        for key in ("embedder1", "embedder2"):
            sns.histplot(
                dataset_values[dataset][key], bins=bin_edges, stat="count", element="step",
                fill=True, alpha=0.38, color=colors[key], label=labels[key], ax=axis,
            )
        axis.set_title(dataset, fontsize=15)
        axis.grid(alpha=0.28)
    for axis in axes[len(DATASETS) :]:
        axis.set_visible(False)
    axes[0].set_ylabel("Documents", fontsize=15)
    fig.supxlabel("Mean pairwise within-document cosine similarity", fontsize=16)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    for axis in axes[: len(DATASETS)]:
        axis.tick_params(axis="both", labelsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_sampled_errors(
    sampled: dict[str, list[tuple[str, int, int, int, int]]], output: Path, sample_size: int
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    # Give every dataset panel room for its rotated query IDs.  A symmetric-log
    # y axis keeps the small error counts legible while fitting rare large
    # counts into the same panel; ticks still show the original error counts.
    fig, axes = plt.subplots(
        len(DATASETS), 1,
        figsize=(max(15, sample_size * 0.42), 4.6 * len(DATASETS)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)
    colors = {"embedder1": ("#C9D8E5", "#7895B2"), "embedder2": ("#E7C7C7", "#C58B8B")}
    for axis, dataset in zip(axes, DATASETS, strict=True):
        rows = sampled[dataset]
        positions = np.arange(len(rows))
        width = 0.38
        for offset, key in ((-width / 2, "embedder1"), (width / 2, "embedder2")):
            i_values = [row[1 if key == "embedder1" else 3] for row in rows]
            ii_values = [row[2 if key == "embedder1" else 4] for row in rows]
            axis.bar(positions + offset, ii_values, width, color=colors[key][0], label=f"{key} II")
            axis.bar(positions + offset, i_values, width, bottom=ii_values, color=colors[key][1], label=f"{key} I")
        axis.set_title(dataset, fontsize=15)
        axis.set_yscale("symlog", linthresh=5, linscale=1)
        axis.set_ylabel("errors\n(symlog)", fontsize=14)
        axis.grid(axis="y", alpha=0.28)
        if rows:
            axis.set_xticks(positions)
            axis.set_xticklabels([row[0] for row in rows], fontsize=9, rotation=90, ha="center")
            axis.tick_params(axis="y", labelsize=11)
            axis.tick_params(axis="x", pad=4)
    axes[0].legend(frameon=False, ncol=4, fontsize=11)
    axes[-1].set_xlabel("Query ID (gold chunk_id)", fontsize=14)
    fig.subplots_adjust(hspace=0.9, left=0.065, right=0.995, top=0.985, bottom=0.045)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_analysis(
    errors: dict[str, dict[str, list[tuple[int, int]]]], output: Path
) -> None:
    rows = [
        # Equality is assigned to the decrease side (<=).  We first choose
        # the total-error branch, then classify the two components; this makes
        # the six rows mutually exclusive and exhaustive, including ties.
        ("总错误减少，I减少，II减少", lambda a, b: sum(b) <= sum(a) and b[0] <= a[0] and b[1] <= a[1]),
        ("总错误减少，I减少，II增加", lambda a, b: sum(b) <= sum(a) and b[0] <= a[0] and b[1] > a[1]),
        ("总错误减少，I增加，II减少", lambda a, b: sum(b) <= sum(a) and b[0] > a[0] and b[1] <= a[1]),
        ("总错误增加，I减少，II增加", lambda a, b: sum(b) > sum(a) and b[0] <= a[0] and b[1] > a[1]),
        ("总错误增加，I增加，II减少", lambda a, b: sum(b) > sum(a) and b[0] > a[0] and b[1] <= a[1]),
        ("总错误增加，I增加，II增加", lambda a, b: sum(b) > sum(a) and b[0] > a[0] and b[1] > a[1]),
    ]
    lines = ["|变化类型|" + "|".join(DATASETS) + "|", "|---|" + "---|" * len(DATASETS)]
    for label, predicate in rows:
        cells = []
        for dataset in DATASETS:
            before = errors[dataset]["embedder1"]
            after = errors[dataset]["embedder2"]
            count = sum(predicate(a, b) for a, b in zip(before, after, strict=True))
            total = len(before)
            cells.append(f"{count}<br>({100 * count / total:.2f}%)" if total else "0<br>(0.00%)")
        lines.append("|" + label + "|" + "|".join(cells) + "|")
    for dataset in DATASETS:
        before = errors[dataset]["embedder1"]
        after = errors[dataset]["embedder2"]
        classified = sum(
            predicate(a, b)
            for predicate in (row[1] for row in rows)
            for a, b in zip(before, after, strict=True)
        )
        assert classified == len(before), f"Six error classes do not cover all queries in {dataset}"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder1 = get_model(args.embedder1_model, False, args.embedder1_type, args.use_prefix, args.extend_context)
    embedder2 = get_model(args.embedder2_model, False, args.embedder2_type, args.use_prefix, args.extend_context)
    similarities: dict[str, dict[str, list[float]]] = {}
    errors: dict[str, dict[str, list[tuple[int, int]]]] = {}
    sampled: dict[str, list[tuple[str, int, int, int, int]]] = {}
    for dataset in DATASETS:
        formatter = DataFormatter(f"{args.data_base_path}/{dataset}", get_split(f"{args.data_base_path}/{dataset}"), query_key="queries")
        dataset_values: dict[str, list[float]] = {}
        dataset_errors: dict[str, list[tuple[int, int]]] = {}
        for key, embedder in (("embedder1", embedder1), ("embedder2", embedder2)):
            queries, gold_ids, chunks, chunk_ids = _embed_dataset(formatter, embedder)
            dataset_values[key] = _internal_similarity_values(chunks, chunk_ids)
            dataset_errors[key] = _ranking_errors(queries, gold_ids, chunks, chunk_ids)
        similarities[dataset] = dataset_values
        errors[dataset] = dataset_errors
        count = min(args.sample_queries, len(dataset_errors["embedder1"]))
        indexes = sorted(random.sample(range(len(dataset_errors["embedder1"])), count))
        query_ids = [str(query_id) for query_id in formatter.queries_dataset["chunk_id"]]
        sampled[dataset] = [
            (
                query_ids[index],
                *dataset_errors["embedder1"][index],
                *dataset_errors["embedder2"][index],
            )
            for index in indexes
        ]
        progress_path = output_dir / "progress.json"
        progress_path.write_text(
            json.dumps({"completed_datasets": list(similarities), "seed": args.seed}, indent=2),
            encoding="utf-8",
        )
    _plot_similarity_histograms(similarities, output_dir / "document_internal_similarity.svg")
    _plot_sampled_errors(sampled, output_dir / "ranking_error_sample.svg", args.sample_queries)
    _write_analysis(errors, output_dir / "ranking_error_analysis.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder1-model", required=True)
    parser.add_argument("--embedder1-type", required=True)
    parser.add_argument("--embedder2-model", required=True)
    parser.add_argument("--embedder2-type", required=True)
    parser.add_argument("--data-base-path", default="illuin-conteb")
    parser.add_argument("--output-dir", default="conteb/motivation-results")
    parser.add_argument(
        "--sample-queries", type=int, default=40
    )
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--use-prefix", action="store_true")
    parser.add_argument("--extend-context", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
