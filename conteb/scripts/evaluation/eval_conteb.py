import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Support running this file directly from a sibling project, e.g.
# ``cd contextual-embeddings && uv run ../conteb/scripts/evaluation/eval_conteb.py``.
# In that case uv installs the contextual-embeddings project, but does not add
# the conteb source tree (which contains cde_benchmark) to sys.path.
_CONTEB_ROOT = Path(__file__).resolve().parents[2]
if str(_CONTEB_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONTEB_ROOT))

import torch
import typer
from huggingface_hub import HfApi, hf_hub_download
# from pylate.models import ColBERT
from sentence_transformers import SentenceTransformer
from contextual_embeddings import HyperContextEmbeddingModel
from contextual_embeddings import HyperContextEmbeddingModel
from datasets import get_dataset_config_names, get_dataset_split_names

from cde_benchmark.embedders.base_embedder import Embedder
from cde_benchmark.embedders.colbert_embedder import ColBERTEmbedder
from cde_benchmark.embedders.contextual_embedder import (
    ContextualEmbedder,
    LateInteractionContextualEmbedder,
    MergedContextualEmbedder,
    PPLXContextualEmbedder,
)
from cde_benchmark.embedders.flex_contextual_embedder import FlexContextualEmbedder
from cde_benchmark.embedders.hyper_contextual_embedder import HyperContextualEmbedder
from cde_benchmark.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from cde_benchmark.embedders.sitemb_embedder import (
    Qwen3Embedder,
    SitEmbChunkOnlyEmbedder,
    SitEmbContextualEmbedder,
)
from cde_benchmark.embedders.qwen3_late_chunking_embedder import (
    Qwen3BOTEmbedder,
    Qwen3LateChunkingEmbedder,
)
from cde_benchmark.embedders.voyage_contextual_embedder import VoyageContextualEmbedder
from cde_benchmark.evaluators.longembed import LongEmbedEvaluator
from cde_benchmark.evaluators.nanobeir import NanoBEIR
from cde_benchmark.formatters.data_formatter import DataFormatter

DATASETS = [
       "geography",
       "covid-qa",
 
    # "esg-reports",
    "insurance",
    "football",
    "squad-conteb-eval",
    "narrative-qa",
    "mldr-conteb-eval",
]

SQUAD_CHUNK_SIZES = [8000, 1000, 500, 300, 250, 200, 175, 150, 125, 100]
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def get_model(
    model_path: str, colbert_model: bool, model_type: str, use_prefix: bool, extend_context: bool
) -> Embedder:
    """
    Get the embedding model based on the type passed by the user.
    Args:
        base_model (SentenceTransformer): The base model to use.
        model_type (str): The type of model to use. Can be one of
          "contextual", "flex_contextual", "hyper_contextual", "merged_contextual", "late_interaction", or
          "sentence_transformer".
        use_prefix (bool): Whether to add prefixes to the chunks and queries.
    """
    if model_type == 'pplx_contextual':
        return PPLXContextualEmbedder(model_path, 
                                      add_prefix=use_prefix,
                                      device=DEVICE)
    if model_type == 'sit_chunk_only':
        return SitEmbChunkOnlyEmbedder(
            adapter_name=model_path,
            device=DEVICE,
            torch_dtype=torch.bfloat16 if DEVICE.startswith("cuda") else None,
        )
    if model_type == 'sit_contextual': 
        return SitEmbContextualEmbedder(
            adapter_name=model_path,
            device=DEVICE,
            torch_dtype=torch.bfloat16 if DEVICE.startswith("cuda") else None,
        )

    if model_type == "sitemb_chunk_only":
        return SitEmbChunkOnlyEmbedder(
            adapter_name=model_path,
            device=DEVICE,
            torch_dtype=torch.bfloat16 if DEVICE.startswith("cuda") else None,
        )

    if model_type == "sitemb_contextual":
        return SitEmbContextualEmbedder(
            adapter_name=model_path,
            device=DEVICE,
            torch_dtype=torch.bfloat16 if DEVICE.startswith("cuda") else None,
        )

    if model_type == "qwen3":
        return Qwen3Embedder(
            model_path=model_path,
            device=DEVICE,
            torch_dtype=torch.bfloat16 if DEVICE.startswith("cuda") else None,
        )
    if model_type in {"qwen3_late", "qwen3_late_chunking"}:
        return Qwen3LateChunkingEmbedder.from_checkpoint(model_path, device=DEVICE)
    if model_type in {"qwen3_bot", "qwen3_bot_late", "qwen3_bot_late_chunking"}:
        return Qwen3BOTEmbedder.from_checkpoint(model_path, device=DEVICE)

    if model_type == 'voyage':
        return VoyageContextualEmbedder(
            model_name=model_path,
            device=DEVICE,
        )

    if colbert_model:
        # Load the ColBERT model
        raise NotImplementedError("pylate is temporarily unavailable")
        # if extend_context:
        #     base_model = ColBERT(model_path, device=DEVICE, document_length=sys.maxsize)
        # else:
        #     base_model = ColBERT(model_path, device=DEVICE)
    if model_type in {"hyper_contextual", "hypernet_contextual"}:
        model = HyperContextEmbeddingModel.from_checkpoint(model_path, add_prefix=use_prefix)
        print(f"Loaded hypernetwork checkpoint from {model_path}")
        return HyperContextualEmbedder(model, device=DEVICE)

    # Flex checkpoints keep replace_local_with/replace_full_with in
    # training_config.yml; load through the adapter so the CLI does not lose
    # those settings by constructing a plain SentenceTransformer first.
    if model_type in {"flex_contextual", "flex"}:
        embedder = FlexContextualEmbedder.from_checkpoint(
            model_path,
            add_prefix=use_prefix,
            device=DEVICE,
        )
        if extend_context:
            embedder.base_model.max_seq_length = sys.maxsize
            embedder.base_model.tokenizer.model_max_length = sys.maxsize
        print(f"Loaded Flex contextual checkpoint from {model_path}")
        return embedder

    else:
        base_model = SentenceTransformer(model_path, trust_remote_code=True, device=DEVICE)
        print(f"Load base model @ {model_path}")

    if extend_context:
        base_model.max_seq_length = sys.maxsize
        base_model.tokenizer.model_max_length = sys.maxsize

    match model_type:
        case "contextual":
            return ContextualEmbedder(base_model, add_prefix=use_prefix, device=DEVICE)
        case "merged_contextual":
            return MergedContextualEmbedder(base_model, add_prefix=use_prefix, device=DEVICE)
        case "late_interaction" | "li":
            return LateInteractionContextualEmbedder(base_model, add_prefix=use_prefix, device=DEVICE)
        case "sentence_transformer":
            return SentenceTransformerEmbedder(base_model, add_prefix=use_prefix, batch_size=32, device=DEVICE,
                                               normalize_embeddings=True)
        case "colbert":
            if extend_context:
                batch_size = 1
            else:
                batch_size = 16
            return ColBERTEmbedder(base_model, batch_size=batch_size, device=DEVICE)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def get_split(dataset_id):
    """
    Get the split for the dataset based on availability.
    Tries "test", then "validation", then "train".
    """
    try:
        splits = get_dataset_split_names(dataset_id)
    except ValueError:
        # Dataset may require a config name (e.g., multiple configs like "documents", "queries")
        configs = get_dataset_config_names(dataset_id)
        if not configs:
            raise ValueError(f"No configs found for dataset {dataset_id}")
        # Try the first available config
        splits = get_dataset_split_names(dataset_id, configs[0])
    if "test" in splits:
        return "test"
    elif "validation" in splits:
        return "validation"
    elif "train" in splits:
        return "train"
    else:
        raise ValueError(f"No split found for dataset {dataset_id}")


def eval_model(
    model_path: str,
    model_type: str,
    model_name: str,
    colbert_model: bool = False,
    datasets: List[str] = DATASETS,
    run_squad_chunked: bool = False,
    run_nanobeir: bool = False,
    run_longembed: bool = False,
    use_prefix: bool = False,
    data_base_path: str = "illuin-conteb",
    save_dir: str = "results-qwen3-gather",
    extend_context: bool = False,
):
    embedder = get_model(model_path, colbert_model, model_type, use_prefix, extend_context)

    output_dic = {
        "model_path": model_path,
        "model_name": model_name,
        "model_type": model_type,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extended_context": extend_context,
        "use_prefix": use_prefix,
        "colbert_model": colbert_model,
    }

    metrics = {}

    # Evaluate on NanoBEIR (non-contextual task, 1 chunk per document)
    if run_nanobeir:
        print("Running NanoBEIR tasks...")
        nanobeir = NanoBEIR(f"zeta-alpha-ai", embedder)
        nanobeir_metrics = nanobeir.run_all_tasks()
        nanobeir_macro_average = nanobeir_metrics.pop("macro_average", {})
        nanobeir_average = nanobeir_metrics.pop("average", {})
        metrics.update(nanobeir_metrics)
        print("NanoBEIR macro-average (equal task weighting):")
        metrics["NanoBEIR_macro_average"] = nanobeir_macro_average
        for name, value in nanobeir_macro_average.items():
            print(f"\t{name}: {value}")
        print("NanoBEIR pooled average (query weighting):")
        metrics["NanoBEIR_average"] = nanobeir_average
        for name, value in nanobeir_average.items():
            print(f"\t{name}: {value}")

    # Evaluate on ConTEB (contextual task, >1 chunks per document)
    for dataset in datasets:
        ds_path = f"{data_base_path}/{dataset}"
        split = get_split(ds_path)
        formatter = DataFormatter(ds_path, split, query_key="queries")

        ds_metrics = embedder.compute_metrics_e2e(formatter)
        metrics[dataset] = ds_metrics

        # pretty print all results in terminal
        print("=" * 20)
        print(f"Dataset: {dataset}, Split: {split}")    
        print("=" * 20)

        for k, v in ds_metrics.items():
            print(f"\t{k}: {v}")

    # intermediary saving
    output_dic["metrics"] = metrics

    # Save the metrics to a JSON file
    output_name = f"metrics_{'use_prefix' if use_prefix else 'no_prefix'}_{model_name}_{model_type}.json"
    output_path = f"{save_dir}/{output_name}"
    os.makedirs(save_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dic, f, indent=4)
    print(f"Metrics saved to {output_path}!")
    if run_squad_chunked:
        for chunk_size in SQUAD_CHUNK_SIZES:
            ds_path = f"{data_base_path}/squad-chunked-par-{chunk_size}"
            formatter = DataFormatter(ds_path, "validation", query_key="queries")
            ds_metrics = embedder.compute_metrics_e2e(formatter)
            metrics[f"squad-chunked-par{chunk_size}"] = ds_metrics

            # pretty print all results in terminal
            print("=" * 20)
            print(f"Dataset: squad-chunked-par-{chunk_size}")
            print("=" * 20)

            for k, v in ds_metrics.items():
                print(f"\t{k}: {v}")



    if run_longembed:
        print("Running LongEmbed tasks...")
        longembed = LongEmbedEvaluator(
            f"{data_base_path}/LongEmbed",
            embedder,
        )
        longembed_metrics = longembed.run_all_tasks()
        metrics.update(longembed_metrics)

    output_dic["metrics"] = metrics

    # Save the metrics to a JSON file
    output_name = f"metrics_{'use_prefix' if use_prefix else 'no_prefix'}_{model_name}_{model_type}.json"
    output_path = f"{save_dir}/{output_name}"
    os.makedirs(save_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dic, f, indent=4)
    print(f"Metrics saved to {output_path}!")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    typer.run(eval_model)
