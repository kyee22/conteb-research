#!/usr/bin/env python3
"""Compare independent and contextual token embeddings for one document.

The two paths use the same base model.  The independent path encodes every
chunk on its own; the contextual path encodes ``sep_token.join(chunks)`` once.
For chunks after the first, the independently encoded leading special token is
not part of the joined document and is therefore excluded from the comparison.

Example:
    python contextual-embeddings/playground/compare_contextual_token_embeddings.py \
        --model illuin-conteb/modernbert-large-insent \
        --chunk 'Ada designed the experiment.' \
        --chunk 'She recorded the result in a notebook.'
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
from sentence_transformers import SentenceTransformer
from torch.nn import functional

# Let the script run from either repository root or the playground directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTEB_ROOT = REPOSITORY_ROOT / "conteb"
CONTEXTUAL_EMBEDDINGS_ROOT = REPOSITORY_ROOT / "contextual-embeddings"
if str(CONTEB_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEB_ROOT))
if str(CONTEXTUAL_EMBEDDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEXTUAL_EMBEDDINGS_ROOT))

from cde_benchmark.embedders.contextual_embedder import ContextualEmbedder  # noqa: E402, I001
from cde_benchmark.embedders.sentence_transformer_embedder import (  # noqa: E402
    SentenceTransformerEmbedder,
)


DEFAULT_CHUNKS = [
    "The old lighthouse keeper trimmed his lamp, its beam cutting a lonely path through the fog.",
    "He remembered nights of violent storms, when the ocean seemed to swallow the sky whole.",
    "Still, he found comfort in his duty, a silent guardian against the treacherous sea."
]

# DEFAULT_CHUNKS = [
#     "A curious fox cub, all rust and wonder, ventured out from its den for the first time.",
#     "Each rustle of leaves, every chirping bird, was a new symphony to its tiny ears.",
#     "Under the watchful eye of its mother, it began to learn the secrets of the whispering forest."
# ]

DEFAULT_CHUNKS = [
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Austria\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000)³     | 8,979           | 2.00%          |\n| (Re)insurance GWP (in million)⁴ | 20,815.873      | 1.5%           |\n| Number of (re)insurance undertakings⁵ | 33              | 1.9%           |\n| Number of registered insurance intermediaries | 17,999          | 2.1%           |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Belgium\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000)⁹     | 11,618          | 2.6%           |\n| (Re)insurance GWP (in million)¹⁰ | 44,172.414       | 3.2%           |\n| Number of (re)insurance undertakings¹¹ | 57              | 3.3%           |\n| Number of registered insurance intermediaries | 8,911          | 1.1%           |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Bulgaria\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000) | 6,839 | 1.5% |\n| (Re)insurance GWP (in million) | 2,730.334 | 0.2% |\n| Number of (re)insurance undertakings | 30 | 1.7% |\n| Number of registered insurance intermediaries | 6,839 | 0.8% |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Croatia\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000) | 4,035.65 | 0.8% |\n| (Re)insurance GWP (in million) | 1,353 | 0.1% |\n| Number of (re)insurance undertakings | 23 | 1.7% |\n| Number of registered insurance intermediaries | 3,985 | 0.5% |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Cyprus\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000) | 904.705 | 0.2% |\n| (Re)insurance GWP (in million) | 1,217.479 | 0.09% |\n| Number of (re)insurance undertakings | 29 | 1.7% |\n| Number of registered insurance intermediaries | 2,322 | 0.3% |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Czechia\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000) | 10,517 | 2.3% |\n| (Re)insurance GWP (in million) | 8,386.338 | 0.6% |\n| Number of (re)insurance undertakings | 22 | 1.3% |\n| Number of registered insurance intermediaries | 32,448 | 3.7% |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Denmark\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000)      | 5,873            | 1.3%             |\n| (Re)insurance GWP (in million) | 40,744.612     | 3.0%             |\n| Number of (re)insurance undertakings | 72           | 4.2%             |\n| Number of registered insurance intermediaries | 2,571 | 0.3%             |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Estonia\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000)            | 1,332            | 0.3%             |\n| (Re)insurance GWP (in million)  | 956,810          | 0.07%            |\n| Number of (re)insurance undertakings | 9           | 0.5%             |\n| Number of registered insurance intermediaries | 394   | 0.05%            |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Finland\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                       | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 5,548            | 1.2%              |\n| (Re)insurance GWP (in million)| 8,449.683        | 0.62%             |\n| Number of (re)insurance undertakings | 43            | 2.5%              |\n| Number of registered insurance intermediaries | 3,812 | 0.4%              |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# France\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                       | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 67,872           | 15%               |\n| (Re)insurance GWP (in million)| 330,940.978      | 24.3%             |\n| Number of (re)insurance undertakings | 151        | 8.7%              |\n| Number of registered insurance intermediaries | 62,284 | 7.2%             |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Germany\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                       | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 83,237           | 18.4%             |\n| (Re)insurance GWP (in million)| 335,298.764      | 24.7%             |\n| Number of (re)insurance undertakings | 256          | 14.8%              |\n| Number of registered insurance intermediaries | 190,708       | 22%              |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Greece\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                       | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 10,460           | 2.3%             |\n| (Re)insurance GWP (in million)| 4,687.315        | 0.4%             |\n| Number of (re)insurance undertakings | 34               | 2.0%             |\n| Number of registered insurance intermediaries | 17,164      | 2%               |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Hungary\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                      | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 9,689            | 2.1%              |\n| (Re)insurance GWP (in million)| 3,562.353        | 0.3%              |\n| Number of (re)insurance undertakings | 22          | 1.3%              |\n| Number of registered insurance intermediaries | 45,376      | 5.2%              |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Iceland\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                           | Share total EEA |\n|----------------------------------|------------------|\n| Population (in 1000)             | 376.248          | 0.08%            |\n| (Re)insurance GWP (in million)   | 572.893          | 0.04%            |\n| Number of (re)insurance undertakings | 8         | 0.5%             |\n| Number of registered insurance intermediaries | 54      | 0.01%            |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Ireland\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                      | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 5,060            | 1.18% |\n| (Re)insurance GWP (in million) | 102,420.675       | 7.5% |\n| Number of (re)insurance undertakings | 163          | 9.4% |\n| Number of registered insurance intermediaries | 2,583   | 0.3% |\n'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Italy\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                      | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 59,030            | 13% |\n| (Re)insurance GWP (in million) | 135,019.570      | 9.9% |\n| Number of (re)insurance undertakings | 87         | 5.0% |\n| Number of registered insurance intermediaries | 235,404  | 27.2% |\n'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Latvia\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                      | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 1,876            | 0.4% |\n| (Re)insurance GWP (in million) | 623,982          | 0.05% |\n| Number of (re)insurance undertakings | 6         | 0.3% |\n| Number of registered insurance intermediaries | 1,397  | 0.2% |\n'}",
]

def model_token_embeddings(model: SentenceTransformer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded token embeddings and their attention mask for ``texts``."""
    features = model.tokenize(texts)
    tensor_features = {
        key: value.to(model.device)
        for key, value in features.items()
        if isinstance(value, torch.Tensor)
    }
    with torch.inference_mode():
        embeddings = model(tensor_features)["token_embeddings"]
    return embeddings, tensor_features["attention_mask"].bool()


def valid_rows(
    token_ids: torch.Tensor,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Iterable[tuple[int, torch.Tensor]]:
    """Yield unpadded ``(token_id, embedding)`` rows in sequence order."""
    for token_id, embedding, is_valid in zip(token_ids, embeddings, attention_mask):
        if is_valid:
            yield token_id.item(), embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="lightonai/modernbert-embed-large",
        help="Hugging Face model identifier or local SentenceTransformer path.",
    )
    parser.add_argument(
        "--chunk",
        action="append",
        dest="chunks",
        help="A document chunk. Repeat this option for each chunk.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used to load the base model.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI token coloring (also useful when redirecting output).",
    )
    return parser.parse_args()


def colorize_token(token: str, score: float, low: float, high: float, enabled: bool) -> str:
    """Color a token from red at the chunk minimum to green at its maximum."""
    if not enabled:
        return token
    ratio = 0.5 if high == low else (score - low) / (high - low)
    ratio = max(0.0, min(1.0, ratio))
    red = round(255 * (1.0 - ratio))
    green = round(180 * ratio)
    return f"\033[38;2;{red};{green};0m{token}\033[0m"


def main() -> None:
    args = parse_args()
    # A trailing newline is ignored when a chunk is encoded independently but
    # becomes an actual token when followed by the literal separator string in
    # the joined document.  Normalize boundaries so both paths tokenize the
    # same text.
    chunks = [chunk.strip() for chunk in (args.chunks or DEFAULT_CHUNKS)]
    if not all(chunk.strip() for chunk in chunks):
        raise ValueError("Chunks must be non-empty strings.")

    base_model = SentenceTransformer(args.model, trust_remote_code=True, device=args.device)
    sentence_embedder = SentenceTransformerEmbedder(
        model=base_model, show_progress_bar=False, device=args.device
    )
    contextual_embedder = ContextualEmbedder(
        model=base_model, show_progress_bar=False, device=args.device
    )

    # The wrapper APIs return pooled chunk vectors.  Their enclosed base models
    # expose the required per-token vectors, while still exercising the two
    # wrapper instances specified above.
    independent_embeddings, independent_mask = model_token_embeddings(
        sentence_embedder.model, chunks
    )
    independent_ids = sentence_embedder.model.tokenize(chunks)["input_ids"]

    sep_token = contextual_embedder.base_model.tokenizer.sep_token
    if sep_token is None:
        raise ValueError("The selected tokenizer has no sep_token and cannot join chunks this way.")
    joined_document = sep_token.join(chunks)
    contextual_embeddings, contextual_mask = model_token_embeddings(
        contextual_embedder.base_model, [joined_document]
    )
    contextual_ids = contextual_embedder.base_model.tokenize([joined_document])["input_ids"][0]

    # A separately encoded BERT-style chunk starts with CLS.  The joined
    # sequence contains that token only once, before the first chunk.
    independent_rows: list[tuple[int, int, torch.Tensor]] = []
    for chunk_index in range(len(chunks)):
        rows = list(valid_rows(
            independent_ids[chunk_index],
            independent_embeddings[chunk_index],
            independent_mask[chunk_index],
        ))
        # Keep the first chunk's leading special token and remove the repeated
        # leading special token from later independently encoded chunks.  The
        # latter is already present once at the start of the joined sequence.
        if chunk_index and rows and rows[0][0] in {
            token_id
            for token_id in (
                tokenizer_id
                for tokenizer_id in (
                    sentence_embedder.model.tokenizer.cls_token_id,
                    sentence_embedder.model.tokenizer.bos_token_id,
                )
                if tokenizer_id is not None
            )
        }:
            rows = rows[1:]
        independent_rows.extend((chunk_index, token_id, embedding) for token_id, embedding in rows)

    contextual_rows = list(
        valid_rows(contextual_ids, contextual_embeddings[0], contextual_mask[0])
    )
    tokenizer = base_model.tokenizer
    independent_token_ids = [token_id for _, token_id, _ in independent_rows]
    contextual_token_ids = [token_id for token_id, _ in contextual_rows]
    if independent_token_ids != contextual_token_ids:
        raise RuntimeError(
            "Token sequences do not align. This experiment assumes that separately "
            "encoded chunks have one leading special token and use the same sep_token."
        )

    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print(f"separator: {sep_token!r}")
    scores = []
    for (chunk_index, token_id, independent), (_, contextual) in zip(
        independent_rows, contextual_rows
    ):
        similarity = functional.cosine_similarity(independent.float(), contextual.float(), dim=0).item()
        scores.append((chunk_index, token_id, similarity))

    ranges = {}
    for chunk_index, _, similarity in scores:
        low, high = ranges.get(chunk_index, (similarity, similarity))
        ranges[chunk_index] = (min(low, similarity), max(high, similarity))

    use_color = not args.no_color and os.environ.get("NO_COLOR") is None
    print("index\tchunk\ttoken\t\tcosine_similarity\tchunk_range")
    for index, (chunk_index, token_id, similarity) in enumerate(scores):
        token = tokenizer.convert_ids_to_tokens(token_id)
        low, high = ranges[chunk_index]
        colored_token = colorize_token(token, similarity, low, high, use_color)
        print(f"{index}\t{chunk_index}\t{colored_token}\t\t{similarity:.6f}\t[{low:.6f}, {high:.6f}]")


if __name__ == "__main__":
    main()
