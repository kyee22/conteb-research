"""Build a Qwen3 SentenceTransformer from a decoder-only checkpoint."""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer, models


def create_sentence_transformer(
    model_name_or_path: str,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    tokenizer_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
    **kwargs: Any,
) -> SentenceTransformer:
    """Use base weights with Qwen3-Embedding tokenization and ST modules.

    The base checkpoint has no SentenceTransformers modules. Reuse the embedding
    tokenizer, including its terminal-token postprocessor, so only the
    Transformer weights come from ``model_name_or_path``. The wrapper still
    reads ``token_embeddings`` and performs its configured contextual pooling.
    """
    transformer = models.Transformer(
        model_name_or_path,
        model_args=model_kwargs or {},
        tokenizer_args=tokenizer_kwargs or {},
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode="lasttoken",
    )
    kwargs.setdefault("prompts", {
        "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "document": "",
    })
    kwargs.setdefault("similarity_fn_name", "cosine")
    return SentenceTransformer(
        modules=[transformer, pooling, models.Normalize()],
        **kwargs,
    )
