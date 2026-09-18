"""Benchmark adapter for the FlexAttention contextual embedder."""

import re
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from contextual_embeddings import FlexLongContextEmbeddingModel

from cde_benchmark.embedders.base_embedder import Embedder


class FlexContextualEmbedder(FlexLongContextEmbeddingModel, Embedder):
    """Contextual benchmark embedder backed by ``FlexLongContextEmbeddingModel``.

    The checkpoint is a regular SentenceTransformers checkpoint.  FlexAttention
    is installed on the ModernBERT instance when the contextual model is
    constructed, so query and document embedding use the model's normal public
    API while document batches receive chunk-isolated attention.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        device: str = "cpu",
        add_prefix: bool = False,
        quantization: str = None,
        replace_local_with: str = "sliding_window",
        replace_full_with: str = "full",
    ):
        FlexLongContextEmbeddingModel.__init__(
            self,
            base_model=model,
            add_prefix=add_prefix,
            quantization=quantization,
            replace_local_with=replace_local_with,
            replace_full_with=replace_full_with,
            show_progress_bar=show_progress_bar,
        )
        Embedder.__init__(self, is_contextual_model=True, device=device)
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs):
        """Load a SentenceTransformers checkpoint and install FlexAttention."""
        device = kwargs.pop("device", "cpu")
        # Training checkpoints retain the FlexAttention choices in the
        # generated training_config.yml.  Use them when available, while
        # allowing callers to override either value explicitly.
        training_config = Path(path) / "training_config.yml"
        if training_config.is_file():
            text = training_config.read_text(encoding="utf-8")
            for name in ("replace_local_with", "replace_full_with"):
                if name not in kwargs:
                    match = re.search(rf"^\s*{name}:\s*[\"']?([A-Za-z_]+)", text, re.MULTILINE)
                    if match:
                        kwargs[name] = match.group(1)
                        print(f"Using {name}={kwargs[name]} from training_config.yml")
        model = SentenceTransformer(path, trust_remote_code=True, device=device)
        return cls(model, device=device, **kwargs)

    def embed_queries(self, queries):
        self.batch_size = getattr(self, "batch_size", 16)
        return FlexLongContextEmbeddingModel.embed_queries(self, queries)

    def embed_documents(self, documents, batch_size: Optional[int] = None):
        return FlexLongContextEmbeddingModel.embed_documents(
            self, documents, batch_size=batch_size or self.batch_size
        )


__all__ = ["FlexContextualEmbedder"]
