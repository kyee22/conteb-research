"""Benchmark adapter for :class:`HyperContextEmbeddingModel`."""

from typing import Optional

from cde_benchmark.embedders.base_embedder import Embedder
from contextual_embeddings import HyperContextEmbeddingModel


class HyperContextualEmbedder(Embedder):
    """Thin wrapper around an already constructed hyper-context model."""

    def __init__(self, model: HyperContextEmbeddingModel, batch_size: int = 16,
                 show_progress_bar: bool = True, device: str = "cpu"):
        super().__init__(is_contextual_model=True, device=device)
        self.model = model.to(device)
        # Match the projection generator to the encoder output dtype (e.g.
        # bfloat16 for ModernBERT checkpoints).
        try:
            encoder_dtype = next(self.model.base_model.parameters()).dtype
            self.model.hypernetwork.to(device=device, dtype=encoder_dtype)
        except (StopIteration, AttributeError):
            pass
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.model.batch_size = batch_size
        self.model.show_progress_bar = show_progress_bar

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs):
        """Construct using the model's canonical checkpoint loader."""
        device = kwargs.pop("device", "cpu")
        model = HyperContextEmbeddingModel.from_checkpoint(path, **kwargs)
        return cls(model, device=device)

    def embed_queries(self, queries):
        return self.model.embed_queries(queries)

    def embed_documents(self, documents, batch_size: Optional[int] = None):
        return self.model.embed_documents(documents, batch_size=batch_size or self.batch_size)


HyperContextEmbedder = HyperContextualEmbedder
__all__ = ["HyperContextualEmbedder", "HyperContextEmbedder"]
