"""Benchmark adapter for Qwen3 late-chunking checkpoints."""

from pathlib import Path
from typing import Optional

from contextual_embeddings import Qwen3LateChunkingModel
from cde_benchmark.embedders.base_embedder import Embedder


class Qwen3LateChunkingEmbedder(Qwen3LateChunkingModel, Embedder):
    """Benchmark adapter exposing Qwen3 contextual query/document embeddings.

    The adapter keeps the benchmark's nested document contract while delegating
    mask selection and pooling to ``Qwen3LateChunkingModel``. ``use_prefix`` is
    intentionally not used: Qwen3 query formatting comes from checkpoint
    metadata and documents do not receive a retrieval instruction.
    """
    def __init__(self, model, batch_size: int = 1, device: str = "cpu", **kwargs):
        """Wrap a Qwen3 SentenceTransformer or checkpoint path.

        Args:
            model: Base Qwen3 SentenceTransformer or saved checkpoint path.
            batch_size: Number of nested documents encoded per call.
            device: Benchmark output/device label.
            **kwargs: Qwen3 attention and pooling options, normally restored
                automatically from checkpoint metadata.
        """
        Qwen3LateChunkingModel.__init__(self, model, **kwargs)
        Embedder.__init__(self, is_contextual_model=True, device=device)
        self.batch_size = batch_size

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs):
        """Load a checkpoint exactly once and restore persisted configuration."""
        device = kwargs.pop("device", "cpu")
        return cls(path, device=device, **kwargs)

    # def embed_queries(self, queries):
    #     """Return a ``(num_queries, hidden_size)`` tensor for retrieval queries."""
    #     return Qwen3LateChunkingModel.embed_queries(self, queries)

    # def embed_documents(self, documents, batch_size: Optional[int] = None):
    #     """Return nested CPU tensors, one embedding for each input chunk.

    #     ``documents`` must be ``Sequence[Sequence[str]]``. The outer structure
    #     is preserved, and empty chunks remain represented by zero vectors as in
    #     the underlying model's offset pooling contract.
    #     """
    #     outputs = []
    #     effective_batch_size = batch_size or self.batch_size
    #     for start in range(0, len(documents), effective_batch_size):
    #         # The adapter overrides this method for the benchmark's optional
    #         # batch-size argument, so call the model implementation explicitly
    #         # instead of recursing through the override.
    #         outputs.extend(
    #             Qwen3LateChunkingModel.embed_documents(
    #                 self,
    #                 documents[start : start + effective_batch_size],
    #                 batch_size=effective_batch_size,
    #             )
    #         )
    #     return outputs


__all__ = ["Qwen3LateChunkingEmbedder"]
