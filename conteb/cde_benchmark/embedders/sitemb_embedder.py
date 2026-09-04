"""Embedders for Qwen3-Embedding and SituatedEmbedding Qwen3 adapters.

The SitEmb repositories contain PEFT adapters rather than a complete base
model. The SitEmb wrappers load an adapter on top of Qwen3-Embedding, while
``Qwen3Embedder`` loads a complete model directly. All classes expose the same
flat/nested interface used by the benchmark embedders.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from cde_benchmark.embedders.base_embedder import Embedder


DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-8B"
CHUNK_ONLY_ADAPTER = "SituatedEmbedding/SitEmb-v1.5-Qwen3-chunk-only"
CONTEXT_ADAPTER = "SituatedEmbedding/SitEmb-v1.5-Qwen3"
DEFAULT_QUERY_TEMPLATE = (
    "Instruct: Given a search query, retrieve relevant chunks from fictions that answer the query\n"
    "Query:{query}"
)
PASSAGE_AFFIX = (
    "The context in which the chunk is situated is given below. "
    "Please encode the chunk by being aware of the context. Context:\n"
)
END_OF_TEXT = "<|endoftext|>"


class _SitEmbBase(Embedder):
    """Shared model loading, batching, and pooling for SitEmb adapters.

    Args:
        model: Optional preloaded PEFT model. Primarily useful for tests or for
            callers that need custom loading options.
        tokenizer: Optional tokenizer paired with ``model``. When omitted, the
            tokenizer is loaded from ``model_name``.
        model_name: Base-model repository used to load the tokenizer.
        adapter_name: PEFT adapter repository. Required when ``model`` is not
            supplied.
        batch_size: Default number of strings encoded per forward pass.
        max_length: Maximum tokenized sequence length, matching the notebook's
            default of 8192.
        normalize_embeddings: Whether to L2-normalize each pooled embedding.
        show_progress_bar: Whether to display encoding progress.
        device: Inference device, such as ``cpu``, ``cuda``, or ``cuda:1``.
        torch_dtype: Optional dtype forwarded to PEFT model loading.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name: str = DEFAULT_BASE_MODEL,
        adapter_name: str | None = None,
        batch_size: int = 16,
        max_length: int = 8192,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
        device: str = "cpu",
        torch_dtype=None,
    ):
        super().__init__(is_contextual_model=False, device=device)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name, use_fast=True, padding_side="left"
        )

        if model is None:
            if adapter_name is None:
                raise ValueError("adapter_name is required when model is not provided")
            from peft import AutoPeftModelForFeatureExtraction

            load_kwargs = {}
            if device.startswith("cuda"):
                cuda_device = torch.device(device)
                load_kwargs["device_map"] = {"": cuda_device.index or 0}
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            model = AutoPeftModelForFeatureExtraction.from_pretrained(adapter_name, **load_kwargs)
            if device != "cpu" and not device.startswith("cuda"):
                model = model.to(device)
        self.model = model
        self.model.eval()

    @property
    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(getattr(self.model, "device", self._device))

    @torch.inference_mode()
    def _encode_passages(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        residual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode strings using notebook-equivalent last/EOS pooling.

        When ``residual`` is true, the second tensor contains the hidden state
        at the first non-padding ``<|endoftext|>`` token. Both tensors are
        independently normalized when ``normalize_embeddings`` is enabled.

        Returns:
            A pair of ``(last_token_embeddings, residual_embeddings)``. The
            second value is ``None`` when residual pooling is disabled.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        outputs = []
        residual_outputs = []
        starts = range(0, len(texts), batch_size)
        if self.show_progress_bar:
            from tqdm import tqdm

            starts = tqdm(starts, desc="Encoding", leave=False)
        for start in starts:
            batch_texts = list(texts[start : start + batch_size])
            batch = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch = {key: value.to(self._model_device) for key, value in batch.items()}
            hidden = self.model(**batch).last_hidden_state
            attention_mask = batch["attention_mask"]
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                pooled = hidden[:, -1]
            else:
                positions = attention_mask.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
            if self.normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=-1)
            outputs.append(pooled)

            if residual:
                eos_positions = []
                for input_ids, mask in zip(batch["input_ids"], attention_mask):
                    matches = (
                        (input_ids == self.tokenizer.pad_token_id) & mask.bool()
                    ).nonzero(as_tuple=False).flatten()
                    if matches.numel() < 2:
                        raise ValueError(
                            "Expected both the explicit separator and final <|endoftext|> token; "
                            "the chunk was likely truncated before its separator"
                        )
                    eos_positions.append(int(matches[0].item()))
                eos_positions = torch.tensor(eos_positions, device=hidden.device)
                residual_pooled = hidden[
                    torch.arange(hidden.shape[0], device=hidden.device), eos_positions
                ]
                if self.normalize_embeddings:
                    residual_pooled = F.normalize(residual_pooled, p=2, dim=-1)
                residual_outputs.append(residual_pooled)

        if not outputs:
            empty = torch.empty((0, 0), device=self._model_device)
            return empty, empty if residual else None
        residual_embeddings = torch.cat(residual_outputs, dim=0) if residual_outputs else None
        return torch.cat(outputs, dim=0), residual_embeddings

    def _encode_texts(self, texts: Sequence[str], batch_size: int | None = None) -> torch.Tensor:
        embeddings, _ = self._encode_passages(texts, batch_size=batch_size)
        return embeddings


class SitEmbChunkOnlyEmbedder(_SitEmbBase):
    """Non-contextual embedder for ``SitEmb-v1.5-Qwen3-chunk-only``.

    Queries receive the retrieval instruction used in ``SitEmb.ipynb`` while
    document chunks are encoded independently. Both methods return a tensor of
    shape ``(num_inputs, embedding_dimension)``.
    """

    def __init__(
        self,
        model=None,
        adapter_name: str = CHUNK_ONLY_ADAPTER,
        query_template: str = DEFAULT_QUERY_TEMPLATE,
        **kwargs,
    ):
        super().__init__(model=model, adapter_name=adapter_name, **kwargs)
        if "{query}" not in query_template:
            raise ValueError("query_template must contain a {query} placeholder")
        self.query_template = query_template

    def embed_queries(self, queries: Iterable[str]) -> torch.Tensor:
        """Encode retrieval queries with the SitEmb instruction prefix."""
        return self._encode_texts([self.query_template.format(query=query) for query in queries])

    def embed_documents(self, documents: Iterable[str]) -> torch.Tensor:
        """Encode document chunks independently without document context."""
        texts = [document if document and document.strip() else "placeholder" for document in documents]
        return self._encode_texts(texts)


class Qwen3Embedder(_SitEmbBase):
    """Non-contextual embedder backed by a complete Qwen3 embedding model.

    Unlike :class:`SitEmbChunkOnlyEmbedder`, ``model_path`` must point to a
    full model rather than a PEFT adapter. Query and document preprocessing is
    otherwise identical to the chunk-only embedder.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_BASE_MODEL,
        model=None,
        tokenizer=None,
        query_template: str = DEFAULT_QUERY_TEMPLATE,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        if "model_name" in kwargs:
            raise TypeError("Qwen3Embedder accepts model_path instead of model_name")
        if "adapter_name" in kwargs:
            raise TypeError("Qwen3Embedder accepts model_path instead of adapter_name")
        if "{query}" not in query_template:
            raise ValueError("query_template must contain a {query} placeholder")

        if model is None:
            device = kwargs.get("device", "cpu")
            load_kwargs = {}
            torch_dtype = kwargs.get("torch_dtype")
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            if attn_implementation is not None:
                load_kwargs["attn_implementation"] = attn_implementation
            if device.startswith("cuda"):
                cuda_device = torch.device(device)
                load_kwargs["device_map"] = {"": cuda_device.index or 0}
            model = AutoModel.from_pretrained(model_path, **load_kwargs)
            if device != "cpu" and not device.startswith("cuda"):
                model = model.to(device)

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            model_name=model_path,
            batch_size=4,
            adapter_name=None,
            **kwargs,
        )
        self.model_path = model_path
        self.query_template = query_template

    def embed_queries(self, queries: Iterable[str]) -> torch.Tensor:
        """Encode retrieval queries with the Qwen3 instruction prefix."""
        return self._encode_texts([self.query_template.format(query=query) for query in queries])

    def embed_documents(self, documents: Iterable[str]) -> torch.Tensor:
        """Encode document chunks independently without an instruction."""
        texts = [document if document and document.strip() else "placeholder" for document in documents]
        return self._encode_texts(texts)


class SitEmbContextualEmbedder(_SitEmbBase):
    """Contextual embedder for ``SitEmb-v1.5-Qwen3``.

    For every chunk, the input exactly follows the notebook's layout::

        {chunk}<|endoftext|>{passage_affix}{full_document_context}

    The final token yields the contextual embedding and the explicit separator
    yields the residual chunk embedding. Each is normalized independently,
    then combined according to ``residual_factor``. Combining embeddings is
    algebraically equivalent to the notebook's weighted sum of dot-product
    scores.
    """

    def __init__(
        self,
        model=None,
        adapter_name: str = CONTEXT_ADAPTER,
        query_template: str = DEFAULT_QUERY_TEMPLATE,
        passage_affix: str = PASSAGE_AFFIX,
        context_separator: str = " ",
        residual_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__(model=model, adapter_name=adapter_name, batch_size=4, **kwargs)
        if "{query}" not in query_template:
            raise ValueError("query_template must contain a {query} placeholder")
        if not 0.0 <= residual_factor <= 1.0:
            raise ValueError("residual_factor must be between 0 and 1")
        self.query_template = query_template
        self.passage_affix = passage_affix
        self.context_separator = context_separator
        self.residual_factor = residual_factor
        self.is_contextual_model = True

    def embed_queries(self, queries: Iterable[str]) -> torch.Tensor:
        """Encode queries without residual pooling, as in the notebook."""
        return self._encode_texts([self.query_template.format(query=query) for query in queries])

    def embed_documents(
        self,
        documents: Iterable[Sequence[str]],
        batch_size: int | None = None,
    ) -> list[list[torch.Tensor]]:
        """Encode chunks using all chunks from the same document as context.

        Args:
            documents: Documents represented as ordered sequences of chunks.
            batch_size: Optional per-call batch-size override.

        Returns:
            Embeddings grouped in the same document/chunk structure as the
            input.
        """
        documents = list(documents)
        texts = []
        lengths = []
        for chunks in documents:
            chunks = list(chunks)
            context = self.context_separator.join(chunks)
            texts.extend(f"{chunk}{END_OF_TEXT}{self.passage_affix}{context}" for chunk in chunks)
            lengths.append(len(chunks))
        contextual, residual = self._encode_passages(texts, batch_size=batch_size, residual=True)
        embeddings = contextual * (1.0 - self.residual_factor) + residual * self.residual_factor
        
        # 1️⃣ embedding-level normalization
        # normalize the combined embeddings if normalization is enabled
        # if self.normalize_embeddings:
        #     embeddings = F.normalize(embeddings, p=2, dim=-1)

        # 2️⃣ similarity-level normalization
        # otherwise, default

        
        result = []
        offset = 0
        for length in lengths:
            result.append([embedding for embedding in embeddings[offset : offset + length]])
            offset += length
        return result
