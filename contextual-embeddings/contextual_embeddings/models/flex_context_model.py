"""FlexAttention variant of the contextual document encoder.

This module is deliberately separate from :mod:`long_context_model`: the
original model keeps its exact implementation and attention backend.  The
class below only enables a chunk-aware ``BlockMask`` for document forwards.
"""

from typing import Optional

import torch
from torch import Tensor

from .long_context_model import LongContextEmbeddingModel


class _FlexModernBertMixin:
    """Forward override mixed into one ModernBERT instance only."""

    def _flex_attention_mask(self, input_ids: Tensor, attention_mask: Optional[Tensor], prefix_length: int, mode: str):
        """Build the bool mask expected by PyTorch SDPA (True means visible)."""
        batch_size, seq_len = input_ids.shape
        sep_id = int(self.config.sep_token_id)
        # -1 is the shared, read-only prefix; non-negative values identify
        # chunks.  Padding gets a separate id and is rejected below.
        chunk_ids = torch.zeros_like(input_ids, dtype=torch.long)
        if prefix_length:
            chunk_ids[:, :prefix_length] = -1
        body = input_ids[:, prefix_length:]
        if body.numel():
            chunk_ids[:, prefix_length:] = torch.cumsum(body.eq(sep_id), dim=1) - body.eq(sep_id).long()
            chunk_ids[:, prefix_length:] = chunk_ids[:, prefix_length:] + 0

        valid = attention_mask.to(torch.bool) if attention_mask is not None else torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
        )

        q = torch.arange(seq_len, device=input_ids.device)[None, :, None]
        k = torch.arange(seq_len, device=input_ids.device)[None, None, :]
        q_valid = valid[:, :, None]
        k_valid = valid[:, None, :]
        if mode == "full":
            visible = q_valid & k_valid
        elif mode == "sliding_window":
            # ModernBertConfig exposes the effective radius as
            # ``sliding_window`` in recent Transformers; older checkpoints
            # only retain ``local_attention`` (the full window width).
            window = getattr(self.config, "sliding_window", None)
            if window is None:
                local_attention = getattr(self.config, "local_attention", None)
                if local_attention is None:
                    raise ValueError("ModernBERT config has no local attention window")
                window = int(local_attention) // 2
            visible = q_valid & k_valid & ((q - k).abs() <= int(window))
        elif mode == "chunk":
            q_group = chunk_ids[:, :, None]
            k_group = chunk_ids[:, None, :]
            # Prefix queries are isolated.  Every chunk may read the shared
            # prefix and its own chunk, but never another chunk.
            visible = q_valid & k_valid & torch.where(
                q_group == -1,
                k_group == -1,
                (k_group == -1) | (q_group == k_group),
            )
        else:
            raise ValueError(f"Unknown attention replacement mode: {mode}")
        return visible[:, None, :, :]

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None,
                is_document=False, chunk_prefix_length=None, **kwargs):
        is_document = is_document or getattr(
            self, "_flex_is_document", False
        )
        if not is_document:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, inputs_embeds=inputs_embeds, **kwargs
            )
        if input_ids is None:
            raise ValueError("is_document requires input_ids")
        if chunk_prefix_length is None:
            chunk_prefix_length = getattr(self, "_flex_chunk_prefix_length", 0)
        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        full_mask = self._flex_attention_mask(
            input_ids, attention_mask, int(chunk_prefix_length), getattr(self, "_flex_replace_full_with", "full")
        )
        local_mask = self._flex_attention_mask(
            input_ids, attention_mask, int(chunk_prefix_length), getattr(self, "_flex_replace_local_with", "chunk")
        )
        attention_mask_mapping = {
            "full_attention": full_mask,
            "sliding_attention": local_mask,
        }
        # print(
        #     "mode:",
        #     getattr(self, "_flex_replace_local_with", None),
        #     getattr(self, "_flex_replace_full_with", None),
        #     "mask shape:",
        #     full_mask.shape,
        #     local_mask.shape,
        #     "full visible:",
        #     full_mask.float().mean().item(),
        #     "local visible:",
        #     local_mask.float().mean().item(),
        # )
        seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask_mapping[encoder_layer.attention_type],
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                **kwargs,
            )
        hidden_states = self.final_norm(hidden_states)
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=hidden_states)


class FlexModernBertModel(_FlexModernBertMixin):
    """Marker class used only for the per-instance ModernBERT override."""


class FlexLongContextEmbeddingModel(LongContextEmbeddingModel):
    """Original contextual model with FlexAttention chunk isolation."""

    def __init__(self, base_model, replace_local_with: str = "chunk", replace_full_with: str = "full", **kwargs):
        super().__init__(base_model=base_model, **kwargs)
        self.replace_local_with = replace_local_with
        self.replace_full_with = replace_full_with
        valid_modes = {"full", "sliding_window", "chunk"}
        if replace_local_with not in valid_modes or replace_full_with not in valid_modes:
            raise ValueError(f"attention replacement modes must be in {sorted(valid_modes)}")
        auto_model = getattr(self.base_model[0], "auto_model", None)
        if auto_model is None or getattr(auto_model.config, "model_type", None) != "modernbert":
            raise ValueError("FlexLongContextEmbeddingModel requires a ModernBERT SentenceTransformer model")
        # Changing the class of this instance leaves the original class and
        # all other model instances untouched.
        auto_model.__class__ = type(
            "FlexModernBertInstance", (FlexModernBertModel, auto_model.__class__), {}
        )
        auto_model._flex_replace_local_with = replace_local_with
        auto_model._flex_replace_full_with = replace_full_with
        self._flex_auto_model = auto_model

    def _encode_document(self, doc_inputs):
        # SentenceTransformer.forward only forwards kwargs explicitly
        # registered by each module. ``is_document`` and
        # ``chunk_prefix_length`` are not registered Transformer-module kwargs,
        # so passing them to self.base_model here silently dropped them and the
        # patched ModernBERT forward took its normal (non-Flex) branch during
        # training. Use per-instance state, as the inference path below does,
        # so the document marker always reaches the underlying auto model.
        self._flex_auto_model._flex_is_document = True
        self._flex_auto_model._flex_chunk_prefix_length = (
            self.n_tokens_prefix if self._current_add_prefix else 0
        )
        try:
            return self.base_model(doc_inputs)["token_embeddings"]
        finally:
            self._flex_auto_model._flex_is_document = False
            self._flex_auto_model._flex_chunk_prefix_length = 0

    def embed_batch_documents(self, documents):
        """Reuse the parent implementation while enabling the document mask.

        The parent method already owns tokenization, truncation, pooling and
        normalization.  Its only missing piece for this model is the marker
        passed to the patched ModernBERT forward.
        """
        self._flex_auto_model._flex_is_document = True
        # The parent implementation owns document tokenization and does not
        # expose the prefix length as a model kwarg. Keep it on this model
        # instance so the patched ModernBERT forward can still isolate prefix
        # tokens during document inference.
        self._flex_auto_model._flex_chunk_prefix_length = (
            self.n_tokens_prefix if self.add_prefix else 0
        )
        try:
            return super().embed_batch_documents(documents)
        finally:
            self._flex_auto_model._flex_is_document = False
            self._flex_auto_model._flex_chunk_prefix_length = 0

    def forward(self, *args, **kwargs):
        query_inputs = args[0]["query_inputs"]
        query_embeddings = self._encode_query_inputs(query_inputs)
        doc_inputs = args[0]["docs_inputs"]
        self._current_add_prefix = bool(kwargs.get("add_prefixes", False))
        doc_token_embeddings = self._encode_document(doc_inputs)
        if self._current_add_prefix:
            doc_token_embeddings = doc_token_embeddings[:, self.n_tokens_prefix:, :]
            for key, value in doc_inputs.items():
                if isinstance(value, Tensor) and value.ndim > 1:
                    doc_inputs[key] = value[:, self.n_tokens_prefix:]
        doc_inputs["input_ids"] = self._add_last_join_token(doc_inputs["input_ids"], doc_inputs["attention_mask"])
        chunk_embeddings, doc_padding_mask = self._late_chunking_pooling(
            doc_inputs["input_ids"], doc_token_embeddings
        )
        loss_type = kwargs.get("loss_type", "weighted")
        if loss_type == "weighted":
            loss = self._compute_loss_weighted_inbatch_inseq(
                query_embeddings, chunk_embeddings, doc_padding_mask,
                kwargs["queries_chunk_indices"]
            )
        elif loss_type == "inbatch_inseq":
            loss = self._compute_loss_inbatch_inseq(
                query_embeddings, chunk_embeddings, doc_padding_mask,
                kwargs["queries_chunk_indices"], kwargs.get("n_docs_per_sample")
            )
        elif loss_type == "late_interaction":
            loss = self._compute_late_interaction_loss(
                query_embeddings, chunk_embeddings, doc_padding_mask,
                kwargs["queries_chunk_indices"]
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return {"loss": loss}
