"""Hypernetwork contextual embedding model.

The implementation deliberately lives beside :class:`LongContextEmbeddingModel`
so existing experiments and checkpoints are unaffected.
"""
from typing import Optional
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from .long_context_model import LongContextEmbeddingModel


class _ChunkHyperNetwork(nn.Module):
    """Maps each chunk embedding ``(…, d)`` to a dynamic matrix ``(…, d, d)``."""
    def __init__(self, dim: int, mode: str = "full", rank: int = 8, dropout: float = 0.0):
        super().__init__()
        mode = mode.lower()
        if mode not in {"full", "lora"}:
            raise ValueError("hypernet_type must be 'full' or 'lora'")
        self.mode, self.dim, self.rank = mode, dim, rank
        if mode == "full":
            self.generator = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, dim * dim))
        else:
            if rank <= 0:
                raise ValueError("hypernet_rank must be positive")
            self.generator_a = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, dim * rank))
            self.generator_b = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, rank * dim))

    def forward(self, chunk: Tensor) -> Tensor:
        if self.mode == "full":
            return self.generator(chunk).view(*chunk.shape[:-1], self.dim, self.dim)
        a = self.generator_a(chunk).view(*chunk.shape[:-1], self.dim, self.rank)
        b = self.generator_b(chunk).view(*chunk.shape[:-1], self.rank, self.dim)
        return a @ b


class HyperContextEmbeddingModel(LongContextEmbeddingModel):
    """Long-context model with chunk-conditioned document projections.

    The inherited encoder/collator contract is unchanged.  Only the model
    loss branch is new, so existing ``LongContextEmbeddingModel`` configs keep
    their original behavior.
    """

    def __init__(self, base_model, hypernet_type: str = "full", hypernet_rank: int = 8,
                 hypernet_dropout: float = 0.0, query_micro_batch_size: Optional[int] = None,
                 chunk_encoding_mode: str = "late",
                 **kwargs):
        # Accept a checkpoint directory directly, matching SentenceTransformer
        # style construction (``HyperContextEmbeddingModel(path)``).
        checkpoint = Path(base_model) if isinstance(base_model, (str, Path)) else None
        if checkpoint is not None:
            meta_path = checkpoint / "hypernetwork_config.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                hypernet_type = meta.get("hypernet_type", hypernet_type)
                hypernet_rank = meta.get("hypernet_rank", hypernet_rank)
                chunk_encoding_mode = meta.get("chunk_encoding_mode", chunk_encoding_mode)
            from sentence_transformers import SentenceTransformer
            base_model = SentenceTransformer(str(checkpoint))
        super().__init__(base_model=base_model, **kwargs)
        if chunk_encoding_mode not in {"late", "separate"}:
            raise ValueError("chunk_encoding_mode must be 'late' or 'separate'")
        self.chunk_encoding_mode = chunk_encoding_mode
        self.query_micro_batch_size = query_micro_batch_size
        dim = int(base_model.get_sentence_embedding_dimension())
        self.hypernetwork = _ChunkHyperNetwork(dim, hypernet_type, hypernet_rank, hypernet_dropout)
        # Keep newly-created parameters in the same dtype/device as the base
        # encoder, avoiding AMP-dependent behavior at inference time.
        try:
            p = next(base_model.parameters())
            self.hypernetwork.to(device=p.device, dtype=p.dtype)
        except StopIteration:
            pass
        if checkpoint is not None:
            assert (checkpoint / "hypernetwork.pt").exists(), "missing hypernetwork weights in checkpoint"
            self.load_hypernetwork(checkpoint)

    def save(self, path: str, *args, **kwargs) -> None:
        """Save the base encoder *and* hypernetwork parameters.

        ``LongContextEmbeddingModel.save`` only delegates to
        ``base_model.save``; without this override the dynamic projection
        generator would silently be omitted from checkpoints.  The additional
        files are portable across sentence-transformers versions and are
        loaded with :meth:`load_hypernetwork` after reconstructing this class.
        """
        super().save(path, *args, **kwargs)
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.hypernetwork.state_dict(), out / "hypernetwork.pt")
        metadata = {
            "hypernet_type": self.hypernetwork.mode,
            "hypernet_rank": self.hypernetwork.rank,
            "hypernet_dim": self.hypernetwork.dim,
            "chunk_encoding_mode": self.chunk_encoding_mode,
        }
        (out / "hypernetwork_config.json").write_text(json.dumps(metadata, indent=2))

    def load_hypernetwork(self, path: str, strict: bool = True):
        """Load hypernetwork weights saved by :meth:`save` into this model.

        Recreate the model with matching ``hypernet_type``/``hypernet_rank``
        (recorded in ``hypernetwork_config.json``) before calling this method.
        The base encoder is loaded separately by ``SentenceTransformer``.
        """
        state_path = Path(path) / "hypernetwork.pt"
        state = torch.load(state_path, map_location=self.hypernetwork.generator[1].weight.device
                           if hasattr(self.hypernetwork, "generator") else "cpu", weights_only=True)
        self.hypernetwork.load_state_dict(state, strict=strict)
        return self

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs):
        """Reconstruct a complete model from a directory produced by ``save``."""
        return cls(path, **kwargs)

    def _infonce(self, queries: Tensor, chunks: Tensor, mask: Tensor,
                 indices, temperature_scale: float) -> Tensor:
        """InfoNCE where every valid chunk in the device batch is a negative."""
        # Flatten ``(batch, max_chunks, d)`` while dropping padding chunks.
        valid = chunks[~mask]
        if valid.numel() == 0:
            raise ValueError("No valid chunks in batch")
        scores = self._compute_sim(queries, valid) * temperature_scale
        labels = []
        offset = 0
        # ``ids`` are local chunk indices; offset converts them to positions in
        # the flattened batch-wide list used by the logits matrix.
        for row, ids in enumerate(indices):
            labels.extend(offset + int(i) for i in ids)
            offset += int((~mask[row]).sum().item())
        labels = torch.tensor(labels, dtype=torch.long, device=scores.device)
        if labels.numel() != queries.shape[0]:
            raise ValueError("queries_chunk_indices does not match query batch")
        return self.cross_entropy_loss(scores, labels)

    def _naive_infonce(self, queries, chunks, mask, indices, n_docs_per_sample=None):
        """Compute naive InfoNCE with *all* chunks as negatives.

        This intentionally calls the inherited ``_compute_loss_inbatch_inseq``.
        Despite its historical name, that method builds one logits matrix over
        every valid chunk in the device batch, so same-document (in-sequence)
        chunks and other-document (in-batch) chunks are both negatives.  The
        positive labels are the matching chunk positions from ``indices``.
        """
        return self._compute_loss_inbatch_inseq(
            queries,
            chunks,
            mask,
            indices,
            n_docs_per_sample=n_docs_per_sample,
        )

    def forward(self, *args, **kwargs):
        loss_type = kwargs.get("loss_type", "infonce")
        if loss_type not in {"hypernet_infonce", "infonce", "weighted"}:
            return super().forward(*args, **kwargs)
        query_inputs, doc_inputs = args[0]["query_inputs"], args[0]["docs_inputs"]
        # Query embeddings are produced exactly as in the original model.
        query_embeddings = self._encode_query_inputs(query_inputs)
        # Document encoding is done once for the concatenated sequence; late
        # chunking below recovers each individual chunk representation.
        # ``docs_context_*`` is present only for separate mode.  It contains
        # one independently encoded chunk per row; normal docs_inputs always
        # represent the complete concatenated document used for e^d.
        context_inputs = {k[len("context_"):]: v for k, v in doc_inputs.items() if k.startswith("context_")}
        # Keep the two encoder inputs separate: the document encoder must see
        # only the concatenated document fields, while context_inputs contains
        # the independently tokenized chunks used for e^b.
        doc_inputs = {k: v for k, v in doc_inputs.items() if not k.startswith("context_")}
        separate = bool(context_inputs)
        token_embeddings = self.base_model(doc_inputs)["token_embeddings"]
        chunk_token_embeddings = self.base_model(context_inputs)["token_embeddings"] if separate else token_embeddings
        if kwargs.get("add_prefixes", False):
            token_embeddings = token_embeddings[:, self.n_tokens_prefix:, :]
            if separate:
                chunk_token_embeddings = chunk_token_embeddings[:, self.n_tokens_prefix:, :]
            doc_inputs = {k: v[:, self.n_tokens_prefix:] if isinstance(v, Tensor) and v.ndim > 1 else v
                          for k, v in doc_inputs.items()}
            if separate:
                context_inputs = {k: v[:, self.n_tokens_prefix:] if isinstance(v, Tensor) and v.ndim > 1 else v
                                  for k, v in context_inputs.items()}
        doc_inputs["input_ids"] = self._add_last_join_token(doc_inputs["input_ids"], doc_inputs["attention_mask"])
        if separate:
            # Independent chunk encoding: each row is one chunk and cannot
            # attend to neighboring chunks.
            cmask = context_inputs["attention_mask"].to(chunk_token_embeddings.dtype)
            chunk_embeddings = (chunk_token_embeddings * cmask.unsqueeze(-1)).sum(1) / cmask.sum(1, keepdim=True).clamp_min(1)
            chunk_embeddings = chunk_embeddings.unsqueeze(1)
            padding_mask = torch.zeros(chunk_embeddings.shape[:2], dtype=torch.bool, device=chunk_embeddings.device)
        else:
            chunk_embeddings, padding_mask = self._late_chunking_pooling(doc_inputs["input_ids"], token_embeddings)
        # Overall document embedding e^d: attention-mask mean over the encoded sequence.
        attn = doc_inputs["attention_mask"].to(token_embeddings.dtype)
        doc_embedding = (token_embeddings * attn.unsqueeze(-1)).sum(1) / attn.sum(1, keepdim=True).clamp_min(1)
        # In single-context collation each input row is one document. Repack
        # those rows into the original samples so labels remain compatible.
        n_docs = kwargs.get("n_docs_per_sample")
        if n_docs is not None and chunk_embeddings.shape[1] == 1:
            if sum(n_docs) != chunk_embeddings.shape[0]:
                raise ValueError("n_docs_per_sample does not match encoded documents")
            max_docs = max(n_docs)
            d, device = chunk_embeddings.shape[-1], chunk_embeddings.device
            packed = chunk_embeddings.new_zeros((len(n_docs), max_docs, d))
            pmask = torch.ones((len(n_docs), max_docs), dtype=torch.bool, device=device)
            pdoc = doc_embedding.new_zeros((len(n_docs), max_docs, d))
            off = 0
            for i, count in enumerate(n_docs):
                packed[i, :count] = chunk_embeddings[off:off + count, 0]
                pdoc[i, :count] = doc_embedding[off:off + count]
                pmask[i, :count] = False
                off += count
            chunk_embeddings, padding_mask, doc_embedding = packed, pmask, pdoc
        if doc_embedding.ndim == 2:
            # Multi-context: one e^d per row, shared by that row's chunks.
            doc_embedding = doc_embedding.unsqueeze(1).expand(-1, chunk_embeddings.shape[1], -1)
        # Generate one projection per chunk and add the residual e^b + W e^d.
        projected = torch.einsum("bnij,bnj->bni", self.hypernetwork(chunk_embeddings), doc_embedding)
        contextual_chunks = chunk_embeddings + projected
        if loss_type in {"infonce", "hypernet_infonce"}:
            # Naive InfoNCE: in-sequence + in-batch negatives together.
            loss = self._naive_infonce(
                query_embeddings,
                contextual_chunks,
                padding_mask,
                kwargs["queries_chunk_indices"],
                kwargs.get("n_docs_per_sample"),
            )
        elif loss_type == "weighted":
            loss = self._compute_loss_weighted_inbatch_inseq(
                query_embeddings, contextual_chunks, padding_mask,
                kwargs["queries_chunk_indices"],
                kwargs.get("n_docs_per_sample"))
        else:
            loss = self._infonce(query_embeddings, contextual_chunks, padding_mask,
                                 kwargs["queries_chunk_indices"], self.sim_score_scale)
        return {"loss": loss}

    def embed_queries(self, queries):
        """Inference query API; delegates prefix handling to the parent."""
        return super().embed_queries(queries)

    def embed_documents(self, documents, batch_size=64):
        """Embed chunked documents with the same hypernetwork used in training.

        ``e^d`` always comes from the concatenated document.  ``e^b`` follows
        ``chunk_encoding_mode``: late chunking (default) or independent chunks.
        """
        self.base_model.eval()
        outputs = []
        with torch.no_grad():
            for chunks in documents:
                text = self.base_model.tokenizer.sep_token.join(chunks)
                if self.add_prefix:
                    text = "search_document: " + text
                whole = self.tokenize([text])
                whole = {k: v.to(self.base_model.device) for k, v in whole.items() if isinstance(v, Tensor)}
                tok = self.base_model(whole)["token_embeddings"]
                if self.add_prefix:
                    tok = tok[:, self.n_tokens_prefix:, :]
                    whole = {k: v[:, self.n_tokens_prefix:] for k, v in whole.items()}
                mask = whole["attention_mask"].to(tok.dtype)
                ed = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
                if getattr(self, "chunk_encoding_mode", "late") == "separate":
                    texts = [("search_document: " + c) if self.add_prefix else c for c in chunks]
                    ci = self.tokenize(texts)
                    ci = {k: v.to(self.base_model.device) for k, v in ci.items() if isinstance(v, Tensor)}
                    ct = self.base_model(ci)["token_embeddings"]
                    if self.add_prefix:
                        ct = ct[:, self.n_tokens_prefix:, :]
                    cm = ci["attention_mask"][:, :ct.shape[1]].to(ct.dtype)
                    eb = (ct * cm.unsqueeze(-1)).sum(1) / cm.sum(1, keepdim=True).clamp_min(1)
                else:
                    eb, pm = self._late_chunking_pooling(whole["input_ids"], tok)
                    eb = eb[0, ~pm[0]]
                ed = ed.expand(eb.shape[0], -1)
                contextual = eb + torch.bmm(self.hypernetwork(eb), ed.unsqueeze(-1)).squeeze(-1)
                if self.normalize_embeddings:
                    contextual = torch.nn.functional.normalize(contextual, p=2, dim=-1)
                outputs.append([x.cpu() for x in contextual])
        return outputs
