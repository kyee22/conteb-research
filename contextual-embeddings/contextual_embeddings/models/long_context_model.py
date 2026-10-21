import copy
from typing import List, Optional

import torch
from pylate.models import ColBERT
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn
from tqdm import tqdm


class LongContextEmbeddingModel(SentenceTransformer):
    def __init__(
        self,
        base_model: SentenceTransformer,
        sim_score_scale: float = 20.0,
        normalize_embeddings: bool = True,
        multi_ctx_training: bool = True,
        lambda_seq: float = 0.5,
        doc_prefix_str: str = "search_document:",
        pooling_mode: str = "average",
        add_prefix: bool = False,
        show_progress_bar: bool = True,
        quantization: str = None,
        query_micro_batch_size: Optional[int] = None,
        document_micro_batch_size: Optional[int] = None,
        self_distillation: bool = False,
        self_distillation_rank_k: int = 2,
        self_distillation_alpha: float = 0.5,
    ):
        super().__init__()

        self.add_prefix = add_prefix
        self.base_model = base_model
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.sim_score_scale = sim_score_scale
        self.normalize_embeddings = normalize_embeddings
        self.multi_ctx_training = multi_ctx_training
        self.lambda_seq = lambda_seq
        self.self_distillation = self_distillation
        self.self_distillation_rank_k = self_distillation_rank_k
        self.self_distillation_alpha = self_distillation_alpha
        if self.self_distillation_rank_k < 0:
            raise ValueError("self_distillation_rank_k must be >= 0")
        if self.self_distillation_rank_k == 0:
            print("[⚠️ Warning] self_distillation_rank_k=0 disables self-distillation; set to >0 to enable")
        if not 0.0 <= self.self_distillation_alpha <= 1.0:
            raise ValueError("self_distillation_alpha must be in [0, 1]")
        self.n_tokens_prefix = (
            len(base_model.tokenizer(doc_prefix_str, add_special_tokens=True)["input_ids"]) - 1
        )  # - 1 to remove the [SEP] token (but count the [CLS] token)
        self.show_progress_bar = show_progress_bar
        
        if query_micro_batch_size is not None and query_micro_batch_size <= 0:
            raise ValueError("query_micro_batch_size must be a positive integer or None")
        self.query_micro_batch_size = query_micro_batch_size
        if document_micro_batch_size is not None and document_micro_batch_size <= 0:
            raise ValueError("document_micro_batch_size must be a positive integer or None")
        self.document_micro_batch_size = document_micro_batch_size

        # test fix for models that do not have a sep_token (decoders)
        if self.base_model.tokenizer.sep_token is None:
            self.base_model.tokenizer.sep_token = self.base_model.tokenizer.pad_token

        # if hasattr(self.base_model, "document_prefix_id"):
        #     self.join_token_id = self.base_model.document_prefix_id
        # else:
        #     self.join_token_id = self.base_model.tokenizer.sep_token_id
        self.join_token_id = self.base_model.tokenizer.sep_token_id

        self.pooling_mode = pooling_mode
        if self.self_distillation and self.pooling_mode != "average":
            raise ValueError("self_distillation currently requires pooling_mode='average'")
        print(f"LongContextEmbeddingModel: lambda_seq={self.lambda_seq}")
        print(
            "LongContextEmbeddingModel: self_distillation="
            f"{self.self_distillation}, rank_k={self.self_distillation_rank_k}, "
            f"alpha={self.self_distillation_alpha}"
        )

        # The teacher is a permanent snapshot of the initial encoder. It is
        # created only for the opt-in training intervention and never receives
        # optimizer updates or gradients.
        self.teacher_base_model = None
        if self.self_distillation:
            self.teacher_base_model = copy.deepcopy(self.base_model)
            for parameter in self.teacher_base_model.parameters():
                parameter.requires_grad_(False)
            self.teacher_base_model.eval()

    def train(self, mode=True):
        """Change student mode while keeping the frozen teacher in eval mode."""
        super().train(mode)
        if self.teacher_base_model is not None:
            self.teacher_base_model.eval()
        return self

    def tokenize(
        # self, texts: list[str] | list[dict] | list[tuple[str, str]]
        self,
        texts: list[str],
        **kwargs,
    ) -> dict[str, Tensor]:
        return self.base_model.tokenize(texts, **kwargs)

    def encode(self, args, **kwargs):
        return self.base_model.encode(args, **kwargs)

    def _offset_tokenize_documents(self, documents, add_prefix=""):
        """Build one context sequence per document and record each chunk span.

        This is the ModernBERT equivalent of Qwen3's offset tokenizer. Each
        document remains one model row, so self-attention still crosses chunk
        boundaries. A single CLS begins the row; every chunk contributes its
        own content tokens followed by SEP. ``offsets[b][c] = (start, end)``
        uses half-open positions into that row and covers chunk content only.
        CLS, the single document-level prefix, separators, and padding are
        excluded from every pooled span. Every source chunk, including an
        empty/truncated one, keeps an explicit span and therefore its index.
        """
        tokenizer = self.base_model.tokenizer
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            raise ValueError("Offset-based ModernBERT encoding requires CLS and SEP token IDs")

        max_length = int(getattr(self.base_model, "max_seq_length", tokenizer.model_max_length))
        prefix_token_count = (
            len(tokenizer(add_prefix, add_special_tokens=False)["input_ids"])
            if add_prefix
            else 0
        )
        rows, offsets = [], []
        for chunks in documents:
            ids_row = [cls_id]
            row_offsets = []
            if len(chunks) + 1 > max_length:
                raise ValueError(
                    f"Document has {len(chunks)} chunks, exceeding ModernBERT sequence "
                    f"capacity {max_length} while preserving one separator per chunk"
                )
            for chunk_idx, chunk in enumerate(chunks):
                encoded = tokenizer(
                    f"{add_prefix}{chunk}" if chunk_idx == 0 else chunk,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
                # Reserve a SEP for every remaining chunk, so truncation can
                # shorten content without silently dropping chunk identities.
                remaining_separators = len(chunks) - chunk_idx
                content_capacity = max_length - len(ids_row) - remaining_separators
                content_ids = encoded[:max(0, content_capacity)]
                # Prefix tokens occur only in the first span's input sequence,
                # but the pooled span starts after both CLS and prefix.
                start = 1 + prefix_token_count if chunk_idx == 0 else len(ids_row)
                ids_row.extend(content_ids)
                row_offsets.append((start, len(ids_row)))
                ids_row.append(sep_id)
            rows.append(torch.tensor(ids_row, dtype=torch.long))
            offsets.append(row_offsets)

        if not rows:
            empty = torch.empty((0, 0), dtype=torch.long)
            return {"input_ids": empty, "attention_mask": empty}, offsets
        input_ids = torch.nn.utils.rnn.pad_sequence(
            rows, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, row in enumerate(rows):
            attention_mask[row_idx, :row.numel()] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}, offsets

    @staticmethod
    def _pool_offset_chunks(token_embeddings, offsets, pooling_mode="average"):
        """Pool offset spans as vectors or token sequences.

        Average mode returns ``(batch, max_chunks, hidden)``. Token mode
        returns ``(batch, max_chunks, max_chunk_tokens, hidden)``. The padding
        mask marks absent chunk slots; empty source chunks remain real entries
        containing zeros.
        """
        max_chunks = max((len(row) for row in offsets), default=0)
        if pooling_mode == "tokens":
            max_chunk_tokens = max(
                1,
                max((end - start for row in offsets for start, end in row), default=0),
            )
            pooled = token_embeddings.new_zeros(
                (len(offsets), max_chunks, max_chunk_tokens, token_embeddings.shape[-1])
            )
        elif pooling_mode == "average":
            pooled = token_embeddings.new_zeros(
                (len(offsets), max_chunks, token_embeddings.shape[-1])
            )
        else:
            raise ValueError("pooling_mode must be 'average' or 'tokens'")
        padding = torch.ones(
            (len(offsets), max_chunks), dtype=torch.bool, device=token_embeddings.device
        )
        for row_idx, row_offsets in enumerate(offsets):
            for chunk_idx, (start, end) in enumerate(row_offsets):
                if start < 0 or end < start or end > token_embeddings.shape[1]:
                    raise ValueError(
                        f"Invalid chunk offset at sample={row_idx}, chunk={chunk_idx}: "
                        f"({start}, {end}) for sequence length {token_embeddings.shape[1]}"
                    )
                if end > start:
                    if pooling_mode == "tokens":
                        pooled[row_idx, chunk_idx, :end - start] = token_embeddings[row_idx, start:end]
                    else:
                        pooled[row_idx, chunk_idx] = token_embeddings[row_idx, start:end].mean(0)
                padding[row_idx, chunk_idx] = False
        return pooled, padding

    def _compute_sim(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        if len(embeddings_a.shape) not in [2, 3] or len(embeddings_b.shape) not in [
            2,
            3,
        ]:
            raise ValueError("Embeddings should have shape (n, dim) or (b, n, dim)")

        normalized_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=-1)
        normalized_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=-1)
        return torch.matmul(normalized_a, normalized_b.transpose(-2, -1))

    def _compute_rowwise_sim(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        if len(embeddings_a.shape) != 2 or len(embeddings_b.shape) != 2:
            raise ValueError("Embeddings should have shape (n, dim) or (b, n, dim)")

        normalized_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=-1)
        normalized_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=-1)
        cosine_similarities = (normalized_a * normalized_b).sum(dim=-1)

        return cosine_similarities

    def _compute_loss_inbatch_inseq(
        self,
        query_embeddings: torch.Tensor,
        chunk_embeddings: torch.Tensor,
        doc_padding_mask: torch.Tensor,
        queries_chunk_indices: List[List[int]],
        n_docs_per_sample: Optional[List[int]] = None,
        targets_per_sample: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Computes a contrastive loss between queries and chunks using in-batch and in-sequence negatives.
        The queries are matched with the corresponding document among all negatives.

        Queries are of shape `(n_queries_in_batch, dim)`, and chunk_embeddings of \
            shape `(batch_size, max_n_docs, dim)`, \
                where `max_n_docs` is the maximum number of documents concatenated in one sample of the batch.
        This implies that the chunk_embeddings are padded for samples with less than `max_n_docs` documents.

        Since all negatives are considered, we directly compare all `n_queries_in_batch` \
            queries with all `batch_size * max_n_docs` documents.
        We then mask out the similarity scores for padding values in the documents (setting them to -inf),\
              so that they do not contribute to the loss.

        Args:
            query_embeddings (torch.Tensor): the computed embeddings for the queries, \
                shape: (n_queries_in_batch, dim)
            chunk_embeddings (torch.Tensor): the computed embeddings for the documents, \
                shape: (batch_size, max_n_docs, dim)
            doc_padding_mask (torch.Tensor): the mask of padding values in the documents, \
                shape: (batch_size, max_n_docs)
            queries_chunk_indices (List[List[int]]): the position in the sequence of the chunk that query `i` refers to
            n_docs_per_sample (List[int], optional): the number of documents for each sample in the batch, \
                only necesary for single context training

        Returns:
            loss (torch.Tensor): the cross-entropy loss between queries and documents
        """
        max_n_docs = chunk_embeddings.shape[1]

        # Validate gold indices against each sample's real chunks before the
        # batch is flattened. Otherwise an index can point at a padding slot
        # (or appear valid only because another sample has more chunks), which
        # hides the same student chunk-layout mismatch detected by weighted.
        for sample_idx, sample_indices in enumerate(queries_chunk_indices):
            valid_chunk_count = int((~doc_padding_mask[sample_idx]).sum().item())
            invalid_indices = [
                int(index)
                for index in sample_indices
                if int(index) < 0 or int(index) >= valid_chunk_count
            ]
            assert not invalid_indices, (
                "Query-chunk indices do not match student late-chunking output: "
                f"sample={sample_idx}, student_chunk_count={valid_chunk_count}, "
                f"gold_indices={list(sample_indices)}, invalid_indices={invalid_indices}"
            )

        # reshape the chunk embeddings and doc mask
        chunk_embeddings = chunk_embeddings.view(-1, chunk_embeddings.shape[-1])  # (batch_size * n_docs, dim)
        doc_padding_mask = doc_padding_mask.view(-1)  # (batch_size * n_docs)

        sim_scores = (
            self._compute_sim(query_embeddings, chunk_embeddings) * self.sim_score_scale
        )  # (n_queries_in_batch, batch_size * n_docs)

        # mask out the similarity scores for padding values
        padding_indices = doc_padding_mask.expand(
            query_embeddings.shape[0], -1
        )  # (n_queries_in_batch, batch_size * n_docs)
        masked_scores = sim_scores.masked_fill(padding_indices, float("-inf"))

        if self.multi_ctx_training:
            labels = torch.cat(
                [
                    torch.tensor(indices, device=sim_scores.device) + batch_idx * max_n_docs
                    for batch_idx, indices in enumerate(queries_chunk_indices)
                ]
            )
        else:
            assert len(n_docs_per_sample) == len(queries_chunk_indices), (
                "There should be the same number of original samples in the batch"
            )

            offsets = torch.tensor([0] + n_docs_per_sample[:-1], device=sim_scores.device).cumsum(dim=0)
            labels = torch.cat(
                [
                    torch.tensor(indices, device=sim_scores.device) + offsets[batch_idx]
                    for batch_idx, indices in enumerate(queries_chunk_indices)
                ]
            )

            assert labels.shape[0] == query_embeddings.shape[0], (
                "The number of labels should match the number of queries"
            )
            assert torch.all(labels < sim_scores.shape[1]), (
                "All labels should be valid indices (i.e. less than the number of documents)"
            )
            assert torch.all(labels >= 0), "All labels should be valid indices (i.e. greater or equal to 0)"

        if targets_per_sample is None:
            loss = self.cross_entropy_loss(masked_scores, labels)
        else:
            targets = torch.zeros_like(masked_scores)
            query_offset = 0
            if self.multi_ctx_training:
                bases = [batch_idx * max_n_docs for batch_idx in range(len(queries_chunk_indices))]
            else:
                if n_docs_per_sample is None:
                    raise ValueError("n_docs_per_sample is required for single-context training")
                offsets = torch.tensor([0] + n_docs_per_sample[:-1], device=masked_scores.device).cumsum(0)
                bases = offsets.tolist()
            for batch_idx, indices in enumerate(queries_chunk_indices):
                count = len(indices)
                width = targets_per_sample[batch_idx].shape[1]
                targets[query_offset:query_offset + count, bases[batch_idx]:bases[batch_idx] + width] = targets_per_sample[batch_idx]
                query_offset += count
            log_probs = torch.log_softmax(masked_scores, dim=-1)
            # Padding columns contain -inf; zero-target positions must be
            # cleared before multiplication because 0 * (-inf) is NaN.
            loss = -(targets * log_probs).masked_fill(targets == 0, 0.0).sum(dim=-1).mean()
        return loss

    def _compute_loss_from_embeddings(self, query_embeddings, doc_embeddings, labels, targets=None):
        # compute similarity scores
        sim_scores = self._compute_sim(query_embeddings, doc_embeddings) * self.sim_score_scale
        if targets is None:
            return self.cross_entropy_loss(sim_scores, labels)
        return -(targets * torch.log_softmax(sim_scores, dim=-1)).sum(dim=-1).mean()

    def _compute_teacher_targets(self, teacher_queries, teacher_chunks, teacher_padding, indices):
        """Create rank-gated soft labels from teacher query/chunk scores."""
        targets_per_sample = []
        total_pairs = enabled_pairs = 0
        query_offset = 0
        for sample_idx, sample_indices in enumerate(indices):
            valid = teacher_chunks[sample_idx][~teacher_padding[sample_idx]]
            sample_queries = teacher_queries[query_offset:query_offset + len(sample_indices)]
            query_offset += len(sample_indices)
            logits = self._compute_sim(sample_queries, valid) * self.sim_score_scale
            rows = []
            for row, gold in enumerate(sample_indices):
                total_pairs += 1
                rank = int((torch.argsort(logits[row], descending=True) == int(gold)).nonzero(as_tuple=False).item()) + 1
                if rank <= self.self_distillation_rank_k:
                    enabled_pairs += 1
                    weights = self.self_distillation_alpha * torch.exp(
                        (logits[row] - logits[row, int(gold)]).clamp_max(0.0)
                    )
                    weights[int(gold)] = 1.0
                    rows.append((weights / weights.sum()).detach())
                else:
                    target = torch.zeros_like(logits[row])
                    target[int(gold)] = 1.0
                    rows.append(target)
            targets_per_sample.append(torch.stack(rows) if rows else logits.new_empty((0, valid.shape[0])))
        return targets_per_sample, total_pairs, enabled_pairs

    def _encode_teacher_embeddings(self, query_inputs, teacher_doc_inputs, chunk_counts):
        """Encode teacher queries and independent chunks in the frozen space."""
        if self.teacher_base_model is None:
            raise ValueError("self-distillation teacher is not initialized")
        teacher_query = self._encode_query_inputs_with_encoder(
            self.teacher_base_model, query_inputs
        )
        flat = self._encode_flat_documents_with_encoder(
            self.teacher_base_model, teacher_doc_inputs
        )
        if self.normalize_embeddings:
            teacher_query = torch.nn.functional.normalize(teacher_query, dim=-1)
            flat = torch.nn.functional.normalize(flat, dim=-1)
        if flat.shape[0] != sum(chunk_counts):
            raise ValueError(
                "teacher input rows do not match teacher_chunk_counts: "
                f"rows={flat.shape[0]}, chunk_counts={list(chunk_counts)}, "
                f"expected={sum(chunk_counts)}"
            )
        max_chunks = max(chunk_counts, default=0)
        chunks = flat.new_zeros((len(chunk_counts), max_chunks, flat.shape[-1]))
        padding = torch.ones((len(chunk_counts), max_chunks), dtype=torch.bool, device=flat.device)
        cursor = 0
        for row, count in enumerate(chunk_counts):
            chunks[row, :count] = flat[cursor:cursor + count]
            padding[row, :count] = False
            cursor += count
        return teacher_query, chunks, padding

    def _encode_query_inputs_with_encoder(self, encoder, query_inputs):
        """Encode queries in micro-batches with the specified encoder."""
        reference = query_inputs.get("input_ids", query_inputs.get("attention_mask"))
        if reference is None:
            raise ValueError("query_inputs must contain input_ids or attention_mask")
        n_queries = reference.shape[0]
        micro = self.query_micro_batch_size or n_queries
        outputs = []
        with torch.no_grad():
            for start in range(0, n_queries, micro):
                end = min(start + micro, n_queries)
                micro_inputs = {
                    key: value[start:end]
                    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == n_queries
                    else value
                    for key, value in query_inputs.items()
                }
                outputs.append(encoder(micro_inputs)["sentence_embedding"])
        return torch.cat(outputs, dim=0)

    def _encode_flat_documents_with_encoder(self, encoder, inputs):
        """Encode flat independent teacher chunks in document micro-batches."""
        reference = inputs.get("input_ids", inputs.get("attention_mask"))
        if reference is None:
            raise ValueError("teacher document inputs must contain input_ids or attention_mask")
        n_rows = reference.shape[0]
        micro = self.document_micro_batch_size or n_rows
        outputs = []
        with torch.no_grad():
            for start in range(0, n_rows, micro):
                end = min(start + micro, n_rows)
                micro_inputs = {
                    key: value[start:end]
                    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == n_rows
                    else value
                    for key, value in inputs.items()
                }
                outputs.append(encoder(micro_inputs)["sentence_embedding"])
        return torch.cat(outputs, dim=0)

    def _encode_documents_microbatched(self, doc_inputs, chunk_offsets):
        """Encode complete student documents in document-sized micro-batches."""
        batch_size = doc_inputs["input_ids"].shape[0]
        micro = self.document_micro_batch_size or batch_size
        pooled_outputs, padding_outputs = [], []
        if chunk_offsets is None:
            raise ValueError("ModernBERT contextual encoding requires explicit chunk_offsets")
        if len(chunk_offsets) != batch_size:
            raise ValueError(
                "chunk_offsets must have one row per student document: "
                f"offset_rows={len(chunk_offsets)}, documents={batch_size}"
            )
        for start in range(0, batch_size, micro):
            end = min(start + micro, batch_size)
            inputs = {
                key: value[start:end]
                if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size
                else value
                for key, value in doc_inputs.items()
            }
            token_embeddings = self.base_model(inputs)["token_embeddings"]
            # Offsets define chunk identity for every ModernBERT training
            # batch; SEP token counts are never used to infer the layout.
            pooled, padding = self._pool_offset_chunks(
                token_embeddings, chunk_offsets[start:end], self.pooling_mode
            )
            pooled_outputs.append(pooled)
            padding_outputs.append(padding)
        max_chunks = max(value.shape[1] for value in pooled_outputs)
        padded_embeddings, padded_masks = [], []
        for embeddings, padding in zip(pooled_outputs, padding_outputs):
            missing = max_chunks - embeddings.shape[1]
            if embeddings.ndim == 4:
                max_chunk_tokens = max(value.shape[2] for value in pooled_outputs)
                embeddings = torch.nn.functional.pad(
                    embeddings,
                    (0, 0, 0, max_chunk_tokens - embeddings.shape[2], 0, missing),
                )
            elif missing:
                embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, missing))
            if missing:
                padding = torch.cat(
                    [padding, torch.ones((padding.shape[0], missing), dtype=torch.bool, device=padding.device)],
                    dim=1,
                )
            padded_embeddings.append(embeddings)
            padded_masks.append(padding)
        return torch.cat(padded_embeddings, dim=0), torch.cat(padded_masks, dim=0)

    def _compute_loss_batch_negatives(
        self,
        query_embeddings: torch.Tensor,
        batch_negatives: torch.Tensor,
        golden_doc_embeddings: torch.Tensor,
    ):
        assert query_embeddings.shape[0] == golden_doc_embeddings.shape[0], (
            "The number of queries should match the number of golden documents"
        )
        # compute similarity scores for the golden documents
        golden_sim_scores = self._compute_rowwise_sim(query_embeddings, golden_doc_embeddings) * self.sim_score_scale

        # compute similarity scores for the batch negatives
        batch_sim_scores = self._compute_sim(query_embeddings, batch_negatives) * self.sim_score_scale

        # concatenate the scores
        all_sim_scores = torch.cat([golden_sim_scores.unsqueeze(1), batch_sim_scores], dim=1)

        # true labels are the first scores for each query
        labels = torch.zeros(query_embeddings.shape[0], dtype=torch.long, device=query_embeddings.device)

        # compute the cross-entropy loss
        return self.cross_entropy_loss(all_sim_scores, labels)

    def _compute_loss_weighted_inbatch_inseq(
        self,
        query_embeddings: torch.Tensor,
        chunk_embeddings: torch.Tensor,
        doc_padding_mask: torch.Tensor,
        queries_chunk_indices: List[List[int]],
        targets_per_sample: Optional[List[torch.Tensor]] = None,
        # loss_type: str = "inbatch_inseq",
    ) -> torch.Tensor:
        batch_size = chunk_embeddings.shape[0]

        query_offset = 0
        in_seq_loss_arr = []
        in_batch_loss_arr = []

        # iterate over each sample in the batch
        for b_idx in range(batch_size):
            # compute the loss for the in-sequence negatives
            sequence_doc_embeddings = chunk_embeddings[b_idx]  # shape (max_n_docs, dim)
            sequence_doc_mask = doc_padding_mask[b_idx]  # shape (max_n_docs)
            sequence_doc_embeddings = sequence_doc_embeddings[~sequence_doc_mask]  # shape (n_docs, dim)

            assert len(sequence_doc_embeddings) > max(queries_chunk_indices[b_idx]), (
                "Query-chunk indices should be less than the number of documents"
            )
            sample_query_embeddings = query_embeddings[
                query_offset : query_offset + len(queries_chunk_indices[b_idx])
            ]  # shape (n_queries, dim)
            query_offset += len(queries_chunk_indices[b_idx])

            in_sequence_loss = self._compute_loss_from_embeddings(
                sample_query_embeddings,
                sequence_doc_embeddings,
                labels=torch.tensor(
                    queries_chunk_indices[b_idx],
                    dtype=torch.long,
                    device=query_embeddings.device,
                ),  # labels for this sample
                targets=None if targets_per_sample is None else targets_per_sample[b_idx],
            )
            in_seq_loss_arr.append(in_sequence_loss)

            # compute the loss for the in-batch negatives (without in-sequence negatives)
            batch_negatives = chunk_embeddings[
                torch.arange(batch_size) != b_idx
            ]  # shape (batch_size - 1, max_n_docs, dim)
            batch_doc_padding_mask = doc_padding_mask[
                torch.arange(batch_size) != b_idx
            ]  # shape (batch_size - 1, max_n_docs)
            batch_negatives = batch_negatives[~batch_doc_padding_mask]
            # TODO: check if line below is necessary
            batch_negatives = batch_negatives.view(-1, batch_negatives.shape[-1])  # shape (n_batch_negs, dim)
            golden_doc_embeddings = sequence_doc_embeddings[queries_chunk_indices[b_idx]]
            in_batch_loss = self._compute_loss_batch_negatives(
                sample_query_embeddings, batch_negatives, golden_doc_embeddings
            )
            in_batch_loss_arr.append(in_batch_loss)

        in_seq_mean_loss = torch.stack(in_seq_loss_arr).mean()
        in_batch_mean_loss = torch.stack(in_batch_loss_arr).mean()

        loss = self.lambda_seq * in_seq_mean_loss + (1 - self.lambda_seq) * in_batch_mean_loss
        return loss

    def _compute_max_sim_scores(
        self,
        q_token_embeddings: torch.Tensor,
        d_token_embeddings: torch.Tensor,
    ):
        # normalize the embeddings
        q_token_embeddings = torch.nn.functional.normalize(q_token_embeddings, p=2, dim=-1)
        d_token_embeddings = torch.nn.functional.normalize(d_token_embeddings, p=2, dim=-1)

        # perform dot product between query and document embeddings
        sim_scores = torch.einsum(
            "qnd,bmd->qbnm", q_token_embeddings, d_token_embeddings
        )  # (n_queries, n_docs, max_q_len, max_doc_len)

        # max_sim: take the max over doc embeddings, then sum over query embeddings
        max_sim_scores = sim_scores.max(dim=3)[0].sum(dim=2)
        return max_sim_scores  # (n_queries, n_docs)

    def _compute_max_sim_loss_in_batch(
        self,
        q_token_embeddings: torch.Tensor,
        bn_d_token_embeddings: torch.Tensor,
        golden_d_token_embeddings: torch.Tensor,
    ):
        assert q_token_embeddings.shape[0] == golden_d_token_embeddings.shape[0], (
            "The number of queries should match the number of golden documents"
        )
        golden_max_sim_scores = self._compute_max_sim_scores(
            q_token_embeddings, golden_d_token_embeddings
        )  # (n_queries, n_golden_docs)

        # take only the score of the golden doc for the corresponding query
        golden_max_sim_scores = golden_max_sim_scores.diag()  # (n_queries,)

        # compute similarity scores for the batch negatives
        bn_max_sim_scores = self._compute_max_sim_scores(
            q_token_embeddings, bn_d_token_embeddings
        )  # (n_queries, n_batch_negs)

        # concatenate the scores
        all_sim_scores = torch.cat([golden_max_sim_scores.unsqueeze(1), bn_max_sim_scores], dim=1)

        # true labels are the first scores for each query
        labels = torch.zeros(q_token_embeddings.shape[0], dtype=torch.long, device=q_token_embeddings.device)

        # compute the cross-entropy loss
        loss = self.cross_entropy_loss(all_sim_scores, labels)

        return loss

    def _compute_late_interaction_loss(
        self,
        q_token_embeddings: torch.Tensor,
        d_token_embeddings: torch.Tensor,
        doc_padding_mask: torch.Tensor,
        queries_chunk_indices: List[List[int]],
    ):
        # query_token_embeddings have shape (n_queries_in_batch, max_query_len, dim)
        # doc_token_embeddings have shape (batch_size, max_n_docs, max_doc_len, dim)

        batch_size = d_token_embeddings.shape[0]
        query_offset = 0
        in_seq_loss_arr = []
        in_batch_loss_arr = []

        for b_idx in range(batch_size):
            # in-sequence loss
            seq_d_token_embeddings = d_token_embeddings[b_idx]  # (max_n_docs, max_doc_len, dim)
            sequence_doc_mask = doc_padding_mask[b_idx]  # shape (max_n_docs)
            seq_d_token_embeddings = seq_d_token_embeddings[~sequence_doc_mask]  # shape (n_docs, max_doc_len, dim)

            sample_q_token_embeddings = q_token_embeddings[
                query_offset : query_offset + len(queries_chunk_indices[b_idx])
            ]  # shape (n_queries, max_q_len, dim)
            query_offset += len(queries_chunk_indices[b_idx])

            # TODO: make sure q_token_embeddings are padded with 0's
            max_sim_scores = self._compute_max_sim_scores(
                sample_q_token_embeddings, seq_d_token_embeddings
            )  # (n_queries, n_docs)

            labels = torch.tensor(
                queries_chunk_indices[b_idx],
                dtype=torch.long,
                device=q_token_embeddings.device,
            )  # labels for this sample
            in_seq_loss = self.cross_entropy_loss(max_sim_scores, labels)
            in_seq_loss_arr.append(in_seq_loss)

            # in-batch loss
            batch_negatives = d_token_embeddings[
                torch.arange(batch_size) != b_idx
            ]  # shape (batch_size - 1, max_n_docs, max_doc_len, dim)
            batch_doc_padding_mask = doc_padding_mask[
                torch.arange(batch_size) != b_idx
            ]  # shape (batch_size - 1, max_n_docs)
            batch_negatives = batch_negatives[~batch_doc_padding_mask]  # (n_batch_negs, max_doc_len, dim)
            golden_doc_token_embeddings = seq_d_token_embeddings[
                queries_chunk_indices[b_idx]
            ]  # (n_queries, max_q_len, dim)

            # compute sim and loss
            in_batch_loss = self._compute_max_sim_loss_in_batch(
                sample_q_token_embeddings, batch_negatives, golden_doc_token_embeddings
            )
            in_batch_loss_arr.append(in_batch_loss)

        # compute the weighted mean loss
        in_seq_mean_loss = torch.stack(in_seq_loss_arr).mean()
        in_batch_mean_loss = torch.stack(in_batch_loss_arr).mean()

        loss = self.lambda_seq * in_seq_mean_loss + (1 - self.lambda_seq) * in_batch_mean_loss
        return loss

    def _encode_query_inputs(self, query_inputs: dict[str, Tensor]) -> Tensor:
        """Encode queries in smaller forwards while preserving a single loss batch.

        The collator flattens every query belonging to every document in the
        device batch. Splitting that tensor here reduces the encoder's peak
        activation memory without changing the set of in-sequence or in-batch
        negatives used by the loss.
        """
        output_key = "sentence_embedding" if self.pooling_mode == "average" else "token_embeddings"
        reference = query_inputs.get("input_ids", query_inputs.get("attention_mask"))
        if reference is None:
            raise ValueError("query_inputs must contain input_ids or attention_mask")

        n_queries = reference.shape[0]
        micro_batch_size = self.query_micro_batch_size
        if micro_batch_size is None or micro_batch_size >= n_queries:
            return self.base_model(query_inputs)[output_key]

        embeddings = []
        original_seq_len = query_inputs.get("attention_mask", reference).shape[1]
        for start in range(0, n_queries, micro_batch_size):
            end = min(start + micro_batch_size, n_queries)
            micro_inputs = {
                key: value[start:end]
                if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == n_queries
                else value
                for key, value in query_inputs.items()
            }

            # Average pooling does not need a common sequence dimension across
            # microbatches, so discard padding introduced by longer queries in
            # other microbatches. Token pooling keeps the original width to
            # preserve the existing MaxSim behavior exactly.
            if self.pooling_mode == "average" and "attention_mask" in micro_inputs:
                micro_seq_len = max(1, int(micro_inputs["attention_mask"].sum(dim=1).max().item()))
                micro_inputs = {
                    key: value[:, :micro_seq_len]
                    if isinstance(value, Tensor) and value.ndim >= 2 and value.shape[1] == original_seq_len
                    else value
                    for key, value in micro_inputs.items()
                }

            embeddings.append(self.base_model(micro_inputs)[output_key])

        return torch.cat(embeddings, dim=0)

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model. Computes the loss for the given batch of queries and documents.
        Args:
            args: A list containing a dictionary with the following keys:
                - "query_inputs": The input tensors for the queries.
                - "docs_inputs": The input tensors for the documents.
            kwargs: A dictionary containing the following keys:
                - "queries_chunk_indices": A list of lists containing the indices of the chunks for each query.
                - "loss_type": The type of loss to compute. Can be "inbatch_inseq", "weighted", or "late_interaction"
                  (default: weighted).
                - "n_docs_per_sample": The number of documents per sample in the batch
                 (only needed for single context training).
                - "teacher_docs_inputs": Flat independently tokenized chunks used
                  only by the frozen teacher when self-distillation is enabled.
                - "teacher_chunk_counts": Original chunk count for each document,
                  used to restore the flat teacher rows to document groups.
                - "add_prefixes": Whether to add prefixes to the document inputs (default: False).
        """
        query_inputs = args[0]["query_inputs"]
        # ⚠️ may encounter spike in memory usage at middle of training
        # query_model_outputs = self.base_model(query_inputs)

        # 😊 so we split the queries into smaller micro-batches
        query_embeddings = self._encode_query_inputs(query_inputs)

        targets_per_sample = None
        self_distill_total = self_distill_enabled = None
        if self.training and self.self_distillation:
            if kwargs.get("loss_type", "weighted") == "late_interaction":
                raise ValueError(
                    "self_distillation currently supports weighted and inbatch_inseq losses, "
                    "not late_interaction"
                )
            teacher_doc_inputs = args[0].get("teacher_docs_inputs")
            chunk_counts = args[0].get("teacher_chunk_counts")
            if teacher_doc_inputs is None or chunk_counts is None:
                raise ValueError(
                    "self_distillation requires flat teacher chunk inputs "
                    "and teacher_chunk_counts"
                )
            if len(chunk_counts) != len(kwargs["queries_chunk_indices"]):
                raise ValueError("n_docs_per_sample must match the document batch")
            teacher_queries, teacher_chunks, teacher_padding = self._encode_teacher_embeddings(
                query_inputs, teacher_doc_inputs, chunk_counts
            )
            # The flat teacher rows are grouped solely by n_docs_per_sample;
            # verify that every query's gold index is valid before constructing
            # labels so a changed collator cannot silently misalign chunks.
            for sample_idx, indices in enumerate(kwargs["queries_chunk_indices"]):
                if any(index < 0 or index >= chunk_counts[sample_idx] for index in indices):
                    raise ValueError("queries_chunk_indices does not match teacher chunk layout")
            with torch.no_grad():
                targets_per_sample, self_distill_total, self_distill_enabled = self._compute_teacher_targets(
                    teacher_queries, teacher_chunks, teacher_padding, kwargs["queries_chunk_indices"]
                )

        doc_inputs = args[0]["docs_inputs"]
        chunk_offsets = args[0].get("chunk_offsets")
        if chunk_offsets is None:
            raise ValueError("ModernBERT contextual training requires explicit chunk_offsets")
        if self.training and self.self_distillation:
            offset_counts = [len(row) for row in chunk_offsets]
            if offset_counts != list(chunk_counts):
                raise ValueError(
                    "student chunk_offsets do not match teacher_chunk_counts: "
                    f"student={offset_counts}, teacher={list(chunk_counts)}"
                )
        chunk_embeddings, doc_padding_mask = self._encode_documents_microbatched(
            doc_inputs, chunk_offsets
        )

        if self.pooling_mode not in {"average", "tokens"}:
            raise ValueError(f"Pooling mode {self.pooling_mode} not supported. Use 'average' or 'tokens'.")

        if "loss_type" not in kwargs or kwargs["loss_type"] == "weighted":
            loss = self._compute_loss_weighted_inbatch_inseq(
                query_embeddings,
                chunk_embeddings,
                doc_padding_mask,
                kwargs["queries_chunk_indices"],
                targets_per_sample,
            )
        elif kwargs["loss_type"] == "inbatch_inseq":
            # in-seq and in-batch negatives
            loss = self._compute_loss_inbatch_inseq(
                query_embeddings,
                chunk_embeddings,
                doc_padding_mask,
                kwargs["queries_chunk_indices"],
                kwargs.get("n_docs_per_sample"),
                targets_per_sample,
            )

        elif kwargs["loss_type"] == "late_interaction":
            loss = self._compute_late_interaction_loss(
                query_embeddings,
                chunk_embeddings,
                doc_padding_mask,
                kwargs["queries_chunk_indices"],
            )
        else:
            raise ValueError(
                f"Unknown loss type: {kwargs['loss_type']}.\
                              Supported types are: inbatch_inseq, weighted, late_interaction."
            )

        outputs = {"loss": loss}
        if self_distill_total is not None:
            outputs["self_distill_query_gold_pairs"] = torch.tensor(
                self_distill_total, dtype=torch.float32, device=loss.device
            )
            outputs["self_distill_enabled_query_gold_pairs"] = torch.tensor(
                self_distill_enabled, dtype=torch.float32, device=loss.device
            )
        return outputs

    def embed_queries(self, queries):
        """
        Embeds a batch of queries.
        Args:
            queries (list[str]): A list of queries to be embedded.
        Returns:
            torch.Tensor: A tensor of shape (n_queries, dim) containing the embeddings of the queries.
        """
        self.base_model.eval()

        kwargs = {
            "show_progress_bar": self.show_progress_bar,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "convert_to_tensor": True,
            "prompt": "search_query: " if self.add_prefix else None,
        }
        if not isinstance(self.base_model, ColBERT):
            output_value = "sentence_embedding" if self.pooling_mode == "average" else "token_embeddings"
            kwargs["output_value"] = output_value

        outputs = self.base_model.encode(
            queries,
            **kwargs,
        )

        # pad token embeddings
        if self.pooling_mode == "tokens":
            outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)

            if self.add_prefix:
                # remove the prefix from the token embeddings
                outputs = outputs[:, self.n_tokens_prefix :, :]

        return outputs

    def _partition_long_doc(self, doc_chunks, n_chunks_overlap: Optional[int] = 10):
        # tokenize all chunks
        if isinstance(self.base_model, ColBERT):
            tokenized_chunks = self.tokenize(doc_chunks, is_query=False)
        else:
            tokenized_chunks = self.tokenize(doc_chunks)

        tokenized_chunks = [
            (
                tokenized_chunks["input_ids"][i, : tokenized_chunks["attention_mask"][i].sum()],
                tokenized_chunks["attention_mask"][i].sum().item(),
            )
            for i in range(tokenized_chunks["input_ids"].shape[0])
        ]

        chunk_buffer = []
        chunk_lists = []  # stores lists of chunks that will be embedded together
        valid_chunks_lists = []  # masks to avoid duplicate chunk embeddings
        overlap_in_buffer = False

        for idx, (_, chunk_size) in enumerate(tokenized_chunks):
            # if buffer is full, empty it
            if chunk_size + sum([line for _, line in chunk_buffer]) >= self.base_model.max_seq_length:
                chunk_lists.append([doc_chunks[idx] for idx, _ in chunk_buffer])
                valid_chunks = [1 for _ in range(len(chunk_buffer))]
                if overlap_in_buffer:
                    valid_chunks[:n_chunks_overlap] = [0 for _ in range(n_chunks_overlap)]
                valid_chunks_lists.append(valid_chunks)

                # empty buffer, leaving some overlapping chunks
                chunk_buffer = chunk_buffer[-n_chunks_overlap:]
                overlap_in_buffer = True

            chunk_buffer.append((idx, chunk_size))

        # add last list
        chunk_lists.append([doc_chunks[idx] for idx, _ in chunk_buffer])
        valid_chunks = [1 for _ in range(len(chunk_buffer))]
        if overlap_in_buffer:
            valid_chunks[:n_chunks_overlap] = [0 for _ in range(n_chunks_overlap)]
        valid_chunks_lists.append(valid_chunks)

        assert sum([c for valid_chunks in valid_chunks_lists for c in valid_chunks]) == len(doc_chunks)

        return chunk_lists, valid_chunks_lists

    def _embed_long_doc(self, doc_chunks):
        """Embed all chunks of a single document using overlapping context windows.

        The document is partitioned into windows that fit within the base model's
        maximum sequence length. Each window is embedded independently, embeddings
        belonging only to overlap context are discarded, and the remaining chunk
        embeddings are flattened in document order.

        Args:
            doc_chunks (list[str]): The ordered text chunks that make up one document.

        Returns:
            list[torch.Tensor]: One embedding for each retained chunk in the document.
        """
        chunk_lists, valid_chunks_lists = self._partition_long_doc(doc_chunks)
        chunk_embeddings = self.embed_batch_documents(chunk_lists)
        # ❌ Bug left by original author
        # valid_chunk_embeddings = [embeds[: sum(valid)] for embeds, valid in zip(chunk_embeddings, valid_chunks_lists)]

        # ✅ Corrected by Codex (GPT-5.6-Sol)
        valid_chunk_embeddings = [
            [
                embedding
                for embedding, keep in zip(embeds, valid)
                if keep
            ]
            for embeds, valid in zip(chunk_embeddings, valid_chunks_lists)
        ]

        # flatten array of embeddings
        valid_chunk_embeddings = [v for valids in valid_chunk_embeddings for v in valids]

        return valid_chunk_embeddings

    def embed_long_document(self, documents, n_chunks_overlap: int = 10):
        """Embed a nested document collection using overlapping context windows.

        Args:
            documents: Documents in the same ``list[list[str]]`` format accepted
                by :meth:`embed_documents`.
            n_chunks_overlap: Number of chunks carried from each window into
                the next window as context for each source document.

        Returns:
            Nested chunk embeddings aligned with the input documents and chunks.

        Window capacity is measured using the tokenizer's content-token count,
        plus one CLS token, one SEP per chunk, and the single document prefix.
        If an individual chunk exceeds the available capacity, the existing
        offset tokenizer truncates that chunk while preserving its output slot.
        """
        if n_chunks_overlap < 0:
            raise ValueError("n_chunks_overlap must be >= 0")
        print(
            "[LongDocument] Sliding-window embedding: "
            f"documents={len(documents)}, chunk_overlap={n_chunks_overlap}"
        )
        return [
            self._embed_one_long_document(doc_chunks, n_chunks_overlap)
            for doc_chunks in documents
        ]

    def _embed_one_long_document(self, doc_chunks, n_chunks_overlap):
        if not doc_chunks:
            return []

        tokenizer = self.base_model.tokenizer
        max_length = int(getattr(self.base_model, "max_seq_length", tokenizer.model_max_length))
        prefix = "search_document: " if self.add_prefix else ""
        prefix_tokens = (
            len(tokenizer(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        )
        available_content = max_length - 1 - prefix_tokens
        chunk_lengths = [
            len(tokenizer(chunk, add_special_tokens=False, truncation=False)["input_ids"])
            for chunk in doc_chunks
        ]

        windows = []
        window_start = 0
        while window_start < len(doc_chunks):
            window_end = window_start
            content_length = 0
            while window_end < len(doc_chunks):
                candidate_length = content_length + chunk_lengths[window_end] + 1
                if window_end > window_start and candidate_length > available_content:
                    break
                content_length = candidate_length
                window_end += 1

            if window_end == window_start:
                window_end += 1
            windows.append((window_start, window_end))
            if window_end == len(doc_chunks):
                break
            next_start = max(window_start + 1, window_end - n_chunks_overlap)
            window_start = next_start

        print(f"[LongDocument] chunks={len(doc_chunks)}, windows={windows}")
        outputs = []
        previous_end = 0
        for window_idx, (start, end) in enumerate(windows):
            window_embeddings = self.embed_batch_documents([doc_chunks[start:end]])[0]
            keep_from = 0 if window_idx == 0 else previous_end - start
            outputs.extend(window_embeddings[keep_from:])
            previous_end = end

        if len(outputs) != len(doc_chunks):
            raise RuntimeError(
                "Sliding-window embedding did not preserve chunk alignment: "
                f"input={len(doc_chunks)}, output={len(outputs)}, windows={windows}"
            )
        return outputs

    def embed_batch_documents(self, documents):
        """
        Embeds a batch of documents, where each document is a list of chunks.
        Args:
            documents (list[list[str]]): A list of documents, where each document is a list of chunks (strings).
                Each chunk is a string that will be tokenized and embedded.
        Returns:
            list[list[torch.Tensor]]: A list of lists of embeddings, where each inner list corresponds to a document,
                and contains one embedding per input chunk. Chunks removed by tokenizer truncation are represented by
                zero-valued placeholder embeddings appended in their original trailing positions.
        """
        # Documents are lists of source chunks. Use the same explicit offset
        # contract as training and Qwen3 inference; never recover boundaries by
        # counting SEP IDs after tokenization.
        self.base_model.eval()
        outputs = []
        for doc in documents:
            if not doc:
                outputs.append([])
                continue

            prefix = "search_document: " if self.add_prefix else ""
            doc_inputs, offsets = self._offset_tokenize_documents([doc], prefix)
            doc_inputs = {
                key: value.to(self.base_model.device)
                for key, value in doc_inputs.items()
                if isinstance(value, torch.Tensor)
            }

            with torch.no_grad():
                doc_embeddings = self.base_model(doc_inputs)["token_embeddings"]
            chunk_embeddings, padding_mask = self._pool_offset_chunks(
                doc_embeddings, offsets, self.pooling_mode
            )
            if self.normalize_embeddings:
                chunk_embeddings = torch.nn.functional.normalize(chunk_embeddings, p=2, dim=-1)

            filtered = chunk_embeddings[0, ~padding_mask[0]]
            embedding_list = [filtered[j].cpu() for j in range(len(doc))]
            if len(embedding_list) != len(doc):
                raise ValueError(
                    "Explicit chunk offsets do not match the input document: "
                    f"expected={len(doc)}, got={len(embedding_list)}"
                )
            outputs.append(embedding_list)

        return outputs

    def embed_documents(self, documents, batch_size=64):
        """

        Embeds a list of documents, where each document is a list of chunks.
        Args:
            documents (list[list[str]]): A list of documents, where each document is a list of chunks (strings).
                Each chunk is a string that will be tokenized and embedded.
            batch_size (int): The size of the batch to use for embedding the documents.
        Returns:
            list[list[torch.Tensor]]: A list of lists of embeddings, where each inner list corresponds to a document,
            and contains the embeddings of the chunks in that document."""

        # return self.embed_documents_right(documents, batch_size)


        # For max{document} <= max_seq_length
        all_outputs = []
        for i in tqdm(range(0, len(documents), batch_size), "Embedding documents", disable=not self.show_progress_bar):
            outputs = self.embed_batch_documents(documents[i : i + batch_size])
            all_outputs.extend(outputs)

        # all_outputs = [
        #     self._embed_long_doc(doc_chunks) for doc_chunks in documents
        # ]

        return all_outputs

    def _embed_documents_directional(self, documents, batch_size, direction):
        """Embed chunks using a prefix-only or suffix-only context window.

        Each target chunk is encoded in its own context window, so the returned
        embedding for chunk ``i`` is pooled from the same chunk position as in
        :meth:`embed_documents`, while the model can only attend to the selected
        side of the document.  The actual encoding is delegated to
        :meth:`embed_batch_documents` to keep tokenization, pooling, truncation,
        prefixes, and normalization identical to the regular late-chunking path.
        """
        if direction not in {"left", "right"}:
            raise ValueError("direction must be either 'left' or 'right'")

        all_outputs = []
        for start in tqdm(
            range(0, len(documents), batch_size),
            f"Embedding documents ({direction} context)",
            disable=not self.show_progress_bar,
        ):
            document_batch = documents[start : start + batch_size]
            context_windows = []
            target_positions = []
            document_indices = []

            for document_index, doc in enumerate(document_batch):
                for chunk_index in range(len(doc)):
                    if direction == "left":
                        context_windows.append(doc[: chunk_index + 1])
                        target_positions.append(chunk_index)
                    else:
                        context_windows.append(doc[chunk_index:])
                        target_positions.append(0)
                    document_indices.append(document_index)

            window_embeddings = self.embed_batch_documents(context_windows)
            batch_outputs = [[] for _ in document_batch]
            for embeddings, document_index, target_position in zip(
                window_embeddings, document_indices, target_positions
            ):
                batch_outputs[document_index].append(embeddings[target_position])

            all_outputs.extend(batch_outputs)

        return all_outputs

    def embed_documents_left(self, documents, batch_size=64):
        """Embed chunks with only their preceding context visible.

        For a document ``[C1, ..., Ci, ..., Cn]``, chunk ``Ci`` is encoded in
        the prefix window ``[C1, ..., Ci]`` and its embedding is extracted from
        that window.  Thus ``Ci`` can attend to itself and all chunks before it,
        but no chunks after it.  This is an analysis/control variant of late
        chunking, useful for measuring the contribution of left context; it is
        not intended as a separate training method.

        Args:
            documents (list[list[str]]): Documents represented as ordered lists
                of text chunks.
            batch_size (int): Number of input documents processed per batch.

        Returns:
            list[list[torch.Tensor]]: One embedding per input chunk, in the same
            document and chunk order as ``documents``.  Embedding shape and
            normalization follow :meth:`embed_documents`.
        """
        return self._embed_documents_directional(documents, batch_size, "left")

    def embed_documents_right(self, documents, batch_size=64):
        """Embed chunks with only their following context visible.

        For a document ``[C1, ..., Ci, ..., Cn]``, chunk ``Ci`` is encoded in
        the suffix window ``[Ci, ..., Cn]`` and its embedding is extracted from
        that window.  Thus ``Ci`` can attend to itself and all chunks after it,
        but no chunks before it.  This is an analysis/control variant of late
        chunking, useful for measuring the contribution of right context; it is
        not intended as a separate training method.

        Args:
            documents (list[list[str]]): Documents represented as ordered lists
                of text chunks.
            batch_size (int): Number of input documents processed per batch.

        Returns:
            list[list[torch.Tensor]]: One embedding per input chunk, in the same
            document and chunk order as ``documents``.  Embedding shape and
            normalization follow :meth:`embed_documents`.
        """
        return self._embed_documents_directional(documents, batch_size, "right")

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Activates gradient checkpointing for the current model (not sure if necessary yet).
        """
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        """
        Enables the gradients for the input embeddings (not sure if necessary yet).
        """
        self.base_model.enable_input_require_grads(**kwargs)

    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        self.base_model.save(
            path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )
