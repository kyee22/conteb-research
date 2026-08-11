from typing import List, Optional, Tuple

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
    ):
        super().__init__()

        self.add_prefix = add_prefix
        self.base_model = base_model
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.sim_score_scale = sim_score_scale
        self.normalize_embeddings = normalize_embeddings
        self.multi_ctx_training = multi_ctx_training
        self.lambda_seq = lambda_seq
        # Compute the number of tokens added by the prefix.
        # Tokenize prefix with special tokens, then strip trailing SEP/EOS
        # (BERT adds [SEP] at end, decoder-only models may add EOS).
        prefix_ids = base_model.tokenizer(doc_prefix_str, add_special_tokens=True)["input_ids"]
        sep_id = base_model.tokenizer.sep_token_id
        eos_id = base_model.tokenizer.eos_token_id
        # Remove trailing sep/eos tokens (BERT's [SEP], or decoder EOS)
        while len(prefix_ids) > 0 and prefix_ids[-1] in (sep_id, eos_id):
            prefix_ids = prefix_ids[:-1]
        self.n_tokens_prefix = len(prefix_ids)
        self.show_progress_bar = show_progress_bar

        # test fix for models that do not have a sep_token (decoders)
        if self.base_model.tokenizer.sep_token is None:
            self.base_model.tokenizer.sep_token = self.base_model.tokenizer.pad_token

        # if hasattr(self.base_model, "document_prefix_id"):
        #     self.join_token_id = self.base_model.document_prefix_id
        # else:
        #     self.join_token_id = self.base_model.tokenizer.sep_token_id
        self.join_token_id = self.base_model.tokenizer.sep_token_id

        self.pooling_mode = pooling_mode

        # Auto-detect quantizer from base_model (e.g. FlexibleQuantizer in pplx-embed)
        self._quantizer = None
        self.quantization = quantization
        for module in base_model.modules():
            if type(module).__name__ == "FlexibleQuantizer":
                print('[DEBUG] found FlexibleQuantizer in base_model, using it for quantization')
                self._quantizer = module
                break
        if self.quantization is None and self._quantizer is not None:
            self.quantization = "int8"

    def tokenize(
        # self, texts: list[str] | list[dict] | list[tuple[str, str]]
        self,
        texts: list[str],
        **kwargs,
    ) -> dict[str, Tensor]:
        return self.base_model.tokenize(texts, **kwargs)

    def encode(self, args, **kwargs):
        return self.base_model.encode(args, **kwargs)

    def _apply_quantization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the same quantization as the base model's FlexibleQuantizer if available.
        Called on document chunk embeddings after mean pooling, before normalization,
        to make them symmetric with query embeddings that go through encode().
        """
        if self._quantizer is None or self.quantization is None:
            return embeddings
        if self.quantization == "int8":
            return self._quantizer._int8_quantizer(embeddings)
        elif self.quantization == "binary":
            return self._quantizer._binary_quantizer(embeddings)
        elif self.quantization == "ubinary":
            return self._quantizer._packed_binary_quantizer(embeddings)
        return embeddings

    def _get_doc_indices(
        self, batch_indices: torch.Tensor, sep_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the start and end indices of each document in a batch of sequences separated by sep_tokens.

        Args:
            sep_indices (torch.Tensor): indices of the sep_tokens in the input sequences of the batch\
                  (result of torch.where)
            batch_indices (torch.Tensor): batch indices corresponding to each sep_token (result of torch.where)

        Returns:
            starts (torch.Tensor): tensor of start indices of each document in the batch (batch_size, max_n_docs).\
                  Contains -1 for padding values.
            ends (torch.Tensor): tensor of end indices of each document in the batch (batch_size, max_n_docs).\
                  Contains -1 for padding values.
        """
        # Get batch size and max number of docs per sample
        unique, counts = torch.unique(batch_indices, return_counts=True)
        batch_size = len(unique)

        # if isinstance(self.base_model, ColBERT):
        #     max_docs = counts.max().item() - 1 # to uncount the last join token
        # else:
        #
        max_docs = counts.max().item()

        # Create output tensor
        ends = (
            torch.zeros((batch_size, max_docs), dtype=torch.long, device=batch_indices.device) - 1
        )  # -1 is a padding value
        starts = ends.clone()
        starts[:, 0] = 0

        # Fill values using masked indexing (for each sample of the batch)
        for i, group_idx in enumerate(unique):
            mask = batch_indices == group_idx
            values = sep_indices[mask]
            # if isinstance(self.base_model, ColBERT):
            #     ends[i, : len(values) - 1] = values[1:]
            #     starts[i, 1 : len(values) - 1] = values[1: -1] # + 1  # shift by 1 to avoid taking the [SEP] token
            # else:
            #     ends[i, : len(values)] = values
            #     starts[i, 1 : len(values)] = values[:-1] + 1  # shift by 1 to avoid taking the [SEP] token
            ends[i, : len(values)] = values
            starts[i, 1 : len(values)] = values[:-1] + 1  # shift by 1 to avoid taking the [SEP] token

        return starts, ends

    def _late_chunking_pooling(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        chunk_boundaries: Optional[List[Tuple[List[int], List[int]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the chunk (or document) embeddings for sequences of multiple documents separated by sep_tokens.
        Returns embeddings pooled depending on the pooling mode of the model (tokens or average pooling).

        Args:
            token_embeddings (Tensor): The token embeddings of the input sequences, of shape (batch_size, seq_len, dim).
            input_ids (Tensor): The input ids of the sequences, of shape (batch_size, seq_len).
            chunk_boundaries (Optional): Pre-computed (starts, ends) per document.
                When provided, avoids searching for sep_token_id (which has false positives for
                decoder-only models where the sep token can appear naturally in text).

        Returns:
            batch_embeddings (Tensor): The chunk embeddings of the input sequences, of shape \
                (batch_size, max_n_chunks, dim), where max_n_chunks is the maximum number of chunks in the batch,
                  in the case of average pooling.
                For token pooling, the shape is (batch_size, max_n_chunks, max_chunk_len, dim),
                  where max_chunk_len is the maximum length of a chunk in the batch.
            padding_mask (Tensor): A mask indicating which chunks are padding, of shape (batch_size, max_n_chunks).
        """
        batch_size, max_seq_len, dim = token_embeddings.shape

        if chunk_boundaries is not None:
            # Use pre-computed boundaries (from embed_batch_documents)
            max_n_chunks = max(len(b[0]) for b in chunk_boundaries)
            starts = torch.full((batch_size, max_n_chunks), -1, dtype=torch.long, device=token_embeddings.device)
            ends = torch.full((batch_size, max_n_chunks), -1, dtype=torch.long, device=token_embeddings.device)
            for i, (s, e) in enumerate(chunk_boundaries):
                starts[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=token_embeddings.device)
                ends[i, : len(e)] = torch.tensor(e, dtype=torch.long, device=token_embeddings.device)
        else:
            # Fallback: search for sep_token_id (works for BERT where [SEP] is a special token)
            batch_indices, sep_indices = torch.where(input_ids == self.base_model.tokenizer.sep_token_id)

            # Decoder-only tokenizers (e.g. Qwen) don't append EOS at the end,
            # so the last chunk lacks a trailing sep_token boundary.
            pad_id = self.base_model.tokenizer.pad_token_id
            if pad_id is not None:
                seq_lens = (input_ids != pad_id).sum(dim=1)
            else:
                seq_lens = torch.full((batch_size,), max_seq_len, device=input_ids.device)
            last_tokens = input_ids[torch.arange(batch_size, device=input_ids.device), seq_lens - 1]
            needs_implicit = last_tokens != self.base_model.tokenizer.sep_token_id
            if needs_implicit.any():
                extra_batch = torch.where(needs_implicit)[0]
                extra_pos = seq_lens[extra_batch]
                batch_indices = torch.cat([batch_indices, extra_batch])
                sep_indices = torch.cat([sep_indices, extra_pos])

            starts, ends = self._get_doc_indices(batch_indices, sep_indices)
            max_n_chunks = starts.shape[1]
        max_chunk_len = (ends - starts).max()

        # Create index tensor (to index all tokens of each chunk)
        indices = torch.arange(max_chunk_len, device=token_embeddings.device)
        # Repeat indices for each chunk
        indices = indices.expand(batch_size, max_n_chunks, -1)  # (batch_size, max_n_chunks, max_chunk_len)
        # Shift indices by start positions
        indices = indices + starts.unsqueeze(-1)

        # Create mask for valid indices
        mask = indices < ends.unsqueeze(-1)  # (batch_size, max_n_chunks, max_chunk_len)

        # Put toy index at invalid positions (for gather)
        indices[~mask] = 0

        all_chunk_lengths = ends - starts  # (batch_size, max_n_chunks)
        all_chunk_lengths = all_chunk_lengths.masked_fill(
            all_chunk_lengths == 0, 1
        )  # avoid division by zero (for empty chunks)

        # Gather values using advanced indexing for each chunk
        batch_embeddings = []
        for i in range(max_n_chunks):
            chunk_indices = indices[:, i, :].unsqueeze(-1)  # (batch_size, max_chunk_len, 1)
            chunk_mask = mask[:, i, :].unsqueeze(-1)
            chunk_lengths = all_chunk_lengths[:, i].unsqueeze(-1)  # (batch_size, 1)

            # Expand the chunk indices to the embedding dimension and gather the values
            gathered = token_embeddings.gather(1, chunk_indices.expand(-1, -1, dim))  # (batch_size, max_chunk_len, dim)

            # Mask invalid positions and compute mean
            gathered = gathered.masked_fill(~chunk_mask, 0)

            if self.pooling_mode == "average":
                chunk_embeddings = gathered.sum(dim=1) / chunk_lengths  # (batch_size, dim)
                batch_embeddings.append(chunk_embeddings)
            elif self.pooling_mode == "tokens":
                # If not average pooling, we keep the token embeddings
                batch_embeddings.append(gathered)  # (batch_size, max_chunk_len, dim)
            else:
                raise ValueError(f"Pooling mode {self.pooling_mode} not supported. Use 'average' or 'tokens'.")

        batch_embeddings = torch.stack(batch_embeddings, dim=1)

        padding_mask = starts == -1
        return (
            batch_embeddings,
            padding_mask,
        )  # return the start indices for masking the loss

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

        loss = self.cross_entropy_loss(masked_scores, labels)
        return loss

    def _compute_loss_from_embeddings(self, query_embeddings, doc_embeddings, labels):
        # compute similarity scores
        sim_scores = self._compute_sim(query_embeddings, doc_embeddings) * self.sim_score_scale
        # compute the cross-entropy loss
        return self.cross_entropy_loss(sim_scores, labels)

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

    def _add_last_join_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # find the end of each sequence with the attention mask
        seq_ends = attention_mask.sum(dim=1)
        # add a column to the input_ids to make sure there is no overflow
        input_ids = torch.cat([input_ids, torch.zeros((input_ids.shape[0], 1), device=input_ids.device)], dim=1)
        # add a join token at that spot in each row
        input_ids[torch.arange(input_ids.shape[0]), seq_ends] = self.join_token_id
        return input_ids

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
                - "add_prefixes": Whether to add prefixes to the document inputs (default: False).
        """
        query_inputs = args[0]["query_inputs"]
        query_model_outputs = self.base_model(query_inputs)

        doc_inputs = args[0]["docs_inputs"]
        doc_token_embeddings = self.base_model(doc_inputs)["token_embeddings"]

        if kwargs.get("add_prefixes", False):
            doc_token_embeddings = doc_token_embeddings[:, self.n_tokens_prefix :, :]
            for k, v in doc_inputs.items():
                doc_inputs[k] = v[:, self.n_tokens_prefix :]

        doc_inputs["input_ids"] = self._add_last_join_token(
            doc_inputs["input_ids"],
            doc_inputs["attention_mask"],
        )  # add the join token at the end of each sequence

        # for now, assume multi-docs
        chunk_embeddings, doc_padding_mask = self._late_chunking_pooling(
            doc_inputs["input_ids"],
            doc_token_embeddings,
        )

        if self.pooling_mode == "average":
            # Apply quantization to doc side (mirrors query side: Pooling → FlexibleQuantizer)
            chunk_embeddings = self._apply_quantization(chunk_embeddings)
            query_embeddings = query_model_outputs["sentence_embedding"]  # (n_queries_in_batch, dim)
        elif self.pooling_mode == "tokens":
            query_embeddings = query_model_outputs["token_embeddings"]  # (n_queries_in_batch, max_query_len, dim)
        else:
            raise ValueError(f"Pooling mode {self.pooling_mode} not supported. Use 'average' or 'tokens'.")

        if "loss_type" not in kwargs or kwargs["loss_type"] == "weighted":
            loss = self._compute_loss_weighted_inbatch_inseq(
                query_embeddings,
                chunk_embeddings,
                doc_padding_mask,
                kwargs["queries_chunk_indices"],
            )
        elif kwargs["loss_type"] == "inbatch_inseq":
            # in-seq and in-batch negatives
            loss = self._compute_loss_inbatch_inseq(
                query_embeddings,
                chunk_embeddings,
                doc_padding_mask,
                kwargs["queries_chunk_indices"],
                kwargs.get("n_docs_per_sample"),
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

        return {"loss": loss}

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
        # ``SentenceTransformer.encode`` rejects unknown kwargs, including
        # ``quantization=None`` in newer sentence-transformers versions.
        # Only pass this option when initialization found/configured one.
        if self.quantization is not None:
            kwargs["quantization"] = self.quantization
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

    def _tokenize_docs(self, documents):
        inputs_list = []

        # iterate over each document, which is a list of chunks
        for docs in documents:
            # tokenize all chunks
            doc_inputs = self.tokenize(docs, is_query=False)
            # remove CLS token from all inputs except the first
            doc_inputs = {k: v[:, 1:] for k, v in doc_inputs.items()}

            input_ids = doc_inputs["input_ids"]
            attention_mask = doc_inputs["attention_mask"]
            # take only the ids of the valid tokens of all chunks
            valid_input_ids_list = [input_ids[i, : attention_mask[i].sum()] for i in range(input_ids.shape[0])]
            # concat all chunks together
            concat_seq = torch.cat(valid_input_ids_list, dim=0)
            # add a join_token at the end for the late chunking pooling logic
            concat_seq = torch.cat(
                [concat_seq, torch.tensor([self.join_token_id], device=concat_seq.device)],
                dim=0,
            )
            inputs_list.append(concat_seq)

        # pad the sequences
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs_list, batch_first=True)
        # add the attention mask
        attention_mask = torch.zeros(padded_inputs.shape[0], padded_inputs.shape[1], device=padded_inputs.device)
        for i in range(padded_inputs.shape[0]):
            attention_mask[i, : len(inputs_list[i]) - 1] = 1

        return {
            "input_ids": padded_inputs,
            "attention_mask": attention_mask,
        }

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
        chunk_lists, valid_chunks_lists = self._partition_long_doc(doc_chunks)
        chunk_embeddings = self.embed_batch_documents(chunk_lists)
        valid_chunk_embeddings = [embeds[: sum(valid)] for embeds, valid in zip(chunk_embeddings, valid_chunks_lists)]
        # flatten array of embeddings
        valid_chunk_embeddings = [v for valids in valid_chunk_embeddings for v in valids]

        return valid_chunk_embeddings

    def embed_batch_documents(self, documents):
        """
        Embeds a batch of documents, where each document is a list of chunks.
        Builds token IDs manually (chunk by chunk) so chunk boundaries are exact,
        avoiding issues with sep_token collisions or BPE boundary merging.

        Args:
            documents (list[list[str]]): A list of documents, where each document is a list of chunks (strings).
                Each chunk is a string that will be tokenized and embedded.
        Returns:
            list[list[torch.Tensor]]: A list of lists of embeddings, where each inner list corresponds to a document,
                and contains the embeddings of the chunks in that document.
        """
        self.base_model.eval()
        sep_id = self.base_model.tokenizer.sep_token_id
        eos_id = self.base_model.tokenizer.eos_token_id
        pad_id = self.base_model.tokenizer.pad_token_id

        batch_ids = []
        batch_boundaries = []

        for doc in documents:
            ids = []
            starts = []
            ends = []

            # Add prefix tokens if enabled
            if self.add_prefix:
                prefix_ids = self.base_model.tokenizer(
                    "search_document: ", add_special_tokens=True
                )["input_ids"]
                # Strip trailing SEP/EOS that tokenizer may have appended
                while prefix_ids and prefix_ids[-1] in (sep_id, eos_id):
                    prefix_ids = prefix_ids[:-1]
                ids.extend(prefix_ids)

            for j, chunk in enumerate(doc):
                chunk_ids = self.base_model.tokenizer(
                    chunk, add_special_tokens=False
                )["input_ids"]
                # Remove pad tokens if any (e.g., from padding in single-item batch)
                if pad_id is not None:
                    chunk_ids = [tid for tid in chunk_ids if tid != pad_id]

                starts.append(len(ids))
                ids.extend(chunk_ids)
                ends.append(len(ids))

                if j < len(doc) - 1:
                    ids.append(sep_id)

            batch_ids.append(ids)
            batch_boundaries.append((starts, ends))

        # Pad to batch max length
        max_len = max(len(ids) for ids in batch_ids)
        pad_val = pad_id if pad_id is not None else 0
        input_ids = []
        attn_mask = []
        for ids in batch_ids:
            input_ids.append(ids + [pad_val] * (max_len - len(ids)))
            attn_mask.append([1] * len(ids) + [0] * (max_len - len(ids)))

        doc_inputs = {
            "input_ids": torch.tensor(input_ids, device=self.base_model.device),
            "attention_mask": torch.tensor(attn_mask, device=self.base_model.device),
        }

        with torch.no_grad():
            doc_embeddings = self.base_model(doc_inputs)["token_embeddings"]

        # Boundaries are exact — no prefix stripping needed
        chunk_embeddings, padding_mask = self._late_chunking_pooling(
            doc_inputs["input_ids"],
            doc_embeddings,
            chunk_boundaries=batch_boundaries,
        )
        # Apply quantization after pooling, before normalization
        # (mirrors query-side pipeline: Pooling → FlexibleQuantizer → normalize)
        if self.pooling_mode == "average":
            chunk_embeddings = self._apply_quantization(chunk_embeddings)
        if self.normalize_embeddings:
            chunk_embeddings = torch.nn.functional.normalize(chunk_embeddings, p=2, dim=-1)

        outputs = []
        for i in range(len(documents)):
            filtered = chunk_embeddings[i, ~padding_mask[i]]
            embedding_list = [filtered[j].cpu() for j in range(filtered.shape[0])]
            outputs.append(embedding_list)

        return outputs

    def embed_documents(self, documents, batch_size=10):
        """

        Embeds a list of documents, where each document is a list of chunks.
        Args:
            documents (list[list[str]]): A list of documents, where each document is a list of chunks (strings).
                Each chunk is a string that will be tokenized and embedded.
            batch_size (int): The size of the batch to use for embedding the documents.
        Returns:
            list[list[torch.Tensor]]: A list of lists of embeddings, where each inner list corresponds to a document,
            and contains the embeddings of the chunks in that document."""
     
        all_outputs = []
        for i in tqdm(range(0, len(documents), batch_size), "Embedding documents", disable=not self.show_progress_bar):
            outputs = self.embed_batch_documents(documents[i : i + batch_size])
            all_outputs.extend(outputs)

        return all_outputs

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
