from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sentence_transformers.data_collator import SentenceTransformerDataCollator
from torch import Tensor


@dataclass
class ContextualDataCollator(SentenceTransformerDataCollator):
    """Collate contextual examples for legacy or offset-based document encoders.

    With ``offset_document_tokenize`` set, the callback owns document
    tokenization and returns ``(features, offsets)``. The collator places those
    features under ``docs_*`` and the Python offset lists under
    ``chunk_offsets``. Without the callback, the historical separator-joined
    ModernBERT path is retained unchanged.
    """
    is_multi_ctx_training: bool = field(default=True, kw_only=True)
    add_prefixes: bool = field(default=False, kw_only=True)
    sep_token: Optional[str] = field(default=None, kw_only=True)
    colbert_tokenize: Optional[bool] = field(default=False, kw_only=True)
    chunk_encoding_mode: str = field(default="late", kw_only=True)
    query_template: Optional[str] = field(default=None, kw_only=True)
    query_prefix: Optional[str] = field(default=None, kw_only=True)
    document_prefix: Optional[str] = field(default=None, kw_only=True)
    offset_document_tokenize: Optional[Any] = field(default=None, kw_only=True)
    self_distillation: bool = field(default=False, kw_only=True)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """Convert dataset rows into trainer-ready query/document features.

        Args:
            features: Rows containing ``queries``, ``docs_list``, and
                ``queries_chunk_ids``. ``docs_list`` is a list of chunks per
                example, while ``queries_chunk_ids`` maps each query to its
                positive chunk index.

        Returns:
            A dictionary with ``queries_input_ids``/``queries_attention_mask``
            and corresponding ``docs_*`` tensors. Offset-based models also get
            ``chunk_offsets``: a list of half-open content-token spans per
            document. ``queries_chunk_indices`` and, for single-context
            training, ``n_docs_per_sample`` preserve the label layout. Query
            templates are applied before tokenization; document prefixes are
            applied inside the selected document tokenizer callback.

        When the callback is absent, documents are joined with ``sep_token`` and
        the old model-specific path is used. When it is present, this method
        returns early after offset tokenization so no second separator-based
        tokenization can overwrite the Qwen3 features.
        """
        # We should always be able to return a loss, label or not:
        batch = {}

        def flatten_teacher_chunks(prefix):
            """Treat every isolated teacher chunk as its own prefixed document."""
            return [
                f"{prefix}{chunk}"
                for row in features
                for chunk in row["docs_list"]
            ]

        def tokenize_teacher_chunks(prefix):
            """Tokenize independent teacher documents using the active tokenizer API.

            SentenceTransformer tokenizers accept text lists directly. The
            ``is_query`` flag belongs only to the ColBERT tokenization path;
            passing it to a regular ModernBERT tokenizer is unsupported.
            """
            chunks = flatten_teacher_chunks(prefix)
            if self.colbert_tokenize:
                return self.tokenize_fn(chunks, is_query=False)
            return self.tokenize_fn(chunks)

        # tokenize the queries

        # merge the lists of queries into one list for the tokenizer
        all_queries = [query for row in features for query in row["queries"]]

        # TODO: put prefixes as parameters
        if self.query_template:
            all_queries = [self.query_template.format(query=query) for query in all_queries]
        elif self.query_prefix is not None:
            all_queries = [f"{self.query_prefix}{query}" for query in all_queries]
        elif self.add_prefixes:
            all_queries = [f"search_query: {query}" for query in all_queries]

        tokenized = self.tokenize_fn(all_queries)
        for key, value in tokenized.items():
            batch[f"queries_{key}"] = value

        # tokenize the documents
        if self.chunk_encoding_mode not in {"late", "separate"}:
            raise ValueError("chunk_encoding_mode must be 'late' or 'separate'")
        if self.is_multi_ctx_training:
            if self.offset_document_tokenize is not None:
                tokenized, offsets = self.offset_document_tokenize(
                    [row["docs_list"] for row in features],
                    add_prefix=self.document_prefix if self.document_prefix is not None else (
                        "search_document: " if self.add_prefixes else ""
                    ),
                )
                batch["chunk_offsets"] = offsets
                for key, value in tokenized.items():
                    batch[f"docs_{key}"] = value
                if self.self_distillation:
                    # The late student sees one complete document per row.  The
                    # frozen teacher instead needs one tokenizer row per chunk;
                    # ``teacher_chunk_counts`` restores those flat rows to the
                    # original ``(document, chunk)`` layout in the model.
                    prefix = self.document_prefix if self.document_prefix is not None else (
                        "search_document: " if self.add_prefixes else ""
                    )
                    teacher_tok = tokenize_teacher_chunks(prefix)
                    for key, value in teacher_tok.items():
                        batch[f"teacher_docs_{key}"] = value
                    batch["teacher_chunk_counts"] = [
                        len(row["docs_list"]) for row in features
                    ]
                batch["queries_chunk_indices"] = [row["queries_chunk_ids"] for row in features]
                batch["add_prefixes"] = self.add_prefixes
                return batch
            concatenated_docs = [self.sep_token.join(row["docs_list"]) for row in features]
            if self.document_prefix is not None:
                concatenated_docs = [f"{self.document_prefix}{doc}" for doc in concatenated_docs]
            elif self.add_prefixes:
                concatenated_docs = [f"search_document: {doc}" for doc in concatenated_docs]

            # In separate mode retain the normal concatenated inputs for e^d,
            # and add a second flattened input set used to compute each e^b.
            if self.colbert_tokenize:
                tokenized = self.tokenize_fn(concatenated_docs, is_query=False)
            else:
                tokenized = self.tokenize_fn(concatenated_docs)
            if self.self_distillation:
                # Preserve the historical late-document student input while
                # adding one independent tokenizer row per chunk for teacher.
                prefix = self.document_prefix if self.document_prefix is not None else (
                    "search_document: " if self.add_prefixes else ""
                )
                teacher_tok = tokenize_teacher_chunks(prefix)
                for key, value in teacher_tok.items():
                    batch[f"teacher_docs_{key}"] = value
                batch["teacher_chunk_counts"] = [len(row["docs_list"]) for row in features]
            if self.chunk_encoding_mode == "separate":
                batch["n_docs_per_sample"] = [len(row["docs_list"]) for row in features]
                flat_chunks = [doc for row in features for doc in row["docs_list"]]
                if self.document_prefix is not None:
                    flat_chunks = [f"{self.document_prefix}{chunk}" for chunk in flat_chunks]
                elif self.add_prefixes:
                    # Independent chunks use the same document prefix as the
                    # concatenated input; the model strips this token span
                    # from both encoding paths in lockstep.
                    flat_chunks = [f"search_document: {chunk}" for chunk in flat_chunks]
                chunk_tok = self.tokenize_fn(flat_chunks, is_query=False) if self.colbert_tokenize else self.tokenize_fn(flat_chunks)
                for key, value in chunk_tok.items():
                    batch[f"docs_context_{key}"] = value
        else:
            if self.offset_document_tokenize is not None:
                tokenized, offsets = self.offset_document_tokenize(
                    [row["docs_list"] for row in features],
                    add_prefix=self.document_prefix if self.document_prefix is not None else (
                        "search_document: " if self.add_prefixes else ""
                    ),
                )
                batch["chunk_offsets"] = offsets
                batch["n_docs_per_sample"] = [len(row["docs_list"]) for row in features]
                for key, value in tokenized.items():
                    batch[f"docs_{key}"] = value
                if self.self_distillation:
                    prefix = self.document_prefix if self.document_prefix is not None else (
                        "search_document: " if self.add_prefixes else ""
                    )
                    teacher_tok = tokenize_teacher_chunks(prefix)
                    for key, value in teacher_tok.items():
                        batch[f"teacher_docs_{key}"] = value
                    batch["teacher_chunk_counts"] = [
                        len(row["docs_list"]) for row in features
                    ]
                batch["queries_chunk_indices"] = [row["queries_chunk_ids"] for row in features]
                batch["add_prefixes"] = self.add_prefixes
                return batch
            single_docs = [row["docs_list"] for row in features]
            batch["n_docs_per_sample"] = [len(docs) for docs in single_docs]
            flattened_docs = [doc for sublist in single_docs for doc in sublist]

            if self.document_prefix is not None:
                flattened_docs = [f"{self.document_prefix}{doc}" for doc in flattened_docs]
            elif self.add_prefixes:
                flattened_docs = [f"search_document: {doc}" for doc in flattened_docs]

            # TODO: add assertion to check that max idx in queries_chunk_indices does not exceed n_docs
            assert len(single_docs) >= 1, "Single context collation requires at least one document"

            if self.colbert_tokenize:
                tokenized = self.tokenize_fn(flattened_docs, is_query=False)
            else:
                tokenized = self.tokenize_fn(flattened_docs)
            if self.self_distillation:
                # ``flattened_docs`` already follows document/chunk order;
                # tokenize the same chunks independently for the teacher.
                prefix = self.document_prefix if self.document_prefix is not None else (
                    "search_document: " if self.add_prefixes else ""
                )
                teacher_tok = tokenize_teacher_chunks(prefix)
                for key, value in teacher_tok.items():
                    batch[f"teacher_docs_{key}"] = value
                batch["teacher_chunk_counts"] = [len(row["docs_list"]) for row in features]
            if self.chunk_encoding_mode == "separate":
                # In single-context mode ``flattened_docs`` already contains
                # one row per chunk, so reuse its tokenization as the
                # independent chunk input set.
                for key, value in tokenized.items():
                    batch[f"docs_context_{key}"] = value

        for key, value in tokenized.items():
            batch[f"docs_{key}"] = value

        # merge the queries_chunk_ids into one tensor
        batch["queries_chunk_indices"] = [row["queries_chunk_ids"] for row in features]

        batch["add_prefixes"] = self.add_prefixes

        return batch
