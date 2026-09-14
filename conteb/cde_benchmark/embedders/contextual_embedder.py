import torch
from contextual_embeddings import LongContextEmbeddingModel
from sentence_transformers import SentenceTransformer

from cde_benchmark.embedders.base_embedder import Embedder
from cde_benchmark.embedders.hyper_contextual_embedder import (
    HyperContextEmbedder,
    HyperContextualEmbedder,
)
from transformers import AutoModel


class ContextualEmbedder(LongContextEmbeddingModel, Embedder):
    def __init__(
        self,
        model: SentenceTransformer = None,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        device: str = "cpu",
        add_prefix: bool = False,
        quantization: str = None,
    ):
        LongContextEmbeddingModel.__init__(
            self, base_model=model, add_prefix=add_prefix, quantization=quantization
        )
        Embedder.__init__(self, is_contextual_model=True, device=device)
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size

class PPLXContextualEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = "perplexity-ai/pplx-embed-context-v1-0.6B",
        batch_size: int = 16,
        show_progress_bar: bool = True,
        device: str = "cpu",
        add_prefix: bool = False,
        normalize_embeddings: bool = True,
        quantization: str = "int8",
    ):
        Embedder.__init__(self, is_contextual_model=True, device=device)
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.quantization = quantization

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    def _check_tensors(self, tensors, label="embeddings"):
        """Check for NaN, Inf, and all-zero tensors."""
        nan_count = 0
        inf_count = 0
        zero_count = 0
        total = 0
        for i, t in enumerate(tensors):
            if isinstance(t, torch.Tensor):
                ft = t.float()
                has_nan = torch.isnan(ft).any().item()
                has_inf = torch.isinf(ft).any().item()
                is_all_zero = (ft.abs().sum() == 0).item()
                if has_nan:
                    nan_count += 1
                    print(f"[WARN] NaN in {label}[{i}], shape={t.shape}, dtype={t.dtype}")
                if has_inf:
                    inf_count += 1
                    print(f"[WARN] Inf in {label}[{i}], shape={t.shape}, dtype={t.dtype}")
                if is_all_zero:
                    zero_count += 1
                    print(f"[WARN] All-zero in {label}[{i}], shape={t.shape}, dtype={t.dtype}")
                total += 1
        if total > 0:
            print(f"[CHECK] {label}: {total} tensors, NaN={nan_count}, Inf={inf_count}, AllZero={zero_count}")
        else:
            ft = tensors.float()
            print(f"[CHECK] {label}: shape={tensors.shape}, dtype={tensors.dtype}, "
                  f"NaN={torch.isnan(ft).any().item()}, Inf={torch.isinf(ft).any().item()}, "
                  f"AllZero={(ft.abs().sum()==0).item()}")

    def embed_queries(self, queries: list[str]) -> list[torch.Tensor]:
        """
        Embeds a batch of queries.
        Args:
            queries (list[str]): A list of queries to be embedded.
        Returns:
            torch.Tensor: A tensor of shape (n_queries, dim) containing the embeddings of the queries.
        """
        docs = [[q] for q in queries]
        results = self.model.encode(
            docs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            device=self._device,
            normalize_embeddings=self.normalize_embeddings,
            quantization=self.quantization,
            convert_to_numpy=False,
        )
        # results: list[tensor(1, dim)] → list[tensor(dim,)]
        stacked = torch.stack([r[0] for r in results])
        self._check_tensors(stacked, "query_embeddings")
        return stacked

    def embed_documents(
        self, documents: list[list[str]], batch_size: int = None
    ) -> list[list[torch.Tensor]]:
        """

        Embeds a list of documents, where each document is a list of chunks.
        Args:
            documents (list[list[str]]): A list of documents, where each document is a list of chunks (strings).
                Each chunk is a string that will be tokenized and embedded.
            batch_size (int): The size of the batch to use for embedding the documents.
        Returns:
            list[list[torch.Tensor]]: A list of lists of embeddings, where each inner list corresponds to a document,
            and contains the embeddings of the chunks in that document.
        """
        bs = batch_size or self.batch_size
        results = self.model.encode(
            documents,
            batch_size=bs,
            show_progress_bar=self.show_progress_bar,
            device=self._device,
            normalize_embeddings=self.normalize_embeddings,
            quantization=self.quantization,
            convert_to_numpy=False,
        )
        # results: list[tensor(n_chunks, dim)] → list[list[tensor(dim,)]]
        flattened = [[chunk for chunk in doc] for doc in results]
        return flattened

class MergedContextualEmbedder(LongContextEmbeddingModel, Embedder):
    def __init__(
        self,
        model: SentenceTransformer = None,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        device: str = "cpu",
        add_prefix: bool = False,
        quantization: str = None,
    ):
        LongContextEmbeddingModel.__init__(
            self, base_model=model, add_prefix=add_prefix, quantization=quantization
        )
        Embedder.__init__(self, is_contextual_model=True, device=device)
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.merge_embeddings = True


class LateInteractionContextualEmbedder(LongContextEmbeddingModel, Embedder):
    def __init__(
        self,
        model: SentenceTransformer = None,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        device: str = "cpu",
        add_prefix: bool = False,
        quantization: str = None,
    ):
        LongContextEmbeddingModel.__init__(
            self, base_model=model, add_prefix=add_prefix, pooling_mode="tokens",
            quantization=quantization,
        )
        Embedder.__init__(self, is_contextual_model=True, device=device)
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size

    def process_documents(self, data_formatter, col="chunk"):
        doc_embeddings, document_ids = super().process_documents(data_formatter, col=col)
        doc_embeddings = torch.nn.utils.rnn.pad_sequence(doc_embeddings, batch_first=True)
        return doc_embeddings, document_ids

    def get_similarities(self, query_embeddings, doc_embeddings):
        # compute late interaction in batch to avoid OOM
        all_scores = []
        for q_embedding in query_embeddings:
            q_embedding = q_embedding.unsqueeze(0)
            # compute similarity
            sim_scores = self._compute_max_sim_scores(q_token_embeddings=q_embedding, d_token_embeddings=doc_embeddings)
            all_scores.append(sim_scores)

        # stack scores
        all_scores = torch.cat(all_scores, dim=0)
        # convert to numpy
        all_scores = all_scores.cpu()
        return all_scores
