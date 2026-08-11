from sentence_transformers import SentenceTransformer

from cde_benchmark.embedders.base_embedder import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model: SentenceTransformer = None,
        batch_size: int = 128,
        show_progress_bar: bool = True,
        add_prefix: bool = False,
        device: str = "cpu",
        quantization: str = None,
        normalize_embeddings: bool = None,
    ):
        super().__init__(is_contextual_model=False, device=device)
        self.model = model
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.query_prompt = "search_query: " if add_prefix else ""
        self.doc_prompt = "search_document: " if add_prefix else ""
        self.quantization = quantization
        self.normalize_embeddings = normalize_embeddings

    def embed_queries(self, queries):
        kwargs = dict(
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            prompt=self.query_prompt,
        )
        if self.quantization is not None:
            kwargs["quantization"] = self.quantization
        if self.normalize_embeddings is not None:
            kwargs["normalize_embeddings"] = self.normalize_embeddings
        return self.model.encode(queries, **kwargs)

    def embed_documents(self, documents):
        # Replace empty docs with placeholder to avoid IndexError in causal mask creation
        documents = [doc if doc and doc.strip() else "placeholder" for doc in documents]
        kwargs = dict(
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            prompt=self.doc_prompt,
        )
        if self.quantization is not None:
            kwargs["quantization"] = self.quantization
        if self.normalize_embeddings is not None:
            kwargs["normalize_embeddings"] = self.normalize_embeddings
        return self.model.encode(documents, **kwargs)
