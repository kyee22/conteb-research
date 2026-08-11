"""Voyage AI contextualized chunk embedder."""

from typing import Any, Sequence

from cde_benchmark.embedders.base_embedder import Embedder
import voyageai

class VoyageContextualEmbedder(Embedder):
    """Embed pre-chunked documents with Voyage contextual embedding models.

    The benchmark already groups chunks by source document, so this embedder
    sends them directly to ``Client.contextualized_embed`` with backend auto
    chunking disabled. By default, the Voyage client reads ``VOYAGE_API_KEY``
    from the environment.

    Args:
        model_name: Either ``voyage-context-3`` or ``voyage-context-4``.
        batch_size: Maximum number of documents or queries in one API call.
        output_dimension: Optional embedding dimension supported by Voyage.
        api_key: Optional Voyage API key. Prefer ``VOYAGE_API_KEY`` in tests
            and production so credentials are not passed through application
            configuration.
        client: Optional preconfigured Voyage client, primarily for testing.
        device: Device to which the benchmark later moves returned embeddings.
    """

    SUPPORTED_MODELS = frozenset({"voyage-context-3", "voyage-context-4"})
    SUPPORTED_DIMENSIONS = frozenset({256, 512, 1024, 2048})

    def __init__(
        self,
        model_name: str = "voyage-context-4",
        batch_size: int = 1,
        output_dimension: int | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        device: str = "cpu",
    ) -> None:
        if model_name not in self.SUPPORTED_MODELS:
            supported = ", ".join(sorted(self.SUPPORTED_MODELS))
            raise ValueError(f"model_name must be one of: {supported}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if output_dimension is not None and output_dimension not in self.SUPPORTED_DIMENSIONS:
            supported = ", ".join(map(str, sorted(self.SUPPORTED_DIMENSIONS)))
            raise ValueError(f"output_dimension must be one of: {supported}")
        if client is not None and api_key is not None:
            raise ValueError("api_key cannot be provided together with client")

        super().__init__(is_contextual_model=True, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dimension = output_dimension
        self.client = voyageai.Client()

    def _embed_batch(self, inputs: Sequence[Any], input_type: str) -> list[list[list[float]]]:
        request: dict[str, Any] = {
            "inputs": list(inputs),
            "model": self.model_name,
            "input_type": input_type,
            "output_dtype": "float",
            "enable_auto_chunking": False,
        }
        if self.output_dimension is not None:
            request["output_dimension"] = self.output_dimension

        response = self.client.contextualized_embed(**request)
        results: list[list[list[float]] | None] = [None] * len(inputs)
        for result in response.results:
            if not 0 <= result.index < len(results) or results[result.index] is not None:
                raise RuntimeError("Voyage returned invalid contextualized embedding indices")
            results[result.index] = result.embeddings

        if any(result is None for result in results):
            raise RuntimeError("Voyage returned fewer contextualized embedding results than requested")
        return [result for result in results if result is not None]

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed search queries with ``input_type='query'``."""
        embeddings: list[list[float]] = []
        for start in range(0, len(queries), self.batch_size):
            batch_results = self._embed_batch(queries[start : start + self.batch_size], "query")
            for result in batch_results:
                if len(result) != 1:
                    raise RuntimeError("Voyage returned more than one embedding for a query")
                embeddings.append(result[0])
        return embeddings

    def embed_documents(
        self,
        documents: list[list[str]],
        batch_size: int | None = None,
    ) -> list[list[list[float]]]:
        """Embed already-split chunks while retaining each document's context.

        Args:
            documents: Documents grouped as lists of pre-split chunk strings.
            batch_size: Optional per-call document count override.

        Returns:
            Embeddings grouped exactly like ``documents`` and in chunk order.
        """
        request_batch_size = self.batch_size if batch_size is None else batch_size
        if request_batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(not document for document in documents):
            raise ValueError("documents must not contain an empty chunk list")

        embeddings: list[list[list[float]]] = []
        for start in range(0, len(documents), request_batch_size):
            batch = documents[start : start + request_batch_size]
            batch_results = self._embed_batch(batch, "document")
            for document, result in zip(batch, batch_results):
                if len(result) != len(document):
                    raise RuntimeError("Voyage returned a different number of embeddings than input chunks")
                embeddings.append(result)
        return embeddings
