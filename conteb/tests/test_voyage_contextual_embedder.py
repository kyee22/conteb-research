from types import SimpleNamespace

import pytest

from cde_benchmark.embedders.voyage_contextual_embedder import VoyageContextualEmbedder


class FakeVoyageClient:
    def __init__(self):
        self.calls = []

    def contextualized_embed(self, **kwargs):
        self.calls.append(kwargs)
        results = []
        for index, item in enumerate(kwargs["inputs"]):
            chunks = item if isinstance(item, list) else [item]
            embeddings = [[float(index), float(chunk_index)] for chunk_index, _ in enumerate(chunks)]
            results.append(SimpleNamespace(index=index, embeddings=embeddings))
        return SimpleNamespace(results=results)


def test_documents_are_sent_as_prechunked_contextual_inputs():
    client = FakeVoyageClient()
    embedder = VoyageContextualEmbedder(
        model_name="voyage-context-3",
        batch_size=2,
        output_dimension=512,
        client=client,
    )
    documents = [["first", "second"], ["third"], ["fourth"]]

    embeddings = embedder.embed_documents(documents)

    assert [len(document) for document in embeddings] == [2, 1, 1]
    assert client.calls == [
        {
            "inputs": documents[:2],
            "model": "voyage-context-3",
            "input_type": "document",
            "output_dtype": "float",
            "enable_auto_chunking": False,
            "output_dimension": 512,
        },
        {
            "inputs": documents[2:],
            "model": "voyage-context-3",
            "input_type": "document",
            "output_dtype": "float",
            "enable_auto_chunking": False,
            "output_dimension": 512,
        },
    ]


def test_queries_use_contextualized_query_api():
    client = FakeVoyageClient()
    embedder = VoyageContextualEmbedder(model_name="voyage-context-4", batch_size=2, client=client)

    embeddings = embedder.embed_queries(["one", "two", "three"])

    assert len(embeddings) == 3
    assert [call["input_type"] for call in client.calls] == ["query", "query"]
    assert [call["inputs"] for call in client.calls] == [["one", "two"], ["three"]]
    assert all("output_dimension" not in call for call in client.calls)


@pytest.mark.parametrize("model_name", ["voyage-context-3", "voyage-context-4"])
def test_supported_model_names(model_name):
    embedder = VoyageContextualEmbedder(model_name=model_name, client=FakeVoyageClient())

    assert embedder.model_name == model_name


def test_invalid_model_name_is_rejected():
    with pytest.raises(ValueError, match="model_name must be one of"):
        VoyageContextualEmbedder(model_name="voyage-3", client=FakeVoyageClient())


def test_embedder_has_docstring():
    assert VoyageContextualEmbedder.__doc__
