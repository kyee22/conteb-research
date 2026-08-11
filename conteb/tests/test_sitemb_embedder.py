from types import SimpleNamespace

import pytest
import torch

from cde_benchmark.embedders.sitemb_embedder import (
    END_OF_TEXT,
    PASSAGE_AFFIX,
    Qwen3Embedder,
    SitEmbChunkOnlyEmbedder,
    SitEmbContextualEmbedder,
)


class DummyTokenizer:
    padding_side = "left"
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(list(texts))
        tokenized = []
        for text in texts:
            if END_OF_TEXT in text:
                tokenized.append([2, self.pad_token_id, 3, 4, self.pad_token_id])
            else:
                tokenized.append([2, 4, self.pad_token_id])
        width = max(map(len, tokenized))
        input_ids = []
        attention_mask = []
        for tokens in tokenized:
            padding = width - len(tokens)
            input_ids.append([self.pad_token_id] * padding + tokens)
            attention_mask.append([0] * padding + [1] * len(tokens))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        del attention_mask
        positions = torch.arange(input_ids.shape[1]).expand_as(input_ids) + 1
        hidden = torch.stack((input_ids.float() + 1, positions.float()), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_chunk_only_embedder_returns_flat_embeddings():
    embedder = SitEmbChunkOnlyEmbedder(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        batch_size=2,
        show_progress_bar=False,
    )

    assert embedder.embed_queries(["one", "two"]).shape == (2, 2)
    assert embedder.embed_documents(["chunk", ""]).shape == (2, 2)


def test_qwen3_embedder_matches_chunk_only_preprocessing():
    tokenizer = DummyTokenizer()
    embedder = Qwen3Embedder(
        model_path="local-qwen3-model",
        model=DummyModel(),
        tokenizer=tokenizer,
        batch_size=2,
        show_progress_bar=False,
    )

    assert embedder.embed_queries(["one"]).shape == (1, 2)
    assert tokenizer.calls[0] == [embedder.query_template.format(query="one")]
    assert embedder.embed_documents(["chunk", ""]).shape == (2, 2)
    assert tokenizer.calls[1] == ["chunk", "placeholder"]


def test_qwen3_embedder_loads_a_full_model(monkeypatch):
    loaded = {}
    model = DummyModel()

    def from_pretrained(model_path, **kwargs):
        loaded["model_path"] = model_path
        loaded["kwargs"] = kwargs
        return model

    monkeypatch.setattr(
        "cde_benchmark.embedders.sitemb_embedder.AutoModel.from_pretrained",
        from_pretrained,
    )
    embedder = Qwen3Embedder(
        model_path="local-qwen3-model",
        tokenizer=DummyTokenizer(),
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        show_progress_bar=False,
    )

    assert embedder.model is model
    assert loaded == {
        "model_path": "local-qwen3-model",
        "kwargs": {
            "torch_dtype": torch.float16,
            "attn_implementation": "flash_attention_2",
        },
    }


def test_contextual_embedder_inserts_full_document_and_restores_groups():
    tokenizer = DummyTokenizer()
    embedder = SitEmbContextualEmbedder(
        model=DummyModel(),
        tokenizer=tokenizer,
        show_progress_bar=False,
    )

    embeddings = embedder.embed_documents([["a", "b"], ["c"]], batch_size=4)

    assert [len(document) for document in embeddings] == [2, 1]
    assert tokenizer.calls == [
        [
            f"a{END_OF_TEXT}{PASSAGE_AFFIX}a b",
            f"b{END_OF_TEXT}{PASSAGE_AFFIX}a b",
            f"c{END_OF_TEXT}{PASSAGE_AFFIX}c",
        ]
    ]
    contextual = torch.nn.functional.normalize(torch.tensor([1.0, 5.0]), dim=0)
    residual = torch.nn.functional.normalize(torch.tensor([1.0, 2.0]), dim=0)
    assert torch.allclose(embeddings[0][0], contextual * 0.5 + residual * 0.5)


def test_batch_size_override_must_be_positive():
    embedder = SitEmbContextualEmbedder(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        show_progress_bar=False,
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        embedder.embed_documents([["chunk"]], batch_size=0)


def test_contextual_embedder_rejects_a_truncated_separator():
    class TruncatedTokenizer(DummyTokenizer):
        def __call__(self, texts, **kwargs):
            batch = super().__call__([text.replace(END_OF_TEXT, "") for text in texts], **kwargs)
            return batch

    embedder = SitEmbContextualEmbedder(
        model=DummyModel(),
        tokenizer=TruncatedTokenizer(),
        show_progress_bar=False,
    )

    with pytest.raises(ValueError, match="truncated before its separator"):
        embedder.embed_documents([["chunk"]])
