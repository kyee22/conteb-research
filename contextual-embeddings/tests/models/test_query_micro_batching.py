import torch
from torch import nn

from contextual_embeddings.models.long_context_model import LongContextEmbeddingModel


class DummyTokenizer:
    sep_token = "[SEP]"
    sep_token_id = 2
    cls_token_id = 1
    pad_token_id = 0
    model_max_length = 12

    def __init__(self):
        self.seen_texts = []

    def __call__(self, text, add_special_tokens=True, truncation=True):
        self.seen_texts.append(text)
        if text == "search_document: ":
            return {"input_ids": [9, 10]}
        if text.startswith("search_document: "):
            return {"input_ids": [9, 10, 3, 4]}
        return {"input_ids": [3, 4]}


class DummyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = DummyTokenizer()
        self.max_seq_length = 12
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.forward_shapes = []

    def forward(self, features):
        self.forward_shapes.append(tuple(features["input_ids"].shape))
        embeddings = features["input_ids"].float().sum(dim=1, keepdim=True) * self.scale
        return {"sentence_embedding": embeddings}


def test_query_micro_batching_preserves_order_and_gradients():
    base_model = DummyBaseModel()
    model = LongContextEmbeddingModel(base_model, query_micro_batch_size=2)
    input_ids = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [2, 3, 0, 0, 0],
            [4, 5, 6, 0, 0],
            [7, 8, 9, 10, 0],
            [11, 12, 13, 14, 15],
        ]
    )
    attention_mask = input_ids.ne(0).long()

    embeddings = model._encode_query_inputs(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )

    assert base_model.forward_shapes == [(2, 2), (2, 4), (1, 5)]
    assert torch.equal(embeddings.squeeze(1), input_ids.sum(dim=1).float())

    embeddings.sum().backward()
    assert base_model.scale.grad is not None
    assert base_model.scale.grad.item() == input_ids.sum().item()


def test_query_micro_batch_size_must_be_positive():
    try:
        LongContextEmbeddingModel(DummyBaseModel(), query_micro_batch_size=0)
    except ValueError as error:
        assert "query_micro_batch_size" in str(error)
    else:
        raise AssertionError("Expected query_micro_batch_size=0 to fail")


def test_offset_tokenization_preserves_chunk_layout_and_pooling():
    model = LongContextEmbeddingModel(DummyBaseModel(), document_micro_batch_size=1)
    model.base_model.tokenizer.seen_texts.clear()
    inputs, offsets = model._offset_tokenize_documents(
        [["first", "second", "third"], ["only"]], add_prefix="search_document: "
    )

    assert model.base_model.tokenizer.seen_texts == [
        "search_document: ", "search_document: first", "second", "third",
        "search_document: only",
    ]
    assert offsets == [[(3, 5), (6, 8), (9, 11)], [(3, 5)]]
    assert inputs["attention_mask"].sum(dim=1).tolist() == [12, 6]
    assert len(offsets[0]) == 3  # Every source chunk keeps its identity and order.

    hidden = torch.arange(2 * 12 * 2, dtype=torch.float32).reshape(2, 12, 2)
    pooled, padding = model._pool_offset_chunks(hidden, offsets)
    assert pooled.shape == (2, 3, 2)
    assert padding.tolist() == [[False, False, False], [False, True, True]]
    assert torch.equal(pooled[0, 0], hidden[0, 3:5].mean(0))

    _, unprefixed_offsets = model._offset_tokenize_documents([["first", "second"]])
    assert unprefixed_offsets == [[(1, 3), (4, 6)]]

    token_chunks, token_padding = model._pool_offset_chunks(
        hidden, offsets, pooling_mode="tokens"
    )
    assert token_chunks.shape == (2, 3, 2, 2)
    assert token_padding.tolist() == [[False, False, False], [False, True, True]]
    assert torch.equal(token_chunks[0, 0], hidden[0, 3:5])
