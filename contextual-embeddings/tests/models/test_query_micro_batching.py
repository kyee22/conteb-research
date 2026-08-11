import torch
from torch import nn

from contextual_embeddings.models.long_context_model import LongContextEmbeddingModel


class DummyTokenizer:
    sep_token = "[SEP]"
    sep_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [1, 2]}


class DummyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = DummyTokenizer()
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
