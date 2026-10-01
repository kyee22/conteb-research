import torch
from torch import nn

from contextual_embeddings.models.qwen3_late_chunking_model import Qwen3LateChunkingModel


class DummyTokenizer:
    padding_side = "left"


class DummyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = DummyTokenizer()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.forward_shapes = []

    def forward(self, features):
        input_ids = features["input_ids"]
        self.forward_shapes.append(tuple(input_ids.shape))
        hidden = input_ids.float().unsqueeze(-1).mul(self.scale)
        return {"token_embeddings": hidden}


def make_model(base_model, query_micro_batch_size=None, document_micro_batch_size=None):
    model = Qwen3LateChunkingModel.__new__(Qwen3LateChunkingModel)
    nn.Module.__init__(model)
    model.base_model = base_model
    model.query_pooling = "last_token"
    model.query_micro_batch_size = query_micro_batch_size
    model.document_micro_batch_size = document_micro_batch_size
    model.document_pooling = "late_chunking_mean"
    model.sim_score_scale = 20.0
    model.multi_ctx_training = True
    model.cross_entropy_loss = nn.CrossEntropyLoss()
    return model


def test_query_micro_batching_splits_batch_and_preserves_gradients():
    base_model = DummyBaseModel()
    model = make_model(base_model, query_micro_batch_size=1)
    input_ids = torch.tensor([[0, 0, 1, 2], [0, 3, 4, 5], [0, 0, 0, 6]])
    attention_mask = input_ids.ne(0).long()

    embeddings = model._encode_queries_microbatched(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )

    assert base_model.forward_shapes == [(1, 4), (1, 4), (1, 4)]
    assert torch.equal(embeddings.squeeze(-1), torch.tensor([2.0, 5.0, 6.0]))
    embeddings.sum().backward()
    assert base_model.scale.grad is not None


def test_document_micro_batching_preserves_order_and_offsets():
    base_model = DummyBaseModel()
    model = make_model(base_model, document_micro_batch_size=1)
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    attention_mask = input_ids.ne(0).long()
    offsets = [[(0, 1)], [(0, 2)]]

    chunks, padding = model._encode_documents_microbatched(
        {"input_ids": input_ids, "attention_mask": attention_mask}, offsets
    )

    assert base_model.forward_shapes == [(1, 2), (1, 3)]
    assert chunks.shape == (2, 1, 1)
    assert torch.equal(chunks[:, 0, 0], torch.tensor([1.0, 3.5]))
    assert not padding.any()


def test_micro_batching_preserves_full_batch_negative_loss():
    input_queries = torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5], [0, 0, 0, 6]])
    query_mask = input_queries.ne(0).long()
    input_docs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    doc_mask = input_docs.ne(0).long()
    offsets = [[(0, 1), (1, 3)], [(0, 2)]]

    full_base = DummyBaseModel()
    micro_base = DummyBaseModel()
    micro_base.load_state_dict(full_base.state_dict())
    full_model = make_model(full_base)
    micro_model = make_model(micro_base, query_micro_batch_size=1, document_micro_batch_size=1)
    full_query = full_model._encode_queries_microbatched(
        {"input_ids": input_queries, "attention_mask": query_mask}
    )
    full_docs, full_padding = full_model._encode_documents_microbatched(
        {"input_ids": input_docs, "attention_mask": doc_mask}, offsets
    )
    micro_query = micro_model._encode_queries_microbatched(
        {"input_ids": input_queries, "attention_mask": query_mask}
    )
    micro_docs, micro_padding = micro_model._encode_documents_microbatched(
        {"input_ids": input_docs, "attention_mask": doc_mask}, offsets
    )

    full_loss = full_model._compute_loss(full_query, full_docs, full_padding, [[0, 1], [0]])
    micro_loss = micro_model._compute_loss(micro_query, micro_docs, micro_padding, [[0, 1], [0]])
    assert torch.allclose(full_loss, micro_loss)
