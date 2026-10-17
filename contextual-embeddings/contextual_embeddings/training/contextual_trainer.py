from typing import Any

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from torch import Tensor


class ContextualTrainer(SentenceTransformerTrainer):
    def __init__(self, training_loss_type: str = "inbatch_inseq", **super_kwargs):
        super().__init__(**super_kwargs)
        self.training_loss_type = training_loss_type

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Collect features and call the model's contextual loss function.

        ``collect_features`` removes the ``queries_`` and ``docs_`` prefixes
        from collator tensors. Non-tensor metadata is forwarded separately:
        ``chunk_offsets`` identifies document content spans and must never be
        sent through tensor device movement or dropped by the trainer.

        Returns either a scalar loss or ``(loss, {})`` when the trainer asks for
        outputs during evaluation.
        """
        features, _ = self.collect_features(inputs)

        # Pass everything to the model and compute the loss in the model directly
        features_dict = {
            "query_inputs": {k: v for k, v in features[0].items()},
            "docs_inputs": {k: v for k, v in features[1].items()},
            "chunk_offsets": inputs.get("chunk_offsets"),
            "queries_chunk_indices": inputs["queries_chunk_indices"],
            "n_docs_per_sample": inputs.get("n_docs_per_sample"),
            "teacher_docs_inputs": {
                k: v for k, v in features[2].items()
            } if len(features) > 2 else None,
            "teacher_chunk_counts": inputs.get("teacher_chunk_counts"),
        }

        model_outputs = model(
            features_dict,
            queries_chunk_indices=inputs["queries_chunk_indices"],
            n_docs_per_sample=inputs.get("n_docs_per_sample"),
            loss_type=self.training_loss_type,
            add_prefixes=inputs.get("add_prefixes", False),
        )
        loss = model_outputs["loss"]
        if "diversity_loss" in model_outputs:
            self.log({"diversity_loss": model_outputs["diversity_loss"].item()})
        if "self_distill_query_gold_pairs" in model_outputs:
            total_pairs = self.accelerator.reduce(
                model_outputs["self_distill_query_gold_pairs"], reduction="sum"
            )
            enabled_pairs = self.accelerator.reduce(
                model_outputs["self_distill_enabled_query_gold_pairs"], reduction="sum"
            )
            self.log({
                "self_distill/query_gold_pairs": total_pairs.item(),
                "self_distill/enabled_query_gold_pairs": enabled_pairs.item(),
            })

        if return_outputs:
            # During prediction/evaluation, `compute_loss` will be called with `return_outputs=True`.
            # However, Sentence Transformer losses do not return outputs, so we return an empty dictionary.
            # This does not result in any problems, as the SentenceTransformerTrainingArguments sets
            # `prediction_loss_only=True` which means that the output is not used.
            return loss, {}
        return loss
