import numbers

from cde_benchmark.embedders.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from cde_benchmark.formatters.data_formatter import BEIRDataFormatter


class NanoBEIR:
    def __init__(self, path, embedder, query_prompt="", doc_prompt=""):
        self.datasets = [
            "NanoClimateFEVER",
            "NanoDBPedia",
            "NanoFEVER",
            "NanoFiQA2018",
            "NanoHotpotQA",
            "NanoMSMARCO",
            "NanoNFCorpus",
            "NanoNQ",
            "NanoQuoraRetrieval",
            "NanoSCIDOCS",
            "NanoArguAna",
            "NanoSciFact",
            "NanoTouche2020",
        ]
        self.path = path
        self.embedder = embedder
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt

    def run_task(self, task):
        formatter = BEIRDataFormatter(
            self.path + "/" + task, "train", query_prompt=self.query_prompt, doc_prompt=self.doc_prompt
        )
        metrics = self.embedder.compute_metrics_e2e(formatter)
        return metrics, len(formatter.queries_dataset)

    def run_all_tasks(self):
        results = {}
        task_support = {}
        for task in self.datasets:
            print(f"Running task: {task}")
            res, support = self.run_task(task)
            print(res)
            results[task] = res
            task_support[task] = support
        # Macro-average each scalar metric across the 13 NanoBEIR tasks.
        # Keep per-task results intact so callers can inspect task variance.
        metric_names = sorted({name for task_metrics in results.values() for name in task_metrics})
        macro_average = {
            name: sum(
                float(task_metrics[name])
                for task_metrics in results.values()
                if name in task_metrics and isinstance(task_metrics[name], numbers.Number)
            ) / sum(
                1
                for task_metrics in results.values()
                if name in task_metrics and isinstance(task_metrics[name], numbers.Number)
            )
            for name in metric_names
            if any(name in task_metrics and isinstance(task_metrics[name], numbers.Number)
                   for task_metrics in results.values())
        }
        average = {
            name: sum(float(results[task][name]) * task_support[task] for task in results
                      if name in results[task] and isinstance(results[task][name], numbers.Number)) / sum(
                task_support[task] for task in results
                if name in results[task] and isinstance(results[task][name], numbers.Number)
            )
            for name in metric_names
            if any(name in results[task] and isinstance(results[task][name], numbers.Number) for task in results)
        }
        results["macro_average"] = macro_average
        results["average"] = average
        return results


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/modernbert-embed-base")
    model._modules["0"].auto_model.config.reference_compile = False

    nanobeir = NanoBEIR("zeta-alpha-ai", SentenceTransformerEmbedder(model))
    results = nanobeir.run_all_tasks()
    print(results)
