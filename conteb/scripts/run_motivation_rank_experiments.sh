#!/usr/bin/env bash
set -euo pipefail

# Edit the embedder model/type and output directory for your experiment.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

uv run python3 scripts/motivation_rank_experiments.py \
  --embedder-model /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-seq0.1_alpha0.5_rank0 \
  --embedder-type contextual \
  --data-base-path illuin-conteb \
  --output-dir motivation-rank-results-seq0.1-rank0

uv run python3 scripts/motivation_rank_experiments.py \
  --embedder-model /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-seq0.5_alpha0.5_rank0 \
  --embedder-type contextual \
  --data-base-path illuin-conteb \
  --output-dir motivation-rank-results-seq0.5-rank0

uv run python3 scripts/motivation_rank_experiments.py \
  --embedder-model /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-seq0.5_alpha0.5_rank1 \
  --embedder-type contextual \
  --data-base-path illuin-conteb \
  --output-dir motivation-rank-results-seq0.5-rank1