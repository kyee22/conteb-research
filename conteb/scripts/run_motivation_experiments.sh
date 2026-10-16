#!/usr/bin/env bash
set -euo pipefail

# Edit the two model paths/types. This script can be called from any directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

uv run python3 scripts/motivation_experiments.py \
  --embedder1-model perplexity-ai/pplx-embed-v1-0.6b \
  --embedder1-type sentence_transformer \
  --embedder2-model perplexity-ai/pplx-embed-context-v1-0.6b \
  --embedder2-type pplx_contextual \
  --data-base-path illuin-conteb \
  --output-dir fixed-motivation-results-pplx \
  --sample-queries 25 \
  --seed 20261008 \

uv run python3 scripts/motivation_experiments.py \
  --embedder1-model lightonai/modernbert-embed-large \
  --embedder1-type sentence_transformer \
  --embedder2-model /root/kyx/checkpoint/modernbert-test_20260806050234 \
  --embedder2-type contextual \
  --data-base-path illuin-conteb \
  --output-dir fixed-motivation-results-MB \
  --sample-queries 25 \
  --seed 20261008 \
  --use-prefix

uv run python3 scripts/motivation_experiments.py \
  --embedder1-model Qwen/Qwen3-Embedding-0.6B \
  --embedder1-type qwen3 \
  --embedder2-model /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-late-chunking-causal-last_token-late_chunking_mean-0.1 \
  --embedder2-type qwen3_late \
  --data-base-path illuin-conteb \
  --output-dir fixed-motivation-results-qwen3-uni \
  --sample-queries 25 \
  --seed 20261008 \


uv run python3 scripts/motivation_experiments.py \
  --embedder1-model Qwen/Qwen3-Embedding-0.6B \
  --embedder1-type qwen3 \
  --embedder2-model /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-late-chunking-bidirectional-last_token-late_chunking_mean-0.1 \
  --embedder2-type qwen3_late \
  --data-base-path illuin-conteb \
  --output-dir fixed-motivation-results-qwen3-bi \
  --sample-queries 25 \
  --seed 20261008 \
