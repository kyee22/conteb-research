#!/usr/bin/env bash
set -euo pipefail

# Edit the two model paths/types. This script can be called from any directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
uv run python3 scripts/motivation_experiments.py \
  --embedder1-model lightonai/modernbert-embed-large \
  --embedder1-type sentence_transformer \
  --embedder2-model /root/kyx/checkpoint/modernbert-test_20260806050234 \
  --embedder2-type contextual \
  --data-base-path illuin-conteb \
  --output-dir motivation-results \
  --cache-dir motivation-results/cache \
  --sample-queries 25 \
  --seed 20261008 \
  --use-prefix
