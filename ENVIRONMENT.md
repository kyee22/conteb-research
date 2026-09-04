# Environment

The benchmark environment is managed from `conteb/` with Python 3.12:

```bash
cd conteb
UV_CACHE_DIR=/tmp/uv-cache uv sync --python /usr/bin/python3.12 --extra dev --no-install-package flash-attn --inexact
```

This installs the CUDA 12.8 PyTorch build (`torch==2.11.0+cu128`) and both
local packages as editable installs. The `contextual-embeddings` source is
wired through `conteb/pyproject.toml`, so the lock file no longer references
the original server's absolute path.

## FlashAttention on a GPU node

`flash-attn==2.7.2.post1` is intentionally deferred on hosts without a
working NVIDIA driver/CUDA compiler. On a GPU node with CUDA 12.x available,
run:

```bash
cd conteb
CUDA_HOME=/usr/local/cuda-12.8 UV_CACHE_DIR=/tmp/uv-cache \
  uv sync --python /usr/bin/python3.12 --extra dev
```

If CUDA is installed at another path, set `CUDA_HOME` accordingly. Verify:

```bash
uv run --no-sync python -c \
  'import torch, flash_attn; print(torch.__version__, torch.cuda.is_available())'
```

Use `uv run --no-sync ...` for tests/commands on this CPU-only host so uv does
not retry the deferred FlashAttention build.
