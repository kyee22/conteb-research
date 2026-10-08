uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-late-chunking-bidirectional-mean-late_chunking_mean \
    qwen3_late \
    qwen3-embedding-0.6b-late-chunking-bidirectional-mean-late_chunking_mean

uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-late-chunking-causal-last_token-late_chunking_mean \
    qwen3_late \
    qwen3-embedding-0.6b-late-chunking-causal-last_token-late_chunking_mean

uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-late-chunking-causal-mean-late_chunking_mean \
    qwen3_late \
    qwen3-embedding-0.6b-late-chunking-causal-mean-late_chunking_mean

uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-late-chunking-chunk_bidirectional_causal-last_token-late_chunking_mean \
    qwen3_late \
    qwen3-embedding-0.6b-late-chunking-chunk_bidirectional_causal-last_token-late_chunking_mean

uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-late-chunking-chunk_bidirectional_causal-mean-late_chunking_mean \
    qwen3_late \
    qwen3-embedding-0.6b-late-chunking-chunk_bidirectional_causal-mean-late_chunking_mean