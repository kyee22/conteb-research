# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-boc-bottleneck_chunk_bidirectional-last_token-last_chunking_mean-0.1-K4 \
#     qwen3_bot \
#     qwen3-embedding-0.6b-boc-bottleneck_chunk_bidirectional-last_token-last_chunking_mean-0.1-K4

# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-boc-bottleneck_chunk_bidirectional-last_token-last_chunking_mean-0.5-K4 \
#     qwen3_bot \
#     qwen3-embedding-0.6b-boc-bottleneck_chunk_bidirectional-last_token-last_chunking_mean-0.5-K4

# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-boc-bottleneck_chunk_causal-last_token-last_chunking_mean-0.1-K4 \
#     qwen3_bot \
#     qwen3-embedding-0.6b-boc-bottleneck_chunk_causal-last_token-last_chunking_mean-0.1-K4

uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-boc-bottleneck_chunk_causal-last_token-last_chunking_mean-0.1-K8 \
    qwen3_bot \
    qwen3-embedding-0.6b-boc-bottleneck_chunk_causal-last_token-last_chunking_mean-0.1-K8

uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-boc-bottleneck_chunk_causal-last_token-last_chunking_mean-0.5-K4 \
    qwen3_bot \
    qwen3-embedding-0.6b-boc-bottleneck_chunk_causal-last_token-last_chunking_mean-0.5-K4

uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/qwen3-embedding-0.6b/qwen3-embedding-0.6b-boc-bottleneck_chunk_causal-last_token-last_chunking_mean-0.5-K8 \
    qwen3_bot \
    qwen3-embedding-0.6b-boc-bottleneck_chunk_causal-last_token-last_chunking_mean-0.5-K8
