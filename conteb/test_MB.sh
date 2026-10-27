# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-naive_alpha0.5_rank1 \
#     contextual \
#     modernbert-self-distill-naive_alpha0.5_rank1 \
#     --use-prefix \
#     --run-nanobeir

# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-naive_alpha0.5_rank0 \
#     contextual \
#     modernbert-self-distill-naive_alpha0.5_rank0 \
#     --use-prefix \
#     --run-nanobeir

# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-seq0.1_alpha0.5_rank0 \
#     contextual \
#     modernbert-self-distill-seq0.1_alpha0.5_rank0 \
#     --use-prefix \
#     --run-nanobeir

# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-seq0.1_alpha0.5_rank1 \
#     contextual \
#     modernbert-self-distill-seq0.1_alpha0.5_rank1 \
#     --use-prefix \
#     --run-nanobeir

# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-seq0.2_alpha0.5_rank0 \
#     contextual \
#     modernbert-self-distill-seq0.2_alpha0.5_rank0 \
#     --use-prefix \
#     --run-nanobeir

# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-seq0.2_alpha0.5_rank1 \
#     contextual \
#     modernbert-self-distill-seq0.2_alpha0.5_rank1 \
#     --use-prefix \
#     --run-nanobeir

# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/conteb-research/conteb/checkpoints/test/modernbert-self-distill-seq0.5_alpha0.5_rank0 \
#     contextual \
#     modernbert-self-distill-seq0.5_alpha0.5_rank0 \
#     --use-prefix \
#     --run-nanobeir

uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/modernbert/self-distill/modernbert-self-distill-seq0.5_alpha0.5_rank1 \
    contextual \
    modernbert-self-distill-seq0.5_alpha0.5_rank0 \
    --use-prefix \

# uv run python scripts/evaluation/eval_conteb.py \
#     /root/kyx/checkpoint/modernbert-test_20260806050234 \
#     contextual \
#     modernbert-self-distill-repro \
#     --use-prefix

uv run python scripts/evaluation/eval_conteb.py \
    /root/kyx/conteb-research/conteb/checkpoints/modernbert/self-distill/modernbert-self-distill-seq0.5_alpha0.5_rank1_full_as_teacher \
    contextual \
    modernbert-self-distill-seq0.5_alpha0.5_rank1_full_as_teacher \
    --use-prefix 