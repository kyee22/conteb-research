#!/bin/zsh
# Run training for every matching config, one after another.
# Usage:
#   ./run_bot_configs.sh                 # default pattern: qwen3_bot*.yaml
#   ./run_bot_configs.sh 'qwen3_bot_k*.yaml'   # custom pattern (quote it)
set -e -u -o pipefail

CONFIG_DIR="../contextual-embeddings/scripts/configs/examples"
SCRIPT="../contextual-embeddings/scripts/training/training.py"
LOG_DIR="./logs"
TARGET="${1:-qwen3_bot*.yaml}"

mkdir -p "$LOG_DIR"

# null_glob: an unmatched pattern expands to nothing instead of an error.
setopt null_glob

# ${~TARGET}: zsh does not glob the result of a parameter expansion by default,
# the ~ turns globbing on for this expansion only.
files=("$CONFIG_DIR"/${~TARGET})

if (( ${#files} == 0 )); then
    echo "没找到任何 $TARGET (目录: $CONFIG_DIR)"
    exit 1
fi

echo "找到 ${#files} 个 $TARGET"
for cfg in "${files[@]}"; do
    echo "  - $cfg"
done

for cfg in "${files[@]}"; do
    name=${cfg:t:r}   # 文件名去掉目录和扩展名
    log="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date)] 启动 $name"
    if uv run accelerate launch "$SCRIPT" "$cfg" > "$log" 2>&1; then
        echo "[$(date)] ✅ $name 完成"
    else
        echo "[$(date)] ❌ $name 失败 -> $log"
    fi
done