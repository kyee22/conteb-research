#!/bin/zsh
set -e -u -o pipefail

CONFIG_DIR="../contextual-embeddings/scripts/configs/examples"
SCRIPT="../contextual-embeddings/scripts/training/training.py"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# 关键：没匹配到时展开为空数组，而不是原样字符串
setopt null_glob
files=("$CONFIG_DIR"/qwen3*.yaml)

if (( ${#files} == 0 )); then
    echo "没找到 qwen3*.yaml"
    exit 1
else
    echo "找到 ${#files} 个 qwen3*.yaml"
    for cfg in "${files[@]}"; do
        echo "  - $cfg"
    done
fi

for cfg in "${files[@]}"; do
    name=${cfg:t:r}          # zsh 内置：取文件名再去掉扩展名
    log="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date)] 启动 $name"
    if uv run accelerate launch "$SCRIPT" "$cfg" > "$log" 2>&1; then
        echo "[$(date)] ✅ $name 完成"
    else
        echo "[$(date)] ❌ $name 失败 -> $log"
    fi
done