# LongContextEmbeddingModel 差异备忘录

对比对象：

- `contextual_embeddings/models/long_context_model.py`（当前模型实现）
- `playground/long-general.py`
- `playground/long-origin.py`
- `playground/long-train.py`

## 与 `long-origin.py`

两者主体代码基本一致，只有两处行为差异：

1. `embed_documents`：当前实现按 `batch_size` 调用 `embed_batch_documents`（默认 64）；`long-origin.py` 改为逐文档调用 `_embed_long_doc`。因此当前实现默认不走重叠窗口的长文切分，超出模型最大长度的文档不会自动按窗口保留上下文；origin 版本会走 `_partition_long_doc`，再按 `valid_chunks` 过滤重叠块。
2. 截断日志：当前实现的调试前缀增加了 `⚠️`，仅影响输出文本。

## 与 `long-general.py`

当前实现相对 general 版本减少或回退了以下能力：

- **量化支持被移除**：general 会自动发现 `FlexibleQuantizer`，支持 `int8`、`binary`、`ubinary`，并在文档池化及查询 `encode` 时保持对称量化。当前构造函数仍保留 `quantization` 参数，但不保存、不应用，也不会传给 `encode`，因此该参数实际无效。
- **前缀 token 计数简化**：general 会同时剥离 tokenizer 追加的尾部 `SEP/EOS` 再计算 `n_tokens_prefix`；当前实现固定减 1，遇到 decoder-only tokenizer 或特殊 token 行为不同时可能产生偏移。
- **精确 chunk 边界被移除**：general 在 `embed_batch_documents` 中手工构造 token id 和 `(start, end)`，并将边界传入 `_late_chunking_pooling`；当前实现改为把 chunks 用 `sep_token` 拼成字符串后再搜索分隔符。这样代码更简单，但无法避免 decoder-only 模型文本中自然出现的 `sep_token` 被误认为边界，也丢失了“最后一个 chunk 无尾部分隔符”的隐式边界补全。
- **批量文档编码改为逐文档编码**：general 一次构造并前向整个 batch；当前实现逐文档 tokenize/forward，降低了不同长度 batch 在 ModernBERT 上出现 NaN 的风险，但吞吐通常更低。
- **默认 `embed_documents(batch_size)` 从 10 改为 64**：提高批次吞吐配置，但当前逐文档内循环仍决定实际前向粒度。

当前实现相对 general 保留/增加的能力：

- 保留了 `_embed_long_doc` 的重叠窗口过滤逻辑，并修复了原先用 `embeds[:sum(valid)]`（只取前 N 个）导致的过滤错误，改为按布尔标记逐项保留。
- 空文档直接返回空列表；当 tokenizer 截断导致 chunk 数不足时，追加零向量占位并打印诊断；若数量反常增加则抛出 `ValueError`。
- 文档编码统一在单文档循环中执行，并对结果按配置做归一化。

## 与 `long-train.py`

- **移除查询微批处理**：train 版新增 `query_micro_batch_size` 参数及 `_encode_query_inputs`，把查询分块前向后再拼接，降低峰值激活显存且不改变负例集合；当前实现直接 `self.base_model(query_inputs)`，显存峰值更高，但实现更简单。
- train 版保留与 origin 相同的批量文档编码和 `batch_size=64`；当前实现还包含上文的长文过滤修复、空文档/截断占位处理，以及逐文档前向改动。
- 两者训练损失、池化模式（`average`/`tokens`）、in-batch/in-sequence 及 late-interaction 损失接口基本相同。

## 结论

当前 `contextual_embeddings/models/long_context_model.py` 是偏“稳定推理/兼容性修补”的版本：通过逐文档编码和截断占位规避部分运行时问题，并修复长文重叠块过滤；代价是 general 的量化与精确边界能力、train 的查询显存优化均不再可用，同时 `embed_documents` 默认路径不会自动启用 `_embed_long_doc` 的重叠窗口策略。
