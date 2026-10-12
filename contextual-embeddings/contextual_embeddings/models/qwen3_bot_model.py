"""Configurable Qwen3 contextual embeddings with explicit chunk offsets and bottleneck (BOT) tokens."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_QUERY_TEMPLATE = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:{query}"
)

DEFAULT_ATTENTION_MODE = "causal"
DEFAULT_QUERY_POOLING = "last_token"
DEFAULT_DOCUMENT_POOLING = "late_chunking_mean"
DEFAULT_NUM_BOT_TOKENS = 4
BOT_TOKEN_TEMPLATE = "<|bot_{index}|>"
VALID_BOT_INIT = {"mean_noise", "random_normal", "sampled_tokens", "token"}
DEFAULT_BOT_INIT = "mean_noise"
DEFAULT_BOT_INIT_TOKEN = "<|endoftext|>"
DEFAULT_BOT_INIT_NOISE = 0.1


def initialize_bot_embeddings(weight, new_ids, reference_rows, mode,
                              token_id=None, noise=DEFAULT_BOT_INIT_NOISE, seed=0):
    """Overwrite the embedding rows ``weight[new_ids]`` in place for the BOT tokens.

    Args:
        weight: ``(rows, hidden)`` input-embedding matrix.
        new_ids: Row indices of the freshly added BOT tokens.
        reference_rows: Number of leading rows that belong to the original
            vocabulary. Statistics (mean, std) are taken over these rows only,
            never over the BOT rows or the spare untrained rows.
        mode: One of ``VALID_BOT_INIT``.

            * ``"mean_noise"``: mean of the original embeddings plus Gaussian
              noise with std ``noise * std(weight)``. Each BOT row gets its own
              noise draw. (Default.)
            * ``"random_normal"``: independent ``N(0, std(weight)^2)`` per
              entry, i.e. a typical-norm random vector.
            * ``"sampled_tokens"``: each BOT row is a copy of the embedding of
              a randomly chosen original vocabulary token.
            * ``"token"``: every BOT row starts from the embedding of the
              vocabulary token ``token_id`` (e.g. ``<|endoftext|>``) plus the
              same noise as ``"mean_noise"``; ``noise=0`` gives identical rows.
        token_id: Source row for ``mode="token"``.
        noise: Relative noise scale for ``"mean_noise"`` and ``"token"``.
        seed: Seed of a private CPU generator, so the same seed gives the same
            rows on every rank and in every run without touching global RNG state.
    """
    if mode not in VALID_BOT_INIT:
        raise ValueError(f"bot_init must be one of {sorted(VALID_BOT_INIT)}")
    generator = torch.Generator().manual_seed(seed)
    hidden = weight.shape[1]
    with torch.no_grad():
        reference = weight[:reference_rows]
        std = reference.std().item()

        def jitter():
            return torch.randn(hidden, generator=generator) * std * noise

        if mode == "mean_noise":
            center = reference.mean(0, dtype=torch.float32).cpu()
            rows = [center + jitter() for _ in new_ids]
        elif mode == "random_normal":
            rows = [torch.randn(hidden, generator=generator) * std for _ in new_ids]
        elif mode == "sampled_tokens":
            picks = torch.randint(0, reference_rows, (len(new_ids),), generator=generator).tolist()
            rows = [reference[i].float().cpu() for i in picks]
        else:
            center = weight[token_id].float().cpu()
            rows = [center + jitter() for _ in new_ids]
        for row_id, row in zip(new_ids, rows):
            weight[row_id] = row.to(device=weight.device, dtype=weight.dtype)


def build_chunk_groups(offsets, batch_size, length):
    """Return a ``(batch, length)`` long tensor with the chunk index of every position.

    Content positions ``[start, end)`` of chunk ``j`` get ``j``. The explicit
    terminal token at position ``end`` also gets ``j`` because it terminates
    that chunk. Everything else (BOT prefix, right padding) stays ``-1``.
    Built on CPU, so there are no GPU syncs; padding is blocked later by the
    validity mask, so a group id on a padded slot is harmless.
    """
    groups = torch.full((batch_size, length), -1, dtype=torch.long)
    for row, row_offsets in enumerate(offsets):
        for group, (start, end) in enumerate(row_offsets):
            groups[row, start:min(end, length)] = group
            if end < length:
                groups[row, end] = group
    return groups


def build_attention_visibility(mode, valid, offsets=None, num_bot_tokens=0):
    """Return a bool tensor ``(batch, length, length)``; True means row may attend to column.

    Args:
        mode: ``"bidirectional"``, ``"chunk_bidirectional_causal"`` or
            ``"bottleneck_chunk"``.
        valid: ``(batch, length)`` mask, 1 for real tokens and 0 for padding.
        offsets: Per-document list of ``(start, end)`` chunk spans in sequence
            coordinates (BOT prefix already included in the coordinates).
        num_bot_tokens: Number of BOT positions at the start of each row.

    ``bottleneck_chunk`` rules (rows = query token, columns = key token):
        * BOT row   -> every valid token (BOT and all chunks).
        * chunk row -> all BOT tokens and the tokens of its own chunk
          (content + terminal), bidirectionally.
        * chunk row -> never another chunk.
    """
    valid = valid.bool()
    batch_size, length = valid.shape
    pair_valid = valid[:, :, None] & valid[:, None, :]
    if mode == "bidirectional":
        return pair_valid
    groups = build_chunk_groups(offsets, batch_size, length).to(valid.device)
    qg, kg = groups[:, :, None], groups[:, None, :]
    if mode == "chunk_bidirectional_causal":
        return pair_valid & ((qg == kg) | (kg < qg))
    if mode == "bottleneck_chunk":
        is_bot = torch.arange(length, device=valid.device) < num_bot_tokens
        return pair_valid & (is_bot[None, :, None] | is_bot[None, None, :] | (qg == kg))
    raise ValueError(f"Unsupported attention mode for a custom mask: {mode}")


class Qwen3LateChunkingModel(SentenceTransformer):
    """Qwen3 SentenceTransformer with configurable masks and contextual pooling.

    The wrapped ``base_model`` remains a normal SentenceTransformer. This class
    adds only the document-context training contract: explicit document chunk
    offsets, a Qwen3-compatible attention mask, configurable query pooling, and
    configurable per-chunk document pooling. It does not infer chunk boundaries
    from ``sep_token``, ``eos_token_id``, or ``pad_token_id``.

    With ``attention_mode="bottleneck_chunk"`` every document row is
    ``[BOT_1..BOT_K] content(c1) T content(c2) T ...``. The BOT tokens read the
    whole document, each chunk reads only the BOT tokens and itself, so chunk
    vectors receive document context exclusively through the BOT tokens.
    Queries are never affected: they keep the native causal mask and
    last-token pooling of Qwen3-Embedding.
    """
    VALID_ATTENTION = {"causal", "bidirectional", "chunk_bidirectional_causal", "bottleneck_chunk"}
    VALID_QUERY_POOLING = {"last_token", "mean"}
    VALID_DOCUMENT_POOLING = {"late_chunking_mean", "late_chunking_last_token"}

    def __init__(self, base_model, 
                    attention_mode=DEFAULT_ATTENTION_MODE, 
                    query_pooling=DEFAULT_QUERY_POOLING,
                    document_pooling=DEFAULT_DOCUMENT_POOLING, 
                    query_template=DEFAULT_QUERY_TEMPLATE,
                    query_micro_batch_size=None, 
                    document_micro_batch_size=None, 
                    num_bot_tokens=None,
                    bot_init=DEFAULT_BOT_INIT,
                    bot_init_token=DEFAULT_BOT_INIT_TOKEN,
                    bot_init_noise=DEFAULT_BOT_INIT_NOISE,
                    bot_init_seed=0,
                    **kwargs):
        """Create a Qwen3 contextual model or restore one from a checkpoint.

        Args:
            base_model: A Qwen3-backed ``SentenceTransformer`` instance, or a
                saved checkpoint directory containing the base model and
                ``qwen3_late_chunking.json`` metadata.
            attention_mode: ``"causal"`` for the native decoder mask,
                ``"bidirectional"`` for all valid tokens to attend to one
                another, ``"chunk_bidirectional_causal"`` for bidirectional
                attention within a document chunk and visibility to earlier
                chunks only, or ``"bottleneck_chunk"`` for K prepended BOT
                tokens that see the whole document while every chunk sees only
                the BOT tokens and itself. Query forwards remain causal in both
                chunk-aware modes because queries have no document chunk
                offsets.
            query_pooling: ``"last_token"`` selects the final attended token
                (Qwen3's official pooling position); ``"mean"`` averages all
                attended query positions, including the terminal token when the
                tokenizer supplies one.
            document_pooling: ``"late_chunking_mean"`` averages hidden states
                in each explicit content span; 
                ``"late_chunking_last_token"`` select the explicit terminal
                ``<|endoftext|>`` state immediately after each content span,
                matching Qwen3's official last-token pooling.
            query_template: Format string containing ``{query}``, applied by
                both the training collator and :meth:`embed_queries`.
            query_micro_batch_size: Optional number of queries encoded per
                forward while keeping the complete query set for the loss.
            document_micro_batch_size: Optional number of documents encoded
                per forward. Each document remains intact, so its chunks keep
                their full contextual attention.
            num_bot_tokens: Number K of BOT tokens prepended to every document.
                Only valid with ``attention_mode="bottleneck_chunk"``, where
                ``None`` means ``DEFAULT_NUM_BOT_TOKENS``. The tokens
                ``<|bot_0|> ... <|bot_{K-1}|>`` are added to the tokenizer
                (if missing) and their embedding rows are initialised according
                to ``bot_init``; they are trained with the rest of the model.
            bot_init: Initialisation of the new BOT embedding rows, see
                :func:`initialize_bot_embeddings`: ``"mean_noise"`` (default),
                ``"random_normal"``, ``"sampled_tokens"`` or ``"token"``. It is
                applied only when the BOT tokens are first created; restoring a
                checkpoint keeps the trained rows untouched.
            bot_init_token: Source vocabulary token for ``bot_init="token"``.
            bot_init_noise: Noise scale relative to the embedding-matrix std,
                used by ``"mean_noise"`` and ``"token"``.
            bot_init_seed: Seed for the initialisation noise / token sampling.
            **kwargs: Runtime options such as normalization and contrastive-loss
                scale. Unknown options are ignored for compatibility with the
                existing configue model construction.
        """
        checkpoint = Path(base_model) if isinstance(base_model, (str, Path)) else None
        if checkpoint:
            metadata_path = checkpoint / "qwen3_late_chunking.json"
            if metadata_path.is_file():
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                attention_mode = meta.get("attention_mode", attention_mode)
                query_pooling = meta.get("query_pooling", query_pooling)
                document_pooling = meta.get("document_pooling", document_pooling)
                query_template = meta.get("query_template", query_template)
                num_bot_tokens = meta.get("num_bot_tokens", num_bot_tokens)
            base_model = SentenceTransformer(str(checkpoint), trust_remote_code=True)
        if attention_mode not in self.VALID_ATTENTION:
            raise ValueError(f"attention_mode must be one of {sorted(self.VALID_ATTENTION)}")
        if query_pooling not in self.VALID_QUERY_POOLING:
            raise ValueError(f"query_pooling must be one of {sorted(self.VALID_QUERY_POOLING)}")
        if document_pooling not in self.VALID_DOCUMENT_POOLING:
            raise ValueError(f"document_pooling must be one of {sorted(self.VALID_DOCUMENT_POOLING)}")
        if attention_mode == "bottleneck_chunk":
            num_bot_tokens = DEFAULT_NUM_BOT_TOKENS if num_bot_tokens is None else int(num_bot_tokens)
            if num_bot_tokens < 1:
                raise ValueError("attention_mode='bottleneck_chunk' requires num_bot_tokens >= 1")
        else:
            if num_bot_tokens:
                raise ValueError("num_bot_tokens is only supported with attention_mode='bottleneck_chunk'")
            num_bot_tokens = 0
        if bot_init not in VALID_BOT_INIT:
            raise ValueError(f"bot_init must be one of {sorted(VALID_BOT_INIT)}")
        if bot_init_noise < 0:
            raise ValueError("bot_init_noise must be >= 0")
        super().__init__()
        self.base_model = base_model
        self.attention_mode = attention_mode
        self.query_pooling = query_pooling
        self.document_pooling = document_pooling
        self.query_template = query_template
        self.num_bot_tokens = num_bot_tokens
        self.bot_init = bot_init
        self.bot_init_token = bot_init_token
        self.bot_init_noise = float(bot_init_noise)
        self.bot_init_seed = int(bot_init_seed)
        print(f"Qwen3LateChunkingModel: attention_mode={attention_mode}, query_pooling={query_pooling}, document_pooling={document_pooling}, num_bot_tokens={num_bot_tokens}")
        print(f"Qwen3LateChunkingModel: query_template={query_template}")
        self.query_prefix = ""
        self.document_prefix = ""
        self.normalize_embeddings = kwargs.pop("normalize_embeddings", True)
        self.sim_score_scale = kwargs.pop("sim_score_scale", 20.0)
        self.multi_ctx_training = kwargs.pop("multi_ctx_training", True)
        self.lambda_seq = kwargs.pop("lambda_seq", 0.5)
        self.show_progress_bar = kwargs.pop("show_progress_bar", True)
        self.batch_size = kwargs.pop("batch_size", 16)
        self.query_micro_batch_size = self._validate_micro_batch_size(
            query_micro_batch_size, "query_micro_batch_size"
        )
        self.document_micro_batch_size = self._validate_micro_batch_size(
            document_micro_batch_size, "document_micro_batch_size"
        )
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.terminal_token_id = self.base_model.tokenizer.pad_token_id
        # Dataset construction only needs a reversible textual placeholder;
        # actual Qwen3 chunk boundaries come from offsets later.
        self.dataset_separator = self.base_model.tokenizer.pad_token or "<|endoftext|>"
        self._active_chunk_offsets = None
        self.bot_token_ids = self._setup_bot_tokens()
        self._install_attention_hook()

    def _setup_bot_tokens(self):
        """Register ``<|bot_i|>`` tokens and return their ids (empty list when K is 0).

        The tokens are added to the tokenizer only if missing, so restoring a
        checkpoint (whose tokenizer already contains them) never re-adds or
        re-initialises them. Qwen3's embedding matrix has more rows than the
        tokenizer has tokens, so the new ids usually fit without resizing; the
        matrix is resized only when they do not. Freshly added rows are set to
        the mean embedding plus small deterministic noise so that the K tokens
        start close to an ordinary token but are not exactly identical.
        """
        if self.num_bot_tokens == 0:
            return []
        tokenizer = self.base_model.tokenizer
        auto_model = self.base_model[0].auto_model
        names = [BOT_TOKEN_TEMPLATE.format(index=i) for i in range(self.num_bot_tokens)]
        vocab = tokenizer.get_vocab()
        missing = [name for name in names if name not in vocab]
        old_size = len(tokenizer)
        token_id = None
        if missing and self.bot_init == "token":
            # Fail before the tokenizer is modified.
            if self.bot_init_token not in vocab:
                raise ValueError(f"bot_init_token {self.bot_init_token!r} is not in the tokenizer vocabulary")
            token_id = vocab[self.bot_init_token]
        if missing:
            tokenizer.add_tokens(missing, special_tokens=True)
        ids = tokenizer.convert_tokens_to_ids(names)
        embeddings = auto_model.get_input_embeddings()
        resized = False
        if max(ids) >= embeddings.weight.shape[0]:
            auto_model.resize_token_embeddings(len(tokenizer))
            embeddings = auto_model.get_input_embeddings()
            resized = True
        if missing:
            initialize_bot_embeddings(
                embeddings.weight,
                tokenizer.convert_tokens_to_ids(missing),
                old_size,
                self.bot_init,
                token_id=token_id,
                noise=self.bot_init_noise,
                seed=self.bot_init_seed,
            )
            print(
                f"Qwen3LateChunkingModel: added {len(missing)} BOT tokens, bot_init={self.bot_init}, "
                f"noise={self.bot_init_noise}, seed={self.bot_init_seed}, tokenizer_size={len(tokenizer)}, "
                f"embedding_rows={embeddings.weight.shape[0]}, resized_embeddings={resized}"
            )
        else:
            print(f"Qwen3LateChunkingModel: all {self.num_bot_tokens} BOT tokens already exist, tokenizer_size={len(tokenizer)}, embedding_rows={embeddings.weight.shape[0]}")
        return list(ids)

    @staticmethod
    def _validate_micro_batch_size(value, name):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be a positive integer or None")
        return value

    @staticmethod
    def _slice_inputs(inputs, start, end, batch_size):
        return {
            key: value[start:end]
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size
            else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _trim_inputs_to_length(inputs, max_length, padding_side="right"):
        """Trim sequence-shaped features without changing token order.

        Tokenizers pad a collated batch to its longest row. After splitting a
        right-padded batch, that padding is often unnecessary and can dominate
        the SDPA allocation for long-context inputs. Left-padded Qwen queries
        are deliberately kept at their original width because removing their
        padding would change the model's implicit RoPE position ids.
        """
        reference = inputs.get("input_ids", inputs.get("attention_mask"))
        if (
            reference is None
            or reference.ndim < 2
            or max_length <= 0
            or reference.shape[1] <= max_length
        ):
            return inputs
        sequence_length = reference.shape[1]
        if padding_side == "left":
            start, end = sequence_length - max_length, sequence_length
        else:
            start, end = 0, max_length
        return {
            key: value[:, start:end]
            if isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[1] == sequence_length
            else value
            for key, value in inputs.items()
        }

    def _trim_micro_inputs(self, inputs, padding_side):
        """Remove padding columns that are not needed by one micro-batch."""
        if padding_side == "left":
            return inputs
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None or attention_mask.ndim != 2:
            return inputs
        max_length = max(1, int(attention_mask.sum(dim=1).max().item()))
        return self._trim_inputs_to_length(inputs, max_length, padding_side)

    @staticmethod
    def _pad_chunk_outputs(outputs):
        """Restore a common chunk dimension after document micro-batching."""
        max_chunks = max((pooled.shape[1] for pooled, _ in outputs), default=0)
        padded_embeddings = []
        padded_masks = []
        for pooled, padding in outputs:
            missing = max_chunks - pooled.shape[1]
            if missing:
                pooled = F.pad(pooled, (0, 0, 0, missing))
                padding = torch.cat(
                    [padding, torch.ones((padding.shape[0], missing), dtype=torch.bool, device=padding.device)],
                    dim=1,
                )
            padded_embeddings.append(pooled)
            padded_masks.append(padding)
        if not padded_embeddings:
            return None, None
        return torch.cat(padded_embeddings, dim=0), torch.cat(padded_masks, dim=0)

    def tokenize(self, texts, **kwargs):
        """Tokenize texts using the wrapped SentenceTransformer.

        Args:
            texts: A sequence of strings accepted by the base tokenizer.
            **kwargs: Tokenizer options, such as ``is_query`` or truncation
                settings, forwarded unchanged.

        Returns:
            A feature dictionary, normally containing batched ``input_ids`` and
            ``attention_mask`` tensors. No document offsets are produced here;
            :meth:`_offset_tokenize_documents` produces those explicitly.
        """
        return self.base_model.tokenize(texts, **kwargs)

    def encode(self, args, **kwargs):
        """Delegate generic encoding to the wrapped SentenceTransformer.

        This compatibility method preserves the ordinary SentenceTransformer
        API. Qwen3 contextual query/document evaluation should use
        :meth:`embed_queries` and :meth:`embed_documents`, because generic
        ``encode`` does not carry document chunk offsets (nor the BOT prefix).
        """
        return self.base_model.encode(args, **kwargs)

    def _offset_tokenize_documents(self, documents, add_prefix=""):
        """Tokenize document chunks and return padded tensors plus exact spans.

        Args:
            documents: A sequence of documents. Each document is an ordered
                sequence of chunk strings: ``[[c1, c2], [c3]]``. Empty chunks
                are retained as zero-length spans.
            add_prefix: Text prepended to every chunk before tokenization.

        Returns:
            ``(features, offsets)``. ``features`` contains right-padded
            ``input_ids`` and a binary ``attention_mask``. Every row has the
            form ``[BOT_1..BOT_K] content(c1), terminal, content(c2), terminal,
            ...`` (the BOT prefix exists only when ``num_bot_tokens > 0``);
            right-padding uses the same terminal/pad id but has mask value zero.
            ``offsets[i][j] == (start, end)`` is a half-open interval into row
            ``i`` covering only content tokens of chunk ``j``, expressed in
            sequence coordinates, i.e. already shifted by the BOT prefix.
            BOT tokens, separators and padding are never inside an offset.

        The tokenizer's automatic final ``<|endoftext|>`` is removed from each
        independently tokenized chunk before the explicit separator is added.
        This is why pooling remains correct even though Qwen3 uses the same
        ``<|endoftext|>`` id as its padding token.
        """
        tokenizer = self.base_model.tokenizer
        bot_ids = list(self.bot_token_ids)
        rows, offsets = [], []
        for chunks in documents:
            ids_row, row_offsets = list(bot_ids), []
            for chunk in chunks:
                encoded = tokenizer(f"{add_prefix}{chunk}", add_special_tokens=True,
                                    truncation=True, return_tensors="pt")
                ids = encoded["input_ids"][0][encoded["attention_mask"][0].bool()].tolist()
                if ids and ids[-1] == self.terminal_token_id:
                    ids = ids[:-1]
                start = len(ids_row)
                ids_row.extend(ids)
                row_offsets.append((start, len(ids_row)))
                ids_row.append(self.terminal_token_id)
            rows.append(torch.tensor(ids_row, dtype=torch.long))
            offsets.append(row_offsets)
        if not rows:
            empty = torch.empty((0, 0), dtype=torch.long)
            return {"input_ids": empty, "attention_mask": empty}, offsets
        padded = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=self.terminal_token_id)
        mask = torch.zeros_like(padded)
        for i, row in enumerate(rows):
            mask[i, :row.numel()] = 1
        return {"input_ids": padded, "attention_mask": mask}, offsets

    @staticmethod
    def _attention_mask_dtype(module, device):
        """dtype for the additive mask: must match the attention query dtype."""
        if device.type == "cuda" and torch.is_autocast_enabled():
            getter = getattr(torch, "get_autocast_dtype", None)
            return getter("cuda") if getter else torch.get_autocast_gpu_dtype()
        return module.get_input_embeddings().weight.dtype

    def _install_attention_hook(self):
        """Install a per-instance Qwen3 pre-hook for the configured mask.

        Qwen3 accepts a 4-D additive mask with shape ``(batch, 1, query_len,
        key_len)``. Zero entries mean visible attention and the minimum finite
        value of the mask dtype means blocked attention. The boolean
        visibility matrix itself comes from :func:`build_attention_visibility`:

        * ``bidirectional``: every valid token pair is visible.
        * ``chunk_bidirectional_causal``: a token sees its own chunk and
          earlier chunks, never later ones.
        * ``bottleneck_chunk``: BOT rows see every valid token; chunk rows see
          the BOT tokens and their own chunk only.

        The explicit terminal token belongs to the chunk it terminates, so the
        content of chunk i never sees a separator of another chunk. Padding is
        blocked by the validity mask. Chunk-aware modes only apply while
        ``_active_chunk_offsets`` is set (document forwards); query forwards
        keep Qwen3's native causal mask. The hook is removed when another Qwen3
        contextual wrapper is installed on the same base model.
        """
        auto_model = getattr(self.base_model[0], "auto_model", None)
        if auto_model is None or getattr(auto_model.config, "model_type", None) != "qwen3":
            raise ValueError("Qwen3LateChunkingModel requires a Qwen3 SentenceTransformer model")
        previous = getattr(auto_model, "_qwen3_late_hook", None)
        if previous is not None:
            previous.remove()

        def hook(module, args, kwargs):
            """Rewrite a 2-D validity mask into the selected 4-D additive mask."""
            if self.attention_mode == "causal":
                return args, kwargs
            ref = kwargs.get("input_ids", args[0] if args else kwargs.get("inputs_embeds"))
            valid = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            if ref is None or valid is None or valid.ndim != 2:
                return args, kwargs
            offsets = getattr(self, "_active_chunk_offsets", None)
            if self.attention_mode in {"chunk_bidirectional_causal", "bottleneck_chunk"} and not offsets:
                # Query sequences have no document chunk offsets. Keep their
                # normal causal mask; the chunk-aware mask is document-only.
                return args, kwargs
            visible = build_attention_visibility(
                self.attention_mode, valid, offsets, self.num_bot_tokens
            )
            bsz, length = visible.shape[:2]
            dtype = self._attention_mask_dtype(module, ref.device)
            additive = torch.zeros((bsz, 1, length, length), dtype=dtype, device=ref.device)
            kwargs["attention_mask"] = additive.masked_fill(~visible[:, None], torch.finfo(dtype).min)
            return args, kwargs
        self.base_model[0].auto_model._qwen3_late_hook = self.base_model[0].auto_model.register_forward_pre_hook(hook, with_kwargs=True)

    def _pool_queries(self, inputs):
        """Pool query token states according to ``query_pooling``.

        Args:
            inputs: A Qwen3 SentenceTransformer feature dictionary with
                ``input_ids``/``attention_mask`` and any other model features.

        Returns:
            A tensor of shape ``(num_queries, hidden_size)``. ``last_token``
            selects the final valid position in each row, which handles both
            left and right padding. ``mean`` averages every position whose mask
            is one and divides by the valid-token count.
            An all-zero mask is clamped to one to avoid division by zero.
        """
        hidden = self.base_model(inputs)["token_embeddings"]
        mask = inputs["attention_mask"].bool()
        if self.query_pooling == "last_token":
            positions = torch.arange(mask.shape[1], device=mask.device).expand(mask.shape[0], -1)
            pos = positions.masked_fill(~mask, -1).max(dim=1).values.clamp_min(0)
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), pos]
        return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def _pool_documents(self, hidden, offsets):
        """Pool document chunks from explicit half-open token spans.

        Args:
            hidden: Token states shaped ``(batch, sequence_length, hidden_size)``.
            offsets: A Python list where ``offsets[i]`` contains ``(start, end)``
                spans for document ``i``. Spans are in the same sequence
                coordinates as ``hidden[i]`` (so they already account for the
                BOT prefix) and must exclude BOT tokens, separators and
                padding. The position ``end`` must contain the explicit
                terminal separator used by last-token pooling.

        Returns:
            ``(embeddings, padding_mask)``. ``embeddings`` has shape
            ``(batch, max_chunks, hidden_size)`` and is zero-filled after each
            row's real chunks. ``padding_mask`` has shape ``(batch, max_chunks)``
            and is true only for those zero-filled slots. Empty spans produce a
            zero vector for mean pooling, while last-token pooling uses the
            following terminal separator. Empty spans remain real, non-padding
            chunks so output order and count match the input.
        """
        rows = []
        for row_hidden, row_offsets in zip(hidden, offsets):
            chunks = []
            for start, end in row_offsets:
                if self.document_pooling in {"late_chunking_last_token"}:
                    # ``end`` is the first position after the content span and
                    # is occupied by the explicit Qwen3 terminal separator
                    # (<|endoftext|>). Qwen3's official pooling extracts this
                    # terminal state, so late_chunking_last must use it rather
                    # than the preceding content token.
                    if end >= row_hidden.shape[0]:
                        raise ValueError(
                            "Document offsets must be followed by an explicit "
                            "terminal token for late_chunking_last pooling"
                        )
                    chunks.append(row_hidden[end])
                elif end <= start:
                    chunks.append(row_hidden.new_zeros(row_hidden.shape[-1]))
                else:
                    chunks.append(row_hidden[start:end].mean(0))
            rows.append(chunks)
        max_n = max((len(x) for x in rows), default=0)
        result = hidden.new_zeros((len(rows), max_n, hidden.shape[-1]))
        padding = torch.ones((len(rows), max_n), dtype=torch.bool, device=hidden.device)
        for i, chunks in enumerate(rows):
            if chunks:
                result[i, :len(chunks)] = torch.stack(chunks)
                padding[i, :len(chunks)] = False
        return result, padding

    def _encode_queries_microbatched(self, inputs):
        """Encode all queries in micro-batches without changing their order."""
        reference = inputs.get("input_ids", inputs.get("attention_mask"))
        if reference is None:
            raise ValueError("query_inputs must contain input_ids or attention_mask")
        batch_size = reference.shape[0]
        micro_batch_size = self.query_micro_batch_size or batch_size
        outputs = []
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            micro_inputs = self._slice_inputs(inputs, start, end, batch_size)
            micro_inputs = self._trim_micro_inputs(
                micro_inputs,
                getattr(self.base_model.tokenizer, "padding_side", "right"),
            )
            outputs.append(self._pool_queries(micro_inputs))
        return torch.cat(outputs, dim=0)

    def _encode_documents_microbatched(self, inputs, offsets):
        """Encode complete documents in micro-batches and restore batch order."""
        batch_size = inputs["input_ids"].shape[0]
        micro_batch_size = self.document_micro_batch_size or batch_size
        outputs = []
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            micro_inputs = self._slice_inputs(inputs, start, end, batch_size)
            # Offset tokenization is always right-padded, so truncating the
            # trailing padding leaves every document span unchanged.
            micro_inputs = self._trim_micro_inputs(micro_inputs, "right")
            self._active_chunk_offsets = offsets[start:end]
            try:
                hidden = self.base_model(micro_inputs)["token_embeddings"]
                outputs.append(self._pool_documents(hidden, offsets[start:end]))
            finally:
                self._active_chunk_offsets = None
        return self._pad_chunk_outputs(outputs)

    def _compute_sim(self, a, b):
        """Compute normalized pairwise cosine similarities.

        Args:
            a: Tensor shaped ``(queries, hidden_size)`` or a compatible batched
                tensor.
            b: Tensor shaped ``(documents, hidden_size)`` or a compatible batched
                tensor.

        Returns:
            Matrix/tensor of dot products after L2 normalization on the final
            dimension. The caller applies the configured logit scale.
        """
        return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-2, -1)

    def _compute_loss(self, queries, chunks, padding, indices, n_docs_per_sample=None):
        """Compute contrastive loss over flattened valid document chunks.

        Args:
            queries: ``(num_queries, hidden_size)`` query embeddings.
            chunks: ``(batch, max_chunks, hidden_size)`` document embeddings.
            padding: Boolean ``(batch, max_chunks)`` mask for artificial chunk
                slots, not token padding.
            indices: Per-example lists mapping each query to its target chunk.
            n_docs_per_sample: Required for single-context training; gives the
                unpadded number of chunks in each example.

        Returns:
            A scalar cross-entropy loss. Every valid chunk from the device batch
            is a negative, while the target index is offset according to the
            configured multi-context or single-context layout.
        """
        scores = self._compute_sim(queries, chunks.reshape(-1, chunks.shape[-1])) * self.sim_score_scale
        scores = scores.masked_fill(padding.reshape(1, -1), float("-inf"))
        max_docs = chunks.shape[1]
        if self.multi_ctx_training:
            labels = torch.cat([torch.tensor(ix, device=scores.device) + i * max_docs for i, ix in enumerate(indices)])
        else:
            offsets = torch.tensor([0] + n_docs_per_sample[:-1], device=scores.device).cumsum(0)
            labels = torch.cat([torch.tensor(ix, device=scores.device) + offsets[i] for i, ix in enumerate(indices)])
        return self.cross_entropy_loss(scores, labels)

    def _compute_loss_from_embeddings(self, queries, documents, labels):
        """Compute in-sequence cross-entropy for one sample's documents.

        ``queries`` has shape ``(n_queries, hidden_size)`` and ``documents``
        has shape ``(n_documents, hidden_size)``. ``labels`` contains the
        document index for each query. This is the in-sequence component used
        by ModernBERT's weighted loss.
        """
        scores = self._compute_sim(queries, documents) * self.sim_score_scale
        return self.cross_entropy_loss(scores, labels)

    def _compute_loss_batch_negatives(self, queries, batch_negatives, gold_documents):
        """Compute the in-batch-negative component for one training sample.

        ``gold_documents`` contains the positive document for each query and
        ``batch_negatives`` contains valid documents from other examples. The
        positive score is placed in column zero, matching the parent model's
        label convention.
        """
        if queries.shape[0] != gold_documents.shape[0]:
            raise ValueError("queries and gold_documents must have equal length")
        gold_scores = (F.normalize(queries, dim=-1) * F.normalize(gold_documents, dim=-1)).sum(-1)
        if batch_negatives.numel():
            negative_scores = self._compute_sim(queries, batch_negatives)
            scores = torch.cat([gold_scores[:, None], negative_scores], dim=1)
        else:
            scores = gold_scores[:, None]
        labels = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)
        return self.cross_entropy_loss(scores * self.sim_score_scale, labels)

    def _compute_loss_inbatch_inseq(self, queries, chunks, padding, indices, n_docs_per_sample=None):
        """Compute the parent's global in-batch/in-sequence objective.

        All valid chunks in the device batch become columns in one logits
        matrix. In multi-context mode, labels are offset by each sample's
        padded chunk width; in single-context mode they are offset by the
        actual number of chunks in preceding samples.
        """
        flat = chunks.reshape(-1, chunks.shape[-1])
        scores = self._compute_sim(queries, flat) * self.sim_score_scale
        scores = scores.masked_fill(padding.reshape(1, -1), float("-inf"))
        if self.multi_ctx_training:
            max_docs = chunks.shape[1]
            labels = torch.cat([
                torch.tensor(sample_indices, device=queries.device) + sample_idx * max_docs
                for sample_idx, sample_indices in enumerate(indices)
            ])
        else:
            if n_docs_per_sample is None:
                raise ValueError("n_docs_per_sample is required when multi_ctx_training is false")
            offsets = torch.tensor([0] + n_docs_per_sample[:-1], device=queries.device).cumsum(0)
            labels = torch.cat([
                torch.tensor(sample_indices, device=queries.device) + offsets[sample_idx]
                for sample_idx, sample_indices in enumerate(indices)
            ])
        return self.cross_entropy_loss(scores, labels)

    def _compute_loss_weighted(self, queries, chunks, padding, indices):
        """Compute ModernBERT's ``lambda_seq`` weighted objective.

        For each sample, the loss is ``lambda_seq * in_sequence_loss +
        (1-lambda_seq) * in_batch_loss``. Thus changing ``lambda_seq`` changes
        the scalar objective while preserving the same positives and negatives.
        """
        batch_size = chunks.shape[0]
        query_offset = 0
        sequence_losses, batch_losses = [], []
        for sample_idx in range(batch_size):
            valid_docs = chunks[sample_idx][~padding[sample_idx]]
            sample_indices = indices[sample_idx]
            sample_queries = queries[query_offset:query_offset + len(sample_indices)]
            query_offset += len(sample_indices)
            labels = torch.tensor(sample_indices, dtype=torch.long, device=queries.device)
            sequence_losses.append(self._compute_loss_from_embeddings(sample_queries, valid_docs, labels))
            other = torch.arange(batch_size, device=chunks.device) != sample_idx
            batch_negatives = chunks[other][~padding[other]]
            batch_losses.append(self._compute_loss_batch_negatives(sample_queries, batch_negatives, valid_docs[labels]))
        return self.lambda_seq * torch.stack(sequence_losses).mean() + (1 - self.lambda_seq) * torch.stack(batch_losses).mean()

    def forward(self, *args, **kwargs):
        """Run one contextual training/evaluation batch.

        Args:
            args[0]: Dictionary containing ``query_inputs``, ``docs_inputs``,
                ``chunk_offsets``, and ``queries_chunk_indices``. The optional
                ``n_docs_per_sample`` is used for single-context labels.
                ``docs_inputs`` / ``chunk_offsets`` must come from
                :meth:`_offset_tokenize_documents` so the BOT prefix and the
                shifted offsets are consistent.
            **kwargs: Compatibility arguments accepted by
                ``ContextualTrainer``; the model reads batch metadata directly.

        Returns:
            ``{"loss": scalar_tensor}``. Query states are pooled independently,
            document states are pooled from explicit offsets, and no BOT,
            separator or right-padding token is included in a document embedding.
        """
        batch = args[0]
        try:
            # Queries are independent sequences. Chunk-specific visibility is
            # only installed for the document forward below.
            self._active_chunk_offsets = None
            queries = self._encode_queries_microbatched(batch["query_inputs"])
            chunks, padding = self._encode_documents_microbatched(
                batch["docs_inputs"], batch["chunk_offsets"]
            )
            loss_type = kwargs.get("loss_type", "weighted")
            if loss_type == "weighted":
                loss = self._compute_loss_weighted(queries, chunks, padding, batch["queries_chunk_indices"])
            elif loss_type == "inbatch_inseq":
                loss = self._compute_loss_inbatch_inseq(
                    queries, chunks, padding, batch["queries_chunk_indices"], batch.get("n_docs_per_sample")
                )
            elif loss_type == "late_interaction":
                raise ValueError(
                    "loss_type='late_interaction' requires token-level query/document pooling; "
                    "Qwen3LateChunkingModel currently produces one vector per chunk"
                )
            else:
                raise ValueError("Unknown loss_type. Supported values: weighted, inbatch_inseq, late_interaction")
            return {"loss": loss}
        finally:
            self._active_chunk_offsets = None

    @torch.inference_mode()
    def embed_queries(self, queries, batch_size=32):
        """Embed retrieval queries in batches with progress reporting.

        Args:
            queries: Iterable of query strings. Each string is formatted with
                ``query_template`` before tokenization.
            batch_size: Number of queries tokenized and encoded per forward.

        Returns:
            A tensor shaped ``(len(queries), hidden_size)`` on the model device.
            Query order is preserved across batches. The output uses
            ``last_token`` or masked ``mean`` pooling and is L2-normalized when
            ``normalize_embeddings`` is true. Queries never receive BOT tokens
            and always use the native causal mask in every attention mode except
            ``bidirectional``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.base_model.eval()
        queries = list(queries)
        outputs = []
        for start in tqdm(
            range(0, len(queries), batch_size),
            desc="Embedding queries",
            disable=not self.show_progress_bar,
        ):
            query_batch = queries[start : start + batch_size]
            inputs = self.tokenize(
                [self.query_template.format(query=q) for q in query_batch]
            )
            inputs = {
                k: v.to(self.base_model.device)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
            outputs.append(self._pool_queries(inputs))

        if not outputs:
            hidden_size = self.base_model.get_sentence_embedding_dimension()
            return torch.empty(
                (0, hidden_size),
                device=self.base_model.device,
                dtype=next(self.base_model.parameters()).dtype,
            )

        out = torch.cat(outputs, dim=0)
        return F.normalize(out, dim=-1) if self.normalize_embeddings else out

    @torch.inference_mode()
    def embed_documents(self, documents, batch_size=1):
        """Embed chunked documents with late contextual pooling.

        Args:
            documents: Iterable of documents, each represented as an ordered
                sequence of chunk strings. For example ``[["c1", "c2"],
                ["c3"]]``. Empty chunks remain in their original position and
                return zero vectors.
            batch_size: Number of documents tokenized and encoded per forward.

        Returns:
            A list with one list per input document. Each inner list contains one
            ``(hidden_size,)`` CPU tensor per input chunk, in input order. The
            configured late mean or late last-token pooling is applied only to
            content spans; BOT tokens, explicit separators and right-padding are
            excluded.
        """
        outputs = []
        for start in tqdm(range(0, len(documents), batch_size), disable=not self.show_progress_bar, desc="Embedding documents"):
            inputs, offsets = self._offset_tokenize_documents(list(documents[start:start + batch_size]), self.document_prefix)
            inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
            self._active_chunk_offsets = offsets
            try:
                hidden = self.base_model(inputs)["token_embeddings"]
                pooled, padding = self._pool_documents(hidden, offsets)
            finally:
                self._active_chunk_offsets = None
            if self.normalize_embeddings:
                pooled = F.normalize(pooled, dim=-1)
            outputs.extend([[x.cpu() for x in row[~mask]] for row, mask in zip(pooled, padding)])
        return outputs

    def save(self, path, *args, **kwargs):
        """Save weights plus the exact Qwen3 train/inference configuration.

        The base SentenceTransformer is written using its normal save format
        (the tokenizer with the BOT tokens and the trained BOT embedding rows
        included). ``qwen3_late_chunking.json`` additionally records attention
        mode, query pooling, document pooling, query template and the number of
        BOT tokens so a later load cannot silently use different behavior.
        """
        self.base_model.save(path, *args, **kwargs)
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "qwen3_late_chunking.json").write_text(json.dumps({
            "attention_mode": self.attention_mode, "query_pooling": self.query_pooling,
            "document_pooling": self.document_pooling, "query_template": self.query_template,
            "num_bot_tokens": self.num_bot_tokens,
            # Provenance only: the trained rows are stored in the weights, so
            # loading never re-initialises them.
            "bot_init": self.bot_init, "bot_init_token": self.bot_init_token,
            "bot_init_noise": self.bot_init_noise, "bot_init_seed": self.bot_init_seed,
        }, indent=2), encoding="utf-8")

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        """Load a saved Qwen3 checkpoint and restore all persisted choices.

        Args:
            path: Directory produced by :meth:`save`.
            **kwargs: Runtime options such as normalization and batch size.
                Persisted mask/pooling/template/BOT values take precedence so
                the checkpoint cannot silently change behavior at load time.

        Returns:
            A ready-to-use ``Qwen3LateChunkingModel`` whose mask and pooling
            behavior matches the saved training configuration.
        """
        return cls(path, **kwargs)