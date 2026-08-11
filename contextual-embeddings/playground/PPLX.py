import inspect
from typing import Callable, Literal
import numpy as np
import torch
from transformers import Qwen3Model
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from .configuration import PPLXQwen3Config
from transformers import AutoTokenizer
from .st_quantize import FlexibleQuantizer

# The transformers `create_causal_mask` signature has shifted over releases
# (the embeds kwarg was renamed `input_embeds` -> `inputs_embeds`, and
# `cache_position` was eventually dropped). Probe the actual signature at import
# time so this works on any installed release, including dev/main builds.
_CCM_PARAMS = inspect.signature(create_causal_mask).parameters
_CCM_EMBEDS_KEY = "inputs_embeds" if "inputs_embeds" in _CCM_PARAMS else "input_embeds"
_CCM_ACCEPTS_CACHE_POSITION = "cache_position" in _CCM_PARAMS


# From modeling_t5gemma.py
def bidirectional_mask_function(attention_mask: torch.Tensor | None) -> Callable:
    """
    This creates bidirectional attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        if attention_mask is None:
            return torch.ones((), dtype=torch.bool)
        return attention_mask[batch_idx, kv_idx].to(torch.bool)

    return inner_mask


class PPLXQwen3Model(Qwen3Model):
    _supports_flash_attn = True
    _supports_sdpa = True

    config_class = PPLXQwen3Config

    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def post_init(self):
        super().post_init()
        # Override to set all layers to non-causal attention. This'll work with attn_implementation="flash_attention_2" or "sdpa"
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            input_ids = None

        mask_kwargs = {
            "config": self.config,
            _CCM_EMBEDS_KEY: inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
            "or_mask_function": bidirectional_mask_function(attention_mask),
        }
        if _CCM_ACCEPTS_CACHE_POSITION:
            mask_kwargs["cache_position"] = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long
            )
        attention_mask = {"full_attention": create_causal_mask(**mask_kwargs)}

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        return outputs


class PPLXQwen3ContextualModel(PPLXQwen3Model):
    """
    Qwen3 model with contextual encoding support for late chunking.

    This model extends PPLXQwen3Model with an encode() method that supports both
    standard encoding (list[str]) and contextual encoding (list[list[str]]) with late chunking.

    IMPORTANT: This model MUST be loaded with trust_remote_code=True:

        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "path/to/model",
            trust_remote_code=True  # REQUIRED!
        )

        embeddings = model.encode([["chunk1", "chunk2"]])

    Loading without trust_remote_code=True will fail to load this custom model class.
    """

    config_class = PPLXQwen3Config

    def __init__(self, config):
        super().__init__(config)

        if not isinstance(config, PPLXQwen3Config):
            raise TypeError(
                f"PPLXQwen3ContextualModel requires PPLXQwen3Config, got {type(config).__name__}. "
                f"Did you forget to load with trust_remote_code=True?"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self._flexible_quantizer = FlexibleQuantizer()

    @staticmethod
    def mean_pooling(
        token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling to token embeddings."""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.inference_mode()
    def encode(
        self,
        documents: list[list[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        device: str | torch.device | None = None,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        quantization: Literal["int8", "binary", "ubinary"] = "int8",
    ) -> list[np.ndarray] | list[torch.Tensor]:
        """
        Encode documents with late chunking (contextual embeddings).

        This model is designed specifically for contextual encoding and always expects
        documents as nested lists where each document is a list of text chunks.

        The encoding process:
        1. Concatenate chunks with separator tokens
        2. Run forward pass to get token embeddings
        3. Extract and pool individual chunk embeddings (late chunking)
        4. Apply quantization (Int8 or binary, always enabled)
        5. Normalize embeddings if requested (applied after quantization)
        6. Convert to numpy or return as tensors

        Args:
            documents: List of documents, where each document is a list of text chunks.
                Example: [["chunk1", "chunk2"], ["chunk1", "chunk2", "chunk3"]]
            batch_size: Batch size for encoding
            show_progress_bar: Show progress bar during encoding
            device: Device to use for computation (defaults to model's device)
            normalize_embeddings: Normalize embeddings to unit length (applied after quantization)
            convert_to_numpy: If True, returns list[np.ndarray], otherwise list[torch.Tensor]
            quantization: Quantization type to apply. Options:
                - "int8": Int8 tanh quantization (default)
                - "binary": Binary tanh quantization (-1.0 or 1.0)
                - "ubinary": Unsigned packed binary (uint8, 8x compression)

        Returns:
            List of numpy arrays or tensors (preserves document structure).
            Each element has shape (n_chunks, hidden_dim) or (n_chunks, hidden_dim // 8) for ubinary.
            Example: embeddings[0].shape = (2, 1024), embeddings[1].shape = (3, 1024)
            Output type depends on quantization method:
            - "int8": int8 dtype, values in range [-128, 127], shape (..., hidden_dim)
            - "binary": float32 dtype, values -1.0 or 1.0, shape (..., hidden_dim)
            - "ubinary": uint8 dtype, packed bits (8x smaller), shape (..., hidden_dim // 8)
        """

        if not isinstance(documents, list) or not all(
            isinstance(doc, list) for doc in documents
        ):
            raise TypeError(
                "Input 'documents' must be a list of lists of strings for contextual encoding."
            )

        if quantization not in ["int8", "binary", "ubinary"]:
            raise ValueError(
                f"Unsupported quantization type: '{quantization}'. "
                f"Supported types are: 'int8', 'binary', 'ubinary'. "
                f"Got: {type(quantization).__name__} = '{quantization}'"
            )

        if normalize_embeddings and quantization == "ubinary":
            raise ValueError(
                "normalize_embeddings=True is incompatible with quantization='ubinary'. "
                "Packed binary embeddings (uint8) cannot be normalized because each byte "
                "represents 8 packed bits, not a single dimension. "
                "Either set normalize_embeddings=False or use 'binary' quantization instead."
            )

        self.eval()

        if device is None:
            device = next(self.parameters()).device

        all_embeddings = []

        range_iter = range(0, len(documents), batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm

                range_iter = tqdm(range_iter, desc="Encoding documents")
            except ImportError:
                pass

        for i in range_iter:
            batch_docs = documents[i : i + batch_size]

            doc_strings = [
                self.tokenizer.sep_token.join(chunks) for chunks in batch_docs
            ]

            inputs = self.tokenizer(
                doc_strings,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.forward(**inputs)
            token_embeddings = outputs.last_hidden_state

            batch_chunk_embeddings = self._extract_chunks_from_concatenated(
                input_ids=inputs["input_ids"],
                token_embeddings=token_embeddings,
                attention_mask=inputs["attention_mask"],
            )

            batch_chunk_embeddings = [
                torch.stack([chunk for chunk in doc_chunks], dim=0)
                for doc_chunks in batch_chunk_embeddings
            ]

            batch_chunk_embeddings = [
                self._flexible_quantizer(
                    {"sentence_embedding": emb}, quantization=quantization
                )["sentence_embedding"]
                for emb in batch_chunk_embeddings
            ]

            if normalize_embeddings:
                batch_chunk_embeddings = [
                    torch.nn.functional.normalize(emb, p=2, dim=-1)
                    for emb in batch_chunk_embeddings
                ]

            batch_chunk_embeddings = [emb.cpu() for emb in batch_chunk_embeddings]

            all_embeddings.extend(batch_chunk_embeddings)

        if convert_to_numpy:
            all_embeddings = [emb.numpy() for emb in all_embeddings]

        return all_embeddings

    def _extract_chunks_from_concatenated(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        """
        Extract individual chunk embeddings from concatenated sequence using late chunking.

        This method splits concatenated sequences like "[chunk1][SEP][chunk2][SEP]..."
        back into individual chunk embeddings by finding SEP token positions.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            token_embeddings: Token embeddings (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            list[list[torch.Tensor]]: List of documents, each containing list of chunk embeddings

        Note:
            The sep_token_id is retrieved from self.tokenizer.sep_token_id.
            Common values: Qwen2=151643, BERT=102, varies by tokenizer.
        """
        sep_token_id = self.tokenizer.sep_token_id
        batch_size = input_ids.shape[0]

        all_doc_chunks = []

        for batch_idx in range(batch_size):
            # non-pad sep tokens
            valid_positions = attention_mask[batch_idx].bool()
            sep_positions = (
                (input_ids[batch_idx] == sep_token_id) & valid_positions
            ).nonzero(as_tuple=True)[0]

            chunk_embeddings = []
            start_pos = 0

            for sep_pos in sep_positions:
                chunk_tokens = token_embeddings[batch_idx, start_pos:sep_pos]
                chunk_mask = attention_mask[batch_idx, start_pos:sep_pos]

                chunk_emb = self.mean_pooling(
                    chunk_tokens.unsqueeze(0), chunk_mask.unsqueeze(0)
                ).squeeze(0)

                chunk_embeddings.append(chunk_emb)

                start_pos = sep_pos + 1

            # Handle the last chunk (after the last SEP token)
            last_valid_pos = attention_mask[batch_idx].sum().item()

            chunk_tokens = token_embeddings[batch_idx, start_pos:last_valid_pos]
            chunk_mask = attention_mask[batch_idx, start_pos:last_valid_pos]

            if chunk_mask.sum() > 0:
                chunk_emb = self.mean_pooling(
                    chunk_tokens.unsqueeze(0), chunk_mask.unsqueeze(0)
                ).squeeze(0)
            else:
                # Empty chunk - create zero embedding
                chunk_emb = torch.zeros(
                    token_embeddings.shape[-1],
                    device=token_embeddings.device,
                    dtype=token_embeddings.dtype,
                )

            chunk_embeddings.append(chunk_emb)

            all_doc_chunks.append(chunk_embeddings)

        return all_doc_chunks
