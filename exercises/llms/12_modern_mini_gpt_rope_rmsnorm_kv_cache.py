"""
============================================================
EXERCISE 12: MODERN MINI GPT ROPE RMSNORM KV CACHE
============================================================

WHAT YOU ARE BUILDING
You are building a tiny decoder-only language model that combines the ideas from the last three exercises.

This is still a small teaching model.
It is not a production LLM.

It combines:
- RoPE from the related exercise
- KV cache from the related exercise
- RMSNorm and SwiGLU from the related exercise
- the mini GPT training flow from earlier decoder exercises

WHY THIS EXERCISE EXISTS
The previous exercises taught important pieces one at a time.
This exercise puts those pieces into one small working model so you can see how they fit together.

This exercise teaches:
- how RoPE fits inside attention
- how RMSNorm fits before attention and feed-forward layers
- how SwiGLU fits inside the feed-forward part
- how a KV cache is used during generation, not normal training
- how to compare cached and non-cached generation

DATASET
Use Tiny Shakespeare.

Default local path:
- `data/input.txt`

Keep one clear split:
- first `60%` of the raw text for train
- next `20%` for validation
- final `20%` for test

INPUTS AND OUTPUTS
Input batch:
- `input_ids`: `[batch_size, context_length]`
- `target_ids`: `[batch_size, context_length]`

Model output during training:
- logits: `[batch_size, context_length, vocab_size]`

KV cache during generation:
- key cache: `[batch_size, num_heads, cached_length, head_dim]`
- value cache: `[batch_size, num_heads, cached_length, head_dim]`

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

Return raw logits.
Do not apply softmax before the loss.

METRICS TO TRACK
Track at least:
- train loss
- validation loss
- validation perplexity
- generated text without KV cache
- generated text with KV cache
- cache length after generation
- simple count of repeated key/value work avoided

STEP-BY-STEP TODO
1. Load Tiny Shakespeare.
2. Split the raw text into train, validation, and test text.
3. Tokenize text into simple word tokens or character tokens.
4. Build the vocabulary from training tokens only.
5. Build next-token datasets and dataloaders.
6. Build RoPE sine and cosine values.
7. Apply RoPE to query and key tensors.
8. Build masked self-attention with optional KV cache.
9. Build RMSNorm.
10. Build SwiGLU.
11. Build one modern decoder block.
12. Stack decoder blocks into a tiny GPT model.
13. Train for a few epochs.
14. Evaluate validation loss and perplexity.
15. Generate text without KV cache.
16. Generate text with KV cache.
17. Compare the two generation paths.

DEFINITION OF DONE
You are done when:
- the model trains without shape errors
- validation loss and perplexity are returned
- generation works without KV cache
- generation works with KV cache
- you can explain which parts are for training and which parts are for inference
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
import sys

from math import sqrt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again

device = "cuda" if torch.cuda.is_available() else "cpu"


# Store the main settings for this combined exercise.
# This config controls:
# - data path
# - sequence length
# - model size
# - training settings
# - vocabulary mappings after the data is loaded
@dataclass
class ModernMiniGPTConfig:
    """Settings for the modern mini GPT exercise."""

    data_path: str = str(BOOTCAMP_ROOT / "data" / "input.txt")
    context_length: int = 16
    embed_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256
    num_layers: int = 4
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 1
    sample_length: int = 200
    rope_base: float = 10000.0
    eps: float = 1e-6
    random_state: int = 42
    vocab_size: int | None = None
    token_to_int: dict[str, int] | None = None
    int_to_token: dict[int, str] | None = None


# Describe the main tensor shapes before writing the model.
# This is a quick map for checking your work later.
# Return:
# - input shape
# - hidden-state shape
# - query/key/value shape per head
# - mask shape
# - logits shape
# - KV cache shape
def describe_model_shapes(config: ModernMiniGPTConfig) -> dict[str, object]:
    """Return the main expected shapes."""
    head_dim = config.embed_dim // config.num_heads
    return {
        "input_ids": (config.batch_size, config.context_length),
        "hidden_states": (config.batch_size, config.context_length, config.embed_dim),
        "qkv_per_head": (config.batch_size, config.num_heads, config.context_length, head_dim),
        "causal_mask": (1, 1, config.context_length, config.context_length),
        "logits": (config.batch_size, config.context_length, "vocab_size"),
        "kv_cache": (config.batch_size, config.num_heads, "cached_length", head_dim),
    }


# Load the Tiny Shakespeare file as one raw string.
# This is the same first data step as the related exercise.
# Return:
# - the full text from `config.data_path`
def load_raw_text(config: ModernMiniGPTConfig) -> str:
    """Return the full Tiny Shakespeare text."""

    with open(file=config.data_path, mode="r", encoding="utf-8") as f:
        raw_text = f.read()
        
    return raw_text
    #raise NotImplementedError("Load config.data_path and return one string.")


# Split the raw text before tokenizing.
# Use one contiguous split so validation and test text stay separate from training text.
# Return:
# - `train_text`: first 60%
# - `val_text`: next 20%
# - `test_text`: final 20%
def split_text_corpus(raw_text: str) -> dict[str, str]:
    """Return a contiguous 60 / 20 / 20 split."""

    num_train, num_val = int(len(raw_text) * 0.6), int(len(raw_text) * 0.2)

    train_text = raw_text[: num_train]
    val_text = raw_text[num_train : num_train + num_val]
    test_text = raw_text[num_train + num_val :]

    return {"train_text": train_text, "val_text": val_text, "test_text": test_text}
    #raise NotImplementedError("Return train_text, val_text, and test_text.")


# Turn text into simple word and punctuation tokens.
# Example:
# - "To be, or not" becomes ["To", "be", ",", "or", "not"]
def basic_word_tokenize(text: str) -> list[str]:
    """Return simple word and punctuation tokens."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


# Build token id mappings from training tokens only.
# Do not use validation or test tokens here.
# Include:
# - `[PAD]` at id 0
# - `[UNK]` at id 1
# Return:
# - `token_to_int`
# - `int_to_token`
# - `vocab_size`
#
# Basic steps:
# 1. Find the unique tokens.
# 2. Sort them so the ids are stable.
# 3. Put `[PAD]` at id 0.
# 4. Put `[UNK]` at id 1.
# 5. Build `token_to_int`.
# 6. Build `int_to_token`.
# 7. Return both mappings and the vocabulary size.
# def build_vocab(train_tokens: list[str]) -> dict[str, object]:
#     """Build token mappings from training tokens only."""
#     raise NotImplementedError("Return token_to_int, int_to_token, and vocab_size.")


class NextTokenDataset(torch.utils.data.Dataset):
    """Dataset for next-token language modeling windows."""

    # Store the tokens and settings needed to make training windows.
    # `token_to_int` turns token strings into integer ids.
    # Keep an unknown-token id so validation/test words missing from training can still be encoded.
    def __init__(self, tokens: list[str], context_length: int, token_to_int: dict[str, int]) -> None:

        self.tokens = tokens
        self.num_tokens = len(tokens)
        self.context_length = context_length
        self.token_to_int = token_to_int

        self.unk_id = 1
        #raise NotImplementedError("Store tokens, context length, vocabulary mapping, and unknown-token id.")

    # The last usable input window starts `context_length` tokens before the end.
    # Example:
    # - 10 tokens
    # - context_length = 4
    # - number of windows = 10 - 4 = 6
    def __len__(self) -> int:
        """Return the number of windows."""
        return self.num_tokens - self.context_length
        #raise NotImplementedError("Return len(tokens) - context_length.")

    # Build one next-token example.
    # If the input tokens are positions i to i + context_length,
    # the target tokens are shifted one step to the right.
    # Return tensors:
    # - `input_ids`: shape `[context_length]`
    # - `target_ids`: shape `[context_length]`
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return input_ids and target_ids for one window."""

        input_tokens = self.tokens[index: index + self.context_length]
        target_tokens = self.tokens[index + 1: index + 1 + self.context_length]

        input_ids, target_ids = [], []
        for token in input_tokens:
            input_ids.append(self.token_to_int.get(token, self.unk_id))
        for token in target_tokens:
            target_ids.append(self.token_to_int.get(token, self.unk_id))

        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), 
                "target_ids": torch.tensor(target_ids, dtype=torch.long)}
        #raise NotImplementedError("Return shifted next-token tensors.")


# Build all dataloaders in one place.
# Steps:
# 1. load raw text
# 2. split text
# 3. tokenize each split
# 4. build the vocab from train tokens only
# 5. save vocab info on `config`
# 6. create train, validation, and test datasets
# 7. wrap them in DataLoader objects
def load_dataloaders(config: ModernMiniGPTConfig) -> dict[str, DataLoader]:
    """Return train, validation, and test dataloaders."""

    raw_text = load_raw_text(config)
    text_dict = split_text_corpus(raw_text)

    all_words = basic_word_tokenize(raw_text)
    vocabulary = sorted(set(all_words))
    vocabulary.insert(0, "[PAD]")
    vocabulary.insert(1, "[UNK]")
    vocab_size = len(vocabulary)

    config.vocab_size = vocab_size
    config.token_to_int = {t:i for i,t in enumerate(vocabulary)}
    config.int_to_token = {i:t for t,i in config.token_to_int.items()}

    train_ds = NextTokenDataset(tokens=basic_word_tokenize(text_dict["train_text"]), context_length=config.context_length, token_to_int=config.token_to_int)
    val_ds = NextTokenDataset(tokens=basic_word_tokenize(text_dict["val_text"]), context_length=config.context_length, token_to_int=config.token_to_int)
    test_ds = NextTokenDataset(tokens=basic_word_tokenize(text_dict["test_text"]), context_length=config.context_length, token_to_int=config.token_to_int)

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}
    #raise NotImplementedError("Load text, tokenize, build vocab, and return dataloaders.")


# Build the left-to-right attention mask.
# Use True for allowed positions and False for future positions.
# Example for length 4:
# [
#   [ True, False, False, False],
#   [ True,  True, False, False],
#   [ True,  True,  True, False],
#   [ True,  True,  True,  True],
# ]
def build_causal_mask(sequence_length: int) -> torch.Tensor:
    """Return a boolean causal mask with shape `[1, 1, seq, seq]`."""

    ones = torch.ones(size=(sequence_length, sequence_length), dtype=torch.bool)
    causal_mask = torch.tril(ones)
    causal_mask = causal_mask.unsqueeze(dim=0).unsqueeze(dim=0)

    return causal_mask
    #raise NotImplementedError("Build a lower-triangular boolean mask.")


# Build the RoPE rotation instructions.
# This comes from the related exercise.
# Return `cos` and `sin` with shape:
# - `[1, 1, sequence_length, head_dim]`
# The first two dimensions are 1 so they broadcast across batch and heads.
#
# Basic steps:
# 1. Compute `head_dim`.
#    Formula:
#    head_dim = config.embed_dim // config.num_heads
#
# 2. Check that `head_dim` is even.
#    RoPE rotates pairs of numbers, so this simple version needs pairs.
#
# 3. Create token positions.
#    Example when sequence_length is 4:
#    positions = [0, 1, 2, 3]
#
# 4. Create one frequency for each pair inside `head_dim`.
#    If head_dim is 8, there are 4 pairs.
#    Formula:
#    frequencies = 1.0 / (config.rope_base ** (pair_index / (head_dim // 2)))
#
#    Meaning:
#    - `pair_index` is [0, 1, 2, ...]
#    - `config.rope_base` is usually 10000
#    - early pairs rotate faster
#    - later pairs rotate slower
#
# 5. Combine positions and frequencies to make angles.
#    Formula:
#    angles = positions[:, None] * frequencies[None, :]
#
#    Shape:
#    [sequence_length, head_dim // 2]
#
# 6. Repeat the angle values across the last dimension.
#    This changes:
#    [sequence_length, head_dim // 2]
#    into:
#    [sequence_length, head_dim]
#
# 7. Take cosine and sine.
#    Then reshape each one to:
#    [1, 1, sequence_length, head_dim]
def build_rope_cache(config: ModernMiniGPTConfig, sequence_length: int) -> dict[str, torch.Tensor]:
    """Return RoPE cosine and sine tensors."""

    head_dim = config.embed_dim // config.num_heads     # 128 // 4 = 32

    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even.")

    token_positions = torch.arange(start=0, end=sequence_length, step=1, dtype=torch.float)        # shape = [16] one for each token

    pair_indexes = torch.arange(start=0, end=head_dim//2, step=1)               # shape = [16]
    frequencies = 1.0 / (config.rope_base ** (pair_indexes / (head_dim // 2)))

    angles = token_positions.unsqueeze(dim=-1) @ frequencies.unsqueeze(dim=0)   # [16, 1] @ [1, 16] = [16, 16]

    angles = torch.concat([angles, angles], dim=-1)                             # shape = [16, 32]  = [seq_length, head_dim]

    cos_table = torch.cos(angles).unsqueeze(dim=0).unsqueeze(dim=0)             # shape = [1, 1, 16, 32] 
    sin_table = torch.sin(angles).unsqueeze(dim=0).unsqueeze(dim=0)             # shape = [1, 1, 16, 32]

    return {"cos": cos_table, "sin": sin_table}
    #raise NotImplementedError("Build RoPE cos and sin values for the given sequence length.")


# Rotate the last dimension in pairs.
# This helper is used by RoPE.
#
# Small example:
# - input pair: [a, b]
# - rotated pair: [-b, a]
#
# For the full tensor:
# 1. split the last dimension into two equal halves
# 2. negate the second half
# 3. concatenate `[-second_half, first_half]`
#
# Input shape:
# - `[batch_size, num_heads, sequence_length, head_dim]`
#
# Output shape:
# - same as input
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Return the rotate-half version of x."""

    batch_size, num_heads, sequence_length, head_dim = x.shape                      # shape = [16, 4, 16, 32]

    split_x = x.reshape(batch_size, num_heads, sequence_length, 2, head_dim//2)     # shape = [16, 4, 16, 2, 16]
    first_half = split_x[:, :, :, 0:1, :]                                           # shape = [16, 4, 16, 1, 16]
    second_half = split_x[:, :, :, 1:, :]                                           # shape = [16, 4, 16, 1, 16]

    second_half = second_half * -1

    merged = torch.concat([second_half, first_half], dim=3)                         # shape = [16, 4, 16, 2, 16]
    merged = merged.reshape(batch_size, num_heads, sequence_length, head_dim)       # shape = [16, 4, 16, 32]

    return merged
    #raise NotImplementedError("Split the last dimension and return [-second_half, first_half].")


# Apply RoPE to a query or key tensor.
# Input:
# - `x`: `[batch_size, num_heads, sequence_length, head_dim]`
# - `rope_cache`: dict with `cos` and `sin`
# - `start_pos`: where this chunk starts in the full generated sequence
#
# Use `start_pos` during cached generation.
# Example:
# - if 5 tokens are already cached, the next token starts at position 5
# - use RoPE values for position 5, not position 0
#
# Formula:
# rotated_x = (x * cos) + (rotate_half(x) * sin)
#
# Basic steps:
# 1. Read `sequence_length` from `x.shape`.
# 2. Take the matching slice from `rope_cache["cos"]`.
#    Use positions from `start_pos` to `start_pos + sequence_length`.
# 3. Take the matching slice from `rope_cache["sin"]`.
# 4. Apply the formula above.
# 5. Return a tensor with the same shape as `x`.
def apply_rope(x: torch.Tensor, rope_cache: dict[str, torch.Tensor], start_pos: int = 0) -> torch.Tensor:
    """Apply RoPE to query or key tensors."""

    batch_size, num_heads, sequence_length, head_dim = x.shape          # shape = [16, 4, 16, 32]  or could also just be 1 token
    # sequence_length is not always the full context_length.
    # During normal training, it usually is the full context length
    # During cached generation, it may be just one new token
    # Example: 
    # prompt length = 5
    # now generating token at position 5
    # then:
    # sequence_length = 1
    # start_pos = 5
    # i.e., use only the RoPE row for the token position you are currently processing

    cos_table = rope_cache["cos"]                                       # shape = [1, 1, 16, 32]
    sin_table = rope_cache["sin"]                                       # shape = [1, 1, 16, 32]

    cos_table = cos_table[: , :, start_pos: start_pos + sequence_length, :]
    sin_table = sin_table[: , :, start_pos: start_pos + sequence_length, :]

    rotated_x = (x * cos_table) + (rotate_half(x) * sin_table)

    return rotated_x
    # raise NotImplementedError("Rotate pairs of query/key features using cos and sin values.")


class RMSNorm(nn.Module):
    """Root mean square normalization."""

    # Build RMSNorm.
    # Store:
    # - a learned scale parameter with shape `[embed_dim]`
    # - the small `eps` value used to avoid divide-by-zero
    #
    # A good starting scale is all ones:
    # `nn.Parameter(torch.ones(embed_dim))`
    def __init__(self, embed_dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

        #raise NotImplementedError("Create the learned RMSNorm scale.")

    # Apply RMSNorm over the last dimension only.
    # Formula:
    # - rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
    # - output = (x / rms) * scale
    #
    # Keep the same shape as the input.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return RMS-normalized hidden states."""

        # x is of shape [batch_size, context_length, embed_dim]

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        norm = (x / rms) * self.scale

        return norm
        #raise NotImplementedError("Normalize over the last dimension and keep the same shape.")


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward block."""

    # Build the three feed-forward projections.
    # Expected layers:
    # - gate projection: `embed_dim -> mlp_dim`
    # - value projection: `embed_dim -> mlp_dim`
    # - output projection: `mlp_dim -> embed_dim`
    #
    # The output must return to `embed_dim` so the residual add works.
    def __init__(self, embed_dim: int, mlp_dim: int) -> None:
        super().__init__()

        self.gate = nn.Linear(in_features=embed_dim, out_features=mlp_dim)
        self.value = nn.Linear(in_features=embed_dim, out_features=mlp_dim)
        self.proj = nn.Linear(in_features=mlp_dim, out_features=embed_dim)

        #raise NotImplementedError("Build gate, value, and output projections.")

    # Run the SwiGLU formula.
    # Formula:
    # - gate = gate_proj(x)
    # - value = value_proj(x)
    # - hidden = SiLU(gate) * value
    # - output = out_proj(hidden)
    #
    # Input and output shape:
    # - `[batch_size, sequence_length, embed_dim]`
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return feed-forward output with shape `[batch, seq, embed_dim]`."""

        # x is shape [batch_size, sequence_length, embed_dim]

        gate = self.gate(x)                             # shape = [batch, seq, mlp_dim]
        value = self.value(x)                           # shape = [batch, seq, mlp_dim]
        hidden = nn.functional.silu(gate) * value       # shape = [batch, seq, mlp_dim]

        output = self.proj(hidden)                      # shape = [batch, seq, embed_dim]

        return output        
        #raise NotImplementedError("Apply SiLU gate times value, then project back to embed_dim.")


# Compute scaled dot-product attention.
# This helper comes from the related exercise, with the RoPE pieces handled before it is called.
#
# Input shapes:
# - `query`: `[batch_size, num_heads, query_length, head_dim]`
# - `key`: `[batch_size, num_heads, key_length, head_dim]`
# - `value`: `[batch_size, num_heads, key_length, head_dim]`
# - `mask`: None or a boolean mask that broadcasts to `[batch_size, num_heads, query_length, key_length]`
#
# Formula:
# scores = (query @ key.transpose(-2, -1)) / sqrt(head_dim)
# weights = softmax(scores, dim=-1)
# output = weights @ value
#
# Basic steps:
# 1. Compute raw attention scores with query and transposed key.
# 2. Divide by `sqrt(head_dim)`.
# 3. If a mask is given, fill blocked positions with `-inf`.
# 4. Apply softmax over the last dimension.
# 5. Multiply the weights by value.
# 6. Return the attention output with shape `[batch_size, num_heads, query_length, head_dim]`.
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return scaled dot-product attention output."""

    # query is shape [batch_size, num_heads, query_length, head_dim]
    # key and value are shape [batch_size, num_heads, key_length, head_dim]
    # During normal training, query_length and key_length are usually the same.
    # During cached generation, query_length may be 1 while key_length includes old cached tokens plus the new token.
    # mask is None or shape [1, 1, query_length, key_length]

    batch_size, num_heads, seq_len, head_dim = query.shape

    scores = query @ key.transpose(dim0=-2, dim1=-1)   # [16, 4, 16, 32] @ [16, 4, 32, 16] = [16, 4, 16, 16] # or could be 1 instead of 16
    scores /= sqrt(head_dim)

    if mask is not None:
        scores = torch.masked_fill(scores, ~mask, value=float("-inf"))

    weights = scores.softmax(dim=-1)

    output = weights @ value                         # [16, 4, 16, 16] @ [16, 4, 16, 32] = [16, 4, 16, 32]

    return output
    #raise NotImplementedError("Compute masked scaled dot-product attention.")


# Create the starting KV cache.
# This comes from the related exercise.
#
# Simple beginner choice:
# - return `None`
#
# Why:
# - before the first generated token, there are no old keys or values
# - the first attention call can create the first real cache
# - later calls append to that cache
def init_empty_cache() -> dict[str, torch.Tensor] | None:
    """Return the starting KV cache value."""

    return None
    #raise NotImplementedError("Return None for the starting cache.")


class ModernSelfAttention(nn.Module):
    """Masked self-attention with RoPE and optional KV cache."""

    # Build attention layers.
    # Expected layers:
    # - query projection
    # - key projection
    # - value projection
    # - output projection
    #
    # Each projection maps `embed_dim -> embed_dim`.
    # Also compute `head_dim = embed_dim // num_heads`.
    def __init__(self, config: ModernMiniGPTConfig) -> None:
        super().__init__()

        self.query = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)
        self.key = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)
        self.value = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)
        self.proj = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)

        self.head_dim = config.embed_dim // config.num_heads
        self.num_heads = config.num_heads

        #raise NotImplementedError("Build Q, K, V projections and output projection.")

    # Run attention in two modes.
    #
    # Training mode:
    # - `x` is a full sequence
    # - `kv_cache` is None
    # - use the causal mask
    #
    # Cached generation mode:
    # - `x` is usually one new token
    # - `kv_cache` stores old keys and values
    # - append the new key and value on dimension 2
    # - usually no mask is needed because the cache only contains past and current tokens
    #
    # Main steps:
    # 1. project x into query, key, value
    # 2. split heads to `[batch, heads, seq, head_dim]`
    # 3. apply RoPE to query and key
    # 4. append key/value to the cache if a cache exists
    # 5. compute scaled dot-product attention
    # 6. merge heads back to `[batch, seq, embed_dim]`
    # 7. apply output projection
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope_cache: dict[str, torch.Tensor],
        kv_cache: dict[str, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return attention output and updated KV cache."""

        # x is shape [batch_size, seq, embed_dim]   seq may be 1 during inference, it is full context length during training

        batch_size, seq, embed_dim = x.shape

        query = self.query(x)                                                               # [batch, seq, embed_dim]
        key = self.key(x)                                                                   # [batch, seq, embed_dim]
        value = self.value(x)                                                               # [batch, seq, embed_dim]

        query = query.reshape(batch_size, seq, self.num_heads, self.head_dim)               # [batch, seq, heads, head_dim]
        key = key.reshape(batch_size, seq, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq, self.num_heads, self.head_dim)

        query = query.permute(dims=(0, 2, 1, 3))                                            # [batch, heads, seq, head_dim]
        key = key.permute(dims=(0, 2, 1, 3))
        value = value.permute(dims=(0, 2, 1, 3))

        rope_query = apply_rope(query, rope_cache, start_pos)
        rope_key = apply_rope(key, rope_cache, start_pos)

        if kv_cache is None:
            key_all = rope_key
            value_all = value
        else:
            # dim=2 is correct because the cache length lives in the sequence dimension:
            key_all = torch.concat([kv_cache["key"], rope_key], dim=2)
            value_all = torch.concat([kv_cache["value"], value], dim=2)

        updated_cache = {"key": key_all, "value": value_all}

        attention_output = scaled_dot_product_attention(rope_query, key_all, value_all, mask)   # [batch, heads, seq, head_dim]

        attention_output = attention_output.permute(dims=(0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq, embed_dim)

        proj = self.proj(attention_output)                                                  # [batch, seq, embed_dim]
        
        return proj, updated_cache
        # raise NotImplementedError("Run attention, append cache during generation, and return updated cache.")


class ModernDecoderBlock(nn.Module):
    """Modern decoder block with RMSNorm, RoPE attention, and SwiGLU."""

    # Build one modern decoder block.
    # The block should contain:
    # - RMSNorm before attention
    # - ModernSelfAttention
    # - RMSNorm before feed-forward
    # - SwiGLUFeedForward
    #
    # This is a pre-norm block because normalization happens before each main layer.
    def __init__(self, config: ModernMiniGPTConfig) -> None:
        super().__init__()

        self.norm1 = RMSNorm(embed_dim=config.embed_dim, eps=config.eps)
        self.self_attention = ModernSelfAttention(config)

        self.norm2 = RMSNorm(embed_dim=config.embed_dim, eps=config.eps)
        self.feed_forward = SwiGLUFeedForward(embed_dim=config.embed_dim, mlp_dim=config.mlp_dim)

        #raise NotImplementedError("Build RMSNorm, attention, RMSNorm, and SwiGLU.")

    # Run the block.
    # Use this order:
    # 1. normalize x
    # 2. run attention
    # 3. add the attention output back to x
    # 4. normalize again
    # 5. run the feed-forward block
    # 6. add the feed-forward output back to x
    #
    # Return:
    # - updated hidden states
    # - updated cache for this block
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope_cache: dict[str, torch.Tensor],
        kv_cache: dict[str, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return updated hidden states and updated block cache."""

        # x is of shape [batch_size, seq, embed_dim]

        residual = x
        x = self.norm1(x)
        x, updated_cache = self.self_attention(x, mask, rope_cache, kv_cache, start_pos)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual
        
        return x, updated_cache
        #raise NotImplementedError("Run pre-norm attention, residual, pre-norm feed-forward, and residual.")


class ModernMiniGPT(nn.Module):
    """Tiny decoder-only GPT using modern block pieces."""

    # Build the full tiny GPT model.
    # Expected parts:
    # - token embedding
    # - a stack of ModernDecoderBlock layers
    # - final RMSNorm
    # - language-model head that maps `embed_dim -> vocab_size`
    #
    # This model uses RoPE, so it does not need a learned positional embedding.
    def __init__(self, vocab_size: int, config: ModernMiniGPTConfig) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.embed_dim, padding_idx=0)
        #self.pos_embedding = nn.Embedding(num_embeddings=config.context_length, embedding_dim=config.embed_dim)

        self.decoder_blocks = nn.ModuleList([ModernDecoderBlock(config) for _ in range(config.num_layers)])

        self.norm = RMSNorm(embed_dim=config.embed_dim, eps=config.eps)

        self.head = nn.Linear(in_features=config.embed_dim, out_features=vocab_size)

        self.config = config

        #raise NotImplementedError("Build embeddings, decoder blocks, final RMSNorm, and LM head.")

    # Run the model for training or generation.
    #
    # Training:
    # - `input_ids`: `[batch_size, context_length]`
    # - `kv_caches`: None
    # - build a causal mask
    # - return logits for every position
    #
    # Cached generation:
    # - `input_ids`: usually `[batch_size, 1]`
    # - `kv_caches`: one cache per decoder block
    # - `start_pos`: position of the new token in the full sequence
    # - return logits for the new token and updated caches
    #
    # Build one RoPE cache for the needed sequence length and pass it into every block.
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[dict[str, torch.Tensor] | None] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Return logits and updated per-layer KV caches."""

        batch_size, seq_length = input_ids.shape

        using_cache = kv_caches is not None

        if using_cache:
            mask = None
        else:
            # build mask
            mask = build_causal_mask(seq_length)
            mask = mask.to(input_ids.device)

        # build rope cache
        rope_cache = build_rope_cache(self.config, start_pos + seq_length)
        rope_cache["cos"] = rope_cache["cos"].to(input_ids.device)
        rope_cache["sin"] = rope_cache["sin"].to(input_ids.device)

        # kv cache
        if kv_caches is None:
            kv_caches = [None] * len(self.decoder_blocks)

        # begin

        token_embeddings = self.token_embedding(input_ids)                  # shape = [batch_size, seq, embed_dim]

        # positions = torch.arange(start=0, end=seq_length, step=1)           # shape = [seq_length]
        # positions = positions.unsqueeze(dim=0)                              # shape = [1, seq_length]
        # positions = positions.expand(size = (batch_size, seq_length))       # shape = [batch_size, sequence_length]
        # pos_embedings = self.pos_embedding(positions)                       # shape = [batch_size, seq, embed_dim]

        embeddings = token_embeddings #+ pos_embedings                       # shape = [batch_size, seq, embed_dim]

        features = embeddings
        new_caches = []        
        for block, layer_cache in zip(self.decoder_blocks, kv_caches):
            features, new_cache = block(features, mask, rope_cache, layer_cache, start_pos)
            new_caches.append(new_cache)

        features = self.norm(features)

        logits = self.head(features)

        return logits, new_caches
        #raise NotImplementedError("Run the model for training or cached generation.")


# Build the model, loss function, and optimizer.
# Use:
# - `ModernMiniGPT` for the model
# - `nn.CrossEntropyLoss()` for next-token prediction
# - Adam or AdamW for the optimizer
#
# Make sure `config.vocab_size` has been set by `load_dataloaders` first.
def build_training_objects(config: ModernMiniGPTConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, loss function, and optimizer."""

    model = ModernMiniGPT(config.vocab_size, config)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(lr=config.learning_rate, params=model.parameters())
    
    return model, loss_fn, optimizer
    #raise NotImplementedError("Build ModernMiniGPT, CrossEntropyLoss, and Adam optimizer.")


# Train with full context windows.
# KV cache is not used during normal training here.
# Steps:
# 1. set model to train mode
# 2. loop over batches
# 3. move input and target tensors to the model device
# 4. call the model with full `input_ids`
# 5. reshape logits and targets for CrossEntropyLoss
# 6. run backward and optimizer step
# 7. return mean loss and perplexity
def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Return mean train loss and perplexity."""

    model.train()

    losses = []
    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        logits, new_caches = model(input_ids=input_ids, kv_caches=None, start_pos=0)
        batch_size, seq, vocab_size = logits.shape
        loss = loss_fn(logits.reshape(batch_size*seq, vocab_size), target_ids.reshape(batch_size*seq))

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean()
    perp = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": perp.item()}
    #raise NotImplementedError("Train one epoch with full sequence windows.")


# Evaluate without changing model weights.
# Use `model.eval()` and do not call backward or optimizer step.
# Return:
# - mean loss
# - perplexity, which is `exp(mean loss)`
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:
    """Return mean evaluation loss and perplexity."""

    model.eval()

    losses = []
    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)       

        logits, new_caches = model(input_ids=input_ids, kv_caches=None, start_pos=0)
        batch_size, seq, vocab_size = logits.shape
        loss = loss_fn(logits.reshape(batch_size*seq, vocab_size), target_ids.reshape(batch_size*seq))

        losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean()
    perp = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": perp.item()}   
    #raise NotImplementedError("Evaluate full sequence windows.")


# Generate text without using a KV cache.
# This is the slower path.
# Each step:
# - pass the current full context into the model
# - take logits from the last position
# - choose the next token
# - append it to the context
# - keep only the last `context_length` tokens
@torch.no_grad()
def generate_without_cache(model: ModernMiniGPT, start_text: str, config: ModernMiniGPTConfig) -> str:
    """Generate text by rerunning the full context each step."""

    temperature = 0.2
    use_temperature=True

    if not start_text:
        start_text = "hello world."

    start_tokens = basic_word_tokenize(start_text)
    start_tokens = start_tokens[-config.context_length:]

    model_device = next(model.parameters()).device

    encode = lambda st: [config.token_to_int.get(t, 1) for t in st]
    decode = lambda l: " ".join([config.int_to_token[i] for i in l])

    token_ids = torch.tensor(encode(start_tokens), dtype=torch.long, device=model_device).unsqueeze(dim=0)     # shape = [1, seq_len]
    generated = []

    for i in range(config.sample_length):
        logits, new_caches = model(input_ids=token_ids, kv_caches=None, start_pos=0)    # shape is [batch_size, seq, vocab_size]
        new_token_vec = logits[:, -1, :]     #shape is [batch_size, vocab_size]

        if use_temperature:
            prob = torch.softmax(new_token_vec/temperature, dim=-1)
            new_token_id = torch.multinomial(prob, num_samples=1)
        else:
            new_token_id = new_token_vec.argmax(dim=-1, keepdim=True)

        token_ids = torch.concat([token_ids, new_token_id], dim=-1)
        token_ids = token_ids[:, -config.context_length:]
        generated.append(new_token_id.squeeze().item())

    return start_text + " " + decode(generated)
    #raise NotImplementedError("Generate one token at a time without passing a KV cache.")


# Generate text with a KV cache.
# This is the faster inference path.
# Main idea:
# - run the prompt once to fill the cache
# - after that, pass only the newest token
# - reuse old keys and values from each decoder block
#
# Track:
# - generated text
# - final cache length
# - a simple estimate of repeated key/value work avoided
@torch.no_grad()
def generate_with_cache(model: ModernMiniGPT, start_text: str, config: ModernMiniGPTConfig) -> dict[str, object]:
    """Generate text while reusing a KV cache."""

    temperature = 0.2
    use_temperature=True

    if not start_text:
        start_text = "hello world."

    start_tokens = basic_word_tokenize(start_text)
    start_tokens = start_tokens[-config.context_length:]

    model_device = next(model.parameters()).device

    encode = lambda st: [config.token_to_int.get(t, 1) for t in st]
    decode = lambda l: " ".join([config.int_to_token[i] for i in l])

    token_ids = torch.tensor(encode(start_tokens), dtype=torch.long, device=model_device).unsqueeze(dim=0)     # shape = [1, seq_len]
    generated = []

    prompt_length = token_ids.shape[1]

    # run the prompt once to fill the cache
    logits, kv_caches  = model(input_ids=token_ids, kv_caches=None, start_pos=0)    # shape is [batch_size, seq, vocab_size]
    new_token_vec = logits[:, -1, :]     #shape is [batch_size, vocab_size]

    if use_temperature:
        prob = torch.softmax(new_token_vec/temperature, dim=-1)
        new_token_id = torch.multinomial(prob, num_samples=1)
    else:
        new_token_id = new_token_vec.argmax(dim=-1, keepdim=True)    
    
    generated.append(new_token_id.squeeze().item())

    current_token = new_token_id
    start_pos = prompt_length
    # now run it in a loop 1 token at a time
    for i in range(1, config.sample_length):

        logits, kv_caches = model(input_ids=current_token, kv_caches=kv_caches , start_pos=start_pos)    # shape is [batch_size, seq, vocab_size]
        new_token_vec = logits[:, -1, :]     #shape is [batch_size, vocab_size]

        if use_temperature:
            prob = torch.softmax(new_token_vec/temperature, dim=-1)
            new_token_id = torch.multinomial(prob, num_samples=1)
        else:
            new_token_id = new_token_vec.argmax(dim=-1, keepdim=True)
        
        current_token = new_token_id
        generated.append(new_token_id.squeeze().item())
        start_pos+=1

    final_cache_length = kv_caches[0]["key"].shape[2]

    return {
                "text": start_text + " " + decode(generated),
                "final_cache_length": final_cache_length,
                "cached_work_estimate": config.sample_length,
                "num_generated_tokens": len(generated),
            }
    #raise NotImplementedError("Generate one token at a time and reuse cached keys and values.")


# Compare the rough work done by cached and non-cached generation.
# This comes from the idea in the related exercise.
#
# Simple estimate:
# - without cache, step 1 processes 1 token, step 2 processes 2 tokens, and so on
# - with cache, each new step only builds key/value work for 1 new token
#
# Example:
# sample_length = 4
# no_cache_work = 1 + 2 + 3 + 4 = 10
# cached_work = 1 + 1 + 1 + 1 = 4
#
# Basic steps:
# 1. Compute the no-cache work estimate.
# 2. Compute the cached work estimate.
# 3. Compute saved work.
# 4. Compute speedup ratio.
# 5. Return those numbers and one plain-language note.
def compare_generation_work(sample_length: int) -> dict[str, float | str]:
    """Compare cached and non-cached generation work."""

    no_cache_work_estimate = sum(range(1, sample_length + 1))
    cached_work_estimate = sample_length

    saved_work = no_cache_work_estimate - cached_work_estimate
    speedup_ratio = no_cache_work_estimate / cached_work_estimate if cached_work_estimate > 0 else 0.0

    return {
        "no_cache_work_estimate": float(no_cache_work_estimate),
        "cached_work_estimate": float(cached_work_estimate),
        "saved_work": float(saved_work),
        "speedup_ratio": float(speedup_ratio),
        "note": "KV cache avoids rebuilding old keys and values for every new token.",
    }
    #raise NotImplementedError("Compare simple cached and non-cached generation work estimates.")


# Run the whole exercise.
# Suggested flow:
# 1. set the seed
# 2. load dataloaders and set vocab info on config
# 3. build model, loss, and optimizer
# 4. train for `config.epochs`
# 5. evaluate on validation data
# 6. generate without cache
# 7. generate with cache
# 8. compare generation work
# 9. return the required summary fields
def run_training(config: ModernMiniGPTConfig) -> dict[str, float | str]:
    """Train, evaluate, and compare cached and non-cached generation."""

    set_seed(config.random_state)

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    for i in range(config.epochs):
        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i+1} -> loss: {train_dict['loss']}, perp: {train_dict['perplexity']}, val_loss: {val_dict['loss']}, val_perp: {val_dict['perplexity']}")

    sample_without_cache = generate_without_cache(model=model, start_text="To be or not", config=config)
    cache_sample = generate_with_cache(model=model, start_text="To be or not", config=config)
    work_comparison = compare_generation_work(config.sample_length)

    print(f"\nGenerated without cache:\n{sample_without_cache}")
    print(f"\nGenerated with cache:\n{cache_sample['text']}")
    print(f"\n\n {work_comparison} \n\n")

    return {
        "train_loss": train_dict["loss"],
        "val_loss": val_dict["loss"],
        "val_perplexity": val_dict["perplexity"],
        "sample_without_cache": sample_without_cache,
        "sample_with_cache": str(cache_sample["text"]),
        "cache_final_length": float(cache_sample["final_cache_length"]),
        "cache_speedup_ratio": float(work_comparison["speedup_ratio"]),
    }

    #raise NotImplementedError("Run the full modern mini GPT experiment.")


# Check the final summary from `run_training`.
# This does not judge text quality.
# It only checks that the full path ran and returned the expected fields.
def check_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when the full experiment returned the expected fields."""
    return (
        "train_loss" in metrics
        and "val_loss" in metrics
        and "val_perplexity" in metrics
        and "sample_without_cache" in metrics
        and "sample_with_cache" in metrics
    )


# Run one scaffold task and catch unfinished TODOs.
# This lets the file stop cleanly at the next function you still need to write.
def report_task_progress(task_name: str, task_fn: Callable, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


# Run the top-level scaffold.
# For this combined exercise, `run_training` owns most of the workflow.
# Build and test smaller functions first, then connect them inside `run_training`.
def main() -> None:
    """Run the modern mini GPT scaffold."""
    config = ModernMiniGPTConfig()
    set_seed(config.random_state)
    print("Main goal: combine RoPE, RMSNorm, SwiGLU, and KV cache in one tiny decoder model.")
    print("Use full-sequence training first. Use KV cache only during generation.")
    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok and metrics is not None:
        print_pass_or_try_again(check_metrics(metrics), "modern mini GPT integration", float(int(check_metrics(metrics))), "training and generation returned expected fields")


if __name__ == "__main__":
    main()
