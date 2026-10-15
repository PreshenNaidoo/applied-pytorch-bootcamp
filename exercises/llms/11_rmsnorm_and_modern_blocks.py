"""
============================================================
EXERCISE 11: RMSNORM AND MODERN BLOCKS
============================================================

WHAT YOU ARE BUILDING
You are building RMSNorm and then using it inside a tiny decoder-only language model.

Plain-language meaning:
- LayerNorm recenters and rescales values
- RMSNorm rescales values but does not recenter them
- many modern language models use RMSNorm
- pre-norm means normalization happens before attention or feed-forward layers

WHY THIS EXERCISE EXISTS
Modern LLM blocks often look a little different from older textbook transformer blocks.
RMSNorm and pre-norm blocks are common.
This exercise teaches:
- RMSNorm from scratch
- how RMSNorm differs from LayerNorm
- pre-norm block structure
- residual connections
- optional SwiGLU feed-forward layers
- how the block fits into a small trainable model
- how to run a short next-token prediction experiment

HOW THIS FITS INTO THE BOOTCAMP
After the basic transformer and decoder exercises,
this file zooms in on several details that show up often in more modern LLM blocks.

What each step is doing in simple words:
1. build the normalization piece
2. build the modern feed-forward piece
3. place them inside a transformer-style residual block
4. place that block inside a tiny decoder-only model
5. train the model to predict the next token
6. generate a short sample so the block has real context

Beginner intuition:
- normalization helps hidden values stay in a more manageable range
- RMSNorm is a simpler normalization rule than LayerNorm
- modern decoder blocks often use this simpler rule before the main sublayers

Small example:
- if hidden values become very large or very uneven,
- normalization can rescale them
- then the next layer gets a cleaner input to work with

DATASET
Use Tiny Shakespeare as the recommended dataset.

Use shape checks to confirm that the RMSNorm block keeps the right tensor shape.
Those checks belong inside `run_shape_checks`.
The main exercise path should still move toward the Tiny Shakespeare language-model experiment.

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- load the Tiny Shakespeare text file
- split the raw text into contiguous `60 / 20 / 20` train, validation, and test sections
- tokenize text into simple word tokens
- build the vocabulary from training text only
- create next-token windows
- train a tiny decoder-only model that uses RMSNorm blocks

INPUTS AND OUTPUTS
Hidden states inside the model:
- shape: `[batch_size, context_length, embed_dim]`
RMSNorm output:
- same shape as input
Block output:
- same shape as input
Language-model input:
- `input_ids`: `[batch_size, context_length]`
- `target_ids`: `[batch_size, context_length]`
Language-model output:
- logits: `[batch_size, context_length, vocab_size]`

MODEL ARCHITECTURE (HIGH LEVEL ONLY)

Default path:
- token embeddings
- positional embeddings
- repeated modern decoder blocks
- RMSNorm
- masked self-attention
- residual add
- RMSNorm
- SwiGLU feed-forward block
- residual add
- final RMSNorm
- linear language-model head

LOSS FUNCTION
Use `nn.CrossEntropyLoss` for the full training experiment.

Return raw logits.
Do not apply softmax before the loss.

METRICS TO TRACK
Track at least:
- RMSNorm shape check
- block shape check
- one note comparing RMSNorm and LayerNorm
- train loss
- validation loss
- validation perplexity
- one generated text sample

COMMON MISTAKES
- changing the hidden-state shape
- forgetting the learned scale parameter
- normalizing over the wrong dimension
- forgetting residual connections
- placing normalization inconsistently

STEP-BY-STEP TODO
1. Implement RMSNorm.
2. Check that RMSNorm keeps the same shape.
3. Compare RMSNorm with `nn.LayerNorm`.
4. Build a tiny pre-norm decoder block.
5. Add residual connections.
6. Add a feed-forward block.
7. Optionally use SwiGLU instead of a plain MLP.
8. Run shape checks.
9. Load Tiny Shakespeare.
10. Build next-token dataloaders.
11. Build a tiny decoder-only language model that uses RMSNorm blocks.
12. Train the model.
13. Evaluate on validation text.
14. Generate a short sample.
15. Write a short note about RMSNorm vs LayerNorm.

DEFINITION OF DONE
You are done when:
- RMSNorm has a learned scale parameter
- RMSNorm keeps shape unchanged
- the pre-norm block keeps shape unchanged
- residual connections are used
- you can explain RMSNorm in plain words
- the tiny model trains without shape errors
- validation loss and perplexity are returned
- one generated text sample is returned
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again

device = "cuda" if torch.cuda.is_available() else "cpu"

# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for RMSNorm and modern block practice.
# Return:
# - a config object with data, model, and training settings
@dataclass
class ModernMiniGPTConfig:
    """Settings for RMSNorm and modern block practice."""
    data_path: str = str(BOOTCAMP_ROOT / "data" / "input.txt")
    batch_size: int = 16
    context_length: int = 16
    embed_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256
    num_layers: int = 4
    learning_rate: float = 1e-4
    epochs: int = 2
    sample_length: int = 200
    target_val_loss: float = 5.0
    eps: float = 1e-6
    random_state: int = 42

    vocab_size: int | None = None
    token_to_int: dict[str, int] | None = None
    int_to_token: dict[int, str] | None = None
    
    tokenizer_name: str = "basic_word_tokenize"

# Describe expected shapes for the exercise.
# Return:
# - dictionary with hidden-state, RMSNorm output, and block output shapes
def describe_modern_block_shapes(config: ModernMiniGPTConfig) -> dict[str, object]:
    """Return expected shapes for this exercise."""
    return {
        "hidden_states": (config.batch_size, config.context_length, config.embed_dim),
        "rmsnorm_output": (config.batch_size, config.context_length, config.embed_dim),
        "block_output": (config.batch_size, config.context_length, config.embed_dim),
        "input_ids": (config.batch_size, config.context_length),
        "target_ids": (config.batch_size, config.context_length),
        "causal_mask": (1, 1, config.context_length, config.context_length),
        "logits": (config.batch_size, config.context_length, "vocab_size"),
    }

# Build the decoder causal mask used by the tiny modern block.
# This keeps the attention part consistent with decoder-only left-to-right behavior.
def build_causal_mask(context_length: int) -> torch.Tensor:
    """Return a causal mask for decoder self-attention."""

    ones = torch.ones(size=(context_length, context_length), dtype=torch.bool)
    causal_mask = torch.tril(ones).unsqueeze(dim=0).unsqueeze(dim=0)
    
    return causal_mask
    #raise NotImplementedError("Return a boolean causal mask with shape [1, 1, seq, seq].")


# Load the raw text corpus.
# Recommended default:
# - Tiny Shakespeare from `data/input.txt`
# If your local file has a different name, update `config.data_path`.
# Return:
# - one Python string with the full text
def load_raw_text(config: ModernMiniGPTConfig) -> str:
    """Return the full Tiny Shakespeare text as one string."""

    with open(file=config.data_path, mode="r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text
    #raise NotImplementedError("Load Tiny Shakespeare from config.data_path and return one string.")


# Split the raw text into train, validation, and test text sections.
# Use a contiguous split so the text order stays natural.
# Return a dict with:
# - `train_text`
# - `val_text`
# - `test_text`
def split_text_corpus(raw_text: str) -> dict[str, str]:
    """Return a contiguous 60 / 20 / 20 text split."""

    num_train, num_val = int(len(raw_text) * 0.6), int(len(raw_text) * 0.2)

    train_text = raw_text[ : num_train]
    val_text = raw_text[num_train : num_train + num_val]
    test_text = raw_text[num_train + num_val : ]

    return {"train_text": train_text, "val_text": val_text, "test_text": test_text}
    #raise NotImplementedError("Split raw text into train_text, val_text, and test_text.")


# Simple fallback tokenizer.
# It keeps words and punctuation as tokens.
# Example:
# - `"To be, or not"` -> `["To", "be", ",", "or", "not"]`
def basic_word_tokenize(text: str) -> list[str]:
    """Return simple word and punctuation tokens."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


# Build a vocabulary from training tokens only.
#
# Return a dictionary with:
# - `token_to_int`: maps token strings to integer ids
# - `int_to_token`: maps integer ids back to token strings
# - `vocab_size`: number of tokens in the vocabulary
#
# Include these two special tokens:
# - `[PAD]` at id `0`
# - `[UNK]` at id `1`
#
# Build this from training tokens only.
# Do not use validation or test text to build the vocabulary.
def build_vocab(train_tokens: list[str]) -> dict[str, object]:
    """Return vocabulary mappings built from training tokens only."""
    raise NotImplementedError("Build token_to_int, int_to_token, and vocab_size from training tokens only.")


# A small Dataset class is a clean way to create next-token training windows.
# In this part, the dataset should work with word tokens instead of single characters.
# It should store:
# - the token list
# - the context length
# - the vocabulary mapping
# `__len__` should return how many short windows can be made from the token list.
# `__getitem__` should return one dict with exactly these keys:
# - `"input_ids"`: tokens from position i to i + context_length
# - `"target_ids"`: the same token sequence shifted one token forward
class WordTokenDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokens: list[str], context_length: int, vocab_dict: dict[str, int]):
        self.tokens = tokens
        self.num_tokens = len(tokens)
        self.context_length = context_length
        self.vocab_dict = vocab_dict

        self.unk_id = 1        

    def __len__(self) -> int:
        return self.num_tokens - self.context_length

    def __getitem__(self, index) -> dict[str, torch.tensor]:        
        input_tokens = self.tokens[index: index + self.context_length]
        target_tokens = self.tokens[index + 1: index + 1 + self.context_length]

        input_ids, target_ids = [], []
        for token in input_tokens:            
            input_ids.append(self.vocab_dict.get(token, self.unk_id))
        
        for token in target_tokens:
            target_ids.append(self.vocab_dict.get(token, self.unk_id))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        return {"input_ids": input_ids, "target_ids": target_ids}

# Build dataloaders for the full RMSNorm language-model experiment.
# Return:
# - `train`
# - `val`
# - `test`
def load_dataloaders(config: ModernMiniGPTConfig) -> dict[str, DataLoader]:
    """Return train, validation, and test dataloaders."""

    raw_text = load_raw_text(config)
    data_splits = split_text_corpus(raw_text)

    all_words = basic_word_tokenize(raw_text)
    vocabulary = sorted(set(all_words))
    vocabulary.insert(0, "[PAD]")
    vocabulary.insert(1, "[UNK]")
    vocab_size = len(vocabulary)

    config.vocab_size = vocab_size
    config.token_to_int = {word: index for index, word in enumerate(vocabulary)}
    config.int_to_token = {index: word for word, index in config.token_to_int.items()}

    train_ds = WordTokenDataset(tokens=basic_word_tokenize(data_splits["train_text"]), context_length=config.context_length, vocab_dict=config.token_to_int)
    val_ds = WordTokenDataset(tokens=basic_word_tokenize(data_splits["val_text"]), context_length= config.context_length, vocab_dict=config.token_to_int)
    test_ds = WordTokenDataset(tokens=basic_word_tokenize(data_splits["test_text"]), context_length= config.context_length, vocab_dict=config.token_to_int)

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}    
    #raise NotImplementedError("Load text, split it, tokenize it, build vocab, and return dataloaders.")


# Beginner hints for this file:
# - RMSNorm is like a simpler cousin of LayerNorm
# - both keep the same shape, but they do not compute exactly the same thing
# - pre-norm means normalize before the attention or feed-forward work
# - residual connections help the model keep and refine information instead of replacing it completely
# - get the plain RMSNorm shape test working first, then build the larger block around it
class RMSNorm(nn.Module):
    """Root mean square normalization."""

    # Build RMSNorm.
    # Expected module state:
    # - learned scale parameter with shape `[embed_dim]`
    # - small epsilon value for numerical stability
    #
    # Formula used in `forward`:
    # - rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
    # - output = (x / rms) * scale
    #
    # Meaning:
    # - `x`: hidden states with shape `[batch_size, context_length, embed_dim]`
    # - `x^2`: square every value in `x`
    # - `mean(x^2)`: average over the last dimension only, which is `embed_dim`
    # - `eps`: tiny number that prevents divide-by-zero
    # - `scale`: learned parameter with shape `[embed_dim]`
    #
    # In code, `scale` is often called `weight`.
    # A good starting value is all ones:
    # - `nn.Parameter(torch.ones(embed_dim))`
    def __init__(self, embed_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps

        self.scale = nn.Parameter(torch.ones(size=(embed_dim,)))

        #raise NotImplementedError("Create the learned RMSNorm scale parameter.")

    # Apply RMSNorm to hidden states.
    # Input:
    # - `x`: `[batch_size, context_length, embed_dim]`
    # Return:
    # - tensor with the same shape as `x`
    #
    # Small example for one token:
    # - x = `[3.0, 4.0]`
    # - x^2 = `[9.0, 16.0]`
    # - mean(x^2) = `(9.0 + 16.0) / 2 = 12.5`
    # - rms = `sqrt(12.5 + eps)`
    # - normalized = `x / rms`
    #
    # Shape check:
    # - input shape: `[batch_size, context_length, embed_dim]`
    # - rms shape if `keepdim=True`: `[batch_size, context_length, 1]`
    # - output shape: `[batch_size, context_length, embed_dim]`
    #
    # Suggested implementation steps:
    # 1. square `x`
    # 2. average over `dim=-1` with `keepdim=True`
    # 3. add `self.eps`
    # 4. take the square root
    # 5. divide `x` by the rms value
    # 6. multiply by the learned scale parameter
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return RMS-normalized hidden states with the same shape as `x`."""
        # LayerNorm and RMSNorm both rescale hidden states,
        # but RMSNorm focuses on overall size and skips mean-centering.
        # For this exercise, the key check is shape stability plus the learned scale.

        # x is of shape [batch_size, context_length, embed_dim]

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        norm = x/rms
        norm = norm * self.scale

        return norm
        #raise NotImplementedError("Apply RMSNorm over the last dimension.")

# Define the SwiGLU feed-forward block.
# Expected behavior:
# - take hidden states in
# - return hidden states with the same final embedding dimension
#
# Plain idea:
# - a normal MLP transforms `x` through one hidden path
# - SwiGLU uses two hidden paths:
#   - a gate path that decides what to let through
#   - a value path that carries the candidate information
# - the gate and value are multiplied together before projecting back down
#
# Formula:
# - gate = gate_proj(x)
# - value = value_proj(x)
# - hidden = SiLU(gate) * value
# - output = out_proj(hidden)
#
# Meaning:
# - `x`: `[batch_size, context_length, embed_dim]`
# - `gate`: `[batch_size, context_length, mlp_dim]`
# - `value`: `[batch_size, context_length, mlp_dim]`
# - `hidden`: `[batch_size, context_length, mlp_dim]`
# - `output`: `[batch_size, context_length, embed_dim]`
#
# `SiLU` is an activation function.
# In PyTorch, use `torch.nn.functional.silu(gate)` or `F.silu(gate)`.
class SwiGLUFeedForward(nn.Module):
    """Small SwiGLU feed-forward block."""

    # Build the feed-forward layers.
    # Expected modules:
    # - gate projection: `nn.Linear(embed_dim, mlp_dim)`
    # - value projection: `nn.Linear(embed_dim, mlp_dim)`
    # - output projection: `nn.Linear(mlp_dim, embed_dim)`
    #
    # The first two projections expand from `embed_dim` to `mlp_dim`.
    # The output projection brings the tensor back to `embed_dim`
    # so the residual connection can add it back to the original hidden states.
    def __init__(self, embed_dim: int, mlp_dim: int) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(in_features=embed_dim, out_features=mlp_dim)
        self.value_proj = nn.Linear(in_features=embed_dim, out_features=mlp_dim)
        self.proj = nn.Linear(in_features=mlp_dim, out_features=embed_dim)

        #raise NotImplementedError("Build the SwiGLU feed-forward layers.")

    # Run the feed-forward block.
    # Input:
    # - `x`: `[batch_size, context_length, embed_dim]`
    # Return:
    # - tensor with shape `[batch_size, context_length, embed_dim]`
    #
    # Small shape example:
    # - if `x` is `[16, 32, 128]`
    # - and `mlp_dim = 256`
    # - gate and value become `[16, 32, 256]`
    # - after multiplying them, hidden is still `[16, 32, 256]`
    # - output projection returns `[16, 32, 128]`
    #
    # Suggested implementation steps:
    # 1. compute `gate = self.gate_proj(x)`
    # 2. compute `value = self.value_proj(x)`
    # 3. compute `hidden = F.silu(gate) * value`
    # 4. return `self.out_proj(hidden)`
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return feed-forward output with shape `[batch, seq, embed_dim]`."""

        #x is shape [batch_size, context_length, embed_dim]

        gate = self.gate_proj(x)

        value = self.value_proj(x)

        hidden = torch.nn.functional.silu(gate) * value

        proj = self.proj(hidden)

        return proj
        #raise NotImplementedError("Run the SwiGLU feed-forward block.")


# Compute scaled dot-product attention.
# Input shapes:
# - `query`: `[batch_size, num_heads, context_length, head_dim]`
# - `key`: `[batch_size, num_heads, context_length, head_dim]`
# - `value`: `[batch_size, num_heads, context_length, head_dim]`
# - `mask`: `[1, 1, context_length, context_length]`
# Return:
# - `[batch_size, num_heads, context_length, head_dim]`
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked scaled dot-product attention output."""

    # query, key and value are all of shape [batch_size, num_heads, context_length, head_dim]
    # mask is of shape [1, 1, context_length, context_length]

    batch_size, num_heads, context_length, head_dim = query.shape

    scores = query @ key.transpose(dim0 = -2, dim1 = -1)        # shape = [batch_size, num_heads, context_length, context_length]
    scale = scores/sqrt(head_dim)

    scale = scale.masked_fill(~mask, value=float("-inf"))

    att_weight = scale.softmax(dim=-1)                          # shape = [batch_size, num_heads, context_length, context_length]

    output = att_weight @ value

    return output
    #raise NotImplementedError("Compute attention scores, apply the causal mask, softmax, and mix values.")


# Masked self-attention for the tiny decoder.
# This is the same main attention idea from the related exercise.
# It is used here so RMSNorm is part of a real trainable decoder block.
class MaskedSelfAttention(nn.Module):
    """Masked multi-head self-attention for a decoder-only model."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if embed_dim % num_heads != 0:
            raise ValueError("Embed dim is not exactly divisible by num_heads.")

        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.key = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.value = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        #raise NotImplementedError("Build Q, K, V projections, output projection, and head_dim.")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return masked self-attention output with shape `[batch, seq, embed_dim]`."""

        batch_size, seq_length, embed_dim = x.shape             # (16, 16, 128)

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.reshape(batch_size, seq_length, 1, embed_dim).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_length, 1, embed_dim).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, 1, embed_dim).reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        query = query.permute(dims=(0, 2, 1, 3))
        key = key.permute(dims=(0, 2, 1, 3))
        value = value.permute(dims=(0, 2, 1, 3))

        att_output = scaled_dot_product_attention(query, key, value, mask)

        output = att_output.permute(dims=(0, 2, 1, 3))
        output = output.reshape(batch_size, seq_length, embed_dim)

        proj = self.proj(output)

        return proj
        #raise NotImplementedError("Project Q/K/V, split heads, run attention, join heads, and project output.")

# Define a tiny pre-norm decoder block.
# Expected behavior:
# - normalize before attention
# - add residual
# - normalize before feed-forward
# - add residual again
#
# This is where the pieces you just built are used.
# In `__init__`, create:
# - `self.norm1 = RMSNorm(config.embed_dim, config.eps)`
# - `self.attention = MaskedSelfAttention(config.embed_dim, config.num_heads)`
# - `self.norm2 = RMSNorm(config.embed_dim, config.eps)`
# - `self.feed_forward = SwiGLUFeedForward(config.embed_dim, config.mlp_dim)`
#
# In `forward`, use them in this order:
# - attention path: `x = x + self.attention(self.norm1(x), causal_mask)`
# - feed-forward path: `x = x + self.feed_forward(self.norm2(x))`
class PreNormDecoderBlock(nn.Module):
    """Tiny modern decoder block with pre-norm and residual connections."""

    # Build the block modules.
    # Expected modules:
    # - RMSNorm before attention
    # - masked self-attention layer
    # - RMSNorm before feed-forward
    # - feed-forward block
    def __init__(self, config: ModernMiniGPTConfig) -> None:
        super().__init__()
        self.config = config

        self.norm1 = RMSNorm(config.embed_dim, config.eps)

        self.masked_self_attention = MaskedSelfAttention(config.embed_dim, config.num_heads)

        self.norm2 = RMSNorm(config.embed_dim, config.eps)

        self.feed_forward = SwiGLUFeedForward(config.embed_dim, config.mlp_dim)
        
        #raise NotImplementedError("Build RMSNorm, self-attention, feed-forward, and residual paths.")

    # Run the pre-norm decoder block.
    # Input:
    # - `x`: `[batch_size, context_length, embed_dim]`
    # - optional causal mask with shape `[1, 1, context_length, context_length]`
    # Return:
    # - tensor with the same shape as `x`
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return updated hidden states with the same shape as input."""
        # Typical pre-norm flow:
        # 1. normalize
        # 2. attention
        # 3. add residual
        # 4. normalize again
        # 5. feed-forward
        # 6. add residual again

        residual = x
        x = self.norm1(x)
        x = self.masked_self_attention(x, causal_mask)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x
        #raise NotImplementedError("Run pre-norm attention, residual, feed-forward, and residual.")


# Tiny decoder-only language model that uses RMSNorm blocks.
# This puts RMSNorm into the same kind of context as the related exercise:
# - token embeddings
# - positional embeddings
# - masked decoder blocks
# - final RMSNorm
# - language-model head
#
# This is where the finished `PreNormDecoderBlock` is used.
# In `__init__`, build:
# - token embedding table: `nn.Embedding(vocab_size, config.embed_dim)`
# - position embedding table: `nn.Embedding(config.context_length, config.embed_dim)`
# - several decoder blocks, usually with `nn.ModuleList`
# - final norm: `RMSNorm(config.embed_dim, config.eps)`
# - language-model head: `nn.Linear(config.embed_dim, vocab_size)`
#
# Each decoder block already contains:
# - RMSNorm before attention
# - masked self-attention
# - RMSNorm before SwiGLU
# - SwiGLU feed-forward
class ModernMiniGPT(nn.Module):
    """Tiny decoder-only language model with RMSNorm blocks."""

    def __init__(self, vocab_size: int, config: ModernMiniGPTConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.embed_dim, padding_idx=0)
        self.positional_embedding = nn.Embedding(num_embeddings=config.context_length, embedding_dim=config.embed_dim)

        self.decoder_blocks = nn.ModuleList([PreNormDecoderBlock(config) for _ in range(config.num_layers)])

        self.norm = RMSNorm(embed_dim=config.embed_dim, eps = config.eps)

        self.classification_head = nn.Linear(in_features=config.embed_dim, out_features=vocab_size)

        #raise NotImplementedError("Build embeddings, RMSNorm decoder blocks, final RMSNorm, and LM head.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return raw next-token logits with shape `[batch, context_length, vocab_size]`."""


        # input_ids is of shape [batch_size, context_length]
        # however the context length shapes differ between training and inference, training will use full context_length
        # whereas for inference, you could pass 1 token id which is essentially saying "hello".
        # we use the sequence_length in this function instead of config.context_length

        batch_size, sequence_length = input_ids.shape

        token_embeddings = self.token_embedding(input_ids)      # shape is [batch_size, sequence_length, embed_dim]
        
        positions = torch.arange(start=0, end=sequence_length, step=1, device=input_ids.device) # shape is [sequence_length]
        positions = positions.unsqueeze(dim=0)                                                  # shape is [1, sequence_length]
        positions = positions.expand(size=(batch_size, sequence_length))                        # shape is [batch_size, sequence_length]
        pos_embeddings = self.positional_embedding(positions)                                   # shape is [batch_size, sequence_length, embed_dim]

        embeddings = token_embeddings + pos_embeddings

        causal_mask = build_causal_mask(sequence_length)
        causal_mask = causal_mask.to(input_ids.device)

        features = embeddings
        for decoder_block in self.decoder_blocks:
            features = decoder_block(features, causal_mask)           # shape is [batch_size, sequence_length, embed_dim]

        features = self.norm(features)

        logits = self.classification_head(features)

        return logits
        #raise NotImplementedError("Run token embeddings, positions, RMSNorm blocks, final norm, and LM head.")

# Compare RMSNorm and LayerNorm outputs.
# Return:
# - output shapes
# - one difference value
# The goal is to see that they normalize differently while keeping the same shape.
def compare_rmsnorm_and_layernorm(x: torch.Tensor, rmsnorm: RMSNorm, layernorm: nn.LayerNorm) -> dict[str, float | tuple[int, ...]]:
    """Compare RMSNorm and LayerNorm outputs."""

    rms_output = rmsnorm(x)
    layer_output = layernorm(x)

    mean_absolute_difference = (rms_output - layer_output).abs().mean().item()

    return {
        "input_shape": tuple(x.shape),
        "rmsnorm_output_shape": tuple(rms_output.shape),
        "layernorm_output_shape": tuple(layer_output.shape),
        "mean_absolute_difference": mean_absolute_difference,
    }
    #raise NotImplementedError("Return output shapes and a simple difference score.")

# Run all shape checks for RMSNorm and the modern block.
#
# This function may create a small hidden-state tensor inside itself.
# That tensor is only for checking shapes.
# The real training data still comes from Tiny Shakespeare through `load_dataloaders`.
#
# Return:
# - dictionary with boolean checks and a short note
def run_shape_checks(config: ModernMiniGPTConfig) -> dict[str, bool | float | str]:
    """Run RMSNorm and block shape checks."""

    expected_hidden_shape = (config.batch_size, config.context_length, config.embed_dim)
    expected_mask_shape = (1, 1, config.context_length, config.context_length)

    hidden_states = torch.randn(expected_hidden_shape)
    causal_mask = build_causal_mask(config.context_length)

    rmsnorm = RMSNorm(config.embed_dim, config.eps).to(hidden_states.device)
    rms_output = rmsnorm(hidden_states)

    layernorm = nn.LayerNorm(config.embed_dim).to(hidden_states.device)
    comparison = compare_rmsnorm_and_layernorm(hidden_states, rmsnorm, layernorm)
    print(comparison)

    feed_forward = SwiGLUFeedForward(config.embed_dim, config.mlp_dim).to(hidden_states.device)
    feed_forward_output = feed_forward(hidden_states)

    attention = MaskedSelfAttention(config.embed_dim, config.num_heads).to(hidden_states.device)
    attention_output = attention(hidden_states, causal_mask)

    block = PreNormDecoderBlock(config).to(hidden_states.device)
    block_output = block(hidden_states, causal_mask)

    return {
        "rmsnorm_shape_ok": tuple(rms_output.shape) == expected_hidden_shape,
        "feed_forward_shape_ok": tuple(feed_forward_output.shape) == expected_hidden_shape,
        "attention_shape_ok": tuple(attention_output.shape) == expected_hidden_shape,
        "block_shape_ok": tuple(block_output.shape) == expected_hidden_shape,
        "mask_shape_ok": tuple(causal_mask.shape) == expected_mask_shape,
        "rmsnorm_vs_layernorm_difference": comparison["mean_absolute_difference"],
        "note": "RMSNorm and LayerNorm both keep the same shape, but RMSNorm rescales without mean-centering.",
    }
    #raise NotImplementedError("Return shape checks and a short RMSNorm note.")


# Build the model, loss, and optimizer for the full experiment.
# This is the point where RMSNorm moves from "one block" into "a model you can train".
# Return:
# - model
# - loss function
# - optimizer
def build_training_objects(config: ModernMiniGPTConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, CrossEntropyLoss, and optimizer."""

    model = ModernMiniGPT(vocab_size=config.vocab_size, config=config)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)

    return model, loss_fn, optimizer
    #raise NotImplementedError("Build ModernMiniGPT, CrossEntropyLoss, and Adam optimizer.")


# Train for one full epoch.
# Return:
# - `loss`: mean training loss
# - `perplexity`: exp(mean loss)
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Return mean train loss and train perplexity."""

    model.train()

    losses = []

    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)       # shape is (batch_size, sequence_length, vocab_size)
        batch_size, sequence_length, vocab_size = logits.shape

        loss = loss_fn(logits.reshape(batch_size*sequence_length, vocab_size), target_ids.reshape(batch_size*sequence_length))
        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean()
    perp = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": perp.item()}
    #raise NotImplementedError("Train for one epoch and return loss and perplexity.")


# Evaluate without updating model weights.
# Return:
# - `loss`: mean loss
# - `perplexity`: exp(mean loss)
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

        logits = model(input_ids)       # shape is (batch_size, sequence_length, vocab_size)
        batch_size, sequence_length, vocab_size = logits.shape

        loss = loss_fn(logits.reshape(batch_size*sequence_length, vocab_size), target_ids.reshape(batch_size*sequence_length))       

        losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean()
    perp = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": perp.item()}
    # raise NotImplementedError("Evaluate the model and return loss and perplexity.")


# Generate one short sample from the trained model.
# This is only a rough sanity check.
# A tiny model trained briefly will not write clean text.
@torch.no_grad()
def generate_text(model: nn.Module, start_text: str, config: ModernMiniGPTConfig) -> str:
    """Generate one short text sample."""

    temperature = 0.8
    use_temperature = True

    if not start_text:
        start_text = "Hello world."

    start_tokens = basic_word_tokenize(start_text)
    start_tokens = start_tokens[-config.context_length:]

    encode = lambda st: [config.token_to_int.get(w, 1) for w in st]
    decode = lambda l: " ".join([config.int_to_token[i] for i in l])

    model_device = next(model.parameters()).device

    context = torch.tensor(encode(start_tokens), dtype=torch.long, device=model_device)
    context = context.unsqueeze(dim=0) # add a fake batch dim

    generated = []
    for i in range(config.sample_length):
        logits = model(context)             # shape = [batch_size, sequence_len, vocab_size]

        last_token_vec = logits[:, -1, :]  # shape = [batch_size, vocab_size]        

        if use_temperature:
            prob = torch.softmax(last_token_vec / temperature, dim=-1)             
            last_token_id = torch.multinomial(prob, num_samples=1)
        else:
            last_token_id = last_token_vec.argmax(dim=-1, keepdim=True)

        context = torch.concat([context, last_token_id], dim=1) # here last_token_id should be shape [1, 1]
        context = context[:, -config.context_length:]
        generated.append(last_token_id.squeeze().long().item())

    return start_text + " " + decode(generated)

    #raise NotImplementedError("Generate tokens one at a time and return one text sample.")


# Run the full training experiment.
# Return a final summary dict with:
# - `train_loss`
# - `val_loss`
# - `val_perplexity`
# - `sample`
def run_training(config: ModernMiniGPTConfig) -> dict[str, float | str]:
    """Run the full RMSNorm mini GPT experiment."""

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    for i in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i+1} -> loss: {train_dict['loss']}, perp: {train_dict['perplexity']}, val_loss: {val_dict['loss']}, val_perp: {val_dict['perplexity']}")

    test_dict = evaluate(model, loaders["test"], loss_fn)
    print(f"Test results -> loss: {test_dict['loss']}, perp: {test_dict['perplexity']}")

    print(f"\n GENERATED: \n {generate_text(model=model, start_text = 'To be or not', config=config)}")

    return {"train_loss": train_dict["loss"], 
            "val_loss": val_dict["loss"], "val_perplexity": val_dict["perplexity"]}

    #raise NotImplementedError("Train, evaluate, and generate a short sample.")

# Check whether the main shape checks passed.
# Return:
# - True when RMSNorm and block outputs keep the expected shapes
def check_shape_metrics(metrics: dict[str, bool | float | str]) -> bool:
    """Return True when the main shape checks pass."""
    return bool(metrics.get("rmsnorm_shape_ok", False)) and bool(metrics.get("block_shape_ok", False))


# Check whether the training run returned the expected values.
# The exact loss can move around because this is a tiny language model.
# So this check mainly proves that the training path ran end to end.
def check_training_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when the training summary has the expected keys."""
    return (
        "train_loss" in metrics
        and "val_loss" in metrics
        and "val_perplexity" in metrics
        and "sample" in metrics
    )

# Run one scaffold task and catch unfinished TODOs.
# Return:
# - result object
# - boolean saying whether the task is done
def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True

# Run the exercise scaffold one step at a time.
# Expected behavior:
# - stop at the first unfinished TODO
# - print progress so the student knows what to implement next
def main() -> None:
    """Run the RMSNorm scaffold one step at a time."""
    config = ModernMiniGPTConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 27
    # Update progress after each scaffold task.
    # Return:
    # - True if the scaffold should keep running
    # - False if it should stop at the current TODO
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    _, ok = report_task_progress("describe_modern_block_shapes", describe_modern_block_shapes, config)
    if not count_or_stop(ok):
        return
    mask, ok = report_task_progress("build_causal_mask", build_causal_mask, config.context_length)
    if not count_or_stop(ok):
        return
    raw_text, ok = report_task_progress("load_raw_text", load_raw_text, config)
    if not count_or_stop(ok):
        return
    split_text, ok = report_task_progress("split_text_corpus", split_text_corpus, raw_text)
    if not count_or_stop(ok):
        return
    train_tokens = basic_word_tokenize(split_text["train_text"])
    # vocab_result, ok = report_task_progress("build_vocab", build_vocab, train_tokens)
    # if not count_or_stop(ok):
    #     return
    # if isinstance(vocab_result, dict):
    #     config.token_to_int = vocab_result.get("token_to_int")  # type: ignore[assignment]
    #     config.int_to_token = vocab_result.get("int_to_token")  # type: ignore[assignment]
    #     config.vocab_size = vocab_result.get("vocab_size")  # type: ignore[assignment]
    
    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if not count_or_stop(ok):
        return

    sample_tokens = train_tokens[: config.context_length + 2]
    dataset, ok = report_task_progress("WordTokenDataset.__init__", WordTokenDataset, sample_tokens, config.context_length, config.token_to_int)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("WordTokenDataset.__len__", len, dataset)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("WordTokenDataset.__getitem__", dataset.__getitem__, 0)
    if not count_or_stop(ok):
        return

    hidden_states = torch.randn(config.batch_size, config.context_length, config.embed_dim)    
    rmsnorm, ok = report_task_progress("RMSNorm.__init__", RMSNorm, config.embed_dim, config.eps)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("RMSNorm.forward", rmsnorm, hidden_states)
    if not count_or_stop(ok):
        return
    feed_forward, ok = report_task_progress("SwiGLUFeedForward.__init__", SwiGLUFeedForward, config.embed_dim, config.mlp_dim)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("SwiGLUFeedForward.forward", feed_forward, hidden_states)
    if not count_or_stop(ok):
        return

    head_dim = config.embed_dim // config.num_heads
    query = torch.randn(config.batch_size, config.num_heads, config.context_length, head_dim)
    key = torch.randn(config.batch_size, config.num_heads, config.context_length, head_dim)
    value = torch.randn(config.batch_size, config.num_heads, config.context_length, head_dim)
    _, ok = report_task_progress("scaled_dot_product_attention", scaled_dot_product_attention, query, key, value, mask)
    if not count_or_stop(ok):
        return
    attention, ok = report_task_progress("MaskedSelfAttention.__init__", MaskedSelfAttention, config.embed_dim, config.num_heads)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("MaskedSelfAttention.forward", attention, hidden_states, mask)
    if not count_or_stop(ok):
        return
    block, ok = report_task_progress("PreNormDecoderBlock.__init__", PreNormDecoderBlock, config)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("PreNormDecoderBlock.forward", block, hidden_states, mask)
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("ModernMiniGPT.__init__", ModernMiniGPT, config.vocab_size, config)
    if not count_or_stop(ok):
        return
    sample_batch = next(iter(loaders["train"]))
    _, ok = report_task_progress("ModernMiniGPT.forward", model, sample_batch["input_ids"])
    if not count_or_stop(ok):
        return
    layernorm = nn.LayerNorm(config.embed_dim)
    _, ok = report_task_progress("compare_rmsnorm_and_layernorm", compare_rmsnorm_and_layernorm, hidden_states, rmsnorm, layernorm)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_shape_checks", run_shape_checks, config)
    print(metrics)
    if not count_or_stop(ok):
        return
    if metrics is not None:
        passed = check_shape_metrics(metrics)
        print_pass_or_try_again(passed, "modern block checks", float(int(passed)), "RMSNorm and block shapes unchanged")

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return
    model, loss_fn, optimizer = training_objects
    _, ok = report_task_progress("train_one_epoch", train_one_epoch, model, loaders["train"], loss_fn, optimizer)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("evaluate", evaluate, model, loaders["val"], loss_fn)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("generate_text", generate_text, model, "To be or not", config)
    if not count_or_stop(ok):
        return
    training_metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and training_metrics is not None:
        passed = check_training_metrics(training_metrics)
        print_pass_or_try_again(passed, "RMSNorm mini GPT experiment", float(int(passed)), "training summary returned expected keys")
if __name__ == "__main__":
    main()
