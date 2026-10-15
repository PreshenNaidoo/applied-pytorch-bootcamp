"""
============================================================
EXERCISE 06: MINI GPT DECODER ONLY
============================================================

WHAT YOU ARE BUILDING
You are building a tiny educational decoder-only GPT-style model.

This is a mini teaching version.
It is not GPT-3 scale.
It is not meant to be large.
It is meant to help you build the main decoder pieces yourself.

You will build:
- token embeddings
- positional embeddings
- masked self-attention
- MLP block
- residual connections
- layer norm
- language-model head
- a simple generation loop

Plain-language meaning:
- `masked self-attention` means each token can look back at earlier tokens, but not forward at future tokens
- `MLP` means a small feed-forward network made of linear layers and activations
- `residual connection` means adding the input of a block back to its output so learning is easier
- `layer norm` means rescaling values inside the model so training stays more stable
- `generation loop` means repeatedly predict one next token, append it, and then continue

Beginner intuition:
- this model learns by playing the game "guess the next character"
- at each position, it is only allowed to use the characters to the left
- masked attention is the rule that stops it from cheating
- the decoder blocks slowly build better features for making that next guess

What "autoregressive next-token prediction" means in plain words:
- `next-token prediction` means:
  - look at the current text
  - guess the one token that should come next
- `autoregressive` means the model keeps extending from earlier tokens to later tokens
- so the full idea is:
  - use the tokens you already have
  - predict the next token
  - then repeat

Small concrete example:
- text: `h e l l o`
- one short training window could be:
  - input:  `h e l l`
  - target: `e l l o`
- position by position, the model is learning:
  - after `h`, guess `e`
  - after `h e`, guess `l`
  - after `h e l`, guess `l`
  - after `h e l l`, guess `o`

Why this is called "autoregressive":
- `auto` here means the model uses the text it already has
- it does not get the future tokens as input
- it must build the sequence one step at a time from left to right

WHY THIS EXERCISE EXISTS
This exercise brings together many ideas from Phase 3.

It teaches:
- causal masking
- autoregressive next-token prediction
- why decoder-only models cannot look at future tokens during training
- how a small generation loop works one step at a time

DATASET
Use Tiny Shakespeare or another tiny public character corpus.

Keep one clear default split rule:
- first `60%` of the raw text for train
- next `20%` for val
- final `20%` for test

Keep the text in order.
Do not shuffle the raw corpus before splitting it.

HOW TO LOAD OR DOWNLOAD THE DATASET
Simple path:
- load the tiny text file
- build a character vocabulary from the training text only
- turn the full text into integer ids
- create input and target windows for next-character prediction
- use the same `context_length` for every batch

INPUTS AND OUTPUTS
Input batch:
- `input_ids`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size, context_length]`
- `target_ids`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size, context_length]`

Example batch:
- `input_ids`: `[32, 128]`
- `target_ids`: `[32, 128]`

What those shapes mean:
- `input_ids: [32, 128]` means 32 examples in the batch and 128 token positions in each input window
- `target_ids: [32, 128]` means one next-token target for each input position

Causal mask:
- type: `torch.Tensor`
- dtype: `torch.bool`
- shape: `[1, 1, context_length, context_length]`
- meaning:
  - `True` means the attention position is allowed
  - `False` means the attention position must be masked out

Model output:
- type: `torch.Tensor`
- shape: `[batch_size, context_length, vocab_size]`
- meaning: one raw logit vector for the next token at every time step

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Default path:
- token embeddings
- positional embeddings
- repeated decoder blocks
- final layer norm
- linear head to vocabulary logits

Each decoder block should include:
- masked self-attention
- residual connection
- layer norm
- MLP
- second residual connection

Decoder idea in plain words:
- this model reads text from left to right
- at each position, it predicts the next token without cheating by looking ahead
- the repeated decoder blocks slowly build better and better features for that prediction

Simple mental picture:
- imagine reading with your finger covering the future words
- you can use everything you have already read
- you cannot use the words still hidden on the right
- that is the basic rule a decoder follows during training

Another simple way to think about training:
- every position in the input window asks one question:
  - "what token should come next here?"
- the model answers that question for all positions in parallel
- but each position is still only allowed to use tokens to its left

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

Important:
- return raw logits
- do not apply softmax before the loss
- compare each position to the next-token target at that position

METRICS TO TRACK
Track at least:
- `train_loss`
  - type: `float`
  - meaning: mean training loss over the whole final training epoch
- `val_loss`
  - type: `float`
  - meaning: mean validation loss over the whole validation loader
- `val_perplexity`
  - type: `float`
  - meaning: `exp(val_loss)` for the whole validation loader

Generation sanity check:
- return one generated string
- read it as a rough qualitative check
- do not expect perfect text from a tiny model

If you pass the test loader to `evaluate(...)`:
- `loss` means mean test loss over the whole test loader
- `perplexity` means `exp(test_loss)` for the whole test loader

TARGET RESULT RANGE
For a tiny educational decoder on a tiny public corpus, a rough validation loss around `1.4` to `2.4` can already be a useful sign.
That usually means validation perplexity is roughly in the `4.1` to `11.0` range.

Also expect only rough text quality.
A tiny model may learn local spelling and short patterns before it learns longer structure.

COMMON MISTAKES
- forgetting causal masking
- using a mask with the wrong shape
- applying softmax before `CrossEntropyLoss`
- forgetting residual connections
- mixing up encoder attention and decoder causal attention
- generating text without feeding the newest token back into the model

STEP-BY-STEP TODO
1. Load the tiny text corpus.
2. Split it into contiguous `60 / 20 / 20` train, val, and test sections.
3. Build the character vocabulary.
4. Create next-token windows.
5. Build the causal mask.
6. Build masked self-attention.
7. Build one decoder block.
8. Stack several decoder blocks.
9. Train the model.
10. Evaluate on validation text.
11. Generate a short sample.

What each step is doing in simple words:
1. read the corpus
2. keep train, val, and test in text order
3. build the character dictionary
4. make one-step-ahead training windows
5. create the no-looking-right mask
6. build the attention part that mixes earlier context
7. package attention and MLP into one decoder block
8. stack blocks so the model can learn richer patterns
9. train the model to guess next characters
10. check whether it works on unseen text
11. use the model in a loop to write new text

STRETCH GOALS
- compare one block and two blocks
- compare different context lengths
- add temperature sampling to generation
- inspect one attention matrix from a finished model

DEFINITION OF DONE
You are done when:
- your model returns raw logits with shape `[batch_size, context_length, vocab_size]`
- your mask prevents attention to future tokens
- you can explain autoregressive next-token prediction in plain words
- the validation loss is in a sensible range
- generation produces text-like local patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.metrics import perplexity_from_loss
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"

class MiniGPTConfig:
    """Settings for the tiny decoder-only GPT exercise."""

    data_path: str = str(BOOTCAMP_ROOT / "data" / "tiny_shakespeare.txt") # or also saved as input.txt
    context_length: int = 128
    embed_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256
    num_layers: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-4 #3e-4
    epochs: int = 10
    random_state: int = 42
    target_val_loss: float = 2.3
    sample_length: int = 250

    vocab_size: int | None = None
    str_to_int: dict | None = None
    int_to_str: dict | None = None



def describe_decoder_shapes(config: MiniGPTConfig) -> dict[str, object]:
    """Return the expected shape flow for the mini GPT.

    Return a dict with exactly these keys:
    - `input_shape`
    - `mask_shape`
    - `hidden_state_shape`
    - `logit_shape`
    - `notes`
    """
    return {
        "input_shape": (config.batch_size, config.context_length),
        "mask_shape": (1, 1, config.context_length, config.context_length),
        "hidden_state_shape": (config.batch_size, config.context_length, config.embed_dim),
        "logit_shape": (config.batch_size, config.context_length, "vocab_size"),
        "notes": [
            "Keep the model and all token tensors on the same device.",
            "Use a causal mask so each position can look left but not right.",
            "Return raw logits. Do not apply softmax before CrossEntropyLoss.",
        ],
    }


        #raise NotImplementedError("Run the tiny decoder-only model and return raw logits.")


# Load the raw text corpus.
# Return:
# - type: `str`
# - meaning: the full text corpus as one Python string
def load_raw_text(config: MiniGPTConfig) -> str:

    with open(file=f"{BOOTCAMP_ROOT}/Data/input.txt", mode="r", encoding="utf-8") as f:
        text = f.read()

    return text

    # raise NotImplementedError("Return the full Tiny Shakespeare text as one Python string.")


# Split the raw text into train, validation, and test text sections.
# Return a dict with exactly these keys:
# - `train_text`
# - `val_text`
# - `test_text`
# Use a contiguous 60% train / 20% val / 20% test split.
def split_text_corpus(raw_text: str) -> dict[str, str]:

    num_train, num_val = int(len(raw_text) * 0.6), int(len(raw_text) * 0.2)

    train_text = raw_text[:num_train]
    val_text = raw_text[num_train: num_train + num_val]
    test_text = raw_text[num_train + num_val:]

    return {"train_text": train_text, "val_text": val_text, "test_text": test_text}


    #raise NotImplementedError("Return the raw text split into contiguous train, val, and test sections.")

class CharTokenDataset(torch.utils.data.Dataset):
    def __init__(self, text, context_length, vocab_dict):
        super().__init__()

        self.text = text
        self.context_length = context_length
        self.vocab_dict = vocab_dict
        self.text_length = len(text)

        self.encode = lambda s: [self.vocab_dict[c] for c in s]

    def __len__(self):
        return self.text_length - self.context_length

    def __getitem__(self, index) -> dict:

        input_ids = torch.tensor(self.encode(self.text[index: index + self.context_length]), dtype=torch.long)
        target_ids = torch.tensor(self.encode(self.text[index + 1: index + 1 + self.context_length]), dtype=torch.long)

        return {"input_ids": input_ids, "target_ids": target_ids}        



@dataclass

# Build the train, validation, and test dataloaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Return type:
# - `dict[str, DataLoader]`
# Each batch should be a dict with exactly these keys:
# - `input_ids`
# - `target_ids`
# Both tensors should have shape `[batch_size, context_length]` and dtype `torch.long`.
# Beginner-friendly window idea:
# - `input_ids` is the current text window
# - `target_ids` is the same window shifted one token ahead
# - example:
#   - input:  `t o _ b`
#   - target: `o _ b e`
# Intuition:
# - the model sees the current text and tries to say what comes next at every position
# - this one-step shift is what teaches autoregressive prediction
# More plain words:
# - the model is not trying to predict the whole future paragraph at once
# - it is only learning one tiny step-ahead prediction at each position
# - putting many of those tiny next-step questions together is what trains a language model
# - this is why the target is just the input shifted left by one token
# A clean way to do this is to make a small Dataset class.
# That class should inherit from `torch.utils.data.Dataset`.
# The Dataset should store the encoded text and the context length.
# `__len__` should return how many short windows can be made from the text.
# `__getitem__` should return one dict with exactly these keys:
# - `"input_ids"`: characters from position i to i + context_length
# - `"target_ids"`: the same text shifted one character forward
# For example, with text "hello" and context length 3:
# - input is  "hel"
# - target is "ell"
def load_dataloaders(config: MiniGPTConfig) -> dict[str, DataLoader]:

    raw_text = load_raw_text(config)
    data_dict = split_text_corpus(raw_text)

    train_text = data_dict["train_text"]
    val_text = data_dict["val_text"]
    test_text = data_dict["test_text"]

    vocabulary = sorted(list(set(raw_text)))
    print(f"VOCAB SIZE: {len(vocabulary)}")

    str_to_int = {c: i for i, c in enumerate(vocabulary)}
    int_to_str = {i: c for c, i in str_to_int.items()}
    # encode = lambda s: [str_to_int[c] for c in s]
    # decode = lambda l: "".join(int_to_str[i] for i in l)

    #print(f"Encoded string: {encode('Hello World!')}")
    #print(f"Decoded string: {decode(encode('Hello World!'))}")

    config.vocab_size = len(vocabulary)
    config.str_to_int = str_to_int
    config.int_to_str = int_to_str

    train_ds = CharTokenDataset(text = train_text, context_length = config.context_length, vocab_dict=str_to_int)
    val_ds = CharTokenDataset(text = val_text, context_length = config.context_length, vocab_dict= str_to_int)
    test_ds = CharTokenDataset(text = test_text, context_length = config.context_length, vocab_dict= str_to_int)

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}



# Build a boolean causal mask for decoder self-attention.
# Return:
# - type: `torch.Tensor`
# - dtype: `torch.bool`
# - shape: `[1, 1, context_length, context_length]`
# Value meaning:
# - `True` means the attention location is allowed
# - `False` means the attention location must be masked out
# Beginner-friendly mask idea:
# - token 0 can look at token 0
# - token 1 can look at tokens 0 and 1
# - token 2 can look at tokens 0, 1, and 2
# - no token is allowed to look to the right at future tokens
# Intuition:
# - this mask is the "no cheating" rule
# - future tokens contain the answer you are trying to predict
# - if the model could see them during training, it would learn the wrong lesson
def build_causal_mask(context_length: int) -> torch.Tensor:

    ones = torch.ones(size=(context_length, context_length), dtype=torch.bool)
    mask = torch.tril(ones)
    mask = mask.unsqueeze(dim=0).unsqueeze(dim=0)

    return mask


    #raise NotImplementedError("Return the boolean decoder causal mask with shape [1, 1, context_length, context_length].")


# Compute scaled dot-product attention for decoder multi-head attention.
# This helper follows the same core math as the related exercise.
# The new part is only the extra `num_heads` axis in the tensors.
# Input:
# - `query`: `[batch_size, num_heads, context_length, head_dim]`
# - `key`: `[batch_size, num_heads, context_length, head_dim]`
# - `value`: `[batch_size, num_heads, context_length, head_dim]`
# - `mask`: `[1, 1, context_length, context_length]`
# Return:
# - type: `torch.Tensor`
# - shape: `[batch_size, num_heads, context_length, head_dim]`
# Step-by-step path:
# 1. compare each query against each key with matrix multiply
# 2. divide by `sqrt(head_dim)`
# 3. apply the causal mask before softmax
# 4. apply softmax over the last dimension
# 5. use the attention weights to mix the value vectors
# Plain-language intuition:
# - each token sends out a `query`, which means:
#   - "what kind of information am I looking for right now?"
# - each token also has a `key`, which means:
#   - "what kind of information do I have?"
# - if a query matches a key well, that position gets more attention
# - the `value` is the actual information that gets mixed into the output
# - the final output is a weighted mix of value vectors
# Decoder-only rule:
# - a token may look at itself and earlier tokens
# - a token may not look at future tokens on the right
# - the mask enforces that "no cheating" rule before softmax
# Multi-head picture:
# - each head does this same matching-and-mixing process on a smaller slice of the features
# - different heads can learn to focus on different kinds of patterns
# Tiny example idea:
# - imagine one head is processing token position `2`
# - one attention row for that head might become `[0.70, 0.20, 0.10, 0.00]`
# - that means this token mostly uses information from token `0`
# - it uses a smaller amount from token `1`
# - it uses a tiny amount from token `2` itself
# - and it uses none from token `3`
# - that final `0.00` fits the decoder rule because token `3` is in the future
# Why this helper is a good teaching split:
# - it keeps the attention math separate from the multi-head reshape logic
# - that makes the related exercise easier to compare with the related exercise
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    # q, k and v are all of shape (32, 4, 128, 32)

    # The head dimension is the feature size that each attention head gets after you split 
    # the full embedding across heads. In this exercise, the full token representation has 
    # size `embed_dim`, and multi-head attention splits that evenly across `num_heads`, so 
    # the correct formula is `head_dim = embed_dim // num_heads`. For example, 
    # if `embed_dim = 128` and `num_heads = 4`, then each head gets `32` features. That means 
    # the full token still has 128 features overall, but attention is done as 4 smaller parallel 
    # attention jobs, each working on 32 features. This only works cleanly if `embed_dim` divides 
    # evenly by `num_heads`, which is why `MaskedSelfAttention.__init__` should check `embed_dim % num_heads == 0`.

    # The intuition is that one token may need to look for different kinds of useful context at the same time, 
    # and multiple heads give the model several separate chances to do that. One head might focus on the very 
    # recent token just to the left, another might notice repeated names or punctuation, and another might 
    # track a longer pattern across the sequence. Instead of forcing one big attention calculation to do every 
    # job at once, multi-head attention splits the work into several smaller attention views running in parallel.
    #  After that, their outputs are joined back together, so each token gets a richer combined summary of the context.

    batch_size, num_heads, context_length, head_dim = key.shape

    # transpose dim2 and dim2 so the matrix multiplication is valid.
    scores = torch.matmul(query, key.transpose(dim0=2, dim1=3)) # (32, 4, 128, 32) @ (32, 4, 128, 32) = (32, 4, 128, 128)
    scaled = scores / sqrt(head_dim)

    scaled = scaled.masked_fill(~mask, float("-inf"))

    # Softmax goes on the last dimension because the last dimension is the list of tokens that the current 
    # token can look at. After the score step, the shape is [batch_size, num_heads, context_length, context_length]. 
    # You can read that as: for each example, for each head, and for each token position, there is one row of 
    # scores over all possible source positions. That row answers the question, “which tokens should I pay 
    # attention to?” Softmax must be applied across that row so the scores turn into attention weights that 
    # behave like proportions and add up to 1.0. In plain words, the current token is choosing how to spread 
    # its attention across the tokens it is allowed to use.

    weights = scaled.softmax(dim=-1)            # (32, 4, 128, 128)

    output = torch.matmul(weights, value)       # (32, 4, 128, 128) @ (32, 4, 128, 32) = (32, 4, 128, 32)

    return output # (32, 4, 128, 32)

    #raise NotImplementedError("Return the masked scaled dot-product attention output for multi-head decoder attention.")


class MaskedSelfAttention(nn.Module):
    """Masked self-attention stub for the tiny decoder."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Beginner-friendly build order:
        # 1. check that `embed_dim` can be split across `num_heads`
        #    - this means each head gets the same number of features
        #    - store that smaller size as something like `head_dim`
        # 2. make linear layers for Q, K, and V
        #    - each one should map `embed_dim -> embed_dim`
        #    - you are still starting from the full token representation here
        # 3. make one output projection layer
        #    - after all heads are joined back together, this layer mixes them
        #
        # Head idea:
        # - if `embed_dim = 128` and `num_heads = 4`
        # - each head gets `32` features
        #
        # Intuition:
        # - multi-head attention means doing several smaller attention calculations in parallel
        # - each head gets a smaller slice of the features
        # - this gives the model a chance to look for different kinds of patterns at the same time
        #
        # Connection to EXERCISE 22:
        # - in the related exercise, you did one attention calculation over the whole feature size
        # - here, keep that same math in `scaled_dot_product_attention(...)`
        # - this class should handle the multi-head setup around that helper
        # - after the helper runs, join the heads back together

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim cannot be split evenly across num_heads. They should be exactly divisible.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.key = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.value = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        #raise NotImplementedError("Build the Q, K, V projections and the output projection for masked multi-head attention.")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return masked self-attention output.

        Input:
        - `x`: tensor with shape `[batch_size, context_length, embed_dim]`
        - `mask`: boolean tensor with shape `[1, 1, context_length, context_length]`
          - `True` means this attention location is allowed
          - `False` means this attention location must be blocked

        Return:
        - type: `torch.Tensor`
        - shape: `[batch_size, context_length, embed_dim]`

        Step-by-step path:
        1. project `x` into Q, K, and V
        2. reshape so there is a separate head dimension
        3. call the scaled dot-product attention helper on each head
        4. combine the head outputs back together
        5. run the output projection

        Why separate heads can help:
        - one head might focus on very nearby characters
        - another head might focus on repeated names or punctuation patterns
        - the model can learn several views of the same context

        Plain build path with shapes:
        1. start with `x` of shape `[B, T, C]`
           - `B` = batch size
           - `T` = context length
           - `C` = embed dim
        2. project into `q`, `k`, and `v`
           - each should still have shape `[B, T, C]`
        3. split each one into heads
           - first reshape to `[B, T, num_heads, head_dim]`
           - then move the head axis forward to get `[B, num_heads, T, head_dim]`
        4. call `scaled_dot_product_attention(...)`
           - pass in the per-head `q`, `k`, `v`, and the mask
           - let that helper do the score math
           - that includes:
             - `q @ k.transpose(-2, -1)`
             - divide by `sqrt(head_dim)`
             - apply the mask before softmax
             - apply softmax
             - mix the values
        5. the helper should return the per-head output
           - result shape should be `[B, num_heads, T, head_dim]`
        6. join the heads back together
           - move back to `[B, T, num_heads, head_dim]`
           - then reshape to `[B, T, C]`
        7. run the output projection

        Simple mental picture:
        - the helper does the actual attention math
        - this `forward` method does the multi-head setup around that math
        - at the end, the head outputs are stitched back into one full token representation

        Common mistakes to avoid:
        - forgetting to split `q`, `k`, and `v` into heads before attention
        - using the wrong transpose when building attention scores
        - applying softmax on the wrong dimension
        - applying the mask after softmax instead of before softmax
        - forgetting that `False` means "block this position"
        - forgetting to join the heads back into `[B, T, C]` before the output projection
        - forgetting that the helper owns the score math and masking
        - trying to do the multi-head reshaping and the attention math all in one big block
        """

        # x shape is (32, 128, 128),  mask shape is (1, 1, 128, 128)

        query = self.query(x)       # (32, 128, 128)
        key = self.key(x)           # (32, 128, 128)
        value = self.value(x)       # (32, 128, 128)

        batch_size, context_length, embed_dim = query.shape

        # scaled_dot_product_attention expects inputs of shape (32, 4, 128, 32)
        # because each embed dim will get split into self.num_heads
        # However, we cant just go from (32, 128, 128) to (32, 4, 128, 32) directly.
        # First reshape to (32, 128, 4, 32), reshape works this way
        # Then permute to get (32, 4, 128, 128)
        # reshape does not really know what a dimension means. It does not know which one 
        # is batch, token position, or features. It only sees one long line of numbers and 
        # then packs that same line into the new shape you ask for. So it does not “choose” 
        # which dimension to divide up. You choose that by giving it a shape that keeps the 
        # meaning in the right order. For example, going from [B, T, C] to [B, T, num_heads, head_dim] 
        # works because you are keeping B and T the same, and only splitting the last part C into two 
        # smaller parts. In simple words, reshape just repacks numbers in order. It follows your shape 
        # exactly, even if that shape mixes things up
        query = torch.reshape(query, shape=(batch_size, context_length, self.num_heads, self.head_dim))
        key = torch.reshape(key, shape=(batch_size, context_length, self.num_heads, self.head_dim))
        value = torch.reshape(value, shape=(batch_size, context_length, self.num_heads, self.head_dim))

        query = query.permute(dims = (0, 2, 1, 3))
        key = key.permute(dims = (0, 2, 1, 3))
        value = value.permute(dims = (0, 2, 1, 3))

        output = scaled_dot_product_attention(query, key, value, mask)  # (32, 4, 128, 32)
        # Don't forget to permute back so reshape can work in the correct order.
        output = output.permute(dims = (0, 2, 1, 3))

        output = torch.reshape(output, shape=(batch_size, context_length, embed_dim))  # (32, 128, 128)

        proj = self.projection(output)  # (32, 128, 128)

        return proj        


        #raise NotImplementedError("Apply masked multi-head self-attention and return the output tensor.")


class MiniGPTBlock(nn.Module):
    """One decoder block stub for the tiny GPT."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        # Beginner-friendly block pieces:
        # - layer norm
        # - masked self-attention
        # - residual connection
        # - second layer norm
        # - MLP
        # - second residual connection
        #
        # Simple MLP path:
        # - `Linear(embed_dim, mlp_dim)`
        # - activation like `GELU` or `ReLU`
        # - `Linear(mlp_dim, embed_dim)`
        #
        # Intuition:
        # - attention mixes information across positions
        # - the MLP then transforms each position's features on its own
        # - residual connections help the block change the features without destroying useful old information
        #         

        self.layer_norm1 = nn.LayerNorm(normalized_shape=config.embed_dim)

        self.masked_self_attention = MaskedSelfAttention(embed_dim = config.embed_dim, num_heads = config.num_heads)

        self.layer_norm2 = nn.LayerNorm(normalized_shape=config.embed_dim)

        self.mlp = nn.Sequential(nn.Linear(in_features=config.embed_dim, out_features=config.mlp_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=config.mlp_dim, out_features=config.embed_dim))

        #raise NotImplementedError("Build one decoder block with masked self-attention, MLP, layer norms, and residual connections.")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the updated hidden states with shape `[batch_size, context_length, embed_dim]`.

        Step-by-step path:
        1. keep the current `x` as the residual path
        2. normalize that `x`, then run masked self-attention on the normalized version
        3. add the attention result back to the original unnormalized `x`
        4. keep that updated tensor as the next residual path
        5. normalize the updated tensor, then run the MLP on the normalized version
        6. add the MLP result back to the updated unnormalized tensor

        Plain-language picture:
        - first the token gathers useful context from earlier tokens
        - then it processes that new information with the MLP
        - each residual add says "keep the old signal, then add a useful update"
        """

        # x shape is (32, 128, 128),  mask shape is (1, 1, 128, 128)

        residual = x
        normalized = self.layer_norm1(x)
        attention_output = self.masked_self_attention(normalized, mask)         # (32, 128, 128)
        x = residual + attention_output

        residual = x
        normalized = self.layer_norm2(x)
        mlp_output = self.mlp(normalized)                                       # (32, 128, 128)
        x = residual + mlp_output

        # So the idea becomes easier to read if each stage says:
        # keep residual
        # normalize residual
        # run sub-layer
        # add result back

        return x

        
        #raise NotImplementedError("Run one decoder block and return the updated hidden states.")


class MiniGPT(nn.Module):
    """Tiny decoder-only GPT stub."""

    def __init__(self, vocab_size: int, config: MiniGPTConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config
        # Beginner-friendly build order:
        # 1. token embedding table
        # 2. positional embedding table
        # 3. a stack of decoder blocks
        # 4. final layer norm
        # 5. linear LM head to vocabulary size
        #
        # Intuition:
        # - embeddings say what the token is
        # - positional embeddings say where it is
        # - decoder blocks build context-aware features
        # - the LM head turns those features into next-token scores

        self.vocab_size = vocab_size
        self.embed_dim = config.embed_dim
        self.context_length = config.context_length

        self.causal_mask = build_causal_mask(context_length = config.context_length)

        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.embed_dim)
        self.positional_embeddings = nn.Embedding(num_embeddings=config.context_length, embedding_dim=config.embed_dim)

        self.decoder_blocks = nn.ModuleList([MiniGPTBlock(config) for _ in range(config.num_layers)])

        self.layer_norm = nn.LayerNorm(normalized_shape=config.embed_dim)

        self.classification_head = nn.Linear(in_features=config.embed_dim, out_features=vocab_size)

        #raise NotImplementedError("Build the token embeddings, positional embeddings, decoder blocks, final layer norm, and LM head.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return raw next-token logits with shape `[batch_size, context_length, vocab_size]`.

        Step-by-step tensor path:
        1. token ids -> token embeddings
        2. add positional embeddings
        3. build or reuse the causal mask
        4. run the decoder blocks
        5. run the final layer norm
        6. run the LM head

        What the model is really doing:
        - it builds one feature vector per position
        - each block makes that feature vector smarter by using left-side context
        - the last layer turns that feature vector into a guess about the next token
        """

        # input_ids is shape (32, 128)  i.e. 128 tokens per sample

        self.causal_mask = self.causal_mask.to(input_ids.device)

        # `current_sequence_length` is the length of the input the model got this time.
        # In training, that is usually the full fixed window size of config.context_length i.e. num_tokens per sample
        # In generation/inference, it can be shorter at the start because you may only have a short prompt.
        # So we use the current input length here instead of always assuming the full context length.

        batch_size, current_sequence_length = input_ids.shape  # 32, 128        current_sequence_length could be different than 128

        token_embeddings = self.token_embeddings(input_ids)                     # (32, 128, 128)

        positions = torch.arange(start=0, end=current_sequence_length, step=1, device=input_ids.device)      # (128) if current_sequence_length=128
        positions = positions.reshape(shape=(1, current_sequence_length))           # (1, 128)
        positions = positions.expand(size = (batch_size, current_sequence_length))  # (32, 128)
        pos_embeddings = self.positional_embeddings(positions)                  # (32, 128, 128)

        embeddings = token_embeddings + pos_embeddings                          # (32, 128, 128)

        mask = self.causal_mask[:, :, : current_sequence_length, : current_sequence_length]

        features = embeddings
        for block in self.decoder_blocks:
            features = block(features, mask)

        features = self.layer_norm(features)                                    # (32, 128, 128)

        logits = self.classification_head(features)                             # (32, 128, 65)

        return logits



    #raise NotImplementedError("Return train, val, and test dataloaders for next-token prediction.")


# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.CrossEntropyLoss`.
def build_training_objects(config: MiniGPTConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    model = MiniGPT(vocab_size = config.vocab_size, config=config)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr = config.learning_rate, params= model.parameters())

    return model, loss_fn, optimizer
    #raise NotImplementedError("Return the tiny GPT model, CrossEntropyLoss, and optimizer.")


# Run one full training epoch.
# Return a dict with exactly these keys:
# - `loss`
# - `perplexity`
# Return type:
# - `dict[str, float]`
# Both values must be Python floats.
# Metric meaning:
# - `loss` is the mean train loss over the whole training epoch
# - `perplexity` is `exp(loss)` for that full training epoch
# Important:
# - move the model, `input_ids`, and `target_ids` to the same device here
# - a simple loss path is to flatten:
#   - logits from `[batch_size, context_length, vocab_size]` to `[batch_size * context_length, vocab_size]`
#   - targets from `[batch_size, context_length]` to `[batch_size * context_length]`
# - do not return one value per batch
# - return one final epoch summary dict
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:

    model.train()

    losses = []    

    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)       # output shape is (32, 128, 65)

        flatten_logits = torch.flatten(logits, start_dim = 0, end_dim=1)    # output shape is (4096, 65)
        flatten_targets = torch.flatten(target_ids)

        loss = loss_fn(flatten_logits, flatten_targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())        

    mean_loss = torch.tensor(losses).mean()
    mean_perp = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": mean_perp.item()}

    #raise NotImplementedError("Return the mean train loss and train perplexity for one full epoch.")


# Run one full evaluation pass.
# Return a dict with exactly these keys:
# - `loss`
# - `perplexity`
# Return type:
# - `dict[str, float]`
# Both values must be Python floats.
# If you pass the validation loader:
# - `loss` means mean validation loss over the whole validation loader
# - `perplexity` means `exp(loss)` for the whole validation loader
# If you pass the test loader:
# - `loss` means mean test loss over the whole test loader
# - `perplexity` means `exp(loss)` for the whole test loader
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    model.eval()

    losses = []    

    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)       # output shape is (32, 128, 65)

        flatten_logits = torch.flatten(logits, start_dim = 0, end_dim=1)    # output shape is (4096, 65)
        flatten_targets = torch.flatten(target_ids)

        loss = loss_fn(flatten_logits, flatten_targets)

        losses.append(loss.item())        

    mean_loss = torch.tensor(losses).mean()
    mean_perp = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": mean_perp.item()}

    #raise NotImplementedError("Return the mean loss and perplexity for the loader you evaluate.")


# Generate a short text sample from a finished decoder-only model.
# Return:
# - type: `str`
# - meaning: one generated character string
# Simple default path:
# - feed the current context into the model
# - take the next token from the last time step
# - append it
# - repeat until you reach `config.sample_length`
# In plain words:
# - run the model on the text you have so far
# - look only at the last position because that is the next-token prediction
# - pick one next token id
# - turn that id back into a character
# - add it to the text and repeat
# Why this is still autoregressive at generation time:
# - first you have a short starting context
# - the model predicts one new token from that context
# - now that new token becomes part of the context
# - then the model predicts the next token after that
# - so the sequence grows one step at a time from left to right
# Small temporary step that is allowed while you debug:
# - you may return `start_text` unchanged first
# - then replace that with real generation once the rest of the file works
def generate_text(model: nn.Module, start_text: str, config: MiniGPTConfig) -> str:
    # Why only the last position matters during generation:
    # - earlier positions predict tokens you already have
    # - the last position is the one that predicts the new token you still need

    if not start_text:
        start_text = "To be"
    
    encode = lambda s: [config.str_to_int[c] for c in s]
    decode = lambda l: "".join(config.int_to_str[i] for i in l)

    context_text = start_text[-config.context_length:]

    # `start_text[-config.context_length:]` takes only the last `context_length` characters 
    # from `start_text`. This is useful during generation because the model was trained to use 
    # a limited text window, not the full text history forever. If `start_text` is longer 
    # than `context_length`, this keeps the most recent characters, which are usually the most 
    # useful for predicting the next character. If `start_text` is shorter than `context_length`, 
    # Python just returns the whole string, so it still works. For example, 
    # if `context_length` is `4` and `start_text` is `"hello"`, the model sees `"ello"`. 
    # If `start_text` is `"hi"`, the model sees `"hi"`.
    # Also In Python, slicing is forgiving. If you ask for a start position that is before 
    # the beginning of the string, Python just starts at the beginning.
    # Example:
    # start_text = "hi"
    # context_length = 4
    # start_text[-context_length:]
    # This means:
    # start_text[-4:]
    # start 4 characters from the end, then go to the end

    # model.parameters() gives an iterator next(...) gets the first parameter, 
    # and .device tells you where the model is, like CPU or GPU.
    model_device = next(model.parameters()).device

    character_ids = torch.tensor(encode(context_text), dtype=torch.long).to(model_device)  # shape = (5)
    character_ids = character_ids.unsqueeze(dim=0)                                              # shape = (1, 5)

    generated = []
    for i in range(config.sample_length):
        logits = model(character_ids)               # shape is (1, 128, 65)
        last_vec = logits[:, -1, :]                 # shape is (65)
        next_char = torch.argmax(last_vec, dim=-1).long()

        next_char = next_char.to(model_device)

        generated.append(next_char.item())

        next_char = next_char.reshape(1, 1)
        character_ids = torch.concat([character_ids, next_char], dim=1)
        character_ids = character_ids[:, -config.context_length:]

    decoded_text = start_text + decode(generated)

    return decoded_text

    #raise NotImplementedError("Generate and return one short text sample from the tiny GPT model.")


# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `val_loss`
# - `val_perplexity`
# Return type:
# - `dict[str, float]`
# This is the final epoch summary, not a history list.
def run_training(config: MiniGPTConfig) -> dict[str, float]:
    torch.manual_seed(config.random_state)

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    for i in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i+1} -> loss: {train_dict['loss']}, perp: {train_dict['perplexity']}, val_loss: {val_dict['loss']}, val_perp: {val_dict['perplexity']}")

    test_dict = evaluate(model, loaders["test"], loss_fn)
    print(f"Test results -> loss: {test_dict['loss']}, perp: {test_dict['perplexity']}")

    print(f"\n GENERATED: \n {generate_text(model=model, start_text = 'To be', config=config)}")

    return {"train_loss": train_dict["loss"], 
            "val_loss": val_dict["loss"], "val_perplexity": val_dict["perplexity"]}

    #raise NotImplementedError("Run the full mini GPT training flow and return the final train and val metrics.")



def check_metrics(metrics: dict[str, float], target: float) -> bool:
    """Return True when validation loss is at or below the target."""
    return float(metrics.get("val_loss", float("inf"))) <= float(target)



def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
        print(f"[DONE] {task_name}")
        return result, True
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False



def main() -> None:
    """Run the tiny GPT scaffold one step at a time."""
    config = MiniGPTConfig()
    set_seed(config.random_state)
    print("Running exercise 06 starter scaffold...")
    print("Main goal: build a tiny educational decoder-only GPT with causal masking.")
    print(f"Device note: keep the model and all token tensors on the same device. Using {device}.")

    completed_steps = 0
    total_steps = 16

    raw_text, ok = report_task_progress("load_raw_text", load_raw_text, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("split_text_corpus", split_text_corpus, raw_text)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_decoder_shapes", describe_decoder_shapes, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("build_causal_mask", build_causal_mask, config.context_length)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    head_dim = config.embed_dim // config.num_heads
    sample_query = torch.zeros(config.batch_size, config.num_heads, config.context_length, head_dim)
    sample_key = torch.zeros(config.batch_size, config.num_heads, config.context_length, head_dim)
    sample_value = torch.zeros(config.batch_size, config.num_heads, config.context_length, head_dim)
    sample_mask = torch.ones(1, 1, config.context_length, config.context_length, dtype=torch.bool)
    _, ok = report_task_progress(
        "scaled_dot_product_attention",
        scaled_dot_product_attention,
        sample_query,
        sample_key,
        sample_value,
        sample_mask,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    attention_module, ok = report_task_progress("MaskedSelfAttention.__init__", MaskedSelfAttention, config.embed_dim, config.num_heads)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_hidden = torch.zeros(config.batch_size, config.context_length, config.embed_dim)
    _, ok = report_task_progress("MaskedSelfAttention.forward", attention_module, sample_hidden, sample_mask)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    block, ok = report_task_progress("MiniGPTBlock.__init__", MiniGPTBlock, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("MiniGPT.__init__", MiniGPT, 65, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_input_ids = torch.zeros(config.batch_size, config.context_length, dtype=torch.long)
    _, ok = report_task_progress("MiniGPT.forward", model, sample_input_ids)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return    

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    training_model, loss_fn, optimizer = training_objects

    # _, ok = report_task_progress("train_one_epoch", train_one_epoch, training_model, loaders["train"], loss_fn, optimizer)
    # if ok:
    #     completed_steps += 1
    # else:
    #     print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    #     return

    _, ok = report_task_progress("evaluate", evaluate, training_model, loaders["val"], loss_fn)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("generate_text", generate_text, model, "To be", config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics, config.target_val_loss)
        print_pass_or_try_again(
            passed,
            "val_loss",
            float(metrics["val_loss"]),
            f"<= {config.target_val_loss:.2f}",
        )
        print(f"Validation perplexity summary: {float(metrics['val_perplexity']):.4f}")


if __name__ == "__main__":
    main()




