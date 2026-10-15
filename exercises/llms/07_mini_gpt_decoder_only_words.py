"""
============================================================
EXERCISE 07: MINI GPT DECODER ONLY WORDS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny educational decoder-only GPT-style model.

This part is the word-token version of the related exercise.

This is a mini teaching version.
It is not GPT-3 scale.
It is not meant to be large.
It is meant to help you build the main decoder pieces yourself.

The main model idea stays the same:
- token embeddings
- positional embeddings
- masked self-attention
- MLP block
- residual connections
- layer norm
- language-model head
- a simple generation loop

The main difference is the token type:
- the related exercise used one token per character
- this exercise uses one token per word

Plain-language meaning:
- `word token` means one token usually stands for one full word
- `masked self-attention` means each token can look back at earlier tokens, but not forward at future tokens
- `MLP` means a small feed-forward network made of linear layers and activations
- `residual connection` means adding the input of a block back to its output so learning is easier
- `layer norm` means rescaling values inside the model so training stays more stable
- `generation loop` means repeatedly predict one next token, append it, and then continue

Beginner intuition:
- this model learns by playing the game "guess the next word"
- at each position, it is only allowed to use the tokens to the left
- masked attention is the rule that stops it from cheating
- the decoder blocks slowly build better features for making that next guess

What "autoregressive next-token prediction" means in plain words:
- `next-token prediction` means:
  - look at the current tokens
  - guess the one token that should come next
- `autoregressive` means the model keeps extending from earlier tokens to later tokens
- so the full idea is:
  - use the tokens you already have
  - predict the next token
  - then repeat

Small concrete example:
- text: `the cat sat on the mat`
- one short training window could be:
  - input:  `the cat sat on`
  - target: `cat sat on the`
- position by position, the model is learning:
  - after `the`, guess `cat`
  - after `the cat`, guess `sat`
  - after `the cat sat`, guess `on`
  - after `the cat sat on`, guess `the`

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
- how word tokens change the data pipeline compared with character tokens

DATASET
Use Tiny Shakespeare or another tiny public text corpus.

Keep one clear default split rule:
- first `60%` of the raw text for train
- next `20%` for val
- final `20%` for test

Keep the text in order.
Do not shuffle the raw corpus before splitting it.

RECOMMENDED TOKENIZER
Start with a simple word tokenizer.

Recommended path:
- use `nltk.word_tokenize` if you want a beginner-friendly word tokenizer
- if you want fewer dependencies, a simple regex tokenizer is also fine
- keep this exercise word-level, not subword-level

HOW TO LOAD OR DOWNLOAD THE DATASET
Simple path:
- load the tiny shakespear text file
- split the text into word tokens
- build a word vocabulary from the training tokens only
- turn the tokenized corpus into integer ids
- create input and target windows for next-word prediction
- use the same `context_length` for every training batch

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
- `input_ids`: `[16, 16]`
- `target_ids`: `[16, 16]`

What those shapes mean:
- `input_ids: [16, 16]` means 16 examples in the batch and 16 token positions in each input window
- `target_ids: [16, 16]` means one next-token target for each input position

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

Beginner note on `exp(test_loss)`:
- the loss is measured in log space
- `exp(...)` moves that value back out of log space
- this gives a number called `perplexity`, which is easier to read than the raw log-based loss
- in simple words, perplexity tells you how confused the model is on average
- if perplexity is low, the model is more confident and usually making better next-token guesses
- if perplexity is high, the model is more unsure and is acting like there are many possible next tokens
- for example, if perplexity is close to `10`, you can think of that as the model behaving like it has about `10` reasonable next-token choices on average
- if perplexity is close to `2`, you can think of that as the model being much less confused, like it is choosing between about `2` strong options on average
- we use `exp(...)` here because `exp` is the opposite of the natural log used inside this kind of loss

Beginner note on `log`, `ln`, and `exp`:
- `log` means logarithm, which asks: "what power made this number?"
- `ln` means natural log, which is log with base `e`
- so `ln(x)` means: "what power of `e` gives me `x`?"
- `exp(x)` means: "raise `e` to the power `x`"
- that means `ln` and `exp` undo each other
- example:
  - `exp(2)` means `e^2`
  - `ln(e^2)` means `2`
- in deep learning, `log` often means the same thing as `ln`
- so when the loss uses natural logs, `exp(loss)` is what moves that value back out of log space

TARGET RESULT RANGE
For a tiny educational decoder on a tiny public corpus, expect training to be slower and the vocabulary to be much larger than the character-token version.

Also expect only rough text quality.
A tiny word-token model may first learn short phrases and local patterns before it learns longer structure.

COMMON MISTAKES
- forgetting causal masking
- using a mask with the wrong shape
- applying softmax before `CrossEntropyLoss`
- forgetting residual connections
- mixing up word tokens and raw strings
- building the vocabulary from the full corpus instead of train only
- generating text without feeding the newest token back into the model

STEP-BY-STEP TODO
1. Load the tiny text corpus.
2. Split it into contiguous `60 / 20 / 20` train, val, and test sections.
3. Tokenize each split into word tokens.
4. Build the word vocabulary.
5. Create next-token windows.
6. Build the causal mask.
7. Build masked self-attention.
8. Build one decoder block.
9. Stack several decoder blocks.
10. Train the model.
11. Evaluate on validation text.
12. Generate a short sample.

What each step is doing in simple words:
1. read the corpus
2. keep train, val, and test in text order
3. split the text into word tokens
4. build the word dictionary
5. make one-step-ahead training windows
6. create the no-looking-right mask
7. build the attention part that mixes earlier context
8. package attention and MLP into one decoder block
9. stack blocks so the model can learn richer patterns
10. train the model to guess next words
11. check whether it works on unseen text
12. use the model in a loop to write new text

STRETCH GOALS
- compare one block and two blocks
- compare different context lengths
- compare a simple regex tokenizer and `nltk.word_tokenize`
- add temperature sampling to generation
  - the current simple path can use `argmax`, which always picks the one token with the highest score
  - temperature sampling is different: it turns the model scores into chances, then randomly picks from those chances
  - `temperature` is a number that controls how bold or safe the random choice is
  - first divide the raw scores by `temperature`
  - then use `softmax` to turn those scores into chances that add up to `1.0`
  - then use `torch.multinomial` to randomly pick one token using those chances
  - example probabilities: `[0.70, 0.20, 0.10]`
  - index `0` has a `70%` chance, index `1` has a `20%` chance, and index `2` has a `10%` chance
  - a lower temperature, like `0.5`, makes the model mostly pick safer, more likely words
  - a higher temperature, like `1.5`, gives less likely words more chance, so the text can be more varied but also more messy
  - example: a token with `70%` chance is picked often, but not every time
  - example: after `the cat sat on the`, low temperature may almost always pick `mat`; high temperature might sometimes pick `chair`, `floor`, or a strange word
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
from typing import Callable
import sys
import re
from math import sqrt
from os import read

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


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

    data_path: str = str(BOOTCAMP_ROOT / "data" / "tiny_shakespeare.txt")
    context_length: int = 16
    embed_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256
    num_layers: int = 2
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 7
    random_state: int = 42
    target_val_loss: float = 4.5
    sample_length: int = 40

    vocab_size: int | None = None
    token_to_int: dict[str, int] | None = None
    int_to_token: dict[int, str] | None = None
    tokenizer_name: str = "nltk.word_tokenize"



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
            "This file uses word tokens, not character tokens.",
        ],
    }

        #raise NotImplementedError("Run the tiny decoder-only model and return raw logits.")


# Load the raw text corpus.
# Return:
# - type: `str`
# - meaning: the full text corpus as one Python string
def load_raw_text(config: MiniGPTConfig) -> str:
    """Return the full text corpus as one Python string."""

    with open(file=f"{BOOTCAMP_ROOT}/data/input.txt", mode="r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text

    #raise NotImplementedError("Return the full Tiny Shakespeare text as one Python string.")


# Split the raw text into train, validation, and test text sections.
# Return a dict with exactly these keys:
# - `train_text`
# - `val_text`
# - `test_text`
# Use a contiguous 60% train / 20% val / 20% test split.
def split_text_corpus(raw_text: str) -> dict[str, str]:
    """Return the raw text split into contiguous train, val, and test sections."""

    num_train, num_val = int(len(raw_text) * 0.6), int(len(raw_text) * 0.2)

    train_text = raw_text[:num_train]
    val_text = raw_text[num_train:num_train+num_val]
    test_text = raw_text[num_train+num_val:]

    return {"train_text": train_text, "val_text": val_text, "test_text": test_text}

    #raise NotImplementedError("Return the raw text split into contiguous train, val, and test sections.")


# Tiny fallback tokenizer helper.
# This is here so you have a simple word-level option if you do not want to depend on NLTK.
# It splits text into:
# - word pieces like `hello`
# - punctuation tokens like `,` or `.`
def basic_word_tokenize(text: str) -> list[str]:
    """Simple fallback tokenizer for word-level experiments."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)



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
    """Dataset stub for word-token next-token windows."""

    def __init__(self, tokens: list[str], context_length: int, vocab_dict: dict[str, int]) -> None:
        self.tokens = tokens
        self.num_tokens = len(tokens)
        self.context_length = context_length
        self.vocab_dict = vocab_dict
        self.unk_id = 1
        #raise NotImplementedError("Store tokenized text and any vocabulary objects the dataset needs.")

    def __len__(self) -> int:
        return self.num_tokens - self.context_length
        #raise NotImplementedError("Return the number of next-token windows available in the token list.")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        inputs = self.tokens[index: index + self.context_length]
        targets = self.tokens[index + 1: index + 1 + self.context_length]
        
        input_ids, target_ids = [], []
        for word in inputs:
            if word in self.vocab_dict:
                input_ids.append(self.vocab_dict[word])
            else:
                input_ids.append(self.unk_id)
        
        for word in targets:
            if word in self.vocab_dict:
                target_ids.append(self.vocab_dict[word])
            else:
                target_ids.append(self.unk_id)

        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "target_ids": torch.tensor(target_ids, dtype=torch.long)}



# Note on ReLU and GELU:
# The MLP needs an activation between its two Linear layers.
# ReLU is simple: it turns negative numbers into 0 and keeps positive numbers.
# GELU is smoother: it lets small negative numbers partly pass through.
# GPT-style models often use GELU because this smoother behavior can help training.
# ReLU is still okay for this small learning exercise, but GELU is closer to GPT-1/GPT-2.


@dataclass
        #raise NotImplementedError("Return one dict with input_ids and target_ids for a word-token window.")


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
# - `input_ids` is the current token window
# - `target_ids` is the same token window shifted one token ahead
# Intuition:
# - the model sees the current tokens and tries to say what comes next at every position
# - this one-step shift is what teaches autoregressive prediction
def load_dataloaders(config: MiniGPTConfig) -> dict[str, DataLoader]:
    """Return train, val, and test dataloaders for next-token prediction."""

    raw_text = load_raw_text(config)
    data_dict = split_text_corpus(raw_text)

    train_text = data_dict["train_text"]
    val_text = data_dict["val_text"]
    test_text = data_dict["test_text"]

    all_words = basic_word_tokenize(raw_text)    
    vocabulary = sorted(set(all_words))
    vocabulary.insert(0, "[PAD]")
    vocabulary.insert(1, "[UNK]")
    vocab_size = len(vocabulary)
    vocab_dict = {w:i for i, w in enumerate(vocabulary)}
    int_to_word = {i: w for w, i in vocab_dict.items()}

    config.vocab_size = vocab_size
    config.token_to_int = vocab_dict
    config.int_to_token = int_to_word

    train_ds = WordTokenDataset(tokens=basic_word_tokenize(train_text), context_length=config.context_length, vocab_dict=vocab_dict)
    val_ds = WordTokenDataset(tokens=basic_word_tokenize(val_text), context_length=config.context_length, vocab_dict=vocab_dict)
    test_ds = WordTokenDataset(tokens=basic_word_tokenize(test_text), context_length=config.context_length, vocab_dict=vocab_dict)

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
# - one head might focus on the word right before the current word
# - another head might focus on the start of the sentence
# - another head might focus on repeated names, punctuation, or quotation marks
# - another head might focus on longer-range patterns like who is speaking or what topic is being discussed
# - this does not mean you manually assign jobs to heads
# - the model learns these different focus patterns during training
# - after that, the head outputs are joined together, so the token gets several different views of the context
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
# - that makes this exercise easier to compare with the related exercise
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    batch_size, num_heads, context_length, head_dim = query.shape  # eg (16, 4, 16, 32)

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

    scores = torch.matmul(query, key.transpose(dim0=2, dim1=3)) # (16, 4, 16, 32) @ (16, 4, 32, 16) = (16, 4, 16, 16)
    scaled = scores / sqrt(head_dim)

    scaled = scaled.masked_fill(~mask, float("-inf")) # (16, 4, 16, 16)

    weights = scaled.softmax(dim=-1) # (16, 4, 16, 16)

    output = torch.matmul(weights, value) # (16, 4, 16, 16) @ (16, 4, 16, 32) = (16, 4, 16, 32)

    return output


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
        # Connection to EXERCISE 23:
        # - in the related exercise, you did one attention calculation over the whole feature size
        # - here, keep that same math in `scaled_dot_product_attention(...)`
        # - this class should handle the multi-head setup around that helper
        # - after the helper runs, join the heads back together

        if embed_dim % num_heads != 0:
            raise ValueError("Embedded dimension is not exactly divisible by num_heads.")

        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.key = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.value = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)

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
        - one head might focus on very nearby words
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
           - let that helper do the score math, scaling, masking, softmax, and value mixing
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
        """

        batch_size, context_length, embed_dim = x.shape # eg. 16, 16, 128

        #mask shape is (1, 1, 16, 16)

        query = self.query(x)   # (16, 16, 128)
        key = self.key(x)       # (16, 16, 128)
        value = self.value(x)   # (16, 16, 128)

        query = query.reshape(shape=(batch_size, context_length, self.num_heads, self.head_dim))    # (16, 16, 4, 32)
        key = key.reshape(shape=(batch_size, context_length, self.num_heads, self.head_dim))        # (16, 16, 4, 32)
        value = value.reshape(shape=(batch_size, context_length, self.num_heads, self.head_dim))    # (16, 16, 4, 32)

        query = torch.permute(query, (0, 2, 1, 3))      # (16, 4, 16, 32)
        key = torch.permute(key, (0, 2, 1, 3))          # (16, 4, 16, 32)
        value = torch.permute(value, (0, 2, 1, 3))      # (16, 4, 16, 32)

        output = scaled_dot_product_attention(query, key, value, mask)              # (16, 4, 16, 32)
        output = torch.permute(output, (0, 2, 1, 3))                                # (16, 16, 4, 32)
        output = output.reshape(shape=(batch_size, context_length, embed_dim))      # (16, 16, 128)

        proj = self.proj(output)        # (16, 16, 128)

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

        # In a transformer, each block keeps adding updates with residual connections. Without normalization, 
        # those values can slowly get too large, too small, or just uneven. When that happens, training gets 
        # shaky and the model becomes harder to optimize. LayerNorm fixes that by taking each token’s feature 
        # vector and rescaling it so the next part of the block sees cleaner, more predictable inputs.

        # attention works better when inputs are on a similar scale, 
        # gradients flow more smoothly during training and,
        # residual connections become safer, because each new update is less likely to blow up the old signal

        self.layer_norm1 = nn.LayerNorm(normalized_shape=config.embed_dim)

        self.masked_self_attention = MaskedSelfAttention(config.embed_dim, config.num_heads)

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

        residual = x
        x = self.layer_norm1(x)
        x = self.masked_self_attention(x, mask)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual

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

        self.causal_mask = build_causal_mask(config.context_length)

        self.token_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.embed_dim)

        self.positional_embed = nn.Embedding(num_embeddings=config.context_length, embedding_dim=config.embed_dim)

        self.decoder_blocks = nn.ModuleList([MiniGPTBlock(config) for i in range(config.num_layers)])

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

        # shape of input_ids = (16, 16)
        batch_size, sequence_length = input_ids.shape

        token_embeddings = self.token_embed(input_ids)              # (16, 16, 128)      # or depending on the actual sequence length

        positions = torch.arange(start=0, end=sequence_length, step=1, device=input_ids.device) # (16)   # or depending on the actual sequence length
        
        # We use `sequence_length` here because it is the length of the input we got right now.
        # That input is not always the full `context_length`.
        # During training, it is usually the full window size.
        # During generation/inference, it can be shorter at the start if the prompt is short.
        # So `context_length` is the maximum allowed length, but `sequence_length` is the current actual length.

        positions = positions.unsqueeze(dim=0)                                          # (1, 16)
        positions = positions.expand(size=(batch_size, sequence_length))                # (16, 16)

        positional_embeddings = self.positional_embed(positions)    # (16, 16, 128)      # or depending on the actual sequence length

        embeddings = token_embeddings + positional_embeddings       # (16, 16, 128)      # or depending on the actual sequence length

        self.causal_mask = self.causal_mask.to(input_ids.device)              # (1, 1, 16, 16) # or depending on the actual sequence length
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]

        features = embeddings
        for decoder_block in self.decoder_blocks:
            features = decoder_block(features, mask)    # (16, 16, 128) 

        features = self.layer_norm(features)            # (16, 16, 128) 

        logits = self.classification_head(features)     # (16, 16, 1000) 

        return logits


    #raise NotImplementedError("Return train, val, and test dataloaders for next-token prediction.")


# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.CrossEntropyLoss`.
def build_training_objects(config: MiniGPTConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return the tiny GPT model, CrossEntropyLoss, and optimizer."""

    model = MiniGPT(vocab_size = config.vocab_size, config=config)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=config.learning_rate, params = model.parameters())

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
    """Return the mean train loss and train perplexity for one full epoch."""

    model.train()

    losses = []
    
    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)

        batch_size, context_length, vocab_size = logits.shape

        # or you can use flatten:
        #loss = loss_fn(logits.flatten(start_dim=0, end_dim=1), target_ids.flatten())
        loss = loss_fn(logits.reshape(batch_size * context_length, vocab_size), target_ids.reshape(batch_size * context_length))

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean()
    perplexity = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": perplexity.item()}
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
# - `perplexity` means `exp(test_loss)` for the whole test loader
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:
    """Return the mean loss and perplexity for the loader you evaluate."""

    model.eval()

    losses = []

    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)

        batch_size, context_length, vocab_size = logits.shape
        loss = loss_fn(logits.reshape(batch_size * context_length, vocab_size), target_ids.reshape(batch_size * context_length))

        losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean()
    perplexity = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": perplexity.item()}
    #raise NotImplementedError("Return the mean loss and perplexity for the loader you evaluate.")


def generate_text(model: nn.Module, start_text: str, config: MiniGPTConfig) -> str:
    """Generate and return one short text sample from the tiny GPT model.

    Return:
    - type: `str`
    - meaning: one generated text string
    Simple default path:
    - feed the current context into the model
    - take the next token from the last time step
    - append it
    - repeat until you reach `config.sample_length`
    In plain words:
    - run the model on the text you have so far
    - look only at the last position because that is the next-token prediction
    - pick one next token id
    - turn that id back into a word or token string
    - add it to the text and repeat
    Why this is still autoregressive at generation time:
    - first you have a short starting context
    - the model predicts one new token from that context
    - now that new token becomes part of the context
    - then the model predicts the next token after that
    - so the sequence grows one step at a time from left to right
    """
    temperature = 0.8
    use_temperature = True

    if not start_text:
        start_text = "Hello world"

    start_tokens = basic_word_tokenize(start_text)
    start_tokens = start_tokens[-config.context_length:]

    encode = lambda st: [config.token_to_int.get(w, 1) for w in st]
    decode = lambda ids: " ".join(config.int_to_token[i] for i in ids)   
    # I added a space by join because in basic_word_tokenize, spaces are not kept as tokens.
    # Only words and punctuations are kept as tokens

    model_device = next(model.parameters()).device    
    
    context_text = torch.tensor(encode(start_tokens), dtype=torch.long, device=model_device)
    context_text = context_text.unsqueeze(dim=0)   # add a fake batch dimension. Model expects (batch_size, sequence_length)

    generated = []

    for i in range(config.sample_length):
        logits = model(context_text)                        # output shape: (1, sequence_length, vocab_size)
        # The last prediction is the one we want.
        last_token_vector = logits[:, -1, :]                # shape is (1, vocab_size)

        if use_temperature:
            probs = torch.softmax(last_token_vector/temperature, dim=-1)
            last_token_id = torch.multinomial(probs, num_samples=1)         # (1, 1)                 
        else:
            last_token_id = last_token_vector.argmax().long()      # shape is scalar  
            last_token_id = last_token_id.unsqueeze(dim=0).unsqueeze(dim=0)   # add that fake batch dimension so we can concat it to the context_text, shape is now (1,1)

        generated.append(last_token_id.squeeze().item())       
        context_text = torch.concat([context_text, last_token_id], dim=1)  #shape is now (1, current_len + 1)
        context_text = context_text[:, -config.context_length:]

    generated_text = start_text + " " + decode(generated)

    return generated_text
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
    """Run the full mini GPT training flow and return the final train and val metrics."""

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

    #raise NotImplementedError("Run the full mini GPT training flow and return the final train and val metrics.")


def check_metrics(metrics: dict[str, float], target: float) -> bool:
    """Return True when validation loss is at or below the target."""
    return float(metrics.get("val_loss", float("inf"))) <= float(target)


def report_task_progress(task_name: str, task_fn: Callable, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
        print(f"[DONE] {task_name}")
        return result, True
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False


def main() -> None:
    set_seed(42)
    config = MiniGPTConfig()

    print("Running exercise 07 starter scaffold...")
    print("Main goal: build a tiny educational decoder-only GPT with word tokens.")
    print(f"Recommended tokenizer: {config.tokenizer_name}")
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

    model, ok = report_task_progress("MiniGPT.__init__", MiniGPT, 1000, config)
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

    # _, ok = report_task_progress("generate_text", generate_text, model, "to be or not", config)
    # if ok:
    #     completed_steps += 1
    # else:
    #     print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    #     return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics, config.target_val_loss)
        print_pass_or_try_again(
            passed,
            "Validation loss reached the target range for the word-token mini GPT.",
            "Keep going. Word-token models often need more care in the tokenizer, vocabulary, and data pipeline.",
            f"<= {config.target_val_loss:.2f}"
        )


if __name__ == "__main__":
    main()
