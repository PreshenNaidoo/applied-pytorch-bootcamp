"""
============================================================
EXERCISE 10: KV CACHE FAST GENERATION
============================================================

WHAT YOU ARE BUILDING
You are building a tiny self-attention module that can use a key-value cache.

Plain-language meaning:
- K means key
- V means value
- during generation, old tokens do not change
- a KV cache stores old keys and values
- then the model only computes keys and values for the new token

WHY THIS EXERCISE EXISTS
Decoder-only language models generate text one token at a time.
Without a cache, the model keeps recomputing work for old tokens.
With a cache, generation can be much faster.
This exercise teaches:
- what is stored in a KV cache
- how cache shapes grow during generation
- why cached generation avoids repeated work
- how to compare no-cache and cached generation

HOW THIS FITS INTO THE BOOTCAMP
After building decoder-only attention and tiny GPT-style generation,
this file teaches one of the main practical speed tricks used during inference.

What each step is doing in simple words:
1. build one tiny masked attention module
2. run the slow full-sequence path once
3. create empty key and value memory
4. generate one token at a time
5. save the new key and value instead of rebuilding old ones
6. watch the cache grow
7. compare repeated work with cached reuse
8. explain why this matters for generation speed

Beginner intuition:
- generation adds one token at a time
- old tokens do not change, so their keys and values do not need to be rebuilt each step
- the cache is just a saved memory of those old keys and values

Small example:
- if you already generated 20 tokens
- token 21 only needs one new key and one new value
- without a cache, the model would rebuild all 20 old ones again too

DATASET
No downloaded dataset is needed.
Use a tiny toy token or hidden-state sequence created inside the file.

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- make a short fake sequence
- run full-sequence attention once
- then run one token at a time with a cache
- compare how the cache grows

INPUTS AND OUTPUTS
Input hidden states:
- full sequence path: `[batch_size, sequence_length, model_dim]`
- one-token path: `[batch_size, 1, model_dim]`
Cache dictionary:
- `key`
  - shape: `[batch_size, num_heads, cached_length, head_dim]`
- `value`
  - shape: `[batch_size, num_heads, cached_length, head_dim]`

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use one tiny masked self-attention module.
The point is to understand cache reuse, not to train a full model.

LOSS FUNCTION
No training loss is required for the main cache exercise.

METRICS TO TRACK
Track at least:
- cache length after each step
- one no-cache summary
- one cached summary
- one note about why cached generation is faster

COMMON MISTAKES
- caching queries instead of keys and values
- forgetting to append the new key and value
- letting cached tokens attend to future tokens
- mixing up `[batch, seq, dim]` and `[batch, heads, seq, head_dim]`

STEP-BY-STEP TODO
1. Build a tiny masked self-attention module.
2. Run attention on a full sequence without a cache.
3. Create an empty KV cache.
4. Feed one token at a time.
5. Append new keys and values to the cache.
6. Check that cache length grows by one each step.
7. Compare full-sequence work with cached one-token work.
8. Write a short note about why this speeds up generation.

DEFINITION OF DONE
You are done when:
- the cache stores keys and values
- cache length grows one token at a time
- cached generation returns one output token per step
- you can explain why old keys and values can be reused
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
import torch.nn as nn

from math import sqrt

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again

# Store settings for the KV-cache exercise.
# Return:
# - a config object with the toy hidden-state and attention shapes
@dataclass
class KVCacheConfig:
    """Settings for the KV cache exercise."""
    batch_size: int = 2
    sequence_length: int = 12
    model_dim: int = 32
    num_heads: int = 4
    random_state: int = 42

# Describe expected cache shapes.
# Return:
# - dictionary with one-token input shape, full input shape, and final cache shapes
def describe_cache_shapes(config: KVCacheConfig) -> dict[str, object]:
    """Return the expected KV cache shapes."""
    head_dim = config.model_dim // config.num_heads
    return {
        "one_token_input": (config.batch_size, 1, config.model_dim),
        "full_sequence_input": (config.batch_size, config.sequence_length, config.model_dim),
        "cache_key_shape_after_full_generation": (config.batch_size, config.num_heads, config.sequence_length, head_dim),
        "cache_value_shape_after_full_generation": (config.batch_size, config.num_heads, config.sequence_length, head_dim),
    }

# Return toy hidden states for generation.
# Shape:
# - `[batch_size, sequence_length, model_dim]`
# These stand in for token representations during decoding.
def make_toy_hidden_states(config: KVCacheConfig) -> torch.Tensor:
    """Return toy hidden states for generation."""
    # Return random hidden states with shape:
    # [batch_size, sequence_length, model_dim]
    #
    # These are not token IDs.
    # They stand in for the hidden vectors that a transformer block would receive.
    #
    # Use the exercise seed so the toy values are repeatable.
    #
    # Concrete shape with the default config:
    # [2, 12, 32]

    gen = torch.Generator().manual_seed(config.random_state)
    hidden_state = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.model_dim,
        generator=gen,
    )

    return hidden_state
    #raise NotImplementedError("Return hidden states with shape [batch, sequence_length, model_dim].")

# Build the decoder causal mask.
# This is the same "no looking ahead" rule you saw in the related exercise.
def build_causal_mask(sequence_length: int) -> torch.Tensor:
    """Return a boolean causal mask."""
    # A causal mask blocks future tokens.
    #
    # Use True for allowed positions.
    # Use False for blocked future positions.
    #
    # Example when sequence_length = 4:
    #
    # [
    #   [ True, False, False, False],
    #   [ True,  True, False, False],
    #   [ True,  True,  True, False],
    #   [ True,  True,  True,  True],
    # ]
    #
    # Row meaning:
    # - row 0: token 0 can only see token 0
    # - row 1: token 1 can see tokens 0 and 1
    # - row 2: token 2 can see tokens 0, 1, and 2
    #
    # Hint:
    # Think about which part of a square matrix should be True:
    # - diagonal should be True
    # - below diagonal should be True
    # - above diagonal should be False

    ones = torch.ones(size=(sequence_length, sequence_length), dtype=bool)
    causal_mask = torch.tril(ones)
    causal_mask = causal_mask.unsqueeze(dim=0).unsqueeze(dim=0)
    
    return causal_mask
    #raise NotImplementedError("Return a mask that blocks future tokens.")

# Create the starting cache value.
# Return:
# - `None` for the simplest beginner path
# - or an empty key/value dictionary if you choose that style
def init_empty_cache() -> dict[str, torch.Tensor] | None:
    """Return an empty cache value.
    Simple default:
    - return `None`
    - create real key/value tensors after the first token
    """
    # Starting with `None` is often the cleanest beginner path.
    # The first generation step creates the cache,
    # and later steps keep extending it.
    #
    # Why `None` is simpler than empty tensors:
    # Before the first token, you do not know the cached key/value tensors yet.
    # On the first step, create a dictionary that stores the first real key and value.
    #
    # After that, each step appends to those tensors.
    return None
    #raise NotImplementedError("Return the starting cache state.")

# Beginner hints for the attention module:
# - generation adds one new token at a time, so old tokens do not need their keys and values recomputed
# - the cache grows with sequence length because each new token adds one more key and one more value
# - the main idea is reuse: store old work once, then keep appending new work
# - first understand the shapes above, then build the attention module
# - watch the cache shapes closely; they are the easiest sanity check in this file
class CachedSelfAttention(nn.Module):
    """Tiny self-attention module with optional KV cache."""

    # Build the attention projections.
    # Expected modules:
    # - query projection
    # - key projection
    # - value projection
    # - output projection
    def __init__(self, config: KVCacheConfig) -> None:
        super().__init__()
        self.config = config
        # Build four linear layers:
        # - query projection: turns hidden states into queries
        # - key projection: turns hidden states into keys
        # - value projection: turns hidden states into values
        # - output projection: mixes attention-head outputs back into model_dim
        #
        # All four layers should map:
        # [model_dim] -> [model_dim]
        #
        # Concrete shape example:
        # if model_dim = 32, each projection takes 32 numbers in
        # and returns 32 numbers out.
        #
        # Why Q, K, and V are separate:
        # The same token hidden state can play three different roles:
        # - query: what this token is looking for
        # - key: what this token offers for matching
        # - value: what information this token passes along if selected
        #
        # Build one linear layer for each projection.
        # Each projection should take a vector of length `model_dim`
        # and return another vector of length `model_dim`.
        #
        # Do not share one layer for Q, K, and V.
        # They need separate learned weights because they play different roles.
        #
        # Also check:
        # config.model_dim must divide cleanly by config.num_heads.
        # head_dim = model_dim // num_heads

        if config.model_dim % config.num_heads != 0:
            raise ValueError("Model dimension should be exactly divisible by num_heads.")

        self.head_dim = config.model_dim // config.num_heads
        self.num_heads = config.num_heads
        
        self.query = nn.Linear(in_features=config.model_dim, out_features=config.model_dim)
        self.key = nn.Linear(in_features=config.model_dim, out_features=config.model_dim)
        self.value = nn.Linear(in_features=config.model_dim, out_features=config.model_dim)
        self.proj = nn.Linear(in_features=config.model_dim, out_features=config.model_dim)       
        #raise NotImplementedError("Build Q, K, V, and output projections.")

    # Run attention with or without a cache.
    # Input:
    # - `x`: `[batch_size, sequence_length, model_dim]` or `[batch_size, 1, model_dim]`
    # - `cache`: `None` or a dictionary with old keys and values
    # Return:
    # - attention output
    # - updated cache containing keys and values
    def forward(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return attention output and the updated KV cache."""
        # Important picture before the steps:
        #
        # `cache is None` can happen in two different situations.
        #
        # Situation 1: full no-cache run.
        # - the caller passes the whole sequence
        # - x.shape = [2, 12, 32]
        # - cache = None
        # - this means "process all 12 tokens at once"
        #
        # Situation 2: first step of cached generation.
        # - the caller passes only the first token
        # - x.shape = [2, 1, 32]
        # - cache = None
        # - this means "there are no old keys and values yet"
        #
        # Later cached generation steps look like this:
        # - the caller passes only the newest token
        # - x.shape = [2, 1, 32]
        # - cache has old keys and values
        #
        # So do not read `cache is None` as "x must be the full sequence".
        # The shape of `x` depends on how the caller uses this function.
        #
        # Simple table:
        # - full no-cache run:        x = [2, 12, 32], cache = None
        # - cached generation step 1: x = [2,  1, 32], cache = None
        # - cached generation step 2: x = [2,  1, 32], cache has old K/V
        #
        # This function has two modes:
        #
        # 1. No cache:
        #    `x` is a full sequence.
        #    shape: [batch_size, sequence_length, model_dim]
        #    The model computes Q, K, and V for every token.
        #
        # 2. With cache:
        #    `x` is usually one new token.
        #    shape: [batch_size, 1, model_dim]
        #    The model computes Q, K, and V only for this new token.
        #    Then it appends the new K and V to the old cached K and V.
        #
        # Step-by-step plan:
        #
        # Step 1: Read the input shape.
        # You need batch size, sequence length, and model width from `x`.
        # You also need the per-head width:
        # head_dim is model_dim divided by num_heads.
        #
        # Step 2: Project `x` into Q, K, and V.
        # Use the query, key, and value projection layers you built in __init__.
        #
        # Step 3: Split heads.
        # Change each tensor from:
        # [batch_size, sequence_length, model_dim]
        # to:
        # [batch_size, num_heads, sequence_length, head_dim]
        #
        # Common idea:
        # first split model_dim into num_heads and head_dim,
        # then move num_heads before sequence_length.
        #
        # Why this works:
        # model_dim is split into num_heads smaller vectors.
        # Example:
        # model_dim = 32, num_heads = 4, head_dim = 8
        #
        # Step 4: Update the cache.
        #
        # If cache is None:
        # - this is the first step
        # - updated key is just the new key
        # - updated value is just the new value
        #
        # If cache already has old keys and values:
        # - append the new key to old key along the sequence/cache dimension
        # - append the new value to old value along the sequence/cache dimension
        #
        # Cache shapes:
        # key:   [batch_size, num_heads, cached_length, head_dim]
        # value: [batch_size, num_heads, cached_length, head_dim]
        #
        # Use dim=2 when appending because dimension 2 is cached_length.
        #
        # Cache growth example:
        #
        # first token:
        # key.shape     = [2, 4, 1, 8]
        # key_all.shape = [2, 4, 1, 8]
        #
        # second token:
        # old cache key = [2, 4, 1, 8]
        # new key       = [2, 4, 1, 8]
        # key_all       = [2, 4, 2, 8]
        #
        # third token:
        # old cache key = [2, 4, 2, 8]
        # new key       = [2, 4, 1, 8]
        # key_all       = [2, 4, 3, 8]
        #
        # Step 5: Compute attention scores.
        #
        # The query is usually only the newest token during cached generation:
        # q shape: [batch, heads, 1, head_dim]
        #
        # The key contains all cached tokens so far:
        # k_all shape: [batch, heads, cached_length, head_dim]
        #
        # Compare the query with every cached key.
        #
        # Result shape:
        # [batch, heads, query_length, cached_length]
        #
        # In one-token generation:
        # [batch, heads, 1, cached_length]
        #
        # Step 6: Apply masking if needed.
        #
        # Full-sequence path:
        # - use a causal mask
        # - it blocks future tokens
        #
        # Cached one-token generation:
        # - the cache should only contain past and current tokens
        # - there are no future keys in the cache
        # - so a causal mask is often not needed for the one-token path
        #
        # Step 7: Turn scores into attention weights with softmax.
        #
        # Step 8: Use attention weights to mix the cached values.
        #
        # Step 9: Merge heads.
        # Change:
        # [batch, heads, sequence_length, head_dim]
        # back to:
        # [batch, sequence_length, model_dim]
        #
        # Common idea:
        # move sequence_length before heads again,
        # then combine heads and head_dim back into model_dim.
        #
        # Step 10: Apply the output projection.
        #
        # Return:
        # - attention output
        # - updated cache dictionary with keys "key" and "value"


        batch_size, sequence_length, model_dim = x.shape  # (2, 12, 32)

        # step 1, 2, 3:

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.unsqueeze(dim=2).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)  # (2, 12, 4, 8)
        key = key.unsqueeze(dim=2).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)      # (2, 12, 4, 8)
        value = value.unsqueeze(dim=2).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)  # (2, 12, 4, 8)

        query = query.permute(dims=(0,2,1,3))       # (2, 4, 12, 8)
        key = key.permute(dims=(0,2,1,3))           # (2, 4, 12, 8)
        value = value.permute(dims=(0,2,1,3))       # (2, 4, 12, 8)

        # Step 4: Update the cache.
        if cache is None:
            key_all = key
            value_all = value
        else:
            key_all = torch.cat([cache["key"], key], dim=2)
            value_all = torch.cat([cache["value"], value], dim=2)

        updated_cache = {"key": key_all, "value": value_all} 

        # key_all is shape [batch_size, num_heads, cached_length, head_dim]

        #step 5, 6, 7, 8, 9:   
        scores = torch.matmul(query, key_all.transpose(dim0=2, dim1=3)) # (2, 4, 1, 8) @ (2, 4, 8, cached_length) = (2, 4, 1, cached_length) this could be 12 or 1 if 1 token is sent in i.e. the new token
        scores/=sqrt(self.head_dim)

        mask = build_causal_mask(sequence_length)
        mask = mask.to(x.device)        
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = scores.softmax(dim=-1)

        output = torch.matmul(weights, value_all)  # (2, 4, 1, cached_length) @ (2, 4, cached_length, 8) = (2, 4, 1, 8)

        output = output.permute(dims=(0, 2, 1, 3)).reshape(batch_size, sequence_length, model_dim)

        #step 10:
        proj = self.proj(output)

        return proj, updated_cache
        #raise NotImplementedError("Run attention and update the KV cache if one is given.")

# Run attention over the full sequence at once.
# Return a summary of:
# - output shape
# - simple work estimate
# This is the no-cache baseline you will compare against.
def run_without_cache(model: CachedSelfAttention, hidden_states: torch.Tensor) -> dict[str, object]:
    """Run attention over the full sequence at once."""
    # This is the slow baseline.
    #
    # For generation without a cache, imagine doing this repeatedly:
    # - step 1: run attention on 1 token
    # - step 2: run attention on 2 tokens
    # - step 3: run attention on 3 tokens
    # - ...
    # - step N: run attention on N tokens
    #
    # This keeps recomputing old keys and values.
    #
    # Simple work estimate:
    # no_cache_work = 1 + 2 + 3 + ... + sequence_length
    #
    # Example:
    # sequence_length = 4
    # no_cache_work = 1 + 2 + 3 + 4 = 10
    #
    # Default config example:
    # run_without_cache:
    # - input:  [2, 12, 32]
    # - model processes all 12 tokens together
    # - output: [2, 12, 32]
    # - work estimate: 1 + 2 + ... + 12 = 78
    #
    # Return a dictionary with:
    # - output_shape
    # - work_estimate

    output, _ = model(hidden_states, cache=None)

    sequence_length = hidden_states.shape[1]
    work_estimate = sum(range(1, sequence_length + 1))

    return {"output_shape": tuple(output.shape), "work_estimate": work_estimate}

    #raise NotImplementedError("Return output shape and a simple work estimate for no-cache attention.")

# Run attention one token at a time while growing the cache.
# Return:
# - cache shapes
# - final cache length
# - simple work estimate
# This is the heart of the KV-cache idea.
def run_with_cache(model: CachedSelfAttention, hidden_states: torch.Tensor) -> dict[str, object]:
    """Run attention one token at a time while growing the KV cache."""
    # This is the heart of the exercise:
    # - take one token slice
    # - run attention
    # - update cache
    # - repeat
    # After enough steps, the final cache length should equal the full sequence length.
    #
    # More concrete plan:
    #
    # 1. Start with the empty cache value from init_empty_cache().
    #
    # 2. Loop over token positions from the first token to the last token.
    #
    # 3. Slice one token while keeping the sequence dimension.
    #    Shape:
    #    [batch_size, 1, model_dim]
    #
    # 4. Run the model on that one-token slice and pass in the current cache.
    #    Save both the one-token output and the updated cache.
    #
    # 5. Record cache key/value shapes after each step.
    #
    # Expected cache length:
    # after position 0 -> 1
    # after position 1 -> 2
    # after position 2 -> 3
    #
    # Simple work estimate:
    # cached_work = 1 + 1 + 1 + ... + 1
    #
    # Example:
    # sequence_length = 4
    # cached_work = 1 + 1 + 1 + 1 = 4
    #
    # Return a dictionary with:
    # - output_shapes
    # - cache_key_shapes
    # - cache_value_shapes
    # - final_cache_length
    # - work_estimate

    cache = init_empty_cache()

    batch_size, sequence_length, model_dim = hidden_states.shape

    output_shapes = []
    cache_key_shapes = []
    cache_value_shapes = []

    for i in range(sequence_length):
        one_token = hidden_states[:, i:i+1, :]
        output, updated_cache = model(one_token, cache)
        cache = updated_cache

        output_shapes.append(tuple(output.shape))
        cache_key_shapes.append(tuple(cache["key"].shape))
        cache_value_shapes.append(tuple(cache["value"].shape))

    final_cache_length = cache["key"].shape[2]
    work_estimate = sequence_length

    return {
        "output_shapes": output_shapes,
        "cache_key_shapes": cache_key_shapes,
        "cache_value_shapes": cache_value_shapes,
        "final_cache_length": final_cache_length,
        "work_estimate": work_estimate,
    }
    #raise NotImplementedError("Return cache shapes and a simple work estimate for cached attention.")

# Compare no-cache and cached work estimates.
# Return:
# - dictionary with numeric comparison and a plain-language note
def compare_cache_work(no_cache: dict[str, object], cached: dict[str, object]) -> dict[str, float | str]:
    """Compare no-cache and cached generation in plain numbers."""
    # Compare the two work estimates.
    #
    # Example:
    # no_cache_work = 10
    # cached_work = 4
    #
    # saved_work = no_cache_work - cached_work
    # speedup_ratio = no_cache_work / cached_work
    #
    # Meaning:
    # - saved_work says how much repeated key/value work was avoided
    # - speedup_ratio says how many times larger the no-cache estimate was
    #
    # Plain-language note example:
    # "The cached path avoids rebuilding keys and values for old tokens.
    #  It only adds one new key and one new value at each generation step."

    no_cache_work = float(no_cache["work_estimate"])
    cached_work = float(cached["work_estimate"])

    saved_work = no_cache_work - cached_work
    speedup_ratio = no_cache_work / cached_work

    note = (
        "The cached path avoids rebuilding keys and values for old tokens. "
        "It only adds one new key and one new value at each generation step."
    )

    return {
        "no_cache_work": no_cache_work,
        "cached_work": cached_work,
        "saved_work": saved_work,
        "speedup_ratio": speedup_ratio,
        "comparison_note": note,
    }

    #raise NotImplementedError("Compare work estimates and write a short note.")

# Run the full KV-cache workflow.
# Return:
# - final cache length
# - expected cache length
# - cache shape check
# - comparison note
def run_cache_demo(config: KVCacheConfig) -> dict[str, object]:
    """Run the full KV cache demo."""
    # Full workflow:
    #
    # 1. Create the model.
    # 2. Create toy hidden states.
    # 3. Run the no-cache summary.
    # 4. Run the cached one-token-at-a-time summary.
    # 5. Compare work estimates.
    # 6. Check that the final cache length equals config.sequence_length.
    # 7. Check that final key and value cache shapes match describe_cache_shapes(config).
    #
    # Return a dictionary with at least:
    # - final_cache_length
    # - expected_cache_length
    # - cache_shape_ok
    # - comparison_note

    model = CachedSelfAttention(config)
    hidden_states = make_toy_hidden_states(config)

    no_cache = run_without_cache(model, hidden_states)
    cached = run_with_cache(model, hidden_states)
    comparison = compare_cache_work(no_cache, cached)

    expected_shapes = describe_cache_shapes(config)
    final_key_shape = cached["cache_key_shapes"][-1]
    final_value_shape = cached["cache_value_shapes"][-1]

    expected_key_shape = expected_shapes["cache_key_shape_after_full_generation"]
    expected_value_shape = expected_shapes["cache_value_shape_after_full_generation"]

    final_cache_length = int(cached["final_cache_length"])
    expected_cache_length = config.sequence_length

    cache_shape_ok = (
        final_cache_length == expected_cache_length
        and final_key_shape == expected_key_shape
        and final_value_shape == expected_value_shape
    )

    return {
        "final_cache_length": final_cache_length,
        "expected_cache_length": expected_cache_length,
        "cache_shape_ok": cache_shape_ok,
        "final_key_shape": final_key_shape,
        "final_value_shape": final_value_shape,
        "no_cache_work": comparison["no_cache_work"],
        "cached_work": comparison["cached_work"],
        "saved_work": comparison["saved_work"],
        "speedup_ratio": comparison["speedup_ratio"],
        "comparison_note": comparison["comparison_note"],
    }


    #raise NotImplementedError("Run no-cache and cached generation and compare them.")

# Check whether the cache demo returned the main proof values.
# Return:
# - True when the expected cache fields are present and shape check passed
def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when cache length and comparison values are present."""
    required = {"final_cache_length", "expected_cache_length", "cache_shape_ok", "comparison_note"}
    return required.issubset(metrics) and bool(metrics.get("cache_shape_ok", False))

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
    """Run the KV cache scaffold one step at a time."""
    config = KVCacheConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 6
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
    _, ok = report_task_progress("describe_cache_shapes", describe_cache_shapes, config)
    if not count_or_stop(ok):
        return
    hidden_states, ok = report_task_progress("make_toy_hidden_states", make_toy_hidden_states, config)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("build_causal_mask", build_causal_mask, config.sequence_length)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("init_empty_cache", init_empty_cache)
    if not count_or_stop(ok):
        return
    model, ok = report_task_progress("CachedSelfAttention.__init__", CachedSelfAttention, config)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_cache_demo", run_cache_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "kv cache checks", float(int(passed)), "cache_shape_ok=True")
if __name__ == "__main__":
    main()
