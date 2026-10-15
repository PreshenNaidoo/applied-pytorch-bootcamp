"""
============================================================
EXERCISE 03: ATTENTION FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building scaled dot-product attention by hand.

This file is intentionally small.
The point is not a big dataset.
The point is to understand exactly what attention does.

You will work with short toy token sequences and tiny tensors.

WHY THIS EXERCISE EXISTS
Attention shows up again and again in modern deep learning.
If you understand this file well, later transformer files will feel much less confusing.

This exercise teaches:
- what queries, keys, and values mean in practice
- how attention scores are computed
- why we divide by the square root of the key dimension
- how masking changes which positions can be used
- what shape each tensor should have

Plain-language idea:
- a query asks, "what information am I looking for right now?"
- a key says, "what kind of information does this token contain?"
- a value is the information that gets mixed into the output
- attention is the rule that decides how much each token should look at the other tokens

Beginner intuition:
- imagine each token is holding a small note card
- each token has:
  - a `key`, which is like the label on the outside of the card
  - a `value`, which is the useful content written inside the card
- the current token sends out a `query`
- the query is like a search request:
  - "what kind of information am I looking for right now?"
- attention works by asking:
  - "which other cards have labels that match my search?"
  - "how much should I read from each of those cards?"
- if a card's label matches the query well, that card's value gets more attention
- the final output for one token is a weighted mix of the values from the tokens it decided to read

Three different roles in plain words:
- `query`: what the current token is looking for
- `key`: what each token can be found by
- `value`: what each token actually gives if chosen

Small example:
- token A sends out a query like:
  - "I need information about who the action is about"
- token B has a key like:
  - "person"
- token B has a value:
  - the useful features it carries
- if token A's query matches token B's key well, token A attends more strongly to token B
- that means token B gets a higher attention weight from token A
- so more of token B's value is mixed into token A's output

A simple picture:
- each row of the attention matrix belongs to one token doing the looking
- each column is one token being looked at
- if the number at row A, column B is large, then A is paying a lot of attention to B

DATASET
No big dataset is needed.
Use a tiny synthetic sequence batch.

Simple default setup in this scaffold:
- small random token embeddings
- batch size `2`
- sequence length `4`
- model dimension `8`

HOW TO LOAD OR DOWNLOAD THE DATASET
You do not need to download anything.

The toy input tensors can be created directly in code.

INPUTS AND OUTPUTS
Attention inputs:
- `query`
  - type: `torch.Tensor`
  - shape: `[batch_size, sequence_length, d_k]`
- `key`
  - type: `torch.Tensor`
  - shape: `[batch_size, sequence_length, d_k]`
- `value`
  - type: `torch.Tensor`
  - shape: `[batch_size, sequence_length, d_v]`

Default toy shape in this file:
- `query`: `[2, 4, 8]`
- `key`: `[2, 4, 8]`
- `value`: `[2, 4, 8]`

What that means:
- `2` = batch size, so there are 2 examples at once
- `4` = sequence length, so each example has 4 token positions
- `8` = feature size, so each token is stored as 8 numbers

Causal mask:
- type: `torch.Tensor`
- dtype: `torch.bool`
- shape: `[1, sequence_length, sequence_length]`
- meaning:
  - `True` means "this position is allowed"
  - `False` means "mask this position out"

Attention weight output:
- type: `torch.Tensor`
- shape: `[batch_size, sequence_length, sequence_length]`
- meaning: for each token, how much weight it gives to every token in the sequence

Attention output:
- type: `torch.Tensor`
- shape: `[batch_size, sequence_length, d_v]`

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
For this file, you only need a tiny single-head attention path:
- optional linear layers to make Q, K, V
- scaled dot-product attention
- optional causal mask

No large transformer block is required here.

More plain words:
- `single-head` means there is one attention calculation, not many in parallel
- `scaled dot-product` means compare queries and keys with multiplication, then divide by a size term so the numbers stay stable
- `causal mask` means a token can look at itself and earlier tokens, but not future tokens

Why this matters:
- attention is a way for each token to choose where to get extra information from
- this is helpful because the meaning of one token often depends on nearby tokens
- for example, in a short phrase like "bank of the river", the word "bank" should pay attention to "river"
- in "money in the bank", the same word should pay attention to different nearby words
- attention gives the model a way to change what it focuses on depending on the context

LOSS FUNCTION
No training loss is required for the first core step.

The main goal is correct tensor math and correct shapes.

If you add a tiny supervised toy task later, you may choose a suitable loss for that task.

METRICS TO TRACK
For the core attention step, track simple sanity checks:
- `weights_row_sum_close_to_one`
  - type: `float`
  - meaning: mean absolute error between each attention row sum and `1.0`
  - smaller is better
- `output_shape_ok`
  - type: `float`
  - meaning: `1.0` if the attention output shape matches the expected shape, else `0.0`
- `mask_respected`
  - type: `float`
  - meaning: `1.0` if masked positions get zero attention after softmax, else `0.0`

These are simple sanity metrics, not dataset metrics.

TARGET RESULT RANGE
A correct implementation should pass shape checks and masking checks exactly.

For example:
- attention row sums should be very close to `1.0`
- masked future positions should get zero or almost-zero attention

COMMON MISTAKES
- forgetting the matrix transpose on the key tensor
- applying softmax on the wrong dimension
- forgetting to divide by `sqrt(d_k)`
- using a float mask in a confusing way instead of a clear boolean mask
- masking after softmax instead of before softmax
- mixing up the attention weights shape and the attention output shape

STEP-BY-STEP TODO
1. Build a tiny toy batch of embeddings.
2. Build a causal mask.
3. Compute raw attention scores with `Q @ K^T`.   #@ means matmul
4. Scale by `sqrt(d_k)`.
5. Apply the mask before softmax.
6. Apply softmax across the key positions.
7. Multiply the weights by `V`.
8. Check the output shape and row sums.

What each step is doing in simple words:
1. make a tiny fake batch so you can study the math without lots of extra details
2. build the rule that blocks looking into the future
3. compare every query against every key to measure how well they match
4. shrink the score size so softmax stays well-behaved
5. block illegal positions before turning scores into probabilities
6. turn scores into weights that add up to `1.0`
7. use those weights to mix information from the value vectors
8. check that the numbers and shapes make sense

STRETCH GOALS
- compare masked attention and unmasked attention
- print one attention matrix and read it row by row
- add a tiny learnable self-attention module with linear Q, K, V projections

DEFINITION OF DONE
You are done when:
- you can explain Q, K, and V in plain words
- your attention weights have shape `[batch_size, sequence_length, sequence_length]`
- each attention row sums to about `1.0`
- your causal mask blocks future positions correctly
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import sys

import torch
import torch.nn as nn


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed


@dataclass
class AttentionExerciseConfig:
    """Settings for the attention-from-scratch exercise."""

    batch_size: int = 2
    sequence_length: int = 4
    model_dim: int = 8
    random_state: int = 42
    atol: float = 1e-5



def describe_attention_shapes(config: AttentionExerciseConfig) -> dict[str, object]:
    """Return the expected tensor shapes for this exercise.

    Return a dict with exactly these keys:
    - `query_shape`
    - `key_shape`
    - `value_shape`
    - `attention_weights_shape`
    - `attention_output_shape`
    - `scale_value`
    """
    return {
        "query_shape": (config.batch_size, config.sequence_length, config.model_dim),
        "key_shape": (config.batch_size, config.sequence_length, config.model_dim),
        "value_shape": (config.batch_size, config.sequence_length, config.model_dim),
        "attention_weights_shape": (config.batch_size, config.sequence_length, config.sequence_length),
        "attention_output_shape": (config.batch_size, config.sequence_length, config.model_dim),
        "scale_value": sqrt(config.model_dim),
    }



def make_toy_attention_inputs(config: AttentionExerciseConfig) -> dict[str, torch.Tensor]:
    """Return a tiny toy attention batch.

    Return a dict with exactly these keys:
    - `query`
    - `key`
    - `value`

    Each value is a tensor of shape `[batch_size, sequence_length, model_dim]`.
    """
    generator = torch.Generator().manual_seed(config.random_state)
    query = torch.randn(config.batch_size, config.sequence_length, config.model_dim, generator=generator)
    key = torch.randn(config.batch_size, config.sequence_length, config.model_dim, generator=generator)
    value = torch.randn(config.batch_size, config.sequence_length, config.model_dim, generator=generator)
    
    return {"query": query, "key": key, "value": value}


# Build a boolean causal mask.
# Return:
# - type: `torch.Tensor`
# - dtype: `torch.bool`
# - shape: `[1, sequence_length, sequence_length]`
# Value meaning:
# - `True` means this position is allowed
# - `False` means this position must be masked out
# A causal mask lets each token look at itself and earlier tokens, but not future tokens.
# Intuition:
# - think of a sentence being read from left to right
# - when you are predicting the current token, you should not peek at later tokens
# - the mask enforces that rule by blocking attention to future positions
# Clear 2D picture for sequence length `4`:
# - row 0: `True,  False, False, False`
# - row 1: `True,  True,  False, False`
# - row 2: `True,  True,  True,  False`
# - row 3: `True,  True,  True,  True`
# Meaning:
# - each row is the current token doing the looking
# - each column is a token position being looked at
# - `True` means "this position is allowed"
# - `False` means "this position is blocked"
# So:
# - token 0 can look only at token 0
# - token 1 can look at tokens 0 and 1
# - token 2 can look at tokens 0, 1, and 2
# - token 3 can look at tokens 0, 1, 2, and 3
# The actual returned tensor has shape `[1, sequence_length, sequence_length]`,
# but the last two dimensions should follow exactly this lower-triangular `True` pattern.
def build_causal_mask(sequence_length: int) -> torch.Tensor:

    raise NotImplementedError("Return a boolean causal mask with shape [1, sequence_length, sequence_length].")

    # for i in range(causal_mask.shape[0]):
    #     print(causal_mask[i])

    # the above was my solution.
    # the alternate way:
    



# Compute scaled dot-product attention by hand.
# Return a dict with exactly these keys:
# - `output`
# - `weights`
# Return type:
# - `dict[str, torch.Tensor]`
# Return meaning:
# - `weights` is the attention matrix after softmax with shape `[batch_size, sequence_length, sequence_length]`
# - `output` is the weighted value tensor with shape `[batch_size, sequence_length, model_dim]`
# Important:
# - compute scores with `query @ key.transpose(-2, -1)`
# - divide the scores by `sqrt(d_k)` before softmax
# - if you use a mask, apply it before softmax
# - apply softmax on the last dimension because that dimension is "which key positions can I look at?"
# More step-by-step tensor guidance:
# - `query` starts as `[batch_size, sequence_length, d_k]`
# - `key.transpose(-2, -1)` becomes `[batch_size, d_k, sequence_length]`
# - after matrix multiply, scores become `[batch_size, sequence_length, sequence_length]`
# - each row of that score matrix means:
#   "for this token, how much should I look at every token position?"
# - after softmax, each row should sum to about `1.0`
# - then multiply the weights by `value` to get the final weighted output
# Intuition for the main math:
# - `query @ key.transpose(-2, -1)` compares each token's question with every token's label
# - bigger score means "this other token looks more relevant to me"
# - smaller score means "this other token looks less relevant to me"
# - softmax turns those scores into a clean set of weights, like percentages
# - those weights say how much information to take from each value vector
# - the output for one token is not copied from just one place
# - instead, it is a blend of information from several tokens
#
# Tiny example idea:
# - if one row of weights is `[0.70, 0.20, 0.10, 0.00]`
# - that means the current token mostly uses information from token 0
# - it uses a smaller amount from token 1
# - it uses a tiny amount from token 2
# - and none from token 3
#
# Why divide by `sqrt(d_k)`?
# - when vectors get bigger, raw dot-product scores can grow large
# - very large scores can make softmax too sharp
# - that can make training unstable and harder to learn from
# - scaling keeps the score size in a healthier range
#
# Why apply the mask before softmax?
# - because masked positions should get no chance at all
# - if you mask after softmax, the row may no longer sum to `1.0`
# - masking before softmax lets softmax distribute all probability only across allowed positions
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:

    #query, key and vale are of shape (2, 4, 8) in this exercise.  2 is the batch_dim, 4 is the sequence_length, 8 is the features per token
    #mask is of shape (1, 4, 4)

    #quick steps to rem
    #matmul q and k
    #scale
    #mask
    #softmax
    #weights * V

    # If you normalized Q and K first, then the score would become much closer to cosine similarity. 
    # That can work, but it changes what the model is allowed to learn. A plain dot product contains two things:
    # - Direction similarity
    # - Magnitude information
    # Cosine similarity keeps only direction and throws away magnitude. In attention, that magnitude can be useful. 
    # The model may want some queries or keys to have stronger or weaker influence, and normalization would remove 
    # part of that freedom.
    # A simple way to think about it:
    # Dot product says: "Are these vectors pointing in a similar direction, and how strong are they?"
    # Cosine similarity says: "Are these vectors pointing in a similar direction, ignoring strength?"
    
    raise NotImplementedError("Return the scaled dot-product attention output and the attention weights.")
    


    


    #`weights` is the attention matrix after softmax with shape `[batch_size, sequence_length, sequence_length]`
    #`output` is the weighted value tensor with shape `[batch_size, sequence_length, model_dim]`


class SingleHeadSelfAttention(nn.Module):
    """Tiny single-head self-attention stub.

    This class is here so you can connect the manual attention math to a learnable module.
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        raise NotImplementedError("Build the linear layers that produce Q, K, and V for one attention head.")
        # Beginner-friendly build order:
        # - make three linear layers:
        #   - one for query
        #   - one for key
        #   - one for value
        # - each one maps `model_dim -> model_dim`
        #
        # Why three layers?
        # - the same input sequence is turned into:
        #   - queries: what each token is asking for
        #   - keys: what each token offers
        #   - values: the information to mix together
        #
        # Intuition:
        # - the input `x` starts as one shared representation per token
        # - we then make three different views of that same token
        # - one view is for asking
        # - one view is for matching
        # - one view is for carrying content
        # - this is why Q, K, and V all come from the same input, but use different linear layers



    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return a dict with exactly these keys:
        - `output`
        - `weights`

        Return meaning:
        - `output` is a tensor with shape `[batch_size, sequence_length, model_dim]`
        - `weights` is a tensor with shape `[batch_size, sequence_length, sequence_length]`

        Step-by-step path:
        - start from `x`
        - project it into `Q`, `K`, and `V`
        - call your `scaled_dot_product_attention(...)`
        - return whatever that function returns

        Plain-language picture:
        - first, each token makes its question, label, and content vectors
        - next, attention decides which tokens matter for each token
        - last, the token receives a mixed summary built from the value vectors
        """

        raise NotImplementedError("Project x into Q, K, and V, then call your attention function.")






# Run simple sanity checks for one finished attention result.
# Return a dict with exactly these keys:
# - `weights_row_sum_close_to_one`
# - `output_shape_ok`
# - `mask_respected`
# Return type:
# - `dict[str, float]`
# All three values should be Python floats.
# Metric meaning:
# - `weights_row_sum_close_to_one` should be one float for the whole result
# - first, add each attention row
# - next, check how far each row sum is from `1.0`
# - then average those row errors into one final number
# - `output_shape_ok` is a simple pass/fail number:
#   - `1.0` means the output shape is correct
#   - `0.0` means the output shape is wrong
# - `mask_respected` is a simple pass/fail number:
#   - `1.0` means masked positions stayed near zero after softmax
#   - `0.0` means masked positions still got attention weight
# Beginner-friendly check idea:
# - row sums: add across the last dimension of `weights`
# - output shape: compare `output.shape` to `expected_output_shape`
# - mask respected: masked positions should have zero or almost-zero attention weight
# Why these checks help:
# - if rows do not sum to about `1.0`, softmax may be on the wrong dimension
# - if the output shape is wrong, one of the matrix multiplies is probably wrong
# - if masked positions still get weight, then future tokens are leaking into the result
def run_attention_sanity_checks(
    attention_result: dict[str, torch.Tensor],
    expected_output_shape: tuple[int, int, int],
    mask: torch.Tensor | None,
    atol: float,
) -> dict[str, float]:

    raise NotImplementedError("Check row sums, output shape, and whether masked positions stayed masked.")


    

        #torch.all() tests if all elements in input evaluate to True.




def check_metrics(metrics: dict[str, float]) -> bool:
    """Return True when the main sanity checks pass."""
    return (
        metrics.get("output_shape_ok", 0.0) == 1.0
        and metrics.get("mask_respected", 0.0) == 1.0
        and metrics.get("weights_row_sum_close_to_one", 1.0) <= 1e-4
    )



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
    """Run the attention scaffold one step at a time."""
    config = AttentionExerciseConfig()
    set_seed(config.random_state)
    print("Running exercise 03 starter scaffold...")
    print("Main goal: implement scaled dot-product attention by hand and check the tensor shapes.")

    completed_steps = 0
    total_steps = 7

    shape_info, ok = report_task_progress("describe_attention_shapes", describe_attention_shapes, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    toy_inputs, ok = report_task_progress("make_toy_attention_inputs", make_toy_attention_inputs, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    mask, ok = report_task_progress("build_causal_mask", build_causal_mask, config.sequence_length)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    attention_result, ok = report_task_progress(
        "scaled_dot_product_attention",
        scaled_dot_product_attention,
        toy_inputs["query"],
        toy_inputs["key"],
        toy_inputs["value"],
        mask,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    module, ok = report_task_progress("SingleHeadSelfAttention.__init__", SingleHeadSelfAttention, config.model_dim)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("SingleHeadSelfAttention.forward", module, toy_inputs["query"], mask)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    metrics, ok = report_task_progress(
        "run_attention_sanity_checks",
        run_attention_sanity_checks,
        attention_result,
        shape_info["attention_output_shape"],
        mask,
        config.atol,
    )
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics)
        print(f"[{'PASS' if passed else 'TRY_AGAIN'}] attention_sanity_checks: {metrics}")


if __name__ == "__main__":
    main()
