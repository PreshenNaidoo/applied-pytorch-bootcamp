"""
============================================================
EXERCISE 09: ROPE POSITIONAL EMBEDDINGS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny RoPE example.
RoPE means rotary positional embeddings.

Plain-language meaning:
- attention needs to know token order
- normal positional embeddings add position vectors to token vectors
- RoPE rotates query and key vectors based on token position
- the vector shape stays the same, but the attention scores change

WHY THIS EXERCISE EXISTS
Many modern language models use RoPE.
It is a common part of decoder-only transformers.
This exercise teaches:
- how RoPE keeps tensor shapes unchanged
- why RoPE is applied to queries and keys
- how position can change attention scores
- how to test the shape and score changes

HOW THIS FITS INTO THE BOOTCAMP
After attention and decoder-only transformer exercises,
this file zooms in on one modern transformer detail that often appears in real language models.

What each step is doing in simple words:
1. make tiny query and key tensors
2. build the position-dependent sine and cosine values
3. write the small rotation helper
4. rotate queries and keys without changing shape
5. compare attention scores before and after
6. explain how position changed behavior even though tensor size stayed the same

Beginner intuition:
- ordinary attention knows content well, but it also needs a way to know order
- RoPE gives order information by rotating parts of the query and key vectors
- so two identical tokens at different positions can behave differently

Small example:
- token A at position 2 and token A at position 7 may start with similar content
- after RoPE, their query/key vectors are rotated differently
- that means attention scores can now depend on position too

DATASET
No downloaded dataset is needed.
Use tiny toy query and key tensors created inside the file.

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- create fake query and key tensors
- create position indices
- build sine and cosine values from those positions
- compare attention scores before and after rotation

INPUTS AND OUTPUTS
Query and key tensors:
- shape: `[batch_size, num_heads, sequence_length, head_dim]`
- `head_dim` must be even for this simple version
RoPE output:
- rotated queries with the same shape
- rotated keys with the same shape

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
This exercise focuses on one transformer piece only:
- rotate queries and keys using position information

LOSS FUNCTION
No training loss is required for the main exercise.

METRICS TO TRACK
Track at least:
- shape check passed
- attention scores changed
- one plain-language explanation

COMMON MISTAKES
- applying RoPE to values instead of queries and keys
- using an odd `head_dim`
- changing the tensor shape by mistake
- mixing up sequence dimension and head dimension
- forgetting that sine and cosine values depend on token position

STEP-BY-STEP TODO
1. Create toy query and key tensors.
2. Build RoPE sine and cosine tensors.
3. Write `rotate_half(...)`.
4. Apply RoPE to queries and keys.
5. Compute attention scores before RoPE.
6. Compute attention scores after RoPE.
7. Check that shapes stayed the same.
8. Write a short note about what changed.

DEFINITION OF DONE
You are done when:
- RoPE keeps query and key shapes unchanged
- attention scores change after RoPE
- shape checks pass
- you can explain "position by rotation" in plain words
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import torch

from math import sqrt

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again

# Store settings for the RoPE exercise.
# Return:
# - a config object with the toy attention shapes and RoPE base value
@dataclass
class RoPEConfig:
    """Settings for the RoPE exercise."""
    batch_size: int = 2
    num_heads: int = 4
    sequence_length: int = 8
    head_dim: int = 16
    base: float = 10000.0
    random_state: int = 42

# Beginner hints for this file:
# - RoPE changes values by position, but it does not change the tensor shape
# - queries and keys get the rotation because attention scores come from comparing queries and keys
# - if the last dimension is split into pairs, rotation becomes easier to code and understand
# - start with tiny toy tensors so you can print shapes and inspect one position at a time
# - if attention scores do not change after RoPE, the rotation likely is not being applied correctly
# Return:
# - a dictionary of expected query/key shapes and simple setup facts
def describe_rope_shapes(config: RoPEConfig) -> dict[str, object]:
    """Return the key RoPE shapes."""
    return {
        "query_shape": (config.batch_size, config.num_heads, config.sequence_length, config.head_dim),
        "key_shape": (config.batch_size, config.num_heads, config.sequence_length, config.head_dim),
        "head_dim_must_be_even": config.head_dim % 2 == 0,
        "rope_applies_to": ["queries", "keys"],
    }

# Create toy query and key tensors.
# Return:
# - dictionary with `query` and `key`
# - each tensor should have shape `[batch_size, num_heads, sequence_length, head_dim]`
# These stand in for the query and key tensors inside attention.
def make_toy_qk(config: RoPEConfig) -> dict[str, torch.Tensor]:
    """Return toy query and key tensors."""

    raise NotImplementedError("Return query and key tensors with shape [batch, heads, seq, head_dim].")



# Build the sine and cosine values used by RoPE.
# Return:
# - `cos`
# - `sin`
# These tell each token position how much to rotate query and key pairs.
def build_rope_cache(config: RoPEConfig) -> dict[str, torch.Tensor]:
    """Build sine and cosine tensors for RoPE.
    Return a dict with keys:
    - `cos`
    - `sin`
    """
    # Think of this as building position-dependent rotation instructions.
    # Later, each token position uses these sine and cosine values
    # to rotate query and key pairs by different amounts.
    #
    # Work in small steps:
    # 1. Check that `config.head_dim` is even.
    #    RoPE rotates pairs of numbers, so an odd head size does not fit this simple version.
    #
    # 2. Create token positions.
    #    Example when sequence_length is 4:
    #    positions = [0, 1, 2, 3]
    #
    # 3. Create one frequency value for each pair in the head dimension.
    #    If head_dim is 16, there are 8 pairs.
    #    Small example when head_dim is 4:
    #    frequencies might look like [1.0000, 0.0100]
    #
    #    Use this RoPE frequency formula:
    #    frequencies = 1.0 / (base ** (pair_index / (head_dim // 2)))
    #
    #    In plain words:
    #    - `pair_index` is [0, 1, 2, ...]
    #    - `base` is usually 10000
    #    - early pairs rotate faster
    #    - later pairs rotate slower
    #
    #    For head_dim = 4 and base = 10000:
    #    pair_index = [0, 1]
    #    frequencies = [
    #        1.0 / (10000 ** (0 / 2)),
    #        1.0 / (10000 ** (1 / 2)),
    #    ]
    #    which is [1.0000, 0.0100]
    #
    # 4. Combine positions and frequencies to make angles.
    #    The angle tensor should have shape:
    #    [sequence_length, head_dim // 2]
    #    Small example when positions are [0, 1, 2, 3]
    #    and frequencies are [1.0000, 0.0100]:
    #    angles =
    #    [
    #        [0.0000, 0.0000],
    #        [1.0000, 0.0100],
    #        [2.0000, 0.0200],
    #        [3.0000, 0.0300],
    #    ]
    #
    # 5. Take cosine and sine of the angles.
    #    With the example above:
    #    cos has the same shape as angles
    #    sin has the same shape as angles
    #    cos[0] is [1.0000, 1.0000] because cos(0) is 1
    #    sin[0] is [0.0000, 0.0000] because sin(0) is 0
    #
    # 6. Concatenate the angle tensor with itself along the last dimension.
    #    This makes the final shape:
    #    [1, 1, sequence_length, head_dim]
    #
    #    Small example:
    #    [a, b, c, d] becomes [a, b, c, d, a, b, c, d]
    #
    #    This matches the common `rotate_half` style used below,
    #    where the tensor is split into two halves.
    #
    # Why that final shape?
    # Query and key tensors have shape:
    # [batch_size, num_heads, sequence_length, head_dim]
    # The first two dimensions are 1 so PyTorch can broadcast them across every batch and head.

    # step1
    raise NotImplementedError("Return sine and cosine tensors for each position.")

    # step 2
    
    # step 3
    # frequencies = 1.0 / (base ** (pair_index / (head_dim // 2)))

    # step 4
    # positions is shape (sequence_length), frequencies is shape (head_dim//2)
    # angle tensor should be shape (sequence_length, head_dim//2)
    # sequence_length = 8, head_dim=16
    # matmul  (8, 1) @ (1, 16//2) = (8, 8)

    # step 5 and 6
    # these should have shape [1, 1, sequence_length, head_dim]


# Plot the RoPE cache as heatmaps.
# This is a visual helper for learning.
# It is not needed for the metric check.
def plot_rope_cache(config: RoPEConfig, show: bool = True, save_path: str | None = None) -> None:
    """Plot RoPE angles, cosine values, and sine values."""
    # This graph answers:
    # "What rotation instruction does each token position get?"
    #
    # The heatmap layout is:
    # - rows: token positions
    # - columns: feature pairs inside the query/key head dimension
    # - color: angle, cos(angle), or sin(angle)
    #
    # Example:
    # If sequence_length = 8 and head_dim = 16,
    # then there are 8 token positions and 8 feature pairs.
    #
    # Why 8 feature pairs?
    # RoPE rotates two numbers at a time.
    # So head_dim 16 becomes 8 pairs.
    #
    # Early feature pairs rotate faster.
    # Later feature pairs rotate slower.
    # This gives the model different ways to sense short and long distance.
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install matplotlib to use plot_rope_cache(...).") from error

    if config.head_dim % 2 != 0:
        raise ValueError("Head dimension should be an even number.")

    token_positions = torch.arange(start=0, end=config.sequence_length, step=1, dtype=torch.float)
    pair_indexes = torch.arange(start=0, end=config.head_dim // 2, step=1, dtype=torch.float)
    frequencies = 1.0 / (config.base ** (pair_indexes / (config.head_dim // 2)))
    angles = token_positions.unsqueeze(dim=-1) @ frequencies.unsqueeze(dim=0)

    angle_table = angles.detach().cpu()
    cos_table = torch.cos(angles).detach().cpu()
    sin_table = torch.sin(angles).detach().cpu()

    tables = [
        ("Rotation angle", angle_table),
        ("cos(angle)", cos_table),
        ("sin(angle)", sin_table),
    ]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), constrained_layout=True)
    for axis, (title, table) in zip(axes, tables):
        image = axis.imshow(table, aspect="auto", origin="lower")
        axis.set_title(title)
        axis.set_xlabel("feature pair")
        axis.set_ylabel("token position")
        axis.set_xticks(range(config.head_dim // 2))
        axis.set_yticks(range(config.sequence_length))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("RoPE cache: position-based rotation instructions")

    if save_path is not None:
        fig.savefig(save_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)

# Rotate a query or key tensor.
# Input:
# - `x` is usually a query tensor or a key tensor
# - shape: `[batch_size, num_heads, sequence_length, head_dim]`
# Return:
# - a rotated version of `x`
# - same shape as `x`
# This small helper makes the RoPE formula easier to write.
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of hidden values.
    Simple idea:
    - split the last dimension into two halves
    - return the rotated version
    - keep the same shape
    """
    # Important:
    # `x` is the tensor being rotated.
    # In this exercise, `x` will be either the query tensor or the key tensor.
    # The RoPE cache is different. The cache contains `cos` and `sin`,
    # which tell us how much of `x` and how much of `rotate_half(x)` to mix later.
    #
    # RoPE needs a small helper that turns:
    # [x1, x2] into [-x2, x1]
    #
    # This is the basic 2D rotation pattern.
    #
    # For a full tensor, do it across the last dimension:
    # 1. Split `x` into two equal parts along the last dimension.
    #    Example:
    #    first_half, second_half = ...
    #
    # 2. Negate the second half.
    #
    # 3. Put the pieces back together as:
    #    [-second_half, first_half]
    #
    # The returned tensor must have the exact same shape as `x`.
    # If the shape changes, the later RoPE formula will not work.

    # x shape is [batch_size, num_heads, sequence_length, head_dim]

    # step 1:
    raise NotImplementedError("Return the rotate-half version of the tensor.")
    #first_half, second_half = split_x[:,:,:,0,:], split_x[:,:,:,1,:]  # both become (2, 4, 8, 8)

    # step 2:

    # step 3:

    # step 4:


# Apply RoPE to one query or key tensor.
# Return a tensor with the same shape.
# In plain words:
# - RoPE adds position information by rotating the vector, not by changing its size
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to one query or key tensor."""
    # Big picture:
    # RoPE means Rotary Positional Embeddings.
    # It adds position information by rotating query and key vectors.
    #
    # `x` is the tensor we are rotating.
    # In this exercise, `x` is either:
    # - a query tensor
    # - a key tensor
    #
    # `cos` and `sin` come from `build_rope_cache`.
    # They are not learned token embeddings.
    # They are position-based rotation instructions.
    #
    # Causal masking and RoPE are different:
    #
    # - causal masking decides which tokens are allowed
    # - RoPE helps attention understand how far away the allowed tokens are
    #
    # Example:
    # text: "The cat sat"
    #
    # If the model is at "sat":
    # - "The" is 2 positions back
    # - "cat" is 1 position back
    # - "sat" is the current position
    #
    # The causal mask says:
    # - "sat" can look at "The", "cat", and "sat"
    # - "sat" cannot look at future tokens
    #
    # RoPE says:
    # - among the tokens that are allowed, position still matters
    # - a nearby past token and a far past token should not have to look the same
    #
    # Visual idea:
    # Think of a query or key vector as an arrow.
    # RoPE rotates that arrow by an amount based on token position.
    #
    # Same content at different positions can start with a similar vector,
    # but after RoPE it points in a different direction.
    #
    # Attention scores use dot products.
    # Dot products are larger when vectors point in similar directions.
    # So rotating query/key vectors changes attention scores.
    #
    # That is how RoPE makes attention position-aware:
    # it changes the query/key comparison based on where tokens are.
    #
    # Important correction:
    # RoPE is not saying that a nearby word should always get a higher score.
    # It is saying that distance should be visible to attention.
    # The model still learns which words matter.
    #
    # Tiny vector example:
    #
    # Suppose the query for "sat" is:
    # query = [1, 0]
    #
    # Suppose both "The" and "cat" have the same key before RoPE:
    # key = [1, 0]
    #
    # Without RoPE:
    # sat query dot The key = [1, 0] dot [1, 0] = 1
    # sat query dot cat key = [1, 0] dot [1, 0] = 1
    #
    # The scores are the same.
    # Attention cannot tell from these vectors alone that "cat" is closer than "The".
    #
    # Now add a simple RoPE-style position rotation:
    # - "sat" is current position, so rotate query by 0 degrees
    # - "cat" is 1 position back, so rotate key by 30 degrees
    # - "The" is 2 positions back, so rotate key by 60 degrees
    #
    # These are simple made-up angles for understanding.
    # Real RoPE angles come from `build_rope_cache`.
    #
    # After rotation:
    # sat query = [1.00, 0.00]
    # cat key   = [0.87, 0.50]
    # The key   = [0.50, 0.87]
    #
    # Dot products:
    # sat dot cat = [1.00, 0.00] dot [0.87, 0.50] = 0.87
    # sat dot The = [1.00, 0.00] dot [0.50, 0.87] = 0.50
    #
    # Now the score is different because position changed the key direction.
    # This does not force "cat" to always win.
    # It only gives attention a way to see position and distance.
    #
    # How `build_rope_cache` fits:
    # `build_rope_cache` creates the `cos` and `sin` values for each position.
    #
    # Simple mental example:
    # position 0 -> cos/sin for 0 degrees
    # position 1 -> cos/sin for 30 degrees
    # position 2 -> cos/sin for 60 degrees
    #
    # Real RoPE does not use those exact degrees.
    # The important idea is:
    # different position -> different rotation instructions.
    #
    # How `rotate_half` fits:
    # A 2D rotation uses this formula:
    #
    # x = [a, b]
    # rotated_x = [
    #     a * cos_angle - b * sin_angle,
    #     a * sin_angle + b * cos_angle
    # ]
    #
    # This is the standard 2D rotation formula.
    # It comes from multiplying the vector by a 2D rotation matrix:
    #
    # [ cos(angle)  -sin(angle) ]   [ a ]
    # [ sin(angle)   cos(angle) ] @ [ b ]
    #
    # The result is:
    #
    # [
    #     a * cos_angle - b * sin_angle,
    #     a * sin_angle + b * cos_angle
    # ]
    #
    # In RoPE code, the same formula can be written as:
    #
    # rotated_x = (x * cos_angle) + (rotate_half(x) * sin_angle)
    #
    # where:
    #
    # x = [a, b]
    # rotate_half(x) = [-b, a]
    #
    # Meaning:
    # - `angle` is the rotation amount for this token position
    # - `cos_angle` means cos(angle)
    # - `sin_angle` means sin(angle)
    #
    # Important:
    # `a * cos_angle` means multiply `a` by cos(angle).
    # It does not mean cos(a).
    #
    # `rotate_half(x)` gives the second piece needed for this formula:
    #
    # rotate_half([a, b]) = [-b, a]
    #
    # Then:
    #
    # x * cos_angle              = [a * cos_angle, b * cos_angle]
    # rotate_half(x) * sin_angle = [-b * sin_angle, a * sin_angle]
    #
    # Add them:
    #
    # [
    #     a * cos_angle - b * sin_angle,
    #     b * cos_angle + a * sin_angle
    # ]
    #
    # That is the normal 2D rotation formula.
    #
    # So this line:
    # rotated_x = (x * cos) + (rotate_half(x) * sin)
    #
    # means:
    # - take the original query/key vector
    # - take the rotate-half version
    # - mix them using the cos/sin values for that token position
    #
    # Final flow:
    # 1. `build_rope_cache` creates position-based `cos` and `sin`.
    # 2. `rotate_half` creates the helper part needed for rotation.
    # 3. `apply_rope` rotates query and key vectors.
    # 4. attention compares the rotated query and key vectors.
    #
    # Because the rotation depends on position,
    # the final attention scores depend on both token content and token position.
    #
    # The usual RoPE pattern is:
    # - keep part of the original tensor
    # - mix in the rotated version
    # - use cos and sin to decide how much of each to keep
    #
    # The formula is:
    # rotated_x = (x * cos) + (rotate_half(x) * sin)
    #
    # Read it in plain words:
    # - `x * cos` keeps some of the original values
    # - `rotate_half(x) * sin` adds the rotated part
    # - cosine and sine decide the mix for each token position
    #
    # Important checks:
    # - `x` should have shape [batch, heads, seq, head_dim]
    # - `cos` and `sin` should broadcast to that same shape
    # - the output shape should be exactly the same as `x.shape`

    # x is [batch, heads, seq, head_dim]

    raise NotImplementedError("Apply rotary position information and keep the same shape.")


# Compute scaled attention scores from query and key.
# Input:
# - `query`: `[batch_size, num_heads, sequence_length, head_dim]`
# - `key`: `[batch_size, num_heads, sequence_length, head_dim]`
# Return:
# - attention scores with shape `[batch_size, num_heads, sequence_length, sequence_length]`
def attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Return scaled attention scores with shape `[batch, heads, seq, seq]`."""
    # Attention scores compare each query token with each key token.
    #
    # Query shape:
    # [batch, heads, seq, head_dim]
    #
    # Key shape:
    # [batch, heads, seq, head_dim]
    #
    # To compare them, the last two dimensions of key need to be swapped:
    # [batch, heads, head_dim, seq]
    #
    # Then matrix multiply:
    # query @ swapped_key
    #
    # The result shape should be:
    # [batch, heads, seq, seq]
    #
    # Last step:
    # divide by sqrt(head_dim)
    # This keeps the scores from getting too large when head_dim is bigger.

    raise NotImplementedError("Compute scaled query-key attention scores.")



# Compare attention scores before and after RoPE.
# Return:
# - shape checks
# - one numeric difference value
# The main point is to show that position changes the scores while shape stays stable.
def compare_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> dict[str, float | tuple[int, ...]]:
    """Compare attention scores before and after RoPE."""
    # This function proves the main lesson of the exercise:
    # RoPE changes attention scores, but it does not change query/key shapes.
    #
    # Suggested steps:
    # 1. Compute attention scores before RoPE.
    #
    # 2. Apply RoPE to the query.
    #
    # 3. Apply RoPE to the key.
    #
    # 4. Compute attention scores after RoPE.
    #
    # 5. Check that the rotated query has the same shape as the original query.
    #
    # 6. Check that the rotated key has the same shape as the original key.
    #
    # 7. Measure how much the scores changed.
    #    A simple way is to take the mean absolute difference:
    #    (after_scores - before_scores).abs().mean()
    #
    # Return a dictionary with simple values, for example:
    # - "query_shape"
    # - "rotated_query_shape"
    # - "key_shape"
    # - "rotated_key_shape"
    # - "score_difference"

    # step 1:
    raise NotImplementedError("Return shape checks and a score-difference value.")

    # step 2 and 3:    

    # step 4

    # step 5 and 6

    # step 7:



# Write the final explanation.
# Return:
# - one short plain-language note that explains how RoPE changed attention scores
def write_rope_note(score_summary: dict[str, float | tuple[int, ...]]) -> str:
    """Write a short plain-language note about RoPE."""
    raise NotImplementedError("Explain what changed after applying RoPE.")

# Run the full RoPE workflow.
# Return:
# - dictionary with shape checks, score difference, and explanation note
def run_rope_demo(config: RoPEConfig) -> dict[str, float | bool | str]:
    """Run the full RoPE demo."""
    raise NotImplementedError("Run the RoPE shape and score comparison.")




# Check whether the RoPE demo returned the main proof values.
# Return:
# - True when shapes are correct and scores changed
def check_metrics(metrics: dict[str, float | bool | str]) -> bool:
    """Return True when the main RoPE checks pass."""
    return bool(metrics.get("shape_ok", False)) and float(metrics.get("score_difference", 0.0)) > 0.0

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
    """Run the RoPE scaffold one step at a time."""
    config = RoPEConfig()
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
    _, ok = report_task_progress("describe_rope_shapes", describe_rope_shapes, config)
    if not count_or_stop(ok):
        return
    qk, ok = report_task_progress("make_toy_qk", make_toy_qk, config)
    if not count_or_stop(ok):
        return
    rope_cache, ok = report_task_progress("build_rope_cache", build_rope_cache, config)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("rotate_half", rotate_half, qk["query"])
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("apply_rope", apply_rope, qk["query"], rope_cache["cos"], rope_cache["sin"])
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_rope_demo", run_rope_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "rope checks", float(int(passed)), "shape_ok=True and score_difference > 0")
        try:
            plot_path = BOOTCAMP_ROOT / "rope_cache.png"
            plot_rope_cache(config, show=True, save_path=str(plot_path))
            print(f"Saved RoPE cache plot to: {plot_path}")
        except ImportError as error:
            print(f"[PLOT SKIPPED] {error}")
if __name__ == "__main__":
    main()
