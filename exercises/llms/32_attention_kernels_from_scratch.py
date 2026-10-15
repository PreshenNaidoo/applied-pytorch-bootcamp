"""
============================================================
EXERCISE 32: ATTENTION KERNELS FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building tiny attention forward passes from scratch.

Plain-language meanings:
- standard attention gives each query head its own key and value head
- MQA means many query heads share one key/value head
- GQA means groups of query heads share key/value heads
- a causal mask blocks a token from looking at future tokens
- a sliding-window mask also blocks tokens that are too far away

WHY THIS EXERCISE EXISTS
the related exercise checks attention shapes and memory.
This part makes the actual tensors and runs the attention math.

Formula:

scores = Q @ K.T / sqrt(head_dim)
weights = softmax(scores after masking)
output = weights @ V

What each symbol means:
- `Q`: queries, shape `[batch, heads, query_length, head_dim]`
- `K`: keys, shape `[batch, kv_heads, key_length, head_dim]`
- `V`: values, shape `[batch, kv_heads, key_length, head_dim]`
- `scores`: token-to-token scores
- `weights`: attention probabilities
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42) -> None:
    """Set the main random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a short result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")


@dataclass
class AttentionKernelConfig:
    """Settings for tiny attention tensor runs."""

    batch_size: int = 2
    sequence_length: int = 8
    num_query_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 8
    window_size: int = 3
    random_state: int = 42


def build_toy_qkv(config: AttentionKernelConfig) -> dict[str, torch.Tensor]:
    """Return toy Q, K, and V tensors."""
    # Goal:
    # Create small random tensors for attention.
    #
    # Output:
    # Return a dictionary with:
    # - `q`: `[batch_size, num_query_heads, sequence_length, head_dim]`
    # - `k`: `[batch_size, num_kv_heads, sequence_length, head_dim]`
    # - `v`: `[batch_size, num_kv_heads, sequence_length, head_dim]`
    #
    # Method:
    # Use `torch.randn(...)`.
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Build toy Q, K, and V tensors.")


def build_causal_mask(sequence_length: int) -> torch.Tensor:
    """Return a causal mask."""
    # Goal:
    # Build a boolean mask with shape `[sequence_length, sequence_length]`.
    #
    # Meaning:
    # - True means this query token may look at that key token.
    # - False means it is blocked.
    #
    # Example for sequence_length=4:
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    #
    # Method:
    # Use `torch.tril(torch.ones(..., dtype=torch.bool))`.
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Build a causal attention mask.")


def build_sliding_window_causal_mask(sequence_length: int, window_size: int) -> torch.Tensor:
    """Return a sliding-window causal mask."""
    # Goal:
    # Build a mask where each token can look only backward within a small window.
    #
    # Token `i` may look at token `j` when:
    # - `j <= i`, so it is not in the future
    # - `i - j < window_size`, so it is close enough
    #
    # Example:
    # - sequence_length: 5
    # - window_size: 2
    # - token 4 may look at tokens 3 and 4 only
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Build a sliding-window causal mask.")


def repeat_kv_heads(k_or_v: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Repeat key or value heads until they match the number of query heads."""
    # Goal:
    # Convert K or V from fewer heads to one head per query head.
    #
    # Input:
    # - `k_or_v`: `[batch_size, num_kv_heads, sequence_length, head_dim]`
    #
    # Output:
    # - tensor with shape `[batch_size, num_query_heads, sequence_length, head_dim]`
    #
    # Steps:
    # 1. Read `num_kv_heads = k_or_v.shape[1]`.
    # 2. Check that `num_query_heads % num_kv_heads == 0`.
    # 3. Compute `repeat_count = num_query_heads // num_kv_heads`.
    # 4. Return `k_or_v.repeat_interleave(repeat_count, dim=1)`.
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Repeat K/V heads to match query heads.")


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run attention from Q, K, V tensors."""
    # Goal:
    # Compute attention output and attention weights by hand.
    #
    # Input:
    # - `q`: `[batch, heads, query_length, head_dim]`
    # - `k`: `[batch, heads, key_length, head_dim]`
    # - `v`: `[batch, heads, key_length, head_dim]`
    # - `mask`: optional `[query_length, key_length]` boolean mask
    #
    # Output:
    # - attention output: `[batch, heads, query_length, head_dim]`
    # - attention weights: `[batch, heads, query_length, key_length]`
    #
    # Steps:
    # 1. Compute `scores = q @ k.transpose(-2, -1)`.
    # 2. Divide by `sqrt(head_dim)`.
    # 3. If a mask is given, fill blocked positions with a large negative value.
    # 4. Apply `F.softmax(scores, dim=-1)`.
    # 5. Compute `output = weights @ v`.
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Run scaled dot-product attention.")


def run_attention_variants(config: AttentionKernelConfig) -> dict[str, object]:
    """Run standard, MQA, and GQA-style attention on tiny tensors."""
    # Goal:
    # Connect the helper functions and return useful shape checks.
    #
    # Output keys:
    # - `standard_output_shape`
    # - `gqa_output_shape`
    # - `causal_mask_shape`
    # - `sliding_mask_shape`
    # - `weights_sum_close`
    # - `attention_note`
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Run tiny attention variants and return checks.")


def check_metrics(summary: dict[str, object], config: AttentionKernelConfig) -> bool:
    """Return True when the output shape is correct."""
    expected = (config.batch_size, config.num_query_heads, config.sequence_length, config.head_dim)
    return tuple(summary.get("gqa_output_shape", ())) == expected


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    print(f"[RUNNING] {task_name}")
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    except Exception as error:
        print(f"[ERROR] {task_name}: {type(error).__name__}: {error}")
        raise
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the attention-kernel scaffold."""
    config = AttentionKernelConfig()
    set_seed(config.random_state)

    qkv, ok = report_task_progress("build_toy_qkv", build_toy_qkv, config)
    if not ok:
        return
    _, ok = report_task_progress("build_causal_mask", build_causal_mask, config.sequence_length)
    if not ok:
        return
    k, ok = report_task_progress("repeat_kv_heads", repeat_kv_heads, qkv["k"], config.num_query_heads)
    if not ok:
        return
    v, ok = report_task_progress("repeat_kv_heads", repeat_kv_heads, qkv["v"], config.num_query_heads)
    if not ok:
        return
    _, ok = report_task_progress("scaled_dot_product_attention", scaled_dot_product_attention, qkv["q"], k, v)
    if not ok:
        return
    summary, ok = report_task_progress("run_attention_variants", run_attention_variants, config)
    if ok and summary is not None:
        passed = check_metrics(summary, config)
        print_pass_or_try_again(passed, "gqa_output_shape", float(int(passed)), "expected shape")


if __name__ == "__main__":
    main()
