"""
============================================================
EXERCISE 31: ATTENTION VARIANTS AND MEMORY
============================================================

WHAT YOU ARE BUILDING
You are building small attention-shape checks for memory-saving attention variants.

Plain-language meanings:
- standard attention gives each head its own queries, keys, and values
- MQA means multi-query attention; many query heads share one key/value head
- GQA means grouped-query attention; groups of query heads share key/value heads
- sliding-window attention means each token only looks at nearby earlier tokens
- FlashAttention is a fast attention method that avoids storing a huge attention matrix

WHY THIS EXERCISE EXISTS
Attention can use a lot of memory.
Frontier labs care about attention memory because it affects context length,
batch size, inference speed, and training cost.

This exercise teaches:
- standard attention shapes
- MQA shapes
- GQA shapes
- sliding-window masks
- attention matrix memory estimates
- why FlashAttention is useful

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises built attention.
This file asks how attention changes when memory matters.

Beginner intuition:
- attention compares tokens with other tokens
- comparing every token with every other token creates a large square matrix
- long context makes that matrix grow quickly
- sharing keys and values can reduce cache memory during generation

DATASET
No downloaded dataset is required.

Use toy hidden-state shapes:
- batch size
- sequence length
- model dimension
- number of attention heads

Why this is enough:
- attention memory depends mostly on tensor shapes
- the actual text content is not needed for the memory lesson

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- create random hidden states if you want to test tensor operations
- or work only with shape tuples for the first version
- build masks and memory estimates from the config values

INPUTS AND OUTPUTS
Input hidden states:
- shape: `[batch_size, sequence_length, model_dim]`

Shape outputs:
- standard attention Q, K, V shapes
- MQA Q, K, V shapes
- GQA Q, K, V shapes

Memory outputs:
- attention score matrix memory
- standard KV cache memory
- MQA KV cache memory
- GQA KV cache memory

STEP-BY-STEP TODO
1. Compute standard Q, K, V shapes.
2. Compute MQA K/V shapes.
3. Compute GQA K/V shapes.
4. Build a sliding-window causal mask.
5. Estimate attention matrix memory.
6. Estimate KV cache memory.
7. Write a note about when each variant helps.

DEFINITION OF DONE
You are done when:
- all shape summaries are returned
- sliding-window mask blocks far-away future tokens
- memory estimates compare standard attention, MQA, and GQA
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.training_checks import print_pass_or_try_again


AttentionShapes = dict[str, tuple[int, ...]]
AttentionMemory = dict[str, float]
AttentionSummary = dict[str, AttentionShapes | AttentionMemory | tuple[int, ...] | bool | str]


# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for attention memory comparisons.
# Expected return:
# - a config object with sequence, head, and memory settings
@dataclass
class AttentionMemoryConfig:
    """Settings for attention memory comparisons."""

    batch_size: int = 2
    sequence_length: int = 128
    model_dim: int = 256
    num_query_heads: int = 8
    num_kv_heads_gqa: int = 2
    num_kv_heads_mqa: int = 1
    window_size: int = 32
    bytes_per_value: int = 2


# Compute standard attention shapes.
# Expected return:
# - dictionary with Q, K, V shapes
def standard_attention_shapes(config: AttentionMemoryConfig) -> AttentionShapes:
    """Return standard Q, K, V shapes."""
    # Goal:
    # - Return Q, K, and V shapes for normal multi-head attention.
    # - In normal attention, each query head has its own key head and value head.
    #
    # Inputs:
    # - Hidden states would have shape `[batch_size, sequence_length, model_dim]`.
    # - Example with the default config:
    #   - hidden states: `[2, 128, 256]`
    #   - `num_query_heads = 8`
    #   - `head_dim = model_dim // num_query_heads = 256 // 8 = 32`
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"q_shape"`: `[batch_size, num_query_heads, sequence_length, head_dim]`
    #   - `"k_shape"`: `[batch_size, num_query_heads, sequence_length, head_dim]`
    #   - `"v_shape"`: `[batch_size, num_query_heads, sequence_length, head_dim]`
    # - With the default config, each shape should be `(2, 8, 128, 32)`.
    #
    # Method:
    # - Use:
    #   `head_dim = config.model_dim // config.num_query_heads`
    # - Then build tuples:
    #   `q_shape = (config.batch_size, config.num_query_heads, config.sequence_length, head_dim)`
    #   `k_shape = q_shape`
    #   `v_shape = q_shape`
    #
    # Steps:
    # 1. Assert `config.model_dim % config.num_query_heads == 0`.
    # 2. Compute `head_dim`.
    # 3. Build the three shape tuples.
    # 4. Return the dictionary.
    #
    # Checks:
    # - `result["q_shape"][-1]` should equal `head_dim`.
    # - `result["k_shape"][1]` should equal `config.num_query_heads`.
    #
    raise NotImplementedError("Return standard attention tensor shapes.")


# Compute MQA shapes.
# Expected return:
# - dictionary with Q, K, V shapes where K and V use one shared head
def mqa_shapes(config: AttentionMemoryConfig) -> AttentionShapes:
    """Return multi-query attention shapes."""
    # Goal:
    # - Return Q, K, and V shapes for multi-query attention.
    # - MQA keeps many query heads but shares one key/value head.
    #
    # Inputs:
    # - Hidden states would have shape `[batch_size, sequence_length, model_dim]`.
    # - Default example:
    #   - query heads: 8
    #   - key/value heads: 1
    #   - head_dim: 32
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"q_shape"`: `[batch_size, num_query_heads, sequence_length, head_dim]`
    #   - `"k_shape"`: `[batch_size, num_kv_heads_mqa, sequence_length, head_dim]`
    #   - `"v_shape"`: `[batch_size, num_kv_heads_mqa, sequence_length, head_dim]`
    # - With defaults:
    #   - Q shape: `(2, 8, 128, 32)`
    #   - K shape: `(2, 1, 128, 32)`
    #   - V shape: `(2, 1, 128, 32)`
    #
    # Method:
    # - Use the same `head_dim` formula as standard attention:
    #   `head_dim = config.model_dim // config.num_query_heads`
    # - K and V use `config.num_kv_heads_mqa`, which should be `1`.
    #
    # Steps:
    # 1. Assert `config.model_dim % config.num_query_heads == 0`.
    # 2. Assert `config.num_kv_heads_mqa == 1`.
    # 3. Compute `head_dim`.
    # 4. Build Q with query heads.
    # 5. Build K and V with one KV head.
    # 6. Return the dictionary.
    #
    # Checks:
    # - `result["q_shape"][1]` should be 8 with defaults.
    # - `result["k_shape"][1]` should be 1 with defaults.
    #
    raise NotImplementedError("Return MQA tensor shapes.")


# Compute GQA shapes.
# Expected return:
# - dictionary with Q, K, V shapes where K and V use fewer heads than Q
def gqa_shapes(config: AttentionMemoryConfig) -> AttentionShapes:
    """Return grouped-query attention shapes."""
    # Goal:
    # - Return Q, K, and V shapes for grouped-query attention.
    # - GQA is between standard attention and MQA: several query heads share each KV head.
    #
    # Inputs:
    # - Hidden states would have shape `[batch_size, sequence_length, model_dim]`.
    # - Default example:
    #   - query heads: 8
    #   - key/value heads: 2
    #   - each KV head serves `8 // 2 = 4` query heads
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"q_shape"`: `[batch_size, num_query_heads, sequence_length, head_dim]`
    #   - `"k_shape"`: `[batch_size, num_kv_heads_gqa, sequence_length, head_dim]`
    #   - `"v_shape"`: `[batch_size, num_kv_heads_gqa, sequence_length, head_dim]`
    # - With defaults:
    #   - Q shape: `(2, 8, 128, 32)`
    #   - K shape: `(2, 2, 128, 32)`
    #   - V shape: `(2, 2, 128, 32)`
    #
    # Method:
    # - Use:
    #   `head_dim = config.model_dim // config.num_query_heads`
    #   `query_heads_per_kv_head = config.num_query_heads // config.num_kv_heads_gqa`
    # - The second line is for checking the grouping is clean.
    #
    # Steps:
    # 1. Assert `config.model_dim % config.num_query_heads == 0`.
    # 2. Assert `config.num_query_heads % config.num_kv_heads_gqa == 0`.
    # 3. Compute `head_dim`.
    # 4. Build Q with query heads.
    # 5. Build K and V with GQA KV heads.
    # 6. Return the dictionary.
    #
    # Checks:
    # - With defaults, `config.num_query_heads // config.num_kv_heads_gqa` should be 4.
    # - K and V should have fewer heads than Q.
    #
    raise NotImplementedError("Return GQA tensor shapes.")


# Build a sliding-window causal mask.
# Expected return:
# - boolean tensor with shape `[sequence_length, sequence_length]`
def build_sliding_window_mask(config: AttentionMemoryConfig) -> torch.Tensor:
    """Return a sliding-window causal attention mask."""
    # Goal:
    # - Build a boolean attention mask for causal sliding-window attention.
    # - Causal means a token cannot look at future tokens.
    # - Sliding-window means a token can only look back a limited number of steps.
    #
    # Inputs:
    # - `config.sequence_length`: number of token positions.
    # - `config.window_size`: how many recent positions each token may look at.
    # - Small example with `sequence_length=5` and `window_size=3`:
    #   - token 3 may look at tokens 1, 2, and 3
    #   - token 3 may not look at token 0 because it is too far away
    #   - token 3 may not look at token 4 because it is in the future
    #
    # Output:
    # - Return a `torch.bool` tensor with shape `[sequence_length, sequence_length]`.
    # - Convention for this exercise:
    #   - `False` means attention is allowed
    #   - `True` means attention is blocked
    # - This matches PyTorch attention masks where boolean `True` usually means "do not attend".
    #
    # Method:
    # - Build query and key position grids:
    #   `positions = torch.arange(config.sequence_length)`
    #   `query_positions = positions[:, None]`
    #   `key_positions = positions[None, :]`
    # - Future tokens are blocked with:
    #   `future_blocked = key_positions > query_positions`
    # - Too-old tokens are blocked with:
    #   `too_old_blocked = key_positions < (query_positions - config.window_size + 1)`
    # - Final mask:
    #   `mask = future_blocked | too_old_blocked`
    #
    # Steps:
    # 1. Assert `config.sequence_length > 0`.
    # 2. Assert `config.window_size > 0`.
    # 3. Build the position grids.
    # 4. Build `future_blocked`.
    # 5. Build `too_old_blocked`.
    # 6. Combine them with `|`.
    # 7. Return the mask.
    #
    # Checks:
    # - `mask.shape` should be `[sequence_length, sequence_length]`.
    # - `mask[0, 1]` should be True because token 0 cannot look at future token 1.
    # - If `window_size=3`, `mask[3, 0]` should be True because token 0 is too far back.
    # - `mask[3, 2]` should be False because token 3 may look at token 2.
    #
    raise NotImplementedError("Build a mask that allows only nearby previous tokens.")


# Estimate full attention matrix memory.
# Expected return:
# - memory in megabytes
def estimate_attention_matrix_memory_mb(config: AttentionMemoryConfig) -> float:
    """Estimate memory for the attention score matrix."""
    # Goal:
    # - Estimate memory for the full attention score matrix.
    # - This is the `[batch, heads, query_positions, key_positions]` matrix.
    #
    # Inputs:
    # - `config.batch_size`: number of sequences in a batch.
    # - `config.num_query_heads`: number of attention heads.
    # - `config.sequence_length`: number of query and key positions.
    # - `config.bytes_per_value`: bytes per score value, such as 2 for fp16/bfloat16.
    #
    # Output:
    # - Return one float in MB.
    # - The attention score shape is:
    #   `[batch_size, num_query_heads, sequence_length, sequence_length]`
    #
    # Method:
    # - Formula:
    #   `num_scores = batch_size * num_query_heads * sequence_length * sequence_length`
    #   `memory_mb = num_scores * bytes_per_value / (1024 * 1024)`
    # - This grows with `sequence_length ** 2`, so long context can get expensive quickly.
    #
    # Steps:
    # 1. Assert all count values are positive.
    # 2. Compute `num_scores`.
    # 3. Multiply by bytes per value.
    # 4. Convert bytes to MB by dividing by `1024 * 1024`.
    # 5. Return the float value.
    #
    # Checks:
    # - With `batch=1`, `heads=2`, `sequence=4`, and `bytes=2`:
    #   `num_scores = 1 * 2 * 4 * 4 = 32`, so memory is `64 / (1024 * 1024) MB`.
    #
    raise NotImplementedError("Estimate attention score memory in MB.")


# Estimate KV cache memory for a chosen number of key/value heads.
# Expected return:
# - memory in megabytes
def estimate_kv_cache_memory_mb(config: AttentionMemoryConfig, num_kv_heads: int) -> float:
    """Estimate KV cache memory."""
    # Goal:
    # - Estimate memory for cached keys and values during generation.
    # - The KV cache stores past K and V tensors so the model does not recompute them.
    #
    # Inputs:
    # - `config.batch_size`: number of sequences.
    # - `config.sequence_length`: number of cached token positions.
    # - `config.model_dim`: hidden size.
    # - `config.num_query_heads`: query-head count used to compute `head_dim`.
    # - `num_kv_heads`: number of key/value heads to cache.
    # - Standard attention uses `num_kv_heads = config.num_query_heads`.
    # - MQA uses `num_kv_heads = config.num_kv_heads_mqa`.
    # - GQA uses `num_kv_heads = config.num_kv_heads_gqa`.
    #
    # Output:
    # - Return one float in MB.
    # - K cache shape: `[batch_size, num_kv_heads, sequence_length, head_dim]`.
    # - V cache shape: `[batch_size, num_kv_heads, sequence_length, head_dim]`.
    # - K and V have the same shape, so the formula has a factor of `2`.
    #
    # Method:
    # - Use:
    #   `head_dim = config.model_dim // config.num_query_heads`
    #   `num_values = 2 * config.batch_size * num_kv_heads * config.sequence_length * head_dim`
    #   `memory_mb = num_values * config.bytes_per_value / (1024 * 1024)`
    # - `2` means one K cache plus one V cache.
    #
    # Steps:
    # 1. Assert `config.model_dim % config.num_query_heads == 0`.
    # 2. Assert `num_kv_heads > 0`.
    # 3. Compute `head_dim`.
    # 4. Compute the number of cached K and V values.
    # 5. Convert bytes to MB.
    # 6. Return the float value.
    #
    # Checks:
    # - MQA KV cache memory should be smaller than standard KV cache memory.
    # - GQA KV cache memory should be between standard and MQA with the default config.
    #
    raise NotImplementedError("Estimate key/value cache memory in MB.")


# Compare memory across standard attention, MQA, and GQA.
# Expected return:
# - dictionary with memory numbers
def compare_attention_memory(config: AttentionMemoryConfig) -> AttentionMemory:
    """Compare attention memory use."""
    # Goal:
    # - Compare attention score memory and KV cache memory for standard attention, MQA, and GQA.
    #
    # Inputs:
    # - `config`: shape and byte settings.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"attention_scores_mb"`: full score matrix memory
    #   - `"standard_kv_cache_mb"`: KV cache with one KV head per query head
    #   - `"mqa_kv_cache_mb"`: KV cache with one shared KV head
    #   - `"gqa_kv_cache_mb"`: KV cache with grouped KV heads
    #   - `"mqa_saving_mb"`: standard KV cache minus MQA KV cache
    #   - `"gqa_saving_mb"`: standard KV cache minus GQA KV cache
    # - All values are MB.
    #
    # Method:
    # - Use:
    #   `attention_scores_mb = estimate_attention_matrix_memory_mb(config)`
    #   `standard_kv_cache_mb = estimate_kv_cache_memory_mb(config, config.num_query_heads)`
    #   `mqa_kv_cache_mb = estimate_kv_cache_memory_mb(config, config.num_kv_heads_mqa)`
    #   `gqa_kv_cache_mb = estimate_kv_cache_memory_mb(config, config.num_kv_heads_gqa)`
    # - Then compute savings by subtracting from standard KV cache memory.
    #
    # Steps:
    # 1. Compute full attention score memory.
    # 2. Compute standard KV cache memory.
    # 3. Compute MQA KV cache memory.
    # 4. Compute GQA KV cache memory.
    # 5. Compute MQA and GQA savings.
    # 6. Return the dictionary with the exact keys above.
    #
    # Checks:
    # - `mqa_kv_cache_mb <= gqa_kv_cache_mb <= standard_kv_cache_mb` should be True
    #   with the default config.
    # - `set(memory)` should equal the six output keys listed above.
    #
    raise NotImplementedError("Return memory estimates for standard attention, MQA, and GQA.")


# Write a note explaining the variants.
# Expected return:
# - short plain-language string
def write_attention_variant_note(memory: AttentionMemory) -> str:
    """Explain attention variants in plain words."""
    # Goal:
    # - Write a short note that explains when each attention variant helps.
    #
    # Inputs:
    # - `memory`: output from `compare_attention_memory(config)`.
    # - Required keys:
    #   `"attention_scores_mb"`, `"standard_kv_cache_mb"`, `"mqa_kv_cache_mb"`,
    #   `"gqa_kv_cache_mb"`, `"mqa_saving_mb"`, and `"gqa_saving_mb"`.
    #
    # Output:
    # - Return one short plain-language string.
    # - It should mention:
    #   - full attention score memory grows with sequence length squared
    #   - MQA saves the most KV cache memory
    #   - GQA saves some KV cache memory while keeping more KV heads than MQA
    #   - FlashAttention helps by avoiding storage of the full attention matrix
    #
    # Method:
    # - Use concrete numbers from `memory`, for example:
    #   `memory["standard_kv_cache_mb"]`
    #   `memory["mqa_kv_cache_mb"]`
    # - Round numbers if you want the sentence to be easy to read:
    #   `round(memory["mqa_saving_mb"], 2)`
    #
    # Steps:
    # 1. Check the required keys are present.
    # 2. Write one or two sentences.
    # 3. Mention at least one concrete MB value.
    # 4. Return the note string.
    #
    # Checks:
    # - The note should contain `"MQA"` or `"multi-query"`.
    # - The note should contain `"FlashAttention"`.
    # - The note should not say FlashAttention changes the mathematical attention result;
    #   it mainly changes how the result is computed and stored.
    #
    raise NotImplementedError("Write a note about attention memory tradeoffs.")


# Run the full attention-memory demo.
# Expected return:
# - summary dictionary with shapes, mask check, memory, and note
def run_attention_memory_demo(config: AttentionMemoryConfig) -> AttentionSummary:
    """Run the attention memory workflow."""
    # Goal:
    # - Run every helper and return one summary dictionary.
    #
    # Inputs:
    # - `config`: shape and memory settings for the toy attention example.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"standard_shapes"`: output of `standard_attention_shapes(config)`
    #   - `"mqa_shapes"`: output of `mqa_shapes(config)`
    #   - `"gqa_shapes"`: output of `gqa_shapes(config)`
    #   - `"mask_shape"`: tuple shape of the sliding-window mask
    #   - `"mask_blocks_future"`: bool check that future tokens are blocked
    #   - `"mask_blocks_too_old"`: bool check that too-old tokens are blocked
    #   - `"memory"`: output of `compare_attention_memory(config)`
    #   - `"attention_note"`: output of `write_attention_variant_note(memory)`
    #
    # Method:
    # - Use this exact call shape:
    #   `standard = standard_attention_shapes(config)`
    #   `mqa = mqa_shapes(config)`
    #   `gqa = gqa_shapes(config)`
    #   `mask = build_sliding_window_mask(config)`
    #   `memory = compare_attention_memory(config)`
    #   `attention_note = write_attention_variant_note(memory)`
    # - For the mask checks:
    #   `mask_blocks_future = bool(mask[0, 1].item())`
    #   `mask_blocks_too_old = bool(mask[config.window_size, 0].item())`
    # - The second check needs `sequence_length > window_size`; the default config has that.
    #
    # Steps:
    # 1. Build all three shape dictionaries.
    # 2. Build the sliding-window mask.
    # 3. Store `tuple(mask.shape)` under `"mask_shape"`.
    # 4. Compute the two mask checks.
    # 5. Compute memory estimates.
    # 6. Write the attention note.
    # 7. Return the summary dictionary with the exact keys above.
    #
    # Checks:
    # - `summary["mask_shape"]` should be `(sequence_length, sequence_length)`.
    # - `summary["mask_blocks_future"]` should be True.
    # - `summary["mask_blocks_too_old"]` should be True with the default config.
    #
    raise NotImplementedError("Return shapes, mask, memory estimates, and note.")


# Check that the summary includes key outputs.
# Expected return:
# - True when the output is complete
def check_metrics(metrics: AttentionSummary) -> bool:
    """Return True when attention memory outputs are present."""
    required = {"standard_shapes", "mqa_shapes", "gqa_shapes", "mask_shape", "memory", "attention_note"}
    return required.issubset(metrics)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the attention variants scaffold."""
    config = AttentionMemoryConfig()

    try:
        metrics = run_attention_memory_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "attention_variant_outputs", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
