"""
============================================================
EXERCISE 30: DISTRIBUTED TRANSFORMER TRAINING BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a map of the main ways large transformer training is split across hardware.

Plain-language meanings:
- data parallelism means each device gets different data but a full model copy
- tensor parallelism means one layer's big matrix math is split across devices
- pipeline parallelism means different layers live on different devices
- ZeRO means optimizer and gradient states are split so each device stores less
- FSDP means fully sharded data parallel; it shards model weights across devices
- shard means split one large thing into smaller pieces

WHY THIS EXERCISE EXISTS
Frontier models are too large for one GPU.
Even if you do not run a giant cluster here, you should understand the training layout.

This exercise teaches:
- why normal data parallelism is not enough for very large models
- what gets copied or split in each parallelism type
- how memory changes when states are sharded
- how to write a simple plan for a transformer training run

HOW THIS FITS INTO THE BOOTCAMP
The general distributed exercise teaches process setup.
This file focuses on transformer-specific training choices.

Beginner intuition:
- data parallelism splits the homework examples
- tensor parallelism splits the big math inside a layer
- pipeline parallelism splits the stack of layers
- sharding means no device has to hold every training state by itself

DATASET
No real training dataset is required.

Use a toy transformer training scenario instead:
- one pretend language-model dataset based on Tiny Shakespeare token counts
- one pretend transformer parameter count
- one pretend number of devices

Why this is enough:
- this exercise is about training layout and memory
- the main lesson is what gets copied and what gets split
- a real dataset would not change that concept

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- create a small metadata dictionary for the pretend training run
- include:
  - number of tokens, using Tiny Shakespeare as the recommended small-data example
  - sequence length
  - batch size
  - number of devices
  - parameter count
- use those numbers for memory and strategy estimates

INPUTS AND OUTPUTS
Input:
- parameter count
- bytes per parameter
- optimizer-state multiplier
- number of devices

Output:
- replicated memory estimate
- sharded memory estimate
- descriptions of data parallelism, tensor parallelism, pipeline parallelism, and sharding
- one plain-language strategy note

STEP-BY-STEP TODO
1. Estimate model parameter memory.
2. Estimate optimizer memory.
3. Compare replicated and sharded memory.
4. Describe data parallelism.
5. Describe tensor parallelism.
6. Describe pipeline parallelism.
7. Describe FSDP or ZeRO-style sharding.
8. Choose a simple strategy for a toy transformer.

DEFINITION OF DONE
You are done when:
- memory estimates are returned
- each parallelism type is described
- the strategy note names what is split and what is copied
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.training_checks import print_pass_or_try_again


MemoryEstimates = dict[str, float]
ParallelismDescription = dict[str, str]
DistributedTransformerSummary = dict[
    str,
    MemoryEstimates | ParallelismDescription | str,
]


# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for distributed transformer estimates.
# Expected return:
# - a config object with model size, memory, and device-count settings
@dataclass
class DistributedTransformerConfig:
    """Settings for transformer distributed-training estimates."""

    parameter_count: int = 50_000_000
    bytes_per_parameter: int = 2
    optimizer_multiplier: float = 8.0
    num_devices: int = 4


# Estimate memory for model weights.
# Expected return:
# - memory in megabytes
def estimate_weight_memory_mb(config: DistributedTransformerConfig) -> float:
    """Estimate model weight memory."""
    # Goal:
    # - Estimate how much memory the model weights use.
    # - Model weights are the learned numbers in the transformer.
    #
    # Inputs:
    # - `config.parameter_count`: number of learned parameters.
    # - `config.bytes_per_parameter`: bytes used by one parameter.
    # - Example:
    #   - `parameter_count = 1_000`
    #   - `bytes_per_parameter = 2`
    #   - total bytes = `1_000 * 2 = 2_000` bytes
    #
    # Output:
    # - Return one `float`.
    # - Unit: megabytes, using `1 MB = 1024 * 1024 bytes`.
    #
    # Method:
    # - Formula:
    #   `weight_memory_mb = parameter_count * bytes_per_parameter / (1024 * 1024)`
    # - `parameter_count * bytes_per_parameter` gives bytes.
    # - Dividing by `1024 * 1024` converts bytes to MB.
    #
    # Steps:
    # 1. Assert `config.parameter_count > 0`.
    # 2. Assert `config.bytes_per_parameter > 0`.
    # 3. Multiply parameters by bytes per parameter.
    # 4. Convert bytes to MB.
    # 5. Return the float value.
    #
    # Checks:
    # - With `1_000` parameters and `2` bytes each, the result should be about `0.0019 MB`.
    #
    raise NotImplementedError("Return parameter memory in MB.")


# Estimate optimizer-state memory.
# Expected return:
# - memory in megabytes
def estimate_optimizer_memory_mb(config: DistributedTransformerConfig) -> float:
    """Estimate optimizer state memory."""
    # Goal:
    # - Estimate extra memory used by optimizer state.
    # - Optimizer state means extra tensors the optimizer keeps, such as Adam's running averages.
    #
    # Inputs:
    # - `config.parameter_count`: number of learned parameters.
    # - `config.bytes_per_parameter`: bytes used by one parameter.
    # - `config.optimizer_multiplier`: how many times bigger optimizer state is than raw weights.
    # - Example:
    #   - if weights use `100 MB`
    #   - and `optimizer_multiplier = 8.0`
    #   - optimizer memory estimate is `800 MB`
    #
    # Output:
    # - Return one `float`.
    # - Unit: megabytes.
    #
    # Method:
    # - First call:
    #   `weight_memory_mb = estimate_weight_memory_mb(config)`
    # - Then use:
    #   `optimizer_memory_mb = weight_memory_mb * config.optimizer_multiplier`
    # - The multiplier is a rough teaching estimate. Real memory depends on optimizer,
    #   precision, gradients, and framework details.
    #
    # Steps:
    # 1. Assert `config.optimizer_multiplier >= 0`.
    # 2. Compute weight memory with `estimate_weight_memory_mb(config)`.
    # 3. Multiply by `config.optimizer_multiplier`.
    # 4. Return the float value.
    #
    # Checks:
    # - If weight memory is `100 MB` and multiplier is `8.0`, result should be `800 MB`.
    #
    raise NotImplementedError("Return optimizer memory in MB.")


# Compare replicated memory with sharded memory.
# Expected return:
# - dictionary with per-device memory estimates
def compare_replicated_and_sharded_memory(config: DistributedTransformerConfig) -> MemoryEstimates:
    """Compare replicated and sharded training memory."""
    # Goal:
    # - Compare memory when every device stores all training state versus when state is split.
    #
    # Inputs:
    # - `config.parameter_count`: model size.
    # - `config.bytes_per_parameter`: bytes per weight.
    # - `config.optimizer_multiplier`: optimizer-state estimate.
    # - `config.num_devices`: number of devices used for sharding.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"weight_memory_mb"`: model weights only
    #   - `"optimizer_memory_mb"`: optimizer state only
    #   - `"replicated_per_device_mb"`: weights plus optimizer state on every device
    #   - `"sharded_per_device_mb"`: weights plus optimizer state split across devices
    #   - `"saving_per_device_mb"`: replicated minus sharded
    # - All values use MB.
    #
    # Method:
    # - Use:
    #   `weight_memory_mb = estimate_weight_memory_mb(config)`
    #   `optimizer_memory_mb = estimate_optimizer_memory_mb(config)`
    #   `replicated_per_device_mb = weight_memory_mb + optimizer_memory_mb`
    #   `sharded_per_device_mb = replicated_per_device_mb / config.num_devices`
    #   `saving_per_device_mb = replicated_per_device_mb - sharded_per_device_mb`
    # - Replicated means each device stores the full state.
    # - Sharded means each device stores only one slice of the state.
    # - This is a simplified estimate. Real FSDP and ZeRO also have temporary all-gather
    #   memory, gradients, activations, and communication buffers.
    #
    # Steps:
    # 1. Assert `config.num_devices > 0`.
    # 2. Compute weight memory.
    # 3. Compute optimizer memory.
    # 4. Add them to get replicated memory per device.
    # 5. Divide by device count to get sharded memory per device.
    # 6. Compute the per-device saving.
    # 7. Return the dictionary with the exact keys above.
    #
    # Checks:
    # - If replicated memory is `900 MB` and there are `3` devices,
    #   sharded memory should be `300 MB` per device.
    # - `set(result)` should equal the five output keys listed above.
    #
    raise NotImplementedError("Return replicated and sharded memory estimates.")


# Describe data parallelism.
# Expected return:
# - plain-language dictionary that says what is copied and what is split
def describe_data_parallelism() -> ParallelismDescription:
    """Describe data parallelism."""
    # Goal:
    # - Explain what data parallelism copies and what it splits.
    #
    # Inputs:
    # - No direct arguments.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"name"`: `"data parallelism"`
    #   - `"copied"`: what each device keeps a full copy of
    #   - `"split"`: what is divided across devices
    #   - `"communication"`: what devices share after backward pass
    #   - `"plain_example"`: one small example with 2 devices
    #
    # Method:
    # - Write plain strings.
    # - Use this idea:
    #   - copied: full model weights
    #   - split: training batches
    #   - communication: gradients are averaged
    #
    # Steps:
    # 1. Create the dictionary with the exact keys above.
    # 2. In `"plain_example"`, say that device 0 gets batch A and device 1 gets batch B.
    # 3. Say both devices still hold the full model.
    # 4. Return the dictionary.
    #
    # Checks:
    # - The `"copied"` value should mention model weights.
    # - The `"split"` value should mention batches or data.
    #
    raise NotImplementedError("Explain data parallelism in simple words.")


# Describe tensor parallelism.
# Expected return:
# - plain-language dictionary that says what is split
def describe_tensor_parallelism() -> ParallelismDescription:
    """Describe tensor parallelism."""
    # Goal:
    # - Explain how one transformer layer's math can be split across devices.
    #
    # Inputs:
    # - No direct arguments.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"name"`: `"tensor parallelism"`
    #   - `"split"`: what part of the layer is split
    #   - `"copied"`: what may still be copied
    #   - `"communication"`: what devices need to exchange
    #   - `"plain_example"`: one small matrix example
    #
    # Method:
    # - Use this small example:
    #   - a linear layer has weight shape `[8, 4]`
    #   - with 2 devices, device 0 can hold rows `[0:4]`
    #   - device 1 can hold rows `[4:8]`
    #   - each device computes part of the output
    # - In real transformer code, tensor parallelism often splits attention projection
    #   matrices or MLP matrices.
    #
    # Steps:
    # 1. Create the dictionary with the exact keys above.
    # 2. Say that big matrix operations inside layers are split.
    # 3. Say that devices must communicate partial outputs or partial gradients.
    # 4. Return the dictionary.
    #
    # Checks:
    # - The `"split"` value should mention a matrix, projection, attention, or MLP.
    # - The `"communication"` value should mention partial outputs or gradients.
    #
    raise NotImplementedError("Explain tensor parallelism in simple words.")


# Describe pipeline parallelism.
# Expected return:
# - plain-language dictionary that says how layers are placed
def describe_pipeline_parallelism() -> ParallelismDescription:
    """Describe pipeline parallelism."""
    # Goal:
    # - Explain how different transformer layers can live on different devices.
    #
    # Inputs:
    # - No direct arguments.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"name"`: `"pipeline parallelism"`
    #   - `"split"`: which part of the model is split
    #   - `"copied"`: what may still be copied
    #   - `"communication"`: what moves between devices
    #   - `"plain_example"`: one small layer-placement example
    #
    # Method:
    # - Use this small example:
    #   - a 4-layer transformer runs on 2 devices
    #   - device 0 stores layers 0 and 1
    #   - device 1 stores layers 2 and 3
    #   - activations from layer 1 are sent to device 1
    # - Activation means the tensor output from a layer.
    #
    # Steps:
    # 1. Create the dictionary with the exact keys above.
    # 2. Say that layers are split across devices.
    # 3. Say activations move forward, and gradients move backward.
    # 4. Return the dictionary.
    #
    # Checks:
    # - The `"split"` value should mention layers.
    # - The `"communication"` value should mention activations.
    #
    raise NotImplementedError("Explain pipeline parallelism in simple words.")


# Describe FSDP or ZeRO-style sharding.
# Expected return:
# - plain-language dictionary that says what gets sharded
def describe_sharded_training() -> ParallelismDescription:
    """Describe sharded transformer training."""
    # Goal:
    # - Explain FSDP or ZeRO-style sharding in plain words.
    #
    # Inputs:
    # - No direct arguments.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"name"`: `"sharded training"`
    #   - `"split"`: what training state is split
    #   - `"copied"`: what may still be copied or temporarily gathered
    #   - `"communication"`: what devices need to gather or reduce
    #   - `"plain_example"`: one small state-splitting example
    #
    # Method:
    # - Use this simple example:
    #   - optimizer state is `800 MB`
    #   - 4 devices shard it
    #   - each device stores about `800 / 4 = 200 MB`
    # - FSDP can shard model weights, gradients, and optimizer state.
    # - ZeRO has stages:
    #   - Stage 1 shards optimizer state
    #   - Stage 2 shards optimizer state and gradients
    #   - Stage 3 shards optimizer state, gradients, and parameters
    #
    # Steps:
    # 1. Create the dictionary with the exact keys above.
    # 2. Say that sharding splits training state, not the dataset.
    # 3. Mention weights, gradients, and optimizer state.
    # 4. Return the dictionary.
    #
    # Checks:
    # - The `"split"` value should mention optimizer state.
    # - The `"plain_example"` value should include a concrete divided number.
    #
    raise NotImplementedError("Explain FSDP or ZeRO-style sharding.")


# Choose a training strategy for the toy setup.
# Expected return:
# - a short plan that names the strategy and why
def choose_training_strategy(config: DistributedTransformerConfig) -> str:
    """Choose a simple distributed training strategy."""
    # Goal:
    # - Choose a reasonable training layout for the toy transformer.
    # - Explain the choice without overclaiming.
    #
    # Inputs:
    # - `config.parameter_count`: toy model size.
    # - `config.num_devices`: number of devices available.
    # - `compare_replicated_and_sharded_memory(config)` returns the memory numbers to compare.
    #
    # Output:
    # - Return one short string.
    # - The note should name:
    #   - the chosen strategy
    #   - what is copied
    #   - what is split
    #   - why this is useful for memory
    #
    # Method:
    # - Use the memory helper:
    #   `memory = compare_replicated_and_sharded_memory(config)`
    # - For this toy setup, a good default answer is:
    #   - use data parallelism for batches
    #   - add FSDP or ZeRO-style sharding for model and optimizer state
    # - If `config.num_devices == 1`, say that sharding does not help because there is
    #   only one device to split across.
    #
    # Steps:
    # 1. Assert `config.num_devices > 0`.
    # 2. Compute `memory = compare_replicated_and_sharded_memory(config)`.
    # 3. If `config.num_devices == 1`, return a note saying single-device training is enough here.
    # 4. Otherwise, return a note that recommends data parallelism plus sharding.
    # 5. Include the replicated and sharded per-device MB numbers in the note.
    #
    # Checks:
    # - The note should mention `"shard"` or `"sharding"` when `num_devices > 1`.
    # - The note should include a concrete MB number from `memory`.
    #
    raise NotImplementedError("Return a plain-language strategy note.")


# Run the full concept workflow.
# Expected return:
# - summary dictionary with estimates and descriptions
def run_distributed_transformer_demo(config: DistributedTransformerConfig) -> DistributedTransformerSummary:
    """Run the distributed transformer training concept demo."""
    # Goal:
    # - Collect all estimates and explanations into one summary dictionary.
    #
    # Inputs:
    # - `config`: model size, memory, and device-count settings.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"memory_estimates"`: output of `compare_replicated_and_sharded_memory(config)`
    #   - `"data_parallel"`: output of `describe_data_parallelism()`
    #   - `"tensor_parallel"`: output of `describe_tensor_parallelism()`
    #   - `"pipeline_parallel"`: output of `describe_pipeline_parallelism()`
    #   - `"sharded_training"`: output of `describe_sharded_training()`
    #   - `"strategy_note"`: output of `choose_training_strategy(config)`
    #
    # Method:
    # - Use this exact call shape:
    #   `memory_estimates = compare_replicated_and_sharded_memory(config)`
    #   `data_parallel = describe_data_parallelism()`
    #   `tensor_parallel = describe_tensor_parallelism()`
    #   `pipeline_parallel = describe_pipeline_parallelism()`
    #   `sharded_training = describe_sharded_training()`
    #   `strategy_note = choose_training_strategy(config)`
    #   `return {...}`
    #
    # Steps:
    # 1. Call each helper once.
    # 2. Store each result under the exact output key listed above.
    # 3. Return the summary dictionary.
    #
    # Checks:
    # - `set(summary)` should equal the six output keys listed above.
    # - `summary["memory_estimates"]` should contain `"replicated_per_device_mb"`.
    #
    raise NotImplementedError("Return memory estimates, descriptions, and strategy note.")


# Check that the summary includes all major concepts.
# Expected return:
# - True when the summary is complete
def check_metrics(metrics: DistributedTransformerSummary) -> bool:
    """Return True when distributed training outputs are present."""
    required = {"memory_estimates", "data_parallel", "tensor_parallel", "pipeline_parallel", "sharded_training", "strategy_note"}
    return required.issubset(metrics)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the distributed transformer scaffold."""
    config = DistributedTransformerConfig()

    try:
        metrics = run_distributed_transformer_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "distributed_transformer_outputs", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
