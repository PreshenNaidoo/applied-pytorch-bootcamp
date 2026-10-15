"""
============================================================
EXERCISE 21: QUANTIZATION AND QLORA BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny quantization comparison.
You are not building a full QLoRA trainer in this exercise.

Plain-language meaning:
- quantization stores model numbers with fewer bits
- fewer bits usually means less memory
- quality can drop if the approximation is too rough
- QLoRA means using a quantized base model with LoRA adapters

Important scope note:
- this exercise uses quantization in code
- this exercise only explains QLoRA in words
- QLoRA is not happening behind the scenes
- PyTorch dynamic quantization does not secretly add LoRA
- this file does not create LoRA adapter weights
- this file does not freeze a base model and train adapters
- this file does not use the LoRA formula in code
- the only QLoRA part is the `explain_qlora` function, which is a plain-English concept check

What real QLoRA would need:
- a large base model
- quantized base weights
- extra LoRA adapter weights
- training code that updates only the adapter weights

Tiny LoRA idea, not used in this file:
- normal layer uses the original weight `W`
- LoRA adds a small trainable update
- rough formula: `new effect = W + A @ B`
- `A` and `B` are small trainable matrices
- this exercise does not build `A` or `B`

WHY THIS EXERCISE EXISTS
Large models can be too big for one GPU.
Quantization helps fit models into less memory.
This exercise teaches:
- model size estimates
- float32 vs lower-bit storage
- simple dynamic quantization
- quantization error
- the basic QLoRA idea

HOW THIS FITS INTO THE BOOTCAMP
After pretrained models, LoRA, and inference-focused topics,
this file teaches one more practical efficiency idea:
- use fewer bits per weight
- and understand the tradeoff between size and quality

Beginner intuition:
- model weights are numbers
- if each number uses fewer bits, the model takes less memory
- the cost is that those numbers become a rougher approximation

Small example:
- float32 stores more detail per weight
- int8 stores less detail per weight
- int8 can save memory, but outputs may change a bit

DATASET
No large dataset is required for the main comparison.
Use one fixed real input batch from a small dataset.

Clear default:
- `sklearn.datasets.load_digits`
- input features: flattened `8 x 8` digit images
- input shape after flattening: `[batch_size, 64]`
- labels are available, but the main comparison only needs model outputs

HOW TO CREATE THE INPUTS

Simple path:
- build a small model
- load one fixed batch from `sklearn.datasets.load_digits`
- run the model before quantization
- run the model after quantization
- compare output difference and size
What "quantization" means in plain words:
- model weights are stored as numbers
- float32 uses more bits per number
- int8 uses fewer bits per number
- fewer bits means less storage
- fewer bits can also mean rougher approximations
Small concrete example:
- if a model has 1,000,000 parameters
- float32 uses about 32 bits per parameter
- int8 uses about 8 bits per parameter
- the int8 version should take about one quarter of the raw weight storage

STEP-BY-STEP TODO
1. Build or load a small model.
2. Count parameters.
3. Estimate float32 model size.
4. Quantize the model or simulate lower-bit storage.
5. Compare output differences before and after quantization.
6. Report memory savings.
7. Explain the QLoRA idea in plain words.

DEFINITION OF DONE
You are done when:
- original model size is estimated
- quantized model size is estimated
- output difference is measured
- memory saving is reported
- you can explain why quantization is useful

INPUTS AND OUTPUTS
Input:
- model
- fixed input batch
Output:
- parameter count
- float32 size estimate
- quantized size estimate
- output difference summary

METRICS TO TRACK
Track at least:
- parameter count
- float32 size estimate
- quantized size estimate
- size reduction percent
- mean output difference
- one plain-language QLoRA note

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small MLP or classifier.
Keep the model simple so the storage tradeoff is easy to see.

LOSS FUNCTION
No training loss is required for the main quantization comparison.

TARGET RESULT RANGE
Focus on the storage tradeoff, not on a magic score.

Useful signs:
- quantized size is clearly smaller than float32 size
- output difference is not zero but stays reasonably small
- you can explain why smaller storage might still change outputs a bit

COMMON MISTAKES
- comparing different input batches before and after quantization
- expecting outputs to stay exactly identical
- mixing up model size on disk with every detail of runtime memory
- treating QLoRA as exactly the same thing as plain inference-time quantization

What each step is doing in simple words:
1. build a small model
2. count how many weights it has
3. estimate storage with float32
4. make a lower-bit version
5. compare outputs on the same inputs
6. report memory saved
7. explain how QLoRA uses a quantized base plus small trainable adapters
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict
import torch
import torch.nn as nn

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again

# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

@dataclass
class QuantizationConfig:
    """Settings for quantization basics."""
    input_dim: int = 64
    hidden_dim: int = 128
    num_classes: int = 4
    batch_size: int = 16
    simulated_bits: int = 8
    random_state: int = 42


class QuantizationPlanSummary(TypedDict):
    """Main settings and idea for the quantization exercise."""

    baseline_dtype: str
    simulated_bits: int
    main_tradeoff: str
    qlora_plain_meaning: str


class OutputDifferenceSummary(TypedDict):
    """Output differences between two model versions."""

    mean_output_difference: float
    max_output_difference: float


class QuantizationRunSummary(TypedDict):
    """Final summary returned by the quantization demo."""

    parameter_count: float
    float32_size_mb: float
    quantized_size_mb: float
    size_reduction_percent: float
    mean_output_difference: float
    max_output_difference: float
    qlora_note: str


def describe_quantization_plan(config: QuantizationConfig) -> QuantizationPlanSummary:
    """Return the main quantization checks."""
    return {
        "baseline_dtype": "float32",
        "simulated_bits": config.simulated_bits,
        "main_tradeoff": "less memory, possible quality loss",
        "qlora_plain_meaning": "quantized base model plus small trainable LoRA adapters",
    }

# Estimate model size from:
# - parameter count
# - bits per parameter
# This is a simple storage estimate, not a full runtime memory model.
def estimate_model_size_mb(parameter_count: int, bits_per_parameter: int) -> float:
    """Estimate model size in MB."""
    # Goal:
    # Estimate how much storage the model weights use.
    # This is a simple weight-only estimate, not full runtime memory.
    #
    # Input:
    # - `parameter_count`: int
    #   Number of model parameters.
    # - `bits_per_parameter`: int
    #   How many bits are used to store each parameter.
    #
    # Output:
    # - `float`
    # - Estimated size in megabytes.
    #
    # Formula:
    # ```text
    # total_bits = parameter_count * bits_per_parameter
    # total_bytes = total_bits / 8
    # size_mb = total_bytes / (1024 * 1024)
    # ```
    #
    # Meaning:
    # - 8 bits make 1 byte.
    # - 1024 * 1024 bytes make 1 MiB, often called MB in model-size notes.
    #
    # Steps:
    # 1. Compute `total_bits`.
    # 2. Divide by `8` to get bytes.
    # 3. Divide by `1024 * 1024` to get MB.
    # 4. Return the result.
    #
    # Checks:
    # - `estimate_model_size_mb(1_000_000, 32)` should be about `3.8147`.
    # - `estimate_model_size_mb(1_000_000, 8)` should be about `0.9537`.

    raise NotImplementedError("Estimate size from parameter count and bits per parameter.")



# Beginner hints for this file:
# - quantization is about storing model numbers more compactly
# - the main tradeoff is smaller memory vs possible output error
# - start with simple size estimates before worrying about advanced kernels
# - if the quantized model output changes too much, quality may drop
# - QLoRA combines a quantized base model with small trainable LoRA adapters
# Build one small model for the size and output comparison.
# Keep it simple so the storage change is easy to understand.
class QuantizationToyModel(nn.Module):
    """Small model used for quantization practice."""
    def __init__(self, config: QuantizationConfig) -> None:
        # Goal:
        # Build a small MLP model.
        # MLP means a simple stack of linear layers with an activation in between.
        #
        # Input:
        # - `config`: QuantizationConfig
        #
        # Output:
        # - No return value.
        # - The model should accept input shape `[batch_size, config.input_dim]`.
        # - The model should return logits with shape `[batch_size, config.num_classes]`.
        #
        # Method:
        # Use:
        # `nn.Sequential(`
        # `    nn.Linear(config.input_dim, config.hidden_dim),`
        # `    nn.ReLU(),`
        # `    nn.Linear(config.hidden_dim, config.num_classes),`
        # `)`
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Create `self.net` with:
        #    `nn.Sequential(nn.Linear(config.input_dim, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, config.num_classes))`.
        # 3. Do not add softmax at the end.
        #
        # Checks:
        # - A fake input with shape `[2, config.input_dim]` should return `[2, config.num_classes]`.
        # - Return raw logits.
        #
        super().__init__()

        raise NotImplementedError("Build a small MLP model.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        # Goal:
        # Return model outputs for one input batch.
        #
        # Input:
        # - `x`: tensor with shape `[batch_size, config.input_dim]`
        #
        # Output:
        # - logits tensor with shape `[batch_size, config.num_classes]`
        #
        # Method:
        # Use:
        # `return self.net(x)`
        #
        # Steps:
        # 1. Pass `x` through `self.net`.
        # 2. Return the logits.
        #
        # Checks:
        # - Do not apply softmax here.


        raise NotImplementedError("Return model outputs.")

# Count how many parameters the model has in total.
# This is the starting point for every size estimate.
def count_parameters(model: nn.Module) -> int:
    """Return total parameter count."""
    # Goal:
    # Count all trainable and frozen parameters in the model.
    #
    # Input:
    # - `model`: nn.Module
    #
    # Output:
    # - `int`
    # - Total number of scalar parameters.
    #
    # Method:
    # Use:
    # `sum(parameter.numel() for parameter in model.parameters())`
    #
    # Meaning:
    # - `model.parameters()` loops over model weights and biases.
    # - `parameter.numel()` counts scalar numbers inside one parameter tensor.
    #
    # Steps:
    # 1. Loop over all parameters.
    # 2. Add up `parameter.numel()`.
    # 3. Return the count.
    #
    # Checks:
    # - For `nn.Linear(3, 2)`, the answer is `3 * 2 + 2 = 8`.

    raise NotImplementedError("Count model parameters.")
    # `param` is not a whole layer.
    # It is one trainable tensor inside a layer.
    # A layer can own more than one trainable tensor.
    # Example: `nn.Linear(3, 2)` has:
    # - one weight tensor with shape `[2, 3]`, which holds 6 numbers
    # - one bias tensor with shape `[2]`, which holds 2 numbers
    # When `model.parameters()` reaches that linear layer, the loop sees them
    # one at a time:
    # - first loop: `param` is the weight tensor
    # - second loop: `param` is the bias tensor
    # So `len(list(model.parameters()))` would count 2 tensors.
    # But the model has 8 scalar parameters: 6 weights + 2 bias values.
    # `param.numel()` means "how many numbers are inside this tensor".


# Return a quantized model or a simple simulated quantized copy.
# The important thing is that this path uses fewer bits than float32.
def quantize_model(model: nn.Module, config: QuantizationConfig) -> nn.Module:
    """Return a quantized model or a simple simulated quantized copy."""
    # Goal:
    # Return a lower-bit version of the model or a simulated lower-bit copy.
    # The important idea is that the weights are stored with fewer bits than float32.
    #
    # Input:
    # - `model`: nn.Module
    # - `config`: QuantizationConfig
    #
    # Output:
    # - `nn.Module`
    # - A model you can call with the same input shape as the original model.
    #
    # Method:
    # Simple PyTorch dynamic quantization path:
    # `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`
    #
    # Meaning:
    # - `{nn.Linear}` says to quantize linear layers.
    # - `dtype=torch.qint8` uses signed 8-bit integer weights.
    #
    # If dynamic quantization is not available in your environment,
    # make a deep copy and round weights to simulate lower precision.
    # Do not modify the original model in-place.
    #
    # Steps:
    # 1. Put the model in eval mode with `model.eval()`.
    # 2. Try dynamic quantization with the call above.
    # 3. Return the quantized model.
    # 4. If you simulate instead, copy the model first.
    # 5. Return the copied simulated model.
    #
    # Checks:
    # - The returned model should accept `[batch_size, config.input_dim]`.
    # - The returned model should return `[batch_size, config.num_classes]`.
    # - The original model should still work after this function.

    raise NotImplementedError("Quantize the model or simulate quantization.")

    # Alternate simulated path, if dynamic quantization is not available.
    # Leave this commented out unless you want to test the fallback idea.
    #
    # This does not create a real int8 model.
    # It only rounds the float32 weights so they act less precise.
    # That makes it useful for learning the idea, but not for measuring real
    # PyTorch int8 model speed or storage.
    #
    # import copy
    #
    # simulated_model = copy.deepcopy(model)
    # simulated_model.eval()
    #
    # levels = 2 ** config.simulated_bits
    #
    # with torch.no_grad():
    #     for param in simulated_model.parameters():
    #         min_value = param.min()
    #         max_value = param.max()
    #
    #         # If all values are the same, rounding would divide by zero.
    #         # In that case, leave that tensor unchanged.
    #         if min_value == max_value:
    #             continue
    #
    #         scale = (max_value - min_value) / (levels - 1)
    #
    #         # Move values so the smallest value starts at 0.
    #         # Round to the nearest level.
    #         # Move values back to the original range.
    #         rounded = torch.round((param - min_value) / scale) * scale + min_value
    #         param.copy_(rounded)
    #
    # return simulated_model


# Compare original and quantized outputs on the same input batch.
# You are not expecting perfect equality.
# The useful question is whether the difference stays reasonably small.
@torch.no_grad()
def compare_outputs(original_model: nn.Module, quantized_model: nn.Module, inputs: torch.Tensor) -> OutputDifferenceSummary:
    """Compare original and quantized model outputs."""
    # Goal:
    # Compare original and quantized model outputs on the exact same input batch.
    #
    # Input:
    # - `original_model`: nn.Module
    # - `quantized_model`: nn.Module
    # - `inputs`: tensor with shape `[batch_size, input_dim]`
    #
    # Output:
    # Return this exact dictionary:
    # - `"mean_output_difference"`: float
    # - `"max_output_difference"`: float
    #
    # Difference means absolute difference:
    # `abs(original_outputs - quantized_outputs)`
    #
    # Method:
    # Use:
    # - `original_model.eval()`
    # - `quantized_model.eval()`
    # - `with torch.no_grad():`
    # - `torch.abs(original_outputs - quantized_outputs)`
    #
    # Steps:
    # 1. Put both models in eval mode.
    # 2. Run both models on the same `inputs`.
    # 3. Compute `difference = torch.abs(original_outputs - quantized_outputs)`.
    # 4. Compute `difference.mean().item()`.
    # 5. Compute `difference.max().item()`.
    # 6. Return the dictionary with the exact keys listed above.
    #
    # Checks:
    # - Use the same input tensor for both models.
    # - Do not expect the difference to be exactly zero.

    raise NotImplementedError("Return output difference between original and quantized models.")




# Explain QLoRA in plain words.
# Important:
# This exercise does not actually build LoRA adapters.
# It only uses quantization and asks for a short QLoRA explanation.
#
# If LoRA were implemented, you would see extra trainable adapter weights.
# You would also see the LoRA idea:
# original weight output + small adapter update
#
# Keep the explanation simple:
# - quantized base model
# - small LoRA adapters
# - cheaper adaptation than full fine-tuning
def explain_qlora(config: QuantizationConfig) -> str:
    """Return a short plain-language QLoRA explanation."""
    # Goal:
    # Explain QLoRA in simple words.
    # This is a concept check only.
    # No LoRA adapter layers are created in this exercise.
    #
    # Input:
    # - `config`: QuantizationConfig
    #
    # Output:
    # - `str`
    # - One or two short sentences.
    #
    # Method:
    # Mention three parts:
    # - the base model is quantized to save memory
    # - the base weights stay mostly frozen
    # - small LoRA adapter weights are trained
    #
    # What is not happening here:
    # - This function does not create LoRA weights.
    # - This function does not train adapters.
    # - This exercise does not use the LoRA formula in code.
    #
    # Steps:
    # 1. Read `config.simulated_bits`.
    # 2. Write one plain sentence about using a quantized base model.
    # 3. Write one plain sentence about training only small LoRA adapters.
    # 4. Return the string.
    #
    # Checks:
    # - The note should include the words `quantized` and `LoRA`.
    # - The note should be easy to say out loud.

    raise NotImplementedError("Explain QLoRA in simple words.")

# Run the full quantization comparison.
# Return a summary with:
# - size before
# - size after
# - output difference
# - QLoRA note
def run_quantization_demo(config: QuantizationConfig) -> QuantizationRunSummary:
    """Run the full quantization comparison."""
    # Goal:
    # Run the full quantization comparison and return a short summary.
    #
    # Input:
    # - `config`: QuantizationConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"parameter_count"`: float
    # - `"float32_size_mb"`: float
    # - `"quantized_size_mb"`: float
    # - `"size_reduction_percent"`: float
    # - `"mean_output_difference"`: float
    # - `"max_output_difference"`: float
    # - `"qlora_note"`: str
    #
    # Method:
    # Use helper functions from this file.
    # Use one fixed input batch so the before/after comparison is fair.
    #
    # Fixed input batch:
    # `inputs = torch.randn(config.batch_size, config.input_dim)`
    #
    # Size reduction formula:
    # ```text
    # size_reduction_percent = 100 * (float32_size_mb - quantized_size_mb) / float32_size_mb
    # ```
    #
    # Steps:
    # 1. Build `model = QuantizationToyModel(config)`.
    # 2. Count parameters with `count_parameters(model)`.
    # 3. Estimate float32 size with `estimate_model_size_mb(parameter_count, 32)`.
    # 4. Estimate quantized size with `estimate_model_size_mb(parameter_count, config.simulated_bits)`.
    # 5. Build one fixed input batch with shape `[config.batch_size, config.input_dim]`.
    # 6. Build `quantized_model = quantize_model(model, config)`.
    # 7. Compare outputs with `compare_outputs(model, quantized_model, inputs)`.
    # 8. Compute `size_reduction_percent`.
    # 9. Build `qlora_note = explain_qlora(config)`.
    # 10. Return the dictionary with the exact keys listed above.
    #
    # Checks:
    # - `quantized_size_mb < float32_size_mb`.
    # - `mean_output_difference >= 0.0`.
    # - `check_metrics(metrics)` should return True.

    raise NotImplementedError("Return model sizes, output difference, and QLoRA note.")







def check_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when quantization summary values are present."""
    required = {"float32_size_mb", "quantized_size_mb", "size_reduction_percent", "mean_output_difference", "qlora_note"}
    return required.issubset(metrics)

def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True

def main() -> None:
    """Run the quantization scaffold."""
    config = QuantizationConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 4
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    _, ok = report_task_progress("describe_quantization_plan", describe_quantization_plan, config)
    if not count_or_stop(ok):
        return
    model, ok = report_task_progress("QuantizationToyModel.__init__", QuantizationToyModel, config)
    if not count_or_stop(ok):
        return
    sample_x = torch.randn(config.batch_size, config.input_dim)
    _, ok = report_task_progress("QuantizationToyModel.forward", model, sample_x)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_quantization_demo", run_quantization_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "quantization summary", float(int(passed)), "quantization keys present")
if __name__ == "__main__":
    main()
