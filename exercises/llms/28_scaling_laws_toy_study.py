"""
============================================================
EXERCISE 28: SCALING LAWS TOY STUDY
============================================================

WHAT YOU ARE BUILDING
You are building a tiny experiment that studies how loss changes with model size and data size.

Plain-language meanings:
- scaling law means a pattern that connects loss with compute, data, or model size
- parameters are learned numbers inside the model
- tokens are the text pieces the model trains on
- compute is the amount of work used for training
- FLOPs means floating point operations, a rough count of math work

WHY THIS EXERCISE EXISTS
Frontier labs care about how much model quality improves when they use more data,
bigger models, or more compute.
You cannot train a frontier model here, but you can learn the shape of the question.

This exercise teaches:
- running the same task at several model sizes
- running the same task at several data sizes
- recording loss in a table
- estimating simple compute
- writing a careful note about what the toy study can and cannot prove

HOW THIS FITS INTO THE BOOTCAMP
Earlier files taught how to train one model.
This file asks a research question across many small runs.

Beginner intuition:
- one run tells you one result
- a sweep tells you a pattern
- bigger is often better, but it costs more
- the useful question is how much better you get for the extra cost

DATASET
Use Tiny Shakespeare as the recommended small text dataset.

Clear default:
- load Tiny Shakespeare and tokenize it into character or word ids
- if the file is missing, use a tiny fallback text string
- use several data sizes, such as `256`, `512`, and `1024` tokens

WHY A SMALL REAL TEXT DATASET IS USEFUL HERE
This exercise is about experiment shape, not text quality.
Tiny Shakespeare gives real repeated text patterns while still being small enough for quick sweeps.

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load Tiny Shakespeare from `data/`
- build a small vocabulary from the training text
- turn the text into one long tensor of token ids
- slice the first `data_size` tokens for each run
- make fixed-length next-token windows
- train each model size on the same kind of data
- record validation loss for every setting

INPUTS AND OUTPUTS
Input:
- model sizes
- data sizes
- token tensors

Output:
- a result row for every model-size and data-size pair
- each row should include:
  - model dimension
  - data size
  - parameter count
  - estimated compute
  - validation loss

STEP-BY-STEP TODO
1. Choose several small model sizes.
2. Choose several small data sizes.
3. Train each setting for a short run.
4. Record validation loss.
5. Estimate parameter count.
6. Estimate token count.
7. Estimate simple compute.
8. Make a results table.
9. Write a note about the trend.

DEFINITION OF DONE
You are done when:
- more than one model size is tested
- more than one data size is tested
- a results table is returned
- the note explains the trend without overclaiming
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


ScalingRow = dict[str, float | int]
ScalingSummary = dict[str, list[ScalingRow] | str]


# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for the scaling-law sweep.
# Expected return:
# - a config object with model sizes, data sizes, and run length
@dataclass
class ScalingLawsToyConfig:
    """Settings for the toy scaling-law study."""

    model_dims: tuple[int, ...] = (32, 64, 128)
    data_sizes: tuple[int, ...] = (256, 512, 1024)
    vocab_size: int = 128
    block_size: int = 32
    train_steps: int = 10
    random_state: int = 42


# Build one tiny language model for a chosen size.
# Expected return:
# - a PyTorch model
def build_toy_lm(vocab_size: int, model_dim: int, block_size: int) -> nn.Module:
    """Build a tiny language model for one scaling setting."""
    # Goal:
    # - Build a very small next-token model.
    # - The model should take token ids and return one score for every possible next token.
    #
    # Inputs:
    # - `vocab_size`: number of possible token ids. If vocab_size is 128, valid ids are 0 to 127.
    # - `model_dim`: hidden size. Larger values make the model have more parameters.
    # - `block_size`: sequence length used by the training windows. Check it is positive.
    # - Example input to the returned model:
    #   `input_ids = torch.tensor([[3, 4, 5, 6]])`
    #   Shape: `[batch_size=1, sequence_length=4]`
    #
    # Output:
    # - Return an `nn.Module`.
    # - When called on `[batch_size, sequence_length]` token ids, it should return logits with
    #   shape `[batch_size, sequence_length, vocab_size]`.
    # - Logits are raw scores. Do not apply softmax before `nn.CrossEntropyLoss`.
    #
    # Method:
    # - Use `nn.Sequential(...)` from `torch.nn` so the model stays simple.
    # - Use this exact shape:
    #   `nn.Sequential(`
    #   `    nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim),`
    #   `    nn.Linear(in_features=model_dim, out_features=model_dim),`
    #   `    nn.GELU(),`
    #   `    nn.Linear(in_features=model_dim, out_features=vocab_size),`
    #   `)`
    # - `nn.Embedding` turns each token id into a vector of length `model_dim`.
    # - `nn.Linear(model_dim, model_dim)` mixes the vector for each position.
    # - The final `nn.Linear(model_dim, vocab_size)` produces one score per possible next token.
    # - This teaching model does not use attention. That is okay here because this file studies
    #   the shape of a sweep, not state-of-the-art language modeling.
    #
    # Steps:
    # 1. Assert `vocab_size > 0`, `model_dim > 0`, and `block_size > 0`.
    # 2. Build the `nn.Sequential(...)` model shown above.
    # 3. Return the model.
    #
    # Checks:
    # - Try `model(torch.tensor([[1, 2, 3, 4]]))`.
    # - The output shape should be `[1, 4, vocab_size]`.
    #
    raise NotImplementedError("Build a model whose size changes with model_dim.")


# Count trainable parameters.
# Expected return:
# - one integer parameter count
def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    # Goal:
    # - Count only the parameters that the optimizer can update.
    #
    # Inputs:
    # - `model`: any PyTorch `nn.Module`.
    # - Example: `nn.Linear(in_features=3, out_features=2)` has:
    #   - weight shape `[2, 3]`, so 6 numbers
    #   - bias shape `[2]`, so 2 numbers
    #   - total trainable parameters: 8
    #
    # Output:
    # - Return one `int`.
    # - The unit is parameters, meaning learned numbers.
    #
    # Method:
    # - Use:
    #   `sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)`
    # - `model.parameters()` gives each parameter tensor.
    # - `parameter.numel()` counts how many numbers are inside one tensor.
    # - `parameter.requires_grad` is True when PyTorch will compute gradients for that tensor.
    #
    # Steps:
    # 1. Loop over `model.parameters()`.
    # 2. Keep only tensors where `requires_grad` is True.
    # 3. Add up `numel()` for those tensors.
    # 4. Return the sum as an int.
    #
    # Checks:
    # - `count_trainable_parameters(nn.Linear(3, 2))` should return 8.
    #
    raise NotImplementedError("Count parameters where requires_grad is True.")


# Build token data for one data size.
# Expected return:
# - tensor of token ids
def build_scaling_token_data(num_tokens: int, config: ScalingLawsToyConfig) -> torch.Tensor:
    """Return toy token data for a chosen size."""
    # Goal:
    # - Create one long tensor of token ids for this data-size setting.
    # - Return one extra token so next-token labels can be made by shifting by one position.
    #
    # Inputs:
    # - `num_tokens`: number of next-token targets you want to train on.
    # - `config.vocab_size`: maximum number of token ids the model can read.
    # - Example: if `num_tokens=8`, return 9 ids such as
    #   `torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])`.
    #   The first 8 ids become inputs, and the last 8 ids become labels.
    #
    # Output:
    # - Return a `torch.long` tensor with shape `[num_tokens + 1]`.
    # - Every value must be between `0` and `config.vocab_size - 1`.
    #
    # Method:
    # - First try to load a local text file:
    #   `text_path = PROJECT_ROOT / "data" / "tiny_shakespeare.txt"`
    #   `text = text_path.read_text(encoding="utf-8")`
    # - If the file is missing, use a small fallback string such as:
    #   `"To be, or not to be, that is the question. " * 200`
    # - Convert characters to ids with:
    #   `chars = sorted(set(text))`
    #   `char_to_id = {char: index for index, char in enumerate(chars)}`
    #   `ids = torch.tensor([char_to_id[char] for char in text], dtype=torch.long)`
    # - Tiny Shakespeare normally has fewer than 128 unique characters, so the default
    #   `config.vocab_size=128` is large enough for this character vocabulary.
    #
    # Steps:
    # 1. Assert `num_tokens > 0`.
    # 2. Load the text from `data/tiny_shakespeare.txt`, or use the fallback text.
    # 3. Build `char_to_id` from the text.
    # 4. Assert `len(char_to_id) <= config.vocab_size`; otherwise the model vocabulary is too small.
    # 5. Convert the full text into a 1D `torch.long` tensor.
    # 6. If the tensor is shorter than `num_tokens + 1`, repeat it with `ids.repeat(repeats)`.
    # 7. Return `ids[: num_tokens + 1]`.
    #
    # Checks:
    # - For `num_tokens=8`, `token_ids.shape` should be `[9]`.
    # - `token_ids.dtype` should be `torch.long`.
    # - `int(token_ids.max()) < config.vocab_size` should be True.
    #
    raise NotImplementedError("Create toy token ids for the scaling study.")


# Make next-token windows from one token stream.
# Expected return:
# - input ids and labels with matching shapes
def make_next_token_windows(token_ids: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split token ids into input windows and next-token labels."""
    # Goal:
    # - Turn one long token stream into fixed-size training examples.
    # - Each input token should predict the token immediately after it.
    #
    # Input:
    # - `token_ids`: 1D `torch.long` tensor with shape `[num_tokens + 1]`.
    # - `block_size`: number of tokens per training example.
    # - Small example:
    #   `token_ids = torch.tensor([10, 11, 12, 13, 14])`
    #   `block_size = 4`
    #
    # Output:
    # - Return `(input_ids, labels)`.
    # - `input_ids` shape: `[num_windows, block_size]`
    # - `labels` shape: `[num_windows, block_size]`
    # - For the example above:
    #   `input_ids = tensor([[10, 11, 12, 13]])`
    #   `labels = tensor([[11, 12, 13, 14]])`
    #
    # Method:
    # - Use simple slicing:
    #   `inputs_1d = token_ids[:-1]`
    #   `labels_1d = token_ids[1:]`
    # - Keep only a whole number of windows:
    #   `usable_tokens = (inputs_1d.numel() // block_size) * block_size`
    # - Reshape with:
    #   `input_ids = inputs_1d[:usable_tokens].view(-1, block_size)`
    #   `labels = labels_1d[:usable_tokens].view(-1, block_size)`
    #
    # Steps:
    # 1. Assert `token_ids.ndim == 1`.
    # 2. Assert `token_ids.dtype == torch.long`.
    # 3. Assert `block_size > 0`.
    # 4. Build `inputs_1d` and `labels_1d` using the shift shown above.
    # 5. Compute `usable_tokens`.
    # 6. Assert `usable_tokens > 0`; otherwise there is not enough text for one window.
    # 7. Reshape both tensors to `[num_windows, block_size]`.
    # 8. Return `(input_ids, labels)`.
    #
    # Checks:
    # - `input_ids.shape == labels.shape` should be True.
    # - `input_ids.shape[1] == block_size` should be True.
    # - `labels[0, 0] == token_ids[1]` should be True.
    raise NotImplementedError("Make input and label windows for next-token training.")


# Train and evaluate one setting.
# Expected return:
# - dictionary with validation loss and simple counts
def run_one_scaling_setting(
    model_dim: int,
    data_size: int,
    config: ScalingLawsToyConfig,
) -> ScalingRow:
    """Run one model-size and data-size setting."""
    # Goal:
    # - Train one model size on one data size.
    # - Return one row for the scaling table.
    #
    # Inputs:
    # - `model_dim`: hidden size for this model, such as 32 or 64.
    # - `data_size`: number of next-token targets to use, such as 256 or 1024.
    # - `config`: sweep settings.
    # - `build_scaling_token_data(data_size, config)` returns shape `[data_size + 1]`.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"model_dim"`: int, the hidden size used for this run
    #   - `"data_size"`: int, the requested number of target tokens
    #   - `"parameter_count"`: int, trainable parameters in the model
    #   - `"token_count"`: int, number of tokens used across all training steps
    #   - `"estimated_compute"`: float, rough FLOPs from `estimate_toy_compute(...)`
    #   - `"validation_loss"`: float, average next-token loss after training
    #
    # Method:
    # - Build the model with:
    #   `model = build_toy_lm(config.vocab_size, model_dim, config.block_size)`
    # - Build token windows with:
    #   `token_ids = build_scaling_token_data(data_size, config)`
    #   `input_ids, labels = make_next_token_windows(token_ids, config.block_size)`
    # - Use AdamW:
    #   `optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)`
    # - Use cross entropy:
    #   `loss_fn = nn.CrossEntropyLoss()`
    # - For logits with shape `[batch, block, vocab_size]`, use:
    #   `loss = loss_fn(logits.reshape(-1, config.vocab_size), labels.reshape(-1))`
    # - The reshape turns all token positions into one long list of predictions.
    #
    # Steps:
    # 1. Assert `model_dim > 0` and `data_size >= config.block_size`.
    # 2. Build the model.
    # 3. Count trainable parameters.
    # 4. Build `token_ids`, then `input_ids` and `labels`.
    # 5. Put the model in training mode with `model.train()`.
    # 6. For each step in `range(config.train_steps)`:
    #    - run `logits = model(input_ids)`
    #    - compute the loss with the reshape shown above
    #    - call `optimizer.zero_grad()`
    #    - call `loss.backward()`
    #    - call `optimizer.step()`
    # 7. Put the model in eval mode with `model.eval()`.
    # 8. Use `with torch.no_grad():` to compute `validation_loss`.
    # 9. Set `token_count = int(input_ids.numel() * config.train_steps)`.
    # 10. Set `estimated_compute = estimate_toy_compute(parameter_count, token_count)`.
    # 11. Return the row dictionary with the exact keys listed above.
    #
    # Checks:
    # - `logits.shape` should be `[input_ids.shape[0], config.block_size, config.vocab_size]`.
    # - `set(row)` should equal the six output keys listed above.
    #
    raise NotImplementedError("Train one tiny model and return loss plus counts.")


# Estimate a simple compute number.
# Expected return:
# - rough compute estimate as a float
def estimate_toy_compute(parameter_count: int, token_count: int) -> float:
    """Return a rough compute estimate."""
    # Goal:
    # - Estimate the rough training compute for one setting.
    # - This is only a simple teaching estimate, not a hardware-accurate profiler number.
    #
    # Inputs:
    # - `parameter_count`: number of trainable model parameters.
    # - `token_count`: number of tokens used across all optimizer steps.
    # - Example: if one run has 1,000 parameters and trains on 200 tokens,
    #   `parameter_count=1000` and `token_count=200`.
    #
    # Output:
    # - Return one `float`.
    # - Unit: rough FLOPs, meaning rough math operations.
    #
    # Method:
    # - Use this common rough formula for transformer-style training:
    #   `estimated_compute = 6 * parameter_count * token_count`
    # - `6` is a rough multiplier for forward pass, backward pass, and gradient work.
    # - This toy model is not a full transformer, but using the same formula makes the
    #   scaling-law idea easy to compare with real training discussions.
    #
    # Steps:
    # 1. Assert both counts are non-negative.
    # 2. Compute `6 * parameter_count * token_count`.
    # 3. Convert the result to `float`.
    # 4. Return it.
    #
    # Checks:
    # - `estimate_toy_compute(1000, 200)` should return `1_200_000.0`.
    #
    raise NotImplementedError("Estimate compute from parameters and tokens.")


# Run the full sweep.
# Expected return:
# - list of result rows
def run_scaling_sweep(config: ScalingLawsToyConfig) -> list[ScalingRow]:
    """Run all model-size and data-size settings."""
    # Goal:
    # - Run every model-size and data-size pair.
    # - Return all rows in one table-like list.
    #
    # Inputs:
    # - `config.model_dims`: tuple such as `(32, 64, 128)`.
    # - `config.data_sizes`: tuple such as `(256, 512, 1024)`.
    # - With those defaults, there are `3 * 3 = 9` settings.
    #
    # Output:
    # - Return `list[ScalingRow]`.
    # - Length should be `len(config.model_dims) * len(config.data_sizes)`.
    # - Each row must have the exact keys returned by `run_one_scaling_setting(...)`.
    #
    # Method:
    # - Use two plain loops:
    #   `results = []`
    #   `for model_dim in config.model_dims:`
    #   `    for data_size in config.data_sizes:`
    #   `        row = run_one_scaling_setting(model_dim, data_size, config)`
    #   `        results.append(row)`
    #
    # Steps:
    # 1. Create an empty list called `results`.
    # 2. Loop through each `model_dim`.
    # 3. Inside that loop, loop through each `data_size`.
    # 4. Run one setting.
    # 5. Append the row to `results`.
    # 6. Return `results`.
    #
    # Checks:
    # - With the default config, `len(results)` should be 9.
    # - The first row should contain `"model_dim"`, `"data_size"`, and `"validation_loss"`.
    #
    raise NotImplementedError("Run the full toy scaling sweep.")


# Write a careful interpretation note.
# Expected return:
# - short plain-language string
def write_scaling_note(results: list[ScalingRow]) -> str:
    """Explain the trend in plain words."""
    # Goal:
    # - Explain what the rows seem to show.
    # - Keep the wording careful because this is a tiny toy study.
    #
    # Inputs:
    # - `results`: list of rows from `run_scaling_sweep(...)`.
    # - Each row has:
    #   `"model_dim"`, `"data_size"`, `"parameter_count"`, `"token_count"`,
    #   `"estimated_compute"`, and `"validation_loss"`.
    #
    # Output:
    # - Return one short plain-language string.
    # - It should mention:
    #   - which row had the lowest validation loss
    #   - whether larger models or more data helped in this toy run
    #   - that this small run cannot prove a real scaling law
    #
    # Method:
    # - Find the best row with:
    #   `best = min(results, key=lambda row: float(row["validation_loss"]))`
    # - Use `float(...)` because the row type allows both ints and floats.
    # - Build the sentence with concrete numbers from `best`.
    #
    # Steps:
    # 1. Assert `len(results) > 0`.
    # 2. Find the row with the lowest `"validation_loss"`.
    # 3. Write one or two sentences using that row's `model_dim`, `data_size`, and loss.
    # 4. Add a careful limit, such as:
    #    "This is only a tiny run, so it shows a pattern to inspect, not a real law."
    # 5. Return the note string.
    #
    # Checks:
    # - The note should include the words "loss" and "tiny" or "toy".
    # - The note should not claim that bigger models always win.
    #
    raise NotImplementedError("Write a careful note about the toy scaling trend.")


# Run the full scaling-law workflow.
# Expected return:
# - summary dictionary with rows and note
def run_scaling_law_demo(config: ScalingLawsToyConfig) -> ScalingSummary:
    """Run the toy scaling-law study."""
    # Goal:
    # - Run the full sweep and add a short interpretation note.
    #
    # Inputs:
    # - `config`: the settings for model sizes, data sizes, and train steps.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"results"`: list of scaling rows from `run_scaling_sweep(config)`
    #   - `"scaling_note"`: string from `write_scaling_note(results)`
    #
    # Method:
    # - Use:
    #   `results = run_scaling_sweep(config)`
    #   `scaling_note = write_scaling_note(results)`
    #   `return {"results": results, "scaling_note": scaling_note}`
    #
    # Steps:
    # 1. Run the sweep.
    # 2. Write the note from the sweep results.
    # 3. Return the summary dictionary with the exact two keys above.
    #
    # Checks:
    # - `set(summary)` should equal `{"results", "scaling_note"}`.
    # - `len(summary["results"])` should equal `len(config.model_dims) * len(config.data_sizes)`.
    #
    raise NotImplementedError("Run the sweep and return results plus a note.")


# Check that the sweep produced enough rows.
# Expected return:
# - True when all planned settings were run
def check_metrics(metrics: ScalingSummary, config: ScalingLawsToyConfig) -> bool:
    """Return True when the scaling summary is complete."""
    expected_rows = len(config.model_dims) * len(config.data_sizes)
    results = metrics.get("results")
    scaling_note = metrics.get("scaling_note")
    return isinstance(results, list) and len(results) == expected_rows and isinstance(scaling_note, str)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the toy scaling-law scaffold."""
    config = ScalingLawsToyConfig()
    set_seed(config.random_state)

    try:
        metrics = run_scaling_law_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics, config), "scaling_rows", int(check_metrics(metrics, config)), "1")


if __name__ == "__main__":
    main()
