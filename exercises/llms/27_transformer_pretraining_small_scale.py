"""
============================================================
EXERCISE 27: TRANSFORMER PRETRAINING SMALL SCALE
============================================================

WHAT YOU ARE BUILDING
You are building a tiny next-token pretraining loop for a decoder-only transformer.

Plain-language meanings:
- pretraining teaches a model a broad text task before a special task
- next-token prediction means predicting the next token from earlier tokens
- tokens per second measures training speed
- validation loss checks learning on held-out text
- a checkpoint saves model and optimizer state so training can resume

WHY THIS EXERCISE EXISTS
Frontier language models are trained with the same basic objective:
predict the next token.
The real systems are much larger, but the core loop should be clear first.

This exercise teaches:
- next-token batches
- training loss
- validation loss
- tokens-per-second tracking
- gradient accumulation
- checkpoint save and load
- resume after interruption

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises built small language models.
This file turns those ideas into a pretraining-style workflow.

Beginner intuition:
- the model reads tokens so far
- it guesses the next token
- the loss says how wrong the guess was
- many small guesses add up to broad language learning

DATASET
Use Tiny Shakespeare as the recommended small text dataset.

Clear default:
- use a Tiny Shakespeare plain-text file under `data/`
- if the file is missing, use a short built-in fallback corpus in the code
- keep the dataset small so the exercise runs quickly

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load one text string
- build a vocabulary from training text only
- turn text into token ids
- split token ids into train and validation sections
- make fixed-length token blocks
- for each block, create:
  - input ids
  - next-token labels

INPUTS AND OUTPUTS
Input batch:
- `input_ids`
  - shape: `[batch_size, block_size]`
  - dtype: `torch.long`
- `labels`
  - shape: `[batch_size, block_size]`
  - dtype: `torch.long`

Model output:
- raw logits with shape `[batch_size, block_size, vocab_size]`

What those shapes mean:
- each row is one text window
- each position predicts the next token
- logits are raw scores, not probabilities

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small decoder-only transformer.

Default path:
- token embeddings
- position embeddings
- causal self-attention blocks
- MLP blocks
- final vocabulary head

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

Important:
- do not apply softmax before the loss
- compare logits to next-token labels

STEP-BY-STEP TODO
1. Build tiny token blocks.
2. Build input ids and next-token labels.
3. Build a small decoder-only transformer.
4. Train for a few steps.
5. Evaluate validation loss.
6. Track tokens per second.
7. Save a checkpoint.
8. Load the checkpoint and continue.
9. Return a summary of the run.

DEFINITION OF DONE
You are done when:
- train and validation loss are reported
- tokens per second is reported
- a checkpoint can be saved
- training can resume from that checkpoint
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


device = "cuda" if torch.cuda.is_available() else "cpu"

# Store settings for tiny transformer pretraining.
# Expected return:
# - a config object with model, data, optimizer, and checkpoint settings
@dataclass
class SmallPretrainingConfig:
    """Settings for tiny transformer pretraining."""

    vocab_size: int = 128
    block_size: int = 32
    batch_size: int = 8
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    learning_rate: float = 3e-4
    max_steps: int = 20
    accumulation_steps: int = 2
    checkpoint_path: str = str(BOOTCAMP_ROOT / "outputs" / "tiny_pretraining_checkpoint.pt")
    random_state: int = 42


# Build toy token blocks.
# Expected return:
# - train loader
# - validation loader
def build_pretraining_loaders(config: SmallPretrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Return tiny train and validation dataloaders."""
    # Goal:
    # Build DataLoaders that return token windows for next-token prediction.
    #
    # Input:
    # - `config`: SmallPretrainingConfig
    #
    # Output:
    # - `(train_loader, val_loader)`
    # - Each loader returns one tensor per batch.
    # - Batch tensor shape: `[batch_size, block_size + 1]`
    # - dtype: `torch.long`
    #
    # Why `block_size + 1`:
    # - A window of 33 tokens can become:
    #   - 32 input tokens
    #   - 32 next-token labels
    # - With `block_size == 4`, `[10, 11, 12, 13, 14]` becomes:
    #   - input ids: `[10, 11, 12, 13]`
    #   - labels: `[11, 12, 13, 14]`
    #
    # Method:
    # Use a tiny text string.
    # If a Tiny Shakespeare file is available, use a small slice from it.
    # Otherwise use a short built-in fallback string.
    #
    # Simple token ids:
    # - Build a character vocabulary from the training text.
    # - Reserve id `0` for unknown characters.
    # - Map each character to an integer id.
    # - Make sure all ids are less than `config.vocab_size`.
    #
    # Build windows:
    # ```python
    # window_size = config.block_size + 1
    # windows = []
    # for start in range(0, len(token_ids) - window_size + 1, window_size):
    #     windows.append(token_ids[start : start + window_size])
    # tensor = torch.tensor(windows, dtype=torch.long)
    # dataset = TensorDataset(tensor)
    # loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # ```
    #
    # DataLoader note:
    # `TensorDataset(tensor)` returns batches as a one-item tuple.
    # In training, unpack it with:
    # `token_block = batch[0]`.
    #
    # Steps:
    # 1. Load or create one text string.
    # 2. Split it into training and validation text.
    # 3. Build the character vocabulary from training text only.
    # 4. Encode train and validation text into token ids.
    # 5. Cut token ids into windows of length `block_size + 1`.
    # 6. Wrap train and validation windows in `TensorDataset`.
    # 7. Return train and validation DataLoaders.
    #
    # Checks:
    # - `batch[0].shape[1]` should equal `config.block_size + 1`.
    # - `batch[0].dtype` should be `torch.long`.
    # - Token ids should be between `0` and `config.vocab_size - 1`.
    raise NotImplementedError("Build DataLoaders that return token blocks.")


# Split one token block into inputs and labels.
# Expected return:
# - input ids
# - labels shifted for next-token prediction
def make_next_token_batch(token_block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create next-token inputs and labels from a token block."""
    # Goal:
    # Return input ids and shifted labels for next-token prediction.
    #
    # Input:
    # - `token_block`: tensor shaped `[batch_size, block_size + 1]`
    # - dtype: `torch.long`
    #
    # Output:
    # - `input_ids`: tensor shaped `[batch_size, block_size]`
    # - `labels`: tensor shaped `[batch_size, block_size]`
    #
    # Small example:
    # ```python
    # token_block = torch.tensor([[10, 11, 12, 13, 14]])
    # input_ids = torch.tensor([[10, 11, 12, 13]])
    # labels = torch.tensor([[11, 12, 13, 14]])
    # ```
    #
    # Method:
    # Use:
    # ```python
    # input_ids = token_block[:, :-1]
    # labels = token_block[:, 1:]
    # return input_ids, labels
    # ```
    #
    # Meaning:
    # - The model sees each token except the last one.
    # - The label at each position is the next token.
    #
    # Steps:
    # 1. Check that `token_block` has 2 dimensions.
    # 2. Slice all tokens except the last for inputs.
    # 3. Slice all tokens except the first for labels.
    # 4. Return both tensors.
    #
    # Checks:
    # - `input_ids.shape == labels.shape`.
    # - `input_ids.shape[1] == token_block.shape[1] - 1`.
    raise NotImplementedError("Return input ids and shifted labels.")



# Define the tiny pretraining model.
# Expected behavior:
# - expose a model class the student can fill in step by step
class TinyPretrainingTransformer(nn.Module):
    """Small decoder-only transformer for pretraining practice."""

    # Build token embeddings, position embeddings, transformer blocks, and a vocab head.
    # Expected return:
    # - an initialized model that maps token ids to logits
    def __init__(self, config: SmallPretrainingConfig) -> None:
        # Goal:
        # Build a tiny decoder-only transformer.
        #
        # Input:
        # - `config`: SmallPretrainingConfig
        #
        # Output:
        # - no return value
        # - creates token embeddings, position embeddings, transformer blocks, and vocab head
        #
        # Shape plan:
        # - input ids: `[batch_size, block_size]`
        # - token embeddings: `[batch_size, block_size, model_dim]`
        # - position embeddings: `[block_size, model_dim]`
        # - hidden states: `[batch_size, block_size, model_dim]`
        # - logits: `[batch_size, block_size, vocab_size]`
        #
        # Method:
        # Store config:
        # ```python
        # self.config = config
        # ```
        #
        # Use:
        # ```python
        # self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        # self.position_embedding = nn.Embedding(config.block_size, config.model_dim)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=config.model_dim,
        #     nhead=config.num_heads,
        #     dim_feedforward=4 * config.model_dim,
        #     batch_first=True,
        # )
        # self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        # self.lm_head = nn.Linear(config.model_dim, config.vocab_size)
        # ```
        #
        # Meaning:
        # - `nn.Embedding(vocab_size, model_dim)` maps token ids to vectors.
        # - Position embeddings tell the model where each token is in the window.
        # - `batch_first=True` means tensors use `[batch, time, features]`.
        # - The language-model head maps hidden vectors back to vocabulary logits.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Store `config`.
        # 3. Build token embeddings.
        # 4. Build position embeddings.
        # 5. Build Transformer encoder blocks.
        # 6. Build the final vocabulary head.
        #
        # Checks:
        # - `config.model_dim` should divide evenly by `config.num_heads`.
        # - Do not apply softmax in the model.
        super().__init__()
        raise NotImplementedError("Build a tiny decoder-only transformer.")

    # Run a forward pass.
    # Expected return:
    # - logits with shape `[batch_size, block_size, vocab_size]`
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits."""
        # Goal:
        # Return logits for each token position.
        #
        # Input:
        # - `input_ids`: tensor shaped `[batch_size, block_size]`
        # - dtype: `torch.long`
        #
        # Output:
        # - logits shaped `[batch_size, block_size, vocab_size]`
        # - logits are raw scores, not probabilities
        #
        # Method:
        # Build positions:
        # ```python
        # batch_size, seq_len = input_ids.shape
        # positions = torch.arange(seq_len, device=input_ids.device)
        # ```
        #
        # Build causal mask:
        # ```python
        # causal_mask = torch.triu(
        #     torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
        #     diagonal=1,
        # )
        # ```
        #
        # Run the model:
        # ```python
        # token_emb = self.token_embedding(input_ids)
        # pos_emb = self.position_embedding(positions).unsqueeze(0)
        # hidden = token_emb + pos_emb
        # hidden = self.blocks(hidden, mask=causal_mask)
        # logits = self.lm_head(hidden)
        # ```
        #
        # Meaning:
        # - The causal mask blocks each token from looking at future tokens.
        # - `.unsqueeze(0)` lets position embeddings broadcast across the batch.
        #
        # Steps:
        # 1. Read batch size and sequence length.
        # 2. Check sequence length is not larger than `config.block_size`.
        # 3. Create position ids.
        # 4. Create a causal attention mask.
        # 5. Add token and position embeddings.
        # 6. Run transformer blocks.
        # 7. Run the vocabulary head.
        # 8. Return logits.
        #
        # Checks:
        # - Input shape `[2, 32]` should return `[2, 32, config.vocab_size]`.
        # - Return raw logits, not softmax probabilities.
        raise NotImplementedError("Return logits for each token position.")


# Run one training step or accumulation step.
# Expected return:
# - loss value
# - number of tokens used for learning
def train_pretraining_steps(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: SmallPretrainingConfig,
) -> dict[str, float | int]:
    """Train for a small number of pretraining steps."""
    # Goal:
    # Train the model and return loss plus tokens per second.
    #
    # Input:
    # - `model`: nn.Module
    # - `loader`: DataLoader
    # - `optimizer`: torch.optim.Optimizer
    # - `config`: SmallPretrainingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"train_loss"`: final or average training loss as a Python float
    # - `"tokens_per_second"`: training speed as a Python float
    # - `"step"`: number of optimizer steps as an int
    #
    # Batch from loader:
    # - `batch` is a one-item tuple from `TensorDataset`
    # - use `token_block = batch[0]`
    # - `token_block` shape is `[batch_size, block_size + 1]`
    #
    # Method:
    # Use:
    # ```python
    # loss_fn = nn.CrossEntropyLoss()
    # input_ids, labels = make_next_token_batch(token_block)
    # logits = model(input_ids)
    # loss = loss_fn(
    #     logits.reshape(-1, logits.shape[-1]),
    #     labels.reshape(-1),
    # )
    # ```
    #
    # Meaning:
    # - `logits` starts as `[batch_size, block_size, vocab_size]`.
    # - Cross-entropy wants `[items, classes]`, so reshape logits to `[batch_size * block_size, vocab_size]`.
    # - Labels reshape to `[batch_size * block_size]`.
    #
    # Gradient accumulation:
    # ```python
    # loss = loss / config.accumulation_steps
    # loss.backward()
    # if micro_step % config.accumulation_steps == 0:
    #     optimizer.step()
    #     optimizer.zero_grad()
    # ```
    #
    # Tokens-per-second formula:
    # ```text
    # tokens_per_second = total_tokens / elapsed_seconds
    # ```
    #
    # Steps:
    # 1. Put the model in train mode.
    # 2. Move batches to `device`.
    # 3. Build input ids and labels.
    # 4. Run the model.
    # 5. Compute cross-entropy loss with the reshape above.
    # 6. Accumulate gradients if requested.
    # 7. Step the optimizer until `config.max_steps` optimizer steps are done.
    # 8. Count tokens with `labels.numel()`.
    # 9. Return the summary dictionary.
    #
    # Checks:
    # - Do not apply softmax before cross-entropy.
    # - Loss should be finite, not NaN.
    # - `tokens_per_second` should be positive.
    _ = time.perf_counter
    raise NotImplementedError("Train the model and return loss plus tokens per second.")


# Evaluate validation loss.
# Expected return:
# - mean validation loss
# Evaluate without tracking gradients.
# Expected return:
# - one float validation loss
@torch.no_grad()
def evaluate_pretraining_loss(model: nn.Module, loader: DataLoader) -> float:
    """Return mean validation next-token loss."""
    # Goal:
    # Compute validation loss without updating the model.
    #
    # Input:
    # - `model`: nn.Module
    # - `loader`: DataLoader
    #
    # Output:
    # - mean validation loss as a Python float
    #
    # Method:
    # Use the same next-token loss shape as training:
    # ```python
    # loss_fn = nn.CrossEntropyLoss()
    # input_ids, labels = make_next_token_batch(token_block)
    # logits = model(input_ids)
    # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    # ```
    #
    # This function already has `@torch.no_grad()`.
    # Still call `model.eval()` inside the function.
    #
    # Steps:
    # 1. Put model in eval mode.
    # 2. Loop over validation batches.
    # 3. Build input ids and labels.
    # 4. Compute cross-entropy loss.
    # 5. Average losses over batches.
    # 6. Return the mean as a Python float.
    #
    # Checks:
    # - Do not call `optimizer.step()` here.
    # - Return one float, not a tensor.
    raise NotImplementedError("Compute validation loss without updating the model.")


# Save model and optimizer state.
# Expected return:
# - path string where the checkpoint was saved
def save_pretraining_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: SmallPretrainingConfig,
) -> str:
    """Save a checkpoint for resuming training."""
    # Goal:
    # Save model, optimizer, and step so training can resume later.
    #
    # Input:
    # - `model`: nn.Module
    # - `optimizer`: torch.optim.Optimizer
    # - `step`: int
    # - `config`: SmallPretrainingConfig
    #
    # Output:
    # - checkpoint path as a string
    #
    # Method:
    # Use:
    # ```python
    # path = Path(config.checkpoint_path)
    # path.parent.mkdir(parents=True, exist_ok=True)
    # checkpoint = {
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     "step": step,
    # }
    # torch.save(checkpoint, path)
    # return str(path)
    # ```
    #
    # Meaning:
    # - Model state stores learned weights.
    # - Optimizer state stores optimizer memory, such as Adam moments.
    # - Step tells training where to resume.
    #
    # Steps:
    # 1. Build a `Path` from `config.checkpoint_path`.
    # 2. Create the parent directory.
    # 3. Build the checkpoint dictionary.
    # 4. Save it with `torch.save`.
    # 5. Return the path string.
    #
    # Checks:
    # - The file should exist after saving.
    # - The checkpoint should contain the three keys shown above.
    raise NotImplementedError("Save model, optimizer, and step.")


# Load model and optimizer state.
# Expected return:
# - resumed step number
def load_pretraining_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> int:
    """Load a checkpoint and return the saved step."""
    # Goal:
    # Load model, optimizer, and step.
    #
    # Input:
    # - `model`: nn.Module
    # - `optimizer`: torch.optim.Optimizer
    # - `checkpoint_path`: str
    #
    # Output:
    # - saved step as an int
    #
    # Method:
    # Use:
    # ```python
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # return int(checkpoint["step"])
    # ```
    #
    # Meaning:
    # - `map_location=device` loads tensors onto CPU or GPU correctly.
    # - Loading optimizer state lets AdamW continue with its stored moments.
    #
    # Steps:
    # 1. Check that the checkpoint path exists.
    # 2. Load the checkpoint with `torch.load`.
    # 3. Load the model state dict.
    # 4. Load the optimizer state dict.
    # 5. Return the saved step.
    #
    # Checks:
    # - Return type should be `int`.
    # - The returned step should match the step saved earlier.
    raise NotImplementedError("Load model, optimizer, and step.")


# Run the full tiny pretraining workflow.
# Expected return:
# - summary dictionary with train loss, validation loss, speed, and checkpoint path
def run_small_pretraining(config: SmallPretrainingConfig) -> dict[str, float | int | str]:
    """Run tiny transformer pretraining."""
    # Goal:
    # Run training, validation, checkpoint save, and resume check.
    #
    # Input:
    # - `config`: SmallPretrainingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"train_loss"`: Python float
    # - `"val_loss"`: Python float
    # - `"tokens_per_second"`: Python float
    # - `"checkpoint_path"`: string path
    # - `"resumed_step"`: int
    #
    # Method:
    # Use:
    # ```python
    # train_loader, val_loader = build_pretraining_loaders(config)
    # model = TinyPretrainingTransformer(config).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # train_metrics = train_pretraining_steps(model, train_loader, optimizer, config)
    # val_loss = evaluate_pretraining_loss(model, val_loader)
    # checkpoint_path = save_pretraining_checkpoint(model, optimizer, train_metrics["step"], config)
    # resumed_step = load_pretraining_checkpoint(model, optimizer, checkpoint_path)
    # ```
    #
    # Steps:
    # 1. Set the seed.
    # 2. Build train and validation loaders.
    # 3. Build model and optimizer.
    # 4. Train for a few steps.
    # 5. Evaluate validation loss.
    # 6. Save checkpoint.
    # 7. Load checkpoint and read resumed step.
    # 8. Return the summary dictionary.
    #
    # Checks:
    # - Returned keys should match `check_metrics`.
    # - `resumed_step` should equal the saved training step.
    # - Loss values should be finite.
    raise NotImplementedError("Run training, validation, checkpoint save, and resume check.")


# Check the pretraining summary.
# Expected return:
# - True when main training outputs are present
def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when pretraining metrics are present."""
    required = {"train_loss", "val_loss", "tokens_per_second", "checkpoint_path", "resumed_step"}
    return required.issubset(metrics)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the small pretraining scaffold."""
    config = SmallPretrainingConfig()
    set_seed(config.random_state)

    try:
        metrics = run_small_pretraining(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "pretraining_outputs", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
