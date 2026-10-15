"""
============================================================
EXERCISE 16: LORA FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building a tiny LoRA layer from scratch in PyTorch.

Plain-language meanings:
- LoRA means low-rank adaptation
- low-rank means "use two small matrices instead of one big matrix"
- adaptation means "learn a small update for a task"
- the base layer stays frozen
- only the small LoRA weights are trained

WHY THIS EXERCISE EXISTS
the related exercise teaches the real Hugging Face and PEFT workflow.
This part teaches what PEFT is doing inside one linear layer.

This exercise teaches:
- how a frozen linear layer can still get a trainable update
- why LoRA uses two small matrices
- how the LoRA scale changes the update size
- how to count total parameters and trainable parameters
- how to prove only the LoRA path is learning

HOW THIS FITS INTO THE BOOTCAMP
Use the main the related exercise for the real tool workflow.
Use this part to understand the math and the tensor shapes.

Formula for one LoRA linear layer:

base_output = x @ W.T + bias
lora_update = ((x @ A.T) @ B.T) * (alpha / rank)
final_output = base_output + lora_update

What each symbol means:
- `x`: input tensor, shape `[batch_size, input_dim]`
- `W`: frozen base weight, shape `[output_dim, input_dim]`
- `bias`: frozen base bias, shape `[output_dim]`
- `A`: trainable down matrix, shape `[rank, input_dim]`
- `B`: trainable up matrix, shape `[output_dim, rank]`
- `alpha`: scale number that controls the update size
- `rank`: small hidden size used by the adapter

Small shape example:
- `x`: `[4, 6]`
- `W`: `[3, 6]`
- `A`: `[2, 6]`
- `B`: `[3, 2]`
- `x @ A.T`: `[4, 2]`
- `(x @ A.T) @ B.T`: `[4, 3]`
- final output: `[4, 3]`

DATASET
Use a tiny synthetic classification dataset.

Each point has:
- features: shape `[input_dim]`
- label: one class id

The data is intentionally simple.
The goal is to learn LoRA mechanics, not data loading.

MODEL
Use a tiny classifier:
- one frozen base linear layer
- one trainable LoRA update on top

INPUTS AND OUTPUTS
Batch fields:
- `features`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, input_dim]`
- `labels`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`

Model output:
- logits
  - type: `torch.Tensor`
  - shape: `[batch_size, num_classes]`

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

Important:
- return raw logits
- do not apply softmax before the loss

METRICS TO TRACK
Track at least:
- total parameters
- trainable parameters
- trainable percent
- train loss
- validation accuracy

TARGET RESULT RANGE
The exact accuracy is less important than the checks:
- base weights stay frozen
- LoRA weights are trainable
- trainable percent is much smaller than 100%
- the forward pass has the expected shape

COMMON MISTAKES
- training the base weights by accident
- using `A @ B` in the wrong order
- forgetting the transpose in the formula
- applying softmax before `CrossEntropyLoss`
- counting buffers or tensors that are not parameters

STEP-BY-STEP TODO
1. Describe the LoRA shapes.
2. Build tiny train, validation, and test dataloaders.
3. Build `LoRALinear`.
4. Freeze the base linear weights.
5. Train only the LoRA weights.
6. Count total and trainable parameters.
7. Evaluate validation accuracy.
8. Explain how this matches the PEFT version from the related exercise.

DEFINITION OF DONE
You are done when:
- `LoRALinear` returns logits with shape `[batch_size, num_classes]`
- base parameters have `requires_grad=False`
- LoRA parameters have `requires_grad=True`
- parameter counts are printed
- you can explain the LoRA formula in plain words
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    """Set the main random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a short result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")


@dataclass
class LoRAFromScratchConfig:
    """Settings for the tiny LoRA-from-scratch exercise."""

    input_dim: int = 12
    num_classes: int = 4
    rank: int = 3
    alpha: float = 6.0
    batch_size: int = 32
    train_examples: int = 512
    val_examples: int = 128
    test_examples: int = 128
    learning_rate: float = 5e-3
    epochs: int = 8
    random_state: int = 42
    target_trainable_percent: float = 50.0


class LoRAShapeSummary(TypedDict):
    """Shape notes returned by `describe_lora_shapes`."""

    base_weight_shape: tuple[int, int]
    lora_A_shape: tuple[int, int]
    lora_B_shape: tuple[int, int]
    scale: float
    formula: str


def describe_lora_shapes(config: LoRAFromScratchConfig) -> LoRAShapeSummary:
    """Return the key LoRA shapes and formula pieces."""
    # Goal:
    # Return a small dictionary that makes the LoRA shape flow easy to check.
    #
    # Input:
    # - `config.input_dim`: number of input features
    # - `config.num_classes`: number of output classes
    # - `config.rank`: small adapter size
    #
    # Output:
    # - dictionary with these keys:
    #   - `base_weight_shape`
    #   - `lora_A_shape`
    #   - `lora_B_shape`
    #   - `scale`
    #   - `formula`
    #
    # Method:
    # Use the formula shown at the top of this file:
    # `final_output = x @ W.T + bias + ((x @ A.T) @ B.T) * (alpha / rank)`.
    #
    # Steps:
    # 1. Build the base weight shape as `(num_classes, input_dim)`.
    # 2. Build `A` shape as `(rank, input_dim)`.
    # 3. Build `B` shape as `(num_classes, rank)`.
    # 4. Compute `scale = alpha / rank`.
    # 5. Return the dictionary.
    #
    # Checks:
    # - If input shape is `[32, 12]`, the final logits should be `[32, 4]`.

    raise NotImplementedError("Return the LoRA shapes and scale.")




def make_toy_classification_tensors(config: LoRAFromScratchConfig) -> dict[str, torch.Tensor]:
    """Create small train, validation, and test tensors."""
    # Goal:
    # Build a tiny classification problem in memory.
    #
    # Input:
    # - `config`: LoRAFromScratchConfig
    #
    # Output:
    # Return a dictionary with exactly these keys:
    # - `train_features`: `[train_examples, input_dim]`, float32
    # - `train_labels`: `[train_examples]`, long
    # - `val_features`: `[val_examples, input_dim]`, float32
    # - `val_labels`: `[val_examples]`, long
    # - `test_features`: `[test_examples, input_dim]`, float32
    # - `test_labels`: `[test_examples]`, long
    #
    # Method:
    # Use these exact PyTorch calls:
    # - `teacher_weight = torch.randn(config.input_dim, config.num_classes)`
    # - `train_features = torch.randn(config.train_examples, config.input_dim)`
    # - `val_features = torch.randn(config.val_examples, config.input_dim)`
    # - `test_features = torch.randn(config.test_examples, config.input_dim)`
    #
    # Use one hidden teacher matrix to make labels:
    # - `teacher_logits = features @ teacher_weight`
    # - `labels = teacher_logits.argmax(dim=1)`
    #
    # Shape example:
    # - features: `[5, 12]`
    # - teacher_weight: `[12, 4]`
    # - teacher_logits: `[5, 4]`
    # - labels: `[5]`
    #
    # Steps:
    # 1. Make one teacher matrix with shape `[input_dim, num_classes]`.
    # 2. Make train, validation, and test feature tensors.
    # 3. Make labels from the teacher logits for each split.
    # 4. Return the dictionary with the exact keys listed above.
    #
    # Checks:
    # - `train_features.dtype` should be `torch.float32`.
    # - `train_labels.dtype` should be `torch.long`.

    raise NotImplementedError("Create toy classification tensors.")






def build_dataloaders(
    tensors: dict[str, torch.Tensor],
    config: LoRAFromScratchConfig,
) -> dict[str, DataLoader]:
    """Return train, validation, and test dataloaders."""
    # Goal:
    # Wrap the tensors in PyTorch `TensorDataset` objects and then DataLoaders.
    #
    # Input:
    # - `tensors`: dictionary from `make_toy_classification_tensors`
    #
    # Output:
    # Return a dictionary with exactly these keys:
    # - `train`
    # - `val`
    # - `test`
    #
    # Method:
    # Use these exact PyTorch classes:
    # - `TensorDataset(features, labels)` makes a dataset from two tensors.
    #   One item from the dataset is `(one_feature_row, one_label)`.
    # - `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`
    #   groups rows into batches.
    #
    # Steps:
    # 1. Build `train_dataset = TensorDataset(tensors["train_features"], tensors["train_labels"])`.
    # 2. Build `val_dataset = TensorDataset(tensors["val_features"], tensors["val_labels"])`.
    # 3. Build `test_dataset = TensorDataset(tensors["test_features"], tensors["test_labels"])`.
    # 4. Build `train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)`.
    # 5. Build `val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)`.
    # 6. Build `test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)`.
    # 7. Return `{"train": train_loader, "val": val_loader, "test": test_loader}`.
    #
    # Checks:
    # - One batch should be a tuple:
    #   - `features`: `[batch_size, input_dim]`
    #   - `labels`: `[batch_size]`    

    raise NotImplementedError("Build train, validation, and test dataloaders.")




class LoRALinear(nn.Module):
    """A frozen base linear layer plus a trainable LoRA update."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        alpha: float,
    ) -> None:
        # Goal:
        # Build one LoRA linear layer from scratch.
        #
        # Input:
        # - `input_dim`: number of input features
        # - `output_dim`: number of output classes or features
        # - `rank`: small adapter size
        # - `alpha`: update scale before dividing by rank
        #
        # Output:
        # This constructor should create:
        # - `self.base`: `nn.Linear(input_dim, output_dim)`
        # - `self.lora_A`: `nn.Linear(input_dim, rank, bias=False)`
        # - `self.lora_B`: `nn.Linear(rank, output_dim, bias=False)`
        # - `self.scale`: `alpha / rank`
        #
        # Method:
        # Use `torch.nn.Linear`.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Create the frozen base linear layer.
        # 3. Create `lora_A`, the down projection from input_dim to rank.
        # 4. Create `lora_B`, the up projection from rank to output_dim.
        # 5. Freeze every parameter in `self.base` by setting `requires_grad=False`.
        # 6. Leave `lora_A` and `lora_B` trainable.
        # 7. Store `self.scale = alpha / rank`.
        #
        # Checks:
        # - `self.base.weight.requires_grad` should be False.
        # - `self.lora_A.weight.requires_grad` should be True.
        # - `self.lora_B.weight.requires_grad` should be True.

        super().__init__()

        raise NotImplementedError("Build the frozen base and trainable LoRA layers.")



        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return base output plus LoRA update."""
        # Goal:
        # Run the LoRA linear layer.
        #
        # Input:
        # - `x`: tensor with shape `[batch_size, input_dim]`
        #
        # Output:
        # - logits or features with shape `[batch_size, output_dim]`
        #
        # Method:
        # Use the modules made in `__init__`.
        #
        # Steps:
        # 1. Compute `base_output = self.base(x)`.
        # 2. Compute `lora_hidden = self.lora_A(x)`.
        # 3. Compute `lora_update = self.lora_B(lora_hidden) * self.scale`.
        # 4. Return `base_output + lora_update`.
        #
        # Checks:
        # - If `x` is `[8, input_dim]`, output should be `[8, output_dim]`.

        raise NotImplementedError("Return base output plus LoRA update.")




class TinyLoRAClassifier(nn.Module):
    """Tiny classifier that uses one LoRA linear layer."""

    def __init__(self, config: LoRAFromScratchConfig) -> None:
        # Goal:
        # Build a classifier that maps features to class logits with LoRA.
        #
        # Input:
        # - `config.input_dim`
        # - `config.num_classes`
        # - `config.rank`
        # - `config.alpha`
        #
        # Output:
        # - a model whose forward pass returns logits with shape
        #   `[batch_size, num_classes]`
        #
        # Method:
        # Use the `LoRALinear` class in this file.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Create:
        #    `self.classifier = LoRALinear(
        #        input_dim=config.input_dim,
        #        output_dim=config.num_classes,
        #        rank=config.rank,
        #        alpha=config.alpha,
        #    )`
        #
        # Checks:
        # - A fake input with shape `[2, input_dim]` should return `[2, num_classes]`.
        super().__init__()

        raise NotImplementedError("Build the tiny LoRA classifier.")
        

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        # Goal:
        # Convert a feature batch into raw class scores.
        #
        # Input:
        # - `features`: `[batch_size, input_dim]`
        #
        # Output:
        # - logits: `[batch_size, num_classes]`
        #
        # Steps:
        # 1. Pass `features` through `self.classifier`.
        # 2. Return the logits.
        #
        # Checks:
        # - Do not apply softmax here.
        # - `nn.CrossEntropyLoss` expects raw logits.

        raise NotImplementedError("Return class logits.")



def count_parameters(model: nn.Module) -> dict[str, float]:
    """Count total and trainable parameters."""
    # Goal:
    # Count how many model parameters exist and how many will be trained.
    #
    # Input:
    # - `model`: any PyTorch model
    #
    # Output:
    # Return a dictionary with exactly these keys:
    # - `total_parameters`
    # - `trainable_parameters`
    # - `trainable_percent`
    #
    # Method:
    # Use `model.parameters()`.
    # A parameter is trainable when `parameter.requires_grad` is True.
    #
    # Steps:
    # 1. Loop over all parameters and add up `parameter.numel()`.
    # 2. Loop over only trainable parameters and add up `parameter.numel()`.
    # 3. Compute `trainable_percent = 100 * trainable / total`.
    # 4. Return the dictionary.
    #
    # Checks:
    # - In this exercise, trainable percent should be much smaller than 100.

    raise NotImplementedError("Count total and trainable parameters.")




def build_training_objects(
    config: LoRAFromScratchConfig,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer, dict[str, float]]:
    """Return model, loss, optimizer, and parameter counts."""
    # Goal:
    # Build everything needed for training.
    #
    # Input:
    # - `config`: LoRAFromScratchConfig
    #
    # Output:
    # Return a tuple:
    # - model: `TinyLoRAClassifier`
    # - loss_fn: `nn.CrossEntropyLoss`
    # - optimizer: optimizer for only trainable parameters
    # - parameter_counts: dictionary from `count_parameters`
    #
    # Method:
    # Use these exact PyTorch calls:
    # - `model = TinyLoRAClassifier(config).to(device)`
    # - `loss_fn = nn.CrossEntropyLoss()`
    # - `trainable_parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)`
    # - `optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)`
    #
    # `nn.CrossEntropyLoss()` expects raw logits with shape `[batch_size, num_classes]`
    # and labels with shape `[batch_size]`.
    #
    # Steps:
    # 1. Build `TinyLoRAClassifier(config)`.
    # 2. Move it to `device`.
    # 3. Build `nn.CrossEntropyLoss()`.
    # 4. Build an optimizer with only parameters where `requires_grad=True`.
    # 5. Count parameters.
    # 6. Print the total and trainable counts.
    # 7. Return all four objects.
    #
    # Checks:
    # - The optimizer should not receive frozen base parameters.

    raise NotImplementedError("Build model, loss, optimizer, and parameter counts.")





def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train the LoRA classifier for one epoch."""
    # Goal:
    # Train only the LoRA parameters for one pass over the training data.
    #
    # Input batch:
    # - `features`: `[batch_size, input_dim]`
    # - `labels`: `[batch_size]`
    #
    # Output:
    # Return:
    # - `{"loss": mean_loss, "accuracy": mean_accuracy}`
    #
    # Steps:
    # 1. Call `model.train()`.
    # 2. For each batch, move `features` and `labels` to `device`.
    # 3. Clear old gradients with `optimizer.zero_grad()`.
    # 4. Run `logits = model(features)`.
    # 5. Compute `loss = loss_fn(logits, labels)`.
    # 6. Call `loss.backward()`.
    # 7. Call `optimizer.step()`.
    # 8. Track average loss:
    #    - `batch_size = features.size(0)`
    #    - add `loss.item() * batch_size` to a running loss total
    #    - add `batch_size` to a running example count
    #    - at the end, divide the loss total by the example count
    # 9. Track average accuracy:
    #    - `predictions = logits.argmax(dim=1)`
    #    - `correct = (predictions == labels).sum().item()`
    #    - add `correct` to a running correct count
    #    - at the end, divide the correct count by the example count
    #
    # Checks:
    # - `logits.shape` should be `[batch_size, num_classes]`.

    raise NotImplementedError("Train one epoch and return loss and accuracy.")






@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:
    """Evaluate the LoRA classifier."""
    # Goal:
    # Measure loss and accuracy without changing the model.
    #
    # Input batch:
    # - `features`: `[batch_size, input_dim]`
    # - `labels`: `[batch_size]`
    #
    # Output:
    # Return:
    # - `{"loss": mean_loss, "accuracy": mean_accuracy}`
    #
    # Steps:
    # 1. Call `model.eval()`.
    # 2. For each batch, move `features` and `labels` to `device`.
    # 3. Run `logits = model(features)`.
    # 4. Compute the loss.
    # 5. Use `logits.argmax(dim=1)` to get class predictions.
    # 6. Track average loss:
    #    - `batch_size = features.size(0)`
    #    - add `loss.item() * batch_size` to a running loss total
    #    - add `batch_size` to a running example count
    #    - at the end, divide the loss total by the example count
    # 7. Track average accuracy:
    #    - `predictions = logits.argmax(dim=1)`
    #    - `correct = (predictions == labels).sum().item()`
    #    - add `correct` to a running correct count
    #    - at the end, divide the correct count by the example count
    #
    # Checks:
    # - This function should not call `backward()` or `optimizer.step()`.

    raise NotImplementedError("Evaluate and return loss and accuracy.")






def explain_lora_from_scratch(parameter_counts: dict[str, float]) -> str:
    """Return a short explanation of what LoRA changed."""
    # Goal:
    # Explain the LoRA tradeoff in plain words.
    #
    # Input:
    # - `parameter_counts`: dictionary from `count_parameters`
    #
    # Output:
    # - one short string
    #
    # Steps:
    # 1. Read `trainable_percent`.
    # 2. Say that the base layer is frozen.
    # 3. Say that only the small adapter matrices are trained.
    # 4. Include the trainable percent rounded to two decimal places.
    #
    # Checks:
    # - The note should be clear enough to say out loud in an interview.
    raise NotImplementedError("explain_lora_from_scratch.")


def run_training(config: LoRAFromScratchConfig) -> dict[str, float | str]:
    """Run the full LoRA-from-scratch workflow."""
    # Goal:
    # Connect all pieces into one full run.
    #
    # Steps:
    # 1. Make toy tensors.
    # 2. Build dataloaders.
    # 3. Build model, loss, optimizer, and parameter counts.
    # 4. Train for `config.epochs`.
    # 5. Evaluate on validation each epoch.
    # 6. Evaluate on test once at the end.
    # 7. Build a short LoRA explanation.
    #
    # Output:
    # Return a dictionary with exactly these keys:
    # - `train_loss`
    # - `train_accuracy`
    # - `val_loss`
    # - `val_accuracy`
    # - `test_loss`
    # - `test_accuracy`
    # - `total_parameters`
    # - `trainable_parameters`
    # - `trainable_percent`
    # - `lora_note`
    #
    # Checks:
    # - The returned dictionary should pass `check_metrics(metrics, config)`.
    # - `trainable_percent` should be below `config.target_trainable_percent`.
    # - `lora_note` should explain why only the adapter weights were trained.

    raise NotImplementedError("Run the full LoRA-from-scratch workflow.")








def check_metrics(metrics: dict[str, float | str], config: LoRAFromScratchConfig) -> bool:
    """Return True when the run returns the expected fields and trainable percent is small."""
    required_keys = {
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "test_loss",
        "test_accuracy",
        "total_parameters",
        "trainable_parameters",
        "trainable_percent",
        "lora_note",
    }
    return required_keys.issubset(metrics) and float(metrics["trainable_percent"]) <= config.target_trainable_percent


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
    """Run the LoRA-from-scratch scaffold one step at a time."""
    config = LoRAFromScratchConfig()
    set_seed(config.random_state)

    shapes, ok = report_task_progress("describe_lora_shapes", describe_lora_shapes, config)
    if not ok:
        return

    tensors, ok = report_task_progress("make_toy_classification_tensors", make_toy_classification_tensors, config)
    if not ok:
        return

    loaders, ok = report_task_progress("build_dataloaders", build_dataloaders, tensors, config)
    if not ok:
        return

    model_result, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not ok:
        return

    model, loss_fn, optimizer, parameter_counts = model_result

    _, ok = report_task_progress("train_one_epoch", train_one_epoch, model, loaders["train"], loss_fn, optimizer)
    if not ok:
        return

    _, ok = report_task_progress("evaluate", evaluate, model, loaders["val"], loss_fn)
    if not ok:
        return

    _, ok = report_task_progress("explain_lora_from_scratch", explain_lora_from_scratch, parameter_counts)
    if not ok:
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok and metrics is not None:
        passed = check_metrics(metrics, config)
        print_pass_or_try_again(
            passed,
            "trainable_percent",
            float(metrics["trainable_percent"]),
            f"<= {config.target_trainable_percent}",
        )

    # Keep `shapes` referenced so a finished run can print or inspect it later.
    _ = shapes


if __name__ == "__main__":
    main()
