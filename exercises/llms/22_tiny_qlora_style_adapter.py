"""
============================================================
EXERCISE 22: TINY QLORA STYLE ADAPTER
============================================================

WHAT YOU ARE BUILDING
You are building a tiny QLoRA-style teaching example.

QLoRA means:
- keep a large base model in a lower-bit form
- freeze that base model
- train small LoRA adapters on top

Important QLoRA detail:
- the base model is quantized and frozen
- the LoRA adapter is not quantized in this exercise
- the adapter stays as normal trainable PyTorch weights
- this is useful because the base model is large, so quantizing it saves memory
- the adapter is small, so keeping it normal precision is cheap and easier to train

This exercise is intentionally small and dependency-light.
If real PEFT or bitsandbytes tools are available, you may use them as a stretch goal.
The default path is a toy PyTorch version that teaches the idea without requiring those packages.

WHY THIS EXERCISE EXISTS
the related exercise taught LoRA.
the related exercise taught quantization.
This exercise connects them.

This exercise teaches:
- why the base model is frozen
- why adapter parameters stay trainable
- how to compare trainable parameter counts
- why QLoRA is not the same thing as plain inference quantization

DATASET
Use a small real classification dataset.

Default:
- `sklearn.datasets.load_digits`

INPUTS AND OUTPUTS
Input:
- features: `[batch_size, input_dim]`
- labels: `[batch_size]`

Model output:
- logits: `[batch_size, num_classes]`

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

METRICS TO TRACK
Track at least:
- full precision size estimate
- quantized base size estimate
- total parameter count
- trainable parameter count
- trainable percent
- validation accuracy
- plain-language QLoRA note

STEP-BY-STEP TODO
1. Load a small classification dataset.
2. Build a small base classifier.
3. Simulate or apply lower-bit storage to the base weights.
4. Freeze the base weights.
5. Add a small LoRA-style adapter path.
6. Train only the adapter parameters.
7. Count total and trainable parameters.
8. Compare validation accuracy with the tiny adapter path.
9. Explain how this is similar to QLoRA and how it is simpler.

DEFINITION OF DONE
You are done when:
- the base weights are frozen
- adapter weights are trainable
- trainable percent is much smaller than 100%
- validation accuracy is returned
- the note clearly explains the QLoRA idea
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict

import torch
from torch._dynamo.utils import count_calls
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TinyQLoRAConfig:
    """Settings for the tiny QLoRA-style exercise."""

    input_dim: int = 64
    hidden_dim: int = 128
    num_classes: int = 10
    adapter_rank: int = 4
    adapter_alpha: float = 8.0
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 5
    quantized_bits: int = 8
    quantization_mode: str = "dynamic"
    random_state: int = 42


class QLoRAStylePlan(TypedDict):
    """Main shapes and idea for the tiny QLoRA-style model."""

    input_shape: tuple[int, int]
    logit_shape: tuple[int, int]
    adapter_rank: int
    adapter_alpha: float
    adapter_scale: float
    plain_meaning: str


class ParameterCountSummary(TypedDict):
    """Parameter counts for the model."""

    total_parameters: float
    trainable_parameters: float
    trainable_percent: float


class EvaluationMetrics(TypedDict):
    """Loss and accuracy from evaluation."""

    loss: float
    accuracy: float


class TinyQLoRARunSummary(TypedDict):
    """Final summary returned by the experiment."""

    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float
    normal_test_accuracy: float
    quantized_test_accuracy: float
    accuracy_difference: float
    full_precision_base_size_bits: float
    quantized_base_size_bits: float
    total_parameters: float
    trainable_parameters: float
    trainable_percent: float
    note: str


def describe_qlora_style_plan(config: TinyQLoRAConfig) -> QLoRAStylePlan:
    """Return the main QLoRA-style pieces."""
    return {
        "input_shape": (config.batch_size, config.input_dim),
        "logit_shape": (config.batch_size, config.num_classes),
        "adapter_rank": config.adapter_rank,
        "adapter_alpha": config.adapter_alpha,
        "adapter_scale": config.adapter_alpha / config.adapter_rank,
        "plain_meaning": "frozen lower-bit base plus small trainable adapters",
    }


def load_dataloaders(config: TinyQLoRAConfig) -> dict[str, DataLoader]:
    """Return train, validation, and test dataloaders."""
    # Goal:
    # Load the digits data, scale it, split it, and return dataloaders.
    #
    # Input:
    # - `config`: TinyQLoRAConfig
    #
    # Output:
    # Return a dictionary with exactly these keys:
    # - `"train"`
    # - `"val"`
    # - `"test"`
    #
    # Each batch should be a tuple:
    # - `features`: float tensor, shape `[batch_size, config.input_dim]`
    # - `labels`: long tensor, shape `[batch_size]`
    #
    # Method:
    # Use:
    # - `load_digits()` from `sklearn.datasets`
    # - `train_test_split(features, labels, test_size=0.3, random_state=config.random_state, stratify=labels)` from `sklearn.model_selection`
    # - `TensorDataset(features, labels)` from `torch.utils.data`
    # - `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`
    #
    # Scale features with:
    # `features = digits.data.astype("float32") / 16.0`
    #
    # Steps:
    # 1. Load digits with `digits = load_digits()`.
    # 2. Build `features` from `digits.data` and scale by `16.0`.
    # 3. Build `labels` from `digits.target`.
    # 4. Split into train and temp with `train_test_split(features, labels, test_size=0.3, random_state=config.random_state, stratify=labels)`.
    # 5. Split temp into val and test with `train_test_split(temp_features, temp_labels, test_size=0.5, random_state=config.random_state, stratify=temp_labels)`.
    # 6. Convert features to `torch.float32`.
    # 7. Convert labels to `torch.long`.
    # 8. Wrap each split in `TensorDataset`.
    # 9. Use `shuffle=True` for train and `shuffle=False` for val and test.
    # 10. Return the loaders dictionary.
    #
    # Checks:
    # - `set(loaders) == {"train", "val", "test"}`
    # - One train batch should have features shape `[batch_size, 64]`.
    # - Labels should have dtype `torch.long`.

    digits = datasets.load_digits()
    features = digits.data.astype("float32") / 16.0
    labels = digits.target

    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=config.random_state,
        stratify=labels,
    )

    val_features, test_features, val_labels, test_labels = train_test_split(
        temp_features,
        temp_labels,
        test_size=0.5,
        random_state=config.random_state,
        stratify=temp_labels,
    )

    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_features, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_features, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long),
    )

    return {
        "train": DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
    }

    #raise NotImplementedError("Load digits data, scale it, split it, and return dataloaders.")


class BaseClassifier(nn.Module):
    """Small base classifier."""

    def __init__(self, config: TinyQLoRAConfig) -> None:
        # Goal:
        # Build the base classifier layers.
        #
        # Input:
        # - `config`: TinyQLoRAConfig
        #
        # Output:
        # - No return value.
        # - This constructor should create `self.net`.
        #
        # Shape:
        # - input: `[batch_size, config.input_dim]`
        # - output logits: `[batch_size, config.num_classes]`
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
        # 2. Build `self.net` with:
        #    `nn.Sequential(nn.Linear(config.input_dim, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, config.num_classes))`.
        # 3. Do not add softmax.
        #
        # Checks:
        # - A fake input with shape `[2, config.input_dim]` should return `[2, config.num_classes]`.
        # - `nn.CrossEntropyLoss` expects raw logits, not softmax probabilities.
        super().__init__()

        self.net = nn.Sequential(nn.Linear(in_features = config.input_dim, out_features=config.hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=config.hidden_dim, out_features=config.num_classes))
        #raise NotImplementedError("Build the base classifier layers.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        # Goal:
        # Run the base classifier.
        #
        # Input:
        # - `x`: tensor with shape `[batch_size, input_dim]`
        #
        # Output:
        # - logits with shape `[batch_size, num_classes]`
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

        logits = self.net(x)
        
        return logits
        #raise NotImplementedError("Run the base classifier.")


class LoRAAdapter(nn.Module):
    """Small low-rank adapter path."""

    def __init__(self, input_dim: int, output_dim: int, rank: int, alpha: float) -> None:
        # Goal:
        # Build the small low-rank adapter path.
        # This adapter is the LoRA-style part of the model.
        #
        # Plain idea:
        # - LoRA does not train one big update matrix directly.
        # - It trains two smaller matrices instead.
        # - In this file, those two matrices are called `lora_A` and `lora_B`.
        # - The final adapter update is scaled by `alpha / rank`.
        #
        # Common LoRA names:
        # - LoRA A is the down projection.
        # - LoRA B is the up projection.
        #
        # In this class:
        # - `self.lora_A` is LoRA A.
        # - `self.lora_B` is LoRA B.
        # - `self.scale` is `alpha / rank`.
        #
        # Input:
        # - `input_dim`: int
        #   Number of features coming into the adapter.
        # - `output_dim`: int
        #   Number of values the adapter update should produce.
        # - `rank`: int
        #   Small hidden size inside the adapter.
        #   Smaller rank means fewer trainable adapter weights.
        # - `alpha`: float
        #   Scale number for the adapter update before dividing by rank.
        #   Bigger alpha makes the adapter update stronger.
        #
        # Output:
        # - No return value.
        # - This constructor should create:
        #   - `self.lora_A`: `nn.Linear(input_dim, rank, bias=False)`
        #   - `self.lora_B`: `nn.Linear(rank, output_dim, bias=False)`
        #   - `self.scale`: `alpha / rank`
        #
        # Shape:
        # - input: `[batch_size, input_dim]`
        # - after down projection: `[batch_size, rank]`
        # - output update: `[batch_size, output_dim]`
        #
        # Small shape example:
        # - if `input_dim = 64`
        # - if `rank = 4`
        # - if `output_dim = 10`
        # - input batch `x` has shape `[batch_size, 64]`
        # - LoRA A / `self.lora_A` changes it to `[batch_size, 4]`
        # - LoRA B / `self.lora_B` changes it to `[batch_size, 10]`
        # - if `alpha = 8.0`, then `scale = 8.0 / 4 = 2.0`
        # - the final adapter update is multiplied by `2.0`
        #
        # Method:
        # Use:
        # - `nn.Linear(input_dim, rank, bias=False)`
        # - `nn.Linear(rank, output_dim, bias=False)`
        #
        # Exact calls:
        # - `self.lora_A = nn.Linear(input_dim, rank, bias=False)`
        # - `self.lora_B = nn.Linear(rank, output_dim, bias=False)`
        # - `self.scale = alpha / rank`
        #
        # Why `bias=False`:
        # - This adapter is meant to be a small weight update path.
        # - Leaving out bias keeps the example closer to the basic LoRA idea.
        #
        # Why scale is used:
        # - Rank controls how small the adapter is.
        # - Alpha controls how strong the adapter update is.
        # - `alpha / rank` keeps the update size easier to compare across ranks.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Create `self.lora_A`, which is LoRA A.
        # 3. Create `self.lora_B`, which is LoRA B.
        # 4. Store `self.scale = alpha / rank`.
        #
        # Checks:
        # - `self.lora_A.weight.requires_grad` should be True.
        # - `self.lora_B.weight.requires_grad` should be True.
        # - `self.scale` should equal `alpha / rank`.

        super().__init__()

        self.lora_A = nn.Linear(in_features=input_dim, out_features=rank, bias=False)
        self.lora_B = nn.Linear(in_features=rank, out_features=output_dim, bias=False)
        self.scale = alpha / rank
        #raise NotImplementedError("Build down and up projections for the adapter.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the adapter update."""
        # Goal:
        # Run the low-rank adapter path.
        #
        # Input:
        # - `x`: tensor with shape `[batch_size, input_dim]`
        #
        # Output:
        # - adapter logits with shape `[batch_size, output_dim]`
        #
        # Method:
        # Use:
        # `return self.lora_B(self.lora_A(x)) * self.scale`
        #
        # Steps:
        # 1. Pass `x` through `self.lora_A`.
        # 2. Pass the result through `self.lora_B`.
        # 3. Multiply by `self.scale`.
        # 4. Return the adapter logits.
        #
        # Checks:
        # - If `x` is `[8, input_dim]`, output should be `[8, output_dim]`.

        down = self.lora_A(x)
        up = self.lora_B(down)

        return up * self.scale
        #raise NotImplementedError("Run the low-rank adapter path.")


class QLoRAStyleClassifier(nn.Module):
    """Frozen base classifier plus trainable LoRA-style adapter."""

    def __init__(self, base: BaseClassifier, config: TinyQLoRAConfig) -> None:
        # Goal:
        # Freeze the base weights and add trainable adapter weights.
        #
        # Important QLoRA-style idea:
        # - The base model may already be quantized before it is passed here.
        # - The base model is frozen in this class.
        # - The LoRA adapter is not quantized here.
        # - The adapter stays as normal trainable PyTorch weights.
        # - That matches the main QLoRA pattern:
        #   quantized frozen base + small normal trainable adapter.
        #
        # Input:
        # - `base`: BaseClassifier
        # - `config`: TinyQLoRAConfig
        #
        # Output:
        # - No return value.
        # - This constructor should create:
        #   - `self.base`: the frozen base classifier
        #   - `self.adapter`: a trainable `LoRAAdapter`
        #
        # Method:
        # Freeze the base with:
        # `for parameter in self.base.parameters():`
        # `    parameter.requires_grad = False`
        #
        # Build the adapter with:
        # `LoRAAdapter(config.input_dim, config.num_classes, config.adapter_rank, config.adapter_alpha)`
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Save `base` as `self.base`.
        # 3. Set every base parameter to `requires_grad=False`.
        # 4. Build `self.adapter`.
        # 5. Leave adapter parameters trainable.
        #
        # Checks:
        # - Base parameters should have `requires_grad=False`.
        # - Adapter parameters should have `requires_grad=True`.
        super().__init__()

        self.base = base

        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_adapter = LoRAAdapter(input_dim=config.input_dim, output_dim=config.num_classes, rank=config.adapter_rank, alpha=config.adapter_alpha)

        #raise NotImplementedError("Freeze base weights and add trainable adapter weights.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return base logits plus adapter logits."""
        # Goal:
        # Run the frozen base path and the trainable adapter path.
        #
        # Input:
        # - `x`: tensor with shape `[batch_size, input_dim]`
        #
        # Output:
        # - logits with shape `[batch_size, num_classes]`
        #
        # Method:
        # Formula:
        # `logits = self.base(x) + self.lora_adapter(x)`
        #
        # The base path is frozen, but it still runs in the forward pass.
        # Frozen means it does not update during training.
        #
        # Why the adapter gets `x` here:
        # - A LoRA adapter should get the same input as the thing it is adapting.
        # - In this toy model, `self.base(x)` turns the original input features
        #   into class logits.
        # - `self.lora_adapter` was built to also take the original input features:
        #   `input_dim=config.input_dim`.
        # - So this adapter must receive `x`, which has shape
        #   `[batch_size, config.input_dim]`.
        #
        # Important:
        # - This toy adapter is not exactly attached to the first hidden layer.
        # - If it were adapting the first layer, its output would need to match
        #   the first layer output shape, such as `[batch_size, hidden_dim]`.
        # - Here it is a simpler teaching version: a small trainable path maps
        #   from input features directly to final logits.
        #
        # Real LLM rule:
        # - In a real LLM, LoRA is usually attached to a specific linear layer,
        #   such as a query projection or value projection.
        # - The LoRA adapter receives the hidden state that enters that layer.
        # - For layer 1, that might be an early hidden state.
        # - For layer n, that would be the hidden state entering layer n.
        # - It is not usually the output of the layer being adapted.
        #
        # Simple real-layer picture:
        # `base_output = layer(hidden_state)`
        # `lora_update = lora_B(lora_A(hidden_state)) * scale`
        # `output = base_output + lora_update`
        #
        # Steps:
        # 1. Compute `base_logits = self.base(x)`.
        # 2. Compute `adapter_logits = self.lora_adapter(x)`.
        # 3. Return `base_logits + adapter_logits`.
        #
        # Checks:
        # - Do not apply softmax here.

        base_output = self.base(x)
        lora_update = self.lora_adapter(x)

        logits = base_output + lora_update
        return logits
        #raise NotImplementedError("Run frozen base path and trainable adapter path.")


def estimate_weight_storage_bits(model: nn.Module, bits_per_parameter: int) -> int:
    """Estimate raw weight storage in bits."""
    # Goal:
    # Estimate raw weight storage in bits.
    # This is a simple weight-only estimate, not full runtime memory.
    #
    # Input:
    # - `model`: nn.Module
    # - `bits_per_parameter`: int
    #
    # Output:
    # - `int`
    # - raw storage estimate in bits
    #
    # Formula:
    # `storage_bits = total_parameters * bits_per_parameter`
    #
    # Steps:
    # 1. Count parameters with `sum(parameter.numel() for parameter in model.parameters())`.
    # 2. Multiply by `bits_per_parameter`.
    # 3. Return the integer bit count.
    #
    # Checks:
    # - A model with 100 parameters at 32 bits should return `3200`.

    count = sum(param.numel() for param in model.parameters())
    size_in_bits = count * bits_per_parameter
    return size_in_bits
    #raise NotImplementedError("Count parameters and multiply by bits_per_parameter.")


def simulate_quantized_weights(model: nn.Module, bits: int) -> nn.Module:
    """Return a copied model with rounded weights."""
    # This is a simple teaching fallback.
    # It keeps normal float tensors, but rounds the weights to fewer levels.
    # That means the base model really changes, but it is not real int8 storage.
    copied_model = copy.deepcopy(model)
    copied_model.eval()

    levels = 2 ** bits

    with torch.no_grad():
        for param in copied_model.parameters():
            min_value = param.min()
            max_value = param.max()

            if min_value == max_value:
                continue

            scale = (max_value - min_value) / (levels - 1)
            rounded = torch.round((param - min_value) / scale) * scale + min_value
            param.copy_(rounded)

    return copied_model


def quantize_base_model(base: nn.Module, config: TinyQLoRAConfig) -> nn.Module:
    """Return a quantized or simulated-quantized base model."""
    # Real PyTorch dynamic quantization runs on CPU.
    # If it fails in an environment, use the simple simulated fallback.
    base = copy.deepcopy(base).cpu()
    base.eval()

    if config.quantization_mode == "dynamic":
        try:
            return torch.quantization.quantize_dynamic(
                base,
                qconfig_spec={nn.Linear},
                dtype=torch.qint8,
            )
        except Exception:
            return simulate_quantized_weights(base, config.quantized_bits)

    return simulate_quantized_weights(base, config.quantized_bits)


def count_parameters(model: nn.Module) -> ParameterCountSummary:
    """Return total, trainable, and trainable percent."""
    # Goal:
    # Count all parameters and trainable parameters.
    #
    # Input:
    # - `model`: nn.Module
    #
    # Output:
    # Return this exact dictionary:
    # - `"total_parameters"`: float
    # - `"trainable_parameters"`: float
    # - `"trainable_percent"`: float
    #
    # Method:
    # Use:
    # - `parameter.numel()` to count values inside one parameter tensor
    # - `parameter.requires_grad` to tell if a parameter will be trained
    #
    # Formula:
    # `trainable_percent = 100 * trainable_parameters / total_parameters`
    #
    # Steps:
    # 1. Sum `numel()` for all parameters.
    # 2. Sum `numel()` only where `requires_grad=True`.
    # 3. Compute the trainable percent.
    # 4. Return the dictionary.
    #
    # Checks:
    # - For this exercise, trainable percent should be far below `100`.

    total_parameters = sum(param.numel() for param in model.parameters())
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    trainable_percent = trainable_parameters/total_parameters * 100.0

    return {"total_parameters": total_parameters, "trainable_parameters": trainable_parameters, "trainable_percent": trainable_percent}
    #raise NotImplementedError("Count all parameters and trainable parameters.")


def build_training_objects(config: TinyQLoRAConfig, quantize_base: bool = False) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer, ParameterCountSummary]:
    """Return model, loss, optimizer, and parameter counts."""
    # Goal:
    # Build the frozen base, adapter model, loss, optimizer, and parameter counts.
    #
    # Input:
    # - `config`: TinyQLoRAConfig
    #
    # Output:
    # Return a tuple:
    # - model: `QLoRAStyleClassifier`
    # - loss_fn: `nn.CrossEntropyLoss`
    # - optimizer: optimizer for trainable adapter parameters only
    # - parameter_counts: dictionary from `count_parameters`
    #
    # Method:
    # Use:
    # - `base = BaseClassifier(config)`
    # - `model = QLoRAStyleClassifier(base, config)`
    # - `loss_fn = nn.CrossEntropyLoss()`
    # - `trainable_parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)`
    # - `optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)`
    #
    # Steps:
    # 1. Build the base classifier.
    # 2. Wrap it in `QLoRAStyleClassifier`.
    # 3. Move the model to the active device if you add a `device` variable.
    # 4. Build `nn.CrossEntropyLoss()`.
    # 5. Build the optimizer with trainable parameters only.
    # 6. Count parameters.
    # 7. Return `(model, loss_fn, optimizer, parameter_counts)`.
    #
    # Checks:
    # - The optimizer should not receive frozen base parameters.
    # - `parameter_counts["trainable_percent"]` should be much smaller than `100`.

    base = BaseClassifier(config)

    if quantize_base:
        base = quantize_base_model(base, config)
    else:
        base = base.to(device)

    model = QLoRAStyleClassifier(base, config)

    if not quantize_base:
        model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    trainable_parameters = (param for param in model.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(params=trainable_parameters, lr=config.learning_rate)

    param_counts = count_parameters(model)

    return model, loss_fn, optimizer, param_counts
    #raise NotImplementedError("Build frozen base plus adapter and optimizer for trainable parameters only.")


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> EvaluationMetrics:
    """Return mean train loss and accuracy."""
    # Goal:
    # Train only the adapter path for one epoch.
    #
    # Input:
    # - `model`: nn.Module
    # - `loader`: DataLoader that yields `(features, labels)`
    # - `loss_fn`: nn.Module
    # - `optimizer`: torch.optim.Optimizer
    #
    # Output:
    # Return this exact dictionary:
    # - `"loss"`: float
    # - `"accuracy"`: float
    #
    # Method:
    # Use the normal PyTorch training loop:
    # - `model.train()`
    # - `optimizer.zero_grad()`
    # - `logits = model(features)`
    # - `loss = loss_fn(logits, labels)`
    # - `predictions = logits.argmax(dim=1)`
    # - `loss.backward()`
    # - `optimizer.step()`
    #
    # Steps:
    # 1. Call `model.train()`.
    # 2. For each batch, move features and labels to the right device if needed.
    # 3. Clear gradients.
    # 4. Compute logits and loss.
    # 5. Backpropagate and step the optimizer.
    # 6. Track loss with `loss.item() * batch_size`.
    # 7. Track correct predictions with `(predictions == labels).sum().item()`.
    # 8. Divide loss and correct count by the total example count.
    # 9. Return loss and accuracy.
    #
    # Checks:
    # - The base weights should stay frozen.
    # - Do not apply softmax before `nn.CrossEntropyLoss`.

    model.train()
    model_device = next(model.parameters()).device

    loss_sum = 0
    num_samples = 0
    correct = 0

    for features, labels in loader:
        features = features.to(model_device)
        labels = labels.to(model_device)

        optimizer.zero_grad()

        logits = model(features)
        loss = loss_fn(logits, labels)        
        loss.backward()
        optimizer.step()

        batch_size = logits.shape[0]

        loss_sum += loss.item() * batch_size

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()

        num_samples += batch_size

    mean_loss = loss_sum / num_samples
    accuracy = correct / num_samples
    
    return {"loss": mean_loss, "accuracy": accuracy}

    #raise NotImplementedError("Train only the adapter path for one epoch.")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> EvaluationMetrics:
    """Return loss and accuracy."""
    # Goal:
    # Evaluate loss and accuracy without changing the model.
    #
    # Input:
    # - `model`: nn.Module
    # - `loader`: DataLoader that yields `(features, labels)`
    # - `loss_fn`: nn.Module
    #
    # Output:
    # Return this exact dictionary:
    # - `"loss"`: float
    # - `"accuracy"`: float
    #
    # Method:
    # Use:
    # - `model.eval()`
    # - `logits = model(features)`
    # - `loss = loss_fn(logits, labels)`
    # - `predictions = logits.argmax(dim=1)`
    #
    # Steps:
    # 1. Call `model.eval()`.
    # 2. Loop over the loader.
    # 3. Compute loss and predictions.
    # 4. Track total loss weighted by batch size.
    # 5. Track correct predictions.
    # 6. Return mean loss and accuracy.
    #
    # Checks:
    # - This function should not call `backward()` or `optimizer.step()`.

    model.eval()
    model_device = next(model.parameters()).device

    loss_sum = 0
    num_samples = 0
    correct = 0

    for features, labels in loader:
        features = features.to(model_device)
        labels = labels.to(model_device)        

        logits = model(features)
        loss = loss_fn(logits, labels)        

        batch_size = logits.shape[0]

        loss_sum += loss.item() * batch_size

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        
        num_samples += batch_size

    mean_loss = loss_sum / num_samples
    accuracy = correct / num_samples
    
    return {"loss": mean_loss, "accuracy": accuracy}

    #raise NotImplementedError("Evaluate loss and accuracy.")


def explain_qlora_style(config: TinyQLoRAConfig) -> str:
    """Return a short plain-language explanation."""
    # Goal:
    # Explain the tiny QLoRA-style idea in plain words.
    #
    # Input:
    # - `config`: TinyQLoRAConfig
    #
    # Output:
    # - `str`
    # - one or two short sentences
    #
    # Method:
    # Mention:
    # - the base model is treated as lower-bit storage
    # - the base model is frozen
    # - only the small adapter is trained
    # - this is a toy version, not full production QLoRA
    #
    # Steps:
    # 1. Read `config.quantized_bits`.
    # 2. Write a short sentence about the frozen lower-bit base.
    # 3. Write a short sentence about the trainable adapter.
    # 4. Return the explanation string.
    #
    # Checks:
    # - The note should include `frozen`, `adapter`, and `QLoRA` or `QLoRA-style`.

    note = (
        f"This is a QLoRA-style toy example: one run uses a normal frozen base, and "
        f"one run uses a frozen {config.quantized_bits}-bit quantized base. Only the "
        "small adapter path is trained, then the two results are compared."
    )
    return note
    #raise NotImplementedError("Explain frozen lower-bit base plus small trainable adapters.")


def run_experiment(config: TinyQLoRAConfig) -> TinyQLoRARunSummary:
    """Run the full tiny QLoRA-style experiment."""
    # Goal:
    # Train the adapter, evaluate it, and return size and parameter summaries.
    #
    # Input:
    # - `config`: TinyQLoRAConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"train_loss"`: float
    # - `"val_loss"`: float
    # - `"val_accuracy"`: float
    # - `"test_loss"`: float
    # - `"test_accuracy"`: float
    # - `"full_precision_base_size_bits"`: float
    # - `"quantized_base_size_bits"`: float
    # - `"total_parameters"`: float
    # - `"trainable_parameters"`: float
    # - `"trainable_percent"`: float
    # - `"note"`: str
    #
    # Method:
    # Use helper functions from this file.
    # Use these storage estimates:
    # - `estimate_weight_storage_bits(base, 32)` for full precision
    # - `estimate_weight_storage_bits(base, config.quantized_bits)` for quantized base storage
    #
    # Steps:
    # 1. Build loaders with `load_dataloaders(config)`.
    # 2. Build model, loss, optimizer, and counts with `build_training_objects(config)`.
    # 3. Estimate full precision base storage in bits.
    # 4. Estimate quantized base storage in bits.
    # 5. Train for `config.epochs`.
    # 6. Evaluate on validation after each epoch.
    # 7. Evaluate once on test at the end.
    # 8. Build the note with `explain_qlora_style(config)`.
    # 9. Return the summary dictionary with the exact keys above.
    #
    # Checks:
    # - `quantized_base_size_bits < full_precision_base_size_bits`.
    # - `trainable_percent` should be much smaller than `100`.
    # - `check_metrics(metrics)` should return True.

    loaders = load_dataloaders(config)

    def train_setting(label: str, quantize_base: bool) -> dict[str, float]:
        model, loss_fn, optimizer, counts = build_training_objects(config, quantize_base=quantize_base)

        if quantize_base:
            base_parameter_count = sum(param.numel() for param in BaseClassifier(config).parameters())
            trainable_parameter_count = sum(param.numel() for param in model.lora_adapter.parameters() if param.requires_grad)
            total_parameter_count = base_parameter_count + trainable_parameter_count
            counts = {
                "total_parameters": total_parameter_count,
                "trainable_parameters": trainable_parameter_count,
                "trainable_percent": 100 * trainable_parameter_count / total_parameter_count,
            }

        train_metrics = {"loss": 0.0, "accuracy": 0.0}
        val_metrics = {"loss": 0.0, "accuracy": 0.0}

        for epoch in range(config.epochs):
            train_metrics = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
            val_metrics = evaluate(model, loaders["val"], loss_fn)

            print(
                f"{label} epoch {epoch + 1} -> "
                f"loss: {train_metrics['loss']}, "
                f"acc: {train_metrics['accuracy']}, "
                f"val_loss: {val_metrics['loss']}, "
                f"val_acc: {val_metrics['accuracy']}"
            )

        test_metrics = evaluate(model, loaders["test"], loss_fn)
        print(f"{label} TEST RESULTS: test_loss: {test_metrics['loss']}, test_acc: {test_metrics['accuracy']}")

        return {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "total_parameters": counts["total_parameters"],
            "trainable_parameters": counts["trainable_parameters"],
            "trainable_percent": counts["trainable_percent"],
        }

    normal_results = train_setting("normal base", quantize_base=False)
    quantized_results = train_setting("quantized base", quantize_base=True)

    base_for_size = BaseClassifier(config)
    full_precision_size = estimate_weight_storage_bits(base_for_size, 32)
    quantized_size = estimate_weight_storage_bits(base_for_size, config.quantized_bits)

    print(f"Normal test accuracy: {normal_results['test_accuracy']}")
    print(f"Quantized test accuracy: {quantized_results['test_accuracy']}")

    return {
        "train_loss": quantized_results["train_loss"],
        "train_accuracy": quantized_results["train_accuracy"],
        "val_loss": quantized_results["val_loss"],
        "val_accuracy": quantized_results["val_accuracy"],
        "test_loss": quantized_results["test_loss"],
        "test_accuracy": quantized_results["test_accuracy"],
        "normal_test_accuracy": normal_results["test_accuracy"],
        "quantized_test_accuracy": quantized_results["test_accuracy"],
        "accuracy_difference": quantized_results["test_accuracy"] - normal_results["test_accuracy"],
        "full_precision_base_size_bits": full_precision_size,
        "quantized_base_size_bits": quantized_size,
        "total_parameters": quantized_results["total_parameters"],
        "trainable_parameters": quantized_results["trainable_parameters"],
        "trainable_percent": quantized_results["trainable_percent"],
        "note": explain_qlora_style(config),
    }
    #raise NotImplementedError("Train the adapter, evaluate it, and return size and parameter summaries.")


def check_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when the experiment returned the expected fields."""
    return (
        "trainable_percent" in metrics
        and "val_accuracy" in metrics
        and "quantized_base_size_bits" in metrics
        and "normal_test_accuracy" in metrics
        and "quantized_test_accuracy" in metrics
        and "accuracy_difference" in metrics
        and "note" in metrics
    )


def report_task_progress(task_name: str, task_fn: Callable, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the tiny QLoRA-style scaffold."""
    config = TinyQLoRAConfig()
    set_seed(config.random_state)
    metrics, ok = report_task_progress("run_experiment", run_experiment, config)
    if ok and metrics is not None:
        print_pass_or_try_again(check_metrics(metrics), "tiny QLoRA-style experiment", float(int(check_metrics(metrics))), "adapter training returned expected fields")


if __name__ == "__main__":
    main()
