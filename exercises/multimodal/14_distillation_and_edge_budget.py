"""
============================================================
EXERCISE 14: DISTILLATION AND EDGE BUDGET
============================================================

WHAT YOU ARE BUILDING
You are training a smaller student model from a stronger teacher model,
then checking whether the student fits an edge-style deployment budget.

Distillation means:

```text
large teacher model -> gives soft targets
small student model -> learns to match them
```

Edge budget means:

```text
latency, memory, model size, and accuracy limits
```

The teacher can be:
- a larger tiny model trained in this exercise
- a model from earlier exercises
- later, a pretrained CLIP or small VLM from the web

WHY THIS EXERCISE EXISTS
Large VLMs are often too slow or too expensive for low-power devices.
Distillation is one way to move useful behavior into a smaller model.

HOW THIS FITS INTO THE INTERVIEW PREP
This helps you connect model training to deployment.
For this job, it is useful to explain not only model quality, but also latency,
memory, size, and whether the model can run near the camera or device.

REAL SYSTEM VS THIS EXERCISE

real system                                             this exercise
---------------------------------------------------------------
large VLM teacher (CLIP ViT-L/14 or Qwen2.5-VL)         larger small CNN teacher or pretrained CLIP
small edge student (MobileNetV3 or EfficientNet-Lite)    smaller CNN student
real device latency                                     CPU latency measurement
memory and power constraints                            model size and rough latency budget
task-specific evaluation                                small CIFAR-10 image task

NEXT STEP AFTER THIS EXERCISE
Exercise `210_distill_embodied_policy.py` uses the same distillation idea for
an embodied policy:

```text
large image-language policy -> soft action targets -> smaller edge policy
```

That later check should measure rollout success, not only action accuracy.

VISUALS TO CREATE
Save:
- teacher vs student accuracy chart
- distillation loss curve
- model size and latency chart

INPUTS AND OUTPUTS
Training image:
- `[3, image_size, image_size]`

Hard label:
- class id such as `0` for one shape/color class

Teacher output:
- logits with shape `[batch_size, num_classes]`

Student output:
- logits with shape `[batch_size, num_classes]`

METRICS TO TRACK
- teacher accuracy
- student accuracy
- distillation loss
- student model size in MB
- student latency in milliseconds
- edge budget pass/fail

For embodied policies in the related exercise, also track:
- teacher rollout success rate
- student rollout success rate
- success drop after distillation

STEP-BY-STEP TODO
1. Build an image classification dataset.
2. Train a larger teacher model on hard labels.
3. Freeze or evaluate the teacher.
4. Build a smaller student model.
5. Get teacher logits for each batch.
6. Compute soft distillation loss.
7. Mix soft teacher loss with hard-label loss.
8. Train the student.
9. Measure accuracy, size, and latency.
10. Decide whether the student meets the edge budget.
11. Then use the related exercise to apply the same idea to an embodied policy.

DEFINITION OF DONE
You are done when:
- teacher and student train on image data
- student learns from teacher logits
- size and latency are measured
- an edge pass/fail summary is printed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DistillationEdgeConfig:
    """Settings for distillation and edge checks."""

    image_size: int = 48
    num_classes: int = 6
    batch_size: int = 16
    teacher_epochs: int = 100
    student_epochs: int = 200
    learning_rate: float = 3e-3
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.7
    max_model_size_mb: float = 2.0
    max_latency_ms: float = 10.0
    min_accuracy: float = 0.70
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    teacher_mode: str = "tiny"
    sample_size: int = 2048


class ImageClassExample(TypedDict):
    """One image classification example."""

    image: torch.Tensor
    label: int
    label_name: str


class EdgeMetrics(TypedDict):
    """Distillation and edge summary."""

    teacher_accuracy: float
    student_accuracy: float
    student_size_mb: float
    student_latency_ms: float
    edge_passed: bool
    loss_curve_path: str
    budget_chart_path: str


class ImageClassDataset(Dataset):
    """Dataset for teacher and student image classification."""

    def __init__(self, examples: list[ImageClassExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        return {
            "image": example["image"],
            "label": torch.tensor(example["label"], dtype=torch.long),
            "label_name": example["label_name"],
        }


def build_image_classification_dataset(config: DistillationEdgeConfig) -> list[ImageClassExample]:
    """Build image-label examples."""
    # Where this fits:
    # This is the shared data for teacher training, student training, and evaluation.
    # The teacher and student should learn from the same image task.
    #
    # Goal:
    # Return image tensors and class labels.
    #
    # Dataset:
    # Use CIFAR-10 from `torchvision.datasets.CIFAR10`.
    #
    # Keep a small subset so training stays quick.
    #
    # Output:
    # - list of dictionaries with:
    #   - `"image"`: `[3, image_size, image_size]`
    #   - `"label"`: integer class id
    #   - `"label_name"`: class name string
    #
    # Method:
    # Use `torchvision.datasets.CIFAR10` from the torchvision package.
    # Use this exact transform shape from `torchvision.transforms`:
    # `torchvision.transforms.Compose([torchvision.transforms.Resize((config.image_size, config.image_size)), torchvision.transforms.ToTensor()])`.
    # The dataset constructor call shape is:
    # `torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=True, transform=transform, download=True)`.
    #
    # Steps:
    # 1. Build the transform with the exact `torchvision.transforms.Compose(...)` call shown above.
    # 2. Load CIFAR-10 with `torchvision.datasets.CIFAR10(...)`.
    # 3. Read class names from `dataset.classes`.
    # 4. Loop over dataset items.
    # 5. Skip labels greater than or equal to `config.num_classes`.
    # 6. Keep only `config.sample_size` examples.
    # 7. Append dictionaries with `"image"`, `"label"`, and `"label_name"`.
    # 8. Return the examples.
    #
    # Checks:
    # - `0 <= label < config.num_classes`.
    # - Images are tensors, not feature vectors.
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((config.image_size, config.image_size)),
            torchvision.transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data"),
        train=True,
        transform=transform,
        download=True,
    )

    examples: list[ImageClassExample] = []
    class_names = dataset.classes

    for image, label in dataset:
        if label >= config.num_classes:
            continue

        examples.append(
            {
                "image": image,
                "label": label,
                "label_name": class_names[label],
            }
        )

        if len(examples) >= config.sample_size:
            break

    return examples
    #raise NotImplementedError("Build image classification dataset for teacher and student.")


class TeacherModel(nn.Module):
    """Larger teacher model."""

    def __init__(self, config: DistillationEdgeConfig):
        super().__init__()
        # Goal:
        # Build a model larger than the student.
        # It should be strong enough to create useful soft targets.
        #
        # Suggested layer shape:
        # - several `nn.Conv2d(...)` layers with more channels than the student
        # - `nn.ReLU()` between convolution layers
        # - `nn.AdaptiveAvgPool2d((1, 1))`
        # - `nn.Flatten()`
        # - `nn.Linear(..., config.num_classes)`
        #
        # Method:
        # Use PyTorch layers from `torch.nn`, usually imported as `nn`.
        # `nn.Sequential(...)` is a simple way to stack the CNN.
        # Use `nn.Conv2d(...)`, `nn.ReLU()`, `nn.MaxPool2d(...)`,
        # `nn.AdaptiveAvgPool2d((1, 1))`, `nn.Flatten()`, and `nn.Linear(...)`.
        #
        # Steps:
        # 1. Build a CNN with more channels than the student.
        # 2. Pool the final feature map with `nn.AdaptiveAvgPool2d((1, 1))`.
        # 3. Flatten with `nn.Flatten()`.
        # 4. End with `nn.Linear(hidden_channels, config.num_classes)`.
        # 5. Store the stack on `self`, for example `self.net = nn.Sequential(...)`.
        #
        # Checks:
        # - The final layer output width is `config.num_classes`.
        # - Teacher parameter count should be greater than student parameter count.
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, config.num_classes),
        )
        #raise NotImplementedError("Build teacher model.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        # Goal:
        # Predict class logits from image tensors.
        #
        # Input:
        # - `images`: `[batch_size, 3, image_size, image_size]`
        #
        # Output:
        # - logits shaped `[batch_size, config.num_classes]`
        #
        # Method:
        # Call the module stack you built in `__init__`, for example `return self.net(images)`.
        #
        # Steps:
        # 1. Pass `images` through the teacher network.
        # 2. Return the raw logits.
        #
        # Check:
        # - Return raw logits. Do not apply softmax in `forward`.
        return self.net(images)
        #raise NotImplementedError("Return teacher logits.")


class StudentModel(nn.Module):
    """Smaller student model."""

    def __init__(self, config: DistillationEdgeConfig):
        super().__init__()
        # Goal:
        # Build a smaller model than the teacher.
        # This is the model we pretend to deploy on an edge device.
        #
        # Suggested layer shape:
        # - fewer `nn.Conv2d(...)` layers than the teacher
        # - fewer channels, such as 16 or 24 instead of 64 or 128
        # - `nn.AdaptiveAvgPool2d((1, 1))`
        # - `nn.Flatten()`
        # - `nn.Linear(..., config.num_classes)`
        #
        # Method:
        # Use PyTorch layers from `torch.nn`, usually imported as `nn`.
        # `nn.Sequential(...)` is a simple way to stack the smaller CNN.
        # Use fewer `nn.Conv2d(...)` layers and fewer channels than the teacher.
        #
        # Steps:
        # 1. Build a small CNN with fewer parameters than `TeacherModel`.
        # 2. Pool with `nn.AdaptiveAvgPool2d((1, 1))`.
        # 3. Flatten with `nn.Flatten()`.
        # 4. End with `nn.Linear(student_hidden_channels, config.num_classes)`.
        # 5. Store the stack on `self`, for example `self.net = nn.Sequential(...)`.
        #
        # Checks:
        # - The final layer output width is `config.num_classes`.
        # - The student should be smaller and faster than the teacher.
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, config.num_classes),
        )
        #raise NotImplementedError("Build student model.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        # Goal:
        # Return class logits from the smaller deployable model.
        #
        # Input:
        # - `images`: `[batch_size, 3, image_size, image_size]`
        #
        # Output:
        # - logits shaped `[batch_size, config.num_classes]`
        #
        # Method:
        # Call the module stack you built in `__init__`, for example `return self.net(images)`.
        #
        # Steps:
        # 1. Pass `images` through the student network.
        # 2. Return the raw logits.
        #
        # Check:
        # - The student should have fewer parameters than the teacher.
        return self.net(images)
        #raise NotImplementedError("Return student logits.")


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    config: DistillationEdgeConfig,
) -> torch.Tensor:
    """Combine soft teacher loss and hard-label loss."""
    # Goal:
    # Train the student with:
    # - teacher soft probabilities
    # - true hard labels
    #
    # Soft distillation:
    # - teacher_probs = softmax(teacher_logits / temperature)
    # - student_log_probs = log_softmax(student_logits / temperature)
    # - KL divergence makes student distribution match teacher distribution
    #
    # Hard-label loss:
    # - normal cross entropy with the true class id
    #
    # Combine:
    # - `alpha * soft_loss + (1 - alpha) * hard_loss`
    #
    # Method:
    # Use PyTorch functions `torch.softmax(...)` and `torch.log_softmax(...)`.
    # Use PyTorch loss classes from `torch.nn`:
    # `nn.KLDivLoss(reduction="batchmean")` and `nn.CrossEntropyLoss()`.
    #
    # Steps:
    # 1. Read `temperature = config.distillation_temperature`.
    # 2. Compute teacher probabilities with `torch.softmax(teacher_logits / temperature, dim=-1)`.
    # 3. Compute student log-probabilities with `torch.log_softmax(student_logits / temperature, dim=-1)`.
    # 4. Compute soft loss with `nn.KLDivLoss(reduction="batchmean")(student_log_probs, teacher_probs)`.
    # 5. Multiply soft loss by `temperature ** 2`.
    # 6. Compute hard-label loss with `nn.CrossEntropyLoss()(student_logits, hard_labels)`.
    # 7. Read `alpha = config.distillation_alpha`.
    # 8. Return `alpha * soft_loss + (1 - alpha) * hard_loss`.
    #
    # Check:
    # - The returned value should be a scalar tensor.
    temperature = config.distillation_temperature
    alpha = config.distillation_alpha

    # `teacher_probs` uses softmax because it is the target distribution: normal
    # probabilities like `[0.70, 0.20, 0.10]`.
    # `student_log_probs` uses log_softmax because `nn.KLDivLoss` computes a
    # formula that subtracts the student's log-probability. PyTorch asks for
    # that log value directly instead of taking `log(student_probs)` inside the
    # loss. This is more stable for very small probabilities, because
    # `log_softmax` combines softmax and log in one safer operation.
    # KL divergence measures how different two probability distributions are.
    # Here it is high when the student puts probability on different classes
    # than the teacher, and low when the student matches the teacher.
    # Formula for one example:
    # KL(teacher || student) = sum(teacher_probs * (log(teacher_probs) - student_log_probs))
    # The teacher part is the target. The student part is what the model changes.
    # Both use the same temperature so the teacher and student are compared at
    # the same softness level.
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)

    soft_loss = nn.KLDivLoss(reduction="batchmean")(student_log_probs, teacher_probs)
    soft_loss = soft_loss * (temperature ** 2)
    hard_loss = nn.CrossEntropyLoss()(student_logits, hard_labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss
    #raise NotImplementedError("Compute teacher-student distillation loss.")


def train_teacher(model: TeacherModel, loader: DataLoader, config: DistillationEdgeConfig) -> list[float]:
    """Train the teacher on hard labels."""
    # Goal:
    # Train the larger teacher model using true labels.
    #
    # Method:
    # Use `torch.optim.AdamW` from PyTorch for the optimizer.
    # Use `nn.CrossEntropyLoss()` from `torch.nn` for hard labels.
    # Use normal training calls: `model.train()`, `optimizer.zero_grad()`,
    # `loss.backward()`, and `optimizer.step()`.
    #
    # Steps:
    # 1. Build `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 2. Build `loss_fn = nn.CrossEntropyLoss()`.
    # 3. For each epoch, call `model.train()`.
    # 4. For each batch, move images and labels to `device`.
    # 5. Compute `logits = model(images)`.
    # 6. Compute `loss = loss_fn(logits, labels)`.
    # 7. Clear old gradients, backprop, and step the optimizer.
    # 8. Store the mean epoch loss in a list.
    # 9. Return one loss value per teacher epoch.
    #
    # Check:
    # - Return one loss value per teacher epoch.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []

    for _ in range(config.teacher_epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_examples += batch_size

        epoch_loss = total_loss / max(total_examples, 1)
        losses.append(epoch_loss)
        print(f"Teacher epoch {len(losses)}/{config.teacher_epochs}: loss={epoch_loss:.4f}")

    return losses
    #raise NotImplementedError("Train teacher model.")


def train_student_with_distillation(teacher: TeacherModel, student: StudentModel, loader: DataLoader, config: DistillationEdgeConfig) -> list[float]:
    """Train student from teacher logits and hard labels."""
    # Goal:
    # Train the smaller student from both teacher logits and hard labels.
    #
    # Method:
    # Use `torch.optim.AdamW(student.parameters(), lr=config.learning_rate)` for the student optimizer.
    # Use `teacher.eval()` so teacher behavior is fixed.
    # Use `student.train()` for the model being updated.
    # Use `torch.no_grad()` around the teacher forward pass.
    #
    # Steps:
    # 1. Build the student optimizer.
    # 2. Put teacher in eval mode.
    # 3. For each epoch, put student in train mode.
    # 4. Move images and labels to `device`.
    # 5. Get `teacher_logits = teacher(images)` inside `with torch.no_grad():`.
    # 6. Get `student_logits = student(images)` normally.
    # 7. Compute `loss = compute_distillation_loss(student_logits, teacher_logits, labels, config)`.
    # 8. Clear old student gradients, backprop, and step the student optimizer.
    # 9. Store the mean epoch loss in a list.
    # 10. Return one loss value per student epoch.
    #
    # Check:
    # - Teacher parameters should not change during this function.
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    teacher.eval()

    for _ in range(config.student_epochs):
        student.train()
        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            batch_size = images.size(0)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = compute_distillation_loss(student_logits, teacher_logits, labels, config)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_examples += batch_size

        epoch_loss = total_loss / max(total_examples, 1)
        losses.append(epoch_loss)
        print(f"Student epoch {len(losses)}/{config.student_epochs}: loss={epoch_loss:.4f}")

    return losses
    #raise NotImplementedError("Train student with distillation.")


def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Return classification accuracy."""
    # Goal:
    # Measure image classification accuracy.
    #
    # Method:
    # Use `model.eval()` and the PyTorch context manager `torch.no_grad()`.
    # Use `logits.argmax(dim=-1)` to get predicted class ids.
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Start `correct = 0` and `total = 0`.
    # 3. Loop over batches inside `with torch.no_grad():`.
    # 4. Move images and labels to `device`.
    # 5. Run `logits = model(images)`.
    # 6. Get predictions with `logits.argmax(dim=-1)`.
    # 7. Count matches with `(predictions == labels).sum().item()`.
    # 8. Return `correct / total`.
    #
    # Check:
    # - Return a float between `0.0` and `1.0`.
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            predictions = logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.numel()

    return correct / max(total, 1)
    #raise NotImplementedError("Evaluate model accuracy.")


def measure_model_size_mb(model: nn.Module) -> float:
    """Estimate model parameter size in MB."""
    # Goal:
    # Estimate how much memory the model parameters use.
    #
    # Method:
    # Use PyTorch parameter methods:
    # `param.numel()` returns the number of values in a tensor.
    # `param.element_size()` returns bytes per value.
    #
    # Steps:
    # 1. Start `total_bytes = 0`.
    # 2. Loop over `model.parameters()`.
    # 3. For each parameter, compute `param.numel() * param.element_size()`.
    # 4. Add that to `total_bytes`.
    # 5. Convert bytes to MB with `total_bytes / (1024 ** 2)`.
    # 6. Return the MB value as a float.
    #
    # Check:
    # - Student size should be smaller than teacher size.
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024 ** 2)
    #raise NotImplementedError("Measure model size.")


def measure_latency_ms(model: nn.Module, example_batch: torch.Tensor, repeats: int = 50) -> float:
    """Measure rough CPU or GPU latency."""
    # Goal:
    # Measure average forward-pass latency in milliseconds.
    #
    # Method:
    # Use `time.perf_counter()` from Python's `time` module for timing.
    # Use `model.eval()` and `torch.no_grad()` so timing only measures inference.
    # Use `torch.cuda.synchronize()` before reading time when `example_batch.is_cuda` is true.
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Run a few warmup forwards inside `with torch.no_grad():`.
    # 3. If using CUDA, call `torch.cuda.synchronize()`.
    # 4. Store `start = time.perf_counter()`.
    # 5. Run `repeats` forward passes inside `with torch.no_grad():`.
    # 6. If using CUDA, call `torch.cuda.synchronize()` again.
    # 7. Store `end = time.perf_counter()`.
    # 8. Compute average milliseconds with `(end - start) * 1000 / repeats`.
    # 9. Return the average latency as a float.
    #
    # Check:
    # - Use `torch.no_grad()` so timing does not include gradient work.
    model.eval()

    with torch.no_grad():
        for _ in range(5):
            model(example_batch)

        if example_batch.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            model(example_batch)

        if example_batch.is_cuda:
            torch.cuda.synchronize()

        end = time.perf_counter()

    return (end - start) * 1000 / repeats
    #raise NotImplementedError("Measure average latency in milliseconds.")


def check_edge_budget(metrics: EdgeMetrics, config: DistillationEdgeConfig) -> bool:
    """Return True when student meets edge budget."""
    return (
        metrics["student_accuracy"] >= config.min_accuracy
        and metrics["student_size_mb"] <= config.max_model_size_mb
        and metrics["student_latency_ms"] <= config.max_latency_ms
    )


def save_distillation_visuals(losses: list[float], metrics: EdgeMetrics, config: DistillationEdgeConfig) -> tuple[str, str]:
    """Save loss and budget charts."""
    # Goal:
    # Save one chart for training and one chart for deployment checks.
    #
    # Method:
    # Use `config.output_dir.mkdir(parents=True, exist_ok=True)` from `pathlib.Path`.
    # Use Matplotlib: `matplotlib.use("Agg")`, `plt.figure()`, `plt.plot(...)`,
    # `plt.bar(...)`, `plt.axhline(...)`, `plt.savefig(path)`, and `plt.close()`.
    #
    # Steps:
    # 1. Create `config.output_dir`.
    # 2. Save `losses` as a line plot.
    # 3. Build a second chart for edge checks.
    # 4. Plot student accuracy against `config.min_accuracy`.
    # 5. Plot student size against `config.max_model_size_mb`.
    # 6. Plot student latency against `config.max_latency_ms`.
    # 7. Save the chart.
    # 8. Return both saved paths as strings.
    #
    # Check:
    # - The budget chart should make pass/fail easy to see.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    loss_curve_path = config.output_dir / "distillation_loss_curve.png"
    budget_chart_path = config.output_dir / "edge_budget_chart.png"

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Student epoch")
    plt.ylabel("Distillation loss")
    plt.title("Student distillation loss")
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()

    names = ["accuracy", "size MB", "latency ms"]
    values = [
        metrics["student_accuracy"],
        metrics["student_size_mb"],
        metrics["student_latency_ms"],
    ]
    limits = [
        config.min_accuracy,
        config.max_model_size_mb,
        config.max_latency_ms,
    ]

    x_positions = range(len(names))
    plt.figure(figsize=(8, 4))
    plt.bar(x_positions, values, label="student")
    plt.scatter(x_positions, limits, color="red", label="budget")
    plt.xticks(list(x_positions), names)
    plt.ylabel("value")
    plt.title("Student edge budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(budget_chart_path)
    plt.close()

    return str(loss_curve_path), str(budget_chart_path)
    #raise NotImplementedError("Save distillation and edge-budget charts.")


def run_exercise(config: DistillationEdgeConfig) -> EdgeMetrics:
    """Run teacher training, student distillation, and edge checks."""
    # Main workflow:
    # image labels -> teacher training -> frozen teacher logits -> student training -> edge checks.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_image_classification_dataset(config)
    dataset = ImageClassDataset(examples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    teacher = TeacherModel(config).to(device)
    student = StudentModel(config).to(device)

    train_teacher(teacher, loader, config)
    student_losses = train_student_with_distillation(teacher, student, loader, config)

    teacher_accuracy = evaluate_accuracy(teacher, loader)
    student_accuracy = evaluate_accuracy(student, loader)
    student_size_mb = measure_model_size_mb(student)
    example_batch = next(iter(loader))["image"].to(device)
    student_latency_ms = measure_latency_ms(student, example_batch)
    metrics: EdgeMetrics = {
        "teacher_accuracy": teacher_accuracy,
        "student_accuracy": student_accuracy,
        "student_size_mb": student_size_mb,
        "student_latency_ms": student_latency_ms,
        "edge_passed": False,
        "loss_curve_path": "",
        "budget_chart_path": "",
    }
    metrics["edge_passed"] = check_edge_budget(metrics, config)
    loss_curve_path, budget_chart_path = save_distillation_visuals(student_losses, metrics, config)
    metrics["loss_curve_path"] = loss_curve_path
    metrics["budget_chart_path"] = budget_chart_path

    print(f"Teacher accuracy: {metrics['teacher_accuracy']:.3f}")
    print(f"Student accuracy: {metrics['student_accuracy']:.3f}")
    print(f"Student size MB: {metrics['student_size_mb']:.3f}")
    print(f"Student latency ms: {metrics['student_latency_ms']:.3f}")
    print(f"Edge budget passed: {metrics['edge_passed']}")
    print(f"Loss curve: {metrics['loss_curve_path']}")
    print(f"Budget chart: {metrics['budget_chart_path']}")
    return metrics


def report_task_progress(task_name: str, task_fn, *args):
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the distillation and edge-budget scaffold one step at a time."""
    config = DistillationEdgeConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 6

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    examples, ok = report_task_progress("build_image_classification_dataset", build_image_classification_dataset, config)
    if not count_or_stop(ok):
        return

    teacher, ok = report_task_progress("TeacherModel.__init__", TeacherModel, config)
    if not count_or_stop(ok):
        return

    student, ok = report_task_progress("StudentModel.__init__", StudentModel, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("compute_distillation_loss", compute_distillation_loss, torch.zeros(2, config.num_classes), torch.zeros(2, config.num_classes), torch.zeros(2, dtype=torch.long), config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("measure_model_size_mb", measure_model_size_mb, student)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_edge_budget(metrics, config)
        print_pass_or_try_again(passed, "edge budget", float(int(passed)), "accuracy, size, and latency meet budget")


if __name__ == "__main__":
    main()
