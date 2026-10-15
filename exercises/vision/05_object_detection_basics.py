"""
============================================================
EXERCISE 05: OBJECT DETECTION BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny object detector for a small real object-detection dataset.

Plain-language meanings:
- object detection means finding what object is in an image and where it is
- a bounding box is a rectangle around an object
- IoU means intersection over union
- IoU measures how much two boxes overlap

WHY THIS EXERCISE EXISTS
Image classification says what is in an image.
Object detection also says where it is.

This exercise teaches:
- how to represent bounding boxes
- how to generate a simple detection dataset
- how to predict one class and one box
- how to compute IoU
- how to inspect bad detections

HOW THIS FITS INTO THE BOOTCAMP
Earlier vision exercises focused on whole-image prediction and segmentation.
This file adds another common vision task:
- find where an object is
- and say what it is

Beginner intuition:
- classification asks "what is in the image?"
- detection asks:
  - "what is in the image?"
  - "where is it?"
- so detection is like classification plus location

Small example:
- imagine one image with a single square
- the class label says `square`
- the bounding box says where that square is in the image

DATASET
Use a small real object-detection dataset.

Simple default path:
- use `torchvision.datasets.VOCDetection`
- keep only images with one clear object for the first version
- keep only a few classes if you want a smaller task
- save the object class label and bounding box

Use one shared split:
- 60% train
- 20% validation
- 20% test

Keep the first version small:
- filter to one-object images first
- resize images to one fixed size
- normalize the target boxes to `0.0` to `1.0`

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load `VOCDetection`
- read the object annotations
- keep one object per image for the first version
- convert the object box to `[x_min, y_min, x_max, y_max]`
- normalize the box by image width and height
- wrap the filtered samples in train, validation, and test datasets

INPUTS AND OUTPUTS
Input image batch:
- `images`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, 3, image_size, image_size]`

Target class batch:
- `class_labels`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`

Target box batch:
- `boxes`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, 4]`
  - format: `[x_min, y_min, x_max, y_max]`
  - values: normalized between `0.0` and `1.0`

Model output:
- class logits with shape `[batch_size, num_classes]`
- box predictions with shape `[batch_size, 4]`

MODEL ARCHITECTURE
Use a small CNN with two heads:
- classification head
- box regression head

Plain-language meaning:
- the shared CNN reads the image
- the class head predicts what the object is
- the box head predicts where the object is

LOSS FUNCTION
Use two losses:
- `nn.CrossEntropyLoss` for the class label
- `nn.SmoothL1Loss` or `nn.MSELoss` for the box

Total loss:
- `classification_loss + box_loss_weight * box_loss`

METRICS TO TRACK
Track at least:
- train loss
- validation class accuracy
- validation mean IoU
- number of examples with IoU above `0.5`

Plain-language meaning:
- mean IoU tells you how close the predicted boxes are on average
- IoU above `0.5` means the predicted box overlaps the real box fairly well

TARGET RESULT RANGE
Focus on getting both parts of the task working:
- the class should become more accurate than random
- the predicted box should overlap the real box better over time
- mean IoU should improve from its starting value

COMMON MISTAKES
- mixing up box order, such as `[x, y, width, height]` vs `[x_min, y_min, x_max, y_max]`
- using pixel box values in one place and normalized box values in another
- letting predicted `x_min` become larger than `x_max`
- checking class accuracy but ignoring box quality
- comparing boxes from different image sizes without normalizing

What each step is doing in simple words:
1. generate one simple object per image so the task stays readable
2. save both the object class and the object location
3. build one model that predicts class and box together
4. train the class head and box head at the same time
5. measure class correctness and box overlap separately
6. inspect bad detections so you can see whether the model missed the class, the box, or both

STEP-BY-STEP TODO
1. Load one small real image with one annotated object.
2. Return the class label and normalized bounding box.
3. Build train, val, and test datasets.
4. Build a small CNN detector with two heads.
5. Compute class loss and box loss.
6. Train for several epochs.
7. Compute IoU for predicted and target boxes.
8. Evaluate class accuracy and mean IoU.
9. Save or print a few bad detections.
10. Write a short failure note.

DEFINITION OF DONE
You are done when:
- each sample has an image, class label, and box
- the model returns class logits and box predictions
- IoU is computed correctly
- validation class accuracy and mean IoU are reported
- you can explain what a bounding box is in plain words
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import check_threshold, print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ObjectDetectionConfig:
    """Settings for the object detection basics exercise."""

    num_samples: int = 1200
    image_size: int = 64
    image_channels: int = 3
    num_classes: int = 3
    class_names: tuple[str, ...] = ("square", "circle", "triangle")
    hidden_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 8
    box_loss_weight: float = 5.0
    iou_threshold: float = 0.5
    random_state: int = 42
    target_mean_iou: float = 0.50



def describe_detection_task(config: ObjectDetectionConfig) -> dict[str, object]:
    """Return the core detection shapes and box format."""
    return {
        "image_shape": (config.image_channels, config.image_size, config.image_size),
        "class_names": list(config.class_names),
        "box_format": "[x_min, y_min, x_max, y_max]",
        "box_value_range": "[0.0, 1.0]",
        "model_outputs": {
            "class_logits": (config.batch_size, config.num_classes),
            "pred_boxes": (config.batch_size, 4),
        },
    }



# Build or adapt one training example.
# Return:
# - image tensor
# - class label
# - normalized box
# This is the simplest place to make the dataset concrete for a beginner.
def build_one_detection_sample(
    config: ObjectDetectionConfig,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Build one image, class label, and normalized box example."""
    # Good first version:
    # - read one image and one object annotation
    # - record its class
    # - convert the box to normalized `[x_min, y_min, x_max, y_max]`
    # Keeping this step simple makes the later box math much easier to trust.
    raise NotImplementedError("Build one object detection sample from a real annotation.")



# Beginner hints for this file:
# - classification asks "what is in the image?"; detection also asks "where is it?"
# - keep one object per image first; that removes a lot of extra complexity
# - use one box format everywhere, or bugs become very hard to track down
# - IoU is a box-overlap score, so read it as "how much do these two boxes cover the same area?"
# - check one or two drawn examples by eye before training for a long time

class DetectionDataset(Dataset):
    """Dataset scaffold for one-object detection samples."""

    def __init__(self, config: ObjectDetectionConfig, split: str) -> None:
        super().__init__()
        self.config = config
        self.split = split
        raise NotImplementedError("Load or store one-object detection samples for this split.")

    def __len__(self) -> int:
        """Return the number of examples in the split."""
        raise NotImplementedError("Return the number of detection examples.")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one detection sample.

        Return a dict with exactly these keys:
        - `image`
        - `class_label`
        - `box`
        """
        raise NotImplementedError("Return one image, class label, and normalized box.")



# Build the train, validation, and test dataloaders.
# Each batch should include:
# - `image`
# - `class_label`
# - `box`
def load_dataloaders(config: ObjectDetectionConfig) -> dict[str, DataLoader]:
    """Return train, val, and test dataloaders."""
    # One clean implementation plan:
    # - import `torchvision`
    # - load `torchvision.datasets.VOCDetection`
    # - read the annotation dict for each image
    # - keep only examples with one object for the first version
    # - map class names to integer ids
    # - resize every image to `config.image_size`
    # - convert every box to normalized `[x_min, y_min, x_max, y_max]`
    # - build one small wrapper dataset that returns:
    #   - `image`
    #   - `class_label`
    #   - `box`
    # - make a shared `60 / 20 / 20` split
    # - return three dataloaders
    # Why this keeps the exercise manageable:
    # - one-object images are much easier than full multi-object detection
    # - the model still learns the real detection idea on real data
    raise NotImplementedError("Return train, val, and test dataloaders for a small real detection task.")





class TinyObjectDetector(nn.Module):
    """Small CNN detector with one class head and one box head."""

    def __init__(self, config: ObjectDetectionConfig) -> None:
        super().__init__()
        self.config = config
        raise NotImplementedError("Build a small CNN backbone, class head, and box head.")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return class logits and predicted boxes.

        Return a dict with exactly these keys:
        - `class_logits`
        - `pred_boxes`
        """
        raise NotImplementedError("Return class logits and normalized box predictions.")

# Build:
# - detector model
# - class loss
# - box loss
# - optimizer
# Keeping these separate makes the two-part detection objective easier to understand.
def build_training_objects(
    config: ObjectDetectionConfig,
) -> tuple[nn.Module, nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, class loss, box loss, and optimizer."""
    raise NotImplementedError("Return detector, CrossEntropyLoss, box loss, and optimizer.")


# Compute IoU for predicted and true boxes.
# Read IoU as:
# - 0.0 means no overlap
# - 1.0 means perfect overlap
def compute_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute IoU for two batches of boxes.

    Expected shapes:
    - `boxes_a`: `[batch_size, 4]`
    - `boxes_b`: `[batch_size, 4]`

    Box format:
    - `[x_min, y_min, x_max, y_max]`
    """
    # One box-overlap recipe:
    # 1. find the overlap rectangle
    # 2. compute overlap area
    # 3. compute each box area
    # 4. union = area_a + area_b - overlap
    # 5. IoU = overlap / union
    raise NotImplementedError("Compute IoU for matching pairs of boxes.")


def detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_loss_fn: nn.Module,
    box_loss_fn: nn.Module,
    config: ObjectDetectionConfig,
) -> dict[str, torch.Tensor]:
    """Return total loss, class loss, and box loss."""
    # The total loss has two jobs:
    # - class loss says what the object is
    # - box loss says where it is
    # The weight lets you decide how strongly the box part should matter.
    raise NotImplementedError("Compute class loss, box loss, and total detection loss.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    class_loss_fn: nn.Module,
    box_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ObjectDetectionConfig,
) -> dict[str, float]:
    """Run one full training epoch."""
    raise NotImplementedError("Return mean train loss, class loss, and box loss.")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_loss_fn: nn.Module,
    box_loss_fn: nn.Module,
    config: ObjectDetectionConfig,
) -> dict[str, float]:
    """Run one full evaluation pass."""
    raise NotImplementedError("Return loss, class accuracy, mean IoU, and IoU-threshold accuracy.")


def inspect_bad_detections(
    model: nn.Module,
    loader: DataLoader,
    config: ObjectDetectionConfig,
) -> list[dict[str, object]]:
    """Return a few examples with poor IoU."""
    raise NotImplementedError("Return bad detections with predicted and target boxes.")


def write_failure_note(bad_detections: list[dict[str, object]]) -> str:
    """Write a short note about one detection failure pattern."""
    raise NotImplementedError("Write a plain-language object detection failure note.")


def run_training(config: ObjectDetectionConfig) -> dict[str, float | str]:
    """Run the full object detection workflow."""
    raise NotImplementedError("Train and evaluate the tiny object detector.")


def check_metrics(metrics: dict[str, float | str], target_mean_iou: float) -> bool:
    """Return True when the detector reaches the rough mean IoU target."""
    required = {"val_class_accuracy", "val_mean_iou", "val_iou_at_50", "failure_note"}
    return required.issubset(metrics) and float(metrics.get("val_mean_iou", 0.0)) >= target_mean_iou


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
    """Run the object detection scaffold one step at a time."""
    config = ObjectDetectionConfig()
    set_seed(config.random_state)

    completed_steps = 0
    total_steps = 10

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if not count_or_stop(ok):
        return


    _, ok = report_task_progress("describe_detection_task", describe_detection_task, config)
    if not count_or_stop(ok):
        return

    generator = torch.Generator().manual_seed(config.random_state)
    _, ok = report_task_progress("build_one_detection_sample", build_one_detection_sample, config, generator)
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("TinyObjectDetector.__init__", TinyObjectDetector, config)
    if not count_or_stop(ok):
        return

    sample_images = torch.zeros(config.batch_size, config.image_channels, config.image_size, config.image_size)
    _, ok = report_task_progress("TinyObjectDetector.forward", model, sample_images)
    if not count_or_stop(ok):
        return

    sample_boxes_a = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.2, 0.2, 0.6, 0.6]], dtype=torch.float32)
    sample_boxes_b = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.3, 0.3, 0.7, 0.7]], dtype=torch.float32)
    _, ok = report_task_progress("compute_iou", compute_iou, sample_boxes_a, sample_boxes_b)
    if not count_or_stop(ok):
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return

    training_model, class_loss_fn, box_loss_fn, optimizer = training_objects
    _, ok = report_task_progress(
        "train_one_epoch",
        train_one_epoch,
        training_model,
        loaders["train"],
        class_loss_fn,
        box_loss_fn,
        optimizer,
        config,
    )
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("evaluate", evaluate, training_model, loaders["val"], class_loss_fn, box_loss_fn, config)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if ok and metrics is not None:
        passed = check_metrics(metrics, config.target_mean_iou)
        print_pass_or_try_again(passed, "val_mean_iou", float(metrics.get("val_mean_iou", 0.0)), f">= {config.target_mean_iou}")

if __name__ == "__main__":
    main()
