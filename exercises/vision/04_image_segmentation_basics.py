"""
============================================================
EXERCISE 04: IMAGE SEGMENTATION BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a small image segmentation model.

Segmentation means:
- the model does not return one label for the whole image
- the model returns one prediction for each pixel

In this beginner version, use binary segmentation:
- pet pixel -> `1`
- not-pet pixel -> `0`

WHY THIS EXERCISE EXISTS
Segmentation is harder than plain classification.

You now need to think about:
- image shape
- mask shape
- resizing images and masks correctly
- pixel-wise loss
- overlap metrics such as Dice and IoU

DATASET
Use `torchvision.datasets.OxfordIIITPet` with `target_types="segmentation"`.

To keep the task simple, turn the trimap into a binary mask with this exact rule:
- raw mask value `2` -> background `0`
- raw mask values `1` and `3` -> pet `1`

This is a simple learning setup.
It is okay that it is a simplified version.

HOW TO LOAD OR DOWNLOAD THE DATASET
Use `torchvision.datasets.OxfordIIITPet(...)`.

Simple path:
- load the dataset
- read images in RGB
- read segmentation masks
- convert the trimap to a binary mask with the rule above
- make your own `60% / 20% / 20%` train / val / test split
- resize images and masks to the same size

Important resize rule:
- image resize: bilinear is fine
- mask resize: use nearest-neighbor so class ids do not get blurred

Use `download=True` the first time.

INPUTS AND OUTPUTS
Input image batch:
- type: `torch.Tensor`
- shape: `[batch_size, 3, height, width]`
- a good beginner size is `128 x 128`

Target mask batch:
- type: `torch.Tensor`
- dtype: `torch.float32`
- shape: `[batch_size, 1, height, width]`
- values: `0.0` or `1.0`

Model output:
- type: `torch.Tensor`
- shape: `[batch_size, 1, height, width]`
- meaning: one raw logit per pixel

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build a small U-Net-like model with:
- encoder
- decoder
- skip connections
- segmentation head

You do not need a full large U-Net.
You need the basic idea only.

LOSS FUNCTION
Use `nn.BCEWithLogitsLoss`.

Important:
- return raw logits from the model
- do not apply sigmoid before the loss

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, move the model, image batch, and mask batch to the same device.
- Device handling belongs inside your training and evaluation code, right before the forward pass.

METRICS TO TRACK
Track at least:
- `train_loss`
  - mean over the whole training epoch
- `val_loss`
  - mean over the whole validation loader
- `val_dice`
  - mean Dice score over the whole validation loader
- `val_iou`
  - mean IoU score over the whole validation loader

Metric meaning:
- Dice measures overlap between predicted mask and true mask
- IoU also measures overlap, but it is a bit stricter

If you pass the test loader to `evaluate(...)`, return test loss, test Dice, and test IoU for that loader.

TARGET RESULT RANGE
For a small toy setup, a reasonable first target is often:
- validation Dice around `0.70` to `0.90`
- validation IoU around `0.55` to `0.80`

Exact numbers can move around because segmentation is harder.

COMMON MISTAKES
- resizing masks with bilinear instead of nearest-neighbor
- forgetting the mask channel dimension
- using integer masks directly with `BCEWithLogitsLoss` instead of float masks
- returning probabilities instead of raw logits
- image and mask shapes not matching
- encoder and decoder skip connection shapes not lining up

STEP-BY-STEP TODO
1. Load Oxford-IIIT Pet images and segmentation masks.
2. Convert the trimap into a binary mask with the exact rule in this file.
3. Resize images and masks to the same size.
4. Build a small encoder.
5. Build a small decoder.
6. Add skip connections.
7. Return one raw logit per pixel.
8. Train with `BCEWithLogitsLoss`.
9. Track Dice and IoU on the validation loader.

STRETCH GOALS
- use a Dice loss term in addition to BCE
- try a slightly larger image size
- save one predicted mask and compare it to the true mask

DEFINITION OF DONE
You are done when:
- the model returns `[batch_size, 1, height, width]`
- the masks use the same spatial size as the model output
- the validation Dice and IoU improve in a sensible way
- you can explain what a skip connection is in simple words
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms.functional as TF


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import check_threshold, print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"





# Build the train, validation, and test loaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Use a 60% train / 20% val / 20% test split.
# Target masks must be float tensors with shape `[batch_size, 1, height, width]`.
def load_dataloaders(config: SegmentationConfig) -> dict[str, DataLoader]:

    # Create a transform object for images:
    raise NotImplementedError("Return train, val, and test loaders for Oxford-IIIT Pet binary segmentation.")

    # create a transform function for the labels:
    # we need to reize masks as well. Also because this is a binary classification task.
    # raw mask value `2` -> background `0`
    # raw mask values `1` and `3` -> pet `1`
    # we can do this in a function


    #It will only hit the breakpoint inside mask_transf when you fetch a sample/batch from dataset/dataloader:
    #_ = data_train[0]

    

@dataclass
class SegmentationConfig:
    """Settings for the segmentation exercise."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    download: bool = False
    image_size: int = 128
    image_channels: int = 3
    batch_size: int = 8
    base_channels: int = 16
    learning_rate: float = 1e-3
    epochs: int = 6
    random_state: int = 42
    target_val_dice: float = 0.80


def describe_segmentation_problem(config: SegmentationConfig) -> dict[str, object]:
    """Return the expected shapes and mask rules.

    Return a dict with exactly these keys:
    - `image_shape`
    - `mask_shape`
    - `output_shape`
    - `binary_mask_rule`
    """
    return {
        "image_shape": (config.image_channels, config.image_size, config.image_size),
        "mask_shape": (1, config.image_size, config.image_size),
        "output_shape": (config.batch_size, 1, config.image_size, config.image_size),
        "binary_mask_rule": "map raw value 2 to 0, and raw values 1 and 3 to 1",
    }


class TinyUNet(nn.Module):
    """Small U-Net-like model stub for binary segmentation."""

    def __init__(self, input_channels: int, base_channels: int) -> None:
        super().__init__()
        raise NotImplementedError("Build a small encoder, decoder, skip connections, and one-channel segmentation head.")

        # Encoder/downsample:
        # lets not call it encoder_block here since we reuse it for the decoder also.

        #Decoder/upsample:
        # skip the function for simplicity
        # we reuse convo_block        

        # base_channels is 16. I hardcoded it here for readability instead of writing base_channels*2 and base_channels*4 etc.



        



        # kernel_size=1 in the head means:
        # for each pixel, mix channel values only, do not look at neighbors.
        # you already extracted spatial info in earlier conv layers
        # now you just convert each pixel’s feature vector (C channels) into one logit (1 channel)
        # So 1x1 conv is like a tiny per-pixel classifier.

        # In segmentation, we downsample first so the model can see bigger context and 
        # learn “what object this is” instead of only tiny local patterns; pooling makes 
        # feature maps smaller, which also lowers compute and lets deeper layers combine 
        # information across larger parts of the image. But downsampling loses fine details 
        # like sharp edges, so we upsample to bring the feature maps back to the original 
        # resolution and produce one prediction per pixel. Skip connections solve the 
        # detail-loss problem: they pass earlier high-resolution features from the encoder 
        # directly to the decoder, so the decoder gets both kinds of information at 
        # once — deep semantic meaning (“this region is pet”) from the bottleneck path, 
        # and precise boundary/location cues (“exactly where the pet edge is”) from 
        # shallow layers — which gives cleaner masks and better object boundaries.

        # In segmentation, we first make the feature maps smaller.
        # This helps the model see more of the image at once and learn
        # what object or region it is looking at.
        # But this also removes small details like exact edges.
        # So we upsample later to get back to the original image size
        # and predict one label for each pixel.
        # Skip connections help by sending earlier, high-detail features
        # straight from the encoder to the decoder.
        # This gives the decoder both big-picture information and fine details,
        # which helps it make cleaner masks.



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw segmentation logits.

        Return:
        - type: `torch.Tensor`
        - shape: `[batch_size, 1, height, width]`
        """

        raise NotImplementedError("Return one raw logit per pixel.")












# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.BCEWithLogitsLoss`.
def build_training_objects(config: SegmentationConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    raise NotImplementedError("Return the segmentation model, BCEWithLogitsLoss, and optimizer.")






# Run one full training epoch.
# Return a dict with exactly these keys:
# - `loss`
# Here:
# - `loss` means mean train loss over the whole epoch
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:

    raise NotImplementedError("Return the mean train loss for one full epoch.")






# Run one full evaluation pass.
# Return a dict with exactly these keys:
# - `loss`
# - `dice`
# - `iou`
# All three values should be Python floats.
# If you pass the validation loader:
# - `loss` means mean validation loss over the whole validation loader
# - `dice` means mean validation Dice score over the whole validation loader
# - `iou` means mean validation IoU score over the whole validation loader
# If you pass the test loader:
# - `loss` means mean test loss over the whole test loader
# - `dice` means mean test Dice score over the whole test loader
# - `iou` means mean test IoU score over the whole test loader
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    raise NotImplementedError("Return the mean loss, mean Dice, and mean IoU for the loader you evaluate.")


    



@torch.no_grad()
def save_prediction_samples(
    model: nn.Module,
    loader: DataLoader,
    output_path: str | Path,
    num_samples: int = 4,
) -> Path:
    """Save a few image and predicted-mask pairs to one figure."""

    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_data, _ = next(iter(loader))
    x_data = x_data[:num_samples].to(device)

    logits = model(x_data)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float().cpu()
    images = x_data.cpu()

    num_rows = images.shape[0]
    fig, axes = plt.subplots(num_rows, 2, figsize=(6, 3 * num_rows))

    if num_rows == 1:
        axes = [axes]

    for i in range(num_rows):
        image = images[i].permute(1, 2, 0).clamp(0, 1)
        pred_mask = preds[i, 0]

        axes[i][0].imshow(image)
        axes[i][0].set_title("Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(pred_mask, cmap="gray")
        axes[i][1].set_title("Predicted mask")
        axes[i][1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `val_loss`
# - `val_dice`
# - `val_iou`
def run_training(config: SegmentationConfig) -> dict[str, float]:

    raise NotImplementedError("Run the full training flow and return the final train and val metrics.")
    







def check_metrics(metrics: dict[str, float], target: float) -> bool:
    """Return True when validation Dice reaches the target."""
    return check_threshold(metrics.get("val_dice", 0.0), target)


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
        print(f"[DONE] {task_name}")
        return result, True
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False


def main() -> None:
    """Run the segmentation scaffold one step at a time."""
    config = SegmentationConfig()
    set_seed(config.random_state)
    print("Running exercise 04 starter scaffold...")
    print("Main goal: predict one binary mask per image.")

    completed_steps = 0
    total_steps = 8

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_segmentation_problem", describe_segmentation_problem, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("TinyUNet.__init__", TinyUNet, config.image_channels, config.base_channels)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.image_channels, config.image_size, config.image_size)
    _, ok = report_task_progress("TinyUNet.forward", model, sample_x)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    training_model, loss_fn, optimizer = training_objects

    _, ok = report_task_progress(
        "train_one_epoch",
        train_one_epoch,
        training_model,
        loaders["train"],
        loss_fn,
        optimizer,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("evaluate", evaluate, training_model, loaders["val"], loss_fn)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics, config.target_val_dice)
        print_pass_or_try_again(
            passed,
            "val_dice",
            float(metrics["val_dice"]),
            f">= {config.target_val_dice:.2f}",
        )


if __name__ == "__main__":
    main()
