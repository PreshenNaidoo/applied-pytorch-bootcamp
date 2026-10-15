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


@dataclass

        #raise NotImplementedError("Return one raw logit per pixel.")


# Build the train, validation, and test loaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Use a 60% train / 20% val / 20% test split.
# Target masks must be float tensors with shape `[batch_size, 1, height, width]`.
def load_dataloaders(config: SegmentationConfig) -> dict[str, DataLoader]:

    # Create a transform object for images:
    resize_transf = torchvision.transforms.Resize(size=(config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BILINEAR)
    conv_transf = torchvision.transforms.ToTensor() # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    image_tf = torchvision.transforms.Compose([resize_transf, conv_transf])

    # create a transform function for the labels:
    # we need to reize masks as well. Also because this is a binary classification task.
    # raw mask value `2` -> background `0`
    # raw mask values `1` and `3` -> pet `1`
    # we can do this in a function
    def mask_transf(mask_pil):
        #resize:
        m = TF.resize(mask_pil, [config.image_size, config.image_size], interpolation=transforms.InterpolationMode.NEAREST)
        # now m shape is (128, 128)
        m = TF.pil_to_tensor(m)
        # now m shape is (1, 128, 128)
        m = ((m==1) | (m == 3)).float()

        return m

    data_train = torchvision.datasets.OxfordIIITPet(root = config.data_root, download = True, split = "trainval", 
                                                    target_types='segmentation', transform=image_tf, target_transform=mask_transf)
    data_test = torchvision.datasets.OxfordIIITPet(root = config.data_root, download = True, split = "test", 
                                                    target_types='segmentation', transform=image_tf, target_transform=mask_transf)

    #It will only hit the breakpoint inside mask_transf when you fetch a sample/batch from dataset/dataloader:
    #_ = data_train[0]

    full_ds = torch.utils.data.ConcatDataset([data_train, data_test])
    train_ds, val_ds, test_ds = torch.utils.data.random_split(full_ds, [0.6, 0.2, 0.2])
    
    return {"train": DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": DataLoader(dataset=test_ds, batch_size=config.batch_size)}

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
        self.input_channels = input_channels
        self.base_channels = base_channels

        # Encoder/downsample:
        # lets not call it encoder_block here since we reuse it for the decoder also.
        def convo_block(in_channels, out_channels):

            sequential = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same'),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )
            return sequential

        #Decoder/upsample:
        # skip the function for simplicity
        # we reuse convo_block        

        # base_channels is 16. I hardcoded it here for readability instead of writing base_channels*2 and base_channels*4 etc.
        self.enc1 = convo_block(input_channels, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = convo_block(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = convo_block(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = convo_block(64, 128)
        
        self.up1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec1 = convo_block(128, 64)

        self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec2 = convo_block(64, 32)

        self.up3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.dec3 = convo_block(32, 16)

        # kernel_size=1 in the head means:
        # for each pixel, mix channel values only, do not look at neighbors.
        # you already extracted spatial info in earlier conv layers
        # now you just convert each pixel’s feature vector (C channels) into one logit (1 channel)
        # So 1x1 conv is like a tiny per-pixel classifier.
        self.head = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

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


        #raise NotImplementedError("Build a small encoder, decoder, skip connections, and one-channel segmentation head.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw segmentation logits.

        Return:
        - type: `torch.Tensor`
        - shape: `[batch_size, 1, height, width]`
        """

        skip1 = self.enc1(x)
        x = self.pool1(skip1)

        skip2 = self.enc2(x)
        x = self.pool2(skip2)

        skip3 = self.enc3(x)
        x = self.pool3(skip3)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = torch.concat([x, skip3], dim=1) #Because dim=1 is the channel dimension for image tensors in PyTorch. [batch, channels, height, width] = [B, C, H, W]
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.concat([x, skip2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.concat([x, skip1], dim=1)
        x = self.dec3(x)

        x = self.head(x)

        return x


    #raise NotImplementedError("Return train, val, and test loaders for Oxford-IIIT Pet binary segmentation.")


# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.BCEWithLogitsLoss`.
def build_training_objects(config: SegmentationConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    model = TinyUNet(input_channels = config.image_channels, base_channels=config.base_channels)
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(lr = config.learning_rate, params=model.parameters())

    return model, loss_fn, optimizer

    #raise NotImplementedError("Return the segmentation model, BCEWithLogitsLoss, and optimizer.")


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

    losses, accs = [], []

    model.train()

    for x_data, y_targets in loader:
        x_data = x_data.to(device)
        y_targets = y_targets.to(device)

        optimizer.zero_grad()

        logits = model(x_data)
        loss = loss_fn(logits, y_targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item()) 

    mean_loss = torch.tensor(losses).mean().item()    

    return {"loss":mean_loss}
    #raise NotImplementedError("Return the mean train loss for one full epoch.")


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

    model.eval()

    losses, dices, ious = [], [], []

    for x_data, y_targets in loader:
        x_data = x_data.to(device)
        y_targets = y_targets.to(device)

        logits = model(x_data)
        loss = loss_fn(logits, y_targets)

        losses.append(loss.item())

        # now for dice and iou
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        intersection = (preds * y_targets).sum(dim=(1, 2, 3))
        pred_sum = preds.sum(dim=(1, 2, 3))
        target_sum = y_targets.sum(dim=(1, 2, 3))
        union = pred_sum + target_sum - intersection

        dice = (2 * intersection) / (pred_sum + target_sum)
        iou = intersection / union

        dices.append(dice.mean().item())
        ious.append(iou.mean().item())
    
    mean_loss = torch.tensor(losses).mean().item()
    mean_dice = torch.tensor(dices).mean().item()
    mean_iou = torch.tensor(ious).mean().item()

    return {"loss":mean_loss, "dice": mean_dice, "iou": mean_iou}
    #raise NotImplementedError("Return the mean loss, mean Dice, and mean IoU for the loader you evaluate.")


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

    torch.manual_seed(config.random_state)
    
    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    train_dict = None
    val_dict = None
    for i in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i} -> loss: {train_dict['loss']}, val_loss: {val_dict['loss']}, val_dice: {val_dict['dice']}, val_iou: {val_dict['iou']}")

    sample_path = BOOTCAMP_ROOT / "outputs" / "segmentation_prediction_samples.png"
    saved_path = save_prediction_samples(model, loaders["test"], sample_path)
    print(f"Saved prediction samples to: {saved_path}")

    return {"train_loss": train_dict['loss'],  "val_loss":val_dict['loss'], "val_dice":val_dict['dice'], "val_iou":val_dict['iou']}


    #raise NotImplementedError("Run the full training flow and return the final train and val metrics.")


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
