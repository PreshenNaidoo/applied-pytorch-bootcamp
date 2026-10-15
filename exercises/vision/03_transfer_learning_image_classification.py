"""
============================================================
EXERCISE 03: TRANSFER LEARNING IMAGE CLASSIFICATION
============================================================

WHAT YOU ARE BUILDING
You are building an image classifier with a pretrained backbone.

The default path in this exercise is:
1. load a pretrained image model
2. replace the classifier head
3. freeze the backbone first
4. train only the new head
5. optionally unfreeze some layers later for fine-tuning

WHY THIS EXERCISE EXISTS
Transfer learning is a very common research and engineering workflow.

It helps because a pretrained backbone already knows many useful visual patterns:
- edges
- textures
- shapes
- simple object parts

That means you usually need less data and less training time than training from scratch.

DATASET
Use `torchvision.datasets.CIFAR10`.

Keep all 10 classes for this exercise.

Because most pretrained image models expect larger images, resize CIFAR10 images to `224 x 224`.

HOW TO LOAD OR DOWNLOAD THE DATASET
Use `torchvision.datasets.CIFAR10(...)`.

Simple path:
- load train and test
- combine them
- make your own `60% / 20% / 20%` train / val / test split
- resize images to `224 x 224`
- use ImageNet normalization if you use ImageNet pretrained weights

Important library note:
- one `torchvision` dataset object only stores one transform
- if you want different train and val/test transforms, use separate dataset objects or a small wrapper after splitting

Use `download=True` the first time.

INPUTS AND OUTPUTS
Input image batch:
- type: `torch.Tensor`
- shape: `[batch_size, 3, 224, 224]`

Label batch:
- type: `torch.Tensor`
- dtype: `torch.long`
- shape: `[batch_size]`
- values: integers from `0` to `9`

Model output:
- type: `torch.Tensor`
- shape: `[batch_size, 10]`
- meaning: one raw logit per class for each image

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a pretrained CNN backbone such as `resnet18`.

High-level steps:
- load pretrained weights
- keep the backbone body
- replace the final classifier layer so it outputs 10 classes
- freeze the backbone first

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

Important:
- return raw logits
- do not apply softmax before the loss

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, move the model, image batch, and label batch to the same device.
- Device handling belongs inside your training and evaluation code, right before the forward pass.

METRICS TO TRACK
Track at least:
- `train_loss`
  - mean over the whole training epoch
- `train_accuracy`
  - mean over the whole training epoch
- `val_loss`
  - mean over the whole validation loader
- `val_accuracy`
  - mean over the whole validation loader

If you pass the test loader to `evaluate(...)`, return test loss and test accuracy for that loader.

TARGET RESULT RANGE
For a short frozen-backbone run, a reasonable validation accuracy is often around `0.60` to `0.80`.

If you later fine-tune part of the backbone, you may do better.

COMMON MISTAKES
- forgetting to replace the classifier head
- forgetting to freeze the backbone at first
- using the wrong output size in the new head
- forgetting to resize images to the backbone input size
- using CIFAR10 normalization while also using ImageNet pretrained weights
- moving the model to GPU but leaving images or labels on CPU

STEP-BY-STEP TODO
1. Load CIFAR10 and make a `60/20/20` split.
2. Resize images to `224 x 224`.
3. Normalize with the pretrained model's expected stats.
4. Load a pretrained backbone.
5. Replace the classifier head.
6. Freeze the backbone first.
7. Train the new head.
8. Evaluate on the validation loader.
9. Optionally fine-tune later as a stretch goal.

STRETCH GOALS
- compare frozen backbone vs partial fine-tuning
- compare two pretrained backbones
- print how many parameters are trainable before and after unfreezing

DEFINITION OF DONE
You are done when:
- the model trains end to end
- the head is replaced correctly
- the frozen-backbone run learns something useful
- you can explain why transfer learning often helps
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import os
import numpy as np

import torch
from torch.ao.quantization.observer import weight_observer_range_neg_127_to_127
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, resnet18


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
# Labels must stay as integer class ids with dtype `torch.long`.
def load_dataloaders(config: TransferLearningConfig) -> dict[str, DataLoader]:

    # later on we use a resnet18 backbone with imagenet weights,
    # this expects the input to be of a certain size and normalised a certain way.
    # so we just use the transform that comes with the weights to help with this.
    # Apparently this is the best place to do it. 
    # not the best place: inside forward() or inside the training loop
    # Also, no need to permute:
    # using the dataset with transform=..., turns each image into a PyTorch tensor in the right order already.

    #weights = torchvision.models.ResNet18_Weights.DEFAULT
    raise NotImplementedError("Return train, val, and test loaders for resized CIFAR10.")



    # dataset object handles image prep, subset or split handles which examples belong to train/val/test, dataloader handles batching
    # the above is correct.
    # when you use data_train.data, it looks at the raw data and ignores the transform.    

    # print(data_train.data.shape)
    # print(type(data_train.targets))

    # data = np.concat([data_train.data, data_test.data])
    # targets = np.concat([data_train.targets, data_test.targets])

    # num_train, num_val = int(data.shape[0]*0.6), int(data.shape[0]*0.2)

    # x_train, y_train = torch.tensor(data[:num_train], dtype=torch.float), torch.tensor(targets[:num_train], dtype=torch.long)
    # x_val, y_val = torch.tensor(data[num_train:num_train+num_val], dtype=torch.float), torch.tensor(targets[num_train:num_train+num_val], dtype=torch.long)
    # x_test, y_test = torch.tensor(data[num_train+num_val:], dtype=torch.float), torch.tensor(targets[num_train+num_val:], dtype=torch.long)
   
    # x_train = x_train.permute(0,3,1,2)
    # x_val = x_val.permute(0,3,1,2)
    # x_test = x_test.permute(0,3,1,2)

    # train_ds = TensorDataset(x_train, y_train)
    # val_ds = TensorDataset(x_val, y_val)
    # test_ds = TensorDataset(x_test, y_test)
    
    # if os.cpu_count() is something like 8, use 8 if os.cpu_count() is None, use 1
    # the or handles the none case


@dataclass
class TransferLearningConfig:
    """Settings for the transfer learning exercise."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    download: bool = False
    image_size: int = 224
    num_classes: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    random_state: int = 42
    target_val_accuracy: float = 0.65


def describe_transfer_plan(config: TransferLearningConfig) -> dict[str, object]:
    """Return the high-level transfer learning plan.

    Return a dict with exactly these keys:
    - `backbone_name`
    - `input_shape`
    - `output_shape`
    - `freeze_backbone_first`
    """
    return {
        "backbone_name": "resnet18",
        "input_shape": (3, config.image_size, config.image_size),
        "output_shape": (config.batch_size, config.num_classes),
        "freeze_backbone_first": True,
    }


class TransferLearningClassifier(nn.Module):
    """Pretrained-model stub for CIFAR10 transfer learning."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError( "Load a pretrained backbone, replace the final classifier head, and freeze the backbone first." )

        ######################## code for resnet18
        # self.weights = torchvision.models.ResNet18_Weights.DEFAULT #imagenet
        # self.backbone = torchvision.models.resnet18(weights=self.weights)        

        # for param in self.backbone.parameters():
        #     param.requires_grad = False 

        # # all old ResNet layers are frozen 
        # # the new fc layer below is trainable by default

        # in_features = self.backbone.fc.in_features
        # self.backbone.fc = nn.Linear(in_features=in_features, out_features=num_classes)
        ########################


        # for param in self.backbone.parameters():
        #     param.requires_grad = False



        # no need to do this below because the new fc layer below is trainable by default
        # for param in self.backbone.fc.parameters():
        #     param.requires_grad = True



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`."""

        raise NotImplementedError("Return raw logits from the transfer learning model.")





# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.CrossEntropyLoss`.
# Only optimize the trainable parameters.
def build_training_objects(config: TransferLearningConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    raise NotImplementedError("Return the transfer model, CrossEntropyLoss, and optimizer.")





# Run one full training epoch.
# Return a dict with exactly these keys:
# - `loss`
# - `accuracy`
# Both values should be Python floats.
# Here:
# - `loss` means mean train loss over the whole epoch
# - `accuracy` means mean train accuracy over the whole epoch
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:

    raise NotImplementedError("Return the mean train loss and mean train accuracy for one full epoch.")






# Run one full evaluation pass.
# Return a dict with exactly these keys:
# - `loss`
# - `accuracy`
# Both values should be Python floats.
# If you pass the validation loader:
# - `loss` means mean validation loss over the whole validation loader
# - `accuracy` means mean validation accuracy over the whole validation loader
# If you pass the test loader:
# - `loss` means mean test loss over the whole test loader
# - `accuracy` means mean test accuracy over the whole test loader
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    raise NotImplementedError("Return the mean loss and mean accuracy for the loader you evaluate.")







# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `train_accuracy`
# - `val_loss`
# - `val_accuracy`
def run_training(config: TransferLearningConfig) -> dict[str, float]:

    raise NotImplementedError("Run the full training flow and return the final train and val metrics.")



    # Freezing the backbone and training only the head gave much lower validation accuracy than 
    # full fine-tuning. The head-only model got stuck around 46.7% validation accuracy, while full 
    # fine-tuning reached about 75.3%. This suggests the pretrained ImageNet features were helpful, 
    # but they did not match CIFAR10 well enough on their own. Letting all layers train gave the model 
    # enough flexibility to adapt to the new dataset and perform much better.

    # The results shown in the above paragraph was actually caused by a small bug I had and it did
    # not use the weights. The bug was caused by using the raw data object i.e. data_train.data

    # With the fixed data pipeline, both transfer learning setups improved a lot, but full fine-tuning 
    # performed best. Training only the classification head reached about 77.2% validation accuracy 
    # after 10 epochs, which already showed that the pretrained ImageNet features were now being used 
    # properly. Full fine-tuning did even better, reaching about 87.3% validation accuracy, so letting 
    # the whole model adapt gave a clear extra gain. This suggests the pretrained features gave a strong 
    # starting point, and once the preprocessing was corrected, the backbone could transfer well to CIFAR10, 
    # with full fine-tuning giving the best final performance.





def check_metrics(metrics: dict[str, float], target: float) -> bool:
    """Return True when validation accuracy reaches the target."""
    return check_threshold(metrics.get("val_accuracy", 0.0), target)


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
    """Run the transfer learning scaffold one step at a time."""
    config = TransferLearningConfig()
    set_seed(config.random_state)
    print("Running exercise 03 starter scaffold...")
    print("Main goal: use a pretrained backbone and replace the classifier head.")

    completed_steps = 0
    total_steps = 8


    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_transfer_plan", describe_transfer_plan, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("TransferLearningClassifier.__init__", TransferLearningClassifier, config.num_classes)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, 3, config.image_size, config.image_size)
    _, ok = report_task_progress("TransferLearningClassifier.forward", model, sample_x)
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
        passed = check_metrics(metrics, config.target_val_accuracy)
        print_pass_or_try_again(
            passed,
            "val_accuracy",
            float(metrics["val_accuracy"]),
            f">= {config.target_val_accuracy:.2f}",
        )


if __name__ == "__main__":
    main()

