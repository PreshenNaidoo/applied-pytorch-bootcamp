"""
============================================================
EXERCISE 07: SELF SUPERVISED VISION PRETRAINING
============================================================

WHAT YOU ARE BUILDING
You are building a two-stage self-supervised workflow.

Stage 1:
- pretrain an encoder on unlabeled images

Stage 2:
- freeze the encoder
- train a small linear classifier on top of the frozen features

Use `STL10` for this exercise.

WHY THIS EXERCISE EXISTS
This is a very common research workflow.

You first learn general features from many images.
Then you test whether those features are useful on a labeled task.

This exercise makes that workflow explicit instead of hiding it.

DATASET
Use `torchvision.datasets.STL10`.

Default path:
- use the `unlabeled` split for self-supervised pretraining
- use the labeled `train` split for linear-probe training
- use the labeled `test` split for linear-probe evaluation

This file does not use a normal `60/20/20` split because the unlabeled pretraining stage is a separate dataset split.
That is the reason for the different setup here.

HOW TO LOAD OR DOWNLOAD THE DATASET
Use `torchvision.datasets.STL10(...)`.

Simple path:
- pretraining loader from `split="unlabeled"`
- linear-probe train loader from `split="train"`
- linear-probe eval loader from `split="test"`
- create two augmented views only for the pretraining stage
- keep plain eval transforms for the linear-probe stage

Use `download=True` the first time.

INPUTS AND OUTPUTS
Pretraining batch:
- type: tuple of tensors
- first view shape: `[batch_size, 3, 96, 96]`
- second view shape: `[batch_size, 3, 96, 96]`

Linear-probe batch:
- images shape: `[batch_size, 3, 96, 96]`
- labels shape: `[batch_size]`
- labels dtype: `torch.long`

Pretraining model output:
- `features`: `[batch_size, feature_dim]`
- `projections`: `[batch_size, projection_dim]`

Linear probe output:
- raw logits with shape `[batch_size, 10]`

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build:
- an encoder
- a projection head for the self-supervised stage
- a small linear probe for the frozen-feature stage

LOSS FUNCTION
For pretraining, use one contrastive-style self-supervised loss.

For linear evaluation, use `nn.CrossEntropyLoss`.

Important:
- the encoder should return useful features
- the projection head is for pretraining
- the linear probe uses frozen encoder features

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, move the model and both pretraining views to the same device.
- For the probe stage, move the frozen encoder, probe, image batch, and label batch to the same device.
- Device handling belongs inside your training and evaluation code, right before the forward pass.

METRICS TO TRACK
Track at least:
- `pretrain_loss`
  - mean over the whole self-supervised training epoch
- `probe_train_loss`
  - mean over the whole probe training epoch
- `probe_test_accuracy`
  - mean over the whole test loader for the probe

Success here is not one exact benchmark number.
Success means the pretrained encoder helps the probe do clearly better than random guessing on the test loader.

TARGET RESULT RANGE
Do not expect a huge score from a short educational run.

For STL10 with 10 classes:
- random guessing is about `0.10` accuracy
- a short run clearly above that can already mean the workflow is working

A very good extra check is to compare against a frozen random encoder.
If the pretrained encoder beats the random encoder, that is a strong success signal.

COMMON MISTAKES
- mixing labels into the self-supervised pretraining stage
- forgetting to freeze the encoder for the linear probe
- using projection-head outputs instead of encoder features for the probe
- applying heavy train augmentations during probe evaluation

STEP-BY-STEP TODO
1. Build the unlabeled pretraining loader.
2. Build the labeled probe loaders.
3. Build the encoder and projection head.
4. Pretrain the encoder.
5. Freeze the encoder.
6. Train a linear probe on top of frozen features.
7. Evaluate the probe.
8. Compare the result to random guessing.

STRETCH GOALS
- compare a pretrained encoder and a random frozen encoder
- unfreeze the encoder later and compare fine-tuning
- save the pretrained encoder to disk

DEFINITION OF DONE
You are done when:
- the two-stage workflow runs end to end
- the encoder is frozen during probe training
- the probe result is clearly above random guessing
- you can explain why pretraining and linear evaluation are separate steps
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SelfSupervisedConfig:
    """Settings for the self-supervised vision exercise."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    download: bool = False
    image_channels: int = 3
    image_size: int = 96
    num_classes: int = 10
    feature_dim: int = 256
    projection_dim: int = 128
    batch_size: int = 64
    learning_rate: float = 1e-3
    temperature: float = 0.1
    pretrain_epochs: int = 10
    probe_epochs: int = 5
    random_state: int = 42



def describe_ssl_workflow(config: SelfSupervisedConfig) -> dict[str, object]:
    """Return the expected shapes for the two-stage SSL workflow.

    Return a dict with exactly these keys:
    - `pretrain_view_shape`
    - `feature_shape`
    - `projection_shape`
    - `probe_output_shape`
    """
    return {
        "pretrain_view_shape": (config.batch_size, config.image_channels, config.image_size, config.image_size),
        "feature_shape": (config.batch_size, config.feature_dim),
        "projection_shape": (config.batch_size, config.projection_dim),
        "probe_output_shape": (config.batch_size, config.num_classes),
    }





class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        view_a = self.transform(image)
        view_b = self.transform(image)
        return view_a, view_b



class PlainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label



# Build the pretraining loader and the probe loaders.
# Return a dict with exactly these keys:
# - `pretrain`
# - `probe_train`
# - `probe_test`
# Use these exact STL10 splits:
# - `pretrain` uses the unlabeled split
# - `probe_train` uses the labeled train split
# - `probe_test` uses the labeled test split
# Beginner-friendly data path:
# - create a small new dataset class for the pretraining split
# - let it inherit from `torch.utils.data.Dataset`
# - give it:
#   - an `__init__` that stores the STL10 unlabeled dataset and the train transform
#   - a `__len__` that returns `len(self.dataset)`
#   - a `__getitem__` that reads one image and returns two views
# - inside the pretraining dataset `__getitem__`:
#   - read one unlabeled image
#   - apply the random train transform once to make `view_a`
#   - apply the same transform again to make `view_b`
#   - return `(view_a, view_b)`
# - for `probe_train` and `probe_test`, you do not need two views
# - those loaders should return normal `(image, label)` batches with a plain eval transform
# - only the self-supervised pretraining stage uses two augmented views
def load_dataloaders(config: SelfSupervisedConfig) -> dict[str, DataLoader]:

    raise NotImplementedError("Return the unlabeled pretraining loader and the labeled probe loaders.")







class SSLPretrainModel(nn.Module):
    """Encoder plus projection-head stub for self-supervised pretraining."""

    def __init__(self, input_channels: int, feature_dim: int, projection_dim: int) -> None:
        super().__init__()
        raise NotImplementedError("Build the encoder and projection head for self-supervised pretraining.")
        # Beginner-friendly build order:
        # 1. build an encoder that turns an image into one feature vector
        # 2. build a small projection head that maps:
        #    `feature_dim -> projection_dim`
        #
        # Simple default path:
        # - encoder: a few conv layers, activations, pooling, then flatten
        # - projection head: `Linear -> ReLU -> Linear`
        #
        # Important:
        # - the encoder output is the more general representation
        # - the projection head output is the one used for the contrastive loss





    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a dict with exactly these keys:
        - `features`
        - `projections`

        Step-by-step tensor path:
        - input image batch goes into the encoder
        - encoder returns one feature vector per image
        - that feature tensor should have shape `[batch_size, feature_dim]`
        - then pass those features into the projection head
        - projection output should have shape `[batch_size, projection_dim]`
        - return both because later stages use different parts
        """

        raise NotImplementedError("Return encoder features and projection features.")






class LinearProbe(nn.Module):
    """Small linear-probe stub used after encoder pretraining."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("Build a simple linear layer that maps frozen features to class logits.")
        # Beginner-friendly build idea:
        # - a linear probe is just a simple classifier on top of frozen features
        # - simplest path: one `nn.Linear(feature_dim, num_classes)`
        # - it reads encoder features and returns class logits



    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`."""

        raise NotImplementedError("Return raw class logits from the linear probe.")





# Build the pretraining model and optimizer.
# Return:
# - model: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
def build_pretraining_objects(config: SelfSupervisedConfig) -> tuple[nn.Module, torch.optim.Optimizer]:
    raise NotImplementedError("Return the pretraining model and optimizer.")





def contrastive_loss(projections_a: torch.tensor, projections_b: torch.tensor, temperature: float):
    
    if projections_a.dim() != 2 or projections_b.dim() != 2:
        raise ValueError("Inavild dimensions.")

    if projections_a.shape != projections_b.shape:
        raise ValueError("Shape mismatch between projections.")

    batch_size = projections_a.shape[0]

    #normalise
    projections_a = F.normalize(projections_a, dim=1)
    projections_b = F.normalize(projections_b, dim=1)

    # Get the cosine similarity scores, now that it has been normalised above, its easier. Just a simple dot product
    # However, we need to dotproduct each sample/row in 'a' with every other sample/row in 'b'. 
    # We can do this using matrix multiplications.
    # We multiply a tensor of shape (batch_size x num_features) with a tensor of shape (num_features x batch_size)
    # We transpose either projections_a or projections_b, depending on which we choose as the second matrix in the multiplication.
    # Remember the number of columns in the first matrix should equal the number of rows in the second matrix.
    # This is why we should transpose one of the tensors, either a or b
    # cosine similarity scores will be a matrix of batch_size x batch_size
    similarity_matrix = torch.matmul(projections_a, torch.transpose(projections_b, 0, 1))
    similarity_matrix = similarity_matrix / temperature    

    targets = torch.arange(start=0, end=batch_size, step=1)
    targets = targets.to(device=projections_a.device)

    # from a->b
    loss1 = F.cross_entropy(similarity_matrix, targets)

    #from b->a
    loss2 = F.cross_entropy(similarity_matrix.transpose(0, 1), targets)

    loss = (loss1+loss2)/2

    return loss


# Run one full self-supervised pretraining epoch.
# Return a dict with exactly these keys:
# - `loss`
# Here, `loss` means mean self-supervised pretraining loss over the whole epoch.
def pretrain_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float,
) -> dict[str, float]:
    # Beginner-friendly training flow:
    # - get `view_a, view_b` from the pretraining loader
    # - run the same model on `view_a`
    # - run the same model again on `view_b`
    # - read `outputs_a["projections"]` and `outputs_b["projections"]`
    # - compute the contrastive-style loss from those projections
    # - backpropagate and step the optimizer
    #
    # Important:
    # - use one shared model, not two different models
    # - this stage does not use class labels

    raise NotImplementedError("Return the mean self-supervised pretraining loss for one full epoch.")






# Build the probe loss and optimizer.
# Return:
# - probe: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# The probe should read frozen encoder features.
# The probe should return raw class logits.
# Important beginner note:
# - "frozen encoder" means the encoder weights should not update during probe training
# - simple ways to do that are:
#   - set encoder parameters to `requires_grad = False`
#   - and do not pass encoder parameters to the probe optimizer
# - the probe is the part you train in this stage, not the encoder
def build_probe_objects(config: SelfSupervisedConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    raise NotImplementedError("Return the linear probe, CrossEntropyLoss, and optimizer.")



# Run the full two-stage workflow.
# Return a final summary dict with exactly these keys:
# - `pretrain_loss`
# - `probe_train_loss`
# - `probe_test_accuracy`
# - `success_note`
# `probe_test_accuracy` means mean probe accuracy over the whole test loader.
def run_training(config: SelfSupervisedConfig) -> dict[str, float | str]:
    # High-level order:
    # 1. build the pretraining loader
    # 2. build and train the pretraining model
    # 3. freeze the encoder
    # 4. build the linear probe
    # 5. train the probe on labeled data
    # 6. evaluate the probe on the test split
    # 7. return the final summary

    raise NotImplementedError("Run the full two-stage workflow and return the final summary.")



    # Freeze weights for the next stage

    ########### Probe:


    #evaluate probe:





def check_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when the final summary contains the expected keys."""
    return "pretrain_loss" in metrics and "probe_train_loss" in metrics and "probe_test_accuracy" in metrics and "success_note" in metrics


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
    """Run the self-supervised vision scaffold one step at a time."""
    config = SelfSupervisedConfig()
    set_seed(config.random_state)
    print("Running exercise 07 starter scaffold...")
    print("Main goal: pretrain an encoder first, then test it with a frozen linear probe.")

    completed_steps = 0
    total_steps = 10


    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_ssl_workflow", describe_ssl_workflow, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress(
        "SSLPretrainModel.__init__",
        SSLPretrainModel,
        config.image_channels,
        config.feature_dim,
        config.projection_dim,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.image_channels, config.image_size, config.image_size)
    _, ok = report_task_progress("SSLPretrainModel.forward", model, sample_x)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    probe, ok = report_task_progress("LinearProbe.__init__", LinearProbe, config.feature_dim, config.num_classes)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_features = torch.randn(config.batch_size, config.feature_dim)
    _, ok = report_task_progress("LinearProbe.forward", probe, sample_features)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    pretraining_objects, ok = report_task_progress("build_pretraining_objects", build_pretraining_objects, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    pretrain_model, pretrain_optimizer = pretraining_objects

    _, ok = report_task_progress(
        "pretrain_one_epoch",
        pretrain_one_epoch,
        pretrain_model,
        loaders["pretrain"],
        pretrain_optimizer,
        config.temperature,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("build_probe_objects", build_probe_objects, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")


if __name__ == "__main__":
    main()

