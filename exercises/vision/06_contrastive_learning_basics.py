"""
============================================================
EXERCISE 06: CONTRASTIVE LEARNING BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a simple contrastive learning pipeline for images.

The core idea is:
- take one image
- make two different augmented views of it
- encode both views
- train the representations so matching views stay close and different images stay apart

Use `CIFAR10` for this first contrastive exercise.

WHY THIS EXERCISE EXISTS
This exercise is your first step into representation learning.

The goal is not just class labels.
The goal is to learn useful image features.

This is a common research idea because a good representation can help later tasks.

DATASET
Use `torchvision.datasets.CIFAR10`.

You can use unlabeled images for the contrastive stage, even though CIFAR10 has labels.
For the downstream check, you can use the labels for a small sanity check.

HOW TO LOAD OR DOWNLOAD THE DATASET
Use `torchvision.datasets.CIFAR10(...)`.

Simple path:
- load train and test
- combine them
- make your own `60% / 20% / 20%` split
- for contrastive pretraining, make a dataset that returns two augmented views of the same image
- for the downstream check, keep one plain-eval loader with labels

Use `download=True` the first time.

INPUTS AND OUTPUTS
Contrastive train batch:
- type: tuple of tensors
- first item shape: `[batch_size, 3, 32, 32]`
- second item shape: `[batch_size, 3, 32, 32]`
- meaning: two different augmented views of the same source images

Downstream eval batch:
- images shape: `[batch_size, 3, 32, 32]`
- labels shape: `[batch_size]`
- labels dtype: `torch.long`

Encoder output:
- type: `torch.Tensor`
- shape: `[batch_size, feature_dim]`

Projection head output:
- type: `torch.Tensor`
- shape: `[batch_size, projection_dim]`

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build:
- an image encoder
- a small projection head on top of the encoder

The encoder learns general features.
The projection head is mainly for the contrastive loss stage.

LOSS FUNCTION
Use one simple contrastive loss path.

A common beginner path is an InfoNCE-style loss.

Important high-level idea:
- matching views from the same image should become similar
- views from different images in the batch should become less similar

Do not fully hide this in a library call.
Write the important loss steps yourself.

METRICS TO TRACK
Track at least:
- `train_loss`
  - mean contrastive loss over the whole training epoch
- `nearest_neighbor_accuracy`
  - one simple downstream sanity-check number

For the sanity check, choose one clear path:
- a small linear probe
or
- nearest-neighbor accuracy in feature space

Use one clear default in this file: nearest-neighbor accuracy.

TARGET RESULT RANGE
Do not chase a perfect number here.

A good beginner success signal is:
- contrastive loss goes down in a sensible way
- downstream sanity-check accuracy is clearly above random guessing

For 10 classes, random guessing is about `0.10` accuracy.
If there are 10 classes, and you guess completely at random, you have a 1 out of 10 chance of being correct on each example.
So the expected accuracy is:  1 / 10 = 0.1
A short educational run doing clearly better than that is already useful.

COMMON MISTAKES
- making two identical views instead of two different views
- forgetting the projection head
- normalizing the wrong tensor in the loss
- using labels during the contrastive training stage by mistake
- expecting benchmark-level accuracy from a tiny educational setup

BEGINNER GUIDANCE
How to prepare the data:
- start from one CIFAR10 image
- make two separate random augmented views from that one image
- return both views together from the contrastive dataset
- do not use labels when computing the contrastive loss

Good beginner augmentations:
- `RandomResizedCrop`
- `RandomHorizontalFlip`
- light `ColorJitter`
- `RandomGrayscale`
- `ToTensor`

Where to apply the augmentations:
- apply them inside the dataset wrapper
- every time `__getitem__` reads one image, create `view_a` and `view_b`
- use the same augmentation pipeline twice, separately
- keep a different plain eval transform for the sanity-check loader

How the model is used:
- yes, a common setup is to call the same `ContrastiveModel` twice
- once for `view_a`
- once for `view_b`
- then use the two projection outputs in `contrastive_loss(...)`

How to think about the loss:
- row `i` in `projections_a` should match row `i` in `projections_b`
- all other rows in the batch are negatives for that anchor
- first normalize the projection vectors
- then compute similarity scores
- then divide by temperature
- then build targets so each row points to the same index in the other batch
- then compute the final scalar loss

Simple mental model:
- ask: "for this item in view A, which item in view B is its true match?"
- the correct answer is the item with the same batch index

STEP-BY-STEP TODO
1. Make a dataset wrapper that returns two augmented views.
2. Build the encoder.
3. Build the projection head.
4. Write the contrastive loss.
5. Train for a few epochs.
6. Run one downstream sanity check.
7. Explain what your sanity-check metric means.

STRETCH GOALS
- compare two augmentation strengths
- compare nearest-neighbor and linear-probe checks
- save the encoder and reuse it later

DEFINITION OF DONE
You are done when:
- the contrastive loader returns two views per image
- the model returns encoder features and projection features
- the contrastive loss runs end to end
- the downstream check is clearly above random guessing
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
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

def load_dataloaders(config: ContrastiveLearningConfig) -> dict[str, DataLoader]:

    train_data = torchvision.datasets.CIFAR10(config.data_root, train=True)
    test_data = torchvision.datasets.CIFAR10(config.data_root, train=False)

    data = torch.utils.data.ConcatDataset([train_data, test_data])

    train_ds, val_ds, test_ds = torch.utils.data.random_split(data, [0.6, 0.2, 0.2])

    resize_tf = torchvision.transforms.RandomResizedCrop(size = config.image_size, scale = (0.7, 1.0))
    horizon_tf = torchvision.transforms.RandomHorizontalFlip()
    # brightness - changes how light or dark the image is 
    # contrast - changes how strong the difference is between dark and bright parts 
    # saturation - changes how strong or faded the colors are
    # hue - shifts the color itself a little
    # For brightness=0.4: the brightness factor is sampled from about: [0.6, 1.4]
    # because it uses 1 - 0.4 to 1 + 0.4
    col_jitter_tf = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    grayscale_tf = torchvision.transforms.RandomGrayscale(p=0.1) # probability 10% of the time converted to grayscale
    to_tensor_tf = torchvision.transforms.ToTensor()
    train_transform = torchvision.transforms.Compose([resize_tf, horizon_tf, col_jitter_tf, grayscale_tf, to_tensor_tf])

    #ToTensor Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    eval_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_ds = ContrastiveDataset(dataset=train_ds, transform=train_transform)
    val_ds = EvalDataset(dataset=val_ds, transform=eval_transform)
    test_ds = EvalDataset(dataset=test_ds, transform=eval_transform)

    return {"contrastive_train": DataLoader(dataset = train_ds, batch_size=config.batch_size, shuffle=True),
            "eval": DataLoader(dataset=val_ds, batch_size=config.batch_size)}

class ContrastiveLearningConfig:
    """Settings for the contrastive learning exercise."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    download: bool = False
    image_channels: int = 3
    image_size: int = 32
    feature_dim: int = 128
    projection_dim: int = 64
    batch_size: int = 128
    learning_rate: float = 1e-3
    temperature: float = 0.1
    epochs: int = 10
    random_state: int = 42


def describe_contrastive_setup(config: ContrastiveLearningConfig) -> dict[str, object]:
    """Return the expected shapes for the contrastive workflow.

    Return a dict with exactly these keys:
    - `view_shape`
    - `feature_shape`
    - `projection_shape`
    - `downstream_metric_name`
    """
    return {
        "view_shape": (config.batch_size, config.image_channels, config.image_size, config.image_size),
        "feature_shape": (config.batch_size, config.feature_dim),
        "projection_shape": (config.batch_size, config.projection_dim),
        "downstream_metric_name": "nearest_neighbor_accuracy or linear_probe_accuracy",
    }


class ContrastiveModel(nn.Module):
    """Encoder plus projection-head stub for contrastive learning."""
  
    def __init__(self, input_channels: int, feature_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim

        self.weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = torchvision.models.mobilenet_v3_small(weights=self.weights)

        backbone_out_dim  = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.feature_head = nn.Linear(backbone_out_dim, feature_dim)

        self.projector = nn.Sequential(
            nn.Linear(in_features = feature_dim, out_features=projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = projection_dim, out_features=projection_dim)
        )

        # raise NotImplementedError("Build an encoder and a projection head for contrastive learning.")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return encoder features and projection features.

        Return a dict with exactly these keys:
        - `features`
        - `projections`

        Shapes:
        - `features`: `[batch_size, feature_dim]`
        - `projections`: `[batch_size, projection_dim]`
        """

        x = self.backbone(x)
        features = self.feature_head(x)

        projections = self.projector(features)

        return {"features": features, "projections":projections}

        #raise NotImplementedError("Return a dict with encoder features and projection features.")


# Build the contrastive train loader and one eval loader.
# Return a dict with exactly these keys:
# - `contrastive_train`
# - `eval`
# Use a 60% train / 20% val / 20% test split.
# You can use the train split for contrastive pretraining and the val split for the first sanity check.
# Beginner-friendly data path:
# - load CIFAR10 train and test
# - combine them
# - make your own 60/20/20 split
# - after splitting, do not build dataloaders from those split datasets yet
# - first create a small new dataset class for contrastive training
# - let it inherit from `torch.utils.data.Dataset`
# - give it:
#   - an `__init__` that stores the split dataset and the transform
#   - a `__len__` that returns `len(self.dataset)`
#   - a `__getitem__` that reads one item and returns the format you need
# - inside `__getitem__` for the contrastive dataset:
#   - read one original item from the split
#   - unpack it as `(image, label)`
#   - ignore the label
#   - apply the random train transform once to make `view_a`
#   - apply the same random train transform again to make `view_b`
#   - return `(view_a, view_b)`
# - create a second small dataset class for evaluation
# - it can also inherit from `torch.utils.data.Dataset`
# - inside its `__getitem__`:
#   - read `(image, label)`
#   - apply a plain eval transform to `image`
#   - return `(image, label)`
# - then make dataset objects from those classes
# - only after that should you create the dataloaders
#
# Good default augmentations for CIFAR10:
# - `transforms.RandomResizedCrop(config.image_size, scale=(0.6, 1.0))`
# - `transforms.RandomHorizontalFlip()`
# - `transforms.ColorJitter(...)`
# - `transforms.RandomGrayscale(p=...)`
# - `transforms.ToTensor()`
#
# Important:
# - the split decides which examples belong to train, val, and test
# - your new dataset class decides what each returned item looks like
# - contrastive train items should look like `(view_a, view_b)`
# - eval items should look like `(image, label)`
# - call the augmentation pipeline twice for the same source image
# - do not make one augmented tensor and reuse it as both views
# - the whole point is to create two related but not identical views
# - if `random_split(...)` does not accept fractions, compute integer sizes first

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

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image_transformed = self.transform(image)
        return image_transformed, label

    #raise NotImplementedError("Return the contrastive train loader and one downstream eval loader.")


# Build the model and optimizer.
# Return:
# - model: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
def build_training_objects(config: ContrastiveLearningConfig) -> tuple[nn.Module, torch.optim.Optimizer]:

    model = ContrastiveModel(input_channels = config.image_channels, feature_dim=config.feature_dim, projection_dim=config.projection_dim)
    model.to(device)

    optimizer = torch.optim.SGD(lr=config.learning_rate, params = model.parameters())

    return model, optimizer
    #raise NotImplementedError("Return the contrastive model and optimizer.")


# Compute one contrastive loss value for one batch.
# Return:
# - type: `torch.Tensor`
# - shape: scalar tensor
# - meaning: mean contrastive loss for the whole batch
#
# Step-by-step guidance for one simple InfoNCE-style path:
# 1. Start by checking the input shapes.
#    - `projections_a` should be `[batch_size, projection_dim]`
#    - `projections_b` should be `[batch_size, projection_dim]`
#    - both tensors should have the same shape
#
# 2. Normalize each row vector.
#    - each row is one example in the batch
#    - normalize along the last dimension, which is the feature dimension
#    - after this, row lengths become 1
#    - this makes the dot product act like cosine similarity
#    - PyTorch hint: look at `torch.nn.functional.normalize(..., dim=1)`
#
# 3. Compute all pair similarities between the two batches.
#    - you want every row in `a` compared with every row in `b`
#    - output shape should be `[batch_size, batch_size]`
#    - row `i`, column `j` means:
#      "how similar is item `i` from `a` to item `j` from `b`?"
#    - PyTorch hint: matrix multiply is the easiest path here
#    - one batch needs to stay `[batch_size, projection_dim]`
#    - the other needs to become `[projection_dim, batch_size]`
#    - that usually means using transpose on one tensor before `@` or `torch.matmul`
#
# 4. Divide the full similarity matrix by `temperature`.
#    - this keeps the shape the same: `[batch_size, batch_size]`
#    - smaller temperature makes the scores sharper
#    - bigger temperature makes the scores softer
#
# 5. Build the targets.
#    - you want a 1D tensor of class indices
#    - shape should be `[batch_size]`
#    - values should be `0, 1, 2, ..., batch_size - 1`
#    - this means:
#      row 0 should match column 0
#      row 1 should match column 1
#      row i should match column i
#    - PyTorch hint: `torch.arange(batch_size, device=...)`
#
# 6. Treat the similarity matrix like classifier logits.
#    - each row is one training example
#    - each column is one possible match in the other batch
#    - the correct class for row `i` is column `i`
#    - this is why a standard classification loss works here
#    - PyTorch hint: `nn.CrossEntropyLoss` expects:
#      logits with shape `[N, C]`
#      targets with shape `[N]`
#
# 7. Compute one loss from `a -> b`.
#    - use the similarity matrix as logits
#    - use the target indices as labels
#    - this asks:
#      "for each item in `a`, did the model give the highest score to the matching item in `b`?"
#
# 8. Compute one loss from `b -> a`.
#    - do the same thing in the other direction
#    - easiest way: reuse the same idea with the transposed similarity matrix
#    - shape after transpose is still valid for cross-entropy
#
# 9. Average the two losses into one scalar.
#    - this final scalar is what you return
#    - this is the value you will call `.backward()` on during training
#
# Tiny batch example:
# - if `batch_size = 3`, your target tensor should be `[0, 1, 2]`
# - row 0 should pick column 0
# - row 1 should pick column 1
# - row 2 should pick column 2
#
# Why do both directions?
# - it gives a more balanced training signal
# - both view sets get treated as anchors
#
# How this compares to classic SimCLR NT-Xent:
# - this exercise uses a simpler symmetric InfoNCE-style setup
# - you keep two separate batches:
#   - `projections_a`: `[N, D]`
#   - `projections_b`: `[N, D]`
# - then you build one similarity matrix with shape `[N, N]`
# - row `i` in `a` should match column `i` in `b`
# - then you do the reverse direction too: `b -> a`
# - then you average the two losses
#
# Classic SimCLR often does the same idea in a different layout:
# - stack both views together into one tensor of shape `[2N, D]`
# - build one big similarity matrix of shape `[2N, 2N]`
# - each example has one positive partner
# - all the other examples in the big matrix act as negatives
#
# So the main difference is:
# - this file uses two smaller `N x N` comparisons
# - classic SimCLR often uses one larger `2N x 2N` comparison
#
# The core goal is still the same:
# - make positive pairs similar
# - make non-matching pairs less similar
#
# Tiny example with `N = 3`:
# - `a = [a0, a1, a2]`
# - `b = [b0, b1, b2]`
# - positive pairs are:
#   - `a0 <-> b0`
#   - `a1 <-> b1`
#   - `a2 <-> b2`
#
# In this simpler exercise version:
# - for anchor `a0`, compare only against `[b0, b1, b2]`
# - positive is `b0`
# - negatives are `b1` and `b2`
# - then do the reverse for `b0`, comparing against `[a0, a1, a2]`
#
# In classic SimCLR:
# - stack everything into `[a0, a1, a2, b0, b1, b2]`
# - for anchor `a0`, the positive is `b0`
# - negatives are `a1, a2, b1, b2`
# - so each anchor usually sees more negatives
#
# Why use this simpler version here?
# - it is easier to read
# - it is easier to debug
# - it still teaches the main contrastive-learning idea clearly
#
# Important:
# - the positive pair is not found by class labels here
# - it is found by matching the same batch index across the two view tensors
# - all other positions in that row act like negatives
def contrastive_loss(projections_a: torch.Tensor, projections_b: torch.Tensor, temperature: float) -> torch.Tensor:

    if projections_a.dim() != 2 or projections_b.dim() != 2:
        raise ValueError("Incorrect input parameter dimensions")

    if projections_a.shape != projections_b.shape:
        raise ValueError("Incorrect input parameter shape mismatch")

    batch_size = projections_a.shape[0]

    # normalise so dot products are cosine similarity score
    projections_a = F.normalize(projections_a, dim=-1)
    projections_b = F.normalize(projections_b, dim=-1)

    # output matrix of scores with shape batch_size x batch_size
    #each sample/row in a is compared with each sample/row in b
    similarity_matrix = torch.matmul(projections_a, projections_b.transpose(0, 1))

    similarity_matrix = similarity_matrix / temperature

    targets = torch.arange(start=0, end=batch_size, step=1, device=projections_a.device)

    # loss from a to b
    loss1 = F.cross_entropy(input = similarity_matrix, target=targets)
    
    # loss from b to a
    # You can transpose here, or you can redo the matmul step but this time transpose_a and then divide by temperature again
    # Transposing is easier:
    loss2 = F.cross_entropy(input = similarity_matrix.transpose(0, 1), target=targets)
    
    # By default, F.cross_entropy(...) already reduces the batch losses to one scalar by taking the mean.
    # So no need to do:
    #loss = loss1.mean(dim=0) + loss2.mean(dim=0)    
    #loss/=2.0

    loss = (loss1 + loss2)/2.0

    return loss

    #raise NotImplementedError("Return one scalar contrastive loss for the batch.")


# Run one full contrastive training epoch.
# Return a dict with exactly these keys:
# - `loss`
# Here, `loss` means mean contrastive train loss over the whole epoch.
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float,
) -> dict[str, float]:
    # Training flow:
    # - get `view_a, view_b` from the loader
    # - run `model(view_a)`
    # - run `model(view_b)`
    # - take the `projections` from both outputs
    # - pass them into `contrastive_loss(...)`
    # - backpropagate that scalar loss
    #
    # Important:
    # - this uses one model with shared weights
    # - you are not building two separate models
    # - calling the same model twice is the normal contrastive pattern

    model.train()

    c, sum = 0,0
    for view_a, view_b in loader:
        view_a = view_a.to(device)
        view_b = view_b.to(device)

        optimizer.zero_grad()

        a_dict = model(view_a)
        b_dict = model(view_b)

        projections_a = a_dict["projections"]
        projections_b = b_dict["projections"]

        loss = contrastive_loss(projections_a, projections_b, temperature)
        loss.backward()
        optimizer.step()

        sum+=loss.item()
        c+=1

    mean_loss = sum/c

    return {"loss": mean_loss}

    #raise NotImplementedError("Return the mean contrastive train loss for one full epoch.")


# Run one simple downstream sanity check.
# Return a dict with exactly these keys:
# - `metric_name`
# - `metric_value`
# `metric_value` should be a Python float.
#
# What this sanity check means, in plain words:
# - after contrastive training, we want to know whether similar images ended up near each other in feature space
# - one simple way to test that is nearest-neighbor accuracy
# - for each eval image, we encode it into a feature vector
# - then we look for the most similar other image feature in the eval set
# - we copy that neighbor's label as the prediction
# - if the predicted label matches the true label, that example counts as correct
# - if many examples are correct, that means the learned feature space has useful structure
# - this is only a sanity check, not a full benchmark
@torch.no_grad()
def run_representation_sanity_check(model: nn.Module, loader: DataLoader) -> dict[str, object]:

    all_features = []
    all_labels = []

    model.eval()

    for images, y_targets in loader:
        images = images.to(device)
        y_targets = y_targets.to(device)
        
        outputs_dict = model(images)
        features = outputs_dict["features"]

        all_features.append(features)
        all_labels.append(y_targets)

    all_features = torch.concat(all_features)
    all_labels = torch.concat(all_labels)

    # If you normalize first, the dot product becomes cosine similarity.
    # That means similar direction in feature space = higher score
    all_features = F.normalize(all_features, dim=1)

    # Imagine you have 5 eval images.
    # After encoding them, you get 5 feature vectors.
    # For image 3:
    # compare feature 3 to features 0, 1, 2, 4
    # suppose feature 2 is the most similar
    # then predict image_3 label as label 2
    # if label 2 equals the true label of image 3, that counts as correct
    # Do that for every example.
    # Then average correctness.

    similarity_matrix = torch.matmul(all_features, all_features.transpose(0, 1))

    # so we dont do self-match because cosine sim with itself will always be 1
    # do not use fill_value = 0 because some real similarities to other examples can be negative
    # if all the other values in a row are less than 0, then the diagonal 0 becomes the biggest value
    # then argmax will pick the same example itself
    similarity_matrix.fill_diagonal_(fill_value=float("-inf"))

    nearest_indices = similarity_matrix.argmax(dim=1)
    predicted_labels = all_labels[nearest_indices]

    accuracy = (predicted_labels == all_labels).float().mean().item()

    return {
        "metric_name": "nearest_neighbor_accuracy",
        "metric_value": accuracy,
    }

    #raise NotImplementedError("Return one simple downstream representation sanity-check result.")


# Run the full contrastive workflow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `sanity_check_name`
# - `nearest_neighbor_accuracy`
def run_training(config: ContrastiveLearningConfig) -> dict[str, float | str]:

    loaders = load_dataloaders(config)
    model, optimizer = build_training_objects(config)

    train_dict = None
    eval_dict = None
    for i in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["contrastive_train"], optimizer, config.temperature)
        eval_dict = run_representation_sanity_check(model, loaders["eval"])

        print(f"Epoch {i} -> contrastive_loss: {train_dict['loss']} , {eval_dict['metric_name']}:{eval_dict['metric_value']}")


    return {"train_loss": train_dict['loss'], "": eval_dict['metric_name'], "": eval_dict['metric_value']}
    #raise NotImplementedError("Run the full contrastive workflow and return the final loss and downstream check.")


def check_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when the final summary contains the expected keys."""
    return "train_loss" in metrics and "sanity_check_name" in metrics and "sanity_check_accuracy" in metrics


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
    """Run the contrastive learning scaffold one step at a time."""
    config = ContrastiveLearningConfig()
    set_seed(config.random_state)
    print("Running exercise 06 starter scaffold...")
    print("Main goal: learn image representations from two augmented views of the same image.")

    completed_steps = 0
    total_steps = 9

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_contrastive_setup", describe_contrastive_setup, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress(
        "ContrastiveModel.__init__",
        ContrastiveModel,
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
    sample_outputs, ok = report_task_progress("ContrastiveModel.forward", model, sample_x)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress(
        "contrastive_loss",
        contrastive_loss,
        sample_outputs["projections"],
        sample_outputs["projections"],
        config.temperature,
    )
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

    training_model, optimizer = training_objects

    _, ok = report_task_progress(
        "train_one_epoch",
        train_one_epoch,
        training_model,
        loaders["contrastive_train"],
        optimizer,
        config.temperature,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress(
        "run_representation_sanity_check",
        run_representation_sanity_check,
        training_model,
        loaders["eval"],
    )
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


