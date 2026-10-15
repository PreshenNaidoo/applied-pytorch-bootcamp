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


        #raise NotImplementedError("Return raw class logits from the linear probe.")



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



@dataclass
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

    unlabelled_data = torchvision.datasets.STL10(config.data_root, split="unlabeled", download=True)
    train_data = torchvision.datasets.STL10(root=config.data_root, split="train", download=True)
    test_data = torchvision.datasets.STL10(root=config.data_root, split="test", download=True)

    resize_tf = torchvision.transforms.RandomResizedCrop(size = config.image_size, scale=(0.7, 1))
    color_tf = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    flip_tf = torchvision.transforms.RandomHorizontalFlip()
    gray_tf = torchvision.transforms.RandomGrayscale(p=0.1)
    to_tf = torchvision.transforms.ToTensor()
    unlabelled_transform = torchvision.transforms.Compose([resize_tf, color_tf, flip_tf, gray_tf, to_tf])

    plain_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    pretrain_ds = ContrastiveDataset(dataset=unlabelled_data, transform=unlabelled_transform)
    train_ds = PlainDataset(dataset=train_data, transform=plain_transform)
    test_ds = PlainDataset(dataset=test_data, transform=plain_transform)

    return {"pretrain":torch.utils.data.DataLoader(dataset=pretrain_ds, batch_size=config.batch_size, shuffle=True),
            "probe_train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "probe_test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}



class SSLPretrainModel(nn.Module):
    """Encoder plus projection-head stub for self-supervised pretraining."""

    def __init__(self, input_channels: int, feature_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
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

        self.image_net_weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = torchvision.models.mobilenet_v3_small(weights=self.image_net_weights)        

        last_output_size = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.feature_head = nn.Linear(in_features=last_output_size, out_features=feature_dim)

        self.projector = nn.Sequential(nn.Linear(in_features = feature_dim, out_features=projection_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(in_features=projection_dim, out_features=projection_dim)
                                      )

        #raise NotImplementedError("Build the encoder and projection head for self-supervised pretraining.")

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

        x = self.backbone(x)
        features = self.feature_head(x)

        projections = self.projector(features)

        return {"features":features, "projections":projections}


        #raise NotImplementedError("Return encoder features and projection features.")


class LinearProbe(nn.Module):
    """Small linear-probe stub used after encoder pretraining."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        # Beginner-friendly build idea:
        # - a linear probe is just a simple classifier on top of frozen features
        # - simplest path: one `nn.Linear(feature_dim, num_classes)`
        # - it reads encoder features and returns class logits

        self.linear_probe = nn.Linear(in_features=feature_dim, out_features=num_classes)

        #raise NotImplementedError("Build a simple linear layer that maps frozen features to class logits.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`."""

        return self.linear_probe(features)



    #raise NotImplementedError("Return the unlabeled pretraining loader and the labeled probe loaders.")


# Build the pretraining model and optimizer.
# Return:
# - model: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
def build_pretraining_objects(config: SelfSupervisedConfig) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = SSLPretrainModel(input_channels=config.image_channels, feature_dim=config.feature_dim, projection_dim=config.projection_dim)
    model.to(device)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate)

    return model, optimizer

    #raise NotImplementedError("Return the pretraining model and optimizer.")


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

    model.train()

    sum, c = 0,0
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

    #raise NotImplementedError("Return the mean self-supervised pretraining loss for one full epoch.")


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
    probe_model = LinearProbe(feature_dim=config.feature_dim, num_classes=config.num_classes)
    probe_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=probe_model.parameters(), lr=config.learning_rate)

    return probe_model, loss_fn, optimizer
    #raise NotImplementedError("Return the linear probe, CrossEntropyLoss, and optimizer.")


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

    torch.manual_seed(config.random_state)

    loaders = load_dataloaders(config)
    pretrain_model, pretrain_optimizer = build_pretraining_objects(config)    

    pretrain_dict = None
    for i in range(config.pretrain_epochs):

        pretrain_dict = pretrain_one_epoch(pretrain_model, loaders["pretrain"], pretrain_optimizer, config.temperature)
        print(f"Epoch: {i}, contrastive_loss: {pretrain_dict['loss']}")

    # Freeze weights for the next stage
    for param in pretrain_model.parameters():
        param.requires_grad = False

    ########### Probe:
    probe_model, probe_loss, probe_optimizer = build_probe_objects(config)
    probe_model.train()

    last_probe_loss = -1
    for i in range(config.probe_epochs):
        sum, c = 0,0
        for images, labels in loaders["probe_train"]:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                out_dict = pretrain_model(images)
                features = out_dict["features"]

            probe_optimizer.zero_grad()

            logits = probe_model(features)
            loss = probe_loss(logits, labels)
            loss.backward()
            probe_optimizer.step()

            sum+=loss.item()
            c+=1

        mean_loss = sum/c
        last_probe_loss = mean_loss

        print(f"Epoch: {i} -> probe_loss: {mean_loss}")

    #evaluate probe:
    probe_model.eval()
    sum, c = 0, 0
    accs = []
    for images, labels in loaders["probe_test"]:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out_dict = pretrain_model(images)
            features = out_dict["features"]
        logits = probe_model(features)

        # loss not necessary here but lets check it anyway for exercises purposes
        loss = probe_loss(logits, labels)
        sum+=loss.item()
        c+=1

        #accuracy
        preds = torch.argmax(logits, dim=1).float()
        correct = (preds == labels).float()
        accs.append(correct.mean().item())

    mean_test_loss = sum/c
    mean_probe_test_acc = torch.tensor(accs).mean().item()

    print(f"Probe Test: mean_test_loss: {mean_test_loss},  mean_probe_test_acc: {mean_probe_test_acc}")

    return {"pretrain_loss": pretrain_dict['loss'], "probe_train_loss": last_probe_loss, "probe_test_accuracy": mean_probe_test_acc, "success_note":"Ok."}
    #raise NotImplementedError("Run the full two-stage workflow and return the final summary.")


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

