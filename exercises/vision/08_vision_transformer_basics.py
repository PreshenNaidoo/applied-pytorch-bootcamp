"""
============================================================
EXERCISE 08: VISION TRANSFORMER BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a toy Vision Transformer, often called a ViT.

Use `FashionMNIST` so the patch shapes stay easy to follow.

WHY THIS EXERCISE EXISTS
A Vision Transformer works very differently from a CNN.

Instead of convolution filters moving across the image, you:
- cut the image into small patches
- turn each patch into an embedding vector
- add positional information
- run transformer encoder blocks over the patch sequence
- classify from a class token or a pooled output

DATASET
Use `torchvision.datasets.FashionMNIST`.

Each image has shape `[1, 28, 28]`.

A simple toy path is:
- patch size `4`
- number of patches per side = `28 / 4 = 7`
- total patches = `7 * 7 = 49`

HOW TO LOAD OR DOWNLOAD THE DATASET
Use `torchvision.datasets.FashionMNIST(...)`.

Simple path:
- load train and test
- combine them
- make your own `60% / 20% / 20%` train / val / test split
- use simple tensor + normalize transforms

Use `download=True` the first time.

INPUTS AND OUTPUTS
Input image batch:
- type: `torch.Tensor`
- shape: `[batch_size, 1, 28, 28]`

Label batch:
- type: `torch.Tensor`
- dtype: `torch.long`
- shape: `[batch_size]`

Patch embedding path for patch size `4`:
- input image: `[batch_size, 1, 28, 28]`
- raw flattened patches: `[batch_size, 49, 16]`
  - because `28 / 4 = 7` patches fit across
  - and `28 / 4 = 7` patches fit down
  - so total patches = `7 * 7 = 49`
  - because each patch has `1 * 4 * 4 = 16` values
- embedded patches: `[batch_size, 49, embed_dim]`
- with class token added: `[batch_size, 50, embed_dim]`

Final model output:
- type: `torch.Tensor`
- shape: `[batch_size, 10]`
- meaning: one raw logit per class

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build these parts:
- patch embedding
- positional embeddings
- transformer encoder blocks
- class token or pooled output
- classifier head

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
A toy ViT on FashionMNIST can work, but it may be less forgiving than a small CNN.

A reasonable target for a simple correct implementation is often around `0.70` to `0.88` validation accuracy.

COMMON MISTAKES
- using an image size that is not divisible by the patch size
- getting the patch flatten shape wrong
- forgetting the positional embeddings
- forgetting the class token or pooled output step
- applying softmax before `CrossEntropyLoss`

STEP-BY-STEP TODO
1. Split the image into patches.
2. Flatten each patch.
3. Project patches into embeddings.
4. Add positional embeddings.
5. Add a class token or use a pooled output.
6. Build transformer encoder blocks.
7. Add the classifier head.
8. Train and evaluate the toy ViT.

STRETCH GOALS
- compare class token vs mean pooling
- compare two patch sizes
- compare the toy ViT to your earlier FashionMNIST CNN

DEFINITION OF DONE
You are done when:
- you can explain the shape at each stage
- the model returns raw logits with shape `[batch_size, 10]`
- the toy ViT trains end to end
- the validation accuracy is in a sensible range
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pickletools import optimize
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
from common.training_checks import check_threshold, print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
        #raise NotImplementedError("Return raw logits from the toy ViT.")


# Build the train, validation, and test loaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Use a 60% train / 20% val / 20% test split.
# Beginner-friendly data path:
# - load FashionMNIST train and test
# - combine them
# - make your own 60/20/20 split
# - use a simple transform like `ToTensor()` plus normalization
# - if `random_split(...)` does not accept fractions, compute integer sizes first
def load_dataloaders(config: VisionTransformerConfig) -> dict[str, DataLoader]:

    plain_tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    data_train = torchvision.datasets.FashionMNIST(root=config.data_root, train=True, transform=plain_tf, download=True)
    data_test = torchvision.datasets.FashionMNIST(root=config.data_root, train=False, transform=plain_tf, download=True)

    data = torch.utils.data.ConcatDataset([data_train, data_test])
    train_ds, val_ds, test_ds = torch.utils.data.random_split(data, [0.6, 0.2, 0.2])

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}

class VisionTransformerConfig:
    """Settings for the toy ViT exercise."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    download: bool = False
    image_channels: int = 1
    image_size: int = 28
    patch_size: int = 4
    num_classes: int = 10
    embed_dim: int = 64
    num_heads: int = 4
    mlp_dim: int = 128
    num_layers: int = 2
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 8
    random_state: int = 42
    target_val_accuracy: float = 0.72


def describe_patch_math(config: VisionTransformerConfig) -> dict[str, object]:
    """Return the expected patch shapes.

    Return a dict with exactly these keys:
    - `num_patches`
    - `raw_patch_dim`
    - `patch_tensor_shape`
    - `token_sequence_shape`
    """
    patches_per_side = config.image_size // config.patch_size
    num_patches = patches_per_side * patches_per_side
    raw_patch_dim = config.image_channels * config.patch_size * config.patch_size
    return {
        "num_patches": num_patches,
        "raw_patch_dim": raw_patch_dim,
        "patch_tensor_shape": (config.batch_size, num_patches, raw_patch_dim),
        "token_sequence_shape": (config.batch_size, num_patches + 1, config.embed_dim),
    }


class PatchEmbedding(nn.Module):
    """Patch-embedding stub for the toy ViT."""

    def __init__(self, image_channels: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Beginner-friendly build idea:
        # - this class has one main job:
        #   turn image patches into embedding vectors
        # - each patch starts as a small flat vector
        # - then a linear layer maps that flat vector to `embed_dim`
        #
        # For FashionMNIST with patch size 4:
        # - one patch has shape `[1, 4, 4]`
        # - flatten that patch to length `16`
        # - then project `16 -> embed_dim`
        #
        # Simple default object to create here:
        # - one `nn.Linear(raw_patch_dim, embed_dim)`
        #
        # `raw_patch_dim` means:
        # - `image_channels * patch_size * patch_size`

        raw_patch_dim = image_channels * patch_size * patch_size
        self.patch_proj = nn.Linear(in_features= raw_patch_dim, out_features=embed_dim)
        
        #raise NotImplementedError("Split images into patches, flatten them, and project them into embedding vectors.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedded patches with shape `[batch_size, num_patches, embed_dim]`.

        Step-by-step tensor path:
        - start with `x` of shape `[batch_size, 1, 28, 28]`
        - split each image into small `patch_size x patch_size` blocks
        - flatten each patch into one short vector
        - this should give shape `[batch_size, num_patches, raw_patch_dim]`
        - then run your linear layer on the last dimension
        - final shape should be `[batch_size, num_patches, embed_dim]`

        Simple mental model:
        - one image becomes a list of patches
        - each patch becomes one token vector

        Beginner PyTorch hints:
        - `unfold(...)` is one common way to extract patches
        - after unfolding, you usually need `reshape(...)` or `flatten(...)`
        - `nn.Linear` can act on the last dimension of a 3D tensor directly
        """

        patches = F.unfold(input=x, kernel_size=self.patch_size, stride=self.patch_size) # This returns a tensor of shape (64, 16, 49)
        patches = patches.permute(0, 2, 1) # This returns a tensor of shape (64, 49, 16)
        embeddings = self.patch_proj(patches) #This returns (64, 49, 64)
        return embeddings

        #raise NotImplementedError("Return the embedded patch sequence.")


class ToyVisionTransformer(nn.Module):
    """Toy ViT stub for FashionMNIST classification."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        # Picture of the full toy ViT:
        #
        # image `[batch_size, 1, 28, 28]`
        #   -> split into `4 x 4` patches
        #   -> 49 flat patch vectors
        #      `[batch_size, 49, 16]`
        #   -> linear layer turns each patch into an embedding
        #      `[batch_size, 49, embed_dim]`
        #   -> add one class token at the front
        #      `[batch_size, 50, embed_dim]`
        #   -> add positional information
        #      `[batch_size, 50, embed_dim]`
        #   -> pass through transformer encoder blocks
        #      `[batch_size, 50, embed_dim]`
        #   -> take the first token as the image summary
        #      `[batch_size, embed_dim]`
        #   -> classifier head
        #      `[batch_size, 10]`
        #
        # Simple picture:
        # [patch1] [patch2] [patch3] ... [patch49]
        #     becomes
        # [CLS] [patch1] [patch2] [patch3] ... [patch49]
        #
        # The transformer lets every token look at every other token.
        # That means the class token can gather information from all patches.
        # At the end, you use that class token to make the final prediction.

        # Beginner-friendly build order:
        # 1. create `self.patch_embed = PatchEmbedding(...)`
        # 2. compute how many patches each image has
        # 3. create one learnable class token
        #    - after patch embedding, one image looks like this:
        #      `[patch1] [patch2] ... [patch49]`
        #    - now add one extra token at the front:
        #      `[CLS] [patch1] [patch2] ... [patch49]`
        #    - this extra token is called the class token
        #    - it is a learned vector
        #    - during the transformer, it can gather information
        #      from all the patch tokens
        #    - at the end, you use it as a summary of the whole image
        #    - shape `[1, 1, embed_dim]` means:
        #      store one class token, and that token has length `embed_dim`
        #    - later in `forward()`, copy that one token across the batch
        # 4. create learnable positional embeddings
        #    - transformers do not know where tokens came from by default
        #    - so you add position information to every token
        #    - this tells the model which patch is first, second, and so on
        #    - the class token also needs a position
        #    - so you need positions for:
        #      `num_patches` patch tokens + `1` class token
        #    - total positions = `num_patches + 1`
        #    - shape `[1, num_patches + 1, embed_dim]` means:
        #      one position vector for every token in the full sequence
        # 5. create one or more transformer encoder blocks
        # 6. create the final classifier head
        #
        # Simple beginner path for step 5:
        # - use `nn.TransformerEncoderLayer(...)`
        # - set `batch_first=True`
        # - then stack them with `nn.TransformerEncoder(...)`
        #
        self.patch_embed = PatchEmbedding(image_channels=config.image_channels, 
                                          patch_size=config.patch_size, 
                                          embed_dim=config.embed_dim)

        self.num_patches = (config.image_size // config.patch_size)**2  # double slash gives an integer

        self.class_token = nn.Parameter(torch.zeros([1, 1, config.embed_dim], dtype=torch.float))

        self.positional_embed = nn.Parameter(torch.zeros([1, self.num_patches+1, config.embed_dim]))
        
        self.encoder = nn.TransformerEncoderLayer(config.embed_dim, 
                                                  nhead=config.num_heads, 
                                                  dim_feedforward=config.mlp_dim,
                                                  batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=config.num_layers)

        self.classifier_head = nn.Linear(in_features=config.embed_dim, out_features=config.num_classes)

        #raise NotImplementedError("Build the patch embedder, positional embeddings, transformer blocks, and classifier head.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`.

        Step-by-step tensor path:
        1. Run patch embedding.
           - input: `[batch_size, 1, 28, 28]`
           - output: `[batch_size, num_patches, embed_dim]`
        2. Expand the class token across the batch.
           - shape should become `[batch_size, 1, embed_dim]`
        3. Concatenate the class token to the front of the patch tokens.
           - new shape: `[batch_size, num_patches + 1, embed_dim]`
        4. Add positional embeddings.
           - shape stays the same
        5. Run the transformer encoder blocks.
           - shape still stays `[batch_size, num_patches + 1, embed_dim]`
        6. Pick one summary vector.
           - simplest default: take the first token, which is the class token
           - summary shape: `[batch_size, embed_dim]`
        7. Run the classifier head.
           - final logits shape: `[batch_size, num_classes]`

        Important beginner point:
        - the transformer does not take the raw image directly
        - it takes the patch-token sequence
        """

        patch_tokens = self.patch_embed(x)                                  # output shape is (64, 49, 64)

        batch_size = x.shape[0] # batch size may not always be 64, depending on the last training step.
        expanded_class_tokens = self.class_token.expand([batch_size, *self.class_token.shape[1:]]) # output shape is (64, 1, 64)

        joined = torch.concat([expanded_class_tokens, patch_tokens], dim=1)  # output shape is (64, 50, 64)

        embeddings = joined + self.positional_embed                          # output shape is (64, 50, 64)

        # joined:
        # [token_content_0] [token_content_1] [token_content_2] ...

        # positional_embed:
        # [position_0]      [position_1]      [position_2]      ...

        # after adding:
        # [content+position] [content+position] [content+position] ...

        # Add positional embeddings so each token knows where it came from.
        # Right now, `joined` tells the model what each token contains,
        # but not where that token belongs in the image.
        # The transformer does not know position by itself, so we add
        # one learned position vector to each token in the sequence.
        # This means each token carries both:
        # - content information from the patch or class token
        # - position information about where it belongs
        # After this addition, the model can tell the difference between
        # a patch from the top-left and a patch from the bottom-right.

        features = self.transformer_encoder(embeddings)             # output shape is (64, 50, 64)
        summary_vector = features[:,0:1,:]                          # output shape is (64, 1, 64)
        summary_vector = summary_vector.squeeze(dim=1)               # output shape is (64, 64)

        output = self.classifier_head(summary_vector)               # output shape is (64, 10)

        return output


    #raise NotImplementedError("Return train, val, and test loaders for FashionMNIST.")


# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Beginner-friendly object list:
# - model: your `ToyVisionTransformer`
# - loss_fn: `nn.CrossEntropyLoss()`
# - optimizer: something simple like `torch.optim.Adam(...)`
def build_training_objects(config: VisionTransformerConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    model = ToyVisionTransformer(config)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())

    return model, loss_fn, optimizer

    #raise NotImplementedError("Return the toy ViT, CrossEntropyLoss, and optimizer.")


# Run one full training epoch.
# Return a dict with exactly these keys:
# - `loss`
# - `accuracy`
# Important shape reminder for this file:
# - the dataloader gives images shaped `[batch_size, 1, 28, 28]`
# - the model should return logits shaped `[batch_size, 10]`
# - `CrossEntropyLoss` expects:
#   - logits: `[batch_size, num_classes]`
#   - labels: `[batch_size]`
# Training loop order:
# - get one batch of `images, labels`
# - move both to the same device as the model
# - call `optimizer.zero_grad()`
# - call `model(images)` to get logits
# - call `loss_fn(logits, labels)`
# - call `loss.backward()`
# - call `optimizer.step()`
# - track total loss and number of correct predictions across the full epoch
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:

    model.train()

    losses = []
    accs = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = logits.argmax(dim=1).float()
        correct = (preds == targets).float()
        accs.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss":mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Return the mean train loss and mean train accuracy for one full epoch.")


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
# Evaluation loop is the same high-level flow as training, except:
# - do not call `optimizer.zero_grad()`
# - do not call `backward()`
# - do not call `optimizer.step()`
# - only run the model forward and collect metrics
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    model.eval()

    losses = []
    accs = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = loss_fn(logits, targets)
        losses.append(loss.item())

        preds = logits.argmax(dim=1).float()
        correct = (preds == targets).float()
        accs.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss": mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Return the mean loss and mean accuracy for the loader you evaluate.")


# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `train_accuracy`
# - `val_loss`
# - `val_accuracy`
# Beginner-friendly order for this function:
# - build loaders
# - build model, loss, and optimizer
# - loop over epochs
# - each epoch:
#   - call `train_one_epoch(...)`
#   - call `evaluate(...)` on the validation loader
# - return the final epoch summary
def run_training(config: VisionTransformerConfig) -> dict[str, float]:

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    train_dict = None
    val_dict = None

    for i in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i} -> loss: {train_dict["loss"]}, acc: {train_dict["accuracy"]}, val_loss: {val_dict["loss"]}, val_acc: {val_dict["accuracy"]}")

    return {"train_loss": train_dict["loss"], "train_accuracy": train_dict["accuracy"],
            "val_loss": val_dict["loss"], "val_accuracy": val_dict["accuracy"]}
    #raise NotImplementedError("Run the full training flow and return the final train and val metrics.")


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
    """Run the toy ViT scaffold one step at a time."""
    config = VisionTransformerConfig()
    set_seed(config.random_state)
    print("Running exercise 08 starter scaffold...")
    print("Main goal: turn images into patches, then classify with transformer blocks.")

    completed_steps = 0
    total_steps = 9

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_patch_math", describe_patch_math, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    patch_embedder, ok = report_task_progress(
        "PatchEmbedding.__init__",
        PatchEmbedding,
        config.image_channels,
        config.patch_size,
        config.embed_dim,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.image_channels, config.image_size, config.image_size)
    _, ok = report_task_progress("PatchEmbedding.forward", patch_embedder, sample_x)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("ToyVisionTransformer.__init__", ToyVisionTransformer, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("ToyVisionTransformer.forward", model, sample_x)
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
