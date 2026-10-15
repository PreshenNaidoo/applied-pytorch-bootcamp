"""
============================================================
EXERCISE 01: MULTIMODAL IMAGE TEXT LEARNING
============================================================

WHAT YOU ARE BUILDING
You are building a small multimodal model that looks at an image and a short text prompt together.

This scaffold uses one clear default task:
- image-text matching
- input: one image and one short text prompt
- output: one raw logit that says whether the image and text match

Plain-language meaning:
- `multimodal` means the model uses more than one kind of input
- here the two kinds are images and text
- a `logit` is just a raw score before turning it into a probability

Beginner intuition:
- the model gets two pieces of information:
  - a picture
  - a short sentence
- its job is to answer one simple question:
  - "do these two belong together?"
- so this is a matching task, not a caption-writing task

This is laptop-friendly and much simpler than a large production multimodal model.

WHY THIS EXERCISE EXISTS
Multimodal learning means a model uses more than one kind of input.
Here the two input types are:
- images
- text

This exercise teaches:
- how to build simple paired image-text data
- how to encode images and text separately
- how to combine two feature vectors into one prediction
- how to keep batching clear when there are two input branches

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises taught:
- image models
- text models
- transfer learning
- attention and small language models

This file brings two modalities together in one task.
It is the first clear step from single-input models toward multimodal systems.

DATASET
Use a small toy paired dataset built from `CIFAR10` plus templated text prompts.

Simple default path:
- use a CIFAR10 image
- make a positive prompt like `"this is a cat"`
- make a negative prompt by pairing the image with the wrong class text
- train a binary matching model

Use one shared `60% train / 20% val / 20% test` split.

HOW TO LOAD OR DOWNLOAD THE DATASET
Simple path:
- load `torchvision.datasets.CIFAR10`
- combine the original train and test splits
- make one shared `60 / 20 / 20` split
- create short text prompts from class names
- build positive and negative pairs

INPUTS AND OUTPUTS
Image-text batch:
- `images`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, 3, 32, 32]`
- `token_ids`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size, max_text_length]`
- `lengths`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`
- `labels`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size]`
  - values: `1.0` for match, `0.0` for mismatch

Example batch:
- `images`: `[32, 3, 32, 32]`
- `token_ids`: `[32, 8]`
- `lengths`: `[32]`
- `labels`: `[32]`

What those shapes mean:
- `images: [32, 3, 32, 32]` means 32 images, 3 color channels, height 32, width 32
- `token_ids: [32, 8]` means 32 text prompts and space for up to 8 token ids per prompt
- `lengths: [32]` means one real-text length per prompt
- `labels: [32]` means one match or mismatch label per image-text pair

Model output:
- type: `torch.Tensor`
- shape: `[batch_size]`
- meaning: one raw matching logit per image-text pair

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Default path:
- small image encoder
- small text encoder
- combine the two feature vectors
- matching head that returns one raw logit

A simple default combine step is:
- concatenate image features and text features
- feed the combined vector into a small MLP

More plain words:
- an `encoder` turns raw input into a smaller learned feature vector
- a `feature vector` is a list of numbers that tries to capture the useful parts of the input
- `concatenate` means put the image numbers and text numbers next to each other into one longer vector

Why this setup works:
- the image encoder tries to summarize what is in the picture
- the text encoder tries to summarize what the prompt says
- the fusion step puts both summaries together so the model can compare them
- the final head decides whether they look like a match

LOSS FUNCTION
Use `nn.BCEWithLogitsLoss`.

Important:
- return raw logits
- do not apply sigmoid before the loss

METRICS TO TRACK
Track at least:
- `train_loss`
  - type: `float`
  - meaning: mean training loss over the whole final training epoch
- `train_accuracy`
  - type: `float`
  - meaning: mean training accuracy over the whole final training epoch
- `val_loss`
  - type: `float`
  - meaning: mean validation loss over the whole validation loader
- `val_accuracy`
  - type: `float`
  - meaning: mean validation accuracy over the whole validation loader

If you pass the test loader to `evaluate(...)`:
- `loss` means mean test loss over the whole test loader
- `accuracy` means mean test accuracy over the whole test loader

TARGET RESULT RANGE
A small correct matching model on this toy paired dataset should usually beat chance by a clear margin.
A rough target is validation accuracy around `0.75` or better.

COMMON MISTAKES
- forgetting that labels for `BCEWithLogitsLoss` should be float values like `0.0` and `1.0`
- applying sigmoid before `BCEWithLogitsLoss`
- mixing up the image batch and the text batch
- building only positive pairs and forgetting negative pairs
- moving images to the GPU but leaving the token tensors on the CPU

STEP-BY-STEP TODO
1. Load CIFAR10 and make one shared `60 / 20 / 20` split.
2. Create short class-name prompts.
3. Build positive and negative image-text pairs.
4. Tokenize the short prompts.
5. Build an image encoder.
6. Build a text encoder.
7. Fuse the two feature vectors.
8. Return one raw matching logit.
9. Train for several epochs.
10. Evaluate on validation data.

What each step is doing in simple words:
1. get image data and make fair train, val, and test splits
2. turn class names into short text prompts
3. make matching and mismatching image-text pairs
4. turn prompts into token ids
5. build a small network that reads images
6. build a small network that reads text
7. join the two summaries
8. make one match or mismatch score
9. train the model on many paired examples
10. check whether it can judge new pairs correctly

STRETCH GOALS
- compare concatenation and cosine-similarity matching
- add more than one prompt template per class
- inspect a few false matches and false mismatches

DEFINITION OF DONE
You are done when:
- your batch contains both image tensors and text tensors
- your model returns one raw matching logit per pair
- you can explain what a positive pair and a negative pair are
- the validation accuracy is clearly above chance
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import check_threshold, print_pass_or_try_again
# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.


device = "cuda" if torch.cuda.is_available() else "cpu"


# Store settings for the image-text matching exercise.
# Return:
# - a config object with data, model, and training settings
@dataclass
class MultimodalConfig:
    """Settings for the toy image-text matching exercise."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    download: bool = False
    image_channels: int = 3
    image_height: int = 32
    image_width: int = 32
    image_encoder_hidden_channels: int = 32
    max_text_length: int = 8
    image_feature_dim: int = 128
    text_embedding_dim: int = 64
    text_feature_dim: int = 128
    fusion_hidden_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 6
    random_state: int = 42
    target_val_accuracy: float = 0.75



# Describe the paired image-text setup.
# Return:
# - split rule
# - input shapes
# - label meaning
# - output shape
def describe_pair_setup(config: MultimodalConfig) -> dict[str, object]:
    """Return the exact paired-data setup for this exercise.

    Return a dict with exactly these keys:
    - `default_split`
    - `input_shapes`
    - `label_meaning`
    - `output_shape`
    """
    return {
        "default_split": {"train": 0.60, "val": 0.20, "test": 0.20},
        "input_shapes": {
            "images": (
                config.batch_size,
                config.image_channels,
                config.image_height,
                config.image_width,
            ),
            "token_ids": (config.batch_size, config.max_text_length),
            "lengths": (config.batch_size,),
            "labels": (config.batch_size,),
        },
        "label_meaning": {0.0: "image and text do not match", 1.0: "image and text match"},
        "output_shape": (config.batch_size,),
    }


# Build the tiny prompt vocabulary.
# Return:
# - vocabulary dictionary
# - pad token id
def build_prompt_vocabulary(class_names: list[str]) -> tuple[dict[str, int], int]:
    """Return a tiny prompt vocabulary and the pad token id."""

    tokens = ["<pad>", "<unk>", "this", "is", "a", *class_names]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    pad_id = vocabulary["<pad>"]
    return vocabulary, pad_id


# Define paired CIFAR10 prompt examples.
# Expected behavior:
# - each image appears once with a matching prompt
# - each image appears once with a mismatching prompt
class CIFAR10PromptPairDataset(Dataset):
    """Wrap an image dataset and return image-text match pairs."""

    # Store the base dataset and prompt settings.
    # Expected stored values:
    # - CIFAR10 base dataset
    # - class names
    # - vocabulary
    # - pad id
    # - max text length
    def __init__(
        self,
        base_dataset: Dataset,
        class_names: list[str],
        vocabulary: dict[str, int],
        pad_id: int,
        max_text_length: int,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.class_names = class_names
        self.vocabulary = vocabulary
        self.pad_id = pad_id
        self.max_text_length = max_text_length
        self.vocab_size = len(vocabulary)

    # Return paired dataset length.
    # Return:
    # - two rows per base image, one positive and one negative
    def __len__(self) -> int:
        # One positive pair and one negative pair for each image.
        return len(self.base_dataset) * 2

    # Encode one prompt.
    # Input:
    # - prompt string such as `"this is a cat"`
    # Return:
    # - fixed-length token ids
    # - real prompt length
    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn one short prompt into fixed-length token ids and its real length."""

        words = prompt.lower().split()
        real_length = min(len(words), self.max_text_length)
        token_ids = [self.vocabulary.get(word, self.vocabulary["<unk>"]) for word in words[: self.max_text_length]]

        while len(token_ids) < self.max_text_length:
            token_ids.append(self.pad_id)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(real_length, dtype=torch.long),
        )

    # Return one image-text pair.
    # Return:
    # - image tensor
    # - token ids
    # - prompt length
    # - float label, where `1.0` means match and `0.0` means mismatch
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base_index = index // 2
        use_positive_prompt = index % 2 == 0

        image, class_id = self.base_dataset[base_index]

        if use_positive_prompt:
            prompt_class_id = class_id
            label = 1.0
        else:
            negative_choices = [i for i in range(len(self.class_names)) if i != class_id]
            prompt_class_id = negative_choices[torch.randint(0, len(negative_choices), (1,)).item()]
            label = 0.0

        prompt = f"this is a {self.class_names[prompt_class_id]}"
        token_ids, length = self.encode_prompt(prompt)

        return {
            "images": image,
            "token_ids": token_ids,
            "lengths": length,
            "labels": torch.tensor(label, dtype=torch.float32),
        }



# Build the multimodal dataloaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Return type:
# - `dict[str, DataLoader]`
# Each batch should be a dict with exactly these keys:
# - `images`
# - `token_ids`
# - `lengths`
# - `labels`
# Also make the dataset expose:
# - `vocab_size`
# - `pad_id`
# so later scaffolds can build the text encoder without guessing those values.
# Batch shape reminder:
# - `images`: `[batch_size, image_channels, image_height, image_width]`
# - `token_ids`: `[batch_size, max_text_length]`
# - `lengths`: `[batch_size]`
# - `labels`: `[batch_size]`
# Beginner-friendly data path:
# - load CIFAR10
# - make one shared 60/20/20 split
# - create a small dataset class that reads one image and its class name
# - for each image, create:
#   - one positive text prompt that matches the class
#   - one negative text prompt from a different class
# - tokenize the prompt into integer ids
# - return one dict with image, token ids, length, and label
# Why positive and negative pairs matter:
# - if you only show matching pairs, the model never learns what a mismatch looks like
# - the negative pairs teach it the difference between "goes together" and "does not go together"
def load_dataloaders(config: MultimodalConfig) -> dict[str, DataLoader]:
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    data_train = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=config.download,
        transform=transforms,
    )
    data_test = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=False,
        download=config.download,
        transform=transforms,
    )

    combined_dataset = torch.utils.data.ConcatDataset([data_train, data_test])

    total_examples = len(combined_dataset)
    train_size = int(total_examples * 0.60)
    val_size = int(total_examples * 0.20)
    test_size = total_examples - train_size - val_size

    split_generator = torch.Generator().manual_seed(config.random_state)
    train_base, val_base, test_base = torch.utils.data.random_split(
        dataset=combined_dataset,
        lengths=[train_size, val_size, test_size],
        generator=split_generator,
    )

    class_names = list(data_train.classes)
    vocabulary, pad_id = build_prompt_vocabulary(class_names)

    train_dataset = CIFAR10PromptPairDataset(
        base_dataset=train_base,
        class_names=class_names,
        vocabulary=vocabulary,
        pad_id=pad_id,
        max_text_length=config.max_text_length,
    )
    val_dataset = CIFAR10PromptPairDataset(
        base_dataset=val_base,
        class_names=class_names,
        vocabulary=vocabulary,
        pad_id=pad_id,
        max_text_length=config.max_text_length,
    )
    test_dataset = CIFAR10PromptPairDataset(
        base_dataset=test_base,
        class_names=class_names,
        vocabulary=vocabulary,
        pad_id=pad_id,
        max_text_length=config.max_text_length,
    )

    return {
        "train": DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
    }


# Define the image encoder.
# Expected behavior:
# - turn one image batch into one feature vector per image
class ImageEncoder(nn.Module):
    """Small image encoder stub."""

    # Build the image encoder layers.
    # Expected modules:
    # - small convolution or image feature extractor
    # - pooling or flattening
    # - linear layer to `image_feature_dim`
    def __init__(self, config: MultimodalConfig) -> None:
        super().__init__()
        raise NotImplementedError("Build a small image encoder that returns one feature vector per image.")
        # Beginner-friendly build idea:
        # - use a small CNN
        # - its job is to turn one image into one feature vector
        # - simple path:
        #   - conv
        #   - activation
        #   - pooling
        #   - flatten
        #   - linear layer to `output_dim`
        #
        # Use the image settings from `config` instead of hiding them inside the class.
        # That keeps this scaffold closer to earlier exercises where important sizes were explicit.
        #
        # Intuition:
        # - the image encoder is trying to compress a whole image into one useful summary vector
        # - that summary should keep the parts that matter for matching, like rough object type and shape



    # Encode images.
    # Input:
    # - `images`: `[batch_size, 3, 32, 32]`
    # Return:
    # - image features with shape `[batch_size, image_feature_dim]`
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return image features with shape `[batch_size, output_dim]`."""

        raise NotImplementedError("Encode the image batch into one feature vector per image.")



# Define the text encoder.
# Expected behavior:
# - turn one prompt batch into one feature vector per prompt
class TextEncoder(nn.Module):
    """Small text encoder stub."""

    # Build the text encoder layers.
    # Expected modules:
    # - embedding layer
    # - pooling path that ignores padding
    # - linear layer to `text_feature_dim`
    def __init__(self, vocab_size: int, pad_id: int, config: MultimodalConfig) -> None:
        super().__init__()
        raise NotImplementedError("Build the text encoder that returns one feature vector per prompt.")
        # Beginner-friendly build idea:
        # - use an embedding layer for token ids
        # - then avg pool across the real tokens
        # - then use a linear layer if you want to map to `text_feature_dim`
        #
        # Simple default path:
        # - embedding
        # - mean over real tokens only
        # - linear layer to final text feature size
        #
        # Intuition:
        # - the text encoder does for prompts what the image encoder does for images
        # - it turns several token vectors into one short summary of the prompt meaning



    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return text features with shape `[batch_size, text_feature_dim]`.

        Step-by-step tensor path:
        - token ids -> embeddings
        - embeddings -> one pooled text vector per example
        - pooled vector -> optional projection layer

        Why use `lengths` here?
        - prompts may have different real lengths
        - `lengths` tells you how many tokens are real and how many are just padding
        - you want to avg pool only over the real prompt tokens
        """

        # token_ids is shape (32, 8),  lengths is shape (32)

        raise NotImplementedError("Encode the prompt batch into one feature vector per prompt.")








# Define the image-text matching model.
# Expected behavior:
# - encode images
# - encode text prompts
# - combine both features
# - return one raw matching logit per pair
class ImageTextMatchingModel(nn.Module):
    """Toy multimodal matching model stub."""

    # Build the full matching model.
    # Expected modules:
    # - image encoder
    # - text encoder
    # - fusion MLP
    # - matching head
    def __init__(self, vocab_size: int, pad_id: int, config: MultimodalConfig) -> None:
        super().__init__()
        raise NotImplementedError("Build the image encoder, text encoder, fusion step, and matching head.")
        # Beginner-friendly build order:
        # 1. image encoder
        # 2. text encoder
        # 3. fusion step
        # 4. matching head
        #
        # Simple default fusion:
        # - first join the image feature vector and text feature vector into one longer vector
        # - then pass that combined vector through a small MLP
        # - the first linear layer mixes information from both branches
        # - the non-linear activation in the middle makes the model more flexible than one linear rule
        # - the final linear layer turns the fused representation into one matching logit
        #
        # Intuition:
        # - one branch says "this is what the image seems to show"
        # - the other branch says "this is what the text seems to mean"
        # - the matching head looks at both summaries together and decides whether they agree




    # Run the multimodal model.
    # Input:
    # - images
    # - token ids
    # - real prompt lengths
    # Return:
    # - raw matching logits with shape `[batch_size]`
    def forward(self, images: torch.Tensor, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return raw matching logits with shape `[batch_size]`.

        Step-by-step path:
        - image batch -> image encoder -> image features
        - token batch -> text encoder -> text features
        - combine the two feature tensors
        - run the matching head
        - squeeze only if needed so the final shape is `[batch_size]`
        """

        raise NotImplementedError("Return one raw matching logit per image-text pair.")






# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.BCEWithLogitsLoss`.
# In the real training flow, pass the `vocab_size` and `pad_id` you stored on the dataset.
def build_training_objects(
    vocab_size: int,
    pad_id: int,
    config: MultimodalConfig,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    raise NotImplementedError("Return the multimodal matching model, BCEWithLogitsLoss, and optimizer.")

    


# Run one full training epoch.
# Return a dict with exactly these keys:
# - `loss`
# - `accuracy`
# Return type:
# - `dict[str, float]`
# Both values must be Python floats.
# Metric meaning:
# - `loss` is the mean train loss over the whole training epoch
# - `accuracy` is the mean train accuracy over the whole training epoch
# Important:
# - move the model, `images`, `token_ids`, `lengths`, and `labels` to the same device here
# - keep `labels` as float values because you are using `BCEWithLogitsLoss`
# - for accuracy, turn logits into predicted match / mismatch decisions with a threshold after sigmoid or directly at logit 0
# Simple accuracy intuition:
# - a positive logit leans toward match
# - a negative logit leans toward mismatch
# - you can turn that into a yes/no prediction and compare it to the label
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
# Return type:
# - `dict[str, float]`
# Both values must be Python floats.
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
# Return type:
# - `dict[str, float]`
# This is the final epoch summary, not a history list.
# Simple flow:
# - call `load_dataloaders(config)`
# - read `vocab_size` and `pad_id` from the train dataset
# - call `build_training_objects(vocab_size, pad_id, config)`
# - train
# - evaluate on validation data
def run_training(config: MultimodalConfig) -> dict[str, float]:

    raise NotImplementedError("Run the full multimodal training flow and return the final train and val metrics.")









def check_metrics(metrics: dict[str, float], target: float) -> bool:
    """Return True when validation accuracy reaches the target."""
    return check_threshold(metrics.get("val_accuracy", 0.0), target)



# Run one scaffold task and catch unfinished TODOs.
# Return:
# - result object
# - boolean saying whether the task is done
def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
        print(f"[DONE] {task_name}")
        return result, True
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False



# Run the exercise scaffold one step at a time.
# Expected behavior:
# - stop at the first unfinished TODO
# - print progress so the student knows what to implement next
def main() -> None:
    """Run the multimodal scaffold one step at a time."""
    config = MultimodalConfig()
    set_seed(config.random_state)
    print("Running exercise 01 starter scaffold...")
    print("Main goal: build a small image-text matching model on toy paired data.")
    print(f"Device note: keep the model, images, token_ids, lengths, and labels on the same device. Using {device}.")

    completed_steps = 0
    total_steps = 12

    _, ok = report_task_progress("describe_pair_setup", describe_pair_setup, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    image_encoder, ok = report_task_progress("ImageEncoder.__init__", ImageEncoder, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_images = torch.zeros(
        config.batch_size,
        config.image_channels,
        config.image_height,
        config.image_width,
    )
    _, ok = report_task_progress("ImageEncoder.forward", image_encoder, sample_images)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    text_encoder, ok = report_task_progress("TextEncoder.__init__", TextEncoder, 64, 0, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_token_ids = torch.zeros(config.batch_size, config.max_text_length, dtype=torch.long)
    sample_lengths = torch.full((config.batch_size,), config.max_text_length, dtype=torch.long)
    _, ok = report_task_progress("TextEncoder.forward", text_encoder, sample_token_ids, sample_lengths)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("ImageTextMatchingModel.__init__", ImageTextMatchingModel, 64, 0, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("ImageTextMatchingModel.forward", model, sample_images, sample_token_ids, sample_lengths)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return    

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, 64, 0, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    training_model, loss_fn, optimizer = training_objects

    _, ok = report_task_progress("train_one_epoch", train_one_epoch, training_model, loaders["train"], loss_fn, optimizer)
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



