"""
============================================================
EXERCISE 14: HUGGINGFACE TEXT FINETUNING
============================================================

WHAT YOU ARE BUILDING
You are building a small text classifier with a pretrained Hugging Face model.

Plain-language meanings:
- a tokenizer turns text into token ids
- an attention mask says which tokens are real and which tokens are padding
- a pretrained model has already learned useful language patterns
- fine-tuning means training that model a bit more for your task

WHY THIS EXERCISE EXISTS
Modern text work usually starts from a pretrained model.
Research engineer roles often need you to load, adapt, train, and evaluate these models.
This exercise teaches:
- how to use a pretrained tokenizer
- how to prepare tokenized batches
- how to fine-tune a pretrained classifier
- how to compare it with a simpler text baseline

HOW THIS FITS INTO THE BOOTCAMP
Earlier text exercises taught tokenization and smaller models from scratch.
This file adds a common real-world path:
- start from a pretrained model
- adapt it to your task
- compare it fairly with your earlier baseline

What each step is doing in simple words:
1. load one shared text task
2. use the tokenizer the pretrained model expects
3. build batches with the right keys and shapes
4. fine-tune the pretrained classifier
5. compare it with a simpler earlier approach

Beginner intuition:
- a pretrained model has already spent a lot of time learning language patterns
- your task is not to teach language from scratch
- your task is to adapt that model to your own labels

Small example:
- if the pretrained model already knows something about words like `market`, `stock`, and `company`
- then adapting it to a news classification task is easier than starting from random weights

DATASET
Use AG News for a 4-class text classification task.

Clear default:
- Hugging Face dataset: `ag_news`
- text field: `text`
- label field: `label`
- labels: 4 news topics

Use the official AG News split:
- split the official training data into train and validation
- keep the official test data for final evaluation

HOW TO LOAD OR DOWNLOAD THE DATASET

Simple path:
- load AG News with `datasets`
- split the official training data into train and validation
- keep the official test split unchanged for final evaluation
- tokenize the text with the pretrained tokenizer
- use the same tokenizer for every split

MODEL
Use a small model by default.

Good beginner-friendly choices:
- `distilbert-base-uncased`
- another small sequence-classification model

INPUTS AND OUTPUTS

Batch fields:
- `input_ids`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size, max_length]`
- `attention_mask`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size, max_length]`
- `labels`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`

Model output:
- raw class logits with shape `[batch_size, num_classes]`
What those tensors mean in plain words:
- `input_ids` are the token numbers the model reads
- `attention_mask` tells the model which positions are real text and which are only padding
- `labels` are the correct class ids

LOSS FUNCTION
Use the model's built-in classification loss or `nn.CrossEntropyLoss`.

METRICS TO TRACK
Track at least:
- train loss
- validation loss
- validation accuracy
- test accuracy

TARGET RESULT RANGE
A small pretrained model should usually beat a simple from-scratch baseline on this task.

Useful signs:
- tokenization runs without shape errors
- validation accuracy improves after training
- the pretrained model reaches a clearly usable score on AG News

MODEL ARCHITECTURE (HIGH LEVEL ONLY)

Default path:
- tokenizer
- pretrained text encoder
- classification head

Simple mental picture:
- the pretrained model already knows a lot about language
- your job is to connect that language knowledge to your label set

COMMON MISTAKES
- forgetting `attention_mask`
- passing text strings directly into the model instead of token ids
- applying softmax before the loss
- using labels with the wrong dtype
- comparing with a baseline that used a different split

STEP-BY-STEP TODO
1. Load AG News.
2. Split the official training data into train and validation.
3. Load a pretrained tokenizer.
4. Tokenize train, val, and test text.
5. Build dataloaders that return `input_ids`, `attention_mask`, and `labels`.
6. Load a pretrained sequence-classification model.
7. Fine-tune for a small number of epochs.
8. Evaluate on validation and test data.
9. Write a short comparison against your simple text classifier.

DEFINITION OF DONE
You are done when:
- tokenized batches contain `input_ids` and `attention_mask`
- the pretrained model returns logits with shape `[batch_size, num_classes]`
- validation accuracy is reported
- you can explain why pretrained models often help
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TYPE_CHECKING, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42) -> None:
    """Set the main random seeds so repeated runs are closer to each other."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a short result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")
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

@dataclass
class HuggingFaceFineTuningConfig:
    """Settings for pretrained text fine-tuning."""
    dataset_name: str = "ag_news"
    text_column: str = "text"
    label_column: str = "label"
    model_name: str = "distilbert-base-uncased"
    num_classes: int = 4
    max_length: int = 96
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 2
    random_state: int = 42
    target_val_accuracy: float = 0.80

# Beginner hints for this file:
# - the tokenizer does not "understand" the task; it just turns text into token ids and masks
# - `attention_mask` tells the model which token positions are real and which are padding
# - pretrained means the model already learned language patterns before your task
# - fine-tuning means adjusting that pretrained model on your own labeled data
# - compare this to your earlier simple text classifier so you can feel what pretraining adds
def describe_hf_workflow(config: HuggingFaceFineTuningConfig) -> dict[str, Any]:
    """Return the main pretrained fine-tuning steps."""
    return {
        "dataset_name": config.dataset_name,
        "model_name": config.model_name,
        "batch_keys": ["input_ids", "attention_mask", "labels"],
        "num_classes": config.num_classes,
    }

# Load the raw dataset splits.
# Return:
# - `train`
# - `val`
# - `test`
# Keep the official test split for final evaluation.
def load_text_splits(config: HuggingFaceFineTuningConfig) -> dict[str, object]:
    """Load AG News and return train, validation, and official test splits."""
    # Step-by-step path:
    #
    # 1. Import the Hugging Face `datasets` library inside this function.
    #    This keeps the import close to the code that needs it.
    #
    # 2. Load the dataset named in `config.dataset_name`.
    #    For the default exercise, this is AG News.
    #
    # 3. Use the official AG News split.
    #    `raw_data["train"]` is the official training data.
    #    `raw_data["test"]` is the official test data.
    #
    # 4. Keep `raw_data["test"]` unchanged.
    #    This is the final test set.
    #    Do not use it for training or choosing settings.
    #
    # 5. Use `.train_test_split(...)` on the official training dataset.
    #    It is not a top-level function called `datasets.train_test_split`.
    #
    #    Example shape:
    #    - `train_data.train_test_split(...)`
    #
    #    Meaning:
    #    - `train_data` is a Hugging Face `Dataset`
    #    - `.train_test_split(...)` is a method attached to that dataset
    #
    # 6. Make a train/validation split from the official training data.
    #    A common small validation split is:
    #    - 80% of the official training data for training
    #    - 20% of the official training data for validation
    #
    # 7. Use `config.random_state` as the seed so the split is repeatable.
    #
    # 8. Return a dictionary with exactly these keys:
    #    - `train`
    #    - `val`
    #    - `test`
    # Keep the official test split for final evaluation.
    # That makes later comparisons with your simpler text model much more honest
    # if both models use the same train, validation, and test data.
    # One clean implementation plan:
    # - import `datasets` inside this function
    # - call `datasets.load_dataset(config.dataset_name)`
    # - load the official train split with `split="train"`
    # - load the official test split with `split="test"`
    # - split the official train split into train and validation with `.train_test_split(...)`
    # - keep the official test split unchanged
    # - return a dict with exactly:
    #   - `train`
    #   - `val`
    #   - `test`

    import datasets as hf_datasets

    raw_data = hf_datasets.load_dataset(config.dataset_name)
    train_data = raw_data["train"]
    test_data = raw_data["test"]

    split_dict = train_data.train_test_split(
        test_size=0.2,
        shuffle=True,
        seed=config.random_state,
    )
    train_data = split_dict["train"]
    val_data = split_dict["test"]

    return {"train": train_data, "val": val_data, "test": test_data}
    #raise NotImplementedError("Load the dataset and return train, val, and test splits.")

# Load the pretrained tokenizer.
# In plain words:
# - this turns text into token ids
# - it also creates masks so the model knows which tokens are real text
def load_tokenizer(config: HuggingFaceFineTuningConfig) -> PreTrainedTokenizerBase:
    """Load the Hugging Face tokenizer.
    Beginner-friendly hint:
    - import `AutoTokenizer` inside this function
    - call `AutoTokenizer.from_pretrained(config.model_name)`
    """
    # Step-by-step path:
    #
    # 1. Import `AutoTokenizer` from `transformers` inside this function.
    #
    # 2. Load the tokenizer using `config.model_name`.
    #    The tokenizer must match the pretrained model.
    #    This does not load the neural network model.
    #    It loads the text-to-token-ids tool that belongs with that model name.
    #
    #    Simple difference:
    #    - `AutoTokenizer.from_pretrained(...)` loads the tokenizer
    #    - `AutoModelForSequenceClassification.from_pretrained(...)` loads the model
    #
    # 3. Return the tokenizer object.
    #
    # Plain check:
    # - for `distilbert-base-uncased`, the tokenizer lowercases text
    # - it also creates token ids in the format DistilBERT expects

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_name)

    return tokenizer
    #raise NotImplementedError("Load and return a pretrained tokenizer.")

# Tokenize each split.
# Return tokenized train, validation, and test data.
# The important fields are usually:
# - `input_ids`
# - `attention_mask`
# - `labels`
def tokenize_splits(
    text_splits: dict[str, object],
    tokenizer: PreTrainedTokenizerBase,
    config: HuggingFaceFineTuningConfig,
) -> dict[str, object]:
    """Tokenize each split and keep labels."""
    # Goal:
    # Turn each raw text example into the fields the pretrained model expects.
    #
    # Raw AG News example:
    # - `text`: one news article string
    # - `label`: one class id, such as 0, 1, 2, or 3
    #
    # Model-ready example:
    # - `input_ids`: token ids made by the tokenizer
    # - `attention_mask`: 1 for real tokens, 0 for padding
    # - `labels`: the correct class id
    #
    # Why there is a helper function:
    # Hugging Face datasets use `.map(...)` to apply one function to every example
    # or every small batch of examples.
    #
    # Here you should use:
    # - `.map(tokenize_batch, batched=True)`
    #
    # `batched=True` means the helper receives many examples at once.
    #
    # In this exercise, the helper receives a batch shaped like:
    # - `batch["text"]`: list of text strings
    # - `batch["label"]`: list of class ids
    #
    # Small example:
    # - `batch["text"] = ["Stocks rose today", "The team won"]`
    # - `batch["label"] = [2, 1]`
    #
    # The helper should return a dictionary shaped like:
    # - `input_ids`: list of token id lists
    # - `attention_mask`: list of mask lists
    # - `labels`: list of class ids
    #
    # Implementation steps:
    # 1. In `tokenize_batch`, read the list of texts from `batch[config.text_column]`.
    # 2. Pass that list of texts to `tokenizer(...)`.
    # 3. Use `padding="max_length"` so all examples become length `config.max_length`.
    # 4. Use `truncation=True` so long text is cut to `config.max_length`.
    # 5. Add `labels` to the tokenizer output using `batch[config.label_column]`.
    # 6. Return the tokenizer output.
    # 7. Outside the helper, call `.map(tokenize_batch, batched=True)` on train, val, and test.
    # 8. Return `{"train": tokenized_train, "val": tokenized_val, "test": tokenized_test}`.
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        """Tokenize one batch and add labels."""
        # This helper is called by `.map(...)`.
        #
        # Input:
        # - `batch[config.text_column]`: list of text strings
        # - `batch[config.label_column]`: list of class ids
        #
        # Do these steps here:
        # 1. Tokenize the list of text strings.
        #    The tokenizer should create both `input_ids` and `attention_mask`.
        #    You do not build the attention mask by hand here.
        #    Important:
        #    - `tokenizer.encode(text)` returns only one list of token ids.
        #    - `tokenizer(texts, padding=..., truncation=..., max_length=...)`
        #      returns a dictionary like:
        #      {"input_ids": [...], "attention_mask": [...]}
        # 2. Store the class ids under the key `labels`.
        # 3. Return the tokenized batch.
        #
        # Return:
        # - dictionary with `input_ids`, `attention_mask`, and `labels`

        texts = batch[config.text_column]
        labels = batch[config.label_column]

        token_dict = tokenizer(texts, padding="max_length", truncation=True, max_length=config.max_length)        
        input_ids = token_dict["input_ids"]
        attention_mask = token_dict["attention_mask"]
            
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        #raise NotImplementedError("Tokenize one batch and add labels.")

    # After `tokenize_batch` works, use it here:
    # - map it over `text_splits["train"]`
    # - map it over `text_splits["val"]`
    # - map it over `text_splits["test"]`
    #
    # Each `.map(...)` call should produce a tokenized Hugging Face dataset.

    tokenized_train = text_splits["train"].map(tokenize_batch, batched=True)

    tokenized_val = text_splits["val"].map(tokenize_batch, batched=True)

    tokenized_test = text_splits["test"].map(tokenize_batch, batched=True)

    return {"train": tokenized_train, "val": tokenized_val, "test": tokenized_test}
    #raise NotImplementedError("Return tokenized train, val, and test data.")

# Build dataloaders from the tokenized splits.
# Return:
# - `train`
# - `val`
# - `test`
# Each batch should be ready to pass into the Hugging Face model.
def build_dataloaders(tokenized_splits: dict[str, object], config: HuggingFaceFineTuningConfig) -> dict[str, DataLoader]:
    """Return dataloaders with input ids, attention masks, and labels."""
    # Goal:
    # Turn tokenized Hugging Face datasets into PyTorch DataLoaders.
    #
    # Input:
    # - `tokenized_splits["train"]`: tokenized training dataset
    # - `tokenized_splits["val"]`: tokenized validation dataset
    # - `tokenized_splits["test"]`: tokenized test dataset
    #
    # Each dataset should already contain:
    # - `input_ids`
    # - `attention_mask`
    # - `labels`
    #
    # Important Hugging Face step:
    # Before using `DataLoader`, tell each dataset to return PyTorch tensors.
    #
    # Use the dataset method `.set_format(...)` like this:
    # - `split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])`
    #
    # Meaning:
    # - `split` is one Hugging Face dataset, such as `tokenized_splits["train"]`
    # - `type="torch"` tells Hugging Face to return PyTorch tensors
    # - `columns=[...]` tells Hugging Face which fields to keep as tensors
    #
    # You need to do this for train, validation, and test splits.
    #
    # Keep only these columns:
    # - `input_ids`
    # - `attention_mask`
    # - `labels`
    #
    # Why:
    # The model does not need the raw text string during training.
    # It needs token ids, attention masks, and labels.
    #
    # DataLoader rules:
    # - train loader: `shuffle=True`
    # - validation loader: `shuffle=False`
    # - test loader: `shuffle=False`
    #
    # Use `batch_size=config.batch_size` for all three loaders.
    #
    # One batch should look like:
    # - `batch["input_ids"]`: shape `[batch_size, max_length]`
    # - `batch["attention_mask"]`: shape `[batch_size, max_length]`
    # - `batch["labels"]`: shape `[batch_size]`
    #
    # Return:
    # - dictionary with `train`, `val`, and `test` DataLoaders

    train_ds = tokenized_splits["train"]
    val_ds = tokenized_splits["val"]
    test_ds = tokenized_splits["test"]

    columns = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)
    test_ds.set_format(type="torch", columns=columns)

    return {
        "train": torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=config.batch_size,
            shuffle=True,
        ),
        "val": torch.utils.data.DataLoader(
            dataset=val_ds,
            batch_size=config.batch_size,
            shuffle=False,
        ),
        "test": torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=config.batch_size,
            shuffle=False,
        ),
    }
    #raise NotImplementedError("Return train, val, and test dataloaders for tokenized text.")

def build_training_objects(
    config: HuggingFaceFineTuningConfig,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Return pretrained model and optimizer.
    Beginner-friendly hint:
    - import `AutoModelForSequenceClassification` inside this function
    - set `num_labels=config.num_classes`
    """
    # Goal:
    # Load the pretrained neural network and make the optimizer.
    #
    # This function loads the model, not the tokenizer.
    #
    # Tokenizer:
    # - turns text into `input_ids` and `attention_mask`
    #
    # Model:
    # - reads `input_ids` and `attention_mask`
    # - returns class logits
    # - can return loss when `labels` are passed in
    #
    # Use:
    # - `AutoModelForSequenceClassification`
    #
    # Why this model class:
    # It adds a classification head on top of the pretrained text encoder.
    #
    # Set:
    # - `num_labels=config.num_classes`
    #
    # For AG News:
    # - `config.num_classes = 4`
    # - output logits should have shape `[batch_size, 4]`
    #
    # After loading:
    # - move the model to `device`
    # - create an `AdamW` optimizer
    # - use `config.learning_rate`
    #
    # Return:
    # - model
    # - optimizer

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_classes)

    # A Hugging Face model is still a PyTorch `nn.Module`.
    # That is why normal PyTorch methods like `.to(device)`,
    # `.train()`, `.eval()`, and `.parameters()` work here.
    model = model.to(device)    

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # We do not return a loss function here.
    # Hugging Face classification models make the cross entropy loss for us
    # when `labels` are passed into the model during training.
    return model, optimizer

    #raise NotImplementedError("Return a pretrained sequence classifier and optimizer.")

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run one fine-tuning epoch."""
    # Goal:
    # Run one full pass over the training DataLoader and update the model.
    #
    # One batch should contain:
    # - `input_ids`: token ids, shape `[batch_size, max_length]`
    # - `attention_mask`: real-token mask, shape `[batch_size, max_length]`
    # - `labels`: correct class ids, shape `[batch_size]`
    #
    # Training mode:
    # Call `model.train()` before the loop.
    # This turns on training behavior such as dropout.
    #
    # For each batch:
    # 1. Move `input_ids`, `attention_mask`, and `labels` to `device`.
    # 2. Clear old gradients with the optimizer.
    # 3. Call the model with `input_ids`, `attention_mask`, and `labels`.
    # 4. Read `loss` from the model output.
    # 5. Read `logits` from the model output.
    # 6. Backpropagate the loss.
    # 7. Take one optimizer step.
    #
    # Why pass `labels` to the model:
    # Hugging Face sequence-classification models can compute cross entropy loss
    # for you when labels are provided.
    #
    # Accuracy:
    # - logits shape should be `[batch_size, num_classes]`
    # - use `argmax` over the class dimension to get predicted labels
    # - compare predicted labels with `labels`
    #
    # Track:
    # - average loss over batches
    # - average accuracy over batches
    #
    # Return:
    # - `{"loss": mean_loss, "accuracy": mean_accuracy}`

    model.train()

    losses, accuracies = [], []

    for data_dict in loader:
        input_ids = data_dict["input_ids"]              # [batch_size, max_length]
        labels = data_dict["labels"]                    # [batch_size]

        # This is a padding mask, not a causal mask.
        # Shape: [batch_size, max_length].
        # 1 means "real token"; 0 means "padding token".
        # Example with batch_size=2 and max_length=6:
        # [[1, 1, 1, 1, 0, 0],   # 4 real tokens, then 2 padding tokens
        #  [1, 1, 1, 1, 1, 1]]   # 6 real tokens, no padding
        # The model uses this during attention so real tokens ignore padding tokens.
        # Without it, the model may treat padding as if it were real text.
        # Hugging Face expands it inside the model to the larger attention shape it needs.
        # A causal mask is different: it blocks a token from looking at future tokens.
        attention_masks = data_dict["attention_mask"]   # [batch_size, max_length]

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_masks = attention_masks.to(device)

        optimizer.zero_grad()

        model_output = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        loss = model_output["loss"]
        logits = model_output["logits"]
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pred = logits.argmax(dim=-1)
        correct = (pred==labels).float()
        accuracies.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accuracies).mean().item()

    return {"loss": mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Return mean train loss and accuracy.")

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """Run one full evaluation pass."""
    # Goal:
    # Measure loss and accuracy without changing the model.
    #
    # Evaluation mode:
    # Call `model.eval()` before the loop.
    # This turns off training behavior such as dropout.
    #
    # No gradients:
    # Use `torch.no_grad()` around the loop.
    # Evaluation should not call `backward()` or `optimizer.step()`.
    #
    # For each batch:
    # 1. Move `input_ids`, `attention_mask`, and `labels` to `device`.
    # 2. Call the model with `input_ids`, `attention_mask`, and `labels`.
    # 3. Read `loss` and `logits` from the model output.
    # 4. Use `argmax` on logits to get predicted labels.
    # 5. Compare predicted labels with `labels`.
    #
    # Track:
    # - average loss over batches
    # - average accuracy over batches
    #
    # Return:
    # - `{"loss": mean_loss, "accuracy": mean_accuracy}`

    model.eval()
    losses, accuracies = [], []

    for data_dict in loader:
        input_ids = data_dict["input_ids"]              # [batch_size, max_length]
        labels = data_dict["labels"]                    # [batch_size]

         # This is a padding mask, not a causal mask.
        # Shape: [batch_size, max_length].
        # 1 means "real token"; 0 means "padding token".
        # Example with batch_size=2 and max_length=6:
        # [[1, 1, 1, 1, 0, 0],   # 4 real tokens, then 2 padding tokens
        #  [1, 1, 1, 1, 1, 1]]   # 6 real tokens, no padding
        # The model uses this during attention so real tokens ignore padding tokens.
        # Without it, the model may treat padding as if it were real text.
        # Hugging Face expands it inside the model to the larger attention shape it needs.
        # A causal mask is different: it blocks a token from looking at future tokens.
        attention_masks = data_dict["attention_mask"]   # [batch_size, max_length]

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_masks = attention_masks.to(device)        

        model_output = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        loss = model_output["loss"]
        logits = model_output["logits"]        

        losses.append(loss.item())

        pred = logits.argmax(dim=-1)
        correct = (pred==labels).float()
        accuracies.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accuracies).mean().item()

    return {"loss": mean_loss, "accuracy": mean_acc}
   # raise NotImplementedError("Return mean loss and accuracy for the loader.")

def compare_with_simple_baseline(pretrained_metrics: dict[str, float], simple_baseline_accuracy: float | None = None) -> str:
    """Write a short note comparing pretrained fine-tuning with a simple baseline."""
    # Goal:
    # Return a short plain-language note about what pretraining changed.
    #
    # Input:
    # - `pretrained_metrics`: metrics from this exercise
    # - `simple_baseline_accuracy`: optional accuracy from an earlier simple text model
    #
    # If a baseline accuracy is provided:
    # - compare it with the pretrained model's accuracy
    # - say which one is higher
    # - keep the wording short
    #
    # If no baseline accuracy is provided:
    # - write a general note
    # - say that a pretrained model starts with useful language knowledge
    # - say that fine-tuning adapts that knowledge to the AG News labels
    #
    # Fair comparison rule:
    # Compare models on the same train, validation, and test data when possible.
    #
    # Return:
    # - one string
    pretrained_accuracy = float(
        pretrained_metrics.get(
            "test_accuracy",
            pretrained_metrics.get("val_accuracy", 0.0),
        )
    )

    if simple_baseline_accuracy is None:
        note = (
            "The pretrained model starts with useful language knowledge. "
            "Fine-tuning adapts that knowledge to the AG News labels."
        )
        print(note)
        return note

    simple_baseline_accuracy = float(simple_baseline_accuracy)
    if pretrained_accuracy > simple_baseline_accuracy:
        note = (
            f"The pretrained model did better: {pretrained_accuracy:.3f} "
            f"vs the simple baseline at {simple_baseline_accuracy:.3f}."
        )
        print(note)
        return note
    if pretrained_accuracy < simple_baseline_accuracy:
        note = (
            f"The simple baseline did better: {simple_baseline_accuracy:.3f} "
            f"vs the pretrained model at {pretrained_accuracy:.3f}."
        )
        print(note)
        return note
    note = f"Both models got the same accuracy: {pretrained_accuracy:.3f}."
    print(note)
    return note

def run_training(config: HuggingFaceFineTuningConfig) -> dict[str, float | str]:
    """Run the full Hugging Face fine-tuning workflow."""
    # Goal:
    # Connect all helper functions into one full training workflow.
    #
    # Build stage:
    # 1. Load the tokenizer.
    # 2. Load train, validation, and test text splits.
    # 3. Tokenize the three splits.
    # 4. Build the three DataLoaders.
    # 5. Load the pretrained classifier and optimizer.
    #
    # Training stage:
    # 1. Loop for `config.epochs`.
    # 2. Call `train_one_epoch(...)` on the train loader.
    # 3. Call `evaluate(...)` on the validation loader.
    # 4. Print or store the train and validation metrics each epoch.
    #
    # Final evaluation stage:
    # 1. After training, call `evaluate(...)` on the test loader once.
    # 2. Build a short comparison note with `compare_with_simple_baseline(...)`.
    #
    # Return:
    # - `train_loss`: latest training loss
    # - `train_accuracy`: latest training accuracy
    # - `val_loss`: latest validation loss
    # - `val_accuracy`: latest validation accuracy
    # - `test_loss`: final test loss
    # - `test_accuracy`: final test accuracy
    # - `comparison_note`: short string from the comparison helper

    raw_splits = load_text_splits(config)
    tokenizer = load_tokenizer(config)
    tokenized_splits = tokenize_splits(text_splits=raw_splits, tokenizer=tokenizer, config=config)
    loaders = build_dataloaders(tokenized_splits, config)
    model, optimizer = build_training_objects(config)    

    for epoch in range(config.epochs):
        train_dict = train_one_epoch(model, loaders["train"], optimizer)
        val_dict = evaluate(model, loaders["val"])

        print(
            f"Epoch {epoch + 1} -> "
            f"train_loss: {train_dict['loss']:.4f}, "
            f"train_accuracy: {train_dict['accuracy']:.4f}, "
            f"val_loss: {val_dict['loss']:.4f}, "
            f"val__accuracy: {val_dict['accuracy']:.4f}"
        )

    test_dict = evaluate(model, loaders["test"])
    comparison_note = compare_with_simple_baseline(
        {
            "val_accuracy": val_dict["accuracy"],
            "test_accuracy": test_dict["accuracy"],
        }
    )

    return {
        "train_loss": train_dict["loss"],
        "train_accuracy": train_dict["accuracy"],
        "val_loss": val_dict["loss"],
        "val_accuracy": val_dict["accuracy"],
        "test_loss": test_dict["loss"],
        "test_accuracy": test_dict["accuracy"],
        "comparison_note": comparison_note,
    }

    #raise NotImplementedError("Fine-tune the pretrained text model and return final metrics.")

def check_metrics(metrics: dict[str, float | str], target: float) -> bool:
    """Return True when validation accuracy reaches the rough target."""
    return "val_accuracy" in metrics and "comparison_note" in metrics and float(metrics["val_accuracy"]) >= target

def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    print(f"[RUNNING] {task_name}")
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    except Exception as error:
        print(f"[ERROR] {task_name}: {type(error).__name__}: {error}")
        raise
    except BaseException as error:
        print(f"[STOPPED] {task_name}: {type(error).__name__}: {error}")
        raise
    print(f"[DONE] {task_name}")
    return result, True

def main() -> None:
    """Run the Hugging Face fine-tuning scaffold one step at a time."""
    config = HuggingFaceFineTuningConfig()
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
    tokenizer, ok = report_task_progress("load_tokenizer", load_tokenizer, config)
    if not count_or_stop(ok):
        return
    text_splits, ok = report_task_progress("load_text_splits", load_text_splits, config)
    if not count_or_stop(ok):
        return
    tokenized_splits, ok = report_task_progress("tokenize_splits", tokenize_splits, text_splits, tokenizer, config)
    if not count_or_stop(ok):
        return
    loaders, ok = report_task_progress("build_dataloaders", build_dataloaders, tokenized_splits, config)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("describe_hf_workflow", describe_hf_workflow, config)
    if not count_or_stop(ok):
        return
    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return
    model, optimizer = training_objects
    _, ok = report_task_progress("train_one_epoch", train_one_epoch, model, loaders["train"], optimizer)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("evaluate", evaluate, model, loaders["val"])
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress(
        "compare_with_simple_baseline",
        compare_with_simple_baseline,
        {"val_accuracy": 0.0, "test_accuracy": 0.0},
        None,
    )
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics, config.target_val_accuracy)
        print_pass_or_try_again(passed, "val_accuracy", float(metrics.get("val_accuracy", 0.0)), f">= {config.target_val_accuracy}")
if __name__ == "__main__":
    main()
