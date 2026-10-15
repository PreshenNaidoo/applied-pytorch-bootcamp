"""
============================================================
EXERCISE 15: LORA PEFT FINETUNING
============================================================

WHAT YOU ARE BUILDING
You are building a small LoRA fine-tuning run.

Plain-language meanings:
- PEFT means parameter-efficient fine-tuning
- parameter-efficient means you train only a small part of the model
- LoRA adds small trainable adapter weights to a pretrained model
- most original model weights stay frozen

WHY THIS EXERCISE EXISTS
Large models are expensive to fully fine-tune.
LoRA is a common way to adapt a model with far fewer trainable weights.
This exercise teaches:
- how to freeze a pretrained model
- how to add LoRA adapters
- how to count total and trainable parameters
- how to compare LoRA with full fine-tuning or head-only training

HOW THIS FITS INTO THE BOOTCAMP
This file comes naturally after pretrained fine-tuning.
First you learn to fine-tune the whole or most of a pretrained model.
Then you learn a cheaper adaptation method that updates only a small part.

Beginner intuition:
- a very large model can be expensive to fully update
- LoRA says:
  - keep the big base model mostly fixed
  - add a small trainable path
  - learn mainly through that small path

Small example:
- full fine-tuning might update millions of weights
- LoRA might update only a much smaller set of adapter weights
- this can save memory and training cost

DATASET
Use AG News for the default LoRA text-classification comparison.

Clear default:
- Hugging Face dataset: `ag_news`
- text field: `text`
- label field: `label`
- labels: 4 news topics
- use the same small split for the baseline and the LoRA model

HOW TO LOAD OR DOWNLOAD THE DATASET

Simple path:
- load `ag_news`
- tokenize it with the same tokenizer as the pretrained model
- keep one shared split for every comparison
- compare LoRA and the baseline on the same data

MODEL
Use a small pretrained text classifier by default.

Good beginner-friendly choice:
- `distilbert-base-uncased`

INPUTS AND OUTPUTS

Batch fields:
- `input_ids`
- `attention_mask`
- `labels`

Key parameter counts:
- total parameters
- trainable parameters
- trainable percent

MODEL ARCHITECTURE (HIGH LEVEL ONLY)

Default path:
- pretrained base model
- LoRA adapters added to chosen linear layers
- small classification head

Simple mental picture:
- the big model stays mostly frozen
- the small LoRA path learns the task-specific adjustment

METRICS TO TRACK
Track at least:
- total parameters
- trainable parameters
- trainable parameter percent
- validation accuracy
- comparison note

LOSS FUNCTION
Use the model's built-in classification loss or `nn.CrossEntropyLoss`.

TARGET RESULT RANGE
Focus on the parameter tradeoff, not only the final score.

Useful signs:
- trainable parameter percent is much smaller than 100%
- validation accuracy is still reasonable
- you can explain the memory and cost tradeoff clearly

COMMON MISTAKES
- thinking LoRA trains the whole model
- forgetting to check which parameters require gradients
- comparing LoRA and full fine-tuning with different splits
- using a target module name that does not exist in the model

What each step is doing in simple words:
1. load one pretrained model to adapt
2. freeze most of the large base model
3. add the small trainable LoRA path
4. count how much of the model is really being updated
5. fine-tune on one small task
6. compare task quality with the much smaller trainable footprint
7. explain the tradeoff between cost and performance

STEP-BY-STEP TODO
1. Load a small pretrained model.
2. Freeze the base model or let PEFT handle trainable adapters.
3. Add LoRA adapters.
4. Count total parameters and trainable parameters.
5. Fine-tune on a small classification task.
6. Evaluate validation accuracy.
7. Compare LoRA with head-only or full fine-tuning.
8. Write a short note about the tradeoff.

DEFINITION OF DONE
You are done when:
- trainable parameter count is much smaller than total parameter count
- LoRA model trains without shape errors
- validation accuracy is reported
- you can explain why LoRA is useful for large models
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

@dataclass
class LoRAFineTuningConfig:
    """Settings for LoRA fine-tuning."""
    dataset_name: str = "ag_news"
    text_column: str = "text"
    label_column: str = "label"

    model_name: str = "distilbert-base-uncased"
    num_classes: int = 4
    max_length: int = 96
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 2
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    random_state: int = 42
    target_val_accuracy: float = 0.75

# Beginner hints for this file:
# - LoRA keeps most base-model weights frozen and trains only small adapter weights
# - this is useful because full fine-tuning can be too expensive for large models
# - always count total parameters and trainable parameters so the saving is concrete
# - if trainable parameter count barely changes, the adapters may not be attached correctly
# - think of LoRA as "small task-specific updates on top of a big frozen model"
def describe_lora_plan(config: LoRAFineTuningConfig) -> dict[str, object]:
    """Return the main LoRA settings."""
    return {
        "model_name": config.model_name,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "main_check": "trainable parameters should be much smaller than total parameters",
    }


# Build the tokenized dataloaders for the LoRA run.
# Return:
# - `train`
# - `val`
# - `test`
# Use the same split for every comparison with LoRA.
def load_tokenized_dataloaders(config: LoRAFineTuningConfig) -> dict[str, DataLoader]:
    """Return tokenized train, val, and test dataloaders."""
    # Stub guidance:
    # Goal:
    # - Return dataloaders that give the model token ids, attention masks, and labels.
    # - Use the same data split for LoRA and any baseline, so the comparison is fair.
    #
    # Inputs:
    # - `config`: LoRAFineTuningConfig
    # - For the default dataset, each raw example looks like:
    #   {"text": "Wall St. Bears Claw Back Into the Black...", "label": 2}
    #
    # Output:
    # - Return type: dict[str, DataLoader]
    # - Return keys: `train`, `val`, and `test`.
    # - Each dataloader should return a batch dictionary with:
    #   - `input_ids`: tensor shaped `[batch_size, config.max_length]`
    #   - `attention_mask`: tensor shaped `[batch_size, config.max_length]`
    #   - `labels`: tensor shaped `[batch_size]`
    # - With the default config, the shapes are:
    #   - `input_ids`: `[16, 96]`
    #   - `attention_mask`: `[16, 96]`
    #   - `labels`: `[16]`
    #
    # Method:
    # - Use `datasets.load_dataset(config.dataset_name)` from Hugging Face Datasets.
    # - Use `AutoTokenizer.from_pretrained(config.model_name)` from Transformers.
    # - Use `dataset.map(tokenize_batch, batched=True)` on each split.
    # - Use `split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])`.
    #
    # Steps:
    # 1. Load `ag_news` with `load_dataset(config.dataset_name)`.
    # 2. Read `raw_data["train"]` and `raw_data["test"]`.
    # 3. Split the training set with:
    #    `train_data.train_test_split(test_size=0.2, shuffle=True, seed=config.random_state)`.
    #    This gives a train split and a validation split.
    # 4. Load the tokenizer with:
    #    `AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_name)`.
    # 5. In `tokenize_batch`, call:
    #    `tokenizer(texts, padding="max_length", truncation=True, max_length=config.max_length)`.
    #    `padding="max_length"` pads short texts to `config.max_length`.
    #    `truncation=True` cuts long texts to `config.max_length`.
    # 6. Add labels with `token_dict["labels"] = batch[config.label_column]`.
    # 7. Map the helper over train, val, and test with `.map(tokenize_batch, batched=True)`.
    # 8. Set torch format on each split with:
    #    `split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])`.
    # 9. Build three `DataLoader` objects and return them in a dictionary.
    #
    # Checks:
    # - `batch = next(iter(loaders["train"]))`
    # - `batch.keys()` should include `input_ids`, `attention_mask`, and `labels`.
    # - `batch["input_ids"].shape` should be `[config.batch_size, config.max_length]`.
    # - `batch["labels"].shape` should be `[config.batch_size]`.

    from datasets import load_dataset
    from transformers import AutoTokenizer

    raw_data = load_dataset(config.dataset_name)
    train_data = raw_data["train"]
    test_data = raw_data["test"]

    split_dict = train_data.train_test_split(
        test_size=0.2,
        shuffle=True,
        seed=config.random_state,
    )
    train_data = split_dict["train"]
    val_data = split_dict["test"]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_name)

    # Implementation steps:
    # 1. In `tokenize_batch`, read the list of texts from `batch[config.text_column]`.
    # 2. Pass that list of texts to:
    #    `tokenizer(texts, padding="max_length", truncation=True, max_length=config.max_length)`.
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
        # - Tiny example:
        #   batch = {"text": ["Stocks rose today", "A team won"], "label": [2, 1]}
        #
        # Do these steps here:
        # 1. Tokenize the list of text strings.
        #    The tokenizer should create both `input_ids` and `attention_mask`.
        #    You do not build the attention mask by hand here.
        #    Important:
        #    - `tokenizer.encode(text)` returns only one list of token ids.
        #    - `tokenizer(texts, padding="max_length", truncation=True, max_length=config.max_length)`
        #      returns a dictionary like:
        #      {"input_ids": [...], "attention_mask": [...]}
        # 2. Store the class ids under the key `labels`.
        # 3. Return the tokenized batch.
        #
        # Return:
        # - dictionary with `input_ids`, `attention_mask`, and `labels`
        # - For the tiny example above with `max_length=6`, the return has keys like:
        #   {
        #       "input_ids": [[101, 15768, 3123, 2651, 102, 0], ...],
        #       "attention_mask": [[1, 1, 1, 1, 1, 0], ...],
        #       "labels": [2, 1],
        #   }

        texts = batch[config.text_column]
        labels = batch[config.label_column]

        token_dict = tokenizer(texts, padding="max_length", truncation=True, max_length=config.max_length)
        input_ids = token_dict["input_ids"]
        attention_mask = token_dict["attention_mask"]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized_train = train_data.map(tokenize_batch, batched=True)

    tokenized_val = val_data.map(tokenize_batch, batched=True)

    tokenized_test = test_data.map(tokenize_batch, batched=True)

    columns = ["input_ids", "attention_mask", "labels"]
    tokenized_train.set_format(type="torch", columns=columns)
    tokenized_val.set_format(type="torch", columns=columns)
    tokenized_test.set_format(type="torch", columns=columns)

    return {
        "train": torch.utils.data.DataLoader(
            dataset=tokenized_train,
            batch_size=config.batch_size,
            shuffle=True,
        ),
        "val": torch.utils.data.DataLoader(
            dataset=tokenized_val,
            batch_size=config.batch_size,
            shuffle=False,
        ),
        "test": torch.utils.data.DataLoader(
            dataset=tokenized_test,
            batch_size=config.batch_size,
            shuffle=False,
        ),
    }

    #raise NotImplementedError("Return tokenized dataloaders for the LoRA fine-tuning task.")

# Load the base pretrained model before adding LoRA.
# Return a classifier that outputs raw class logits.
def load_base_model(config: LoRAFineTuningConfig) -> nn.Module:
    """Load the pretrained base model for sequence classification."""
    # Stub guidance:
    # Goal:
    # - Load a pretrained text classifier that can read tokenized AG News batches.
    # - The model should return raw class scores, called logits.
    #
    # Inputs:
    # - `config`: LoRAFineTuningConfig
    # - Important fields:
    #   - `config.model_name`: default `"distilbert-base-uncased"`
    #   - `config.num_classes`: default `4`
    #
    # Output:
    # - Return type: nn.Module
    # - Return a Transformers sequence-classification model.
    # - When called with a batch, it should return an object with:
    #   - `outputs.loss`: scalar tensor when `labels` are passed
    #   - `outputs.logits`: tensor shaped `[batch_size, config.num_classes]`
    # - With the default config and batch size 16, logits should be shaped `[16, 4]`.
    #
    # Method:
    # - Import `AutoModelForSequenceClassification` from `transformers`.
    # - Use this exact call shape:
    #   `AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_classes)`.
    # - `num_labels=config.num_classes` tells the model to create 4 output scores for AG News.
    #
    # Steps:
    # 1. Import `AutoModelForSequenceClassification`.
    # 2. Load the model with the `.from_pretrained(...)` call shown above.
    # 3. Move the model to `device` with `model.to(device)`.
    # 4. Return the model.
    #
    # Checks:
    # - Do not apply softmax in this function.
    # - Cross-entropy expects raw logits, not probabilities.
    # - A quick forward pass should make `outputs.logits.shape[-1] == config.num_classes`.
    
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_classes)
    model = model.to(device)

    return model
    #raise NotImplementedError("Load and return a pretrained sequence-classification model.")

# Add LoRA adapters to the base model.
# Return the wrapped model.
# In plain words:
# - the big base model stays mostly fixed
# - the small adapter path learns the task change
def add_lora_adapters(model: nn.Module, config: LoRAFineTuningConfig) -> nn.Module:
    """Add LoRA adapters to the model.
    Beginner-friendly hint:
    - import PEFT tools inside this function
    - use a LoRA config that matches the model
    - print trainable parameters after adding adapters
    """
    # Stub guidance:
    # Goal:
    # - Add small trainable LoRA adapters to the pretrained classifier.
    # - Return the wrapped PEFT model.
    #
    # Inputs:
    # - `model`: nn.Module
    #   A sequence-classification model from `load_base_model`.
    # - `config`: LoRAFineTuningConfig
    #
    # Output:
    # - Return type: nn.Module
    # - Return a PEFT model that still accepts:
    #   - `input_ids`: `[batch_size, config.max_length]`
    #   - `attention_mask`: `[batch_size, config.max_length]`
    #   - `labels`: `[batch_size]`
    # - The forward output should still have `.loss` and `.logits`.
    #
    # Method:
    # - Import these from `peft`:
    #   - `LoraConfig`
    #   - `TaskType`
    #   - `get_peft_model`
    # - Use this call shape:
    #   `LoraConfig(`
    #   `    task_type=TaskType.SEQ_CLS,`
    #   `    r=config.lora_rank,`
    #   `    lora_alpha=config.lora_alpha,`
    #   `    lora_dropout=config.lora_dropout,`
    #   `    target_modules=["q_lin", "v_lin"],`
    #   `)`
    # - What each argument means:
    #   - `task_type=TaskType.SEQ_CLS` tells PEFT this is sequence classification.
    #     Sequence classification means one whole text gets one label, like AG News topic labels.
    #   - `r=config.lora_rank` sets the LoRA rank.
    #     Rank means the size of the small adapter path.
    #     Bigger rank means more trainable weights and more room to learn.
    #     Smaller rank means cheaper training but less room to adapt.
    #   - `lora_alpha=config.lora_alpha` sets the strength of the LoRA update.
    #     PEFT commonly scales the adapter update by `lora_alpha / r`.
    #     With the default config, that is `16 / 8 = 2`.
    #   - `lora_dropout=config.lora_dropout` drops part of the LoRA path during training.
    #     With the default config, `0.05` means about 5 percent is dropped.
    #     This can help the adapter avoid memorizing the training set too much.
    #   - `target_modules=["q_lin", "v_lin"]` tells PEFT where to attach LoRA.
    #     In DistilBERT, `q_lin` and `v_lin` are attention linear layers.
    #     `q_lin` helps decide what each token looks at.
    #     `v_lin` helps pass along the information from the tokens being looked at.
    # - For DistilBERT, `q_lin` and `v_lin` are common attention projection module names.
    # - If the model changes, print or inspect `model.named_modules()` and choose real linear layer names.
    # - Then call:
    #   `get_peft_model(model, peft_config)`.
    #
    # Steps:
    # 1. Build a `LoraConfig`.
    # 2. Pass the base model and config to `get_peft_model(model, peft_config)`.
    # 3. Move the returned model to `device`.
    # 4. Print trainable parameters with `lora_model.print_trainable_parameters()` if that method exists.
    # 5. Return the LoRA model.
    #
    # Checks:
    # - `count_parameters(lora_model)["trainable_percent"]` should be much smaller than 100.
    # - A quick forward pass should still produce logits shaped `[batch_size, config.num_classes]`.

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r = config.lora_rank, 
                                lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                                target_modules=["q_lin", "v_lin"])

    lora_model = get_peft_model(model = model, peft_config = lora_config)
    lora_model = lora_model.to(device)

    lora_model.print_trainable_parameters()
    #the above line prints: trainable params: 741,124 || all params: 67,697,672 || trainable%: 1.0948

    return lora_model
    #raise NotImplementedError("Add LoRA adapters and return the PEFT model.")

# Count parameters and summarize how many are trainable.
# Return:
# - total parameters
# - trainable parameters
# - trainable percent
# This is one of the main proof points for LoRA.
def count_parameters(model: nn.Module) -> dict[str, float]:
    """Return total, trainable, and trainable percent."""
    # Stub guidance:
    # Goal:
    # - Count all model weights.
    # - Count only the weights that training will update.
    # - Return the percent of weights that are trainable.
    #
    # Inputs:
    # - `model`: nn.Module
    #
    # Output:
    # - Return type: dict[str, float]
    # - Return exactly these keys:
    #   - `total_parameters`
    #   - `trainable_parameters`
    #   - `trainable_percent`
    # - Values are counts of parameters, except `trainable_percent`, which is a percent.
    #
    # Formula:
    # - `total_parameters = sum(p.numel() for p in model.parameters())`
    # - `trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)`
    # - `trainable_percent = 100 * trainable_parameters / total_parameters`
    # - `p.numel()` means "how many numbers are inside this tensor".
    #
    # Tiny example:
    # - A weight shaped `[3, 4]` has `3 * 4 = 12` parameters.
    # - A bias shaped `[4]` has `4` parameters.
    # - Together they have `16` parameters.
    #
    # Steps:
    # 1. Loop over `model.parameters()`.
    # 2. Add every parameter size to `total_parameters`.
    # 3. Add only parameters with `p.requires_grad == True` to `trainable_parameters`.
    # 4. Compute the percent.
    # 5. Return the dictionary with the exact keys listed above.
    #
    # This is one of the most important checks in the whole file.
    # If trainable parameters are not much smaller,
    # the exercise is not really demonstrating LoRA properly.

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = 100 * trainable_parameters / total_parameters

    return {'total_parameters': total_parameters, 'trainable_parameters': trainable_parameters, 'trainable_percent': trainable_percent}
    #raise NotImplementedError("Count total and trainable model parameters.")

def build_training_objects(config: LoRAFineTuningConfig) -> tuple[nn.Module, torch.optim.Optimizer, dict[str, float]]:
    """Return LoRA model, optimizer, and parameter counts."""
    # Stub guidance:
    # Goal:
    # - Build the base model, add LoRA adapters, count parameters, and create the optimizer.
    #
    # Inputs:
    # - `config`: LoRAFineTuningConfig
    #
    # Output:
    # - Return type: tuple[nn.Module, torch.optim.Optimizer, dict[str, float]]
    # - Return exactly:
    #   `(lora_model, optimizer, parameter_counts)`
    # - `parameter_counts` should have:
    #   - `total_parameters`
    #   - `trainable_parameters`
    #   - `trainable_percent`
    #
    # Method:
    # - Call your own helpers:
    #   - `load_base_model(config)`
    #   - `add_lora_adapters(base_model, config)`
    #   - `count_parameters(lora_model)`
    # - Build the optimizer with:
    #   `torch.optim.AdamW((p for p in lora_model.parameters() if p.requires_grad), lr=config.learning_rate)`.
    # - `AdamW` is a common optimizer for transformer fine-tuning.
    #
    # Steps:
    # 1. Load the base classifier.
    # 2. Wrap it with LoRA adapters.
    # 3. Count parameters after LoRA is added.
    # 4. Build the optimizer using only trainable parameters.
    # 5. Return `(lora_model, optimizer, parameter_counts)`.
    #
    # Checks:
    # - The optimizer should not receive frozen parameters.
    # - `parameter_counts["trainable_percent"]` should be much smaller than 100.
    #

    base_model = load_base_model(config)
    lora_model = add_lora_adapters(base_model, config)
    count_params = count_parameters(lora_model)

    optimizer = torch.optim.AdamW(params=lora_model.parameters(), lr=config.learning_rate)

    return lora_model, optimizer, count_params
    #raise NotImplementedError("Build the LoRA model and optimizer.")

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train one LoRA fine-tuning epoch."""
    # Stub guidance:
    # Goal:
    # - Train the model for one pass over the dataloader.
    # - Return the average loss and accuracy for that pass.
    #
    # Inputs:
    # - `model`: nn.Module
    # - `loader`: DataLoader that returns batches with:
    #   - `input_ids`: `[batch_size, max_length]`
    #   - `attention_mask`: `[batch_size, max_length]`
    #   - `labels`: `[batch_size]`
    # - `optimizer`: torch.optim.Optimizer
    #
    # Output:
    # - Return type: dict[str, float]
    # - Return exactly:
    #   - `train_loss`: mean loss across all examples
    #   - `train_accuracy`: mean accuracy across all examples
    #
    # Method:
    # - Put the model in training mode with `model.train()`.
    # - Move each batch tensor to `device`.
    # - Call the model with:
    #   `model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])`.
    # - The output has:
    #   - `outputs.loss`: scalar loss tensor
    #   - `outputs.logits`: raw class scores shaped `[batch_size, num_classes]`
    #
    # Steps:
    # 1. Set `model.train()`.
    # 2. For each batch, move `input_ids`, `attention_mask`, and `labels` to `device`.
    # 3. Call `optimizer.zero_grad()`.
    # 4. Run the forward pass and read `outputs.loss`.
    # 5. Call `loss.backward()`.
    # 6. Call `optimizer.step()`.
    # 7. Compute predictions with `outputs.logits.argmax(dim=-1)`.
    # 8. Count correct predictions with `(preds == labels).sum().item()`.
    # 9. Average loss by number of examples, not by the last batch only.
    #
    # Checks:
    # - Do not call `torch.no_grad()` while training.
    # - Do not apply softmax before `argmax`; the largest logit gives the predicted class.
    
    model.train()

    losses, accs = [], []

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
        accs.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss": mean_loss, "accuracy": mean_acc}

    #raise NotImplementedError("Return mean train loss and accuracy.")

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """Evaluate the LoRA model."""
    # Stub guidance:
    # Goal:
    # - Measure loss and accuracy without updating model weights.
    #
    # Inputs:
    # - `model`: nn.Module
    # - `loader`: DataLoader that returns batches with:
    #   - `input_ids`: `[batch_size, max_length]`
    #   - `attention_mask`: `[batch_size, max_length]`
    #   - `labels`: `[batch_size]`
    #
    # Output:
    # - Return type: dict[str, float]
    # - Return exactly:
    #   - `eval_loss`: mean loss across all examples
    #   - `eval_accuracy`: mean accuracy across all examples
    #
    # Method:
    # - Put the model in eval mode with `model.eval()`.
    # - Wrap the loop in `with torch.no_grad():`.
    # - Call the model with:
    #   `model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])`.
    #
    # Steps:
    # 1. Set `model.eval()`.
    # 2. Start `with torch.no_grad():`.
    # 3. Move each batch tensor to `device`.
    # 4. Run the forward pass.
    # 5. Add `outputs.loss.item() * batch_size` to the total loss.
    # 6. Compute predictions with `outputs.logits.argmax(dim=-1)`.
    # 7. Count correct predictions and total examples.
    # 8. Return mean loss and mean accuracy.
    #
    # Checks:
    # - Evaluation should not call `loss.backward()` or `optimizer.step()`.
    # - Accuracy should be `correct / total`.
    #

    model.train()

    losses, accs = [], []

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
        accs.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss": mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Return mean evaluation loss and accuracy.")

def compare_lora_tradeoff(parameter_counts: dict[str, float], metrics: dict[str, float]) -> str:
    """Write a short note about the LoRA tradeoff."""
    # Stub guidance:
    # Goal:
    # - Return a short note that explains the LoRA tradeoff in plain English.
    #
    # Inputs:
    # - `parameter_counts`: dict[str, float]
    #   Expected keys:
    #   - `total_parameters`
    #   - `trainable_parameters`
    #   - `trainable_percent`
    # - `metrics`: dict[str, float]
    #   Expected useful key:
    #   - `val_accuracy`
    #
    # Output:
    # - Return type: str
    # - Return one or two plain sentences.
    # - Mention concrete numbers when they are available.
    #
    # Example output:
    # - "LoRA trained 0.8% of the model weights and reached 84.0% validation accuracy.
    #    That means it used far fewer trainable weights while still learning the task."
    #
    # Steps:
    # 1. Read `trainable_percent` from `parameter_counts`.
    # 2. Read `val_accuracy` from `metrics`.
    # 3. Format both values so the note is easy to read.
    # 4. Say the tradeoff clearly: fewer trainable weights, hopefully similar task quality.
    #
    # Checks:
    # - Do not use vague words like "good" without numbers.
    # - Keep the note beginner-friendly.
    #
    trainable_percent = float(parameter_counts["trainable_percent"])
    trainable_parameters = int(parameter_counts["trainable_parameters"])
    total_parameters = int(parameter_counts["total_parameters"])

    accuracy = metrics.get("val_accuracy", metrics.get("accuracy"))
    if accuracy is None:
        return (
            f"LoRA trained {trainable_parameters:,} of {total_parameters:,} weights "
            f"({trainable_percent:.2f}%). That means training updated only a small part "
            "of the model instead of the whole model."
        )

    accuracy_percent = float(accuracy) * 100.0
    return (
        f"LoRA trained {trainable_parameters:,} of {total_parameters:,} weights "
        f"({trainable_percent:.2f}%) and reached {accuracy_percent:.1f}% validation accuracy. "
        "That means it used far fewer trainable weights while still learning the task."
    )
    #raise NotImplementedError("compare_lora_tradeoff.")

def run_training(config: LoRAFineTuningConfig) -> dict[str, float | str]:
    """Run the full LoRA fine-tuning workflow."""
    # Stub guidance:
    # Goal:
    # - Run the whole LoRA workflow and return the final numbers.
    #
    # Inputs:
    # - `config`: LoRAFineTuningConfig
    #
    # Output:
    # - Return type: dict[str, float | str]
    # - Return exactly these required keys:
    #   - `val_accuracy`
    #   - `total_parameters`
    #   - `trainable_parameters`
    #   - `trainable_percent`
    #   - `comparison_note`
    # - You may also include helpful extra keys such as `train_loss`, `val_loss`, or `test_accuracy`.
    #
    # Method:
    # - Use your own helpers:
    #   - `load_tokenized_dataloaders(config)`
    #   - `build_training_objects(config)`
    #   - `train_one_epoch(model, loaders["train"], optimizer)`
    #   - `evaluate(model, loaders["val"])`
    #   - `compare_lora_tradeoff(parameter_counts, metrics)`
    #
    # Steps:
    # 1. Build the dataloaders.
    # 2. Build the LoRA model, optimizer, and parameter counts.
    # 3. For each epoch, call `train_one_epoch(...)`.
    # 4. After each epoch, call `evaluate(...)` on the validation loader.
    # 5. Keep the final validation metrics.
    # 6. Build `comparison_note`.
    # 7. Return one final metrics dictionary with the required keys.
    #
    # Checks:
    # - Validation accuracy should come from `evaluate(model, loaders["val"])`.
    # - The parameter counts should come from the LoRA model, not the base model before LoRA.
    # - The returned dictionary must pass `check_metrics(...)`.
    #

    loaders = load_tokenized_dataloaders(config)
    lora_model, optimizer, param_counts = build_training_objects(config)
    train_dict: dict[str, float] = {"loss": 0.0, "accuracy": 0.0}
    val_dict: dict[str, float] = {"loss": 0.0, "accuracy": 0.0}

    for epoch in range(config.epochs):
        train_dict = train_one_epoch(lora_model, loaders["train"], optimizer)
        val_dict = evaluate(lora_model, loaders["val"])

        print(
            f"Epoch {epoch + 1} -> "
            f"train_loss: {train_dict['loss']:.4f}, "
            f"train_accuracy: {train_dict['accuracy']:.4f}, "
            f"val_loss: {val_dict['loss']:.4f}, "
            f"val_accuracy: {val_dict['accuracy']:.4f}"
        )

    test_dict = evaluate(lora_model, loaders["test"])
    print(f"test_loss: {test_dict['loss']:.4f}, "
          f"test_accuracy: {test_dict['accuracy']:.4f}"
        )

    metrics = {
        "train_loss": train_dict["loss"],
        "train_accuracy": train_dict["accuracy"],
        "val_loss": val_dict["loss"],
        "val_accuracy": val_dict["accuracy"],
        "test_loss": test_dict["loss"],
        "test_accuracy": test_dict["accuracy"],
        "total_parameters": param_counts["total_parameters"],
        "trainable_parameters": param_counts["trainable_parameters"],
        "trainable_percent": param_counts["trainable_percent"],
    }
    metrics["comparison_note"] = compare_lora_tradeoff(param_counts, metrics)

    return metrics

    #raise NotImplementedError("Fine-tune with LoRA and return final metrics.")

def check_metrics(metrics: dict[str, float | str], target: float) -> bool:
    """Return True when the LoRA summary has the expected values."""
    required = {"val_accuracy", "total_parameters", "trainable_parameters", "trainable_percent", "comparison_note"}
    return required.issubset(metrics) and float(metrics.get("val_accuracy", 0.0)) >= target

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
    """Run the LoRA scaffold one step at a time."""
    config = LoRAFineTuningConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 7
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    loaders, ok = report_task_progress("load_tokenized_dataloaders", load_tokenized_dataloaders, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("describe_lora_plan", describe_lora_plan, config)
    if not count_or_stop(ok):
        return
    base_model, ok = report_task_progress("load_base_model", load_base_model, config)
    if not count_or_stop(ok):
        return
    lora_model, ok = report_task_progress("add_lora_adapters", add_lora_adapters, base_model, config)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("count_parameters", count_parameters, lora_model)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics, config.target_val_accuracy)
        print_pass_or_try_again(
            passed,
            "val_accuracy",
            float(metrics.get("val_accuracy", 0.0)),
            f">= {config.target_val_accuracy}",
        )
if __name__ == "__main__":
    main()
