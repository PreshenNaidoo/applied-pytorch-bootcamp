"""
============================================================
EXERCISE 04: MINI TRANSFORMER TEXT CLASSIFIER
============================================================

WHAT YOU ARE BUILDING
You are building a small transformer encoder for text classification.

This is still a teaching model.
It is not meant to be large.
It is meant to help you connect the ideas from tokenization, padding, and attention.

Default path in this scaffold:
- token embeddings
- positional embeddings
- one or two transformer encoder blocks
- mean pooling over real tokens or a CLS-style token
- classifier head

Plain-language meaning:
- token embeddings turn token ids into learned vectors
- positional embeddings tell the model where each token sits in the sequence
- an encoder block is one layer that mixes token information with attention and then processes it with a small feed-forward network
- mean pooling means average the token features into one text summary vector
- a classifier head is the final small part that turns that summary into class scores

Beginner intuition:
- a text classifier is trying to answer one question about the whole sentence or article
- but the model first builds a feature vector for each token
- attention lets one token borrow useful information from other tokens in the same text
- after that, pooling turns many token vectors into one summary vector for the whole text
- the classifier head reads that final summary and chooses the class

WHY THIS EXERCISE EXISTS
This exercise connects the earlier text exercises to a transformer encoder.

It teaches:
- why token order still matters even when you use attention
- how positional information gets added
- how self-attention can mix information across the whole sequence
- how to turn a sequence of token features into one class prediction

DATASET
Reuse AG News.

Use the same shared split rule as the other text exercises:
- combine the original train and test splits
- make your own `60% train / 20% val / 20% test` split

HOW TO LOAD OR DOWNLOAD THE DATASET
Use the same beginner-friendly path as before:
- `load_dataset("ag_news")`
- combine the original splits
- make one shared `60 / 20 / 20` split
- build the vocabulary from the training text only
- pad or truncate to one fixed `max_length`

INPUTS AND OUTPUTS
Input batch:
- `token_ids`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size, max_length]`
- `lengths`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`
- `labels`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`

Example batch:
- `token_ids`: `[32, 64]`
- `lengths`: `[32]`
- `labels`: `[32]`

What those shapes mean:
- `token_ids: [32, 64]` means 32 examples in the batch and space for up to 64 token ids per example
- `lengths: [32]` means one real-text length for each example
- `labels: [32]` means one class label for each example

Transformer hidden states:
- type: `torch.Tensor`
- shape: `[batch_size, max_length, embed_dim]`

Final model output:
- type: `torch.Tensor`
- shape: `[batch_size, 4]`
- meaning: one raw logit per class

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Default path:
- token embedding table
- positional embedding table
- one or more transformer encoder blocks
- a pooling step that turns `[batch_size, max_length, embed_dim]` into `[batch_size, embed_dim]`
- classifier head

Helpful note:
- a transformer encoder can let each token look at other tokens on both the left and the right
- that is different from a decoder, which usually blocks future tokens

Why that helps for classification:
- in classification, you already have the full text
- so there is no need to hide future words
- this means a word near the start can use words near the end to understand the full meaning
- for example, an early word like "apple" may mean something different once the model also sees later words like "iphone" or "pie"

Clear default pooling choice:
- mean pool only across real tokens, not across padding tokens

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

Important:
- return raw logits
- do not apply softmax before the loss

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
A tiny correct transformer encoder on AG News should often reach about `0.84` to `0.91` validation accuracy.

COMMON MISTAKES
- forgetting positional embeddings
- averaging over padded positions as if they were real tokens
- using float token ids instead of `torch.long`
- applying softmax before `CrossEntropyLoss`
- confusing encoder self-attention with causal decoder masking

STEP-BY-STEP TODO
1. Reuse the text split and vocabulary pipeline.
2. Build token embeddings.
3. Build positional embeddings.
4. Build one or more encoder blocks.
5. Add a padding-aware pooling step.
6. Add the classifier head.
7. Train for several epochs.
8. Evaluate on validation data.
9. Check whether the metrics make sense.

What each step is doing in simple words:
1. turn raw text into token ids the model can read
2. give each token a learned vector
3. add position information so the model knows order
4. let tokens share information with attention and small feed-forward layers
5. turn all token vectors into one text summary
6. map that summary to class scores
7. teach the model by comparing scores to the true label
8. check how well it works on unseen data
9. make sure the numbers are believable and not clearly broken

STRETCH GOALS
- compare mean pooling and CLS-style pooling
- compare one encoder block and two encoder blocks
- inspect the effect of different `max_length` values

DEFINITION OF DONE
You are done when:
- the model returns raw logits with shape `[batch_size, 4]`
- you can explain what positional embeddings do in plain words
- you can explain why encoder attention can look both left and right
- the validation accuracy is in a sensible range
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets as hf_datasets


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data_utils import describe_text_batch_rules
from common.reproducibility import set_seed
from common.training_checks import check_threshold, print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"

class MiniTransformerClassifierConfig:
    """Settings for the mini transformer text-classifier exercise."""

    dataset_name: str = "ag_news"
    num_classes: int = 4
    max_vocab_size: int = 20000
    max_length: int = 64
    embed_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256
    num_layers: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 6
    random_state: int = 42
    target_val_accuracy: float = 0.85




def describe_encoder_shapes(config: MiniTransformerClassifierConfig) -> dict[str, object]:
    """Return the expected shape flow for the mini transformer.

    Return a dict with exactly these keys:
    - `token_batch_shape`
    - `hidden_state_shape`
    - `pooled_shape`
    - `logit_shape`
    - `notes`
    """
    return {
        "token_batch_shape": (config.batch_size, config.max_length),
        "hidden_state_shape": (config.batch_size, config.max_length, config.embed_dim),
        "pooled_shape": (config.batch_size, config.embed_dim),
        "logit_shape": (config.batch_size, config.num_classes),
        "notes": describe_text_batch_rules(),
    }


        #raise NotImplementedError("Run the mini transformer encoder and return raw class logits.")


# Load the text split plus the vocabulary pieces you need.
# Return a dict with exactly these keys:
# - `train_texts`
# - `val_texts`
# - `test_texts`
# - `train_labels`
# - `val_labels`
# - `test_labels`
# - `token_to_id`
# - `pad_id`
# Beginner-friendly rule:
# - build the vocabulary from the training text only
# - keep one pad token id for shorter sequences
# - later, the pad id helps you avoid pooling over fake padding tokens
# Intuition:
# - the vocabulary is the dictionary the model uses to turn words or tokens into ids
# - building it from train text only avoids peeking at val or test text
# - the pad id is a special filler token so shorter examples can sit in the same batch as longer ones
def load_text_artifacts(config: MiniTransformerClassifierConfig) -> dict[str, object]:

    data = hf_datasets.load_dataset(path=config.dataset_name)
    train_data = data["train"]
    test_data = data["test"]

    train_list = list(train_data["text"])
    train_labels_list = list(train_data["label"])

    test_texts = list(test_data["text"])
    test_labels = list(test_data["label"])

    num_train = int(len(train_labels_list) * 0.80)
    train_texts, train_labels = train_list[:num_train], train_labels_list[:num_train]
    val_texts, val_labels = train_list[num_train:], train_labels_list[num_train:]

    return {"train_texts": train_texts,"train_labels": train_labels,
            "val_texts": val_texts,"val_labels": val_labels,
            "test_texts": test_texts,"test_labels": test_labels}


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocabulary_dict, max_length):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.vocabulary_dict = vocabulary_dict
        self.max_length = max_length
        self.pad_id = 0
        self.unk_id = 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> dict:
        text = self.texts[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)

        words_list = [word.strip(string.punctuation) for word in text.split()]

        token_ids = []
        c = 0
        
        for word in words_list:
            if not word:
                continue

            if word in self.vocabulary_dict:
                token_ids.append(self.vocabulary_dict[word])                
            else:
                token_ids.append(self.unk_id)

            c+=1
            if c == self.max_length:
                break

        length = len(token_ids)
        token_ids += [self.pad_id] * (self.max_length - length)

        length = torch.tensor(len(token_ids), dtype=torch.long)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        return {"token_ids":token_ids, "lengths":length, "labels":label}



@dataclass

# Build the train, validation, and test dataloaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Each batch should be a dict with exactly these keys:
# - `token_ids`
# - `lengths`
# - `labels`
# Beginner-friendly path:
# - create a small dataset class that turns raw text into padded token ids
# - in `__getitem__`, return one dict with:
#   - `token_ids`
#   - `lengths`
#   - `labels`
# - `lengths` is still useful here because mean pooling should only use the real tokens
# Intuition:
# - a dataloader is just the part that serves batches to the model
# - each batch needs both the token ids and the real lengths
# - the real lengths tell you where the true text ends and where padding begins
def load_dataloaders(config: MiniTransformerClassifierConfig) -> dict[str, DataLoader]:

    data_dict = load_text_artifacts(config)
    train_texts = data_dict["train_texts"]

    vocabulary = {"[PAD]":0, "[UNK]":1}
    index = 2
    for text in train_texts:
        words_list = [word.strip(string.punctuation) for word in text.split()]
        for word in words_list:
            if not word:
                continue
            if word not in vocabulary:
                vocabulary[word] = index
                index+=1
            if index == config.max_vocab_size:
                break
        if index == config.max_vocab_size:
            break

    train_ds = TokenDataset(texts=data_dict["train_texts"], labels=data_dict["train_labels"], vocabulary_dict=vocabulary, max_length=config.max_length)
    val_ds = TokenDataset(data_dict["val_texts"], data_dict["val_labels"], vocabulary, config.max_length)
    test_ds = TokenDataset(data_dict["test_texts"], data_dict["test_labels"], vocabulary, config.max_length)

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}



class MiniTransformerTextClassifier(nn.Module):
    """Toy transformer encoder classifier stub."""

    def __init__(self, vocab_size: int, pad_id: int, config: MiniTransformerClassifierConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.config = config
        # Beginner-friendly build order:
        # 1. token embedding layer
        # 2. positional embedding layer
        # 3. transformer encoder blocks
        # 4. classifier head
        #
        # Simple default objects:
        # - `nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)`
        # - `nn.Embedding(max_length, embed_dim)` for positions
        # - `nn.TransformerEncoderLayer(..., batch_first=True)`
        # - `nn.TransformerEncoder(...)`
        # - `nn.Linear(embed_dim, num_classes)`
        #
        # Intuition:
        # - token embeddings answer "what token is this?"
        # - positional embeddings answer "where is this token?"
        # - the encoder mixes information across the whole text
        # - the classifier head uses the final text summary to make one decision

        # Both layers below use `nn.Embedding`, but they learn different lookup tables.
        #
        # `self.token_embed` looks up token ids.
        # It answers: "what word or token is this?"
        # Example:
        # - token id 57 might mean "market"
        # - token id 103 might mean "football"
        # Each token id gets its own learned vector.
        #
        # `self.positional_embed` looks up position ids.
        # It answers: "where is this token sitting in the sequence?"
        # Example:
        # - position 0 means "first token"
        # - position 1 means "second token"
        # - position 2 means "third token"
        # Each position gets its own learned vector too.
        #
        # So the layer type is the same, but the meaning is different:
        # - token embedding = token identity
        # - positional embedding = token location
        #
        # Why do we need both?
        # A transformer does not know order by itself.
        # If you only give token embeddings, the model can see which words are present,
        # but it does not clearly know where they appear.
        # Positional embeddings add that missing order signal.
        #
        # Intuition:
        # - token embedding says "this is the word"
        # - positional embedding says "this is the seat number"
        # - adding them says "this word is sitting in this place"
        #
        # Shape idea:
        # - token ids start as `[batch_size, max_length]`
        # - after token lookup they become `[batch_size, max_length, embed_dim]`
        # - position ids also become `[batch_size, max_length, embed_dim]`
        # - then we add those two tensors together
        #
        # Table sizes:
        # - token embedding table shape: `[vocab_size, embed_dim]`
        # - positional embedding table shape: `[max_length, embed_dim]`
        # They are different tables because one stores token meaning and the other stores position meaning.

        self.token_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.embed_dim, padding_idx=pad_id)
        self.positional_embed = nn.Embedding(num_embeddings=config.max_length, embedding_dim=config.embed_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_heads, 
                                                                    dim_feedforward=config.mlp_dim, batch_first=True)

        self.transformer_enc = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, 
                                                    num_layers=config.num_layers)

        self.classifier_head = nn.Linear(in_features = config.embed_dim, out_features=config.num_classes)


        # raise NotImplementedError(
        #     "Build token embeddings, positional embeddings, encoder blocks, a pooling step, and the classifier head."
        # )

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`.

        Clear default path:
        - add token embeddings and positional embeddings
        - run the encoder blocks
        - mean pool across real tokens only
        - return raw logits

        Step-by-step tensor path:
        1. token ids -> token embeddings
           shape: `[batch_size, max_length, embed_dim]`
        2. build position ids like `0, 1, 2, ... max_length - 1`
        3. look up positional embeddings
        4. add token embeddings and positional embeddings
        5. run the encoder blocks
        6. pool across real tokens only
           result shape: `[batch_size, embed_dim]`
        7. run the classifier head
           result shape: `[batch_size, num_classes]`

        Why pool only real tokens?
        - padding tokens are fake placeholders added just to make batch shapes line up
        - they are not part of the real text
        - if you average over them, the text summary gets watered down by fake information
        """

        batch_size, max_length = token_ids.shape[0], token_ids.shape[1]  #batch_size=32, max_length=64

        token_embeddings = self.token_embed(token_ids)  # (32, 64, 128)
        
        positions = torch.arange(start=0, end=max_length, device=token_ids.device)  # (64)
        position_ids = positions.unsqueeze(0).expand(size=(batch_size, max_length)) # (32, 64)

        positional_embeddings = self.positional_embed(position_ids)                 # (32, 64, 128)

        embeddings = token_embeddings + positional_embeddings                       # (32, 64, 128)

        x = self.transformer_enc(embeddings)                                        # (32, 64, 128)
        
        mask = positions.unsqueeze(dim=0) < lengths.unsqueeze(dim=1)                # (32, 64)

        masked_embeddings = embeddings * mask.unsqueeze(dim=2)                      # (32, 64, 128) * (32, 64)

        summed = masked_embeddings.sum(dim=1)
        pooled = summed / lengths.unsqueeze(dim=1)                                  #(32, 64) / (32, 1)

        x = self.classifier_head(pooled)

        return x


   # raise NotImplementedError("Return the AG News split plus the training-text vocabulary and pad token id.")

import string


    #raise NotImplementedError("Return train, val, and test dataloaders for the mini transformer text classifier.")


# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.CrossEntropyLoss`.
def build_training_objects(config: MiniTransformerClassifierConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    # Beginner-friendly object list:
    # - model: `MiniTransformerTextClassifier(...)`
    # - loss_fn: `nn.CrossEntropyLoss()`
    # - optimizer: something simple like `torch.optim.Adam(...)`

    model = MiniTransformerTextClassifier(vocab_size = config.max_vocab_size, pad_id=0, config=config)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())

    return model, loss_fn, optimizer
    #raise NotImplementedError("Return the mini transformer text classifier, CrossEntropyLoss, and optimizer.")


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
# - move the model, `token_ids`, `lengths`, and `labels` to the same device here
# - if you use `nn.TransformerEncoderLayer`, set `batch_first=True` to keep shapes easy: `[batch_size, max_length, embed_dim]`
# - do not return a history list here
# - if you mean-pool the hidden states, do not average across padded token positions
# Why this matters:
# - the model can only learn if all the needed tensors are in the same place
# - `batch_first=True` keeps the tensor order easier to read for beginners
# - ignoring padded positions keeps the text summary honest
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:

    model.train()

    losses, accuracies = [], []

    for data_dict in loader:
        token_ids = data_dict["token_ids"]
        labels = data_dict["labels"]
        lengths = data_dict["lengths"]

        token_ids = token_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        logits = model(token_ids, lengths)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = logits.argmax(dim=1).float()
        correct = (preds == labels).float()
        accuracies.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accuracies).mean().item()

    return {"loss":mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Return the mean train loss and mean train accuracy for one full epoch.")


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

    model.eval()

    losses, accs = [], []

    for data_dict in loader:
        token_ids = data_dict["token_ids"]
        labels = data_dict["labels"]
        lengths = data_dict["lengths"]

        token_ids = token_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(token_ids, lengths)
        loss = loss_fn(logits, labels)

        losses.append(loss.item())

        preds = logits.argmax(dim=1).float()
        correct = (preds == labels).float()
        accs.append(correct.mean().float())

    
    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss":mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Return the mean loss and mean accuracy for the loader you evaluate.")


# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `train_accuracy`
# - `val_loss`
# - `val_accuracy`
# Return type:
# - `dict[str, float]`
# This is the final epoch summary, not a history list.
def run_training(config: MiniTransformerClassifierConfig) -> dict[str, float]:

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    for i in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i} -> loss: {train_dict['loss']}, acc: {train_dict['accuracy']}, val_loss: {val_dict['loss']}, val_acc: {val_dict['accuracy']}")

    test_dict = evaluate(model, loaders["test"], loss_fn)
    print(f"Test results -> loss: {test_dict['loss']}, accuracy: {test_dict['accuracy']}")

    return {"train_loss": train_dict["loss"], "train_accuracy": train_dict["accuracy"],
            "val_loss": val_dict["loss"], "val_accuracy": val_dict["accuracy"]}

    #raise NotImplementedError("Run the full mini transformer training flow and return the final train and val metrics.")



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
    """Run the mini transformer text-classifier scaffold one step at a time."""
    config = MiniTransformerClassifierConfig()
    set_seed(config.random_state)
    print("Running exercise 04 starter scaffold...")
    print("Main goal: classify text with a small transformer encoder.")
    print(f"Device note: keep the model, token_ids, lengths, and labels on the same device. Using {device}.")

    completed_steps = 0
    total_steps = 9

    _, ok = report_task_progress("load_text_artifacts", load_text_artifacts, config)
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

    _, ok = report_task_progress("describe_encoder_shapes", describe_encoder_shapes, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress(
        "MiniTransformerTextClassifier.__init__",
        MiniTransformerTextClassifier,
        config.max_vocab_size,
        0,
        config,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_token_ids = torch.zeros(config.batch_size, config.max_length, dtype=torch.long)
    sample_lengths = torch.full((config.batch_size,), config.max_length, dtype=torch.long)
    _, ok = report_task_progress("MiniTransformerTextClassifier.forward", model, sample_token_ids, sample_lengths)
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

