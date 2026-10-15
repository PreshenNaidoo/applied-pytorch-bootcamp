"""
============================================================
EXERCISE 02: SEQUENCE MODELS RNN LSTM
============================================================

WHAT YOU ARE BUILDING
You are building a text classifier that reads tokens in order.

This is different from the previous exercise.
Before, a simple baseline could treat text like an unordered bag of words.
Now the order matters.

Default path in this scaffold:
- embedding layer
- one `nn.LSTM`
- last hidden state
- classifier head

You may use `nn.RNN` instead if you want, but `nn.LSTM` is the recommended default.
An embedding layer turns each token id into a learned vector of numbers.
An RNN or LSTM then reads those vectors one step at a time, in order.
The hidden state is the model's running memory while it reads.

WHY THIS EXERCISE EXISTS
A sequence model reads one token after another.
That makes it a good next step after simple text classification.

This exercise teaches:
- what token order means in practice
- what hidden states are
- how padding changes batching
- why sequence lengths matter

DATASET
Reuse AG News from the related exercise.

Use the same shared split rule:
- combine the original train and test splits
- make your own `60% train / 20% val / 20% test` split

HOW TO LOAD OR DOWNLOAD THE DATASET
Use the same beginner-friendly path as EXERCISE 17:
- `load_dataset("ag_news")`
- combine the original splits
- make one shared `60 / 20 / 20` split
- build the vocabulary from the training text only
- pad or truncate sequences to one fixed `max_length`

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
  - meaning: number of real tokens before padding
- `labels`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`
  - values: `0, 1, 2, 3`

Example batch:
- `token_ids`: `[32, 50]`
- `lengths`: `[32]`
- `labels`: `[32]`

What those shapes mean:
- `token_ids: [32, 50]` means 32 examples in the batch and space for up to 50 token ids per example
- `lengths: [32]` means one real-text length for each example
- `labels: [32]` means one class label for each example

Model output:
- type: `torch.Tensor`
- shape: `[batch_size, 4]`
- meaning: one raw logit per class

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Default path:
- embed the token ids
- run the sequence through one LSTM
- take one summary vector from the last layer hidden state
- feed that summary vector to a classifier head

What these words mean:
- `LSTM` is a kind of sequence model with a stronger memory than a basic `RNN`
- `last hidden state` means the memory after the model has read the whole sequence
- `classifier head` means the final small part of the model that turns features into class scores

Hidden state, in plain words:
- it is the model's running memory while it reads the sequence
- each new token updates that memory
- the final hidden state is a compact summary of the sequence

Padding and sequence length, in plain words:
- padding makes all examples in a batch the same length
- padded tokens are fake tokens added only for batching
- `lengths` tells you where the real text stops
- a clear default path is to use `pack_padded_sequence(..., enforce_sorted=False)` so the LSTM can ignore padding more cleanly

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
A small correct LSTM text classifier on AG News should often reach about `0.83` to `0.91` validation accuracy.

COMMON MISTAKES
- forgetting that token ids must be `torch.long`
- sending padded tokens through the model without using lengths clearly
- taking the wrong hidden state from the LSTM output
- applying softmax before `CrossEntropyLoss`
- mixing up validation accuracy and test accuracy

STEP-BY-STEP TODO
1. Reuse the AG News text split.
2. Reuse the training-text vocabulary.
3. Build padded token-id batches.
4. Build an embedding layer.
5. Build an RNN or LSTM.
6. Take a clean sequence summary vector.
7. Add the classifier head.
8. Train for several epochs.
9. Evaluate on validation data.
10. Check whether the metrics make sense.

STRETCH GOALS
- compare `nn.RNN` and `nn.LSTM`
- compare using the final hidden state against mean pooling over outputs
- inspect a few long and short examples

DEFINITION OF DONE
You are done when:
- your model returns raw logits with shape `[batch_size, 4]`
- you can explain what the hidden state means in simple words
- you can explain why lengths matter when padding is present
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

class SequenceModelConfig:
    """Settings for the RNN/LSTM text exercise."""

    dataset_name: str = "ag_news"
    num_classes: int = 4
    max_vocab_size: int = 20000
    max_length: int = 50
    embedding_dim: int = 96
    hidden_dim: int = 128
    num_layers: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 6
    random_state: int = 42
    target_val_accuracy: float = 0.84
    default_sequence_model: str = "lstm"




def describe_sequence_shapes(config: SequenceModelConfig) -> dict[str, object]:
    """Return the expected sequence-model shapes.

    Return a dict with exactly these keys:
    - `default_split`
    - `input_shapes`
    - `hidden_state_shape`
    - `output_shape`
    - `notes`
    """
    return {
        "default_split": {"train": 0.60, "val": 0.20, "test": 0.20},
        "input_shapes": {
            "token_ids": (config.batch_size, config.max_length),
            "lengths": (config.batch_size,),
            "labels": (config.batch_size,),
        },
        "hidden_state_shape": (config.num_layers, config.batch_size, config.hidden_dim),
        "output_shape": (config.batch_size, config.num_classes),
        "notes": describe_text_batch_rules(),
    }


        #raise NotImplementedError("Run the sequence model and return raw class logits.")


# Load the raw text split and vocabulary pieces you need.
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
# - build `token_to_id` from the training text only
# - do not let validation or test text add new vocabulary items
# - keep one special pad token id so shorter sequences can be padded to one batch shape
def load_text_artifacts(config: SequenceModelConfig) -> dict[str, object]:

    data = hf_datasets.load_dataset("ag_news")
    train_data = data["train"]
    test_data = data["test"]

    train_list = list(train_data["text"])
    train_labels_list = list(train_data["label"])

    test_texts = list(test_data["text"])
    test_labels = list(test_data["label"])

    num_train = int(len(train_labels_list) * 0.8)
    train_texts = train_list[:num_train]
    train_labels = train_labels_list[:num_train]
    val_texts = train_list[num_train:]
    val_labels = train_labels_list[num_train:]

    return {"train_texts": train_texts,"train_labels": train_labels,
            "val_texts": val_texts,"val_labels": val_labels,
            "test_texts": test_texts,"test_labels": test_labels}


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocabulary, max_length):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.pad_id = 0
        self.unk_id = 1 #unkown word, OOV (out of vocabulary)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)

        token_ids = []
        c = 0

        words = [word.strip(string.punctuation) for word in text.split()]

        for word in words:
            if not word:
                continue
            if word in self.vocabulary:
                token_ids.append(self.vocabulary[word])
            else:
                token_ids.append(self.unk_id)

            c+=1
            if c==self.max_length:
                break

        length = len(token_ids)
        token_ids+=[self.pad_id] * (self.max_length - length)

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        length = torch.tensor(length, dtype = torch.long)

        return token_ids, length, label



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
# - create a small dataset class that stores texts, labels, the vocabulary, and `max_length`
# - in `__getitem__`, return:
#   - `token_ids`: padded or truncated ids
#   - `lengths`: number of real tokens before padding
#   - `labels`: class id
# - `lengths` matters because the LSTM should know where the real text stops
def load_dataloaders(config: SequenceModelConfig) -> dict[str, DataLoader]:

    data_dict = load_text_artifacts(config)

    vocabulary = {"[PAD]":0, "[UNK]":1}
    train_texts = data_dict["train_texts"]
    index = 2

    for texts in train_texts:
        words = [word.strip(string.punctuation) for word in texts.split()]
        distinct_words = set(words)
        for word in distinct_words:
            if not word:
                continue
            if word not in vocabulary:
                vocabulary[word] = index
                index+=1
            if index == config.max_vocab_size:
                break
        if index == config.max_vocab_size:
            break

    train_ds = TokenDataset(texts= data_dict["train_texts"], labels = data_dict["train_labels"], vocabulary=vocabulary, max_length=config.max_length)
    val_ds = TokenDataset(texts= data_dict["val_texts"], labels = data_dict["val_labels"], vocabulary=vocabulary, max_length=config.max_length)
    test_ds = TokenDataset(texts= data_dict["test_texts"], labels = data_dict["test_labels"], vocabulary=vocabulary, max_length=config.max_length)

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}



class SequenceTextClassifier(nn.Module):
    """Text classifier stub that uses token order."""

    def __init__(self, vocab_size: int, pad_id: int, config: SequenceModelConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.config = config
        # Beginner-friendly build order:
        # 1. embedding layer
        # 2. one sequence model
        #    - recommended: `nn.LSTM`
        # 3. classifier head
        #
        # Simple default path:
        # - `nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)`
        # - `nn.LSTM(embedding_dim, hidden_dim, batch_first=True)`
        # - `nn.Linear(hidden_dim, num_classes)`

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(input_size=config.embedding_dim, hidden_size=config.hidden_dim, num_layers=config.num_layers, batch_first=True)
        self.classification_head = nn.Linear(in_features=config.hidden_dim, out_features=config.num_classes)

        # raise NotImplementedError(
        #     "Build the embedding layer, one LSTM or RNN, and the classifier head. Return raw logits."
        # )

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`.

        Clear default path:
        - embed the token ids
        - pack the sequence using `lengths`
            - without packing, the LSTM reads all 50 positions for every example
            - with packing, if one example really has length 7, the LSTM only reads 7 steps for that example
              it ignores the padded part at the end
            Why this helps:
            - the final hidden state better matches the real text
            - padding does not keep changing the sequence memory
            - training is usually cleaner
        - run the LSTM or RNN
        - use the final hidden state from the last layer
        - return raw logits

        Shape idea:
        - token ids: `[batch_size, max_length]`
        - embeddings: `[batch_size, max_length, embedding_dim]`
        - final hidden state from one-layer LSTM:
          `[1, batch_size, hidden_dim]`
        - after taking the last layer: `[batch_size, hidden_dim]`
        - logits: `[batch_size, num_classes]`
        """

        # token_ids shape is (32, 40)
        # lengths: shape [batch_size]

        embeddings = self.embed(token_ids)

        packed = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed)
        final_hidden = hidden[-1]

        logits = self.classification_head(final_hidden)

        return logits


    #raise NotImplementedError("Return the AG News split plus the training-text vocabulary and pad token id.")


import string



# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.CrossEntropyLoss`.
def build_training_objects(config: SequenceModelConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    # Beginner-friendly object list:
    # - model: `SequenceTextClassifier(...)`
    # - loss_fn: `nn.CrossEntropyLoss()`
    # - optimizer: something simple like `torch.optim.Adam(...)`

    model = SequenceTextClassifier(vocab_size = config.max_vocab_size, pad_id=0, config=config)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(lr = config.learning_rate, params = model.parameters())

    return model, loss_fn, optimizer

    #raise NotImplementedError("Return the sequence model, CrossEntropyLoss, and optimizer.")


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
# - if you use `pack_padded_sequence`, a simple safe path is `lengths.cpu()`
# - do not return one value per batch
# - do not return a history list
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:

    model.train()

    losses, accs = [],[]
    for batch_token_ids, lengths, targets in loader:

        batch_token_ids = batch_token_ids.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(batch_token_ids, lengths)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = logits.argmax(dim=1).float()
        correct = (preds == targets).float()
        accs.append(correct.mean().item())


    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss":mean_loss, "accuracy":mean_acc}

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
    for batch_token_ids, lengths, targets in loader:

        batch_token_ids = batch_token_ids.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        logits = model(batch_token_ids, lengths)
        loss = loss_fn(logits, targets)
        losses.append(loss.item())

        preds = logits.argmax(dim=1).float()
        correct = (preds == targets).float()
        accs.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss": mean_loss, "accuracy":mean_acc}
    
    # raise NotImplementedError("Return the mean loss and mean accuracy for the loader you evaluate.")


# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `train_accuracy`
# - `val_loss`
# - `val_accuracy`
# Return type:
# - `dict[str, float]`
# This is the final epoch summary, not a history list.
def run_training(config: SequenceModelConfig) -> dict[str, float]:

    torch.manual_seed(config.random_state)

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    train_dict = {}
    val_dict = {}

    for i in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i} -> loss: {train_dict['loss']}, acc: {train_dict['accuracy']}, val_loss: {val_dict['loss']}, val_acc: {val_dict['accuracy']}")

    test_dict = evaluate(model, loaders["test"], loss_fn)
    print(f"Test results -> loss: {test_dict['loss']}, accuracy: {test_dict['accuracy']}")

    return {"train_loss": train_dict["loss"], "train_accuracy": train_dict["accuracy"],
            "val_loss": val_dict["loss"], "val_accuracy": val_dict["accuracy"]}

    #raise NotImplementedError("Run the full RNN/LSTM training flow and return the final train and val metrics.")



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
    """Run the sequence-model scaffold one step at a time."""
    config = SequenceModelConfig()
    set_seed(config.random_state)
    print("Running exercise 02 starter scaffold...")
    print("Main goal: classify text with an LSTM or RNN that reads tokens in order.")
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

    _, ok = report_task_progress("describe_sequence_shapes", describe_sequence_shapes, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress(
        "SequenceTextClassifier.__init__",
        SequenceTextClassifier,
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
    _, ok = report_task_progress("SequenceTextClassifier.forward", model, sample_token_ids, sample_lengths)
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


