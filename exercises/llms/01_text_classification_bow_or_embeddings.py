"""
============================================================
EXERCISE 01: TEXT CLASSIFICATION BOW OR EMBEDDINGS
============================================================

WHAT YOU ARE BUILDING
You are building your first text classifier in this bootcamp.

You will take short news headlines and descriptions, turn them into numbers, and predict the news class.

This scaffold gives you one simple default path:
- turn each token id into a learned vector
- ignore padding and average only the real token vectors
- use that average vector as one summary of the text
- pass that summary into a small classifier

An embedding is a learned list of numbers for each token.
You can think of it as a compact numeric meaning for a word or subword.
Words used in similar ways can learn similar embeddings.

If you want, you may replace that with a bag-of-words baseline instead.
A bag-of-words model counts which words appear in the text and how many times they appear.
It does not care about word order.
For example, it treats "dog bites man" and "man bites dog" as the same set of word counts.
This is different from embeddings:
- bag-of-words keeps simple word counts
- embeddings give each token its own learned vector
- bag-of-words throws away order right away
- embeddings let you build richer text representations before the final classifier
But keep the same function names and return values.

WHY THIS EXERCISE EXISTS
Text data is different from image data and vector data.

Before a model can use text, you need to:
- split text into tokens
- build a vocabulary
- turn each token into an integer id
- pad shorter examples so a batch has one shape

This exercise is here to make those steps feel normal instead of mysterious.

DATASET
Use the `ag_news` dataset from Hugging Face `datasets`.

It has 4 classes:
- World
- Sports
- Business
- Sci/Tech

To keep one consistent split across this repo, use this default:
- combine the original train and test splits
- make your own `60% train / 20% val / 20% test` split

HOW TO LOAD OR DOWNLOAD THE DATASET
Simple path:
- install the Hugging Face `datasets` package
  - if you use pip: run `pip install datasets`
  - if you use conda: run `conda install -c conda-forge datasets`
  - if `pip` does not point to your Python, try `python -m pip install datasets`
- call `load_dataset("ag_news")`
- combine the train and test text examples
- make one shared `60 / 20 / 20` split

If you are offline, you may temporarily replace it with a tiny local toy dataset just to debug shapes.
But the real exercise should use AG News.

INPUTS AND OUTPUTS
Text batch, embedding path:
- `token_ids`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size, max_length]`
- `lengths`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`
  - meaning: number of real tokens before padding for each example
- `labels`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`
  - values: `0, 1, 2, 3`

One example batch:
- `token_ids`: `[32, 40]`
  - this means 32 examples in the batch
  - each example is stored as up to 40 token ids
- `lengths`: `[32]`
  - this means one length value for each of the 32 examples
- `labels`: `[32]`
  - this means one class label for each of the 32 examples

Model output:
- type: `torch.Tensor`
- shape: `[batch_size, 4]`
- meaning: one raw logit per class

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Simple default approach:
- turn each token id into a learned vector with `nn.Embedding`
- ignore the padding tokens and average only the real token vectors
- use that average vector as one summary for the whole text
- pass that summary into a small classifier to predict the class
- return raw logits

If you choose bag-of-words instead:
- bag-of-words means one large count vector for the whole text
- each position in that vector matches one word in the vocabulary
- the value says how many times that word appeared
- build a count vector of shape `[batch_size, vocab_size]`
- feed it into a small classifier
- still return raw logits of shape `[batch_size, 4]`

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
A small correct baseline on AG News should usually reach about `0.80` to `0.90` validation accuracy.

You do not need a fancy model to get a useful result here.

COMMON MISTAKES
- using float token ids instead of integer token ids
- forgetting that `nn.Embedding` expects `torch.long` ids
- averaging across padding tokens as if they were real words
- applying softmax before `CrossEntropyLoss`
- returning probabilities instead of raw logits
- mixing up train accuracy and validation accuracy

STEP-BY-STEP TODO
1. Load AG News and make one shared `60 / 20 / 20` split.
2. Tokenize the text with a simple beginner-friendly tokenizer.
3. Build a vocabulary from the training text only.
4. Turn each text into token ids.
5. Pad or truncate each example to one fixed `max_length`.
6. Build train, val, and test dataloaders.
7. Build the text classifier.
8. Train for several epochs.
9. Evaluate on validation data.
10. Check whether validation accuracy reaches the target range.

STRETCH GOALS
- compare mean pooling against max pooling
- compare bag-of-words against embeddings
- print a few wrong predictions and read the text

DEFINITION OF DONE
You are done when:
- your text batches have the shapes written above
- your model returns raw logits with shape `[batch_size, 4]`
- you can explain vocabulary, padding, and token ids in plain words
- your validation accuracy is in a sensible range
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
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

class TextClassificationConfig:
    """Settings for the beginner text-classification exercise."""

    dataset_name: str = "ag_news"
    text_column: str = "text"
    num_classes: int = 4
    max_vocab_size: int = 20000
    max_length: int = 40
    embedding_dim: int = 64
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 6
    random_state: int = 42
    target_val_accuracy: float = 0.82
    default_model_kind: str = "embedding"




def describe_expected_pipeline(config: TextClassificationConfig) -> dict[str, object]:
    """Return the exact beginner-friendly pipeline summary.

    Return a dict with exactly these keys:
    - `dataset_name`
    - `default_split`
    - `default_model_kind`
    - `batch_example_shapes`
    - `notes`
    """
    return {
        "dataset_name": config.dataset_name,
        "default_split": {"train": 0.60, "val": 0.20, "test": 0.20},
        "default_model_kind": config.default_model_kind,
        "batch_example_shapes": {
            "token_ids": (config.batch_size, config.max_length),
            "lengths": (config.batch_size,),
            "labels": (config.batch_size,),
            "logits": (config.batch_size, config.num_classes),
        },
        "notes": describe_text_batch_rules(),
    }

        #raise NotImplementedError("Embed the tokens, pool across real tokens, and return raw logits.")


# Load the raw text dataset and build one shared split.
# Return a dict with exactly these keys:
# - `train_texts`: list[str]
# - `val_texts`: list[str]
# - `test_texts`: list[str]
# - `train_labels`: list[int]
# - `val_labels`: list[int]
# - `test_labels`: list[int]
# Use one shared 60% train / 20% val / 20% test split.
# Beginner-friendly path:
# - call `load_dataset("ag_news")`
# - read the original train split texts and labels
# - read the original test split texts and labels
# - combine them into one text list and one label list
# - make one shared 60/20/20 split
# - if your split helper wants counts instead of fractions, compute integer sizes first
def load_text_splits(config: TextClassificationConfig) -> dict[str, object]:

    data = hf_datasets.load_dataset("ag_news")
    train_data = data["train"]
    test_data = data["test"]

    train_list = list(train_data["text"])
    train_labels_list = list(train_data["label"])
    
    test_texts = list(test_data["text"])
    test_labels = list(test_data["label"])

    num_train = int(len(train_labels_list)*0.8)
    train_texts = train_list[:num_train]
    train_labels = train_labels_list[:num_train]
    val_texts = train_list[num_train:]
    val_labels = train_labels_list[num_train:]

    return {"train_texts": train_texts,"train_labels": train_labels,
            "val_texts": val_texts,"val_labels": val_labels,
            "test_texts": test_texts,"test_labels": test_labels}


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list, labels: list, vocabulary: dict, max_length=40):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.pad_id = 0
        self.unk_id = 1  #Unkown word, OOV (out of vocabulary)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> dict:
        text = self.texts[index]
        label = self.labels[index]

        token_ids = []
        c = 0

        words = [word.strip(string.punctuation) for word in text.split()]
        for word in words:
            if not word: # if word is an emty string ""
                continue
            if word in self.vocabulary:
                token_id = self.vocabulary[word]
                token_ids.append(token_id)
            else:
                token_ids.append(self.unk_id)

            c+=1
            if c == self.max_length:
                break
        
        length = len(token_ids)                         #length of actual text before padding
        token_ids+= [self.pad_id] * (self.max_length - length)    #padidng with 0's if length is less than max_length
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return {"token_ids": token_ids, "lengths": length, "labels": label}



@dataclass

def load_dataloaders(config: TextClassificationConfig) -> dict[str, DataLoader]:

    data_dicts = load_text_splits(config)

    #build vocabulary from train split:
    #lets assign zero as 
    vocabulary = {"[PAD]":0, "[UNK]":1}  #Unknown word, #out of vocabulary (OOV)
    train_texts = data_dicts["train_texts"]
    index=2
    for text in train_texts:
        words = [word.strip(string.punctuation) for word in text.split()]
        distinct_words = set(words)
        for word in distinct_words:
            if not word: # if word is an emty string ""
                continue
            if word not in vocabulary:
                vocabulary[word]=index
                index+=1
                if index==config.max_vocab_size:
                    break
        if index==config.max_vocab_size:
            break

    train_ds = TokenDataset(texts= data_dicts["train_texts"], labels = data_dicts["train_labels"], vocabulary=vocabulary, max_length=config.max_length)
    val_ds = TokenDataset(texts= data_dicts["val_texts"], labels = data_dicts["val_labels"], vocabulary=vocabulary, max_length=config.max_length)
    test_ds = TokenDataset(texts= data_dicts["test_texts"], labels = data_dicts["test_labels"], vocabulary=vocabulary, max_length=config.max_length)

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}



class SimpleTextClassifier(nn.Module):
    """Beginner text classifier stub.

    Default path:
    - embedding layer
    - mean pool over real tokens only
    - small classifier head
    """

    def __init__(self, vocab_size: int, config: TextClassificationConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config
        # Beginner-friendly build order:
        # 1. create an embedding layer
        #    - maps token ids to vectors
        # 2. create a classifier head
        #    - simple path: one linear layer
        #    - slightly bigger path: `Linear -> ReLU -> Linear`
        #
        # What the embedding layer does:
        # - each integer token id becomes one learned vector
        # - shape changes from `[batch_size, max_length]`
        #   to `[batch_size, max_length, embedding_dim]`

        # The embedding layer is learnable. PyTorch creates a weight matrix that the model learns during training.
        # Its shape is: [vocab_size, embedding_dim]
        # That means one row for each token id, each row is that token's vector
        # if vocab_size = 20000 and embedding_dim = 64, then the embedding table has shape: [20000, 64]
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.embedding_dim, padding_idx=0)
        self.classifier_head = nn.Linear(in_features=config.embedding_dim, out_features=config.num_classes)

        # raise NotImplementedError(
        #     "Build the default embedding-based text classifier. Return raw logits with shape [batch_size, num_classes]."
        # )

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return raw class logits.

        Inputs:
        - `token_ids`: tensor with shape `[batch_size, max_length]`
        - `lengths`: tensor with shape `[batch_size]`

        Return:
        - type: `torch.Tensor`
        - shape: `[batch_size, num_classes]`
        - meaning: raw class logits for AG News

        Step-by-step tensor path:
        1. embed the token ids
           - shape becomes `[batch_size, max_length, embedding_dim]`
        2. make sure padding tokens do not count like real words
        3. pool across the sequence dimension
           - simple default: mean over real tokens only
           - result shape becomes `[batch_size, embedding_dim]`
        4. pass the pooled vector into the classifier head
           - final logits shape becomes `[batch_size, num_classes]`
        """
        # token_ids shape is (32, 40)
        embeddings = self.embed(token_ids)      # [batch_size, max_length, embedding_dim]  (32, 40, 64)
        # You want to average only the token embeddings that come from real words, and ignore the padding embeddings.
        # Having pad token id 0 does not automatically mean its embedding vector is all zeros.
        # token_ids contains 0 at padding positions and nn.Embedding looks up row 0 in the embedding table
        # row 0 is just another learned vector unless you handle it specially
        # so we create a mask 
        # sum across the sequence dimension 
        # divide by the number of real tokens, not by max_length

        positions = torch.arange(token_ids.size(1), device=token_ids.device)            # (40)
        # shape: [max_length]

        # Over here, positions is shape 40 and lengths is shape 32
        # When you unsqueeze positions on dim 0, it becomes shape (1, 40)
        # When you unsqueeze lengths on dim 1, it becomes shape (32, 1)
        # Now PyTorch compares: [1, 40] < [32, 1],  These do not match exactly, so PyTorch uses broadcasting.
        # [1, 40] becomes [32, 40] by copying the same row 32 times
        # [32, 1] becomes [32, 40] by copying each length across 40 columns
        # Each row compares all positions 0..39 against one example's length.
        # For example, if one length is 5, then that row becomes:
        # [0, 1, 2, 3, 4, 5, 6, ...] < 5
        # [True, True, True, True, True, False, False, ..., False]
        # That means positions 0 to 4 are real tokens and positions 5 onward are padding
        # Basically, for each example in the batch, mark which token positions are before the real length.
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        # shape: [batch_size, max_length]
        # True for real tokens, False for padding

        mask = mask.unsqueeze(-1)                           #(32, 40, 1)
        # shape: [batch_size, max_length, 1]

        masked_embeddings = embeddings * mask               #(32, 40, 64) * (32, 40, 1) = (32, 40, 64)
        # padding positions become 0        

        summed = masked_embeddings.sum(dim=1)               #(32, 64)
        # [batch_size, embedding_dim] 

        pooled = summed / lengths.unsqueeze(1)              #(32, 64) / (32, 1)
        logits = self.classifier_head(pooled)

        return logits


    #raise NotImplementedError("Load AG News, combine the original splits, and return one shared 60/20/20 split.")


# Build the train, validation, and test dataloaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Each batch should be a dict with exactly these keys:
# - `token_ids`
# - `lengths`
# - `labels`
# Batch value meaning:
# - `token_ids` is a tensor of shape `[batch_size, max_length]`
# - `lengths` is a tensor of shape `[batch_size]`
# - `labels` is a tensor of shape `[batch_size]`
# Beginner-friendly data path after you have the raw split:
# - build the vocabulary from `train_texts` only
# - create a small dataset class that stores texts, labels, the vocabulary, and `max_length`
# - in `__getitem__`:
#   - tokenize one text
#   - map each token to an integer id
#   - truncate if it is too long
#   - save the real token count as `length`
#   - pad with the pad id until it reaches `max_length`
#   - return one dict with `token_ids`, `lengths`, and `labels`

import string


    #raise NotImplementedError("Return train, val, and test text dataloaders with padded token-id batches.")


# Build the three training objects.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.CrossEntropyLoss`.
def build_training_objects(config: TextClassificationConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    # Beginner-friendly object list:
    # - model: `SimpleTextClassifier(...)`
    # - loss_fn: `nn.CrossEntropyLoss()`
    # - optimizer: something simple like `torch.optim.Adam(...)`

    model = SimpleTextClassifier(vocab_size=config.max_vocab_size, config=config)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())

    return model, loss_fn, optimizer

    #raise NotImplementedError("Return the text classifier, CrossEntropyLoss, and optimizer.")


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
# - do not return one value per batch
# - do not return a history list
# - return one final epoch summary dict
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    # Training loop order:
    # - get one batch dict
    # - unpack `token_ids`, `lengths`, and `labels`
    # - move them to the same device as the model
    # - zero the gradients
    # - run the model forward
    # - compute the loss
    # - backpropagate
    # - step the optimizer
    # - track total loss and total correct predictions across the epoch

    losses, accs = [],[]

    model.train()
    
    for batch_dict in loader:
        tokens_batch = batch_dict["token_ids"]
        lengths = batch_dict["lengths"]
        targets = batch_dict["labels"]

        tokens_batch = tokens_batch.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(tokens_batch, lengths)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = logits.argmax(dim=1).float()
        correct = (preds == targets).float()
        accs.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss": mean_loss, "accuracy":mean_acc}
    #raise NotImplementedError("Return the mean training loss and mean training accuracy for one full epoch.")


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
# Important:
# - move the model, `token_ids`, `lengths`, and `labels` to the same device here
# - do not return one value per sample
# - do not return a history list
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    losses, accs = [],[]

    model.eval()
    
    for batch_dict in loader:
        tokens_batch = batch_dict["token_ids"]
        lengths = batch_dict["lengths"]
        targets = batch_dict["labels"]

        tokens_batch = tokens_batch.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        logits = model(tokens_batch, lengths)
        loss = loss_fn(logits, targets)
        losses.append(loss.item())

        preds = logits.argmax(dim=1).float()
        correct = (preds == targets).float()
        accs.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss": mean_loss, "accuracy":mean_acc}

    #raise NotImplementedError("Return the mean loss and mean accuracy for the loader you evaluate.")


# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `train_accuracy`
# - `val_loss`
# - `val_accuracy`
# Return type:
# - `dict[str, float]`
# All four values should be Python floats.
# Important:
# - this function should return the final epoch summary only
# - it should not return a history list unless you add that as an extra value yourself
# - `train_loss` means final train epoch mean loss
# - `train_accuracy` means final train epoch mean accuracy
# - `val_loss` means final validation loader mean loss
# - `val_accuracy` means final validation loader mean accuracy
def run_training(config: TextClassificationConfig) -> dict[str, float]:

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
    #raise NotImplementedError("Run the full text-classification training flow and return the final train and val metrics.")



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
    """Run the text-classification scaffold one step at a time."""
    config = TextClassificationConfig()
    set_seed(config.random_state)
    print("Running exercise 01 starter scaffold...")
    print("Main goal: turn raw text into padded token-id batches and classify the text.")
    print(f"Device note: keep the model, token_ids, lengths, and labels on the same device. Using {device}.")

    completed_steps = 0
    total_steps = 9

    text_splits, ok = report_task_progress("load_text_splits", load_text_splits, config)
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

    _, ok = report_task_progress("describe_expected_pipeline", describe_expected_pipeline, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("SimpleTextClassifier.__init__", SimpleTextClassifier, config.max_vocab_size, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_token_ids = torch.zeros(config.batch_size, config.max_length, dtype=torch.long)
    sample_lengths = torch.full((config.batch_size,), config.max_length, dtype=torch.long)
    _, ok = report_task_progress("SimpleTextClassifier.forward", model, sample_token_ids, sample_lengths)
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
