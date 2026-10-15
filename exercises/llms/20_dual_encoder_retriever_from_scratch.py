"""
============================================================
EXERCISE 20: DUAL ENCODER RETRIEVER FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building a tiny trainable retriever from scratch.

Plain-language meanings:
- a document encoder turns a document into a vector
- a query encoder turns a query into a vector
- a dual encoder uses two encoders, one for queries and one for documents
- contrastive loss teaches the matching query and document to be close
- in-batch negatives means other documents in the same batch act as wrong answers

WHY THIS EXERCISE EXISTS
the related exercise builds retrieval with simple embeddings.
This part teaches how a retriever can learn its embeddings.

HOW THIS CONNECTS TO RAG AND VECTOR DATABASES
This is the small, from-scratch version of the idea behind vector search in RAG.

In a RAG system:
- documents are split into chunks
- each chunk is turned into a vector
- those vectors are stored in a vector database
- the user question is turned into a vector
- the vector database finds document chunks with vectors close to the question vector
- those chunks are sent to the language model as extra context

This exercise focuses on the model part:
- the query encoder turns the question into a vector
- the document encoder turns the document into a vector
- the similarity score checks how well the query and document match

A vector database is the storage and fast search part.
The dual encoder is what creates useful vectors to store and search.

Small example:
- query: `How do I make a DataLoader?`
- correct document: `A DataLoader groups Dataset items into batches.`
- wrong document: `A convolution scans an image with a small filter.`

The model should give the query a higher score with the correct document.

DATASET
Use the local dummy dataset:
- `data/dual_encoder_retriever_pairs.json`

The JSON file has:
- `pair_count`: the number of pairs
- `pairs`: a list of dictionaries

Each item in `pairs` looks like:
- `{"query": "what is dataloader", "document": "DataLoader groups Dataset items into batches for training."}`

INPUTS AND OUTPUTS
Batch fields:
- `query_ids`: `[batch_size, max_length]`
- `document_ids`: `[batch_size, max_length]`
- `query_lengths`: `[batch_size]`
- `document_lengths`: `[batch_size]`

Similarity matrix:
- shape: `[batch_size, batch_size]`
- row `i` is query `i`
- column `j` is document `j`
- the correct match is on the diagonal, where `i == j`

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

For a batch of 4 matching pairs:
- similarity matrix shape is `[4, 4]`
- labels are `[0, 1, 2, 3]`
- label `2` means query row 2 should pick document column 2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import string
import sys
import tokenize
from typing import TypedDict
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    """Set the main random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a short result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")


@dataclass
class DualEncoderRetrieverConfig:
    """Settings for the tiny dual-encoder retriever."""

    max_length: int = 18
    embedding_dim: int = 48
    hidden_dim: int = 64
    batch_size: int = 8
    learning_rate: float = 2e-3
    epochs: int = 20
    random_state: int = 42
    target_top1_accuracy: float = 0.80
    max_vocab_size = 2000


class QueryDocumentPair(TypedDict):
    """One matching query and document pair."""

    query: str
    document: str


class RetrieverBatch(TypedDict):
    """One encoded query/document batch item."""

    query_ids: torch.Tensor
    document_ids: torch.Tensor
    query_length: torch.Tensor
    document_length: torch.Tensor


class RetrieverMetrics(TypedDict):
    """Metrics returned by train and eval loops."""

    loss: float
    accuracy: float


class RetrieverRunSummary(TypedDict):
    """Final metrics returned by the full workflow."""

    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float
    retrieval_note: str


def build_tiny_pairs() -> list[QueryDocumentPair]:
    """Return query/document training pairs from the local JSON file."""
    # Goal:
    # Load matching query/document pairs from the local JSON dataset.
    #
    # Input:
    # - No arguments.
    #
    # Output:
    # - `list[QueryDocumentPair]`
    # - each dictionary has exactly:
    #   - `query`: one search question
    #   - `document`: the matching note
    #
    # Small example:
    # `{"query": "what is dataloader", "document": "DataLoader groups Dataset items into batches for training."}`
    #
    # Method:
    # Load this file:
    # `BOOTCAMP_ROOT / "data" / "dual_encoder_retriever_pairs.json"`
    #
    # Use:
    # `with dataset_path.open("r", encoding="utf-8") as file:`
    # `    payload = json.load(file)`
    #
    # The JSON object has a `pairs` key.
    # `payload["pairs"]` is the list you should return.
    #
    # Steps:
    # 1. Build `dataset_path`.
    # 2. Open the JSON file with UTF-8 text encoding.
    # 3. Load it with `json.load(file)`.
    # 4. Read `pairs = payload["pairs"]`.
    # 5. Return `pairs`.
    #
    # Checks:
    # - `len(pairs) >= 500`
    # - `set(pairs[0]) == {"query", "document"}`
    # - `pairs[0]["query"]` and `pairs[0]["document"]` should be non-empty strings.

    raise NotImplementedError("Return tiny query/document pairs.")




def simple_tokenize(text: str) -> list[str]:
    """Split text into simple lowercase word tokens."""
    # Goal:
    # Turn a short text string into word tokens.
    #
    # Input:
    # - `text`: string like `"How do DataLoaders make batches?"`
    #
    # Output:
    # - list of lowercase tokens like `["how", "do", "dataloaders", "make", "batches"]`
    #
    # Method:
    # Use `text.lower().split()` and remove punctuation with `string.punctuation`.
    #
    # Steps:
    # 1. Lowercase the text.
    # 2. Split on spaces.
    # 3. Strip punctuation from each word.
    # 4. Skip empty tokens.
    #
    # Check:
    # - `simple_tokenize("Hello!")` should return `["hello"]`.

    raise NotImplementedError("Tokenize text into lowercase words.")





def build_vocab(pairs: list[QueryDocumentPair], max_vocab_size: int = 2000) -> dict[str, int]:
    """Build a vocabulary from queries and documents."""
    # Goal:
    # Build the word-to-id dictionary used by both encoders.
    #
    # Input:
    # - `pairs`: list of dictionaries with `query` and `document`
    # - `max_vocab_size`: maximum number of tokens to keep
    #
    # Output:
    # - `dict[str, int]`
    # - must include:
    #   - `"[PAD]": 0`
    #   - `"[UNK]": 1`
    #
    # Required special ids:
    # - `[PAD]`: 0
    # - `[UNK]`: 1
    #
    # Method:
    # Use `simple_tokenize(text)` for both queries and documents.
    # Add tokens in a stable order so runs are repeatable.
    #
    # Steps:
    # 1. Start with `{"[PAD]": 0, "[UNK]": 1}`.
    # 2. Tokenize every query and every document.
    # 3. Collect normal tokens in a list or set.
    # 4. Sort the tokens with `sorted(tokens)` if you used a set.
    # 5. Add each token until `len(vocab) == max_vocab_size`.
    # 6. Return the vocabulary.
    #
    # Checks:
    # - `vocab["[PAD]"] == 0`
    # - `vocab["[UNK]"] == 1`
    # - A known token from the pairs should have an integer id.

    raise NotImplementedError("Build a shared query/document vocabulary.")






def encode_text(text: str, vocab: dict[str, int], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded token ids and the real length."""
    # Goal:
    # Convert one text string into fixed-length token ids.
    #
    # Input:
    # - `text`: one query or document string
    # - `vocab`: dictionary from token string to integer id
    # - `max_length`: fixed output length
    #
    # Output:
    # - token ids tensor with shape `[max_length]`
    # - length tensor with shape `[]`
    #
    # Small example:
    # - text: `"make batches"`
    # - max_length: 5
    # - ids after padding: `[7, 9, 0, 0, 0]`
    # - length: `2`
    #
    # Steps:
    # 1. Tokenize the text.
    # 2. Convert each token to an id.
    # 3. Use `[UNK]` id for missing tokens.
    # 4. Truncate to `max_length`.
    # 5. Pad with `[PAD]` id until length is `max_length`.
    # 6. Build `ids_tensor = torch.tensor(ids, dtype=torch.long)`.
    # 7. Build `length_tensor = torch.tensor(real_length, dtype=torch.long)`.
    # 8. Return `(ids_tensor, length_tensor)`.
    #
    # Checks:
    # - `ids_tensor.shape == (max_length,)`
    # - `length_tensor.shape == ()`
    # - The length should count real tokens before padding, after truncation.

    raise NotImplementedError("Encode text as padded token ids and length.")






class QueryDocumentPairDataset(Dataset):
    """Dataset of matching query/document pairs."""

    def __init__(self, pairs: list[QueryDocumentPair], vocab: dict[str, int], max_length: int) -> None:
        self.pairs = pairs
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> RetrieverBatch:
        # Goal:
        # Return one encoded query/document pair.
        #
        # Input:
        # - `index`: which pair to read
        #
        # Output keys:
        # - `query_ids`: `[max_length]`
        # - `document_ids`: `[max_length]`
        # - `query_length`: scalar tensor
        # - `document_length`: scalar tensor
        #
        # Method:
        # Use `encode_text(text, self.vocab, self.max_length)` for the query
        # and again for the document.
        #
        # Steps:
        # 1. Read `pair = self.pairs[index]`.
        # 2. Encode `pair["query"]`.
        # 3. Encode `pair["document"]`.
        # 4. Return a dictionary with the exact keys listed above.
        #
        # Checks:
        # - `item["query_ids"].shape == (self.max_length,)`
        # - `item["document_ids"].shape == (self.max_length,)`
        # - `item["query_length"].shape == ()`

        raise NotImplementedError("Return one encoded query/document pair.")
        


class MeanPoolTextEncoder(nn.Module):
    """Tiny text encoder with embeddings and mean pooling."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pad_id: int = 0) -> None:
        # Goal:
        # Build a small encoder that turns token ids into one vector.
        #
        # Input:
        # - `vocab_size`: number of token ids
        # - `embedding_dim`: size of each token vector
        # - `hidden_dim`: size of the final text vector
        # - `pad_id`: token id for padding, usually `0`
        #
        # Layers:
        # - `nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)`
        # - `nn.Linear(embedding_dim, hidden_dim)`
        super().__init__()
        #
        # Method:
        # Store:
        # - `self.pad_id = pad_id`
        # - `self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)`
        # - `self.projection = nn.Linear(embedding_dim, hidden_dim)`
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Save `pad_id`.
        # 3. Build the embedding layer.
        # 4. Build the projection layer.
        #
        # Checks:
        # - `self.embedding.padding_idx == pad_id`
        # - The projection output size should be `hidden_dim`.

        raise NotImplementedError("Build the mean-pooling text encoder.")




    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return one embedding per text."""
        # Goal:
        # Turn `[batch_size, max_length]` token ids into `[batch_size, hidden_dim]`.
        #
        # Input:
        # - `token_ids`: tensor with shape `[batch_size, max_length]`
        # - `lengths`: tensor with shape `[batch_size]`
        #
        # Output:
        # - normalized text embeddings with shape `[batch_size, hidden_dim]`
        #
        # Method:
        # Mean pooling means:
        # 1. embed every token
        # 2. ignore padding embeddings
        # 3. average the real token embeddings
        #
        # Steps:
        # 1. Embed token ids.
        # 2. Build a padding mask from `token_ids != pad_id`.
        # 3. Sum only real token embeddings.
        # 4. Divide by the real lengths.
        # 5. Project to hidden_dim.
        # 6. Normalize with `torch.nn.functional.normalize(projected, p=2, dim=1)`.
        #
        # Shape example:
        # - token ids: `[8, 18]`
        # - embedded tokens: `[8, 18, 48]`
        # - pooled text vectors: `[8, 48]`
        # - projected vectors: `[8, 64]`
        #
        # Checks:
        # - Output shape should be `[batch_size, hidden_dim]`.
        # - Padding tokens should not change the mean.
        # - Non-zero output rows should have length close to `1`.

        raise NotImplementedError("Encode token ids into normalized text embeddings.")









class DualEncoderRetriever(nn.Module):
    """Two encoders: one for queries and one for documents."""

    def __init__(self, vocab_size: int, config: DualEncoderRetrieverConfig, pad_id: int = 0) -> None:
        # Goal:
        # Build separate query and document encoders.
        super().__init__()
        #
        # Input:
        # - `vocab_size`: number of token ids
        # - `config.embedding_dim`: token embedding size
        # - `config.hidden_dim`: final query/document embedding size
        # - `pad_id`: padding token id
        #
        # Method:
        # Create two `MeanPoolTextEncoder` modules:
        # - `self.query_encoder = MeanPoolTextEncoder(vocab_size, config.embedding_dim, config.hidden_dim, pad_id)`
        # - `self.document_encoder = MeanPoolTextEncoder(vocab_size, config.embedding_dim, config.hidden_dim, pad_id)`
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Build the query encoder.
        # 3. Build the document encoder.
        #
        # Checks:
        # - The query and document encoders should be separate modules.
        # - They should return vectors with the same hidden size.

        raise NotImplementedError("Build query and document encoders.")




    def forward(
        self,
        query_ids: torch.Tensor,
        query_lengths: torch.Tensor,
        document_ids: torch.Tensor,
        document_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return query-document similarity scores."""
        # Goal:
        # Return the score for every query against every document in the batch.
        #
        # Output:
        # - scores: `[batch_size, batch_size]`
        #
        # Input shapes:
        # - `query_ids`: `[batch_size, max_length]`
        # - `query_lengths`: `[batch_size]`
        # - `document_ids`: `[batch_size, max_length]`
        # - `document_lengths`: `[batch_size]`
        #
        # Method:
        # Use:
        # `scores = query_embeddings @ document_embeddings.T`
        #
        # Shape example:
        # `[8, 64] @ [64, 8] -> [8, 8]`
        #
        # Steps:
        # 1. Encode queries with `self.query_encoder(query_ids, query_lengths)`.
        # 2. Encode documents with `self.document_encoder(document_ids, document_lengths)`.
        # 3. Compute `query_embeddings @ document_embeddings.T`.
        # 4. Return the score matrix.
        #
        # Checks:
        # - The output shape should be `[batch_size, batch_size]`.
        # - The diagonal is where the correct matches are.

        raise NotImplementedError("Return query-document score matrix.")


        # The encoders normalized these vectors, so this dot product is cosine similarity.
        # Example: if a query vector points in the same direction as a document vector,
        # its score is close to 1. If it points in a different direction, its score is lower.



def build_dataloaders(config: DualEncoderRetrieverConfig) -> tuple[dict[str, DataLoader], dict[str, int]]:
    """Build train, validation, and test dataloaders plus vocabulary."""
    # Goal:
    # Build everything needed to serve query/document batches.
    #
    # Input:
    # - `config`: DualEncoderRetrieverConfig
    #
    # Output:
    # - loaders dictionary with keys:
    #   - `"train"`
    #   - `"val"`
    #   - `"test"`
    # - vocabulary dictionary with token ids
    #
    # Each loader should yield a batch with:
    # - `query_ids`: `[batch_size, max_length]`
    # - `document_ids`: `[batch_size, max_length]`
    # - `query_length`: `[batch_size]`
    # - `document_length`: `[batch_size]`
    #
    # Method:
    # Use:
    # - `pairs = build_tiny_pairs()`
    #   This loads `data/dual_encoder_retriever_pairs.json`.
    # - `vocab = build_vocab(pairs)`
    # - `dataset = QueryDocumentPairDataset(pairs, vocab, config.max_length)`
    # - `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`
    #
    # Steps:
    # 1. Build pairs.
    # 2. Build the vocabulary from all pairs.
    # 3. Build one dataset.
    # 4. Split the dataset into train, validation, and test.
    # 5. Use `DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)` for train.
    # 6. Use `DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)` for validation.
    # 7. Use `DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)` for test.
    # 8. Return `({"train": train_loader, "val": val_loader, "test": test_loader}, vocab)`.
    #
    # Checks:
    # - `set(loaders) == {"train", "val", "test"}`
    # - One train batch should include `query_ids` and `document_ids`.
    # - `batch["query_ids"].shape[1] == config.max_length`

    raise NotImplementedError("Build retriever dataloaders and vocabulary.")





def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> RetrieverMetrics:
    """Train one epoch with in-batch negatives."""
    # Goal:
    # Teach each query to pick its matching document in the same batch.
    #
    # Loss setup:
    # - scores shape: `[batch_size, batch_size]`
    # - labels: `torch.arange(batch_size)`
    # - label `i` means query row `i` should pick document column `i`
    #
    # Input batch keys:
    # - `query_ids`: `[batch_size, max_length]`
    # - `document_ids`: `[batch_size, max_length]`
    # - `query_length`: `[batch_size]`
    # - `document_length`: `[batch_size]`
    #
    # Output:
    # - `{"loss": mean_loss, "accuracy": mean_accuracy}`
    #
    # Method:
    # Use `loss_fn(scores, labels)` where:
    # - `scores` is `[batch_size, batch_size]`
    # - `labels = torch.arange(batch_size, device=device)`
    #
    # Steps:
    # 1. Call `model.train()`.
    # 2. For each batch, move all tensors to `device`.
    # 3. Clear old gradients with `optimizer.zero_grad()`.
    # 4. Run the model to get `scores`.
    # 5. Build labels with `torch.arange(scores.size(0), device=device)`.
    # 6. Compute the loss.
    # 7. Call `loss.backward()`.
    # 8. Call `optimizer.step()`.
    # 9. Track average loss with `loss.item() * batch_size`.
    # 10. Track accuracy with `scores.argmax(dim=1) == labels`.
    # 11. Return the averaged metrics.
    #
    # Checks:
    # - `scores.shape == (batch_size, batch_size)`
    # - If the batch size is 4, `labels.tolist()` should be `[0, 1, 2, 3]`.
    # - Do not apply softmax before `nn.CrossEntropyLoss`.

    raise NotImplementedError("Train one dual-encoder retrieval epoch.")






@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> RetrieverMetrics:
    """Evaluate top-1 retrieval accuracy."""
    # Goal:
    # Measure whether each query retrieves its paired document.
    #
    # Output:
    # - `{"loss": mean_loss, "accuracy": mean_accuracy}`
    #
    # Method:
    # Use the same score matrix and labels as training:
    # - `scores`: `[batch_size, batch_size]`
    # - `labels = torch.arange(batch_size, device=device)`
    #
    # `@torch.no_grad()` is already on this function, so it should not update weights.
    #
    # Steps:
    # 1. Call `model.eval()`.
    # 2. For each batch, move tensors to `device`.
    # 3. Run the model to get `scores`.
    # 4. Build labels with `torch.arange(scores.size(0), device=device)`.
    # 5. Compute loss.
    # 6. Compute top-1 accuracy with `scores.argmax(dim=1)`.
    # 7. Track average loss and accuracy over all examples.
    # 8. Return the averaged metrics.
    #
    # Checks:
    # - This function should not call `backward()` or `optimizer.step()`.
    # - Accuracy is correct when the highest score is on the diagonal.


    raise NotImplementedError("Evaluate dual-encoder retrieval.")






def run_training(config: DualEncoderRetrieverConfig) -> RetrieverRunSummary:
    """Run the full dual-encoder retriever workflow."""
    # Goal:
    # Train the tiny dual encoder and return train, validation, and test metrics.
    #
    # Input:
    # - `config`: DualEncoderRetrieverConfig
    #
    # Output keys:
    # - `train_loss`
    # - `train_accuracy`
    # - `val_loss`
    # - `val_accuracy`
    # - `test_loss`
    # - `test_accuracy`
    # - `retrieval_note`
    #
    # Method:
    # Use the helper functions in this file.
    # Build the model with:
    # `DualEncoderRetriever(vocab_size=len(vocab), config=config, pad_id=vocab["[PAD]"])`
    #
    # Use:
    # - `loss_fn = nn.CrossEntropyLoss()`
    # - `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    #
    # Steps:
    # 1. Build dataloaders and vocabulary with `build_dataloaders(config)`.
    # 2. Build the model and move it to `device`.
    # 3. Build loss and optimizer.
    # 4. For each epoch, run `train_one_epoch(model, loaders["train"], optimizer, loss_fn)`.
    # 5. Evaluate on the validation loader each epoch.
    # 6. Evaluate once on the test loader at the end.
    # 7. Write a short retrieval note, such as:
    #    `"The model learns by making matching query/document pairs score highest."`
    # 8. Return the dictionary with the exact output keys listed above.
    #
    # Checks:
    # - `test_accuracy` should be a float between `0.0` and `1.0`.
    # - `retrieval_note` should explain in-batch negatives in plain language.

    raise NotImplementedError("Run the full dual-encoder retriever workflow.")












def check_metrics(metrics: dict[str, float | str], config: DualEncoderRetrieverConfig) -> bool:
    """Return True when test accuracy reaches the target."""
    return "test_accuracy" in metrics and float(metrics["test_accuracy"]) >= config.target_top1_accuracy


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
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the dual-encoder retriever scaffold."""
    config = DualEncoderRetrieverConfig()
    set_seed(config.random_state)

    pairs, ok = report_task_progress("build_tiny_pairs", build_tiny_pairs)
    if not ok:
        return
    vocab, ok = report_task_progress("build_vocab", build_vocab, pairs)
    if not ok:
        return
    loaders, ok = report_task_progress("build_dataloaders", build_dataloaders, config)
    if not ok:
        return
    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok and metrics is not None:
        passed = check_metrics(metrics, config)
        print_pass_or_try_again(passed, "test_accuracy", float(metrics["test_accuracy"]), f">= {config.target_top1_accuracy}")
    _ = vocab, loaders


if __name__ == "__main__":
    main()
