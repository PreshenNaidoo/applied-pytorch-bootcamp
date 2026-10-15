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

    dataset_path = Path("data/dual_encoder_retriever_pairs.json")
    if not dataset_path.exists():
        raise FileNotFoundError("dual_encoder_retriever_pairs.json not found.")

    with dataset_path.open(mode="r", encoding="utf-8") as file:
        data_dict = json.load(file)

    pairs = data_dict["pairs"]
    return pairs
    #raise NotImplementedError("Return tiny query/document pairs.")


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

    tokens = []

    for word in text.lower().split():
        clean_word = word.strip(string.punctuation)
        if clean_word:
            tokens.append(clean_word)

    return tokens

    #raise NotImplementedError("Tokenize text into lowercase words.")


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

    from collections import Counter

    vocab = {"[PAD]": 0, "[UNK]": 1}
    counts = Counter()

    for pair in pairs:
        counts.update(simple_tokenize(pair["query"]))
        counts.update(simple_tokenize(pair["document"]))

    for word, count in counts.most_common(max_vocab_size - len(vocab)):
        vocab[word] = len(vocab)

    return vocab
    #raise NotImplementedError("Build a shared query/document vocabulary.")


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

    unk_id = vocab["[UNK]"]
    pad_id = vocab["[PAD]"]

    tokens = simple_tokenize(text)
    token_ids = []
    for token in tokens:
        token_id = vocab.get(token, unk_id)
        token_ids.append(token_id)

    token_ids = token_ids[:max_length]
    actual_length_before_pad = len(token_ids)

    if len(token_ids) < max_length:
        token_ids += [pad_id] * (max_length - len(token_ids))

    return (torch.tensor(token_ids, dtype=torch.long), torch.tensor(actual_length_before_pad, dtype=torch.long))
    #raise NotImplementedError("Encode text as padded token ids and length.")


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

        pair = self.pairs[index]
        query_token_ids, real_query_length = encode_text(pair["query"], self.vocab, self.max_length)
        doc_token_ids, real_doc_length = encode_text(pair["document"], self.vocab, self.max_length)
        
        return {"query_ids": query_token_ids, "query_length": real_query_length, "document_ids": doc_token_ids, "document_length": real_doc_length}
        #raise NotImplementedError("Return one encoded query/document pair.")


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

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_id = pad_id

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)

        self.proj = nn.Linear(in_features=embedding_dim, out_features=hidden_dim)

        #raise NotImplementedError("Build the mean-pooling text encoder.")

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

        batch_size, max_length = token_ids.shape            # shape = [8, 18]

        token_embeddings = self.embedding(token_ids)        # shape = [8, 18, 48]

        mask = token_ids != self.pad_id                     # shape = [8, 18]
        mask = mask.unsqueeze(dim=-1)                       # shape = [8, 18, 1]    for broadcasting in next line

        sum_tokens = (token_embeddings * mask).sum(dim=1)   # [8, 18, 48] * [8, 18, 1] = [8, 18, 48], then sum across dim = 1 is shape = [8, 48]

        lengths = lengths.unsqueeze(dim=-1)                 # [8, 1]    for broadcasting in next line
        avg_tokens = sum_tokens / lengths

        proj = self.proj(avg_tokens)

        output = nn.functional.normalize(proj, p=2, dim=-1) # shape = [8, 64]

        return output
        #raise NotImplementedError("Encode token ids into normalized text embeddings.")


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

        self.query_encoder = MeanPoolTextEncoder(vocab_size, config.embedding_dim, config.hidden_dim, pad_id)

        self.doc_encoder = MeanPoolTextEncoder(vocab_size, config.embedding_dim, config.hidden_dim, pad_id)


        #raise NotImplementedError("Build query and document encoders.")

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

        queries = self.query_encoder(query_ids, query_lengths)      # shape = [8, 64]

        docs = self.doc_encoder(document_ids, document_lengths)     # shape = [8, 64]

        # The encoders normalized these vectors, so this dot product is cosine similarity.
        # Example: if a query vector points in the same direction as a document vector,
        # its score is close to 1. If it points in a different direction, its score is lower.
        scores = queries @ docs.T                                   # shape = [8, 8]

        return scores
        #raise NotImplementedError("Return query-document score matrix.")


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

    pairs = build_tiny_pairs()
    vocab = build_vocab(pairs, config.max_vocab_size)

    data_ds = QueryDocumentPairDataset(pairs, vocab, config.max_length)

    gen = torch.Generator().manual_seed(config.random_state)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(data_ds, lengths=[0.6, 0.2, 0.2], generator=gen)

    return {"train": torch.utils.data.DataLoader(train_ds, config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size)}, vocab
    #raise NotImplementedError("Build retriever dataloaders and vocabulary.")


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

    model.train()

    loss_sum = 0
    num_elements = 0
    accs = []

    for data_dict in loader:
        query_ids = data_dict["query_ids"].to(device)
        query_lengths = data_dict["query_length"].to(device)
        doc_ids = data_dict["document_ids"].to(device)
        doc_lengths = data_dict["document_length"].to(device)

        optimizer.zero_grad()

        scores = model(query_ids, query_lengths, doc_ids, doc_lengths)      #shape = [8, 8]
        # scores was computed by comparing every query against every document in the batch

        labels = torch.arange(scores.size(0), device=device)                #shape = [8], i.e. value = [0, 1, 2, 3, 4, 5, 6, 7]
        # This means for the first sample, the comparison score with its corresponding document should be in position 0
        # For the second sample, the comparison score with its corresponding document should be in position 1, and so on ... 
        # This is a contrastive loss setup:
        # each query should score its matching document higher than the other documents in the batch.
        # One row in `scores` is one query compared with every document in the batch.
        # `scores` are raw scores, not probabilities. CrossEntropyLoss applies softmax inside.
        # Example: row 3 contains the cosine similarity between query 3 and document 0, 1, 2, 3, and so on.
        # The correct match is document 3, so score[3, 3] should become higher than the other scores in row 3.
        # The other scores in row 3 are wrong documents, so they should become smaller.
        # `scores` are raw scores, not probabilities. CrossEntropyLoss applies softmax inside.
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()

        batch_size = query_ids.shape[0]
        loss_sum += loss.item() * batch_size
        num_elements+=batch_size

        preds = scores.argmax(dim=-1)
        correct = (preds == labels).float()
        accs.append(correct.mean().item())

    mean_loss = loss_sum/num_elements
    mean_acc = torch.tensor(accs).mean().item()        

    return {"loss": mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Train one dual-encoder retrieval epoch.")


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


    model.eval()

    loss_sum = 0
    num_elements = 0
    accs = []

    for data_dict in loader:
        query_ids = data_dict["query_ids"].to(device)
        query_lengths = data_dict["query_length"].to(device)
        doc_ids = data_dict["document_ids"].to(device)
        doc_lengths = data_dict["document_length"].to(device)
       

        scores = model(query_ids, query_lengths, doc_ids, doc_lengths)      #shape = [8, 8]
        # scores was computed by comparing every query against every document in the batch

        labels = torch.arange(scores.size(0), device=device)                #shape = [8], i.e. value = [0, 1, 2, 3, 4, 5, 6, 7]
        # This means for the first sample, the comparison score with its corresponding document should be in position 0
        # For the second sample, the comparison score with its corresponding document should be in position 1, and so on ... 
        # This is a contrastive loss setup:
        # each query should score its matching document higher than the other documents in the batch.
        # One row in `scores` is one query compared with every document in the batch.
        # `scores` are raw scores, not probabilities. CrossEntropyLoss applies softmax inside.
        # Example: row 3 contains the cosine similarity between query 3 and document 0, 1, 2, 3, and so on.
        # The correct match is document 3, so score[3, 3] should become higher than the other scores in row 3.
        # The other scores in row 3 are wrong documents, so they should become smaller.
        # `scores` are raw scores, not probabilities. CrossEntropyLoss applies softmax inside.
        loss = loss_fn(scores, labels)        

        batch_size = query_ids.shape[0]
        loss_sum += loss.item() * batch_size
        num_elements+=batch_size

        preds = scores.argmax(dim=-1)
        correct = (preds == labels).float()
        accs.append(correct.mean().item())

    mean_loss = loss_sum/num_elements
    mean_acc = torch.tensor(accs).mean().item()        

    return {"loss": mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Evaluate dual-encoder retrieval.")


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

    loaders, vocab = build_dataloaders(config)

    model = DualEncoderRetriever(vocab_size=len(vocab), config=config, pad_id=vocab["[PAD]"])
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(lr=config.learning_rate, params=model.parameters())

    train_dict = {"loss": 0.0, "accuracy": 0.0}
    val_dict = {"loss": 0.0, "accuracy": 0.0}

    for epoch in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["train"], optimizer, loss_fn)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {epoch+1} - loss: {train_dict['loss']}, acc: {train_dict['accuracy']}, val_loss: {val_dict['loss']}, val_acc: {val_dict['accuracy']}")

    test_dict = evaluate(model, loaders["test"], loss_fn)
    print(f"TEST RESULTS: {test_dict['loss']}, acc: {test_dict['accuracy']}")

    test_batch = next(iter(loaders["test"]))
    query_ids = test_batch["query_ids"].to(device)
    query_lengths = test_batch["query_length"].to(device)
    document_ids = test_batch["document_ids"].to(device)
    document_lengths = test_batch["document_length"].to(device)

    model.eval()
    with torch.no_grad():
        scores = model(query_ids, query_lengths, document_ids, document_lengths)

    retrieved_position = int(scores[0].argmax().item())
    is_correct = retrieved_position == 0

    test_subset = loaders["test"].dataset
    original_dataset = test_subset.dataset
    test_indices = list(test_subset.indices)
    query_pair = original_dataset.pairs[test_indices[0]]
    retrieved_pair = original_dataset.pairs[test_indices[retrieved_position]]

    print("TEST EXAMPLE")
    print(f"query: {query_pair['query']}")
    print(f"correct document: {query_pair['document']}")
    print(f"retrieved document: {retrieved_pair['document']}")
    print(f"correct: {is_correct}")

    return {
        "train_loss": train_dict["loss"],
        "train_accuracy": train_dict["accuracy"],
        "val_loss": val_dict["loss"],
        "val_accuracy": val_dict["accuracy"],
        "test_loss": test_dict["loss"],
        "test_accuracy": test_dict["accuracy"],
        "retrieval_note": "In-batch negatives means each query treats the other documents in the same batch as wrong matches.",
    }
    #raise NotImplementedError("Run the full dual-encoder retriever workflow.")


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
