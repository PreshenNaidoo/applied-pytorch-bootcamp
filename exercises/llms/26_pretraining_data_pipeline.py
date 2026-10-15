"""
============================================================
EXERCISE 26: PRETRAINING DATA PIPELINE
============================================================

WHAT YOU ARE BUILDING
You are building a small language-model pretraining data pipeline.

Plain-language meanings:
- pretraining means training a model on a large general text task before any special task
- a document is one text item, such as an article, page, or file
- deduplication means removing repeated documents
- filtering means removing bad or unsuitable examples
- data leakage means validation text accidentally appears in training
- packing means joining tokenized text into fixed-length chunks

WHY THIS EXERCISE EXISTS
Frontier labs spend a lot of effort on data.
A model can only learn from the text it sees.
Bad data can waste compute, leak answers, or teach unwanted behavior.

This exercise teaches:
- loading documents
- cleaning text
- removing duplicates
- filtering low-quality examples
- making train and validation splits
- tokenizing and packing text into training blocks
- checking for leakage

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises trained models.
This file focuses on what happens before training starts.

Beginner intuition:
- training data is the model's study material
- repeated pages can make the model memorize too much
- validation must be separate so the score is honest
- fixed-length token blocks make batching simple

DATASET
Use Tiny Shakespeare or a tiny local document dataset written inside this file.

Clear default:
- split Tiny Shakespeare into small document-like chunks if it is available
- if it is not available, use 10 to 20 short paragraph strings
- include at least:
  - one exact duplicate
  - one very short document
  - one noisy document with mostly symbols
  - several normal clean documents

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load Tiny Shakespeare from `data/` and split it into chunks
- if the file is missing, write the documents as a Python list
- normalize whitespace
- remove exact duplicate strings
- filter documents that are too short or too noisy
- split the remaining documents into train and validation
- build the vocabulary from training documents only
- tokenize both train and validation documents
- pack tokens into fixed-length blocks

INPUTS AND OUTPUTS
Input:
- list of raw document strings

Output:
- filtered train documents
- filtered validation documents
- train token blocks with shape `[num_train_blocks, block_size]`
- validation token blocks with shape `[num_val_blocks, block_size]`
- a summary of how many documents were removed

Important rule:
- validation documents must not also appear in training

STEP-BY-STEP TODO
1. Build or load a tiny set of documents.
2. Normalize whitespace.
3. Remove exact duplicates.
4. Filter documents that are too short or too noisy.
5. Split documents into train and validation.
6. Check that validation documents are not in train.
7. Tokenize text.
8. Pack tokens into fixed-length blocks.
9. Return a summary of document counts and token counts.

DEFINITION OF DONE
You are done when:
- duplicate documents are removed
- train and validation documents do not overlap
- packed token blocks have fixed length
- the summary reports what changed at each pipeline step
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import random

import torch


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again
# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for the data pipeline exercise.
# Expected return:
# - a config object with filtering, split, and block-size settings
@dataclass
class PretrainingDataConfig:
    """Settings for the pretraining data pipeline."""

    min_chars: int = 40
    validation_fraction: float = 0.2
    block_size: int = 32
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    random_state: int = 42


# Return small raw documents.
# Expected return:
# - a list of strings
# Include a duplicate and a low-quality document so the checks have something to remove.
def load_raw_documents() -> list[str]:
    """Return raw documents for pipeline practice."""
    # Goal:
    # Return a small list of raw document strings.
    # Include good documents and a few bad ones so the pipeline has something to clean.
    #
    # Input:
    # - no direct arguments
    #
    # Output:
    # - `list[str]`
    # - About 10 to 20 short documents is enough.
    #
    # Required examples:
    # - at least one exact duplicate
    # - at least one very short document, such as `"Hi."`
    # - at least one noisy document, such as `"@@@@ #### $$$$ !!!!"`
    # - several normal documents with words and punctuation
    #
    # Method:
    # Simple path:
    # Return a hard-coded list.
    #
    # Optional path:
    # If `data/tiny_shakespeare.txt` exists, you may load a short slice from it
    # and split it into paragraph-like chunks.
    #
    # Keep this exercise small. The goal is the pipeline shape, not a large dataset.
    #
    # Steps:
    # 1. Create a list of document strings.
    # 2. Add one exact duplicate.
    # 3. Add one very short string.
    # 4. Add one noisy symbol-heavy string.
    # 5. Return the list.
    #
    # Checks:
    # - The returned list should not be empty.
    # - `len(documents)` should be at least 10.
    # - There should be at least one duplicate before deduplication.
    raise NotImplementedError("Return a small list of raw document strings.")


# Normalize one document.
# Expected return:
# - one cleaned string
# Keep this simple: trim spaces and collapse repeated whitespace.
def normalize_document(text: str) -> str:
    """Clean one document string."""
    # Goal:
    # Normalize whitespace and trim the document.
    #
    # Input:
    # - `text`: str
    #
    # Output:
    # - `str`
    # - one cleaned document
    #
    # Small example:
    # ```python
    # normalize_document("  hello\\n\\n   world  ") == "hello world"
    # ```
    #
    # Method:
    # Use:
    # ```python
    # " ".join(text.split())
    # ```
    #
    # Meaning:
    # - `text.split()` breaks on any whitespace and removes empty pieces.
    # - `" ".join(...)` puts single spaces back between the words.
    #
    # Steps:
    # 1. Split text on whitespace.
    # 2. Join pieces with one space.
    # 3. Return the cleaned string.
    #
    # Checks:
    # - The result should not start or end with a space.
    # - Repeated spaces and newlines should be collapsed.
    raise NotImplementedError("Normalize whitespace and trim the document.")


# Remove exact duplicate documents.
# Expected return:
# - a list with the first copy kept and later exact copies removed
def deduplicate_documents(documents: list[str]) -> list[str]:
    """Remove exact duplicate documents."""
    # Goal:
    # Remove repeated documents while keeping the first copy.
    #
    # Input:
    # - `documents`: list[str]
    #
    # Output:
    # - `list[str]`
    # - same order as input, but later exact duplicates removed
    #
    # Small example:
    # ```python
    # ["a", "b", "a", "c"] -> ["a", "b", "c"]
    # ```
    #
    # Method:
    # Use:
    # ```python
    # seen = set()
    # unique = []
    # for document in documents:
    #     if document not in seen:
    #         unique.append(document)
    #         seen.add(document)
    # ```
    #
    # Steps:
    # 1. Create `seen` and `unique`.
    # 2. Loop over documents in order.
    # 3. Keep a document only the first time it appears.
    # 4. Return `unique`.
    #
    # Checks:
    # - The first copy should be kept.
    # - `len(unique)` should be less than input length when duplicates exist.
    raise NotImplementedError("Remove repeated documents while keeping order.")


# Filter bad documents.
# Expected return:
# - a list of documents that are long enough and useful enough
def filter_documents(documents: list[str], config: PretrainingDataConfig) -> list[str]:
    """Filter low-quality documents."""
    # Goal:
    # Remove documents that are too short or too noisy.
    #
    # Input:
    # - `documents`: list[str]
    # - `config`: PretrainingDataConfig
    #
    # Output:
    # - `list[str]`
    # - only documents that pass the quality checks
    #
    # Filter rules:
    # - Keep documents with `len(document) >= config.min_chars`.
    # - Keep documents where at least half the characters are letters, numbers, or spaces.
    #
    # Noise ratio formula:
    # ```text
    # useful_ratio = useful_characters / total_characters
    # ```
    #
    # Useful character test:
    # ```python
    # char.isalnum() or char.isspace()
    # ```
    #
    # Small example:
    # - `"hello world " * 5` should pass.
    # - `"@@@@ #### $$$$ !!!!"` should fail because most characters are symbols.
    #
    # Method:
    # Use:
    # ```python
    # useful = sum(1 for char in document if char.isalnum() or char.isspace())
    # useful_ratio = useful / max(len(document), 1)
    # ```
    #
    # Steps:
    # 1. Create an empty output list.
    # 2. Loop over documents.
    # 3. Skip documents shorter than `config.min_chars`.
    # 4. Compute useful character ratio.
    # 5. Skip documents where useful ratio is below `0.5`.
    # 6. Return the kept documents.
    #
    # Checks:
    # - A very short document should be removed.
    # - A mostly-symbol document should be removed.
    # - Normal text should remain.
    raise NotImplementedError("Remove documents that are too short or too noisy.")


# Split documents into train and validation sets.
# Expected return:
# - train document list
# - validation document list
def split_train_validation(
    documents: list[str],
    config: PretrainingDataConfig,
) -> tuple[list[str], list[str]]:
    """Split documents into train and validation sets."""
    # Goal:
    # Create a deterministic train/validation split.
    #
    # Input:
    # - `documents`: list[str]
    # - `config`: PretrainingDataConfig
    #
    # Output:
    # - `(train_docs, val_docs)`
    # - both are `list[str]`
    #
    # Split formula:
    # ```text
    # val_count = max(1, round(len(documents) * config.validation_fraction))
    # train_count = len(documents) - val_count
    # ```
    #
    # Method:
    # Use a deterministic shuffle:
    # ```python
    # rng = random.Random(config.random_state)
    # shuffled = list(documents)
    # rng.shuffle(shuffled)
    # ```
    #
    # Then use the first `val_count` documents for validation
    # and the rest for training.
    #
    # Steps:
    # 1. Check that there are at least 2 documents.
    # 2. Copy and shuffle documents with the fixed seed.
    # 3. Compute `val_count`.
    # 4. Put the first `val_count` documents in validation.
    # 5. Put the rest in training.
    # 6. Return `(train_docs, val_docs)`.
    #
    # Checks:
    # - Both splits should be non-empty.
    # - Running the function twice with the same seed should give the same split.
    raise NotImplementedError("Create a deterministic train/validation split.")


# Check whether train and validation overlap.
# Expected return:
# - True if there is no overlap
def check_no_document_leakage(train_docs: list[str], val_docs: list[str]) -> bool:
    """Return True when validation documents are not in training."""
    # Goal:
    # Check that train and validation documents do not overlap.
    #
    # Input:
    # - `train_docs`: list[str]
    # - `val_docs`: list[str]
    #
    # Output:
    # - `True` when no validation document appears in training.
    # - `False` when there is overlap.
    #
    # Small example:
    # ```python
    # check_no_document_leakage(["a", "b"], ["c"]) == True
    # check_no_document_leakage(["a", "b"], ["b"]) == False
    # ```
    #
    # Method:
    # Use:
    # ```python
    # train_set = set(train_docs)
    # val_set = set(val_docs)
    # return train_set.isdisjoint(val_set)
    # ```
    #
    # Steps:
    # 1. Convert train docs to a set.
    # 2. Convert validation docs to a set.
    # 3. Return whether the sets are disjoint.
    #
    # Checks:
    # - The two small examples above should pass.
    # - This check should happen after splitting.
    raise NotImplementedError("Check that train and validation documents do not overlap.")


# Build a simple word-level vocabulary.
# Expected return:
# - token_to_id dictionary
def build_pipeline_vocabulary(documents: list[str], config: PretrainingDataConfig) -> dict[str, int]:
    """Build a small vocabulary from training documents."""
    # Goal:
    # Build a token vocabulary from training documents only.
    # Do not use validation documents to build the vocabulary.
    #
    # Input:
    # - `documents`: list[str]
    # - `config`: PretrainingDataConfig
    #
    # Output:
    # - `dict[str, int]`
    # - maps token string to integer id
    #
    # Required ids:
    # - `config.pad_token` should have id `0`
    # - `config.unk_token` should have id `1`
    #
    # Tokenization rule:
    # - lowercase text
    # - split on spaces
    #
    # Small example:
    # ```python
    # {"<pad>": 0, "<unk>": 1, "the": 2, "model": 3}
    # ```
    #
    # Method:
    # Use:
    # ```python
    # vocab_tokens = set()
    # for document in documents:
    #     vocab_tokens.update(document.lower().split())
    # tokens = [config.pad_token, config.unk_token] + sorted(vocab_tokens)
    # token_to_id = {token: index for index, token in enumerate(tokens)}
    # ```
    #
    # Meaning:
    # - Sorting keeps ids stable.
    # - The unknown token handles validation words that never appeared in training.
    #
    # Steps:
    # 1. Create an empty set of tokens.
    # 2. Loop over training documents.
    # 3. Lowercase and split each document.
    # 4. Add tokens to the set.
    # 5. Add pad and unknown tokens first.
    # 6. Build and return `token_to_id`.
    #
    # Checks:
    # - `token_to_id[config.pad_token]` should be `0`.
    # - `token_to_id[config.unk_token]` should be `1`.
    # - Do not use validation documents here.
    raise NotImplementedError("Build a token vocabulary from training documents.")


# Tokenize documents into ids.
# Expected return:
# - one list of token ids per document
def tokenize_documents(documents: list[str], token_to_id: dict[str, int], config: PretrainingDataConfig) -> list[list[int]]:
    """Turn documents into token-id lists."""
    # Goal:
    # Tokenize each document into ids.
    #
    # Input:
    # - `documents`: list[str]
    # - `token_to_id`: dict[str, int]
    # - `config`: PretrainingDataConfig
    #
    # Output:
    # - `list[list[int]]`
    # - one list of token ids per document
    #
    # Tokenization rule:
    # - lowercase text
    # - split on spaces
    # - map each token to its id
    # - use `config.unk_token` id for missing tokens
    #
    # Small example:
    # ```python
    # token_to_id = {"<pad>": 0, "<unk>": 1, "hello": 2}
    # tokenize_documents(["hello new"], token_to_id, config) == [[2, 1]]
    # ```
    #
    # Method:
    # Use:
    # ```python
    # unk_id = token_to_id[config.unk_token]
    # ids = [token_to_id.get(token, unk_id) for token in document.lower().split()]
    # ```
    #
    # Steps:
    # 1. Read `unk_id`.
    # 2. Create an empty output list.
    # 3. Loop over documents.
    # 4. Lowercase and split each document.
    # 5. Convert tokens to ids with unknown fallback.
    # 6. Return the list of id lists.
    #
    # Checks:
    # - Every output item should be a list of integers.
    # - Unknown validation tokens should map to the unknown id.
    raise NotImplementedError("Tokenize each document into ids.")


# Pack token ids into fixed-length blocks.
# Expected return:
# - tensor with shape `[num_blocks, block_size]`
def pack_token_blocks(token_lists: list[list[int]], config: PretrainingDataConfig, pad_id: int) -> torch.Tensor:
    """Pack token ids into fixed-length training blocks."""
    # Goal:
    # Return fixed-length token blocks.
    #
    # Input:
    # - `token_lists`: list[list[int]]
    # - `config`: PretrainingDataConfig
    # - `pad_id`: int
    #
    # Output:
    # - `torch.Tensor`
    # - shape `[num_blocks, config.block_size]`
    # - dtype `torch.long`
    #
    # Packing example:
    # If `block_size == 4` and all ids are:
    # ```python
    # [5, 6, 7, 8, 9]
    # ```
    # then blocks are:
    # ```python
    # [[5, 6, 7, 8],
    #  [9, pad_id, pad_id, pad_id]]
    # ```
    #
    # Method:
    # Use:
    # ```python
    # all_ids = [token_id for doc_ids in token_lists for token_id in doc_ids]
    # for start in range(0, len(all_ids), config.block_size):
    #     block = all_ids[start : start + config.block_size]
    #     if len(block) < config.block_size:
    #         block = block + [pad_id] * (config.block_size - len(block))
    # ```
    #
    # Meaning:
    # - Packing joins documents into one token stream.
    # - Padding makes the final block the same length as all other blocks.
    #
    # Steps:
    # 1. Flatten all token id lists into one list.
    # 2. If the list is empty, return an empty tensor shaped `[0, block_size]`.
    # 3. Slice ids into blocks of length `config.block_size`.
    # 4. Pad the last block if needed.
    # 5. Convert to `torch.tensor(blocks, dtype=torch.long)`.
    # 6. Return the tensor.
    #
    # Checks:
    # - `blocks.shape[1]` should equal `config.block_size` when blocks exist.
    # - `blocks.dtype` should be `torch.long`.
    # - The final block should be padded when needed.
    raise NotImplementedError("Return fixed-length token blocks.")


# Run the full data pipeline.
# Expected return:
# - summary dictionary with counts and leakage check
def run_pretraining_data_pipeline(config: PretrainingDataConfig) -> dict[str, int | bool | tuple[int, ...]]:
    """Run the pretraining data pipeline."""
    # Goal:
    # Run the full document-to-block pipeline.
    #
    # Input:
    # - `config`: PretrainingDataConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"raw_documents"`: number of raw documents
    # - `"normalized_documents"`: number after normalization
    # - `"deduplicated_documents"`: number after duplicate removal
    # - `"filtered_documents"`: number after quality filtering
    # - `"train_documents"`: number of train documents
    # - `"val_documents"`: number of validation documents
    # - `"vocab_size"`: number of tokens in the vocabulary
    # - `"no_leakage"`: bool
    # - `"train_blocks_shape"`: shape tuple for train blocks
    # - `"val_blocks_shape"`: shape tuple for validation blocks
    #
    # Shape examples:
    # - train blocks: `(num_train_blocks, config.block_size)`
    # - validation blocks: `(num_val_blocks, config.block_size)`
    #
    # Method:
    # Use the helper functions in this order:
    # ```python
    # raw_docs = load_raw_documents()
    # normalized_docs = [normalize_document(doc) for doc in raw_docs]
    # deduped_docs = deduplicate_documents(normalized_docs)
    # filtered_docs = filter_documents(deduped_docs, config)
    # train_docs, val_docs = split_train_validation(filtered_docs, config)
    # no_leakage = check_no_document_leakage(train_docs, val_docs)
    # token_to_id = build_pipeline_vocabulary(train_docs, config)
    # train_token_lists = tokenize_documents(train_docs, token_to_id, config)
    # val_token_lists = tokenize_documents(val_docs, token_to_id, config)
    # pad_id = token_to_id[config.pad_token]
    # train_blocks = pack_token_blocks(train_token_lists, config, pad_id)
    # val_blocks = pack_token_blocks(val_token_lists, config, pad_id)
    # ```
    #
    # Important:
    # Build the vocabulary from train documents only.
    # Validation text should never help choose the vocabulary.
    #
    # Steps:
    # 1. Load raw documents.
    # 2. Normalize every document.
    # 3. Remove exact duplicates.
    # 4. Filter short or noisy documents.
    # 5. Split into train and validation.
    # 6. Check no document leakage.
    # 7. Build vocabulary from training documents.
    # 8. Tokenize train and validation documents.
    # 9. Pack token ids into fixed-length blocks.
    # 10. Return the summary dictionary.
    #
    # Checks:
    # - `no_leakage` should be `True`.
    # - `train_blocks_shape[1]` should equal `config.block_size` when train blocks exist.
    # - `filtered_documents` should be less than or equal to `deduplicated_documents`.
    # - `deduplicated_documents` should be less than or equal to `normalized_documents`.
    raise NotImplementedError("Run the full document-to-block pipeline.")


# Check the pipeline summary.
# Expected return:
# - True when the important pipeline checks passed
def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when the pipeline summary looks complete."""
    required = {"raw_documents", "deduplicated_documents", "filtered_documents", "no_leakage", "train_blocks_shape"}
    return required.issubset(metrics) and bool(metrics.get("no_leakage", False))


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the pretraining data pipeline scaffold."""
    config = PretrainingDataConfig()
    set_seed(config.random_state)

    try:
        metrics = run_pretraining_data_pipeline(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "data_pipeline_checks", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
