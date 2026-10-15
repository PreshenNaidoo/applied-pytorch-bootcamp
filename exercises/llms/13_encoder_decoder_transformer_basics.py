"""
============================================================
EXERCISE 13: ENCODER DECODER TRANSFORMER BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny encoder-decoder transformer.

Plain-language meaning:
- the encoder reads the input sequence
- the decoder writes the output sequence
- decoder self-attention can only look left
- cross-attention lets the decoder look at encoder outputs

WHY THIS EXERCISE EXISTS
Decoder-only models are common for chat and text generation.
Encoder-decoder models are also important.
They are used for tasks like translation, summarization, and text-to-text learning.

This exercise teaches:
- encoder self-attention
- decoder causal self-attention
- cross-attention
- teacher forcing
- sequence-to-sequence generation

HOW THIS FITS INTO THE BOOTCAMP
Earlier transformer exercises focused mostly on encoder-only or decoder-only ideas.
This file adds the classic sequence-to-sequence setup
where one side reads and the other side writes.

Beginner intuition:
- the encoder is the reader
- the decoder is the writer
- the reader first builds a useful summary of the source text
- then the writer uses that summary while producing the target text step by step

Small example:
- source sentence: `I like cats`
- target sentence: `J'aime les chats`
- the encoder reads the English words
- the decoder writes the French words using the encoder output

DATASET
Use a tiny real English-to-French translation task.

Clear default:
- Hugging Face dataset: `Helsinki-NLP/opus_books`
- config: `en-fr`
- field to use: `translation`
- source text: `translation["en"]`
- target text: `translation["fr"]`
- start with the first 1,000 examples, then use a smaller slice like 100 examples while debugging

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load `Helsinki-NLP/opus_books` with config `en-fr`
- keep only the source text and target text fields
- tokenize both sides into words
- lowercase the text first so `The` and `the` become the same word
- start with `basic_word_tokenize`
- build one shared word vocabulary from the training source and target text
- keep the most common words that fit inside `vocab_size`
- use the unknown token for words that are not in the vocabulary
- shift the target right for decoder inputs
- keep the unshifted target as labels

Word token example:
- English text: `I like cats`
- word tokens: `["i", "like", "cats"]`
- word ids: `[31, 82, 143]`
- French text: `J'aime les chats`
- word tokens: `["j'aime", "les", "chats"]`
- decoder input ids: `[start, 91, 47, 204]`
- target labels: `[91, 47, 204, end]`

INPUTS AND OUTPUTS
Source token ids:
- shape: `[batch_size, source_length]`
- each id means one source word, or a special token such as pad or unknown

Target input ids:
- shape: `[batch_size, target_length]`
- this is the target shifted right
- each id means one target word, or a special token such as start or pad

Target labels:
- shape: `[batch_size, target_length]`
- this is what the decoder should predict
- each id means the next target word, or the end token

Model output:
- shape: `[batch_size, target_length, vocab_size]`

What those shapes mean:
- source ids are what the encoder reads
- decoder input ids are the target shifted right by one step
- target labels are the answers the decoder should predict
- logits give one vocabulary score vector for each target position

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Default path:
- source token embeddings
- encoder blocks
- target token embeddings
- decoder blocks with causal masking
- cross-attention from decoder to encoder output
- final vocabulary head

Simple mental picture:
- the encoder reads the full source first
- the decoder writes the target one token at a time
- decoder self-attention looks left inside the target
- cross-attention looks across to the encoder output

LOSS FUNCTION
Use `nn.CrossEntropyLoss` over target tokens.

METRICS TO TRACK
Track at least:
- train loss
- validation loss
- token accuracy
- one generated example

TARGET RESULT RANGE
On a tiny word-level translation task, the model should learn a visible pattern.

Useful signs:
- token accuracy rises above random
- generated outputs start to match the expected target more often

COMMON MISTAKES
- letting the decoder see future target tokens
- forgetting cross-attention
- using target labels as decoder inputs without shifting
- mixing source length and target length
- applying softmax before `CrossEntropyLoss`

STEP-BY-STEP TODO
1. Build a tiny real source-target dataset.
2. Tokenize source and target sentences into words.
3. Build a word vocabulary.
4. Turn word lists into fixed-length token id tensors.
5. Build train, validation, and test dataloaders.
6. Build the decoder causal mask.
7. Build encoder embeddings and positional embeddings.
8. Build decoder embeddings and positional embeddings.
9. Build an encoder block.
10. Build a decoder block with causal self-attention and cross-attention.
11. Train with teacher forcing.
12. Generate a short output sequence.
13. Track token accuracy.

What each step is doing in simple words:
1. make source-target examples
2. turn words into token ids
3. give each common word one number
4. pad or cut each sentence to the same length
5. batch examples so training can use them
6. block the decoder from looking at future target words
7. build the reader part
8. build the writer part
9. let the encoder build source features
10. let the decoder use both past target tokens and encoder output
11. train with the correct earlier target tokens
12. try real generation one step at a time
13. check how many target tokens are predicted correctly

DEFINITION OF DONE
You are done when:
- encoder output has shape `[batch_size, source_length, embed_dim]`
- decoder logits have shape `[batch_size, target_length, vocab_size]`
- decoder future tokens are masked
- cross-attention reads encoder output
- you can explain self-attention vs cross-attention
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from math import sqrt

import datasets as hf_datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.metrics import token_accuracy_from_logits
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


device = "cuda" if torch.cuda.is_available() else "cpu"


# Store settings for the encoder-decoder exercise.
# Return:
# - a config object with token ids, sequence lengths, model sizes, and training settings
@dataclass
class EncoderDecoderConfig:
    """Settings for the encoder-decoder transformer exercise."""

    hf_data_path = "Helsinki-NLP/opus_books"
    
    # Source and target lengths can be different.
    # Here the source side is English and the target side is French.
    # French can use more word tokens than English for the same meaning.
    # The target side also needs room for special tokens like start and end.
    # So 28 gives the target a little extra space compared with source length 24.
    source_length: int = 24
    target_length: int = 28

    embed_dim: int = 192
    num_heads: int = 4
    mlp_dim: int = 384
    num_layers: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 20
    random_state: int = 42
    target_token_accuracy: float = 0.45

    pad_token_id: int = 0
    start_token_id: int = 1
    end_token_id: int = 2
    unknown_token_id: int = 3

    # A word-level English-French vocabulary needs more than a few thousand words.
    # If this is too small, many real words become `[UNK]`, and the model can learn
    # to generate `[UNK]` too often.
    vocab_size: int = 12000
    
    token_to_int: dict[str, int] | None = None
    int_to_token: dict[int, str] | None = None


# Describe the key tensor shapes.
# Return:
# - dictionary with source, decoder input, logits, and encoder output shapes
def describe_encoder_decoder_shapes(config: EncoderDecoderConfig) -> dict[str, object]:
    """Return the key sequence-to-sequence shapes."""
    return {
        "source_ids": (config.batch_size, config.source_length),
        "decoder_input_ids": (config.batch_size, config.target_length),
        "target_logits": (config.batch_size, config.target_length, config.vocab_size),
        "encoder_output": (config.batch_size, config.source_length, config.embed_dim),
    }


# Split one sentence into simple word tokens.
# Input:
# - one text string, such as "I like cats"
# Return:
# - lowercase words, such as ["i", "like", "cats"]
def basic_word_tokenize(text: str) -> list[str]:
    """Split text into lowercase words."""
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


# Build one shared vocabulary for English source words and French target words.
# Input:
# - source_texts: English sentences from the training split
# - target_texts: French sentences from the training split
# - config with vocabulary size and special token ids
# Return:
# - dictionary with at least:
#   - `token_to_int`: maps each word string to one integer id
#   - `int_to_token`: maps each integer id back to one word string
# The first ids are special tokens:
# - 0 means padding
# - 1 means start of target sentence
# - 2 means end of target sentence
# - 3 means unknown word
def build_word_vocab(source_texts: list[str], target_texts: list[str], config: EncoderDecoderConfig) -> dict[str, object]:
    """Build a shared word vocabulary from source and target text."""

    raise NotImplementedError("Build token_to_int and int_to_token for word tokens.")












# Turn one word list into a fixed-length tensor of token ids.
# Input:
# - words: a list such as ["i", "like", "cats"]
# - token_to_int: vocabulary lookup table
# - max_length: final sequence length
# - config with pad, end, and unknown token ids
# - add_end_token: True when encoding target labels
# Return:
# - tensor with shape `[max_length]`
# Example:
# - words: ["i", "like"]
# - ids before padding: [31, 82, end]
# - final ids for max_length 5: [31, 82, end, pad, pad]
# This helper does not add the start token.
# The start token is only for decoder inputs.
# Add it later when you shift the target right.
def encode_word_sequence(
    words: list[str],
    token_to_int: dict[str, int],
    max_length: int,
    config: EncoderDecoderConfig,
    add_end_token: bool = False,
) -> torch.Tensor:
    """Turn words into a fixed-length token id tensor."""

    raise NotImplementedError("Map words to ids, add end if needed, then pad or cut.")




    # Add padding after the real tokens.



# Beginner hints for this file:
# - the encoder reads the whole source sequence
# - the decoder predicts the target sequence one step at a time
# - decoder self-attention must be causal, but cross-attention is allowed to read the encoder output
# - teacher forcing means the decoder gets the correct earlier target tokens during training
# - when you shift targets, decoder input and target labels should look almost the same except for the one-step offset

# Build word-level source-target examples.
# Return a dict with exactly these keys:
# - `source_ids`
# - `decoder_input_ids`
# - `target_labels`
# This is where the "shift target right" idea becomes concrete.
# This dataset gets the actual English and French text.
# `build_dataloaders` decides which examples are train, val, or test
# before it creates this dataset.
#
# Small example:
# - source_texts: ["i like cats", "you like books"]
# - target_texts: ["j'aime les chats", "tu aimes les livres"]
# - token_to_int: {"[PAD]": 0, "[START]": 1, "[END]": 2, "[UNK]": 3, ...}
# - source_texts and target_texts must have the same length
# - source_texts[i] and target_texts[i] must be one matching pair
#
# For the first pair:
# - English source text: "i like cats"
# - French target text: "j'aime les chats"
# - source_ids should contain the English word ids:
#   [i, like, cats, pad, pad]
# - decoder_input_ids should contain the French ids shifted right:
#   [start, j'aime, les, chats, pad]
# - target_labels should contain what the decoder should predict:
#   [j'aime, les, chats, end, pad]
class WordSeq2SeqDataset(Dataset):
    """Tiny word-level source-target sequence dataset."""

    # Build source-target examples from English and French text.
    # Input:
    # - source_texts: English source sentences
    # - target_texts: French target sentences
    # - token_to_int: word-to-id lookup table
    # - config: lengths and special token ids
    # Rule:
    # - `source_texts[index]` and `target_texts[index]` must be one matching translation pair
    # Expected stored data:
    # - source ids
    # - decoder input ids
    # - target labels
    def __init__(
        self,
        source_texts: list[str],
        target_texts: list[str],
        token_to_int: dict[str, int],
        config: EncoderDecoderConfig,
    ) -> None:
        super().__init__()
        raise NotImplementedError("Create source-target examples from English and French text.")


    # Return dataset length.
    # Return:
    # - integer number of source-target examples
    def __len__(self) -> int:
        """Return the number of examples."""
        raise NotImplementedError("Return the dataset length.")

    # Return one source-target example.
    # Return:
    # - dictionary with `source_ids`, `decoder_input_ids`, and `target_labels`
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one source-target training example."""
        raise NotImplementedError("Return source_ids, decoder_input_ids, and target_labels.")


        # - source_ids should contain the English word ids:
        #   [i, like, cats, pad, pad]
        # - decoder_input_ids should contain the French ids shifted right:
        #   [start, j'aime, les, chats, pad]
        # - target_labels should contain what the decoder should predict:
        #   [j'aime, les, chats, end, pad]

        



# Build the train, validation, and test dataloaders for the word seq2seq task.
# Each batch should contain:
# - `source_ids`
# - `decoder_input_ids`
# - `target_labels`
def build_dataloaders(config: EncoderDecoderConfig) -> dict[str, DataLoader]:
    """Return train, val, and test dataloaders."""
    # One clean implementation plan:
    # - import `load_dataset` from `datasets` inside this function
    # - load a tiny subset from a real seq2seq dataset such as `opus_books`
    # - keep only a small number of examples at first
    # - use `basic_word_tokenize` to split each sentence into words
    # - use `build_word_vocab` on the training split
    # - save the vocabulary on the config:
    #   - `config.token_to_int = vocab["token_to_int"]`
    #   - `config.int_to_token = vocab["int_to_token"]`
    #   - `config.vocab_size = len(config.token_to_int)`
    # - map each source word to a word id to make `source_ids`
    # - map each target word to a word id to make one target sequence
    # - use `encode_word_sequence` so every sequence has a fixed length
    # - build:
    #   - `decoder_input_ids` by shifting the target right and adding the start token
    #   - `target_labels` as the unshifted target tokens
    # - pad or cut every sequence to the fixed source and target lengths
    # - wrap those tensors in a small dataset class
    # - return train, val, and test dataloaders
    # Exact training rule:
    # - decoder inputs are what the decoder sees
    # - target labels are what the decoder should predict one step later

    
    #download_config = hf_datasets.DownloadConfig(local_files_only=True)
    raise NotImplementedError("Return dataloaders for the word seq2seq task.")

    # row = data[0]
    # print(row["translation"]["en"])
    # print(row["translation"]["fr"])


    # Filter to short sentence pairs
    # The model will see easier examples and generation will look less random.









# Build the decoder causal mask.
# This is the decoder-side "no future tokens" rule.
# The encoder does not use this kind of left-to-right mask.
# Return:
# - boolean tensor with shape `[target_length, target_length]`
# - True means "block this position"
# Example for target_length = 4:
# - token 0 can see token 0
# - token 1 can see tokens 0 and 1
# - token 2 can see tokens 0, 1, and 2
# - token 3 can see tokens 0, 1, 2, and 3
def build_decoder_causal_mask(target_length: int) -> torch.Tensor:
    """Return a causal mask for decoder self-attention."""
    # The decoder has two very different views:
    # - in self-attention, it must not look at future target tokens
    # - in cross-attention, it is allowed to read the full encoder output

    raise NotImplementedError("Return a mask that blocks future target tokens.")





# Build the tiny encoder-decoder model.
# High-level path:
# - embed source tokens
# - run encoder
# - embed decoder tokens
# - run decoder with self-attention and cross-attention
# - return raw vocabulary logits
class TinyEncoderDecoderTransformer(nn.Module):
    """Small encoder-decoder transformer."""

    # Build the encoder-decoder model parts.
    #
    # What to create in this method:
    #
    # 1. Store the config.
    #    This is already done below with `self.config = config`.
    #
    # 2. Create an embedding table for source tokens.
    #    Source tokens are the input sentence tokens, such as English words.
    #    This turns token ids with shape `[batch_size, source_length]`
    #    into vectors with shape `[batch_size, source_length, embed_dim]`.
    #
    # 3. Create an embedding table for target tokens.
    #    Target tokens are the decoder input tokens, such as French words seen so far.
    #    This has the same shape idea as the source embedding table.
    #
    # 4. Create positional embeddings for source positions.
    #    The source token embedding tells the model which word it sees.
    #    The source positional embedding tells the model where that word is.
    #
    # 5. Create positional embeddings for target positions.
    #    The decoder also needs position information, because word order matters.
    #
    # 6. Create the encoder stack.
    #    The encoder reads the full source sentence.
    #    It should use self-attention, because source tokens need to look at each other.
    #
    # 7. Create the decoder stack.
    #    The decoder has two attention jobs:
    #    - masked self-attention over target tokens seen so far
    #    - cross-attention over the encoder output
    #
    # 8. Create the output head.
    #    This maps decoder hidden states from `embed_dim` to `vocab_size`.
    #    The output is raw logits, not probabilities.
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        raise NotImplementedError("Build encoder, decoder, embeddings, positions, and output head.")









    # Run the encoder-decoder model.
    # Input:
    # - `source_ids`: `[batch_size, source_length]`
    # - `decoder_input_ids`: `[batch_size, target_length]`
    # Return:
    # - raw target-token logits with shape `[batch_size, target_length, vocab_size]`
    #
    # Step-by-step path:
    #
    # 1. Read the shapes.
    #    `source_ids` gives `batch_size` and `source_length`.
    #    `decoder_input_ids` gives `target_length`.
    #
    # 2. Build source position ids.
    #    For a source length of 4, the positions are `[0, 1, 2, 3]`.
    #    These positions are shared across the batch.
    #
    # 3. Build target position ids.
    #    For a target length of 5, the positions are `[0, 1, 2, 3, 4]`.
    #
    # 4. Embed the source tokens and add source position embeddings.
    #    After this, the source tensor should have shape
    #    `[batch_size, source_length, embed_dim]`.
    #
    # 5. Embed the decoder input tokens and add target position embeddings.
    #    After this, the target tensor should have shape
    #    `[batch_size, target_length, embed_dim]`.
    #
    # 6. Run the encoder on the source tensor.
    #    The encoder output is the "memory" the decoder will read from.
    #
    # 7. Build the decoder causal mask.
    #    This stops target position 2 from seeing target positions 3, 4, and later.
    #    The decoder is only allowed to use earlier target tokens.
    #
    # 8. Run the decoder.
    #    Give it:
    #    - the target tensor
    #    - the encoder output
    #    - the causal mask for target self-attention
    #
    # 9. Pass the decoder output through the output head.
    #    This gives one score for every vocabulary token at every target position.
    #    The final shape should be `[batch_size, target_length, vocab_size]`.
    def forward(self, source_ids: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """Return target token logits with shape `[batch, target_length, vocab_size]`."""

        raise NotImplementedError("Run encoder-decoder transformer and return raw logits.")





        # In decoder cross-attention:
        # - decoder hidden states become the queries
        # - encoder features become the keys
        # - encoder features also become the values
        # Plain meaning: the decoder asks which source tokens matter
        # for each target token it is trying to predict.


# Build the main training objects.
# Return:
# - encoder-decoder model
# - loss function
# - optimizer
# Use a per-token loss so training can ignore tokens that should not teach the model.
# Padding should not count because it is only there to make sequences the same length.
# Unknown target tokens should not count because they can make the model learn to
# generate `[UNK]` too often.
def build_training_objects(config: EncoderDecoderConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, loss function, and optimizer."""
    raise NotImplementedError("Return encoder-decoder model, CrossEntropyLoss, and optimizer.")



# Run one full teacher-forcing training epoch.
# Return:
# - mean loss
# - token accuracy
# Teacher forcing means the decoder sees the correct earlier target tokens during training.
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run one full teacher-forcing training epoch."""

    raise NotImplementedError("Return mean train loss and token accuracy.")



    


# Run one full evaluation pass.
# Return:
# - validation loss
# - token accuracy
# This measures the model without updating weights.
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:
    """Run one full evaluation pass."""

    raise NotImplementedError("Return mean validation loss and token accuracy.")



    


# Generate one target sequence from one source sequence.
# In plain words:
# - run the encoder once
# - start the decoder with the start token
# - predict one token at a time
# - append each new token and repeat
# Input:
# - `source_ids` should have shape `[source_length]`
# - if you have a batch, pass one row, such as `batch["source_ids"][0]`
@torch.no_grad()
def generate_sequence(model: nn.Module, source_ids: torch.Tensor, config: EncoderDecoderConfig) -> torch.Tensor:
    """Generate one target sequence from one source sequence."""
    # Generation idea:
    # - run the encoder once on the source
    # - start the decoder with the start token
    # - predict one next target token
    # - append it
    # - repeat until end token or max length    

    raise NotImplementedError("Generate target token ids one step at a time.")


    


# Run the full training workflow.
# Return:
# - final train loss
# - validation token accuracy
# - one generated example or note
def run_training(config: EncoderDecoderConfig) -> dict[str, float | str]:
    """Run the full encoder-decoder training workflow."""

    raise NotImplementedError("Train and evaluate the tiny encoder-decoder transformer.")









# Check whether the validation token accuracy reached the rough target.
# Return:
# - True when the expected metric is present and high enough
def check_metrics(metrics: dict[str, float | str], target: float) -> bool:
    """Return True when token accuracy reaches the rough target."""
    return "val_token_accuracy" in metrics and float(metrics["val_token_accuracy"]) >= target


# Run one scaffold task and catch unfinished TODOs.
# Return:
# - result object
# - boolean saying whether the task is done
def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


# Run the exercise scaffold one step at a time.
# Expected behavior:
# - stop at the first unfinished TODO
# - print progress so the student knows what to implement next
def main() -> None:
    """Run the encoder-decoder scaffold one step at a time."""
    config = EncoderDecoderConfig()
    set_seed(config.random_state)

    completed_steps = 0
    total_steps = 16

    # Update progress after each scaffold task.
    # Return:
    # - True if the scaffold should keep running
    # - False if it should stop at the current TODO
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    _, ok = report_task_progress("describe_encoder_decoder_shapes", describe_encoder_decoder_shapes, config)
    if not count_or_stop(ok):
        return

    sample_source_texts = ["I like cats", "You like books"]
    sample_target_texts = ["J'aime les chats", "Tu aimes les livres"]

    sample_tokens, ok = report_task_progress("basic_word_tokenize", basic_word_tokenize, sample_source_texts[0])
    if not count_or_stop(ok):
        return

    vocab_result, ok = report_task_progress("build_word_vocab", build_word_vocab, sample_source_texts, sample_target_texts, config)
    if not count_or_stop(ok):
        return

    token_to_int = vocab_result["token_to_int"]
    _, ok = report_task_progress("encode_word_sequence", encode_word_sequence, sample_tokens, token_to_int, config.source_length, config)
    if not count_or_stop(ok):
        return

    dataset, ok = report_task_progress(
        "WordSeq2SeqDataset.__init__",
        WordSeq2SeqDataset,
        sample_source_texts,
        sample_target_texts,
        token_to_int,
        config,
    )
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("WordSeq2SeqDataset.__len__", len, dataset)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("WordSeq2SeqDataset.__getitem__", dataset.__getitem__, 0)
    if not count_or_stop(ok):
        return

    loaders, ok = report_task_progress("build_dataloaders", build_dataloaders, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("build_decoder_causal_mask", build_decoder_causal_mask, config.target_length)
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("TinyEncoderDecoderTransformer.__init__", TinyEncoderDecoderTransformer, config)
    if not count_or_stop(ok):
        return

    source_ids = torch.zeros(config.batch_size, config.source_length, dtype=torch.long)
    decoder_input_ids = torch.zeros(config.batch_size, config.target_length, dtype=torch.long)
    _, ok = report_task_progress("TinyEncoderDecoderTransformer.forward", model, source_ids, decoder_input_ids)
    if not count_or_stop(ok):
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return

    training_model, loss_fn, optimizer = training_objects
    _, ok = report_task_progress("train_one_epoch", train_one_epoch, training_model, loaders["train"], loss_fn, optimizer)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("evaluate", evaluate, training_model, loaders["val"], loss_fn)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("generate_sequence", generate_sequence, training_model, source_ids[0], config)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if ok and metrics is not None:
        passed = check_metrics(metrics, config.target_token_accuracy)
        print_pass_or_try_again(passed, "val_token_accuracy", float(metrics.get("val_token_accuracy", 0.0)), f">= {config.target_token_accuracy}")

if __name__ == "__main__":
    main()
