"""
============================================================
EXERCISE 25: TOKENIZER INTERNALS FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building a tiny tokenizer from scratch.

Plain-language meanings:
- a tokenizer turns text into token ids
- a token is one piece of text, such as a word, word part, or character
- a vocabulary is the table that maps tokens to numbers
- BPE means byte-pair encoding
- BPE repeatedly joins common neighboring pieces into larger pieces

WHY THIS EXERCISE EXISTS
Large language models do not read raw text directly.
They read token ids.
Tokenization affects cost, context length, spelling behavior, code behavior,
multilingual text, and strange model mistakes.

This exercise teaches:
- why tokenization exists
- how text becomes ids
- how a simple BPE merge is chosen
- how unknown text is handled
- how decoding turns ids back into text pieces

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises used tokenizers as helper tools.
This file opens the box and shows the main idea inside.

Beginner intuition:
- character tokenization is simple, but long
- word tokenization is shorter, but breaks on unknown words
- BPE is a middle path
- it can learn useful word parts like `train`, `ing`, or `tion`

Small example:
- start with characters: `l o w`
- if `l` and `o` often appear together, BPE may merge them into `lo`
- later `lo` and `w` may merge into `low`

DATASET
Use Tiny Shakespeare or a tiny local text corpus written inside this file.

Clear default:
- use a short slice of Tiny Shakespeare if it is available
- if it is not available, use short words that share pieces, such as:
  - `low`
  - `lower`
  - `lowest`
  - `newer`
  - `wider`
- keep the corpus small enough that you can print every merge and understand it

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load a short Tiny Shakespeare text file from `data/`
- if the file is missing, write a Python list of short fallback strings
- split each string into words
- split each word into character pieces
- count repeated words so common words matter more
- learn merges from this tiny corpus only

INPUTS AND OUTPUTS
Input:
- raw text string
- example: `"lower newer"`

Intermediate objects:
- tokenized words as tuples of pieces
- pair-frequency dictionary
- learned merge list

Output:
- encoded token ids
- decoded token pieces
- a vocabulary dictionary

What those pieces mean:
- encoded ids are what a model would receive
- decoded pieces let you check that the ids still represent readable text

STEP-BY-STEP TODO
1. Build a tiny text corpus.
2. Split text into starting pieces.
3. Count neighboring token pairs.
4. Pick the most common pair.
5. Apply one merge.
6. Repeat for several merges.
7. Build token-to-id and id-to-token tables.
8. Encode new text into ids.
9. Decode ids back into tokens or text.
10. Explain one tradeoff in plain words.

DEFINITION OF DONE
You are done when:
- a vocabulary is returned
- merges are learned from the corpus
- text can be encoded into ids
- ids can be decoded back into readable pieces
- you can explain why tokenization matters
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys


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

# Store settings for the tokenizer exercise.
# Expected return:
# - a config object with merge count and special tokens
@dataclass
class TokenizerInternalsConfig:
    """Settings for the tiny tokenizer exercise."""

    num_merges: int = 12
    unk_token: str = "<unk>"
    end_word_token: str = "</w>"
    random_state: int = 42


# Return a tiny corpus.
# Expected return:
# - a list of short strings
# The strings should share some word parts so BPE has useful pairs to merge.
def build_tiny_corpus() -> list[str]:
    """Return a tiny text corpus for tokenizer practice."""
    # Goal:
    # Return a list of short training strings.
    # The strings should share word parts so BPE has useful pairs to merge.
    #
    # Input:
    # - no direct arguments
    #
    # Output:
    # - `list[str]`
    # - each string is one short training line
    #
    # Small example:
    # ```python
    # ["low lower lowest", "newer wider lower", "low lowest"]
    # ```
    #
    # Method:
    # Simple path:
    # Return a hard-coded list like the example above.
    #
    # Optional path:
    # If you have a small text file in `data/`, you may load a short slice.
    # Keep it small enough that you can print and understand the learned merges.
    #
    # Steps:
    # 1. Create a list of short strings.
    # 2. Make sure some words share pieces, such as `low`, `lower`, and `lowest`.
    # 3. Return the list.
    #
    # Checks:
    # - The list should not be empty.
    # - At least one word should appear more than once.
    # - The corpus should be small enough to debug by printing it.
    raise NotImplementedError("Return a list of short training strings.")


# Split one word into starting pieces.
# Expected return:
# - a tuple of character pieces
# Include the end-word token so the tokenizer can learn word endings.
def split_word_to_pieces(word: str, config: TokenizerInternalsConfig) -> tuple[str, ...]:
    """Return starting pieces for one word."""
    # Goal:
    # Split a word into character pieces plus an end marker.
    #
    # Input:
    # - `word`: str
    # - `config`: TokenizerInternalsConfig
    #
    # Output:
    # - `tuple[str, ...]`
    #
    # Small example:
    # If `word == "low"` and `config.end_word_token == "</w>"`,
    # return:
    # ```python
    # ("l", "o", "w", "</w>")
    # ```
    #
    # Method:
    # Use:
    # ```python
    # pieces = tuple(word) + (config.end_word_token,)
    # ```
    #
    # Meaning:
    # - `tuple(word)` splits a word into characters.
    # - The end-word token lets BPE learn that some pieces happen at word endings.
    #
    # Steps:
    # 1. Check that `word` is not empty.
    # 2. Convert characters to a tuple.
    # 3. Add `config.end_word_token` at the end.
    # 4. Return the tuple.
    #
    # Checks:
    # - The last piece should be `config.end_word_token`.
    # - The result for `"low"` should match the example above.
    raise NotImplementedError("Split a word into character pieces plus an end marker.")


# Count how often each neighboring pair appears.
# Expected return:
# - dictionary mapping `(left_piece, right_piece)` to count
def count_pair_frequencies(tokenized_words: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    """Count neighboring piece pairs."""
    # Goal:
    # Count all neighboring token pairs.
    #
    # Input:
    # - `tokenized_words`: dict[tuple[str, ...], int]
    # - Keys are tokenized words.
    # - Values are word counts.
    #
    # Output:
    # - `dict[tuple[str, str], int]`
    # - Each key is a neighboring pair like `("l", "o")`.
    # - Each value is how many times that pair appears, including repeated words.
    #
    # Small example:
    # ```python
    # tokenized_words = {("l", "o", "w", "</w>"): 2}
    # ```
    # Pair counts should include:
    # ```python
    # {("l", "o"): 2, ("o", "w"): 2, ("w", "</w>"): 2}
    # ```
    #
    # Method:
    # Use:
    # ```python
    # pair_counts: dict[tuple[str, str], int] = {}
    # for pieces, word_count in tokenized_words.items():
    #     for left, right in zip(pieces, pieces[1:]):
    #         pair_counts[(left, right)] = pair_counts.get((left, right), 0) + word_count
    # ```
    #
    # Meaning:
    # - `zip(pieces, pieces[1:])` gives neighboring pairs.
    # - `word_count` makes repeated words count more.
    #
    # Steps:
    # 1. Create an empty dictionary.
    # 2. Loop over tokenized words and their counts.
    # 3. Loop over neighboring pairs inside each word.
    # 4. Add the word count to that pair.
    # 5. Return the pair-count dictionary.
    #
    # Checks:
    # - Test the small example above by hand.
    # - Empty input should return an empty dictionary.
    raise NotImplementedError("Count all neighboring token pairs.")


# Merge one pair everywhere it appears.
# Expected return:
# - a new tokenized word dictionary after the merge
def apply_merge(
    tokenized_words: dict[tuple[str, ...], int],
    pair: tuple[str, str],
) -> dict[tuple[str, ...], int]:
    """Apply one BPE merge to all tokenized words."""
    # Goal:
    # Replace the chosen neighboring pair with one joined token.
    #
    # Input:
    # - `tokenized_words`: dict[tuple[str, ...], int]
    # - `pair`: tuple[str, str]
    #
    # Output:
    # - `dict[tuple[str, ...], int]`
    # - Same counts as input, but with the pair merged everywhere it appears.
    #
    # Small example:
    # ```python
    # tokenized_words = {("l", "o", "w", "</w>"): 2}
    # pair = ("l", "o")
    # ```
    # Return:
    # ```python
    # {("lo", "w", "</w>"): 2}
    # ```
    #
    # Method:
    # Use a while loop so you can skip over the right side of a merged pair.
    #
    # Merge token:
    # ```python
    # merged_token = pair[0] + pair[1]
    # ```
    #
    # Main loop shape:
    # ```python
    # new_pieces = []
    # index = 0
    # while index < len(pieces):
    #     if index < len(pieces) - 1 and (pieces[index], pieces[index + 1]) == pair:
    #         new_pieces.append(merged_token)
    #         index += 2
    #     else:
    #         new_pieces.append(pieces[index])
    #         index += 1
    # ```
    #
    # Steps:
    # 1. Build `merged_token`.
    # 2. Create an empty output dictionary.
    # 3. For each tokenized word, scan pieces from left to right.
    # 4. Replace each exact neighboring pair with the merged token.
    # 5. Keep the original word count.
    # 6. Return the new dictionary.
    #
    # Checks:
    # - The small example above should return `("lo", "w", "</w>")`.
    # - Counts should not change.
    # - Non-matching pairs should stay unchanged.
    raise NotImplementedError("Replace the chosen neighboring pair with one joined token.")


# Learn several BPE merges from the corpus.
# Expected return:
# - a list of merge pairs in the order they were learned
def learn_bpe_merges(texts: list[str], config: TokenizerInternalsConfig) -> list[tuple[str, str]]:
    """Learn BPE merges from text."""
    # Goal:
    # Learn the most common pair merges from the training texts.
    #
    # Input:
    # - `texts`: list[str]
    # - `config`: TokenizerInternalsConfig
    #
    # Output:
    # - `list[tuple[str, str]]`
    # - Merge pairs in the order they were learned.
    #
    # Example output:
    # ```python
    # [("l", "o"), ("lo", "w"), ("e", "r")]
    # ```
    #
    # Method:
    # First count tokenized words:
    # ```python
    # word_counts = Counter()
    # for text in texts:
    #     for word in text.split():
    #         word_counts[word] += 1
    # tokenized_words = {
    #     split_word_to_pieces(word, config): count
    #     for word, count in word_counts.items()
    # }
    # ```
    #
    # Then repeat:
    # ```python
    # pair_counts = count_pair_frequencies(tokenized_words)
    # best_pair = max(pair_counts, key=pair_counts.get)
    # merges.append(best_pair)
    # tokenized_words = apply_merge(tokenized_words, best_pair)
    # ```
    #
    # Stop early if `pair_counts` is empty.
    #
    # Steps:
    # 1. Count words from the text corpus with `Counter`.
    # 2. Convert each word into starting pieces.
    # 3. Create an empty `merges` list.
    # 4. Repeat up to `config.num_merges`.
    # 5. Count pair frequencies.
    # 6. Pick the most common pair.
    # 7. Store that pair.
    # 8. Apply the merge.
    # 9. Return the merge list.
    #
    # Checks:
    # - `len(merges)` should be at most `config.num_merges`.
    # - Every merge should be a pair of strings.
    # - Merges should come from corpus counts, not hard-coded answers.
    raise NotImplementedError("Learn the most common pair merges.")


# Build a token vocabulary from starting pieces and learned merges.
# Expected return:
# - token_to_id dictionary
# - id_to_token dictionary
def build_token_vocabulary(
    texts: list[str],
    merges: list[tuple[str, str]],
    config: TokenizerInternalsConfig,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build token id lookup tables."""
    # Goal:
    # Return token-to-id and id-to-token lookup tables.
    #
    # Input:
    # - `texts`: list[str]
    # - `merges`: list[tuple[str, str]]
    # - `config`: TokenizerInternalsConfig
    #
    # Output:
    # - `(token_to_id, id_to_token)`
    # - `token_to_id`: maps token string to integer id
    # - `id_to_token`: maps integer id back to token string
    #
    # Required special token:
    # - `config.unk_token`, usually `"<unk>"`
    #
    # Small example:
    # ```python
    # token_to_id["<unk>"] == 0
    # id_to_token[0] == "<unk>"
    # ```
    #
    # Method:
    # Build the final token pieces by:
    # 1. splitting each word into starting pieces
    # 2. applying every learned merge in order
    # 3. collecting every final token string into a set
    #
    # Use deterministic ids:
    # ```python
    # tokens = [config.unk_token] + sorted(vocab_tokens)
    # token_to_id = {token: index for index, token in enumerate(tokens)}
    # id_to_token = {index: token for token, index in token_to_id.items()}
    # ```
    #
    # Meaning:
    # - Sorting keeps ids stable across runs.
    # - The unknown token gives a fallback id for pieces not in the vocabulary.
    #
    # Steps:
    # 1. Create an empty set for vocabulary tokens.
    # 2. Loop over all words in all texts.
    # 3. Split each word into pieces.
    # 4. Apply every merge in order to that one word.
    # 5. Add the final pieces to the token set.
    # 6. Add `config.unk_token` as id `0`.
    # 7. Build both dictionaries.
    # 8. Return them.
    #
    # Checks:
    # - `config.unk_token` should be in `token_to_id`.
    # - Every id in `token_to_id` should exist in `id_to_token`.
    # - Vocabulary ids should be integers.
    raise NotImplementedError("Return token_to_id and id_to_token.")


# Encode new text with the learned merges.
# Expected return:
# - one list of integer token ids
def encode_text(
    text: str,
    merges: list[tuple[str, str]],
    token_to_id: dict[str, int],
    config: TokenizerInternalsConfig,
) -> list[int]:
    """Encode text into token ids."""
    # Goal:
    # Apply learned merges and return token ids.
    #
    # Input:
    # - `text`: str
    # - `merges`: list[tuple[str, str]]
    # - `token_to_id`: dict[str, int]
    # - `config`: TokenizerInternalsConfig
    #
    # Output:
    # - `list[int]`
    # - one id per final token piece
    #
    # Small example:
    # If `"low"` becomes pieces `("low", "</w>")`,
    # then return `[token_to_id["low"], token_to_id["</w>"]]`.
    #
    # Method:
    # For each word:
    # 1. split into starting pieces with `split_word_to_pieces`
    # 2. apply every merge in order
    # 3. map each final piece to an id
    #
    # Unknown-token fallback:
    # ```python
    # unk_id = token_to_id[config.unk_token]
    # token_id = token_to_id.get(piece, unk_id)
    # ```
    #
    # Steps:
    # 1. Create an empty id list.
    # 2. Split the text on spaces into words.
    # 3. Split each word into starting pieces.
    # 4. Apply merges in order.
    # 5. Convert each final piece into an id.
    # 6. Use the unknown id for missing pieces.
    # 7. Return the id list.
    #
    # Checks:
    # - The return value should be a list of integers.
    # - Unknown pieces should not crash encoding.
    # - Encoding an empty string should return an empty list.
    raise NotImplementedError("Apply merges and return token ids.")


# Decode token ids back into readable token pieces.
# Expected return:
# - a list of token strings, or one readable string
def decode_ids(ids: list[int], id_to_token: dict[int, str]) -> list[str]:
    """Decode token ids into token strings."""
    # Goal:
    # Map ids back to token strings.
    #
    # Input:
    # - `ids`: list[int]
    # - `id_to_token`: dict[int, str]
    #
    # Output:
    # - `list[str]`
    # - one token string per id
    #
    # Small example:
    # ```python
    # ids = [1, 2]
    # id_to_token = {1: "low", 2: "</w>"}
    # ```
    # Return:
    # ```python
    # ["low", "</w>"]
    # ```
    #
    # Method:
    # Use:
    # ```python
    # return [id_to_token[token_id] for token_id in ids]
    # ```
    #
    # If you want a safe version, use:
    # ```python
    # return [id_to_token.get(token_id, "<missing>") for token_id in ids]
    # ```
    #
    # Steps:
    # 1. Create an empty token list.
    # 2. Loop over ids.
    # 3. Look up each token string.
    # 4. Return the token list.
    #
    # Checks:
    # - The return length should match the input id length.
    # - Every output item should be a string.
    raise NotImplementedError("Map ids back to token strings.")


# Run the full tokenizer workflow.
# Expected return:
# - a summary dictionary with vocabulary size, merge count, ids, and decoded pieces
def run_tokenizer_demo(config: TokenizerInternalsConfig) -> dict[str, object]:
    """Run the tiny tokenizer workflow."""
    # Goal:
    # Build the corpus, learn merges, encode text, and return a summary.
    #
    # Input:
    # - `config`: TokenizerInternalsConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"vocab_size"`: int
    # - `"num_merges"`: int
    # - `"encoded_ids"`: list[int]
    # - `"decoded_pieces"`: list[str]
    # - `"tokenization_note"`: short plain-English string
    #
    # Demo text:
    # Use a small text such as:
    # ```python
    # demo_text = "lower newer"
    # ```
    #
    # Method:
    # Use:
    # ```python
    # texts = build_tiny_corpus()
    # merges = learn_bpe_merges(texts, config)
    # token_to_id, id_to_token = build_token_vocabulary(texts, merges, config)
    # encoded_ids = encode_text(demo_text, merges, token_to_id, config)
    # decoded_pieces = decode_ids(encoded_ids, id_to_token)
    # ```
    #
    # Good note shape:
    # - Say that BPE starts from characters.
    # - Say that it joins common neighboring pieces.
    # - Say that this can make common words shorter in token ids.
    #
    # Steps:
    # 1. Build the tiny corpus.
    # 2. Learn BPE merges.
    # 3. Build vocabulary dictionaries.
    # 4. Encode one demo text.
    # 5. Decode the ids back to pieces.
    # 6. Write a short note.
    # 7. Return the summary dictionary.
    #
    # Checks:
    # - The returned keys should match `check_metrics`.
    # - `vocab_size` should equal `len(token_to_id)`.
    # - `num_merges` should equal `len(merges)`.
    # - `decoded_pieces` should be readable token pieces.
    raise NotImplementedError("Build corpus, learn merges, encode text, and return a summary.")


# Check that the key tokenizer pieces were returned.
# Expected return:
# - True when the scaffold output has the needed fields
def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when tokenizer outputs are present."""
    required = {"vocab_size", "num_merges", "encoded_ids", "decoded_pieces", "tokenization_note"}
    return required.issubset(metrics)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the tokenizer internals scaffold."""
    config = TokenizerInternalsConfig()
    set_seed(config.random_state)

    try:
        metrics = run_tokenizer_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "tokenizer_outputs", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
