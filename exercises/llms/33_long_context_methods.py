"""
============================================================
EXERCISE 33: LONG CONTEXT METHODS
============================================================

WHAT YOU ARE BUILDING
You are building small tests that show what long-context work is trying to solve.

Plain-language meanings:
- context length is how many tokens a model can read at once
- long context means the model can read many tokens in one request
- RoPE scaling changes position values so a model can handle longer positions
- retrieval means finding useful outside text instead of putting everything in context
- needle test means hiding one answer in a long text and checking if the model can use it

WHY THIS EXERCISE EXISTS
Frontier labs care about long context, but longer context is not magic.
A model must both fit the tokens and actually use information far away.

This exercise teaches:
- context length limits
- simple RoPE scaling intuition
- retrieval versus longer context
- building a needle-in-a-haystack style test
- measuring whether the far-away answer was found

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises introduced RoPE and retrieval.
This file connects those ideas to long-context evaluation.

Beginner intuition:
- adding more pages to the prompt is only useful if the model can find the right page
- retrieval tries to choose the useful pages before the model reads
- long context tries to let the model read more at once

DATASET
Use Tiny Shakespeare as the recommended source for a tiny "needle in a haystack" dataset.

Clear default:
- split Tiny Shakespeare into many short text chunks
- most chunks are filler
- one chunk contains the answer
- the query asks for that hidden answer

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load Tiny Shakespeare from `data/`
- if the file is missing, build a list of fallback text chunks in Python
- place the secret answer in one chosen chunk
- create a query that should retrieve or find that answer
- compare:
  - using all chunks
  - using only retrieved chunks

INPUTS AND OUTPUTS
Input:
- `chunks`: list of text chunks
- `query`: one question string
- `secret_answer`: known answer string

Output:
- full-context success flag
- retrieval success flag
- RoPE scaling summary
- long-context tradeoff note

STEP-BY-STEP TODO
1. Build a long toy context.
2. Hide a specific answer inside it.
3. Build a simple retrieval function.
4. Build a simple answer finder.
5. Compare full context with retrieved context.
6. Compute a position-scaling summary.
7. Write a note about long context tradeoffs.

DEFINITION OF DONE
You are done when:
- the hidden answer can be placed at different positions
- retrieval returns a smaller context
- the answer check reports success or failure
- the note explains long context versus retrieval
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


RopeScalingSummary = dict[str, float | int]
LongContextComparison = dict[str, int | bool | str]
LongContextSummary = dict[str, int | bool | str | RopeScalingSummary]


# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for long-context tests.
# Expected return:
# - a config object with context size, hidden answer, and query settings
@dataclass
class LongContextConfig:
    """Settings for long-context toy tests."""

    num_chunks: int = 40
    needle_chunk_index: int = 31
    query: str = "What is the secret code?"
    secret_answer: str = "blue-17"
    base_context_length: int = 2048
    target_context_length: int = 8192
    random_state: int = 42


# Build a long context with one hidden answer.
# Expected return:
# - list of text chunks
def build_needle_context(config: LongContextConfig) -> list[str]:
    """Return context chunks with one hidden answer."""
    # Goal:
    # - Build a list of text chunks.
    # - Put the secret answer in exactly one chosen chunk.
    #
    # Inputs:
    # - `config.num_chunks`: how many chunks to create.
    # - `config.needle_chunk_index`: where to place the secret answer.
    # - `config.secret_answer`: the answer string, such as `"blue-17"`.
    # - Example with `num_chunks=4` and `needle_chunk_index=2`:
    #   - chunk 0: filler text
    #   - chunk 1: filler text
    #   - chunk 2: contains `"The secret code is blue-17."`
    #   - chunk 3: filler text
    #
    # Output:
    # - Return `list[str]` with length `config.num_chunks`.
    # - Exactly one chunk should contain `config.secret_answer`.
    # - Each chunk should be a normal text string.
    #
    # Method:
    # - First try to load a local text file:
    #   `text_path = PROJECT_ROOT / "data" / "tiny_shakespeare.txt"`
    #   `text = text_path.read_text(encoding="utf-8")`
    # - If the file is missing, use fallback chunks like:
    #   `"This is filler text about kings, letters, and journeys."`
    # - A simple chunking method is:
    #   `words = text.split()`
    #   `chunk_size = max(1, len(words) // config.num_chunks)`
    #   `chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]`
    # - Trim or pad the list so it has exactly `config.num_chunks` chunks.
    # - Put the answer into the needle chunk with a clear pattern:
    #   `chunks[index] += f" The secret code is {config.secret_answer}."`
    # - That exact phrase lets `find_answer_in_context(...)` find the answer later.
    #
    # Steps:
    # 1. Assert `config.num_chunks > 0`.
    # 2. Assert `0 <= config.needle_chunk_index < config.num_chunks`.
    # 3. Load Tiny Shakespeare text or use fallback text.
    # 4. Split the text into chunks.
    # 5. Pad with fallback chunks if there are too few chunks.
    # 6. Trim to exactly `config.num_chunks`.
    # 7. Add the exact phrase `"The secret code is {config.secret_answer}."`
    #    to `chunks[config.needle_chunk_index]`.
    # 8. Return the chunks.
    #
    # Checks:
    # - `len(chunks)` should equal `config.num_chunks`.
    # - `config.secret_answer in chunks[config.needle_chunk_index]` should be True.
    # - The answer should appear in exactly one chunk.
    #
    raise NotImplementedError("Build chunks and hide the answer in one chunk.")


# Retrieve chunks that look useful for the query.
# Expected return:
# - smaller list of chunks
def retrieve_relevant_chunks(query: str, chunks: list[str], top_k: int) -> list[str]:
    """Return the most relevant chunks."""
    # Goal:
    # - Pick the chunks that look most useful for answering the query.
    # - Keep this as a simple keyword retriever, not a neural retriever.
    #
    # Inputs:
    # - `query`: one question string, such as `"What is the secret code?"`.
    # - `chunks`: list of context chunks from `build_needle_context(...)`.
    # - `top_k`: number of chunks to return.
    # - Example:
    #   - query words: `{"what", "is", "the", "secret", "code"}`
    #   - a chunk containing `"The secret code is blue-17."` should score highly.
    #
    # Output:
    # - Return `list[str]`.
    # - Length should be `min(top_k, len(chunks))`.
    # - The original chunk strings should be returned, not indexes.
    #
    # Method:
    # - Use a tiny word-overlap score:
    #   `query_words = set(query.lower().replace("?", "").split())`
    #   `chunk_words = set(chunk.lower().replace(".", "").split())`
    #   `score = len(query_words & chunk_words)`
    # - Also give a bonus if the phrase `"secret code"` appears in the chunk.
    # - Sort chunks by score from high to low.
    # - Keep original order as a tie-breaker so the result is stable.
    #
    # Steps:
    # 1. Assert `top_k > 0`.
    # 2. Turn the query into lowercase words.
    # 3. For each chunk, compute a score.
    # 4. Sort `(score, index, chunk)` values by highest score first and lowest index first.
    # 5. Return the top `min(top_k, len(chunks))` chunk strings.
    #
    # Checks:
    # - `len(retrieved)` should be no larger than `top_k`.
    # - With the default hidden phrase, one retrieved chunk should contain `"secret code"`.
    #
    raise NotImplementedError("Return a small set of relevant chunks.")


# Find the answer in a context.
# Expected return:
# - answer string or empty string
def find_answer_in_context(query: str, chunks: list[str]) -> str:
    """Return the answer found in the provided context."""
    # Goal:
    # - Find the secret answer in the provided chunks.
    # - This is a toy answer finder, not a real language model.
    #
    # Inputs:
    # - `query`: question string. For this exercise it asks for the secret code.
    # - `chunks`: context chunks.
    # - Expected hidden phrase:
    #   `"The secret code is blue-17."`
    #
    # Output:
    # - Return the answer string, such as `"blue-17"`.
    # - Return `""` if no answer is found.
    #
    # Method:
    # - Search each chunk for the marker:
    #   `marker = "secret code is "`
    # - Use lowercase for matching, but return the answer from the original text.
    # - After the marker, read until the next period:
    #   `answer = chunk[start:end].strip()`
    # - Example:
    #   - chunk: `"Noise. The secret code is blue-17. More noise."`
    #   - answer: `"blue-17"`
    #
    # Steps:
    # 1. Ignore the query except for checking that it asks about the secret code.
    # 2. Loop over chunks.
    # 3. Find `"secret code is "` in each chunk.
    # 4. If found, slice the text after the marker up to the next period.
    # 5. Strip whitespace and return the answer.
    # 6. If no chunk contains the marker, return `""`.
    #
    # Checks:
    # - `find_answer_in_context(query, ["The secret code is blue-17."])` should return `"blue-17"`.
    # - `find_answer_in_context(query, ["no answer here"])` should return `""`.
    #
    raise NotImplementedError("Find the hidden answer from the context chunks.")


# Check whether an answer is correct.
# Expected return:
# - True when the answer matches the known secret
def check_answer(answer: str, config: LongContextConfig) -> bool:
    """Return True when the answer is correct."""
    # Goal:
    # - Check whether a found answer matches the known secret answer.
    #
    # Inputs:
    # - `answer`: string returned by `find_answer_in_context(...)`.
    # - `config.secret_answer`: the correct answer.
    # - Example:
    #   - answer: `" blue-17 "`
    #   - secret: `"blue-17"`
    #   - result: True after stripping spaces
    #
    # Output:
    # - Return `True` if the cleaned answer equals `config.secret_answer`.
    # - Return `False` otherwise.
    #
    # Method:
    # - Use:
    #   `answer.strip().lower() == config.secret_answer.strip().lower()`
    # - Lowercasing makes the check focus on the answer, not capitalization.
    #
    # Steps:
    # 1. Strip whitespace from both strings.
    # 2. Lowercase both strings.
    # 3. Compare them.
    # 4. Return the bool result.
    #
    # Checks:
    # - `"blue-17"` should pass.
    # - `""` should fail.
    #
    raise NotImplementedError("Compare the answer with the known secret.")


# Compute a simple RoPE scaling factor.
# Expected return:
# - dictionary with base length, target length, and scale factor
def compute_rope_scaling_summary(config: LongContextConfig) -> RopeScalingSummary:
    """Return a simple RoPE scaling summary."""
    # Goal:
    # - Compute a simple position-scaling summary for RoPE.
    # - RoPE scaling is one way to stretch position handling to longer contexts.
    #
    # Inputs:
    # - `config.base_context_length`: original context length, such as 2048.
    # - `config.target_context_length`: desired context length, such as 8192.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"base_context_length"`: int
    #   - `"target_context_length"`: int
    #   - `"scale_factor"`: float
    # - With defaults, scale factor is `8192 / 2048 = 4.0`.
    #
    # Method:
    # - Formula:
    #   `scale_factor = config.target_context_length / config.base_context_length`
    # - This is only a simple summary. Real RoPE scaling methods can use extra details,
    #   but the main idea here is that positions are stretched to cover a longer range.
    #
    # Steps:
    # 1. Assert `config.base_context_length > 0`.
    # 2. Assert `config.target_context_length > 0`.
    # 3. Compute the scale factor.
    # 4. Return the dictionary with the exact keys above.
    #
    # Checks:
    # - With defaults, `summary["scale_factor"]` should be `4.0`.
    #
    raise NotImplementedError("Compute target_length / base_length and return a summary.")


# Compare full context with retrieval.
# Expected return:
# - dictionary with token or chunk counts and success flags
def compare_long_context_and_retrieval(config: LongContextConfig) -> LongContextComparison:
    """Compare full context and retrieved context."""
    # Goal:
    # - Compare two ways to answer the query:
    #   - give the answer finder all chunks
    #   - give the answer finder only retrieved chunks
    #
    # Inputs:
    # - `config`: settings for the hidden-answer test.
    # - `build_needle_context(config)` returns `config.num_chunks` chunks.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"num_chunks"`: total chunks in the full context
    #   - `"retrieved_chunks"`: number of chunks used after retrieval
    #   - `"full_context_answer"`: answer found using all chunks
    #   - `"retrieved_answer"`: answer found using retrieved chunks
    #   - `"full_context_success"`: bool
    #   - `"retrieval_success"`: bool
    #   - `"needle_chunk_index"`: where the secret was hidden
    # - This dictionary uses chunk counts, not token counts, to keep the exercise simple.
    #
    # Method:
    # - Use:
    #   `chunks = build_needle_context(config)`
    #   `retrieved = retrieve_relevant_chunks(config.query, chunks, top_k=5)`
    #   `full_answer = find_answer_in_context(config.query, chunks)`
    #   `retrieved_answer = find_answer_in_context(config.query, retrieved)`
    #   `full_success = check_answer(full_answer, config)`
    #   `retrieval_success = check_answer(retrieved_answer, config)`
    # - `top_k=5` means retrieval should use a smaller context than the full 40 chunks.
    #
    # Steps:
    # 1. Build the hidden-answer context.
    # 2. Retrieve the top 5 chunks.
    # 3. Find the answer in all chunks.
    # 4. Find the answer in retrieved chunks.
    # 5. Check both answers against the secret.
    # 6. Return the dictionary with the exact keys above.
    #
    # Checks:
    # - `results["retrieved_chunks"] < results["num_chunks"]` should be True.
    # - `results["full_context_success"]` should be True when the answer finder works.
    # - Retrieval success depends on whether the retriever found the needle chunk.
    #
    raise NotImplementedError("Return success checks for full and retrieved context.")


# Write a note about the tradeoff.
# Expected return:
# - plain-language string
def write_long_context_note(results: LongContextComparison) -> str:
    """Explain long context and retrieval tradeoffs."""
    # Goal:
    # - Explain the tradeoff between giving the model everything and retrieving fewer chunks.
    #
    # Inputs:
    # - `results`: output from `compare_long_context_and_retrieval(config)`.
    # - Important keys:
    #   `"num_chunks"`, `"retrieved_chunks"`, `"full_context_success"`, and `"retrieval_success"`.
    #
    # Output:
    # - Return one short plain-language string.
    # - It should say:
    #   - long context can read more chunks at once
    #   - retrieval can reduce the number of chunks the model reads
    #   - retrieval can fail if it misses the chunk with the answer
    #
    # Method:
    # - Use concrete numbers:
    #   `results["num_chunks"]`
    #   `results["retrieved_chunks"]`
    # - Example sentence:
    #   `"The full context checked 40 chunks, while retrieval checked 5 chunks."`
    #
    # Steps:
    # 1. Read the full and retrieved chunk counts.
    # 2. Read the two success flags.
    # 3. Write one or two clear sentences.
    # 4. Return the note.
    #
    # Checks:
    # - The note should include the words `"retrieval"` and `"long context"`.
    # - The note should include at least one chunk count.
    #
    raise NotImplementedError("Write a note about long context versus retrieval.")


# Run the full long-context workflow.
# Expected return:
# - summary dictionary with checks and note
def run_long_context_demo(config: LongContextConfig) -> LongContextSummary:
    """Run the long-context toy workflow."""
    # Goal:
    # - Run the hidden-answer comparison and add a RoPE scaling summary and note.
    #
    # Inputs:
    # - `config`: settings for the long-context toy test.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - all keys from `compare_long_context_and_retrieval(config)`
    #   - `"rope_scaling"`: output of `compute_rope_scaling_summary(config)`
    #   - `"long_context_note"`: output of `write_long_context_note(results)`
    #
    # Method:
    # - Use:
    #   `results = compare_long_context_and_retrieval(config)`
    #   `rope_scaling = compute_rope_scaling_summary(config)`
    #   `long_context_note = write_long_context_note(results)`
    #   `return {**results, "rope_scaling": rope_scaling, "long_context_note": long_context_note}`
    #
    # Steps:
    # 1. Run the full-context versus retrieval comparison.
    # 2. Compute RoPE scaling.
    # 3. Write the tradeoff note.
    # 4. Merge everything into one summary dictionary.
    # 5. Return it.
    #
    # Checks:
    # - `summary["rope_scaling"]["scale_factor"]` should be `4.0` with defaults.
    # - `check_metrics(summary)` should return True.
    #
    raise NotImplementedError("Build context, retrieve chunks, check answer, and return summary.")


# Check that the summary includes key outputs.
# Expected return:
# - True when required fields are present
def check_metrics(metrics: LongContextSummary) -> bool:
    """Return True when long-context outputs are present."""
    required = {"full_context_success", "retrieval_success", "rope_scaling", "long_context_note"}
    return required.issubset(metrics)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the long-context scaffold."""
    config = LongContextConfig()
    set_seed(config.random_state)

    try:
        metrics = run_long_context_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "long_context_outputs", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
