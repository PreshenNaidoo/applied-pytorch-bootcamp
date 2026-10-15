"""
============================================================
EXERCISE 17: LANGUAGE MODEL SEQUENCE PACKING
============================================================

WHAT YOU ARE BUILDING
You are building packed training sequences for language modeling.

Plain-language meaning:
- padding fills unused space in a batch
- too much padding wastes compute
- packing puts several short texts into one fixed-length sequence
- labels use `-100` where the loss should ignore a position

WHY THIS EXERCISE EXISTS
Real language model training tries to use tokens efficiently.
Sequence packing is one way to waste less space.
This exercise teaches:
- next-token labels
- fixed-length packed sequences
- separator tokens
- ignore labels
- padding waste vs packing efficiency

HOW THIS FITS INTO THE BOOTCAMP
After earlier language-model batching exercises,
this file teaches a practical data-efficiency idea:
- waste fewer token positions on padding
- use more of each batch for real training targets

What each step is doing in simple words:
1. build a few short text examples
2. tokenize them
3. create labels for next-token training
4. compare padded batches with packed batches
5. count how many positions are actually useful for learning
6. explain why packing can improve efficiency

Beginner intuition:
- padding is easy, but it can waste a lot of space
- packing tries to fill that wasted space with more useful tokens
- the model still learns next-token prediction, but now the batches use space better

Small example:
- if one text has length 5 and another has length 6,
- padding them to length 32 wastes a lot of positions
- packing can often place both into one longer fixed-length training sequence

DATASET
Use Tiny Shakespeare lines as the default text source.

Clear default:
- load `data/input.txt`
- split the text into non-empty lines
- keep the first 100 short lines for the first run
- if the file is missing, use the fallback list of short example sentences written inside this file

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- load short lines from Tiny Shakespeare
- fall back to a short local list only if the file is missing
- tokenize each text
- build padded batches
- build packed fixed-length sequences
- compare how many real token positions each method uses

INPUTS AND OUTPUTS
Padded batch:
- `input_ids`: `[batch_size, max_length]`
- `labels`: `[batch_size, max_length]`
Packed batch:
- `input_ids`: `[num_packed_sequences, packed_length]`
- `labels`: `[num_packed_sequences, packed_length]`
Ignore label:
- `-100`
- `CrossEntropyLoss` ignores this value by default when you set `ignore_index=-100`

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
No new model is required.
This exercise is mainly about language-model data preparation.

LOSS FUNCTION
Use next-token cross-entropy with `ignore_index=-100`.

METRICS TO TRACK
Track at least:
- number of real label tokens in padded batches
- number of real label tokens in packed batches
- padding waste
- one packing summary note

COMMON MISTAKES
- labels are not shifted by one token
- the model learns across document boundaries when it should not
- separator tokens are missing
- padding tokens still contribute to loss
- counting batch size instead of real training tokens

STEP-BY-STEP TODO
1. Build a small list of short text examples.
2. Tokenize each example.
3. Build padded batches.
4. Build packed fixed-length sequences.
5. Add separator tokens between examples.
6. Create next-token labels.
7. Use `-100` where loss should be ignored.
8. Count useful training tokens in padded and packed forms.
9. Compare padding waste and packing efficiency.

DEFINITION OF DONE
You are done when:
- packed sequences have fixed length
- labels are shifted for next-token prediction
- ignored labels use `-100`
- packing uses more real token positions than padding
- you can explain why packing saves compute
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict
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

@dataclass
class SequencePackingConfig:
    """Settings for language-model sequence packing."""
    max_length: int = 24
    packed_length: int = 32
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    sep_token: str = "<sep>"
    ignore_index: int = -100
    random_state: int = 42


class PackingGoalSummary(TypedDict):
    """Settings shown at the start of the exercise."""

    padded_shape: str
    packed_shape: str
    max_length: int
    packed_length: int
    ignore_index: int


class LanguageModelBatch(TypedDict):
    """Input ids and labels for language-model training."""

    input_ids: torch.Tensor
    labels: torch.Tensor


class PackingComparison(TypedDict):
    """Useful-token counts for padded and packed batches."""

    padded_useful_tokens: int
    packed_useful_tokens: int
    padded_total_positions: int
    packed_total_positions: int
    padded_waste_tokens: int
    packed_waste_tokens: int
    packing_efficiency_gain: float
    packing_note: str


# Beginner hints for this file:
# - padding wastes space when many sequences are short
# - packing tries to use that wasted space by placing short texts next to each other
# - separator tokens mark boundaries so you know where one text ends and the next begins
# - `-100` is a common ignore label for language-model loss
# - count real label tokens, not just tensor size, when you compare padded and packed batches
def describe_packing_goal(config: SequencePackingConfig) -> PackingGoalSummary:
    """Return the main packing settings."""
    return {
        "padded_shape": "[batch_size, max_length]",
        "packed_shape": "[num_packed_sequences, packed_length]",
        "max_length": config.max_length,
        "packed_length": config.packed_length,
        "ignore_index": config.ignore_index,
    }

# Return a short local list of texts for packing practice.
# Keep the texts short and different lengths so padding waste is easy to see.
def build_text_examples() -> list[str]:
    """Return short text examples for packing practice."""
    # Goal:
    # Return short text examples with different lengths.
    # Different lengths make padding waste easy to see.
    #
    # Input:
    # - No arguments.
    #
    # Output:
    # - `list[str]`
    # - Example:
    #   `["to be or not to be", "the king speaks", "a horse"]`
    # - Return at least 8 texts so packing has enough examples to combine.
    #
    # Method:
    # Use `Path.exists()` to check for `data/input.txt`.
    # Use `Path.read_text(encoding="utf-8")` to read the file if it exists.
    # Use `.splitlines()` to split it into lines.
    # Use `.strip()` to remove spaces at the start and end of each line.
    #
    # Steps:
    # 1. Set `data_path = PROJECT_ROOT / "data" / "input.txt"`.
    # 2. If the file exists, read it with `data_path.read_text(encoding="utf-8")`.
    # 3. Split the text into lines with `.splitlines()`.
    # 4. Keep only non-empty lines after `.strip()`.
    # 5. Keep short lines first, for example lines with at most 12 words.
    # 6. Return the first 100 short lines.
    # 7. If the file is missing or no short lines are found, return a fixed local list.
    #
    # Checks:
    # - `len(texts) >= 8`
    # - `isinstance(texts[0], str)`
    # - `texts[0]` should not be an empty string.
    #

    raise NotImplementedError("Return a small list of short text strings.")





def build_vocabulary(texts: list[str], config: SequencePackingConfig) -> dict[str, int]:
    """Build a simple token vocabulary."""
    # Goal:
    # Build a token-to-id dictionary from the text examples.
    # A token-to-id dictionary maps each word-like token to one integer.
    #
    # Input:
    # - `texts`: list[str]
    #   Example: `["to be or not to be", "the king speaks"]`
    # - `config`: SequencePackingConfig
    #
    # Output:
    # - `dict[str, int]`
    # - Must include:
    #   - `config.pad_token`
    #   - `config.unk_token`
    #   - `config.sep_token`
    # - Example:
    #   `{"<pad>": 0, "<unk>": 1, "<sep>": 2, "be": 3, "king": 4}`
    #
    # Method:
    # Use simple whitespace tokenization:
    # `tokens = text.lower().split()`
    #
    # This is a simple teaching tokenizer.
    # Real LLMs usually use subword tokenizers such as BPE or SentencePiece.
    #
    # Steps:
    # 1. Start the dictionary with the special tokens:
    #    `{config.pad_token: 0, config.unk_token: 1, config.sep_token: 2}`.
    # 2. Loop over each text in `texts`.
    # 3. Convert the text to lowercase with `text.lower()`.
    # 4. Split it into tokens with `.split()`.
    # 5. Collect all normal tokens in a `set`.
    # 6. Sort the set with `sorted(...)` so ids are stable across runs.
    # 7. Add each token that is not already in the dictionary.
    #
    # Checks:
    # - `token_to_id[config.pad_token] == 0`
    # - `token_to_id[config.unk_token] == 1`
    # - `token_to_id[config.sep_token] == 2`
    # - A known word from the texts should have an integer id.

    raise NotImplementedError("Build token-to-id vocabulary from the text examples.")



def tokenize_texts(texts: list[str], token_to_id: dict[str, int], config: SequencePackingConfig) -> list[list[int]]:
    """Turn text examples into token-id lists."""
    # Goal:
    # Turn each text into a list of token ids.
    # Do not add padding or separator tokens in this function.
    #
    # Input:
    # - `texts`: list[str]
    #   Example: `["to be", "unknown word"]`
    # - `token_to_id`: dict[str, int]
    #   Example: `{"<unk>": 1, "to": 7, "be": 8}`
    # - `config`: SequencePackingConfig
    #
    # Output:
    # - `list[list[int]]`
    # - Example:
    #   `[[7, 8], [1, 1]]`
    # - Each inner list is one text.
    # - The inner lists can have different lengths.
    #
    # Method:
    # Use the same simple tokenization as `build_vocabulary`:
    # `tokens = text.lower().split()`
    #
    # Use dictionary `.get(...)` for unknown tokens:
    # `token_to_id.get(token, unk_id)`
    #
    # Steps:
    # 1. Read `unk_id = token_to_id[config.unk_token]`.
    # 2. Create an empty list called `token_lists`.
    # 3. For each text, lowercase it and split it into tokens.
    # 4. Convert each token to an id with `token_to_id.get(token, unk_id)`.
    # 5. Skip empty token lists, because a language model cannot learn from an empty text.
    # 6. Return `token_lists`.
    #
    # Checks:
    # - `len(token_lists) > 0`
    # - `isinstance(token_lists[0][0], int)`
    # - If the text is `"to be"` and ids are `to=7`, `be=8`, the result should be `[7, 8]`.

    raise NotImplementedError("Tokenize each text example into token ids.")


# Build next-token labels from input ids.
# Main idea:
# - current token = clue
# - next token = answer
# Use `ignore_index` for positions that should not contribute to loss.
def build_next_token_labels(input_ids: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Return next-token labels for a batch of input ids."""
    # Goal:
    # Build labels for next-token prediction.
    # The current token is the clue, and the next token is the answer.
    #
    # Input:
    # - `input_ids`: torch.Tensor with shape `[batch_size, sequence_length]`
    #   Example:
    #   `torch.tensor([[10, 11, 12, 0]])`
    # - `ignore_index`: int
    #   Usually `-100`.
    #
    # Output:
    # - `torch.Tensor` with the same shape as `input_ids`
    # - Example output:
    #   `torch.tensor([[11, 12, 0, -100]])`
    #
    # Shape rule:
    # - Input shape: `[batch_size, sequence_length]`
    # - Label shape: `[batch_size, sequence_length]`
    #
    # Method:
    # Use `torch.full_like(input_ids, fill_value=ignore_index)` to create the label tensor.
    # Then copy the next token into each earlier label position.
    #
    # Steps:
    # 1. Create `labels = torch.full_like(input_ids, fill_value=ignore_index)`.
    # 2. Fill every label except the last one:
    #    `labels[:, :-1] = input_ids[:, 1:]`
    # 3. Leave the last label as `ignore_index`.
    #    There is no next token after the last position.
    # 4. Return `labels`.
    #
    # Checks:
    # - If `input_ids` is `[[10, 11, 12, 0]]`, labels should be `[[11, 12, 0, -100]]`.
    # - `labels.shape == input_ids.shape`

    raise NotImplementedError("Shift labels one step left and ignore positions that should not train.")

# Build the simple padded version of the batch.
# This is the baseline that may waste space.
def build_padded_batch(token_lists: list[list[int]], config: SequencePackingConfig, pad_id: int) -> LanguageModelBatch:
    """Return padded input ids and labels."""
    # Goal:
    # Build a padded batch for next-token language-model training.
    # This is the simple baseline that may waste many positions on padding.
    #
    # Input:
    # - `token_lists`: list[list[int]]
    #   Example: `[[5, 6, 7], [8, 9]]`
    # - `config`: SequencePackingConfig
    # - `pad_id`: int
    #   Example: `0`
    #
    # Output:
    # Return this exact dictionary:
    # - `"input_ids"`: tensor with shape `[num_texts, config.max_length]`
    # - `"labels"`: tensor with shape `[num_texts, config.max_length]`
    #
    # Small example with `max_length = 5`, `pad_id = 0`, and `ignore_index = -100`:
    #
    # ```text
    # token ids:  [5, 6, 7]
    # input_ids:  [5, 6, 7, 0, 0]
    # labels:     [6, 7, -100, -100, -100]
    # ```
    #
    # Why the label at position 2 is `-100`:
    # the next token would be padding, and padding should not train the model.
    #
    # Method:
    # Use `torch.full(...)` to create fixed-size tensors:
    # - `torch.full((len(token_lists), config.max_length), pad_id, dtype=torch.long)`
    # - `torch.full((len(token_lists), config.max_length), config.ignore_index, dtype=torch.long)`
    #
    # Steps:
    # 1. Create `input_ids` filled with `pad_id`.
    # 2. Create `labels` filled with `config.ignore_index`.
    # 3. For each token list, keep at most `config.max_length` tokens.
    # 4. Copy those tokens into the matching row of `input_ids`.
    # 5. For next-token labels, use each next real token as the answer:
    #    if the kept tokens are `[5, 6, 7]`, labels should start `[6, 7]`.
    # 6. Do not create labels that point to padding.
    # 7. Return `{"input_ids": input_ids, "labels": labels}`.
    #
    # Checks:
    # - `input_ids.shape == (len(token_lists), config.max_length)`
    # - `labels.shape == input_ids.shape`
    # - `(labels != config.ignore_index).sum()` counts real next-token targets.
    #

    raise NotImplementedError("Build a padded language-model batch.")



# Build the packed version of the batch.
# This tries to use more token positions for real learning instead of padding.
def build_packed_batch(
    token_lists: list[list[int]],
    config: SequencePackingConfig,
    pad_id: int,
    sep_id: int,
) -> LanguageModelBatch:
    """Return packed input ids and labels."""
    # Goal:
    # Build fixed-length packed sequences.
    # Packing puts several short texts into one row before adding padding.
    #
    # Input:
    # - `token_lists`: list[list[int]]
    #   Example: `[[5, 6], [7, 8, 9]]`
    # - `config`: SequencePackingConfig
    # - `pad_id`: int
    #   Example: `0`
    # - `sep_id`: int
    #   Example: `2`
    #
    # Output:
    # Return this exact dictionary:
    # - `"input_ids"`: tensor with shape `[num_packed_sequences, config.packed_length]`
    # - `"labels"`: tensor with shape `[num_packed_sequences, config.packed_length]`
    #
    # Small example with `packed_length = 8`, `sep_id = 2`, `pad_id = 0`:
    #
    # ```text
    # text 1 ids: [5, 6]
    # text 2 ids: [7, 8, 9]
    # packed input row: [5, 6, 2, 7, 8, 9, 2, 0]
    # ```
    #
    # Label rule:
    # - normal token positions predict the next token
    # - a token before `<sep>` may predict `<sep>`
    # - a `<sep>` position should use `-100`
    # - a padding position should use `-100`
    #
    # Example labels for the row above:
    #
    # ```text
    # [6, 2, -100, 8, 9, 2, -100, -100]
    # ```
    #
    # Method:
    # Use plain Python lists first.
    # Convert the final rows to tensors with:
    # `torch.tensor(rows, dtype=torch.long)`
    #
    # Steps:
    # 1. Start an empty `current` list.
    # 2. For each token list, make `example = tokens + [sep_id]`.
    # 3. If `example` is longer than `config.packed_length`, cut it to that length.
    # 4. If `current + example` would be too long, pad `current` and save it as one row.
    # 5. Start a new `current` row.
    # 6. Add `example` to `current`.
    # 7. After the loop, save the last `current` row if it is not empty.
    # 8. Pad every row to `config.packed_length` with `pad_id`.
    # 9. Build labels from each padded row using the label rule above.
    # 10. Return `{"input_ids": input_ids, "labels": labels}`.
    #
    # Checks:
    # - `input_ids.ndim == 2`
    # - `input_ids.shape[1] == config.packed_length`
    # - `labels.shape == input_ids.shape`
    # - No label should teach the model to predict the first token of a new text from `<sep>`.

    raise NotImplementedError("build_packed_batch.")





def count_useful_label_tokens(labels: torch.Tensor, ignore_index: int) -> int:
    """Count labels that will contribute to the loss."""
    # Goal:
    # Count how many label positions will train the model.
    # A useful label is any label that is not `ignore_index`.
    #
    # Input:
    # - `labels`: torch.Tensor
    #   Shape can be `[batch_size, sequence_length]`.
    #   Example:
    #   `torch.tensor([[6, 7, -100], [3, -100, -100]])`
    # - `ignore_index`: int
    #   Usually `-100`.
    #
    # Output:
    # - `int`
    # - Example output for the labels above: `3`
    #
    # Method:
    # Use a boolean mask:
    # `labels != ignore_index`
    #
    # Use `.sum().item()` to turn the count into a Python number.
    #
    # Steps:
    # 1. Build `mask = labels != ignore_index`.
    # 2. Count True values with `mask.sum().item()`.
    # 3. Convert the result to `int`.
    # 4. Return the count.
    #
    # Checks:
    # - For `[[6, 7, -100], [3, -100, -100]]`, the answer should be `3`.

    #labels is shape [batch_size, sequence_length]
    raise NotImplementedError("Count labels that are not ignore_index.")


# Compare padded and packed batches fairly.
# The best comparison is:
# - how many label positions actually train the model
# not only:
# - how large the tensors are
def compare_padding_and_packing(
    padded_batch: LanguageModelBatch,
    packed_batch: LanguageModelBatch,
    config: SequencePackingConfig,
    sep_id: int | None = None,
) -> PackingComparison:
    """Compare useful tokens in padded and packed batches."""
    # Goal:
    # Compare padded batching and packed batching using useful label tokens.
    # Useful label tokens are the positions that actually contribute to the loss.
    #
    # Input:
    # - `padded_batch`: dictionary with `"input_ids"` and `"labels"`
    # - `packed_batch`: dictionary with `"input_ids"` and `"labels"`
    # - `config`: SequencePackingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"padded_useful_tokens"`: int
    # - `"packed_useful_tokens"`: int
    # - `"padded_total_positions"`: int
    # - `"packed_total_positions"`: int
    # - `"padded_waste_tokens"`: int
    # - `"packed_waste_tokens"`: int
    # - `"packing_efficiency_gain"`: float
    # - `"packing_note"`: str
    #
    # Method:
    # Use `count_useful_label_tokens(labels, config.ignore_index)` for useful labels.
    # Use `.numel()` to count total tensor positions.
    #
    # Formula:
    # - `waste_tokens = total_positions - useful_tokens`
    # - `packing_efficiency_gain = packed_useful_tokens / padded_useful_tokens`
    #
    # Plain meaning:
    # if the gain is `1.25`, packing gives 25 percent more useful labels.
    #
    # Steps:
    # 1. Count useful labels in `padded_batch["labels"]`.
    # 2. Count useful labels in `packed_batch["labels"]`.
    # 3. Count total positions with `.numel()` for each label tensor.
    # 4. Compute waste for padded and packed batches.
    # 5. Compute efficiency gain.
    # 6. Build a short note, such as:
    #    `"Packing used 40 useful labels instead of 28."`
    # 7. Return the dictionary with the exact keys listed above.
    #
    # Checks:
    # - `packed_useful_tokens >= padded_useful_tokens` for the default examples.
    # - `packing_note` should mention the concrete useful-token counts.

    raise NotImplementedError("Return useful token counts and a plain-language comparison.")


    # `.numel()` counts every position in the tensor.
    # For shape `[100, 24]`, this is `100 * 24 = 2400`.
    # It does not mean "number of padding tokens".


    



def run_packing_demo(config: SequencePackingConfig) -> PackingComparison:
    """Run the full sequence-packing workflow."""
    # Goal:
    # Run the whole sequence-packing workflow from text examples to comparison metrics.
    #
    # Input:
    # - `config`: SequencePackingConfig
    #
    # Output:
    # - Return the same dictionary as `compare_padding_and_packing`.
    # - It must include:
    #   - `"padded_useful_tokens"`
    #   - `"packed_useful_tokens"`
    #   - `"packing_note"`
    #
    # Method:
    # Use the helper functions in this file in order.
    # Do not rebuild tokenization logic inside this function.
    #
    # Steps:
    # 1. Build texts with `build_text_examples()`.
    # 2. Build `token_to_id` with `build_vocabulary(texts, config)`.
    # 3. Build token lists with `tokenize_texts(texts, token_to_id, config)`.
    # 4. Read `pad_id = token_to_id[config.pad_token]`.
    # 5. Read `sep_id = token_to_id[config.sep_token]`.
    # 6. Build the padded batch with `build_padded_batch(token_lists, config, pad_id)`.
    # 7. Build the packed batch with `build_packed_batch(token_lists, config, pad_id, sep_id)`.
    # 8. Return `compare_padding_and_packing(padded_batch, packed_batch, config, sep_id)`.
    #
    # Checks:
    # - The returned dictionary should pass `check_metrics(metrics)`.

    raise NotImplementedError("Run padded and packed batching and compare them.")






    # example output:
    # padded_batch shape: torch.Size([100, 24])
    # packed_batch shape: torch.Size([13, 32])
    # {'padded_useful_tokens': 200, 'packed_useful_tokens': 200, 'padded_total_positions': 2400, 'packed_total_positions': 416, 'padded_waste_tokens': 2200, 'packed_waste_tokens': 216, 'packing_efficiency_gain': 5.76923076923077, 'packing_note': 'Packing used 200 useful non-separator labels instead of 200.'}   
    # Plain meaning: packing uses tensor space about 5.77x more efficiently than padding for this data.




def check_metrics(metrics: dict[str, float | int | str]) -> bool:
    """Return True when packing comparison values are present."""
    required = {"padded_useful_tokens", "packed_useful_tokens", "packing_note"}
    return required.issubset(metrics)

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
    """Run the sequence-packing scaffold one step at a time."""
    config = SequencePackingConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 5
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    _, ok = report_task_progress("describe_packing_goal", describe_packing_goal, config)
    if not count_or_stop(ok):
        return
    texts, ok = report_task_progress("build_text_examples", build_text_examples)
    if not count_or_stop(ok):
        return
    vocab, ok = report_task_progress("build_vocabulary", build_vocabulary, texts, config)
    if not count_or_stop(ok):
        return
    token_lists, ok = report_task_progress("tokenize_texts", tokenize_texts, texts, vocab, config)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_packing_demo", run_packing_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "packing summary", float(int(passed)), "packing comparison keys present")
if __name__ == "__main__":
    main()
