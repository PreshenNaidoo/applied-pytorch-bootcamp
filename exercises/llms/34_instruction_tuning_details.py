"""
============================================================
EXERCISE 34: INSTRUCTION TUNING DETAILS
============================================================

WHAT YOU ARE BUILDING
You are building the data formatting pieces needed for assistant-style instruction tuning.

Plain-language meanings:
- instruction tuning teaches a model to follow prompts
- a chat template turns messages into one training string
- a user message asks for something
- an assistant message is the answer the model should learn to produce
- assistant-only loss means loss is computed only on assistant tokens
- masking means marking some label positions as ignored

WHY THIS EXERCISE EXISTS
Many fine-tuning bugs come from formatting, not model math.
If the model trains on user tokens by mistake, it learns the wrong task.

This exercise teaches:
- chat message format
- chat templates
- assistant-only labels
- loss masks
- padding masks
- checking that user tokens are ignored

HOW THIS FITS INTO THE BOOTCAMP
The SFT exercise introduced the idea.
This file focuses on the details that make SFT correct.

Beginner intuition:
- the prompt is the question
- the assistant answer is what the model should learn
- the model should read the question but not be punished for failing to predict it

DATASET
Use a tiny local chat dataset.

Clear default:
- 4 to 8 chat examples written inside this file
- each example should include:
  - one user message
  - one assistant answer

Example:
- user: `Explain what a tensor is.`
- assistant: `A tensor is a grid of numbers that PyTorch can compute with.`

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- write a list of chat dictionaries
- apply one chat template to every example
- tokenize the formatted text
- find the assistant-answer span
- build labels that ignore user tokens
- pad examples into one batch

INPUTS AND OUTPUTS
Input:
- chat messages with roles and text

Output batch:
- `input_ids`
  - shape: `[batch_size, sequence_length]`
- `labels`
  - shape: `[batch_size, sequence_length]`
  - uses `-100` outside assistant answer tokens
- `attention_mask`
  - shape: `[batch_size, sequence_length]`

Important:
- user tokens should be readable by the model
- user tokens should not contribute to the loss

STEP-BY-STEP TODO
1. Build tiny chat examples.
2. Apply one clear chat template.
3. Tokenize formatted text.
4. Find assistant answer spans.
5. Build labels that ignore user tokens.
6. Pad examples into a batch.
7. Check that only assistant tokens train.
8. Return a formatting summary.

DEFINITION OF DONE
You are done when:
- chat examples have user and assistant messages
- formatted strings are returned
- labels use ignore index outside assistant answers
- the summary proves user tokens are masked
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


ChatMessage = dict[str, str]
ChatExample = dict[str, list[ChatMessage]]
InstructionBatch = dict[str, torch.Tensor]
InstructionSummary = dict[str, int | bool | str | tuple[int, int]]


# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for instruction-tuning format checks.
# Expected return:
# - a config object with special tags and ignore-label settings
@dataclass
class InstructionTuningDetailsConfig:
    """Settings for instruction tuning format checks."""

    ignore_index: int = -100
    pad_token: str = "<pad>"
    user_tag: str = "<user>"
    assistant_tag: str = "<assistant>"
    random_state: int = 42


# Build tiny chat examples.
# Expected return:
# - list of examples, each with messages containing roles and content
def build_chat_examples() -> list[ChatExample]:
    """Return tiny chat examples."""
    # Goal:
    # - Return a tiny local instruction-tuning dataset.
    # - Each example should have one user message and one assistant message.
    #
    # Inputs:
    # - No direct arguments.
    #
    # Output:
    # - Return `list[ChatExample]`.
    # - Each example must look like:
    #   `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
    # - Use roles exactly as `"user"` and `"assistant"`.
    # - Return at least 4 examples so padding and batching are worth checking.
    #
    # Method:
    # - Build the list by hand in Python.
    # - Good beginner examples:
    #   - user: `"Explain what a tensor is."`
    #   - assistant: `"A tensor is a grid of numbers that PyTorch can compute with."`
    # - Keep answers short so token and label checks are easy to inspect.
    #
    # Steps:
    # 1. Create a list.
    # 2. Add at least 4 dictionaries.
    # 3. In each dictionary, add a `"messages"` list.
    # 4. Put the user message first and assistant message second.
    # 5. Return the list.
    #
    # Checks:
    # - `len(examples) >= 4` should be True.
    # - `examples[0]["messages"][0]["role"]` should be `"user"`.
    # - `examples[0]["messages"][1]["role"]` should be `"assistant"`.
    #
    raise NotImplementedError("Return chat examples with user and assistant messages.")


# Apply a chat template.
# Expected return:
# - one formatted string
def apply_chat_template(example: ChatExample, config: InstructionTuningDetailsConfig) -> str:
    """Format one chat example as training text."""
    # Goal:
    # - Turn one message list into one training string.
    # - The tags must make the assistant answer easy to find later.
    #
    # Inputs:
    # - `example`: one item from `build_chat_examples()`.
    # - `config.user_tag`: default `"<user>"`.
    # - `config.assistant_tag`: default `"<assistant>"`.
    #
    # Output:
    # - Return one string.
    # - Use this exact template:
    #   `"<user> {user_text}\n<assistant> {assistant_text}"`
    # - Example:
    #   `"<user> Explain what a tensor is.\n<assistant> A tensor is a grid of numbers."`
    #
    # Method:
    # - Read `messages = example["messages"]`.
    # - Use `messages[0]["content"]` for the user text.
    # - Use `messages[1]["content"]` for the assistant text.
    # - Build the string with an f-string:
    #   `f"{config.user_tag} {user_text}\n{config.assistant_tag} {assistant_text}"`
    #
    # Steps:
    # 1. Assert the example has a `"messages"` key.
    # 2. Assert the first role is `"user"`.
    # 3. Assert the second role is `"assistant"`.
    # 4. Format the string with the exact template above.
    # 5. Return it.
    #
    # Checks:
    # - The formatted text should contain both `config.user_tag` and `config.assistant_tag`.
    # - The assistant tag should appear after the user tag.
    #
    raise NotImplementedError("Return one formatted chat string.")


# Build a simple vocabulary from formatted texts.
# Expected return:
# - token_to_id dictionary
def build_chat_vocabulary(texts: list[str], config: InstructionTuningDetailsConfig) -> dict[str, int]:
    """Build a vocabulary for the formatted chats."""
    # Goal:
    # - Build a tiny word-level vocabulary from the formatted chat strings.
    # - Include the pad token and chat tags with known ids.
    #
    # Inputs:
    # - `texts`: formatted strings from `apply_chat_template(...)`.
    # - `config.pad_token`: default `"<pad>"`.
    # - `config.user_tag`: default `"<user>"`.
    # - `config.assistant_tag`: default `"<assistant>"`.
    #
    # Output:
    # - Return `dict[str, int]`.
    # - Required ids:
    #   - `token_to_id[config.pad_token] == 0`
    #   - `config.user_tag` must be present
    #   - `config.assistant_tag` must be present
    # - Every token from every formatted text should have an id.
    #
    # Method:
    # - Use whitespace tokenization for this exercise:
    #   `tokens = text.split()`
    # - Start the vocabulary like:
    #   `token_to_id = {config.pad_token: 0}`
    # - Then add tokens in the order they first appear.
    # - Do not sort the tokens if you want ids to be easy to trace from the examples.
    #
    # Steps:
    # 1. Create `token_to_id = {config.pad_token: 0}`.
    # 2. Loop over each formatted text.
    # 3. Split each text with `text.split()`.
    # 4. Add any new token with the next id.
    # 5. Return `token_to_id`.
    #
    # Checks:
    # - `token_to_id[config.pad_token]` should be `0`.
    # - `config.assistant_tag in token_to_id` should be True.
    # - If a text starts with `"<user>"`, its token id should be found in the vocabulary.
    #
    raise NotImplementedError("Build token ids for chat template tokens and words.")


# Tokenize formatted chat text.
# Expected return:
# - list of token ids
def tokenize_chat_text(text: str, token_to_id: dict[str, int]) -> list[int]:
    """Tokenize formatted chat text."""
    # Goal:
    # - Convert one formatted chat string into token ids.
    #
    # Inputs:
    # - `text`: formatted chat text.
    # - `token_to_id`: vocabulary from `build_chat_vocabulary(...)`.
    # - Example:
    #   - text: `"<user> Hi\n<assistant> Hello"`
    #   - split tokens: `["<user>", "Hi", "<assistant>", "Hello"]`
    #
    # Output:
    # - Return `list[int]`.
    # - Length should equal `len(text.split())`.
    # - Each id should come from `token_to_id`.
    #
    # Method:
    # - Use:
    #   `tokens = text.split()`
    #   `ids = [token_to_id[token] for token in tokens]`
    # - This is a simple teaching tokenizer. Real tokenizers may split words into smaller pieces.
    #
    # Steps:
    # 1. Split the text with `text.split()`.
    # 2. Look up each token in `token_to_id`.
    # 3. Return the ids.
    #
    # Checks:
    # - `len(ids) == len(text.split())` should be True.
    # - The first id should equal `token_to_id["<user>"]` for the default template.
    #
    raise NotImplementedError("Turn formatted chat text into token ids.")


# Find which token positions belong to the assistant answer.
# Expected return:
# - boolean list where True means this token should train
def build_assistant_loss_mask(
    formatted_text: str,
    token_ids: list[int],
    token_to_id: dict[str, int],
    config: InstructionTuningDetailsConfig,
) -> list[bool]:
    """Return a mask for assistant-answer tokens."""
    # Goal:
    # - Mark exactly the assistant answer tokens as trainable.
    # - User tokens and chat tags should not train.
    #
    # Inputs:
    # - `formatted_text`: output from `apply_chat_template(...)`.
    # - `token_ids`: output from `tokenize_chat_text(...)`.
    # - `token_to_id`: vocabulary.
    # - `config.assistant_tag`: token that marks where the assistant answer starts.
    # - Example tokens:
    #   `["<user>", "Hi", "<assistant>", "Hello", "there"]`
    # - Expected mask:
    #   `[False, False, False, True, True]`
    #
    # Output:
    # - Return `list[bool]` with the same length as `token_ids`.
    # - `True` means the token should contribute to loss.
    # - `False` means the label should become `ignore_index`.
    #
    # Method:
    # - Use whitespace tokens so the mask lines up with `tokenize_chat_text(...)`:
    #   `tokens = formatted_text.split()`
    # - Find the assistant tag:
    #   `assistant_index = tokens.index(config.assistant_tag)`
    # - Mark tokens after the assistant tag:
    #   `mask = [index > assistant_index for index in range(len(tokens))]`
    # - Do not mark the assistant tag itself as trainable. The model should learn the answer,
    #   not be trained to predict the formatting tag.
    #
    # Steps:
    # 1. Split `formatted_text` with `formatted_text.split()`.
    # 2. Assert `len(tokens) == len(token_ids)`.
    # 3. Find `config.assistant_tag` in `tokens`.
    # 4. Create a bool list that is True only after the assistant tag.
    # 5. Return the mask.
    #
    # Checks:
    # - `len(mask) == len(token_ids)` should be True.
    # - The user tag position should be False.
    # - The assistant tag position should be False.
    # - At least one answer token should be True.
    #
    raise NotImplementedError("Mark only assistant answer tokens as trainable.")


# Build labels for assistant-only loss.
# Expected return:
# - tensor labels with ignore_index outside assistant answer positions
def build_assistant_only_labels(
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Return labels for assistant-only language-model loss."""
    # Goal:
    # - Create next-token labels.
    # - Keep labels only where the next token is part of the assistant answer.
    #
    # Inputs:
    # - `input_ids`: integer tensor with shape `[batch_size, sequence_length]`.
    # - `assistant_mask`: bool tensor with shape `[batch_size, sequence_length]`.
    # - `ignore_index`: value used by `nn.CrossEntropyLoss(ignore_index=...)`, usually `-100`.
    # - Small example:
    #   - `input_ids = [[10, 11, 12, 13]]`
    #   - assistant tokens are ids `12` and `13`
    #   - `assistant_mask = [[False, False, True, True]]`
    #
    # Output:
    # - Return labels with shape `[batch_size, sequence_length]`.
    # - Positions that should not train must equal `ignore_index`.
    # - For the example above, labels should be:
    #   `[[-100, 12, 13, -100]]`
    # - Why position 1 gets label 12:
    #   - token 11 predicts the next token, 12
    #   - token 12 is an assistant token, so that prediction should train
    #
    # Method:
    # - Create all-ignore labels:
    #   `labels = torch.full_like(input_ids, fill_value=ignore_index)`
    # - Shift next-token ids into labels:
    #   `next_token_ids = input_ids[:, 1:]`
    # - Shift the assistant mask the same way:
    #   `next_token_is_assistant = assistant_mask[:, 1:]`
    # - Fill trainable positions:
    #   `label_targets = labels[:, :-1]`
    #   `label_targets[next_token_is_assistant] = next_token_ids[next_token_is_assistant]`
    # - Keep the last label as `ignore_index` because there is no next token in this sequence.
    #
    # Steps:
    # 1. Assert `input_ids.shape == assistant_mask.shape`.
    # 2. Create `labels` filled with `ignore_index`.
    # 3. Read next-token ids from `input_ids[:, 1:]`.
    # 4. Read next-token assistant mask from `assistant_mask[:, 1:]`.
    # 5. Copy next-token ids into `labels[:, :-1]` only where the shifted mask is True.
    # 6. Return labels.
    #
    # Checks:
    # - `labels.shape == input_ids.shape` should be True.
    # - User-token positions should be `ignore_index`.
    # - At least one assistant-answer label should not be `ignore_index`.
    #
    raise NotImplementedError("Create shifted labels and ignore non-assistant positions.")


# Pad examples into a batch.
# Expected return:
# - dictionary with input_ids, labels, and attention_mask tensors
def collate_instruction_batch(
    token_lists: list[list[int]],
    masks: list[list[bool]],
    pad_id: int,
    config: InstructionTuningDetailsConfig,
) -> InstructionBatch:
    """Pad instruction examples into a batch."""
    # Goal:
    # - Pad variable-length token lists into one tensor batch.
    # - Build attention masks and assistant-only labels.
    #
    # Inputs:
    # - `token_lists`: token ids for each formatted example.
    # - `masks`: assistant-loss masks from `build_assistant_loss_mask(...)`.
    # - `pad_id`: id for `config.pad_token`.
    # - `config.ignore_index`: label value for ignored tokens.
    # - Example:
    #   - token lengths: `[5, 8]`
    #   - max length: `8`
    #   - first example gets 3 pad tokens at the end
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"input_ids"`: long tensor with shape `[batch_size, max_length]`
    #   - `"labels"`: long tensor with shape `[batch_size, max_length]`
    #   - `"attention_mask"`: long tensor with shape `[batch_size, max_length]`
    #   - `"assistant_mask"`: bool tensor with shape `[batch_size, max_length]`
    # - `attention_mask` should be 1 for real tokens and 0 for padding.
    # - `labels` should use `config.ignore_index` outside assistant answer labels.
    #
    # Method:
    # - Compute:
    #   `batch_size = len(token_lists)`
    #   `max_length = max(len(tokens) for tokens in token_lists)`
    # - Create padded tensors:
    #   `input_ids = torch.full((batch_size, max_length), fill_value=pad_id, dtype=torch.long)`
    #   `attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)`
    #   `assistant_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)`
    # - Fill each row up to its real length.
    # - Build labels by calling:
    #   `labels = build_assistant_only_labels(input_ids, assistant_mask, config.ignore_index)`
    #
    # Steps:
    # 1. Assert `len(token_lists) == len(masks)`.
    # 2. Assert there is at least one example.
    # 3. Compute `batch_size` and `max_length`.
    # 4. Create padded `input_ids`, `attention_mask`, and `assistant_mask`.
    # 5. Loop over examples and copy values into the left side of each row.
    # 6. Call `build_assistant_only_labels(...)`.
    # 7. Return the dictionary with the exact keys above.
    #
    # Checks:
    # - `batch["input_ids"].shape == batch["labels"].shape` should be True.
    # - `batch["attention_mask"].sum().item()` should equal the total number of real tokens.
    # - All padding labels should equal `config.ignore_index`.
    #
    raise NotImplementedError("Return padded input ids, labels, and attention mask.")


# Check that user tokens are ignored.
# Expected return:
# - True when non-assistant positions are ignore_index
def verify_user_tokens_are_masked(batch: InstructionBatch, config: InstructionTuningDetailsConfig) -> bool:
    """Return True when user tokens do not contribute to loss."""
    # Goal:
    # - Check that labels outside assistant-answer positions are ignored.
    #
    # Inputs:
    # - `batch`: output from `collate_instruction_batch(...)`.
    # - Required keys:
    #   - `"labels"`
    #   - `"assistant_mask"`
    #   - `"attention_mask"`
    # - `config.ignore_index`: usually `-100`.
    #
    # Output:
    # - Return True only when:
    #   - every non-training or padding position has label `ignore_index`
    #   - at least one assistant position has a real label
    #
    # Method:
    # - Use:
    #   `labels = batch["labels"]`
    #   `assistant_mask = batch["assistant_mask"]`
    #   `attention_mask = batch["attention_mask"].bool()`
    # - Labels are next-token labels, so the trainable label position is one step before
    #   the assistant token it predicts.
    # - Build train label positions with:
    #   `train_label_positions = torch.zeros_like(assistant_mask, dtype=torch.bool)`
    #   `train_label_positions[:, :-1] = assistant_mask[:, 1:] & attention_mask[:, 1:]`
    # - Non-train positions are:
    #   `non_train_positions = ~train_label_positions`
    # - Check ignored labels:
    #   `(labels[non_train_positions] == config.ignore_index).all()`
    # - Check there is at least one real label:
    #   `(labels != config.ignore_index).any()`
    #
    # Steps:
    # 1. Read `labels`, `assistant_mask`, and `attention_mask`.
    # 2. Assert their shapes match.
    # 3. Build `train_label_positions` by shifting `assistant_mask` left by one position.
    # 4. Build `non_train_positions`.
    # 5. Check all non-train labels are ignored.
    # 6. Check at least one label is not ignored.
    # 7. Return the combined bool.
    #
    # Checks:
    # - If user labels are accidentally real token ids, this function should return False.
    # - If every label is `ignore_index`, this function should return False.
    #
    raise NotImplementedError("Check that only assistant tokens have real labels.")


# Run the full formatting workflow.
# Expected return:
# - summary dictionary with counts and checks
def run_instruction_format_demo(config: InstructionTuningDetailsConfig) -> InstructionSummary:
    """Run the instruction tuning format workflow."""
    # Goal:
    # - Run the full instruction-format pipeline.
    # - Return a small summary that proves assistant-only labels were built correctly.
    #
    # Inputs:
    # - `config`: tags, pad token, and ignore-label settings.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"num_examples"`: number of chat examples
    #   - `"vocab_size"`: number of tokens in the toy vocabulary
    #   - `"batch_shape"`: tuple `(batch_size, sequence_length)`
    #   - `"assistant_label_count"`: number of labels not equal to `ignore_index`
    #   - `"ignored_label_count"`: number of labels equal to `ignore_index`
    #   - `"user_tokens_masked"`: bool from `verify_user_tokens_are_masked(...)`
    #   - `"format_note"`: short plain-language note
    #
    # Method:
    # - Use this exact workflow:
    #   `examples = build_chat_examples()`
    #   `texts = [apply_chat_template(example, config) for example in examples]`
    #   `token_to_id = build_chat_vocabulary(texts, config)`
    #   `token_lists = [tokenize_chat_text(text, token_to_id) for text in texts]`
    #   `masks = [build_assistant_loss_mask(text, ids, token_to_id, config) for text, ids in zip(texts, token_lists)]`
    #   `batch = collate_instruction_batch(token_lists, masks, token_to_id[config.pad_token], config)`
    #   `user_tokens_masked = verify_user_tokens_are_masked(batch, config)`
    # - Count labels with:
    #   `assistant_label_count = int((batch["labels"] != config.ignore_index).sum().item())`
    #   `ignored_label_count = int((batch["labels"] == config.ignore_index).sum().item())`
    #
    # Steps:
    # 1. Build chat examples.
    # 2. Format each example with the chat template.
    # 3. Build the vocabulary.
    # 4. Tokenize each formatted string.
    # 5. Build assistant masks.
    # 6. Collate into one padded batch.
    # 7. Verify user tokens are masked.
    # 8. Count assistant labels and ignored labels.
    # 9. Return the summary dictionary with the exact keys above.
    #
    # Checks:
    # - `summary["assistant_label_count"] > 0` should be True.
    # - `summary["user_tokens_masked"]` should be True.
    # - `check_metrics(summary)` should be True.
    #
    raise NotImplementedError("Build formatted chats, labels, masks, and return checks.")


# Check the summary.
# Expected return:
# - True when the formatting checks are present and passed
def check_metrics(metrics: InstructionSummary) -> bool:
    """Return True when instruction-format outputs are present."""
    required = {"num_examples", "assistant_label_count", "user_tokens_masked", "format_note"}
    return required.issubset(metrics) and bool(metrics.get("user_tokens_masked", False))


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the instruction tuning details scaffold."""
    config = InstructionTuningDetailsConfig()
    set_seed(config.random_state)

    try:
        metrics = run_instruction_format_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "instruction_format_checks", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
