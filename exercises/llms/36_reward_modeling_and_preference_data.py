"""
============================================================
EXERCISE 36: REWARD MODELING AND PREFERENCE DATA
============================================================

WHAT YOU ARE BUILDING
You are building a tiny reward-modeling workflow from preference examples.

Plain-language meanings:
- preference data compares two answers to the same prompt
- chosen means the better answer
- rejected means the worse answer
- a reward model gives a score to an answer
- reward modeling trains scores so chosen answers score higher than rejected answers
- rejection sampling means generate several answers and keep the highest-scoring one

WHY THIS EXERCISE EXISTS
Many post-training pipelines use preference data.
Before using DPO or RLHF ideas, students should understand how preference examples
can become reward scores.

This exercise teaches:
- preference example format
- chosen/rejected ordering
- reward scores
- pairwise reward loss
- preference margin
- rejection sampling with a reward model

HOW THIS FITS INTO THE BOOTCAMP
The DPO exercise showed direct preference optimization.
This file adds the reward-model path that is often used in RLHF-style systems.

Beginner intuition:
- the reward model is like a judge
- it reads a prompt and an answer
- it gives a higher score to better answers
- later, another model can use those scores as a training signal

DATASET
Use a tiny local preference dataset.

Clear default:
- 6 to 12 beginner-helpfulness examples written inside this file
- each example has:
  - `prompt`
  - `chosen`
  - `rejected`
- prompts should be about safe, simple learning help
- the chosen answer should be clearer, safer, or more helpful than the rejected answer

Example:
- prompt: `Give a safe cooking tip.`
- chosen: `Wash your hands before handling food.`
- rejected: `Ignore food safety rules.`

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- write the preference examples as dictionaries
- format each prompt-answer pair into one text string
- tokenize chosen and rejected pairs
- score both with the reward model
- train so chosen scores become higher

INPUTS AND OUTPUTS
Input:
- prompt text
- chosen answer text
- rejected answer text

Model output:
- one scalar reward score per prompt-answer pair

Metrics:
- final pairwise loss
- chosen-minus-rejected reward margin
- best answer selected by rejection sampling

STEP-BY-STEP TODO
1. Build tiny preference examples.
2. Format prompt-answer pairs for scoring.
3. Build a small reward model.
4. Score chosen and rejected answers.
5. Compute pairwise reward loss.
6. Train for a few steps.
7. Measure the chosen-minus-rejected margin.
8. Use scores for simple rejection sampling.
9. Write a note about reward modeling.

DEFINITION OF DONE
You are done when:
- preference examples have prompt, chosen, and rejected fields
- chosen and rejected are not mixed up
- reward margin is reported
- rejection sampling returns the best-scored answer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


PreferenceExample = dict[str, str]
RewardTrainingMetrics = dict[str, float]
RejectionSampleResult = dict[str, str | float | int]
RewardModelingSummary = dict[str, float | str | int]


# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for reward-modeling practice.
# Expected return:
# - a config object with model, data, and training settings
@dataclass
class RewardModelingConfig:
    """Settings for reward modeling practice."""

    vocab_size: int = 256
    hidden_dim: int = 64
    max_length: int = 64
    learning_rate: float = 1e-3
    train_steps: int = 20
    random_state: int = 42


# Build preference examples.
# Expected return:
# - list of dicts with prompt, chosen, and rejected
def build_preference_dataset() -> list[PreferenceExample]:
    """Return tiny preference examples."""
    # Goal:
    # - Build a tiny local preference dataset.
    # - Each example compares two answers to the same prompt.
    #
    # Inputs:
    # - No direct arguments.
    #
    # Output:
    # - Return `list[PreferenceExample]`.
    # - Each example must have exactly these keys:
    #   - `"prompt"`: str
    #   - `"chosen"`: str, the better answer
    #   - `"rejected"`: str, the worse answer
    # - Return at least 6 examples.
    # - Example:
    #   `{"prompt": "Give a safe cooking tip.", "chosen": "Wash your hands before handling food.", "rejected": "Ignore food safety rules."}`
    #
    # Method:
    # - Write the examples by hand in a Python list.
    # - Make the chosen answer clearer, safer, or more useful than the rejected answer.
    # - Do not swap the fields. The loss assumes `"chosen"` should score higher.
    #
    # Steps:
    # 1. Create a list of dictionaries.
    # 2. Add prompt, chosen, and rejected strings to each dictionary.
    # 3. Check that chosen and rejected are different.
    # 4. Return the list.
    #
    # Checks:
    # - `len(examples) >= 6` should be True.
    # - `set(examples[0])` should be `{"prompt", "chosen", "rejected"}`.
    # - `examples[0]["chosen"] != examples[0]["rejected"]` should be True.
    #
    raise NotImplementedError("Return prompt, chosen, rejected examples.")


# Format prompt and answer for reward scoring.
# Expected return:
# - one string that includes both prompt and answer
def format_reward_input(prompt: str, answer: str) -> str:
    """Format one prompt-answer pair."""
    # Goal:
    # - Format one prompt and one answer into the text that the reward model scores.
    #
    # Inputs:
    # - `prompt`: user request.
    # - `answer`: candidate answer to score.
    #
    # Output:
    # - Return one string.
    # - Use this exact template:
    #   `"Prompt: {prompt}\nAnswer: {answer}"`
    # - Example:
    #   `"Prompt: Give a safe cooking tip.\nAnswer: Wash your hands before handling food."`
    #
    # Method:
    # - Use an f-string:
    #   `f"Prompt: {prompt}\nAnswer: {answer}"`
    # - Keep the same template for chosen, rejected, and candidate answers.
    #
    # Steps:
    # 1. Insert the prompt after `"Prompt: "`.
    # 2. Insert the answer after `"Answer: "`.
    # 3. Return the formatted string.
    #
    # Checks:
    # - The result should contain `"Prompt:"` and `"Answer:"`.
    # - The prompt should appear before the answer.
    #
    raise NotImplementedError("Return text for reward model scoring.")


# Build a vocabulary for reward inputs.
# Expected return:
# - token_to_id dictionary
def build_reward_vocabulary(examples: list[PreferenceExample], config: RewardModelingConfig) -> dict[str, int]:
    """Build a vocabulary from preference text."""
    # Goal:
    # - Build word ids for all prompt-answer texts used by the reward model.
    # - Include a pad token at id 0.
    #
    # Inputs:
    # - `examples`: output from `build_preference_dataset()`.
    # - `config.vocab_size`: maximum number of tokens to keep.
    #
    # Output:
    # - Return `dict[str, int]`.
    # - Required ids:
    #   - `"<pad>"` should be 0
    #   - `"<unk>"` should be 1
    # - Every other id should be less than `config.vocab_size`.
    #
    # Method:
    # - For each example, format both pairs:
    #   `format_reward_input(example["prompt"], example["chosen"])`
    #   `format_reward_input(example["prompt"], example["rejected"])`
    # - Tokenize by lowercasing and splitting on spaces:
    #   `tokens = text.lower().split()`
    # - Add new tokens in first-seen order until `len(token_to_id) == config.vocab_size`.
    # - Keep `"<unk>"` for words that are not in the vocabulary.
    #
    # Steps:
    # 1. Create `token_to_id = {"<pad>": 0, "<unk>": 1}`.
    # 2. Loop over examples.
    # 3. Format the chosen pair and rejected pair.
    # 4. Add lowercase whitespace tokens in first-seen order.
    # 5. Stop adding new tokens if the vocab reaches `config.vocab_size`.
    # 6. Return the vocabulary.
    #
    # Checks:
    # - `token_to_id["<pad>"] == 0` should be True.
    # - `token_to_id["<unk>"] == 1` should be True.
    # - `len(token_to_id) <= config.vocab_size` should be True.
    #
    raise NotImplementedError("Build token ids from preference examples.")


# Tokenize reward inputs.
# Expected return:
# - tensor with shape `[num_examples, max_length]`
def tokenize_reward_inputs(texts: list[str], token_to_id: dict[str, int], config: RewardModelingConfig) -> torch.Tensor:
    """Tokenize reward model inputs."""
    # Goal:
    # - Convert formatted reward texts into one padded integer tensor.
    #
    # Inputs:
    # - `texts`: formatted prompt-answer strings.
    # - `token_to_id`: vocabulary from `build_reward_vocabulary(...)`.
    # - `config.max_length`: fixed output sequence length.
    # - Small example with `max_length=5`:
    #   - text tokens: `["prompt:", "hi", "answer:", "hello"]`
    #   - ids before padding: `[2, 3, 4, 5]`
    #   - ids after padding: `[2, 3, 4, 5, 0]`
    #
    # Output:
    # - Return a `torch.long` tensor with shape `[len(texts), config.max_length]`.
    # - Use pad id 0 for padding.
    # - Use unknown id 1 for tokens missing from the vocabulary.
    # - Truncate texts longer than `config.max_length`.
    #
    # Method:
    # - Create the output with:
    #   `input_ids = torch.full((len(texts), config.max_length), fill_value=0, dtype=torch.long)`
    # - For each text:
    #   `tokens = text.lower().split()[: config.max_length]`
    #   `ids = [token_to_id.get(token, token_to_id["<unk>"]) for token in tokens]`
    # - Copy ids into the left side of the row.
    #
    # Steps:
    # 1. Assert `config.max_length > 0`.
    # 2. Create a padded tensor filled with 0.
    # 3. Loop over texts.
    # 4. Lowercase, split, and truncate each text.
    # 5. Convert tokens to ids, using `"<unk>"` for unknown tokens.
    # 6. Copy ids into the row.
    # 7. Return the tensor.
    #
    # Checks:
    # - `input_ids.shape == (len(texts), config.max_length)` should be True.
    # - `input_ids.dtype == torch.long` should be True.
    # - Padding positions should contain 0.
    #
    raise NotImplementedError("Return padded token ids for reward scoring.")



# Define the tiny reward model.
# Expected behavior:
# - expose a model class that scores prompt-answer pairs
class TinyRewardModel(nn.Module):
    """Small model that scores prompt-answer text."""

    # Build the reward model.
    # Expected return:
    # - a model that maps token ids to one scalar reward per example
    def __init__(self, config: RewardModelingConfig) -> None:
        # Goal:
        # - Build a small model that maps token ids to one reward score.
        # - Higher reward should mean the answer is judged better.
        #
        # Inputs:
        # - `config.vocab_size`: number of token ids the embedding can read.
        # - `config.hidden_dim`: embedding and hidden size.
        #
        # Output:
        # - Set up modules on `self`.
        # - The forward pass should return shape `[batch_size]`.
        #
        # Method:
        # - Use this exact module shape:
        #   `self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=0)`
        #   `self.scorer = nn.Sequential(`
        #   `    nn.Linear(config.hidden_dim, config.hidden_dim),`
        #   `    nn.Tanh(),`
        #   `    nn.Linear(config.hidden_dim, 1),`
        #   `)`
        # - `padding_idx=0` means pad tokens use a fixed zero-like embedding and do not train.
        # - `self.scorer` turns one pooled vector into one scalar score.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Store `config` if useful.
        # 3. Create `self.embedding`.
        # 4. Create `self.scorer`.
        #
        # Checks:
        # - `model(torch.ones(2, config.max_length, dtype=torch.long)).shape` should be `[2]`.
        # - Do not apply sigmoid or softmax to rewards. They are raw scores.
        #
        super().__init__()
        raise NotImplementedError("Build a tiny reward model.")

    # Score a batch of token ids.
    # Expected return:
    # - tensor with shape `[batch_size]`
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return reward scores."""
        # Goal:
        # - Score each prompt-answer pair in the batch.
        #
        # Inputs:
        # - `input_ids`: long tensor with shape `[batch_size, sequence_length]`.
        # - Pad token id is 0.
        #
        # Output:
        # - Return a float tensor with shape `[batch_size]`.
        # - Each value is one reward score.
        #
        # Method:
        # - Embed tokens:
        #   `embeddings = self.embedding(input_ids)`
        #   shape `[batch_size, sequence_length, hidden_dim]`
        # - Build a non-pad mask:
        #   `mask = (input_ids != 0).unsqueeze(-1)`
        #   shape `[batch_size, sequence_length, 1]`
        # - Mean-pool only non-pad tokens:
        #   `masked_embeddings = embeddings * mask`
        #   `lengths = mask.sum(dim=1).clamp_min(1)`
        #   `pooled = masked_embeddings.sum(dim=1) / lengths`
        # - Score:
        #   `scores = self.scorer(pooled).squeeze(-1)`
        #
        # Steps:
        # 1. Assert `input_ids.ndim == 2`.
        # 2. Embed the ids.
        # 3. Build the non-pad mask.
        # 4. Mean-pool non-pad token embeddings.
        # 5. Pass the pooled vectors through `self.scorer`.
        # 6. Remove the last size-1 dimension with `.squeeze(-1)`.
        # 7. Return scores.
        #
        # Checks:
        # - If `input_ids.shape` is `[4, 64]`, scores should have shape `[4]`.
        # - Scores can be negative or positive. Only their order matters.
        #
        raise NotImplementedError("Return one reward score per example.")


# Compute pairwise reward loss.
# Expected return:
# - scalar loss tensor
def pairwise_reward_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    """Return preference loss for reward modeling."""
    # Goal:
    # - Train the model so chosen answers score higher than rejected answers.
    #
    # Inputs:
    # - `chosen_scores`: tensor with shape `[batch_size]`.
    # - `rejected_scores`: tensor with shape `[batch_size]`.
    # - Larger score means the reward model likes the answer more.
    #
    # Output:
    # - Return one scalar loss tensor with shape `[]`.
    # - Lower loss means the chosen scores are higher than rejected scores.
    #
    # Method:
    # - Use the common pairwise logistic loss:
    #   `margin = chosen_scores - rejected_scores`
    #   `loss = -torch.nn.functional.logsigmoid(margin).mean()`
    # - If margin is large and positive, loss is small.
    # - If margin is negative, loss is large because the rejected answer scored higher.
    #
    # Steps:
    # 1. Assert both input tensors have the same shape.
    # 2. Compute the margin.
    # 3. Apply `torch.nn.functional.logsigmoid`.
    # 4. Negate and average.
    # 5. Return the scalar loss.
    #
    # Checks:
    # - `pairwise_reward_loss(torch.tensor([2.0]), torch.tensor([0.0]))`
    #   should be smaller than `pairwise_reward_loss(torch.tensor([0.0]), torch.tensor([2.0]))`.
    # - The returned tensor should have `loss.ndim == 0`.
    #
    raise NotImplementedError("Make chosen scores higher than rejected scores.")


# Train the reward model.
# Expected return:
# - dictionary with final loss and margin
def train_reward_model(
    model: TinyRewardModel,
    examples: list[PreferenceExample],
    token_to_id: dict[str, int],
    config: RewardModelingConfig,
) -> RewardTrainingMetrics:
    """Train the tiny reward model."""
    # Goal:
    # - Train the reward model on chosen/rejected pairs.
    # - Report the final pairwise loss and average reward margin.
    #
    # Inputs:
    # - `model`: `TinyRewardModel`.
    # - `examples`: preference examples with `"prompt"`, `"chosen"`, and `"rejected"`.
    # - `token_to_id`: vocabulary.
    # - `config.train_steps`: number of optimizer steps.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"final_loss"`: float
    #   - `"preference_margin"`: float, mean chosen score minus rejected score
    # - Positive margin means chosen answers score higher on average.
    #
    # Method:
    # - Build chosen and rejected texts:
    #   `chosen_texts = [format_reward_input(e["prompt"], e["chosen"]) for e in examples]`
    #   `rejected_texts = [format_reward_input(e["prompt"], e["rejected"]) for e in examples]`
    # - Tokenize both:
    #   `chosen_ids = tokenize_reward_inputs(chosen_texts, token_to_id, config)`
    #   `rejected_ids = tokenize_reward_inputs(rejected_texts, token_to_id, config)`
    # - Optimizer:
    #   `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    # - Each step:
    #   `chosen_scores = model(chosen_ids)`
    #   `rejected_scores = model(rejected_ids)`
    #   `loss = pairwise_reward_loss(chosen_scores, rejected_scores)`
    #
    # Steps:
    # 1. Put the model in train mode with `model.train()`.
    # 2. Build and tokenize chosen and rejected texts.
    # 3. Create AdamW.
    # 4. Loop for `config.train_steps`.
    # 5. Compute chosen and rejected scores.
    # 6. Compute pairwise loss.
    # 7. Clear gradients with `optimizer.zero_grad()`.
    # 8. Backpropagate with `loss.backward()`.
    # 9. Step the optimizer with `optimizer.step()`.
    # 10. After training, compute scores again under `torch.no_grad()`.
    # 11. Return final loss and mean margin.
    #
    # Checks:
    # - `chosen_scores.shape == rejected_scores.shape == (len(examples),)` should be True.
    # - Returned keys should be `{"final_loss", "preference_margin"}`.
    #
    raise NotImplementedError("Train on chosen and rejected pairs.")


# Pick the best answer from candidates.
# Expected return:
# - chosen answer string and score
def rejection_sample_answer(
    prompt: str,
    candidates: list[str],
    model: TinyRewardModel,
    token_to_id: dict[str, int],
    config: RewardModelingConfig,
) -> RejectionSampleResult:
    """Choose the highest-scoring candidate answer."""
    # Goal:
    # - Score several candidate answers and keep the highest-scoring answer.
    # - This is the basic idea of rejection sampling with a reward model.
    #
    # Inputs:
    # - `prompt`: the user request shared by all candidates.
    # - `candidates`: possible answers to score.
    # - `model`: trained reward model.
    # - `token_to_id`: vocabulary.
    # - `config`: tokenization settings.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"best_answer"`: str
    #   - `"best_score"`: float
    #   - `"best_index"`: int
    # - `best_index` is the index into the original `candidates` list.
    #
    # Method:
    # - Format all candidates:
    #   `texts = [format_reward_input(prompt, candidate) for candidate in candidates]`
    # - Tokenize:
    #   `input_ids = tokenize_reward_inputs(texts, token_to_id, config)`
    # - Score under eval mode:
    #   `model.eval()`
    #   `with torch.no_grad(): scores = model(input_ids)`
    # - Pick best:
    #   `best_index = int(scores.argmax().item())`
    #
    # Steps:
    # 1. Assert `len(candidates) > 0`.
    # 2. Format prompt-answer texts.
    # 3. Tokenize them.
    # 4. Score them with the reward model.
    # 5. Find `argmax`.
    # 6. Return best answer, score, and index.
    #
    # Checks:
    # - `0 <= result["best_index"] < len(candidates)` should be True.
    # - `result["best_answer"] == candidates[result["best_index"]]` should be True.
    #
    raise NotImplementedError("Score candidates and return the best one.")


# Run the full reward-modeling workflow.
# Expected return:
# - summary dictionary with margin and rejection sampling result
def run_reward_modeling_demo(config: RewardModelingConfig) -> RewardModelingSummary:
    """Run the reward modeling workflow."""
    # Goal:
    # - Run the full tiny reward-model workflow.
    # - Return training metrics and the rejection-sampling result.
    #
    # Inputs:
    # - `config`: data, model, and training settings.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"final_loss"`: float
    #   - `"preference_margin"`: float
    #   - `"best_answer"`: str
    #   - `"best_score"`: float
    #   - `"best_index"`: int
    #   - `"reward_model_note"`: str
    #
    # Method:
    # - Use:
    #   `examples = build_preference_dataset()`
    #   `token_to_id = build_reward_vocabulary(examples, config)`
    #   `model = TinyRewardModel(config)`
    #   `training_metrics = train_reward_model(model, examples, token_to_id, config)`
    # - For rejection sampling, use one prompt and candidates such as:
    #   `prompt = "Give a safe cooking tip."`
    #   `candidates = ["Wash your hands before handling food.", "Ignore food safety rules."]`
    # - Then call:
    #   `sample = rejection_sample_answer(prompt, candidates, model, token_to_id, config)`
    #
    # Steps:
    # 1. Build preference examples.
    # 2. Build the vocabulary.
    # 3. Build the reward model.
    # 4. Train the reward model.
    # 5. Run rejection sampling on a small candidate list.
    # 6. Write a short note that says chosen answers should score higher than rejected answers.
    # 7. Return the summary dictionary with the exact keys above.
    #
    # Checks:
    # - `summary["preference_margin"]` should ideally be positive after training.
    # - `check_metrics(summary)` should be True.
    #
    raise NotImplementedError("Build data, train reward model, sample best answer, and return summary.")


# Check the summary.
# Expected return:
# - True when reward modeling outputs are present
def check_metrics(metrics: RewardModelingSummary) -> bool:
    """Return True when reward-modeling outputs are present."""
    required = {"final_loss", "preference_margin", "best_answer", "reward_model_note"}
    return required.issubset(metrics)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the reward modeling scaffold."""
    config = RewardModelingConfig()
    set_seed(config.random_state)

    try:
        metrics = run_reward_modeling_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "reward_model_outputs", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
