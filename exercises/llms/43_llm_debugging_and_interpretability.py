"""
============================================================
EXERCISE 43: LLM DEBUGGING AND INTERPRETABILITY
============================================================

WHAT YOU ARE BUILDING
You are building tiny tools for inspecting language-model behavior.

Plain-language meanings:
- logits are raw scores before probabilities
- logit lens means looking at intermediate hidden states through the output head
- activation means a tensor value inside the model
- activation patching means replacing one run's activation with another run's activation
- attention pattern means which tokens a token looks at
- memorization means the model repeats text it has seen before

WHY THIS EXERCISE EXISTS
Frontier labs need to debug model behavior, not only train models.
Small inspection tools help connect model outputs to internal signals.

This exercise teaches:
- collecting logits
- checking top predicted tokens
- simple logit lens
- attention pattern summaries
- toy activation patching
- memorization checks

HOW THIS FITS INTO THE BOOTCAMP
The interpretability basics exercise gives a wider view.
This file focuses on LLM-specific debugging habits.

Beginner intuition:
- the final answer is not the only thing you can inspect
- hidden states can show what token the model is leaning toward
- attention can show which earlier tokens were used
- patching can test whether one internal value matters

DATASET
Use Tiny Shakespeare plus a tiny local prompt dataset.

Clear default:
- two or three short prompts written as token ids
- one small list of training texts taken from Tiny Shakespeare if available
- one small list of generated texts to compare against training text

Why this is enough:
- debugging tools need model inputs and internal tensors
- the text does not need to be large
- small prompts make attention and patching easier to inspect

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load a short Tiny Shakespeare text file from `data/`
- write toy prompts
- convert them to token ids
- run the tiny model on each prompt
- collect logits, hidden states, and attention
- compare outputs before and after patching
- compare generated strings with training strings

INPUTS AND OUTPUTS
Input:
- prompt token ids with shape `[1, sequence_length]`

Model outputs:
- logits
- hidden states
- attention weights

Debugging outputs:
- top token predictions
- logit-lens predictions
- attention summary
- activation patching comparison
- memorization report

STEP-BY-STEP TODO
1. Build a tiny language model or load one from an earlier exercise.
2. Run two prompts.
3. Collect logits and hidden states.
4. Compute top-token predictions.
5. Apply a simple logit lens.
6. Summarize attention patterns.
7. Patch one activation from one prompt into another.
8. Check for memorized text.
9. Write a debugging note.

DEFINITION OF DONE
You are done when:
- top token predictions are returned
- hidden-state inspection is returned
- attention summary is returned
- patching result is compared
- memorization check is reported
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


DebugPrompts = dict[str, torch.Tensor]
TokenPrediction = dict[str, float | int]
AttentionDebugSummary = dict[str, int | float | list[int] | list[float]]
PatchingResult = dict[str, int | float | list[TokenPrediction]]
MemorizationReport = dict[str, int | float | list[str]]
LLMDebuggingSummary = dict[
    str,
    str
    | list[TokenPrediction]
    | AttentionDebugSummary
    | PatchingResult
    | MemorizationReport,
]
# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for LLM debugging practice.
# Expected return:
# - a config object with toy model and inspection settings
@dataclass
class LLMDebuggingConfig:
    """Settings for LLM debugging practice."""

    vocab_size: int = 64
    model_dim: int = 32
    sequence_length: int = 12
    top_k: int = 5
    random_state: int = 42


# Build toy prompts as token ids.
# Expected return:
# - dictionary mapping prompt names to token-id tensors
def build_debug_prompts(config: LLMDebuggingConfig) -> DebugPrompts:
    """Return toy prompts for debugging."""
    # Goal:
    # - Return at least two toy prompts as token id tensors.
    # - The prompts should be similar enough that patching one into the other is meaningful.
    #
    # Input:
    # - `config.sequence_length`: number of token ids per prompt.
    # - `config.vocab_size`: token ids must be in `[0, config.vocab_size - 1]`.
    #
    # Output:
    # - Return `DebugPrompts`.
    # - Exact keys:
    #   - `"source"`: `torch.long` tensor with shape `[1, sequence_length]`
    #   - `"target"`: `torch.long` tensor with shape `[1, sequence_length]`
    # - Small example for `sequence_length=5`:
    #   `source = torch.tensor([[2, 5, 7, 9, 3]], dtype=torch.long)`
    #   `target = torch.tensor([[2, 5, 8, 9, 3]], dtype=torch.long)`
    #
    # Method:
    # - Use `torch.tensor(..., dtype=torch.long)`.
    # - Keep ids small and readable. Do not use random ids for this first version.
    # - If you need a quick pattern, use:
    #   `base = torch.arange(config.sequence_length).unsqueeze(0) % config.vocab_size`
    #
    # Steps:
    # 1. Build a source tensor with shape `[1, config.sequence_length]`.
    # 2. Clone it to make the target tensor.
    # 3. Change one or two positions in the target tensor.
    # 4. Make sure all ids are less than `config.vocab_size`.
    # 5. Return `{"source": source, "target": target}`.
    #
    # Checks:
    # - `set(prompts) == {"source", "target"}`
    # - `prompts["source"].shape == (1, config.sequence_length)`
    # - `prompts["source"].dtype == torch.long`
    # - `int(prompts["source"].max()) < config.vocab_size`
    #
    raise NotImplementedError("Return at least two prompts as token ids.")



# Define the tiny inspectable language model.
# Expected behavior:
# - expose a model class that can return internal tensors
class TinyInspectableLM(nn.Module):
    """Small language model that exposes internal tensors."""

    # Build a tiny inspectable language model.
    # Expected return:
    # - a model that can return logits, hidden states, and attention weights
    def __init__(self, config: LLMDebuggingConfig) -> None:
        # Goal:
        # - Build a tiny language model that exposes logits, hidden states, and attention.
        #
        # Input:
        # - `config.vocab_size`: number of token ids the model can read and predict.
        # - `config.model_dim`: hidden size for each token.
        # - `config.sequence_length`: maximum prompt length.
        #
        # Output:
        # - Return `None`, but create all modules used by `forward`.
        # - The model should predict one logit per vocabulary token.
        #
        # Method:
        # - Use these exact PyTorch modules:
        #   `self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)`
        #   `self.position_embedding = nn.Embedding(config.sequence_length, config.model_dim)`
        #   `self.attention = nn.MultiheadAttention(config.model_dim, num_heads=4, batch_first=True)`
        #   `self.norm = nn.LayerNorm(config.model_dim)`
        #   `self.output_head = nn.Linear(config.model_dim, config.vocab_size)`
        # - `batch_first=True` means tensors use shape `[batch, sequence, hidden]`.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Save `self.config = config`.
        # 3. Create token and position embeddings.
        # 4. Create one self-attention layer.
        # 5. Create a layer norm.
        # 6. Create the output head that maps hidden states to vocabulary logits.
        #
        # Checks:
        # - `self.token_embedding.weight.shape == (config.vocab_size, config.model_dim)`
        # - `self.output_head.out_features == config.vocab_size`
        # - Do not add softmax. Debugging functions need raw logits.
        #
        super().__init__()
        raise NotImplementedError("Build a tiny language model with inspectable internals.")

    # Run the model and return internal values.
    # Expected return:
    # - dictionary with logits, hidden_states, and attention
    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits and internal tensors."""
        # Goal:
        # - Run the model and return both final logits and internal tensors.
        #
        # Input:
        # - `input_ids`: `torch.long` tensor with shape `[batch_size, sequence_length]`.
        # - Small example shape:
        #   `input_ids.shape == (1, 12)`
        #
        # Output:
        # - Return a dictionary with exactly these keys:
        #   - `"logits"`: tensor with shape `[batch_size, sequence_length, vocab_size]`
        #   - `"hidden_states"`: tensor with shape `[batch_size, sequence_length, model_dim]`
        #   - `"attention"`: tensor with shape `[batch_size, num_heads, sequence_length, sequence_length]`
        #
        # Method:
        # - Build positions with:
        #   `positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)`
        # - Add token and position embeddings:
        #   `x = self.token_embedding(input_ids) + self.position_embedding(positions)`
        # - Call attention with:
        #   `attn_output, attention = self.attention(x, x, x, need_weights=True, average_attn_weights=False)`
        # - Use `self.norm(x + attn_output)` for a simple residual connection.
        # - Get logits with:
        #   `logits = self.output_head(hidden_states)`
        #
        # Steps:
        # 1. Read `batch_size` and `sequence_length` from `input_ids.shape`.
        # 2. Create position ids on the same device as `input_ids`.
        # 3. Add token embeddings and position embeddings.
        # 4. Run self-attention.
        # 5. Build hidden states with the residual and layer norm.
        # 6. Project hidden states to vocabulary logits.
        # 7. Return the three exact keys listed above.
        #
        # Checks:
        # - `outputs["logits"].shape == (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size)`
        # - `outputs["hidden_states"].shape[-1] == self.config.model_dim`
        # - `outputs["attention"].ndim == 4`
        #
        raise NotImplementedError("Return logits, hidden states, and attention weights.")


# Get top predicted tokens from logits.
# Expected return:
# - list of top token ids and scores
def get_top_token_predictions(logits: torch.Tensor, top_k: int) -> list[TokenPrediction]:
    """Return top token predictions."""
    # Goal:
    # - Return the highest-scoring next-token predictions from the final sequence position.
    #
    # Input:
    # - `logits`: tensor with shape `[batch_size, sequence_length, vocab_size]`.
    # - `top_k`: number of predictions to return.
    # - Use only the first batch item and the final position for this toy tool.
    #
    # Output:
    # - Return `list[TokenPrediction]`.
    # - Each dictionary must have exactly these keys:
    #   - `"token_id"`: int
    #   - `"logit"`: float raw score
    # - Small example:
    #   `[{"token_id": 12, "logit": 3.4}, {"token_id": 7, "logit": 2.9}]`
    #
    # Method:
    # - Select final-position logits with:
    #   `final_logits = logits[0, -1]`
    # - Use:
    #   `top_values, top_indices = torch.topk(final_logits, k=min(top_k, final_logits.numel()))`
    # - Do not apply softmax. The raw logits are enough for ranking.
    #
    # Steps:
    # 1. Check that `logits.ndim == 3`.
    # 2. Read the final-position logits for batch item 0.
    # 3. Clamp `top_k` so it is not larger than the vocabulary size.
    # 4. Use `torch.topk`.
    # 5. Convert token ids to Python ints and logits to Python floats.
    # 6. Return one dictionary per top token.
    #
    # Checks:
    # - `len(predictions) == min(top_k, logits.shape[-1])`
    # - `set(predictions[0]) == {"token_id", "logit"}`
    # - Logits should be sorted from highest to lowest.
    #
    raise NotImplementedError("Return top-k token ids and scores for the final position.")


# Apply a simple logit lens.
# Expected return:
# - top token predictions from an intermediate hidden state
def apply_logit_lens(hidden_state: torch.Tensor, output_head: nn.Module, top_k: int) -> list[TokenPrediction]:
    """Project a hidden state through the output head."""
    # Goal:
    # - Use the output head to inspect what a hidden state already points toward.
    # - This is called logit lens: look through the final output layer at an earlier hidden state.
    #
    # Input:
    # - `hidden_state`: tensor with shape `[batch_size, sequence_length, model_dim]`.
    # - `output_head`: usually `model.output_head`, an `nn.Linear(model_dim, vocab_size)`.
    # - `top_k`: number of token predictions to return.
    #
    # Output:
    # - Return the same format as `get_top_token_predictions`:
    #   `list[TokenPrediction]` with `"token_id"` and `"logit"` keys.
    #
    # Method:
    # - Project hidden states with:
    #   `lens_logits = output_head(hidden_state)`
    # - `lens_logits` should have shape `[batch_size, sequence_length, vocab_size]`.
    # - Then call:
    #   `get_top_token_predictions(lens_logits, top_k)`
    #
    # Steps:
    # 1. Check that `hidden_state.ndim == 3`.
    # 2. Pass it through `output_head`.
    # 3. Reuse `get_top_token_predictions` so the output format stays the same.
    # 4. Return the top token list.
    #
    # Checks:
    # - `lens_logits.shape[:2] == hidden_state.shape[:2]`
    # - `len(result) == min(top_k, lens_logits.shape[-1])`
    #
    raise NotImplementedError("Return top tokens from an intermediate hidden state.")


# Summarize attention patterns.
# Expected return:
# - dictionary with simple attention facts
def summarize_attention_patterns(attention: torch.Tensor) -> AttentionDebugSummary:
    """Summarize attention weights."""
    # Goal:
    # - Summarize which earlier positions receive the most attention.
    # - Attention is useful to inspect, but it is not a full explanation by itself.
    #
    # Input:
    # - `attention`: tensor with shape `[batch_size, num_heads, sequence_length, sequence_length]`.
    # - Last two dimensions mean `[query_position, key_position]`.
    # - Example: `attention[0, 2, 5, 3]` is head 2, query token 5, looking at key token 3.
    #
    # Output:
    # - Return `AttentionDebugSummary`.
    # - Exact keys:
    #   - `"sequence_length"`: int
    #   - `"top_key_positions"`: list[int]
    #   - `"top_attention_values"`: list[float]
    #   - `"mean_self_attention"`: float average attention to the same position
    #
    # Method:
    # - Average attention over batch, heads, and query positions:
    #   `mean_by_key = attention.mean(dim=(0, 1, 2))`
    # - Get top key positions with:
    #   `top_values, top_positions = torch.topk(mean_by_key, k=min(3, mean_by_key.numel()))`
    # - Self-attention is the diagonal of the last two dims:
    #   `mean_self_attention = attention.diagonal(dim1=-2, dim2=-1).mean().item()`
    #
    # Steps:
    # 1. Check that `attention.ndim == 4`.
    # 2. Compute `mean_by_key`.
    # 3. Use `torch.topk` for the top 3 key positions.
    # 4. Compute diagonal self-attention.
    # 5. Convert tensors to Python ints and floats.
    # 6. Return the four exact keys listed above.
    #
    # Checks:
    # - `summary["sequence_length"] == attention.shape[-1]`
    # - `len(summary["top_key_positions"]) <= 3`
    # - `0.0 <= summary["mean_self_attention"] <= 1.0`
    #
    raise NotImplementedError("Return which positions receive the most attention.")


# Patch an activation from one run into another.
# Expected return:
# - dictionary comparing original and patched predictions
def patch_activation_and_compare(
    model: TinyInspectableLM,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    position: int,
) -> PatchingResult:
    """Patch one activation and compare outputs."""
    # Goal:
    # - Replace one hidden state from the target run with the source run's hidden state.
    # - Then compare final-token predictions before and after the patch.
    #
    # Input:
    # - `model`: the inspectable language model.
    # - `source_ids`: `torch.long` tensor with shape `[1, sequence_length]`.
    # - `target_ids`: `torch.long` tensor with shape `[1, sequence_length]`.
    # - `position`: token position to patch. It must satisfy `0 <= position < sequence_length`.
    #
    # Output:
    # - Return `PatchingResult`.
    # - Exact keys:
    #   - `"position"`: patched token position
    #   - `"target_top_token"`: top token id before patching
    #   - `"patched_top_token"`: top token id after patching
    #   - `"logit_gap"`: mean absolute difference between target and patched final logits
    #   - `"patched_top_tokens"`: top token predictions after patching
    #
    # Method:
    # - Run source and target with:
    #   `source_outputs = model(source_ids)`
    #   `target_outputs = model(target_ids)`
    # - Copy hidden states:
    #   `patched_hidden = target_outputs["hidden_states"].clone()`
    #   `patched_hidden[:, position, :] = source_outputs["hidden_states"][:, position, :]`
    # - Project patched hidden states with:
    #   `patched_logits = model.output_head(patched_hidden)`
    # - Compare final-position logits:
    #   `logit_gap = (target_outputs["logits"][0, -1] - patched_logits[0, -1]).abs().mean().item()`
    #
    # Steps:
    # 1. Check that source and target shapes match.
    # 2. Check that `position` is inside the sequence.
    # 3. Put the model in eval mode and use `torch.no_grad()`.
    # 4. Run source and target prompts.
    # 5. Clone target hidden states.
    # 6. Copy the source hidden state at `position` into the target hidden states.
    # 7. Project patched hidden states through `model.output_head`.
    # 8. Get top token ids before and after patching.
    # 9. Return the exact keys listed above.
    #
    # Checks:
    # - `source_ids.shape == target_ids.shape`
    # - `0 <= result["position"] < source_ids.shape[1]`
    # - `result["logit_gap"] >= 0.0`
    # - A changed top token is a clue that this position matters, not proof by itself.
    #
    raise NotImplementedError("Replace one hidden state and compare final predictions.")


# Check for memorized text.
# Expected return:
# - dictionary with match count and examples
def check_memorized_spans(generated_texts: list[str], training_texts: list[str]) -> MemorizationReport:
    """Check whether generated text copies training text."""
    # Goal:
    # - Find generated strings that appear inside the training text.
    # - This is a simple exact-substring check for possible memorization.
    #
    # Input:
    # - `generated_texts`: strings made by the model or toy generator.
    #   Example: `["to be or not to be", "a new short answer"]`
    # - `training_texts`: strings from training data.
    #   Example: `["to be or not to be, that is the question"]`
    #
    # Output:
    # - Return `MemorizationReport`.
    # - Exact keys:
    #   - `"num_generated"`: number of generated strings checked
    #   - `"num_matches"`: number of generated strings found in training text
    #   - `"match_rate"`: `num_matches / num_generated`
    #   - `"matched_texts"`: list of generated strings that matched
    #
    # Method:
    # - Join the training text into one lowercase string:
    #   `training_blob = "\n".join(training_texts).lower()`
    # - For each generated string, check:
    #   `generated.lower() in training_blob`
    # - Formula:
    #   `match_rate = num_matches / num_generated`
    #
    # Steps:
    # 1. Handle an empty `generated_texts` list by returning zeros and an empty match list.
    # 2. Build `training_blob`.
    # 3. Loop over generated strings.
    # 4. Add exact substring matches to `matched_texts`.
    # 5. Compute `num_matches` and `match_rate`.
    # 6. Return the four exact keys listed above.
    #
    # Checks:
    # - `report["num_generated"] == len(generated_texts)`
    # - `0.0 <= report["match_rate"] <= 1.0`
    # - This check only catches exact copied spans, not paraphrases.
    #
    raise NotImplementedError("Find generated strings that appear in training text.")


# Run the full debugging workflow.
# Expected return:
# - summary dictionary with prediction, logit lens, attention, patching, and memorization checks
def run_llm_debugging_demo(config: LLMDebuggingConfig) -> LLMDebuggingSummary:
    """Run the LLM debugging workflow."""
    # Goal:
    # - Run all debugging tools and return one summary dictionary.
    # - This connects model outputs, hidden-state inspection, attention, patching, and memorization checks.
    #
    # Input:
    # - `config`: toy model size and inspection settings.
    #
    # Output:
    # - Return `LLMDebuggingSummary`.
    # - Exact keys:
    #   - `"top_tokens"`: final-position top token predictions
    #   - `"logit_lens"`: top token predictions from hidden states
    #   - `"attention_summary"`: result from `summarize_attention_patterns`
    #   - `"patching_result"`: result from `patch_activation_and_compare`
    #   - `"memorization_check"`: result from `check_memorized_spans`
    #   - `"debugging_note"`: short plain-English note
    #
    # Method:
    # - Call the helpers in this order:
    #   `build_debug_prompts(config)`
    #   `TinyInspectableLM(config)`
    #   `outputs = model(prompts["target"])`
    #   `get_top_token_predictions(outputs["logits"], config.top_k)`
    #   `apply_logit_lens(outputs["hidden_states"], model.output_head, config.top_k)`
    #   `summarize_attention_patterns(outputs["attention"])`
    #   `patch_activation_and_compare(model, prompts["source"], prompts["target"], position=0)`
    #   `check_memorized_spans(generated_texts, training_texts)`
    #
    # Steps:
    # 1. Seed the run with `set_seed(config.random_state)`.
    # 2. Build source and target prompts.
    # 3. Build the tiny model and set it to eval mode.
    # 4. Run the target prompt through the model.
    # 5. Get top final-token predictions.
    # 6. Run logit lens on hidden states.
    # 7. Summarize attention.
    # 8. Patch one activation from source into target.
    # 9. Create tiny `generated_texts` and `training_texts` lists for memorization checking.
    # 10. Write a debugging note that says these tools give clues, not perfect explanations.
    # 11. Return the six exact keys listed above.
    #
    # Checks:
    # - `set(summary) == {"top_tokens", "logit_lens", "attention_summary", "patching_result", "memorization_check", "debugging_note"}`
    # - `len(summary["top_tokens"]) == min(config.top_k, config.vocab_size)`
    # - `isinstance(summary["debugging_note"], str)`
    #
    raise NotImplementedError("Run inspection tools and return a summary.")


# Check the summary.
# Expected return:
# - True when all debugging outputs are present
def check_metrics(metrics: LLMDebuggingSummary) -> bool:
    """Return True when debugging outputs are present."""
    required = {"top_tokens", "logit_lens", "attention_summary", "patching_result", "memorization_check", "debugging_note"}
    return required.issubset(metrics)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the LLM debugging scaffold."""
    config = LLMDebuggingConfig()
    set_seed(config.random_state)

    try:
        metrics = run_llm_debugging_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "llm_debugging_outputs", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
