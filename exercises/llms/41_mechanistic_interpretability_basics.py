"""
============================================================
EXERCISE 41: MECHANISTIC INTERPRETABILITY BASICS
============================================================

WHAT YOU ARE BUILDING
You are building tiny tools for looking inside a model.

Plain-language meaning:
- interpretability means trying to understand what a model is doing
- activations are hidden values inside the model
- attention patterns show which tokens looked at which other tokens
- a probe is a small model that checks what information is present in activations

WHY THIS EXERCISE EXISTS
Some research engineer work asks not only "does it work?" but also "why does it work?"
This is especially useful for language model research and safety work.
This exercise teaches:
- saving activations
- inspecting attention weights
- simple logit lens idea
- tiny probing classifier
- activation patching idea on a toy task

HOW THIS FITS INTO THE BOOTCAMP
Earlier files asked whether a model works.
This file starts asking a different research question:
- what is happening inside while it works or fails?

Beginner intuition:
- hidden states are the model's working notes
- interpretability tools are ways to inspect those notes
- they do not give perfect truth, but they can reveal useful clues

Small example:
- if a probe can predict a label from one layer's hidden states,
- that suggests the layer contains information about that label

DATASET
Use a tiny real text classification dataset.

Clear default:
- Hugging Face dataset: `ag_news`
- text field: `text`
- label field: `label`
- use short headlines or truncate each article to a short fixed length
- start with a tiny subset, such as 512 training examples and 128 validation examples

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- load a tiny subset from AG News
- tokenize the text into short fixed-length token sequences
- keep the labels
- train or load one tiny transformer-like model
- inspect one small batch carefully
Small concrete example:
- imagine short news headlines with four labels
- you can ask:
  - do the hidden states contain topic information?
  - do attention weights point toward useful words?
  - can a small probe recover the label from one layer?
What "mechanistic interpretability" means in plain words:
- try to inspect the inside parts of the model
- look at hidden values, attention patterns, and simple probes
- use those tools to form careful clues about what the model may be doing

STEP-BY-STEP TODO
1. Build or load a tiny transformer classifier.
2. Save activations from one layer.
3. Save attention weights from one batch.
4. Inspect which tokens get high attention.
5. Build a small probe on hidden states.
6. Try a tiny activation patch.
7. Write a short note about what you found.

METRICS TO TRACK
Track at least:
- activation cache saved
- attention summary saved
- probe accuracy
- patch result
- one plain-language note

TARGET RESULT RANGE
Focus on whether the tools produce readable signals.

Useful signs:
- activations have the shapes you expect
- attention summary points to some repeated pattern
- the probe score is above random if the hidden states contain useful information
- the final note says what you observed and what remains uncertain

COMMON MISTAKES
- treating interpretability output as perfect proof
- reading attention as if it explains everything
- using too many examples at once and making the outputs hard to inspect
- writing conclusions without showing the supporting evidence

DEFINITION OF DONE
You are done when:
- activations are captured
- attention weights are inspected
- probe accuracy is reported
- one toy patch result is reported
- you can explain what these tools do in plain words

INPUTS AND OUTPUTS
Input:
- token ids
- hidden states
- attention weights
Output:
- activation cache
- attention summary
- probe score
- patch result

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a tiny transformer classifier.
Keep the model small so the inside of the model is easier to inspect.

LOSS FUNCTION
Use `nn.CrossEntropyLoss` for the toy task.

What each step is doing in simple words:
1. build a tiny model and task
2. save the hidden values
3. save the attention pattern
4. inspect where attention goes
5. test what information is present with a probe
6. change one internal value and see what happens
7. write down what you learned and what you still do not know
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


ToolDescription = dict[str, list[str] | tuple[int, int]]
AttentionSummary = dict[str, int | float | list[int]]
ProbeMetrics = dict[str, float]
PatchMetrics = dict[str, float]
InterpretabilitySummary = dict[str, tuple[int, ...] | AttentionSummary | float | PatchMetrics | str]
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

@dataclass
class InterpretabilityConfig:
    """Settings for mechanistic interpretability basics."""
    vocab_size: int = 32
    sequence_length: int = 8
    model_dim: int = 64
    num_heads: int = 4
    batch_size: int = 16
    random_state: int = 42

def describe_interpretability_tools(config: InterpretabilityConfig) -> ToolDescription:
    """Return the tools this exercise should practice."""
    return {
        "tools": ["activation_cache", "attention_inspection", "probe", "activation_patch"],
        "token_shape": (config.batch_size, config.sequence_length),
    }

# Build a tiny real dataset for the interpretability demo.
# Keep it small so you can actually inspect one batch by hand.
def build_toy_dataloader(config: InterpretabilityConfig) -> DataLoader:
    """Return a toy token classification dataloader."""
    # Goal:
    # - Return a tiny dataloader for a topic-classification task.
    # - Each batch should contain token ids and labels so the model can learn a simple class task.
    #
    # Input:
    # - `config.sequence_length`: number of tokens per example.
    # - `config.batch_size`: number of examples per batch.
    # - `config.vocab_size`: token ids must be in `[0, config.vocab_size - 1]`.
    #
    # Output:
    # - Return a PyTorch `DataLoader`.
    # - Each batch must be a dictionary with exactly these keys:
    #   - `"token_ids"`: `torch.long` tensor with shape `[batch_size, sequence_length]`
    #   - `"label"`: `torch.long` tensor with shape `[batch_size]`
    # - Small batch example:
    #   `token_ids = [[2, 5, 7, 0, 0, 0, 0, 0], [3, 4, 9, 1, 0, 0, 0, 0]]`
    #   `label = [0, 1]`
    #
    # Method:
    # - Use Hugging Face `datasets.load_dataset` if available:
    #   `from datasets import load_dataset`
    #   `raw = load_dataset("ag_news", split="train[:512]")`
    # - `raw[i]` is a dictionary like `{"text": "...", "label": 2}`.
    # - For a simple tokenizer, split on spaces, lowercase each word, and map words to small ids.
    # - Use a small custom `torch.utils.data.Dataset` whose `__getitem__` returns
    #   `{"token_ids": token_ids[index], "label": labels[index]}`.
    # - Build the loader with:
    #   `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`
    #
    # Steps:
    # 1. Load a tiny AG News split.
    # 2. Build a small word-to-id dictionary. Reserve id 0 for padding and id 1 for unknown words.
    # 3. Convert each text to a list of token ids.
    # 4. Truncate longer lists to `config.sequence_length`.
    # 5. Pad shorter lists with 0 until they have length `config.sequence_length`.
    # 6. Convert token ids to a `torch.long` tensor with shape `[num_examples, sequence_length]`.
    # 7. Convert labels to a `torch.long` tensor with shape `[num_examples]`.
    # 8. Return a `DataLoader` that yields the batch dictionary described above.
    #
    # Checks:
    # - `batch = next(iter(loader))`
    # - `set(batch) == {"token_ids", "label"}`
    # - `batch["token_ids"].shape == (config.batch_size, config.sequence_length)`
    # - `batch["token_ids"].dtype == torch.long`
    # - `batch["label"].shape == (config.batch_size,)`
    raise NotImplementedError("Return a dataloader for a tiny real token task.")


# Beginner hints for this file:
# - interpretability asks what the model is doing inside, not only what output it gives
# - activations are the hidden values produced during the forward pass
# - attention weights can show which token positions influenced another token
# - a probe asks whether some useful information is present in hidden states
# - start by inspecting one tiny batch so the hidden values feel less abstract
class TinyInterpretableTransformer(nn.Module):
    """Small transformer that exposes hidden states and attention weights."""
    def __init__(self, config: InterpretabilityConfig) -> None:
        # Goal:
        # - Build a tiny transformer classifier that can expose its inside values.
        #
        # Input:
        # - `config.vocab_size`: number of token ids the embedding can read.
        # - `config.sequence_length`: max sequence length.
        # - `config.model_dim`: hidden size for each token.
        # - `config.num_heads`: number of attention heads.
        #
        # Output:
        # - Return `None`, but create the modules used by `forward`.
        # - The final classifier should output 4 logits because AG News has 4 classes.
        #
        # Method:
        # - Use these exact PyTorch modules:
        #   `self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)`
        #   `self.position_embedding = nn.Embedding(config.sequence_length, config.model_dim)`
        #   `self.attention = nn.MultiheadAttention(config.model_dim, config.num_heads, batch_first=True)`
        #   `self.norm = nn.LayerNorm(config.model_dim)`
        #   `self.classifier = nn.Linear(config.model_dim, 4)`
        # - `batch_first=True` means tensors use shape `[batch, sequence, hidden]`.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Save `self.config = config`.
        # 3. Create token and position embeddings.
        # 4. Create the attention layer.
        # 5. Create a layer norm.
        # 6. Create the classifier head.
        #
        # Checks:
        # - `model.token_embedding.weight.shape == (config.vocab_size, config.model_dim)`
        # - `model.classifier.out_features == 4`
        # - Do not add softmax here. `nn.CrossEntropyLoss` expects raw logits.
        #
        super().__init__()
        self.config = config
        raise NotImplementedError("Build a tiny transformer classifier.")
    def forward(self, token_ids: torch.Tensor, return_cache: bool = False) -> dict[str, torch.Tensor]:
        """Return logits and optionally hidden states and attention weights."""
        # Goal:
        # - Run the tiny transformer and optionally return hidden states and attention weights.
        #
        # Input:
        # - `token_ids`: `torch.long` tensor with shape `[batch_size, sequence_length]`.
        # - `return_cache`: if `True`, include hidden states and attention weights in the output.
        # - Small example shape:
        #   `token_ids.shape == (16, 8)`
        #
        # Output:
        # - Always return a dictionary with:
        #   - `"logits"`: tensor with shape `[batch_size, 4]`
        # - If `return_cache=True`, also return:
        #   - `"hidden_states"`: tensor with shape `[batch_size, sequence_length, model_dim]`
        #   - `"attention_weights"`: tensor with shape `[batch_size, num_heads, sequence_length, sequence_length]`
        #
        # Method:
        # - Create positions with:
        #   `positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0)`
        # - Embed tokens and positions, then add them:
        #   `x = self.token_embedding(token_ids) + self.position_embedding(positions)`
        # - Call attention with:
        #   `attn_output, attn_weights = self.attention(x, x, x, need_weights=True, average_attn_weights=False)`
        # - Pool by averaging over sequence positions:
        #   `pooled = hidden_states.mean(dim=1)`
        #
        # Steps:
        # 1. Read `batch_size` and `sequence_length` from `token_ids.shape`.
        # 2. Build position ids on the same device as `token_ids`.
        # 3. Add token and position embeddings.
        # 4. Run self-attention.
        # 5. Add or normalize hidden states in the simple way you chose in `__init__`.
        # 6. Mean-pool hidden states across sequence positions.
        # 7. Project pooled states with `self.classifier`.
        # 8. Return logits, and return the cache only when requested.
        #
        # Checks:
        # - `outputs["logits"].shape == (token_ids.shape[0], 4)`
        # - If `return_cache=True`, `outputs["hidden_states"].shape[-1] == self.config.model_dim`
        # - Attention weights should have one row per query position and one column per key position.
        #
        raise NotImplementedError("Return logits, activations, and attention weights when requested.")

# Capture hidden activations from the model.
# In plain words:
# - this saves the model's internal working values so you can inspect them later
def capture_activations(model: nn.Module, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return hidden activations from a forward pass."""
    # Goal:
    # - Run the model once and return the cached hidden states.
    # - Activations are the hidden numbers inside the model.
    #
    # Input:
    # - `model`: the `TinyInterpretableTransformer`.
    # - `token_ids`: `torch.long` tensor with shape `[batch_size, sequence_length]`.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"hidden_states"`: tensor with shape `[batch_size, sequence_length, model_dim]`
    #   - `"attention_weights"`: tensor with shape `[batch_size, num_heads, sequence_length, sequence_length]`
    # - You may also keep `"logits"` if it helps, but the two cache keys above are required.
    #
    # Method:
    # - Put the model in eval mode with:
    #   `model.eval()`
    # - Avoid gradient tracking with:
    #   `with torch.no_grad():`
    # - Call:
    #   `outputs = model(token_ids, return_cache=True)`
    #
    # Steps:
    # 1. Move `token_ids` to the same device as the model if needed.
    # 2. Set eval mode.
    # 3. Run the forward pass with `return_cache=True`.
    # 4. Read `"hidden_states"` and `"attention_weights"` from the output dictionary.
    # 5. Return the cache dictionary.
    #
    # Checks:
    # - `cache["hidden_states"].ndim == 3`
    # - `cache["attention_weights"].ndim == 4`
    # - The first dimension of both tensors should match `token_ids.shape[0]`.
    raise NotImplementedError("Run the model and return saved activations.")

# Inspect which token positions attended to which other positions.
# This is a descriptive tool, not perfect proof of model reasoning.
def inspect_attention_patterns(attention_weights: torch.Tensor) -> AttentionSummary:
    """Return a simple summary of attention weights."""
    # Goal:
    # - Summarize which source token positions receive the most attention.
    # - This is a clue about model behavior, not complete proof of the model's reasoning.
    #
    # Input:
    # - `attention_weights`: tensor with shape `[batch_size, num_heads, sequence_length, sequence_length]`.
    # - The last two dimensions mean `[query_position, key_position]`.
    # - Example: `attention_weights[0, 1, 3, 2]` is head 1, query token 3, looking at key token 2.
    #
    # Output:
    # - Return `AttentionSummary`.
    # - Exact keys:
    #   - `"top_key_positions"`: list of int positions with highest average attention
    #   - `"top_attention_values"`: list of float average attention values for those positions
    #   - `"sequence_length"`: int
    #
    # Method:
    # - Average over batch, heads, and query positions:
    #   `mean_by_key = attention_weights.mean(dim=(0, 1, 2))`
    # - Get top positions with:
    #   `top_values, top_positions = torch.topk(mean_by_key, k=min(3, mean_by_key.numel()))`
    #
    # Steps:
    # 1. Check that `attention_weights.ndim == 4`.
    # 2. Compute `mean_by_key`.
    # 3. Use `torch.topk` to find the top 3 key positions.
    # 4. Convert positions to plain Python ints with `.tolist()`.
    # 5. Convert values to plain Python floats with `.tolist()`.
    # 6. Return the three exact keys listed above.
    #
    # Checks:
    # - `len(summary["top_key_positions"]) <= 3`
    # - `summary["sequence_length"] == attention_weights.shape[-1]`
    # - Attention values should usually be between 0.0 and 1.0.
    #
    raise NotImplementedError("Summarize which positions receive high attention.")

# Run a tiny logit-lens check.
# In plain words:
# - take an intermediate hidden state
# - push it through the output head
# - see what tokens it seems to support already
def run_logit_lens(hidden_states: torch.Tensor, output_head: nn.Module) -> torch.Tensor:
    """Project hidden states through the output head as a logit-lens check."""
    # Goal:
    # - Project hidden states through an output head to see what the hidden states already support.
    # - This is called a logit lens because it views hidden states through the final prediction layer.
    #
    # Input:
    # - `hidden_states`: tensor with shape `[batch_size, sequence_length, model_dim]`.
    # - `output_head`: a module like `nn.Linear(model_dim, num_classes)` or `nn.Linear(model_dim, vocab_size)`.
    #
    # Output:
    # - Return logits with shape `[batch_size, sequence_length, output_size]`.
    # - `output_size` is whatever the last dimension of `output_head` produces.
    #
    # Method:
    # - Call the head directly:
    #   `lens_logits = output_head(hidden_states)`
    # - PyTorch `nn.Linear` can read a 3D tensor and applies the linear layer to the last dimension.
    #
    # Steps:
    # 1. Check that `hidden_states.ndim == 3`.
    # 2. Pass `hidden_states` through `output_head`.
    # 3. Return the logits without softmax.
    #
    # Checks:
    # - `lens_logits.shape[:2] == hidden_states.shape[:2]`
    # - Do not apply softmax here. Raw logits are easier to compare.
    #
    raise NotImplementedError("Return logits from intermediate hidden states.")

# Train a tiny probe on hidden activations.
# If the probe predicts well, that suggests useful information is present in those activations.
def train_probe(activations: torch.Tensor, labels: torch.Tensor) -> ProbeMetrics:
    """Train a tiny probe to test what information activations contain."""
    # Goal:
    # - Train a small classifier on frozen activations and report whether it can predict labels.
    # - If the probe does well, the activations likely contain information about the labels.
    #
    # Input:
    # - `activations`: tensor with shape `[batch_size, sequence_length, model_dim]`.
    # - `labels`: `torch.long` tensor with shape `[batch_size]`.
    # - Labels are AG News classes, so there are 4 classes: 0, 1, 2, and 3.
    #
    # Output:
    # - Return `ProbeMetrics`.
    # - Exact keys:
    #   - `"probe_loss"`: final cross-entropy loss as a float
    #   - `"probe_accuracy"`: final accuracy from 0.0 to 1.0
    #
    # Method:
    # - Mean-pool the activations first:
    #   `features = activations.detach().mean(dim=1)`
    # - Build the probe with:
    #   `probe = nn.Linear(features.shape[-1], 4).to(features.device)`
    # - Use:
    #   `loss_fn = nn.CrossEntropyLoss()`
    #   `optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2)`
    # - Cross entropy expects raw logits with shape `[batch_size, 4]` and labels with shape `[batch_size]`.
    #
    # Steps:
    # 1. Detach activations so the probe does not train the original model.
    # 2. Mean-pool across sequence positions to get `[batch_size, model_dim]`.
    # 3. Create the linear probe, loss function, and optimizer.
    # 4. Train for a small fixed number of steps, such as 50.
    # 5. In each step, run logits, compute loss, clear gradients, backpropagate, and step.
    # 6. Compute final predictions with `logits.argmax(dim=-1)`.
    # 7. Compute accuracy as `(predictions == labels).float().mean().item()`.
    # 8. Return the two exact keys listed above.
    #
    # Checks:
    # - `features.shape == (labels.shape[0], activations.shape[-1])`
    # - `0.0 <= metrics["probe_accuracy"] <= 1.0`
    # - A high probe score is a clue, not proof that the model uses that information.
    #
    raise NotImplementedError("Return probe loss and accuracy.")

# Patch one set of activations into another run.
# This is a tiny version of asking:
# - if I replace this internal value, does the output change?
def activation_patch(
    model: nn.Module,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
) -> PatchMetrics:
    """Run a tiny activation patch experiment."""
    # Goal:
    # - Compare clean and corrupted runs, then report how different their logits are.
    # - This is a simple teaching version of activation patching.
    #
    # Input:
    # - `model`: the `TinyInterpretableTransformer`.
    # - `clean_tokens`: tensor with shape `[batch_size, sequence_length]`.
    # - `corrupted_tokens`: tensor with the same shape as `clean_tokens`.
    # - Example:
    #   clean and corrupted batches may differ in one token position.
    #
    # Output:
    # - Return `PatchMetrics`.
    # - Exact keys:
    #   - `"clean_corrupted_logit_gap"`: mean absolute difference between clean and corrupted logits
    #   - `"patched_logit_gap"`: mean absolute difference between clean and patched logits
    # - Smaller patched gap means the patched run became more like the clean run.
    #
    # Method:
    # - For this exercise, use a small direct approximation:
    #   1. Run clean tokens with `return_cache=True`.
    #   2. Run corrupted tokens with `return_cache=True`.
    #   3. Build patched hidden states by copying clean hidden states into the corrupted run.
    #   4. Mean-pool patched hidden states and pass them through `model.classifier`.
    # - Gap formula:
    #   `gap = (first_logits - second_logits).abs().mean().item()`
    #
    # Steps:
    # 1. Check that clean and corrupted tokens have the same shape.
    # 2. Put the model in eval mode.
    # 3. Use `torch.no_grad()`.
    # 4. Run clean and corrupted forward passes with cache.
    # 5. Compute the clean-corrupted logit gap.
    # 6. Copy clean hidden states as the patched hidden states.
    # 7. Mean-pool patched hidden states and run `model.classifier(pooled)`.
    # 8. Compute the clean-patched logit gap.
    # 9. Return the two exact keys listed above.
    #
    # Checks:
    # - `clean_tokens.shape == corrupted_tokens.shape`
    # - Both returned gaps should be non-negative floats.
    # - This toy patch is simple, so explain it as a clue, not proof.
    raise NotImplementedError("Patch activations and report how logits changed.")

def write_interpretability_note(summary: InterpretabilitySummary) -> str:
    """Write a short note about what the tools showed."""
    # Goal:
    # - Write a short note that explains what the tools showed and what they do not prove.
    # - Keep the note careful. Interpretability results are clues, not final proof.
    #
    # Input:
    # - `summary`: dictionary from `run_interpretability_demo`.
    # - Useful keys include:
    #   - `"activation_shape"`
    #   - `"attention_summary"`
    #   - `"probe_accuracy"`
    #   - `"patch_result"`
    #
    # Output:
    # - Return one plain-English string.
    # - Mention at least:
    #   - the activation shape
    #   - the probe accuracy
    #   - one caution, such as "attention is not a full explanation"
    #
    # Method:
    # - Use an f-string, for example:
    #   `note = f"The activation cache had shape {activation_shape}."`
    # - Keep the wording direct and short.
    #
    # Steps:
    # 1. Read `activation_shape` from the summary.
    # 2. Read `probe_accuracy` from the summary.
    # 3. Optionally read the top attention positions.
    # 4. Write one or two sentences.
    # 5. Say that the result is useful evidence but not proof of the model's true reasoning.
    #
    # Checks:
    # - `isinstance(note, str)`
    # - `len(note) > 0`
    # - The note should not claim that attention fully explains the model.
    #
    raise NotImplementedError("Explain the interpretability result in plain words.")

def run_interpretability_demo(config: InterpretabilityConfig) -> InterpretabilitySummary:
    """Run the full interpretability demo."""
    # Goal:
    # - Run the full interpretability workflow on one small batch.
    # - Return shapes and scores that show the tools ran correctly.
    #
    # Input:
    # - `config`: settings for sequence length, hidden size, batch size, and random seed.
    #
    # Output:
    # - Return `InterpretabilitySummary`.
    # - Exact keys:
    #   - `"activation_shape"`: tuple like `(batch_size, sequence_length, model_dim)`
    #   - `"attention_summary"`: result from `inspect_attention_patterns`
    #   - `"probe_accuracy"`: float from 0.0 to 1.0
    #   - `"patch_result"`: result from `activation_patch`
    #   - `"interpretability_note"`: result from `write_interpretability_note`
    #
    # Method:
    # - Call the helpers in this order:
    #   `build_toy_dataloader(config)`
    #   `TinyInterpretableTransformer(config).to(device)`
    #   `capture_activations(model, token_ids)`
    #   `inspect_attention_patterns(cache["attention_weights"])`
    #   `train_probe(cache["hidden_states"], labels)`
    #   `activation_patch(model, clean_tokens, corrupted_tokens)`
    #   `write_interpretability_note(summary)`
    #
    # Steps:
    # 1. Seed the run with `set_seed(config.random_state)`.
    # 2. Build the dataloader and read one batch.
    # 3. Move `token_ids` and `labels` to `device`.
    # 4. Build the model and move it to `device`.
    # 5. Capture activations from the batch.
    # 6. Summarize attention weights.
    # 7. Train a probe on the cached hidden states.
    # 8. Make `corrupted_tokens` by cloning `token_ids` and changing one token id.
    #    Example: `corrupted_tokens[:, 0] = 1`
    # 9. Run the patch experiment.
    # 10. Build the summary dictionary.
    # 11. Add the note from `write_interpretability_note`.
    #
    # Checks:
    # - `summary["activation_shape"] == tuple(cache["hidden_states"].shape)`
    # - `0.0 <= summary["probe_accuracy"] <= 1.0`
    # - `"interpretability_note"` should be a non-empty string.
    #
    raise NotImplementedError("Capture activations, inspect attention, run probe, and patch.")

def check_metrics(metrics: InterpretabilitySummary) -> bool:
    """Return True when interpretability outputs are present."""
    required = {"activation_shape", "attention_summary", "probe_accuracy", "interpretability_note"}
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
    """Run the interpretability scaffold."""
    config = InterpretabilityConfig()
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
    loader, ok = report_task_progress("build_toy_dataloader", build_toy_dataloader, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("describe_interpretability_tools", describe_interpretability_tools, config)
    if not count_or_stop(ok):
        return
    model, ok = report_task_progress("TinyInterpretableTransformer.__init__", TinyInterpretableTransformer, config)
    if not count_or_stop(ok):
        return
    sample_tokens = torch.zeros(config.batch_size, config.sequence_length, dtype=torch.long)
    _, ok = report_task_progress("TinyInterpretableTransformer.forward", model, sample_tokens, True)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_interpretability_demo", run_interpretability_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "interpretability summary", float(int(passed)), "interpretability keys present")
if __name__ == "__main__":
    main()
