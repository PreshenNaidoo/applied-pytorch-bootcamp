"""
============================================================
EXERCISE 46: MIXTURE OF EXPERTS BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny mixture-of-experts layer.

Plain-language meaning:
- an expert is a small neural network
- a router chooses which expert should process each token
- sparse MoE means only a few experts run for each token
- load balancing means trying not to send everything to one expert

WHY THIS EXERCISE EXISTS
Mixture-of-experts is a common way to scale large models.
It increases model size without using all weights for every token.
This exercise teaches:
- expert networks
- router logits
- top-k routing
- combining expert outputs
- load balancing loss
- routing summaries

HOW THIS FITS INTO THE BOOTCAMP
After modern transformer block exercises,
this file adds another scaling idea:
- make the total model larger
- but only use part of it for each token

What each step is doing in simple words:
1. build several small expert networks
2. build a router that scores experts
3. choose the top experts per token
4. send tokens through those experts
5. combine the outputs back into one tensor
6. measure whether the router is overusing one expert
7. explain why sparse compute can scale model size

Beginner intuition:
- not every token may need the same computation
- the router tries to send each token to a useful expert
- this lets the full model be large while using only part of it at a time

Small example:
- one token might be sent to expert 1
- another token might be sent to expert 3
- the router is deciding which expert should handle each token

DATASET
No external dataset is required for the main routing exercise.
Use random token embeddings.

HOW TO CREATE THE INPUTS

Simple path:
- create fake token embeddings
- run the router
- select top experts
- send tokens through the selected experts
- combine the outputs

STEP-BY-STEP TODO
1. Build several small expert MLPs.
2. Build a router that scores each expert.
3. Pick top-1 or top-2 experts per token.
4. Send tokens to selected experts.
5. Combine expert outputs.
6. Compute a simple load balancing loss.
7. Report how often each expert was used.
8. Write a short note about sparse compute.

DEFINITION OF DONE
You are done when:
- router logits have shape `[batch, seq, num_experts]`
- expert outputs keep model dimension unchanged
- expert usage counts are reported
- load balancing loss is returned
- you can explain why MoE can scale model size

INPUTS AND OUTPUTS
Input:
- token embeddings with shape `[batch, seq, model_dim]`
Output:
- router logits
- selected expert ids
- combined output
- load-balancing loss

MODEL ARCHITECTURE (HIGH LEVEL ONLY)

Default path:
- router
- several expert MLPs
- top-k routing
- combine selected expert outputs

LOSS FUNCTION
No main task loss is required for the routing demo.
Add a simple load-balancing loss.
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


MoEShapes = dict[str, tuple[int, ...] | int]
MoEForwardOutput = dict[str, torch.Tensor]
ExpertUsageSummary = dict[str, int | float | list[int] | list[float]]
MoEDemoSummary = dict[str, tuple[int, ...] | ExpertUsageSummary | float | str]
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
class MoEConfig:
    """Settings for mixture-of-experts basics."""
    batch_size: int = 2
    sequence_length: int = 8
    model_dim: int = 64
    hidden_dim: int = 128
    num_experts: int = 4
    top_k: int = 1
    random_state: int = 42

def describe_moe_shapes(config: MoEConfig) -> MoEShapes:
    """Return expected MoE shapes."""
    return {
        "input": (config.batch_size, config.sequence_length, config.model_dim),
        "router_logits": (config.batch_size, config.sequence_length, config.num_experts),
        "output": (config.batch_size, config.sequence_length, config.model_dim),
        "top_k": config.top_k,
    }

# Return toy token embeddings for the routing demo.
# Shape:
# - `[batch_size, sequence_length, model_dim]`
def make_hidden_states(config: MoEConfig) -> torch.Tensor:
    """Return toy hidden states."""
    # Goal:
    # - Return fake token embeddings for testing the MoE layer.
    # - These are not real model tokens. They are just shaped tensors for routing practice.
        #
    # Input:
    # - `config.batch_size`: number of sequences.
    # - `config.sequence_length`: number of token positions per sequence.
    # - `config.model_dim`: hidden size for each token.
        #
    # Output:
    # - Return a `torch.float32` tensor with shape:
    #   `[config.batch_size, config.sequence_length, config.model_dim]`
    # - Small example shape:
    #   if batch is 2, sequence length is 8, and model dim is 64, return shape `[2, 8, 64]`.
        #
    # Method:
    # - Use:
    #   `torch.randn(config.batch_size, config.sequence_length, config.model_dim)`
    # - Seed is set in `main`, so the random tensor stays repeatable for this exercise.
        #
    # Steps:
    # 1. Create the random tensor with the exact shape above.
    # 2. Make sure it is floating point.
    # 3. Return it directly.
        #
    # Checks:
    # - `hidden_states.shape == (config.batch_size, config.sequence_length, config.model_dim)`
    # - `hidden_states.dtype == torch.float32`
        #
    raise NotImplementedError("Return hidden states for MoE testing.")


# Beginner hints for this file:
# - a router chooses which expert should handle each token
# - sparse means not every expert runs for every token
# - this can make the total model larger without using all weights each time
# - if one expert gets almost all tokens, load balancing becomes important
# - report expert usage counts so routing is visible, not hidden
class ExpertMLP(nn.Module):
    """One small expert network."""
    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        # Goal:
        # - Build one expert MLP that maps from `model_dim` back to `model_dim`.
        # - The expert changes the token representation but keeps the same shape.
        #
        # Input:
        # - `model_dim`: input and output hidden size.
        # - `hidden_dim`: wider hidden size inside the expert.
        #
        # Output:
        # - Return `None`, but create the module used by `forward`.
        #
        # Method:
        # - Use this exact PyTorch shape:
        #   `self.net = nn.Sequential(`
        #   `    nn.Linear(model_dim, hidden_dim),`
        #   `    nn.GELU(),`
        #   `    nn.Linear(hidden_dim, model_dim),`
        #   `)`
        # - The last layer returns to `model_dim` so all expert outputs can be combined.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Create `self.net` with the layers above.
        # 3. Do not change batch or sequence dimensions in this module.
        #
        # Checks:
        # - `expert(torch.randn(5, model_dim)).shape == (5, model_dim)`
        # - The expert should not include a router. Routing belongs in `TinyMoELayer`.
        #
        super().__init__()
        raise NotImplementedError("Build one expert MLP.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return expert output with the same final dimension."""
        # Goal:
        # - Run this expert on a batch of token vectors.
        #
        # Input:
        # - `x`: tensor with shape `[num_tokens, model_dim]`.
        # - `num_tokens` means the tokens assigned to this expert.
        #
        # Output:
        # - Return a tensor with shape `[num_tokens, model_dim]`.
        #
        # Method:
        # - Call:
        #   `return self.net(x)`
        #
        # Steps:
        # 1. Pass `x` through `self.net`.
        # 2. Return the result.
        #
        # Checks:
        # - `output.shape == x.shape`
        #
        raise NotImplementedError("Run one expert.")

class TinyMoELayer(nn.Module):
    """Tiny sparse mixture-of-experts layer."""
    def __init__(self, config: MoEConfig) -> None:
        # Goal:
        # - Build the router and the list of expert MLPs.
        # - The router scores experts for each token.
        #
        # Input:
        # - `config.model_dim`: token hidden size.
        # - `config.hidden_dim`: hidden size inside each expert.
        # - `config.num_experts`: number of experts.
        # - `config.top_k`: number of selected experts per token. This exercise starts with `top_k=1`.
        #
        # Output:
        # - Return `None`, but create:
        #   - `self.router`
        #   - `self.experts`
        #
        # Method:
        # - Router:
        #   `self.router = nn.Linear(config.model_dim, config.num_experts)`
        # - Experts:
        #   `self.experts = nn.ModuleList([ExpertMLP(config.model_dim, config.hidden_dim) for _ in range(config.num_experts)])`
        # - `ModuleList` tells PyTorch to track the expert parameters.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Save `self.config = config`.
        # 3. Build `self.router`.
        # 4. Build `self.experts`.
        # 5. For this first exercise, raise a clear error if `config.top_k != 1`, or implement top-k carefully.
        #
        # Checks:
        # - `len(self.experts) == config.num_experts`
        # - `self.router.out_features == config.num_experts`
        #
        super().__init__()
        self.config = config
        raise NotImplementedError("Build router and experts.")
    def forward(self, x: torch.Tensor) -> MoEForwardOutput:
        """Return MoE output, router logits, expert ids, and load balancing loss."""
        # Goal:
        # - Route each token to an expert, run the selected experts, and combine outputs back into the original shape.
        #
        # Input:
        # - `x`: hidden states with shape `[batch_size, sequence_length, model_dim]`.
        # - Small example shape: `[2, 8, 64]`.
        #
        # Output:
        # - Return `MoEForwardOutput`.
        # - Exact keys:
        #   - `"output"`: tensor with shape `[batch_size, sequence_length, model_dim]`
        #   - `"router_logits"`: tensor with shape `[batch_size, sequence_length, num_experts]`
        #   - `"router_probs"`: tensor with shape `[batch_size, sequence_length, num_experts]`
        #   - `"expert_ids"`: tensor with shape `[batch_size, sequence_length, top_k]`
        #   - `"load_balancing_loss"`: scalar tensor with shape `[]`
        #
        # Method:
        # - Router logits:
        #   `router_logits = self.router(x)`
        # - Router probabilities:
        #   `router_probs = torch.softmax(router_logits, dim=-1)`
        # - Top-k routing:
        #   `top_probs, expert_ids = torch.topk(router_probs, k=self.config.top_k, dim=-1)`
        # - For top-1, flatten tokens:
        #   `flat_x = x.reshape(batch_size * sequence_length, model_dim)`
        #   `flat_expert_ids = expert_ids.squeeze(-1).reshape(-1)`
        # - Create the output buffer:
        #   `flat_output = torch.zeros_like(flat_x)`
        # - For each expert id, run only the tokens assigned to that expert.
        #
        # Steps:
        # 1. Check that `x.ndim == 3`.
        # 2. Compute router logits and router probabilities.
        # 3. Pick top experts with `torch.topk`.
        # 4. Flatten tokens from `[batch, seq, dim]` to `[batch * seq, dim]`.
        # 5. For each expert index:
        #    - build a mask like `mask = flat_expert_ids == expert_index`
        #    - select tokens with `expert_input = flat_x[mask]`
        #    - run `self.experts[expert_index](expert_input)`
        #    - write the result back into `flat_output[mask]`
        # 6. Reshape output back to `[batch, seq, model_dim]`.
        # 7. Compute load balancing loss with `compute_load_balancing_loss`.
        # 8. Return the five exact keys listed above.
        #
        # Checks:
        # - `result["output"].shape == x.shape`
        # - `result["router_logits"].shape == (x.shape[0], x.shape[1], self.config.num_experts)`
        # - `result["expert_ids"].shape == (x.shape[0], x.shape[1], self.config.top_k)`
        # - At least one expert should be selected.
        #
        raise NotImplementedError("Route tokens to experts and combine outputs.")

# Count how often each expert was selected.
# This is one of the main sanity checks in the MoE exercise.
def compute_expert_usage(expert_ids: torch.Tensor, num_experts: int) -> ExpertUsageSummary:
    """Return how often each expert was selected."""
    # Goal:
    # - Count how often each expert was selected by the router.
    # - This makes routing visible so you can see if one expert gets most tokens.
    #
    # Input:
    # - `expert_ids`: tensor with shape `[batch_size, sequence_length, top_k]`.
    # - Each value is an expert id from `0` to `num_experts - 1`.
    # - `num_experts`: total number of experts.
    #
    # Output:
    # - Return `ExpertUsageSummary`.
    # - Exact keys:
    #   - `"counts"`: list of int counts, one per expert
    #   - `"fractions"`: list of float fractions, one per expert
    #   - `"total_assignments"`: total number of expert assignments
    #   - `"most_used_expert"`: expert id with the highest count
    #
    # Method:
    # - Flatten selected ids:
    #   `flat_ids = expert_ids.reshape(-1)`
    # - Count with:
    #   `counts = torch.bincount(flat_ids, minlength=num_experts)`
    # - Fractions:
    #   `fractions = counts.float() / counts.sum().clamp_min(1)`
    #
    # Steps:
    # 1. Flatten `expert_ids`.
    # 2. Count how many times each expert id appears.
    # 3. Compute fractions from counts.
    # 4. Find the most-used expert with `counts.argmax().item()`.
    # 5. Convert tensors to Python lists and ints.
    # 6. Return the four exact keys listed above.
    #
    # Checks:
    # - `len(summary["counts"]) == num_experts`
    # - `sum(summary["counts"]) == summary["total_assignments"]`
    # - The fractions should sum to about 1.0 when there is at least one assignment.
    #
    raise NotImplementedError("Count expert usage.")

# Compute a small load-balancing loss.
# This encourages the router not to send almost everything to one expert.
def compute_load_balancing_loss(router_probs: torch.Tensor, expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Return a simple load balancing loss."""
    # Goal:
    # - Return a scalar loss that gets larger when routing is very uneven.
    # - This encourages the router not to send almost every token to the same expert.
    #
    # Input:
    # - `router_probs`: tensor with shape `[batch_size, sequence_length, num_experts]`.
    # - `expert_ids`: tensor with shape `[batch_size, sequence_length, top_k]`.
    # - `num_experts`: total number of experts.
    #
    # Output:
    # - Return one scalar tensor with shape `[]`.
    #
    # Method:
    # - Compute two vectors of length `num_experts`:
    #   - `importance`: average router probability per expert
    #   - `load`: fraction of selected assignments per expert
    # - Formula:
    #   `importance = router_probs.mean(dim=(0, 1))`
    #   `counts = torch.bincount(expert_ids.reshape(-1), minlength=num_experts).float()`
    #   `load = counts / counts.sum().clamp_min(1)`
    #   `loss = num_experts * torch.sum(importance * load)`
    # - If routing is perfectly even, this loss is closer to 1.
    # - If one expert dominates, this loss becomes larger.
    #
    # Steps:
    # 1. Check that the last dimension of `router_probs` is `num_experts`.
    # 2. Average router probabilities over batch and sequence.
    # 3. Count selected expert ids.
    # 4. Convert counts to fractions.
    # 5. Apply the formula above.
    # 6. Return the scalar loss tensor.
    #
    # Checks:
    # - `loss.ndim == 0`
    # - `loss.item() >= 0.0`
    # - Check a tiny case where all experts are used evenly and one where all tokens go to expert 0.
    raise NotImplementedError("Compute a small load balancing loss.")

def write_moe_note(usage_summary: ExpertUsageSummary) -> str:
    """Write a short note about routing and sparse compute."""
    # Goal:
    # - Explain the routing result and sparse compute idea in plain words.
    #
    # Input:
    # - `usage_summary`: result from `compute_expert_usage`.
    # - Useful keys include `"counts"`, `"fractions"`, and `"most_used_expert"`.
    #
    # Output:
    # - Return one short string.
    # - Mention which expert was used most and why uneven usage can be a problem.
    #
    # Method:
    # - Use an f-string, for example:
    #   `f"Expert {most_used} was used most. Sparse MoE runs only selected experts for each token."`
    #
    # Steps:
    # 1. Read the most-used expert id.
    # 2. Read the usage counts or fractions.
    # 3. Write one or two direct sentences.
    # 4. Say that sparse compute means only selected experts run for each token.
    # 5. Say that load balancing matters if routing is too concentrated.
    #
    # Checks:
    # - `isinstance(note, str)`
    # - The note should mention `"expert"` and `"selected"` or `"sparse"`.
    #
    raise NotImplementedError("Explain MoE routing in plain words.")

# Run the full MoE layer demo.
# The most useful outputs are often:
# - which experts were chosen
# - how often they were used
# - whether routing became too concentrated
def run_moe_demo(config: MoEConfig) -> MoEDemoSummary:
    """Run the full MoE demo."""
    # Goal:
    # - Run the MoE layer on toy hidden states and return routing summaries.
    # - The main point is to inspect which experts were used and whether usage was balanced.
    #
    # Input:
    # - `config`: MoE shape and routing settings.
    #
    # Output:
    # - Return `MoEDemoSummary`.
    # - Exact keys:
    #   - `"output_shape"`: tuple like `(batch_size, sequence_length, model_dim)`
    #   - `"router_shape"`: tuple like `(batch_size, sequence_length, num_experts)`
    #   - `"expert_usage"`: result from `compute_expert_usage`
    #   - `"load_balancing_loss"`: float from the scalar loss tensor
    #   - `"moe_note"`: result from `write_moe_note`
    #
    # Method:
    # - Call the helpers in this order:
    #   `hidden_states = make_hidden_states(config)`
    #   `moe = TinyMoELayer(config)`
    #   `outputs = moe(hidden_states)`
    #   `usage = compute_expert_usage(outputs["expert_ids"], config.num_experts)`
    #   `note = write_moe_note(usage)`
    #
    # Steps:
    # 1. Seed the run with `set_seed(config.random_state)`.
    # 2. Create hidden states.
    # 3. Build the MoE layer.
    # 4. Run the forward pass.
    # 5. Compute expert usage from selected expert ids.
    # 6. Convert the load-balancing loss to a Python float with `.item()`.
    # 7. Write the MoE note.
    # 8. Return the five exact keys listed above.
    #
    # Checks:
    # - `summary["output_shape"] == (config.batch_size, config.sequence_length, config.model_dim)`
    # - `summary["router_shape"] == (config.batch_size, config.sequence_length, config.num_experts)`
    # - `"counts"` should be present in `summary["expert_usage"]`.
    raise NotImplementedError("Run MoE layer and return routing summaries.")

def check_metrics(metrics: MoEDemoSummary) -> bool:
    """Return True when MoE outputs are present."""
    required = {"output_shape", "router_shape", "expert_usage", "load_balancing_loss", "moe_note"}
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
    """Run the MoE scaffold."""
    config = MoEConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 4
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    _, ok = report_task_progress("describe_moe_shapes", describe_moe_shapes, config)
    if not count_or_stop(ok):
        return
    moe, ok = report_task_progress("TinyMoELayer.__init__", TinyMoELayer, config)
    if not count_or_stop(ok):
        return
    hidden_states = torch.zeros(config.batch_size, config.sequence_length, config.model_dim)
    _, ok = report_task_progress("TinyMoELayer.forward", moe, hidden_states)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_moe_demo", run_moe_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "moe summary", float(int(passed)), "MoE keys present")
if __name__ == "__main__":
    main()
