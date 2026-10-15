"""
============================================================
EXERCISE 38: TINY RLHF LOOP FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building a tiny RLHF-style training loop from scratch.

This is not a real language model.
It is a small toy version of the part of RLHF where an LLM is trained from
reward scores.

The most important map is:

```text
real RLHF                         this toy exercise
----------------------------------------------------------
LLM                               TinyPolicy
real prompt text                  prompt id, like 0
full generated answer             answer id, like 2
reward model                      reward_table
LLM token probabilities           log_probs
LLM fine-tuning update            loss.backward() + optimizer.step()
old frozen LLM                    reference_policy
```

Why the class is called `TinyPolicy`:
- in reinforcement learning, a policy is the model that chooses an action
- in RLHF, the LLM is treated as the policy
- the LLM's action is the answer it generates
- in this toy exercise, the action is just one answer id

So:

```text
normal RL:
state or observation -> policy -> action -> reward

RLHF:
prompt -> LLM policy -> generated answer -> reward model score

this exercise:
prompt id -> TinyPolicy -> answer id -> reward_table score
```

WHAT THIS EXERCISE SKIPS
Real RLHF usually has two jobs.

Job 1 trains the reward model:

```text
prompt + answer A + answer B + human choice
-> train reward model to score better answers higher
```

Job 2 trains the LLM with that reward model:

```text
1. freeze the reward model
2. give prompts to the LLM
3. LLM samples answers
4. reward model scores those answers
5. use the score and answer log-probabilities to compute a loss
6. backprop updates the LLM weights
7. repeat
```

This exercise only covers Job 2.
It skips reward model training.
The fixed `reward_table` is pretending to be an already-trained reward model.

HOW THE LLM UPDATE IS REPRESENTED HERE
In a real LLM, an answer is many tokens.

Example:

```text
prompt = "Explain gravity simply."
generated answer = "Gravity pulls objects toward each other."
```

The LLM generated that answer token by token.
Training keeps the log-probabilities of the tokens it chose.
If the reward is high, the update makes those chosen tokens more likely in
that prompt context.

This toy exercise compresses the full answer into one id:

```text
prompt id 0 -> answer id 2 -> reward 1.0
```

That means:

```text
for prompt 0, make answer 2 more likely next time
```

The actual model update happens later in the training loop with:

```text
loss.backward()
optimizer.step()
```

`loss.backward()` computes how the policy weights should change.
`optimizer.step()` applies that change.

Tiny world:
- there are `num_prompts` prompts, numbered `0` to `num_prompts - 1`
- there are `num_answers` possible answers, numbered `0` to `num_answers - 1`
- the policy sees a prompt id and chooses one answer id
- the reward table gives a score for each prompt-answer pair
- the reference policy is a frozen copy of the starting policy

Plain-language meanings:
- policy means the model that chooses an action
- in this exercise, the action is an answer id
- in real RLHF, the policy is the LLM and the action is the generated answer
- reference policy means the older model we do not want to drift too far from
- reward model means the scoring rule that says how good an answer was
- reward table means the toy version of that reward model
- KL-style penalty means a stay-close penalty against the reference policy
- RLHF update means improve reward while limiting policy drift

WHY THIS EXERCISE EXISTS
the related exercise connects RL and preference optimization.
This part turns that bridge into a tiny full loop.
It keeps the pieces small, but the workflow is end to end.

HOW THIS FITS INTO THE BOOTCAMP
Earlier RL exercises taught:
- sample an action
- get a reward
- update the policy

the related exercise part 1 connected those ideas to RLHF and DPO.
This part now runs the small RLHF-style loop itself.

Beginner intuition:
- prompt id `0` is a tiny stand-in for a user prompt
- answer id `3` is a tiny stand-in for a full generated answer
- the reward table is a tiny stand-in for a trained reward model
- `TinyPolicy` is a tiny stand-in for an LLM being used as an RL policy
- the policy should learn to choose higher-reward answers
- the reference policy keeps the new policy from changing too wildly

Tiny concrete example:
- prompt id: `0`
- answer rewards for prompt `0`: `[0.0, 0.2, 1.0, 0.1, -0.2]`
- answer id `2` is best because it has reward `1.0`
- if the policy samples answer id `2`, the update should make answer id `2`
  more likely for prompt id `0`
- if the new policy already made answer id `2` much more likely than the
  reference policy, the stay-close penalty should push back a little

WHAT IS FIXED AND WHAT CHANGES
Fixed:
- the reward table
- the reference policy after it is copied and frozen
- the list of prompt ids used for training

Changes:
- the current policy weights
- the probabilities that the current policy gives to each answer

The object that makes decisions:
- the current policy chooses answer ids

The object that gives feedback:
- the reward table gives the reward score

The object that limits drift:
- the frozen reference policy gives baseline log-probabilities

Main loop in plain words:
1. look at all prompt ids
2. use the current policy to sample one answer id per prompt
3. look up the reward for each sampled answer
4. score the same sampled answers under the current policy
5. score the same sampled answers under the frozen reference policy
6. compute a reward-seeking loss with a stay-close penalty
7. call `loss.backward()` to compute weight changes
8. call `optimizer.step()` to update only the current policy
9. repeat
10. evaluate whether the learned policy gets better rewards

The real LLM version of the same loop is:

```text
1. give text prompts to the LLM
2. the LLM samples full text answers
3. the reward model scores those full answers
4. the code reads the log-probabilities of the tokens the LLM generated
5. the loss uses reward * log-probability to decide the update direction
6. backprop updates the LLM weights
7. repeat
```

INPUTS AND OUTPUTS, WITH SHAPES
Inputs:
- `prompt_ids`: tensor with shape `[batch_size]`
- `answer_ids`: tensor with shape `[batch_size]`
- `reward_table`: tensor with shape `[num_prompts, num_answers]`
- policy logits: tensor with shape `[batch_size, num_answers]`
- sampled log-probabilities: tensor with shape `[batch_size]`

Output summary:
- `initial_mean_reward`: mean greedy-policy reward before training
- `final_mean_reward`: mean greedy-policy reward after training
- `mean_kl_penalty`: average stay-close penalty during training
- `learning_succeeded`: whether final reward improved
- `rlhf_note`: one plain-language note about what happened

LOSS FUNCTION
Use a tiny policy-gradient-style loss:

```text
rlhf_score = reward - kl_coef * (new_log_prob - reference_log_prob)
loss = -(rlhf_score.detach() * new_log_prob).mean()
```

What each part means:
- `reward`: score for the sampled answer
- `kl_coef`: how strongly we punish drift from the reference policy
- `new_log_prob`: log-probability from the current policy for the sampled answer
- `reference_log_prob`: log-probability from the frozen reference policy for the same sampled answer
- `rlhf_score`: high when the answer got good reward and did not drift too far
- `.detach()`: treats `rlhf_score` like a fixed score for this update

Why the loss uses `new_log_prob`:
- reward is just a number from the table
- by itself, reward has no gradient back to the policy
- multiplying by `new_log_prob` tells PyTorch which sampled answer should become more or less likely

This is a teaching version, not full PPO.
Real RLHF often adds value models, advantage estimates, PPO clipping, batching over generated text,
and a more careful KL calculation.

METRICS TO TRACK
Track at least:
- initial mean reward
- final mean reward
- mean stay-close penalty
- whether final reward improved

Higher mean reward is better.
If reward does not improve, try more update steps, a smaller learning rate, or another random seed.

STEP-BY-STEP TODO
1. Build the tiny policy.
2. Build a fixed reward table.
3. Sample answers from the current policy.
4. Score already-sampled answers under any policy.
5. Evaluate the greedy policy before training.
6. Compute the tiny RLHF-style policy loss.
7. Run the training loop.
8. Evaluate the greedy policy after training.
9. Return a summary that says whether learning worked.

DEFINITION OF DONE
You are done when:
- the reward table has shape `[num_prompts, num_answers]`
- sampled answers and log-probabilities have shape `[batch_size]`
- the reference policy is frozen
- only the current policy is updated
- final mean reward is usually higher than initial mean reward
- you can explain how reward, log-probability, and reference-policy penalty fit together

COMMON MISTAKES
- treating reward as a loss by itself
- forgetting that reward must affect the sampled answer log-probability
- updating the reference policy by accident
- swapping prompt ids and answer ids
- checking only loss and not checking whether reward improved
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


device = "cuda" if torch.cuda.is_available() else "cpu"


# Learning order for this exercise:
# 1. Build the reward table first so the goal is visible.
# 2. Build the tiny policy and check one forward pass.
# 3. Sample answers and inspect their log-probabilities.
# 4. Score the same answers under the current and reference policies.
# 5. Write the loss only after the shapes are clear.
# 6. Run the full loop and check whether reward improved.


def set_seed(seed: int = 42) -> None:
    """Set the main random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a short result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")


@dataclass
class TinyRLHFLoopConfig:
    """Settings for the tiny RLHF loop."""

    num_prompts: int = 4
    num_answers: int = 5
    hidden_dim: int = 32
    kl_coef: float = 0.1
    learning_rate: float = 1e-2
    update_steps: int = 20
    random_state: int = 42


class TinyRLHFSummary(TypedDict):
    """Final summary from the tiny RLHF loop."""

    initial_mean_reward: float
    final_mean_reward: float
    mean_kl_penalty: float
    learning_succeeded: bool
    rlhf_note: str


class TinyPolicy(nn.Module):
    """Tiny stand-in for an LLM used as an RL policy."""

    def __init__(self, config: TinyRLHFLoopConfig) -> None:
        # Where this fits:
        # This is the current policy, the small model that learns.
        # In real RLHF, the current policy is the LLM being fine-tuned.
        # Here, `TinyPolicy` is the toy LLM.
        #
        # A real LLM maps text tokens to next-token scores.
        # This toy policy maps one prompt id to scores over answer ids.
        #
        # The answer id is pretending to be a whole generated answer.
        #
        # Goal:
        # Build two layers:
        # - an embedding table that turns prompt ids into vectors
        # - a linear layer that turns each vector into answer scores
        #
        # Input:
        # - `config.num_prompts`: number of prompt ids, for example `4`
        # - `config.hidden_dim`: vector size, for example `32`
        # - `config.num_answers`: number of answer ids, for example `5`
        #
        # Output:
        # - Store layers on `self`:
        #   - `self.prompt_embedding`
        #   - `self.answer_head`
        #
        # Method:
        # Use:
        # - `nn.Embedding(config.num_prompts, config.hidden_dim)`
        # - `nn.Linear(config.hidden_dim, config.num_answers)`
        #
        # What the layers mean:
        # - `nn.Embedding(...)` gives each prompt id its own learned vector
        # - `nn.Linear(...)` converts that vector into one score per answer id
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Create `self.prompt_embedding`.
        # 3. Create `self.answer_head`.
        #
        # Checks:
        # - `self.prompt_embedding.weight.shape` should be `[num_prompts, hidden_dim]`.
        # - `self.answer_head.weight.shape` should be `[num_answers, hidden_dim]`.
        super().__init__()

        self.prompt_embedding = nn.Embedding(num_embeddings=config.num_prompts, embedding_dim=config.hidden_dim)
        self.linear = nn.Linear(in_features=config.hidden_dim, out_features=config.num_answers)

        raise NotImplementedError("Build the tiny policy.")

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Return answer logits."""
        # Where this fits:
        # The training loop calls this to get answer scores before sampling.
        # In a real LLM, the forward pass gives token scores.
        # In this toy version, the forward pass gives answer-id scores.
        #
        # Goal:
        # Turn prompt ids into one score per possible answer.
        #
        # Input:
        # - `prompt_ids`: shape `[batch_size]`, dtype `torch.long`
        # - Example: `torch.tensor([0, 1, 2, 3])`
        #
        # Output:
        # - logits with shape `[batch_size, num_answers]`
        # - Example shape with defaults: `[4, 5]`
        #
        # Method:
        # Use the layers created in `__init__`:
        # ```python
        # hidden = self.prompt_embedding(prompt_ids)
        # logits = self.answer_head(hidden)
        # ```
        #
        # Meaning:
        # - `hidden[i]` is the learned vector for prompt `prompt_ids[i]`
        # - `logits[i, j]` is the policy score for answer `j` on that prompt
        #
        # Steps:
        # 1. Pass `prompt_ids` through `self.prompt_embedding`.
        # 2. Pass the hidden vectors through `self.answer_head`.
        # 3. Return the logits.
        #
        # Checks:
        # - With default config and 4 prompt ids, output shape should be `[4, 5]`.
        # - Do not apply softmax here. Sampling will handle that later.
        raise NotImplementedError("Return answer logits.")


def build_reward_table(config: TinyRLHFLoopConfig) -> torch.Tensor:
    """Return fixed reward scores for prompt-answer pairs."""
    # Where this fits:
    # This is the tiny reward model.
    # It does not learn. It gives the feedback used by the policy update.
    #
    # Real RLHF note:
    # In a real RLHF pipeline, this reward model was trained earlier from
    # human comparisons like "answer B is better than answer A".
    # This exercise skips that training step.
    # The fixed table below is pretending that the reward model already exists.
    #
    # Goal:
    # Build a tiny reward model as a table.
    #
    # Input:
    # - `config.num_prompts`: number of prompt rows
    # - `config.num_answers`: number of answer columns
    #
    # Output:
    # - `reward_table`: tensor with shape `[num_prompts, num_answers]`
    # - dtype should be floating point, for example `torch.float32`
    #
    # Meaning:
    # - `reward_table[prompt_id, answer_id]` is the reward for that answer.
    #
    # Tiny example:
    # ```text
    # reward_table[0] = [0.0, 0.2, 1.0, 0.1, -0.2]
    # ```
    # For prompt `0`, answer `2` is best because reward `1.0` is highest.
    #
    # Method:
    # Use `torch.full(...)` to start with small low rewards.
    # Then set one preferred answer per prompt to `1.0`.
    #
    # Suggested pattern:
    # ```python
    # reward_table = torch.full((config.num_prompts, config.num_answers), -0.2)
    # preferred_answers = torch.arange(config.num_prompts) % config.num_answers
    # reward_table[torch.arange(config.num_prompts), preferred_answers] = 1.0
    # ```
    #
    # You may also add a few medium rewards like `0.2` if you want the table
    # to look less all-or-nothing.
    #
    # Steps:
    # 1. Create a tensor with shape `[config.num_prompts, config.num_answers]`.
    # 2. Choose one preferred answer id for each prompt id.
    # 3. Set each preferred prompt-answer pair to reward `1.0`.
    # 4. Return the reward table on `device`.
    #
    # Checks:
    # - `reward_table.shape == (config.num_prompts, config.num_answers)`
    # - `reward_table.max()` should be `1.0`.
    # - Every prompt row should have at least one clearly best answer.
    raise NotImplementedError("Build reward table.")


def sample_answers(policy: nn.Module, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample answers and return their log probabilities."""
    # Where this fits:
    # This is the "policy chooses an action" step.
    # Here, an action is an answer id.
    #
    # Goal:
    # Sample one answer per prompt from the current policy.
    #
    # Input:
    # - `policy`: TinyPolicy
    # - `prompt_ids`: shape `[batch_size]`, dtype `torch.long`
    # - Example: `torch.tensor([0, 1, 2, 3])`
    #
    # Output:
    # - `answer_ids`: `[batch_size]`
    # - `log_probs`: `[batch_size]`
    #
    # Tiny numeric example:
    # If the policy gives answer scores `[2.0, 0.5]`, then:
    # - probabilities are about `[0.82, 0.18]`
    # - answer `0` is sampled about 82% of the time
    # - if answer `0` was sampled, `log_prob` is `log(0.82)`, about `-0.20`
    #
    # A log-probability closer to `0` means the sampled answer was more likely.
    # A more negative log-probability means it was less likely.
    #
    # Method:
    # Use `torch.distributions.Categorical(logits=logits)`.
    # This object turns raw answer scores into a distribution you can sample from.
    #
    # Close simple version:
    # ```python
    # probs = torch.softmax(logits, dim=-1)
    # answer_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    # ```
    #
    # Prefer `Categorical` here because it also gives `.log_prob(...)`.
    #
    # Steps:
    # 1. Move `prompt_ids` to `device`.
    # 2. Get `logits = policy(prompt_ids)`.
    # 3. Build `dist = torch.distributions.Categorical(logits=logits)`.
    # 4. Sample `answer_ids = dist.sample()`.
    # 5. Read `log_probs = dist.log_prob(answer_ids)`.
    # 6. Return `(answer_ids, log_probs)`.
    #
    # Checks:
    # - `answer_ids.shape == prompt_ids.shape`
    # - `log_probs.shape == prompt_ids.shape`
    # - `answer_ids.dtype` should be `torch.long`.
    raise NotImplementedError("Sample answers from the policy.")


def log_prob_of_answers(policy: nn.Module, prompt_ids: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
    """Return log probabilities for chosen answers."""
    # Where this fits:
    # The same sampled answer must be scored under both policies:
    # - current policy
    # - frozen reference policy
    #
    # Goal:
    # Score already-chosen answers under a policy.
    #
    # Input:
    # - `policy`: TinyPolicy
    # - `prompt_ids`: shape `[batch_size]`
    # - `answer_ids`: shape `[batch_size]`
    #
    # Output:
    # - selected log-probabilities with shape `[batch_size]`
    #
    # Tiny example:
    # ```text
    # log_probs for one prompt = [-0.20, -1.70]
    # answer_id = 0
    # returned value = -0.20
    # ```
    #
    # Method:
    # Use:
    # - `logits = policy(prompt_ids)`
    # - `all_log_probs = F.log_softmax(logits, dim=-1)`
    # - `selected = all_log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)`
    #
    # What `gather` does:
    # - `all_log_probs` has one log-probability for every answer id
    # - `answer_ids` says which answer was actually sampled
    # - `gather` picks only those sampled answer log-probabilities
    #
    # Steps:
    # 1. Move `prompt_ids` and `answer_ids` to `device`.
    # 2. Get logits from the policy.
    # 3. Convert logits to log-probabilities with `F.log_softmax(logits, dim=-1)`.
    # 4. Use `gather(...)` to pick the sampled answer columns.
    # 5. Return a tensor with shape `[batch_size]`.
    #
    # Checks:
    # - Output shape should be `[batch_size]`.
    # - Values should usually be `0` or negative.
    # - Do not sample again inside this function. Score the given `answer_ids`.
    raise NotImplementedError("Compute log probabilities for chosen answers.")


def evaluate_policy_reward(policy: nn.Module, reward_table: torch.Tensor) -> float:
    """Return mean reward from the policy's greedy answers."""
    # Where this fits:
    # This is the final test step.
    # It measures behavior. It does not train the policy.
    #
    # Goal:
    # For every prompt id, choose the answer with the highest policy probability.
    # Then look up the reward for those chosen answers.
    #
    # Input:
    # - `policy`: TinyPolicy
    # - `reward_table`: shape `[num_prompts, num_answers]`
    #
    # Output:
    # - Python `float`
    # - Mean reward across all prompt ids.
    #
    # Tiny example:
    # ```text
    # greedy answers = [2, 0, 4, 1]
    # rewards = [1.0, 0.5, 1.0, 0.2]
    # mean reward = 0.675
    # ```
    #
    # Method:
    # Use:
    # - `prompt_ids = torch.arange(reward_table.shape[0], device=device)`
    # - `logits = policy(prompt_ids)`
    # - `answer_ids = logits.argmax(dim=-1)`
    # - `rewards = reward_table[prompt_ids, answer_ids]`
    #
    # Steps:
    # 1. Put the policy in eval mode with `policy.eval()`.
    # 2. Build one prompt id for every row in the reward table.
    # 3. Use the policy logits to choose the highest-scoring answer id.
    # 4. Look up reward for each prompt-answer pair.
    # 5. Return the mean reward as a Python float with `.item()`.
    #
    # Checks:
    # - Do this inside `torch.no_grad()` because this function only measures.
    # - Higher is better.
    # - This is a behavior check, not a training loss.
    raise NotImplementedError("Evaluate greedy policy reward.")


def rlhf_policy_loss(
    rewards: torch.Tensor,
    new_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    kl_coef: float,
) -> torch.Tensor:
    """Return a tiny RLHF-style policy loss."""
    # Where this fits:
    # This is the update rule.
    # It tells the policy which sampled answers should become more likely.
    #
    # Goal:
    # Increase the log-probability of high-scoring sampled answers.
    # Also reduce the push when the current policy drifts too far from the reference policy.
    #
    # Input:
    # - `rewards`: shape `[batch_size]`
    #   Reward for each sampled answer.
    # - `new_log_probs`: shape `[batch_size]`
    #   Log-probability from the current policy for each sampled answer.
    # - `reference_log_probs`: shape `[batch_size]`
    #   Log-probability from the frozen reference policy for the same sampled answer.
    # - `kl_coef`: float
    #   How strongly to push back against policy drift.
    #
    # Output:
    # - scalar tensor with shape `[]`
    # - This is the loss that will receive `.backward()`.
    #
    # Important idea:
    # Reward is just a number from the table.
    # Reward has no gradient back to the policy by itself.
    # The policy changes because the reward-weighted score is multiplied by `new_log_probs`.
    #
    # Real LLM connection:
    # In a real LLM, `new_log_probs` would come from the tokens the model generated.
    # If the answer was:
    # ```text
    # "Gravity pulls objects together."
    # ```
    # the training code would keep the log-probability of each chosen token.
    # A high reward would push those chosen tokens to become more likely in
    # the same kind of prompt context.
    #
    # This toy exercise compresses that whole generated answer into one
    # `answer_id`, so there is only one selected log-probability per prompt.
    #
    # Tiny formula:
    # ```text
    # drift = new_log_prob - reference_log_prob
    # rlhf_score = reward - kl_coef * drift
    # loss = -(rlhf_score.detach() * new_log_prob).mean()
    # ```
    #
    # What this means:
    # - high reward makes the sampled answer more likely next time
    # - high drift reduces the push if the new policy already moved far from the reference
    # - `.detach()` treats `rlhf_score` as a fixed score for this one update
    #
    # Small numeric example:
    # ```text
    # reward = 1.0
    # new_log_prob = -0.30
    # reference_log_prob = -0.80
    # kl_coef = 0.1
    # drift = -0.30 - (-0.80) = 0.50
    # rlhf_score = 1.0 - 0.1 * 0.50 = 0.95
    # loss contribution = -(0.95 * -0.30) = 0.285
    # ```
    #
    # If the sampled answer had low reward, the score is smaller,
    # so the update does not push that answer up as much.
    #
    # Method:
    # Use:
    # ```python
    # drift = new_log_probs - reference_log_probs
    # rlhf_scores = rewards - kl_coef * drift
    # loss = -(rlhf_scores.detach() * new_log_probs).mean()
    # ```
    #
    # Steps:
    # 1. Check that all three tensors have shape `[batch_size]`.
    # 2. Compute `drift`.
    # 3. Compute `rlhf_scores`.
    # 4. Detach `rlhf_scores`.
    # 5. Multiply detached scores by `new_log_probs`.
    # 6. Negate the mean because PyTorch minimizes losses.
    # 7. Return the scalar loss.
    #
    # Checks:
    # - The output should be a scalar tensor.
    # - Do not detach `new_log_probs`; gradients must flow through it.
    # - `reference_log_probs` should come from the frozen reference policy.
    # - This is a tiny teaching loss, not full PPO.
    raise NotImplementedError("Compute tiny RLHF policy loss.")


def run_tiny_rlhf_loop(config: TinyRLHFLoopConfig) -> TinyRLHFSummary:
    """Run the full tiny RLHF-style loop."""
    # Where this fits:
    # This is the full tiny workflow.
    # It should call the helper functions instead of hiding all logic in one place.
    #
    # Goal:
    # Train the current policy to choose higher-reward answers while staying near
    # a frozen reference policy.
    #
    # Real LLM connection:
    # This is the toy version of fine-tuning the LLM after the reward model
    # has already been trained and frozen.
    # The line `optimizer.step()` is where the toy policy weights change.
    # In real RLHF, that step changes the LLM weights instead.
    #
    # Input:
    # - `config`: TinyRLHFLoopConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"initial_mean_reward"`: float
    # - `"final_mean_reward"`: float
    # - `"mean_kl_penalty"`: float
    # - `"learning_succeeded"`: bool
    # - `"rlhf_note"`: str
    #
    # Output keys:
    # - `initial_mean_reward`
    # - `final_mean_reward`
    # - `mean_kl_penalty`
    # - `learning_succeeded`
    # - `rlhf_note`
    #
    # Main tensors:
    # - `prompt_ids`: shape `[num_prompts]`
    # - `answer_ids`: shape `[num_prompts]`
    # - `rewards`: shape `[num_prompts]`
    # - `new_log_probs`: shape `[num_prompts]`
    # - `reference_log_probs`: shape `[num_prompts]`
    #
    # Steps:
    # 1. Build current policy and frozen reference policy.
    # 2. Copy current policy weights into reference policy.
    # 3. Freeze reference policy.
    # 4. Build reward table.
    # 5. Build `prompt_ids = torch.arange(config.num_prompts, device=device)`.
    # 6. Measure `initial_mean_reward = evaluate_policy_reward(policy, reward_table)`.
    # 7. Build an optimizer:
    #    `torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)`.
    # 8. For each update step:
    #    - sample answers and current log-probs with `sample_answers(policy, prompt_ids)`
    #    - look up rewards with `reward_table[prompt_ids, answer_ids]`
    #    - score the same answers under the reference policy inside `torch.no_grad()`
    #    - compute loss with `rlhf_policy_loss(...)`
    #    - call `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`
    #    - track the mean drift value `new_log_probs - reference_log_probs`
    # 9. Measure `final_mean_reward = evaluate_policy_reward(policy, reward_table)`.
    # 10. Set `learning_succeeded = final_mean_reward > initial_mean_reward`.
    # 11. Return the summary dictionary.
    #
    # Important:
    # - The reference policy is only used for scoring.
    # - The reference policy should not receive gradients.
    # - Only the current policy should be passed to the optimizer.
    #
    # Method:
    # Use helpers from this file:
    # - `TinyPolicy(...)`
    # - `build_reward_table(...)`
    # - `sample_answers(...)`
    # - `log_prob_of_answers(...)`
    # - `evaluate_policy_reward(...)`
    # - `rlhf_policy_loss(...)`
    #
    # What to print or inspect:
    # - initial mean reward
    # - final mean reward
    # - mean drift penalty
    # - whether reward improved
    #
    # Checks:
    # - `learning_succeeded` should often be True with the default config.
    # - If it is False, that does not always mean the code is wrong.
    #   Try more update steps or another seed before changing the logic.
    # - Do not use training loss alone as proof that RL worked.
    raise NotImplementedError("Run tiny RLHF-style loop.")


def check_metrics(summary: TinyRLHFSummary) -> bool:
    """Return True when expected RLHF fields are present."""
    required = {
        "initial_mean_reward",
        "final_mean_reward",
        "mean_kl_penalty",
        "learning_succeeded",
        "rlhf_note",
    }
    return required.issubset(summary) and summary["final_mean_reward"] >= summary["initial_mean_reward"]


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    print(f"[RUNNING] {task_name}")
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    except Exception as error:
        print(f"[ERROR] {task_name}: {type(error).__name__}: {error}")
        raise
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the tiny RLHF loop scaffold."""
    config = TinyRLHFLoopConfig()
    set_seed(config.random_state)

    completed_steps = 0
    total_steps = 8

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    reward_table, ok = report_task_progress("build_reward_table", build_reward_table, config)
    if not count_or_stop(ok):
        return

    policy, ok = report_task_progress("TinyPolicy", TinyPolicy, config)
    if not count_or_stop(ok):
        return

    prompt_ids = torch.arange(config.num_prompts, device=device)
    _, ok = report_task_progress("policy.forward", policy, prompt_ids)
    if not count_or_stop(ok):
        return

    sample_result, ok = report_task_progress("sample_answers", sample_answers, policy, prompt_ids)
    if not count_or_stop(ok):
        return
    answer_ids, new_log_probs = sample_result

    _, ok = report_task_progress("log_prob_of_answers", log_prob_of_answers, policy, prompt_ids, answer_ids)
    if not count_or_stop(ok):
        return

    rewards = reward_table[prompt_ids, answer_ids]
    _, ok = report_task_progress(
        "rlhf_policy_loss",
        rlhf_policy_loss,
        rewards,
        new_log_probs,
        new_log_probs.detach(),
        config.kl_coef,
    )
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("evaluate_policy_reward", evaluate_policy_reward, policy, reward_table)
    if not count_or_stop(ok):
        return

    summary, ok = report_task_progress("run_tiny_rlhf_loop", run_tiny_rlhf_loop, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and summary is not None:
        passed = check_metrics(summary)
        print_pass_or_try_again(passed, "rlhf_loop_fields_and_reward", float(int(passed)), "all fields and reward improved")


if __name__ == "__main__":
    main()
