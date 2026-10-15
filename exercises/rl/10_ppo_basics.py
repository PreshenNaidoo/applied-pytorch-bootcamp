"""
============================================================
EXERCISE 10: PPO BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny PPO-style learner.

Plain-language meanings:
- PPO means proximal policy optimization
- rollout means collected experience from the current policy
- ratio means how much the new policy changed compared with the old one
- clipping means limiting that change so training stays more stable
- entropy bonus encourages the policy to stay somewhat exploratory
- old log-probability means what the policy believed about a sampled action before the update

WHY THIS EXERCISE EXISTS
PPO is one of the most common policy-gradient methods in practice.
A beginner does not need the full production version first.
They need the core idea:
make policy updates, but do not let each update jump too far.
This exercise teaches:
- rollout collection
- old log-probabilities
- advantages
- clipped policy objective
- value loss
- entropy bonus
- multi-epoch updates over one rollout batch

WHAT THE AGENT KNOWS
The agent only sees the current LineWorld position as a one-hot vector.
For example, position `0` in a 5-position world is:

```text
[1, 0, 0, 0, 0]
```

The agent does not get a map or a shortest-path answer.
It must learn from actions, rewards, and done flags.

WHAT IS FIXED OR HIDDEN
- The environment rules are fixed: action `0` moves left, action `1` moves right.
- The reward rule is fixed: reaching the goal gives `+1`, and other steps give a small negative reward.
- The agent should not use those rules directly to choose actions.
- The policy only uses the state vector and its learned weights.

WHAT CHANGES DURING TRAINING
- The policy head changes how likely each action is.
- The value head changes its guess for how good each state is.
- The rollout batch changes after each fresh collection.
- The old log-probabilities inside one rollout batch do not change.

WHICH OBJECT DOES WHAT
- `TinyPPOEnv` is the world. It gives the next state, reward, and done flag.
- `PPOPolicyValueNetwork` is the learner. It chooses actions and predicts state values.
- `Categorical(logits=...)` turns action scores into action probabilities and samples one action.
- `old_log_probs` store what the policy believed before the PPO update.
- `returns` are the reward targets.
- `advantages` tell the policy whether the sampled actions were better or worse than expected.

HOW THIS FITS INTO THE RL SEQUENCE
This file comes after actor-critic.
The next idea is not a new RL branch.
It is a safer update rule for policy-gradient training.
In simple words:
- actor-critic gives you policy and value learning
- PPO adds a guardrail so policy updates do not jump too far at once

Beginner intuition:
- PPO is like telling the policy:
  - improve, but do not change too much all at once
- clipping is a safety rail
- it helps prevent one bad update from pushing the policy too far

Small example:
- if the new policy suddenly makes an action much more likely
  than the old policy did,
  PPO clips the update instead of trusting that large jump fully

Tiny numeric example:

```text
old probability of action right = 0.50
new probability of action right = 0.90
ratio = 0.90 / 0.50 = 1.80
clip_epsilon = 0.20
clipped ratio = 1.20
```

Plain meaning:
the new policy is trying to make a very large jump.
PPO limits that jump in the loss, so the update is less likely to break the policy.

IMPORTANT SCOPE NOTE
This is a teaching version of PPO.
Keep the environment tiny and local.
The goal is to understand the clipped objective and rollout shape.

DATASET
No dataset file is needed.
Collect rollouts from a tiny local LineWorld environment.

Clear default:
- environment name: `LineWorld`
- states: 5 positions in a line, numbered `0` to `4`
- start state: position `0`
- goal state: position `4`
- actions: `0 = left`, `1 = right`
- reward: `+1` at the goal, small negative reward such as `-0.01` otherwise
- rollout batch: a short fixed number of steps collected from the current policy

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- collect one rollout batch from the current policy
- save states, actions, rewards, dones, old log-probabilities, and values
- reuse that rollout batch for a few PPO update epochs
- then collect fresh rollout data again

INPUTS AND OUTPUTS
Input:
- rollout states
- actions
- old log-probabilities
- rewards
- dones
- predicted values
Output:
- mean return
- final policy loss
- final value loss
- number of PPO updates
- short clipping note

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a tiny policy-value network:
- shared trunk
- policy head
- value head

LOSS FUNCTION
Use a small PPO training objective with:
- clipped policy loss
- value loss
- entropy bonus

METRICS TO TRACK
Track at least:
- mean return
- last policy loss
- last value loss
- number of updates

TARGET RESULT RANGE
Focus on understanding old vs new policy comparison, not on benchmark reward.

Useful signs:
- old log-probabilities are saved before the update
- ratios can be computed without shape errors
- clipping changes the objective when the new policy moves too far
- value loss and entropy bonus are included in the final objective

COMMON MISTAKES
- recomputing old log-probabilities after the policy already changed
- clipping the wrong quantity
- forgetting to include the critic loss
- forgetting the entropy term when the exercise expects it
- treating PPO as magic instead of "actor-critic with a clipped update"

What each step is doing in simple words:
1. collect a batch of recent experience from the current policy
2. save what the old policy thought about those sampled actions
3. compute returns and advantages from the rollout
4. compare old and new policy probabilities on the same sampled actions
5. clip overly large changes
6. update the value head as well
7. keep some exploration pressure with entropy
8. repeat over several update rounds and explain why the guardrail helps

Main loop in exact order:

```text
reset the environment
look at the current state
use the policy to sample an action
save the state, action, reward, done flag, old log-probability, and value guess
repeat until the rollout batch is full
compute returns and advantages
run the current policy on the saved states again
compare new log-probabilities with old log-probabilities
clip very large policy changes
add value loss and entropy bonus
update the network
collect fresh data and repeat
```

STEP-BY-STEP TODO
1. Build a tiny environment.
2. Build a policy-value network.
3. Collect one rollout batch.
4. Save old log-probabilities.
5. Compute returns and advantages.
6. Compute the PPO clipped policy loss.
7. Add value loss.
8. Add entropy bonus.
9. Train for several update rounds.
10. Explain why clipping helps.

DEFINITION OF DONE
You are done when:
- rollouts are collected
- old log-probabilities are stored
- clipped policy loss is computed
- value loss and entropy bonus are included
- you can explain PPO clipping in plain words
- the returned summary has the expected keys
- the final result reports a mean return, losses, update count, and clipping note
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TypedDict
import torch
import torch.nn as nn

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

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class PPOConfig:
    """Settings for the PPO exercise."""
    state_dim: int = 5
    num_actions: int = 2
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    rollout_steps: int = 64
    updates: int = 40
    ppo_epochs_per_rollout: int = 3
    random_state: int = 42


class PPOPlanSummary(TypedDict):
    """Main PPO pieces used in this exercise."""

    clip_epsilon: float
    main_parts: list[str]


class PPORolloutBatch(TypedDict):
    """One PPO rollout batch collected before policy updates."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    old_log_probs: torch.Tensor
    predicted_values: torch.Tensor


class PPOSummary(TypedDict):
    """Final summary returned by the tiny PPO workflow."""

    mean_return: float
    last_policy_loss: float
    last_value_loss: float
    num_updates: int
    note: str


def describe_ppo_plan(config: PPOConfig) -> PPOPlanSummary:
    """Return the main PPO pieces."""
    return {
        "clip_epsilon": config.clip_epsilon,
        "main_parts": ["rollout_batch", "old_log_probs", "clipped_policy_loss", "value_loss", "entropy_bonus"],
    }


# Beginner hints for this file:
# - PPO still uses policy gradients, but with an extra guardrail
# - the old log-probabilities are needed so you can compare old and new policies
# - the policy ratio says how much the policy changed on the sampled actions
# - clipping limits very large changes
# - entropy bonus helps stop the policy from becoming too certain too early


class TinyPPOEnv:
    """Tiny local environment used for PPO."""
    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.position = 0

    def reset(self) -> torch.Tensor:
        """Reset and return the starting state."""
        # Goal:
        # Reset LineWorld to the first position and return its one-hot state vector.
        #
        # Input:
        # - No direct arguments.
        # - This method uses `self.config.state_dim` and `self.position`.
        #
        # Output:
        # - `torch.Tensor`
        # - Shape: `[state_dim]`
        # - Example for 5 positions at position 0:
        #   `tensor([1., 0., 0., 0., 0.])`
        #
        # Method:
        # Use:
        # - `self.position = 0`
        # - `state = torch.zeros(self.config.state_dim, dtype=torch.float32)`
        # - `state[self.position] = 1.0`
        #
        # Steps:
        # 1. Set the current position to `0`.
        # 2. Create a zero vector with length `state_dim`.
        # 3. Put `1.0` at the current position.
        # 4. Return the state vector.
        #
        # Checks:
        # - `state.shape` should be `[self.config.state_dim]`.
        # - `state.sum()` should be `1.0`.
        # - `self.position` should be `0` after reset.

        self.position = 0
        state = torch.zeros(self.config.state_dim, dtype=torch.float32)
        state[self.position] = 1.0

        return state
        #raise NotImplementedError("Return the starting state vector.")


    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        """Run one environment step."""
        # Goal:
        # Move one step in LineWorld and return the next state, reward, and done flag.
        #
        # Input:
        # - `action`: int
        # - `0` means move left.
        # - `1` means move right.
        #
        # Output:
        # - `next_state`: torch.Tensor with shape `[state_dim]`
        # - `reward`: float
        # - `done`: bool
        #
        # Example:
        # If the current position is `0` and action is `1`, the next position should be `1`.
        #
        # Method:
        # Use `max(...)` and `min(...)` to keep the position inside the line:
        # - left: `self.position = max(0, self.position - 1)`
        # - right: `self.position = min(self.config.state_dim - 1, self.position + 1)`
        #
        # The goal position is `self.config.state_dim - 1`.
        # Use reward `1.0` at the goal and `-0.01` otherwise.
        #
        # Steps:
        # 1. Update `self.position` based on the action.
        # 2. Set `done = self.position == self.config.state_dim - 1`.
        # 3. Set reward to `1.0` if done, otherwise `-0.01`.
        # 4. Build the next one-hot state vector.
        # 5. Return `(next_state, reward, done)`.
        #
        # Checks:
        # - The position should never go below `0`.
        # - The position should never go above `state_dim - 1`.
        # - `done` should be `True` only at the goal.

        goal_position = self.config.state_dim - 1  # 5 - 1 = 4
        if action == 0:
            self.position = max(self.position - 1, 0)
        elif action == 1:
            self.position = min(self.position + 1, goal_position)
        
        done = self.position == goal_position
        reward = 1.0 if done else -0.01

        next_state_vec = torch.zeros(self.config.state_dim, dtype=torch.float32)
        next_state_vec[self.position] = 1.0

        return next_state_vec, reward, done
        #raise NotImplementedError("Return next state, reward, and done flag.")

class PPOPolicyValueNetwork(nn.Module):
    """Tiny network with policy and value outputs."""
    def __init__(self, config: PPOConfig) -> None:
        # Goal:
        # Build one small shared network with two heads:
        # - the policy head outputs action logits
        # - the value head outputs one state value
        #
        # Input:
        # - `config`: PPOConfig
        #
        # Output:
        # - `None`
        # - Store the layers on `self`.
        #
        # Method:
        # Use this exact layer shape:
        # ```python
        # self.trunk = nn.Sequential(
        #     nn.Linear(config.state_dim, config.hidden_dim),
        #     nn.ReLU(),
        # )
        # self.policy_head = nn.Linear(config.hidden_dim, config.num_actions)
        # self.value_head = nn.Linear(config.hidden_dim, 1)
        # ```
        #
        # Meaning:
        # - `trunk` reads the state and makes shared features.
        # - `policy_head` turns those features into one score per action.
        # - `value_head` turns those features into one value estimate.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Build `self.trunk`.
        # 3. Build `self.policy_head`.
        # 4. Build `self.value_head`.
        # 5. Do not apply softmax in this method.
        # 6. Do not return anything.
        #
        # Checks:
        # - For input shape `[batch_size, state_dim]`, policy logits should be `[batch_size, num_actions]`.
        # - For input shape `[batch_size, state_dim]`, values should be `[batch_size]`.
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(in_features = config.state_dim, out_features=config.hidden_dim),
                                    nn.ReLU(inplace = True))

        self.policy_head = nn.Linear(config.hidden_dim, config.num_actions)
        self.value_head = nn.Linear(config.hidden_dim, 1)
        #raise NotImplementedError("Build a tiny shared network with policy and value heads.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and state values."""
        # Goal:
        # Return both PPO model outputs for a batch of states.
        #
        # Input:
        # - `x`: torch.Tensor
        # - Shape: `[batch_size, state_dim]`
        # - Example: a rollout batch of 64 states has shape `[64, 5]`.
        #
        # Output:
        # Return a tuple:
        # - `logits`: torch.Tensor with shape `[batch_size, num_actions]`
        # - `values`: torch.Tensor with shape `[batch_size]`
        #
        # Method:
        # Use:
        # - `features = self.trunk(x)`
        # - `logits = self.policy_head(features)`
        # - `values = self.value_head(features).squeeze(-1)`
        #
        # Steps:
        # 1. Pass `x` through the shared trunk.
        # 2. Pass the shared features through the policy head.
        # 3. Pass the shared features through the value head.
        # 4. Remove only the last size-1 dimension from values with `.squeeze(-1)`.
        # 5. Return `(logits, values)`.
        #
        # Checks:
        # - Do not apply softmax to logits in `forward`.
        # - Do not use `.squeeze()` without a dimension, because a batch of size 1 could become shape `[]`.

        features = self.trunk(x)  # shape: [batch_size, hidden_dim]
        logits = self.policy_head(features)  # shape: [batch_size, num_actions]
        values = self.value_head(features).squeeze(dim=-1) # shape [batch_size, 1] becomes [batch_size]

        return logits, values
        #raise NotImplementedError("Return policy logits and critic values.")

# Collect one PPO rollout batch.
# Return at least:
# - states
# - actions
# - rewards
# - dones
# - old_log_probs
# - predicted_values
def collect_rollout_batch(
    model: nn.Module,
    env: TinyPPOEnv,
    config: PPOConfig,
) -> PPORolloutBatch:
    """Collect one rollout batch for PPO."""
    # Where this fits:
    # This is the data collection step.
    # PPO first acts in the environment using the current policy.
    # Only after this batch is collected should the policy be updated.
    #
    # Why this helper exists:
    # PPO must compare the policy before the update with the policy after the update.
    # That is why this function saves `old_log_probs`.
    # They are called "old" because they come from the policy that collected the actions.
    #
    # Goal:
    # Collect one fixed-size rollout batch before any PPO update happens.
    # Save old log-probabilities from the policy that collected the actions.
    #
    # Input:
    # - `model`: nn.Module
    # - `env`: TinyPPOEnv
    # - `config`: PPOConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"states"`: float tensor with shape `[rollout_steps, state_dim]`
    # - `"actions"`: long tensor with shape `[rollout_steps]`
    # - `"rewards"`: float tensor with shape `[rollout_steps]`
    # - `"dones"`: float tensor with shape `[rollout_steps]`, where done is `1.0` and not done is `0.0`
    # - `"old_log_probs"`: float tensor with shape `[rollout_steps]`
    # - `"predicted_values"`: float tensor with shape `[rollout_steps]`
    #
    # Tiny example of one saved step:
    # ```text
    # state = [1, 0, 0, 0, 0]
    # logits = [0.2, 0.8]
    # sampled action = 1
    # old_log_prob = log(probability of action 1 under these logits)
    # reward after env.step(1) = -0.01
    # done = 0.0
    # predicted_value = value head output for this state
    # ```
    #
    # Method:
    # Use PyTorch's categorical distribution:
    # - `logits, values = model(state.unsqueeze(0))`
    # - `dist = torch.distributions.Categorical(logits=logits.squeeze(0))`
    # - `action_tensor = dist.sample()`
    # - `old_log_prob = dist.log_prob(action_tensor)`
    # - `action = int(action_tensor.item())`
    #
    # Store old log-probabilities with `.detach()`.
    # PPO needs the old policy numbers to stay fixed while the new policy updates.
    #
    # Steps:
    # 1. Put the model in training mode with `model.train()`.
    # 2. Reset the environment with `state = env.reset()`.
    # 3. Create empty Python lists for states, actions, rewards, dones, old_log_probs, and predicted_values.
    # 4. Loop exactly `config.rollout_steps` times.
    # 5. Run the model on `state.unsqueeze(0)`.
    # 6. Build a `Categorical` distribution from the logits.
    # 7. Sample one action.
    # 8. Save the current state, sampled action, reward, done flag, old log-probability, and value.
    # 9. If `done` is true, reset the environment; otherwise keep the next state.
    # 10. Stack or convert every list into tensors.
    # 11. Return the rollout dictionary.
    #
    # Checks:
    # - All returned tensors should have first dimension `config.rollout_steps`.
    # - `actions` must use dtype `torch.long`.
    # - `old_log_probs` should not require gradients.
    # - `predicted_values` should not require gradients in the stored rollout batch.

    model.train()
    model_device = next(model.parameters()).device

    state = env.reset()

    states, actions, rewards, dones, old_log_probs, predicted_values = [], [], [], [], [], []

    for i in range(config.rollout_steps):

        model_state = state.to(model_device).unsqueeze(dim=0)  # shape: [1, state_dim]
        logits, values = model(model_state)

        # `logits` has shape `[1, num_actions]` because the policy saw one state.
        # The first `1` is the batch size.
        # `logits.squeeze(0)` removes that batch wrapper, so the scores become shape `[num_actions]`.
        #
        # `Categorical(logits=...)` is the PyTorch distribution for choosing one item
        # from a short list of choices. Here the choices are actions:
        # - action 0 = left
        # - action 1 = right
        #
        # This is very similar to:
        # ```python
        # probs = torch.softmax(logits.squeeze(0), dim=-1)
        # action_tensor = torch.multinomial(probs, num_samples=1)
        # ```
        #
        # Both versions mean:
        # turn raw action scores into probabilities, then randomly pick one action
        # using those probabilities.
        #
        # Small example:
        # ```python
        # action_scores = torch.tensor([2.0, 0.5])
        # probs = torch.softmax(action_scores, dim=-1)
        # # probs is about tensor([0.8176, 0.1824])
        # ```
        #
        # So action 0 is picked about 82% of the time,
        # and action 1 is picked about 18% of the time.
        #
        # `Categorical` is nicer here because PPO needs `log_prob`.
        # `log_prob` means "the log of the probability of the action we actually picked".
        #
        # Example:
        # If the policy probabilities are:
        # - action 0: 0.82
        # - action 1: 0.18
        #
        # and the sampled action is action 0, then:
        # ```python
        # log_prob = torch.log(torch.tensor(0.82))
        # about -0.20
        # ```
        #
        # If the sampled action is action 1, then:
        # ```python
        # log_prob = torch.log(torch.tensor(0.18))
        # about -1.71
        # ```
        #
        # A more likely action has a log_prob closer to 0.
        # A less likely action has a more negative log_prob.

        dist = torch.distributions.Categorical(logits = logits.squeeze(dim=0))
        action_tensor = dist.sample()
        old_log_prob = dist.log_prob(action_tensor)
        action = int(action_tensor.item())

        next_state, reward, done = env.step(action)

        #actions, rewards, dones, old_log_probs, predicted_values
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        old_log_probs.append(float(old_log_prob.detach().cpu().item()))
        predicted_values.append(float(values.squeeze().detach().cpu().item()))

        if done:
            state = env.reset()            
        else:
            state = next_state
    
    states = torch.stack(states)  # shape: [rollout_steps, state_dim]
    actions = torch.tensor(actions, dtype=torch.long)  # shape: [rollout_steps]
    rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [rollout_steps]
    dones = torch.tensor(dones, dtype=torch.float32)  # shape: [rollout_steps]
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)  # shape: [rollout_steps]
    predicted_values = torch.tensor(predicted_values)  # shape: [rollout_steps]

    return {"states":states, "actions":actions, "rewards":rewards, "dones":dones, "old_log_probs": old_log_probs, "predicted_values": predicted_values}
    #raise NotImplementedError("Return one PPO rollout batch with old policy information.")

def compute_returns_and_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rollout returns and advantages."""
    # Where this fits:
    # This helper turns raw rewards into training targets.
    # The policy loss needs advantages.
    # The value loss needs returns.
    #
    # PPO does not only ask, "Did this action get reward?"
    # It asks, "Was this action better or worse than the value head expected?"
    #
    # Goal:
    # Return discounted returns and advantages for one rollout batch.
    #
    # Input:
    # - `rewards`: torch.Tensor
    #   Shape: `[rollout_steps]`
    # - `dones`: torch.Tensor
    #   Shape: `[rollout_steps]`, where done is `1.0` and not done is `0.0`
    # - `values`: torch.Tensor
    #   Shape: `[rollout_steps]`
    # - `gamma`: float
    #
    # Output:
    # Return a tuple:
    # - `returns`: torch.Tensor with shape `[rollout_steps]`
    # - `advantages`: torch.Tensor with shape `[rollout_steps]`
    #
    # Formula:
    # ```text
    # running_return = reward_t + gamma * running_return * (1 - done_t)
    # advantage_t = return_t - value_t
    # ```
    #
    # Meaning:
    # - `(1 - done_t)` removes future reward after an episode has ended.
    # - `advantage_t` says better or worse than the critic expected.
    #
    # Simple meanings:
    # - `return` means what actually happened after this step.
    # - `value` means the critic's prediction for the state.
    # - `done` means the episode ended, so no later rewards should be added.
    # - `advantage` means how much better or worse the real return was than the critic predicted.
    #
    # In one sentence:
    # advantage = what happened - what the critic expected
    #
    # The actor and critic answer different questions.
    #
    # The actor picks actions.
    # For one state, the actor head outputs action scores:
    # ```text
    # state = position 0
    # actor logits = [left_score, right_score]
    # ```
    #
    # Those logits become action probabilities with:
    # ```python
    # probs = torch.softmax(logits, dim=-1)
    # ```
    #
    # For example:
    # ```text
    # left probability  = 0.30
    # right probability = 0.70
    # ```
    #
    # These probabilities come from the actor, not the critic.
    #
    # The critic does not see the chosen action.
    # The critic only sees the state and outputs one number:
    # ```text
    # value(position 0)
    # ```
    #
    # That number means:
    # "When the agent is in this state, how much reward does it usually get later,
    # using the actor as it currently behaves?"
    #
    # Why "usually"?
    # From the same state, the actor may sometimes pick right and sometimes pick left.
    #
    # Example:
    # ```text
    # position 0
    # actor sometimes picks right -> good return, maybe 0.79
    # actor sometimes picks left  -> bad return, maybe -0.03
    # ```
    #
    # The critic only has one value for `position 0`.
    # So it learns the usual result for that state under the current actor.
    #
    # Short version:
    # ```text
    # actor = picks actions
    # critic = predicts the usual future reward from this state
    # return = what actually happened this time
    # advantage = actual result - critic's usual prediction
    # ```
    #
    # If the critic predicted:
    # ```text
    # value(position 0) = 0.50
    # ```
    #
    # and this rollout picked right and got:
    # ```text
    # return = 0.79
    # ```
    #
    # then:
    # ```text
    # advantage = 0.79 - 0.50 = +0.29
    # ```
    #
    # Plain meaning:
    # this sampled action did better than usual for this state,
    # so PPO should make that action more likely.
    #
    # Small return example with `gamma = 0.9`:
    # ```text
    # positions = [0, 1, 2]
    # actions   = [right, right, stop]
    # rewards   = [-0.01, -0.01, 1.00]
    # dones     = [ 0.00,  0.00, 1.00]
    # values    = [ 0.20,  0.50, 0.70]
    # ```
    #
    # The agent starts at position 0.
    # It moves right, moves right again, then reaches the goal.
    #
    # `returns` are calculated backward because each earlier step needs to know
    # what reward came later.
    #
    # Step 2 is the goal step:
    # ```text
    # return_2 = 1.00 + 0.9 * 0.00 * (1 - 1.00)
    #          = 1.00
    # ```
    #
    # This means:
    # at step 2, the agent reached the goal and got `1.00`.
    # `done = 1.00`, so no later reward is added.
    #
    # Step 1 is one step before the goal:
    # ```text
    # return_1 = -0.01 + 0.9 * 1.00 * (1 - 0.00)
    #          = 0.89
    # ```
    #
    # This means:
    # at step 1, the agent got a small cost now, but it reached the goal next.
    # So this step gets credit for leading to the goal.
    #
    # Step 0 is two steps before the goal:
    # ```text
    # return_0 = -0.01 + 0.9 * 0.89 * (1 - 0.00)
    #          = 0.791
    # ```
    #
    # This means:
    # if the agent is at position 0 and follows this good path,
    # the critic should learn to predict about `0.791` for that state.
    #
    # So:
    # ```text
    # returns = [0.791, 0.89, 1.00]
    # ```
    #
    # These are the targets for the critic:
    # - for the state representing position 0, learn to predict about `0.791`
    # - for the state representing position 1, learn to predict about `0.89`
    # - at the goal step, learn to predict about `1.00`
    #
    # ```text
    # advantages = returns - values
    #            = [0.791 - 0.20, 0.89 - 0.50, 1.00 - 0.70]
    #            = [0.591, 0.39, 0.30]
    # ```
    #
    # Plain meaning:
    # The critic predicted `[0.20, 0.50, 0.70]`.
    # What actually happened was better: `[0.791, 0.89, 1.00]`.
    # So all three advantages are positive.
    # PPO should make the sampled actions from these steps more likely.
    #
    # Method:
    # Work backward through the rollout:
    # - start with `running_return = 0.0`
    # - loop from the last step down to the first step
    # - insert each return at the front, or fill a preallocated tensor
    # - compute `advantages = returns - values`
    #
    # Use `.detach()` on `advantages` before returning if you plan to use them only for the policy loss.
    #
    # Steps:
    # 1. Create `returns = torch.zeros_like(rewards)`.
    # 2. Set `running_return = 0.0`.
    # 3. Loop backward over indexes from `len(rewards) - 1` to `0`.
    # 4. Apply the formula above.
    # 5. Store each `running_return` in `returns[index]`.
    # 6. Compute `advantages = returns - values`.
    # 7. Return `(returns, advantages)`.
    #
    # Checks:
    # - If `done_t` is `1.0`, the return at that step should not include later rewards.
    # - `returns.shape` and `advantages.shape` should both match `rewards.shape`.
    # - With `values` all zeros, advantages should equal returns.


    num = rewards.shape[0]

    returns = torch.zeros_like(rewards)

    running_return = 0.0

    for i in range(num-1, -1, -1):
        running_return = rewards[i] + gamma * running_return * (1 - dones[i])
        returns[i] = running_return

    advantages = returns - values

    return returns, advantages
    #raise NotImplementedError("Return rollout returns and advantages.")

# Compute the PPO clipped policy loss.
# In plain words:
# - compare new policy probability with old policy probability
# - clip very large changes
def ppo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """Return the PPO clipped policy loss."""
    # Where this fits:
    # This is the main PPO idea.
    # It compares the new policy with the old policy on the exact same sampled actions.
    #
    # Do not use rewards directly here.
    # The policy update uses `advantages`, because advantages say whether the sampled actions were better than expected.
    #
    # Goal:
    # Return one scalar PPO policy loss.
    # This loss limits how much the new policy can move away from the old policy.
    #
    # Input:
    # - `new_log_probs`: torch.Tensor
    #   Shape: `[rollout_steps]`
    # - `old_log_probs`: torch.Tensor
    #   Shape: `[rollout_steps]`
    # - `advantages`: torch.Tensor
    #   Shape: `[rollout_steps]`
    # - `clip_epsilon`: float
    #   Example: `0.2`
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[]`, meaning one scalar loss.
    #
    # Formula:
    # ```text
    # ratio = exp(new_log_prob - old_log_prob)
    # unclipped = ratio * advantage
    # clipped_ratio = clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    # clipped = clipped_ratio * advantage
    # policy_loss = -mean(min(unclipped, clipped))
    # ```
    #
    # Meaning:
    # - `ratio` is how much the new policy probability changed for the same action.
    # - `clamp` limits the ratio to a small range, such as `0.8` to `1.2`.
    # - the minus sign is needed because PyTorch optimizers minimize loss.
    #
    # Why `ratio = torch.exp(new_log_probs - old_log_probs)` works:
    #
    # PPO needs this normal probability ratio:
    # ```text
    # ratio = new_probability_of_sampled_action / old_probability_of_sampled_action
    # ```
    #
    # Practical example:
    # The agent was at position 0.
    # During rollout collection, the old actor sampled action `right`.
    #
    # At collection time:
    # ```text
    # old probability of right = 0.50
    # old_log_prob = log(0.50)
    # ```
    #
    # During the PPO update, the model looks at the same saved state and action again.
    # Now the updated actor might say:
    # ```text
    # new probability of right = 0.75
    # new_log_prob = log(0.75)
    # ```
    #
    # PPO wants to know how much more likely the same sampled action became:
    # ```text
    # normal ratio = 0.75 / 0.50 = 1.50
    # ```
    #
    # But the code stores log-probabilities because policy-gradient code usually
    # works with log-probabilities.
    #
    # Log rule:
    # ```text
    # log(a / b) = log(a) - log(b)
    # ```
    #
    # So:
    # ```text
    # new_log_prob - old_log_prob
    # = log(0.75) - log(0.50)
    # = log(0.75 / 0.50)
    # = log(1.50)
    # ```
    #
    # That is still in log form.
    # `torch.exp(...)` changes it back into the normal ratio:
    # ```text
    # exp(log(1.50)) = 1.50
    # ```
    #
    # Plain meaning:
    # - ratio `1.00` means the new policy gives the sampled action the same probability as before.
    # - ratio `1.50` means the new policy made that sampled action 50% more likely.
    # - ratio `0.50` means the new policy made that sampled action half as likely.
    #
    # Method:
    # Use:
    # - `ratio = torch.exp(new_log_probs - old_log_probs)`
    # - `clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)`
    # - `unclipped = ratio * advantages.detach()`
    # - `clipped = clipped_ratio * advantages.detach()`
    # - `loss = -torch.min(unclipped, clipped).mean()`
    #
    # Steps:
    # 1. Compute the policy ratio from new and old log-probabilities.
    # 2. Clip the ratio to `[1 - clip_epsilon, 1 + clip_epsilon]`.
    # 3. Build the unclipped objective.
    # 4. Build the clipped objective.
    # 5. Take the smaller objective at each step.
    # 6. Negate the mean and return it.
    #
    # Checks:
    # - If `new_log_probs == old_log_probs`, the ratio should be `1.0`.
    # - With `clip_epsilon=0.2`, ratios should be clipped to the range `[0.8, 1.2]`.
    # - The returned loss should be a scalar tensor.

    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped = ratio * advantages.detach()
    clipped = clipped_ratio * advantages.detach()
    loss = -torch.min(unclipped, clipped).mean()

    return loss
    # Why `advantages.detach()` is used above:
    #
    # `detach` means:
    # use this tensor as a number, but stop gradients through it.
    #
    # Example:
    # ```python
    # advantage = returns - values
    # ```
    #
    # If `values` came from the critic, then `advantage` is connected to the critic.
    #
    # Without detach, this line:
    # ```python
    # policy_loss = ratio * advantage
    # ```
    #
    # could send gradients into:
    # ```text
    # actor through ratio
    # critic through advantage
    # ```
    #
    # But that is not what we want.
    # We want:
    # ```text
    # policy loss trains actor
    # value loss trains critic
    # ```
    #
    # So we use:
    # ```python
    # policy_loss = ratio * advantage.detach()
    # ```
    #
    # Meaning:
    # ```text
    # actor, use this advantage number as a fixed score
    # critic, do not learn from this policy loss
    # ```
    #raise NotImplementedError("Return the PPO clipped policy loss.")

def ppo_value_loss(predicted_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Return the PPO critic loss."""
    # Where this fits:
    # This trains the value head.
    # The value head should predict the return from each state.
    # Better value guesses make future advantages less noisy.
    #
    # Goal:
    # Return one scalar loss that trains the value head.
    #
    # Input:
    # - `predicted_values`: torch.Tensor
    #   Shape: `[rollout_steps]`
    # - `returns`: torch.Tensor
    #   Shape: `[rollout_steps]`
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[]`, meaning one scalar loss.
    #
    # Formula:
    # ```text
    # value_loss = mean((predicted_value - return) ** 2)
    # ```
    #
    # Method:
    # Use:
    # `nn.functional.mse_loss(predicted_values, returns)`
    #
    # Steps:
    # 1. Make sure both tensors have shape `[rollout_steps]`.
    # 2. Compute mean squared error.
    # 3. Return the scalar value loss.
    #
    # Checks:
    # - Do not detach `predicted_values`, because this loss should train the value head.
    # - Detaching `returns` is fine because returns are targets, not model outputs.

    loss = nn.functional.mse_loss(predicted_values, returns)
    return loss
    #raise NotImplementedError("Return the value loss for PPO.")

def entropy_bonus(action_logits: torch.Tensor) -> torch.Tensor:
    """Return the policy entropy bonus."""
    # Where this fits:
    # This keeps the policy from becoming too certain too early.
    # In this tiny world, it gives the learner more chances to try both left and right.
    #
    # Goal:
    # Return the mean entropy of the policy distribution.
    # Higher entropy means the policy is less certain and explores more.
    #
    # Input:
    # - `action_logits`: torch.Tensor
    #   Shape: `[batch_size, num_actions]`
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[]`, meaning one scalar entropy value.
    #
    # Method:
    # Use PyTorch's categorical distribution:
    # - `dist = torch.distributions.Categorical(logits=action_logits)`
    # - `entropy = dist.entropy().mean()`
    #
    # Same idea written out with softmax.
    #
    # Start from the entropy formula:
    # ```text
    # entropy = -sum(p * log(p))
    # ```
    #
    # In this formula:
    # - `p` is the probability of one action.
    # - `log(p)` is the log-probability of that action.
    # - `sum(...)` adds this over all actions.
    #
    # Code version:
    # ```python
    # probs = torch.softmax(action_logits, dim=-1)
    # log_probs = torch.log(probs + 1e-8)
    # entropy_per_state = -(probs * log_probs).sum(dim=-1)
    # entropy = entropy_per_state.mean()
    # ```
    #
    # The direct formula-to-code mapping is:
    # ```text
    # p        -> probs
    # log(p)   -> log_probs
    # p*log(p) -> probs * log_probs
    # -sum(...) over actions -> -(probs * log_probs).sum(dim=-1)
    # ```
    #
    # What this means:
    # - `probs` turns action scores into action probabilities.
    # - `log_probs` gives the log of each action probability.
    # - `probs * log_probs` is the `p * log(p)` part of the entropy formula.
    # - the minus sign makes entropy positive.
    # - `.sum(dim=-1)` adds over actions for each state.
    # - `.mean()` averages over the batch.
    #
    # `Categorical(...).entropy()` does the same calculation for you.
    #
    # Steps:
    # 1. Build the categorical distribution from logits.
    # 2. Compute entropy for each row.
    # 3. Average the entropy values.
    # 4. Return the scalar entropy bonus.
    #
    # Checks:
    # - For two equally likely actions, entropy should be near `0.693`.
    # - Entropy should be lower when one action is much more likely than the other.

    dist = torch.distributions.Categorical(logits=action_logits)
    entropy = dist.entropy().mean()

    return entropy
    #raise NotImplementedError("Return the entropy bonus from the policy logits.")

def train_ppo(config: PPOConfig) -> PPOSummary:
    """Run the full tiny PPO workflow."""
    # Where this fits:
    # This connects all earlier helpers into one full training loop.
    # If a shape error happens here, check the smaller helpers first.
    #
    # What learns in this function:
    # - the policy head learns which action to pick
    # - the value head learns how good each state is
    #
    # What only measures in this function:
    # - `mean_return` reports how much reward the collected rollouts got
    # - `note` explains the clipping idea
    #
    # Goal:
    # Collect rollout batches, run PPO update epochs, and return a summary.
    #
    # Input:
    # - `config`: PPOConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"mean_return"`: float, average return from collected rollout batches
    # - `"last_policy_loss"`: float, policy loss from the final PPO update
    # - `"last_value_loss"`: float, value loss from the final PPO update
    # - `"num_updates"`: int, number of optimizer steps run
    # - `"note"`: str, short explanation from `explain_clipping(...)`
    #
    # Method:
    # Use these objects:
    # - `env = TinyPPOEnv(config)`
    # - `model = PPOPolicyValueNetwork(config).to(device)`
    # - `optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)`
    #
    # Use these helper functions:
    # - `batch = collect_rollout_batch(model, env, config)`
    # - `returns, advantages = compute_returns_and_advantages(...)`
    # - `policy_loss = ppo_policy_loss(new_log_probs, batch["old_log_probs"], advantages, config.clip_epsilon)`
    # - `value_loss = ppo_value_loss(new_values, returns)`
    # - `entropy = entropy_bonus(logits)`
    #
    # Combine losses with:
    # ```python
    # loss = (
    #     policy_loss
    #     + config.value_loss_weight * value_loss
    #     - config.entropy_weight * entropy
    # )
    # ```
    #
    # The entropy term is subtracted because higher entropy is good, but optimizers minimize loss.
    #
    # Steps:
    # 1. Build the environment, model, and optimizer.
    # 2. Create an empty list for rollout returns.
    # 3. Start `last_policy_loss = 0.0`, `last_value_loss = 0.0`, and `num_updates = 0`.
    # 4. Loop for `config.updates`.
    # 5. Collect one rollout batch.
    # 6. Compute returns and advantages from the rollout rewards, dones, and stored values.
    # 7. Loop for `config.ppo_epochs_per_rollout`.
    # 8. Run the current model on `batch["states"]`.
    # 9. Build `dist = torch.distributions.Categorical(logits=logits)`.
    # 10. Compute new log-probabilities with `dist.log_prob(batch["actions"])`.
    # 11. Compute policy loss, value loss, and entropy.
    # 12. Combine the losses with the formula above.
    # 13. Clear gradients, backpropagate, and step the optimizer.
    # 14. Add 1 to `num_updates`.
    # 15. Save the final policy and value loss values as Python floats.
    # 16. Save the rollout return, such as `float(batch["rewards"].sum().item())`.
    # 17. Build the summary dictionary.
    # 18. Add the note from `explain_clipping(summary)`.
    # 19. Return the summary.
    #
    # Checks:
    # - `num_updates` should equal `config.updates * config.ppo_epochs_per_rollout`.
    # - Old log-probabilities should come from collection time, not after the model update.
    # - `new_log_probs`, `old_log_probs`, `returns`, and `advantages` should all have shape `[rollout_steps]`.


    env = TinyPPOEnv(config)
    model = PPOPolicyValueNetwork(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    rollout_returns: list[float] = []
    last_policy_loss = 0.0
    last_value_loss = 0.0
    num_updates = 0

    # How to read the two PPO training loops:
    # ```text
    # 40 times:
    #     collect one rollout batch with 64 environment steps
    #     train on that same batch 3 times
    #     throw that batch away
    # ```
    #
    # Outer loop: collect a new rollout batch.
    # Inner loop: train on that same batch.
    # Total optimizer updates: 40 * 3 = 120.
    
    for _ in range(config.updates):

        batch_dict = collect_rollout_batch(model, env, config)
        batch_dict = {key: value.to(device) for key, value in batch_dict.items()}

        returns, advantages = compute_returns_and_advantages(batch_dict["rewards"], batch_dict["dones"], batch_dict["predicted_values"], config.gamma)

        for _ in range(config.ppo_epochs_per_rollout):
            logits, new_values = model(batch_dict["states"])
            
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_dict["actions"])

            policy_loss = ppo_policy_loss(new_log_probs, batch_dict["old_log_probs"], advantages, config.clip_epsilon)
            value_loss = ppo_value_loss(new_values, returns)
            entropy = entropy_bonus(logits)

            loss = (policy_loss + config.value_loss_weight * value_loss - config.entropy_weight * entropy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_updates += 1
            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())

        rollout_returns.append(float(batch_dict["rewards"].sum().item()))

    mean_return = sum(rollout_returns) / len(rollout_returns)
    summary: PPOSummary = {
        "mean_return": float(mean_return),
        "last_policy_loss": last_policy_loss,
        "last_value_loss": last_value_loss,
        "num_updates": num_updates,
        "note": "",
    }
    summary["note"] = explain_clipping(summary)

    model.eval()
    eval_env = TinyPPOEnv(config)
    eval_episodes = 5
    eval_max_steps = config.state_dim * 2
    eval_returns: list[float] = []
    eval_successes = 0

    with torch.no_grad():
        for _ in range(eval_episodes):
            state = eval_env.reset()
            episode_return = 0.0
            reached_goal = False

            for _ in range(eval_max_steps):
                logits, _ = model(state.to(device).unsqueeze(0))
                action = int(torch.argmax(logits.squeeze(0)).item())
                state, reward, done = eval_env.step(action)
                episode_return += reward

                if done:
                    reached_goal = True
                    break

            eval_returns.append(episode_return)
            eval_successes += int(reached_goal)

    eval_mean_return = sum(eval_returns) / len(eval_returns)
    expected_updates = config.updates * config.ppo_epochs_per_rollout

    # How to read the eval return:
    # In this LineWorld, the agent gets `+1.00` when it reaches the goal
    # and `-0.01` for each non-goal step.
    #
    # A good greedy policy moves right until it reaches the goal:
    # ```text
    # position 0 -> position 1 -> position 2 -> position 3 -> position 4 goal
    # ```
    #
    # That gives:
    # ```text
    # -0.01 - 0.01 - 0.01 + 1.00 = 0.97
    # ```
    #
    # So a greedy eval mean return near `0.970` means:
    # the trained agent reached the goal quickly.
    #
    # Do not compare `greedy eval mean return` and `training rollout mean return` directly.
    # They measure different things:
    # - greedy eval mean return is the reward for one test episode
    # - training rollout mean return is total reward across one 64-step rollout batch
    #
    # A 64-step rollout can contain several short successful episodes because
    # the environment resets after the agent reaches the goal.
    # So a training rollout return like `6.279` can be good:
    # it means the agent reached the goal several times inside the 64 collected steps.
    #
    # Example:
    # one successful episode gives about `0.97`.
    # A 64-step rollout can contain several successes:
    # ```text
    # about 6 successful episodes * 0.97 = about 5.82
    # ```
    #
    # Plus some extra collected steps, so `6.279` is not bad.
    # It means the agent is reaching the goal multiple times inside the rollout batch.
    print("PPO learned-policy check:")
    print(f"- greedy eval success: {eval_successes}/{eval_episodes} episodes reached the goal")
    print(f"- greedy eval mean return: {eval_mean_return:.3f} (higher is better; near 1.0 means it reaches the goal quickly)")
    print(f"- training rollout mean return: {summary['mean_return']:.3f} (total reward per {config.rollout_steps}-step rollout)")
    print(f"- optimizer updates: {summary['num_updates']} / {expected_updates}")
    print(f"- final losses: policy={summary['last_policy_loss']:.3f}, value={summary['last_value_loss']:.3f} (losses are debug numbers, not the main pass check)")
    return summary
    #raise NotImplementedError("Collect rollouts, run PPO updates, and return a summary.")

def explain_clipping(summary: PPOSummary) -> str:
    """Write a short note explaining PPO clipping."""
    # Goal:
    # Explain why PPO clips the policy update.
    #
    # Input:
    # - `summary`: PPOSummary
    # - Useful keys:
    #   - `summary["mean_return"]`
    #   - `summary["last_policy_loss"]`
    #   - `summary["num_updates"]`
    #
    # Output:
    # - `str`
    # - One or two plain sentences.
    #
    # Meaning:
    # - PPO compares the new policy with the old policy on the same sampled actions.
    # - Clipping limits that change so one update cannot push the policy too far.
    # - This makes training less jumpy than a plain policy-gradient update.
    #
    # Steps:
    # 1. Mention old and new policy comparison.
    # 2. Mention that clipping limits large changes.
    # 3. Use one concrete number from the summary if useful.
    #
    # Checks:
    # - Keep the note short and plain.
    # - Do not say clipping prevents all bad updates.
    # - Do not say old log-probabilities should be recomputed after training.
    return (
        "PPO compares the new policy with the old policy on the same sampled actions, "
        f"then clips large probability changes so one update does not jump too far. "
        f"This run used {summary['num_updates']} optimizer updates."
    )
    #raise NotImplementedError("Explain PPO clipping in simple words.")

def check_metrics(metrics: PPOSummary) -> bool:
    """Return True when the PPO summary has the expected keys."""
    required = {"mean_return", "last_policy_loss", "last_value_loss", "num_updates", "note"}
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
    """Run the PPO scaffold."""
    config = PPOConfig()
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
    random.seed(config.random_state)
    _, ok = report_task_progress("describe_ppo_plan", describe_ppo_plan, config)
    if not count_or_stop(ok):
        return
    model, ok = report_task_progress("PPOPolicyValueNetwork.__init__", PPOPolicyValueNetwork, config)
    if not count_or_stop(ok):
        return
    sample_x = torch.randn(config.rollout_steps, config.state_dim)
    _, ok = report_task_progress("PPOPolicyValueNetwork.forward", model, sample_x)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("train_ppo", train_ppo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "PPO summary", float(int(passed)), "ppo keys present")
if __name__ == "__main__":
    main()
