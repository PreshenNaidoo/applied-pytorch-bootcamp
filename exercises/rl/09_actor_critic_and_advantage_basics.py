"""
============================================================
EXERCISE 09: ACTOR CRITIC AND ADVANTAGE BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny actor-critic learner.

Plain-language meanings:
- actor means the part that chooses actions
- critic means the part that predicts how good a state is
- advantage means how much better or worse an action did than expected
- baseline means a reference value used to reduce noise
- bootstrap target means a target that uses a reward plus a learned next-state value estimate

WHY THIS EXERCISE EXISTS
REINFORCE is simple but often noisy.
Actor-critic methods reduce that noise
by learning a value baseline with the critic.
This exercise teaches:
- actor outputs
- critic value prediction
- bootstrap targets
- advantage estimates
- actor loss
- critic loss

HOW THIS FITS INTO THE RL SEQUENCE
This file comes after REINFORCE.
REINFORCE taught direct policy learning.
This file adds the next stabilizing idea:
- let a critic estimate how good a state should be
- compare actual outcome with that estimate
- use the difference as the advantage

Beginner intuition:
- the actor says what to do
- the critic says how promising the current state looks
- if the result was better than the critic expected,
  that action should become more likely
- that difference is the advantage

Small example:
- critic predicted value `0.4`
- actual return looked closer to `0.9`
- the advantage is positive
- that means the chosen action was better than expected

DATASET
No dataset file is needed.
Use a tiny local LineWorld environment.

Clear default:
- environment name: `LineWorld`
- states: 5 positions in a line, numbered `0` to `4`
- start state: position `0`
- goal state: position `4`
- actions: `0 = left`, `1 = right`
- reward: `+1` at the goal, small negative reward such as `-0.01` otherwise
- rollout length: keep it short, such as 16 to 32 steps

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- build the tiny `LineWorld` environment in this file
- collect states, actions, rewards, values, and log-probabilities
- compute value targets and advantages from the rollout
- train actor and critic together

INPUTS AND OUTPUTS
Input:
- rollout states
- actions
- rewards
- predicted values
- action log-probabilities
Output:
- mean return
- final actor loss
- final critic loss
- short note about the advantage idea

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use either:
- one shared trunk with actor and critic heads
- or two tiny separate networks

LOSS FUNCTION
Use two losses:
- actor loss from log-probabilities and advantages
- critic loss from predicted values and value targets

METRICS TO TRACK
Track at least:
- mean return
- last actor loss
- last critic loss
- number of rollouts

TARGET RESULT RANGE
Focus on understanding the actor-critic split, not on perfect training curves.

Useful signs:
- actor and critic outputs are both present
- advantages have both sign and magnitude
- actor and critic losses can both be computed
- the training summary shows both optimization signals

COMMON MISTAKES
- confusing the actor output with the critic output
- using raw returns directly without comparing them to predicted values
- forgetting to detach advantages when needed
- treating the critic as another policy head
- not seeing that the critic is there to reduce noise, not choose actions directly

What each step is doing in simple words:
1. build one model that both acts and judges state quality
2. collect a short rollout from the current policy
3. compute what the critic should have predicted
4. compare those targets with the critic outputs
5. compute the advantage as better-or-worse-than-expected
6. update the actor with that advantage signal
7. update the critic to make future predictions better
8. explain why this can be less noisy than REINFORCE

STEP-BY-STEP TODO
1. Build a tiny environment.
2. Build a shared actor-critic network or two small heads.
3. Collect one rollout.
4. Compute value targets.
5. Compute advantages.
6. Compute actor loss.
7. Compute critic loss.
8. Train for several rollouts.
9. Explain why the critic can reduce variance.

DEFINITION OF DONE
You are done when:
- the actor and critic outputs are both present
- advantages are computed
- actor and critic losses are both used
- training runs for several rollouts
- you can explain advantage in plain words
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
class ActorCriticConfig:
    """Settings for the actor-critic exercise."""
    state_dim: int = 5
    num_actions: int = 2
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    rollouts: int = 100
    max_steps_per_rollout: int = 20
    random_state: int = 42


class ActorCriticPlanSummary(TypedDict):
    """Main actor-critic pieces used in this exercise."""

    state_dim: int
    num_actions: int
    main_parts: list[str]


class ActorCriticRollout(TypedDict):
    """One rollout collected from the current actor-critic model."""

    states: list[torch.Tensor]
    actions: list[int]
    rewards: list[float]
    dones: list[bool]
    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]


class ActorCriticSummary(TypedDict):
    """Final summary returned by the tiny actor-critic workflow."""

    mean_return: float
    last_actor_loss: float
    last_critic_loss: float
    num_rollouts: int
    note: str


def describe_actor_critic_plan(config: ActorCriticConfig) -> ActorCriticPlanSummary:
    """Return the main actor-critic pieces."""
    return {
        "state_dim": config.state_dim,
        "num_actions": config.num_actions,
        "main_parts": ["actor_head", "critic_head", "advantage", "two_losses"],
    }


# Beginner hints for this file:
# - the actor and critic solve different jobs
# - the actor should output action logits
# - the critic should output one value per state
# - advantage is often "target return minus predicted value"
# - if the critic improves, the actor updates often become less noisy
class TinyActorCriticEnv:
    """Tiny local environment used for actor-critic."""
    def __init__(self, config: ActorCriticConfig) -> None:
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
        """Run one environment step and return next state, reward, and done."""
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

        goal_position = self.config.state_dim - 1

        if action == 0:
            self.position = max(0, self.position-1)
        elif action == 1:
            self.position = min(goal_position, self.position + 1)

        done = self.position == goal_position

        reward = -0.01
        if done:
            reward = 1.0

        state = torch.zeros(self.config.state_dim, dtype=torch.float32)
        state[self.position] = 1.0

        return state, reward, done
        #raise NotImplementedError("Return the next state, reward, and done flag.")

class ActorCriticNetwork(nn.Module):
    """Tiny network with actor and critic outputs."""
    def __init__(self, config: ActorCriticConfig) -> None:
        # Goal:
        # Build one small shared network with two heads:
        # - the actor head outputs action logits
        # - the critic head outputs one state value
        #
        # Input:
        # - `config`: ActorCriticConfig
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
        # self.actor_head = nn.Linear(config.hidden_dim, config.num_actions)
        # self.critic_head = nn.Linear(config.hidden_dim, 1)
        # ```
        #
        # Meaning:
        # - `trunk` reads the state and makes shared features.
        # - `actor_head` turns those features into one score per action.
        # - `critic_head` turns those features into one value estimate.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Build `self.trunk`.
        # 3. Build `self.actor_head`.
        # 4. Build `self.critic_head`.
        # 5. Do not apply softmax in this method.
        # 6. Do not return anything.
        #
        # Checks:
        # - For input shape `[batch_size, state_dim]`, actor logits should be `[batch_size, num_actions]`.
        # - For input shape `[batch_size, state_dim]`, critic values should be `[batch_size]`.
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(in_features=config.state_dim, out_features=config.hidden_dim),
                                    nn.ReLU(inplace=True))

        self.actor_head = nn.Linear(in_features=config.hidden_dim, out_features=config.num_actions)
        self.critic_head = nn.Linear(in_features=config.hidden_dim, out_features=1)

        #raise NotImplementedError("Build a tiny shared trunk with actor and critic heads.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and state values."""
        # Goal:
        # Return both model outputs for a batch of states.
        #
        # Input:
        # - `x`: torch.Tensor
        # - Shape: `[batch_size, state_dim]`
        # - Example: a batch of 20 states has shape `[20, 5]`.
        #
        # Output:
        # Return a tuple:
        # - `logits`: torch.Tensor with shape `[batch_size, num_actions]`
        # - `values`: torch.Tensor with shape `[batch_size]`
        #
        # Example:
        # - `logits[0]` might be `tensor([0.2, -0.1])` for left and right.
        # - `values[0]` might be `tensor(0.4)`, meaning the critic thinks this state is worth about `0.4`.
        #
        # Method:
        # Use:
        # - `features = self.trunk(x)`
        # - `logits = self.actor_head(features)`
        # - `values = self.critic_head(features).squeeze(-1)`
        #
        # Steps:
        # 1. Pass `x` through the shared trunk.
        # 2. Pass the shared features through the actor head.
        # 3. Pass the shared features through the critic head.
        # 4. Remove only the last size-1 dimension from critic values with `.squeeze(-1)`.
        # 5. Return `(logits, values)`.
        #
        # Checks:
        # - Do not apply softmax to logits in `forward`.
        # - Do not use `.squeeze()` without a dimension, because a batch of size 1 could become shape `[]`.

        features = self.trunk(x)
        logits = self.actor_head(features)
        values = self.critic_head(features)

        return logits, values.squeeze(dim=-1)
        #raise NotImplementedError("Return actor logits and critic values.")

# Collect one rollout.
# Return at least:
# - states
# - actions
# - rewards
# - dones
# - log_probs
# - values
def collect_rollout(
    model: nn.Module,
    env: TinyActorCriticEnv,
    config: ActorCriticConfig,
) -> ActorCriticRollout:
    """Collect one rollout from the current actor-critic policy."""
    # Goal:
    # Run one short rollout with the current model.
    # Store both actor data and critic data for training.
    #
    # Input:
    # - `model`: nn.Module
    # - `env`: TinyActorCriticEnv
    # - `config`: ActorCriticConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"states"`: list[torch.Tensor], each tensor has shape `[state_dim]`
    # - `"actions"`: list[int], each action is `0` or `1`
    # - `"rewards"`: list[float], one reward per step
    # - `"dones"`: list[bool], one done flag per step
    # - `"log_probs"`: list[torch.Tensor], one scalar log-probability per chosen action
    # - `"values"`: list[torch.Tensor], one scalar critic value per state
    #
    # Tiny example:
    # If the rollout takes 4 steps, all six lists should have length `4`.
    #
    # Method:
    # Use PyTorch's categorical distribution:
    # - `logits, values = model(state.unsqueeze(0))`
    # - `dist = torch.distributions.Categorical(logits=logits.squeeze(0))`
    # - `action_tensor = dist.sample()`
    # - `log_prob = dist.log_prob(action_tensor)`
    # - `action = int(action_tensor.item())`
    #
    # The actor uses `log_prob`.
    # The critic uses `values.squeeze(0)`.
    #
    # Steps:
    # 1. Put the model in training mode with `model.train()`.
    # 2. Reset the environment with `state = env.reset()`.
    # 3. Create empty lists for states, actions, rewards, dones, log_probs, and values.
    # 4. Loop up to `config.max_steps_per_rollout`.
    # 5. Run the model on `state.unsqueeze(0)`.
    # 6. Build a `Categorical` distribution from the actor logits.
    # 7. Sample one action and save its log-probability.
    # 8. Save the critic value for the current state.
    # 9. Step the environment with `next_state, reward, done = env.step(action)`.
    # 10. Save the current state, action, reward, done, log-probability, and value.
    # 11. Set `state = next_state`.
    # 12. Stop early if `done` is true.
    # 13. Return the rollout dictionary.
    #
    # Checks:
    # - The six returned lists should have the same length.
    # - Each log-probability should be a scalar tensor that still has gradients.
    # - Each value should be a scalar tensor that still has gradients.

    model.train()

    model_device = next(model.parameters()).device
    state = env.reset().to(model_device)

    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    for i in range(config.max_steps_per_rollout):

        logits, value_batch = model(state.unsqueeze(dim=0))  # add a fake batch dim

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
        # `Categorical` is nicer here because actor-critic also needs `log_prob`.
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
        # # about -0.20
        # ```
        #
        # If the sampled action is action 1, then:
        # ```python
        # log_prob = torch.log(torch.tensor(0.18))
        # # about -1.71
        # ```
        #
        # A more likely action has a log_prob closer to 0.
        # A less likely action has a more negative log_prob.
        
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        action = int(action_tensor.item())

        values.append(value_batch.squeeze(dim=0))   # save critic value

        next_state, reward, done = env.step(action)
        next_state = next_state.to(model_device)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)

        state = next_state

        if done:
            break

    return {"states": states, "actions": actions, "rewards": rewards, "dones": dones, "log_probs": log_probs, "values": values}
    #raise NotImplementedError("Return one rollout with actor and critic information.")

def compute_value_targets(rewards: list[float], gamma: float) -> list[float]:
    """Return discounted value targets for the rollout."""
    # Goal:
    # Turn the rewards from one rollout into one target value per step.
    #
    # The critic looks at a state and tries to predict:
    # "How much reward will I probably get from here until the rollout ends?"
    #
    # This function builds the correct answer for each step after the rollout is done.
    # Those answers are called value targets.
    #
    # Input:
    # - `rewards`: list[float]
    #   One reward for each step the agent took.
    #   Example: `[-0.01, -0.01, -0.01, 1.0]`
    #   This means:
    #   - step 0 got `-0.01`
    #   - step 1 got `-0.01`
    #   - step 2 got `-0.01`
    #   - step 3 reached the goal and got `1.0`
    # - `gamma`: float
    #   Example: `0.99`.
    #   This is the discount factor.
    #   It controls how much later rewards count.
    #   If `gamma` is close to `1.0`, future rewards count almost fully.
    #   If `gamma` is smaller, future rewards count less.
    #
    # Output:
    # - `list[float]`
    # - Same length as `rewards`.
    # - `targets[i]` is the total future reward from step `i` onward.
    # - Each target is what the critic should have predicted for that step.
    #
    # Tiny example with no discount:
    # If `gamma = 1.0` and rewards are `[-0.01, -0.01, -0.01, 1.0]`,
    # the targets are:
    # - step 3 target: `1.0`
    # - step 2 target: `-0.01 + 1.0 = 0.99`
    # - step 1 target: `-0.01 - 0.01 + 1.0 = 0.98`
    # - step 0 target: `-0.01 - 0.01 - 0.01 + 1.0 = 0.97`
    #
    # So the result is `[0.97, 0.98, 0.99, 1.0]`.
    #
    # Tiny example with discount:
    # If `gamma = 0.5` and rewards are `[1.0, 1.0]`,
    # then the last step target is `1.0`.
    # The first step target is `1.0 + 0.5 * 1.0 = 1.5`.
    # So the result is `[1.5, 1.0]`.
    #
    # Formula:
    # ```text
    # target_t = reward_t + gamma * target_(t+1)
    # ```
    #
    # Meaning:
    # - `target_t` is what the critic should predict at step `t`.
    # - `reward_t` is the reward at step `t`.
    # - `target_(t+1)` is the already-computed target for the next step.
    # - `gamma * target_(t+1)` means "count the future, but maybe discount it".
    #
    # Why work backward?
    # To compute the target for step 0, we need to know the target for step 1.
    # To compute the target for step 1, we need to know the target for step 2.
    # The final step is easiest because there is no later reward after it.
    # So we start from the end and move backward.
    #
    # `running_target` means:
    # "the total future reward from the step we are currently looking at."
    #
    # As we move backward:
    # - first it holds the target for the final step
    # - then it holds the target for the step before that
    # - then the step before that
    # - and so on
    #
    # Steps:
    # 1. Create an empty `targets` list.
    # 2. Set `running_target = 0.0` because there is no reward after the rollout ends.
    # 3. Loop through the rewards backward with `reversed(rewards)`.
    # 4. Update `running_target` with:
    #    `running_target = reward + gamma * running_target`
    # 5. Put the new target at the front with `targets.insert(0, running_target)`.
    #    We insert at the front because we are reading rewards backward,
    #    but the returned list should still be in normal time order.
    # 6. Return `targets`.
    #
    # Checks:
    # - With `rewards=[1.0]`, the result should be `[1.0]`.
    # - With `rewards=[1.0, 1.0]` and `gamma=0.5`, the result should be `[1.5, 1.0]`.
    # - With `rewards=[-0.01, -0.01, -0.01, 1.0]` and `gamma=1.0`,
    #   the result should be `[0.97, 0.98, 0.99, 1.0]`.
    # - The final target should equal the final reward because there is no later reward.

    targets = []
    running_target = 0.0

    for reward in reversed(rewards):
        running_target = reward + gamma * running_target
        targets.insert(0, running_target)

    return targets
    #raise NotImplementedError("Return discounted value targets.")

def compute_advantages(value_targets: list[float], predicted_values: list[torch.Tensor]) -> torch.Tensor:
    """Return the rollout advantages."""
    # Goal:
    # Return how much better or worse each step was than the critic expected.
    #
    # Input:
    # - `value_targets`: list[float]
    #   Example: `[0.8, 0.9, 1.0]`
    # - `predicted_values`: list[torch.Tensor]
    #   Each tensor is one scalar critic prediction from the rollout.
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[num_steps]`
    #
    # Formula:
    # ```text
    # advantage = value_target - predicted_value
    # ```
    #
    # Meaning:
    # - positive advantage means the result was better than expected
    # - negative advantage means the result was worse than expected
    #
    # Method:
    # Use:
    # - `value_tensor = torch.stack(predicted_values).squeeze(-1)`
    # - `target_tensor = torch.tensor(value_targets, dtype=torch.float32, device=value_tensor.device)`
    # - `advantages = target_tensor - value_tensor.detach()`
    #
    # `detach()` means the actor loss will not train the critic by accident.
    #
    # Steps:
    # 1. Stack predicted values into shape `[num_steps]`.
    # 2. Convert targets into a tensor on the same device.
    # 3. Subtract detached predicted values from targets.
    # 4. Return the advantages.
    #
    # Checks:
    # - If target is `0.9` and predicted value is `0.4`, advantage should be `0.5`.
    # - The output length should equal `len(value_targets)`.

    value_tensor = torch.stack(predicted_values).squeeze(-1)
    target_tensor = torch.tensor(value_targets, dtype=torch.float32, device=value_tensor.device)
    advantages = target_tensor - value_tensor.detach()
    return advantages
    #raise NotImplementedError("Return target value minus predicted value.")

def actor_loss(log_probs: list[torch.Tensor], advantages: torch.Tensor) -> torch.Tensor:
    """Return the actor loss."""
    # Goal:
    # Return one scalar loss for the actor.
    # Positive advantages should make the chosen actions more likely.
    # Negative advantages should make the chosen actions less likely.
    #
    # Input:
    # - `log_probs`: list[torch.Tensor]
    #   Each item is the log-probability of the action that was sampled.
    # - `advantages`: torch.Tensor
    #   Shape: `[num_steps]`
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[]`, meaning one scalar loss.
    #
    # Formula:
    # ```text
    # actor_loss = mean(-log_prob_of_chosen_action * advantage)
    # ```
    #
    # Meaning:
    # - `log_prob_of_chosen_action` comes from the actor.
    # - `advantage` says whether that action did better or worse than expected.
    # - The minus sign is needed because PyTorch optimizers minimize loss.
    #
    # Method:
    # Use:
    # - `log_prob_tensor = torch.stack(log_probs)`
    # - `loss = -(log_prob_tensor * advantages.detach()).mean()`
    #
    # `advantages.detach()` keeps this actor loss from changing the critic values directly.
    #
    # Steps:
    # 1. Stack log-probabilities into shape `[num_steps]`.
    # 2. Multiply each log-probability by the matching advantage.
    # 3. Negate the mean.
    # 4. Return the scalar actor loss.
    #
    # Checks:
    # - `len(log_probs)` should equal `advantages.shape[0]`.
    # - Do not call `.item()` before `loss.backward()`.

    log_prob_tensor = torch.stack(log_probs)
    loss = -(log_prob_tensor * advantages.detach()).mean()

    return loss
    #raise NotImplementedError("Return the actor loss using log_probs and advantages.")

def critic_loss(predicted_values: list[torch.Tensor], value_targets: list[float]) -> torch.Tensor:
    """Return the critic loss."""
    # Goal:
    # Return one scalar loss that trains the critic to predict value targets.
    #
    # Input:
    # - `predicted_values`: list[torch.Tensor]
    #   Each item is one scalar critic prediction from the rollout.
    # - `value_targets`: list[float]
    #   Each item is the discounted return for the same step.
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[]`, meaning one scalar loss.
    #
    # Formula:
    # ```text
    # critic_loss = mean((predicted_value - value_target) ** 2)
    # ```
    #
    # Meaning:
    # - the critic is doing normal regression
    # - it should move its predicted value closer to the target return
    #
    # Method:
    # Use:
    # - `value_tensor = torch.stack(predicted_values).squeeze(-1)`
    # - `target_tensor = torch.tensor(value_targets, dtype=torch.float32, device=value_tensor.device)`
    # - `nn.functional.mse_loss(value_tensor, target_tensor)`
    #
    # Steps:
    # 1. Stack predicted values into shape `[num_steps]`.
    # 2. Convert targets into a tensor on the same device.
    # 3. Compute mean squared error.
    # 4. Return the scalar critic loss.
    #
    # Checks:
    # - `len(predicted_values)` should equal `len(value_targets)`.
    # - Do not detach `value_tensor` here, because this loss should train the critic.

    value_tensor = torch.stack(predicted_values).squeeze(-1)
    target_tensor = torch.tensor(value_targets, dtype=torch.float32, device=value_tensor.device)
    loss = nn.functional.mse_loss(value_tensor, target_tensor)

    return loss
    #raise NotImplementedError("Return the critic loss between predicted and target values.")

def train_actor_critic(config: ActorCriticConfig) -> ActorCriticSummary:
    """Run the full actor-critic workflow."""
    # Goal:
    # Collect rollouts, compute actor and critic losses, update the model, and return a summary.
    #
    # Input:
    # - `config`: ActorCriticConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"mean_return"`: float, average rollout return
    # - `"last_actor_loss"`: float, actor loss from the final update
    # - `"last_critic_loss"`: float, critic loss from the final update
    # - `"num_rollouts"`: int, should equal `config.rollouts`
    # - `"note"`: str, short explanation from `explain_advantage(...)`
    #
    # Method:
    # Use these objects:
    # - `env = TinyActorCriticEnv(config)`
    # - `model = ActorCriticNetwork(config).to(device)`
    # - `optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)`
    #
    # Use these helper functions:
    # - `rollout = collect_rollout(model, env, config)`
    # - `value_targets = compute_value_targets(rollout["rewards"], config.gamma)`
    # - `advantages = compute_advantages(value_targets, rollout["values"])`
    # - `a_loss = actor_loss(rollout["log_probs"], advantages)`
    # - `c_loss = critic_loss(rollout["values"], value_targets)`
    #
    # Combine losses with:
    # `loss = a_loss + c_loss`
    #
    # Steps:
    # 1. Build the environment, model, and optimizer.
    # 2. Create an empty list for rollout returns.
    # 3. Start `last_actor_loss = 0.0` and `last_critic_loss = 0.0`.
    # 4. Loop for `config.rollouts`.
    # 5. Collect one rollout.
    # 6. Compute value targets from rewards.
    # 7. Compute advantages from targets and critic values.
    # 8. Compute actor loss.
    # 9. Compute critic loss.
    # 10. Add the two losses.
    # 11. Clear gradients with `optimizer.zero_grad()`.
    # 12. Run `loss.backward()`.
    # 13. Update the model with `optimizer.step()`.
    # 14. Save `sum(rollout["rewards"])` in rollout returns.
    # 15. Save both loss values as Python floats.
    # 16. Build the summary dictionary.
    # 17. Add the note from `explain_advantage(summary)`.
    # 18. Return the summary.
    #
    # Checks:
    # - `num_rollouts` should equal `config.rollouts`.
    # - `mean_return` should be a Python float.
    # - Actor and critic losses should both be used in the update.

    env = TinyActorCriticEnv(config)
    model = ActorCriticNetwork(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    rollout_returns = []

    last_actor_loss, last_critic_loss = 0.0, 0.0

    for i in range(config.rollouts):

        rollout = collect_rollout(model, env, config)

        # Build one target per step: "from this step onward, how good did things turn out?"
        # These are the answers the critic is trying to learn.
        value_targets = compute_value_targets(rollout["rewards"], config.gamma)

        # Compare what really happened with what the critic guessed.
        # Positive advantage means the step turned out better than the critic expected.
        # Negative advantage means it turned out worse than the critic expected.
        advantages = compute_advantages(value_targets, rollout["values"])

        # Train the actor with the advantage signal.
        # If an action did better than expected, make that action more likely next time.
        # If it did worse than expected, make that action less likely next time.
        a_loss = actor_loss(rollout["log_probs"], advantages)

        # Train the critic to make better value guesses.
        # It compares the critic's old guesses with the value targets from this rollout.
        c_loss = critic_loss(rollout["values"], value_targets)

        loss = a_loss + c_loss

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        rollout_returns.append(sum(rollout["rewards"]))
        last_actor_loss = float(a_loss.item())
        last_critic_loss = float(c_loss.item())

        if (i + 1) % 10 == 0 or i == 0:
            reached_goal = bool(rollout["dones"][-1])
            print(
                f"rollout {i + 1:03d}: "
                f"return={rollout_returns[-1]:.4f}, "
                f"steps={len(rollout['rewards'])}, "
                f"reached_goal={reached_goal}, "
                f"actor_loss={last_actor_loss:.4f}, "
                f"critic_loss={last_critic_loss:.4f}"
            )

    mean_return = float(sum(rollout_returns) / len(rollout_returns))
    check_window = min(10, len(rollout_returns))
    early_mean_return = float(sum(rollout_returns[:check_window]) / check_window)
    late_mean_return = float(sum(rollout_returns[-check_window:]) / check_window)

    # In this LineWorld, the agent gets -0.01 for each non-goal step
    # and +1.0 when it reaches the goal.
    # So a final return near 0.97 usually means:
    # - the agent reached the goal
    # - it took about 3 small penalty steps first
    # - the total was roughly -0.01 - 0.01 - 0.01 + 1.0 = 0.97
    print("Training check:")
    print(f"  first {check_window} rollout mean return: {early_mean_return:.4f}")
    print(f"  last {check_window} rollout mean return: {late_mean_return:.4f}")
    print(f"  final rollout return: {rollout_returns[-1]:.4f}")
    print("  Higher return is better here because the agent gets +1 at the goal.")

    summary: ActorCriticSummary = {
        "mean_return": mean_return,
        "last_actor_loss": last_actor_loss,
        "last_critic_loss": last_critic_loss,
        "num_rollouts": config.rollouts,
        "note": "",
    }
    summary["note"] = explain_advantage(summary)

    return summary
    #raise NotImplementedError("Collect rollouts, compute advantages, train actor and critic, and return a summary.")

def explain_advantage(summary: ActorCriticSummary) -> str:
    """Write a short note about the meaning of advantage."""
    # Goal:
    # Explain the advantage idea and the critic's job in simple words.
    #
    # Input:
    # - `summary`: ActorCriticSummary
    # - Useful keys:
    #   - `summary["mean_return"]`
    #   - `summary["last_actor_loss"]`
    #   - `summary["last_critic_loss"]`
    #
    # Output:
    # - `str`
    # - One or two plain sentences.
    #
    # Meaning:
    # - The critic guesses how good a state is.
    # - The advantage compares the real target with that guess.
    # - The actor uses this difference to know whether the sampled action was better or worse than expected.
    #
    # Steps:
    # 1. Mention that the critic gives a baseline guess.
    # 2. Mention that advantage is target minus that guess.
    # 3. Use one concrete number from the summary if useful.
    #
    # Checks:
    # - Keep the note short and plain.
    # - Do not say the critic chooses the action.
    # - Do not say advantage is always positive.
    mean_return = summary["mean_return"]
    return (
        f"The critic gives a baseline guess for each state. "
        f"Advantage is the target return minus that guess, and this run had mean return {mean_return:.3f}."
    )
    #raise NotImplementedError("Explain advantage and the critic in simple words.")

def check_metrics(metrics: ActorCriticSummary) -> bool:
    """Return True when the actor-critic summary has the expected keys."""
    required = {"mean_return", "last_actor_loss", "last_critic_loss", "num_rollouts", "note"}
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
    """Run the actor-critic scaffold."""
    config = ActorCriticConfig()
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
    _, ok = report_task_progress("describe_actor_critic_plan", describe_actor_critic_plan, config)
    if not count_or_stop(ok):
        return
    model, ok = report_task_progress("ActorCriticNetwork.__init__", ActorCriticNetwork, config)
    if not count_or_stop(ok):
        return
    sample_x = torch.randn(config.max_steps_per_rollout, config.state_dim)
    _, ok = report_task_progress("ActorCriticNetwork.forward", model, sample_x)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("train_actor_critic", train_actor_critic, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        print("Actor-critic results:")
        print(f"  mean_return: {metrics['mean_return']:.4f}")
        print(f"  last_actor_loss: {metrics['last_actor_loss']:.4f}")
        print(f"  last_critic_loss: {metrics['last_critic_loss']:.4f}")
        print(f"  num_rollouts: {metrics['num_rollouts']}")
        print(f"  note: {metrics['note']}")
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "actor-critic summary", float(int(passed)), "actor-critic keys present")
if __name__ == "__main__":
    main()
