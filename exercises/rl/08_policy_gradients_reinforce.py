"""
============================================================
EXERCISE 08: POLICY GRADIENTS REINFORCE
============================================================

WHAT YOU ARE BUILDING
You are building a tiny policy-gradient agent with REINFORCE.

Plain-language meanings:
- a policy is the rule that chooses actions
- policy gradient means directly improving that action rule
- a trajectory is the list of states, actions, and rewards from one episode
- return means total future reward from a time step onward
- stochastic means the policy samples actions from probabilities instead of always picking one fixed action

The full picture:

```text
state -> policy network -> action probabilities -> sample action
take action in environment -> get reward
save the log-probability of the sampled action
repeat until the episode ends
compute returns from the reward list
use those returns to update the policy
```

The important thing to picture is:

```text
REINFORCE does not learn Q-values first.
It directly changes the policy so useful sampled actions become more likely.
```

WHY THIS EXERCISE EXISTS
Q-learning is one branch of RL.
Policy gradients are another.
They are important because many modern RL methods build on them.
This exercise teaches:
- stochastic policy outputs
- sampling actions from a policy
- trajectories
- discounted returns
- REINFORCE loss
- episode-based training

HOW THIS FITS INTO THE RL SEQUENCE
This file starts the policy-gradient branch of the RL track.
Earlier files taught value-based methods such as Q-learning and DQN.
Now the focus changes:
- instead of learning action values first,
- learn the action rule directly

Short comparison:

```text
DQN:
learn Q-values for actions
choose actions from those values

REINFORCE:
sample actions from the policy
use episode returns to make good sampled actions more likely
```

Beginner intuition:
- instead of learning a Q-table or Q-network first,
  you can directly train the action rule itself
- if an episode went well,
  the chosen actions in that episode should become more likely
- if an episode went badly,
  those actions should become less likely

Small example:
- policy outputs probabilities for `left` and `right`
- you sample one action
- after the episode, you use the return to push good actions up
  and bad actions down

Why `log_prob` matters:
- the action was sampled from probabilities
- later, PyTorch needs to know how likely the policy said that exact action was
- `log_prob` keeps that connection so gradients can make the sampled action more or less likely

Tiny example:

```text
policy says:
left = 0.30
right = 0.70

sampled action = right
saved value = log_prob(right)
```

If the episode return is high, the update makes `right` more likely in that state.
If the episode return is low, the update makes `right` less likely in that state.

IMPORTANT SCOPE NOTE
REINFORCE is simple but noisy.
That is normal.
This exercise exists so you can see why episode returns may jump around and why later methods add baselines.

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
- episode ends at the goal or after a short max step count

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- build the tiny `LineWorld` environment in this file
- collect one full episode at a time
- save states, actions, rewards, and log-probabilities
- train after the episode finishes

INPUTS AND OUTPUTS
Input:
- state
- sampled action
- reward
- log-probability of the action
Output:
- episode returns: total reward from each full episode
- mean return: average episode return across training
- final loss: loss from the final episode update
- short note about policy-gradient noise

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a tiny policy network:
- state vector in
- action logits out

LOSS FUNCTION
Use the REINFORCE loss:
- log-probability of chosen actions
- weighted by discounted returns

Plain meaning:

```text
loss = -log_prob(chosen_action) * return
```

PyTorch minimizes loss.
The minus sign is there because we want high-return actions to become more likely, not less likely.

METRICS TO TRACK
Track at least:
- episode returns: total reward from each episode
- mean return: average episode return
- last loss: the final policy-gradient loss value
- number of episodes

How to read the metrics:
- higher episode return is better
- mean return is the main score to watch
- last loss alone does not prove the policy is good
- REINFORCE can look jumpy because each update uses sampled episode data

TARGET RESULT RANGE
Focus on understanding trajectories and returns, not on stable high reward.

Useful signs:
- trajectories are collected correctly
- discounted returns can be computed from the reward list
- the loss uses both log-probabilities and returns
- training runs for many episodes without shape errors
- later episode returns are sometimes better than early episode returns

COMMON MISTAKES
- treating logits like probabilities without sampling properly
- forgetting to store the log-probability of the chosen action
- using immediate reward instead of return
- treating the loss value as the main success metric
- expecting REINFORCE to look smooth even in tiny runs
- forgetting that this method updates after an episode, not after a classification batch

What each step is doing in simple words:
1. build a tiny world and a policy that can sample actions
2. let the current policy act for one full episode
3. save what happened in that episode
4. compute how good each step turned out in the long run
5. build the REINFORCE loss from those returns
6. update the policy
7. repeat and watch the episode returns
8. explain why direct policy learning can be noisy 

STEP-BY-STEP TODO
1. Build a tiny environment.
2. Build a policy network.
3. Roll out one episode by sampling actions.
4. Compute discounted returns.
5. Compute the REINFORCE loss.
6. Update the policy.
7. Track episode returns over training.
8. Explain why policy gradients can be noisy.

DEFINITION OF DONE
You are done when:
- episodes are collected
- returns are computed
- the REINFORCE loss uses log-probabilities and returns
- the policy updates run
- episode returns are reported so you can check whether learning helped
- you can explain why high-return episodes matter more
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
class ReinforceConfig:
    """Settings for the REINFORCE exercise."""
    state_dim: int = 5
    num_actions: int = 2
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    episodes: int = 100
    max_steps_per_episode: int = 20
    random_state: int = 42


class ReinforcePlanSummary(TypedDict):
    """Main REINFORCE pieces used in this exercise."""

    state_dim: int
    num_actions: int
    main_parts: list[str]


class EpisodeTrajectory(TypedDict):
    """One episode collected from the current policy."""

    states: list[torch.Tensor]
    actions: list[int]
    rewards: list[float]
    log_probs: list[torch.Tensor]


class ReinforceSummary(TypedDict):
    """Final summary returned by the tiny REINFORCE workflow."""

    episode_returns: list[float]
    mean_return: float
    last_loss: float
    num_episodes: int
    note: str


def describe_reinforce_plan(config: ReinforceConfig) -> ReinforcePlanSummary:
    """Return the main REINFORCE pieces."""
    return {
        "state_dim": config.state_dim,
        "num_actions": config.num_actions,
        "main_parts": ["policy_network", "trajectory_collection", "discounted_returns", "log_prob_loss"],
    }


# Beginner hints for this file:
# - the policy network should output action logits, not Q-values
# - `torch.distributions.Categorical(logits=...)` turns those logits into action probabilities
# - sampling matters because REINFORCE learns from the actions actually taken
# - returns should be larger near the start of a good episode
# - log-probability is the usual way to write the policy-gradient loss
# - keep the environment tiny so you can inspect one full trajectory
# - the policy learns after the episode, not after every single step
class TinyPolicyEnv:
    """Tiny local environment used for REINFORCE."""
    def __init__(self, config: ReinforceConfig) -> None:
        self.config = config
        self.position = 0

    def reset(self) -> torch.Tensor:
        """Reset the environment and return the starting state."""
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
        # Metric meaning:
        # In this tiny world, a better episode usually reaches the goal faster.
        # The agent gets `1.0` at the goal and `-0.01` on other steps.
        # So a higher episode return means the policy made better moves overall.
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
            self.position = min(goal_position, self.position+1)

        done = self.position == goal_position
        if done:
            reward = 1.0
        else:
            reward = -0.01

        state = torch.zeros(self.config.state_dim, dtype=torch.float32)
        state[self.position] = 1

        return state, reward, done
        #raise NotImplementedError("Return the next state, reward, and done flag.")

class PolicyNetwork(nn.Module):
    """Tiny policy network."""
    def __init__(self, config: ReinforceConfig) -> None:
        # Goal:
        # Build a tiny MLP that maps a state vector to one logit per action.
        # A logit is a raw score. It is not a probability yet.
        #
        # Input:
        # - `config`: ReinforceConfig
        #
        # Output:
        # - `None`
        # - Store the network layers on `self`.
        #
        # Method:
        # Use `torch.nn.Sequential` with this shape:
        # ```python
        # self.net = nn.Sequential(
        #     nn.Linear(config.state_dim, config.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_dim, config.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_dim, config.num_actions),
        # )
        # ```
        #
        # Meaning:
        # - `config.state_dim` is the one-hot state length.
        # - `config.num_actions` is the number of action scores to output.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Build `self.net`.
        # 3. Do not apply softmax here.
        # 4. Do not return anything.
        #
        # Checks:
        # - For input shape `[batch_size, state_dim]`, output shape should be `[batch_size, num_actions]`.
        super().__init__()

        self.net = nn.Sequential(nn.Linear(in_features=config.state_dim, out_features=config.hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=config.hidden_dim, out_features=config.hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=config.hidden_dim, out_features=config.num_actions))

        #raise NotImplementedError("Build a tiny policy network that returns action logits.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw action logits."""
        # Goal:
        # Return one raw score for each action and each state in the batch.
        #
        # Input:
        # - `x`: torch.Tensor
        # - Shape: `[batch_size, state_dim]`
        # - Example: a batch of 20 states has shape `[20, 5]`.
        #
        # Output:
        # - `torch.Tensor`
        # - Shape: `[batch_size, num_actions]`
        # - Example: `[20, 2]` for left and right actions.
        #
        # Method:
        # Return:
        # `self.net(x)`
        #
        # Steps:
        # 1. Pass `x` through the network.
        # 2. Return the logits.
        #
        # Checks:
        # - Do not apply softmax in `forward`.
        # - The second dimension should equal `num_actions`.

        logits = self.net(x)
        return logits
        #raise NotImplementedError("Return one logit per action.")

# Roll out one episode.
# Return at least:
# - states
# - actions
# - rewards
# - log_probs
def collect_episode(policy_model: nn.Module, env: TinyPolicyEnv, config: ReinforceConfig) -> EpisodeTrajectory:
    """Collect one episode from the current policy."""
    # Goal:
    # Run one full episode with the current policy.
    # Store the action, reward, and log-probability for each step.
    #
    # Where this fits:
    # REINFORCE needs a complete episode before it updates.
    # This function collects that episode.
    # It does not call `loss.backward()` or `optimizer.step()`.
    #
    # Input:
    # - `policy`: nn.Module
    # - `env`: TinyPolicyEnv
    # - `config`: ReinforceConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"states"`: list[torch.Tensor], each tensor has shape `[state_dim]`
    # - `"actions"`: list[int], each action is `0` or `1`
    # - `"rewards"`: list[float], one reward per step
    # - `"log_probs"`: list[torch.Tensor], one scalar log-probability per chosen action
    #
    # Tiny example:
    # If an episode takes 4 steps, all four lists should have length `4`.
    #
    # Method:
    # Use PyTorch's categorical distribution:
    # - `logits = policy(state.unsqueeze(0))`
    # - `dist = torch.distributions.Categorical(logits=logits.squeeze(0))`
    # - `action_tensor = dist.sample()`
    # - `log_prob = dist.log_prob(action_tensor)`
    # - `action = int(action_tensor.item())`
    #
    # `Categorical(logits=...)` turns raw action scores into a distribution.
    # It is better than sampling from `torch.softmax(...)` by hand because it also gives `log_prob`.
    #
    # Why save `log_prob`:
    # The action is sampled, so later we need to know how likely the policy said that sampled action was.
    # The REINFORCE loss uses `log_prob * return` to push that sampled action up or down.
    #
    # Steps:
    # 1. Put the policy in training mode with `policy.train()`.
    # 2. Reset the environment with `state = env.reset()`.
    # 3. Create empty lists for states, actions, rewards, and log_probs.
    # 4. Loop up to `config.max_steps_per_episode`.
    # 5. Run the policy on `state.unsqueeze(0)`.
    # 6. Build a `Categorical` distribution from the logits.
    # 7. Sample one action and save its log-probability.
    # 8. Step the environment with `next_state, reward, done = env.step(action)`.
    # 9. Save the current state, action, reward, and log-probability.
    # 10. Set `state = next_state`.
    # 11. Stop early if `done` is true.
    # 12. Return the trajectory dictionary.
    #
    # Checks:
    # - `len(rewards)` should be no greater than `config.max_steps_per_episode`.
    # - The four returned lists should have the same length.
    # - Each log-probability should be a scalar tensor that still has gradients.

    model_device = next(policy_model.parameters()).device

    policy_model.train()

    state = env.reset()

    states, actions, rewards, log_probs = [], [], [], []

    for i in range(config.max_steps_per_episode):
        
        logits = policy_model(state.unsqueeze(dim=0).to(model_device))   # add a fake batch dim  shape = [1, state_dim]   batch_size 1

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
        # `Categorical` is nicer here because REINFORCE also needs `log_prob`.
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
        #
        # REINFORCE saves this value because the loss later says:
        # "if the return was good, make this sampled action more likely next time;
        # if the return was bad, make this sampled action less likely next time."
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        action = int(action_tensor.item())

        next_state, reward, done = env.step(action)

        actions.append(action)
        states.append(state)
        rewards.append(reward)
        log_probs.append(log_prob)
        
        state = next_state

        if done:
            break

    return {"states":states, "actions":actions, "rewards": rewards, "log_probs": log_probs}
    #raise NotImplementedError("Return one trajectory of states, actions, rewards, and log_probs.")

def compute_discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Return discounted returns for each time step."""
    # Goal:
    # Return the total future reward from each step.
    # Future rewards count less when they are farther away.
    #
    # Input:
    # - `rewards`: list[float]
    #   Example: `[-0.01, -0.01, 1.0]`
    # - `gamma`: float
    #   Example: `0.99`
    #
    # Output:
    # - `list[float]`
    # - Same length as `rewards`.
    # - `returns[i]` is the discounted return starting at step `i`.
    #
    # Tiny example:
    # ```text
    # rewards = [0.0, 0.0, 1.0]
    # gamma = 0.9
    #
    # returns = [0.81, 0.9, 1.0]
    # ```
    #
    # Plain meaning:
    # The first action gets some credit for the final reward because it helped start the path to success.
    #
    # Formula:
    # ```text
    # G_t = reward_t + gamma * G_(t+1)
    # ```
    #
    # Meaning:
    # - `G_t` is the return at step `t`.
    # - `reward_t` is the reward at step `t`.
    # - `gamma` is the discount. It is usually close to `1.0`.
    # - `G_(t+1)` is the return from the next step.
    #
    # Method:
    # Work backward through the reward list:
    # - start with `running_return = 0.0`
    # - update with `running_return = reward + gamma * running_return`
    # - insert each new value at the front of the returns list
    #
    # Steps:
    # 1. Create an empty `returns` list.
    # 2. Set `running_return = 0.0`.
    # 3. Loop through `reversed(rewards)`.
    # 4. Apply the formula above.
    # 5. Insert the new return at the front with `returns.insert(0, running_return)`.
    # 6. Return `returns`.
    #
    # Checks:
    # - With `rewards=[1.0]`, the result should be `[1.0]`.
    # - With `rewards=[1.0, 1.0]` and `gamma=0.5`, the result should be `[1.5, 1.0]`.
    # - The final return should equal the final reward because there is no future reward after it.

    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    return returns

    #raise NotImplementedError("Return the discounted return from each step onward.")

# Compute the REINFORCE loss.
# In plain words:
# - increase probability of actions from good returns
# - decrease probability of actions from poor returns
def reinforce_loss(log_probs: list[torch.Tensor], returns: list[float]) -> torch.Tensor:
    """Return the REINFORCE loss."""
    # Goal:
    # Return one scalar loss that trains the policy from one episode.
    # High-return actions should become more likely.
    #
    # Input:
    # - `log_probs`: list[torch.Tensor]
    #   Each item is the log-probability of the action that was actually sampled.
    # - `returns`: list[float]
    #   Each item is the discounted return for the same time step.
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[]`, meaning one scalar loss.
    #
    # Formula:
    # ```text
    # loss = mean(-log_prob_of_chosen_action * return)
    # ```
    #
    # Meaning:
    # - `log_prob_of_chosen_action` comes from the policy distribution.
    # - `return` says how good the rest of the episode was after that action.
    # - The minus sign is needed because PyTorch optimizers minimize loss.
    #
    # Why multiply `log_prob` by `return`:
    # `log_prob` says which action the policy chose and how likely that action was.
    # `return` says how good things turned out after that action.
    #
    # So this:
    # ```text
    # -log_prob * return
    # ```
    #
    # means:
    # ```text
    # change the policy based on the action that was actually chosen,
    # but scale the change by how good the outcome was
    # ```
    #
    # Example:
    # If the chosen action helped reach the goal, its return is high.
    # A high return gives that action a bigger update.
    #
    # If the chosen action led to a bad result, its return may be low or negative.
    # A low or negative return gives a weak update or pushes the action down.
    #
    # Without the return, the loss would know which action was chosen,
    # but it would not know whether that action was useful.
    #
    # Why the sign is negative:
    # If the return is high, we want the chosen action to become more likely.
    # Optimizers minimize loss, so the negative sign turns "make this more likely" into a minimization problem.
    #
    # Small example:
    # Suppose the chosen action had probability `0.20`.
    # Its log-probability is about `log(0.20) = -1.61`.
    # If the return is `10.0`, then:
    # ```text
    # loss = -log_prob * return
    # loss = -(-1.61) * 10.0
    # loss = 16.1
    # ```
    #
    # Now suppose the policy makes that same action more likely, with probability `0.80`.
    # Its log-probability is about `log(0.80) = -0.22`.
    # With the same return:
    # ```text
    # loss = -(-0.22) * 10.0
    # loss = 2.2
    # ```
    #
    # The optimizer tries to make loss smaller.
    # For a good return, making the chosen action more likely makes the loss smaller.
    # That is why the negative sign points learning in the right direction.
    #
    # What this loss does not do:
    # It does not learn a Q-table.
    # It does not compare against a target network.
    # It directly changes the policy from sampled actions and returns.
    #
    # Method:
    # Use:
    # - `log_prob_tensor = torch.stack(log_probs)`
    # - `return_tensor = torch.tensor(returns, dtype=torch.float32, device=log_prob_tensor.device)`
    # - `loss = -(log_prob_tensor * return_tensor).mean()`
    #
    # Steps:
    # 1. Stack the log-probabilities into one tensor with shape `[num_steps]`.
    # 2. Convert returns into a float tensor on the same device.
    # 3. Multiply each log-probability by the matching return.
    # 4. Negate the mean.
    # 5. Return the scalar loss.
    #
    # Checks:
    # - `len(log_probs)` should equal `len(returns)`.
    # - The returned loss should be a scalar tensor.
    # - Do not call `.item()` before `loss.backward()`, because that would remove gradients.

    log_prob_tensor = torch.stack(log_probs)
    return_tensor = torch.tensor(returns, dtype=torch.float32, device=log_prob_tensor.device)
    loss = -(log_prob_tensor * return_tensor).mean()

    return loss
    #raise NotImplementedError("Return the REINFORCE loss from log_probs and returns.")

def train_reinforce(config: ReinforceConfig) -> ReinforceSummary:
    """Run the full REINFORCE workflow."""
    # Goal:
    # Collect episodes, update the policy after each episode, and return a summary.
    #
    # Where this fits:
    # This is the first function that actually learns.
    # Earlier helpers build the environment, collect one episode, compute returns, and build the loss.
    # This function connects those pieces and calls `optimizer.step()`.
    #
    # Input:
    # - `config`: ReinforceConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"episode_returns"`: list[float], one total reward per episode
    # - `"mean_return"`: float, average of all episode returns
    # - `"last_loss"`: float, loss from the final episode update
    # - `"num_episodes"`: int, should equal `config.episodes`
    # - `"note"`: str, short explanation from `explain_policy_gradient_noise(...)`
    #
    # Metric meaning:
    # - `episode_returns` is the score history.
    # - `mean_return` is the main training score in this tiny exercise.
    # - `last_loss` only tells you the final update loss. It is not the main proof that learning worked.
    #
    # Method:
    # Use these objects:
    # - `env = TinyPolicyEnv(config)`
    # - `policy = PolicyNetwork(config).to(device)`
    # - `optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)`
    #
    # Use these helper functions:
    # - `trajectory = collect_episode(policy, env, config)`
    # - `returns = compute_discounted_returns(trajectory["rewards"], config.gamma)`
    # - `loss = reinforce_loss(trajectory["log_probs"], returns)`
    #
    # Steps:
    # 1. Build the environment, policy, and optimizer.
    # 2. Create an empty list for episode returns.
    # 3. Start `last_loss = 0.0`.
    # 4. Loop for `config.episodes`.
    # 5. Collect one episode.
    # 6. Compute discounted returns from that episode's rewards.
    # 7. Compute the REINFORCE loss.
    # 8. Clear gradients with `optimizer.zero_grad()`.
    # 9. Run `loss.backward()`.
    # 10. Update the policy with `optimizer.step()`.
    # 11. Save `sum(trajectory["rewards"])` in episode returns.
    # 12. Save `float(loss.item())` as `last_loss`.
    # 13. Build the summary dictionary.
    # 14. Add the note from `explain_policy_gradient_noise(summary)`.
    # 15. Return the summary.
    #
    # Checks:
    # - `len(episode_returns)` should equal `config.episodes`.
    # - `mean_return` should be a Python float.
    # - This training loop should update after each full episode, not after each single step.
    # - A useful practical check is whether later returns are often better than early returns.
    # - Do not expect a perfectly smooth curve because actions are sampled.

    env = TinyPolicyEnv(config)
    policy_model = PolicyNetwork(config).to(device)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.learning_rate)

    episode_returns = []
    last_loss = 0.0

    for episode_idx in range(config.episodes):
        trajectory = collect_episode(policy_model, env, config)

        returns = compute_discounted_returns(trajectory["rewards"], config.gamma)
        loss = reinforce_loss(trajectory["log_probs"], returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = sum(trajectory["rewards"])
        episode_returns.append(episode_return)
        last_loss = float(loss.item())

        print_every = max(1, config.episodes // 5)
        if (episode_idx + 1) % print_every == 0 or episode_idx == 0:
            print(
                f"Episode {episode_idx + 1}/{config.episodes}: "
                f"return={episode_return:.3f}, loss={last_loss:.3f}"
            )

    mean_return = float(sum(episode_returns) / len(episode_returns))
    first_window = episode_returns[: min(10, len(episode_returns))]
    last_window = episode_returns[-min(10, len(episode_returns)) :]

    summary: ReinforceSummary = {
        "episode_returns": episode_returns,
        "mean_return": mean_return,
        "last_loss": last_loss,
        "num_episodes": config.episodes,
        "note": "",
    }
    summary["note"] = explain_policy_gradient_noise(summary)

    print(f"Mean return over {config.episodes} episodes: {mean_return:.3f}")
    # These two lines compare early training to late training.
    # More exactly, they compare total episode reward, also called episode return.
    #
    # Example output:
    # ```text
    # Mean return in first 10 episodes: 0.450
    # Mean return in last 10 episodes: 0.959
    # ```
    #
    # That means:
    # - early training: average total reward per episode was about `0.45`
    # - late training: average total reward per episode was about `0.96`
    #
    # In this environment:
    # - goal reward is `+1.0`
    # - each non-goal step is `-0.01`
    #
    # So a return close to `1.0` means the agent reached the goal quickly.
    #
    # Example:
    # ```text
    # return = 0.97
    # ```
    #
    # This usually means the agent reached the goal after about 4 steps:
    # ```text
    # -0.01 - 0.01 - 0.01 + 1.0 = 0.97
    # ```
    #
    # So if the last 10 episodes have a bigger mean return than the first 10,
    # that is a useful sign that the policy learned better actions.
    print(f"Mean return in first {len(first_window)} episodes: {sum(first_window) / len(first_window):.3f}")
    print(f"Mean return in last {len(last_window)} episodes: {sum(last_window) / len(last_window):.3f}")
    print(f"Last loss: {last_loss:.3f}")
    print(f"Training note: {summary['note']}")

    return summary
    #raise NotImplementedError("Collect episodes, update the policy, and return training summaries.")

def explain_policy_gradient_noise(summary: ReinforceSummary) -> str:
    """Write a short note about why REINFORCE can be noisy."""
    # Goal:
    # Explain why REINFORCE training can jump around from episode to episode.
    #
    # Input:
    # - `summary`: ReinforceSummary
    # - Useful keys:
    #   - `summary["mean_return"]`
    #   - `summary["num_episodes"]`
    #   - `summary["last_loss"]`
    #
    # Output:
    # - `str`
    # - One or two plain sentences.
    #
    # Meaning:
    # - REINFORCE learns from sampled episodes.
    # - One lucky or unlucky episode can push the policy more than it deserves.
    # - Later actor-critic methods add a baseline to reduce this noise.
    #
    # Plain comparison:
    # DQN updates from replay batches and a TD target.
    # REINFORCE updates from full sampled episode returns.
    # Because one episode can be lucky or unlucky, the return curve can jump around.
    #
    # Steps:
    # 1. Mention that the policy samples actions.
    # 2. Mention that each update uses one whole episode's returns.
    # 3. Use one concrete number from the summary if useful.
    #
    # Checks:
    # - Keep the note short and plain.
    # - Do not say the method is broken just because the curve is noisy.
    # - Do not mention baselines as if this exercise already implemented one.
    return (
        f"REINFORCE sampled actions for {summary['num_episodes']} episodes, "
        f"then updated from each whole episode's returns. "
        f"The mean return was {summary['mean_return']:.3f}, but one lucky or unlucky episode can still move the policy a lot."
    )
    #raise NotImplementedError("Explain REINFORCE noise in simple words.")

def check_metrics(metrics: ReinforceSummary) -> bool:
    """Return True when the REINFORCE summary has the expected keys."""
    required = {"episode_returns", "mean_return", "last_loss", "num_episodes", "note"}
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
    """Run the REINFORCE scaffold."""
    config = ReinforceConfig()
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
    _, ok = report_task_progress("describe_reinforce_plan", describe_reinforce_plan, config)
    if not count_or_stop(ok):
        return
    policy, ok = report_task_progress("PolicyNetwork.__init__", PolicyNetwork, config)
    if not count_or_stop(ok):
        return
    sample_x = torch.randn(config.max_steps_per_episode, config.state_dim)
    _, ok = report_task_progress("PolicyNetwork.forward", policy, sample_x)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("train_reinforce", train_reinforce, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "REINFORCE summary", float(int(passed)), "reinforce keys present")
if __name__ == "__main__":
    main()
