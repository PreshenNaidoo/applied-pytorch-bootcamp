"""
============================================================
EXERCISE 06: DEEP Q NETWORK DQN BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny Deep Q-Network learner.

Plain-language meanings:
- DQN means using a neural network instead of a Q-table
- replay buffer stores old experience so you can train on mini-batches
- target network is a slower copy used for more stable targets
- TD means temporal difference
- temporal difference means "learn from the difference between what you guessed before
  and what one new step says now"
- TD target means the new value you want the network to move toward
- TD loss means the gap between the Q-value you predicted and that TD target

WHY THIS EXERCISE EXISTS
Tabular Q-learning works for tiny state spaces.
When the state is too large for a table,
you need a function approximator such as a neural network.
This exercise teaches:
- neural-network Q-function
- replay buffer
- target network
- TD loss
- epsilon-greedy data collection
- simple DQN training loop

HOW THIS FITS INTO THE RL SEQUENCE
This file comes after tabular Q-learning.
The main new idea is:
- keep the Q-learning logic
- replace the table with a neural network
That is the bridge from tiny RL problems to larger state spaces.

Beginner intuition:
- a Q-table gives one box for every state-action pair
- that breaks when there are too many states
- DQN replaces that huge table with a neural network that tries to predict the Q-values

Small example:
- input state: a short vector
- network output: one Q-value per action
- the largest output is the action the network currently prefers

Tiny TD example:
- the agent chose action `right`
- the environment gave reward `1.0`
- the target network says the best next action is worth `4.0`
- gamma is `0.9`, so future reward is slightly discounted
- TD target = `1.0 + 0.9 * 4.0 = 4.6`
- if the online network predicted `3.0` for `right`, the TD loss pushes it closer to `4.6`
- if the step ended the episode, there is no next value, so TD target = reward only

IMPORTANT SCOPE NOTE
This is a teaching version of DQN.
Keep the environment tiny and local.
The goal is to understand the moving parts,
not to match Atari results.

DATASET
No dataset file is needed.
Collect transitions from a tiny local LineWorld environment.

Clear default:
- environment name: `LineWorld`
- states: 5 positions in a line, numbered `0` to `4`
- start state: position `0`
- goal state: position `4`
- actions: `0 = left`, `1 = right`
- reward: `+1` at the goal, small negative reward such as `-0.01` otherwise
- state tensor: one-hot vector with shape `[5]`

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- build the tiny `LineWorld` environment in this file
- collect transitions online while the agent acts
- store those transitions in replay
- train from replay mini-batches instead of only the newest step

INPUTS AND OUTPUTS
Input:
- state tensor
- action
- reward
- next-state tensor
- done flag
Network output:
- one Q-value per action
Training output:
- episode returns
- replay size
- mean loss
- target update count
- short plain-language note

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a tiny MLP:
- state vector in
- one or two hidden layers
- one Q-value per action out

LOSS FUNCTION
Use a simple TD regression loss such as mean squared error between:
- predicted Q for the chosen action
- TD target from reward and next-state value

METRICS TO TRACK
Track at least:
- episode returns
- mean loss
- replay size
- target update count

TARGET RESULT RANGE
Focus on understanding the moving parts, not on large rewards.

Useful signs:
- replay starts filling up
- the online network produces one value per action
- target-network copies happen at the expected interval
- loss can be computed from replay batches without shape errors

COMMON MISTAKES
- training before replay has enough data
- using the online network instead of the target network for TD targets
- taking the max over the wrong dimension
- forgetting to gather the Q-value for the chosen action
- expecting a tiny DQN to look stable on every short run

What each step is doing in simple words:
1. create a tiny world with vector states
2. store old experience so learning is not based only on the newest step
3. build the online Q-network that will be trained
4. build the slower target network used for more stable targets
5. collect experience with exploration
6. sample mixed old experience from replay
7. build the TD target the network should move toward
8. update the online network from that target
9. sometimes refresh the target network
10. return a summary that shows each DQN part actually ran

STEP-BY-STEP TODO
1. Build a tiny vector-state environment.
2. Build the replay buffer.
3. Build the Q-network.
4. Build the target network.
5. Collect transitions with epsilon-greedy.
6. Sample replay batches.
7. Compute TD targets.
8. Train the online Q-network.
9. Periodically copy weights into the target network.
10. Return reward and loss summaries.

DEFINITION OF DONE
You are done when:
- transitions are stored in replay
- the network returns one Q-value per action
- TD targets are computed
- target network updates happen
- you can explain why replay and target networks help
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TypeAlias, TypedDict
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
class DQNConfig:
    """Settings for the DQN exercise."""
    state_dim: int = 5
    num_actions: int = 2
    hidden_dim: int = 64
    replay_capacity: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon: float = 0.2
    episodes: int = 80
    max_steps_per_episode: int = 20
    target_update_every: int = 10
    random_state: int = 42


Transition: TypeAlias = tuple[torch.Tensor, int, float, torch.Tensor, bool]


class DQNPlanSummary(TypedDict):
    """Main DQN pieces used in this exercise."""

    state_dim: int
    num_actions: int
    main_parts: list[str]


class ReplayBatch(TypedDict):
    """One mini-batch sampled from replay."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class DQNBatchSummary(TypedDict):
    """Metrics from one DQN optimization step."""

    loss: float
    mean_predicted_q: float


class DQNSummary(TypedDict):
    """Final summary returned by the tiny DQN workflow."""

    episode_returns: list[float]
    mean_loss: float
    replay_size: int
    target_updates: int
    note: str


def describe_dqn_plan(config: DQNConfig) -> DQNPlanSummary:
    """Return the main DQN pieces used in this exercise."""
    return {
        "state_dim": config.state_dim,
        "num_actions": config.num_actions,
        "main_parts": ["online_network", "target_network", "replay_buffer", "epsilon_greedy"],
    }


# Beginner hints for this file:
# - DQN predicts one Q-value per action from the current state
# - the replay buffer breaks the strong correlation between back-to-back steps
# - the target network makes the learning target change more slowly
# - start with a tiny local environment and small batches
# - if the pieces feel abstract, print one transition and one TD target by hand
class TinyLineWorld:
    """Tiny local environment used for DQN practice."""
    def __init__(self, config: DQNConfig) -> None:
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

        raise NotImplementedError("Return the starting state vector.")
        

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
        # Use reward `1.0` at the goal and a small step reward such as `-0.01` otherwise.
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

        raise NotImplementedError("Return the next state, reward, and done flag.")
        





class ReplayBuffer:
    """Simple replay buffer for DQN."""
    def __init__(self, capacity: int) -> None:
        # Goal:
        # Initialize replay storage with a maximum capacity.
        #
        # Input:
        # - `capacity`: int
        # - Maximum number of transitions to keep.
        #
        # Output:
        # - `None`
        # - This method should create attributes on `self`.
        #
        # Method:
        # Store:
        # - `self.capacity = capacity`
        # - `self.storage: list[Transition] = []`
        #
        # Steps:
        # 1. Save the capacity.
        # 2. Create an empty list for transitions.
        #
        # Checks:
        # - `len(buffer)` should be `0` right after initialization.

        raise NotImplementedError("Initialize replay storage.")


    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store one transition."""
        # Goal:
        # Add one transition to replay.
        # If replay is full, remove the oldest transition first.
        #
        # Input:
        # - `state`: torch.Tensor with shape `[state_dim]`
        # - `action`: int, chosen action
        # - `reward`: float, reward from the step
        # - `next_state`: torch.Tensor with shape `[state_dim]`
        # - `done`: bool, whether the episode ended
        #
        # Output:
        # - `None`
        #
        # Method:
        # A transition has this shape:
        # `(state, action, reward, next_state, done)`
        #
        # To remove the oldest item from a list, use:
        # `self.storage.pop(0)`
        #
        # To add the new item, use:
        # `self.storage.append((state, action, reward, next_state, done))`
        #
        # Steps:
        # 1. If storage is already at capacity, remove the oldest transition.
        # 2. Add the new transition.
        #
        # Checks:
        # - `len(self.storage)` should never be greater than `self.capacity`.
        # - This method should not train the network.

        raise NotImplementedError("Add one transition to replay.")
        


    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample one replay batch."""
        # Goal:
        # Return one random mini-batch from replay.
        #
        # Input:
        # - `batch_size`: int
        #
        # Output:
        # Return this exact dictionary:
        # - `"states"`: float tensor with shape `[batch_size, state_dim]`
        # - `"actions"`: long tensor with shape `[batch_size]`
        # - `"rewards"`: float tensor with shape `[batch_size]`
        # - `"next_states"`: float tensor with shape `[batch_size, state_dim]`
        # - `"dones"`: float tensor with shape `[batch_size]`, where done is `1.0` and not done is `0.0`
        #
        # Method:
        # Use:
        # - `random.sample(self.storage, batch_size)`
        # - `torch.stack(states)`
        # - `torch.tensor(actions, dtype=torch.long)`
        # - `torch.tensor(rewards, dtype=torch.float32)`
        # - `torch.tensor(dones, dtype=torch.float32)`
        #
        # Steps:
        # 1. Sample `batch_size` transitions.
        # 2. Split them into states, actions, rewards, next_states, and dones.
        # 3. Stack or convert each field into one tensor.
        # 4. Return the dictionary.
        #
        # Checks:
        # - Do not call this until `len(self.storage) >= batch_size`.
        # - The returned keys should match the list above exactly.

        raise NotImplementedError("Return one replay batch of states, actions, rewards, next_states, and dones.")



    def __len__(self) -> int:
        """Return the number of stored transitions."""
        # Goal:
        # Return how many transitions are currently stored.
        #
        # Input:
        # - No direct arguments.
        #
        # Output:
        # - `int`
        #
        # Method:
        # Return:
        # `len(self.storage)`
        #
        # Steps:
        # 1. Count the stored transitions.
        # 2. Return that count.
        #
        # Checks:
        # - The result should be between `0` and `self.capacity`.

        raise NotImplementedError("Return replay size.")


class DQNNetwork(nn.Module):
    """Tiny Q-network."""
    def __init__(self, config: DQNConfig) -> None:
        # Goal:
        # Build a tiny MLP that maps a state vector to one Q-value per action.
        #
        # Input:
        # - `config`: DQNConfig
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
        # - `config.num_actions` is the number of Q-values to output.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Build `self.net`.
        # 3. Do not return anything.
        #
        # Checks:
        # - For input shape `[batch_size, state_dim]`, output shape should be `[batch_size, num_actions]`.
        super().__init__()

        raise NotImplementedError("Build a tiny MLP that returns one Q-value per action.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values with shape `[batch_size, num_actions]`."""
        # Goal:
        # Return one Q-value per action for each state in the batch.
        #
        # Input:
        # - `x`: torch.Tensor
        # - Shape: `[batch_size, state_dim]`
        # - Example: a batch of 32 states has shape `[32, 5]`.
        #
        # Output:
        # - `torch.Tensor`
        # - Shape: `[batch_size, num_actions]`
        # - Example: `[32, 2]` for left and right actions.
        #
        # Method:
        # Return:
        # `self.net(x)`
        #
        # Steps:
        # 1. Pass `x` through the network.
        # 2. Return the Q-values.
        #
        # Checks:
        # - Do not apply softmax. Q-values are scores, not probabilities.
        # - The second dimension should equal `num_actions`.

        raise NotImplementedError("Return one Q-value per action.")

@torch.no_grad()
def choose_action(q_network: nn.Module, state: torch.Tensor, epsilon: float) -> int:
    """Choose one action with epsilon-greedy."""
    # Goal:
    # Return one action using epsilon-greedy and the network outputs.
    #
    # Input:
    # - `q_network`: nn.Module
    # - `state`: torch.Tensor
    #   Shape: `[state_dim]`
    # - `epsilon`: float
    #
    # Output:
    # - `int`
    # - Valid actions are `0` and `1` in this exercise.
    #
    # Method:
    # Use:
    # - `random.random()` to decide whether to explore
    # - `random.randrange(num_actions)` for a random action
    # - `q_network(state.unsqueeze(0))` to add a batch dimension
    # - `torch.argmax(q_values, dim=1).item()` for the greedy action
    #
    # Wrap the network call in:
    # `with torch.no_grad():`
    # because action choice should not build a training graph.
    #
    # Steps:
    # 1. If a random number is below epsilon, return a random action.
    # 2. Otherwise, run the network on `state.unsqueeze(0)`.
    # 3. Choose the action with the largest Q-value.
    # 4. Return it as a Python int.
    #
    # Checks:
    # - With `epsilon=0.0`, this should always use the network.
    # - With `epsilon=1.0`, this should always explore.

    raise NotImplementedError("Return one action using epsilon-greedy and the network outputs.")

    #else Exploit:


# TD means temporal difference.
# In this file, the "difference" is:
# - what the online network predicted for the chosen action
# - what one new replay step says that value should be
#
# Build the TD target:
# - reward + gamma * max target-network Q(next_state) for non-terminal steps
# - reward only for terminal steps
def compute_td_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_states: torch.Tensor,
    target_network: nn.Module,
    gamma: float,
) -> torch.Tensor:
    """Return the DQN TD targets."""
    # Goal:
    # Return one TD target per replay item.
    # A TD target is the value you want the chosen-action Q-value to move toward.
    # It uses the real reward from the replay batch plus the target network's guess
    # about the best value available from the next state.
    #
    # Input:
    # - `rewards`: torch.Tensor
    #   Shape: `[batch_size]`
    # - `dones`: torch.Tensor
    #   Shape: `[batch_size]`, where done is `1.0` and not done is `0.0`
    # - `next_states`: torch.Tensor
    #   Shape: `[batch_size, state_dim]`
    # - `target_network`: nn.Module
    # - `gamma`: float
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[batch_size]`
    #
    # Formula:
    # TD target = reward + gamma * max_next_q * (1 - done)
    #
    # Plain meaning:
    # - `reward` is what the environment gave for this one step.
    # - `gamma` says how much future reward matters.
    # - `max_next_q` is the best next-state Q-value from the target network.
    # - `done` is `1.0` when the episode ended, so the next-state part becomes zero.
    #
    # Small example:
    # - reward = 1.0
    # - gamma = 0.9
    # - max_next_q = 4.0
    # - done = 0.0
    # - TD target = 1.0 + 0.9 * 4.0 * (1.0 - 0.0) = 4.6
    #
    # Terminal example:
    # - reward = 1.0
    # - done = 1.0
    # - TD target = 1.0 because there is no next state to learn from
    # ```text
    # target = reward + gamma * max_next_q * (1 - done)
    # ```
    #
    # Meaning:
    # - `max_next_q` is the best target-network Q-value for the next state.
    # - `(1 - done)` removes future value when the episode is over.
    #
    # Method:
    # Use the target network, not the online network:
    # - `next_q_values = target_network(next_states)`
    # - `max_next_q = next_q_values.max(dim=1).values`
    #
    # Wrap target computation in:
    # `with torch.no_grad():`
    #
    # Steps:
    # 1. Run `next_states` through the target network.
    # 2. Take the max Q-value across the action dimension.
    # 3. Apply the formula above.
    # 4. Return the targets.
    #
    # Checks:
    # - If `done == 1.0`, target should equal reward.
    # - Do not let gradients flow into the target network.

    raise NotImplementedError("Return the TD targets for the replay batch.")

    # This is the batched version of:
    # if done: 
    #   target = reward
    # else: 
    #   target = reward + gamma * best_next_q
    # When done is 1.0, (1.0 - dones) is 0.0, so the future part disappears.


def hard_update_target_network(online_network: nn.Module, target_network: nn.Module) -> None:
    """Copy online weights into the target network."""
    # Goal:
    # Copy all parameters from the online network into the target network.
    #
    # Input:
    # - `online_network`: nn.Module
    # - `target_network`: nn.Module
    #
    # Output:
    # - `None`
    #
    # Method:
    # Use the PyTorch module method:
    # `target_network.load_state_dict(online_network.state_dict())`
    #
    # Steps:
    # 1. Read the online network weights with `.state_dict()`.
    # 2. Load them into the target network with `.load_state_dict(...)`.
    # 3. Return nothing.
    #
    # Checks:
    # - After copying, matching parameters should be equal.
    # - This function should not call `optimizer.step()`.

    raise NotImplementedError("Copy parameters from the online network into the target network.")


# Train the online network on one replay batch.
# Return at least:
# - loss
# - mean predicted Q
def train_one_dqn_batch(
    online_network: nn.Module,
    target_network: nn.Module,
    replay_batch: ReplayBatch,
    optimizer: torch.optim.Optimizer,
    gamma: float,
) -> DQNBatchSummary:
    """Run one DQN optimization step."""
    # Goal:
    # Train the online network on one replay batch and return simple metrics.
    #
    # Input:
    # - `online_network`: nn.Module
    # - `target_network`: nn.Module
    # - `replay_batch`: ReplayBatch
    # - `optimizer`: torch.optim.Optimizer
    # - `gamma`: float
    #
    # Output:
    # Return this exact dictionary:
    # - `"loss"`: float
    # - `"mean_predicted_q"`: float
    #
    # Important tensor shapes:
    # - `states`: `[batch_size, state_dim]`
    # - `actions`: `[batch_size]`
    # - `q_values = online_network(states)`: `[batch_size, num_actions]`
    # - chosen action Q-values: `[batch_size]`
    # - TD targets: `[batch_size]`
    #
    # Method:
    # Use:
    # - `online_network.train()`
    # - `q_values.gather(1, actions.unsqueeze(1)).squeeze(1)`
    # - `compute_td_targets(...)`
    # - `nn.functional.mse_loss(predicted_q, targets)`
    # - `optimizer.zero_grad()`
    # - `loss.backward()`
    # - `optimizer.step()`
    #
    # Steps:
    # 1. Read `states`, `actions`, `rewards`, `next_states`, and `dones` from the batch.
    # 2. Run the online network on `states`.
    # 3. Gather the Q-value for each chosen action.
    # 4. Compute TD targets with the target network.
    # 5. Compute MSE loss between predicted Q-values and targets.
    # 6. Clear gradients, backpropagate, and step the optimizer.
    # 7. Return loss and mean predicted Q as Python floats.
    #
    # Checks:
    # - `actions` must have dtype `torch.long` for `.gather(...)`.
    # - TD targets should not require gradients.

    raise NotImplementedError("Return one DQN batch update summary.")








def run_dqn_training(config: DQNConfig) -> DQNSummary:
    """Run the full tiny DQN workflow."""
    # Goal:
    # Collect replay, train the online network, update the target network, and return a summary.
    #
    # Input:
    # - `config`: DQNConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"episode_returns"`: list[float], one total reward per episode
    # - `"mean_loss"`: float, average of batch losses that were actually run
    # - `"replay_size"`: int, final number of stored transitions
    # - `"target_updates"`: int, number of hard target-network copies
    # - `"note"`: str, short explanation from `explain_replay_and_target_network(...)`
    #
    # Method:
    # Use these objects:
    # - `env = TinyLineWorld(config)`
    # - `replay = ReplayBuffer(config.replay_capacity)`
    # - `online_network = DQNNetwork(config).to(device)`
    # - `target_network = DQNNetwork(config).to(device)`
    # - `optimizer = torch.optim.Adam(online_network.parameters(), lr=config.learning_rate)`
    #
    # Copy weights into the target network before training:
    # `hard_update_target_network(online_network, target_network)`
    #
    # Only train when:
    # `len(replay) >= config.batch_size`
    #
    # Steps:
    # 1. Build environment, replay buffer, online network, target network, and optimizer.
    # 2. Copy online weights into the target network.
    # 3. Create empty lists for episode returns and losses.
    # 4. Loop over episodes.
    # 5. Reset the environment and start episode return at `0.0`.
    # 6. Loop up to `config.max_steps_per_episode`.
    # 7. Choose an action with epsilon-greedy.
    # 8. Step the environment.
    # 9. Store the transition in replay.
    # 10. If replay has enough data, sample a batch and train one DQN batch.
    # 11. Every `config.target_update_every` training steps, copy online weights into target.
    # 12. Stop the episode early if `done` is true.
    # 13. Return the summary dictionary.
    #
    # Checks:
    # - `len(episode_returns)` should equal `config.episodes`.
    # - `replay_size` should be no larger than `config.replay_capacity`.
    # - `mean_loss` should handle the case where no training batch ran yet.

    raise NotImplementedError("Collect replay, train the online network, update the target network, and return a summary.")








    # Test whether the trained agent can reach the goal without exploration.

    # In this line world, the shortest path needs 4 moves to reach the goal.
    # The first 3 moves are non-goal moves, so each gives -0.1.
    # The 4th move reaches the goal, so it gives +1.0.
    # Total return: -0.1 + -0.1 + -0.1 + 1.0 = 0.7.
    # So a good greedy test should usually reach the goal with return near 0.7.


def explain_replay_and_target_network(summary: DQNSummary) -> str:
    """Write a short note about replay and target networks."""
    # Goal:
    # Explain why replay and target networks help DQN.
    #
    # Input:
    # - `summary`: DQNSummary
    # - Useful keys:
    #   - `summary["replay_size"]`
    #   - `summary["target_updates"]`
    #   - `summary["mean_loss"]`
    #
    # Output:
    # - `str`
    # - One or two plain sentences.
    #
    # Meaning:
    # - Replay helps because the network trains on a mix of old steps, not only the newest step.
    # - The target network helps because the target changes more slowly than the online network.
    #
    # Steps:
    # 1. Mention replay in plain words.
    # 2. Mention the target network in plain words.
    # 3. Use one concrete number from the summary if useful.
    #
    # Checks:
    # - Keep the note short.
    # - Do not say replay or target networks are required for tabular Q-learning.
    raise NotImplementedError("Explain why replay and target networks help DQN.")

def check_metrics(metrics: DQNSummary) -> bool:
    """Return True when the DQN summary has the expected keys."""
    required = {"episode_returns", "mean_loss", "replay_size", "target_updates", "note"}
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
    """Run the DQN scaffold."""
    config = DQNConfig()
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
    _, ok = report_task_progress("describe_dqn_plan", describe_dqn_plan, config)
    if not count_or_stop(ok):
        return
    model, ok = report_task_progress("DQNNetwork.__init__", DQNNetwork, config)
    if not count_or_stop(ok):
        return
    sample_x = torch.randn(config.batch_size, config.state_dim)
    _, ok = report_task_progress("DQNNetwork.forward", model, sample_x)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_dqn_training", run_dqn_training, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "DQN summary", float(int(passed)), "dqn keys present")


if __name__ == "__main__":
    main()
