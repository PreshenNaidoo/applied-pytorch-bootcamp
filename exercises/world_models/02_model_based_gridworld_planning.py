"""
============================================================
EXERCISE 02: MODEL BASED GRIDWORLD PLANNING
============================================================

WHAT YOU ARE BUILDING
You are building a small model-based planning agent.

Plain meaning:
- a dynamics model predicts what may happen next
- a planner tries possible future action sequences inside that model
- the agent takes the first action from the best imagined sequence
- the real environment checks whether that imagined plan worked

The core idea is:

```text
current state -> try action sequences in learned model -> choose action -> test in real world
```

WHY THIS IS MODEL-BASED RL

In RL, an environment model predicts what may happen after an action.
It can predict the next state, reward, safety cost, and whether the episode may
end.

This exercise is model-based because action choice uses an explicit learned
environment model:

```text
1. collect real grid-world transitions
2. train the dynamics model to predict the next step
3. freeze the trained dynamics model
4. use it to imagine several future action sequences
5. choose the first action from the sequence with the best predicted result
6. test that action with the true grid-world rules
```

The dynamics model is the part that makes this model-based. The planner asks
it questions like:

```text
If I move down and then right, where do you predict I will go?
What reward and safety cost do you predict?
```

A model-free method such as DQN or PPO would not use an explicit environment
model to predict these future transitions for planning. It would learn action
values or a policy directly from experience.

The shortest-path baseline is different from the learned planner. It uses the
known grid map and hand-written path rules. It does not use the learned
dynamics model. It is included so the learned planner can be compared with a
strong route finder that already knows the map.

WHY THIS EXERCISE EXISTS
the related exercise trains the first state-based world model.
This exercise uses the same idea for planning.

World models are useful because they let an agent ask:

```text
What do I think will happen if I do this?
```

before it spends a real environment step.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
robot, game, or simulator           local visual grid-world
camera and sensor state             four-channel complete grid state
motor or keyboard action            action id 0, 1, 2, or 3
learned world model                 small PyTorch dynamics model
planning/search                     try many short action sequences
real-world test                     roll out in common.embodied_gridworld
planning report                     success, return, cost, and path plots

WHY THIS IS NOT RANDOM SYNTHETIC DATA
The training data comes from the shared bootcamp environment in
`common.embodied_gridworld`.

Each row is a real logged transition from that environment:

```text
state, action -> next_state, reward, cost, done
```

The evaluation also uses real grid-world scenes from the same environment.
The model is allowed to imagine futures, but the final sequence result comes from
real environment steps.

WHAT IS PROVIDED BY THE ENVIRONMENT
`common.embodied_gridworld` provides:
- a grid with walls, hazards, a goal, and an agent
- real step rules
- a scripted policy for collecting logged transitions
- complete four-channel grid states
- rendered images

WHAT THE MODEL KNOWS
For training, the model sees:
- complete grid state shaped `[4, 7, 7]`
- action id `0`, `1`, `2`, or `3`

For planning, the model sees:
- the complete current grid state
- a candidate action

WHAT THE MODEL PREDICTS
The model predicts:
- next-position logits shaped `[49]`, one possible result for each grid cell
- reward shaped `[1]`
- cost shaped `[1]`
- done logit shaped `[1]`

WHAT THE PLANNER DOES
The planner does not learn weights.
It uses the trained model.

For each real state:

```text
1. build many candidate action sequences
2. roll each sequence forward inside the learned model
3. calculate a predicted sequence score for each imagined sequence
4. choose the first action from the best sequence
5. take that action in the real environment
6. repeat
```

WHERE EXPLORATION AND EXPLOITATION HAPPEN

These words describe why an action is chosen:

- Explore: try an action to collect experience that may improve what the
  system knows. The action does not have to look best right now.
- Exploit: use what the system currently knows and choose the action that
  looks best right now.

In this exercise, exploration happens while collecting the training data.
The shared scripted policy takes a random action 15% of the time. Those random
actions help the dataset include transitions away from the shortest safe path.

The other 85% of data-collection actions exploit the map by following the
current shortest safe path when one is available.

After training, the planner exploits its learned dynamics model. It scores all
candidate action sequences and chooses the first action from the sequence with
the highest predicted score. The model is frozen during this evaluation, so
the planner is not exploring to learn more.

Trying many sequences inside the frozen model is planning search. It is not
exploration in the learning sense because no real experience is collected and
no model weights are updated during that search.

DATASET
Default dataset:
- no download
- use `common.embodied_gridworld`
- call `collect_visual_gridworld_episodes(...)`
- call `flatten_logged_episodes(...)`

One logged row contains both the older small state and the complete state used
by this exercise:

```text
state: [6]
full_state: [4, 7, 7]
action: int
reward: float
cost: float
next_state: [6]
next_full_state: [4, 7, 7]
done: bool
```

Complete state meaning:

```text
channel  meaning
----------------------------------------
0        current position
1        goal position
2        every hazard position
3        every wall position
```

Tiny concrete example:

```text
full_state[0, 0, 0] = 1.0 means the current position is `(0, 0)`
full_state[3, 2, 3] = 1.0 means there is a wall at `(2, 3)`
action = 1
model predicts the next position, reward, safety cost, and episode end
planner uses those predictions to calculate a predicted sequence score
real environment checks what actually happened
```

INPUTS AND OUTPUTS
One model input:
- full state flattened from `[4, 7, 7]` to `[196]`
- action one-hot: `[num_actions]`
- model input: `[200]`

One model output:
- next_position_logits: `[49]`
- reward: `[1]`
- cost: `[1]`
- done_logit: `[1]`

One planner output:
- one action id

LOSS FUNCTION
Use the same prediction losses as a small world model:

```text
next_position_loss = cross_entropy(predicted_next_position_logits, true_next_position_index)
reward_loss = mse(predicted_reward, true_reward)
cost_loss = mse(predicted_cost, true_cost)
done_loss = binary_cross_entropy_with_logits(done_logit, done_target)

total_loss = next_position_loss + reward_loss + cost_loss + done_loss
```

PLANNING SCORE
Use predicted reward, predicted cost, and an early stop for done:

```text
predicted_sequence_score = sum(predicted_reward - cost_weight * max(predicted_cost, 0))
```

If the model predicts `done=True`, stop imagining that sequence early.
Clamping cost at zero stops a negative regression mistake from becoming a fake
safety bonus.

FAIR PREDICTED-VS-REAL COMPARISON

Both sequence-scoring paths now start with the same complete layout:

```text
current position
goal position
every hazard position
every wall position
```

`predict_action_sequence_score(...)` uses the learned dynamics model.
`calculate_real_action_sequence_score(...)` uses the true grid-world rules.
Any gap can now come from model prediction error, not from hiding walls or the
second hazard from the learned model.

METRICS TO TRACK
Training metrics:
- validation next-position accuracy: higher is better
- validation reward MSE: lower is better
- validation cost MSE: lower is better
- validation done accuracy: higher is better

Planning metrics:
- planned success rate: higher is better
- planned mean return: higher is better
- planned mean cost: lower is better
- planner mean episode steps: lower is better when success is high
- shortest-path baseline success rate: compare against a hand-written route finder
- planner reward prediction gap: lower means imagined rewards matched reality
- planner cost prediction gap: lower means imagined safety estimates matched reality
- selected-sequence score gap: model score versus real score for the sequence
  the planner ranked highest
- random-sequence score gap: the same check for a random sequence
- selected-vs-random model error gap: selected gap minus random gap; values above zero
  suggest search is taking advantage of model errors, not only finding good
  plans. Here, "exploitation" means exploiting a model mistake. It is related
  to, but more specific than, normal explore-versus-exploit action choice.

VISUALS TO CREATE
Save these files:
- `outputs/gridworld_planning/training_curves.png`
- `outputs/gridworld_planning/planning_rollout.png`
- `outputs/gridworld_planning/shortest_path_baseline_vs_planner.png`

COMMON MISTAKES
- using random tensors instead of logged environment transitions
- treating the learned model as the real environment
- reporting only model loss and not planning behavior
- forgetting that the planner chooses only the first action
- applying sigmoid before `binary_cross_entropy_with_logits`
- using validation transitions for training
- trusting imagined reward or cost without comparing it with the real rollout
- checking only one-step predictions when the planner searches multi-step sequences
- making a plot that does not show what the planner actually did

STEP-BY-STEP TODO
1. Collect complete logged episodes from `common.embodied_gridworld`.
2. Split episodes, then build train and validation tensors.
3. Train a small dynamics model.
4. Evaluate one-step prediction quality.
5. Build candidate action sequences.
6. Score imagined rollouts inside the learned model.
7. Choose actions with the planner.
8. Roll out the planner in real grid-world scenes.
9. Compare planner behavior with the shortest-path baseline.
10. Save metrics and visuals.

DEFINITION OF DONE
You are done when:
- data comes from real logged grid-world transitions
- no episode appears in both training and validation
- the dynamics model trains after you complete the stubs
- validation metrics are printed
- planning is evaluated in real grid-world scenes
- predicted and real planner return and cost are compared
- overoptimistic planner episodes are reported
- selected imagined sequences are replayed in the real environment and compared
  with random sequences to test whether search finds model errors
- the planner is compared with the shortest-path baseline
- all three visual files are saved
- every completed stub keeps its original `raise NotImplementedError(...)` line commented out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.embodied_gridworld import (
    ACTION_DELTAS,
    LoggedGridStep,
    VisualGridWorldConfig,
    collect_visual_gridworld_episodes,
    encode_full_grid_state,
    flatten_logged_episodes,
    make_grid_scene,
    render_grid_image,
    shortest_path_action,
    step_grid_world,
)
from common.reproducibility import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"


# Learning order for this exercise:
# 1. Finish the dataset helpers first.
# 2. Build and train the dynamics model.
# 3. Finish one-step model prediction.
# 4. Finish sequence scoring and action choice.
# 5. Finish real-environment rollout evaluation.
# 6. Finish the three plots.


Position = tuple[int, int]

# Keep these two ideas separate:
# - the planning agent is the whole system that chooses actions
# - a Position is only one `(row, column)` location in the grid
#
# GridScene stores the fixed layout at the start of an episode. During a
# rollout, `current_position` starts at `scene.start_position` and then changes
# after each action. The other scene fields stay fixed.


@dataclass
class GridScene:
    """One starting grid-world layout used for evaluation."""

    start_position: Position
    goal_position: Position
    hazard_positions: set[Position]
    wall_positions: set[Position]


class PlanningDatasetSplits(TypedDict):
    """Train and validation tensors for the planning dynamics model."""

    train_inputs: torch.Tensor
    train_next_position_indices: torch.Tensor
    train_rewards: torch.Tensor
    train_costs: torch.Tensor
    train_dones: torch.Tensor
    val_inputs: torch.Tensor
    val_next_position_indices: torch.Tensor
    val_rewards: torch.Tensor
    val_costs: torch.Tensor
    val_dones: torch.Tensor


class OneStepMetrics(TypedDict):
    """Validation metrics for the learned dynamics model."""

    val_next_position_accuracy: float
    val_reward_mse: float
    val_cost_mse: float
    val_done_accuracy: float


class PlanningMetrics(TypedDict):
    """Final training, planning, and visual results."""

    final_train_loss: float
    val_next_position_accuracy: float
    val_reward_mse: float
    val_cost_mse: float
    val_done_accuracy: float
    planner_success_rate: float
    planner_mean_return: float
    planner_mean_cost: float
    planner_mean_episode_steps: float
    shortest_path_baseline_success_rate: float
    shortest_path_baseline_mean_return: float
    shortest_path_baseline_mean_cost: float
    shortest_path_baseline_mean_episode_steps: float
    planner_reward_prediction_gap: float
    planner_cost_prediction_gap: float
    planner_overoptimistic_episode_rate: float
    selected_sequence_score_gap: float
    random_sequence_score_gap: float
    selected_vs_random_model_error_gap: float
    selected_sequence_overestimate_rate: float
    training_curve_path: str
    rollout_plot_path: str
    comparison_plot_path: str


class RolloutResult(TypedDict):
    """One real environment rollout summary."""

    visited_positions: list[Position]
    actions: list[int]
    real_return: float
    real_cost: float
    predicted_return: float
    predicted_cost: float
    reached_goal: bool
    steps: int


@dataclass
class GridworldPlanningConfig:
    """Settings for model-based grid-world planning."""

    state_channels: int = 4
    num_actions: int = 4
    grid_size: int = 7
    image_size: int = 28
    num_episodes: int = 10000#320
    max_steps: int = 18
    val_fraction: float = 0.2
    batch_size: int = 64
    hidden_dim: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    planning_horizon: int = 4
    cost_weight: float = 8.0 # 1.0
    num_eval_scenes: int = 24
    random_seed: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "gridworld_planning"


def make_visual_grid_config(config: GridworldPlanningConfig) -> VisualGridWorldConfig:
    """Create the shared grid-world config used for data and evaluation."""
    # This helper is provided so every stub uses the same environment settings.
    return VisualGridWorldConfig(
        grid_size=config.grid_size,
        image_size=config.image_size,
        num_episodes=config.num_episodes,
        max_steps=config.max_steps,
        random_state=config.random_seed,
    )


def encode_full_state_action(
    full_state: torch.Tensor,
    action: int,
    config: GridworldPlanningConfig,
) -> torch.Tensor:
    """Flatten one complete grid state and append a one-hot action."""
    # Where this fits:
    # This creates one input row/sample for the dynamics model.
    #
    # Goal:
    # Flatten the complete grid and concatenate a one-hot action vector.
    #
    # Input:
    # - `full_state`: tensor shaped `[state_channels, grid_size, grid_size]`.
    # - `action`: integer from `0` to `num_actions - 1`.
    #
    # Output:
    # - tensor shaped `[state_channels * grid_size * grid_size + num_actions]`.
    # - with defaults, shape is `[4 * 7 * 7 + 4] = [200]`.
    #
    # Method:
    # 1. Check the complete grid-state shape.
    # 2. Check `0 <= action < config.num_actions`.
    # 3. Build `action_one_hot = torch.zeros(config.num_actions, dtype=torch.float32)`.
    # 4. Set `action_one_hot[action] = 1.0`.
    # 5. Flatten `full_state` and concatenate the action.
    #
    # Checks:
    # - The last four values should contain exactly one `1.0`.

    expected_shape = (config.state_channels, config.grid_size, config.grid_size)
    if full_state.shape != expected_shape:
        raise ValueError(f"Expected full_state shape {expected_shape}, got {tuple(full_state.shape)}.")

    if not 0 <= action < config.num_actions:
        raise ValueError("Invalid action input.")

    action_one_hot = torch.zeros(config.num_actions, dtype=torch.float32, device=full_state.device)
    action_one_hot[action] = 1.0

    return torch.cat([full_state.float().flatten(), action_one_hot], dim=0)

    #raise NotImplementedError("Encode one state-action pair.")


def collect_logged_episodes(config: GridworldPlanningConfig) -> list[list[LoggedGridStep]]:
    """Collect complete real episodes from the shared grid-world."""
    # Where this fits:
    # This is the real data source for model training.
    #
    # Goal:
    # Return complete logged episodes from `common.embodied_gridworld`.
    #
    # Method:
    # 1. Build `grid_config = make_visual_grid_config(config)`.
    # 2. Collect `episodes = collect_visual_gridworld_episodes(grid_config)`.
    # 3. Check that the episode list and every kept episode are not empty.
    # 4. Return the episodes without flattening them.
    #
    # Checks:
    # - Do not create random tensors here.
    # - Do not invent reward, cost, or done labels.
    #
    # Explore versus exploit:
    # `collect_visual_gridworld_episodes(...)` uses the shared scripted policy.
    # With its default `explore_probability=0.15`:
    # - explore: 15% of actions are random so the logs contain more varied
    #   experience
    # - exploit: otherwise, use the shortest safe path that currently looks
    #   best from the full grid map
    #
    # This is where exploration happens in this exercise. The model later
    # learns from these saved episodes.


    grid_config = make_visual_grid_config(config)
    episodes = collect_visual_gridworld_episodes(grid_config)
    return episodes

    #raise NotImplementedError("Collect complete logged planning episodes.")


def build_planning_dataset(config: GridworldPlanningConfig) -> PlanningDatasetSplits:
    """Build train and validation tensors from logged transitions."""
    # Where this fits:
    # This turns real logs into tensors the dynamics model can train on.
    #
    # Goal:
    # Build aligned train and validation tensors without leaking transitions
    # from one episode across both groups.
    #
    # Method:
    # 1. Call `collect_logged_episodes(config)`.
    # 2. Shuffle and split episode indices by `config.val_fraction`.
    # 3. Flatten the train episodes and validation episodes separately with
    #    `flatten_logged_episodes(...)`.
    # 4. For each step in each group, build:
    #    - input from `encode_full_state_action(step["full_state"], ...)`
    #    - next-position cell index from `step["next_full_state"]`
    #    - reward target shaped `[1]`
    #    - cost target shaped `[1]`
    #    - done target shaped `[1]`, where done is `1.0` or `0.0`
    # 5. Stack each train list and validation list separately with
    #    `torch.stack(...)`.
    #
    # Checks:
    # - Inputs and targets must stay aligned.
    # - No episode may appear in both splits.

    episodes = collect_logged_episodes(config)

    shuffled_indices = torch.randperm(n=len(episodes), device=device)

    num_val = int(len(episodes) * config.val_fraction)

    val_indices = shuffled_indices[:num_val]
    train_indices = shuffled_indices[num_val:]

    val_episodes = [episodes[index.item()] for index in val_indices]
    train_episodes = [episodes[index.item()] for index in train_indices]

    train_flat_episodes = flatten_logged_episodes(train_episodes)
    val_flat_episodes = flatten_logged_episodes(val_episodes)

    def build_tensors(steps: list[LoggedGridStep]) -> tuple[torch.Tensor, ...]:
        inputs: list[torch.Tensor] = []
        next_position_indices: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        costs: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []

        for step in steps:
            full_state = step["full_state"].to(device=device, dtype=torch.float32)
            inputs.append(encode_full_state_action(full_state, int(step["action"]), config))

            next_position_channel = step["next_full_state"][0]
            next_position_index = int(next_position_channel.flatten().argmax().item())
            next_position_indices.append(torch.tensor(next_position_index, dtype=torch.long, device=device))
            rewards.append(torch.tensor([step["reward"]], dtype=torch.float32, device=device))
            costs.append(torch.tensor([step["cost"]], dtype=torch.float32, device=device))
            dones.append(torch.tensor([float(step["done"])], dtype=torch.float32, device=device))

        return (
            torch.stack(inputs),
            torch.stack(next_position_indices),
            torch.stack(rewards),
            torch.stack(costs),
            torch.stack(dones),
        )

    train_tensors = build_tensors(train_flat_episodes)
    val_tensors = build_tensors(val_flat_episodes)

    return {
        "train_inputs": train_tensors[0],
        "train_next_position_indices": train_tensors[1],
        "train_rewards": train_tensors[2],
        "train_costs": train_tensors[3],
        "train_dones": train_tensors[4],
        "val_inputs": val_tensors[0],
        "val_next_position_indices": val_tensors[1],
        "val_rewards": val_tensors[2],
        "val_costs": val_tensors[3],
        "val_dones": val_tensors[4],
    }

    #raise NotImplementedError("Build train and validation tensors.")


class GridworldPlanningDynamicsModel(nn.Module):
    """Predict next position, reward, cost, and episode end for planning."""

    def __init__(self, config: GridworldPlanningConfig) -> None:
        # Where this fits:
        # This is the learned model used by the planner to imagine futures.
        #
        # Goal:
        # Build one shared trunk and four heads.
        #
        # Method:
        # 1. Call `super().__init__()`.
        # 2. Include every cell from all four state channels and the action.
        # 3. Build `self.trunk = nn.Sequential(...)`.
        # 4. Use two `nn.Linear(...)` layers with `nn.ReLU()` between them.
        # 5. Build:
        #    - `self.next_position_head` returns one logit for every grid cell
        #    - `self.reward_head = nn.Linear(config.hidden_dim, 1)`
        #    - `self.cost_head = nn.Linear(config.hidden_dim, 1)`
        #    - `self.done_head = nn.Linear(config.hidden_dim, 1)`
        #
        # Checks:
        # - Do not apply sigmoid inside the done head.
        super().__init__()

        self.input_dim = config.state_channels * config.grid_size * config.grid_size + config.num_actions

        self.trunk = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features = config.hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=config.hidden_dim, out_features=config.hidden_dim),
                                    nn.ReLU(inplace=True))

        self.next_position_head = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.grid_size * config.grid_size,
        )
        self.reward_head = nn.Linear(in_features=config.hidden_dim, out_features=1)
        self.cost_head = nn.Linear(in_features=config.hidden_dim, out_features=1)
        self.done_head = nn.Linear(in_features=config.hidden_dim, out_features=1)

        #raise NotImplementedError("Build the planning dynamics model.")

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return next-position, reward, cost, and episode-end predictions."""
        # Goal:
        # Run `inputs` through the trunk and all four heads.
        #
        # Input:
        # - `inputs`: `[batch_size, state_channels * grid_size * grid_size + num_actions]`.
        #
        # Output:
        # - next_position_logits: `[batch_size, grid_size * grid_size]`
        # - reward: `[batch_size, 1]`
        # - cost: `[batch_size, 1]`
        # - done_logit: `[batch_size, 1]`

        features = self.trunk(inputs)
        next_position_logits = self.next_position_head(features)
        reward = self.reward_head(features)
        cost = self.cost_head(features)
        done_logit = self.done_head(features)

        return next_position_logits, reward, cost, done_logit
        #raise NotImplementedError("Run the planning dynamics model.")


def compute_dynamics_loss(
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return total dynamics loss and its parts."""
    # Where this fits:
    # This trains the world model used by the planner.
    #
    # Method:
    # 1. Unpack predictions into next-position logits, reward, cost, and done logit.
    # 2. Unpack targets into the true next-position index, reward, cost, and done.
    # 3. Use cross-entropy for the next grid cell and MSE for reward and cost.
    # 4. Use `nn.functional.binary_cross_entropy_with_logits(...)` for done.
    # 5. Add all four losses.
    # 6. Return `(total_loss, loss_parts)`.
    #
    # Checks:
    # - `done` should contain floats shaped `[batch_size, 1]`.
    # - `done_logit` should be raw logits, not probabilities.

    next_position_loss = nn.functional.cross_entropy(predictions[0], targets[0].long())
    reward_loss = nn.functional.mse_loss(predictions[1], targets[1])
    cost_loss = nn.functional.mse_loss(predictions[2], targets[2])
    done_loss = nn.functional.binary_cross_entropy_with_logits(predictions[3], targets[3])

    loss = next_position_loss + reward_loss + cost_loss + done_loss

    loss_parts = {
        "next_position_loss": next_position_loss.item(),
        "reward_loss": reward_loss.item(),
        "cost_loss": cost_loss.item(),
        "done_loss": done_loss.item(),
    }

    return loss, loss_parts

    #raise NotImplementedError("Compute dynamics-model loss.")


def train_one_epoch(
    model: GridworldPlanningDynamicsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train for one epoch and return mean loss."""
    # Where this fits:
    # This updates the learned model, not the planner.
    #
    # Method:
    # 1. Call `model.train()`.
    # 2. Loop over batches from the loader.
    # 3. Move every tensor to `device`.
    # 4. Run the model.
    # 5. Compute loss with `compute_dynamics_loss(...)`.
    # 6. Call `optimizer.zero_grad()`.
    # 7. Call `loss.backward()`.
    # 8. Call `optimizer.step()`.
    # 9. Return mean loss weighted by batch size.

    model.train()

    total_loss = 0
    total_samples = 0

    for inputs, next_position_indices, rewards, costs, dones in loader:
        inputs = inputs.to(device)

        next_position_indices = next_position_indices.to(device)
        rewards = rewards.to(device)                        # shape is [batch_size, 1]
        costs = costs.to(device)                            # shape is [batch_size, 1]
        dones = dones.to(device)                            # shape is [batch_size, 1]

        pred_next_position_logits, pred_reward, pred_cost, pred_done_logit = model(inputs)

        optimizer.zero_grad()

        loss, loss_parts = compute_dynamics_loss(
            predictions=(pred_next_position_logits, pred_reward, pred_cost, pred_done_logit),
            targets=(next_position_indices, rewards, costs, dones),
        )

        loss.backward()
        optimizer.step()

        batch_size = dones.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

    #raise NotImplementedError("Train one planning dynamics epoch.")

@torch.no_grad()
def evaluate_one_step_model(
    model: GridworldPlanningDynamicsModel,
    splits: PlanningDatasetSplits,
) -> OneStepMetrics:
    """Evaluate one-step prediction quality on validation data."""
    # Where this fits:
    # This checks whether the learned model predicts held-out transitions.
    #
    # Goal:
    # Return validation metrics without updating model weights.
    #
    # Method:
    # 1. Call `model.eval()`.
    # 2. Use `torch.no_grad()`.
    # 3. Move validation tensors to `device`.
    # 4. Run the model.
    # 5. Compute next-position accuracy and MSE for reward and cost.
    # 6. Convert done logits with `torch.sigmoid(...)`.
    # 7. Compute done accuracy with threshold `0.5`.
    #
    # Output keys:
    # - `val_next_position_accuracy`
    # - `val_reward_mse`
    # - `val_cost_mse`
    # - `val_done_accuracy`

    model.eval()

    val_inputs = splits["val_inputs"].to(device)
    val_next_position_indices = splits["val_next_position_indices"].to(device)
    val_rewards = splits["val_rewards"].to(device)
    val_costs = splits["val_costs"].to(device)
    val_dones = splits["val_dones"].to(device)

    pred_next_position_logits, pred_reward, pred_cost, pred_done_logit = model(val_inputs)

    pred_next_position_indices = pred_next_position_logits.argmax(dim=-1)
    val_next_position_accuracy = (
        pred_next_position_indices == val_next_position_indices
    ).float().mean().item()
    val_reward_mse = nn.functional.mse_loss(pred_reward, val_rewards).item()
    val_cost_mse = nn.functional.mse_loss(pred_cost, val_costs).item()

    pred_done = torch.sigmoid(pred_done_logit) >= 0.5
    true_done = val_dones >= 0.5
    val_done_accuracy = (pred_done == true_done).float().mean().item()

    return {
        "val_next_position_accuracy": val_next_position_accuracy,
        "val_reward_mse": val_reward_mse,
        "val_cost_mse": val_cost_mse,
        "val_done_accuracy": val_done_accuracy,
    }

    #raise NotImplementedError("Evaluate one-step dynamics metrics.")


def train_planning_dynamics_model(
    config: GridworldPlanningConfig,
) -> tuple[GridworldPlanningDynamicsModel, PlanningDatasetSplits, list[float], list[float], OneStepMetrics]:
    """Train the learned model that the planner will use."""
    # Where this fits:
    # This prepares the model used for imagined rollouts.
    #
    # Method:
    # 1. Call `set_seed(config.random_seed)`.
    # 2. Build `splits = build_planning_dataset(config)`.
    # 3. Build `TensorDataset(...)` from the training tensors.
    # 4. Build `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`.
    # 5. Build `GridworldPlanningDynamicsModel(config).to(device)`.
    # 6. Build `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 7. For each epoch:
    #    - call `train_one_epoch(...)`
    #    - call `evaluate_one_step_model(...)`
    #    - append train loss and validation next-position accuracy to lists
    # 8. Return the model, splits, histories, and final metrics.

    set_seed(config.random_seed)

    splits = build_planning_dataset(config)

    train_dataset = TensorDataset(
        splits["train_inputs"],
        splits["train_next_position_indices"],
        splits["train_rewards"],
        splits["train_costs"],
        splits["train_dones"],
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    model = GridworldPlanningDynamicsModel(config).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate)

    train_losses: list[float] = []
    val_next_position_accuracies: list[float] = []
    final_metrics: OneStepMetrics | None = None

    print(
        "Training planning dynamics model "
        f"on {len(train_dataset)} train rows and {splits['val_inputs'].shape[0]} validation rows."
    )

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        metrics = evaluate_one_step_model(model, splits)

        train_losses.append(train_loss)
        val_next_position_accuracies.append(metrics["val_next_position_accuracy"])
        final_metrics = metrics

        print(
            f"epoch {epoch + 1:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_next_position_accuracy={metrics['val_next_position_accuracy']:.3f} | "
            f"val_reward_mse={metrics['val_reward_mse']:.4f} | "
            f"val_cost_mse={metrics['val_cost_mse']:.4f} | "
            f"val_done_accuracy={metrics['val_done_accuracy']:.3f}"
        )

    if final_metrics is None:
        raise ValueError("Training did not run because config.epochs is 0.")

    return model, splits, train_losses, val_next_position_accuracies, final_metrics

    #raise NotImplementedError("Train the planning dynamics model.")


def candidate_action_sequences(config: GridworldPlanningConfig) -> list[tuple[int, ...]]:
    """Return every action sequence for the planning horizon."""
    # Where this fits:
    # The planner will score these possible futures.
    #
    # Goal:
    # Build all action tuples of length `config.planning_horizon`.
    #
    # Method:
    # 1. Validate that `config.planning_horizon > 0`.
    # 2. Use `itertools.product(range(config.num_actions), repeat=config.planning_horizon)`.
    # 3. Convert the result to a list.
    #
    # Checks:
    # - With 4 actions and horizon 4, there should be `4 ** 4 = 256` sequences.
    # - Every sequence should have length `config.planning_horizon`.
    #
    # Why `itertools.product` is used:
    # With horizon 4, you could write four nested loops by hand:
    #
    # ```text
    # for a0 in range(num_actions):
    #     for a1 in range(num_actions):
    #         for a2 in range(num_actions):
    #             for a3 in range(num_actions):
    #                 sequence = (a0, a1, a2, a3)
    # ```
    #
    # `itertools.product(range(num_actions), repeat=planning_horizon)` does the
    # same thing, but it still works if the horizon changes to 2, 5, or 10.
    import itertools

    if config.planning_horizon <= 0:
        raise ValueError("planning_horizon is zero.")

    sequences = list(itertools.product(range(config.num_actions), repeat=config.planning_horizon))

    # Each item in `sequences` is one possible future plan.
    # Example: `(1, 1, 3, 3)` means:
    # - step 1: action 1
    # - step 2: action 1
    # - step 3: action 3
    # - step 4: action 3
    #
    # In this grid-world:
    # - action 0 means up
    # - action 1 means down
    # - action 2 means left
    # - action 3 means right
    #
    # So `(1, 1, 3, 3)` means down, down, right, right.
    # `predict_action_sequence_score(...)` scores one of these fixed plans.
    # `choose_planned_action(...)` later scores many plans and picks the best one.
    # Listing all possible plans is a search inside the frozen learned model.
    # It is not exploration for learning because it does not take real actions,
    # collect new training data, or update model weights.
    return sequences

    #raise NotImplementedError("Build candidate action sequences.")


def move_position_in_full_state(
    full_state: torch.Tensor,
    next_position_index: int,
    config: GridworldPlanningConfig,
) -> torch.Tensor:
    """Copy a full state and move only its current-position marker."""

    num_grid_cells = config.grid_size * config.grid_size
    if not 0 <= next_position_index < num_grid_cells:
        raise ValueError(f"next_position_index must be between 0 and {num_grid_cells - 1}.")

    next_full_state = full_state.clone()
    next_full_state[0].zero_()
    next_row, next_col = divmod(next_position_index, config.grid_size)
    next_full_state[0, next_row, next_col] = 1.0
    return next_full_state


@torch.no_grad()
def predict_one_step(
    model: GridworldPlanningDynamicsModel,
    full_state: torch.Tensor,
    action: int,
    config: GridworldPlanningConfig,
) -> tuple[torch.Tensor, float, float, bool]:
    """Predict one imagined future step with the learned model."""
    # Where this fits:
    # This is one step inside an imagined rollout.
    #
    # Input:
    # - `full_state`: tensor shaped `[4, grid_size, grid_size]`.
    # - `action`: integer action.
    #
    # Output:
    # - predicted next full state shaped `[4, grid_size, grid_size]`
    # - predicted reward as a Python float
    # - predicted cost as a Python float
    # - predicted done as a Python bool
    #
    # Method:
    # 1. Call `model.eval()`.
    # 2. Build input with `encode_full_state_action(full_state, action, config)`.
    # 3. Add a batch dimension with `.unsqueeze(0)`.
    # 4. Move the input to `device`.
    # 5. Use `torch.no_grad()`.
    # 6. Run the model.
    # 7. Choose the most likely next grid cell with `argmax`.
    # 8. Move only the position marker; keep goal, hazards, and walls fixed.
    # 9. Convert reward, cost, and episode end to plain Python values.
    #
    # Checks:
    # - Do not call the real environment here.
    # - This function imagines one step using the learned model.

    model.eval()

    model_input = encode_full_state_action(full_state, action, config)

    model_input = model_input.unsqueeze(dim=0)
    model_input = model_input.to(device)

    pred_next_position_logits, pred_reward, pred_cost, pred_done_logit = model(model_input)

    pred_next_position_index = int(pred_next_position_logits.argmax(dim=-1).item())
    pred_next_full_state = move_position_in_full_state(full_state, pred_next_position_index, config)
    pred_reward = pred_reward.item()
    pred_cost = pred_cost.item()

    pred_done = torch.sigmoid(pred_done_logit).item() >= 0.5

    return pred_next_full_state, pred_reward, pred_cost, pred_done

    #raise NotImplementedError("Predict one imagined step.")

@torch.no_grad()
def predict_action_sequence_score(
    model: GridworldPlanningDynamicsModel,
    start_full_state: torch.Tensor,
    actions: tuple[int, ...],
    config: GridworldPlanningConfig,
) -> float:
    """Score one imagined action sequence."""
    # Where this fits:
    # The planner uses this predicted sequence score to rank possible futures.
    #
    # Formula:
    # `safe_cost = max(predicted_cost, 0.0)`
    # `predicted_sequence_score = sum(predicted_reward - config.cost_weight * safe_cost)`.
    #
    # Method:
    # 1. Set `current_full_state = start_full_state.clone()`.
    # 2. Set `predicted_sequence_score = 0.0`.
    # 3. Loop over actions.
    # 4. Call `predict_one_step(model, current_full_state, action, config)`.
    # 5. Clamp predicted cost at zero, then add
    #    `reward - config.cost_weight * safe_cost` to `predicted_sequence_score`.
    # 6. Move `current_full_state` to the predicted next full state.
    # 7. Stop early if predicted done is true.
    # 8. Return the predicted sequence score as a float.
    #
    # Checks:
    # - Do not use true future rewards from the environment here.
    # - This function should only use model predictions.

    model.eval()

    current_full_state = start_full_state.clone()
    predicted_sequence_score = 0.0

    # `actions` is one fixed future plan, such as `(1, 1, 3, 3)`.
    # In this grid-world:
    # - action 0 means up
    # - action 1 means down
    # - action 2 means left
    # - action 3 means right
    #
    # So `(1, 1, 3, 3)` means down, down, right, right.
    # This function does not choose between plans. It only gives this one plan
    # a predicted sequence score using the learned model's imagined next steps.

    for action in actions:
        pred_next_full_state, pred_reward, pred_cost, pred_done = predict_one_step(
            model, current_full_state, action, config
        )

        safe_cost = max(pred_cost, 0.0)
        predicted_sequence_score += pred_reward - config.cost_weight * safe_cost

        if pred_done:
            break

        current_full_state = pred_next_full_state

    return predicted_sequence_score
    #raise NotImplementedError("Score one imagined action sequence.")


def calculate_real_action_sequence_score(
    scene: GridScene,
    actions: tuple[int, ...],
    config: GridworldPlanningConfig,
) -> float:
    """Execute one fixed action sequence in a copied real scene."""
    # Where this fits:
    # `predict_action_sequence_score(...)` imagines a sequence. This helper
    # executes the same sequence with the true grid-world rules so the two
    # sequence scores can be compared directly.
    #
    # Input:
    # - `scene.start_position` and `scene.goal_position` are `(row, col)`
    #   tuples like `(0, 3)`.
    # - `scene.hazard_positions` and `scene.wall_positions` are sets of
    #   `(row, col)` tuples.
    # - `actions` is a fixed tuple such as `(1, 1, 3, 3)`.
    #
    # Output:
    # - one Python float: the real sequence score for that fixed action sequence.
    #
    # Exact environment call:
    # `step_grid_world(current_position, action, goal_position,
    # hazard_positions, wall_positions, grid_config)` returns
    # `(next_position, reward, cost, done)`.
    #
    # Tiny example:
    # If `reward=-0.01`, `cost=0.2`, and `config.cost_weight=1.0`, then this
    # step adds `-0.01 - 1.0 * 0.2 = -0.21` to the real sequence score.
    #
    # Method:
    # 1. Set `current_position = scene.start_position`.
    # 2. Build the shared environment config with `make_visual_grid_config(...)`.
    # 3. Set `real_sequence_score = 0.0`.
    # 4. Loop over each action in `actions`.
    # 5. Call:
    #    `next_position, reward, cost, done = step_grid_world(...)`.
    # 6. Add `reward - config.cost_weight * cost` to `real_sequence_score`.
    # 7. Set `current_position = next_position` so the next action starts from
    #    the new real position.
    # 8. If `done` is true, stop the loop early.
    # 9. Return `float(real_sequence_score)`.
    #
    # Checks:
    # - Do not call the learned model in this function.
    # - Do not change `scene` itself. Only update `current_position`.
    # - This real sequence score uses raw real cost. Do not clamp real cost here.


    current_position = scene.start_position

    grid_config = make_visual_grid_config(config)

    real_sequence_score = 0.0

    for action in actions:
        next_position, reward, cost, done = step_grid_world(
            current_position,
            action,
            scene.goal_position,
            scene.hazard_positions,
            scene.wall_positions,
            grid_config,
        )
        real_sequence_score += reward - config.cost_weight * cost
        current_position = next_position

        if done:
            break

    return float(real_sequence_score)

    #raise NotImplementedError("Score a fixed action sequence in the real scene.")

@torch.no_grad()
def audit_selected_sequence_model_error(
    model: GridworldPlanningDynamicsModel,
    scenes: list[GridScene],
    config: GridworldPlanningConfig,
) -> dict[str, float]:
    """Check whether search prefers sequences where the model is unusually wrong."""
    # Why this check matters:
    # The planner scores many possible action sequences inside the learned
    # model. That is useful, but it creates one extra risk:
    #
    # The planner may find plans that only look good because the learned model
    # is wrong.
    #
    # Example:
    # - The learned model imagines that `(1, 1, 3, 3)` gets score `0.80`.
    # - The real grid-world gives that same fixed sequence score `0.10`.
    # - The gap is `abs(0.80 - 0.10) = 0.70`.
    #
    # A large gap means the model's imagined score did not match reality.
    #
    # This function compares two kinds of fixed action sequence:
    # - selected sequence: the sequence the planner thinks is best
    # - random sequence: one random sequence, used as a normal comparison
    #
    # If selected sequences have much bigger gaps than random sequences, the
    # planner may be exploiting model mistakes. In plain words, search may be
    # finding places where the model is too optimistic.
    #
    # This use of "exploiting" is narrower than the usual explore-versus-exploit
    # choice:
    # - normal exploitation means choosing the action that currently looks best
    # - exploiting a model mistake means that action looks best only because the
    #   learned model predicts an unrealistically good result
    #
    # This function does not train the model.
    # This function does not take a real action for the agent.
    # It only audits whether imagined scores match real scores.
    #
    # Names used below:
    # - imagined score: score from `predict_action_sequence_score(...)`, using
    #   the model
    # - real score: score from `calculate_real_action_sequence_score(...)`,
    #   using the true grid-world rules
    # - gap: `abs(imagined_score - real_score)`
    #
    # Method:
    # 1. Build all candidate action sequences with `candidate_action_sequences(config)`.
    # 2. Build a seeded random number generator:
    #    `rng = random.Random(config.random_seed + 20_000)`.
    # 3. Create empty lists:
    #    - `selected_gaps = []`
    #    - `random_gaps = []`
    # 4. Create a counter:
    #    - `selected_overestimates = 0`
    # 5. For each held-out `scene`:
    #    - build `grid_config = make_visual_grid_config(config)`
    #    - encode the complete starting state with
    #      `encode_full_grid_state(...)`
    # 6. Score every candidate sequence from that starting state with
    #    `predict_action_sequence_score(model, start_full_state, actions, config)`.
    # 7. Keep the highest-scoring pair:
    #    - `selected_imagined_score`
    #    - `selected_actions`
    # 8. Pick one random sequence:
    #    `random_actions = rng.choice(sequences)`.
    #    This random sequence is only an audit comparison. The acting planner
    #    does not take it to explore or collect new training experience.
    # 9. Compute `random_imagined_score` with
    #    `predict_action_sequence_score(...)`.
    # 10. Replay both fixed sequences in the real scene with
    #     `calculate_real_action_sequence_score(...)`.
    # 11. Compute:
    #     - `selected_gap = abs(selected_imagined_score - selected_real_score)`
    #     - `random_gap = abs(random_imagined_score - random_real_score)`
    # 12. Append those gaps to their lists.
    # 13. If `selected_imagined_score > selected_real_score`, add 1 to
    #     `selected_overestimates`.
    # 14. After all scenes, return:
    #    - `selected_sequence_score_gap`
    #    - `random_sequence_score_gap`
    #    - `selected_vs_random_model_error_gap = selected_gap - random_gap`
    #    - `selected_sequence_overestimate_rate`
    #
    # Interpretation:
    # - `selected_sequence_score_gap` is the average gap for planner-picked
    #   sequences. Lower is better.
    # - `random_sequence_score_gap` is the average gap for random sequences.
    #   It is a comparison point.
    # - `selected_vs_random_model_error_gap` is selected gap minus random gap.
    #   If it is positive, planner-picked sequences are more wrong than random
    #   sequences.
    # - `selected_sequence_overestimate_rate` is the fraction of scenes where
    #   the model thought the selected sequence was better than it really was.
    #_ = random

    sequences = candidate_action_sequences(config)
    rng = random.Random(config.random_seed + 20_000)

    selected_gaps = []
    random_gaps = []
    selected_overestimates = 0

    for scene in scenes:
        grid_config = make_visual_grid_config(config)

        start_full_state = encode_full_grid_state(
            scene.start_position,
            scene.goal_position,
            scene.hazard_positions,
            scene.wall_positions,
            grid_config,
        )

        scored_sequences = []
        for actions in sequences:
            imagined_score = predict_action_sequence_score(model, start_full_state, actions, config)
            scored_sequences.append((imagined_score, actions))

        # Get the highest-scoring pair.
        selected_imagined_score, selected_actions = max(scored_sequences, key=lambda item: item[0])

        random_actions = rng.choice(sequences)
        #this is not a random score, it is the actual score for a randomly chosen set of actions from sequences.
        random_imagined_score = predict_action_sequence_score(
            model, start_full_state, random_actions, config
        )

        selected_real_score = calculate_real_action_sequence_score(scene, selected_actions, config)
        random_real_score = calculate_real_action_sequence_score(scene, random_actions, config)

        selected_gaps.append(abs(selected_imagined_score - selected_real_score))
        random_gaps.append(abs(random_imagined_score - random_real_score))

        if selected_imagined_score > selected_real_score:
            selected_overestimates += 1

    selected_sequence_score_gap_avg = sum(selected_gaps) / len(selected_gaps)
    random_sequence_score_gap_avg = sum(random_gaps) / len(random_gaps)

    return {
        "selected_sequence_score_gap": selected_sequence_score_gap_avg,
        "random_sequence_score_gap": random_sequence_score_gap_avg,
        "selected_vs_random_model_error_gap": selected_sequence_score_gap_avg - random_sequence_score_gap_avg,
        "selected_sequence_overestimate_rate": selected_overestimates / len(scenes),
    }

    #raise NotImplementedError("Audit selected-sequence model error.")


def choose_planned_action(
    model: GridworldPlanningDynamicsModel,
    current_full_state: torch.Tensor,
    config: GridworldPlanningConfig,
) -> int:
    """Choose the first action from the best imagined sequence."""
    # Where this fits:
    # This is the actual model-based planner.
    #
    # Explore versus exploit:
    # This function exploits the frozen learned model. It chooses the sequence
    # with the highest predicted score, then returns that sequence's first
    # action. It does not deliberately try a worse-looking real action to learn
    # more, so no exploration happens here.
    #
    # Method:
    # 1. Build sequences with `candidate_action_sequences(config)`.
    # 2. Score every sequence with `predict_action_sequence_score(...)`.
    # 3. Keep the sequence with the highest predicted sequence score.
    # 4. Return the first action in that sequence.
    #
    # Checks:
    # - The returned action should be an `int`.
    # - The action should be in `0 <= action < config.num_actions`.
    sequences = candidate_action_sequences(config)

    best_predicted_sequence_score = float("-inf")
    best_sequence: tuple[int, ...] | None = None

    for actions in sequences:
        predicted_sequence_score = predict_action_sequence_score(
            model, current_full_state, actions, config
        )
        if predicted_sequence_score > best_predicted_sequence_score:
            best_predicted_sequence_score = predicted_sequence_score
            best_sequence = actions

    action = int(best_sequence[0])

    return action

    #raise NotImplementedError("Choose an action with model-based planning.")


def choose_shortest_path_baseline_action(
    current_position: Position,
    goal_position: Position,
    hazard_positions: set[Position],
    wall_positions: set[Position],
    config: GridworldPlanningConfig,
) -> int:
    """Choose a shortest-path baseline action without using the learned model."""
    # Where this fits:
    # This shortest-path baseline tells us whether learned planning helped.
    #
    # Goal:
    # Return a wall-avoiding shortest-path action when possible.
    #
    # Method:
    # 1. Build `grid_config = make_visual_grid_config(config)`.
    # 2. Use `shortest_path_action(current_position, goal_position,
    #    wall_positions | hazard_positions, grid_config)` first.
    # 3. If that returns `None`, try again with only `wall_positions` blocked.
    # 4. If that also returns `None`, return action `0`.
    #
    # Why this baseline is allowed:
    # It uses environment map information, so it is a strong hand-written
    # shortest-path baseline.
    # The planner should still be evaluated against it so you can see what the
    # learned model does well and where it is weaker.


    grid_config = make_visual_grid_config(config)

    action = shortest_path_action(
        current_position,
        goal_position,
        wall_positions | hazard_positions,
        grid_config,
    )

    if action is None:
        action = shortest_path_action(current_position, goal_position, wall_positions, grid_config)

    if action is None:
        action = 0

    return action
    #raise NotImplementedError("Choose one shortest-path baseline action.")


def run_policy_in_real_gridworld(
    model: GridworldPlanningDynamicsModel,
    scene: GridScene,
    config: GridworldPlanningConfig,
    use_planner: bool,
) -> RolloutResult:
    """Roll out either the learned planner or the shortest-path baseline."""
    # Where this fits:
    # This function runs one full episode in the real grid-world rules.
    #
    # A "rollout" means:
    # start from one scene, choose an action, step the world, save what happened,
    # then repeat until the goal is reached or `config.max_steps` is hit.
    #
    # There are two different things happening in this loop:
    #
    # 1. Choosing the action.
    #    If `use_planner=True`, the learned model helps choose the action by
    #    imagining possible futures.
    #    If `use_planner=False`, the shortest-path baseline chooses the action.
    #
    # 2. Testing the action.
    #    The chosen action is always tested with `step_grid_world(...)`, which
    #    is the real grid-world rule. The learned model is not allowed to decide
    #    the real reward, real cost, next position, or done flag.
    #
    # The learned model is still called in this function for one reason:
    # we save what the model predicted before the real step. Later we compare
    # those predictions with what really happened.
    #
    # So the loop keeps two sets of totals:
    # - `real_return` and `real_cost`: what the real grid-world returned
    # - `predicted_return` and `predicted_cost`: what the learned model guessed
    #
    # Nothing in this function trains the model. There is no `loss.backward()`
    # and no `optimizer.step()`. The model is only used for action choice and
    # prediction logging.
    #
    # Input:
    # - `scene`: the fixed starting layout for this episode.
    # - `use_planner=True`: choose actions with the learned-model planner.
    # - `use_planner=False`: choose actions with the shortest-path baseline.
    #
    # Output:
    # Return a `RolloutResult` dictionary with:
    # - `visited_positions`: positions visited, including the start position.
    # - `actions`: action ids actually taken in the real environment.
    # - `real_return`: sum of real rewards from `step_grid_world(...)`.
    # - `real_cost`: sum of real costs from `step_grid_world(...)`.
    # - `predicted_return`: sum of model-predicted rewards for chosen actions.
    # - `predicted_cost`: sum of model-predicted costs for chosen actions.
    # - `reached_goal`: `True` if the final real position reached the goal.
    # - `steps`: number of real environment steps taken.
    #
    # Exact helper calls:
    # - `encode_full_grid_state(...)` returns the complete grid shaped
    #   `[4, grid_size, grid_size]`.
    # - `predict_one_step(model, current_full_state, action, config)` returns
    #   `(pred_next_full_state, pred_reward, pred_cost, pred_done)`.
    # - `step_grid_world(current_position, action, scene.goal_position,
    #   scene.hazard_positions, scene.wall_positions, grid_config)` returns
    #   `(next_position, reward, cost, done)`.
    #
    # Tiny one-step example:
    # Start with `current_position=(0, 0)`.
    # Suppose the planner or shortest-path baseline chooses action `1`, meaning down.
    # The model predicts reward `-0.02` and cost `0.0`, so add those to
    # `predicted_return` and `predicted_cost`.
    # The real environment returns `next_position=(1, 0)`, reward `-0.01`, cost
    # `0.0`, and `done=False`, so add the real values to `real_return` and
    # `real_cost`, append action `1`, append state `(1, 0)`, and continue.
    #
    # Method:
    # 1. Set `current_position = scene.start_position`.
    # 2. Build `grid_config = make_visual_grid_config(config)`.
    # 3. Create:
    #    - `visited_positions = [current_position]`
    #    - `actions = []`
    #    - `real_return = 0.0`
    #    - `real_cost = 0.0`
    #    - `predicted_return = 0.0`
    #    - `predicted_cost = 0.0`
    # 4. Loop for at most `config.max_steps`.
    # 5. Encode the current state with:
    #    `current_full_state = encode_full_grid_state(...)`.
    # 6. Choose one action:
    #    - planner: `choose_planned_action(model, current_full_state, config)`
    #    - shortest-path baseline:
    #      `choose_shortest_path_baseline_action(current_position, ...)`
    # 7. Before the real step, call `predict_one_step(...)` for the chosen
    #    action and add its predicted reward and cost to separate totals.
    # 8. Step the real environment with:
    #    `next_position, reward, cost, done = step_grid_world(...)`.
    # 9. Add `reward` to `real_return` and `cost` to `real_cost`.
    # 10. Append the chosen action to `actions`.
    # 11. Set `current_position = next_position` and append it to
    #     `visited_positions`.
    # 12. Stop early if `done` is true.
    # 13. After the loop, compare `current_position` with `scene.goal_position`.
    # 14. Return the dictionary with all fields listed above.
    #
    # Checks:
    # - The real score must come from `step_grid_world(...)`.
    # - Do not use predicted reward as the final real score.
    # - `len(visited_positions)` should usually equal `len(actions) + 1`.


    current_position = scene.start_position

    grid_config = make_visual_grid_config(config)

    visited_positions = [current_position]
    actions = []
    real_return = 0.0
    real_cost = 0.0
    predicted_return = 0.0
    predicted_cost = 0.0

    for _ in range(config.max_steps):
        current_full_state = encode_full_grid_state(
            current_position,
            scene.goal_position,
            scene.hazard_positions,
            scene.wall_positions,
            grid_config,
        )

        # If this uses the planner, `choose_planned_action(...)` will call
        # `predict_one_step(...)` many times while it imagines possible future
        # action sequences. Those calls are for choosing the action.
        #
        # After the action is chosen, this function calls `predict_one_step(...)`
        # once more below. That later call is not for choosing. It only logs what
        # the model predicted for the exact action that will be tested in the
        # real grid-world.
        #
        # This keeps the learning idea simple:
        # 1. choose an action
        # 2. log the model's prediction for that chosen action
        # 3. step the real environment and compare what really happened
        if use_planner:
            action = choose_planned_action(model, current_full_state, config)
        else:
            action = choose_shortest_path_baseline_action(
                current_position,
                scene.goal_position,
                scene.hazard_positions,
                scene.wall_positions,
                config,
            )

        _pred_next_full_state, pred_reward, pred_cost, _pred_done = predict_one_step(
            model, current_full_state, action, config
        )
        predicted_return += pred_reward
        predicted_cost += pred_cost

        next_position, reward, cost, done = step_grid_world(
            current_position,
            action,
            scene.goal_position,
            scene.hazard_positions,
            scene.wall_positions,
            grid_config,
        )

        real_return += reward
        real_cost += cost
        actions.append(action)

        current_position = next_position
        visited_positions.append(current_position)

        if done:
            break

    return {
        "visited_positions": visited_positions,
        "actions": actions,
        "real_return": real_return,
        "real_cost": real_cost,
        "predicted_return": predicted_return,
        "predicted_cost": predicted_cost,
        "reached_goal": current_position == scene.goal_position,
        "steps": len(actions),
    }

    #raise NotImplementedError("Roll out one policy in the real grid-world.")


def build_eval_scenes(config: GridworldPlanningConfig) -> list[GridScene]:
    """Build real grid-world scenes for planner evaluation."""
    # Where this fits:
    # These are held-out real scenes used after training.
    #
    # Method:
    # 1. Build `grid_config = make_visual_grid_config(config)`.
    # 2. Build `rng = random.Random(config.random_seed + 10_000)`.
    # 3. Repeatedly call `make_grid_scene(grid_config, rng)`.
    # 4. Return `config.num_eval_scenes` scenes.
    #
    # Checks:
    # - Use a different seed offset from training data collection.
    # - Do not evaluate only on one scene.
    grid_config = make_visual_grid_config(config)
    # Use a fixed seed so evaluation scenes repeat across runs.
    # Add `10_000` so these scenes differ from the training-data scenes.
    rng = random.Random(config.random_seed + 10_000)

    scenes: list[GridScene] = []
    for _ in range(config.num_eval_scenes):
        start_position, goal_position, hazard_positions, wall_positions = make_grid_scene(grid_config, rng)
        scenes.append(
            GridScene(
                start_position=start_position,
                goal_position=goal_position,
                hazard_positions=hazard_positions,
                wall_positions=wall_positions,
            )
        )

    return scenes

    #raise NotImplementedError("Build held-out evaluation scenes.")


def evaluate_planner(
    model: GridworldPlanningDynamicsModel,
    config: GridworldPlanningConfig,
) -> tuple[dict[str, float], list[RolloutResult], list[RolloutResult], list[GridScene]]:
    """Compare the planner with the shortest-path baseline in real scenes."""
    # Where this fits:
    # This answers the main question: did planning help real behavior?
    #
    # Input:
    # - `model`: the trained dynamics model.
    # - `config`: settings for the environment and evaluation count.
    #
    # Output:
    # Return four objects:
    # - `metrics`: one dictionary of float results.
    # - `planner_rollouts`: one rollout result per scene using the learned planner.
    # - `shortest_path_baseline_rollouts`: one rollout result per scene using
    #   the shortest-path baseline.
    # - `scenes`: the held-out scenes that were tested.
    #
    # Helper pattern:
    # You can write a small local helper such as:
    # `mean = sum(values) / len(values)`.
    #
    # For one list of rollouts:
    # - success rate is the mean of `float(rollout["reached_goal"])`.
    # - mean return is the mean of `rollout["real_return"]`.
    # - mean cost is the mean of `rollout["real_cost"]`.
    # - mean steps is the mean of `rollout["steps"]`.
    #
    # Prediction-gap example:
    # If one planner rollout has `predicted_return=0.7` and `real_return=0.5`,
    # its reward gap is `abs(0.7 - 0.5) = 0.2`.
    #
    # Method:
    # 1. Build held-out scenes with `build_eval_scenes(config)`.
    # 2. For each scene, run:
    #    - planner rollout with `use_planner=True`
    #    - shortest-path baseline rollout with `use_planner=False`
    # 3. Compute success rate, mean return, mean cost, and mean steps for both.
    # 4. For planner rollouts, compare summed one-step predictions with the
    #    real totals:
    #    - reward gap = `abs(predicted_return - real_return)`
    #    - cost gap = `abs(predicted_cost - real_cost)`
    # 5. Count an episode as overoptimistic when predicted return is higher
    #    than real return or predicted cost is lower than real cost.
    # 6. Call `audit_selected_sequence_model_error(model, scenes, config)` and
    #    merge its
    #    four metrics into the result.
    # 7. Return metrics, planner rollouts, shortest-path baseline rollouts,
    #    and scenes.
    #
    # Output metric keys:
    # - `planner_success_rate`
    # - `planner_mean_return`
    # - `planner_mean_cost`
    # - `planner_mean_episode_steps`
    # - `shortest_path_baseline_success_rate`
    # - `shortest_path_baseline_mean_return`
    # - `shortest_path_baseline_mean_cost`
    # - `shortest_path_baseline_mean_episode_steps`
    # - `planner_reward_prediction_gap`
    # - `planner_cost_prediction_gap`
    # - `planner_overoptimistic_episode_rate`
    # - `selected_sequence_score_gap`
    # - `random_sequence_score_gap`
    # - `selected_vs_random_model_error_gap`
    # - `selected_sequence_overestimate_rate`
    #
    # Checks:
    # - Use the same `scenes` list for planner and shortest-path baseline.
    # - Do not train the model here.
    # - `len(planner_rollouts) == len(shortest_path_baseline_rollouts) == len(scenes)`.
    # - Every metric value should be a plain Python float.
    scenes = build_eval_scenes(config)
    if not scenes:
        raise ValueError("No evaluation scenes were created.")

    planner_rollouts = []
    shortest_path_baseline_rollouts = []

    for scene in scenes:
        # Run the learned planner in this real scene.
        # `use_planner=True` means actions are chosen by searching inside the
        # learned model, then each chosen action is tested in the real grid-world.
        planner_rollouts.append(run_policy_in_real_gridworld(model, scene, config, use_planner=True))

        # Run the hand-written shortest-path baseline in the same real scene.
        # `use_planner=False` means actions come from the shortest-path helper,
        # not from the learned model planner.
        shortest_path_baseline_rollouts.append(
            run_policy_in_real_gridworld(model, scene, config, use_planner=False)
        )

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values))

    planner_reward_gaps = [
        abs(rollout["predicted_return"] - rollout["real_return"])
        for rollout in planner_rollouts
    ]
    planner_cost_gaps = [
        abs(rollout["predicted_cost"] - rollout["real_cost"])
        for rollout in planner_rollouts
    ]
    planner_overoptimistic = [
        rollout["predicted_return"] > rollout["real_return"]
        or rollout["predicted_cost"] < rollout["real_cost"]
        for rollout in planner_rollouts
    ]

    metrics = {
        "planner_success_rate": mean([float(rollout["reached_goal"]) for rollout in planner_rollouts]),
        "planner_mean_return": mean([rollout["real_return"] for rollout in planner_rollouts]),
        "planner_mean_cost": mean([rollout["real_cost"] for rollout in planner_rollouts]),
        "planner_mean_episode_steps": mean([float(rollout["steps"]) for rollout in planner_rollouts]),
        "shortest_path_baseline_success_rate": mean(
            [float(rollout["reached_goal"]) for rollout in shortest_path_baseline_rollouts]
        ),
        "shortest_path_baseline_mean_return": mean(
            [rollout["real_return"] for rollout in shortest_path_baseline_rollouts]
        ),
        "shortest_path_baseline_mean_cost": mean(
            [rollout["real_cost"] for rollout in shortest_path_baseline_rollouts]
        ),
        "shortest_path_baseline_mean_episode_steps": mean(
            [float(rollout["steps"]) for rollout in shortest_path_baseline_rollouts]
        ),
        "planner_reward_prediction_gap": mean(planner_reward_gaps),
        "planner_cost_prediction_gap": mean(planner_cost_gaps),
        "planner_overoptimistic_episode_rate": mean([float(value) for value in planner_overoptimistic]),
    }

    # Add the selected-sequence model-error audit metrics into the same dictionary.
    # `audit_selected_sequence_model_error(...)` returns:
    # - `selected_sequence_score_gap`: average model-vs-real score gap for the
    #   sequences the planner chose as best
    # - `random_sequence_score_gap`: average model-vs-real score gap for random
    #   sequences
    # - `selected_vs_random_model_error_gap`: selected gap minus random gap
    # - `selected_sequence_overestimate_rate`: fraction of scenes where the
    #   model gave the selected sequence a higher score than it got in reality
    metrics.update(audit_selected_sequence_model_error(model, scenes, config))

    # These returned objects answer two different questions.
    #
    # `planner_rollouts` and `shortest_path_baseline_rollouts` answer:
    # "What happened when each policy acted in the real grid-world?"
    # They store the real path, real reward, real cost, and number of steps.
    #
    # The audit metrics inside `metrics` answer:
    # "Did the planner choose action sequences because the learned model gave
    # them unrealistically good scores?"
    #
    # A planner can fail for two different reasons:
    # - it may choose actions that are bad in the real grid-world
    # - it may choose actions that only looked good inside the learned model
    #
    # Returning both rollouts and audit metrics lets the final report show both
    # the real behavior and whether the learned model was being trusted too much.
    return metrics, planner_rollouts, shortest_path_baseline_rollouts, scenes

    #raise NotImplementedError("Evaluate planner and shortest-path baseline in real scenes.")


def save_training_curves(
    train_losses: list[float],
    val_next_position_accuracies: list[float],
    config: GridworldPlanningConfig,
) -> str:
    """Save dynamics-model training curves."""
    # Goal:
    # Save `outputs/gridworld_planning/training_curves.png`.
    #
    # Method:
    # 1. Import Matplotlib inside this function:
    #    `import matplotlib`
    #    `matplotlib.use("Agg")`
    #    `import matplotlib.pyplot as plt`
    # 2. Create `config.output_dir`.
    # 3. Plot train loss and validation next-position accuracy.
    # 4. Label both axes.
    # 5. Save the figure and close it.
    #
    # Checks:
    # - The plot should show whether the learned model improved.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "training_curves.png"

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_next_position_accuracies, label="val next-position accuracy")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Planning dynamics model training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return str(path)

    #raise NotImplementedError("Save training curve plot.")


def save_rollout_plot(
    scene: GridScene,
    planner_rollout: RolloutResult,
    shortest_path_baseline_rollout: RolloutResult,
    config: GridworldPlanningConfig,
) -> str:
    """Save one comparison of planner and shortest-path baseline paths."""
    # Goal:
    # Save `outputs/gridworld_planning/planning_rollout.png`.
    #
    # Input:
    # - `scene`: the fixed starting layout.
    # - `planner_rollout["visited_positions"]`: positions visited by the planner.
    # - `shortest_path_baseline_rollout["visited_positions"]`: positions visited
    #   by the shortest-path baseline.
    #
    # Exact render call:
    # `render_grid_image(scene.start_position, scene.goal_position,
    # scene.hazard_positions, scene.wall_positions, grid_config)` returns a
    # tensor shaped `[3, image_size, image_size]`.
    #
    # Matplotlib shape note:
    # `plt.imshow(...)` expects image shape `[height, width, channels]`, so
    # convert the rendered tensor with `.permute(1, 2, 0).cpu().numpy()`.
    #
    # Coordinate note:
    # Grid positions are `(row, col)`.
    # Matplotlib line plots use x first and y second.
    # So for a path like `[(0, 0), (1, 0), (1, 1)]`:
    # - x values are columns: `[0, 0, 1]`
    # - y values are rows: `[0, 1, 1]`
    #
    # Image pixel note:
    # Each grid cell is `cell = config.image_size // config.grid_size` pixels.
    # To draw through the center of cells, plot:
    # - `x = col * cell + cell / 2`
    # - `y = row * cell + cell / 2`
    #
    # Method:
    # 1. Import Matplotlib inside this function:
    #    `import matplotlib`
    #    `matplotlib.use("Agg")`
    #    `import matplotlib.pyplot as plt`
    # 2. Create `config.output_dir`.
    # 3. Read the named positions from `scene`.
    # 4. Build `grid_config = make_visual_grid_config(config)`.
    # 5. Render the background image and convert it for Matplotlib.
    # 6. Plot the image with `plt.imshow(...)`.
    # 7. Convert planner and shortest-path baseline paths into x and y pixel-center
    #    lists.
    # 8. Overlay planner positions as one line.
    # 9. Overlay shortest-path baseline positions as another line.
    # 10. Add a legend and a plain title.
    # 11. Save to `config.output_dir / "planning_rollout.png"`.
    # 12. Close the figure.
    # 13. Return the saved path as a string.
    #
    # Checks:
    # - The plot should make it clear where the planner moved.
    # - The path lines should sit on top of cell centers, not between cells.
    # - Save the file even if both action selectors take the same path.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "planning_rollout.png"

    grid_config = make_visual_grid_config(config)
    image = render_grid_image(
        scene.start_position,
        scene.goal_position,
        scene.hazard_positions,
        scene.wall_positions,
        grid_config,
    )
    image_for_plot = image.permute(1, 2, 0).cpu().numpy()

    cell = config.image_size // config.grid_size

    def path_to_xy(positions: list[Position]) -> tuple[list[float], list[float]]:
        xs = [col * cell + cell / 2 for row, col in positions]
        ys = [row * cell + cell / 2 for row, col in positions]
        return xs, ys

    planner_xs, planner_ys = path_to_xy(planner_rollout["visited_positions"])
    shortest_path_xs, shortest_path_ys = path_to_xy(
        shortest_path_baseline_rollout["visited_positions"]
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(image_for_plot)
    plt.plot(planner_xs, planner_ys, marker="o", linewidth=2, label="planner")
    plt.plot(
        shortest_path_xs,
        shortest_path_ys,
        marker="x",
        linewidth=2,
        label="shortest-path baseline",
    )
    plt.title("Planner and shortest-path baseline rollout")
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return str(path)

    #raise NotImplementedError("Save planner rollout plot.")


def save_shortest_path_baseline_vs_planner_plot(
    metrics: dict[str, float],
    config: GridworldPlanningConfig,
) -> str:
    """Save a chart comparing planner and shortest-path baseline metrics."""
    # Goal:
    # Save `outputs/gridworld_planning/shortest_path_baseline_vs_planner.png`.
    #
    # Method:
    # 1. Build a small bar chart for success rate, mean return, and mean cost.
    # 2. Put planner and shortest-path baseline bars next to each other.
    # 3. Label the axes plainly.
    # 4. Save and close the plot.
    #
    # Checks:
    # - Higher success and return are better.
    # - Lower cost is better.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "shortest_path_baseline_vs_planner.png"

    names = ["success rate", "mean return", "mean cost"]
    planner_values = [
        metrics["planner_success_rate"],
        metrics["planner_mean_return"],
        metrics["planner_mean_cost"],
    ]
    shortest_path_baseline_values = [
        metrics["shortest_path_baseline_success_rate"],
        metrics["shortest_path_baseline_mean_return"],
        metrics["shortest_path_baseline_mean_cost"],
    ]

    xs = list(range(len(names)))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar([x - width / 2 for x in xs], planner_values, width=width, label="planner")
    plt.bar(
        [x + width / 2 for x in xs],
        shortest_path_baseline_values,
        width=width,
        label="shortest-path baseline",
    )
    plt.xticks(xs, names)
    plt.ylabel("value")
    plt.title("Planner versus shortest-path baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return str(path)

    #raise NotImplementedError("Save planner and shortest-path baseline comparison plot.")


def run_model_based_planning_study(config: GridworldPlanningConfig) -> PlanningMetrics:
    """Train the model, run planning, and save metrics and visuals."""
    # Where this fits:
    # This connects the whole exercise end to end.
    #
    # Method:
    # 1. Train the dynamics model with `train_planning_dynamics_model(config)`.
    # 2. Save training curves.
    # 3. Evaluate the planner with `evaluate_planner(model, config)`.
    # 4. Confirm the multi-step exploitation audit is present in the metrics.
    # 5. Save one rollout plot using the first scene and first rollouts.
    # 6. Save the shortest-path-baseline-vs-planner comparison plot.
    # 7. Return one `PlanningMetrics` dictionary.
    #
    # Checks:
    # - This function should return both model metrics and planning metrics.
    # - Do not report only loss.
    model, _splits, train_losses, val_next_position_accuracies, one_step_metrics = (
        train_planning_dynamics_model(config)
    )
    training_curve_path = save_training_curves(train_losses, val_next_position_accuracies, config)

    planning_metrics, planner_rollouts, shortest_path_baseline_rollouts, scenes = evaluate_planner(
        model, config
    )

    required_audit_keys = [
        "selected_sequence_score_gap",
        "random_sequence_score_gap",
        "selected_vs_random_model_error_gap",
        "selected_sequence_overestimate_rate",
    ]
    for key in required_audit_keys:
        if key not in planning_metrics:
            raise KeyError(f"Missing planner audit metric: {key}")

    if not scenes or not planner_rollouts or not shortest_path_baseline_rollouts:
        raise ValueError("Planner evaluation did not return any rollout data.")

    rollout_plot_path = save_rollout_plot(
        scenes[0],
        planner_rollouts[0],
        shortest_path_baseline_rollouts[0],
        config,
    )
    comparison_plot_path = save_shortest_path_baseline_vs_planner_plot(planning_metrics, config)

    return {
        "final_train_loss": train_losses[-1],
        "val_next_position_accuracy": one_step_metrics["val_next_position_accuracy"],
        "val_reward_mse": one_step_metrics["val_reward_mse"],
        "val_cost_mse": one_step_metrics["val_cost_mse"],
        "val_done_accuracy": one_step_metrics["val_done_accuracy"],
        "planner_success_rate": planning_metrics["planner_success_rate"],
        "planner_mean_return": planning_metrics["planner_mean_return"],
        "planner_mean_cost": planning_metrics["planner_mean_cost"],
        "planner_mean_episode_steps": planning_metrics["planner_mean_episode_steps"],
        "shortest_path_baseline_success_rate": planning_metrics[
            "shortest_path_baseline_success_rate"
        ],
        "shortest_path_baseline_mean_return": planning_metrics[
            "shortest_path_baseline_mean_return"
        ],
        "shortest_path_baseline_mean_cost": planning_metrics[
            "shortest_path_baseline_mean_cost"
        ],
        "shortest_path_baseline_mean_episode_steps": planning_metrics[
            "shortest_path_baseline_mean_episode_steps"
        ],
        "planner_reward_prediction_gap": planning_metrics["planner_reward_prediction_gap"],
        "planner_cost_prediction_gap": planning_metrics["planner_cost_prediction_gap"],
        "planner_overoptimistic_episode_rate": planning_metrics["planner_overoptimistic_episode_rate"],
        "selected_sequence_score_gap": planning_metrics["selected_sequence_score_gap"],
        "random_sequence_score_gap": planning_metrics["random_sequence_score_gap"],
        "selected_vs_random_model_error_gap": planning_metrics[
            "selected_vs_random_model_error_gap"
        ],
        "selected_sequence_overestimate_rate": planning_metrics["selected_sequence_overestimate_rate"],
        "training_curve_path": training_curve_path,
        "rollout_plot_path": rollout_plot_path,
        "comparison_plot_path": comparison_plot_path,
    }

    #raise NotImplementedError("Run the full model-based planning study.")


def explain_model_based_planning() -> str:
    """Return a short explanation of the main idea."""
    return (
        "Model-based planning means the agent uses a learned model to imagine "
        "short futures before it acts. The final test still happens in the real "
        "environment, because imagined rewards are only guesses."
    )


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def print_planning_metrics(metrics: PlanningMetrics) -> None:
    """Print final metrics in groups that are easy to read."""
    print("\nFINAL MODEL METRICS")
    print(f"final train loss:              {metrics['final_train_loss']:.4f}")
    print(
        f"validation next-position accuracy: {metrics['val_next_position_accuracy']:.3f}"
    )
    print(f"validation reward MSE:         {metrics['val_reward_mse']:.4f}")
    print(f"validation cost MSE:           {metrics['val_cost_mse']:.4f}")
    print(f"validation done accuracy:      {metrics['val_done_accuracy']:.3f}")

    print("\nREAL GRID-WORLD PERFORMANCE")
    print("planner:")
    print(f"  success rate:                {metrics['planner_success_rate']:.3f}")
    print(f"  mean real return:            {metrics['planner_mean_return']:.3f}")
    print(f"  mean real cost:              {metrics['planner_mean_cost']:.3f}")
    print(f"  mean episode steps:          {metrics['planner_mean_episode_steps']:.2f}")
    print("shortest-path baseline:")
    print(
        f"  success rate:                {metrics['shortest_path_baseline_success_rate']:.3f}"
    )
    print(
        f"  mean real return:            {metrics['shortest_path_baseline_mean_return']:.3f}"
    )
    print(f"  mean real cost:              {metrics['shortest_path_baseline_mean_cost']:.3f}")
    print(
        "  mean episode steps:          "
        f"{metrics['shortest_path_baseline_mean_episode_steps']:.2f}"
    )

    print("\nMODEL TRUST CHECKS")
    print(f"planner reward prediction gap: {metrics['planner_reward_prediction_gap']:.3f}")
    print(f"planner cost prediction gap:   {metrics['planner_cost_prediction_gap']:.3f}")
    print(f"overoptimistic episode rate:   {metrics['planner_overoptimistic_episode_rate']:.3f}")
    print(f"selected sequence score gap:   {metrics['selected_sequence_score_gap']:.3f}")
    print(f"random sequence score gap:     {metrics['random_sequence_score_gap']:.3f}")
    print(
        "selected vs random model error gap: "
        f"{metrics['selected_vs_random_model_error_gap']:.3f}"
    )
    print(f"selected overestimate rate:    {metrics['selected_sequence_overestimate_rate']:.3f}")

    print("\nSAVED PLOTS")
    print(f"training curves:               {metrics['training_curve_path']}")
    print(f"rollout plot:                  {metrics['rollout_plot_path']}")
    print(f"planner comparison plot:       {metrics['comparison_plot_path']}")


def main() -> None:
    """Run the model-based planning scaffold."""
    config = GridworldPlanningConfig()
    metrics, ok = report_task_progress("run_model_based_planning_study", run_model_based_planning_study, config)
    if ok:
        print_planning_metrics(metrics)
        print(explain_model_based_planning())


if __name__ == "__main__":
    main()
