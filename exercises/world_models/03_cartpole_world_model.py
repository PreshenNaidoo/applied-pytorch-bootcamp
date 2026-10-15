"""
============================================================
EXERCISE 03: CARTPOLE WORLD MODEL
============================================================

WHAT YOU ARE BUILDING
You are training a small world model for Gymnasium `CartPole-v1`.

A world model predicts what the environment will do next.

In this exercise, the model learns:

```text
current observation + action -> next observation, reward, done
```

Plain meaning:
- current observation: the four CartPole numbers before the action
- action: push the cart left or right
- next observation: the four CartPole numbers after the action
- reward: the score from the environment
- done: whether the episode ended

WHY THIS EXERCISE EXISTS
Exercises `world_models/01` and `world_models/02` use the local grid-world.
This exercise moves to a real Gymnasium control task.

That matters because Gymnasium is a common API used in RL projects.
Many model-based RL systems start by learning this same kind of transition
model, then use it for planning or imagined training.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
robot or simulator                  Gymnasium CartPole-v1
sensor reading                      observation shaped `[4]`
motor command                       action `0` or `1`
future sensor reading               next observation shaped `[4]`
reward and episode stop signal      reward and done from `env.step(...)`
world model                         small PyTorch model with three heads
dataset                             logged CartPole transitions

WHY THIS IS NOT RANDOM SYNTHETIC DATA
The dataset comes from stepping the real Gymnasium `CartPole-v1` environment.

Each row is collected like this:

```text
obs, action -> env.step(action) -> next_obs, reward, terminated, truncated
```

Use:

```text
done = terminated or truncated
```

Do not create random observation tensors.
Random actions are okay for data collection because the environment still
returns real observations, rewards, and done flags.

WHAT CARTPOLE OBSERVATIONS MEAN
`CartPole-v1` returns four numbers:

```text
index  meaning
-------------------------------------
0      cart position
1      cart velocity
2      pole angle
3      pole angular velocity
```

Actions:

```text
0      push cart left
1      push cart right
```

Tiny concrete example:

```text
obs = [0.02, 0.10, -0.03, -0.20]
action = 1
next_obs = [0.03, 0.29, -0.04, -0.49]
reward = 1.0
done = False
```

INPUTS AND OUTPUTS
One model input:
- observation: `[4]`
- action one-hot: `[2]`
- model input: `[6]`

One model output:
- next observation: `[4]`
- reward: `[1]`
- done logit: `[1]`

One batch:
- inputs: `[batch_size, 6]`
- next_obs_targets: `[batch_size, 4]`
- reward_targets: `[batch_size, 1]`
- done_targets: `[batch_size, 1]`

LOSS FUNCTION
Use one loss for each prediction head:

```text
next_obs_loss = mse(predicted_next_obs, true_next_obs)
reward_loss = mse(predicted_reward, true_reward)
done_loss = binary_cross_entropy_with_logits(done_logit, done_target)

total_loss = next_obs_loss + reward_loss + done_loss
```

METRICS TO TRACK
- validation next-observation MSE: lower is better
- validation reward MSE: lower is better
- validation done accuracy: higher is better
- final training loss: lower is better

VISUALS TO CREATE
Save these files:
- `outputs/cartpole_world_model/training_curves.png`
- `outputs/cartpole_world_model/next_obs_predictions.png`
- `outputs/cartpole_world_model/done_probability_histogram.png`

TARGET RESULT RANGE
CartPole rewards are usually `1.0` for each step until the episode ends.
So reward MSE should become very small.

Next-observation MSE should fall if the model learns the physics.
Done accuracy can be noisy because most random-policy steps are not terminal.
If done accuracy looks high but the model always predicts "not done", inspect
the done probability histogram.

COMMON MISTAKES
- making random tensors instead of stepping `CartPole-v1`
- using only training loss and no validation metrics
- applying sigmoid before `binary_cross_entropy_with_logits`
- treating `terminated` and `truncated` as unrelated labels
- shuffling inputs without shuffling targets the same way
- forgetting to close the Gymnasium environment
- making no plots, which makes the model hard to inspect

STEP-BY-STEP TODO
1. Collect real CartPole transitions.
2. Encode observation-action inputs.
3. Build train and validation tensors.
4. Build the three-head world model.
5. Train the model.
6. Evaluate validation metrics.
7. Save training curves.
8. Save next-observation prediction plots.
9. Save done probability plots.
10. Explain what the model learned and what it did not learn.

DEFINITION OF DONE
You are done when:
- data comes from real Gymnasium `CartPole-v1` steps
- no CartPole episode appears in both training and validation
- the model trains end to end after you complete the stubs
- validation metrics are printed
- all three visual files are saved
- you can explain `terminated`, `truncated`, and `done`
- every completed stub keeps its original `raise NotImplementedError(...)` line commented out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"


# Learning order for this exercise:
# 1. Finish `collect_cartpole_episodes`.
# 2. Finish `encode_observation_action`.
# 3. Finish `build_cartpole_dataset`.
# 4. Build the model and run one forward pass.
# 5. Finish the loss, training loop, and validation metrics.
# 6. Finish the three plots.
# 7. Run `main()` and read the metrics.


class CartPoleTransition(TypedDict):
    """One real transition from CartPole."""

    obs: torch.Tensor
    action: int
    next_obs: torch.Tensor
    reward: float
    terminated: bool
    truncated: bool
    done: bool


class CartPoleDatasetSplits(TypedDict):
    """Train and validation tensors for the CartPole world model."""

    train_inputs: torch.Tensor
    train_next_obs: torch.Tensor
    train_rewards: torch.Tensor
    train_dones: torch.Tensor
    val_inputs: torch.Tensor
    val_next_obs: torch.Tensor
    val_rewards: torch.Tensor
    val_dones: torch.Tensor


class CartPoleWorldModelMetrics(TypedDict):
    """Final metrics and visual paths."""

    final_train_loss: float
    val_next_obs_mse: float
    val_reward_mse: float
    val_done_accuracy: float
    training_curve_path: str
    next_obs_prediction_path: str
    done_probability_path: str


@dataclass
class CartPoleWorldModelConfig:
    """Settings for the CartPole world model."""

    env_name: str = "CartPole-v1"
    obs_dim: int = 4
    num_actions: int = 2
    transitions: int = 6000
    batch_size: int = 128
    hidden_dim: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "cartpole_world_model"


def make_cartpole_env(config: CartPoleWorldModelConfig) -> Any:
    """Create one Gymnasium CartPole environment."""
    # This helper is provided because creating the environment is not the main lesson.
    # It imports Gymnasium inside the function so the file can still be imported
    # before Gymnasium is installed.
    import gymnasium as gym

    return gym.make(config.env_name)


def collect_cartpole_episodes(config: CartPoleWorldModelConfig) -> list[list[CartPoleTransition]]:
    """Collect complete CartPole episodes by stepping Gymnasium."""
    # Where this fits:
    # This is the dataset source for the whole exercise.
    #
    # Goal:
    # Return real episodes from `CartPole-v1` while keeping episode boundaries.
    #
    # Input:
    # - `config.env_name`: usually `"CartPole-v1"`.
    # - `config.transitions`: number of rows to collect.
    # - `config.random_state`: seed for repeatable collection.
    #
    # Output:
    # A list of episodes. Each episode is a list of dictionaries with:
    # - `obs`: tensor shaped `[4]`
    # - `action`: int, either `0` or `1`
    # - `next_obs`: tensor shaped `[4]`
    # - `reward`: float
    # - `terminated`: bool
    # - `truncated`: bool
    # - `done`: bool, equal to `terminated or truncated`
    #
    # Method:
    # 1. Build `env = make_cartpole_env(config)`.
    # 2. Reset with `obs, info = env.reset(seed=config.random_state)`.
    # 3. Seed the action space with `env.action_space.seed(config.random_state)`.
    # 4. Create `episodes = []`, `current_episode = []`, and a transition count.
    # 5. Until the transition count reaches `config.transitions`:
    #    - sample `action = env.action_space.sample()`
    #    - call `next_obs, reward, terminated, truncated, info = env.step(action)`
    #    - set `done = terminated or truncated`
    #    - append the transition dictionary to `current_episode`
    #    - if `done`, append `current_episode` to `episodes`, start a new
    #      empty episode, and reset the environment
    #    - otherwise set `obs = next_obs`
    # 6. If collection stops in the middle of an episode, keep that episode as
    #    one complete split unit. Do not break it into different data groups.
    # 7. Close the environment with `env.close()`.
    # 8. Return `episodes`.
    #
    # Tensor conversion:
    # Use `torch.tensor(obs, dtype=torch.float32)` and the same for `next_obs`.
    #
    # Checks:
    # - Do not create fake random observations.
    # - Every observation tensor should have shape `(4,)`.
    # - The total number of rows should equal `config.transitions`.

    raise NotImplementedError("Collect real CartPole episodes.")








def encode_observation_action(obs: torch.Tensor, action: int, config: CartPoleWorldModelConfig) -> torch.Tensor:
    """Encode one observation and one action as one model input."""
    # Where this fits:
    # This creates one input row from one transition.
    #
    # Goal:
    # Concatenate the four observation numbers with a one-hot action vector.
    #
    # Input:
    # - `obs`: tensor shaped `[4]`.
    # - `action`: integer `0` or `1`.
    #
    # Output:
    # - tensor shaped `[6]`.
    #
    # Method:
    # 1. Check `obs.shape == (config.obs_dim,)`.
    # 2. Check `0 <= action < config.num_actions`.
    # 3. Build `action_one_hot = torch.zeros(config.num_actions, dtype=torch.float32)`.
    # 4. Set `action_one_hot[action] = 1.0`.
    # 5. Return `torch.cat([obs.float(), action_one_hot], dim=0)`.
    #
    # Checks:
    # - The output shape should be `(6,)`.
    # - The final two values should contain exactly one `1.0`.

    raise NotImplementedError("Encode one CartPole observation-action pair.")






def build_cartpole_dataset(config: CartPoleWorldModelConfig) -> CartPoleDatasetSplits:
    """Build train and validation tensors from real CartPole transitions."""
    # Where this fits:
    # This turns environment logs into tensors for supervised learning.
    #
    # Goal:
    # Build aligned train and validation splits at episode level.
    #
    # Method:
    # 1. Call `collect_cartpole_episodes(config)`.
    # 2. Shuffle and split whole episode indices by `config.val_fraction`.
    # 3. Flatten the selected train episodes and validation episodes
    #    separately with a list comprehension.
    # 4. For each transition in each group, build:
    #    - input from `encode_observation_action(...)`
    #    - next observation from `transition["next_obs"]`
    #    - reward tensor shaped `[1]`
    #    - done tensor shaped `[1]`, with `1.0` for done and `0.0` otherwise
    # 5. Stack each group separately. Convert rewards and dones with
    #    `torch.tensor(...).unsqueeze(1)`.
    #
    # Checks:
    # - Inputs and targets must stay aligned.
    # - No CartPole episode may appear in both groups.
    # - Do not recollect validation data separately with a different policy.

    raise NotImplementedError("Build CartPole train and validation tensors.")











class CartPoleWorldModel(nn.Module):
    """Predict next observation, reward, and done."""

    def __init__(self, config: CartPoleWorldModelConfig) -> None:
        # Where this fits:
        # This is the learned world model.
        #
        # Goal:
        # Build a shared trunk with three prediction heads.
        #
        # Method:
        # 1. Call `super().__init__()`.
        # 2. Set `input_dim = config.obs_dim + config.num_actions`.
        # 3. Build:
        #    `self.trunk = nn.Sequential(nn.Linear(input_dim, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU())`
        # 4. Build:
        #    `self.next_obs_head = nn.Linear(config.hidden_dim, config.obs_dim)`
        #    `self.reward_head = nn.Linear(config.hidden_dim, 1)`
        #    `self.done_head = nn.Linear(config.hidden_dim, 1)`
        #
        # Checks:
        # - Do not apply sigmoid in the done head.
        # - `binary_cross_entropy_with_logits` expects raw logits.
        super().__init__()

        raise NotImplementedError("Build the CartPole world model.")




    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return next observation, reward, and done logit."""
        # Goal:
        # Run the model input through the trunk and three heads.
        #
        # Input:
        # - `inputs`: `[batch_size, 6]`.
        #
        # Output:
        # - next observation: `[batch_size, 4]`
        # - reward: `[batch_size, 1]`
        # - done logit: `[batch_size, 1]`
        #
        # Method:
        # 1. Run `hidden = self.trunk(inputs)`.
        # 2. Run all three heads on `hidden`.
        # 3. Return `(pred_next_obs, pred_reward, pred_done_logit)`.

        raise NotImplementedError("Run the CartPole world model.")




def compute_world_model_loss(
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return total world-model loss and its parts."""
    # Where this fits:
    # This tells the model what a bad prediction means.
    #
    # Method:
    # 1. Unpack predictions into `pred_next_obs`, `pred_reward`, `pred_done_logit`.
    # 2. Unpack targets into `next_obs`, `rewards`, `dones`.
    # 3. Compute:
    #    `next_obs_loss = nn.functional.mse_loss(pred_next_obs, next_obs)`.
    # 4. Compute:
    #    `reward_loss = nn.functional.mse_loss(pred_reward, rewards)`.
    # 5. Compute:
    #    `done_loss = nn.functional.binary_cross_entropy_with_logits(pred_done_logit, dones)`.
    # 6. Add the three losses.
    # 7. Return `(total_loss, loss_parts)`.
    #
    # Checks:
    # - `dones` should contain floats shaped `[batch_size, 1]`.
    # - Do not apply `torch.sigmoid` before the loss.

    raise NotImplementedError("Compute CartPole world-model loss.")





def train_one_epoch(model: CartPoleWorldModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train for one epoch and return mean total loss."""
    # Where this fits:
    # This updates the model weights using the training split.
    #
    # Method:
    # 1. Call `model.train()`.
    # 2. Loop over batches from `loader`.
    # 3. Move every tensor to `device`.
    # 4. Run the model.
    # 5. Compute loss with `compute_world_model_loss(...)`.
    # 6. Call `optimizer.zero_grad()`.
    # 7. Call `loss.backward()`.
    # 8. Call `optimizer.step()`.
    # 9. Return mean loss weighted by batch size.


    raise NotImplementedError("Train one CartPole world-model epoch.")





@torch.no_grad()
def evaluate_cartpole_world_model(
    model: CartPoleWorldModel,
    splits: CartPoleDatasetSplits,
) -> dict[str, float]:
    """Return validation metrics without updating the model."""
    # Where this fits:
    # This checks whether the model predicts held-out transitions.
    #
    # Method:
    # 1. Call `model.eval()`.
    # 2. Use `torch.no_grad()`.
    # 3. Move validation tensors to `device`.
    # 4. Run the model.
    # 5. Compute validation next-observation MSE.
    # 6. Compute validation reward MSE.
    # 7. Convert done logits with `torch.sigmoid(...)`.
    # 8. Compute done accuracy using threshold `0.5`.
    #
    # Output keys:
    # - `val_next_obs_mse`
    # - `val_reward_mse`
    # - `val_done_accuracy`
    #
    # Check:
    # - This function should not call `backward()` or `optimizer.step()`.

    raise NotImplementedError("Evaluate CartPole world-model validation metrics.")


   


def save_training_curves(
    train_losses: list[float],
    val_next_obs_mses: list[float],
    config: CartPoleWorldModelConfig,
) -> str:
    """Save train loss and validation next-observation MSE curves."""
    # Goal:
    # Save `outputs/cartpole_world_model/training_curves.png`.
    #
    # Method:
    # 1. Import Matplotlib inside this function:
    #    `import matplotlib`
    #    `matplotlib.use("Agg")`
    #    `import matplotlib.pyplot as plt`
    # 2. Create `config.output_dir`.
    # 3. Plot train loss and validation next-observation MSE by epoch.
    # 4. Label axes and add a legend.
    # 5. Save the figure and close it.
    raise NotImplementedError("Save CartPole training curve plot.")






def save_next_obs_prediction_plot(
    model: CartPoleWorldModel,
    splits: CartPoleDatasetSplits,
    config: CartPoleWorldModelConfig,
) -> str:
    """Save a true-vs-predicted next-observation plot."""
    # Goal:
    # Save `outputs/cartpole_world_model/next_obs_predictions.png`.
    #
    # Method:
    # 1. Run the model on validation inputs with `torch.no_grad()`.
    # 2. Move predictions and targets to CPU.
    # 3. Make a scatter plot:
    #    - x-axis: true next-observation values
    #    - y-axis: predicted next-observation values
    # 4. Add a diagonal reference line so perfect predictions are easy to see.
    # 5. Save and close the plot.
    #
    # Check:
    # - Good predictions should sit close to the diagonal.
    raise NotImplementedError("Save CartPole next-observation prediction plot.")









def save_done_probability_histogram(
    model: CartPoleWorldModel,
    splits: CartPoleDatasetSplits,
    config: CartPoleWorldModelConfig,
) -> str:
    """Save a histogram of predicted done probabilities."""
    # Goal:
    # Save `outputs/cartpole_world_model/done_probability_histogram.png`.
    #
    # Why this visual exists:
    # Done labels are imbalanced in CartPole. Most steps are not done.
    # Accuracy alone can hide a model that always predicts "not done".
    #
    # Method:
    # 1. Run the model on validation inputs with `torch.no_grad()`.
    # 2. Convert done logits to probabilities with `torch.sigmoid(...)`.
    # 3. Plot two histograms:
    #    - probabilities for rows where true done is `0`
    #    - probabilities for rows where true done is `1`
    # 4. Save and close the plot.
    raise NotImplementedError("Save CartPole done probability histogram.")








def train_cartpole_world_model(config: CartPoleWorldModelConfig) -> CartPoleWorldModelMetrics:
    """Train the CartPole world model and save metrics and visuals."""
    # Where this fits:
    # This connects the full exercise.
    #
    # Method:
    # 1. Call `set_seed(config.random_state)`.
    # 2. Build `splits = build_cartpole_dataset(config)`.
    # 3. Build a training `TensorDataset`.
    # 4. Build `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`.
    # 5. Build `CartPoleWorldModel(config).to(device)`.
    # 6. Build `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 7. Train and evaluate for `config.epochs`.
    # 8. Save all three visual files.
    # 9. Return a `CartPoleWorldModelMetrics` dictionary.
    #
    # Checks:
    # - Return validation metrics, not only training loss.
    # - Return paths for all saved visuals.
    
    raise NotImplementedError("Train and evaluate the CartPole world model.")













def explain_cartpole_world_model() -> str:
    """Return a short explanation of what this model learns."""
    return (
        "This world model learns one-step CartPole dynamics from real "
        "Gymnasium transitions. It predicts the next observation, reward, "
        "and whether the episode ended after a chosen action."
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


def main() -> None:
    """Run the CartPole world-model scaffold."""
    config = CartPoleWorldModelConfig()
    metrics, ok = report_task_progress("train_cartpole_world_model", train_cartpole_world_model, config)
    if ok:
        print(explain_cartpole_world_model())


if __name__ == "__main__":
    main()
