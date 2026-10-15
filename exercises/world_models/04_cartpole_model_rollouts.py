"""
============================================================
EXERCISE 04: CARTPOLE MODEL ROLLOUTS
============================================================

WHAT YOU ARE BUILDING
You are training a CartPole world model and checking how it behaves over many
future steps.

A one-step world model learns:

```text
current observation + action -> next observation
```

A rollout check asks a harder question:

```text
if the model keeps using its own predictions, how fast does it drift away
from the real environment?
```

Plain meaning:
- one-step error checks one prediction
- rollout error checks many predictions in a row
- drift means the imagined path slowly moves away from the real path

WHY THIS EXERCISE EXISTS
the related exercise trains a one-step CartPole world model.
This exercise shows why one-step validation loss is not enough.

For planning, the model is often used for several imagined steps.
A model can look good for one step but become bad after 10 or 20 imagined
steps because each small mistake becomes the next input.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
robot or simulator                  Gymnasium CartPole-v1
sensor reading                      observation shaped `[4]`
action sequence                     list of left/right pushes
learned world model                 small PyTorch transition model
real rollout                        Gymnasium rollout with fixed actions
imagined rollout                    model rollout with the same fixed actions
drift report                        error by future step and saved plots

WHY THIS IS NOT RANDOM SYNTHETIC DATA
The data comes from stepping the real Gymnasium `CartPole-v1` environment.

Training rows come from real transitions:

```text
obs, action -> env.step(action) -> next_obs, reward, done
```

Rollout targets also come from real environment rollouts.
The action list can be sampled, but the observations and targets must come
from Gymnasium.

WHAT THE MODEL KNOWS
At each imagined step, the model receives:
- the current observation vector
- one action id

WHAT IS HIDDEN
The model does not see the real future observation during imagined rollout.
It must feed its own predicted next observation into the next step.

That is why drift can grow.

FULL WORKFLOW

```text
1. collect real CartPole training transitions
2. train a one-step world model
3. collect held-out real action rollouts
4. replay the same actions inside the learned model
5. measure error at each future step
6. save drift curves and rollout comparison plots
```

DATASET
Default dataset:
- no download
- use Gymnasium `CartPole-v1`
- collect transitions and rollouts by calling `env.step(action)`

One transition row contains:

```text
obs: [4]
action: int, 0 or 1
next_obs: [4]
reward: float
done: bool
```

One rollout example contains:

```text
start_obs: [4]
actions: [rollout_horizon]
real_obs: [rollout_horizon + 1, 4]
```

Tiny concrete example:

```text
real_obs[0] = [0.02, 0.10, -0.03, -0.20]
actions = [1, 1, 0]
real_obs[1] comes from env.step(1)
imagined_obs[1] comes from the model
error[1] compares those two observations
```

INPUTS AND OUTPUTS
Model input:
- observation: `[4]`
- one-hot action: `[2]`
- concatenated input: `[6]`

Model output:
- next observation: `[4]`

Rollout output:
- real observations: `[horizon + 1, 4]`
- imagined observations: `[horizon + 1, 4]`
- error curve: `[horizon + 1]`

LOSS FUNCTION
Use next-observation mean squared error:

```text
loss = mean((predicted_next_obs - true_next_obs) ** 2)
```

ROLLOUT ERROR
At each time step:

```text
error_t = mean((real_obs_t - imagined_obs_t) ** 2)
```

METRICS TO TRACK
- validation one-step MSE: lower is better
- free-running MSE at horizons 1, 3, 5, and 10: lower is better
- mean rollout MSE: lower is better
- final-step rollout MSE: lower is better
- drift ratio: final-step MSE divided by step-1 MSE

VISUALS TO CREATE
Save these files:
- `outputs/cartpole_rollout_drift/training_curves.png`
- `outputs/cartpole_rollout_drift/rollout_error_curve.png`
- `outputs/cartpole_rollout_drift/real_vs_imagined_observation.png`

COMMON MISTAKES
- using an untrained placeholder model
- measuring only one-step loss
- using random tensors instead of real CartPole rollouts
- comparing rollouts that used different action sequences
- feeding real observations into every imagined step
- forgetting that `imagined_obs[0]` and `real_obs[0]` should match
- not saving plots, which makes drift hard to inspect

STEP-BY-STEP TODO
1. Collect complete real CartPole episodes.
2. Split episodes, then build train and validation tensors.
3. Train a one-step next-observation model.
4. Collect held-out real rollouts with fixed action lists.
5. Replay the same action lists inside the learned model.
6. Compute rollout error curves.
7. Summarize drift metrics.
8. Save training and drift visuals.
9. Explain why rollout drift matters for planning.

DEFINITION OF DONE
You are done when:
- training data comes from real Gymnasium `CartPole-v1` steps
- held-out rollout targets come from real Gymnasium `CartPole-v1` rollouts
- no episode appears in both training and validation
- the model trains after you complete the stubs
- validation one-step MSE is reported
- free-running MSE is reported separately at horizons 1, 3, 5, and 10
- all three visual files are saved
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
# 1. Finish the transition dataset helpers.
# 2. Train the one-step model.
# 3. Collect held-out real rollouts.
# 4. Create imagined rollouts from the trained model.
# 5. Compute drift metrics.
# 6. Save the three plots.


class CartPoleTransition(TypedDict):
    """One real CartPole transition."""

    obs: torch.Tensor
    action: int
    next_obs: torch.Tensor
    done: bool


class CartPoleRollout(TypedDict):
    """One real CartPole rollout with a fixed action list."""

    start_obs: torch.Tensor
    actions: list[int]
    real_obs: torch.Tensor
    done_step: int | None


class RolloutDatasetSplits(TypedDict):
    """Train and validation tensors for the one-step model."""

    train_inputs: torch.Tensor
    train_next_obs: torch.Tensor
    val_inputs: torch.Tensor
    val_next_obs: torch.Tensor


class RolloutDriftMetrics(TypedDict):
    """Final metrics and saved visual paths."""

    final_train_loss: float
    val_one_step_mse: float
    rollout_mse_h1: float
    rollout_mse_h3: float
    rollout_mse_h5: float
    rollout_mse_h10: float
    mean_rollout_mse: float
    final_step_mse: float
    drift_ratio: float
    training_curve_path: str
    rollout_error_curve_path: str
    observation_plot_path: str


@dataclass
class CartPoleRolloutDriftConfig:
    """Settings for CartPole rollout-drift checks."""

    env_name: str = "CartPole-v1"
    obs_dim: int = 4
    num_actions: int = 2
    transitions: int = 6000
    rollout_count: int = 24
    rollout_horizon: int = 25
    batch_size: int = 128
    hidden_dim: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "cartpole_rollout_drift"


def make_cartpole_env(config: CartPoleRolloutDriftConfig) -> Any:
    """Create one Gymnasium CartPole environment."""
    # This helper is complete because environment creation is not the lesson.
    import gymnasium as gym

    return gym.make(config.env_name)


def encode_obs_action(obs: torch.Tensor, action: int, config: CartPoleRolloutDriftConfig) -> torch.Tensor:
    """Encode one observation and action for the world model."""
    # Where this fits:
    # This creates one model input row.
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
    # - The final two values should contain exactly one `1.0`.
    raise NotImplementedError("Encode one CartPole observation-action pair.")



def collect_cartpole_episodes(config: CartPoleRolloutDriftConfig) -> list[list[CartPoleTransition]]:
    """Collect real CartPole episodes for one-step training."""
    # Where this fits:
    # This is the training dataset source.
    #
    # Goal:
    # Step `CartPole-v1` and keep real transitions grouped by episode.
    #
    # Method:
    # 1. Build `env = make_cartpole_env(config)`.
    # 2. Reset with `obs, info = env.reset(seed=config.random_state)`.
    # 3. Seed the action space with `env.action_space.seed(config.random_state)`.
    # 4. Create an episode list and one current-episode list.
    # 5. Loop until you have `config.transitions` rows.
    # 6. Sample an action from `env.action_space.sample()`.
    # 7. Step the environment.
    # 8. Store `obs`, `action`, `next_obs`, and `done` in the current episode.
    # 9. After done, save the episode, start a new list, and reset.
    # 10. Keep a final partial episode as one split unit, then close the environment.
    #
    # Checks:
    # - Do not create random observation tensors.
    # - Every row must come from `env.step(...)`.
    raise NotImplementedError("Collect real CartPole episodes for rollout training.")






def build_rollout_dataset(config: CartPoleRolloutDriftConfig) -> RolloutDatasetSplits:
    """Build train and validation tensors for one-step prediction."""
    # Where this fits:
    # This turns real transitions into supervised training data.
    #
    # Method:
    # 1. Call `collect_cartpole_episodes(config)`.
    # 2. Shuffle and split whole episodes by `config.val_fraction`.
    # 3. Flatten train episodes and validation episodes separately.
    # 4. Build one input with `encode_obs_action(row["obs"], row["action"], config)`.
    # 5. Use `row["next_obs"]` as the target.
    # 6. Stack train tensors and validation tensors separately.
    #
    # Checks:
    # - Inputs and targets must stay aligned.
    # - No episode may appear in both splits.
    raise NotImplementedError("Build CartPole rollout train and validation tensors.")






class CartPoleNextObsModel(nn.Module):
    """Predict the next CartPole observation."""

    def __init__(self, config: CartPoleRolloutDriftConfig) -> None:
        # Where this fits:
        # This model is rolled forward many times during drift checks.
        #
        # Goal:
        # Build a small MLP that maps `[obs, one_hot_action]` to `next_obs`.
        #
        # Method:
        # 1. Call `super().__init__()`.
        # 2. Set `input_dim = config.obs_dim + config.num_actions`.
        # 3. Build `self.net = nn.Sequential(...)` with:
        #    - `nn.Linear(input_dim, config.hidden_dim)`
        #    - `nn.ReLU()`
        #    - `nn.Linear(config.hidden_dim, config.hidden_dim)`
        #    - `nn.ReLU()`
        #    - `nn.Linear(config.hidden_dim, config.obs_dim)`
        super().__init__()
        raise NotImplementedError("Build the CartPole next-observation model.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predicted next observations."""
        # Input:
        # - `inputs`: `[batch_size, 6]`.
        #
        # Output:
        # - predicted next observations: `[batch_size, 4]`.
        raise NotImplementedError("Run the CartPole next-observation model.")


def train_one_epoch(model: CartPoleNextObsModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train one epoch and return mean MSE loss."""
    # Method:
    # 1. Call `model.train()`.
    # 2. Loop over `inputs, next_obs` batches.
    # 3. Move tensors to `device`.
    # 4. Run `pred_next_obs = model(inputs)`.
    # 5. Compute `loss = nn.functional.mse_loss(pred_next_obs, next_obs)`.
    # 6. Clear gradients, backpropagate, and step the optimizer.
    # 7. Return mean loss weighted by batch size.
    raise NotImplementedError("Train one rollout-drift model epoch.")




def evaluate_one_step_mse(model: CartPoleNextObsModel, splits: RolloutDatasetSplits) -> float:
    """Return validation one-step MSE."""
    # Method:
    # 1. Call `model.eval()`.
    # 2. Use `torch.no_grad()`.
    # 3. Run the model on `splits["val_inputs"]`.
    # 4. Compare with `splits["val_next_obs"]`.
    # 5. Return the MSE as a Python float.
    #
    # Check:
    # - This function should not update model weights.
    raise NotImplementedError("Evaluate one-step validation MSE.")


def train_next_obs_model(
    config: CartPoleRolloutDriftConfig,
) -> tuple[CartPoleNextObsModel, RolloutDatasetSplits, list[float], list[float]]:
    """Train the one-step model used for rollout drift."""
    # Method:
    # 1. Call `set_seed(config.random_state)`.
    # 2. Build `splits = build_rollout_dataset(config)`.
    # 3. Build a training `TensorDataset` and `DataLoader`.
    # 4. Build `CartPoleNextObsModel(config).to(device)`.
    # 5. Build `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 6. For each epoch:
    #    - call `train_one_epoch(...)`
    #    - call `evaluate_one_step_mse(...)`
    #    - append both values to history lists
    # 7. Return the model, splits, train losses, and validation MSEs.
    raise NotImplementedError("Train the CartPole next-observation model.")





def collect_real_rollout(seed: int, actions: list[int], config: CartPoleRolloutDriftConfig) -> CartPoleRollout:
    """Collect one real rollout by replaying a fixed action list."""
    # Where this fits:
    # This creates the real target path for a rollout-drift check.
    #
    # Method:
    # 1. Build `env = make_cartpole_env(config)`.
    # 2. Reset with `obs, info = env.reset(seed=seed)`.
    # 3. Store the reset observation first.
    # 4. Step the real environment with each action in `actions`.
    # 5. Store every next observation.
    # 6. If the environment ends early, record the done step and stop.
    # 7. Close the environment.
    # 8. Return a `CartPoleRollout` dictionary.
    #
    # Checks:
    # - Do not call the learned model here.
    # - The first observation is the start observation.
    raise NotImplementedError("Collect one real CartPole rollout.")




def build_action_rollouts(config: CartPoleRolloutDriftConfig) -> list[CartPoleRollout]:
    """Build held-out real rollouts with fixed action lists."""
    # Where this fits:
    # These rollouts are held-out tests for drift.
    #
    # Method:
    # 1. Create a `torch.Generator().manual_seed(config.random_state + 10_000)`.
    # 2. For each rollout:
    #    - sample `config.rollout_horizon` actions with `torch.randint(...)`
    #    - convert them to a Python list
    #    - call `collect_real_rollout(seed, actions, config)`
    # 3. Return the rollout list.
    #
    # Checks:
    # - Use held-out seeds, not the same reset seed as training.
    # - Each rollout should use the same actions for real and imagined paths later.
    raise NotImplementedError("Build held-out CartPole rollouts.")

    # Default config makes 24 separate held-out rollouts.
    # Each rollout samples 25 actions, so each real path can have up to
    # 26 observations: the start observation plus one next observation per action.



def imagined_rollout(
    model: CartPoleNextObsModel,
    start_obs: torch.Tensor,
    actions: list[int],
    config: CartPoleRolloutDriftConfig,
) -> torch.Tensor:
    """Roll the learned model forward with a fixed action list."""
    # Where this fits:
    # This is the imagined path.
    #
    # Method:
    # 1. Check `start_obs.shape == (config.obs_dim,)`.
    # 2. Set `obs = start_obs.float()`.
    # 3. Store `obs` as the first imagined observation.
    # 4. For each action:
    #    - encode `obs` and action
    #    - add a batch dimension with `.unsqueeze(0)`
    #    - run the model with `torch.no_grad()`
    #    - remove the batch dimension with `.squeeze(0)`
    #    - use the predicted next observation as the next `obs`
    # 5. Stack and return all imagined observations.
    #
    # Checks:
    # - Do not call Gymnasium here.
    # - Do not feed real future observations into the imagined rollout.
    raise NotImplementedError("Build one imagined CartPole rollout.")





def rollout_error_mse(real_obs: torch.Tensor, imagined_obs: torch.Tensor) -> torch.Tensor:
    """Return one MSE value per rollout step."""
    # Formula:
    # `error_t = mean((real_obs_t - imagined_obs_t) ** 2)`.
    #
    # Method:
    # 1. If the two rollouts have different lengths, compare only the shared prefix.
    # 2. Compute squared difference.
    # 3. Average over observation dimension `1`.
    # 4. Return a tensor shaped `[shared_steps]`.
    #
    # Check:
    # - Step 0 error should be close to 0 because both paths start from the same observation.
    raise NotImplementedError("Compute one rollout error curve.")


def evaluate_rollout_drift(
    model: CartPoleNextObsModel,
    real_rollouts: list[CartPoleRollout],
    config: CartPoleRolloutDriftConfig,
) -> tuple[dict[str, float], list[torch.Tensor], list[torch.Tensor]]:
    """Evaluate rollout drift over many held-out rollouts."""
    # Method:
    # 1. For each real rollout:
    #    - build imagined observations with `imagined_rollout(...)`
    #    - compute one MSE value per step with `rollout_error_mse(...)`
    # 2. Pad or stack the curves in a clear way so you can average by step.
    # 3. Average the free-running error at exact horizons 1, 3, 5, and 10.
    #    Horizon 1 uses the first prediction after the shared start state.
    #    If a rollout ended before a requested horizon, leave it out of that
    #    horizon's average rather than padding it with a made-up value.
    # 4. Compute:
    #    - `mean_rollout_mse`
    #    - `final_step_mse`
    #    - `drift_ratio = final_step_mse / max(step_1_mse, 1e-8)`
    #    - `rollout_mse_h1`, `rollout_mse_h3`, `rollout_mse_h5`, `rollout_mse_h10`
    # 5. Return metrics, error curves, and imagined observation rollouts.
    #
    # Checks:
    # - The metric must be based on real-vs-imagined observations.
    # - Do not compare two imagined rollouts.
    raise NotImplementedError("Evaluate CartPole rollout drift.")

    # For each real test rollout, make the model imagine the same rollout.
    # Both paths start from the same start observation and use the same actions.
    # The per-step errors say how far the model's imagined observations drift from the real ones.
    # Each `errors_per_step` tensor has one value per shared observation step.
    # `errors_per_step` has the same length as the shared part of the two rollouts.
    # For the default full 25-action rollout, that is 26 values:
    # one error for the start observation, then one error after each action.
    # Shared means the steps that both rollouts actually have.
    # Example: if the model imagines 26 observations but the real CartPole
    # rollout ended after 12 observations, only the first 12 observations are
    # compared because there is no real observation after that.


    # `values_at_step(3)` gets the step-3 error from every rollout.
    # Step 3 means: after replaying 3 actions inside the model.




def save_training_curves(train_losses: list[float], val_mses: list[float], config: CartPoleRolloutDriftConfig) -> str:
    """Save one-step training curves."""
    # Goal:
    # Save `outputs/cartpole_rollout_drift/training_curves.png`.
    #
    # Method:
    # Use Matplotlib with `matplotlib.use("Agg")`.
    # Plot training MSE and validation one-step MSE by epoch.
    raise NotImplementedError("Save rollout-drift training curves.")






def save_rollout_error_curve(error_curves: list[torch.Tensor], config: CartPoleRolloutDriftConfig) -> str:
    """Save mean rollout error by future step."""
    # Goal:
    # Save `outputs/cartpole_rollout_drift/rollout_error_curve.png`.
    #
    # Method:
    # 1. Average error curves by future step.
    # 2. Plot future step on the x-axis.
    # 3. Plot mean MSE on the y-axis.
    # 4. Save and close the figure.
    raise NotImplementedError("Save rollout error curve plot.")







def save_observation_rollout_plot(
    rollout: CartPoleRollout,
    imagined_obs: torch.Tensor,
    config: CartPoleRolloutDriftConfig,
) -> str:
    """Save a real-vs-imagined observation plot for one rollout."""
    # Goal:
    # Save `outputs/cartpole_rollout_drift/real_vs_imagined_observation.png`.
    #
    # Method:
    # 1. Compare the shared prefix of `rollout["real_obs"]` and `imagined_obs`.
    # 2. Plot at least pole angle, which is observation index `2`.
    # 3. Show real and imagined values as separate lines.
    # 4. Save and close the plot.
    #
    # Check:
    # - This plot should make drift visible over time.
    raise NotImplementedError("Save real-vs-imagined observation plot.")






def run_cartpole_rollout_drift_study(config: CartPoleRolloutDriftConfig) -> RolloutDriftMetrics:
    """Train the model, measure drift, and save visuals."""
    # Method:
    # 1. Train the model with `train_next_obs_model(config)`.
    # 2. Build held-out real rollouts with `build_action_rollouts(config)`.
    # 3. Evaluate rollout drift with `evaluate_rollout_drift(...)`.
    # 4. Save the three visual files.
    # 5. Return one `RolloutDriftMetrics` dictionary.
    #
    # Checks:
    # - Return both one-step and rollout metrics.
    # - Do not report only training loss.
    raise NotImplementedError("Run the CartPole rollout-drift study.")






def explain_rollout_drift() -> str:
    """Return a short explanation of rollout drift."""
    return (
        "Rollout drift happens when a world model feeds its own predicted "
        "observation back into the next prediction. Small one-step mistakes can "
        "compound, which matters for planning because planning uses imagined "
        "multi-step futures."
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
    """Run the CartPole rollout-drift scaffold."""
    config = CartPoleRolloutDriftConfig()
    metrics, ok = report_task_progress("run_cartpole_rollout_drift_study", run_cartpole_rollout_drift_study, config)
    if ok:
        print(explain_rollout_drift())


if __name__ == "__main__":
    main()
