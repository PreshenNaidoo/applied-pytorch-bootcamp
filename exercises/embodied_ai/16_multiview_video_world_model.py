"""
============================================================
EXERCISE 16: MULTIVIEW VIDEO WORLD MODEL
============================================================

WHAT YOU ARE BUILDING
You are testing whether a model predicts the next state better when it sees two
views of the same world instead of one.

The first version uses the local visual grid-world.
There is no download.

The model learns:

```text
current view or views + action -> next state
```

Single-view version:

```text
agent-local view + action -> next state
```

Multi-view version:

```text
agent-local view + full-scene view + action -> next state
```

WHY THIS EXERCISE EXISTS
Physical AI systems often use more than one camera.
One camera may miss useful information.
Another camera may see the missing part.

This exercise makes that idea concrete.

Example:

```text
agent-local view:
the agent can see itself and nearby cells, but not the far goal

full-scene view:
the model can see the whole grid layout, including the goal and hazards

next-state target:
where the agent actually moved after the action
```

If the second view contains useful information, the multi-view model should
usually predict the next state better than the single-view model.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
robot with several cameras          grid-world with two rendered views
front camera                        local crop around the agent
overhead or side camera             full top-down grid image
robot action                        grid action id
next robot state or future latent   next grid state vector
large physical AI dataset           local generated rollouts

WHY THIS TOY EXISTS
The real Cosmos-style setup uses large physical-world video and many sensors.
That is too much for a first exercise.

This toy keeps the important learning idea:

```text
extra view -> less hidden information -> better prediction
```

The toy is small, but the comparison is real:
both models train on the same rows, with the same actions and targets.
The only difference is whether the model gets one view or two views.

WHAT IS PROVIDED
- `common.embodied_gridworld` generates real stepped grid-world rollouts.
- Each logged step has:
  - `image`: full rendered grid frame
  - `action`: the action that was taken
  - `next_state`: the true state after the action

WHAT YOU CREATE IN THIS EXERCISE
- `view_0`: an agent-local crop made from `step["image"]`
- `view_1`: the full scene image from `step["image"]`
- `target_next_state`: `step["next_state"]`

WHAT IS HIDDEN
The single-view model only gets `view_0`.
It should not see `view_1`.

The multi-view model gets both views.

WHAT CHANGES DURING THE LOOP
- the environment state changes after the action
- `next_state` is the target the model must predict
- model weights change during training

WHAT PROVIDES THE TARGET
The environment provides `step["next_state"]`.
Do not invent the target with random numbers.

FULL WORKFLOW

```text
1. collect real grid-world transitions
2. turn each transition into two views
3. train a single-view model
4. train a multi-view model
5. evaluate both on the same validation rows
6. plot single-view error vs multi-view error
```

This exercise covers the multi-view prediction check.
It does not train a policy and it does not plan.

DATASET
Default dataset:
- no download
- use `common.embodied_gridworld`
- call `collect_visual_gridworld_episodes(...)`
- call `flatten_logged_episodes(...)`

Each training example should contain:

```text
views: [num_views, 3, image_size, image_size]
action: int
target_next_state: [state_dim]
```

For the first version:
- `num_views = 2`
- `state_dim = 6`
- `view_0` is a local crop around the agent, padded back to full image size
- `view_1` is the full top-down image

Optional later dataset:
- after the local version works, use a simulator or robot dataset with multiple cameras
- keep the same output tensors before changing the model

INPUTS AND OUTPUTS
One example:
- views: `[2, 3, image_size, image_size]`
- action: scalar integer
- target_next_state: `[state_dim]`

One batch:
- views: `[batch_size, 2, 3, image_size, image_size]`
- actions: `[batch_size]`
- targets: `[batch_size, state_dim]`

Single-view model input:
- views sliced to `[batch_size, 1, 3, image_size, image_size]`

Multi-view model input:
- views kept as `[batch_size, 2, 3, image_size, image_size]`

Model output:
- predicted next state: `[batch_size, state_dim]`

LOSS FUNCTION
Use mean squared error:

```text
loss = nn.functional.mse_loss(predicted_next_state, target_next_state)
```

Plain meaning:
- compare the predicted next-state numbers with the true next-state numbers
- square the difference
- average the errors
- lower is better

METRICS TO TRACK
- single-view validation MSE: lower is better
- multi-view validation MSE: lower is better
- improvement: `single_view_mse - multiview_mse`
- improvement percent: `improvement / single_view_mse`

TARGET RESULT RANGE
Do not expect a huge win every run.
The useful first check is:

```text
multi-view validation MSE <= single-view validation MSE
```

If multi-view is worse, check:
- are the two views identical by mistake?
- is `view_0` really missing information?
- are actions aligned with the right targets?
- are both models trained for the same number of epochs?
- are both models evaluated on the same validation rows?

VISUALS TO CREATE
Save `outputs/multiview_world_model/single_vs_multiview_error.png`.
Use a simple bar chart:

```text
single-view MSE | multi-view MSE
```

Also save `outputs/multiview_world_model/example_views.png`.
It should show:

```text
view_0: local crop | view_1: full scene
```

COMMON MISTAKES
- making `view_0` and `view_1` identical
- training the two models on different rows
- using random targets instead of `step["next_state"]`
- comparing training loss instead of validation loss
- forgetting to keep the action with the right transition
- flattening `[batch, views, channels, height, width]` in the wrong order
- saying this is video prediction while not using time-aligned transitions

STEP-BY-STEP TODO
1. Collect grid-world transitions.
2. Build `view_0` and `view_1` for each transition.
3. Build train and validation tensors.
4. Save an example two-view image.
5. Build the multi-view model.
6. Train one model with only `view_0`.
7. Train another model with both views.
8. Evaluate both on the same validation set.
9. Plot the validation errors.
10. Explain when extra views help.

DEFINITION OF DONE
You are done when:
- examples come from real grid-world transitions
- `view_0` and `view_1` are clearly different
- train and validation splits are built once and shared by both models
- single-view validation MSE is reported
- multi-view validation MSE is reported
- a bar chart compares the two errors
- an example two-view image is saved
- you can explain why the second view helped or did not help
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    LoggedGridStep,
    VisualGridWorldConfig,
    collect_visual_gridworld_episodes,
    flatten_logged_episodes,
)


# Learning order for this exercise:
# 1. Build the two-view dataset first.
# 2. Save `example_views.png` before training any model.
# 3. Check batch shapes.
# 4. Train the single-view model.
# 5. Train the multi-view model on the same split.
# 6. Compare validation errors and plot them.


class MultiviewDatasetSplits(TypedDict):
    """Train and validation tensors for multi-view prediction."""

    train_views: torch.Tensor
    train_actions: torch.Tensor
    train_targets: torch.Tensor
    val_views: torch.Tensor
    val_actions: torch.Tensor
    val_targets: torch.Tensor


class MultiviewMetrics(TypedDict):
    """Single-view and multi-view comparison metrics."""

    single_view_error: float
    multiview_error: float
    improvement: float
    improvement_percent: float
    error_chart_path: str
    example_views_path: str


@dataclass
class MultiviewWorldModelConfig:
    """Settings for a multi-view world-model study."""

    image_size: int = 64
    channels: int = 3
    num_views: int = 2
    num_actions: int = 4
    state_dim: int = 6
    hidden_dim: int = 128
    num_episodes: int = 96
    max_steps: int = 18
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    val_fraction: float = 0.2
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multiview_world_model"


def make_agent_local_view(image: torch.Tensor, state: torch.Tensor, config: MultiviewWorldModelConfig) -> torch.Tensor:
    """Return a local view around the agent."""
    # Where this fits:
    # This creates `view_0`, the weaker single view.
    # It should hide some far-away scene information.
    #
    # Goal:
    # Build a local crop around the agent from the full grid image.
    #
    # Input:
    # - `image`: full frame shaped `[3, image_size, image_size]`.
    # - `state`: `[state_dim]`.
    #   For `common.embodied_gridworld`, the first two numbers are normalized
    #   agent row and agent column.
    #
    # Output:
    # - local view shaped `[3, image_size, image_size]`.
    #
    # Method:
    # 1. Convert normalized agent row and column back to pixel center.
    # 2. Crop a square region around that center.
    # 3. If the crop goes outside the image, pad with white pixels.
    # 4. Place the crop back into a tensor shaped like the input image.
    #
    # Simple first version:
    # - create a white image of the same shape
    # - copy only a centered crop from the original image into the middle
    #
    # Checks:
    # - Return the same shape as `image`.
    # - The local view should not be identical to the full image for most rows.
    _ = (image, state, config)
    raise NotImplementedError("Create an agent-local view.")


def make_two_views_from_step(step: LoggedGridStep, config: MultiviewWorldModelConfig) -> torch.Tensor:
    """Return local and full-scene views for one logged step."""
    # Where this fits:
    # This converts one environment transition into the multi-view input.
    #
    # Goal:
    # Build the two views used by both models.
    #
    # Input:
    # One `LoggedGridStep` from `common.embodied_gridworld`.
    # Important keys:
    # - `step["image"]`: full current frame, `[3, image_size, image_size]`
    # - `step["state"]`: current state, `[state_dim]`
    #
    # Output:
    # - tensor shaped `[2, 3, image_size, image_size]`
    # - index 0 is the local view
    # - index 1 is the full-scene view
    #
    # Method:
    # 1. Set `full_view = step["image"]`.
    # 2. Build `local_view = make_agent_local_view(full_view, step["state"], config)`.
    # 3. Return `torch.stack([local_view, full_view])`.
    #
    # Checks:
    # - `views.shape == (2, 3, image_size, image_size)`.
    # - `views[0]` and `views[1]` should usually differ.
    _ = (step, config)
    raise NotImplementedError("Build two views from one logged step.")


def build_multiview_dataset(config: MultiviewWorldModelConfig) -> MultiviewDatasetSplits:
    """Build train and validation tensors from real grid-world transitions."""
    # Where this fits:
    # This is the shared dataset. Both single-view and multi-view models must
    # use this same split.
    #
    # Goal:
    # Collect real transitions and turn each one into:
    # `views, action, target_next_state`.
    #
    # Output:
    # Return a dictionary with:
    # - `train_views`: `[num_train, 2, 3, image_size, image_size]`
    # - `train_actions`: `[num_train]`
    # - `train_targets`: `[num_train, state_dim]`
    # - `val_views`: `[num_val, 2, 3, image_size, image_size]`
    # - `val_actions`: `[num_val]`
    # - `val_targets`: `[num_val, state_dim]`
    #
    # Method:
    # 1. Build:
    #    `grid_config = VisualGridWorldConfig(image_size=config.image_size, num_episodes=config.num_episodes, max_steps=config.max_steps, random_state=config.random_state)`.
    # 2. Collect episodes with `collect_visual_gridworld_episodes(grid_config)`.
    # 3. Flatten with `flatten_logged_episodes(episodes)`.
    # 4. For each step:
    #    - views = `make_two_views_from_step(step, config)`
    #    - action = `step["action"]`
    #    - target = `step["next_state"]`
    # 5. Stack views, actions, and targets.
    # 6. Shuffle once with `torch.Generator().manual_seed(config.random_state)`.
    # 7. Split into train and validation tensors.
    #
    # Checks:
    # - Actions and targets must stay aligned after shuffling.
    # - Do not create separate splits for the two models.
    _ = (VisualGridWorldConfig, collect_visual_gridworld_episodes, flatten_logged_episodes)
    raise NotImplementedError("Build the multi-view grid-world dataset.")


def save_example_views(views: torch.Tensor, config: MultiviewWorldModelConfig) -> str:
    """Save one local view next to one full-scene view."""
    # Where this fits:
    # This visual check happens before training.
    # It proves the two views are different.
    #
    # Input:
    # - `views`: `[2, 3, image_size, image_size]`.
    #
    # Output:
    # - path to `outputs/multiview_world_model/example_views.png`.
    #
    # Method:
    # 1. Use `matplotlib.use("Agg")`.
    # 2. Create `config.output_dir`.
    # 3. Plot `views[0]` as "local view".
    # 4. Plot `views[1]` as "full view".
    # 5. Save and return the path.
    #
    # Check:
    # - If the two views look identical, fix `make_agent_local_view(...)`.
    _ = (views, config)
    raise NotImplementedError("Save an example local/full view pair.")


class MultiviewWorldModel(nn.Module):
    """Predict next grid state from one or more image views and an action."""

    def __init__(self, config: MultiviewWorldModelConfig) -> None:
        # Where this fits:
        # This model is used twice:
        # - once with only the local view
        # - once with both views
        #
        # Goal:
        # Build a network that can accept `[batch, views, channels, height, width]`.
        #
        # Suggested structure:
        # - shared CNN view encoder:
        #   `nn.Conv2d(3, 32, kernel_size=3, padding=1)`
        #   `nn.ReLU()`
        #   `nn.AdaptiveAvgPool2d((1, 1))`
        #   `nn.Flatten()`
        #   `nn.Linear(32, hidden_dim)`
        # - action embedding:
        #   `nn.Embedding(config.num_actions, config.hidden_dim)`
        # - predictor MLP:
        #   concatenate visual feature and action feature
        #   output `[batch_size, state_dim]`
        #
        # How to combine views:
        # - encode each view with the same CNN
        # - reshape back to `[batch_size, num_views, hidden_dim]`
        # - average across the view dimension with `.mean(dim=1)`
        super().__init__()
        _ = config
        raise NotImplementedError("Build the multi-view world model.")

    def forward(self, images: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return predicted next state."""
        # Goal:
        # Predict the next state from image views and an action.
        #
        # Input:
        # - `images`: `[batch_size, num_views, 3, image_size, image_size]`
        # - `actions`: `[batch_size]`
        #
        # Output:
        # - predicted next state: `[batch_size, state_dim]`
        #
        # Method:
        # 1. Read `batch_size, num_views = images.shape[:2]`.
        # 2. Reshape images to `[batch_size * num_views, 3, image_size, image_size]`.
        # 3. Encode each view with the CNN.
        # 4. Reshape encoded views to `[batch_size, num_views, hidden_dim]`.
        # 5. Average view features with `.mean(dim=1)`.
        # 6. Embed actions with `self.action_embedding(actions)`.
        # 7. Concatenate visual features and action features.
        # 8. Return the MLP prediction.
        #
        # Checks:
        # - Passing one view should work: `images[:, :1]`.
        # - Passing two views should also work.
        _ = (images, actions)
        raise NotImplementedError("Run the multi-view world model.")


def train_prediction_model(
    model: MultiviewWorldModel,
    views: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    config: MultiviewWorldModelConfig,
) -> float:
    """Train one next-state prediction model and return final train loss."""
    # Where this fits:
    # This function is reused for the single-view and multi-view models.
    #
    # Goal:
    # Train a model to predict `target_next_state`.
    #
    # Method:
    # 1. Wrap tensors in `TensorDataset(views, actions, targets)`.
    # 2. Build `DataLoader(..., batch_size=config.batch_size, shuffle=True)`.
    # 3. Use `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 4. For each batch:
    #    - run `prediction = model(batch_views, batch_actions)`
    #    - compute `loss = nn.functional.mse_loss(prediction, batch_targets)`
    #    - call `optimizer.zero_grad()`
    #    - call `loss.backward()`
    #    - call `optimizer.step()`
    # 5. Return the final mean train loss.
    #
    # Check:
    # - This function learns. It should call `backward()` and `step()`.
    _ = (model, views, actions, targets, config, TensorDataset, DataLoader)
    raise NotImplementedError("Train one next-state prediction model.")


def evaluate_prediction_error(
    model: MultiviewWorldModel,
    views: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> float:
    """Return validation next-state MSE."""
    # Where this fits:
    # This is the fair comparison number for single-view vs multi-view.
    #
    # Goal:
    # Measure validation error without updating the model.
    #
    # Method:
    # 1. Build a DataLoader from the validation tensors.
    # 2. Call `model.eval()`.
    # 3. Use `with torch.no_grad():`.
    # 4. Compute MSE for each batch.
    # 5. Return the mean MSE.
    #
    # Check:
    # - This function should not call `backward()` or `step()`.
    _ = (model, views, actions, targets, batch_size)
    raise NotImplementedError("Evaluate next-state prediction error.")


def plot_single_vs_multiview_error(
    single_view_error: float,
    multiview_error: float,
    config: MultiviewWorldModelConfig,
) -> str:
    """Save a bar chart comparing validation errors."""
    # Where this fits:
    # This is the final visual result for the exercise.
    #
    # Goal:
    # Make it easy to see whether the second view helped.
    #
    # Output:
    # - path to `outputs/multiview_world_model/single_vs_multiview_error.png`.
    #
    # Method:
    # 1. Use `matplotlib.use("Agg")`.
    # 2. Plot two bars:
    #    - single-view MSE
    #    - multi-view MSE
    # 3. Label the y-axis `validation MSE`.
    # 4. Save and return the path.
    _ = (single_view_error, multiview_error, config)
    raise NotImplementedError("Plot single-view vs multi-view validation error.")


def run_multiview_world_model_study(config: MultiviewWorldModelConfig) -> MultiviewMetrics:
    """Compare single-view and multi-view next-state prediction."""
    # Where this fits:
    # This runs the full exercise from data to final chart.
    #
    # Goal:
    # Train two models on the same split and compare validation MSE.
    #
    # Method:
    # 1. Build `splits = build_multiview_dataset(config)`.
    # 2. Save an example view pair with `save_example_views(splits["train_views"][0], config)`.
    # 3. For the single-view model:
    #    - train with `splits["train_views"][:, :1]`
    #    - evaluate with `splits["val_views"][:, :1]`
    # 4. For the multi-view model:
    #    - train with all views
    #    - evaluate with all views
    # 5. Compute:
    #    `improvement = single_view_error - multiview_error`.
    # 6. Compute:
    #    `improvement_percent = improvement / max(single_view_error, 1e-8)`.
    # 7. Save the error bar chart.
    # 8. Return all metrics and paths.
    #
    # Checks:
    # - Both models must use the same train and validation rows.
    # - If multi-view is worse, inspect `example_views.png`.
    raise NotImplementedError("Run the multi-view world-model study.")


def explain_when_multiview_helps() -> str:
    """Return a short explanation of when extra views help."""
    return (
        "Extra views help when one view hides useful information and another "
        "view shows it. If both views show the same thing, the second view may "
        "not help much."
    )


def main() -> None:
    """Run the multi-view world-model scaffold."""
    config = MultiviewWorldModelConfig()
    try:
        metrics = run_multiview_world_model_study(config)
    except NotImplementedError as error:
        print(f"[TODO] run_multiview_world_model_study: {error}")
        return
    print(metrics)
    print(explain_when_multiview_helps())


if __name__ == "__main__":
    main()
