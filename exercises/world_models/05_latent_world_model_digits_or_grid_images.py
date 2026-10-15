"""
============================================================
EXERCISE 05: LATENT WORLD MODEL DIGITS OR GRID IMAGES
============================================================

WHAT YOU ARE BUILDING
You are building a tiny image-based world model.

The model learns from real logged image transitions from the shared visual
grid-world:

```text
current frame + action -> next frame
```

It does this through a smaller latent vector:

```text
frame -> latent
latent + action -> next latent
next latent -> next frame
```

Plain meaning:
- frame: the RGB image the agent sees now
- latent: a small learned vector that stores useful frame information
- action: the move the agent takes
- next frame: the RGB image after the action

WHY THIS EXERCISE EXISTS
Exercises `world_models/01` to `world_models/04` use clean state vectors.
This exercise moves toward visual world models.

Predicting full images directly is harder than predicting clean state numbers.
A latent model first compresses an image into a smaller vector, predicts in
that smaller space, then decodes the result back to an image.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
robot/game camera                   RGB grid-world frame
motor or keyboard action            action id 0, 1, 2, or 3
large visual encoder                small PyTorch encoder
compact world state                 learned latent vector
future camera frame                 decoded next grid-world frame
video dataset                       logged rollouts from common.embodied_gridworld

WHY THIS IS NOT RANDOM SYNTHETIC DATA
The data comes from stepping the shared environment in
`common.embodied_gridworld`.

Each row is a real logged transition:

```text
image, action -> next_image
```

Do not create random image tensors.
Do not use a separate toy renderer inside this file.

WHAT THE MODEL KNOWS
For each training row, the model receives:
- the current RGB frame shaped `[3, image_size, image_size]`
- the action id

WHAT THE MODEL MUST PREDICT
The model predicts:
- a reconstruction of the current frame
- a predicted next latent
- a decoded next frame

WHAT PROVIDES THE TARGET
The environment log provides:
- `step["image"]`
- `step["action"]`
- `step["next_image"]`

FULL WORKFLOW

```text
1. collect logged image/action transitions
2. split them into train and validation sets
3. encode current and next frames into latents
4. reconstruct current frames
5. predict next latents from current latents and actions
6. decode predicted next latents into predicted next frames
7. track validation losses
8. save visual previews
```

DATASET
Default dataset:
- no download
- use `common.embodied_gridworld`
- call `collect_visual_gridworld_episodes(...)`
- call `flatten_logged_episodes(...)`

One logged row contains:

```text
image: [3, image_size, image_size]
action: int
next_image: [3, image_size, image_size]
```

INPUTS AND OUTPUTS
One example:
- frame: `[3, image_size, image_size]`
- action one-hot: `[4]`
- latent: `[latent_dim]`
- predicted next frame: `[3, image_size, image_size]`

One batch:
- frames: `[batch_size, 3, image_size, image_size]`
- actions: `[batch_size, 4]`
- next_frames: `[batch_size, 3, image_size, image_size]`

LOSS FUNCTION
Use three losses:

```text
reconstruction_loss = mse(reconstructed_frame, current_frame)
latent_prediction_loss = mse(predicted_next_latent, encoded_next_frame.detach())
next_frame_loss = mse(decoded_predicted_next_frame, true_next_frame)

total_loss = reconstruction_loss + latent_prediction_loss + next_frame_loss
```

Why `.detach()` appears:
- the encoded next frame is used as a target for the dynamics head
- `.detach()` stops that target from being changed by the dynamics loss

METRICS TO TRACK
- validation reconstruction MSE: lower is better
- validation latent prediction MSE: lower is better
- validation next-frame MSE: lower is better
- action-branch MSE: lower means predictions follow different actions correctly
- action-branch changed-pixel MSE: checks the small image areas that really moved
- action-branch sensitivity ratio: predicted branch separation divided by real
  branch separation; near `1.0` is useful and near `0.0` means actions are ignored
- final training loss: lower is better

VISUALS TO CREATE
Save these files:
- `outputs/latent_gridworld_images/training_curves.png`
- `outputs/latent_gridworld_images/reconstruction_preview.png`
- `outputs/latent_gridworld_images/next_frame_prediction_preview.png`
- `outputs/latent_gridworld_images/action_branch_preview.png`

COMMON MISTAKES
- using random image tensors instead of environment frames
- training without a validation split
- predicting next frames but never saving visual previews
- comparing predicted next frame to the current frame by mistake
- forgetting to one-hot encode actions
- letting validation data leak into training
- testing each action from a different frame instead of holding the frame fixed

STEP-BY-STEP TODO
1. Collect complete logged image episodes.
2. Split episodes, then build train and validation tensors.
3. Build the encoder, dynamics model, and decoder.
4. Train reconstruction and next-frame prediction.
5. Evaluate validation metrics.
6. Test all actions from the same held-out frame.
7. Save training curves and all three prediction previews.
8. Explain why latent prediction is useful for visual world models.

DEFINITION OF DONE
You are done when:
- frames come from `common.embodied_gridworld`
- no episode appears in both training and validation
- the model trains after you complete the stubs
- validation metrics are printed
- same-frame action-branch MSE is reported
- changed-pixel error and branch sensitivity are reported so unchanged background
  pixels cannot hide an action-ignoring model
- all four visual files are saved
- you can explain what the latent vector is doing
- every completed stub keeps its original `raise NotImplementedError(...)` line commented out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
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
    make_grid_scene,
    render_grid_image,
    step_grid_world,
)
from common.reproducibility import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"


# Learning order for this exercise:
# 1. Finish the data helpers.
# 2. Build the latent image model.
# 3. Finish the loss and training loop.
# 4. Finish validation metrics.
# 5. Save the three visual outputs.


class ImageWorldModelSplits(TypedDict):
    """Train and validation tensors for the latent image world model."""

    train_frames: torch.Tensor
    train_actions: torch.Tensor
    train_next_frames: torch.Tensor
    val_frames: torch.Tensor
    val_actions: torch.Tensor
    val_next_frames: torch.Tensor


class LatentImageMetrics(TypedDict):
    """Final metrics and visual paths."""

    final_train_loss: float
    val_reconstruction_mse: float
    val_latent_prediction_mse: float
    val_next_frame_mse: float
    heldout_action_branch_mse: float
    heldout_action_branch_changed_pixel_mse: float
    heldout_action_branch_sensitivity_ratio: float
    training_curve_path: str
    reconstruction_preview_path: str
    next_frame_preview_path: str
    action_branch_preview_path: str


@dataclass
class LatentWorldModelConfig:
    """Settings for the latent image world model."""

    image_size: int = 28
    image_channels: int = 3
    num_actions: int = 4
    latent_dim: int = 32
    hidden_dim: int = 256
    num_episodes: int = 256
    max_steps: int = 18
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    action_branch_scenes: int = 24
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "latent_gridworld_images"


def collect_logged_image_episodes(config: LatentWorldModelConfig) -> list[list[LoggedGridStep]]:
    """Collect complete real image episodes from the shared grid-world."""
    # Where this fits:
    # This is the dataset source for the whole exercise.
    #
    # Goal:
    # Return complete image/action episodes from `common.embodied_gridworld`.
    #
    # Method:
    # 1. Build:
    #    `grid_config = VisualGridWorldConfig(image_size=config.image_size, num_episodes=config.num_episodes, max_steps=config.max_steps, random_state=config.random_state)`.
    # 2. Collect:
    #    `episodes = collect_visual_gridworld_episodes(grid_config)`.
    # 3. Check the list and every kept episode are not empty.
    # 4. Return `episodes` without flattening.
    #
    # Checks:
    # - Do not create random images.
    # - `step["image"]` and `step["next_image"]` should come from environment logs.
    _ = (VisualGridWorldConfig, collect_visual_gridworld_episodes)

    grid_config = VisualGridWorldConfig(image_size=config.image_size, 
                                        num_episodes=config.num_episodes, 
                                        max_steps=config.max_steps, 
                                        random_state=config.random_state)

    episodes = collect_visual_gridworld_episodes(grid_config)

    return episodes
    
    #raise NotImplementedError("Collect complete logged grid-world image episodes.")


def one_hot_action(action: int, config: LatentWorldModelConfig) -> torch.Tensor:
    """Encode one action id as a one-hot vector."""
    # Where this fits:
    # The dynamics model needs action information beside the image latent.
    #
    # Goal:
    # Convert action `0`, `1`, `2`, or `3` into a tensor shaped `[4]`.
    #
    # Method:
    # 1. Check `0 <= action < config.num_actions`.
    # 2. Build `encoded = torch.zeros(config.num_actions, dtype=torch.float32)`.
    # 3. Set `encoded[action] = 1.0`.
    # 4. Return `encoded`.
    #
    # Check:
    # - The returned vector should sum to `1.0`.

    if not 0 <= action < config.num_actions:
        raise ValueError("Inavlid action")

    encoded = torch.zeros(config.num_actions, dtype=torch.float)
    encoded[action] = 1.0
    return encoded
    #raise NotImplementedError("One-hot encode one grid-world action.")


def build_image_world_model_splits(config: LatentWorldModelConfig) -> ImageWorldModelSplits:
    """Build train and validation tensors from logged image transitions."""
    # Where this fits:
    # This turns real rollout rows into tensors for training.
    #
    # Method:
    # 1. Call `collect_logged_image_episodes(config)`.
    # 2. Shuffle and split whole episode indices by `config.val_fraction`.
    # 3. Flatten train episodes and validation episodes separately with
    #    `flatten_logged_episodes(...)`.
    # 4. Stack `step["image"]` and `step["next_image"]` separately for each group.
    # 5. Build one-hot action tensors with `one_hot_action(...)`.
    # 6. Stack actions into `[num_examples, num_actions]`.
    #
    # Checks:
    # - No episode may appear in both groups.
    # - `frames.shape[1:]` should be `[3, image_size, image_size]`.
    # - Action rows should sum to `1.0`.

    episodes = collect_logged_image_episodes(config)

    num_val = int(len(episodes) * config.val_fraction)

    shuffled_indices = torch.randperm(n=len(episodes), generator = torch.Generator().manual_seed(config.random_state))

    val_eps = [episodes[shuffled_indices[index]] for index in range(0, num_val)]
    train_eps = [episodes[shuffled_indices[index]] for index in range(num_val, len(episodes))]

    val_flat_eps = flatten_logged_episodes(val_eps) #[step for episode in val_eps for step in episode]
    train_flat_eps = flatten_logged_episodes(train_eps) #[step for episode in train_eps for step in episode]

    def build_tensors(steps: list[LoggedGridStep]) -> list[torch.tensor]:

        images = []
        actions = []
        next_images = []

        for step in steps:
            images.append(step["image"])
            actions.append(one_hot_action(step["action"], config))
            next_images.append(step["next_image"])

        return torch.stack(images, dim=0), torch.stack(actions, dim=0), torch.stack(next_images, dim=0)

    train_tensors = build_tensors(train_flat_eps)
    val_tensors = build_tensors(val_flat_eps)

    return ImageWorldModelSplits(train_frames=train_tensors[0], train_actions=train_tensors[1], train_next_frames=train_tensors[2],
                                val_frames=val_tensors[0], val_actions=val_tensors[1], val_next_frames=val_tensors[2])

    #raise NotImplementedError("Build latent image world-model train and validation tensors.")


class LatentWorldModel(nn.Module):
    """Encode frames, predict next latents, and decode frames."""

    def __init__(self, config: LatentWorldModelConfig) -> None:
        # Where this fits:
        # This is the visual world model.
        #
        # Goal:
        # Build an encoder, dynamics model, and decoder.
        #
        # Method:
        # 1. Call `super().__init__()`.
        # 2. Store `config` as `self.config`.
        # 3. Compute:
        #    `flat_image_dim = config.image_channels * config.image_size * config.image_size`.
        # 4. Build `self.encoder`:
        #    `nn.Flatten()`, Linear, ReLU, Linear to `config.latent_dim`.
        # 5. Build `self.dynamics`:
        #    Linear from `config.latent_dim + config.num_actions`, ReLU, Linear to `config.latent_dim`.
        # 6. Build `self.decoder`:
        #    Linear, ReLU, Linear back to `flat_image_dim`, `nn.Sigmoid()`.
        #
        # Why `nn.Sigmoid()` is used:
        # Image pixels are between `0.0` and `1.0`, so sigmoid keeps decoded pixels in that range.
        super().__init__()

        self.config = config

        self.flat_image_dim = config.image_channels * config.image_size * config.image_size

        self.encoder = nn.Sequential(nn.Flatten(), 
                                    nn.Linear(in_features=self.flat_image_dim, out_features=config.hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=config.hidden_dim, out_features=config.latent_dim))

        self.dynamics = nn.Sequential(nn.Linear(in_features=config.latent_dim + config.num_actions, out_features=config.hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=config.hidden_dim, out_features=config.latent_dim))

        self.decoder = nn.Sequential(nn.Linear(in_features=config.latent_dim, out_features=config.hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=config.hidden_dim, out_features=self.flat_image_dim),
                                    nn.Sigmoid())

        #raise NotImplementedError("Build the latent image world model.")

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Return latent vectors for frames."""
        # Input:
        # - `frames`: `[batch_size, 3, image_size, image_size]`.
        #
        # Output:
        # - latents: `[batch_size, latent_dim]`.
        #
        # Method:
        # 1. Check that `frames` has 4 dimensions.
        # 2. Return `self.encoder(frames)`.

        if len(frames.shape) != 4:
            raise ValueError("Shape of frames is incorrect.")

        return self.encoder(frames)
        # raise NotImplementedError("Encode image frames into latents.")

    def predict_next_latent(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict next latent vectors from current latents and actions."""
        # Input:
        # - `latents`: `[batch_size, latent_dim]`.
        # - `actions`: `[batch_size, num_actions]`.
        #
        # Output:
        # - predicted next latents: `[batch_size, latent_dim]`.
        #
        # Method:
        # 1. Check that batch sizes match.
        # 2. Build `dynamics_input = torch.cat([latents, actions], dim=-1)`.
        # 3. Return `self.dynamics(dynamics_input)`.

        dynamics_input = torch.cat([latents, actions], dim=-1)
        return self.dynamics(dynamics_input)
        # raise NotImplementedError("Predict next image latent.")

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents into image frames."""
        # Input:
        # - `latents`: `[batch_size, latent_dim]`.
        #
        # Output:
        # - frames: `[batch_size, 3, image_size, image_size]`.
        #
        # Method:
        # 1. Run `flat = self.decoder(latents)`.
        # 2. Reshape with:
        #    `flat.view(latents.shape[0], self.config.image_channels, self.config.image_size, self.config.image_size)`.
        # 3. Return the frames.

        flat = self.decoder(latents)
        frames = flat.view(latents.shape[0], self.config.image_channels, self.config.image_size, self.config.image_size)
        return frames
        #raise NotImplementedError("Decode image latents into frames.")

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, predicted next latent, and predicted next frame."""
        # Method:
        # 1. Encode current frames.
        # 2. Decode current latents to reconstruct frames.
        # 3. Predict next latents from current latents and actions.
        # 4. Decode predicted next latents into predicted next frames.
        # 5. Return `(reconstructed_frames, predicted_next_latents, predicted_next_frames)`.

        encoded_current_frames = self.encode(frames)
        reconstructed_current_frames = self.decode(encoded_current_frames)

        predicted_next_latents = self.predict_next_latent(encoded_current_frames, actions)
        predicted_next_frames = self.decode(predicted_next_latents)

        return reconstructed_current_frames, predicted_next_latents, predicted_next_frames

        #raise NotImplementedError("Run the latent image world model.")


def compute_latent_world_model_loss(
    model: LatentWorldModel,
    frames: torch.Tensor,
    actions: torch.Tensor,
    next_frames: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return total latent world-model loss and its parts."""
    # Where this fits:
    # This is the training objective.
    #
    # Method:
    # 1. Run the model on `frames` and `actions`.
    # 2. Compute reconstruction MSE by comparing the rebuilt current frames
    #    with the original current frames.
    # 3. Encode `next_frames` with `model.encode(next_frames)`. These encoded
    #    next-frame latents are the target for the predicted next latents.
    # 4. Compute latent prediction MSE against `next_latents.detach()`.
    # 5. Compute next-frame MSE.
    # 6. Add all three losses.
    # 7. Return `(total_loss, loss_parts)`.
    #
    # Check:
    # - The next-frame loss must compare predicted next frames to `next_frames`.

    reconstructed_current_frames, predicted_next_latents, predicted_next_frames = model(frames, actions)
    
    reconstruction_mse = nn.functional.mse_loss(reconstructed_current_frames, frames)

    next_latents = model.encode(next_frames)

    latent_pred_mse = nn.functional.mse_loss(predicted_next_latents, next_latents.detach())
    next_frame_mse = nn.functional.mse_loss(predicted_next_frames, next_frames)

    loss = reconstruction_mse + latent_pred_mse + next_frame_mse

    loss_parts = {
        "reconstruction_mse": float(reconstruction_mse.detach().cpu().item()),
        "latent_prediction_mse": float(latent_pred_mse.detach().cpu().item()),
        "next_frame_mse": float(next_frame_mse.detach().cpu().item()),
    }

    return loss, loss_parts

    #raise NotImplementedError("Compute latent image world-model loss.")


def train_one_epoch(model: LatentWorldModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train for one epoch and return mean total loss."""
    # Method:
    # 1. Call `model.train()`.
    # 2. Loop over `frames, actions, next_frames`.
    # 3. Move every tensor to `device`.
    # 4. Compute loss with `compute_latent_world_model_loss(...)`.
    # 5. Clear gradients, backpropagate, and step the optimizer.
    # 6. Return mean loss weighted by batch size.

    model.train()

    total_loss = 0
    num_samples = 0

    for frames, actions, next_frames in loader:
        frames = frames.to(device)
        actions = actions.to(device)
        next_frames = next_frames.to(device)

        optimizer.zero_grad()

        loss, loss_parts = compute_latent_world_model_loss(model, frames, actions, next_frames)
        loss.backward()
        optimizer.step()

        batch_size = frames.shape[0]
        num_samples+=batch_size
        total_loss += loss.item() * batch_size

    return total_loss/num_samples
    #raise NotImplementedError("Train one latent image world-model epoch.")

@torch.no_grad()
def evaluate_latent_world_model(model: LatentWorldModel, splits: ImageWorldModelSplits) -> dict[str, float]:
    """Return validation metrics without updating the model."""
    # Method:
    # 1. Call `model.eval()`.
    # 2. Use `torch.no_grad()`.
    # 3. Move validation frames, actions, and next frames to `device`.
    # 4. Compute the same three loss parts as training.
    # 5. Return:
    #    - `val_reconstruction_mse`
    #    - `val_latent_prediction_mse`
    #    - `val_next_frame_mse`
    #
    # Check:
    # - This function should not call `backward()` or `optimizer.step()`.
    model.eval()

    frames = splits["val_frames"].to(device)
    actions = splits["val_actions"].to(device)
    next_frames = splits["val_next_frames"].to(device)

    _, loss_parts = compute_latent_world_model_loss(model, frames, actions, next_frames)

    return {
        "val_reconstruction_mse": loss_parts["reconstruction_mse"],
        "val_latent_prediction_mse": loss_parts["latent_prediction_mse"],
        "val_next_frame_mse": loss_parts["next_frame_mse"],
    }

    #raise NotImplementedError("Evaluate latent image world-model validation metrics.")


def build_action_branch_batch(
    config: LatentWorldModelConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build counterfactual next-frame targets for every action from each scene."""
    # Where this fits:
    # A useful action-conditioned model must predict different correct futures
    # when only the proposed action changes.
    #
    # Output:
    # - current frames: `[action_branch_scenes * num_actions, 3, H, W]`
    # - one-hot actions: `[action_branch_scenes * num_actions, num_actions]`
    # - real next frames: `[action_branch_scenes * num_actions, 3, H, W]`
    #
    # Method:
    # 1. Build a new `VisualGridWorldConfig(...)` for held-out test scenes.
    #    Use the same image size and grid settings, but use a different seed
    #    from training by adding `20_000` to `config.random_state`.
    # 2. Build `rng = random.Random(config.random_state + 20_000)`.
    #    This gives repeatable held-out scenes.
    # 3. For each held-out scene:
    #    - call `make_grid_scene(grid_config, rng)`
    #    - keep the returned `agent`, `goal`, `hazards`, and `walls`
    #    - render the current frame once with `render_grid_image(...)`
    # 4. From that same unchanged scene, try every possible action.
    #    For `num_actions = 4`, this means:
    #    - action 0 from the original scene
    #    - action 1 from the original scene
    #    - action 2 from the original scene
    #    - action 3 from the original scene
    # 5. For each action:
    #    - call `step_grid_world(agent, action, goal, hazards, walls, grid_config)`
    #    - use the returned `next_agent` to render the true next frame
    #    - append the same current frame again
    #    - append `one_hot_action(action, config)`
    #    - append the true next frame for that action
    # 6. Stack the saved current frames, actions, and true next frames.
    #
    # Small example:
    # If one scene starts with the agent in the middle, this helper creates four
    # rows from that one scene:
    #
    # ```text
    # same current frame + up    -> true next frame after up
    # same current frame + down  -> true next frame after down
    # same current frame + left  -> true next frame after left
    # same current frame + right -> true next frame after right
    # ```
    #
    # Checks:
    # - Reset to the same original scene before every action branch.
    # - Do not feed one action's next state into the next action branch.    

    grid_config = VisualGridWorldConfig(image_size=config.image_size, 
                                        num_episodes=config.num_episodes, 
                                        max_steps=config.max_steps, 
                                        random_state=config.random_state + 20000)

    rng = random.Random(config.random_state + 20_000)

    current_frames = []
    actions = []
    next_frames = []

    for scene_index in range(config.action_branch_scenes):
        scene = make_grid_scene(grid_config, rng)
        agent = scene[0]
        goals = scene[1]
        hazards = scene[2]
        walls = scene[3]

        rendered_curr_frame = render_grid_image(agent, goals, hazards, walls, grid_config)

        for action_id in range(config.num_actions):
            next_agent, reward, cost, done = step_grid_world(agent, action_id, goals, hazards, walls, grid_config)
            next_rendered_frame = render_grid_image(next_agent, goals, hazards, walls, grid_config)

            current_frames.append(rendered_curr_frame)
            actions.append(one_hot_action(action_id, config))
            next_frames.append(next_rendered_frame)

    return torch.stack(current_frames, dim=0), torch.stack(actions, dim=0), torch.stack(next_frames, dim=0)
    #raise NotImplementedError("Build held-out counterfactual action branches.")


def measure_action_branch_quality(
    current_frames: torch.Tensor,
    predicted_next_frames: torch.Tensor,
    true_next_frames: torch.Tensor,
    config: LatentWorldModelConfig,
) -> dict[str, float]:
    """Measure branch accuracy without letting static pixels dominate."""
    # Input:
    # Each group of `config.num_actions` rows shares one current frame and is
    # ordered by action id.
    #
    # Method:
    # 1. Compute ordinary prediction MSE over every pixel.
    #    Compare the model prediction with the real target:
    #    `(predicted_next_frames - true_next_frames).pow(2).mean()`.
    #    This uses every RGB pixel, including pixels that did not change.
    # 2. Build a changed-pixel mask with
    #    `(true_next_frames - current_frames).abs().amax(dim=1, keepdim=True) > 1e-6`.
    # 3. Compute changed-pixel MSE only on pixels where the true next frame
    #    changed from the current frame.
    #    - First compute squared prediction error:
    #      `(predicted_next_frames - true_next_frames).pow(2)`.
    #    - Expand `change_mask` so it has the same shape as the squared error.
    #    - Select only values where the expanded mask is true.
    #    - Average those selected values.
    #    If a row has no changed pixels, skip that row because there is no
    #    changed area to measure for that example.
    # 4. Group the flat batch back into scenes.
    #    `build_action_branch_batch(...)` saved rows in this order:
    #
    #    ```text
    #    scene 0, action 0
    #    scene 0, action 1
    #    scene 0, action 2
    #    scene 0, action 3
    #    scene 1, action 0
    #    scene 1, action 1
    #    ...
    #    ```
    #
    #    So the first dimension is:
    #
    #    ```text
    #    batch_size = num_scenes * num_actions
    #    ```
    #
    #    Compute:
    #    `num_scenes = batch_size // config.num_actions`.
    #
    #    Then reshape predictions and targets from:
    #
    #    ```text
    #    [batch_size, 3, H, W]
    #    ```
    #
    #    to:
    #
    #    ```text
    #    [num_scenes, num_actions, 3 * H * W]
    #    ```
    #
    # 5. Compare action branches inside each scene.
    #    For one scene, there are `num_actions` predicted next frames and
    #    `num_actions` true next frames.
    #
    #    Use `torch.pdist(scene_predictions)` to measure how different the
    #    predicted action branches are from each other.
    #    Use `torch.pdist(scene_targets)` to measure how different the true
    #    action branches are from each other.
    #
    #    Example with 4 actions:
    #
    #    ```text
    #    compare up vs down
    #    compare up vs left
    #    compare up vs right
    #    compare down vs left
    #    compare down vs right
    #    compare left vs right
    #    ```
    #
    #    Average those pair distances for predictions and for targets.
    #
    # 6. Return a dictionary with:
    #    - `"action_branch_mse"`: ordinary prediction MSE over every pixel
    #    - `"action_branch_changed_pixel_mse"`: prediction MSE only where the
    #      true next frame changed from the current frame
    #    - `"action_branch_sensitivity_ratio"`:
    #      `predicted_pair_distance / true_pair_distance`
    #
    # Interpretation:
    # - A ratio near zero means the model gives almost the same future for every action.
    # - A ratio near one means its branch separation is similar to the real world.
    # - A very large ratio means it reacts too strongly, so MSE must still be checked.
    # - Skip a scene when its true pair distance is zero.

    mse = torch.pow(predicted_next_frames - true_next_frames, 2).mean().item()

    # true_next_frames.shape == [batch_size, 3, H, W]
    # current_frames.shape == [batch_size, 3, H, W]

    # amax returns the largest value.
    # argmax returns the index position of the largest value.
    change_mask = (true_next_frames - current_frames).abs().amax(dim=1, keepdim=True) > 1e-6 # [batch_size, 1, H, W]  True or false mask indicating change or not
    # true_next_frames - current_frames -> [batch_size, 3, H, W]
    # .abs()                            -> [batch_size, 3, H, W]
    # .amax(dim=1, keepdim=True)        -> [batch_size, 1, H, W]
    # > 1e-6                            -> [batch_size, 1, H, W]

    squared_error = (predicted_next_frames - true_next_frames).pow(2)           # [batch_size, 3, H, W]
    changed_pixel_mses = []

    for row_index in range(squared_error.shape[0]):
        row_mask = change_mask[row_index].expand_as(squared_error[row_index])   # [3, H, W]
        if not row_mask.any():
            continue
        # Keep only the squared errors for pixels that truly changed, then average them.
        changed_pixel_mses.append(squared_error[row_index][row_mask].mean())    # scalar

    if changed_pixel_mses:
        changed_pixel_mse = torch.stack(changed_pixel_mses).mean().item()       # scalar float
    else:
        changed_pixel_mse = 0.0

    batch_size, channels, height, width = predicted_next_frames.shape
    num_scenes = batch_size // config.num_actions

    reshaped_pred_next_frames = predicted_next_frames.reshape(num_scenes, config.num_actions, channels * height * width)
    reshaped_true_next_frames = true_next_frames.reshape(num_scenes, config.num_actions, channels * height * width)

    # this is computed per scene:
    total_predicted_pair_distance = 0
    total_true_pair_distance = 0
    for scene_index in range(reshaped_pred_next_frames.shape[0]): #i.e. num_scenes
        scene_predictions = reshaped_pred_next_frames[scene_index]
        scene_targets = reshaped_true_next_frames[scene_index]

        # torch.pdist(x) computes the distance between every pair of rows in a 2D tensor.
        # Example:
        # x = torch.tensor([  [0.0, 0.0],
        #                     [3.0, 4.0],
        #                     [6.0, 8.0], ])
        # torch.pdist(x)
        #
        # this compares:
        # row 0 vs row 1
        # row 0 vs row 2
        # row 1 vs row 2
        # 
        # the result is [5.0, 10.0, 5.0]
        # because
        # distance between [0, 0] and [3, 4] = 5
        # distance between [0, 0] and [6, 8] = 10
        # distance between [3, 4] and [6, 8] = 5
        #
        # In this exercise, one scene has one predicted next frame for each action:
        # up prediction
        # down prediction
        # left prediction
        # right prediction
        # After flattening each image into a long vector, torch.pdist(...) measures how different those action branches are from each other.
        # So it helps answer:
        # Does the model predict different futures for different actions?
        # In this exercise, the model should predict different next frames for different actions, when the current frame is the same.
        # If the model ignores the action, then all four predictions may look almost the same. That would be a problem 
        # because it means the model has not learned the action-conditioned part of the world.

        pred_pair_dist = torch.pdist(scene_predictions)
        true_pair_dist = torch.pdist(scene_targets)

        mean_true_pair_distance = true_pair_dist.mean().item()
        if mean_true_pair_distance == 0.0:
            # Skip scenes where the real actions all lead to the same frame.
            # There is no true branch separation to compare against.
            # Example: the agent is boxed in by walls or grid edges. 
            # If every action hits a wall or boundary, the agent does 
            # not move, so every action gives the same next frame.
            continue

        total_predicted_pair_distance += pred_pair_dist.mean().item()
        total_true_pair_distance += mean_true_pair_distance

    sensitivity_ratio = (
        total_predicted_pair_distance / total_true_pair_distance
        if total_true_pair_distance > 0.0
        else 0.0
    )

    return {"action_branch_mse": mse, "action_branch_changed_pixel_mse": changed_pixel_mse, "action_branch_sensitivity_ratio": sensitivity_ratio}

    #raise NotImplementedError("Measure action-branch accuracy and sensitivity.")


@torch.no_grad()
def evaluate_action_branches(
    model: LatentWorldModel,
    config: LatentWorldModelConfig,
) -> dict[str, float]:
    """Evaluate real next-frame branches for every action from the same frame."""
    # Method:
    # 1. Call `build_action_branch_batch(config)`.
    # 2. Move the tensors to `device`.
    # 3. Run the same current scenes with every possible action.
    # 4. Pass current, predicted, and real frames to
    #    `measure_action_branch_quality(...)`.
    # 5. Return all three metrics as plain floats.
    #
    # Check:
    # - A low ordinary validation loss is not enough. This metric checks that
    #   the action changes the prediction in the correct way.
    model.eval()

    current_frames, actions, true_next_frames = build_action_branch_batch(config)
    current_frames = current_frames.to(device)
    actions = actions.to(device)
    true_next_frames = true_next_frames.to(device)

    _, _, predicted_next_frames = model(current_frames, actions)

    return measure_action_branch_quality(
        current_frames=current_frames,
        predicted_next_frames=predicted_next_frames,
        true_next_frames=true_next_frames,
        config=config,
    )

    #raise NotImplementedError("Evaluate counterfactual action-conditioned predictions.")


def save_training_curves(train_losses: list[float], val_next_frame_mses: list[float], config: LatentWorldModelConfig) -> str:
    """Save training loss and validation next-frame MSE curves."""
    # Goal:
    # Save `outputs/latent_gridworld_images/training_curves.png`.
    #
    # Method:
    # Use Matplotlib with `matplotlib.use("Agg")`.
    # Plot train loss and validation next-frame MSE by epoch.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "training_curves.png"

    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="train loss")
    ax.plot(epochs, val_next_frame_mses, label="val next-frame MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Latent image world-model training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)

    #raise NotImplementedError("Save latent image training curves.")


def save_reconstruction_preview(model: LatentWorldModel, splits: ImageWorldModelSplits, config: LatentWorldModelConfig) -> str:
    """Save current-frame reconstruction examples."""
    # Goal:
    # Save `outputs/latent_gridworld_images/reconstruction_preview.png`.
    #
    # Method:
    # 1. Run the model on a small validation batch.
    # 2. Show original current frames and reconstructed current frames.
    # 3. Save and close the figure.
    #
    # Check:
    # - The preview should show whether the latent keeps useful visual information.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "reconstruction_preview.png"

    model.eval()
    num_examples = min(6, splits["val_frames"].shape[0])
    frames = splits["val_frames"][:num_examples].to(device)

    with torch.no_grad():
        reconstructed_frames, _, _ = model(frames, torch.zeros(num_examples, config.num_actions, device=device))

    originals = frames.detach().cpu()
    reconstructions = reconstructed_frames.detach().cpu()

    fig, axes = plt.subplots(2, num_examples, figsize=(2 * num_examples, 4))
    for index in range(num_examples):
        axes[0, index].imshow(originals[index].permute(1, 2, 0).clamp(0, 1))
        axes[0, index].set_title("original")
        axes[0, index].axis("off")
        axes[1, index].imshow(reconstructions[index].permute(1, 2, 0).clamp(0, 1))
        axes[1, index].set_title("rebuilt")
        axes[1, index].axis("off")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)

    #raise NotImplementedError("Save reconstruction preview.")


def save_next_frame_prediction_preview(model: LatentWorldModel, splits: ImageWorldModelSplits, config: LatentWorldModelConfig) -> str:
    """Save next-frame prediction examples."""
    # Goal:
    # Save `outputs/latent_gridworld_images/next_frame_prediction_preview.png`.
    #
    # Method:
    # 1. Run the model on a small validation batch.
    # 2. Show true next frames and predicted next frames.
    # 3. Save and close the figure.
    #
    # Check:
    # - The preview should compare predicted next frame to true next frame, not current frame.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "next_frame_prediction_preview.png"

    model.eval()
    num_examples = min(6, splits["val_frames"].shape[0])
    frames = splits["val_frames"][:num_examples].to(device)
    actions = splits["val_actions"][:num_examples].to(device)
    true_next_frames = splits["val_next_frames"][:num_examples].to(device)

    with torch.no_grad():
        _, _, predicted_next_frames = model(frames, actions)

    targets = true_next_frames.detach().cpu()
    predictions = predicted_next_frames.detach().cpu()

    fig, axes = plt.subplots(2, num_examples, figsize=(2 * num_examples, 4))
    for index in range(num_examples):
        axes[0, index].imshow(targets[index].permute(1, 2, 0).clamp(0, 1))
        axes[0, index].set_title("true next")
        axes[0, index].axis("off")
        axes[1, index].imshow(predictions[index].permute(1, 2, 0).clamp(0, 1))
        axes[1, index].set_title("pred next")
        axes[1, index].axis("off")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)

    #raise NotImplementedError("Save next-frame prediction preview.")


def save_action_branch_preview(model: LatentWorldModel, config: LatentWorldModelConfig) -> str:
    """Save predicted and real next frames for every action from one scene."""
    # Goal:
    # Save `outputs/latent_gridworld_images/action_branch_preview.png`.
    #
    # Method:
    # 1. Use the first scene returned by `build_action_branch_batch(config)`.
    # 2. Run all `config.num_actions` branches through the model.
    # 3. Make one column per action.
    # 4. Show predicted next frames in one row and real next frames below them.
    # 5. Put the action id in each title, then save and close the figure.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "action_branch_preview.png"

    model.eval()
    current_frames, actions, true_next_frames = build_action_branch_batch(config)
    num_actions = config.num_actions
    current_frames = current_frames[:num_actions].to(device)
    actions = actions[:num_actions].to(device)
    true_next_frames = true_next_frames[:num_actions].to(device)

    with torch.no_grad():
        _, _, predicted_next_frames = model(current_frames, actions)

    predictions = predicted_next_frames.detach().cpu()
    targets = true_next_frames.detach().cpu()

    fig, axes = plt.subplots(2, num_actions, figsize=(2 * num_actions, 4))
    for action_id in range(num_actions):
        axes[0, action_id].imshow(predictions[action_id].permute(1, 2, 0).clamp(0, 1))
        axes[0, action_id].set_title(f"pred action {action_id}")
        axes[0, action_id].axis("off")
        axes[1, action_id].imshow(targets[action_id].permute(1, 2, 0).clamp(0, 1))
        axes[1, action_id].set_title(f"true action {action_id}")
        axes[1, action_id].axis("off")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)

    #raise NotImplementedError("Save counterfactual action-branch preview.")


def train_latent_world_model(config: LatentWorldModelConfig) -> LatentImageMetrics:
    """Train the latent image world model and save metrics and visuals."""
    # Method:
    # 1. Call `set_seed(config.random_state)`.
    # 2. Build `splits = build_image_world_model_splits(config)`.
    # 3. Build a training `TensorDataset` and `DataLoader`.
    # 4. Build `LatentWorldModel(config).to(device)`.
    # 5. Build `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 6. Train and evaluate for `config.epochs`.
    # 7. Evaluate all three counterfactual action-branch metrics with
    #    `evaluate_action_branches(model, config)` and map its keys to
    #    `heldout_action_branch_mse`,
    #    `heldout_action_branch_changed_pixel_mse`, and
    #    `heldout_action_branch_sensitivity_ratio`.
    # 8. Save all four visual files, including the action-branch preview.
    # 9. Return a `LatentImageMetrics` dictionary.
    #
    # Checks:
    # - Return validation metrics, not only training loss.
    # - Return paths for all saved visuals.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_image_world_model_splits(config)
    train_dataset = TensorDataset(splits["train_frames"], splits["train_actions"], splits["train_next_frames"])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = LatentWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_losses: list[float] = []
    val_next_frame_mses: list[float] = []
    latest_val_metrics: dict[str, float] = {}

    print("epoch | train_loss | val_recon_mse | val_latent_mse | val_next_frame_mse")
    print("------|------------|---------------|----------------|-------------------")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_metrics = evaluate_latent_world_model(model, splits)

        train_losses.append(train_loss)
        val_next_frame_mses.append(val_metrics["val_next_frame_mse"])
        latest_val_metrics = val_metrics

        print(
            f"{epoch:5d} | "
            f"{train_loss:10.6f} | "
            f"{val_metrics['val_reconstruction_mse']:13.6f} | "
            f"{val_metrics['val_latent_prediction_mse']:14.6f} | "
            f"{val_metrics['val_next_frame_mse']:17.6f}"
        )

    action_branch_metrics = evaluate_action_branches(model, config)

    training_curve_path = save_training_curves(train_losses, val_next_frame_mses, config)
    reconstruction_preview_path = save_reconstruction_preview(model, splits, config)
    next_frame_preview_path = save_next_frame_prediction_preview(model, splits, config)
    action_branch_preview_path = save_action_branch_preview(model, config)

    final_metrics: LatentImageMetrics = {
        "final_train_loss": train_losses[-1],
        "val_reconstruction_mse": latest_val_metrics["val_reconstruction_mse"],
        "val_latent_prediction_mse": latest_val_metrics["val_latent_prediction_mse"],
        "val_next_frame_mse": latest_val_metrics["val_next_frame_mse"],
        "heldout_action_branch_mse": action_branch_metrics["action_branch_mse"],
        "heldout_action_branch_changed_pixel_mse": action_branch_metrics["action_branch_changed_pixel_mse"],
        "heldout_action_branch_sensitivity_ratio": action_branch_metrics["action_branch_sensitivity_ratio"],
        "training_curve_path": training_curve_path,
        "reconstruction_preview_path": reconstruction_preview_path,
        "next_frame_preview_path": next_frame_preview_path,
        "action_branch_preview_path": action_branch_preview_path,
    }

    print("\nfinal metrics")
    print("-------------")
    print(f"final_train_loss: {final_metrics['final_train_loss']:.6f}")
    print(f"val_reconstruction_mse: {final_metrics['val_reconstruction_mse']:.6f}")
    print(f"val_latent_prediction_mse: {final_metrics['val_latent_prediction_mse']:.6f}")
    print(f"val_next_frame_mse: {final_metrics['val_next_frame_mse']:.6f}")
    print(f"heldout_action_branch_mse: {final_metrics['heldout_action_branch_mse']:.6f}")
    print(f"heldout_action_branch_changed_pixel_mse: {final_metrics['heldout_action_branch_changed_pixel_mse']:.6f}")
    print(f"heldout_action_branch_sensitivity_ratio: {final_metrics['heldout_action_branch_sensitivity_ratio']:.6f}")
    print(f"training_curve_path: {final_metrics['training_curve_path']}")
    print(f"reconstruction_preview_path: {final_metrics['reconstruction_preview_path']}")
    print(f"next_frame_preview_path: {final_metrics['next_frame_preview_path']}")
    print(f"action_branch_preview_path: {final_metrics['action_branch_preview_path']}")

    return final_metrics

    #raise NotImplementedError("Train and evaluate the latent image world model.")


def explain_latent_image_world_model() -> str:
    """Return a short explanation of latent image prediction."""
    return (
        "A latent image world model first compresses each frame into a smaller "
        "vector. It predicts how that vector changes after an action, then "
        "decodes the predicted vector back into an image."
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
    """Run the latent image world-model scaffold."""
    config = LatentWorldModelConfig()
    metrics, ok = report_task_progress("train_latent_world_model", train_latent_world_model, config)
    if ok:
        print(metrics)
        print(explain_latent_image_world_model())


if __name__ == "__main__":
    main()
