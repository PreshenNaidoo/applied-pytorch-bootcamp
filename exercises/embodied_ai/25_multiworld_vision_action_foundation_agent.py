"""
============================================================
EXERCISE 25: MULTI-WORLD VISION-ACTION FOUNDATION AGENT
============================================================

WHAT YOU ARE BUILDING
You are training one fast visual policy across several small worlds, then
testing whether that pretraining helps it learn a held-out world.

The experiment is:

```text
source worlds:
pixels -> shared gamepad-style action chunks

held-out target world:
compare a multi-world-pretrained policy with the same target-only baseline
```

This is a small NitroGen-style experiment.

REAL SYSTEM FIRST
NitroGen is an open foundation model for generalist gaming agents.
Its full workflow has three important parts:

```text
internet gameplay videos with recovered controller actions
-> one shared vision-action model trained across many games
-> one simulator interface used to evaluate and adapt the model
```

The released model is a fast visual controller.
It maps a game frame to a short gamepad-action chunk.
It uses a vision encoder and a flow-matching Diffusion Transformer, often
shortened to DiT.

Plain meaning:

```text
look at the current game frame
-> start with a noisy short action sequence
-> repeatedly clean that sequence
-> execute the first action
-> observe the real next frame
-> repeat
```

The full NitroGen model was trained at very large scale.
It is not a language planner and it does not solve long tasks by itself.
The project describes it as a fast, reactive "system 1" controller.

In this bootcamp:

```text
embodied_ai/25 = fast shared visual controller
embodied_ai/22 = temporal memory
embodied_ai/23 = slower language and subgoal planner
embodied_ai/24 = correction and self-improvement loop
```

WHY THIS EXERCISE EXISTS
The earlier behavior-cloning exercises train one policy in one world.
A foundation policy asks a different question:

```text
Does learning broad visual-action patterns in several worlds make a new
world easier to learn?
```

That question needs a fair transfer experiment.
Do not compare different architectures or give the pretrained model more
target-world data.

FULL WORKFLOW

```text
1. collect successful demonstrations from several real local environments
2. map every environment action into one shared gamepad-style vector
3. build fixed-length action chunks without crossing episode boundaries
4. hold one complete world out of pretraining
5. pretrain one shared vision-action policy on the source worlds
6. measure a zero-shot target-world diagnostic
7. copy the pretrained model and fine-tune it on a small target dataset
8. train the same architecture only on that target dataset
9. evaluate both models on the same untouched target episodes
10. compare action error, success, cost, and data efficiency
```

REAL SYSTEM VS THIS EXERCISE

```text
real NitroGen                         this exercise
---------------------------------------------------------------------------
40,000 hours of internet gameplay    small local rendered demonstrations
more than 1,000 games                two source worlds and one held-out world
recovered gamepad overlays           exact actions logged by environments
24-value gamepad actions              8-value shared controller actions
16-action chunks                      8-action chunks
large visual encoder                  pretrained ResNet-18 visual encoder
roughly 500M-parameter Action DiT      tiny flow-matching action transformer
commercial-game simulator wrapper     one local world-adapter interface
large post-training runs              equal small target-data budgets
```

WHY THIS SMALLER VERSION EXISTS
The original dataset and pretraining run are too large for one bootcamp
exercise.

This version keeps the important method:

- real environment images
- real actions
- multiple worlds
- one shared action space
- action chunks
- flow matching
- a transformer action head
- held-out-world adaptation
- pretrained-versus-target-only evaluation

It scales down the number of worlds, data hours, image size, action size,
model width, and training steps.

DATA
Use real outputs from the local environments:

```text
source world 1:
standard `common.embodied_gridworld`

source world 2:
a visually and mechanically changed grid-world from `embodied_ai/08`

held-out target world:
first-person navigation episodes from `embodied_ai/10` and `embodied_ai/11`
```

Use scripted successful demonstrations first.
Random actions are useful for a data-pipeline smoke test, but they do not
provide a strong controller target.

Keep the first-person target world completely out of source pretraining.
Within each world, split whole episodes.
Never split individual frames from one episode across train and test.

OPTIONAL OFFICIAL MODEL CHECK
After the small experiment works, the open NitroGen repository provides its
own model and inference workflow.

That is an optional comparison, not the main exercise.
Do not download commercial games or run game processes automatically.
Use only games you own and only after reading the official setup and license.

SHARED CONTROLLER ACTION
Every local action becomes an 8-value controller vector:

```text
index  meaning
------------------------------------------------
0      left-stick horizontal: strafe or move left/right
1      left-stick vertical: move forward/backward
2      right-stick horizontal: turn left/right
3      right-stick vertical: camera up/down
4      stop button
5      interact button
6      jump button
7      spare button
```

Values for stick axes are in `[-1, 1]`.
Button values are `0` or `1`.

Examples:

```text
grid-world move up:
[0, 1, 0, 0, 0, 0, 0, 0]

first-person turn right:
[0, 0, 1, 0, 0, 0, 0, 0]

first-person stop:
[0, 0, 0, 0, 1, 0, 0, 0]
```

Use one documented mapping for every world.
Do not let the same vector mean unrelated actions in different worlds.

FLOW MATCHING
The target action chunk is:

```text
action_chunk: [batch, horizon, action_dim]
```

Sample Gaussian noise with the same shape:

```text
noise ~ Normal(0, I)
```

Sample one time value per batch row:

```text
t ~ Uniform(0, 1)
```

Mix noise and the real action:

```text
noisy_action = (1 - t) * noise + t * action_chunk
```

The correct velocity is:

```text
target_velocity = action_chunk - noise
```

The model receives:

```text
current image
noisy action chunk
time t
```

It predicts:

```text
predicted velocity
```

Training loss:

```text
valid = action_valid_mask.unsqueeze(-1)

per_value_loss = (
    predicted_velocity - target_velocity
).square()

loss = (
    per_value_loss * valid
).sum() / (valid.sum() * action_dim)
```

The mask removes padded action positions near the end of an episode.

ACTION SAMPLING
At evaluation time:

```text
start with Gaussian action noise at t=0
predict velocity
move the action a small step in that direction
repeat until t=1
clamp stick values to [-1, 1]
threshold button values
```

Use Euler updates for the small exercise:

```text
action = action + step_size * predicted_velocity
```

The model predicts a whole chunk, but the environment executes only the first
action before observing a real new image.
This is receding-horizon control.

INPUTS AND OUTPUTS

```text
images:
[batch, 3, image_size, image_size]

noisy_actions:
[batch, action_horizon, action_dim]

flow_times:
[batch]

predicted_velocity:
[batch, action_horizon, action_dim]

action_valid_mask:
[batch, action_horizon]
```

TRANSFER EXPERIMENT
Train these exact comparisons:

```text
A. source-pretrained -> small target fine-tune
B. no source pretraining -> same small target fine-tune
C. source-pretrained -> zero-shot target diagnostic
```

Both A and B may start from the same general ImageNet-pretrained ResNet.
Here, `scratch` means no multi-world action pretraining.
Keeping the general image initialization the same isolates the value of
source-world visual-action experience.

Models A and B must use:

- the same architecture
- the same target training rows
- the same target batch size
- the same number of optimizer steps
- the same evaluation seeds

The zero-shot result is a diagnostic, not a required success.
The published NitroGen work also reports that game-specific mechanics still
need target-world adaptation.

METRICS TO TRACK

- source-world flow loss: lower is better
- target action mean absolute error: lower is better
- target action-chunk error by horizon: lower is better
- target success rate: higher is better
- target collision or hazard cost: lower is better
- target data efficiency: success at several target-data budgets
- relative transfer gain: higher is better
- per-world source success: higher is better
- balanced mean source success: higher is better
- policy latency in milliseconds: lower is better

Relative transfer gain:

```text
pretrained target success - scratch target success
```

Balanced mean means each world counts equally.
Do not let the largest source dataset dominate the headline metric.

TRAINING LOSS IS NOT THE FINAL RESULT
Flow loss only says how well the model predicts demonstration actions.
The main question is whether the policy completes tasks in the environment.

Always run real closed-loop episodes:

```text
reset world
-> observe image
-> sample action chunk
-> execute first decoded action
-> observe next image
-> repeat
```

VISUALS TO SAVE
Save to `outputs/multiworld_foundation_agent/`:

- one frame strip from every world
- source-world data balance chart
- action chunk target versus prediction
- target success versus number of fine-tuning episodes
- pretrained versus scratch success bars
- zero-shot, fine-tuned, and scratch rollout strips
- failure gallery with world name and decoded action

COMMON MISTAKES

- calling a model generalist after training it in only one world
- using target-world rows during source pretraining
- comparing pretrained and scratch models with different target data
- randomly splitting frames instead of whole episodes
- using incompatible action meanings across worlds
- predicting only one action and dropping the action-chunk part of the method
- replacing flow matching with plain classification without explaining it
- training on padded future actions
- reporting only flow loss instead of closed-loop success
- claiming zero-shot mastery when the target world needed fine-tuning
- confusing this fast visual policy with the language planner in exercise 23

STEP-BY-STEP TODO

1. Build one adapter interface for all local worlds.
2. Collect successful real demonstrations from each world.
3. Convert every action into the shared 8-value controller vector.
4. Check action meanings and tensor ranges.
5. Split complete episodes and hold the first-person world out.
6. Build padded action chunks and their valid masks.
7. Build the small visual encoder.
8. Build the flow-matching action transformer.
9. Train on balanced source-world batches.
10. Run the zero-shot target diagnostic.
11. Fine-tune a pretrained copy on a small target dataset.
12. Train the same model from scratch on the same target dataset.
13. Evaluate both on fixed untouched target episodes.
14. Repeat with several target-data budgets.
15. Save plots, rollouts, latency, and failure cases.

LEARNING ORDER
Complete the world adapters and inspect real rows before building the model.
Then complete action chunking, flow noise, the model, source pretraining, and
the transfer comparison in that order.

DEFINITION OF DONE

You are done when:

- at least two source worlds provide real rendered demonstrations
- one complete target world is excluded from source pretraining
- all worlds use one checked controller-action meaning
- the model predicts action chunks with flow matching
- padded actions do not contribute to the loss
- source batches are balanced across worlds
- pretrained and scratch models receive identical target-data budgets
- zero-shot performance is reported without exaggeration
- real target-world rollout success is reported
- transfer gain and data-efficiency curves are saved
- policy latency is measured
- you can explain what the fast controller still lacks

OPTIONAL NEXT STEP
After the required experiment works:

1. Use `embodied_ai/22` to add frame and action history.
2. Use `embodied_ai/23` to add language goals and slow subgoal reasoning.
3. Use `embodied_ai/24` to collect failure corrections or successful
   self-imitation rollouts.
4. Compare the small model with the official open NitroGen checkpoint on a
   legally available supported game.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class MultiWorldStep(TypedDict):
    """One real visual-action step in the shared controller format."""

    image: torch.Tensor
    action: torch.Tensor
    reward: float
    cost: float
    done: bool
    success: bool
    world_name: str
    episode_id: int
    step_id: int
    split: str


class ActionChunkExample(TypedDict):
    """One image paired with a fixed-size future action sequence."""

    image: torch.Tensor
    action_chunk: torch.Tensor
    action_valid_mask: torch.Tensor
    world_name: str
    episode_id: int
    step_id: int
    split: str


class WorldTransition(TypedDict):
    """Result after one shared controller action is executed."""

    image: torch.Tensor
    reward: float
    cost: float
    done: bool
    success: bool
    info: dict[str, object]


class GeneralistWorldAdapter(Protocol):
    """Common interface around one local visual environment."""

    @property
    def name(self) -> str:
        """Return the stable world name used in reports."""

    def collect_demonstrations(
        self,
        *,
        num_episodes: int,
        first_episode_id: int,
        seed: int,
    ) -> list[list[MultiWorldStep]]:
        """Return ordered real demonstrations with shared action vectors."""

    def reset(self, *, seed: int) -> torch.Tensor:
        """Reset the world and return its first RGB image."""

    def step(self, controller_action: torch.Tensor) -> WorldTransition:
        """Decode and execute one shared controller action."""


class MultiWorldTransferMetrics(TypedDict):
    """Main measurements for one transfer comparison."""

    zero_shot_success_rate: float
    pretrained_finetune_success_rate: float
    scratch_finetune_success_rate: float
    relative_transfer_gain: float
    pretrained_target_action_mae: float
    scratch_target_action_mae: float
    pretrained_mean_cost: float
    scratch_mean_cost: float
    balanced_source_success_rate: float
    mean_policy_latency_ms: float


@dataclass
class MultiWorldAgentConfig:
    """Settings for the small multi-world foundation-policy experiment."""

    source_worlds: tuple[str, ...] = (
        "gridworld_standard",
        "gridworld_shifted",
    )
    held_out_world: str = "mini3d_navigation"
    output_dir: Path = Path("outputs/multiworld_foundation_agent")
    image_size: int = 96
    action_dim: int = 8
    action_horizon: int = 8
    source_episodes_per_world: int = 96
    target_train_episode_budgets: tuple[int, ...] = (4, 12, 32)
    target_validation_episodes: int = 16
    target_test_episodes: int = 32
    vision_dim: int = 256
    action_model_dim: int = 192
    action_model_depth: int = 3
    action_model_heads: int = 6
    action_model_mlp_dim: int = 384
    batch_size: int = 32
    source_epochs: int = 8
    target_finetune_steps: int = 500
    learning_rate: float = 1e-4
    flow_sample_steps: int = 8
    max_rollout_steps: int = 60
    random_state: int = 42

    def __post_init__(self) -> None:
        """Catch shape settings that would break the small action transformer."""
        if self.action_model_dim % self.action_model_heads != 0:
            raise ValueError("action_model_dim must divide evenly by action_model_heads.")
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be at least 1.")
        if self.action_dim != 8:
            raise ValueError("This exercise's shared controller action has 8 values.")
        if self.held_out_world in self.source_worlds:
            raise ValueError("The held-out world must not be a source world.")
        if self.flow_sample_steps < 1:
            raise ValueError("flow_sample_steps must be at least 1.")
        if not self.target_train_episode_budgets:
            raise ValueError("At least one target training budget is required.")
        if any(budget < 1 for budget in self.target_train_episode_budgets):
            raise ValueError("Every target training budget must be positive.")


def discrete_action_to_controller_vector(
    action: int,
    world_name: str,
    config: MultiWorldAgentConfig,
) -> torch.Tensor:
    """Map one local action id into the shared 8-value controller action."""
    # Where this fits:
    # The same policy can train across worlds only when action meanings agree.
    #
    # Goal:
    # Return a float tensor shaped `[config.action_dim]`.
    #
    # Grid-world mapping:
    # - up -> left-stick vertical `+1`
    # - down -> left-stick vertical `-1`
    # - left -> left-stick horizontal `-1`
    # - right -> left-stick horizontal `+1`
    #
    # Mini 3D mapping:
    # - move forward -> left-stick vertical `+1`
    # - turn left -> right-stick horizontal `-1`
    # - turn right -> right-stick horizontal `+1`
    # - strafe left -> left-stick horizontal `-1`
    # - strafe right -> left-stick horizontal `+1`
    # - stop -> stop button `1`
    #
    # Method:
    # 1. Create `vector = torch.zeros(config.action_dim, dtype=torch.float32)`.
    # 2. Use a separate explicit dictionary for each action convention.
    # 3. Fill the matching axis or button value.
    # 4. Raise `ValueError` for an unknown world or action id.
    #
    # Checks:
    # - stick values must stay in `[-1, 1]`
    # - button values must be exactly `0` or `1`
    # - print the complete mapping table before collecting data
    _ = (action, world_name, config)
    raise NotImplementedError("Map one environment action to the shared controller.")


def controller_vector_to_discrete_action(
    controller_action: torch.Tensor,
    world_name: str,
    config: MultiWorldAgentConfig,
) -> int:
    """Decode the first predicted controller action for one local world."""
    # Where this fits:
    # The model outputs one shared vector, but each environment still expects
    # its own integer action id.
    #
    # Goal:
    # Return the closest valid local action.
    #
    # Method:
    # 1. Build every valid controller vector for this world by calling
    #    `discrete_action_to_controller_vector(...)`.
    # 2. Stack them to `[num_local_actions, action_dim]`.
    # 3. Compute Euclidean distance from the predicted vector.
    # 4. Return the action id with the smallest distance.
    #
    # This nearest-valid-action check is better than decoding each axis
    # separately because it cannot create an action the environment does not
    # support.
    _ = (controller_action, world_name, config)
    raise NotImplementedError("Decode one shared controller action.")


def build_local_world_adapters(
    config: MultiWorldAgentConfig,
) -> dict[str, GeneralistWorldAdapter]:
    """Build adapters for the real local rendered environments."""
    # Where this fits:
    # NitroGen uses one simulator wrapper across games.
    # These adapters are the small local version of that interface.
    #
    # Goal:
    # Return adapters for:
    # - `"gridworld_standard"`
    # - `"gridworld_shifted"`
    # - `"mini3d_navigation"`
    #
    # Method:
    # 1. Wrap `common.embodied_gridworld` for the standard source.
    # 2. Build the shifted grid-world with changed colors, observation noise,
    #    wall count, and movement rules from `embodied_ai/08`.
    # 3. Wrap the first-person environment and scripted expert from
    #    `embodied_ai/10`.
    # 4. Resize every returned image to
    #    `[3, config.image_size, config.image_size]`.
    # 5. Convert local actions with
    #    `discrete_action_to_controller_vector(...)`.
    # 6. Decode model outputs with
    #    `controller_vector_to_discrete_action(...)`.
    #
    # Checks:
    # - every adapter returns float RGB values in `[0, 1]`
    # - every demonstration episode stays ordered
    # - every adapter can replay one scripted episode successfully
    # - hidden map state may create expert labels but is not a model input
    _ = config
    raise NotImplementedError("Build the local multi-world adapter set.")


def collect_multiworld_demonstrations(
    adapters: dict[str, GeneralistWorldAdapter],
    config: MultiWorldAgentConfig,
) -> dict[str, list[list[MultiWorldStep]]]:
    """Collect source and held-out target episodes without leakage."""
    # Goal:
    # Return one ordered episode list for every configured world.
    #
    # Method:
    # 1. Collect `source_episodes_per_world` successful or near-successful
    #    demonstrations from each source adapter.
    # 2. Collect enough separate target episodes for the largest training
    #    budget, validation, and test.
    # 3. Give every episode a globally unique `episode_id`.
    # 4. Assign whole episodes to train, validation, or test.
    # 5. Keep all target episodes out of source pretraining.
    #
    # Checks:
    # - report expert success and mean cost in each world
    # - no episode id may occur in two splits
    # - no target-world row may enter the source training set
    # - do not silently keep failed expert episodes as correct demonstrations
    _ = (adapters, config)
    raise NotImplementedError("Collect real multi-world demonstrations.")


def build_action_chunk_examples(
    episodes_by_world: dict[str, list[list[MultiWorldStep]]],
    config: MultiWorldAgentConfig,
) -> list[ActionChunkExample]:
    """Pair each current image with its future controller-action chunk."""
    # Where this fits:
    # NitroGen predicts a short future action sequence instead of one action.
    #
    # Tiny example with `action_horizon=4`:
    #
    # ```text
    # episode actions: [a0, a1, a2]
    #
    # row at step 1:
    # action_chunk     = [a1, a2, zero, zero]
    # action_valid_mask = [1, 1, 0, 0]
    # ```
    #
    # Steps:
    # 1. Loop over worlds, episodes, and current step positions.
    # 2. Read at most `config.action_horizon` actions from the same episode.
    # 3. Right-pad missing future positions with zero action vectors.
    # 4. Mark real positions `True` and padded positions `False`.
    # 5. Copy the current image and split metadata.
    #
    # Checks:
    # - action chunks must never cross episode or world boundaries
    # - every mask must contain at least one valid position
    # - output action shape must be `[action_horizon, action_dim]`
    _ = (episodes_by_world, config)
    raise NotImplementedError("Build padded multi-world action chunks.")


class MultiWorldActionDataset(Dataset):
    """Return real images and action chunks for flow-matching training."""

    def __init__(
        self,
        examples: list[ActionChunkExample],
        world_to_id: dict[str, int],
    ) -> None:
        # Goal:
        # Store the rows and stable world ids used for metrics.
        #
        # Output from `__getitem__`:
        # - image `[3, H, W]`
        # - action chunk `[horizon, action_dim]`
        # - valid mask `[horizon]`
        # - world id scalar
        self.examples = examples
        self.world_to_id = world_to_id
        raise NotImplementedError("Build the multi-world action dataset.")

    def __len__(self) -> int:
        """Return the number of action-chunk rows."""
        raise NotImplementedError("Return the multi-world dataset length.")

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return image, action chunk, valid mask, and world id."""
        raise NotImplementedError("Return one multi-world training row.")


def build_world_balanced_loader(
    dataset: MultiWorldActionDataset,
    config: MultiWorldAgentConfig,
) -> DataLoader:
    """Give every source world equal expected sampling weight."""
    # Goal:
    # Stop the largest world dataset from dominating source pretraining.
    #
    # Method:
    # 1. Count rows for every source world id.
    # 2. Give a row weight of `1.0 / rows_in_its_world`.
    # 3. Build:
    #    `WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)`.
    # 4. Return:
    #    `DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)`.
    #
    # Checks:
    # - do not also set `shuffle=True`
    # - print the expected sampled fraction for every world
    _ = (dataset, config, WeightedRandomSampler)
    raise NotImplementedError("Build a world-balanced source dataloader.")


class SmallGameVisionEncoder(nn.Module):
    """Encode one current game frame with a small pretrained image model."""

    def __init__(self, config: MultiWorldAgentConfig) -> None:
        # Where this fits:
        # NitroGen uses a much larger pretrained visual encoder.
        # This exercise uses a real pretrained ResNet-18 so the main compute can
        # go into learning the shared action model.
        #
        # Method:
        # Import:
        # `from torchvision.models import ResNet18_Weights, resnet18`
        #
        # Build:
        # `self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)`
        # `feature_dim = self.backbone.fc.in_features`
        # `self.backbone.fc = nn.Identity()`
        # `self.projection = nn.Linear(feature_dim, config.vision_dim)`
        #
        # Start with the backbone frozen.
        # After the action head works, optionally unfreeze the last ResNet block.
        super().__init__()
        _ = config
        raise NotImplementedError("Build the small pretrained game-frame encoder.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return image features shaped `[batch, vision_dim]`."""
        # Input images are float values in `[0, 1]`.
        #
        # Import:
        # `from torchvision.transforms.functional import normalize`
        #
        # Preserve the exercise's 96 by 96 image size and normalize with:
        # `normalize(`
        # `    images,`
        # `    mean=[0.485, 0.456, 0.406],`
        # `    std=[0.229, 0.224, 0.225],`
        # `)`
        #
        # Do not divide the image by 255 again.
        _ = images
        raise NotImplementedError("Encode current game frames.")


class TinyActionFlowTransformer(nn.Module):
    """Predict flow velocity for a noisy controller-action chunk."""

    def __init__(self, config: MultiWorldAgentConfig) -> None:
        # Where this fits:
        # This is the small version of NitroGen's Action DiT.
        #
        # Build:
        # - `SmallGameVisionEncoder(config)`
        # - action input projection:
        #   `nn.Linear(config.action_dim, config.action_model_dim)`
        # - learned action-position embedding:
        #   `[1, action_horizon, action_model_dim]`
        # - image projection:
        #   `nn.Linear(config.vision_dim, config.action_model_dim)`
        # - time MLP:
        #   `nn.Sequential(`
        #   `nn.Linear(1, action_model_dim),`
        #   `nn.SiLU(),`
        #   `nn.Linear(action_model_dim, action_model_dim),`
        #   `)`
        # - `nn.TransformerEncoder` using `config.action_model_depth` layers
        # - output projection:
        #   `nn.Linear(config.action_model_dim, config.action_dim)`
        #
        # Transformer layer:
        # `nn.TransformerEncoderLayer(`
        # `    d_model=config.action_model_dim,`
        # `    nhead=config.action_model_heads,`
        # `    dim_feedforward=config.action_model_mlp_dim,`
        # `    dropout=0.0,`
        # `    activation="gelu",`
        # `    batch_first=True,`
        # `)`
        super().__init__()
        _ = config
        raise NotImplementedError("Build the tiny flow-matching action transformer.")

    def forward(
        self,
        images: torch.Tensor,
        noisy_actions: torch.Tensor,
        flow_times: torch.Tensor,
    ) -> torch.Tensor:
        """Return velocity shaped `[batch, action_horizon, action_dim]`."""
        # Steps:
        # 1. Encode images to `[B, vision_dim]`.
        # 2. Project noisy actions to `[B, horizon, model_dim]`.
        # 3. Add the learned action-position embedding.
        # 4. Project image features and expand them across the horizon.
        # 5. Run `flow_times[:, None]` through the time MLP and expand it.
        # 6. Add action, image, and time features.
        # 7. Run the transformer.
        # 8. Project every token to `action_dim` velocity values.
        #
        # Check:
        # - do not apply sigmoid or softmax to the training velocity
        _ = (images, noisy_actions, flow_times)
        raise NotImplementedError("Predict action flow velocity.")


def make_flow_matching_inputs(
    action_chunks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create noisy actions, flow times, and target velocities."""
    # Input:
    # - `action_chunks`: `[B, horizon, action_dim]`
    #
    # Output:
    # - noisy actions: same shape
    # - flow times: `[B]`
    # - target velocity: same shape as actions
    #
    # Exact steps:
    # 1. `noise = torch.randn_like(action_chunks)`.
    # 2. Sample:
    #    `flow_times = torch.rand(batch_size, device=action_chunks.device)`.
    # 3. Reshape time:
    #    `time_view = flow_times[:, None, None]`.
    # 4. Mix:
    #    `noisy_actions = (1.0 - time_view) * noise + time_view * action_chunks`.
    # 5. Target:
    #    `target_velocity = action_chunks - noise`.
    #
    # Checks:
    # - at time near `0`, noisy actions should look mostly like noise
    # - at time near `1`, they should look mostly like real actions
    _ = action_chunks
    raise NotImplementedError("Build one conditional flow-matching batch.")


def compute_masked_flow_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    action_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute flow loss without padded future actions."""
    # Input:
    # - velocity tensors: `[B, horizon, action_dim]`
    # - valid mask: `[B, horizon]`
    #
    # Method:
    # 1. Check both velocity tensors have the same shape.
    # 2. Compute elementwise squared error.
    # 3. Convert the mask to the loss dtype and add a final size-1 dimension.
    # 4. Sum valid error.
    # 5. Divide by valid positions times `action_dim`.
    #
    # Check:
    # - changing a padded target must not change the returned loss
    _ = (predicted_velocity, target_velocity, action_valid_mask)
    raise NotImplementedError("Compute masked action flow loss.")


@torch.no_grad()
def sample_controller_action_chunk(
    model: TinyActionFlowTransformer,
    image: torch.Tensor,
    config: MultiWorldAgentConfig,
) -> torch.Tensor:
    """Generate one controller-action chunk with Euler flow steps."""
    # Goal:
    # Return `[action_horizon, action_dim]` for one current image.
    #
    # Steps:
    # 1. Put the model in evaluation mode.
    # 2. Add a batch dimension to the image when needed.
    # 3. Start with:
    #    `actions = torch.randn(1, horizon, action_dim, device=image.device)`.
    # 4. Set `step_size = 1.0 / config.flow_sample_steps`.
    # 5. For times from `0` to just below `1`:
    #    - create one `[1]` time tensor
    #    - predict velocity
    #    - update `actions = actions + step_size * velocity`
    # 6. Clamp the four stick axes to `[-1, 1]`.
    # 7. Convert button values to `0` or `1` with threshold `0.5`.
    # 8. Remove the batch dimension.
    #
    # Check:
    # - the policy still executes only `actions[0]` before observing again
    _ = (model, image, config)
    raise NotImplementedError("Sample one controller-action chunk.")


def pretrain_on_source_worlds(
    source_examples: list[ActionChunkExample],
    config: MultiWorldAgentConfig,
) -> tuple[TinyActionFlowTransformer, dict[str, list[float]]]:
    """Train one shared flow policy on balanced source-world data."""
    # Steps:
    # 1. Keep only source-world training rows.
    # 2. Build a stable `world_to_id` mapping for metrics.
    # 3. Build `MultiWorldActionDataset`.
    # 4. Use `build_world_balanced_loader(...)`.
    # 5. Build `TinyActionFlowTransformer(config)`.
    # 6. Use `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 7. For every batch:
    #    - move tensors to the device
    #    - call `make_flow_matching_inputs(...)`
    #    - predict velocity
    #    - call `compute_masked_flow_loss(...)`
    #    - zero gradients, backpropagate, and step the optimizer
    # 8. Report validation flow loss separately for every source world.
    #
    # Checks:
    # - save model, config, action mapping, and image normalization together
    # - do not read any held-out target-world row
    _ = (source_examples, config)
    raise NotImplementedError("Pretrain the shared source-world action model.")


def evaluate_policy_in_world(
    model: TinyActionFlowTransformer,
    adapter: GeneralistWorldAdapter,
    config: MultiWorldAgentConfig,
    *,
    evaluation_seeds: list[int],
) -> dict[str, float]:
    """Run fixed real episodes and report action-policy behavior."""
    # Goal:
    # Return:
    # - success rate
    # - mean reward
    # - mean cost
    # - mean episode steps
    # - mean policy latency in milliseconds
    #
    # Loop:
    # 1. Reset one episode.
    # 2. Time `sample_controller_action_chunk(...)`.
    # 3. Decode only the first action through the adapter.
    # 4. Step the real environment.
    # 5. Move to the returned image.
    # 6. Stop on done or `max_rollout_steps`.
    #
    # Checks:
    # - this function does not train
    # - call any policy history reset before each episode
    # - use the same seeds for pretrained and scratch models
    _ = (model, adapter, config, evaluation_seeds)
    raise NotImplementedError("Evaluate the shared action model in one real world.")


def compare_pretraining_with_scratch(
    pretrained_model: TinyActionFlowTransformer,
    target_examples: list[ActionChunkExample],
    target_adapter: GeneralistWorldAdapter,
    config: MultiWorldAgentConfig,
) -> tuple[MultiWorldTransferMetrics, dict[int, dict[str, float]]]:
    """Run the fair held-out-world transfer experiment."""
    # Goal:
    # Compare target adaptation from source-pretrained weights against the same
    # architecture without source-world action pretraining.
    #
    # Method for every target episode budget:
    # 1. Select the exact same target training episodes for both models.
    # 2. Deep-copy the source-pretrained model.
    # 3. Create one new `TinyActionFlowTransformer(config)`. It may use the
    #    same ImageNet-pretrained vision backbone, but its action model must
    #    not receive source-world training.
    # 4. Fine-tune both for exactly `config.target_finetune_steps`.
    # 5. Use the same batches, optimizer settings, and random seed.
    # 6. Evaluate on fixed untouched target seeds.
    # 7. Also evaluate the source-pretrained model before target fine-tuning.
    #
    # Return:
    # - the headline `MultiWorldTransferMetrics`
    # - one result dictionary for every target-data budget
    #
    # Checks:
    # - never choose the best target test checkpoint
    # - use validation data for checkpoint choice
    # - report negative transfer if scratch performs better
    # - do not describe fine-tuned success as zero-shot success
    _ = (pretrained_model, target_examples, target_adapter, config)
    raise NotImplementedError("Compare pretrained and scratch target adaptation.")


def save_multiworld_transfer_report(
    metrics: MultiWorldTransferMetrics,
    budget_results: dict[int, dict[str, float]],
    examples_by_world: dict[str, ActionChunkExample],
    config: MultiWorldAgentConfig,
) -> dict[str, str]:
    """Save data, transfer, action, and failure visuals."""
    # Save:
    # - `world_frame_strips.png`
    # - `source_world_balance.png`
    # - `target_action_chunk.png`
    # - `target_data_efficiency.png`
    # - `pretrained_vs_scratch_success.png`
    # - `transfer_failure_gallery.png`
    #
    # Use Matplotlib's `Agg` backend.
    # Put world names, episode ids, and decoded actions on failure examples.
    _ = (metrics, budget_results, examples_by_world, config)
    raise NotImplementedError("Save the multi-world transfer report.")


def run_multiworld_foundation_experiment(
    config: MultiWorldAgentConfig,
) -> MultiWorldTransferMetrics:
    """Run source pretraining and the held-out-world comparison."""
    # Steps:
    # 1. Build adapters.
    # 2. Collect real demonstrations.
    # 3. Build action-chunk rows.
    # 4. Pretrain only on source worlds.
    # 5. Compare pretrained and scratch target adaptation.
    # 6. Evaluate real rollouts.
    # 7. Save the report.
    _ = config
    raise NotImplementedError("Run the multi-world foundation-agent experiment.")


def main() -> None:
    """Run the NitroGen-style multi-world exercise scaffold."""
    config = MultiWorldAgentConfig()
    try:
        print(run_multiworld_foundation_experiment(config))
    except NotImplementedError as error:
        print(f"[TODO] multiworld_vision_action_foundation_agent: {error}")


if __name__ == "__main__":
    main()
