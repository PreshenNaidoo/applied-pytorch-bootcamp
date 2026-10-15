"""
============================================================
EXERCISE 17: LEROBOT DROID-100 DATASET ADAPTER
============================================================

WHAT YOU ARE BUILDING
You are loading real robot demonstrations and converting them into the shared
vision-language-action format used by this bootcamp.

Source:

```text
lerobot/droid_100
```

This is a manageable 100-episode subset of DROID.
It contains real robot manipulation trajectories recorded across many tasks.

The shared training row is:

```text
camera image
language instruction
robot state
action
action padding mask
reward
done
episode id
step id
timestamp
```

REAL SYSTEM FIRST
Large robot policies train across datasets that store:

- several camera streams
- robot joint or end-effector state
- continuous actions
- natural-language task names
- time and episode boundaries
- success or reward information

The files may be split across Parquet metadata and MP4 video files.
After the data is in current LeRobot v3 format, training code should use the
dataset's Python API instead of depending on those storage paths.

LeRobot v3 uses this design:

```text
storage:
Parquet rows + video shards + metadata

Python sample:
dictionary of PyTorch tensors and task fields
```

WHY THIS EXERCISE EXISTS
The earlier version of this exercise generated fake robot rows.
That was useful for testing a dictionary conversion, but it did not teach the
real data-loading problems found in current robot learning.

This exercise uses real DROID images, states, actions, task ids, and episode
boundaries.

REAL SYSTEM VS THIS EXERCISE

large robot foundation-model run      this exercise
----------------------------------------------------------------
full DROID dataset                    `lerobot/droid_100`
many thousands of episodes            at most a few chosen episodes
all camera streams                     one chosen camera first
large VLA policy                       small continuous behavior-cloning model
distributed training                   one local dataloader
full benchmark deployment              held-out action and data checks

WHY THIS SMALLER VERSION EXISTS
The full DROID dataset is very large.
The 100-episode subset still contains real robot behavior and many tasks, but
it is small enough for learning the data pipeline.

IMPORTANT FORMAT CHECK
At the time this exercise was reviewed, the published
`lerobot/droid_100` repository reports:

```text
100 episodes
32,212 frames
47 tasks
15 frames per second
7-value robot state
7-value action
LeRobot v2.0 storage
```

Current LeRobot uses v3 storage.
Do not pass this older v2.0 repository into a current v3 workflow and hope the
formats match.

The exercise therefore has two data names:

```text
source_repo_id:
the published `lerobot/droid_100` metadata you inspect first

converted_repo_id:
your current LeRobot v3 copy used by `LeRobotDataset`
```

If an official v3 copy is published later, point `converted_repo_id` at it and
skip the conversion step.

CURRENT DATA PREPARATION
Use the official LeRobot DROID porting guide.
For a local learning run:

1. Install current LeRobot and the DROID conversion dependencies.
2. Download only the roughly 2 GB raw `droid_100` sample from the official
   DROID Google Cloud location.
3. Run the current LeRobot `examples/port_datasets/port_droid.py` script on
   that small raw directory.
4. Give the result a new repository id ending in `_v3`.
5. Set `config.converted_repo_id` to that id.

INSTALLATION
LeRobot is optional for the rest of this repository.
Install it only when starting this exercise:

```text
uv pip install "lerobot==0.6.0" tensorflow tensorflow-datasets
```

One concrete v0.6.0 preparation path is:

```text
gsutil -m cp -r gs://gresearch/robotics/droid_100 data/droid_100_raw

git clone --depth 1 --branch v0.6.0 https://github.com/huggingface/lerobot.git external/lerobot-v0.6.0

python external/lerobot-v0.6.0/examples/port_datasets/port_droid.py --raw-dir data/droid_100_raw --repo-id YOUR_HF_NAME/droid_100_v3
```

The download is still large enough to take time and disk space.
Run the metadata-only first stub before downloading it.

If the pinned port script rejects the raw sample layout, stop and compare the
script's expected folder with the official porting guide. Do not silently
rename or invent fields.

LOADING CALL
Use the real LeRobot API:

```python
from lerobot.datasets import LeRobotDataset

frame_seconds = 1.0 / metadata.fps

dataset = LeRobotDataset(
    config.converted_repo_id,
    episodes=list(range(config.max_episodes)),
    delta_timestamps={
        config.camera_key: [-2 * frame_seconds, -frame_seconds, 0.0],
        "action": [0.0, frame_seconds, 2 * frame_seconds],
    },
)
```

`delta_timestamps` asks LeRobot for an ordered time window around one row.
For the camera example:

```text
-2 frames: older image
-1 frame: previous image
 0 frames: current image
```

For the action example:

```text
0 frames: current action
1 frame: next action
2 frames: action after that
```

This creates an action chunk.
An action chunk is a short sequence of actions predicted together.

Do not use `datasets.load_dataset(...)` as the main loader here.
That can expose table fields, but the LeRobot API also joins video frames,
episode metadata, and requested time windows.

FINDING THE CAMERA KEY
Do not guess whether the dataset uses:

```text
observation.images.exterior_image_1_left
observation.images.wrist_image_left
observation.images.front
```

Inspect:

```python
print(dataset.meta.features)
print(dataset[0].keys())
```

Then choose one RGB field and store its exact name in `config.camera_key`.

MAPPING TO THE SHARED FORMAT

real LeRobot field                   shared field
----------------------------------------------------------------
chosen `observation.images.*`         `image_history`
`observation.state`                   `robot_state`
`action`                              `action_chunk`
`action_is_pad`                       `action_is_pad`
task text or task lookup              `instruction`
`next.reward`                         `reward`
`next.done`                           `done`
`episode_index`                       `episode_id`
`frame_index`                         `step_id`
`timestamp`                           `timestamp`

INPUTS AND OUTPUTS

```text
image_history:
[history_frames, 3, image_size, image_size]

robot_state:
[state_dim]

action_chunk:
[action_horizon, action_dim]

action_is_pad:
[action_horizon]
True means this future position crossed the episode boundary and must not
contribute to the loss.

instruction:
string
```

The exact state and action sizes must come from metadata.
Do not hard-code `7` before checking.

BEHAVIOR-CLONING SMOKE TEST
Train:

```text
current image + robot state -> action chunk
```

Use a small CNN for the image, an MLP for state, and one linear head for the
flattened action chunk.

Loss:

```python
per_value_loss = nn.functional.smooth_l1_loss(
    predicted_action_chunks,
    target_action_chunks,
    reduction="none",
)

valid = (~action_is_pad).unsqueeze(-1)
loss = (per_value_loss * valid).sum() / (valid.sum() * action_dim)
```

`Smooth L1` behaves like squared error near the target but is less affected by
very large individual errors.

METRICS TO TRACK

- rows and episodes loaded
- missing required field rate
- timestamp-gap error
- action mean absolute error
- action Smooth L1 loss
- per-action-dimension error
- held-out task error
- held-out episode error
- image decode failures

Split by episode.
Also create a stricter split where some task ids appear only in the test set.

VISUALS TO SAVE
Save to `outputs/droid100_adapter/`:

- one real image-history strip
- one target versus predicted action-chunk plot
- error by action dimension
- error by task
- dataset schema report as JSON

COMMON MISTAKES

- reading raw MP4 shard paths instead of using `LeRobotDataset`
- assuming one row equals one episode
- randomly splitting frames from the same episode
- guessing the camera key
- losing the time dimension returned by `delta_timestamps`
- training on padded future actions at the end of an episode
- confusing robot state with the action target
- normalizing test data with statistics computed from the full dataset
- downloading all of DROID instead of starting with `droid_100`

STEP-BY-STEP TODO

1. Inspect the published `lerobot/droid_100` source metadata.
2. Confirm its current storage version instead of assuming v3.
3. Prepare or locate a current v3 copy of the 100-episode sample.
4. Load v3 metadata and print the real schema.
5. Choose one real RGB camera field.
6. Turn frame offsets into seconds using the real frames-per-second value.
7. Request image history and future action chunks.
8. Inspect one returned sample and every tensor shape.
9. Convert samples into `DroidVLAStep`.
10. Split by episode and then make a held-out-task test.
11. Fit normalization statistics on training actions only.
12. Train the small action-chunk smoke test.
13. Report per-dimension and held-out-task errors.
14. Save the real frame and action visuals.

LEARNING ORDER
Complete the schema inspection stub first.
Do not write the adapter from a remembered field list.

DEFINITION OF DONE

You are done when:

- the main data comes from real DROID-100 episodes
- the published source version and the training copy version are both recorded
- the training copy uses current LeRobot v3 format
- one camera key is chosen from the actual schema
- time windows line up with the dataset frame rate
- padded future actions do not contribute to loss or error metrics
- train and test rows come from different episodes
- a held-out-task result is reported
- the smoke-test model predicts real continuous action chunks
- real frames and action predictions are plotted
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata


class DroidVLAStep(TypedDict):
    """One adapted real DROID training row."""

    image_history: torch.Tensor
    instruction: str
    robot_state: torch.Tensor
    action_chunk: torch.Tensor
    action_is_pad: torch.Tensor
    reward: float
    done: bool
    episode_id: int
    step_id: int
    task_id: int
    timestamp: float


class DroidAdapterMetrics(TypedDict):
    """Data and action-prediction checks."""

    num_rows: int
    num_episodes: int
    num_tasks: int
    missing_field_rate: float
    timestamp_gap_error: float
    action_mae: float
    action_smooth_l1: float
    held_out_episode_mae: float
    held_out_task_mae: float
    image_decode_failures: int


@dataclass
class DroidAdapterConfig:
    """Settings for the real LeRobot DROID subset."""

    source_repo_id: str = "lerobot/droid_100"
    converted_repo_id: str = "SET_TO_CURRENT_V3_DROID100_REPO"
    output_dir: Path = Path("outputs/droid100_adapter")
    max_episodes: int = 12
    camera_key: str = "SET_AFTER_SCHEMA_INSPECTION"
    image_history_frame_offsets: tuple[int, ...] = (-2, -1, 0)
    action_chunk_frame_offsets: tuple[int, ...] = (0, 1, 2)
    image_size: int = 96
    hidden_dim: int = 128
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 3e-4
    random_state: int = 42


def inspect_published_droid100_metadata(config: DroidAdapterConfig) -> dict[str, object]:
    """Inspect the published DROID-100 source before choosing a loader."""
    # Where this fits:
    # This must happen before choosing fields or model shapes.
    #
    # Goal:
    # Download only `meta/info.json` from the published source and return a
    # dictionary containing:
    # - codebase version
    # - frames per second
    # - total episodes
    # - total frames
    # - task count
    # - feature names, shapes, and dtypes
    # - available RGB camera keys
    #
    # Method:
    # Import:
    # `from huggingface_hub import hf_hub_download`
    #
    # Call:
    # `info_path = hf_hub_download(`
    # `    repo_id=config.source_repo_id,`
    # `    filename="meta/info.json",`
    # `    repo_type="dataset",`
    # `)`
    #
    # Read:
    # `info = json.loads(Path(info_path).read_text(encoding="utf-8"))`
    #
    # Save:
    # `config.output_dir / "published_source_schema.json"`
    # with `json.dump(report, file, indent=2)`.
    #
    # Checks:
    # - Print `info["codebase_version"]`.
    # - Do not describe v2.0 metadata as v3.
    # - Print all keys beginning with `"observation.images."`.
    # - Do not continue if no RGB camera field exists.
    _ = (config, json)
    raise NotImplementedError("Inspect the published DROID-100 source metadata.")


def load_current_droid100_metadata(
    config: DroidAdapterConfig,
) -> LeRobotDatasetMetadata:
    """Load and validate metadata for the converted current-format copy."""
    # Where this fits:
    # The published source check proves what format exists.
    # This function checks the copy that current LeRobot will actually load.
    #
    # Method:
    # Import `LeRobotDatasetMetadata` from `lerobot.datasets`.
    # Call:
    # `metadata = LeRobotDatasetMetadata(config.converted_repo_id)`.
    #
    # Steps:
    # 1. Reject the unchanged placeholder `converted_repo_id`.
    # 2. Load the metadata.
    # 3. Check the reported codebase version starts with `"v3"`.
    # 4. Print `metadata.camera_keys`, `metadata.features`, `metadata.fps`,
    #    and `metadata.tasks`.
    # 5. Save a second report named `current_v3_schema.json`.
    # 6. Return the metadata object.
    #
    # Check:
    # - Choose `config.camera_key` only from `metadata.camera_keys`.
    _ = config
    raise NotImplementedError("Load and validate current DROID-100 metadata.")


def load_droid100_dataset(
    config: DroidAdapterConfig,
    metadata: LeRobotDatasetMetadata,
) -> LeRobotDataset:
    """Load selected episodes with image history and future actions."""
    # Where this fits:
    # LeRobot joins the table, video, and time-window fields for the adapter.
    #
    # Goal:
    # Return a real `LeRobotDataset`.
    #
    # Method:
    # Import:
    # `from lerobot.datasets import LeRobotDataset`
    #
    # Exact call:
    # First compute:
    # `frame_seconds = 1.0 / metadata.fps`
    # `image_history_seconds = [`
    # `    offset * frame_seconds`
    # `    for offset in config.image_history_frame_offsets`
    # `]`
    # `action_chunk_seconds = [`
    # `    offset * frame_seconds`
    # `    for offset in config.action_chunk_frame_offsets`
    # `]`
    #
    # Then call:
    # `LeRobotDataset(`
    # `    config.converted_repo_id,`
    # `    episodes=list(range(config.max_episodes)),`
    # `    delta_timestamps={`
    # `        config.camera_key: image_history_seconds,`
    # `        "action": action_chunk_seconds,`
    # `    },`
    # `)`
    #
    # Checks:
    # - `camera_key` must have been copied from the schema report.
    # - Every delta timestamp must be an integer frame offset divided by fps.
    # - Inspect `dataset[0]` before writing conversion code.
    _ = (config, metadata)
    raise NotImplementedError("Load the selected real DROID-100 episodes.")


def resolve_task_instruction(
    dataset: LeRobotDataset,
    sample: dict[str, torch.Tensor | str],
) -> tuple[str, int]:
    """Return the natural-language task and integer task id for one sample."""
    # Goal:
    # Decode task metadata instead of using one fixed fake instruction.
    #
    # Method:
    # 1. Read `task_index` from the sample.
    # 2. Use the dataset metadata task lookup to find its text.
    # 3. Return `(instruction_text, task_id)`.
    #
    # Check:
    # - Raise a clear error when task metadata is unavailable.
    _ = (dataset, sample)
    raise NotImplementedError("Resolve one real DROID task instruction.")


def adapt_droid_sample(
    dataset: LeRobotDataset,
    sample: dict[str, torch.Tensor | str],
    config: DroidAdapterConfig,
) -> DroidVLAStep:
    """Convert one LeRobot sample into the shared VLA format."""
    # Goal:
    # Hide LeRobot-specific names behind `DroidVLAStep`.
    #
    # Steps:
    # 1. Read `sample[config.camera_key]`.
    # 2. Confirm shape `[history, 3, H, W]`.
    # 3. Resize every frame to `config.image_size` with
    #    `torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)`.
    # 4. Read `sample["observation.state"]`.
    # 5. Read the action window from `sample["action"]`.
    # 6. Read `sample["action_is_pad"]`, a boolean tensor shaped `[horizon]`.
    # 7. Use `resolve_task_instruction(...)`.
    # 8. Read reward, done, episode index, frame index, and timestamp.
    # 9. Return all `DroidVLAStep` keys.
    #
    # Checks:
    # - Preserve the image-history and action-chunk dimensions.
    # - Convert scalar tensors with `.item()`.
    # - Do not invent missing images or actions.
    # - Keep padded actions for fixed tensor shapes, but keep their mask too.
    _ = (dataset, sample, config)
    raise NotImplementedError("Adapt one real DROID sample.")


def build_droid_vla_rows(
    dataset: LeRobotDataset,
    config: DroidAdapterConfig,
) -> tuple[list[DroidVLAStep], dict[str, int]]:
    """Adapt selected rows and count decode or schema failures."""
    # Goal:
    # Convert valid rows and report why any rows were skipped.
    #
    # Method:
    # 1. Loop over dataset indices in their stored order.
    # 2. Call `adapt_droid_sample(...)`.
    # 3. Catch only expected decode or missing-field errors.
    # 4. Count errors by reason.
    # 5. Return `(rows, failure_counts)`.
    #
    # Checks:
    # - Do not catch every exception and silently continue.
    # - Print the first failure from each reason.
    _ = (dataset, config)
    raise NotImplementedError("Build shared VLA rows from DROID-100.")


def split_droid_rows_by_episode_and_task(
    rows: list[DroidVLAStep],
    config: DroidAdapterConfig,
) -> dict[str, list[DroidVLAStep]]:
    """Create episode-safe train, validation, and held-out-task splits."""
    # Goal:
    # Return:
    # - `"train"`
    # - `"validation"`
    # - `"held_out_episode_test"`
    # - `"held_out_task_test"`
    #
    # Method:
    # 1. Group rows by `episode_id`.
    # 2. Group episode ids by `task_id`.
    # 3. Reserve at least one task for held-out-task testing.
    # 4. Split the remaining episode ids into train, validation, and test.
    # 5. Collect rows only after episode ids are assigned.
    #
    # Checks:
    # - No episode id may occur in two splits.
    # - No held-out task id may occur in training.
    _ = (rows, config)
    raise NotImplementedError("Split DROID rows by episode and task.")


class DroidActionChunkDataset(Dataset):
    """Return normalized real images, states, and action chunks."""

    def __init__(
        self,
        rows: list[DroidVLAStep],
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        # Goal:
        # Store rows and training-only action normalization values.
        #
        # Check:
        # - Replace action standard deviations below `1e-6` with `1.0`.
        self.rows = rows
        self.action_mean = action_mean
        self.action_std = action_std
        raise NotImplementedError("Build the DROID action-chunk dataset.")

    def __len__(self) -> int:
        """Return the number of adapted rows."""
        raise NotImplementedError("Return DROID dataset length.")

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return image, state, action chunk, padding mask, and task id."""
        # Output:
        # - current image: `[3, H, W]`
        # - robot state: `[state_dim]`
        # - normalized action chunk: `[action_horizon, action_dim]`
        # - action padding mask: `[action_horizon]`
        # - task id: scalar long tensor
        raise NotImplementedError("Return one DROID action-training row.")


class DroidActionChunkPolicy(nn.Module):
    """Predict a short continuous action sequence from image and robot state."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        config: DroidAdapterConfig,
    ) -> None:
        # Goal:
        # Build:
        #
        # image encoder:
        # `nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)`
        # `nn.ReLU()`
        # `nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)`
        # `nn.ReLU()`
        # `nn.AdaptiveAvgPool2d((1, 1))`
        # `nn.Flatten()`
        #
        # state encoder:
        # `nn.Linear(state_dim, config.hidden_dim)`
        # `nn.ReLU()`
        #
        # action head:
        # concatenate image and state features, then output
        # `action_horizon * action_dim` values
        #
        # Save `action_dim` and `action_horizon` for reshaping in `forward`.
        super().__init__()
        _ = (state_dim, action_dim, action_horizon, config)
        raise NotImplementedError("Build the DROID action-chunk policy.")

    def forward(self, images: torch.Tensor, robot_states: torch.Tensor) -> torch.Tensor:
        """Return action chunks shaped `[batch, horizon, action_dim]`."""
        # Steps:
        # 1. Encode images.
        # 2. Encode robot states.
        # 3. Concatenate features.
        # 4. Run the action head.
        # 5. Reshape with `.view(batch_size, action_horizon, action_dim)`.
        _ = (images, robot_states)
        raise NotImplementedError("Predict a real DROID action chunk.")


def train_droid_action_smoke_test(
    splits: dict[str, list[DroidVLAStep]],
    config: DroidAdapterConfig,
) -> tuple[DroidActionChunkPolicy, DroidAdapterMetrics, dict[str, torch.Tensor]]:
    """Train and evaluate the small continuous action-chunk policy."""
    # Goal:
    # Prove that the real adapted rows can feed a normal PyTorch model.
    #
    # Steps:
    # 1. Stack training action chunks.
    # 2. Compute action mean and standard deviation from training only.
    # 3. Build one `DroidActionChunkDataset` per split.
    # 4. Build `DataLoader` objects.
    # 5. Infer `state_dim`, `action_dim`, and `action_horizon` from one row.
    # 6. Build `DroidActionChunkPolicy(...)`.
    # 7. Use `torch.optim.AdamW(...)`.
    # 8. Compute elementwise Smooth L1 with `reduction="none"`.
    # 9. Use `action_is_pad` to remove padded future positions from the loss.
    # 10. Evaluate MAE after undoing normalization, again ignoring padding.
    # 11. Report held-out-episode and held-out-task MAE separately.
    #
    # Return:
    # - trained model
    # - all `DroidAdapterMetrics`
    # - per-action-dimension error tensors for plotting
    _ = (splits, config, DataLoader)
    raise NotImplementedError("Train the real DROID action smoke test.")


def save_droid_adapter_visuals(
    example: DroidVLAStep,
    target_chunk: torch.Tensor,
    predicted_chunk: torch.Tensor,
    per_dimension_error: torch.Tensor,
    config: DroidAdapterConfig,
) -> dict[str, str]:
    """Save real frames, action chunks, and error plots."""
    # Save:
    # - `real_image_history.png`
    # - `target_vs_predicted_action_chunk.png`
    # - `error_by_action_dimension.png`
    #
    # Use Matplotlib's `Agg` backend.
    # Label each action dimension by index if semantic names are unavailable.
    _ = (
        example,
        target_chunk,
        predicted_chunk,
        per_dimension_error,
        config,
    )
    raise NotImplementedError("Save the DROID adapter visuals.")


def main() -> None:
    """Run the real DROID-100 adapter scaffold."""
    config = DroidAdapterConfig()
    try:
        print(inspect_published_droid100_metadata(config))
        metadata = load_current_droid100_metadata(config)
        dataset = load_droid100_dataset(config, metadata)
        rows, failures = build_droid_vla_rows(dataset, config)
        print(f"adapted rows: {len(rows)}, failures: {failures}")
    except NotImplementedError as error:
        print(f"[TODO] lerobot_droid100_dataset_adapter: {error}")


if __name__ == "__main__":
    main()
