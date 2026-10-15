"""
============================================================
EXERCISE 18: WORLD MODEL DATA QUALITY CHECKS
============================================================

WHAT YOU ARE BUILDING
You are writing checks for world-model datasets before training.

WHY THIS EXERCISE EXISTS
World models are very sensitive to broken rollout data. Missing frames, repeated
frames, bad episode boundaries, or train/test leakage can make the model look
better or worse than it really is.

DATASET AND DOWNLOADS
Default dataset:
- no download
- run checks on local rollout rows from:
  - `common.embodied_gridworld`
  - `outputs/mini3d_vla_dataset.pt`
  - the fake robot rows from the related exercise

No model should train before this report passes.

CHECKS TO BUILD
- missing frames
- repeated frames
- action imbalance
- broken episode boundaries
- bad reward or cost labels
- train/test leakage

INPUT ROW FORMAT
Each row should be a dictionary with keys like:

```text
image
next_image
action
reward
cost
done
episode_id
step_id
split
```

METRICS TO TRACK
- missing frame count
- repeated frame count
- broken episode count
- action imbalance score
- bad reward/cost count
- train/test leak count
- final passed flag

HOW TO READ THE REPORT
- `missing_frame_count > 0`: training will fail or learn from incomplete rows
- `repeated_frame_count` high: the logger may not be stepping the environment
- `action_imbalance_score` near `1.0`: one action dominates the dataset
- `broken_episode_count > 0`: rollout order may be wrong
- `train_test_leak_count > 0`: validation scores may be too optimistic

DEFINITION OF DONE
You are done when:
- the checks run before training
- the report names blocking data problems
- the report gives counts, not only pass/fail text
- at least one intentionally bad dataset is caught
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class DataQualityReport(TypedDict):
    """World-model dataset quality report."""

    num_steps: int
    missing_frame_count: int
    repeated_frame_count: int
    broken_episode_count: int
    action_imbalance_score: float
    bad_reward_cost_count: int
    train_test_leak_count: int
    passed: bool


@dataclass
class DataQualityConfig:
    """Settings for world-model data checks."""

    num_actions: int = 6
    repeated_frame_threshold: float = 1e-6
    max_action_imbalance_score: float = 0.8


def find_missing_frames(steps: list[dict[str, object]]) -> int:
    """Count rows with missing current or next frames."""
    # Goal:
    # Count rows that cannot be used for frame prediction.
    #
    # Method:
    # A row is missing frames if:
    # - `"image"` is missing or None
    # - `"next_image"` is missing or None
    #
    # Check:
    # - Return an integer count, not True or False.
    _ = steps
    raise NotImplementedError("Count missing frame rows.")


def find_repeated_frames(steps: list[dict[str, object]], config: DataQualityConfig) -> int:
    """Count rows where current and next frame are almost identical."""
    # Method:
    # Use tensor mean squared error when both frames are tensors.
    #
    # Formula:
    # `mse = ((image - next_image) ** 2).mean()`
    #
    # Count the row if `mse <= config.repeated_frame_threshold`.
    #
    # Check:
    # - A few repeated frames may be valid if the agent hits a wall.
    # - Many repeated frames usually mean the logger is broken.
    _ = (steps, config)
    raise NotImplementedError("Count repeated frame rows.")


def compute_action_imbalance_score(actions: torch.Tensor, config: DataQualityConfig) -> float:
    """Return a simple action imbalance score."""
    # Goal:
    # Report when one action dominates the dataset.
    #
    # Simple score:
    # `largest_action_fraction`
    #
    # Example:
    # If 90 out of 100 actions are action `0`, the score is `0.90`.
    # That is a warning because the model can look good by always predicting `0`.
    _ = (actions, config)
    raise NotImplementedError("Compute action imbalance score.")


def run_world_model_data_quality_checks(
    steps: list[dict[str, object]],
    config: DataQualityConfig,
) -> DataQualityReport:
    """Run all data-quality checks and return one report."""
    # Goal:
    # Build one clear report before training a world model.
    #
    # Method:
    # 1. Count missing frames with `find_missing_frames(...)`.
    # 2. Count repeated frames with `find_repeated_frames(...)`.
    # 3. Build an action tensor from row actions.
    # 4. Compute action imbalance.
    # 5. Check episode order:
    #    within each `episode_id`, `step_id` should count upward.
    # 6. Check reward and cost:
    #    they should be finite numbers, not strings, NaN, or None.
    # 7. Check train/test leakage:
    #    the same `(episode_id, step_id)` should not appear in more than one split.
    # 8. Set `passed=False` if any blocking count is nonzero or action imbalance
    #    is above `config.max_action_imbalance_score`.
    #
    # Output:
    # Return every count in `DataQualityReport`.
    _ = (steps, config)
    raise NotImplementedError("Run world-model data quality checks.")


def main() -> None:
    """Run the world-model data-quality scaffold."""
    config = DataQualityConfig()
    try:
        print(run_world_model_data_quality_checks([], config))
    except NotImplementedError as error:
        print(f"[TODO] run_world_model_data_quality_checks: {error}")


if __name__ == "__main__":
    main()
