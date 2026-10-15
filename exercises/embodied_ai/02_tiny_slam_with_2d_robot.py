"""
============================================================
EXERCISE 02: TINY SLAM WITH 2D ROBOT
============================================================

WHAT YOU ARE BUILDING
You are building a tiny SLAM-style loop for a 2D robot.

Plain-language meanings:
- SLAM means simultaneous localization and mapping
- localization means estimating where the robot is
- mapping means estimating where landmarks or obstacles are
- noisy movement means the robot does not move exactly as commanded
- noisy sensors mean measurements are not perfectly correct

WHY THIS EXERCISE EXISTS
Embodied agents often need to build a map while also tracking where they are.
This exercise gives the smallest useful picture of that loop.

This exercise should produce real logged robot-style data from the simulator:
movement commands, noisy observations, estimated poses, and a learned or
updated landmark map. The final check is map error and pose error over a full
trajectory.

DATASET OPTIONS
Default: use the local 2D robot simulator in this file.

Real dataset option after the default works:
- KITTI Odometry: use stereo/LiDAR-style odometry data and ground-truth poses for larger pose-error checks.

CORE LOOP IN PLAIN WORDS

```text
start with a robot pose guess
apply one movement command
receive noisy landmark observations
update the robot pose estimate
update the landmark map
repeat
```

DEFINITION OF DONE
You are done when:
- a robot path is simulated
- noisy observations are created
- pose estimates are updated
- a simple landmark map is built
- pose error and landmark error are reported
- you can explain why localization and mapping depend on each other
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict
import torch


class LandmarkObservation(TypedDict):
    """One observed landmark from the robot."""

    landmark_id: int
    range_meters: float
    bearing_radians: float


@dataclass
class TinySLAMConfig:
    """Settings for the tiny 2D SLAM exercise."""

    num_steps: int = 20
    movement_noise: float = 0.02
    sensor_noise: float = 0.05
    random_state: int = 42


def simulate_robot_motion(config: TinySLAMConfig) -> list[torch.Tensor]:
    """Return a simulated true robot path."""
    # Goal:
    # Simulate where the robot really goes.
    #
    # Input:
    # - `config.num_steps`: number of poses.
    #
    # Output:
    # - list of pose tensors.
    # - Each pose is shaped `[3]` and stores `[x, y, theta]`.
    #
    # Method:
    # Use a simple command like move forward a little and turn slowly.
    #
    # Steps:
    # 1. Start at pose `[0.0, 0.0, 0.0]`.
    # 2. For each step, update x, y, and theta.
    # 3. Add small random movement noise.
    # 4. Store every pose.
    #
    # Checks:
    # - The returned list length should equal `config.num_steps`.
    # - Each pose should have shape `[3]`.
    raise NotImplementedError("Simulate true 2D robot motion.")


def observe_landmarks(
    pose: torch.Tensor,
    landmarks: dict[int, torch.Tensor],
    config: TinySLAMConfig,
) -> list[LandmarkObservation]:
    """Return noisy landmark observations from one robot pose."""
    # Goal:
    # Simulate what the robot sensor sees from one pose.
    #
    # Input:
    # - `pose`: tensor shaped `[3]`, storing `[x, y, theta]`
    # - `landmarks`: maps landmark id to a tensor shaped `[2]`
    #
    # Output:
    # - list of observations.
    # - Each observation has:
    #   - `landmark_id`
    #   - `range_meters`
    #   - `bearing_radians`
    #
    # Method:
    # For each landmark:
    # - compute vector from robot to landmark
    # - range is vector length
    # - bearing is angle to landmark minus robot heading
    #
    # Steps:
    # 1. Read robot x, y, and theta.
    # 2. Loop over landmarks.
    # 3. Compute range with `torch.linalg.norm(...)`.
    # 4. Compute bearing with `torch.atan2(dy, dx) - theta`.
    # 5. Add sensor noise.
    # 6. Return observations.
    #
    # Checks:
    # - Ranges should be positive.
    raise NotImplementedError("Simulate noisy landmark observations.")


def update_landmark_map(
    pose_estimate: torch.Tensor,
    observations: list[LandmarkObservation],
    landmark_map: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Update landmark position estimates."""
    # Goal:
    # Convert robot-relative observations into world landmark positions.
    #
    # Input:
    # - `pose_estimate`: `[x, y, theta]`
    # - `observations`: list from `observe_landmarks`
    # - `landmark_map`: current estimated landmark positions
    #
    # Output:
    # - updated landmark map
    #
    # Method:
    # Convert range and bearing back to world x and y:
    # `world_angle = theta + bearing`
    # `landmark_x = robot_x + range * cos(world_angle)`
    # `landmark_y = robot_y + range * sin(world_angle)`
    #
    # Steps:
    # 1. Convert each observation to a world position.
    # 2. If the landmark is new, add it.
    # 3. If it already exists, average the old and new estimates.
    # 4. Return the updated map.
    #
    # Checks:
    # - The map values should be tensors shaped `[2]`.
    raise NotImplementedError("Update the landmark map.")


def explain_slam_dependency() -> str:
    """Return a short SLAM explanation."""
    return "A better pose estimate makes a better map, and a better map helps correct the pose estimate."


def run_slam_mini_project(config: TinySLAMConfig) -> dict[str, float]:
    """Run the full tiny SLAM loop and report map quality."""
    # Goal:
    # Connect motion, observations, pose estimates, and landmark map updates.
    #
    # Output:
    # - dictionary with:
    #   - `"mean_pose_error"`
    #   - `"mean_landmark_error"`
    #   - `"num_steps"`
    #
    # Method:
    # 1. Simulate the true robot path.
    # 2. Create fixed true landmarks.
    # 3. For each pose, collect noisy observations.
    # 4. Update the estimated landmark map.
    # 5. Compare estimated landmarks with true landmark positions.
    # 6. Compare estimated poses with true poses if your solution keeps pose estimates.
    #
    # Checks:
    # - Error should be reported over a full trajectory, not one step.
    # - More sensor noise should usually increase error.
    raise NotImplementedError("Run the tiny SLAM mini-project.")


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
    """Run the full tiny SLAM scaffold."""
    config = TinySLAMConfig()
    metrics, ok = report_task_progress("run_slam_mini_project", run_slam_mini_project, config)
    if ok:
        print(metrics)


if __name__ == "__main__":
    main()
