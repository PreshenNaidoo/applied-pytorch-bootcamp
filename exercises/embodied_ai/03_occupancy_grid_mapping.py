"""
============================================================
EXERCISE 03: OCCUPANCY GRID MAPPING
============================================================

WHAT YOU ARE BUILDING
You are building a tiny occupancy grid map.

Plain-language meanings:
- occupancy grid means a map made of cells
- free means the robot thinks the cell is open
- blocked means the robot thinks an obstacle is there
- unknown means the robot has not seen enough yet
- sensor ray means one distance measurement from the robot into the world

WHY THIS EXERCISE EXISTS
Robots often plan using maps of where they can and cannot move.
This exercise teaches the map shape used by many navigation systems.

This should be a full mapping-and-navigation mini-project. The data is a set of
sensor rays collected from a real grid scene, and the final check is whether a
planner can use the built map to reach a goal.

DATASET OPTIONS
Default: use a local hidden grid scene and simulated sensor rays.

Real dataset option after the default works:
- KITTI Odometry: use LiDAR or stereo-derived obstacle points to practice larger occupancy-grid mapping.

CORE LOOP IN PLAIN WORDS

```text
start with every cell unknown
cast sensor rays from the robot
mark cells before the obstacle as free
mark the hit cell as blocked
plan a path through free cells
```

DEFINITION OF DONE
You are done when:
- sensor rays are simulated
- free, blocked, and unknown cells are updated
- a simple path can be planned
- map accuracy and path success are reported
- you can explain why unknown is not the same as free
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import torch


@dataclass
class OccupancyGridConfig:
    """Settings for a tiny occupancy grid."""

    height: int = 10
    width: int = 10
    unknown_value: int = -1
    free_value: int = 0
    blocked_value: int = 1


def create_unknown_grid(config: OccupancyGridConfig) -> torch.Tensor:
    """Return a grid where every cell is unknown."""
    # Goal:
    # Start the map before the robot has seen anything.
    #
    # Input:
    # - grid height and width from config.
    #
    # Output:
    # - integer tensor shaped `[height, width]`
    # - every value should be `config.unknown_value`
    #
    # Method:
    # Use `torch.full((config.height, config.width), config.unknown_value, dtype=torch.long)`.
    #
    # Checks:
    # - Grid shape should be `(height, width)`.
    raise NotImplementedError("Create an unknown occupancy grid.")


def cast_sensor_ray(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Return cells along one simple sensor ray."""
    # Goal:
    # List grid cells between the robot and a sensed obstacle.
    #
    # Input:
    # - `start`: `(row, col)` robot cell
    # - `end`: `(row, col)` hit cell
    #
    # Output:
    # - list of `(row, col)` cells from start toward end.
    #
    # Method:
    # Use a simple teaching version:
    # - support horizontal and vertical rays first
    # - add diagonal rays later if you want
    #
    # Steps:
    # 1. If the row is the same, walk columns from start to end.
    # 2. If the column is the same, walk rows from start to end.
    # 3. Include the end cell.
    # 4. Return the cell list.
    #
    # Checks:
    # - The last cell should equal `end`.
    raise NotImplementedError("Cast a simple grid sensor ray.")


def update_grid_with_ray(
    grid: torch.Tensor,
    ray_cells: list[tuple[int, int]],
    config: OccupancyGridConfig,
) -> torch.Tensor:
    """Update free and blocked cells from one ray."""
    # Goal:
    # Mark cells along a sensor ray.
    #
    # Input:
    # - `grid`: tensor shaped `[height, width]`
    # - `ray_cells`: cells from robot to obstacle
    #
    # Output:
    # - updated grid tensor
    #
    # Method:
    # - every cell before the final cell becomes free
    # - the final cell becomes blocked
    #
    # Steps:
    # 1. Copy the grid so the original is not changed by surprise.
    # 2. Mark `ray_cells[:-1]` as free.
    # 3. Mark `ray_cells[-1]` as blocked.
    # 4. Return the updated grid.
    #
    # Checks:
    # - Unknown cells not touched by the ray should stay unknown.
    raise NotImplementedError("Update occupancy grid with one sensor ray.")


def plan_path_bfs(
    grid: torch.Tensor,
    start: tuple[int, int],
    goal: tuple[int, int],
    config: OccupancyGridConfig,
) -> list[tuple[int, int]]:
    """Plan a path through free cells with BFS."""
    # Goal:
    # Find a path from start to goal using only free cells.
    #
    # Input:
    # - `grid`: `[height, width]`
    # - `start`: `(row, col)`
    # - `goal`: `(row, col)`
    #
    # Output:
    # - list of cells from start to goal.
    # - return an empty list if no path exists.
    #
    # Method:
    # Use breadth-first search, called BFS.
    # BFS checks nearby cells before farther cells.
    #
    # Steps:
    # 1. Use `deque([start])` as the queue.
    # 2. Store parents in a dictionary so you can rebuild the path.
    # 3. Try four neighbors: up, down, left, right.
    # 4. Skip cells outside the grid.
    # 5. Skip blocked and unknown cells.
    # 6. Stop when the goal is found.
    # 7. Rebuild and return the path.
    #
    # Checks:
    # - The path should start at `start` and end at `goal`.
    _ = deque
    raise NotImplementedError("Plan a path with BFS.")


def explain_unknown_vs_free() -> str:
    """Return a short map explanation."""
    return "Unknown means the robot has not checked the cell yet. Free means the robot has evidence the cell is open."


def run_occupancy_grid_mini_project(config: OccupancyGridConfig) -> dict[str, float]:
    """Build a map from rays, plan a path, and report navigation checks."""
    # Goal:
    # Run the complete occupancy-grid workflow on a small scene.
    #
    # Output:
    # - dictionary with:
    #   - `"known_cell_fraction"`
    #   - `"blocked_cell_accuracy"`
    #   - `"path_found"`
    #   - `"path_length"`
    #
    # Method:
    # 1. Create a hidden true grid with free and blocked cells.
    # 2. Start with an unknown map.
    # 3. Cast several sensor rays from robot positions.
    # 4. Update the map with `update_grid_with_ray(...)`.
    # 5. Plan a path with `plan_path_bfs(...)`.
    # 6. Compare the built map with the hidden true grid.
    #
    # Checks:
    # - The planner should avoid blocked and unknown cells.
    # - A map with more useful rays should usually find better paths.
    raise NotImplementedError("Run the occupancy grid mini-project.")


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
    """Run the full occupancy-grid scaffold."""
    config = OccupancyGridConfig()
    metrics, ok = report_task_progress("run_occupancy_grid_mini_project", run_occupancy_grid_mini_project, config)
    if ok:
        print(metrics)


if __name__ == "__main__":
    main()
