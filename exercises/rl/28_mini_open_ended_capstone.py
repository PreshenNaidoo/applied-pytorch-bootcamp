"""
============================================================
EXERCISE 28: MINI OPEN ENDED CAPSTONE
============================================================

WHAT YOU ARE BUILDING
You are building a small open-ended learning system.

This capstone combines ideas from exercises 94 to 100:
- world-model uncertainty
- image or state-based task checks
- curriculum learning
- population self-play
- novelty search
- quality-diversity
- POET-style environment generation

The goal is not to build a large research system.
The goal is to connect the parts into one clear loop.

MAIN LOOP
At a high level, your system should:

```text
1. keep a population of agents
2. keep an archive of different useful behaviors
3. keep a set of generated gridworld tasks
4. train agents on tasks
5. generate new tasks
6. keep tasks that are solvable and useful
7. track progress over time
```

WHY THIS EXERCISE EXISTS
Open-endedness is about continuing to find new useful things.
That requires more than a single reward number.

This capstone asks you to track:
- quality: how good an agent is
- diversity: how different its behavior is
- difficulty: how hard a task is
- uncertainty: how much the model disagrees or lacks confidence

Keep the implementation small.
Use simple gridworlds and simple policies.
The lesson is the system design.

HOW THIS FITS INTO THE BOOTCAMP
This is the capstone for the open-ended learning block.
It should reuse the ideas from exercises 94 to 100 instead of adding many new ideas.

Beginner intuition:
- tasks give agents things to practice
- agents create behavior
- the archive keeps useful and different behavior
- task generation keeps the system from running out of challenges
- uncertainty gives a warning when a rollout may be less trustworthy

COMMON MISTAKES
- making the capstone too large
- adding new complex models before the simple loop works
- keeping unsolvable tasks
- tracking only best score and ignoring diversity
- using uncertainty without explaining what it means in this toy setup
- updating the archive before checking descriptor and score shapes

STEP-BY-STEP TODO
1. Build the seed task.
2. Check task solvability.
3. Estimate task difficulty.
4. Mutate tasks while keeping them solvable.
5. Initialize a population of agents.
6. Roll out policies on tasks.
7. Compute behavior descriptors.
8. Map descriptors to archive bins.
9. Estimate simple rollout uncertainty.
10. Update the behavior archive.
11. Improve the population.
12. Add new tasks when there is room.
13. Run the full open-ended loop and report progress.

DEFINITION OF DONE
You are done when:
- the system keeps a set of solvable tasks
- the population contains valid policies
- the behavior archive stores different useful behaviors
- new tasks can be generated during the run
- the summary reports task count, agent count, filled bins, and best score
- you can explain what makes the loop open-ended
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import TypedDict

import torch


# Learning order for this exercise:
# 1. Make tasks valid and solvable first.
# 2. Make one policy rollout work on one task.
# 3. Build descriptors and archive bins before updating the archive.
# 4. Add population improvement after the archive works.
# 5. Add task generation last, then run the full loop.
#
# The goal is to connect the earlier ideas into one small system.

State = tuple[int, int]
Policy = torch.Tensor
TaskId = str
BehaviorBin = tuple[int, int]


@dataclass
class OpenEndedTask:
    """One generated task."""

    task_id: TaskId
    rows: int
    cols: int
    start: State
    goal: State
    walls: set[State]
    difficulty: float


class AgentRecord(TypedDict):
    """One agent in the population."""

    policy: Policy
    score: float
    task_id: TaskId


class BehaviorElite(TypedDict):
    """One behavior kept in the archive."""

    policy: Policy
    score: float
    descriptor: torch.Tensor
    task_id: TaskId


class CapstoneSummary(TypedDict):
    """Final numbers returned by the capstone."""

    task_count: int
    agent_count: int
    filled_behavior_bins: int
    best_score: float
    mean_task_difficulty: float
    note: str


@dataclass
class OpenEndedCapstoneConfig:
    """Settings for the mini open-ended capstone."""

    action_count: int = 4
    initial_rows: int = 5
    initial_cols: int = 5
    max_tasks: int = 10
    population_size: int = 12
    outer_iterations: int = 15
    rollout_steps: int = 35
    bin_count: int = 5
    mutation_scale: float = 0.35
    uncertainty_penalty: float = 0.2
    random_state: int = 42


def normalize_policy(raw_policy: torch.Tensor) -> Policy:
    """Return action probabilities."""
    # Goal:
    # Convert raw action values into a valid policy.
    #
    # Input:
    # - tensor shaped `[4]`.
    #
    # Output:
    # - tensor shaped `[4]`.
    # - values sum to `1.0`.
    #
    # Method:
    # Use `torch.softmax(raw_policy.float(), dim=0)`.
    raise NotImplementedError("Normalize capstone policy.")


def build_seed_task(config: OpenEndedCapstoneConfig) -> OpenEndedTask:
    """Return the first task."""
    # Goal:
    # Create the first simple gridworld task.
    #
    # Output:
    # - one `OpenEndedTask`.
    #
    # Suggested values:
    # - `task_id="task_000"`
    # - start `(0, 0)`
    # - goal `(rows - 1, cols - 1)`
    # - no walls
    # - difficulty from `estimate_task_difficulty(task)`
    #
    # Checks:
    # - The task should be solvable.
    raise NotImplementedError("Build seed open-ended task.")


def step_task(state: State, action: int, task: OpenEndedTask) -> State:
    """Move one step in a task."""
    # Goal:
    # Apply grid movement while respecting walls.
    #
    # Rule:
    # If an action would leave the grid or hit a wall, stay in place.
    #
    # Checks:
    # - The returned state should be inside the grid.
    # - The returned state should not be a wall.
    raise NotImplementedError("Step one capstone task.")


def shortest_path_length(task: OpenEndedTask) -> int | None:
    """Return the shortest path length, or None if no path exists."""
    # Goal:
    # Find the fewest steps from `task.start` to `task.goal`.
    #
    # Input:
    # - `task`: one generated gridworld task.
    #
    # Output:
    # - integer number of steps if the task is solvable.
    # - `None` if no path exists.
    #
    # Method:
    # Use breadth-first search.
    # Breadth-first search checks all states 1 step away,
    # then all states 2 steps away, and so on.
    #
    # Use:
    # ```python
    # from collections import deque
    # queue = deque([(task.start, 0)])
    # visited = {task.start}
    # ```
    #
    # Neighbor actions:
    # - use actions `0`, `1`, `2`, and `3`
    # - call `step_task(state, action, task)` to get the neighbor
    #
    # Steps:
    # 1. Return `None` if start or goal is a wall.
    # 2. Put `(task.start, 0)` in the queue.
    # 3. Pop one `(state, distance)` pair.
    # 4. If `state == task.goal`, return `distance`.
    # 5. Add valid unvisited neighbors with distance plus 1.
    # 6. Return `None` if the queue becomes empty.
    #
    # Checks:
    # - A no-wall 5 by 5 grid from `(0, 0)` to `(4, 4)` should return `8`.
    # - A fully blocked goal should return `None`.
    raise NotImplementedError("Return shortest path length for a capstone task.")


def is_task_solvable(task: OpenEndedTask) -> bool:
    """Return whether the task has a path from start to goal."""
    # Goal:
    # Reject generated tasks that cannot be solved.
    #
    # Method:
    # Use `shortest_path_length(task)`.
    #
    # Return:
    # ```python
    # shortest_path_length(task) is not None
    # ```
    #
    # Steps:
    # 1. Call `shortest_path_length(task)`.
    # 2. Return `True` if it is not `None`.
    # 3. Return `False` if it is `None`.
    #
    # Checks:
    # - A no-wall grid should be solvable.
    # - A task with the goal fully blocked should be unsolvable.
    raise NotImplementedError("Check task solvability.")


def estimate_task_difficulty(task: OpenEndedTask) -> float:
    """Return a simple task difficulty score."""
    # Goal:
    # Give each task a rough difficulty number.
    #
    # Simple formula:
    # ```text
    # difficulty = shortest_path_length + wall_count * 0.25
    # ```
    #
    # where:
    # - `shortest_path_length` is the fewest steps from start to goal
    # - `wall_count` is `len(task.walls)`
    #
    # Method:
    # Use the helper `shortest_path_length(task)`.
    # If it returns `None`, return a large value such as `float("inf")`.
    #
    # Steps:
    # 1. Get `path_length = shortest_path_length(task)`.
    # 2. If `path_length is None`, return `float("inf")`.
    # 3. Set `wall_count = len(task.walls)`.
    # 4. Return `float(path_length + wall_count * 0.25)`.
    #
    # Checks:
    # - Adding useful walls should usually increase difficulty.
    raise NotImplementedError("Estimate task difficulty.")


def mutate_task(parent: OpenEndedTask, new_task_id: TaskId, config: OpenEndedCapstoneConfig) -> OpenEndedTask | None:
    """Generate a new task from an old task."""
    # Goal:
    # Create a new solvable task.
    #
    # Mutation choices:
    # - add a wall
    # - remove a wall
    # - move the goal to another valid cell
    #
    # Rules:
    # - Never put a wall on the start.
    # - Never put a wall on the goal.
    # - Return `None` when the task is not solvable.
    #
    # Steps:
    # 1. Copy the parent fields.
    # 2. Apply one random mutation.
    # 3. Build the new task.
    # 4. Check solvability.
    # 5. Estimate difficulty.
    # 6. Return the new task or `None`.
    raise NotImplementedError("Mutate one capstone task.")


def rollout_policy(policy: Policy, task: OpenEndedTask, config: OpenEndedCapstoneConfig) -> tuple[list[State], float, bool]:
    """Run one policy on one task."""
    # Goal:
    # Return visited states, total reward, and whether the goal was reached.
    #
    # Reward rule:
    # - `1.0` for reaching the goal.
    # - `-0.01` for each step before the goal.
    #
    # Method:
    # Start at `task.start`.
    # Sample actions from the policy with `torch.multinomial(policy, 1).item()`.
    #
    # Checks:
    # - Stop early when the goal is reached.
    # - Always include the start state in the returned state list.
    raise NotImplementedError("Roll out capstone policy.")


def behavior_descriptor(states: list[State], task: OpenEndedTask) -> torch.Tensor:
    """Return a normalized final-position descriptor."""
    # Goal:
    # Summarize one behavior in two numbers.
    #
    # Output:
    # - tensor shaped `[2]`.
    #
    # Values:
    # - final row divided by `task.rows - 1`
    # - final col divided by `task.cols - 1`
    #
    # Checks:
    # - Values should be between `0.0` and `1.0`.
    raise NotImplementedError("Compute capstone behavior descriptor.")


def descriptor_to_bin(descriptor: torch.Tensor, config: OpenEndedCapstoneConfig) -> BehaviorBin:
    """Map a descriptor to an archive bin."""
    # Goal:
    # Convert a continuous descriptor into a discrete bin.
    #
    # Formula:
    # ```text
    # bin_index = int(value * bin_count)
    # ```
    #
    # Clamp to `bin_count - 1` so value `1.0` is valid.
    #
    # Checks:
    # - Returned indexes should be inside the archive grid.
    raise NotImplementedError("Map capstone descriptor to bin.")


def estimate_uncertainty(states: list[State], task: OpenEndedTask) -> float:
    """Return a simple uncertainty score for one rollout."""
    # Goal:
    # Mark rollouts as uncertain when they visit areas with little signal.
    #
    # Simple teaching version:
    # Count how often the agent fails to move.
    # Since this function receives only states, a failed move appears as two
    # repeated states in a row.
    # More repeated states means higher uncertainty.
    #
    # Output:
    # - float between `0.0` and `1.0`.
    #
    # Checks:
    # - A rollout that moves smoothly should have lower uncertainty.
    # - If every state is different, uncertainty should be `0.0`.
    # - This is a simple proxy, not a full learned uncertainty model.
    raise NotImplementedError("Estimate simple rollout uncertainty.")


def update_behavior_archive(
    archive: dict[BehaviorBin, BehaviorElite],
    policy: Policy,
    task: OpenEndedTask,
    states: list[State],
    score: float,
    config: OpenEndedCapstoneConfig,
) -> None:
    """Keep the best behavior for each archive bin."""
    # Goal:
    # Store useful and different behaviors.
    #
    # Quality score:
    # ```text
    # adjusted_score = score - uncertainty_penalty * uncertainty
    # ```
    #
    # where:
    # - `score` is the rollout reward
    # - `uncertainty` comes from `estimate_uncertainty(...)`
    # - `uncertainty_penalty` controls how much uncertainty hurts the score
    #
    # Steps:
    # 1. Compute descriptor.
    # 2. Convert descriptor to a bin.
    # 3. Compute uncertainty.
    # 4. Compute adjusted score.
    # 5. Store this policy if the bin is empty or the adjusted score is better.
    #
    # Checks:
    # - This function should update `archive` in place.
    raise NotImplementedError("Update capstone behavior archive.")


def improve_population_on_tasks(
    agents: list[AgentRecord],
    tasks: dict[TaskId, OpenEndedTask],
    archive: dict[BehaviorBin, BehaviorElite],
    config: OpenEndedCapstoneConfig,
) -> list[AgentRecord]:
    """Improve agents and update the archive."""
    # Goal:
    # Train the population with mutation and selection.
    #
    # Method:
    # For each agent:
    # - get its task from `tasks`
    # - mutate its policy
    # - roll out old and new policies
    # - keep the better one
    # - update the behavior archive
    #
    # Checks:
    # - Return exactly `config.population_size` agents.
    raise NotImplementedError("Improve capstone population.")


def maybe_add_new_tasks(tasks: dict[TaskId, OpenEndedTask], config: OpenEndedCapstoneConfig) -> int:
    """Generate new tasks when there is room."""
    # Goal:
    # Grow the task set with solvable mutated tasks.
    #
    # Output:
    # - number of tasks added.
    #
    # Steps:
    # 1. Stop if `len(tasks) >= config.max_tasks`.
    # 2. Pick parent tasks at random.
    # 3. Create new task ids like `"task_003"`.
    # 4. Call `mutate_task(...)`.
    # 5. Add non-None tasks to the dictionary.
    #
    # Checks:
    # - Do not exceed `config.max_tasks`.
    # - Every added task should be solvable.
    raise NotImplementedError("Maybe add new capstone tasks.")


def initialize_agents(tasks: dict[TaskId, OpenEndedTask], config: OpenEndedCapstoneConfig) -> list[AgentRecord]:
    """Create the first population of agents."""
    # Goal:
    # Build agents and assign them to available tasks.
    #
    # Output:
    # - list of `AgentRecord` dictionaries.
    #
    # Method:
    # Use random policies from `normalize_policy(torch.randn(config.action_count))`.
    # Assign each agent to a task id from `tasks`.
    #
    # Checks:
    # - List length should equal `config.population_size`.
    # - Every `task_id` should exist in `tasks`.
    raise NotImplementedError("Initialize capstone agents.")


def run_open_ended_capstone(config: OpenEndedCapstoneConfig) -> CapstoneSummary:
    """Run the mini open-ended capstone."""
    # Goal:
    # Combine task generation, population improvement, and behavior archiving.
    #
    # Output keys:
    # - `task_count`
    # - `agent_count`
    # - `filled_behavior_bins`
    # - `best_score`
    # - `mean_task_difficulty`
    # - `note`
    #
    # Steps:
    # 1. Set random seeds.
    # 2. Build the seed task.
    # 3. Initialize agents.
    # 4. Create an empty behavior archive.
    # 5. For each outer iteration:
    #    - improve the population
    #    - update the archive
    #    - maybe add new tasks
    # 6. Compute final stats.
    # 7. Return the summary.
    #
    # Checks:
    # - The note should explain what made the loop open-ended.
    raise NotImplementedError("Run mini open-ended capstone.")


def main() -> None:
    """Run the mini open-ended capstone scaffold."""
    config = OpenEndedCapstoneConfig()
    try:
        summary = run_open_ended_capstone(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return
    print(summary)


if __name__ == "__main__":
    main()
