"""
============================================================
EXERCISE 23: CURRICULUM LEARNING
============================================================

WHAT YOU ARE BUILDING
You are building a small curriculum for an RL agent.

Curriculum learning means:
- start with easy tasks
- train until the agent is good enough
- then move to harder tasks

Small example:
- task 1: 3 by 3 grid, goal is close
- task 2: 4 by 4 grid, goal is farther away
- task 3: 5 by 5 grid, goal is even farther away

WHY THIS EXERCISE EXISTS
Self-improving systems often need a good order of practice tasks.
If the first task is too hard, the agent may learn nothing.
If the task never gets harder, the agent stops improving.

This exercise uses a simple tabular Q-learning agent so the focus stays on the curriculum.

Q-table shape:
- `[num_states, action_count]`

Q-learning update:

```text
new_q = old_q + learning_rate * (target - old_q)
target = reward + discount * best_next_q
```

where:
- `old_q` is the old value for the action the agent took
- `reward` is the reward from the environment
- `discount` is how much future reward matters
- `best_next_q` is the best Q-value from the next state

HOW THIS FITS INTO THE BOOTCAMP
Earlier RL exercises train on one task at a time.
This exercise teaches how the order of tasks can help learning.

Beginner intuition:
- an easy task gives the agent a first useful skill
- a harder task can build on that skill
- a task that is too hard too early may give mostly useless feedback

COMMON MISTAKES
- making all tasks the same difficulty
- evaluating with random exploration still turned on
- moving to the next task before the current task is learned
- using the wrong Q-table shape after the grid size changes
- adding future reward even when the episode is done

STEP-BY-STEP TODO
1. Build easy-to-hard grid tasks.
2. Convert grid states to Q-table indexes.
3. Implement gridworld stepping for one task.
4. Create a Q-table.
5. Choose epsilon-greedy actions.
6. Apply the Q-learning update.
7. Train on one task.
8. Evaluate without exploration.
9. Move through the curriculum only when the success rate is high enough.

DEFINITION OF DONE
You are done when:
- tasks are ordered from easier to harder
- Q-learning runs on each task
- evaluation uses greedy actions only
- the summary reports tasks completed and final success rate
- you can explain why easy-to-hard training can help
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TypedDict

import torch


# Learning order for this exercise:
# 1. Build the task list first.
# 2. Make state indexing and stepping correct before training.
# 3. Test one Q-learning update by hand.
# 4. Train and evaluate one task before running the full curriculum.
# 5. Move to harder tasks only through the curriculum loop.
#
# The goal is to learn how task order changes training.

State = tuple[int, int]


class CurriculumSummary(TypedDict):
    """Final numbers returned by the curriculum demo."""

    tasks_completed: int
    final_success_rate: float
    total_episodes: int
    note: str


@dataclass
class GridTask:
    """One curriculum task."""

    rows: int
    cols: int
    goal: State
    name: str


@dataclass
class CurriculumConfig:
    """Settings for curriculum learning."""

    action_count: int = 4
    episodes_per_task: int = 120
    eval_episodes: int = 30
    max_steps_per_episode: int = 40
    learning_rate: float = 0.2
    discount: float = 0.95
    epsilon: float = 0.15
    pass_success_rate: float = 0.75
    random_state: int = 42


def build_curriculum() -> list[GridTask]:
    """Return tasks from easiest to hardest."""
    # Goal:
    # Build a list of gridworld tasks in learning order.
    #
    # Output:
    # - list of `GridTask` objects.
    #
    # Small example:
    # - `GridTask(rows=3, cols=3, goal=(2, 2), name="easy_3x3")`
    #
    # Method:
    # Write the list directly in this function.
    #
    # Steps:
    # 1. Start with a small grid.
    # 2. Add a medium grid.
    # 3. Add a larger grid.
    # 4. Put easier tasks before harder tasks.
    # 5. Return the list.
    #
    # Checks:
    # - The first task should be easier than the last task.
    # - Every goal should be inside its grid.
    raise NotImplementedError("Build curriculum tasks.")


def state_to_index(state: State, task: GridTask) -> int:
    """Convert a grid state into one integer index."""
    # Goal:
    # Map `(row, col)` to one row in the Q-table.
    #
    # Input:
    # - `state`: `(row, col)`.
    # - `task.rows` and `task.cols`: grid size.
    #
    # Output:
    # - integer state index from `0` to `rows * cols - 1`.
    #
    # Formula:
    # ```text
    # index = row * task.cols + col
    # ```
    #
    # Example:
    # - On a 4-column grid, state `(2, 1)` becomes `2 * 4 + 1 = 9`.
    #
    # Checks:
    # - `(0, 0)` should map to `0`.
    # - The bottom-right cell should map to `rows * cols - 1`.
    raise NotImplementedError("Convert state to index.")


def step_task(state: State, action: int, task: GridTask) -> tuple[State, float, bool]:
    """Take one step in one curriculum task."""
    # Goal:
    # Move inside the task grid and return reward.
    #
    # Output:
    # - next state
    # - reward
    # - done flag
    #
    # Reward rule:
    # - `1.0` if the agent reaches `task.goal`
    # - `-0.01` for every other step
    #
    # Steps:
    # 1. Check state and action.
    # 2. Move by action.
    # 3. Clamp inside the grid.
    # 4. Check whether the next state equals the goal.
    # 5. Return `(next_state, reward, done)`.
    #
    # Checks:
    # - The agent should never leave the grid.
    # - Reaching the goal should return `done=True`.
    raise NotImplementedError("Step one curriculum task.")


def make_q_table(task: GridTask, config: CurriculumConfig) -> torch.Tensor:
    """Create a Q-table for one task."""
    # Goal:
    # Return a zero Q-table.
    #
    # Shape:
    # - `[task.rows * task.cols, config.action_count]`
    #
    # Method:
    # Use:
    # `torch.zeros((task.rows * task.cols, config.action_count), dtype=torch.float32)`.
    #
    # Checks:
    # - Every starting Q-value should be `0.0`.
    raise NotImplementedError("Create a Q-table.")


def choose_epsilon_greedy_action(q_table: torch.Tensor, state: State, task: GridTask, config: CurriculumConfig) -> int:
    """Choose one action with epsilon-greedy exploration."""
    # Goal:
    # Sometimes explore, sometimes pick the best known action.
    #
    # Input:
    # - `q_table`: `[num_states, action_count]`.
    # - `state`: current grid state.
    #
    # Method:
    # - With probability `config.epsilon`, return a random action.
    # - Otherwise, return the action with the largest Q-value for this state.
    #
    # Use:
    # - `random.random()`
    # - `random.randrange(config.action_count)`
    # - `torch.argmax(q_values).item()`
    #
    # Checks:
    # - The returned action must be between `0` and `action_count - 1`.
    raise NotImplementedError("Choose an epsilon-greedy action.")


def q_learning_update(
    q_table: torch.Tensor,
    state: State,
    action: int,
    reward: float,
    next_state: State,
    done: bool,
    task: GridTask,
    config: CurriculumConfig,
) -> None:
    """Update one Q-table entry in place."""
    # Goal:
    # Apply one Q-learning update.
    #
    # Formula:
    # ```text
    # target = reward                         if done
    # target = reward + discount * best_next_q if not done
    # new_q = old_q + learning_rate * (target - old_q)
    # ```
    #
    # Input:
    # - `q_table`: updated in place.
    # - `state`, `action`, `reward`, `next_state`, `done`: one transition.
    #
    # Steps:
    # 1. Convert `state` and `next_state` to indexes.
    # 2. Read `old_q`.
    # 3. Compute `best_next_q`.
    # 4. Compute `target`.
    # 5. Compute `new_q`.
    # 6. Store `new_q` back into `q_table[state_index, action]`.
    #
    # Checks:
    # - If `done=True`, do not add future reward.
    # - Only one table entry should change.
    raise NotImplementedError("Apply one Q-learning update.")


def train_on_task(task: GridTask, config: CurriculumConfig, q_table: torch.Tensor | None = None) -> torch.Tensor:
    """Train the agent on one task."""
    # Goal:
    # Train with Q-learning on one grid task.
    #
    # Input:
    # - `task`: the current curriculum task.
    # - `q_table`: optional old table. Use it only if it has the right shape.
    #
    # Output:
    # - trained Q-table for this task.
    #
    # Steps:
    # 1. Create a Q-table if needed.
    # 2. For each episode, start at `(0, 0)`.
    # 3. Choose actions with epsilon-greedy.
    # 4. Step the task.
    # 5. Apply the Q-learning update.
    # 6. Stop the episode when done or after max steps.
    # 7. Return the Q-table.
    #
    # Checks:
    # - The returned Q-table shape should match the current task.
    raise NotImplementedError("Train on one curriculum task.")


def evaluate_task(q_table: torch.Tensor, task: GridTask, config: CurriculumConfig) -> float:
    """Return the success rate for one task."""
    # Goal:
    # Measure how often the greedy policy reaches the goal.
    #
    # Method:
    # Use no random exploration during evaluation.
    # At each state, choose `torch.argmax(q_values)`.
    #
    # Output:
    # - success rate between `0.0` and `1.0`.
    #
    # Steps:
    # 1. Run `config.eval_episodes` episodes.
    # 2. Start each episode at `(0, 0)`.
    # 3. Pick greedy actions.
    # 4. Count episodes that reach the goal.
    # 5. Return `successes / eval_episodes`.
    #
    # Checks:
    # - Do not use epsilon during evaluation.
    raise NotImplementedError("Evaluate one curriculum task.")


def run_curriculum(config: CurriculumConfig) -> CurriculumSummary:
    """Run the full curriculum learning demo."""
    # Goal:
    # Train tasks in order and move on only when the agent is good enough.
    #
    # Output keys:
    # - `tasks_completed`
    # - `final_success_rate`
    # - `total_episodes`
    # - `note`
    #
    # Steps:
    # 1. Build the curriculum.
    # 2. Train on the first task.
    # 3. Evaluate it.
    # 4. Move to the next task if success rate is high enough.
    # 5. Stop if a task is not passed.
    # 6. Return the summary.
    #
    # Checks:
    # - Tasks should be attempted in the order returned by `build_curriculum()`.
    # - The note should explain why easy-to-hard training helps.
    raise NotImplementedError("Run curriculum learning.")


def main() -> None:
    """Run the curriculum learning scaffold."""
    config = CurriculumConfig()
    try:
        summary = run_curriculum(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return
    print(summary)


if __name__ == "__main__":
    main()
