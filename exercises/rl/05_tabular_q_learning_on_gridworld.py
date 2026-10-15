"""
============================================================
EXERCISE 05: TABULAR Q LEARNING ON GRIDWORLD
============================================================

WHAT YOU ARE BUILDING
You are building a tiny gridworld agent that learns with tabular Q-learning.

Plain-language meanings:
- tabular means you store one number per state-action pair in a table
- Q-value means "how good is this action in this state?"
- temporal-difference update means you update from one step of experience
- episode means one run from start state until terminal state

WHY THIS EXERCISE EXISTS
This is often the first real RL algorithm students implement.
It is simple enough to inspect by hand,
but it already teaches learning from trial and error over time.
This exercise teaches:
- state and action
- Q-table
- epsilon-greedy exploration
- Bellman-style update
- terminal states
- training over episodes

HOW THIS FITS INTO THE RL SEQUENCE
This is the first full trial-and-error RL exercise in the track.
The previous MDP file used known transition rules.
This file changes the learning style:
- now the agent must learn from sampled experience
- now the agent updates from episodes and steps

Beginner intuition:
- imagine a tiny maze
- the agent starts in one cell
- it wants to reach the goal cell
- it gets a small negative reward for wasting steps
- it gets a positive reward at the goal
- the Q-table slowly learns which moves are worth taking

Small example:
- state `(0, 0)`, action `right`
- reward `-0.1`
- next state `(0, 1)`
- if the next state looks promising,
  the value of `right` at `(0, 0)` should go up

DATASET
No dataset file is needed.
Build a tiny local gridworld directly in code.

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- use a very small square grid such as `4 x 4`
- choose one start cell and one goal cell
- give a small negative reward for each step
- give a positive reward at the goal
- keep actions simple: up, down, left, right

INPUTS AND OUTPUTS
Input:
- current state
- chosen action
- reward
- next state
- done flag
Output:
- learned Q-table
- episode returns
- goal reach count
- greedy policy
- short Q-value note

METRICS TO TRACK
Track at least:
- episode returns
- goal reach count
- learned Q-table
- greedy policy

TARGET RESULT RANGE
Focus on sensible learning behavior, not on a perfect route every run.

Useful signs:
- the agent reaches the goal more often later in training
- good actions near the goal get larger Q-values
- the greedy policy points roughly toward the goal

COMMON MISTAKES
- forgetting to explore
- updating the wrong state-action entry
- bootstrapping from the next state even when the episode is done
- confusing one episode return with the learned Q-values
- making the grid too large for a first tabular exercise

What each step is doing in simple words:
1. build a tiny world with a clear goal
2. give every state-action pair a starting guess
3. choose actions with some exploration
4. see where the action leads and what reward it gives
5. update the chosen state-action value from that one step
6. repeat over many episodes so the table improves
7. read off the best action in each state
8. explain how the table stores useful experience

STEP-BY-STEP TODO
1. Build the tiny gridworld.
2. Build a Q-table with one value per state-action pair.
3. Choose actions with epsilon-greedy.
4. Step through the environment.
5. Update the Q-table from reward and next-state values.
6. Train for several episodes.
7. Extract the greedy policy from the learned Q-table.
8. Explain what the Q-values mean in plain words.

DEFINITION OF DONE
You are done when:
- the gridworld runs step by step
- the Q-table updates each episode
- the agent reaches the goal sometimes
- a greedy policy can be extracted
- you can explain the update in simple words
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TypeAlias, TypedDict

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.training_checks import print_pass_or_try_again
# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

@dataclass
class QLearningConfig:
    """Settings for the tabular Q-learning exercise."""
    grid_size: int = 4
    learning_rate: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    episodes: int = 200
    max_steps_per_episode: int = 30
    random_state: int = 42


State: TypeAlias = tuple[int, int]
QTable: TypeAlias = dict[State, dict[str, float]]
GreedyPolicy: TypeAlias = dict[State, str]


class QLearningTermSummary(TypedDict):
    """Plain-language Q-learning words used in this exercise."""

    gamma: float
    epsilon: float
    plain_words: dict[str, str]


class Gridworld(TypedDict):
    """A tiny gridworld written as plain Python data."""

    grid_size: int
    start_state: State
    goal_state: State
    actions: list[str]
    step_reward: float
    goal_reward: float


class QLearningSummary(TypedDict):
    """Final summary returned by Q-learning training."""

    q_table: QTable
    episode_returns: list[float]
    goal_reach_count: int
    greedy_policy: GreedyPolicy
    note: str


# Beginner hints for this file:
# - the Q-table stores one number for each state and action pair
# - a larger Q-value means "this action seems better from this state"
# - epsilon-greedy is still important because the agent does not know the best route at first
# - use a tiny grid so you can print the learned policy and inspect it
# - terminal states should stop the episode
def describe_q_learning_terms(config: QLearningConfig) -> QLearningTermSummary:
    """Return the main Q-learning words used here."""
    return {
        "gamma": config.gamma,
        "epsilon": config.epsilon,
        "plain_words": {
            "q_value": "how good one action looks in one state",
            "episode": "one run from start until finish",
            "td_update": "a one-step value update using reward and next-state estimate",
        },
    }

# Build the gridworld.
# Return a structure that includes at least:
# - start state
# - goal state
# - list of actions
# - state transition rules
def build_gridworld(config: QLearningConfig) -> Gridworld:
    """Return a tiny local gridworld."""
    # Goal:
    # Return the small gridworld settings the agent will use.
    # The environment should be small enough to inspect by hand.
    #
    # Input:
    # - `config`: QLearningConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"grid_size"`: int, grid width and height
    # - `"start_state"`: tuple[int, int], usually `(0, 0)`
    # - `"goal_state"`: tuple[int, int], usually `(grid_size - 1, grid_size - 1)`
    # - `"actions"`: list[str], exactly `["up", "down", "left", "right"]`
    # - `"step_reward"`: float, small negative reward such as `-0.1`
    # - `"goal_reward"`: float, positive reward such as `1.0`
    #
    # Coordinate example:
    # - `(row, col) = (0, 0)` is the top-left cell.
    # - `(row, col) = (3, 3)` is the bottom-right cell in a `4 x 4` grid.
    #
    # Method:
    # Use plain Python data only. No PyTorch is needed for tabular Q-learning here.
    #
    # Steps:
    # 1. Read `config.grid_size`.
    # 2. Set the start state to `(0, 0)`.
    # 3. Set the goal state to `(config.grid_size - 1, config.grid_size - 1)`.
    # 4. Store the four action strings.
    # 5. Store the step reward and goal reward.
    # 6. Return the dictionary.
    #
    # Checks:
    # - `grid_size` should be at least `2`.
    # - The start state and goal state should be different.
    # - The goal state is the only terminal state in this simple exercise.

    gridworld = {"grid_size": config.grid_size,
                 "start_state": (0,0),
                 "goal_state": (config.grid_size - 1, config.grid_size - 1),
                 "actions": ["up", "down", "left", "right"],
                 "step_reward": -0.1,
                 "goal_reward": 1.0}

    return gridworld
    #raise NotImplementedError("Return the gridworld structure with states, actions, start, and goal.")

# Return the starting state for one new episode.
def reset_environment(gridworld: Gridworld) -> State:
    """Return the start state."""
    # Goal:
    # Return the starting state for a new episode.
    #
    # Input:
    # - `gridworld`: Gridworld
    # - Example key: `gridworld["start_state"] == (0, 0)`
    #
    # Output:
    # - `tuple[int, int]`
    # - Example: `(0, 0)`
    #
    # Method:
    # Return:
    # `gridworld["start_state"]`
    #
    # Steps:
    # 1. Read the start state from the gridworld dictionary.
    # 2. Return it.
    #
    # Checks:
    # - The returned state should be inside the grid.
    # - Reset should not change the Q-table.

    return gridworld["start_state"]
    #raise NotImplementedError("Return the gridworld start state.")

# Apply one action in the environment.
# Return:
# - next state
# - reward
# - done flag
def step_environment(
    gridworld: Gridworld,
    state: State,
    action: str,
) -> tuple[State, float, bool]:
    """Run one gridworld step."""
    # Goal:
    # Apply one action in the gridworld and return what happened.
    #
    # Input:
    # - `gridworld`: Gridworld
    # - `state`: tuple[int, int], current `(row, col)`
    # - `action`: str, one of `"up"`, `"down"`, `"left"`, `"right"`
    #
    # Output:
    # - `next_state`: tuple[int, int]
    # - `reward`: float
    # - `done`: bool
    #
    # Example:
    # If `state == (0, 0)` and `action == "right"`, then `next_state` should be `(0, 1)`.
    # If the action would leave the grid, keep the agent in the same state.
    #
    # Default 4 x 4 grid:
    # ```text
    #          col 0   col 1   col 2   col 3
    # row 0      S       .       .       .
    # row 1      .       .       .       .
    # row 2      .       .       .       .
    # row 3      .       .       .       G
    # ```
    #
    # `S` is the start state `(0, 0)`.
    # `G` is the goal state `(3, 3)`.
    #
    # Method:
    # Split the state into row and column:
    # `row, col = state`
    #
    # Action effects:
    # - `"up"` means `row - 1`
    # - `"down"` means `row + 1`
    # - `"left"` means `col - 1`
    # - `"right"` means `col + 1`
    #
    # Clamp the new row and column so they stay between:
    # - `0`
    # - `gridworld["grid_size"] - 1`
    #
    # Steps:
    # 1. Compute the candidate next row and column from the action.
    # 2. Clamp the candidate position inside the grid.
    # 3. Build `next_state = (next_row, next_col)`.
    # 4. Set `done = next_state == gridworld["goal_state"]`.
    # 5. Use `goal_reward` if done, otherwise use `step_reward`.
    # 6. Return `(next_state, reward, done)`.
    #
    # Checks:
    # - `next_state` should always be inside the grid.
    # - `done` should be `True` only at the goal.
    # - This function should not update Q-values.

    actions = gridworld["actions"]

    row, col = state
    next_row, next_col = state
    if action == actions[0]: #up
        next_row = row-1 if row-1>=0 else 0
    elif action == actions[1]: #down
        next_row = row+1 if row+1 < gridworld["grid_size"] else row
    elif action == actions[2]: #left
        next_col = col-1 if col-1>=0 else 0
    elif action == actions[3]: #right
        next_col = col+1 if col+1 < gridworld["grid_size"] else col

    next_state = (next_row, next_col)
    done = next_state == gridworld["goal_state"]
    reward = gridworld["goal_reward"] if done else gridworld["step_reward"]

    return next_state, reward, done
    #raise NotImplementedError("Return next state, reward, and done flag.")

# Return the starting Q-table.
# A simple first version can use nested dictionaries.
def initialize_q_table(gridworld: Gridworld) -> QTable:
    """Return zero-initialized Q-values."""
    # Goal:
    # Return one starting Q-value for every state-action pair.
    # At the start, the agent has not learned anything, so all values can be `0.0`.
    #
    # Input:
    # - `gridworld`: Gridworld
    #
    # Output:
    # - `QTable`
    # - Shape in plain words: one outer key per grid cell, one inner key per action.
    # - Example for one state:
    #   `{(0, 0): {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}}`
    #
    # Method:
    # Use nested dictionaries:
    # - outer key: `(row, col)`
    # - inner key: action string
    # - inner value: Q-value float
    #
    # Steps:
    # 1. Create an empty dictionary called `q_table`.
    # 2. Loop over every row from `0` to `gridworld["grid_size"] - 1`.
    # 3. Loop over every column from `0` to `gridworld["grid_size"] - 1`.
    # 4. Create `state = (row, col)`.
    # 5. Set each action value to `0.0`.
    # 6. Return `q_table`.
    #
    # Checks:
    # - `len(q_table)` should equal `grid_size * grid_size`.
    # - `set(q_table[(0, 0)])` should equal `set(gridworld["actions"])`.

    q_table = {}
    for i in range(gridworld["grid_size"]):
        for j in range(gridworld["grid_size"]):
            state = (i, j)
            pairs = {action:0.0 for action in gridworld["actions"]}
            q_table[state] = pairs

    return q_table
    #raise NotImplementedError("Return one Q-value per state-action pair.")

def choose_action(
    q_table: QTable,
    state: State,
    epsilon: float,
) -> str:
    """Choose one action with epsilon-greedy."""
    # Goal:
    # Return one action using epsilon-greedy.
    # This lets the agent sometimes explore instead of always using its current best guess.
    #
    # Input:
    # - `q_table`: QTable
    # - `state`: tuple[int, int]
    # - `epsilon`: float
    #
    # Output:
    # - `str`
    # - One action from `q_table[state]`.
    #
    # Method:
    # Use:
    # - `actions = list(q_table[state])`
    # - `random.random()` to decide whether to explore
    # - `random.choice(actions)` for exploration
    # - `max(actions, key=lambda action: q_table[state][action])` for exploitation
    #
    # Rule:
    # - if `random.random() < epsilon`, choose a random action
    # - otherwise choose the action with the highest Q-value in this state
    #
    # Steps:
    # 1. Read the available actions for this state.
    # 2. Draw a random number.
    # 3. Explore if the number is below epsilon.
    # 4. Otherwise return the action with the largest Q-value.
    #
    # Checks:
    # - With `epsilon=0.0`, this should always choose the current best action.
    # - With `epsilon=1.0`, this should choose random actions.

    available_actions = list(q_table[state])  #returns ['up', 'down', 'left', 'right']
    # the above is equivalent to available_actions = list(q_table[state].keys())
    # Because looping over a dictionary gives its keys by default.

    random_num = random.random()
    if random_num < epsilon:
        # Exploration
        random_int = random.randint(0, len(available_actions)-1)
        action = available_actions[random_int]
    else:
        # Exploitation
        action = max(available_actions, key=lambda action: q_table[state][action])

    return action
    #raise NotImplementedError("Return one action using epsilon-greedy.")

# Update one Q-value.
# Standard shape:
# - current estimate
# - reward
# - best next-state value
# - learning rate
# - gamma
def q_learning_update(
    current_q: float,
    reward: float,
    best_next_q: float,
    learning_rate: float,
    gamma: float,
) -> float:
    """Return the updated Q-value."""
    # Goal:
    # Return the new Q-value for one `(state, action)` pair.
    #
    # Input:
    # - `current_q`: float
    #   The old Q-value for the chosen state and action.
    # - `reward`: float
    #   The reward from the step.
    # - `best_next_q`: float
    #   The largest Q-value in the next state.
    #   Use `0.0` here if the next state is terminal.
    # - `learning_rate`: float
    #   How much the new target changes the old value.
    # - `gamma`: float
    #   How much future reward counts.
    #
    # Output:
    # - `float`
    # - The updated Q-value.
    #
    # Formula:
    # ```text
    # target = reward + gamma * best_next_q
    # new_q = current_q + learning_rate * (target - current_q)
    # ```
    #
    # Meaning:
    # - `target` is the better guess from the new experience.
    # - `target - current_q` is the error in the old guess.
    # - `learning_rate` controls how far to move toward the target.
    #
    # Steps:
    # 1. Compute `target`.
    # 2. Compute `new_q`.
    # 3. Return `new_q`.
    #
    # Checks:
    # - If `learning_rate=1.0`, the result should equal `target`.
    # - If `best_next_q=0.0`, the target is just the immediate reward.

    target = reward + gamma * best_next_q
    difference = target - current_q
    new_q = current_q + learning_rate * difference

    return new_q
    #raise NotImplementedError("Return the tabular Q-learning update.")

# Train the agent for several episodes.
# Return a summary with at least:
# - q_table
# - episode returns
# - goal reach count
def train_q_learning_agent(config: QLearningConfig) -> QLearningSummary:
    """Run tabular Q-learning on the tiny gridworld."""
    # Goal:
    # Train the Q-learning agent over many episodes and return a summary.
    #
    # Input:
    # - `config`: QLearningConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"q_table"`: QTable, learned Q-values
    # - `"episode_returns"`: list[float], total reward for each episode
    # - `"goal_reach_count"`: int, how many episodes ended at the goal
    # - `"greedy_policy"`: dict[State, str], best learned action per state
    # - `"note"`: str, short plain-language Q-value explanation
    #
    # Method:
    # Use the helpers in this file:
    # - `gridworld = build_gridworld(config)`
    # - `q_table = initialize_q_table(gridworld)`
    # - `state = reset_environment(gridworld)`
    # - `action = choose_action(q_table, state, config.epsilon)`
    # - `next_state, reward, done = step_environment(gridworld, state, action)`
    # - `new_q = q_learning_update(current_q, reward, best_next_q, config.learning_rate, config.gamma)`
    # - `greedy_policy = extract_greedy_policy(q_table, gridworld)`
    #
    # Terminal rule:
    # If `done` is `True`, set `best_next_q = 0.0`.
    # Do not bootstrap from the goal state after the episode is done.
    #
    # Steps:
    # 1. Build the gridworld.
    # 2. Create the Q-table.
    # 3. Create an empty list for `episode_returns`.
    # 4. Start `goal_reach_count = 0`.
    # 5. Loop over `config.episodes`.
    # 6. Reset the environment and set this episode's return to `0.0`.
    # 7. Loop up to `config.max_steps_per_episode`.
    # 8. Choose an action with epsilon-greedy.
    # 9. Step the environment.
    # 10. Read `current_q = q_table[state][action]`.
    # 11. Compute `best_next_q`.
    # 12. Update only `q_table[state][action]`.
    # 13. Add reward to the episode return.
    # 14. Move to `next_state`.
    # 15. If `done`, increase `goal_reach_count` and stop this episode.
    # 16. Store the episode return.
    # 17. After training, extract the greedy policy.
    # 18. Return the summary dictionary.
    #
    # Checks:
    # - `len(episode_returns)` should equal `config.episodes`.
    # - `goal_reach_count` should be between `0` and `config.episodes`.
    # - Only the chosen `(state, action)` Q-value should update on each step.

    random.seed(config.random_state)
    gridworld = build_gridworld(config)
    q_table = initialize_q_table(gridworld)    
    episode_returns = []
    goal_reach_count = 0

    for i in range(config.episodes):
        # Start a new run from the start cell.
        state = reset_environment(gridworld)
        episode_reward = 0.0

        for j in range(config.max_steps_per_episode):

            # Use the policy to choose what to do next.
            # Epsilon-greedy means sometimes explore and sometimes use the best known action.
            action = choose_action(q_table, state, config.epsilon)

            # Ask the environment what happened after this move.
            # It gives the next state, the reward, and whether the episode ended.
            next_state, reward, done = step_environment(gridworld, state, action)

            # Read the old guess for this exact state-action pair.
            current_q = q_table[state][action]
            if done:
                # If the episode ended, there is no next-state value to add.
                best_next_q = 0.0
            else:
                # Otherwise, use the best action value from the next state.
                best_next_q = max(q_table[next_state].values())

            # Use the new experience to update only the chosen state-action value.
            updated_q = q_learning_update(current_q, reward, best_next_q, config.learning_rate, config.gamma)
            q_table[state][action] = updated_q

            episode_reward+=reward
            # Move the agent's current state forward.
            state = next_state

            if done:
                # The goal was reached, so this episode is finished.
                goal_reach_count+=1
                break

        episode_returns.append(episode_reward)

    greedy_policy = extract_greedy_policy(q_table, gridworld)

    print("Q-learning training summary:")
    print(f"  episodes: {config.episodes}")
    print(f"  goal reaches: {goal_reach_count}")
    print(f"  final episode return: {episode_returns[-1]}")
    print("  greedy policy:")
    print(greedy_policy)

    return {
        "q_table": q_table,
        "episode_returns": episode_returns,
        "goal_reach_count": goal_reach_count,
        "greedy_policy": greedy_policy,
        "note": "Q-values are learned guesses for how good each action is in each state.",
    }
    #raise NotImplementedError("Train the Q-learning agent and return a summary.")

def extract_greedy_policy(
    q_table: QTable,
    gridworld: Gridworld,
) -> GreedyPolicy:
    """Return the best action in each non-terminal state."""
    # Goal:
    # Return the best learned action for each non-terminal state.
    # Greedy means no random exploration here.
    #
    # Input:
    # - `q_table`: QTable
    # - `gridworld`: Gridworld
    #
    # Output:
    # - `GreedyPolicy`
    # - Keys: all states except the goal state.
    # - Values: the action with the largest Q-value in that state.
    #
    # Method:
    # For one state, use:
    # `best_action = max(q_table[state], key=lambda action: q_table[state][action])`
    #
    # Steps:
    # 1. Create an empty dictionary called `policy`.
    # 2. Loop over every state in `q_table`.
    # 3. Skip `gridworld["goal_state"]`.
    # 4. Choose the action with the largest Q-value.
    # 5. Store it in the policy.
    # 6. Return the policy.
    #
    # Checks:
    # - The goal state should not be in the returned policy.
    # - Every policy action should be one of `gridworld["actions"]`.
    policy = {}
    goal_state = gridworld["goal_state"]

    for state, action_values in q_table.items():
        if state == goal_state:
            continue
        best_action = max(action_values, key=lambda action: action_values[action])
        policy[state] = best_action

    return policy
    #raise NotImplementedError("Return the greedy policy from the learned Q-table.")

def explain_q_values(summary: QLearningSummary) -> str:
    """Write a short note explaining what the Q-values mean."""
    # Goal:
    # Explain learned Q-values in simple words.
    #
    # Input:
    # - `summary`: QLearningSummary
    # - Useful keys:
    #   - `summary["goal_reach_count"]`
    #   - `summary["greedy_policy"]`
    #   - `summary["q_table"]`
    #
    # Output:
    # - `str`
    # - One or two plain sentences.
    #
    # Meaning:
    # A Q-value answers:
    # "If I am in this state and take this action, how much reward do I expect from now on?"
    #
    # Small example:
    # If `q_table[(2, 3)]["down"]` is larger than `q_table[(2, 3)]["left"]`,
    # the learned policy should prefer `"down"` from state `(2, 3)`.
    #
    # Steps:
    # 1. Mention that higher Q-values mean better learned actions.
    # 2. Mention the goal reach count if useful.
    # 3. Keep the note short and plain.
    #
    # Checks:
    # - Do not say Q-values are the same thing as one episode return.
    # - Q-values are learned estimates for state-action pairs.
    return (
        "A Q-value is a learned guess for one state and one action. "
        f"Higher Q-values mean the agent expects better future reward; it reached the goal in {summary['goal_reach_count']} episodes."
    )
    #raise NotImplementedError("Explain learned Q-values in simple words.")

def check_metrics(metrics: QLearningSummary) -> bool:
    """Return True when the Q-learning summary has the expected keys."""
    required = {"q_table", "episode_returns", "goal_reach_count", "greedy_policy", "note"}
    return required.issubset(metrics)

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
    """Run the Q-learning scaffold."""
    config = QLearningConfig()
    random.seed(config.random_state)
    completed_steps = 0
    total_steps = 9

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    _, ok = report_task_progress("describe_q_learning_terms", describe_q_learning_terms, config)
    if not count_or_stop(ok):
        return

    gridworld, ok = report_task_progress("build_gridworld", build_gridworld, config)
    if not count_or_stop(ok):
        return

    state, ok = report_task_progress("reset_environment", reset_environment, gridworld)
    if not count_or_stop(ok):
        return

    q_table, ok = report_task_progress("initialize_q_table", initialize_q_table, gridworld)
    if not count_or_stop(ok):
        return

    action, ok = report_task_progress("choose_action", choose_action, q_table, state, config.epsilon)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("step_environment", step_environment, gridworld, state, action)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("q_learning_update", q_learning_update, 0.0, 1.0, 0.0, config.learning_rate, config.gamma)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("train_q_learning_agent", train_q_learning_agent, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if ok and metrics is not None:
        _, note_ok = report_task_progress("explain_q_values", explain_q_values, metrics)
        if note_ok:
            completed_steps += 1
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "tabular Q-learning summary", float(int(passed)), "q-learning keys present")
if __name__ == "__main__":
    main()
