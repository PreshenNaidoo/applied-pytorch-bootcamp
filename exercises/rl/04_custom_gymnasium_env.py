"""
============================================================
EXERCISE 04: CUSTOM GYMNASIUM ENV
============================================================

WHAT YOU ARE BUILDING
You are turning a tiny hand-written LineWorld environment into a Gymnasium-style environment.

This is the bridge from bootcamp toy environments to real RL tooling.

WHY THIS EXERCISE EXISTS
the related exercise taught a custom gridworld with plain Python functions.
Real RL libraries usually expect an environment object with:

```python
observation, info = env.reset()
next_observation, reward, terminated, truncated, info = env.step(action)
```

This exercise teaches how to wrap a small environment in that shape.

THE FULL PICTURE
The environment is a line of positions:

```text
0 -- 1 -- 2 -- 3 -- 4
```

The agent starts at position `0`.
The goal is position `4`.

Actions:
- `0`: move left
- `1`: move right

The environment should:
- keep the agent inside the line
- give a small step penalty
- give a positive reward at the goal
- end the episode at the goal
- end the episode if the max step limit is reached

Plain-English RL loop in this exercise:

```text
start a new episode at position 0
look at the current position
choose action 0 for left or action 1 for right
ask the environment to move the agent
get the next position and reward
stop if the goal is reached or the step limit is hit
repeat while the episode is still running
```

The main object is `LineWorldEnv`.
It is the environment object that a real RL library will call.

The library does not know your custom rules.
It only knows how to call:

```python
env.reset()
env.step(action)
```

So your job is to hide the custom LineWorld rules behind that standard API.

Gymnasium code in plain English:

- `env.reset()` means start a new episode and return the first position.
- `env.step(action)` means take one left/right move and tell us what happened.
- `terminated` means the agent reached the goal.
- `truncated` means the step limit stopped the episode.
- `observation = np.array([position], dtype=np.float32)` means the agent sees one number: its current position.

Tiny concrete example:

```text
current position = 0
action = 1
next position = 1
reward = -0.01
terminated = False
truncated = False
```

If the current position is `3` and action `1` moves to the goal:

```text
next position = 4
reward = 1.0
terminated = True
```

INPUTS AND OUTPUTS
Input:
- a position on the line
- an action, `0` for left or `1` for right

Output from `step`:
- next position as an observation
- reward
- `terminated`
- `truncated`
- `info`

METRICS TO TRACK
Track at least:
- whether `check_env` passes
- random episode return
- observation space
- action space

STEP-BY-STEP TODO
1. Define the Gymnasium observation and action spaces.
2. Implement `reset`.
3. Implement `step`.
4. Run the Gymnasium environment checker.
5. Run one random episode.
6. Optionally train a tiny SB3 agent on the custom environment.

DEFINITION OF DONE
You are done when:
- the custom environment passes `check_env`
- `reset` and `step` return the correct Gymnasium shapes
- a random episode can run
- you can explain why real RL libraries need a standard environment API
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, TypedDict

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.training_checks import print_pass_or_try_again

# Learning order for this exercise:
# 1. Build the custom LineWorld environment with `make_lineworld_env`.
# 2. Run `run_env_checker` before trusting the environment.
# 3. Run one random episode to check the loop by hand.
# 4. Connect the pieces in `run_custom_env_workflow`.
# 5. Treat SB3 training as optional; API correctness comes first.


@dataclass
class CustomGymEnvConfig:
    """Settings for the custom Gymnasium environment."""

    size: int = 5
    goal_position: int = 4
    max_steps: int = 20
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    random_state: int = 42
    sb3_train_steps: int = 1_000


class CustomGymEnvSummary(TypedDict):
    """Final summary for the custom environment exercise."""

    observation_space: str
    action_space: str
    checker_passed: bool
    random_episode_return: float
    note: str


def describe_custom_env_goal(config: CustomGymEnvConfig) -> dict[str, int | float | str]:
    """Return the main custom environment settings."""
    return {
        "environment": "LineWorld",
        "size": config.size,
        "goal_position": config.goal_position,
        "main_api": "reset(...) and step(action)",
    }


def make_lineworld_env(config: CustomGymEnvConfig) -> Any:
    """Return one custom Gymnasium-style LineWorld environment."""
    # Goal:
    # Create and return a custom environment object.
    #
    # Where this fits:
    # This is how a real simulator, business problem, or game becomes usable by RL libraries.
    #
    # Method:
    # Import Gymnasium inside this function:
    # ```python
    # import gymnasium as gym
    # from gymnasium import spaces
    # import numpy as np
    # ```
    #
    # Define an inner class:
    # ```python
    # class LineWorldEnv(gym.Env):
    #     observation_space = spaces.Box(...)
    #     action_space = spaces.Discrete(2)
    #     def reset(...): ...
    #     def step(...): ...
    # ```
    #
    # Observation:
    # - use one integer position, stored as a NumPy array shaped `(1,)`
    # - example: `np.array([0], dtype=np.float32)`
    # - use `spaces.Box(low=0, high=config.size - 1, shape=(1,), dtype=np.float32)`
    #
    # Plain meaning:
    # The observation is not the whole line.
    # It is one number that tells the agent where it is now.
    #
    # Action:
    # - `0` means left
    # - `1` means right
    # - use `spaces.Discrete(2)` because there are two allowed actions
    #
    # Reset return:
    # ```python
    # return observation, info
    # ```
    #
    # Step logic:
    # - if action is `0`, move one position left
    # - if action is `1`, move one position right
    # - clamp the position so it stays between `0` and `config.size - 1`
    # - increase `self.steps` by `1`
    # - set `terminated` when `self.position == config.goal_position`
    # - set `truncated` when `self.steps >= config.max_steps` and the goal was not reached
    #
    # Step return:
    # ```python
    # return observation, reward, terminated, truncated, info
    # ```
    #
    # Checks:
    # - `env.reset(seed=...)` should return `(observation, info)`.
    # - `env.step(action)` should return five values.
    # - The position should never leave `0` to `config.size - 1`.

    raise NotImplementedError("Build and return a custom Gymnasium LineWorld environment.")




def run_env_checker(env: Any) -> bool:
    """Run Stable-Baselines3's environment checker."""
    # Goal:
    # Check that the custom environment follows the Gymnasium API.
    #
    # Where this fits:
    # The checker catches API mistakes before training starts.
    # This is better than debugging a training failure caused by a bad `step` return.
    #
    # Method:
    # Import inside this function:
    # ```python
    # from stable_baselines3.common.env_checker import check_env
    # check_env(env, warn=True)
    # ```
    #
    # Output:
    # - `True` if the checker runs without raising an error
    #
    # Checks:
    # - If this fails, fix `reset`, `step`, `observation_space`, or `action_space`.
    raise NotImplementedError("Run check_env on the custom environment.")



def run_random_custom_episode(env: Any, seed: int, max_steps: int) -> float:
    """Run one random episode in the custom environment."""
    # Goal:
    # Confirm the environment can run from reset to done.
    #
    # Where this fits:
    # A random episode is a simple smoke test.
    # If this fails, training with SB3 will also fail.
    #
    # Input:
    # - `env`: the custom LineWorld environment
    # - `seed`: reset seed for repeatable starts
    # - `max_steps`: safety cap for this rollout
    #
    # Output:
    # - total reward collected during the episode
    #
    # Plain-English loop:
    # ```text
    # reset the environment
    # choose a random left/right action
    # ask the environment what happened
    # add the reward
    # move to the next observation
    # stop when terminated, truncated, or max_steps is reached
    # ```
    #
    # Steps:
    # 1. Reset the env with `env.reset(seed=seed)`.
    # 2. Start `total_reward = 0.0`.
    # 3. Start `done = False`.
    # 4. Loop while not done and while the local step count is below `max_steps`.
    # 5. Sample actions with `env.action_space.sample()`.
    # 6. Step with `env.step(action)`.
    # 7. Add rewards into `total_reward`.
    # 8. Set `done = terminated or truncated`.
    # 9. Return `total_reward`.
    #
    # Checks:
    # - The return should be a float.
    # - The loop should not call learning code.
    # - The loop should stop even if the environment has a bug.

    raise NotImplementedError("Run one random episode in the custom environment.")





def train_tiny_sb3_agent(env: Any, config: CustomGymEnvConfig) -> dict[str, str | int]:
    """Optionally train a tiny SB3 agent on the custom environment."""
    # Goal:
    # Show that a real RL library can use your custom environment.
    #
    # Method:
    # Import inside this function:
    # ```python
    # from stable_baselines3 import DQN
    # model = DQN("MlpPolicy", env, verbose=0, seed=config.random_state)
    # model.learn(total_timesteps=config.sb3_train_steps)
    # ```
    #
    # Output:
    # - dictionary with a short status summary
    #
    # Note:
    # Keep this tiny. The main goal is API compatibility, not high reward.

    raise NotImplementedError("Train a tiny SB3 agent on the custom environment.")



def run_custom_env_workflow(config: CustomGymEnvConfig) -> CustomGymEnvSummary:
    """Run the custom environment workflow."""
    # Goal:
    # Build the environment, check it, run one episode, and return a summary.
    #
    # Where this fits:
    # This connects the environment, checker, and random rollout into one workflow.
    #
    # Output:
    # Return this exact dictionary:
    # - `"observation_space"`
    # - `"action_space"`
    # - `"checker_passed"`
    # - `"random_episode_return"`
    # - `"note"`
    #
    # Steps:
    # 1. Build the environment with `make_lineworld_env(config)`.
    # 2. Save `str(env.observation_space)` and `str(env.action_space)`.
    # 3. Run `checker_passed = run_env_checker(env)`.
    # 4. Run one random episode with `run_random_custom_episode(...)`.
    # 5. Close the environment with `env.close()`.
    # 6. Return the summary dictionary.
    #
    # Checks:
    # - The summary should contain every key listed above.
    # - `checker_passed` should be `True` if `check_env` did not raise an error.
    # - `random_episode_return` should be a float.

    raise NotImplementedError("Run the custom Gymnasium environment workflow.")



def check_metrics(metrics: CustomGymEnvSummary) -> bool:
    """Return True when the custom env summary has the expected keys."""
    required = {"observation_space", "action_space", "checker_passed", "random_episode_return", "note"}
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
    """Run the custom Gymnasium environment scaffold."""
    config = CustomGymEnvConfig()
    completed_steps = 0
    total_steps = 5

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    _, ok = report_task_progress("describe_custom_env_goal", describe_custom_env_goal, config)
    if not count_or_stop(ok):
        return

    env, ok = report_task_progress("make_lineworld_env", make_lineworld_env, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("run_env_checker", run_env_checker, env)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("run_random_custom_episode", run_random_custom_episode, env, config.random_state, config.max_steps)
    if not count_or_stop(ok):
        return

    summary, ok = report_task_progress("run_custom_env_workflow", run_custom_env_workflow, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and summary is not None:
        passed = check_metrics(summary)
        print_pass_or_try_again(passed, "custom gymnasium env summary", float(int(passed)), "summary keys present")

    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
