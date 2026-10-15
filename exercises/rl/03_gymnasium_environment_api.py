"""
============================================================
EXERCISE 03: GYMNASIUM ENVIRONMENT API
============================================================

WHAT YOU ARE BUILDING
You are learning the real environment API used by many RL projects.

The library is called Gymnasium.
It gives a common shape for environments:

```python
observation, info = env.reset()
next_observation, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

WHY THIS EXERCISE EXISTS
Exercises 72 and 73 taught the idea of an environment in plain Python.

This file shows the same idea using a real API.
This matters because many RL tools expect environments to look like Gymnasium
environments.

THE FULL PICTURE
The environment is the world.
The agent chooses an action.
The environment returns feedback.

In this exercise, the policy is random.
That means the agent is not learning yet.
The point is to understand the environment API before training a model.

In Gymnasium, the feedback is:

- `next_observation`: what the agent sees after the action
- `reward`: the score from the action
- `terminated`: true when the task ended naturally, such as failure or success
- `truncated`: true when the run stopped because of a limit, such as max time
- `info`: extra debugging data

The main loop is:

```python
observation, info = env.reset(seed=config.random_state)

while not done:
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    observation = next_observation
```

Read that as:
- reset the environment to start a new episode
- choose one action
- try the action in the environment
- receive the next observation and reward
- stop if the task ended or hit a time limit

DATASET
There is no dataset file.
The data comes from stepping a real Gymnasium environment.

Default environment:
- `CartPole-v1`
- observation shape: `[4]`
- actions: `0` or `1`

INPUTS AND OUTPUTS
Input:
- Gymnasium environment name
- random seed
- number of episodes

Output:
- observation space
- action space
- episode returns
- mean return
- short note about `terminated` and `truncated`

METRICS TO TRACK
Track at least:
- one return per episode
- mean return across random episodes
- observation and action spaces

STEP-BY-STEP TODO
1. Create one Gymnasium environment.
2. Inspect the observation space and action space.
3. Run one random episode.
4. Run several random episodes.
5. Explain `terminated` vs `truncated`.

DEFINITION OF DONE
You are done when:
- `env.reset(...)` is used correctly
- `env.step(...)` is used correctly
- random episodes produce returns
- you can explain the five values returned by `env.step(...)`
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


@dataclass
class GymnasiumApiConfig:
    """Settings for the Gymnasium API exercise."""

    env_name: str = "CartPole-v1"
    random_state: int = 42
    episodes: int = 5
    max_steps_per_episode: int = 500


class GymnasiumApiSummary(TypedDict):
    """Final summary from random Gymnasium rollouts."""

    env_name: str
    observation_space: str
    action_space: str
    episode_returns: list[float]
    mean_return: float
    note: str


def describe_gymnasium_api(config: GymnasiumApiConfig) -> dict[str, str | int]:
    """Return the main Gymnasium API idea."""
    return {
        "env_name": config.env_name,
        "reset_returns": "observation, info",
        "step_returns": "next_observation, reward, terminated, truncated, info",
        "main_check": "done should be terminated or truncated",
    }


def make_gymnasium_env(config: GymnasiumApiConfig) -> Any:
    """Create one Gymnasium environment."""
    # Goal:
    # Return a real Gymnasium environment.
    #
    # Where this fits:
    # This is the first step in using real RL tooling.
    # Later functions call `env.reset(...)` and `env.step(...)` on this object.
    #
    # Input:
    # - `config.env_name`: string such as `"CartPole-v1"`.
    #
    # Output:
    # - a Gymnasium environment object
    #
    # Method:
    # Import Gymnasium inside this function:
    # ```python
    # import gymnasium as gym
    # env = gym.make(config.env_name)
    # return env
    # ```
    #
    # If this import fails, install Gymnasium in the project environment first.
    #
    # Checks:
    # - The returned object should have `.reset(...)`.
    # - The returned object should have `.step(...)`.
    # - The returned object should have `.observation_space` and `.action_space`.

    import gymnasium as gym

    env = gym.make(id=config.env_name)

    return env
    #raise NotImplementedError("Create and return a Gymnasium environment.")


def inspect_spaces(env: Any) -> dict[str, str]:
    """Return simple strings for observation and action spaces."""
    # Goal:
    # Return readable descriptions of what observations and actions look like.
    #
    # Where this fits:
    # Before training an agent, inspect what the environment expects.
    # The observation space tells you what the model will receive.
    # The action space tells you what the policy is allowed to output.
    #
    # Input:
    # - `env`: Gymnasium environment from `make_gymnasium_env`.
    #
    # Output:
    # Return this exact dictionary:
    # - `"observation_space"`: string form of `env.observation_space`
    # - `"action_space"`: string form of `env.action_space`
    #
    # Method:
    # Use:
    # ```python
    # str(env.observation_space)
    # str(env.action_space)
    # ```
    #
    # Checks:
    # - For CartPole, observation space should mention shape `(4,)`.
    # - For CartPole, action space should be `Discrete(2)`.

    return {
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space)
    }
    #raise NotImplementedError("Return readable observation and action spaces.")


def run_random_episode(env: Any, seed: int, max_steps: int) -> float:
    """Run one random-policy episode and return total reward."""
    # Goal:
    # Let a random policy interact with the environment for one episode.
    #
    # Where this fits:
    # This is the simplest possible policy.
    # It checks that the environment loop works before adding learning.
    #
    # Input:
    # - `env`: Gymnasium environment
    # - `seed`: reset seed for repeatability
    # - `max_steps`: safety cap so the loop cannot run forever
    #
    # Output:
    # - total reward as a float
    #
    # Method:
    # Use the real Gymnasium calls:
    # ```python
    # observation, info = env.reset(seed=seed)
    # action = env.action_space.sample()
    # next_observation, reward, terminated, truncated, info = env.step(action)
    # done = terminated or truncated
    # ```
    #
    # Steps:
    # 1. Reset the environment with the seed.
    # 2. Start `total_reward = 0.0`.
    # 3. Loop until `done` is true or `max_steps` is reached.
    # 4. Sample a random action from `env.action_space`.
    # 5. Step the environment.
    # 6. Add reward to `total_reward`.
    # 7. Set `done = terminated or truncated`.
    # 8. Return `total_reward`.
    #
    # Checks:
    # - The return should be a float.
    # - CartPole random returns are usually positive.


    observation, inform = env.reset(seed=seed)

    total_reward = 0.0
    done = False
    step=0

    while not done:
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)

        total_reward+=reward
        step+=1

        done = terminated or truncated or step==max_steps
    
    return total_reward
    #raise NotImplementedError("Run one random Gymnasium episode.")


def run_random_episodes(config: GymnasiumApiConfig) -> GymnasiumApiSummary:
    """Run several random episodes and summarize the result."""
    # Goal:
    # Create one environment, run several random episodes, and return a summary.
    #
    # Input:
    # - `config`: GymnasiumApiConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"env_name"`
    # - `"observation_space"`
    # - `"action_space"`
    # - `"episode_returns"`
    # - `"mean_return"`
    # - `"note"`
    #
    # Steps:
    # 1. Create the environment with `make_gymnasium_env(config)`.
    # 2. Inspect spaces with `inspect_spaces(env)`.
    # 3. Run `config.episodes` random episodes.
    # 4. Use a different seed for each episode, such as `config.random_state + i`.
    # 5. Close the environment with `env.close()`.
    # 6. Return the summary.
    #
    # Checks:
    # - `len(episode_returns)` should equal `config.episodes`.
    # - `mean_return` should be the average of `episode_returns`.

    env = make_gymnasium_env(config)
    spaces_dict = inspect_spaces(env)
    print(spaces_dict)

    episode_rewards = []
    for i in range(config.episodes):
        reward = run_random_episode(env, seed=config.random_state + i, max_steps=config.max_steps_per_episode)
        episode_rewards.append(reward)

    env.close()

    mean_return = sum(episode_rewards) / len(episode_rewards)

    return {
        "env_name": config.env_name,
        "observation_space": spaces_dict["observation_space"],
        "action_space": spaces_dict["action_space"],
        "episode_returns": episode_rewards,
        "mean_return": mean_return,
        "note": "terminated means the task ended; truncated means an outside limit stopped the episode.",
    }
    #raise NotImplementedError("Run several random episodes and return a summary.")


def explain_terminated_vs_truncated() -> str:
    """Explain Gymnasium's two done flags."""
    # Goal:
    # Explain the difference between `terminated` and `truncated`.
    #
    # Output:
    # - one or two plain sentences
    #
    # Meaning:
    # - `terminated` means the task itself ended.
    # - `truncated` means the run was cut off by an outside limit, often a time limit.
    # - `done = terminated or truncated` means the episode should stop either way.
    raise NotImplementedError("Explain terminated and truncated in plain words.")


def check_metrics(metrics: GymnasiumApiSummary) -> bool:
    """Return True when the Gymnasium summary has the expected keys."""
    required = {"env_name", "observation_space", "action_space", "episode_returns", "mean_return", "note"}
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
    """Run the Gymnasium API scaffold."""
    config = GymnasiumApiConfig()
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

    _, ok = report_task_progress("describe_gymnasium_api", describe_gymnasium_api, config)
    if not count_or_stop(ok):
        return

    env, ok = report_task_progress("make_gymnasium_env", make_gymnasium_env, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("inspect_spaces", inspect_spaces, env)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("run_random_episode", run_random_episode, env, config.random_state, config.max_steps_per_episode)
    if not count_or_stop(ok):
        return

    summary, ok = report_task_progress("run_random_episodes", run_random_episodes, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and summary is not None:
        passed = check_metrics(summary)
        print_pass_or_try_again(passed, "gymnasium api summary", float(int(passed)), "summary keys present")

    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
