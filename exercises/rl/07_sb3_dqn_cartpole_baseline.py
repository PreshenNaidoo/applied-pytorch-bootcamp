"""
============================================================
EXERCISE 07: SB3 DQN CARTPOLE BASELINE
============================================================

WHAT YOU ARE BUILDING
You are training a DQN agent on `CartPole-v1` with Stable-Baselines3.

the related exercise built the main DQN pieces by hand.
This exercise uses a common RL library to train the same kind of agent.

The goal is not only to call the library.
The goal is to check whether the trained policy behaves better than random actions.

WHY THIS EXERCISE EXISTS
Before trusting your own RL code, first run a known library version.

That gives you a simple check:

```text
Can a standard DQN implementation learn in this environment?
```

If the library version beats random actions, your custom DQN has a fair score to compare against.
If the library version cannot beat random actions, check the setup, training budget, and evaluation before blaming your custom code.

THE CARTPOLE TASK
CartPole is a small control task.

The agent sees four numbers:

- cart position
- cart velocity
- pole angle
- pole angular velocity

The agent chooses one of two actions:

- push the cart left
- push the cart right

DQN fits this task because the action space is small and discrete.

CartPole in plain words:

- the cart moves left or right
- the pole starts upright
- the agent gets reward for each step the pole stays balanced
- the episode ends when the pole falls too far, the cart moves too far, or the time limit is reached

THE SCORE YOU PRINT
CartPole usually gives reward `+1` for every time step the pole is still balanced.

So:

```text
episode reward = total number of balanced steps in one episode
mean episode reward = average balanced steps over several episodes
```

Example:

```text
mean episode reward = 180
```

means the policy kept the pole balanced for about 180 steps per episode on average.

This is why the final print says:

```text
SB3 DQN mean episode reward: random=..., trained=..., loaded=..., passed=...
```

Higher is better because it means the pole stayed up for longer.

THE TWO ENVIRONMENTS
This exercise uses two environment objects:

```text
train_env = practice environment
eval_env = test environment
```

`train_env` is passed into `model.learn(...)`.
The model is allowed to learn from this environment.

`eval_env` is passed into evaluation functions.
The model is not updated there.
It only acts, collects rewards, and gets scored.

Using a separate eval environment makes the comparison easier to reason about.

THE RL LOOP
The core RL loop is still:

```text
look at the current observation
choose an action
ask the environment what happened
get the next observation and reward
update the agent from that experience
repeat
```

WHAT STABLE-BASELINES3 HIDES
In this file, Stable-Baselines3 runs most of the DQN training loop inside:

```python
model.learn(total_timesteps=config.total_timesteps)
```

That one call:

- chooses actions
- steps the training environment
- stores experience in a replay buffer
- samples mini-batches
- updates the online Q-network
- copies weights into the target network sometimes
- changes epsilon for exploration

You do not write those details here because SB3 already has them.
Your job in this exercise is to set up the run and check the result.

WHAT YOU STILL HANDLE
You still write the code that decides:

- training timesteps
- how many evaluation episodes to average
- where to save the model
- how to load the model back
- how to compare random actions with the trained model

RANDOM POLICY VS TRAINED POLICY

```text
random policy:
choose random actions
no model
no update step
this gives a weak score to beat

trained DQN policy:
use the neural network trained by SB3
choose actions from model.predict(...)
this should score higher after enough training
```

If `passed=False`, the trained model did not beat random actions in that run.
That does not always mean the code is broken.
It can mean the run needs more training steps, more evaluation episodes, or a different seed.

INPUTS AND OUTPUTS
Input:

- Gymnasium environment name
- training timesteps
- evaluation episode count
- random seed

Output:

- random-policy mean episode reward
- trained-model mean episode reward
- loaded-model mean episode reward
- whether the trained model passed a simple evaluation check
- saved model path
- short baseline note

METRICS TO TRACK
Track at least:

- random mean episode reward
- trained mean episode reward
- loaded mean episode reward
- evaluation passed flag

The most important check is not training loss.
The main RL question is:

```text
after training, does the policy behave better in the environment?
```

For this exercise, that means the trained DQN should usually get a higher mean episode reward than random actions.

COMMON MISTAKES
- comparing against no baseline
- thinking mean episode reward is a loss value
- forgetting that CartPole reward roughly means steps balanced
- evaluating on only one episode and trusting a noisy score
- forgetting to use a separate eval environment
- saving a model but never checking that the loaded model still works
- assuming one short failed run means DQN can never learn

STEP-BY-STEP TODO
1. Make train and eval environments.
2. Evaluate random actions so you know the weak baseline.
3. Train an SB3 DQN model.
4. Evaluate the trained model.
5. Save and load the model.
6. Check whether the trained policy passed the simple evaluation test.
7. Explain why baseline-first workflow matters.

DEFINITION OF DONE
You are done when:
- DQN trains on CartPole
- mean episode reward is reported
- the model can be saved and loaded
- random and trained performance are compared
- the final check says whether training worked better than random
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
# 1. Create separate train and eval environments.
# 2. Train one SB3 DQN model.
# 3. Evaluate the trained model over several episodes.
# 4. Evaluate a random policy for comparison.
# 5. Save and load the trained model.
# 6. Connect the full baseline workflow.
#
# Finish the functions in that order so each later step has a working object to use.


@dataclass
class SB3DQNConfig:
    """Settings for the SB3 DQN baseline exercise."""

    env_name: str = "CartPole-v1"
    total_timesteps: int = 10_000
    eval_episodes: int = 10
    random_state: int = 42
    model_path: str = "outputs/sb3_dqn_cartpole.zip"


class SB3DQNSummary(TypedDict):
    """Final summary for the SB3 DQN baseline."""

    env_name: str
    random_mean_return: float
    trained_mean_return: float
    loaded_mean_return: float
    evaluation_passed: bool
    model_path: str
    note: str


def describe_sb3_dqn_baseline(config: SB3DQNConfig) -> dict[str, str | int]:
    """Return the main SB3 DQN baseline settings."""
    return {
        "env_name": config.env_name,
        "algorithm": "DQN",
        "library": "Stable-Baselines3",
        "main_reason": "compare your from-scratch DQN idea with a trusted baseline",
    }


def make_train_and_eval_envs(config: SB3DQNConfig) -> tuple[Any, Any]:
    """Create separate train and eval environments."""
    # Goal:
    # Return two Gymnasium environments, one for training and one for evaluation.
    #
    # Where this fits:
    # `train_env` is where the model practices.
    # SB3 uses it inside `model.learn(...)` and updates the model from that experience.
    #
    # `eval_env` is where the model is tested.
    # The model is not updated in `eval_env`.
    # It only acts and collects rewards so we can score it.
    #
    # Input:
    # - `config.env_name`: environment name such as `"CartPole-v1"`
    #
    # Output:
    # - `(train_env, eval_env)`
    # - both are Gymnasium environment objects
    #
    # Plain English:
    # Make two copies of the same world.
    # One copy is for learning.
    # The other copy is for measuring.
    #
    # Method:
    # Import Gymnasium inside this function:
    # ```python
    # import gymnasium as gym
    # train_env = gym.make(config.env_name)
    # eval_env = gym.make(config.env_name)
    # return train_env, eval_env
    # ```
    #
    # Checks:
    # - Both envs should have the same observation and action spaces.
    # - For CartPole, action space should be `Discrete(2)`.
    # - For CartPole, observation space should have shape `(4,)`.


    import gymnasium as gym
    train_env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    return train_env, eval_env
    #raise NotImplementedError("Create train and eval Gymnasium environments.")


def train_dqn_model(train_env: Any, config: SB3DQNConfig) -> Any:
    """Train one Stable-Baselines3 DQN model."""
    # Goal:
    # Train a trusted DQN baseline.
    #
    # Where this fits:
    # the related exercise showed the DQN pieces by hand.
    # This function lets SB3 handle those pieces.
    # You use its score as a reference point for your own DQN code.
    #
    # Input:
    # - `train_env`: Gymnasium environment used for training
    # - `config.total_timesteps`: number of environment steps SB3 should collect while training
    #
    # Output:
    # - trained SB3 `DQN` model
    #
    # What SB3 does inside `learn(...)`:
    # - chooses actions
    # - calls `train_env.step(action)`
    # - stores experience in a replay buffer
    # - samples mini-batches
    # - updates the Q-network
    # - updates the target network
    # - handles the exploration schedule
    #
    # Method:
    # Import inside this function:
    # ```python
    # from stable_baselines3 import DQN
    # model = DQN("MlpPolicy", train_env, verbose=0, seed=config.random_state)
    # model.learn(total_timesteps=config.total_timesteps)
    # return model
    # ```
    #
    # Checks:
    # - The returned model should have `.predict(...)`.
    # - The returned model should have `.save(...)`.
    # - Keep `verbose=0` so normal runs do not print too much.

    from stable_baselines3 import DQN
    model = DQN(policy="MlpPolicy", env=train_env, verbose=0, seed=config.random_state)
    model.learn(total_timesteps=config.total_timesteps)

    return model
    #raise NotImplementedError("Train and return an SB3 DQN model.")


def evaluate_model(model: Any, eval_env: Any, episodes: int) -> float:
    """Evaluate a trained SB3 model."""
    # Goal:
    # Return mean episode reward over several episodes.
    #
    # Where this fits:
    # Training is practice.
    # Evaluation is the test score.
    # This function tests the trained model without updating it.
    #
    # Input:
    # - `model`: trained SB3 model with `.predict(...)`
    # - `eval_env`: separate Gymnasium environment for scoring
    # - `episodes`: number of full episodes to average over
    #
    # Output:
    # - mean episode reward as a Python `float`
    # - for CartPole, this is roughly the average number of steps the pole stayed up
    #
    # Plain English:
    # Run the trained policy for several full episodes.
    # Add up the reward inside each episode.
    # Average those episode totals.
    #
    # CartPole gives about `+1` per balanced step.
    # So a mean episode reward of `180` means about 180 balanced steps per episode.
    #
    # Method:
    # Use SB3's helper:
    # ```python
    # from stable_baselines3.common.evaluation import evaluate_policy
    # mean_reward, std_reward = evaluate_policy(
    #     model,
    #     eval_env,
    #     n_eval_episodes=episodes,
    #     deterministic=True,
    # )
    # return float(mean_reward)
    # ```
    #
    # Checks:
    # - Use deterministic evaluation for the trained model.
    # - Do not report only one episode.

    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=episodes, deterministic=True)

    return float(mean_reward)
    #raise NotImplementedError("Evaluate the trained DQN model.")


def evaluate_random_policy(eval_env: Any, episodes: int, max_steps: int = 500) -> float:
    """Evaluate a random policy for comparison."""
    # Goal:
    # Return mean reward for random actions.
    #
    # Why:
    # A trained model only means something if you compare it with a simple baseline.
    # Random actions are the simplest baseline.
    # There is no learning and no update call in this function.
    #
    # Input:
    # - `eval_env`: Gymnasium environment
    # - `episodes`: number of random episodes to run
    # - `max_steps`: safety cap per episode
    #
    # Output:
    # - mean random-policy episode reward as a Python `float`
    # - for CartPole, this is roughly the average number of random-action steps before failure
    #
    # Plain-English loop:
    # ```text
    # reset the environment
    # sample a random action
    # ask the environment what happened
    # add the reward
    # stop when terminated, truncated, or max_steps is reached
    # repeat for several episodes
    # average the returns
    # ```
    #
    # This is only a measuring loop.
    # It should not call `model.learn(...)`, `model.predict(...)`, or an optimizer.
    #
    # Steps:
    # 1. Run `episodes` episodes.
    # 2. Reset the environment at the start of each episode.
    # 3. Sample actions with `eval_env.action_space.sample()`.
    # 4. Stop on `terminated or truncated`.
    # 5. Return the mean episode reward.
    #
    # Checks:
    # - Do not call `model.predict(...)` here.
    # - The random policy should use only `eval_env.action_space.sample()`.
    # - Return the average, not the list of returns.

    episode_returns = []

    for _ in range(episodes):
        obs, info = eval_env.reset()
        episode_return = 0.0

        for _ in range(max_steps):
            action = eval_env.action_space.sample()
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_return += float(reward)

            done = terminated or truncated
            if done:
                break

        episode_returns.append(episode_return)

    # raise NotImplementedError("Evaluate a random policy.")
    return sum(episode_returns) / len(episode_returns)


def check_trained_policy_beats_random(random_mean_return: float, trained_mean_return: float) -> bool:
    """Return True when the trained policy beats the random policy."""
    # Goal:
    # Check whether the trained DQN policy worked better than random actions.
    #
    # Where this fits:
    # This is the final "did learning work?" check for this RL exercise.
    # Training loss is not enough in RL.
    # We care whether the learned policy gets better rewards in the environment.
    #
    # Input:
    # - `random_mean_return`: mean return from random actions
    # - `trained_mean_return`: mean return from the trained DQN policy
    #
    # Output:
    # - `bool`
    # - `True` means the trained policy scored higher than random
    #
    # Method:
    # Use:
    # ```python
    # return trained_mean_return > random_mean_return
    # ```
    #
    # Plain meaning:
    # If the trained policy cannot beat random actions, do not trust the run yet.
    # Try more training steps, more eval episodes, or another seed.
    # If it still fails, check the setup.
    #
    # Checks:
    # - This should compare evaluation returns, not training loss.
    # - This is a simple baseline check, not proof that DQN is optimal.

    return trained_mean_return > random_mean_return
    #raise NotImplementedError("Check whether trained DQN beats random actions.")


def save_and_load_model(model: Any, eval_env: Any, config: SB3DQNConfig) -> Any:
    """Save a model and load it back."""
    # Goal:
    # Practice the save/load workflow used in real experiments.
    #
    # Where this fits:
    # Real experiments need checkpoints so results can be reused and inspected later.
    #
    # Input:
    # - `model`: trained SB3 DQN model
    # - `eval_env`: environment to attach to the loaded model
    # - `config.model_path`: file path where the model should be saved
    #
    # Output:
    # - loaded SB3 DQN model
    #
    # Plain English:
    # Save the trained model to disk, then load it back and check it still works.
    #
    # Method:
    # Use:
    # ```python
    # Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    # model.save(config.model_path)
    # from stable_baselines3 import DQN
    # loaded_model = DQN.load(config.model_path, env=eval_env)
    # return loaded_model
    # ```
    #
    # Checks:
    # - The saved path should exist after saving.
    # - The loaded model should still evaluate.
    # - Loading with `env=eval_env` lets the model keep using the same environment API.

    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(config.model_path)
    from stable_baselines3 import DQN
    loaded_model = DQN.load(config.model_path, env=eval_env)
    return loaded_model
    #raise NotImplementedError("Save and load the trained DQN model.")


def run_sb3_dqn_baseline(config: SB3DQNConfig) -> SB3DQNSummary:
    """Run the full SB3 DQN baseline workflow."""
    # Goal:
    # Train, evaluate, save, load, and compare against random.
    #
    # Where this fits:
    # This is the full baseline experiment.
    # It gives you one summary that is easy to compare with custom DQN code.
    #
    # Plain picture:
    # 1. Score random actions first.
    # 2. Train SB3 DQN.
    # 3. Score the trained model.
    # 4. Save and load the model.
    # 5. Score the loaded model to check that loading worked.
    # 6. Check whether trained DQN beat random actions.
    #
    # Output:
    # Return this exact dictionary:
    # - `"env_name"`
    # - `"random_mean_return"`
    # - `"trained_mean_return"`
    # - `"loaded_mean_return"`
    # - `"evaluation_passed"`
    # - `"model_path"`
    # - `"note"`
    #
    # Steps:
    # 1. Create train and eval envs with `make_train_and_eval_envs(config)`.
    # 2. Evaluate the random policy first using `eval_env`.
    # 3. Train the SB3 DQN model.
    # 4. Evaluate the trained model using the same `eval_env`.
    # 5. Save and load the trained model.
    # 6. Evaluate the loaded model using the same `eval_env`.
    # 7. Check whether trained DQN beats random actions.
    # 8. Close both environments.
    # 9. Return the summary dictionary.
    #
    # Checks:
    # - The summary should include every key listed above.
    # - `trained_mean_return` should usually beat `random_mean_return` after enough training.
    # - `loaded_mean_return` should be close to `trained_mean_return`.
    # - `evaluation_passed` should be `True` for a successful baseline run.
    # - For CartPole, these mean rewards are roughly average steps balanced.
    # - Close environments even if an earlier step fails.

    train_env, eval_env = make_train_and_eval_envs(config)

    try:
        random_mean_return = evaluate_random_policy(eval_env, config.eval_episodes)

        model = train_dqn_model(train_env, config)
        trained_mean_return = evaluate_model(model, eval_env, config.eval_episodes)

        loaded_model = save_and_load_model(model, eval_env, config)
        loaded_mean_return = evaluate_model(loaded_model, eval_env, config.eval_episodes)

        evaluation_passed = check_trained_policy_beats_random(random_mean_return, trained_mean_return)
        note = (
            "SB3 DQN gives a reference score for the same environment. "
            "Compare custom DQN against this score and against random actions."
        )

        summary: SB3DQNSummary = {
            "env_name": config.env_name,
            "random_mean_return": random_mean_return,
            "trained_mean_return": trained_mean_return,
            "loaded_mean_return": loaded_mean_return,
            "evaluation_passed": evaluation_passed,
            "model_path": config.model_path,
            "note": note,
        }
        print(
            "SB3 DQN mean episode reward, about average steps balanced: "
            f"random={random_mean_return:.1f}, "
            f"trained={trained_mean_return:.1f}, "
            f"loaded={loaded_mean_return:.1f}, "
            f"passed={evaluation_passed}"
        )
        return summary
    finally:
        train_env.close()
        eval_env.close()

    # raise NotImplementedError("Run the full SB3 DQN baseline workflow.")


def explain_baseline_first_workflow(summary: SB3DQNSummary) -> str:
    """Explain why a trusted baseline is useful."""
    # Goal:
    # Write one or two plain sentences about why you should run SB3 before trusting custom code.
    #
    # Use:
    # - `summary["random_mean_return"]`
    # - `summary["trained_mean_return"]`
    #
    # Plain answer shape:
    # ```text
    # SB3 DQN gives a reference score for this environment.
    # If it beats random actions, your custom DQN has a fair target to compare against.
    # ```
    #
    # Checks:
    # - Mention the random policy comparison.
    # - Do not claim SB3 proves your custom implementation is correct.
    # - Keep it short and natural.
    # raise NotImplementedError("Explain the baseline-first workflow.")
    return (
        "SB3 DQN gives a reference score for this environment. "
        f"Here, random actions scored {summary['random_mean_return']:.1f} mean episode reward, "
        f"while SB3 DQN scored {summary['trained_mean_return']:.1f}. "
        "For CartPole, that reward is roughly the average number of steps the pole stayed balanced."
    )


def check_metrics(metrics: SB3DQNSummary) -> bool:
    """Return True when the SB3 DQN summary has the expected keys."""
    required = {
        "env_name",
        "random_mean_return",
        "trained_mean_return",
        "loaded_mean_return",
        "evaluation_passed",
        "model_path",
        "note",
    }
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
    """Run the SB3 DQN baseline scaffold."""
    config = SB3DQNConfig()
    completed_steps = 0
    total_steps = 7

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    _, ok = report_task_progress("describe_sb3_dqn_baseline", describe_sb3_dqn_baseline, config)
    if not count_or_stop(ok):
        return

    envs, ok = report_task_progress("make_train_and_eval_envs", make_train_and_eval_envs, config)
    if not count_or_stop(ok):
        return
    train_env, eval_env = envs

    model, ok = report_task_progress("train_dqn_model", train_dqn_model, train_env, config)
    if not count_or_stop(ok):
        return

    trained_mean_return, ok = report_task_progress("evaluate_model", evaluate_model, model, eval_env, config.eval_episodes)
    if not count_or_stop(ok):
        return

    random_mean_return, ok = report_task_progress("evaluate_random_policy", evaluate_random_policy, eval_env, config.eval_episodes)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress(
        "check_trained_policy_beats_random",
        check_trained_policy_beats_random,
        random_mean_return,
        trained_mean_return,
    )
    if not count_or_stop(ok):
        return

    summary, ok = report_task_progress("run_sb3_dqn_baseline", run_sb3_dqn_baseline, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and summary is not None:
        passed = check_metrics(summary)
        print_pass_or_try_again(passed, "sb3 dqn summary", float(int(passed)), "summary keys present")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
