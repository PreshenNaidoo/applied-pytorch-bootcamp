"""
============================================================
EXERCISE 12: PPO STABLE BASELINES3 BASELINE
============================================================

WHAT YOU ARE BUILDING
You are training PPO with Stable-Baselines3 on a real Gymnasium environment.

the related exercise built PPO pieces by hand.
This exercise shows the practical baseline version.

WHY THIS EXERCISE EXISTS
PPO is common in RL practice.
Real projects usually start by checking a known PPO implementation before changing the algorithm.

This file teaches:
- SB3 PPO training
- a separate evaluation environment
- `EvalCallback`
- saving the best model
- deterministic evaluation

THE FULL PICTURE
Your from-scratch PPO exercise taught:
- rollout collection
- old log-probabilities
- advantages
- clipped policy loss
- value loss
- entropy bonus

SB3 hides those details inside the library.
Your job here is to use the library correctly and evaluate honestly.

The agent starts untrained.
During training, PPO collects rollouts, updates the policy, and checks performance with the eval environment.

INPUTS AND OUTPUTS
Input:
- Gymnasium environment name
- total training timesteps
- evaluation frequency
- evaluation episode count

Output:
- mean evaluation return
- path to the best saved model
- total training timesteps
- short note about what SB3 handled

METRICS TO TRACK
Track at least:
- mean evaluation return
- best model path
- total timesteps

STEP-BY-STEP TODO
1. Create train and eval environments.
2. Create an `EvalCallback`.
3. Train SB3 PPO.
4. Load the best model if one was saved.
5. Evaluate the final or best model.
6. Explain what SB3 handles for you.

DEFINITION OF DONE
You are done when:
- PPO trains on a Gymnasium environment
- evaluation return is reported
- best-model saving is used
- you can connect SB3 PPO to the from-scratch PPO pieces
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
class SB3PPOConfig:
    """Settings for the SB3 PPO baseline exercise."""

    env_name: str = "CartPole-v1"
    total_timesteps: int = 20_000
    eval_freq: int = 2_000
    eval_episodes: int = 10
    random_state: int = 42
    output_dir: str = "outputs/sb3_ppo_cartpole"


class SB3PPOSummary(TypedDict):
    """Final summary for the SB3 PPO baseline."""

    env_name: str
    mean_eval_return: float
    best_model_path: str
    total_timesteps: int
    note: str


def describe_sb3_ppo_baseline(config: SB3PPOConfig) -> dict[str, str | int]:
    """Return the main SB3 PPO settings."""
    return {
        "env_name": config.env_name,
        "algorithm": "PPO",
        "library": "Stable-Baselines3",
        "main_check": "evaluate with a separate environment",
    }


def make_ppo_envs(config: SB3PPOConfig) -> tuple[Any, Any]:
    """Create train and eval environments for PPO."""
    # Goal:
    # Return separate Gymnasium environments for training and evaluation.
    #
    # Where this fits:
    # The train environment supplies experience.
    # The eval environment gives a cleaner check of the current policy.
    #
    # Method:
    # ```python
    # import gymnasium as gym
    # train_env = gym.make(config.env_name)
    # eval_env = gym.make(config.env_name)
    # return train_env, eval_env
    # ```
    raise NotImplementedError("Create PPO train and eval environments.")


def make_eval_callback(eval_env: Any, config: SB3PPOConfig) -> Any:
    """Create an SB3 EvalCallback."""
    # Goal:
    # Create a callback that periodically evaluates and saves the best model.
    #
    # Where this fits:
    # This is the practical version of checking learning while training.
    # It saves the best checkpoint automatically.
    #
    # Method:
    # ```python
    # from stable_baselines3.common.callbacks import EvalCallback
    # output_dir = Path(config.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    # callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=str(output_dir),
    #     log_path=str(output_dir),
    #     eval_freq=config.eval_freq,
    #     n_eval_episodes=config.eval_episodes,
    #     deterministic=True,
    # )
    # return callback
    # ```
    #
    # Checks:
    # - Use deterministic evaluation.
    # - Save the best model under `config.output_dir`.
    raise NotImplementedError("Create an EvalCallback for PPO.")


def train_ppo_model(train_env: Any, callback: Any, config: SB3PPOConfig) -> Any:
    """Train one SB3 PPO model."""
    # Goal:
    # Train PPO using Stable-Baselines3.
    #
    # Where this fits:
    # the related exercise showed the PPO pieces.
    # This function uses the production-style library implementation.
    #
    # Method:
    # ```python
    # from stable_baselines3 import PPO
    # model = PPO("MlpPolicy", train_env, verbose=0, seed=config.random_state)
    # model.learn(total_timesteps=config.total_timesteps, callback=callback)
    # return model
    # ```
    raise NotImplementedError("Train an SB3 PPO model.")


def load_best_or_use_final(model: Any, eval_env: Any, config: SB3PPOConfig) -> tuple[Any, str]:
    """Load the saved best PPO model if it exists."""
    # Goal:
    # Use the best model saved by `EvalCallback` when available.
    #
    # Method:
    # The best model path is usually:
    # ```python
    # best_model_path = Path(config.output_dir) / "best_model.zip"
    # ```
    #
    # If it exists:
    # ```python
    # from stable_baselines3 import PPO
    # return PPO.load(best_model_path, env=eval_env), str(best_model_path)
    # ```
    #
    # Otherwise:
    # ```python
    # return model, "final model in memory"
    # ```
    raise NotImplementedError("Load the saved best PPO model if available.")


def evaluate_ppo_model(model: Any, eval_env: Any, episodes: int) -> float:
    """Evaluate a PPO model."""
    # Goal:
    # Return mean evaluation reward.
    #
    # Method:
    # Use:
    # ```python
    # from stable_baselines3.common.evaluation import evaluate_policy
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=episodes, deterministic=True)
    # return float(mean_reward)
    # ```
    raise NotImplementedError("Evaluate the PPO model.")


def run_sb3_ppo_baseline(config: SB3PPOConfig) -> SB3PPOSummary:
    """Run the full SB3 PPO baseline workflow."""
    # Goal:
    # Train PPO with evaluation callback and return a summary.
    #
    # Output:
    # Return this exact dictionary:
    # - `"env_name"`
    # - `"mean_eval_return"`
    # - `"best_model_path"`
    # - `"total_timesteps"`
    # - `"note"`
    raise NotImplementedError("Run the full SB3 PPO baseline workflow.")


def explain_what_sb3_handles() -> str:
    """Explain what the library handles for PPO."""
    # Goal:
    # Name the PPO pieces SB3 handles internally.
    #
    # Mention:
    # - rollout collection
    # - advantage estimates
    # - clipped policy loss
    # - value loss
    # - optimizer steps
    raise NotImplementedError("Explain what SB3 handles inside PPO.")


def check_metrics(metrics: SB3PPOSummary) -> bool:
    """Return True when the SB3 PPO summary has the expected keys."""
    required = {"env_name", "mean_eval_return", "best_model_path", "total_timesteps", "note"}
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
    """Run the SB3 PPO baseline scaffold."""
    config = SB3PPOConfig()
    completed_steps = 0
    total_steps = 6

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    _, ok = report_task_progress("describe_sb3_ppo_baseline", describe_sb3_ppo_baseline, config)
    if not count_or_stop(ok):
        return

    envs, ok = report_task_progress("make_ppo_envs", make_ppo_envs, config)
    if not count_or_stop(ok):
        return
    train_env, eval_env = envs

    callback, ok = report_task_progress("make_eval_callback", make_eval_callback, eval_env, config)
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("train_ppo_model", train_ppo_model, train_env, callback, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("evaluate_ppo_model", evaluate_ppo_model, model, eval_env, config.eval_episodes)
    if not count_or_stop(ok):
        return

    summary, ok = report_task_progress("run_sb3_ppo_baseline", run_sb3_ppo_baseline, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and summary is not None:
        passed = check_metrics(summary)
        print_pass_or_try_again(passed, "sb3 ppo summary", float(int(passed)), "summary keys present")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
