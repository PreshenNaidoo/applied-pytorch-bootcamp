"""
============================================================
EXERCISE 14: REAL RL EXPERIMENT RELIABILITY
============================================================

WHAT YOU ARE BUILDING
You are running a small real-tooling RL experiment across several seeds.

This exercise uses Gymnasium and Stable-Baselines3.
It is the practical version of the related exercise.

WHY THIS EXERCISE EXISTS
RL results can vary a lot.
One lucky seed can make an algorithm look better than it really is.
One unlucky seed can make a working setup look broken.

Industry and research workflows need:
- multiple seeds
- separate evaluation
- saved checkpoints
- clear result summaries
- failure notes

THE FULL PICTURE
For each seed:

1. Create a fresh train environment.
2. Create a fresh eval environment.
3. Train the same algorithm.
4. Evaluate for several episodes.
5. Save the model.
6. Store the result.

Then compare all seeds.

The model is not the only thing being tested.
You are also testing whether the result is repeatable.

If seed returns are close together, the result is easier to trust.
If seed returns are very spread out, one run is not enough evidence.

INPUTS AND OUTPUTS
Input:
- environment name
- algorithm name
- seed list
- training timesteps
- evaluation episode count

Output:
- one evaluation return per seed
- mean return
- return standard deviation
- saved report path
- short reliability note

METRICS TO TRACK
Track at least:
- seed returns
- mean return
- return standard deviation
- saved model paths

STEP-BY-STEP TODO
1. Build the list of seeds.
2. Train one SB3 run for one seed.
3. Evaluate one trained model.
4. Repeat for all seeds.
5. Compute mean and spread.
6. Save a plain-text report.
7. Explain why one seed is not enough.

DEFINITION OF DONE
You are done when:
- at least three seeds are run
- evaluation returns are summarized
- model paths are saved
- a report is written
- you can explain why seed spread matters
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from typing import Any, TypedDict

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.training_checks import print_pass_or_try_again


@dataclass
class RealRLReliabilityConfig:
    """Settings for the real RL reliability exercise."""

    env_name: str = "CartPole-v1"
    algorithm: str = "PPO"
    seeds: tuple[int, ...] = (0, 1, 2)
    total_timesteps: int = 10_000
    eval_episodes: int = 10
    output_dir: str = "outputs/real_rl_reliability"


class SeedRunResult(TypedDict):
    """Result from one seed."""

    seed: int
    eval_mean_return: float
    model_path: str
    note: str


class RealRLReliabilitySummary(TypedDict):
    """Final reliability summary across seeds."""

    env_name: str
    algorithm: str
    seed_returns: list[float]
    mean_return: float
    return_std: float
    report_path: str
    note: str


def describe_reliability_goal(config: RealRLReliabilityConfig) -> dict[str, str | int]:
    """Return the main real-RL reliability settings."""
    return {
        "env_name": config.env_name,
        "algorithm": config.algorithm,
        "num_seeds": len(config.seeds),
        "main_check": "do not trust one seed",
    }


def build_seed_list(config: RealRLReliabilityConfig) -> list[int]:
    """Return the seeds to run."""
    # Goal:
    # Return the seed list as a normal Python list.
    #
    # Where this fits:
    # The seed list defines how many independent runs you will compare.
    # One seed is not enough for an RL reliability check.
    #
    # Checks:
    # - Use at least three seeds for this exercise.
    # - Keep the list small enough that the exercise runs in reasonable time.
    raise NotImplementedError("Return the list of seeds to run.")


def make_algorithm(env: Any, seed: int, config: RealRLReliabilityConfig) -> Any:
    """Create the SB3 algorithm object."""
    # Goal:
    # Return one SB3 model for the chosen algorithm.
    #
    # Where this fits:
    # This keeps algorithm construction in one place.
    # The rest of the workflow can then loop over seeds.
    #
    # Method:
    # Import inside this function:
    # ```python
    # from stable_baselines3 import PPO, DQN
    # if config.algorithm == "PPO":
    #     return PPO("MlpPolicy", env, verbose=0, seed=seed)
    # if config.algorithm == "DQN":
    #     return DQN("MlpPolicy", env, verbose=0, seed=seed)
    # raise ValueError(...)
    # ```
    #
    # Checks:
    # - The model should have `.learn(...)`, `.predict(...)`, and `.save(...)`.
    raise NotImplementedError("Create one SB3 algorithm object.")


def train_one_seed(seed: int, config: RealRLReliabilityConfig) -> SeedRunResult:
    """Train and evaluate one seed."""
    # Goal:
    # Run one complete train-evaluate-save workflow.
    #
    # Where this fits:
    # This is one independent trial.
    # `run_real_rl_reliability(...)` will call this once per seed.
    #
    # Method:
    # Use:
    # ```python
    # import gymnasium as gym
    # train_env = gym.make(config.env_name)
    # eval_env = gym.make(config.env_name)
    # model = make_algorithm(train_env, seed, config)
    # model.learn(total_timesteps=config.total_timesteps)
    # ```
    #
    # Evaluate with:
    # ```python
    # from stable_baselines3.common.evaluation import evaluate_policy
    # mean_return, std_return = evaluate_policy(model, eval_env, n_eval_episodes=config.eval_episodes, deterministic=True)
    # ```
    #
    # Save with a seed-specific path:
    # ```python
    # outputs/real_rl_reliability/seed_0_model.zip
    # ```
    #
    # Output:
    # Return this exact dictionary:
    # - `"seed"`
    # - `"eval_mean_return"`
    # - `"model_path"`
    # - `"note"`
    raise NotImplementedError("Train and evaluate one seed.")


def summarize_seed_results(results: list[SeedRunResult], config: RealRLReliabilityConfig) -> RealRLReliabilitySummary:
    """Summarize several seed results."""
    # Goal:
    # Compute mean and standard deviation across seeds.
    #
    # Input:
    # - `results`: list of seed result dictionaries
    #
    # Output:
    # Return a summary dictionary with:
    # - `"seed_returns"`
    # - `"mean_return"`
    # - `"return_std"`
    #
    # Method:
    # Use:
    # ```python
    # returns = [result["eval_mean_return"] for result in results]
    # mean_return = statistics.mean(returns)
    # return_std = statistics.pstdev(returns)
    # ```
    raise NotImplementedError("Summarize the seed results.")


def write_reliability_report(summary: RealRLReliabilitySummary, results: list[SeedRunResult], config: RealRLReliabilityConfig) -> str:
    """Write a plain-text reliability report."""
    # Goal:
    # Save a short report to `config.output_dir`.
    #
    # Include:
    # - environment
    # - algorithm
    # - one line per seed
    # - mean return
    # - return standard deviation
    # - a note about whether the result looks stable
    #
    # Output:
    # - path to the saved report
    raise NotImplementedError("Write the reliability report.")


def run_real_rl_reliability(config: RealRLReliabilityConfig) -> RealRLReliabilitySummary:
    """Run the full multi-seed reliability workflow."""
    # Goal:
    # Train one model per seed, summarize results, write a report, and return the summary.
    #
    # Steps:
    # 1. Build the seed list.
    # 2. Train one seed at a time.
    # 3. Summarize seed results.
    # 4. Write the report.
    # 5. Add `"report_path"` to the summary.
    # 6. Return the summary.
    raise NotImplementedError("Run the real RL reliability workflow.")


def explain_seed_reliability(summary: RealRLReliabilitySummary) -> str:
    """Explain why multiple seeds matter."""
    # Goal:
    # Write one or two plain sentences using `mean_return` and `return_std`.
    #
    # Meaning:
    # - A small spread means runs were similar.
    # - A large spread means the result is less reliable.
    raise NotImplementedError("Explain why one seed is not enough.")


def check_metrics(metrics: RealRLReliabilitySummary) -> bool:
    """Return True when the reliability summary has the expected keys."""
    required = {"env_name", "algorithm", "seed_returns", "mean_return", "return_std", "report_path", "note"}
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
    """Run the real RL reliability scaffold."""
    config = RealRLReliabilityConfig()
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

    _, ok = report_task_progress("describe_reliability_goal", describe_reliability_goal, config)
    if not count_or_stop(ok):
        return

    seeds, ok = report_task_progress("build_seed_list", build_seed_list, config)
    if not count_or_stop(ok):
        return

    first_result, ok = report_task_progress("train_one_seed", train_one_seed, seeds[0], config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("summarize_seed_results", summarize_seed_results, [first_result], config)
    if not count_or_stop(ok):
        return

    summary, ok = report_task_progress("run_real_rl_reliability", run_real_rl_reliability, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and summary is not None:
        passed = check_metrics(summary)
        print_pass_or_try_again(passed, "real rl reliability summary", float(int(passed)), "summary keys present")


if __name__ == "__main__":
    main()
