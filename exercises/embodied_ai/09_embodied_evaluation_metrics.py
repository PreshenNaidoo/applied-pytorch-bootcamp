"""
============================================================
EXERCISE 09: EMBODIED EVALUATION METRICS
============================================================

WHAT YOU ARE BUILDING
You are building a small benchmark for embodied agents.

Plain-language meanings:
- success rate means how often the agent finishes the task
- path length means how many steps or meters the agent used
- collision rate means how often it hit obstacles or hazards
- recovery rate means how often it gets back on track after a forced mistake
- intervention count means how often a person or expert had to correct it
- action latency means how long it takes to choose one action
- calibration means whether the agent's confidence matches how often it is right
- reward measures task score
- cost measures unsafe or unwanted behavior
- rollout prediction error measures world-model prediction quality

WHY THIS EXERCISE EXISTS
Embodied AI needs more than one loss number.
An agent can get high reward while taking unsafe or wasteful actions.

This scorecard should be used for the flagship 2D and 3D projects.
Every serious agent comparison should include task success, safety, path
length, return, cost, and world-model prediction quality when a world model is
part of the agent.

A strong agent should also work when:
- the instruction is reworded
- the task is longer
- the scene or object position changes
- one bad action is forced and the agent must recover
- memory is removed or shuffled

These checks help separate real skill from memorizing a small test setup.

DATASET OPTIONS
Default: evaluate agents in `common.embodied_gridworld`.

Exact local data method:
1. Use `VisualGridWorldConfig(num_episodes=config.num_episodes, max_steps=config.max_steps, random_state=config.random_state)`.
2. Run at least two baselines: random actions and scripted shortest path.
3. Store each result as an `EpisodeResult`.

Real dataset options after the default works:
- `lerobot/pusht` on Hugging Face for success, reward, done, action, and observation checks.
- LeRobot ALOHA simulation datasets for imitation-policy evaluation.
- Matterport3D Simulator / Room-to-Room later for seen versus unseen indoor navigation metrics.

METRICS TO TRACK
- success rate: higher is better
- mean steps: lower is better after success
- timeout rate: lower is better
- collision rate: lower is better
- mean reward: higher is better
- mean cost: lower is better
- mean rollout prediction error: lower is better for world models
- recovery rate after a forced bad action: higher is better
- mean replans and expert interventions: lower is usually better
- mean action latency and peak GPU memory: lower is better at the same success rate
- calibration error: lower is better

VISUALS TO CREATE
Save `outputs/embodied_evaluation/agent_scorecard.png`.
It should compare agents across success, cost, and path length.

DEFINITION OF DONE
You are done when:
- at least five metrics are computed
- several agents can be compared
- seen and unseen settings can be reported separately
- short and long tasks can be reported separately
- original and reworded instructions can be compared
- recovery after a forced mistake is measured
- action latency, GPU memory, and confidence quality are measured
- the metrics can summarize real environment rollouts
- metric tradeoffs are reported
- an agent scorecard chart is saved
- you can explain why high reward is not enough
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal, TypedDict

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.embodied_gridworld import VisualGridWorldConfig


class EpisodeResult(TypedDict):
    """One evaluation episode result."""

    success: bool
    timed_out: bool
    steps: int
    collisions: int
    total_reward: float
    total_cost: float
    rollout_prediction_error: float
    split: str
    task_length_bucket: str
    instruction_variant: str
    perturbation: str
    forced_bad_action: bool
    recovered: bool
    replans: int
    interventions: int
    action_latency_ms: float
    peak_gpu_memory_mb: float
    action_confidences: list[float]
    action_was_acceptable: list[bool]


@dataclass
class EvaluationConfig:
    """Settings for embodied evaluation."""

    max_good_path_length: int = 25
    num_episodes: int = 50
    max_steps: int = 18
    calibration_bins: int = 10
    random_state: int = 42


def compute_success_rate(results: list[EpisodeResult]) -> float:
    """Return fraction of successful episodes."""
    # Goal:
    # Count how often the agent completed the task.
    #
    # Input:
    # - list of EpisodeResult dictionaries.
    #
    # Output:
    # - float between 0 and 1.
    #
    # Method:
    # `success_rate = number_of_successes / number_of_episodes`
    #
    # Checks:
    # - Return `0.0` if the list is empty.
    raise NotImplementedError("Compute success rate.")


def compute_collision_rate(results: list[EpisodeResult]) -> float:
    """Return fraction of episodes with at least one collision."""
    # Goal:
    # Measure how often the agent was unsafe.
    #
    # Method:
    # Count episodes where `collisions > 0`.
    #
    # Checks:
    # - This is episode-level collision rate, not total collision count.
    raise NotImplementedError("Compute collision rate.")


def compute_recovery_rate(results: list[EpisodeResult]) -> float:
    """Return success rate after the evaluator forces one bad action."""
    # Goal:
    # Measure whether the agent can notice a mistake and get back on track.
    #
    # Input:
    # - all episode results, including normal episodes and forced-mistake episodes.
    #
    # Method:
    # 1. Keep only rows where `forced_bad_action` is `True`.
    # 2. Count the kept rows where `recovered` is `True`.
    # 3. Divide by the number of forced-mistake rows.
    #
    # Checks:
    # - Return `0.0` when there are no forced-mistake episodes.
    # - Do not count a normal successful episode as a recovery test.
    raise NotImplementedError("Compute recovery rate after a forced mistake.")


def compute_calibration_error(
    results: list[EpisodeResult],
    num_bins: int,
) -> float:
    """Measure whether confidence matches decision accuracy."""
    # Goal:
    # Check whether an agent that says "90% confident" is right about 90% of
    # the time. An agent can choose good actions but still be dangerously
    # overconfident when it is wrong.
    #
    # Input:
    # - `action_confidences`: confidence for each checked action in an episode.
    # - `action_was_acceptable`: whether each checked action was accepted by
    #   an expert or task-specific action validator.
    # - `num_bins`: number of confidence groups.
    #
    # Output:
    # - expected calibration error, a float from 0 to 1.
    # - lower is better, and `0.0` means confidence perfectly matched accuracy.
    #
    # Method:
    # 1. Flatten action confidences and acceptability labels across episodes.
    # 2. Split confidence into equal ranges such as 0.0-0.1 and 0.1-0.2.
    # 3. In each non-empty range, compute:
    #    `abs(mean_confidence - fraction_correct)`.
    # 4. Weight that gap by the fraction of actions in the range.
    # 5. Add the weighted gaps.
    #
    # Tiny example:
    # If two decisions both have confidence `0.8` but only one is correct,
    # their accuracy is `0.5` and the gap is `abs(0.8 - 0.5) = 0.3`.
    #
    # Checks:
    # - reject `num_bins < 1`.
    # - reject confidence values outside 0 to 1.
    # - each episode must have the same number of confidence values and labels.
    # - return `float("nan")` when no action-level labels were collected.
    raise NotImplementedError("Compute expected calibration error.")


def summarize_embodied_metrics(results: list[EpisodeResult]) -> dict[str, float]:
    """Return a full embodied-agent metric summary."""
    # Goal:
    # Compute several metrics from episode results.
    #
    # Output:
    # - dictionary with:
    #   - `"success_rate"`
    #   - `"mean_steps_on_success"`
    #   - `"timeout_rate"`
    #   - `"collision_rate"`
    #   - `"mean_reward"`
    #   - `"mean_cost"`
    #   - `"mean_rollout_prediction_error"`
    #   - `"recovery_rate"`
    #   - `"mean_replans"`
    #   - `"mean_interventions"`
    #   - `"mean_action_latency_ms"`
    #   - `"peak_gpu_memory_mb"`
    #   - `"calibration_error"`
    #
    # Method:
    # - Use simple averages for mean metrics.
    # - Average `steps` only across successful episodes.
    # - `timeout_rate` is timed-out episodes divided by all episodes.
    # - `peak_gpu_memory_mb` is the maximum episode peak, not an average.
    #
    # Checks:
    # - Do not report only reward.
    # - Lower cost, prediction error, latency, memory use, and calibration
    #   error are better.
    # - A faster agent is not better if its success rate falls sharply. Report
    #   both numbers together.
    raise NotImplementedError("Summarize embodied evaluation metrics.")


def summarize_by_split(results: list[EpisodeResult]) -> dict[str, dict[str, float]]:
    """Return separate metrics for seen and unseen settings."""
    # Goal:
    # Check whether the agent only memorized familiar worlds or also works in changed ones.
    #
    # Input:
    # - episode results where each result has `split`, for example `"seen"` or `"unseen"`.
    #
    # Output:
    # - dictionary like:
    #   - `"seen"` -> metric summary
    #   - `"unseen"` -> metric summary
    #
    # Method:
    # 1. Group results by `result["split"]`.
    # 2. Call `summarize_embodied_metrics(...)` for each group.
    #
    # Checks:
    # - Unseen success rate is usually the stricter number.
    raise NotImplementedError("Summarize embodied metrics by split.")


def summarize_by_condition(
    results: list[EpisodeResult],
    condition: Literal[
        "split",
        "task_length_bucket",
        "instruction_variant",
        "perturbation",
    ],
) -> dict[str, dict[str, float]]:
    """Report the same metrics for each evaluation condition."""
    # Where this fits:
    # One overall average can hide important failures. For example, an agent
    # may solve short tasks but fail every long task.
    #
    # Goal:
    # Group results by one named condition and summarize every group.
    #
    # Examples:
    # - `condition="task_length_bucket"` -> `"short"`, `"medium"`, `"long"`
    # - `condition="instruction_variant"` -> `"original"`, `"paraphrase"`
    # - `condition="perturbation"` -> `"none"`, `"object_moved"`, `"camera_noise"`
    #
    # Steps:
    # 1. Make a dictionary from condition value to result rows.
    # 2. Call `summarize_embodied_metrics(...)` for each group.
    # 3. Return all group summaries.
    raise NotImplementedError("Summarize results by evaluation condition.")


def compare_agents(agent_results: dict[str, list[EpisodeResult]]) -> dict[str, dict[str, float]]:
    """Return metrics for several agents."""
    # Goal:
    # Compare random, reactive, memory, and world-model agents in the same format.
    #
    # Input:
    # - dictionary mapping agent name to episode results.
    #
    # Output:
    # - dictionary mapping agent name to metric summary.
    #
    # Method:
    # Call `summarize_embodied_metrics(...)` for each agent.
    raise NotImplementedError("Compare embodied agents.")


def collect_gridworld_episode_results(config: EvaluationConfig) -> list[EpisodeResult]:
    """Collect real visual grid-world rollouts in the shared metric format."""
    # Goal:
    # Turn real environment rollouts into the metric rows used by this file.
    #
    # Output:
    # - list of EpisodeResult dictionaries.
    #
    # Method:
    # 1. Build `VisualGridWorldConfig(num_episodes=..., max_steps=...)`.
    # 2. Run at least two agent baselines:
    #    - random actions
    #    - scripted shortest-path actions with full map access
    # 3. For each episode, record:
    #    - whether it reached the goal
    #    - whether the episode ended because the step limit was reached
    #    - number of steps
    #    - number of collisions or hazard hits
    #    - total reward
    #    - total cost
    #    - rollout prediction error if a world model is being tested, else `0.0`
    #    - split name, such as `"seen"` or `"unseen"`
    #    - task length group, such as `"short"`, `"medium"`, or `"long"`
    #    - instruction version, such as `"original"` or `"paraphrase"`
    #    - perturbation name, such as `"none"` or `"object_moved"`
    #    - whether a bad action was forced and whether the agent recovered
    #    - replan and expert-intervention counts
    #    - mean action-decision time in milliseconds
    #    - peak GPU memory used during the episode
    #    - action confidence values and whether a task-specific validator
    #      considered each checked action acceptable
    #
    # Checks:
    # - Use the same metric names for every agent.
    # - This makes website charts and tables much easier later.
    # - Run the same fixed episode seeds for each agent.
    # - Change only one test condition at a time so failures are explainable.
    _ = VisualGridWorldConfig
    raise NotImplementedError("Collect grid-world results in the embodied metric format.")


def explain_reward_not_enough() -> str:
    """Return a short evaluation explanation."""
    return "High reward is not enough because the agent may still collide, take too long, or fail in unseen settings."


def main() -> None:
    """Run the full embodied-evaluation scaffold."""
    config = EvaluationConfig()
    try:
        results = collect_gridworld_episode_results(config)
        print(summarize_embodied_metrics(results))
    except NotImplementedError as error:
        print(f"[TODO] collect_gridworld_episode_results: {error}")


if __name__ == "__main__":
    main()
