"""
============================================================
EXERCISE 13: RL DEBUGGING AND EVALUATION
============================================================

WHAT YOU ARE BUILDING
You are building a tiny debugging and evaluation workflow for RL experiments.

Plain-language meanings:
- training return is the reward seen while the agent is learning
- evaluation return is the reward from a cleaner test run
- seed variance means different random seeds can give different results
- instability means the learning curve jumps around too much to trust easily

WHY THIS EXERCISE EXISTS
RL is often harder to debug than supervised learning.
A curve can look good for the wrong reason.
Another curve can look bad just because one seed was unlucky.
Research engineers need habits for reading RL results carefully.
This exercise teaches:
- train vs evaluation split
- mean return across seeds
- instability checks
- reward-curve reading
- failure example notes
- plain-language evaluation report

HOW THIS FITS INTO THE RL SEQUENCE
This file comes after the first RL algorithms.
Once you can train an agent,
you need to learn how to read the result honestly.
This is important because RL can look better or worse than it really is
if you only inspect one noisy curve.

Beginner intuition:
- one good RL curve is not enough
- you want to know:
  - did the result repeat across seeds?
  - did evaluation also improve?
  - did learning stay stable?
  - where did the agent still fail?

Small example:
- training reward goes up
- evaluation reward stays flat
- that suggests the training signal may be misleading
  or the agent may be overfitting to exploration noise

DATASET
Use tiny local logged results from earlier RL exercises.
The point is reading and checking RL experiments carefully.

Clear default:
- use logged runs from the same `LineWorld` setup used in exercises 75 to 78
- include at least 3 random seeds
- each run should include:
  - training returns by episode
  - evaluation returns by episode
  - final success rate
  - one short note if the run was unstable

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- reuse a few small run histories from earlier RL exercises
- include several random seeds
- include both training and evaluation returns
- add one or two short failure notes to make the numbers easier to interpret

INPUTS AND OUTPUTS
Input:
- run histories from several seeds
- training-return curves
- evaluation-return curves
- failure notes
Output:
- mean train return
- mean eval return
- spread across seeds
- instability flags
- short debugging report

METRICS TO TRACK
Track at least:
- mean train return
- mean eval return
- spread across seeds
- instability flags

TARGET RESULT RANGE
Focus on honest reading of results, not on achieving a high return.

Useful signs:
- training and evaluation are reported separately
- more than one seed is included
- suspicious patterns are written down, not hidden
- failure cases are connected to the curves in plain language

COMMON MISTAKES
- trusting one lucky seed too much
- reporting only training return
- averaging curves without looking at spread
- hiding failure cases because the average looks decent
- treating RL evaluation as if it were a single clean classification accuracy

What each step is doing in simple words:
1. gather several small run histories instead of one
2. compute the average shape across seeds
3. measure how spread out those seeds are
4. compare learning-time reward with cleaner evaluation reward
5. flag patterns that look unstable or suspicious
6. connect the numbers to concrete failure stories

STEP-BY-STEP TODO
1. Build a few tiny RL run histories.
2. Compute mean and spread across seeds.
3. Compare training return and evaluation return.
4. Flag unstable or suspicious runs.
5. List a few failure cases worth reading by hand.
6. Write a short RL debugging report.

DEFINITION OF DONE
You are done when:
- several runs are compared
- mean return across seeds is computed
- training and evaluation are compared
- suspicious patterns are flagged
- you can explain why one seed is not enough
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from typing import TypedDict

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
class RLDebuggingConfig:
    """Settings for the RL debugging exercise."""
    report_path: str = str(BOOTCAMP_ROOT / "reports" / "79_rl_debugging_and_evaluation.md")


class DebuggingPlanSummary(TypedDict):
    """Main checks used in this exercise."""

    checks: list[str]
    report_path: str


class RunHistory(TypedDict):
    """One small RL run history from one random seed."""

    seed: int
    train_returns: list[float]
    eval_returns: list[float]
    success_rate: float
    failure_notes: list[str]
    unstable_note: str


class DebugReportInputs(TypedDict):
    """Values used to write the debugging report."""

    mean_train_return: list[float]
    mean_eval_return: list[float]
    eval_seed_spread: list[float]
    instability_flags: list[str]
    failure_cases: list[str]


class RLDebuggingSummary(DebugReportInputs):
    """Final summary returned by the RL debugging workflow."""

    report_path: str

# Beginner hints for this file:
# - RL curves are often noisy, so one run is rarely enough
# - evaluation should usually be cleaner than training because exploration may be reduced
# - a high training curve with weak eval can be a warning sign
# - write down suspicious patterns instead of only showing one average number
# - failed examples give the reward curve a real story
def describe_debugging_plan(config: RLDebuggingConfig) -> DebuggingPlanSummary:
    """Return the main debugging checks used in this exercise."""
    return {
        "checks": ["mean_return", "seed_variance", "train_vs_eval", "instability_flags", "failure_notes"],
        "report_path": config.report_path,
    }

# Return several tiny RL run histories.
# Each run can include:
# - seed
# - train_returns
# - eval_returns
# - short failure notes
def build_run_histories() -> list[RunHistory]:
    """Return tiny local RL run histories."""
    # Goal:
    # Return a few small RL run histories that look like logged experiment results.
    # Each run should come from a different random seed.
    #
    # Input:
    # - No direct arguments.
    #
    # Output:
    # Return a list of at least 3 dictionaries.
    # Each dictionary must have these exact keys:
    # - `"seed"`: int
    # - `"train_returns"`: list[float], one value per episode checkpoint
    # - `"eval_returns"`: list[float], one value per episode checkpoint
    # - `"success_rate"`: float between `0.0` and `1.0`
    # - `"failure_notes"`: list[str]
    # - `"unstable_note"`: str
    #
    # Small example:
    # ```python
    # {
    #     "seed": 0,
    #     "train_returns": [-0.2, 0.1, 0.4, 0.8],
    #     "eval_returns": [-0.1, 0.0, 0.3, 0.7],
    #     "success_rate": 0.75,
    #     "failure_notes": ["Still walks left in some starts."],
    #     "unstable_note": "Learning improves but has one dip.",
    # }
    # ```
    #
    # Method:
    # Write the histories by hand as small Python lists.
    # Keep all `train_returns` lists the same length.
    # Keep all `eval_returns` lists the same length.
    #
    # Steps:
    # 1. Create a list with 3 run dictionaries.
    # 2. Give each run a different `seed`.
    # 3. Include both training and evaluation curves.
    # 4. Include at least one failure note across the runs.
    # 5. Return the list.
    #
    # Checks:
    # - `len(run_histories)` should be at least `3`.
    # - Every run should have the keys listed above.
    # - All train curves should have the same length.
    # - All eval curves should have the same length.
    raise NotImplementedError("Return a few RL run histories for debugging and evaluation.")

def compute_mean_return_across_seeds(run_histories: list[RunHistory], key: str) -> list[float]:
    """Return the mean curve across seeds for the chosen key."""
    # Goal:
    # Return the mean return curve across all seeds for one curve key.
    #
    # Input:
    # - `run_histories`: list[RunHistory]
    #   Each run must contain a list under `key`.
    # - `key`: str
    #   Use `"train_returns"` or `"eval_returns"`.
    #
    # Output:
    # - `list[float]`
    # - Length: number of return points in each run.
    #
    # Formula:
    # ```text
    # mean_at_step = sum(run[key][step] for each run) / number_of_runs
    # ```
    #
    # Small example:
    # If seed curves are `[1.0, 2.0]` and `[3.0, 4.0]`,
    # the mean curve is `[2.0, 3.0]`.
    #
    # Method:
    # Use:
    # - `curve_length = len(run_histories[0][key])`
    # - `statistics.mean(values_at_step)`
    #
    # `statistics.mean(...)` is from Python's standard library.
    #
    # Steps:
    # 1. Check that `run_histories` is not empty.
    # 2. Read the curve length from the first run.
    # 3. For each step index, collect that step's value from every run.
    # 4. Compute the mean for that step.
    # 5. Return the mean curve.
    #
    # Checks:
    # - The result length should equal `len(run_histories[0][key])`.
    # - Every run should have the same curve length for this key.
    raise NotImplementedError("Return the mean return at each step across runs.")

def compute_seed_spread(run_histories: list[RunHistory], key: str) -> list[float]:
    """Return a simple spread measure across seeds for the chosen key."""
    # Goal:
    # Return one spread value per step across seeds.
    # Spread means how different the seeds are from each other.
    #
    # Input:
    # - `run_histories`: list[RunHistory]
    # - `key`: str
    #   Use `"train_returns"` or `"eval_returns"`.
    #
    # Output:
    # - `list[float]`
    # - Length: number of return points in each run.
    #
    # Formula:
    # ```text
    # spread_at_step = max(values_at_step) - min(values_at_step)
    # ```
    #
    # Small example:
    # If seed values at one step are `[0.1, 0.5, 0.9]`,
    # the spread is `0.9 - 0.1 = 0.8`.
    #
    # Method:
    # Use Python `max(...)` and `min(...)` on the values from the same step.
    #
    # Steps:
    # 1. Check that `run_histories` is not empty.
    # 2. Read the curve length from the first run.
    # 3. For each step index, collect that step's value from every run.
    # 4. Compute `max(values_at_step) - min(values_at_step)`.
    # 5. Return the spread curve.
    #
    # Checks:
    # - Spread should never be negative.
    # - Large spread means the result depends a lot on the random seed.
    raise NotImplementedError("Return one spread value per step across seeds.")

# Flag suspicious training behavior.
# Example warning signs:
# - training rises but evaluation does not
# - large jumpiness
# - one seed much better than the others
def detect_instability_signs(run_histories: list[RunHistory]) -> list[str]:
    """Return plain-language instability warnings."""
    # Goal:
    # Return warning messages for suspicious RL result patterns.
    #
    # Input:
    # - `run_histories`: list[RunHistory]
    #
    # Output:
    # - `list[str]`
    # - Each string should be one plain-language warning.
    #
    # Warning examples:
    # - `"Evaluation is much lower than training at the final checkpoint."`
    # - `"Seed spread is large, so one seed is not enough to trust the result."`
    # - `"One run reports unstable learning: ..."`
    #
    # Method:
    # Use helpers from this file:
    # - `mean_train = compute_mean_return_across_seeds(run_histories, "train_returns")`
    # - `mean_eval = compute_mean_return_across_seeds(run_histories, "eval_returns")`
    # - `eval_spread = compute_seed_spread(run_histories, "eval_returns")`
    #
    # Simple rules you can use:
    # - If final train mean is more than `0.2` above final eval mean, add a train-vs-eval warning.
    # - If final eval spread is more than `0.5`, add a seed-spread warning.
    # - If any run has a non-empty `unstable_note`, add that note as a warning.
    #
    # Steps:
    # 1. Create an empty warnings list.
    # 2. Compute mean train and eval curves.
    # 3. Compare their final values.
    # 4. Compute eval seed spread and check its final value.
    # 5. Read each run's `unstable_note`.
    # 6. Return the warnings list.
    #
    # Checks:
    # - If no warning is found, return an empty list instead of `None`.
    # - Warnings should explain the problem in plain words.
    raise NotImplementedError("Return suspicious training or evaluation patterns.")

def summarize_failure_cases(run_histories: list[RunHistory]) -> list[str]:
    """Return a few concrete failure notes worth reading."""
    # Goal:
    # Return concrete failure notes from the run histories.
    # These notes connect reward curves to behavior a person can inspect.
    #
    # Input:
    # - `run_histories`: list[RunHistory]
    #
    # Output:
    # - `list[str]`
    # - Each string should mention the seed and one failure note.
    #
    # Example output:
    # - `"Seed 2: agent still moves left at the start."`
    #
    # Method:
    # Read each run's `"failure_notes"` list.
    # Prefix each note with its seed.
    #
    # Steps:
    # 1. Create an empty `failure_cases` list.
    # 2. Loop through every run.
    # 3. Read `seed = run["seed"]`.
    # 4. Loop through `run["failure_notes"]`.
    # 5. Add strings like `f"Seed {seed}: {note}"`.
    # 6. Return the list.
    #
    # Checks:
    # - Return an empty list if there are no failure notes.
    # - Do not hide failure notes just because the average return looks good.
    raise NotImplementedError("Return a few failure examples from the runs.")

def write_debug_report(config: RLDebuggingConfig, summary: DebugReportInputs) -> str:
    """Write and save a short RL debugging report."""
    # Goal:
    # Write a short Markdown report that explains the RL results in plain language.
    #
    # Input:
    # - `config`: RLDebuggingConfig
    # - `summary`: DebugReportInputs
    #
    # Output:
    # - `str`
    # - The path where the report was saved.
    #
    # Summary keys this function should read:
    # - `"mean_train_return"`: list[float]
    # - `"mean_eval_return"`: list[float]
    # - `"eval_seed_spread"`: list[float]
    # - `"instability_flags"`: list[str]
    # - `"failure_cases"`: list[str]
    #
    # Method:
    # Use `pathlib.Path`:
    # - `report_path = Path(config.report_path)`
    # - `report_path.parent.mkdir(parents=True, exist_ok=True)`
    # - `report_path.write_text(report_text, encoding="utf-8")`
    #
    # Keep the report short.
    # Include headings such as:
    # - `# RL Debugging Report`
    # - `## Mean Returns`
    # - `## Warnings`
    # - `## Failure Cases`
    #
    # Steps:
    # 1. Convert `config.report_path` into a `Path`.
    # 2. Create the parent folder.
    # 3. Build a Markdown string with the key numbers and notes.
    # 4. Write the file with UTF-8 encoding.
    # 5. Return `str(report_path)`.
    #
    # Checks:
    # - The report should say training and evaluation are different checks.
    # - The report should include warnings and failure cases when they exist.
    # - Do not make the report sound more certain than the numbers support.
    raise NotImplementedError("Write a plain-language RL debugging report.")

def run_debugging_workflow(config: RLDebuggingConfig) -> RLDebuggingSummary:
    """Run the full RL debugging and evaluation workflow."""
    # Goal:
    # Analyze several RL runs and return a debugging summary.
    #
    # Input:
    # - `config`: RLDebuggingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"mean_train_return"`: list[float]
    # - `"mean_eval_return"`: list[float]
    # - `"eval_seed_spread"`: list[float]
    # - `"instability_flags"`: list[str]
    # - `"failure_cases"`: list[str]
    # - `"report_path"`: str
    #
    # Method:
    # Use the helper functions in this file:
    # - `run_histories = build_run_histories()`
    # - `mean_train = compute_mean_return_across_seeds(run_histories, "train_returns")`
    # - `mean_eval = compute_mean_return_across_seeds(run_histories, "eval_returns")`
    # - `eval_spread = compute_seed_spread(run_histories, "eval_returns")`
    # - `instability_flags = detect_instability_signs(run_histories)`
    # - `failure_cases = summarize_failure_cases(run_histories)`
    # - `report_path = write_debug_report(config, summary)`
    #
    # Steps:
    # 1. Build the run histories.
    # 2. Compute mean training return.
    # 3. Compute mean evaluation return.
    # 4. Compute evaluation spread across seeds.
    # 5. Detect instability warnings.
    # 6. Summarize failure cases.
    # 7. Build the summary dictionary without `report_path` first.
    # 8. Write the report.
    # 9. Add `"report_path"` to the summary.
    # 10. Return the summary.
    #
    # Checks:
    # - The summary should include both train and eval results.
    # - The report path should point to the file that was written.
    # - Do not report only one seed.
    raise NotImplementedError("Analyze several runs and return a debugging summary.")

def check_metrics(metrics: RLDebuggingSummary) -> bool:
    """Return True when the RL debugging summary has the expected keys."""
    required = {"mean_train_return", "mean_eval_return", "instability_flags", "report_path"}
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
    """Run the RL debugging scaffold."""
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
    config = RLDebuggingConfig()
    _, ok = report_task_progress("describe_debugging_plan", describe_debugging_plan, config)
    if not count_or_stop(ok):
        return

    histories, ok = report_task_progress("build_run_histories", build_run_histories)
    if not count_or_stop(ok):
        return

    mean_train_return, ok = report_task_progress("compute_mean_return_across_seeds", compute_mean_return_across_seeds, histories, "train_returns")
    if not count_or_stop(ok):
        return

    mean_eval_return, ok = report_task_progress("compute_mean_return_across_seeds", compute_mean_return_across_seeds, histories, "eval_returns")
    if not count_or_stop(ok):
        return

    eval_seed_spread, ok = report_task_progress("compute_seed_spread", compute_seed_spread, histories, "eval_returns")
    if not count_or_stop(ok):
        return

    instability_flags, ok = report_task_progress("detect_instability_signs", detect_instability_signs, histories)
    if not count_or_stop(ok):
        return

    failure_cases, ok = report_task_progress("summarize_failure_cases", summarize_failure_cases, histories)
    if not count_or_stop(ok):
        return

    debug_summary = {
        "mean_train_return": mean_train_return,
        "mean_eval_return": mean_eval_return,
        "eval_seed_spread": eval_seed_spread,
        "instability_flags": instability_flags,
        "failure_cases": failure_cases,
    }
    _, ok = report_task_progress("write_debug_report", write_debug_report, config, debug_summary)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_debugging_workflow", run_debugging_workflow, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "RL debugging summary", float(int(passed)), "rl debugging keys present")
if __name__ == "__main__":
    main()
