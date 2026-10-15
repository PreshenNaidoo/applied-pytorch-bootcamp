"""
============================================================
EXERCISE 15: OFFLINE RL AND DATASET BASED RL INTRO
============================================================

WHAT YOU ARE BUILDING
You are building a tiny offline RL teaching example.

Plain-language meanings:
- offline RL means learning from a fixed dataset of past experience
- behavior policy means the policy that created that dataset
- support means which actions the dataset actually contains
- distribution shift means the learned policy may ask for actions the dataset did not show enough
- logged data means saved transitions collected earlier, before current training started

WHY THIS EXERCISE EXISTS
Not all RL learns by collecting fresh experience online.
Sometimes you only have old logged data.
That is useful, but it creates new problems.
This exercise teaches:
- fixed transition dataset
- behavior-policy coverage
- support in simple words
- offline value learning
- why out-of-dataset actions are risky

HOW THIS FITS INTO THE RL SEQUENCE
The earlier RL files mostly learned from fresh online experience.
This file changes the setup:
- now the dataset is fixed
- the learner does not get to ask the environment for more data
That is a very important shift in both intuition and risk.

Beginner intuition:
- imagine learning to drive only from old recorded trips
- if the dataset almost never shows a certain road choice,
  you should be careful about trusting a learned value for that choice
- offline RL is powerful, but it must respect what the dataset really covers

Small example:
- in one state, the dataset mostly shows action `left`
- action `right` appears only once or never
- if your learned policy loves `right`,
  that may be a warning sign because the data did not support it well

IMPORTANT SCOPE NOTE
This is a very small teaching version of offline RL.
It is here to teach the core data problem, not the full research literature.

DATASET
Use a tiny local transition dataset written directly in code.

Clear default:
- use the same 5-state `LineWorld` idea from the earlier RL exercises
- write 12 to 20 logged transitions by hand
- make the logged behavior mostly choose `right`
- include a few weakly covered actions so the support problem is visible

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- write a short list of logged transitions by hand
- include state, action, reward, next_state, and done
- make some states well covered and some actions weakly covered
- keep the dataset tiny so the support problem is easy to see

INPUTS AND OUTPUTS
Input:
- fixed transition dataset
- discount factor
- number of fitted update rounds
Output:
- action-support counts
- weak-support states
- learned offline Q-values
- greedy policy
- short coverage note

METRICS TO TRACK
Track at least:
- support counts
- weak-support states
- learned greedy policy
- short distribution-shift note

TARGET RESULT RANGE
Focus on understanding coverage, not on high reward.

Useful signs:
- you can clearly see which actions were common and which were rare
- the offline value updates use only the logged dataset
- the note explains why unsupported actions are risky

COMMON MISTAKES
- forgetting that the learner cannot collect new data here
- treating a never-seen action as if its value were trustworthy
- ignoring support counts and jumping straight to a policy
- making the dataset so random that the coverage lesson is hidden
- mixing online-RL intuition directly into the offline setting

What each step is doing in simple words:
1. build a fixed history of old experience
2. count what the dataset actually shows
3. spot states where action coverage is weak
4. fit value estimates only from the logged data
5. see which actions the learned policy prefers
6. explain why poor coverage can make those preferences risky

STEP-BY-STEP TODO
1. Build a tiny offline transition dataset.
2. Count which actions appear in each state.
3. Measure simple support coverage.
4. Fit a tiny tabular value estimate from the fixed data.
5. Compare covered and weakly covered actions.
6. Explain distribution shift risk in plain words.

DEFINITION OF DONE
You are done when:
- a fixed dataset is built
- action support is measured
- value learning uses only the logged data
- weak-coverage warnings are reported
- you can explain why offline RL is different from online RL
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
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
class OfflineRLConfig:
    """Settings for the offline RL exercise."""
    gamma: float = 0.95
    num_fitted_iterations: int = 20


class OfflineRLTerms(TypedDict):
    """Main offline RL words used in this exercise."""

    gamma: float
    plain_words: dict[str, str]


class Transition(TypedDict):
    """One logged transition in the fixed offline dataset."""

    state: str
    action: str
    reward: float
    next_state: str
    done: bool


class OfflineRLSummary(TypedDict):
    """Final summary returned by the offline RL demo."""

    support_counts: dict[str, dict[str, int]]
    weak_support_states: list[str]
    q_values: dict[str, dict[str, float]]
    greedy_policy: dict[str, str]
    note: str

# Beginner hints for this file:
# - in offline RL the dataset is fixed; you do not ask the environment for more data
# - some actions may be well covered in the data and some may be almost absent
# - absent actions are dangerous because the value estimate can become unrealistic there
# - count the action coverage first so the dataset stops feeling abstract
# - keep the first example tiny enough to inspect by hand
def describe_offline_rl_terms(config: OfflineRLConfig) -> OfflineRLTerms:
    """Return the main offline RL words used here."""
    return {
        "gamma": config.gamma,
        "plain_words": {
            "offline_rl": "learning from a fixed log of old experience",
            "support": "which actions the dataset actually shows in each state",
            "distribution_shift": "the learned policy asks for situations the dataset did not cover well",
        },
    }

# Build a tiny fixed transition dataset.
# Each transition should include at least:
# - state
# - action
# - reward
# - next_state
# - done
def build_offline_dataset() -> list[Transition]:
    """Return a tiny local offline RL dataset."""
    # Goal:
    # Return a fixed list of logged transitions.
    # Fixed means the learner must use only these rows and cannot ask the environment for more data.
    #
    # Input:
    # - No direct arguments.
    #
    # Output:
    # - `list[Transition]`
    # - Use 12 to 20 transitions.
    #
    # Each transition must have these exact keys:
    # - `"state"`: str, such as `"s0"`
    # - `"action"`: str, either `"left"` or `"right"`
    # - `"reward"`: float
    # - `"next_state"`: str, such as `"s1"`
    # - `"done"`: bool
    #
    # Small example:
    # ```python
    # {"state": "s0", "action": "right", "reward": -0.01, "next_state": "s1", "done": False}
    # ```
    #
    # Method:
    # Write the dataset by hand as a Python list.
    # Make `"right"` common in most states.
    # Include at least one weakly covered action, such as `"left"` appearing only once in `"s2"`.
    # Include at least one terminal transition into the goal state `"s4"`.
    #
    # Steps:
    # 1. Create an empty list or write the list directly.
    # 2. Add transitions from `"s0"` through `"s4"`.
    # 3. Use reward `1.0` for reaching `"s4"`.
    # 4. Use reward `-0.01` for normal non-terminal steps.
    # 5. Set `done=True` only when the transition reaches the goal.
    # 6. Return the dataset list.
    #
    # Checks:
    # - `len(dataset)` should be at least `12`.
    # - Every transition should have the five keys listed above.
    # - The dataset should include both `"left"` and `"right"` somewhere.
    # - This function should not train values or choose a policy.
    raise NotImplementedError("Return a fixed list of logged transitions.")

def count_action_support(dataset: list[Transition]) -> dict[str, dict[str, int]]:
    """Return how often each action appears in each state."""
    # Goal:
    # Count how often each action appears for each state in the fixed dataset.
    #
    # Input:
    # - `dataset`: list[Transition]
    #
    # Output:
    # - `dict[str, dict[str, int]]`
    #
    # Example:
    # ```python
    # {
    #     "s0": {"left": 0, "right": 4},
    #     "s1": {"left": 1, "right": 3},
    # }
    # ```
    #
    # Meaning:
    # - `"s0"` is a state.
    # - `"right": 4` means the dataset took action `"right"` four times from `"s0"`.
    #
    # Method:
    # Use normal dictionaries.
    # Start every seen state with both actions:
    # `{"left": 0, "right": 0}`
    #
    # Steps:
    # 1. Create an empty `support_counts` dictionary.
    # 2. Loop through each transition.
    # 3. Read `state = transition["state"]`.
    # 4. Read `action = transition["action"]`.
    # 5. If the state is new, create `{"left": 0, "right": 0}` for it.
    # 6. Add 1 to the count for that state and action.
    # 7. Return `support_counts`.
    #
    # Checks:
    # - The total of all counts should equal `len(dataset)`.
    # - States that appear in the dataset should have both `"left"` and `"right"` keys.
    # - This function should only count; it should not learn Q-values.
    raise NotImplementedError("Return the action coverage counts for each state.")

def find_weak_support_states(support_counts: dict[str, dict[str, int]]) -> list[str]:
    """Return states where one or more actions have weak or zero support."""
    # Goal:
    # Return states where at least one action is missing or rare.
    #
    # Input:
    # - `support_counts`: dict[str, dict[str, int]]
    #
    # Output:
    # - `list[str]`
    # - Each string is a state name, such as `"s2"`.
    #
    # Rule:
    # A state has weak support if either action count is less than `2`.
    #
    # Example:
    # - `{"left": 0, "right": 5}` is weak because `"left"` never appears.
    # - `{"left": 1, "right": 4}` is weak because `"left"` appears only once.
    # - `{"left": 2, "right": 3}` is not weak by this simple rule.
    #
    # Method:
    # Use `min(action_counts.values()) < 2`.
    #
    # Steps:
    # 1. Create an empty `weak_states` list.
    # 2. Loop through `support_counts.items()`.
    # 3. For each state, find the smallest action count.
    # 4. If that count is less than `2`, add the state to the list.
    # 5. Return the list.
    #
    # Checks:
    # - Return an empty list if no state is weak.
    # - Do not include a state twice.
    raise NotImplementedError("Return states with weak action coverage.")

# Fit a tiny tabular value estimate using only the offline dataset.
# A simple first version can use repeated fitted updates over the logged transitions.
def fit_tabular_offline_q(
    dataset: list[Transition],
    gamma: float,
    num_iterations: int,
) -> dict[str, dict[str, float]]:
    """Return a tiny offline Q estimate from logged transitions."""
    # Goal:
    # Return tabular Q-values learned only from the fixed offline dataset.
    # Do not collect new transitions.
    #
    # Input:
    # - `dataset`: list[Transition]
    # - `gamma`: float
    # - `num_iterations`: int
    #
    # Output:
    # - `dict[str, dict[str, float]]`
    #
    # Example:
    # ```python
    # {
    #     "s0": {"left": 0.0, "right": 0.6},
    #     "s1": {"left": 0.1, "right": 0.8},
    # }
    # ```
    #
    # Formula:
    # ```text
    # target = reward + gamma * max_next_q * (1 - done)
    # ```
    #
    # Meaning:
    # - `reward` is the logged reward from one transition.
    # - `max_next_q` is the best current Q-value for `next_state`.
    # - `(1 - done)` removes future value after a terminal transition.
    #
    # Teaching update:
    # For this small exercise, replace the Q-value directly:
    # `q_values[state][action] = target`
    #
    # Method:
    # Use plain Python dictionaries.
    # Initialize every seen state with:
    # `{"left": 0.0, "right": 0.0}`
    #
    # Include both `state` and `next_state` in the Q-table so `max_next_q` can be read safely.
    #
    # Steps:
    # 1. Create a Q-table with both actions set to `0.0` for every state and next state seen in the dataset.
    # 2. Repeat for `num_iterations`.
    # 3. Loop through each logged transition.
    # 4. Read state, action, reward, next_state, and done.
    # 5. Compute `max_next_q = max(q_values[next_state].values())`.
    # 6. Compute the target with the formula above.
    # 7. Replace `q_values[state][action]` with the target.
    # 8. Return the Q-table.
    #
    # Checks:
    # - This function should never call an environment step function.
    # - Terminal transitions should use target equal to reward.
    # - Unsupported actions may stay near their initial value, which is part of the offline RL lesson.
    raise NotImplementedError("Return tabular Q-values learned from the fixed dataset.")

def choose_greedy_offline_policy(q_values: dict[str, dict[str, float]]) -> dict[str, str]:
    """Return the greedy action in each state."""
    # Goal:
    # Return the action with the largest Q-value in each state.
    #
    # Input:
    # - `q_values`: dict[str, dict[str, float]]
    #
    # Output:
    # - `dict[str, str]`
    #
    # Example:
    # If `q_values["s1"] = {"left": 0.2, "right": 0.7}`,
    # then the policy should include `"s1": "right"`.
    #
    # Method:
    # Use:
    # `max(action_values, key=action_values.get)`
    #
    # Here `action_values` is one inner dictionary, such as `{"left": 0.2, "right": 0.7}`.
    #
    # Steps:
    # 1. Create an empty `policy` dictionary.
    # 2. Loop through `q_values.items()`.
    # 3. For each state, find the action with the largest value.
    # 4. Store that action in `policy[state]`.
    # 5. Return `policy`.
    #
    # Checks:
    # - Every state in `q_values` should appear in the returned policy.
    # - If two actions tie, `max(...)` can pick either one. That is fine for this exercise.
    raise NotImplementedError("Return the greedy policy from the learned offline Q-values.")

def explain_distribution_shift(
    support_counts: dict[str, dict[str, int]],
    learned_policy: dict[str, str],
) -> str:
    """Write a short note about coverage and distribution shift."""
    # Goal:
    # Explain why weakly covered actions can be risky in offline RL.
    #
    # Input:
    # - `support_counts`: dict[str, dict[str, int]]
    # - `learned_policy`: dict[str, str]
    #
    # Output:
    # - `str`
    # - One or two plain sentences.
    #
    # Method:
    # Check whether the learned policy chooses an action with low support.
    # Low support means the action count is less than `2` in that state.
    #
    # Example:
    # If `learned_policy["s2"] == "left"` and `support_counts["s2"]["left"] == 1`,
    # say the policy is trusting an action that only appeared once.
    #
    # Steps:
    # 1. Create an empty list of risky choices.
    # 2. Loop through the learned policy.
    # 3. Look up the support count for the chosen action.
    # 4. If the count is less than `2`, add a short description to risky choices.
    # 5. Write one or two sentences explaining the risk.
    # 6. Return the note.
    #
    # Checks:
    # - Keep the note short and plain.
    # - Mention that offline RL cannot easily check missing actions by trying them live.
    raise NotImplementedError("Explain why weakly covered actions can be risky in offline RL.")

def run_offline_rl_demo(config: OfflineRLConfig) -> OfflineRLSummary:
    """Run the full tiny offline RL workflow."""
    # Goal:
    # Build the dataset, measure support, fit offline Q-values, and return a summary.
    #
    # Input:
    # - `config`: OfflineRLConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"support_counts"`: dict[str, dict[str, int]]
    # - `"weak_support_states"`: list[str]
    # - `"q_values"`: dict[str, dict[str, float]]
    # - `"greedy_policy"`: dict[str, str]
    # - `"note"`: str
    #
    # Method:
    # Use the helper functions in this file:
    # - `dataset = build_offline_dataset()`
    # - `support_counts = count_action_support(dataset)`
    # - `weak_support_states = find_weak_support_states(support_counts)`
    # - `q_values = fit_tabular_offline_q(dataset, config.gamma, config.num_fitted_iterations)`
    # - `greedy_policy = choose_greedy_offline_policy(q_values)`
    # - `note = explain_distribution_shift(support_counts, greedy_policy)`
    #
    # Steps:
    # 1. Build the fixed offline dataset.
    # 2. Count action support.
    # 3. Find weak-support states.
    # 4. Fit tabular Q-values from the fixed dataset.
    # 5. Choose the greedy policy from the Q-values.
    # 6. Write the distribution-shift note.
    # 7. Return the summary dictionary.
    #
    # Checks:
    # - The summary should include both support counts and the learned policy.
    # - `weak_support_states` should come from support counts, not from Q-values.
    # - The workflow should not collect new environment experience.
    raise NotImplementedError("Build the dataset, measure support, fit offline values, and return a summary.")

def check_metrics(metrics: OfflineRLSummary) -> bool:
    """Return True when the offline RL summary has the expected keys."""
    required = {"support_counts", "weak_support_states", "greedy_policy", "note"}
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
    """Run the offline RL scaffold."""
    completed_steps = 0
    total_steps = 8
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    dataset, ok = report_task_progress("build_offline_dataset", build_offline_dataset)
    if not count_or_stop(ok):
        return
    config = OfflineRLConfig()
    _, ok = report_task_progress("describe_offline_rl_terms", describe_offline_rl_terms, config)
    if not count_or_stop(ok):
        return
    support_counts, ok = report_task_progress("count_action_support", count_action_support, dataset)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("find_weak_support_states", find_weak_support_states, support_counts)
    if not count_or_stop(ok):
        return
    q_values, ok = report_task_progress("fit_tabular_offline_q", fit_tabular_offline_q, dataset, config.gamma, config.num_fitted_iterations)
    if not count_or_stop(ok):
        return
    learned_policy, ok = report_task_progress("choose_greedy_offline_policy", choose_greedy_offline_policy, q_values)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("explain_distribution_shift", explain_distribution_shift, support_counts, learned_policy)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_offline_rl_demo", run_offline_rl_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "offline RL summary", float(int(passed)), "offline rl keys present")
if __name__ == "__main__":
    main()
