"""
============================================================
EXERCISE 02: MARKOV DECISION PROCESSES AND DYNAMIC PROGRAMMING
============================================================

WHAT YOU ARE BUILDING
You are building a tiny environment with a few named states.

The agent can choose actions.
Each action moves the agent to a next state and gives a reward.
In this exercise, the environment rules are already known.
That means the code can look up:

- where each action goes next
- what reward each action gives

The formal name for this kind of known environment is a Markov decision process, or MDP.

WHY THIS EXERCISE EXISTS
the related exercise had actions and rewards, but no state.

This exercise adds state.
Now the agent can be in different places, and the best action may depend on where it is.

The main idea is:

1. Write down the known environment rules.
2. Start with a simple policy.
3. Evaluate that policy by estimating how good each state is.
4. Improve the policy by choosing better actions from those values.
5. Repeat until the policy stops changing.

This is still simpler than later RL exercises.
The agent is not learning the environment rules from trial and error.
The environment rules are already written down in dictionaries.

THE FULL PICTURE
Think of a tiny map:

```text
start -> middle -> goal
```

There are two possible actions from the non-terminal states:

- `"left"` is the bad action because it wastes time or moves backward
- `"right"` is the good action because it moves toward the goal

The environment tells us what happens:

```python
transitions[("start", "left")] = "start"
rewards[("start", "left")] = -0.1
transitions[("start", "right")] = "middle"
rewards[("start", "right")] = 0.0
transitions[("middle", "left")] = "start"
rewards[("middle", "left")] = 0.0
transitions[("middle", "right")] = "goal"
rewards[("middle", "right")] = 1.0
```

This is the point of the exercise:

```text
The first policy may choose bad actions.
Policy evaluation scores that policy.
Policy improvement uses those scores to choose better actions.
```

Read one transition line like this:

```python
transitions[("start", "right")] = "middle"
```

It means:

- the current state is `"start"`
- the action is `"right"`
- the next state becomes `"middle"`

The key is the pair `("start", "right")`.
That pair means `(current_state, action)`.
The value is `"middle"`.
That value means `next_state`.

So the full sentence is:

```text
If the agent is in "start" and chooses "right",
the environment moves the agent to "middle".
```

Read one reward line like this:

```python
rewards[("start", "right")] = 0.0
```

It means:

- the current state is `"start"`
- the action is `"right"`
- the reward from the environment is `0.0`

The key is the pair `("start", "right")`.
That pair means `(current_state, action)`.
The value is `0.0`.
That value means `reward`.

So the full sentence is:

```text
If the agent is in "start" and chooses "right",
the environment gives reward 0.0.
```

The policy tells the agent what action to take in each non-terminal state.

A non-terminal state means a state where the episode is not over yet.
The agent still needs to choose an action there.

A terminal state means the episode is already over.
The agent does not need to choose another action there.

In this tiny environment:

- `"start"` is non-terminal because the agent still needs to move
- `"middle"` is non-terminal because the agent still needs to move
- `"goal"` is terminal because the episode is done

So the policy only needs actions for `"start"` and `"middle"`.

A weak starting policy might choose `"left"`:

```python
policy = {"start": "left", "middle": "left"}
```

After policy improvement, a better policy should choose `"right"`:

```python
policy = {"start": "right", "middle": "right"}
```

Keep these two ideas separate:

The environment says:

```text
If you take this action here,
I will move you there and give this reward.
```

The policy says:

```text
When I am here,
I will choose this action.
```

The values tell us how good each state looks under the current policy:

```python
values = {"start": 0.0, "middle": 0.0, "goal": 0.0}
```

The main update is:

```python
new_value[state] = reward + gamma * old_value[next_state]
```

Read that as:
- reward is what you get now
- next state value is how good the next state looks
- gamma controls how much future reward matters now

There are two repeated jobs:

1. Policy evaluation
   - keep the policy fixed
   - update `values` until they describe that policy better

2. Policy improvement
   - keep the values fixed
   - choose the best action for each state using those values

Together, those two jobs are called policy iteration.

DATASET
No dataset file is needed.
Build a tiny environment directly in code.

INPUTS AND OUTPUTS
Input:
- tiny environment definition
- starting policy
- discount factor `gamma`

Output:
- state values
- improved policy
- number of improvement rounds
- short note about discounting

MAIN OBJECTS
- `states`: every place the agent can be
- `actions`: allowed actions from each non-terminal state
- `transitions`: the environment rule for where each `(current_state, action)` goes next
- `rewards`: reward from each `(state, action)`
- `terminal_states`: states where the episode is done, so the agent stops choosing actions
- `policy`: chosen action for each non-terminal state
- `values`: current estimate of how good each state is

METRICS TO TRACK
Track at least:
- final policy
- state values
- gamma
- number of improvement rounds

TARGET RESULT RANGE
Focus on understandable values and policy changes, not on benchmark scores.

Useful signs:
- values stop changing a lot after repeated evaluation updates
- the improved policy is at least as good as the starting one
- terminal states are handled cleanly

COMMON MISTAKES
- confusing a state value with an action value
- forgetting to stop improvement on terminal states
- using the wrong next-state value in the Bellman-style update
- treating gamma as a random scaling trick instead of future discounting
- building an environment so large that you cannot inspect it anymore

STEP-BY-STEP TODO
Complete the functions in this order:

1. `build_tiny_mdp`
   - write down the environment rules

2. `build_initial_policy`
   - choose one starting action for each non-terminal state

3. `initialize_state_values`
   - start every state value at `0.0`

4. `evaluate_policy`
   - keep the policy fixed and update values

5. `improve_policy`
   - use the values to choose better actions

6. `policy_changed`
   - check whether improvement changed any action

7. `explain_discount_factor`
   - explain `gamma` in plain words

8. `run_policy_iteration`
   - connect everything into the full loop

DEFINITION OF DONE
You are done when:
- the environment is clearly written down
- policy evaluation updates the values
- policy improvement chooses better actions
- the final policy is returned
- you can explain why future rewards are discounted
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypeAlias, TypedDict

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.training_checks import print_pass_or_try_again
# Learning order for this exercise:
# 1. Build the known environment rules with `build_tiny_mdp`.
# 2. Build a simple starting policy with `build_initial_policy`.
# 3. Start all state values at zero with `initialize_state_values`.
# 4. Evaluate the policy with `evaluate_policy`.
# 5. Improve the policy with `improve_policy`.
# 6. Connect the repeated loop in `run_policy_iteration`.
#
# The first helpers are small pieces of the full policy-iteration loop.
# Finish them in order so the larger loop has clear parts to call.

@dataclass
class DynamicProgrammingConfig:
    """Settings for the dynamic programming exercise."""
    gamma: float = 0.9
    evaluation_steps: int = 20
    improvement_rounds: int = 10


TransitionTable: TypeAlias = dict[tuple[str, str], str]
RewardTable: TypeAlias = dict[tuple[str, str], float]


class MdpTermSummary(TypedDict):
    """Plain-language MDP words used in this exercise."""

    gamma: float
    plain_words: dict[str, str]


class TinyMdp(TypedDict):
    """A tiny known MDP written as plain Python data."""

    states: list[str]
    actions: dict[str, list[str]]
    transitions: TransitionTable
    rewards: RewardTable
    terminal_states: set[str]


class DynamicProgrammingSummary(TypedDict):
    """Final summary returned by policy iteration."""

    final_policy: dict[str, str]
    state_values: dict[str, float]
    gamma: float
    num_improvement_rounds: int
    note: str


# Beginner hints for this file:
# - an MDP is a known environment written with states, actions, transitions, and rewards
# - policy evaluation asks "how good is each state if I keep following this policy?"
# - policy improvement asks "given those values, should I change any actions?"
# - gamma controls how much future reward matters
# - start with a tiny environment you can inspect by hand
def describe_mdp_terms(config: DynamicProgrammingConfig) -> MdpTermSummary:
    """Return the main words used in this exercise."""
    return {
        "gamma": config.gamma,
        "plain_words": {
            "state": "the current situation",
            "policy": "the action rule for each state",
            "value": "how good a state is under a policy",
            "dynamic_programming": "repeated small updates using known transition rules",
        },
    }

# Build one tiny deterministic MDP.
# Return a structure that includes at least:
# - states
# - actions
# - transitions
# - rewards
# - terminal states
def build_tiny_mdp() -> TinyMdp:
    """Return a tiny local MDP."""
    # Goal:
    # Return the full known environment that dynamic programming will solve.
    # "Known" means the code can look up every next state and reward.
    #
    # Where this fits:
    # This function creates the environment rules.
    # Later functions do not guess the rules.
    # They read `transitions` to find the next state and `rewards` to find the reward.
    # Keep this tiny so you can check the whole environment by eye.
    #
    # Input:
    # - No direct arguments.
    #
    # Output:
    # Return this exact dictionary:
    # - `"states"`: list[str], every state name
    # - `"actions"`: dict[str, list[str]], allowed actions for each non-terminal state
    # - `"transitions"`: dict[tuple[str, str], str], maps `(state, action)` to `next_state`
    # - `"rewards"`: dict[tuple[str, str], float], maps `(state, action)` to the immediate reward
    # - `"terminal_states"`: set[str], states where the episode is already done
    #
    # Tiny example shape:
    # ```python
    # {
    #     "states": ["start", "middle", "goal"],
    #     "actions": {"start": ["left", "right"], "middle": ["left", "right"]},
    #     "transitions": {
    #         ("start", "left"): "start",
    #         ("start", "right"): "middle",
    #         ("middle", "left"): "start",
    #         ("middle", "right"): "goal",
    #     },
    #     "rewards": {
    #         ("start", "left"): -0.1,
    #         ("start", "right"): 0.0,
    #         ("middle", "left"): 0.0,
    #         ("middle", "right"): 1.0,
    #     },
    #     "terminal_states": {"goal"},
    # }
    # ```
    #
    # Why `"actions"` is a dictionary:
    # `"actions"` shows the allowed actions from each state.
    # It is not just `"right"` because different states can have different allowed actions.
    #
    # Example:
    # ```python
    # "actions": {"start": ["left", "right"], "middle": ["left", "right"]}
    # ```
    #
    # means:
    # - from `"start"`, the agent is allowed to choose `"left"` or `"right"`
    # - from `"middle"`, the agent is allowed to choose `"left"` or `"right"`
    #
    # The first action in each list is the weak starting choice.
    # Policy improvement should learn to switch both states to `"right"`.
    #
    # How to read one transition:
    # `transitions[("start", "right")] = "middle"` means:
    # "If the current state is start and the chosen action is right,
    # the next state is middle."
    #
    # How to read one reward:
    # `rewards[("start", "right")] = 0.0` means:
    # "If the current state is start and the chosen action is right,
    # the environment gives reward 0.0."
    #
    # Method:
    # Use normal Python lists, dictionaries, tuples, strings, floats, and a set.
    # No PyTorch or external library is needed.
    #
    # Steps:
    # 1. Choose 3 to 5 state names.
    # 2. Choose at least one terminal state, such as `"goal"` or `"bad"`.
    # 3. Give each non-terminal state one or more allowed actions.
    # 4. For every allowed `(state, action)`, write one next state in `"transitions"`.
    # 5. For every allowed `(state, action)`, write one reward in `"rewards"`.
    # 6. Return the dictionary.
    #
    # Checks:
    # - Every `(state, action)` in `"transitions"` should also exist in `"rewards"`.
    # - Every next state should be in `"states"`.
    # - Terminal states should not need actions.

    raise NotImplementedError("Return states, actions, transitions, rewards, and terminal states.")


# Return one simple starting policy.
# A plain first version can choose one default action per non-terminal state.
def build_initial_policy(mdp: TinyMdp) -> dict[str, str]:
    """Return a simple starting policy."""
    # Goal:
    # Return one starting action for each non-terminal state.
    # The starting policy does not need to be good.
    #
    # Where this fits:
    # This creates the first action rule the agent will evaluate.
    # Policy evaluation asks, "How good is this exact rule?"
    # Policy improvement later asks, "Can we choose better actions than this?"
    #
    # Input:
    # - `mdp`: TinyMdp
    # - `mdp["actions"]` maps each non-terminal state to its allowed actions.
    # - Example: `{"start": ["left", "right"], "middle": ["right"]}`
    #
    # Output:
    # - `dict[str, str]`
    # - Keys: non-terminal state names.
    # - Values: one allowed action for that state.
    # - Example: `{"start": "left", "middle": "right"}`
    #
    # Method:
    # A simple choice is the first listed action:
    # `first_action = mdp["actions"][state][0]`
    #
    # Steps:
    # 1. Create an empty dictionary called `policy`.
    # 2. Loop over `mdp["actions"].items()`.
    # 3. For each state, choose one action from its action list.
    # 4. Store it as `policy[state] = chosen_action`.
    # 5. Return `policy`.
    #
    # Checks:
    # - `set(policy)` should equal `set(mdp["actions"])`.
    # - Every chosen action should be inside that state's action list.

    raise NotImplementedError("Return one starting action for each non-terminal state.")



# Return starting values for all states.
# A simple first version can use zero for every state.
def initialize_state_values(mdp: TinyMdp) -> dict[str, float]:
    """Return zero-initialized state values."""
    # Goal:
    # Return one starting value for every state.
    # A simple starting point is `0.0` for all states.
    #
    # Where this fits:
    # These values are the starting guesses for how good each state is.
    # Policy evaluation updates them using rewards and next-state values.
    # They are not rewards by themselves.
    #
    # Input:
    # - `mdp`: TinyMdp
    # - `mdp["states"]` is a list of state names.
    # - Example: `["start", "middle", "goal"]`
    #
    # Output:
    # - `dict[str, float]`
    # - One value per state.
    # - Example: `{"start": 0.0, "middle": 0.0, "goal": 0.0}`
    #
    # Method:
    # Use a dictionary comprehension:
    # `{state: 0.0 for state in mdp["states"]}`
    #
    # Steps:
    # 1. Loop over all states.
    # 2. Set each state value to `0.0`.
    # 3. Return the values dictionary.
    #
    # Checks:
    # - `set(values)` should equal `set(mdp["states"])`.
    # - Terminal state values can start at `0.0`.

    raise NotImplementedError("Return one starting value per state.")



# Evaluate the current policy.
# In plain words:
# - keep the policy fixed
# - update state values again and again
def evaluate_policy(
    mdp: TinyMdp,
    policy: dict[str, str],
    values: dict[str, float],
    gamma: float,
    num_steps: int,
) -> dict[str, float]:
    """Return updated state values for the current policy."""
    # Goal:
    # Return better state values for the current fixed policy.
    # Fixed policy means the action for each state does not change inside this function.
    #
    # Input:
    # - `mdp`: TinyMdp
    # - `policy`: dict[str, str]
    #   Example: `{"start": "right", "middle": "right"}`
    # - `values`: dict[str, float]
    #   Example: `{"start": 0.0, "middle": 0.0, "goal": 0.0}`
    # - `gamma`: float
    #   Future reward weight, usually between `0.0` and `1.0`.
    # - `num_steps`: int
    #   Number of repeated evaluation updates.
    #
    # Output:
    # - `dict[str, float]`
    # - Same keys as `values`.
    # - Values are updated estimates of how good each state is under `policy`.
    #
    # Formula for one non-terminal state:
    # ```text
    # new_value[state] = reward + gamma * old_value[next_state]
    # ```
    #
    # Meaning:
    # - `reward` is the immediate reward for taking the policy action.
    # - `gamma` says how much future value counts.
    # - `old_value[next_state]` is the value estimate from the previous pass.
    #
    # Method:
    # Use these lookups:
    # - `action = policy[state]`
    # - `next_state = mdp["transitions"][(state, action)]`
    # - `reward = mdp["rewards"][(state, action)]`
    #
    # Steps:
    # 1. Copy the starting values so you do not overwrite the input dictionary by accident.
    # 2. Repeat `num_steps` times.
    # 3. Inside each pass, create a fresh copy of the values.
    # 4. Loop over each non-terminal state in `policy`.
    # 5. Read the policy action, next state, and reward.
    # 6. Apply the formula above.
    # 7. Keep terminal states unchanged, usually at `0.0`.
    # 8. Replace the old values with the new values for the next pass.
    # 9. Return the final values.
    #
    # Checks:
    # - The returned dictionary should have one key for every state.
    # - Terminal states should not try to read `policy[state]`.
    # - With `gamma=0.0`, each non-terminal value should only use immediate reward.

    
    raise NotImplementedError("Return the policy-evaluated state values.")
    



# Improve the policy using the current values.
# For each state:
# - compare the value of each action
# - choose the best one
def improve_policy(
    mdp: TinyMdp,
    values: dict[str, float],
    gamma: float,
) -> dict[str, str]:
    """Return an improved greedy policy."""
    # Goal:
    # Return a greedy policy using the current state values.
    # Greedy means choose the action with the highest one-step lookahead value.
    #
    # Input:
    # - `mdp`: TinyMdp
    # - `values`: dict[str, float]
    # - `gamma`: float
    #
    # Output:
    # - `dict[str, str]`
    # - One chosen action per non-terminal state.
    # - Example: `{"start": "right", "middle": "right"}`
    #
    # Formula for one action:
    # ```text
    # action_score = reward + gamma * values[next_state]
    # ```
    #
    # Method:
    # For each action, use:
    # - `next_state = mdp["transitions"][(state, action)]`
    # - `reward = mdp["rewards"][(state, action)]`
    # - `score = reward + gamma * values[next_state]`
    # Then keep the action with the highest score.
    #
    # Steps:
    # 1. Create an empty dictionary called `new_policy`.
    # 2. Loop over `mdp["actions"].items()`.
    # 3. For each possible action, compute the action score with the formula above.
    # 4. Choose the action with the largest score.
    # 5. Store it as `new_policy[state] = best_action`.
    # 6. Return `new_policy`.
    #
    # Checks:
    # - Do not create policy entries for terminal states.
    # - Every chosen action should be allowed in `mdp["actions"][state]`.

    raise NotImplementedError("Return the better action for each non-terminal state.")



def policy_changed(old_policy: dict[str, str], new_policy: dict[str, str]) -> bool:
    """Return True when the policy changed."""
    # Goal:
    # Return whether any state changed its chosen action.
    #
    # Input:
    # - `old_policy`: dict[str, str]
    # - `new_policy`: dict[str, str]
    # - Example:
    #   `old_policy = {"start": "left"}`
    #   `new_policy = {"start": "right"}`
    #
    # Output:
    # - `bool`
    # - `True` if at least one action changed.
    # - `False` if the policies are the same.
    #
    # Method:
    # In Python, dictionaries compare by keys and values:
    # `old_policy != new_policy`
    #
    # Steps:
    # 1. Compare the two dictionaries.
    # 2. Return the boolean result.
    #
    # Checks:
    # - Same dictionaries should return `False`.
    # - Different actions for any state should return `True`.
    raise NotImplementedError("Return whether any policy action changed.")

# Run policy iteration:
# - evaluate policy
# - improve policy
# - stop when policy no longer changes or rounds are done
def run_policy_iteration(config: DynamicProgrammingConfig) -> DynamicProgrammingSummary:
    """Run the full dynamic programming workflow."""
    # Goal:
    # Run policy iteration until the policy stops changing or the max round limit is reached.
    #
    # Input:
    # - `config`: DynamicProgrammingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"final_policy"`: dict[str, str], chosen action per non-terminal state
    # - `"state_values"`: dict[str, float], final value estimate per state
    # - `"gamma"`: float, copied from `config.gamma`
    # - `"num_improvement_rounds"`: int, number of improvement rounds actually run
    # - `"note"`: str, short plain-language discount explanation
    #
    # Method:
    # Use the helpers in this file:
    # - `mdp = build_tiny_mdp()`
    # - `policy = build_initial_policy(mdp)`
    # - `values = initialize_state_values(mdp)`
    # - `values = evaluate_policy(mdp, policy, values, config.gamma, config.evaluation_steps)`
    # - `new_policy = improve_policy(mdp, values, config.gamma)`
    # - `changed = policy_changed(policy, new_policy)`
    # - `note = explain_discount_factor(config.gamma)`
    #
    # Steps:
    # 1. Build the MDP.
    # 2. Build the starting policy.
    # 3. Build starting state values.
    # 4. Loop up to `config.improvement_rounds`.
    # 5. Evaluate the current policy.
    # 6. Improve the policy from the evaluated values.
    # 7. If the policy did not change, stop early.
    # 8. Otherwise set `policy = new_policy` and continue.
    # 9. Return the summary dictionary.
    #
    # Checks:
    # - `final_policy` should not include terminal states.
    # - `state_values` should include every state.
    # - `num_improvement_rounds` should be at most `config.improvement_rounds`.

    raise NotImplementedError("Run policy evaluation and improvement until stable or until max rounds.")






def explain_discount_factor(gamma: float) -> str:
    """Explain the discount factor in simple words."""
    # Goal:
    # Write a short note about why future rewards are discounted.
    #
    # Input:
    # - `gamma`: float
    # - Example: `0.9`
    #
    # Output:
    # - `str`
    # - One or two plain sentences.
    #
    # Meaning:
    # `gamma` controls how much future reward counts now.
    # - `gamma = 0.0` means only the immediate reward matters.
    # - `gamma = 0.9` means a reward one step later counts as `0.9` of its value now.
    # - `gamma = 1.0` means future reward is not discounted in this tiny teaching setup.
    #
    # Tiny formula example:
    # ```text
    # current_value = reward_now + gamma * next_state_value
    # ```
    #
    # If `reward_now = 0`, `gamma = 0.9`, and `next_state_value = 10`,
    # then `current_value = 0 + 0.9 * 10 = 9`.
    #
    # Steps:
    # 1. Mention that future reward usually matters less than reward now.
    # 2. Mention the concrete `gamma` value.
    # 3. Keep the sentence short and plain.
    #
    # Checks:
    # - The returned text should explain `gamma` without using unexplained jargon.
    # - If you use a formula, define every part of it in words.
    raise NotImplementedError("Write a short note about why future rewards are discounted.")

def check_metrics(metrics: DynamicProgrammingSummary) -> bool:
    """Return True when the DP summary has the expected keys."""
    required = {"final_policy", "state_values", "gamma", "num_improvement_rounds", "note"}
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
    """Run the dynamic programming scaffold."""
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
    config = DynamicProgrammingConfig()
    _, ok = report_task_progress("describe_mdp_terms", describe_mdp_terms, config)
    if not count_or_stop(ok):
        return

    mdp, ok = report_task_progress("build_tiny_mdp", build_tiny_mdp)
    if not count_or_stop(ok):
        return

    policy, ok = report_task_progress("build_initial_policy", build_initial_policy, mdp)
    if not count_or_stop(ok):
        return

    values, ok = report_task_progress("initialize_state_values", initialize_state_values, mdp)
    if not count_or_stop(ok):
        return

    evaluated_values, ok = report_task_progress("evaluate_policy", evaluate_policy, mdp, policy, values, config.gamma, config.evaluation_steps)
    if not count_or_stop(ok):
        return

    improved_policy, ok = report_task_progress("improve_policy", improve_policy, mdp, evaluated_values, config.gamma)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("policy_changed", policy_changed, policy, improved_policy)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("explain_discount_factor", explain_discount_factor, config.gamma)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_policy_iteration", run_policy_iteration, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "dynamic programming summary", float(int(passed)), "dp keys present")
if __name__ == "__main__":
    main()
