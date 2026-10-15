"""
============================================================
EXERCISE 01: INTRO TO REINFORCEMENT LEARNING AND BANDITS
============================================================

WHAT YOU ARE BUILDING
You are building the smallest useful reinforcement learning example:
a tiny environment with 3 possible actions.

The formal RL name for this setup is a multi-armed bandit.
That name is not important for the first pass.
Think of it as a tiny environment.

The environment is like 3 slot machines.
Each slot machine is one action the agent can choose.
The agent does not know which action is best.
It has to try actions, get rewards, and update its guesses.

The environment has the hidden reward rules.
The agent does not get to see those rules.
The agent only sees the reward after it chooses an action.

So the roles are:
- the environment knows how good each action really is
- the agent starts with no idea which actions are good or bad
- exploration lets the agent try actions and collect evidence
- value estimates are the agent's guesses after seeing rewards

WHY THIS EXERCISE EXISTS
This file teaches one core RL loop:

1. The agent has guesses about actions.
2. The agent chooses one action.
3. The world gives one reward.
4. The agent updates the guess for that one action.
5. The loop repeats.

This is different from supervised learning.
In supervised learning, the data usually gives the correct label.
In RL, the agent acts first and only gets a reward afterward.

THE FULL PICTURE
There are two important lists in this file.

1. The hidden truth inside the environment:

```python
environment = [0.2, 0.5, 0.8]
```

This is the real reward chance for each action:
- action `0` gives reward about 20% of the time
- action `1` gives reward about 50% of the time
- action `2` gives reward about 80% of the time

The agent is not allowed to use `environment` to choose actions.
This list is the answer key.
It belongs to the environment, not to the agent.
The agent can only learn about it by trying actions.

2. The agent's current guesses:

```python
value_estimates = [0.0, 0.0, 0.0]
```

These are the agent's learned guesses.
They start at zero because the agent has not tried anything yet.
At the start, the agent does not know that action `2` is best.
To the agent, all actions look equally unknown.
The agent chooses actions using this list.
After it gets a reward, it updates only the chosen action's estimate.

The main loop is:

```python
action = choose_action(value_estimates, epsilon)
reward = sample_reward(environment[action])
action_counts[action] += 1
value_estimates[action] = update_action_value(
    old_estimate=value_estimates[action],
    reward=reward,
    action_count=action_counts[action],
)
```

Read that as:
- the agent uses its current guesses to choose an action
- the agent tries that action in the environment
- the environment gives back a reward, either `0.0` or `1.0`
- the agent records that it tried this action one more time
- the agent updates only this action's guess based on the reward

DATASET
There is no dataset file.
The tiny environment is created inside the code:

```python
action_probabilities = [0.2, 0.5, 0.8]
```

A reward is sampled from those probabilities.
For example, action `2` has probability `0.8`, so it often gives reward `1.0`,
but it can still sometimes give reward `0.0`.

INPUTS AND OUTPUTS
Input:
- `action_probabilities`: the hidden reward chances
- `epsilon`: how often to explore
- `num_steps`: how many actions to try

Output:
- `action_counts`: how often each action was chosen
- `value_estimates`: the learned guess for each action
- `mean_reward`: average reward over all steps
- `best_action_guess`: the action the agent thinks is best
- `true_best_action`: the real best action from the hidden truth

EXPLORATION VS EXPLOITATION
The action-choice rule is the policy.
In this exercise, the policy is inside `choose_action(...)`.

This policy is called epsilon-greedy.
It uses `epsilon` to decide whether to explore or exploit.

```python
if guess < epsilon:
    # explore
    action_index = random.randint(0, len(value_estimates) - 1)
else:
    # exploit
    max_val = max(value_estimates)
    action_index = value_estimates.index(max_val)
```

Explore means try a random action so the agent can learn more.
Exploit means use the best learned guess so far.

Important:
- `epsilon = 0.0` means never explore
- this can get stuck on the first tied action
- `epsilon = 1.0` means always explore
- this learns about all actions, but keeps choosing randomly
- `epsilon = 0.1` means usually exploit, sometimes explore

METRICS TO TRACK
Track at least:
- action counts
- value estimates
- mean reward
- best action guess

TARGET RESULT RANGE
Focus on behavior, not on one perfect number.

Useful signs:
- the best true arm is chosen more often over time
- estimated values move toward the true reward rates
- exploration still happens sometimes when epsilon is above zero

COMMON MISTAKES
- mixing up the true reward probabilities and the learned value estimates
- never exploring, so the agent gets stuck with an early guess
- exploring forever and never using what was learned
- updating the estimate for the wrong arm
- forgetting that this tiny environment has no next state

STEP-BY-STEP TODO
Complete the functions in this order:

1. `build_bandit`
   - create the environment's hidden reward rules, such as `[0.2, 0.5, 0.8]`

2. `initialize_agent_state`
   - create the agent's starting guesses and counts
   - this returns `value_estimates` and `action_counts`

3. `choose_action`
   - choose either a random action or the best current guess
   - this is where explore vs exploit happens

4. `sample_reward`
   - use the environment's hidden rules to return `1.0` or `0.0`

5. `update_action_value`
   - update the running average for one chosen action

6. `run_bandit_experiment`
   - connect all helper functions into the full learning loop

7. `explain_exploration_vs_exploitation`
   - explain the result in plain English

DEFINITION OF DONE
You are done when:
- the environment has several actions
- epsilon-greedy picks random actions sometimes
- value estimates update over time
- action counts are tracked
- you can explain why exploration is needed
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TypedDict

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.training_checks import print_pass_or_try_again

# Learning order for this exercise:
# 1. Build the tiny environment with `build_bandit`.
# 2. Build the agent's empty memory with `initialize_agent_state`.
# 3. Write the action choice rule with `choose_action`.
# 4. Write the reward sampler with `sample_reward`.
# 5. Write the one-action update rule with `update_action_value`.
# 6. Connect everything in `run_bandit_experiment`.
#
# The first functions are small on purpose.
# They are the pieces that make the full loop easier to understand.

@dataclass
class BanditConfig:
    """Settings for the tiny environment exercise."""
    action_probabilities: tuple[float, float, float] = (0.2, 0.5, 0.8)
    epsilon: float = 0.1
    num_steps: int = 200
    random_state: int = 42


class BanditTermSummary(TypedDict):
    """Plain-language RL words used in this exercise."""

    num_actions: int
    epsilon: float
    plain_words: dict[str, str]


class BanditAgentState(TypedDict):
    """The values the agent tracks while learning."""

    value_estimates: list[float]
    action_counts: list[int]


class BanditRunSummary(TypedDict):
    """Final summary returned by the environment experiment."""

    action_counts: list[int]
    value_estimates: list[float]
    mean_reward: float
    best_action_guess: int
    true_best_action: int
    note: str


# Beginner hints for this file:
# - this environment has actions and rewards, but no state transitions
# - epsilon-greedy means "usually use the best guess so far, but sometimes explore"
# - your value estimate can start at zero for every arm
# - keep the first version tiny so you can print all important numbers
# - if one action is truly best, the agent should slowly favor it more often
def describe_bandit_terms(config: BanditConfig) -> BanditTermSummary:
    """Return the main RL words used in this first exercise."""
    return {
        "num_actions": len(config.action_probabilities),
        "epsilon": config.epsilon,
        "plain_words": {
            "action": "the choice the agent makes",
            "reward": "the score returned after that choice",
            "exploration": "trying actions to learn more",
            "exploitation": "using the action that looks best so far",
        },
    }

# Build the environment.
# A simple first version can return:
# - one reward probability per action
def build_bandit(config: BanditConfig) -> list[float]:
    """Return the reward probability for each action."""
    # Goal:
    # Return the hidden reward probability for each action.
    # Hidden means the agent should learn these values from rewards, not read them during action choice.
    #
    # Where this fits:
    # This function creates the tiny environment.
    # It is the answer key, not the agent's memory.
    # Later, `sample_reward(environment[action])` uses this list to decide whether the agent gets reward.
    # `choose_action(...)` should not use this list.
    #
    # Input:
    # - `config`: BanditConfig
    #
    # Output:
    # - `list[float]`
    # - Length: `len(config.action_probabilities)`
    # - Example: `[0.2, 0.5, 0.8]`
    #
    # Method:
    # Convert the tuple in the config to a list:
    # `list(config.action_probabilities)`
    #
    # Steps:
    # 1. Read `config.action_probabilities`.
    # 2. Convert it to a list.
    # 3. Return the list.
    #
    # Checks:
    # - Every probability should be between `0.0` and `1.0`.
    # - For the default config, action `2` is the true best action.
    # - There is no state and no next state in this tiny environment.

    action_prob = list(config.action_probabilities)

    return action_prob
    #raise NotImplementedError("Return the local environment reward probabilities.")

# Return initial value estimates and action counts.
# Keep the first version simple:
# - one value estimate per action
# - one count per action
def initialize_agent_state(config: BanditConfig) -> BanditAgentState:
    """Return starting action values and counts."""
    # Goal:
    # Return the starting value estimates and action counts.
    # At the start, the agent has not tried any action.
    # `value_estimates` is the agent's belief, not the hidden truth.
    # It starts at zero for every action because the agent has no reward evidence yet.
    # Later, each value becomes a running average of the rewards seen for that action.
    #
    # Where this fits:
    # This function creates the agent's memory.
    # `value_estimates` says how good each action looks to the agent right now.
    # `action_counts` says how many times each action has been tried.
    # The learning loop updates these two lists after each reward.
    #
    # Input:
    # - `config`: BanditConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"value_estimates"`: list of floats, one per action
    # - `"action_counts"`: list of ints, one per action
    #
    # Example for 3 actions:
    # `{"value_estimates": [0.0, 0.0, 0.0], "action_counts": [0, 0, 0]}`
    #
    # Method:
    # Use:
    # `num_actions = len(config.action_probabilities)`
    #
    # Steps:
    # 1. Count the number of actions.
    # 2. Create `[0.0] * num_actions` for value estimates.
    # 3. Create `[0] * num_actions` for action counts.
    # 4. Return the dictionary.
    #
    # Checks:
    # - Both lists should have length `num_actions`.
    # - All counts should start at `0`.
    # - The agent should not know which action is truly best yet.

    num_actions = len(config.action_probabilities)
    value_estimates = [0.0] * num_actions
    action_counts = [0] * num_actions

    return {"value_estimates": value_estimates, "action_counts": action_counts}

    #raise NotImplementedError("Return zero-filled value estimates and action counts.")

# Choose one action with epsilon-greedy.
# In plain words:
# - with probability epsilon, choose a random action
# - otherwise choose the action with the highest estimated value
def choose_action(
    value_estimates: list[float],
    epsilon: float,
) -> int:
    """Choose one action with epsilon-greedy."""
    # Goal:
    # Return one action index using epsilon-greedy.
    # Epsilon-greedy means sometimes explore and usually use the current best guess.
    #
    # Input:
    # - `value_estimates`: list[float]
    #   Example: `[0.1, 0.4, 0.3]`
    # - `epsilon`: float
    #   Example: `0.1`
    #
    # Output:
    # - `int`
    # - The action index to pull.
    # - For 3 actions, valid actions are `0`, `1`, and `2`.
    #
    # Method:
    # Use:
    # - `random.random()` to decide whether to explore
    # - `random.randrange(len(value_estimates))` for a random action
    # - `max(range(len(value_estimates)), key=lambda action: value_estimates[action])` for the current best action
    #
    # Rule:
    # - if `random.random() < epsilon`, choose a random action
    # - otherwise choose the action with the highest estimate
    #
    # Which line is exploration?
    # - The `if guess < epsilon:` branch is exploration.
    # - It returns a random action with `random.randint(...)`.
    # - Random means the agent may try an action that does not look best yet.
    #
    # Which line is exploitation?
    # - The `else:` branch is exploitation.
    # - `max(value_estimates)` finds the highest learned estimate.
    # - `value_estimates.index(max_val)` returns the action index for that estimate.
    # - This means the agent uses what it has learned so far.
    #
    # Steps:
    # 1. Count actions with `len(value_estimates)`.
    # 2. Draw a random number with `random.random()`.
    # 3. If the random number is below epsilon, return a random action.
    # 4. Otherwise return the action with the highest value estimate.
    #
    # Tie note:
    # At the start all estimates may be equal, like `[0.0, 0.0, 0.0]`.
    # The `max(...)` call above will pick the first tied action.
    # That is okay here because epsilon still gives the agent chances to try other actions.
    #
    # Checks:
    # - With `epsilon=0.0`, the function should always exploit.
    # - With `epsilon=1.0`, the function should always explore.
    # - The return value should be an integer between `0` and `len(value_estimates) - 1`.

    action_index = 0

    guess = random.random()
    if guess < epsilon:
        # Explore: pick a random action to learn more about it.
        action_index = random.randint(0, len(value_estimates)-1)
    else:
        # Exploit: pick the action with the best learned estimate so far.
        max_val = max(value_estimates)
        action_index = value_estimates.index(max_val)

    return action_index
    #raise NotImplementedError("Return one action index using epsilon-greedy.")

# Sample one reward from the chosen arm.
# A simple first version can return:
# - `1.0` for success
# - `0.0` for failure
def sample_reward(action_probability: float) -> float:
    """Sample one reward from one environment action."""
    # Goal:
    # Return one sampled reward for the chosen action.
    # The reward is `1.0` for success and `0.0` for failure.
    #
    # Input:
    # - `action_probability`: float
    #   This is the chance of getting reward `1.0`.
    #
    # Output:
    # - `float`
    # - Either `1.0` or `0.0`.
    #
    # Method:
    # Use:
    # `random.random() < action_probability`
    #
    # Example:
    # - if `action_probability = 0.8`, reward `1.0` should happen often
    # - if `action_probability = 0.2`, reward `1.0` should happen less often
    #
    # Steps:
    # 1. Draw a random number with `random.random()`.
    # 2. If it is less than `action_probability`, return `1.0`.
    # 3. Otherwise return `0.0`.
    #
    # Checks:
    # - The return value should always be `0.0` or `1.0`.
    # - This function should not update value estimates.

    guess = random.random()

    if guess < action_probability:
        return 1.0
    else:
        return 0.0

    #raise NotImplementedError("Return one sampled reward for the chosen arm.")

# Update the estimate of the chosen action.
# Use the running-average update.
# In plain words:
# - if this action has been tried many times already,
#   one new reward should change the estimate only a little
def update_action_value(
    old_estimate: float,
    reward: float,
    action_count: int,
) -> float:
    """Return the updated action value estimate."""
    # Goal:
    # Return the new value estimate for the chosen action.
    # Only the chosen action should be updated.
    #
    # Input:
    # - `old_estimate`: float
    #   Current estimate for this action before the new reward.
    # - `reward`: float
    #   The new reward, usually `0.0` or `1.0`.
    # - `action_count`: int
    #   How many times this action has been chosen after adding this new try.
    #
    # Output:
    # - `float`
    # - Updated running-average estimate.
    #
    # Formula:
    # ```text
    # new_estimate = old_estimate + (reward - old_estimate) / action_count
    # ```
    #
    # Meaning:
    # - `reward - old_estimate` is the new surprise.
    # - dividing by `action_count` makes each new reward matter less after many tries.
    # - `action_count` means the count after adding the new action, not before it.
    # - the estimate is not a sum of old estimates.
    # - it is one current running-average estimate.
    # - `old_estimate` is the previous estimate from the last time this action was updated.
    #
    # Tiny example:
    # Suppose action 2 has rewards `[1.0, 0.0]`.
    # After the first reward, the estimate is `1.0`.
    # After the second reward, use:
    # `1.0 + (0.0 - 1.0) / 2 = 0.5`
    # So the estimate becomes the average of the two rewards.
    #
    # Steps:
    # 1. Use the formula above.
    # 2. Return the new estimate.
    #
    # Checks:
    # - If `old_estimate=0.0`, `reward=1.0`, and `action_count=1`, result should be `1.0`.
    # - If `old_estimate=0.5`, `reward=1.0`, and `action_count=2`, result should be `0.75`.

    new_estimate = old_estimate + (reward - old_estimate) / action_count
    return new_estimate
    #raise NotImplementedError("Return the running-average value update.")

# Run the full environment experiment.
# Return a summary with at least:
# - action counts
# - value estimates
# - mean reward
# - best action guess
def run_bandit_experiment(config: BanditConfig) -> BanditRunSummary:
    """Run the full environment learning loop."""
    # Goal:
    # Run the environment loop from start to finish.
    # The agent should learn from sampled rewards, not from reading the true probabilities.
    #
    # Input:
    # - `config`: BanditConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"action_counts"`: list[int], one count per action
    # - `"value_estimates"`: list[float], one learned value per action
    # - `"mean_reward"`: float, total reward divided by `config.num_steps`
    # - `"best_action_guess"`: int, action with the highest learned estimate
    # - `"true_best_action"`: int, action with the highest hidden reward probability
    # - `"note"`: str, short plain-language explanation
    #
    # Method:
    # Use the helper functions in this file:
    # - `build_bandit(config)`
    # - `initialize_agent_state(config)`
    # - `choose_action(state["value_estimates"], config.epsilon)`
    # - `sample_reward(environment[action])`
    # - `update_action_value(old_estimate, reward, action_count)`
    #
    # Formula:
    # ```text
    # mean_reward = total_reward / config.num_steps
    # ```
    #
    # Important:
    # `choose_action` should only see learned value estimates.
    # It should not see `environment`, because `environment` contains the hidden true probabilities.
    #
    # Steps:
    # 1. Build the hidden environment with `environment = build_bandit(config)`.
    # 2. Create the agent state with `state = initialize_agent_state(config)`.
    # 3. Start `total_reward = 0.0`.
    # 4. Loop for `config.num_steps`.
    # 5. Pick an action with `choose_action(state["value_estimates"], config.epsilon)`.
    # 6. Sample a reward with `sample_reward(environment[action])`.
    # 7. Add 1 to `state["action_counts"][action]`.
    # 8. Update only `state["value_estimates"][action]` with `update_action_value(...)`.
    # 9. Add the reward to `total_reward`.
    # 10. After the loop, compute `mean_reward`.
    # 11. Find the learned best action with
    #     `max(range(len(state["value_estimates"])), key=lambda action: state["value_estimates"][action])`.
    # 12. Find the true best action with
    #     `max(range(len(environment)), key=lambda action: environment[action])`.
    # 13. Build the summary dictionary and add a short `note`.
    #
    # Checks:
    # - `sum(action_counts)` should equal `config.num_steps`.
    # - `len(value_estimates)` should equal `len(config.action_probabilities)`.
    # - `mean_reward` should be between `0.0` and `1.0`.
    # - In this tiny environment, there is no next state.

    random.seed(config.random_state)

    environment = build_bandit(config) # This is like the hidden rules of the world that the agent does not know yet.

    state_dict = initialize_agent_state(config)

    total_reward = 0.0

    for i in range(config.num_steps):
        action_index = choose_action(state_dict["value_estimates"], config.epsilon)

        reward = sample_reward(environment[action_index])
        state_dict["action_counts"][action_index]+=1        

        old_estimate = state_dict["value_estimates"][action_index]
        action_count = state_dict["action_counts"][action_index]

        new_estimate = update_action_value(old_estimate=old_estimate, reward=reward, action_count=action_count)
        state_dict["value_estimates"][action_index] = new_estimate

        total_reward += reward

    mean_reward = total_reward / config.num_steps

    learned_best_action = max(range(len(state_dict["value_estimates"])), key=lambda action_id: state_dict["value_estimates"][action_id])
    true_best_action = max(range(len(environment)), key=lambda action_id: environment[action_id])

    print("Environment run summary:")
    print(f"  action counts: {state_dict['action_counts']}")
    print(f"  value estimates: {[round(value, 3) for value in state_dict['value_estimates']]}")
    print(f"  total reward: {total_reward:.3f}")
    print(f"  mean reward: {mean_reward:.3f}")
    print(f"  learned best action: {learned_best_action}")
    print(f"  true best action: {true_best_action}")

    return {
        "action_counts": state_dict["action_counts"],
        "value_estimates": state_dict["value_estimates"],
        "mean_reward": mean_reward,
        "best_action_guess": learned_best_action,
        "true_best_action": true_best_action,
        "note": "The agent used rewards from the environment to learn which action looked best.",
    }

    #raise NotImplementedError("Run the environment loop and return a summary.")

def explain_exploration_vs_exploitation(summary: BanditRunSummary) -> str:
    """Write a short note about exploration and exploitation."""
    # Goal:
    # Explain the tradeoff in simple words using the run summary.
    #
    # Input:
    # - `summary`: BanditRunSummary
    # - Example:
    #   `{"action_counts": [12, 30, 158], "value_estimates": [0.1, 0.4, 0.78],
    #     "mean_reward": 0.66, "best_action_guess": 2, "true_best_action": 2, "note": ""}`
    #
    # Output:
    # - `str`
    # - One or two plain sentences.
    # - Mention that exploration tries actions to learn and exploitation uses the best current estimate.
    #
    # Method:
    # Use values from the summary:
    # - `summary["best_action_guess"]`
    # - `summary["true_best_action"]`
    # - `summary["action_counts"]`
    #
    # Steps:
    # 1. Read the learned best action from `summary["best_action_guess"]`.
    # 2. Read the true best action from `summary["true_best_action"]`.
    # 3. Optionally find the most chosen action from `summary["action_counts"]`.
    # 4. Write one or two sentences about what happened.
    #
    # Checks:
    # - Keep the note short and plain.
    # - Do not say the agent used the hidden true probabilities to choose actions.
    # - Say "no next state" if you compare this to full RL.
    learned_best_action = summary["best_action_guess"]
    true_best_action = summary["true_best_action"]
    action_counts = summary["action_counts"]
    most_chosen_action = max(range(len(action_counts)), key=lambda action: action_counts[action])

    return (
        f"Exploration tried different actions so the agent could learn from rewards. "
        f"Exploitation used the best current estimate, so action {most_chosen_action} was chosen most often; "
        f"the learned best action was {learned_best_action}, and the true best action was {true_best_action}."
    )

    #raise NotImplementedError("Explain the tradeoff in simple words using the run summary.")

def check_metrics(metrics: BanditRunSummary) -> bool:
    """Return True when the environment summary has the expected keys."""
    required = {
        "action_counts",
        "value_estimates",
        "mean_reward",
        "best_action_guess",
        "true_best_action",
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
    """Run the environment scaffold."""
    config = BanditConfig()
    random.seed(config.random_state)
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
    _, ok = report_task_progress("describe_bandit_terms", describe_bandit_terms, config)
    if not count_or_stop(ok):
        return

    environment, ok = report_task_progress("build_bandit", build_bandit, config)
    if not count_or_stop(ok):
        return

    agent_state, ok = report_task_progress("initialize_agent_state", initialize_agent_state, config)
    if not count_or_stop(ok):
        return

    action, ok = report_task_progress("choose_action", choose_action, agent_state["value_estimates"], config.epsilon)
    if not count_or_stop(ok):
        return

    reward, ok = report_task_progress("sample_reward", sample_reward, environment[action])
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("update_action_value", update_action_value, 0.0, reward, 1)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_bandit_experiment", run_bandit_experiment, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "environment summary", float(int(passed)), "environment keys present")

    if not ok or metrics is None:
        return

    _, ok = report_task_progress("explain_exploration_vs_exploitation", explain_exploration_vs_exploitation, metrics)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
if __name__ == "__main__":
    main()
