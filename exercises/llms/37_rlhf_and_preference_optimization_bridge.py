"""
============================================================
EXERCISE 37: RLHF AND PREFERENCE OPTIMIZATION BRIDGE
============================================================

WHAT YOU ARE BUILDING
You are building a tiny bridge between standard RL ideas and preference optimization for language models.

Plain-language meanings:
- RLHF means reinforcement learning from human feedback
- a reward model scores answers based on feedback
- a reference policy is the older policy you do not want to drift too far from
- policy ratio compares how much the new policy changed
- DPO means direct preference optimization
- DPO is a preference method that can be understood as a simpler bridge from preferred answers

WHY THIS EXERCISE EXISTS
A lot of modern LLM post-training uses ideas that connect back to RL.
If a beginner already learned bandits, policy gradients, and PPO basics,
they should now see how those ideas connect to preference learning.
This exercise teaches:
- tiny preference dataset
- tiny reward scores
- policy log-probabilities
- PPO-style policy-ratio intuition
- reference-policy anchoring
- plain-language comparison between RLHF-style PPO and DPO-style preference updates

HOW THIS FITS INTO THE RL SEQUENCE
This is the last file in the beginner RL track.
It connects the earlier RL ideas to language-model post-training.
In simple words:
- bandits and policy gradients taught reward-based updates
- PPO taught clipped policy updates
- this file shows how those ideas appear again in RLHF-style training and preference optimization

Beginner intuition:
- in normal RL, reward often comes from the environment
- in RLHF, reward can come from human preference signals
- the policy still changes to get higher reward
- but for language models, we also care about not drifting too far from the original policy

Small example:
- prompt: `Be helpful.`
- answer A is polite and correct
- answer B is rude or wrong
- preference data says A is better
- a reward model gives A a higher score
- policy optimization should make A-style answers more likely

IMPORTANT SCOPE NOTE
This is a small teaching bridge.
It is not a production RLHF pipeline.
The goal is to connect ideas clearly:
- reward
- policy update
- reference policy
- preference learning

DATASET
Use a tiny local preference dataset.

Clear default:
- 6 to 10 beginner-helpfulness preference examples written inside this file
- prompts should ask for simple ML or PyTorch help
- chosen answers should be clearer, safer, or more useful
- rejected answers should be vague, wrong, unsafe, or unhelpful

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- write a few prompt, chosen, rejected examples
- keep them tiny and readable
- make the preference difference obvious to a beginner
- use toy scores so the main ideas stay visible

INPUTS AND OUTPUTS
Input:
- prompt
- chosen answer
- rejected answer
- toy reward scores
- toy current-policy and reference-policy log-probabilities
Output:
- chosen and rejected reward summary
- policy-ratio comparison
- PPO-style objective term
- short bridge note
- saved report path

METRICS TO TRACK
Track at least:
- number of examples
- mean chosen reward
- mean rejected reward
- report path
- bridge note

TARGET RESULT RANGE
Focus on concept connection, not on training a strong language model.

Useful signs:
- chosen answers get higher reward than rejected answers
- current and reference policy scores can both be compared
- the bridge note explains reward, preference, and policy update in one clear story

COMMON MISTAKES
- mixing up chosen and rejected answers
- treating the toy reward model like a real production reward model
- forgetting why a reference policy matters
- trying to turn this bridge exercise into a full RLHF implementation
- missing the main point that DPO and RLHF-style PPO are related through preference-improving policy updates

What each step is doing in simple words:
1. build tiny preference data that clearly shows better and worse answers
2. score those answers with a toy reward signal
3. compare how the current and reference policies score the same answers
4. turn that comparison into a PPO-style update intuition
5. compare that reward-based view with a DPO-style preference view
6. write one plain-language explanation that connects the whole path

STEP-BY-STEP TODO
1. Build a tiny prompt, chosen, rejected dataset.
2. Build or define tiny reward scores.
3. Build a tiny policy scoring function.
4. Compute log-probabilities under current and reference policies.
5. Compute a tiny PPO-style objective from reward and policy ratio.
6. Compare that idea with a DPO-style preference view.
7. Write a short bridge note in plain words.

DEFINITION OF DONE
You are done when:
- chosen and rejected answers are clearly separated
- reward scores are computed
- current and reference policy scores are compared
- a PPO-style policy objective is written
- you can explain how RLHF-style PPO and DPO are related at a high level
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict
import torch

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
class RLHFBridgeConfig:
    """Settings for the RLHF bridge exercise."""
    beta: float = 0.1
    clip_epsilon: float = 0.2
    report_path: str = str(BOOTCAMP_ROOT / "reports" / "81_rlhf_and_preference_optimization_bridge.md")


class RLHFBridgeTerms(TypedDict):
    """Main RLHF bridge words used in this exercise."""

    beta: float
    clip_epsilon: float
    plain_words: dict[str, str]


class PreferenceExample(TypedDict):
    """One preference example with a better and worse answer."""

    prompt: str
    chosen: str
    rejected: str


class RLHFDPOToyComparison(TypedDict):
    """Tiny comparison between RLHF-style PPO and DPO-style views."""

    chosen_reward: float
    rejected_reward: float
    reward_advantage: float
    chosen_ratio: float
    ppo_objective: float
    dpo_preference_gap: float
    note: str


class RLHFBridgeSummary(TypedDict):
    """Final summary returned by the RLHF bridge demo."""

    num_examples: int
    mean_chosen_reward: float
    mean_rejected_reward: float
    report_path: str
    bridge_note: str

# Beginner hints for this file:
# - keep the examples tiny and readable so the preference signal is obvious
# - the reward model is a teaching stand-in here, not a real large model
# - the reference policy is important because post-training often should not drift too far
# - think in terms of "make better answers more likely"
# - the goal is understanding the bridge, not building a full RLHF stack
def describe_rlhf_bridge_terms(config: RLHFBridgeConfig) -> RLHFBridgeTerms:
    """Return the main RLHF bridge words used in this exercise."""
    return {
        "beta": config.beta,
        "clip_epsilon": config.clip_epsilon,
        "plain_words": {
            "reward_model": "a model or scoring rule that gives higher scores to better answers",
            "reference_policy": "the older policy used as a stay-close anchor",
            "dpo": "a preference-learning method that pushes chosen answers above rejected ones",
        },
    }

# Build a tiny preference dataset.
# Each example should include:
# - prompt
# - chosen
# - rejected
def build_preference_dataset() -> list[PreferenceExample]:
    """Return a tiny local preference dataset."""
    # Goal:
    # Return small prompt, chosen, rejected examples.
    # The chosen answer should clearly be better than the rejected answer.
    #
    # Input:
    # - No direct arguments.
    #
    # Output:
    # - `list[PreferenceExample]`
    # - Include 6 to 10 examples.
    #
    # Each example must have these exact keys:
    # - `"prompt"`: str
    # - `"chosen"`: str, the better answer
    # - `"rejected"`: str, the worse answer
    #
    # Small example:
    # ```python
    # {
    #     "prompt": "How do I check a tensor shape in PyTorch?",
    #     "chosen": "Use `x.shape`. It tells you the size of each tensor dimension.",
    #     "rejected": "Just run it and hope it works.",
    # }
    # ```
    #
    # Method:
    # Write the dataset by hand as a Python list.
    # Use short prompts about ML, PyTorch, or debugging.
    # Make chosen answers clear, safe, and useful.
    # Make rejected answers vague, wrong, or unhelpful.
    #
    # Steps:
    # 1. Create a list of dictionaries.
    # 2. Add at least 6 examples.
    # 3. Make sure every dictionary has `prompt`, `chosen`, and `rejected`.
    # 4. Return the list.
    #
    # Checks:
    # - `len(examples)` should be at least `6`.
    # - `examples[0].keys()` should include `prompt`, `chosen`, and `rejected`.
    # - Do not make the rejected answer better than the chosen answer.
    raise NotImplementedError("Return prompt, chosen, rejected examples.")

# Return a tiny reward score.
# The simplest path is:
# - higher score for clearly better answers
# - lower score for worse answers
def reward_model_score(example: PreferenceExample, answer_key: str) -> float:
    """Return a tiny reward score for one answer."""
    # Goal:
    # Return a toy reward score for either the chosen or rejected answer.
    # This is a teaching score, not a real reward model.
    #
    # Input:
    # - `example`: PreferenceExample
    # - `answer_key`: str
    #   Use `"chosen"` or `"rejected"`.
    #
    # Output:
    # - `float`
    # - Higher means the answer looks better.
    #
    # Simple scoring rule:
    # - start chosen answers at `1.0`
    # - start rejected answers at `0.0`
    # - add `0.2` if the answer includes a concrete code word such as `shape`, `tensor`, `loss`, or `gradient`
    # - subtract `0.2` if the answer includes an unhelpful phrase such as `hope`, `whatever`, or `ignore`
    #
    # Method:
    # Use:
    # - `answer = example[answer_key].lower()`
    # - simple `if ... in answer` checks
    #
    # Steps:
    # 1. Check that `answer_key` is `"chosen"` or `"rejected"`.
    # 2. Read and lowercase the answer text.
    # 3. Start the score based on whether the key is chosen or rejected.
    # 4. Add or subtract small bonuses from the simple rule above.
    # 5. Return the score as a float.
    #
    # Checks:
    # - For most examples, `reward_model_score(example, "chosen")` should be higher than rejected.
    # - This function should not train a model.

    raise NotImplementedError("Return a reward score for the chosen or rejected answer.")






# Return a tiny policy log-probability.
# This is a teaching stand-in for a language model log-probability.
def policy_log_probability(example: PreferenceExample, answer_key: str, policy_name: str) -> float:
    """Return a tiny current-policy or reference-policy log-probability."""
    # Goal:
    # Return a toy log-probability for one answer under one policy.
    # This stands in for what a language model would return.
    #
    # Input:
    # - `example`: PreferenceExample
    # - `answer_key`: str
    #   Use `"chosen"` or `"rejected"`.
    # - `policy_name`: str
    #   Use `"new"` or `"reference"`.
    #
    # Output:
    # - `float`
    # - Log-probabilities are usually negative.
    # - A less negative value means the policy thinks the answer is more likely.
    #
    # Small example:
    # - `-0.4` is more likely than `-1.2`
    #
    # Simple rule:
    # - reference policy: chosen `-0.8`, rejected `-0.9`
    # - new policy: chosen `-0.4`, rejected `-1.2`
    #
    # This means the new policy already prefers the chosen answer more strongly.
    #
    # Method:
    # Use a small nested dictionary:
    # ```python
    # scores = {
    #     "reference": {"chosen": -0.8, "rejected": -0.9},
    #     "new": {"chosen": -0.4, "rejected": -1.2},
    # }
    # ```
    #
    # Steps:
    # 1. Check that `policy_name` is `"new"` or `"reference"`.
    # 2. Check that `answer_key` is `"chosen"` or `"rejected"`.
    # 3. Return the matching score from the dictionary.
    #
    # Checks:
    # - The new chosen log-probability should be higher than the reference chosen log-probability.
    # - The return value should be a Python float.
    raise NotImplementedError("Return a toy log-probability for one answer under one policy.")



def policy_ratio(new_log_prob: float, old_log_prob: float) -> float:
    """Return the PPO-style policy ratio."""
    # Goal:
    # Return how much more likely the new policy made the same answer.
    #
    # Input:
    # - `new_log_prob`: float
    # - `old_log_prob`: float
    #
    # Output:
    # - `float`
    #
    # Formula:
    # ```text
    # ratio = exp(new_log_prob - old_log_prob)
    # ```
    #
    # Meaning:
    # - ratio above `1.0` means the new policy made the answer more likely
    # - ratio below `1.0` means the new policy made the answer less likely
    #
    # Method:
    # Use PyTorch:
    # - `torch.exp(torch.tensor(new_log_prob - old_log_prob)).item()`
    #
    # `.item()` converts a one-value tensor into a Python float.
    #
    # Steps:
    # 1. Subtract old log-probability from new log-probability.
    # 2. Apply exponential with `torch.exp(...)`.
    # 3. Return the result as a float.
    #
    # Checks:
    # - If both log-probabilities are equal, ratio should be `1.0`.
    # - If `new_log_prob` is greater than `old_log_prob`, ratio should be greater than `1.0`.

    # Since log(a / b) = log(a) - log(b), exp(new_log_prob - old_log_prob)
    # gives the same idea as new_probability / old_probability.
    raise NotImplementedError("Return exp(new_log_prob - old_log_prob).")

# Return a tiny PPO-style objective term.
# In plain words:
# - reward says which answer looks better
# - policy ratio says how much the new policy changed
# - clipping keeps the update from becoming too large
def clipped_policy_objective(ratio: float, reward_advantage: float, clip_epsilon: float) -> float:
    """Return a tiny clipped PPO-style objective term."""
    # Goal:
    # Return a tiny PPO-style clipped objective term.
    # This shows how PPO limits a policy update when the ratio gets too large.
    #
    # Input:
    # - `ratio`: float
    # - `reward_advantage`: float
    #   Positive means the answer was better than expected.
    # - `clip_epsilon`: float
    #   Example: `0.2`
    #
    # Output:
    # - `float`
    #
    # Formula:
    # ```text
    # unclipped = ratio * reward_advantage
    # clipped_ratio = min(max(ratio, 1 - clip_epsilon), 1 + clip_epsilon)
    # clipped = clipped_ratio * reward_advantage
    # objective = min(unclipped, clipped)
    # ```
    #
    # Meaning:
    # - `ratio` compares new and old policy probabilities.
    # - clipping keeps the ratio in a small range, such as `0.8` to `1.2`.
    #
    # Method:
    # Use Python `min(...)` and `max(...)`.
    #
    # Steps:
    # 1. Compute the unclipped objective.
    # 2. Clip the ratio to `[1 - clip_epsilon, 1 + clip_epsilon]`.
    # 3. Compute the clipped objective.
    # 4. Return the smaller of the two objective values.
    #
    # Checks:
    # - With `ratio=2.0`, `reward_advantage=1.0`, and `clip_epsilon=0.2`, result should be `1.2`.
    # - With `ratio=1.0`, clipping should not change the objective.

    raise NotImplementedError("Return the clipped objective term.")


def compare_rlhf_style_and_dpo(example: PreferenceExample, config: RLHFBridgeConfig) -> RLHFDPOToyComparison:
    """Return a tiny comparison between RLHF-style PPO intuition and DPO intuition."""
    # Goal:
    # Return numbers and a short note that compare RLHF-style PPO and DPO-style thinking.
    #
    # Input:
    # - `example`: PreferenceExample
    # - `config`: RLHFBridgeConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"chosen_reward"`: float
    # - `"rejected_reward"`: float
    # - `"reward_advantage"`: float
    # - `"chosen_ratio"`: float
    # - `"ppo_objective"`: float
    # - `"dpo_preference_gap"`: float
    # - `"note"`: str
    #
    # Formula:
    # ```text
    # reward_advantage = chosen_reward - rejected_reward
    # chosen_ratio = exp(new_chosen_log_prob - reference_chosen_log_prob)
    # dpo_preference_gap =
    #     beta * ((new_chosen - ref_chosen) - (new_rejected - ref_rejected))
    # ```
    #
    # Meaning:
    # - RLHF-style PPO uses reward and a policy ratio.
    # - DPO-style thinking compares chosen and rejected answers against a reference policy.
    #
    # Method:
    # Use helpers from this file:
    # - `reward_model_score(...)`
    # - `policy_log_probability(...)`
    # - `policy_ratio(...)`
    # - `clipped_policy_objective(...)`
    #
    # Steps:
    # 1. Score the chosen and rejected answers.
    # 2. Compute `reward_advantage`.
    # 3. Get new and reference log-probabilities for the chosen answer.
    # 4. Compute the chosen policy ratio.
    # 5. Compute the clipped PPO-style objective.
    # 6. Get new and reference log-probabilities for the rejected answer.
    # 7. Compute the DPO preference gap with the formula above.
    # 8. Return the dictionary.
    #
    # Checks:
    # - `chosen_reward` should usually be higher than `rejected_reward`.
    # - The note should say the two views both push preferred answers up.

    raise NotImplementedError("Return a plain comparison of reward-based and preference-based views.")








def write_bridge_note(comparisons: list[RLHFDPOToyComparison], config: RLHFBridgeConfig) -> str:
    """Write and save a short bridge note."""
    # Goal:
    # Write a short Markdown note that explains the bridge from RL to preference optimization.
    #
    # Input:
    # - `comparisons`: list[RLHFDPOToyComparison]
    # - `config`: RLHFBridgeConfig
    #
    # Output:
    # - `str`
    # - Path where the report was saved.
    #
    # Method:
    # Use `pathlib.Path`:
    # - `report_path = Path(config.report_path)`
    # - `report_path.parent.mkdir(parents=True, exist_ok=True)`
    # - `report_path.write_text(note_text, encoding="utf-8")`
    #
    # Include:
    # - one sentence about reward models
    # - one sentence about reference policies
    # - one sentence about DPO comparing chosen and rejected answers
    #
    # Steps:
    # 1. Convert `config.report_path` into a `Path`.
    # 2. Create the parent folder.
    # 3. Build a short Markdown string.
    # 4. Include a few mean values from `comparisons`.
    # 5. Save the file.
    # 6. Return `str(report_path)`.
    #
    # Checks:
    # - The note should not claim this is a full RLHF pipeline.
    # - The note should use plain language.
    raise NotImplementedError("Write a plain-language bridge note and return the saved path.")






def run_rlhf_bridge_demo(config: RLHFBridgeConfig) -> RLHFBridgeSummary:
    """Run the full RLHF bridge workflow."""
    # Goal:
    # Build preference examples, compare RLHF-style and DPO-style views, and return a summary.
    #
    # Input:
    # - `config`: RLHFBridgeConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"num_examples"`: int
    # - `"mean_chosen_reward"`: float
    # - `"mean_rejected_reward"`: float
    # - `"report_path"`: str
    # - `"bridge_note"`: str
    #
    # Method:
    # Use the helpers in this file:
    # - `examples = build_preference_dataset()`
    # - `comparisons = [compare_rlhf_style_and_dpo(example, config) for example in examples]`
    # - `report_path = write_bridge_note(comparisons, config)`
    #
    # Steps:
    # 1. Build the preference dataset.
    # 2. Compare RLHF-style and DPO-style views for every example.
    # 3. Compute the mean chosen reward.
    # 4. Compute the mean rejected reward.
    # 5. Save the bridge note.
    # 6. Write a one-sentence `bridge_note`.
    # 7. Return the summary dictionary.
    #
    # Checks:
    # - `mean_chosen_reward` should usually be higher than `mean_rejected_reward`.
    # - The report path should point to the saved Markdown note.
    raise NotImplementedError("Build preference examples, compare RLHF-style and DPO-style views, and return a summary.")





def check_metrics(metrics: RLHFBridgeSummary) -> bool:
    """Return True when the RLHF bridge summary has the expected keys."""
    required = {"num_examples", "mean_chosen_reward", "mean_rejected_reward", "report_path", "bridge_note"}
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
    """Run the RLHF bridge scaffold."""
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
    examples, ok = report_task_progress("build_preference_dataset", build_preference_dataset)
    if not count_or_stop(ok):
        return
    config = RLHFBridgeConfig()
    _, ok = report_task_progress("describe_rlhf_bridge_terms", describe_rlhf_bridge_terms, config)
    if not count_or_stop(ok):
        return
    example = examples[0]
    _, ok = report_task_progress("reward_model_score", reward_model_score, example, "chosen")
    if not count_or_stop(ok):
        return
    new_log_prob, ok = report_task_progress("policy_log_probability", policy_log_probability, example, "chosen", "new")
    if not count_or_stop(ok):
        return
    old_log_prob, ok = report_task_progress("policy_log_probability", policy_log_probability, example, "chosen", "reference")
    if not count_or_stop(ok):
        return
    ratio, ok = report_task_progress("policy_ratio", policy_ratio, new_log_prob, old_log_prob)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("clipped_policy_objective", clipped_policy_objective, ratio, 1.0, config.clip_epsilon)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("compare_rlhf_style_and_dpo", compare_rlhf_style_and_dpo, example, config)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_rlhf_bridge_demo", run_rlhf_bridge_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "RLHF bridge summary", float(int(passed)), "rlhf bridge keys present")
if __name__ == "__main__":
    main()
