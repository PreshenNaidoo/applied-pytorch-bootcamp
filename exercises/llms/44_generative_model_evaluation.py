"""
============================================================
EXERCISE 44: GENERATIVE MODEL EVALUATION
============================================================

WHAT YOU ARE BUILDING
You are building a tiny evaluation set for generated text.

Plain-language meaning:
- generative models create answers, not only class labels
- one answer can be partly right and partly wrong
- evaluation needs examples, rules, and failure notes

WHY THIS EXERCISE EXISTS
Accuracy is not enough for generated text.
Research engineers need to build clear evals that catch real failures.
This exercise teaches:
- exact match
- simple overlap score
- rubric-based scoring
- pairwise preference
- hallucination checks
- writing an eval report

HOW THIS FITS INTO THE BOOTCAMP
Earlier evaluation exercises focused more on classification-style metrics.
This file teaches the extra care needed when the output is free-form text instead of one fixed label.

What each step is doing in simple words:
1. build a tiny set of prompts, references, and outputs
2. score them with one strict metric
3. score them again with a softer overlap metric
4. add a small rubric for quality
5. compare two outputs directly
6. flag suspicious unsupported claims
7. write an eval note that a human can trust

Beginner intuition:
- generated answers are harder to score than class labels
- one answer can be correct in meaning but use different words
- another answer can sound fluent but still contain false claims

Small example:
- prompt: `What color is the sky on a clear day?`
- reference: `blue`
- output A: `The sky is blue.`
- output B: `The sky is usually green.`
- exact match, overlap, and rubric scores would treat these answers differently

DATASET
Use a tiny local prompt-answer evaluation set.

Clear default:
- 8 to 12 prompts about PyTorch and ML basics
- one reference answer for each prompt
- two candidate model outputs for each prompt:
  - one mostly correct answer
  - one answer with a clear mistake or unsupported claim
- write the examples directly inside this file

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- write a short list of prompts
- write one reference answer for each prompt
- write one or two model outputs for each prompt
- score those outputs with several simple rules

STEP-BY-STEP TODO
1. Build a tiny prompt-answer eval set.
2. Add model outputs.
3. Compute exact match.
4. Compute a simple token overlap score.
5. Score answers with a small rubric.
6. Compare two model outputs for the same prompt.
7. Flag unsupported claims.
8. Write a short eval report.

DEFINITION OF DONE
You are done when:
- exact match is computed
- overlap score is computed
- rubric score is computed
- failed examples are listed
- you can explain why generative eval is hard

INPUTS AND OUTPUTS
Input:
- prompt
- reference answer
- model output
Output:
- exact-match score
- overlap score
- rubric score
- unsupported-claim flag
- eval report

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
No new model is required.
This exercise is about evaluation design and reading failures.

LOSS FUNCTION
No training loss is required.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.training_checks import print_pass_or_try_again


EvalPlan = dict[str, list[str] | str]
EvalExample = dict[str, str]
RubricScore = dict[str, float | str]
EvaluationSummary = dict[str, float | str | list[EvalExample] | list[str]]
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
class GenerativeEvalConfig:
    """Settings for generative model evaluation."""
    report_path: str = str(BOOTCAMP_ROOT / "reports" / "62_generative_model_evaluation.md")

# Beginner hints for this file:
# - generated answers are often partly right and partly wrong
# - that is why one metric is usually not enough
# - exact match is strict, overlap is softer, and rubric scores can capture quality better
# - failed examples are very important here; read them directly
# - keep the eval set small and clear at first so you trust what each metric means

def describe_eval_plan(config: GenerativeEvalConfig) -> EvalPlan:
    """Return the eval pieces this exercise should build."""
    return {
        "metrics": ["exact_match", "token_overlap", "rubric_score", "pairwise_preference"],
        "report_path": config.report_path,
    }

# Return a tiny local evaluation set for generated answers.
# Each example should include:
# - prompt
# - reference answer
# - one or two model outputs
def build_eval_examples() -> list[EvalExample]:
    """Return prompts, reference answers, and model outputs."""
    # Goal:
    # - Return a tiny dataset for checking generated answers.
    # - Each example should include one reference answer and two candidate outputs.
    #
    # Input:
    # - no direct arguments
    #
    # Output:
    # - Return `list[EvalExample]`.
    # - Each example must have exactly these keys:
    #   - `"prompt"`: the question or instruction
    #   - `"reference"`: a short correct answer
    #   - `"output_a"`: first model answer
    #   - `"output_b"`: second model answer
    # - Small example:
    #   `{"prompt": "What does a DataLoader do?", "reference": "It batches and loads data.", "output_a": "It batches data for training.", "output_b": "It changes the model weights."}`
    #
    # Method:
    # - Use a plain Python list of dictionaries.
    # - Make `output_a` mostly correct and `output_b` clearly worse for at least some examples.
    # - This makes pairwise preference easy to check by hand.
    #
    # Steps:
    # 1. Write 8 to 12 examples.
    # 2. Keep prompts about PyTorch or ML basics.
    # 3. Keep references short enough to compare by eye.
    # 4. Write two candidate outputs for each prompt.
    # 5. Include at least a few outputs with a clear unsupported claim.
    #
    # Checks:
    # - `len(examples) >= 8`
    # - `set(examples[0]) == {"prompt", "reference", "output_a", "output_b"}`
    # - `examples[0]["reference"]` should not be an empty string.
    #
    raise NotImplementedError("Return a tiny generative eval dataset.")

# Return exact match for one prediction-reference pair.
# This is strict:
# - tiny wording changes can make the score drop to zero
def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when prediction exactly matches reference, else 0.0."""
    # Goal:
    # - Return 1.0 only when the prediction and reference are the same after simple cleanup.
    #
    # Input:
    # - `prediction`: generated answer string.
    # - `reference`: target answer string.
    # - Small example:
    #   prediction `"PyTorch"` and reference `"pytorch"` should match after lowercasing.
    #
    # Output:
    # - Return `1.0` for a match and `0.0` for a non-match.
    #
    # Method:
    # - Clean both strings with:
    #   `text.strip().lower()`
    # - Formula:
    #   `score = 1.0 if cleaned_prediction == cleaned_reference else 0.0`
    #
    # Steps:
    # 1. Strip whitespace from both strings.
    # 2. Lowercase both strings.
    # 3. Compare them.
    # 4. Return `1.0` or `0.0`.
    #
    # Checks:
    # - `exact_match("PyTorch", "pytorch") == 1.0`
    # - `exact_match("PyTorch library", "PyTorch") == 0.0`
    #
    raise NotImplementedError("Compute exact match.")

# Return a softer overlap score than exact match.
# This is useful when wording changes but meaning may still be similar.
def token_overlap_score(prediction: str, reference: str) -> float:
    """Return a simple overlap score between prediction and reference tokens."""
    # Goal:
    # - Return a softer score that gives partial credit for shared words.
    #
    # Input:
    # - `prediction`: generated answer string.
    # - `reference`: target answer string.
    # - Small example:
    #   prediction `"it batches data"` and reference `"it loads data"` share `{"it", "data"}`.
    #
    # Output:
    # - Return a float from 0.0 to 1.0.
    # - Use 0.0 if the reference has no tokens.
    #
    # Method:
    # - Tokenize with lowercase whitespace splitting:
    #   `prediction_tokens = set(prediction.lower().split())`
    #   `reference_tokens = set(reference.lower().split())`
    # - Formula:
    #   `overlap = len(prediction_tokens & reference_tokens) / len(reference_tokens)`
    #
    # Steps:
    # 1. Lowercase and split both strings.
    # 2. Convert token lists to sets.
    # 3. If the reference set is empty, return 0.0.
    # 4. Count shared tokens.
    # 5. Divide by the number of reference tokens.
    #
    # Checks:
    # - `token_overlap_score("it batches data", "it loads data") == 2 / 3`
    # - The returned score should always be between 0.0 and 1.0.
    #
    raise NotImplementedError("Compute a simple token overlap score.")

def rubric_score(example: EvalExample) -> RubricScore:
    """Score one answer with a tiny hand-written rubric."""
    # Goal:
    # - Score `output_a` with a small, clear rubric.
    # - Rubric means a short scoring rule a human can understand.
    #
    # Input:
    # - `example`: one dictionary from `build_eval_examples`.
    # - Required keys: `"prompt"`, `"reference"`, `"output_a"`, `"output_b"`.
    #
    # Output:
    # - Return `RubricScore`.
    # - Exact keys:
    #   - `"correctness"`: float from 0.0 to 1.0
    #   - `"clarity"`: float from 0.0 to 1.0
    #   - `"groundedness"`: float from 0.0 to 1.0
    #   - `"mean_score"`: average of the three scores
    #   - `"note"`: short reason for the score
    #
    # Method:
    # - Use `token_overlap_score(example["output_a"], example["reference"])` as a simple correctness signal.
    # - Use a simple clarity rule, such as:
    #   `clarity = 1.0 if len(example["output_a"].split()) <= 30 else 0.5`
    # - Use `flag_unsupported_claims(example)` for groundedness:
    #   `groundedness = 1.0 if len(flags) == 0 else 0.0`
    # - Formula:
    #   `mean_score = (correctness + clarity + groundedness) / 3`
    #
    # Steps:
    # 1. Compute correctness from overlap with the reference.
    # 2. Compute clarity from answer length.
    # 3. Compute groundedness from unsupported-claim flags.
    # 4. Average the three numbers.
    # 5. Return the rubric dictionary.
    #
    # Checks:
    # - `0.0 <= score["mean_score"] <= 1.0`
    # - `set(score) == {"correctness", "clarity", "groundedness", "mean_score", "note"}`
    raise NotImplementedError("Return correctness, clarity, and groundedness scores.")

def pairwise_preference(example: EvalExample) -> str:
    """Choose the better of two outputs for one prompt."""
    # Goal:
    # - Choose whether `output_a` or `output_b` is better for one prompt.
    # - Include a short reason so the choice is easy to audit.
    #
    # Input:
    # - `example`: one dictionary with `"reference"`, `"output_a"`, and `"output_b"`.
    #
    # Output:
    # - Return one string.
    # - Use a simple format like:
    #   `"output_a: closer to the reference"`
    #   or `"output_b: fewer unsupported claims"`.
    #
    # Method:
    # - Score both outputs against the reference:
    #   `score_a = token_overlap_score(example["output_a"], example["reference"])`
    #   `score_b = token_overlap_score(example["output_b"], example["reference"])`
    # - Prefer the output with the higher score.
    # - If tied, prefer the shorter answer because it is less likely to add unsupported detail in this toy eval.
    #
    # Steps:
    # 1. Compute overlap for output A.
    # 2. Compute overlap for output B.
    # 3. Compare the scores.
    # 4. Return a string naming the better output and a short reason.
    #
    # Checks:
    # - The returned string should contain `"output_a"` or `"output_b"`.
    # - The reason should be plain enough that another person can inspect it.
    #
    raise NotImplementedError("Return which output is better and why.")

def flag_unsupported_claims(example: EvalExample) -> list[str]:
    """Return claims that are not supported by the reference."""
    # Goal:
    # - Return short text snippets from `output_a` that look unsupported by the reference.
    # - This is a tiny version of hallucination checking.
    #
    # Input:
    # - `example`: one dictionary with `"reference"` and `"output_a"`.
    # - Small example:
    #   reference `"A tensor stores numbers."`
    #   output `"A tensor stores numbers and always lives on a GPU."`
    #   unsupported claim could be `"always lives on a GPU"`.
    #
    # Output:
    # - Return `list[str]`.
    # - Return an empty list when no suspicious claim is found.
    #
    # Method:
    # - Keep this simple and rule-based.
    # - Split `output_a` into short claim-like chunks with:
    #   `claims = [part.strip() for part in output.split(".") if part.strip()]`
    # - For each claim, check whether its important words appear in the reference.
    # - A simple rule is: flag a claim if none of its non-trivial words appear in the reference.
    # - You can define non-trivial words by dropping tiny words like `"the"`, `"a"`, `"is"`, and `"to"`.
    #
    # Steps:
    # 1. Lowercase the reference.
    # 2. Split `output_a` into claim chunks.
    # 3. Remove very common short words from each claim.
    # 4. If a claim has meaningful words but none appear in the reference, add it to the flags.
    # 5. Return the list of flags.
    #
    # Checks:
    # - `isinstance(flags, list)`
    # - Return `[]` for a short answer that simply repeats the reference.
    # - This simple check will miss many real hallucinations; say that in the report.
    raise NotImplementedError("Return unsupported or suspicious claims.")

# Run all the evaluation checks over the tiny eval set.
# Return aggregate scores plus failed examples worth reading by hand.
def evaluate_outputs(examples: list[EvalExample]) -> EvaluationSummary:
    """Run all eval checks."""
    # Goal:
    # - Score every example and return aggregate metrics plus examples worth reading by hand.
    #
    # Input:
    # - `examples`: list from `build_eval_examples`.
    # - Each example has `"prompt"`, `"reference"`, `"output_a"`, and `"output_b"`.
    #
    # Output:
    # - Return `EvaluationSummary`.
    # - Exact keys:
    #   - `"exact_match"`: mean exact-match score for `output_a`
    #   - `"token_overlap"`: mean token-overlap score for `output_a`
    #   - `"mean_rubric_score"`: mean rubric score for `output_a`
    #   - `"failed_examples"`: list of examples with low rubric score or unsupported claims
    #   - `"preferences"`: list of pairwise preference strings
    #   - `"unsupported_claims"`: flat list of unsupported claim strings
    #
    # Method:
    # - For each example, call:
    #   `exact_match(example["output_a"], example["reference"])`
    #   `token_overlap_score(example["output_a"], example["reference"])`
    #   `rubric_score(example)`
    #   `pairwise_preference(example)`
    #   `flag_unsupported_claims(example)`
    # - Mean formula:
    #   `mean_score = sum(scores) / len(scores)`
    #
    # Steps:
    # 1. Handle an empty `examples` list by returning zeros and empty lists.
    # 2. Create empty lists for exact scores, overlap scores, rubric scores, failures, preferences, and flags.
    # 3. Loop over every example.
    # 4. Score `output_a` with exact match and token overlap.
    # 5. Get the rubric dictionary and store its `"mean_score"`.
    # 6. Add low-scoring examples to `failed_examples`.
    # 7. Add all unsupported claims to the flat flags list.
    # 8. Compute means.
    # 9. Return the six exact keys listed above.
    #
    # Checks:
    # - `0.0 <= summary["exact_match"] <= 1.0`
    # - `0.0 <= summary["token_overlap"] <= 1.0`
    # - `0.0 <= summary["mean_rubric_score"] <= 1.0`
    # - `len(summary["preferences"]) == len(examples)`
    #
    raise NotImplementedError("Return aggregate metrics and failed examples.")

def write_eval_report(config: GenerativeEvalConfig, summary: EvaluationSummary) -> str:
    """Write and save a short markdown eval report."""
    # Goal:
    # - Write a short markdown report that explains the eval results in plain language.
    # - Save the report to `config.report_path` and return that path.
    #
    # Input:
    # - `config.report_path`: where to save the markdown file.
    # - `summary`: dictionary from `evaluate_outputs`.
    #
    # Output:
    # - Return the report path as a string.
    # - The file should exist after this function runs.
    #
    # Method:
    # - Use `pathlib.Path`:
    #   `path = Path(config.report_path)`
    #   `path.parent.mkdir(parents=True, exist_ok=True)`
    #   `path.write_text(report_text, encoding="utf-8")`
    # - Include metric values and a short limitations section.
    #
    # Steps:
    # 1. Build a markdown string with a title.
    # 2. Add exact match, token overlap, and mean rubric score.
    # 3. Add a short section for failed examples.
    # 4. Add a short section for unsupported claims.
    # 5. Add a short limitations note, such as "This tiny eval does not prove real-world quality."
    # 6. Create the parent folder.
    # 7. Write the file.
    # 8. Return `str(path)`.
    #
    # Checks:
    # - `Path(returned_path).exists()`
    # - The report should mention that exact match is strict and overlap is only a rough signal.
    #
    raise NotImplementedError("Write a plain-language generative eval report.")

def run_evaluation(config: GenerativeEvalConfig) -> EvaluationSummary:
    """Run the full generative evaluation workflow."""
    # Goal:
    # - Run the full generative eval workflow and return the final summary.
    #
    # Input:
    # - `config`: settings object with the report path.
    #
    # Output:
    # - Return `EvaluationSummary`.
    # - It must include the keys from `evaluate_outputs` plus:
    #   - `"report_path"`: string path returned by `write_eval_report`
    #
    # Method:
    # - Call:
    #   `examples = build_eval_examples()`
    #   `summary = evaluate_outputs(examples)`
    #   `report_path = write_eval_report(config, summary)`
    #
    # Steps:
    # 1. Build examples.
    # 2. Evaluate all outputs.
    # 3. Write the report.
    # 4. Add `"report_path"` to the summary.
    # 5. Return the summary.
    #
    # Checks:
    # - `check_metrics(summary)` should return `True`.
    # - `summary["report_path"] == config.report_path`
    #
    raise NotImplementedError("Run evals and return summary metrics.")

def check_metrics(metrics: EvaluationSummary) -> bool:
    """Return True when main eval outputs are present."""
    required = {"exact_match", "token_overlap", "mean_rubric_score", "report_path"}
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
    """Run the generative eval scaffold."""
    completed_steps = 0
    total_steps = 4
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    config = GenerativeEvalConfig()
    _, ok = report_task_progress("describe_eval_plan", describe_eval_plan, config)
    if not count_or_stop(ok):
        return
    examples, ok = report_task_progress("build_eval_examples", build_eval_examples)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("evaluate_outputs", evaluate_outputs, examples)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_evaluation", run_evaluation, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "generative eval summary", float(int(passed)), "eval keys present")
if __name__ == "__main__":
    main()
