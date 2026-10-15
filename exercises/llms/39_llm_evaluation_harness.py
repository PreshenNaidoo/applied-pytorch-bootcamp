"""
============================================================
EXERCISE 39: LLM EVALUATION HARNESS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny evaluation harness for language-model behavior.

Plain-language meanings:
- an evaluation harness is a repeatable script that runs many checks
- exact match means the answer must match the target text exactly
- multiple choice means choosing one option from a fixed list
- calibration asks whether confidence matches correctness
- contamination means test examples appeared in training data
- judge-based evaluation uses another model or rule to grade open answers

WHY THIS EXERCISE EXISTS
Frontier labs use many evaluations.
One score is not enough.
The evaluation code must be repeatable and careful.

This exercise teaches:
- exact-match checks
- multiple-choice checks
- simple generation checks
- contamination checks
- confidence and calibration tracking
- a small evaluation report

HOW THIS FITS INTO THE BOOTCAMP
Earlier files measured accuracy and loss.
This file builds a broader language-model evaluation shape.

Beginner intuition:
- evals are tests for model behavior
- different tests catch different failures
- a good harness keeps prompts, answers, scores, and notes together

DATASET
Use tiny local evaluation datasets.

Clear default:
- exact-match examples about PyTorch basics
- multiple-choice examples about tensor shapes and model training
- short generation examples where the expected answer is one or two sentences
- a small list of training texts for contamination checks
- write all examples directly inside this file

Why these are separate:
- exact match checks facts with fixed answers
- multiple choice checks ranking among options
- generation checks open text behavior
- contamination checks whether eval prompts appeared in training text

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- write each eval set as a Python list of dictionaries
- keep prompts short and readable
- use a toy model-calling function first
- score every example in the same repeatable way

INPUTS AND OUTPUTS
Input:
- eval examples
- model predictions
- optional confidence scores

Output:
- exact-match score
- multiple-choice score
- generation score
- contamination report
- calibration summary
- final eval note

STEP-BY-STEP TODO
1. Build tiny exact-match examples.
2. Build tiny multiple-choice examples.
3. Build simple generation examples.
4. Write one model-calling interface.
5. Score exact match.
6. Score multiple choice.
7. Check contamination against training text.
8. Track confidence.
9. Return an evaluation report.

DEFINITION OF DONE
You are done when:
- each eval type returns a score
- contamination check is reported
- report rows include prompts and predictions
- the final note explains what the eval does and does not prove
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


ExactMatchExample = dict[str, str]
MultipleChoiceExample = dict[str, str | list[str] | int]
GenerationExample = dict[str, str]
ModelPrediction = dict[str, str | float]
EvalRow = dict[str, str | int | float | bool]
EvalScore = dict[str, float | list[EvalRow]]
ContaminationReport = dict[str, int | float | list[str]]
CalibrationSummary = dict[str, float]
LLMEvalReport = dict[str, str | EvalScore | ContaminationReport | CalibrationSummary]
# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for the LLM evaluation harness.
# Expected return:
# - a config object for repeatable evaluation setup
@dataclass
class LLMEvalConfig:
    """Settings for the LLM evaluation harness."""

    random_state: int = 42


# Build exact-match examples.
# Expected return:
# - list of examples with prompt and answer
def build_exact_match_examples() -> list[ExactMatchExample]:
    """Return exact-match eval examples."""
    # Goal:
    # - Return small questions where the target answer must match exactly.
    # - These are best for facts or short labels, not long free-text answers.
    #
    # Input:
    # - no direct arguments
    #
    # Output:
    # - Return `list[ExactMatchExample]`.
    # - Each example must have exactly these keys:
    #   - `"prompt"`: the question string
    #   - `"answer"`: the exact target answer string
    # - Small example:
    #   `{"prompt": "What library uses tensors in this bootcamp?", "answer": "PyTorch"}`
    #
    # Method:
    # - Use a plain Python list of dictionaries. No library call is needed.
    #
    # Steps:
    # 1. Write at least 3 examples.
    # 2. Keep answers short, such as `"PyTorch"`, `"batch"`, or `"gradient"`.
    # 3. Use the same key names for every example.
    # 4. Do not include choices here. Choices belong in `build_multiple_choice_examples`.
    #
    # Checks:
    # - `len(examples) >= 3`
    # - `set(examples[0]) == {"prompt", "answer"}`
    # - `isinstance(examples[0]["answer"], str)`
    #
    raise NotImplementedError("Return prompts with exact target answers.")


# Build multiple-choice examples.
# Expected return:
# - list of examples with prompt, choices, and correct index
def build_multiple_choice_examples() -> list[MultipleChoiceExample]:
    """Return multiple-choice eval examples."""
    # Goal:
    # - Return questions where the model must choose one answer from a fixed list.
    #
    # Input:
    # - no direct arguments
    #
    # Output:
    # - Return `list[MultipleChoiceExample]`.
    # - Each example must have exactly these keys:
    #   - `"prompt"`: the question string
    #   - `"choices"`: a list of answer strings, such as `["2", "4", "8"]`
    #   - `"correct_index"`: the integer index of the right choice
    # - Small example:
    #   `{"prompt": "A tensor with shape [2, 3] has how many rows?", "choices": ["2", "3"], "correct_index": 0}`
    #
    # Method:
    # - Use a plain Python list of dictionaries.
    # - Indexing starts at 0, so `correct_index=0` means the first choice is right.
    #
    # Steps:
    # 1. Write at least 3 examples.
    # 2. Give each example 2 to 4 choices.
    # 3. Make sure `0 <= correct_index < len(choices)`.
    # 4. Keep wrong choices plausible enough that the eval is useful.
    #
    # Checks:
    # - `set(examples[0]) == {"prompt", "choices", "correct_index"}`
    # - `isinstance(examples[0]["choices"], list)`
    # - `0 <= examples[0]["correct_index"] < len(examples[0]["choices"])`
    #
    raise NotImplementedError("Return prompts, answer choices, and correct index.")


# Build generation examples.
# Expected return:
# - list of examples with prompt, reference, and required phrase
def build_generation_examples() -> list[GenerationExample]:
    """Return simple generation eval examples."""
    # Goal:
    # - Return short open-answer prompts and simple rules for grading them.
    # - This is a toy version of a generation eval, so use clear rules.
    #
    # Input:
    # - no direct arguments
    #
    # Output:
    # - Return `list[GenerationExample]`.
    # - Each example must have exactly these keys:
    #   - `"prompt"`: the instruction string
    #   - `"reference"`: a short example of a good answer
    #   - `"required_phrase"`: a phrase that must appear in the prediction
    # - Small example:
    #   `{"prompt": "Explain why batching helps.", "reference": "Batching shares work across examples.", "required_phrase": "shares work"}`
    #
    # Method:
    # - Use a plain Python list of dictionaries.
    # - Use `required_phrase` for scoring later, not a hidden rule in your head.
    #
    # Steps:
    # 1. Write at least 2 generation examples.
    # 2. Keep each prompt short.
    # 3. Pick a `required_phrase` that a good answer should include.
    # 4. Make sure `required_phrase` is also present in `reference`.
    #
    # Checks:
    # - `set(examples[0]) == {"prompt", "reference", "required_phrase"}`
    # - `examples[0]["required_phrase"].lower() in examples[0]["reference"].lower()`
    #
    raise NotImplementedError("Return prompts for open-ended generation checks.")


# Call the model or toy model.
# Expected return:
# - dictionary with text prediction and confidence
def call_model(prompt: str) -> ModelPrediction:
    """Return a model prediction for one prompt."""
    # Goal:
    # - Return one toy model answer for one prompt.
    # - This function is the model interface. Later scoring functions should call this instead of making up answers themselves.
    #
    # Input:
    # - `prompt`: one string, such as `"What does a DataLoader do?"`
    #
    # Output:
    # - Return `ModelPrediction`.
    # - Exact keys:
    #   - `"prediction"`: the answer text string
    #   - `"confidence"`: a float from 0.0 to 1.0
    # - Small example:
    #   `{"prediction": "PyTorch", "confidence": 0.80}`
    #
    # Method:
    # - Use simple `if` / `elif` checks on lowercase prompt text.
    # - Example call shape:
    #   `prompt_lower = prompt.lower()`
    # - This exercise does not need a real LLM API. A toy rule-based model is enough.
    #
    # Steps:
    # 1. Convert the prompt to lowercase.
    # 2. If the prompt contains a known keyword, return a matching answer and higher confidence.
    # 3. Otherwise, return a short fallback answer and lower confidence.
    # 4. Keep confidence in the closed range `[0.0, 1.0]`.
    #
    # Checks:
    # - `set(prediction) == {"prediction", "confidence"}`
    # - `0.0 <= prediction["confidence"] <= 1.0`
    #
    raise NotImplementedError("Return prediction text and confidence.")


# Score exact-match examples.
# Expected return:
# - score dictionary and per-example rows
def score_exact_match(examples: list[ExactMatchExample]) -> EvalScore:
    """Score exact-match examples."""
    # Goal:
    # - Run `call_model` on exact-match examples and compute exact-match accuracy.
    # - Exact match means the cleaned prediction text equals the cleaned target answer text.
    #
    # Input:
    # - `examples`: list of dictionaries from `build_exact_match_examples`.
    # - Each item has:
    #   `{"prompt": str, "answer": str}`
    #
    # Output:
    # - Return `EvalScore`.
    # - Exact top-level keys:
    #   - `"accuracy"`: a float from 0.0 to 1.0
    #   - `"rows"`: one row per example
    # - Each row should have:
    #   - `"prompt"`, `"target"`, `"prediction"`, `"correct"`, `"confidence"`
    #
    # Method:
    # - Use Python string cleanup:
    #   `clean = text.strip().lower()`
    # - Formula:
    #   `accuracy = number_correct / number_of_examples`
    # - `number_correct` is the count of rows where `correct` is `True`.
    #
    # Steps:
    # 1. Start with an empty `rows` list.
    # 2. For each example, call `call_model(example["prompt"])`.
    # 3. Read `prediction["prediction"]` and `prediction["confidence"]`.
    # 4. Compare cleaned prediction with cleaned answer.
    # 5. Append a row with the keys listed above.
    # 6. Compute accuracy from the rows.
    #
    # Checks:
    # - `len(result["rows"]) == len(examples)`
    # - `0.0 <= result["accuracy"] <= 1.0`
    #
    raise NotImplementedError("Return exact-match accuracy and rows.")


# Score multiple-choice examples.
# Expected return:
# - score dictionary and per-example rows
def score_multiple_choice(examples: list[MultipleChoiceExample]) -> EvalScore:
    """Score multiple-choice examples."""
    # Goal:
    # - Run `call_model` and score whether it picked the right choice.
    # - This toy harness lets the model answer with either choice text or a choice number.
    #
    # Input:
    # - `examples`: list of dictionaries from `build_multiple_choice_examples`.
    # - Each item has:
    #   `{"prompt": str, "choices": list[str], "correct_index": int}`
    #
    # Output:
    # - Return `EvalScore`.
    # - Exact top-level keys:
    #   - `"accuracy"`: a float from 0.0 to 1.0
    #   - `"rows"`: one row per example
    # - Each row should have:
    #   - `"prompt"`, `"target_index"`, `"predicted_index"`, `"correct"`, `"confidence"`
    #
    # Method:
    # - Use a small helper inside this function or a short loop:
    #   compare `prediction_text.strip().lower()` with each choice lowercased.
    # - If the prediction is `"0"`, `"1"`, `"2"`, or `"3"`, treat it as a choice index.
    # - Formula:
    #   `accuracy = number_correct / number_of_examples`
    #
    # Steps:
    # 1. For each example, check that `correct_index` is a valid index.
    # 2. Call `call_model(example["prompt"])`.
    # 3. Convert the model answer into `predicted_index`.
    # 4. Set `correct = predicted_index == correct_index`.
    # 5. Add a row and compute accuracy.
    #
    # Checks:
    # - `len(result["rows"]) == len(examples)`
    # - Every row has an integer `"target_index"`.
    # - `0.0 <= result["accuracy"] <= 1.0`
    #
    raise NotImplementedError("Return multiple-choice accuracy and rows.")


# Score generation examples with a simple rule.
# Expected return:
# - score dictionary and per-example rows
def score_generation_examples(examples: list[GenerationExample]) -> EvalScore:
    """Score simple generation examples."""
    # Goal:
    # - Run `call_model` on open-answer prompts and score each answer with a simple phrase rule.
    # - This is not a full judge model. It is a clear, repeatable teaching version.
    #
    # Input:
    # - `examples`: list of dictionaries from `build_generation_examples`.
    # - Each item has:
    #   `{"prompt": str, "reference": str, "required_phrase": str}`
    #
    # Output:
    # - Return `EvalScore`.
    # - Exact top-level keys:
    #   - `"accuracy"`: fraction of examples that passed the phrase rule
    #   - `"rows"`: one row per example
    # - Each row should have:
    #   - `"prompt"`, `"reference"`, `"prediction"`, `"required_phrase"`, `"correct"`, `"confidence"`
    #
    # Method:
    # - Use lowercase substring matching:
    #   `correct = required_phrase.lower() in prediction_text.lower()`
    # - Formula:
    #   `accuracy = number_correct / number_of_examples`
    #
    # Steps:
    # 1. For each example, call `call_model(example["prompt"])`.
    # 2. Read the prediction text and confidence.
    # 3. Check whether the required phrase appears in the prediction.
    # 4. Add one row per example.
    # 5. Compute accuracy from the rows.
    #
    # Checks:
    # - `len(result["rows"]) == len(examples)`
    # - `0.0 <= result["accuracy"] <= 1.0`
    #
    raise NotImplementedError("Return generation scores and rows.")


# Check whether eval examples appear in training text.
# Expected return:
# - contamination report dictionary
def check_contamination(eval_prompts: list[str], training_texts: list[str]) -> ContaminationReport:
    """Check for possible eval contamination."""
    # Goal:
    # - Report eval prompts that also appear inside the toy training text.
    # - Contamination matters because a model may look good by remembering test data.
    #
    # Input:
    # - `eval_prompts`: prompt strings from all eval sets.
    #   Example: `["What is PyTorch?", "What shape is this tensor?"]`
    # - `training_texts`: text strings that stand in for model training data.
    #   Example: `["A tutorial says: What is PyTorch? PyTorch is a tensor library."]`
    #
    # Output:
    # - Return `ContaminationReport`.
    # - Exact keys:
    #   - `"num_prompts"`: total number of eval prompts
    #   - `"num_contaminated"`: how many prompts were found in training text
    #   - `"contamination_rate"`: `num_contaminated / num_prompts`
    #   - `"contaminated_prompts"`: list of prompt strings that were found
    #
    # Method:
    # - Join all training strings into one lowercase string:
    #   `training_blob = "\n".join(training_texts).lower()`
    # - For each prompt, check:
    #   `prompt.lower() in training_blob`
    # - Formula:
    #   `contamination_rate = num_contaminated / num_prompts`
    #
    # Steps:
    # 1. Build `training_blob`.
    # 2. Loop over every eval prompt.
    # 3. Add matching prompts to `contaminated_prompts`.
    # 4. Count matches.
    # 5. Return the four keys listed above.
    #
    # Checks:
    # - `report["num_prompts"] == len(eval_prompts)`
    # - `0.0 <= report["contamination_rate"] <= 1.0`
    #
    raise NotImplementedError("Report which eval prompts appear in training text.")


# Compute a simple calibration summary.
# Expected return:
# - dictionary comparing confidence and correctness
def compute_calibration_summary(rows: list[EvalRow]) -> CalibrationSummary:
    """Return a simple confidence calibration summary."""
    # Goal:
    # - Compare average confidence with average correctness.
    # - A model is better calibrated when confidence is close to how often it is right.
    #
    # Input:
    # - `rows`: rows from exact-match, multiple-choice, and generation scoring.
    # - Each row must include:
    #   - `"correct"`: bool
    #   - `"confidence"`: float from 0.0 to 1.0
    # - Small example:
    #   `[{"correct": True, "confidence": 0.9}, {"correct": False, "confidence": 0.8}]`
    #
    # Output:
    # - Return `CalibrationSummary`.
    # - Exact keys:
    #   - `"average_confidence"`: mean confidence
    #   - `"accuracy"`: mean correctness where True counts as 1.0 and False counts as 0.0
    #   - `"calibration_gap"`: absolute difference between those two numbers
    #
    # Method:
    # - Use Python sums and lengths.
    # - Formulas:
    #   `average_confidence = sum(confidences) / len(rows)`
    #   `accuracy = sum(1.0 if row["correct"] else 0.0 for row in rows) / len(rows)`
    #   `calibration_gap = abs(average_confidence - accuracy)`
    #
    # Steps:
    # 1. Handle the empty-list case by returning zeros for all three keys.
    # 2. Read every row's confidence and correctness.
    # 3. Compute the three formulas above.
    # 4. Return only the exact keys listed above.
    #
    # Checks:
    # - `0.0 <= summary["average_confidence"] <= 1.0`
    # - `0.0 <= summary["accuracy"] <= 1.0`
    # - `0.0 <= summary["calibration_gap"] <= 1.0`
    #
    raise NotImplementedError("Compare confidence with correctness.")


# Run the full evaluation harness.
# Expected return:
# - report dictionary with all eval scores and notes
def run_llm_evaluation_harness(config: LLMEvalConfig) -> LLMEvalReport:
    """Run the full LLM evaluation harness."""
    # Goal:
    # - Run all eval types, check contamination, compute calibration, and return one report.
    # - This is the function that connects all smaller helpers.
    #
    # Input:
    # - `config`: `LLMEvalConfig`.
    # - Use `set_seed(config.random_state)` before running evals so toy choices stay repeatable.
    #
    # Output:
    # - Return `LLMEvalReport`.
    # - Exact keys:
    #   - `"exact_match"`: result from `score_exact_match`
    #   - `"multiple_choice"`: result from `score_multiple_choice`
    #   - `"generation"`: result from `score_generation_examples`
    #   - `"contamination"`: result from `check_contamination`
    #   - `"calibration"`: result from `compute_calibration_summary`
    #   - `"eval_note"`: one short plain-English note
    #
    # Method:
    # - Call the helpers in this order:
    #   `build_exact_match_examples()`
    #   `build_multiple_choice_examples()`
    #   `build_generation_examples()`
    #   `score_exact_match(exact_examples)`
    #   `score_multiple_choice(mc_examples)`
    #   `score_generation_examples(generation_examples)`
    #   `check_contamination(eval_prompts, training_texts)`
    #   `compute_calibration_summary(all_rows)`
    #
    # Steps:
    # 1. Seed the run with `set_seed(config.random_state)`.
    # 2. Build all three eval datasets.
    # 3. Score each dataset.
    # 4. Build `eval_prompts` by collecting every example's `"prompt"`.
    # 5. Create a tiny `training_texts` list inside this function.
    # 6. Run the contamination check.
    # 7. Combine rows from all three score outputs:
    #    `all_rows = exact["rows"] + multiple_choice["rows"] + generation["rows"]`
    # 8. Run calibration on `all_rows`.
    # 9. Return the six report keys listed above.
    #
    # Checks:
    # - `set(report) == {"exact_match", "multiple_choice", "generation", "contamination", "calibration", "eval_note"}`
    # - `len(report["exact_match"]["rows"]) == len(exact_examples)`
    # - The eval note should say that this tiny harness is useful for practice but does not prove real-world model quality.
    #
    raise NotImplementedError("Run evals, contamination checks, calibration, and return report.")


# Check the report.
# Expected return:
# - True when key eval outputs are present
def check_metrics(metrics: LLMEvalReport) -> bool:
    """Return True when evaluation outputs are present."""
    required = {"exact_match", "multiple_choice", "generation", "contamination", "calibration", "eval_note"}
    return required.issubset(metrics)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the LLM eval harness scaffold."""
    config = LLMEvalConfig()
    set_seed(config.random_state)

    try:
        metrics = run_llm_evaluation_harness(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "llm_eval_outputs", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
