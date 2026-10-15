"""
============================================================
EXERCISE 17: PAPER REPRODUCTION MINI PROJECT
============================================================

WHAT YOU ARE BUILDING
You are building a small paper reproduction project.

Plain-language meaning:
- reproduction means trying to rebuild an idea from a paper
- you do not need to match a huge result
- the goal is to test one core claim in a small setting

WHY THIS EXERCISE EXISTS
Research engineers often read papers and turn ideas into working code.
This exercise teaches that workflow in a safe, small way.
This exercise teaches:
- choosing one paper claim
- writing the core idea in plain words
- building a tiny version
- running a baseline
- running an ablation
- writing a short reproduction report

HOW THIS FITS INTO THE BOOTCAMP
This file is a capstone-style exercise for the earlier modeling and experiment skills.
It asks you to combine:
- clear problem framing
- fair comparison
- implementation
- reporting

Beginner intuition:
- a paper may contain many ideas, but a good reproduction starts with one clear claim
- you do not need full scale to learn something useful
- a tiny fair comparison can still show whether the core idea helps

Small example:
- baseline: simple model
- paper-inspired method: same model plus one new idea
- ablation: remove that new idea again
- if the paper-inspired method wins and the ablation hurts, the idea may be helping

DATASET
Use one small dataset you already understand well.
Recommended beginner choices:
- `sklearn.datasets.load_wine` for small tabular classification
- `sklearn.datasets.load_digits` for small image-style classification
- AG News if the paper idea fits text

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- choose one dataset
- make one fixed split
- reuse that split for baseline, paper-inspired method, and ablation

INPUTS AND OUTPUTS
Input:
- baseline config
- paper-inspired config
- ablation config
Output:
- comparable metric summary
- markdown report

METRICS TO TRACK
Track at least:
- baseline result
- paper-inspired result
- ablation result
- best method
- one comparison note

TARGET RESULT RANGE
Focus on whether the comparison is fair and readable.

Useful signs:
- all methods use the same split
- all methods use the same training budget unless the claim requires otherwise
- the final comparison clearly says whether the paper idea helped in this small setup

COMMON MISTAKES
- changing more than one thing at a time
- using different splits for different methods
- picking a paper claim that is too large or vague for a small test
- writing a report without showing the actual comparison numbers

STEP-BY-STEP TODO
1. Pick one simple paper or one idea from a paper.
2. Write the core claim in plain language.
3. Build a tiny baseline.
4. Build the paper-inspired method.
5. Use the same data split for both.
6. Run at least one ablation.
7. Compare real numbers.
8. Explain what matched and what did not.
9. Save a short markdown report.

DEFINITION OF DONE
You are done when:
- the claim is written clearly
- baseline and paper-inspired method both run
- at least one ablation is included
- results use the same split
- the report says what worked and what failed

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Keep the models small.
The goal is a fair comparison, not large-scale reproduction.

LOSS FUNCTION
Use the standard loss for your chosen task.
For classification, use `nn.CrossEntropyLoss`.

What each step is doing in simple words:
1. choose one claim you can actually test
2. write that claim clearly
3. build a simple baseline
4. build the paper-inspired version
5. keep the data split fixed
6. remove one part for the ablation
7. compare the numbers fairly
8. explain what matched the paper idea and what did not
9. save the report
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


ReproductionScope = dict[str, str | list[str]]
DataloaderSplits = dict[str, DataLoader]
MethodResult = dict[str, float | str]
ComparisonSummary = dict[str, str | float | int | list[MethodResult]]
ProjectSummary = dict[str, str | int]

# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class PaperReproductionConfig:
    """Settings for the paper reproduction mini project."""
    project_name: str = "tiny_paper_reproduction"
    paper_title: str = "Fill in the paper or idea title"
    dataset_name: str = "sklearn.datasets.load_wine"
    input_dim: int = 13
    hidden_dim: int = 64
    num_classes: int = 3
    num_samples: int = 1200
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    random_state: int = 42
    report_path: str = str(BOOTCAMP_ROOT / "reports" / "71_paper_reproduction_mini_project.md")

# Beginner hints for this file:
# - a reproduction does not need to be huge to be useful
# - pick one claim you can test clearly in a small setup
# - use the same split and training budget for every method so the comparison is fair
# - an ablation should change one thing at a time
# - the report matters because reproduction is as much about clear evidence as it is about code
def describe_reproduction_scope(config: PaperReproductionConfig) -> ReproductionScope:
    """Return the required project pieces."""
    return {
        "project_name": config.project_name,
        "paper_title": config.paper_title,
        "required_methods": ["baseline", "paper_inspired_method", "ablation"],
        "report_path": config.report_path,
    }

# Write the one core paper claim you want to test.
# This may be the most important sentence in the whole exercise.
# If the claim is vague, the reproduction becomes hard to judge.
def write_core_claim(config: PaperReproductionConfig) -> str:
    """Write the paper idea or claim in plain language."""
    # Goal:
    # - Write one clear claim you can test in this small project.
    # - The claim should name the baseline, the paper-inspired change, and the metric.
    #
    # Input:
    # - `config.paper_title`: the paper or idea title.
    # - `config.dataset_name`: the default dataset for the small test.
    #
    # Output:
    # - Return one plain string.
    # - Good shape:
    #   `"The paper idea claims that adding X to baseline model B improves validation accuracy on dataset D."`
    # - Beginner default example:
    #   `"The paper idea claims that adding batch normalization to a small MLP improves validation accuracy on load_wine compared with the same MLP without batch normalization."`
    #
    # Method:
    # - Pick one claim, not the whole paper.
    # - Name the dataset and metric before training.
    # - Use simple words. If you use a term like "regularization", explain it as "a change that helps reduce overfitting".
    #
    # Steps:
    # 1. Choose one small paper idea.
    # 2. Name the baseline.
    # 3. Name the paper-inspired change.
    # 4. Name the dataset and metric.
    # 5. Return the claim as one sentence.
    #
    # Checks:
    # - The claim should include `"baseline"` or name the reference method.
    # - The claim should include the metric, such as `"validation accuracy"`.
    raise NotImplementedError("Write the core claim you want to test.")

# Load one shared dataset split for every method.
# Baseline, paper-inspired method, and ablation must all use the same split.
def load_shared_dataloaders(config: PaperReproductionConfig) -> DataloaderSplits:
    """Return shared train, val, and test dataloaders."""
    # Goal:
    # - Return one fixed train, validation, and test split.
    # - Baseline, paper-inspired method, and ablation must all use these same loaders.
    #
    # Input:
    # - `config.dataset_name`: default is `sklearn.datasets.load_wine`.
    # - `config.input_dim`: number of input features, 13 for `load_wine`.
    # - `config.num_classes`: number of classes, 3 for `load_wine`.
    # - `config.batch_size`: batch size.
    # - `config.random_state`: seed for the split.
    #
    # Output:
    # - Return `DataloaderSplits`.
    # - Exact keys:
    #   - `"train"`
    #   - `"val"`
    #   - `"test"`
    # - Each loader yields `(x, y)`.
    # - `x` shape: `[batch_size, config.input_dim]`, dtype `torch.float32`.
    # - `y` shape: `[batch_size]`, dtype `torch.long`.
    # - Small example for `load_wine`:
    #   `x.shape == (32, 13)` and `y.shape == (32,)`.
    #
    # Method:
    # - Use these imports inside the function:
    #   `from sklearn.datasets import load_wine`
    #   `from sklearn.model_selection import train_test_split`
    #   `from sklearn.preprocessing import StandardScaler`
    #   `from torch.utils.data import TensorDataset, DataLoader`
    # - Load the data:
    #   `wine = load_wine()`
    # - Split twice with stratified labels:
    #   `x_train, x_temp, y_train, y_temp = train_test_split(wine.data, wine.target, test_size=0.4, random_state=config.random_state, stratify=wine.target)`
    #   `x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=config.random_state, stratify=y_temp)`
    # - Fit `StandardScaler()` on `x_train` only:
    #   `scaler = StandardScaler().fit(x_train)`
    # - Transform train, val, and test with the same scaler.
    # - Convert features to `torch.float32` and labels to `torch.long`.
    #
    # Steps:
    # 1. Load `load_wine`.
    # 2. Make a 60% train, 20% validation, 20% test split.
    # 3. Fit the scaler on train features only.
    # 4. Transform all splits with that train-fitted scaler.
    # 5. Convert each split to tensors.
    # 6. Wrap each split with `TensorDataset(x_split, y_split)`.
    # 7. Return DataLoaders with the exact keys above.
    #
    # Checks:
    # - `set(loaders) == {"train", "val", "test"}`
    # - `x_batch, y_batch = next(iter(loaders["train"]))`
    # - `x_batch.shape[1] == config.input_dim`
    # - `y_batch.dtype == torch.long`
    raise NotImplementedError("Return one shared split used by all methods.")

# Build the simple baseline model.
# This is the "before adding the paper idea" reference point.
def build_baseline_model(config: PaperReproductionConfig) -> nn.Module:
    """Build the baseline model."""
    # Goal:
    # - Return the reference model before adding the paper idea.
    # - This baseline should be simple and easy to compare against.
    #
    # Input:
    # - `config.input_dim`: input feature count.
    # - `config.hidden_dim`: hidden layer size.
    # - `config.num_classes`: output class count.
    #
    # Output:
    # - Return an `nn.Module`.
    # - For a batch `x` with shape `[batch_size, input_dim]`, output logits should have shape `[batch_size, num_classes]`.
    #
    # Method:
    # - Beginner default baseline:
    #   `nn.Sequential(`
    #   `    nn.Linear(config.input_dim, config.hidden_dim),`
    #   `    nn.ReLU(),`
    #   `    nn.Linear(config.hidden_dim, config.num_classes),`
    #   `)`
    # - Do not add softmax. `nn.CrossEntropyLoss` expects raw logits.
    #
    # Steps:
    # 1. Build the `nn.Sequential` model above.
    # 2. Move it to `device` with `.to(device)`.
    # 3. Return the model.
    #
    # Checks:
    # - `model(torch.randn(4, config.input_dim, device=device)).shape == (4, config.num_classes)`
    raise NotImplementedError("Return a simple baseline model.")

# Build the paper-inspired method.
# Keep only the core idea you want to test.
# Do not add extra unrelated changes at the same time.
def build_paper_inspired_model(config: PaperReproductionConfig) -> nn.Module:
    """Build the paper-inspired model or method."""
    # Goal:
    # - Return the model that adds the one paper-inspired idea.
    # - Beginner default idea: add batch normalization to the baseline MLP.
    #
    # Input:
    # - Same config fields as `build_baseline_model`.
    #
    # Output:
    # - Return an `nn.Module`.
    # - Output shape for `[batch_size, input_dim]` input: `[batch_size, num_classes]`.
    #
    # Method:
    # - Beginner default paper-inspired model:
    #   `nn.Sequential(`
    #   `    nn.Linear(config.input_dim, config.hidden_dim),`
    #   `    nn.BatchNorm1d(config.hidden_dim),`
    #   `    nn.ReLU(),`
    #   `    nn.Linear(config.hidden_dim, config.num_classes),`
    #   `)`
    # - `nn.BatchNorm1d(config.hidden_dim)` normalizes hidden features in each batch.
    # - Keep every other setting the same as the baseline.
    #
    # Steps:
    # 1. Build the `nn.Sequential` model above.
    # 2. Move it to `device`.
    # 3. Return the model.
    #
    # Checks:
    # - `model(torch.randn(4, config.input_dim, device=device)).shape == (4, config.num_classes)`
    # - The only main difference from baseline should be the paper-inspired layer or idea.
    raise NotImplementedError("Return the method inspired by the paper.")

def build_ablation_model(config: PaperReproductionConfig) -> nn.Module:
    """Build one ablation model.
    Plain-language meaning:
    - an ablation changes or removes one thing
    - this helps you see what mattered
    """
    # Goal:
    # - Return one ablation model that removes or changes one part of the paper-inspired method.
    # - Beginner default ablation: replace batch normalization with dropout.
    #
    # Input:
    # - Same config fields as the baseline and paper-inspired model.
    #
    # Output:
    # - Return an `nn.Module`.
    # - Output logits shape: `[batch_size, config.num_classes]`.
    #
    # Method:
    # - Beginner default ablation:
    #   `nn.Sequential(`
    #   `    nn.Linear(config.input_dim, config.hidden_dim),`
    #   `    nn.Dropout(p=0.2),`
    #   `    nn.ReLU(),`
    #   `    nn.Linear(config.hidden_dim, config.num_classes),`
    #   `)`
    # - This tests whether the result was specific to batch normalization or just any extra layer/change.
    #
    # Steps:
    # 1. Build the ablation model.
    # 2. Move it to `device`.
    # 3. Return it.
    #
    # Checks:
    # - Output shape should match the baseline and paper-inspired model.
    # - The ablation should change one clear thing.
    raise NotImplementedError("Return one ablation model.")

# Train and evaluate one method under the shared setup.
# Return the same keys for every method so comparison stays simple and fair.
def train_and_evaluate_method(
    model: nn.Module,
    loaders: DataloaderSplits,
    config: PaperReproductionConfig,
    method_name: str,
) -> MethodResult:
    """Train and evaluate one method."""
    # Goal:
    # - Train one model and return the same metric keys for every method.
    #
    # Input:
    # - `model`: one of the baseline, paper-inspired, or ablation models.
    # - `loaders`: shared loaders with keys `"train"`, `"val"`, and `"test"`.
    # - `method_name`: name to store in the result, such as `"baseline"`.
    #
    # Output:
    # - Return `MethodResult`.
    # - Exact keys:
    #   - `"method_name"`
    #   - `"train_loss"`
    #   - `"val_accuracy"`
    #   - `"test_accuracy"`
    # - Accuracy should be a percentage-like fraction from `0.0` to `1.0`.
    #
    # Method:
    # - Use exact training objects:
    #   `loss_fn = nn.CrossEntropyLoss()`
    #   `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    # - For each train batch:
    #   `x, y = batch`
    #   `x = x.to(device)`
    #   `y = y.to(device)`
    #   `logits = model(x)`
    #   `loss = loss_fn(logits, y)`
    # - For accuracy:
    #   `predictions = logits.argmax(dim=1)`
    #   `correct += (predictions == y).sum().item()`
    #
    # Steps:
    # 1. Create loss function and optimizer.
    # 2. Train for `config.epochs`.
    # 3. Average train loss over all train examples.
    # 4. Evaluate validation accuracy with `model.eval()` and `torch.no_grad()`.
    # 5. Evaluate test accuracy the same way.
    # 6. Return the result dictionary with exact keys above.
    #
    # Checks:
    # - `set(result) == {"method_name", "train_loss", "val_accuracy", "test_accuracy"}`
    # - `0.0 <= result["val_accuracy"] <= 1.0`
    # - Do not use a different loader split for different methods.
    raise NotImplementedError("Return train, validation, and test metrics for one method.")

# Compare baseline, paper-inspired method, and ablation.
# The key questions are:
# - did the paper-inspired method beat the baseline?
# - did the ablation hurt?
# - what does that suggest about the idea?
def compare_methods(results: list[MethodResult]) -> ComparisonSummary:
    """Compare baseline, paper-inspired method, and ablation."""
    # Goal:
    # - Compare the three method results in a fair, readable way.
    # - Decide which method had the best validation accuracy.
    #
    # Input:
    # - `results`: list of three `MethodResult` dictionaries.
    # - Each result should include `"method_name"`, `"val_accuracy"`, and `"test_accuracy"`.
    #
    # Output:
    # - Return `ComparisonSummary`.
    # - Exact keys:
    #   - `"best_method"`
    #   - `"best_val_accuracy"`
    #   - `"paper_minus_baseline_val_accuracy"`
    #   - `"paper_minus_ablation_val_accuracy"`
    #   - `"num_methods_compared"`
    #   - `"comparison_note"`
    #   - `"results"`
    #
    # Method:
    # - Find the best method with:
    #   `best = max(results, key=lambda item: float(item["val_accuracy"]))`
    # - Look up baseline, paper-inspired, and ablation rows by `"method_name"`.
    # - Difference formula:
    #   `paper_minus_baseline = paper_val_accuracy - baseline_val_accuracy`
    #   `paper_minus_ablation = paper_val_accuracy - ablation_val_accuracy`
    #
    # Steps:
    # 1. Check that three methods are present.
    # 2. Find the best validation accuracy.
    # 3. Compute paper-inspired minus baseline.
    # 4. Compute paper-inspired minus ablation.
    # 5. Write a short comparison note in plain language.
    # 6. Return the summary dictionary.
    #
    # Checks:
    # - `comparison["num_methods_compared"] == 3`
    # - The comparison note should say whether the paper idea helped in this small setup.
    raise NotImplementedError("Return best method and comparison note.")

# Write and save the final markdown report.
# A good report should answer:
# - what claim was tested?
# - what methods were compared?
# - what numbers were observed?
# - what conclusion seems reasonable?
def write_reproduction_report(
    config: PaperReproductionConfig,
    core_claim: str,
    comparison: ComparisonSummary,
) -> str:
    """Write and save the markdown reproduction report."""
    # Goal:
    # - Save a short markdown report with the claim, methods, results, ablation, and conclusion.
    #
    # Input:
    # - `config.report_path`: markdown path to write.
    # - `core_claim`: the claim being tested.
    # - `comparison`: dictionary from `compare_methods`.
    #
    # Output:
    # - Return `config.report_path` as a string.
    # - The file should exist after the function runs.
    #
    # Method:
    # - Use `Path`:
    #   `report_path = Path(config.report_path)`
    #   `report_path.parent.mkdir(parents=True, exist_ok=True)`
    #   `report_path.write_text(markdown, encoding="utf-8")`
    # - Include headings:
    #   `# Paper Reproduction Mini Project`
    #   `## Core Claim`
    #   `## Methods`
    #   `## Results`
    #   `## Ablation`
    #   `## Conclusion`
    #
    # Steps:
    # 1. Create the report folder.
    # 2. Build a markdown string.
    # 3. Include the exact comparison numbers.
    # 4. Explain whether the paper-inspired method beat the baseline.
    # 5. Explain what the ablation suggests.
    # 6. Save the file and return the path.
    #
    # Checks:
    # - `Path(report_path).exists()`
    # - The report should include `"## Results"` and the best method name.
    raise NotImplementedError("Write a report with claim, methods, results, ablation, and conclusion.")

def run_reproduction_project(config: PaperReproductionConfig) -> ProjectSummary:
    """Run the full paper reproduction mini project."""
    # Goal:
    # - Run the whole mini reproduction and return a small summary.
    #
    # Input:
    # - `config`: dataset, model, training, and report settings.
    #
    # Output:
    # - Return `ProjectSummary`.
    # - Exact keys:
    #   - `"best_method"`
    #   - `"num_methods_compared"`
    #   - `"report_path"`
    #
    # Method:
    # - Call helpers in this order:
    #   `loaders = load_shared_dataloaders(config)`
    #   `core_claim = write_core_claim(config)`
    #   `baseline_model = build_baseline_model(config)`
    #   `paper_model = build_paper_inspired_model(config)`
    #   `ablation_model = build_ablation_model(config)`
    #   `baseline_result = train_and_evaluate_method(baseline_model, loaders, config, "baseline")`
    #   `paper_result = train_and_evaluate_method(paper_model, loaders, config, "paper_inspired")`
    #   `ablation_result = train_and_evaluate_method(ablation_model, loaders, config, "ablation")`
    #   `comparison = compare_methods([baseline_result, paper_result, ablation_result])`
    #   `report_path = write_reproduction_report(config, core_claim, comparison)`
    #
    # Steps:
    # 1. Seed the run with `set_seed(config.random_state)`.
    # 2. Load shared dataloaders once.
    # 3. Build the three methods.
    # 4. Train and evaluate each method with the same loaders and budget.
    # 5. Compare results.
    # 6. Save the report.
    # 7. Return the summary keys above.
    #
    # Checks:
    # - `check_metrics(summary)` should return `True`.
    # - `summary["num_methods_compared"] == 3`.
    raise NotImplementedError("Run baseline, paper-inspired method, ablation, and report.")

def check_metrics(metrics: ProjectSummary) -> bool:
    """Return True when reproduction project outputs are present."""
    required = {"best_method", "num_methods_compared", "report_path"}
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
    """Run the paper reproduction scaffold."""
    config = PaperReproductionConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 12
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    loaders, ok = report_task_progress("load_shared_dataloaders", load_shared_dataloaders, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("describe_reproduction_scope", describe_reproduction_scope, config)
    if not count_or_stop(ok):
        return
    core_claim, ok = report_task_progress("write_core_claim", write_core_claim, config)
    if not count_or_stop(ok):
        return
    baseline_model, ok = report_task_progress("build_baseline_model", build_baseline_model, config)
    if not count_or_stop(ok):
        return
    paper_model, ok = report_task_progress("build_paper_inspired_model", build_paper_inspired_model, config)
    if not count_or_stop(ok):
        return
    ablation_model, ok = report_task_progress("build_ablation_model", build_ablation_model, config)
    if not count_or_stop(ok):
        return
    baseline_result, ok = report_task_progress("train_and_evaluate_method", train_and_evaluate_method, baseline_model, loaders, config, "baseline")
    if not count_or_stop(ok):
        return
    paper_result, ok = report_task_progress("train_and_evaluate_method", train_and_evaluate_method, paper_model, loaders, config, "paper_inspired")
    if not count_or_stop(ok):
        return
    ablation_result, ok = report_task_progress("train_and_evaluate_method", train_and_evaluate_method, ablation_model, loaders, config, "ablation")
    if not count_or_stop(ok):
        return
    comparison, ok = report_task_progress("compare_methods", compare_methods, [baseline_result, paper_result, ablation_result])
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("write_reproduction_report", write_reproduction_report, config, core_claim, comparison)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_reproduction_project", run_reproduction_project, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "paper reproduction summary", float(int(passed)), "project keys present")
if __name__ == "__main__":
    main()
