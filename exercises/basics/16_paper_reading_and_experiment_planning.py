"""
============================================================
EXERCISE 16: PAPER READING AND EXPERIMENT PLANNING
============================================================

WHAT YOU ARE BUILDING
You are building a small research plan before writing training code.

Plain-language meaning:
- a paper claim is the main idea you want to test
- an experiment plan says exactly how you will test it
- a baseline is the simple reference method you compare against
- an ablation changes one thing so you can see what mattered

WHY THIS EXERCISE EXISTS
Many beginners rush into coding too early.
That often leads to unfair comparisons, vague goals, and wasted time.
Research engineers usually do better work when they decide the test plan first.
This exercise teaches:
- how to read a paper for the core claim
- how to write the idea in plain words
- how to choose the smallest testable version
- how to choose a fair baseline
- how to choose metrics before coding
- how to plan ablations and failure checks
- how to write exact next steps another person could follow

HOW THIS FITS INTO THE BOOTCAMP
Earlier project exercises asked you to compare methods and write reports.
This exercise moves one step earlier in the workflow:
- before you code
- before you train
- before you compare numbers
It teaches the planning habit that strong research engineers use to avoid unclear experiments.

Beginner intuition:
- coding before planning is like building before drawing the map
- you may still build something,
  but it is much harder to know if it answers the right question

Small example:
- paper claim: a new attention change helps long-context language modeling
- smallest testable version:
  - baseline: tiny decoder without the new change
  - paper-inspired method: same decoder plus the new change
  - same split
  - same training budget
  - compare validation loss

IMPORTANT SCOPE NOTE
This exercise is mostly about thinking and writing clearly.
It is not mainly a coding exercise.
The main output is a markdown plan, not a trained model.

DATASET
Choose one concrete dataset before writing the plan.

Clear beginner defaults:
- tabular claim: `sklearn.datasets.load_wine`
- small image-style claim: `sklearn.datasets.load_digits`
- text-classification claim: Hugging Face `ag_news`
- language-modeling claim: Tiny Shakespeare from `data/input.txt`

Use one fixed split:
- 60% train
- 20% validation
- 20% test

Do not leave the plan saying "some dataset".
Write the exact dataset name, fields, split rule, and metric.

PAPER CHOICE
Pick one paper or one small claim from a paper.

Good beginner-friendly choices:
- one architectural change
- one training trick
- one evaluation idea
Avoid at first:
- a full giant system with many moving parts
- a paper where the claim is too vague to test clearly

OUTPUT
Save a markdown file with:
- paper title
- link or citation note
- core claim
- plain-language summary
- baseline
- paper-inspired method
- dataset and split
- metric
- ablations
- risks
- success rule
- exact next coding steps

TARGET RESULT RANGE
Focus on clarity and fairness, not on paper length or fancy wording.

Useful signs:
- a beginner could read the claim and understand it
- the baseline and paper-inspired method are close enough for a fair comparison
- the metric matches the claim
- the next coding steps are concrete enough that someone could actually start implementing

STEP-BY-STEP TODO
1. Pick one paper or one paper claim.
2. Write the title and citation note.
3. Write the core claim in plain language.
4. Write the smallest testable version of the claim.
5. Choose the baseline.
6. Choose the paper-inspired method.
7. Choose one dataset and one fixed split.
8. Choose the main metric.
9. Plan one or two ablations.
10. List likely risks and failure cases.
11. Write what result would support the claim.
12. Save the full markdown plan.

DEFINITION OF DONE
You are done when:
- the claim is specific
- the baseline is clear
- the comparison is fair
- the metric matches the claim
- the ablations change one thing at a time
- another person could read the plan and start implementing

COMMON MISTAKES
- summarizing the whole paper instead of one testable claim
- changing many things at once
- forgetting to define the baseline clearly
- choosing a metric after coding instead of before coding
- saying "it worked better" without deciding what that means

What each step is doing in simple words:
1. choose one idea small enough to test honestly
2. record the paper source so the plan stays grounded
3. turn the paper claim into a sentence a beginner can understand
4. shrink that claim into the smallest fair experiment
5. define the plain reference method
6. define the version that adds the paper idea
7. freeze the dataset and split so the comparison is fair
8. choose the number that will decide whether the idea helped
9. plan one or two controlled changes to test what mattered
10. write down likely failure modes before you code
11. state what result would count as support
12. save the plan so the later implementation has a clear map
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


PlanningWorkflowDescription = dict[str, str | list[str]]
PaperInfo = dict[str, str]
MethodPlan = dict[str, str]
DatasetSplitPlan = dict[str, str | tuple[float, float, float]]
MetricPlan = dict[str, str]
AblationPlan = dict[str, str]
ExperimentPlan = dict[
    str,
    PaperInfo
    | str
    | MethodPlan
    | DatasetSplitPlan
    | MetricPlan
    | list[AblationPlan]
    | list[str],
]
PlanningSummary = dict[str, str | int]

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
class ExperimentPlanningConfig:
    """Settings for the paper reading and planning exercise."""
    project_name: str = "paper_reading_and_plan"
    suggested_topic: str = "one small claim from a paper you can test fairly"
    dataset_name: str = "choose_one_dataset"
    random_state: int = 42
    report_path: str = str(BOOTCAMP_ROOT / "reports" / "70_paper_reading_and_experiment_plan.md")

# Beginner hints for this file:
# - one good sentence about the claim is better than one vague paragraph
# - choose the smallest fair test first, not the most impressive one
# - write the metric before coding so you do not move the goal later
# - a baseline is the "before the paper idea" version
# - exact next steps turn a vague plan into an actionable one
def describe_planning_workflow(config: ExperimentPlanningConfig) -> PlanningWorkflowDescription:
    """Return the main pieces that the planning report must include."""
    return {
        "project_name": config.project_name,
        "required_sections": [
            "paper_title",
            "citation_note",
            "core_claim",
            "plain_language_summary",
            "smallest_testable_version",
            "baseline",
            "paper_inspired_method",
            "dataset_and_split",
            "main_metric",
            "ablations",
            "risks",
            "success_rule",
            "next_coding_steps",
        ],
        "report_path": config.report_path,
    }

# Pick the paper or claim you want to test.
# Return at least:
# - `paper_title`
# - `citation_note`
# - `paper_link_or_source`
def choose_paper_or_claim(config: ExperimentPlanningConfig) -> PaperInfo:
    """Choose one paper or one paper claim."""
    # Goal:
    # - Pick one paper, or one small claim from a paper, that can be tested fairly.
    # - Return the source details needed for the report.
    #
    # Input:
    # - `config.suggested_topic`: the kind of claim to look for.
    # - `config.project_name`: the name of this planning project.
    #
    # Output:
    # - Return `PaperInfo`.
    # - Exact keys:
    #   - `"paper_title"`
    #   - `"citation_note"`
    #   - `"paper_link_or_source"`
    # - Small example:
    #   `{"paper_title": "Attention Is All You Need", "citation_note": "Vaswani et al., 2017", "paper_link_or_source": "arXiv:1706.03762"}`
    #
    # Method:
    # - If you already know the paper, fill the three strings directly.
    # - If you do not, choose a small claim from a paper you recently studied in this bootcamp.
    # - Do not choose a whole large system as the test target. Choose one claim.
    #
    # Steps:
    # 1. Write the paper title.
    # 2. Write a short citation note with author names or venue/year.
    # 3. Add a link, arXiv id, DOI, or local note about where the paper came from.
    # 4. Return the dictionary with the exact keys listed above.
    #
    # Checks:
    # - `set(paper_info) == {"paper_title", "citation_note", "paper_link_or_source"}`
    # - None of the three values should be an empty string.
    raise NotImplementedError("Return the paper title, citation note, and link or source.")

# Write the core claim in plain language.
# This is the most important sentence in the plan.
# A beginner should be able to read it and understand what is being tested.
def write_core_claim(paper_info: PaperInfo) -> str:
    """Return the one core claim in plain words."""
    # Goal:
    # - Write one clear sentence that says exactly what you plan to test.
    # - The claim should be small enough to test in this bootcamp.
    #
    # Input:
    # - `paper_info["paper_title"]`: the paper title.
    # - `paper_info["citation_note"]`: short source note.
    #
    # Output:
    # - Return one string.
    # - Good shape:
    #   `"The paper claims that adding X improves Y on task Z compared with baseline B."`
    # - Small example:
    #   `"The paper claims that using rotary position embeddings improves validation loss on longer text sequences compared with learned absolute positions."`
    #
    # Method:
    # - Use one claim, not the whole paper.
    # - Name the change, the outcome, the task, and the baseline.
    # - Avoid vague words like "better" unless you also say what number will measure it.
    #
    # Steps:
    # 1. Read the title and source note.
    # 2. Pick one testable claim from the paper.
    # 3. Write it as one plain sentence.
    # 4. Return that sentence.
    #
    # Checks:
    # - The claim should name both methods being compared.
    # - The claim should name the outcome you will measure.
    raise NotImplementedError("Write the main claim in simple plain language.")

# Write a short plain-language summary of the paper idea.
# Keep it short.
# The goal is not to rewrite the whole paper.
def write_plain_language_summary(paper_info: PaperInfo, core_claim: str) -> str:
    """Return a short plain-language summary."""
    # Goal:
    # - Explain the paper idea in a few beginner-friendly sentences.
    # - Keep the focus on the chosen claim, not the whole paper.
    #
    # Input:
    # - `paper_info`: dictionary from `choose_paper_or_claim`.
    # - `core_claim`: one-sentence claim from `write_core_claim`.
    #
    # Output:
    # - Return one short string.
    # - Aim for 3 to 5 sentences.
    # - Good content:
    #   - what problem the paper is trying to improve
    #   - what change the paper makes
    #   - why that change might help
    #   - what you will test in the small version
    #
    # Method:
    # - Use plain words.
    # - If you need a technical term, explain it right away.
    # - Do not copy the paper abstract. Write your own short explanation.
    #
    # Steps:
    # 1. Start with the problem.
    # 2. State the paper's change.
    # 3. Explain why it might help.
    # 4. Tie it back to your `core_claim`.
    # 5. Return the summary string.
    #
    # Checks:
    # - The summary should be understandable without reading the paper.
    # - It should not be longer than one short paragraph.
    raise NotImplementedError("Write a short beginner-friendly summary of the idea.")

# Write the smallest version of the claim you can test fairly.
# In plain words:
# - keep the setup small
# - keep the comparison fair
# - test one clear idea
def write_smallest_testable_version(core_claim: str) -> str:
    """Return the smallest fair test of the claim."""
    # Goal:
    # - Shrink the paper claim into the smallest fair experiment you can run.
    #
    # Input:
    # - `core_claim`: the one claim you chose to test.
    #
    # Output:
    # - Return one string that names:
    #   - baseline
    #   - paper-inspired method
    #   - dataset
    #   - fixed training budget
    #   - metric
    # - Small example:
    #   `"Train the same tiny decoder twice on Tiny Shakespeare for 1 epoch: once with learned absolute positions and once with rotary positions. Compare validation loss."`
    #
    # Method:
    # - Keep everything the same except the paper idea.
    # - Choose a budget small enough to actually run.
    # - Do not add extra improvements to the paper-inspired method.
    #
    # Steps:
    # 1. Name the two methods.
    # 2. Name the dataset.
    # 3. Name the fixed budget.
    # 4. Name the metric.
    # 5. Return the short experiment description.
    #
    # Checks:
    # - The test should change one main thing.
    # - A second person should be able to start coding from this sentence.
    raise NotImplementedError("Write the smallest testable version of the claim.")

# Choose:
# - one baseline
# - one paper-inspired method
# Keep the two choices close enough that the comparison stays fair.
def choose_methods(core_claim: str) -> MethodPlan:
    """Return the baseline and paper-inspired method descriptions."""
    # Goal:
    # - Define the two methods you will compare.
    # - Keep them close enough that the comparison is fair.
    #
    # Input:
    # - `core_claim`: the chosen paper claim.
    #
    # Output:
    # - Return `MethodPlan`.
    # - Exact keys:
    #   - `"baseline"`
    #   - `"paper_inspired_method"`
    #   - `"fairness_note"`
    # - Small example:
    #   baseline: `"Tiny decoder with learned absolute position embeddings."`
    #   paper-inspired method: `"Same decoder, but replace learned absolute positions with rotary position embeddings."`
    #   fairness note: `"Same data split, model size, optimizer, batch size, and training steps."`
    #
    # Method:
    # - The baseline is the simple reference method before adding the paper idea.
    # - The paper-inspired method should add only the paper idea.
    # - The fairness note should say what stays the same.
    #
    # Steps:
    # 1. Write the baseline.
    # 2. Write the paper-inspired method.
    # 3. Write the fairness note.
    # 4. Return the dictionary with the exact keys above.
    #
    # Checks:
    # - The baseline and paper-inspired method should differ by one main change.
    # - `set(methods) == {"baseline", "paper_inspired_method", "fairness_note"}`
    raise NotImplementedError("Return the baseline and the paper-inspired method.")

# Choose one dataset and one fixed split.
# The split should be shared by the baseline and paper-inspired method.
def choose_dataset_and_split(config: ExperimentPlanningConfig, core_claim: str) -> DatasetSplitPlan:
    """Return dataset choice and split plan."""
    # Goal:
    # - Choose one dataset and one split rule before coding.
    # - The same split must be used by both methods.
    #
    # Input:
    # - `config.dataset_name`: optional chosen dataset name.
    # - `core_claim`: the claim being tested.
    #
    # Output:
    # - Return `DatasetSplitPlan`.
    # - Exact keys:
    #   - `"dataset_name"`
    #   - `"input_fields"`
    #   - `"target_field"`
    #   - `"split_rule"`
    #   - `"split_percentages"`
    #   - `"fairness_note"`
    # - `split_percentages` should be `(0.6, 0.2, 0.2)` for train, validation, and test.
    #
    # Method:
    # - Choose a dataset that matches the claim.
    # - For beginner defaults:
    #   - image-style claim: `sklearn.datasets.load_digits`
    #   - tabular claim: `sklearn.datasets.load_wine`
    #   - text classification claim: Hugging Face `ag_news`
    #   - language modeling claim: Tiny Shakespeare from `data/input.txt`
    # - Do not return `"some dataset"`.
    #
    # Steps:
    # 1. Pick the dataset name.
    # 2. Name the input fields.
    # 3. Name the target field, if the task has one.
    # 4. Write the split rule.
    # 5. Add the fixed percentages `(0.6, 0.2, 0.2)`.
    # 6. Say that both methods use the same split.
    # 7. Return the dictionary.
    #
    # Checks:
    # - `dataset_plan["split_percentages"] == (0.6, 0.2, 0.2)`
    # - The dataset name should be specific.
    raise NotImplementedError("Return the dataset name, split rule, and fairness note.")

# Choose the main metric and the success rule.
# Example:
# - metric: validation accuracy
# - success rule: paper-inspired method beats baseline by a meaningful amount
def choose_metric_and_success_rule(core_claim: str) -> MetricPlan:
    """Return the main metric and what result would count as support."""
    # Goal:
    # - Choose the main metric before coding.
    # - State what result would support the claim.
    #
    # Input:
    # - `core_claim`: the claim being tested.
    #
    # Output:
    # - Return `MetricPlan`.
    # - Exact keys:
    #   - `"main_metric"`
    #   - `"success_rule"`
    #   - `"why_metric_fits_claim"`
    # - Small example:
    #   main metric: `"validation loss"`
    #   success rule: `"The paper-inspired method has lower validation loss than the baseline after the same number of training steps."`
    #
    # Method:
    # - Match the metric to the claim.
    # - For accuracy claims, use accuracy or F1.
    # - For language modeling claims, use validation loss or perplexity.
    # - For speed claims, use examples per second or wall-clock time.
    #
    # Steps:
    # 1. Choose one main metric.
    # 2. Write the success rule using that metric.
    # 3. Explain why the metric fits the claim.
    # 4. Return the dictionary.
    #
    # Checks:
    # - The success rule should not be vague.
    # - It should say which method must beat which baseline.
    raise NotImplementedError("Return the metric, success rule, and why the metric fits the claim.")

# Plan one or two ablations.
# Each ablation should change one thing at a time.
def plan_ablations(core_claim: str) -> list[AblationPlan]:
    """Return one or two ablation ideas."""
    # Goal:
    # - Plan one or two controlled changes that test what mattered.
    # - Each ablation should change one thing.
    #
    # Input:
    # - `core_claim`: the claim being tested.
    #
    # Output:
    # - Return `list[AblationPlan]`.
    # - Each list item must have exact keys:
    #   - `"name"`
    #   - `"change"`
    #   - `"what_it_tests"`
    # - Small example:
    #   `{"name": "remove rotary positions", "change": "Use learned absolute positions again", "what_it_tests": "Whether the position method caused the gain"}`
    #
    # Method:
    # - Start from the paper-inspired method.
    # - Remove or change one part.
    # - Say what that single change tests.
    #
    # Steps:
    # 1. Write one ablation name.
    # 2. Write the one change it makes.
    # 3. Write what it tests.
    # 4. Add a second ablation only if it is still simple.
    # 5. Return the list.
    #
    # Checks:
    # - `len(ablations) >= 1`
    # - Each ablation should change one thing only.
    raise NotImplementedError("Return small ablations that isolate one change at a time.")

# List likely risks or failure cases before coding.
# This is good research habit.
# It helps you think clearly about what may go wrong.
def list_risks_and_failure_cases(core_claim: str) -> list[str]:
    """Return likely risks and failure cases."""
    # Goal:
    # - List likely ways the planned experiment could fail or mislead you.
    #
    # Input:
    # - `core_claim`: the claim being tested.
    #
    # Output:
    # - Return `list[str]`.
    # - Aim for 3 to 5 short risks.
    # - Small examples:
    #   - `"The dataset may be too small to show the claimed effect."`
    #   - `"One random seed may make the result noisy."`
    #   - `"The small model may not match the setting where the paper idea helps."`
    #
    # Method:
    # - Think about data size, model size, random seed, metric choice, and implementation mistakes.
    # - Write risks before coding so the later report can be honest.
    #
    # Steps:
    # 1. Write one risk about data.
    # 2. Write one risk about model or training budget.
    # 3. Write one risk about measurement or randomness.
    # 4. Return the list.
    #
    # Checks:
    # - `len(risks) >= 3`
    # - Each risk should be a full sentence.
    raise NotImplementedError("Return a short list of likely risks or failure cases.")

# Write the exact next coding steps.
# These should be concrete enough that another person could follow them.
def write_exact_next_steps(core_claim: str) -> list[str]:
    """Return the exact next coding steps."""
    # Goal:
    # - Write ordered coding steps that another person could follow.
    #
    # Input:
    # - `core_claim`: the claim being tested.
    #
    # Output:
    # - Return `list[str]`.
    # - Aim for 5 to 8 steps.
    # - Small examples:
    #   - `"Create a dataset loader that returns the fixed train, validation, and test split."`
    #   - `"Implement the baseline model first and save its validation metric."`
    #   - `"Implement the paper-inspired change without changing the training budget."`
    #
    # Method:
    # - Start with data loading.
    # - Then baseline.
    # - Then paper-inspired method.
    # - Then metric comparison.
    # - Then ablations and report.
    #
    # Steps:
    # 1. Write the data-loading step.
    # 2. Write the baseline implementation step.
    # 3. Write the paper-inspired implementation step.
    # 4. Write the training and evaluation step.
    # 5. Write the ablation step.
    # 6. Write the report step.
    # 7. Return the ordered list.
    #
    # Checks:
    # - `len(next_steps) >= 5`
    # - Each item should start with a concrete action verb like "Create", "Train", or "Compare".
    raise NotImplementedError("Return a short ordered list of next coding steps.")

# Save the markdown experiment plan.
# The saved report should include all the required sections in a clean order.
def save_experiment_plan(
    config: ExperimentPlanningConfig,
    plan: ExperimentPlan,
) -> str:
    """Write the experiment plan to markdown and return the saved path."""
    # Goal:
    # - Save the full experiment plan as a markdown file.
    # - Return the saved path.
    #
    # Input:
    # - `config.report_path`: where to save the markdown file.
    # - `plan`: dictionary built in `main` or `run_planning_workflow`.
    # - Required plan keys:
    #   - `"paper_info"`
    #   - `"core_claim"`
    #   - `"plain_language_summary"`
    #   - `"smallest_testable_version"`
    #   - `"methods"`
    #   - `"dataset_and_split"`
    #   - `"metric_and_success_rule"`
    #   - `"ablations"`
    #   - `"risks"`
    #   - `"next_coding_steps"`
    #
    # Output:
    # - Return `config.report_path` as a string.
    # - The markdown file should contain all required sections in a readable order.
    #
    # Method:
    # - Use `pathlib.Path`:
    #   `report_path = Path(config.report_path)`
    #   `report_path.parent.mkdir(parents=True, exist_ok=True)`
    #   `report_path.write_text(markdown, encoding="utf-8")`
    # - Build the markdown with headings such as:
    #   `# Paper Reading And Experiment Plan`
    #   `## Core Claim`
    #   `## Dataset And Split`
    #   `## Next Coding Steps`
    #
    # Steps:
    # 1. Check that all required plan keys are present.
    # 2. Create the report folder.
    # 3. Format the plan into markdown.
    # 4. Write the markdown with UTF-8 encoding.
    # 5. Return the path string.
    #
    # Checks:
    # - `Path(saved_path).exists()`
    # - The saved text should include `"## Core Claim"` and `"## Next Coding Steps"`.
    raise NotImplementedError("Save the full markdown plan and return the report path.")

# Run the full planning workflow and return a summary.
# This exercise is complete when the plan is clear and saved.
def run_planning_workflow(config: ExperimentPlanningConfig) -> PlanningSummary:
    """Run the full paper reading and planning workflow."""
    # Goal:
    # - Run every planning helper, save the markdown report, and return a small summary.
    #
    # Input:
    # - `config`: project name, dataset choice, random seed, and report path.
    #
    # Output:
    # - Return `PlanningSummary`.
    # - Exact keys:
    #   - `"paper_title"`
    #   - `"core_claim"`
    #   - `"main_metric"`
    #   - `"report_path"`
    #   - `"next_steps_count"`
    #
    # Method:
    # - Call the helpers in this order:
    #   `paper_info = choose_paper_or_claim(config)`
    #   `core_claim = write_core_claim(paper_info)`
    #   `plain_summary = write_plain_language_summary(paper_info, core_claim)`
    #   `smallest_test = write_smallest_testable_version(core_claim)`
    #   `methods = choose_methods(core_claim)`
    #   `dataset_plan = choose_dataset_and_split(config, core_claim)`
    #   `metric_plan = choose_metric_and_success_rule(core_claim)`
    #   `ablations = plan_ablations(core_claim)`
    #   `risks = list_risks_and_failure_cases(core_claim)`
    #   `next_steps = write_exact_next_steps(core_claim)`
    #   `report_path = save_experiment_plan(config, plan)`
    #
    # Steps:
    # 1. Seed the run with `set_seed(config.random_state)`.
    # 2. Build each plan section.
    # 3. Assemble the `plan` dictionary with the required keys from `save_experiment_plan`.
    # 4. Save the report.
    # 5. Return the summary dictionary with the exact keys above.
    #
    # Checks:
    # - `check_metrics(summary)` should return `True`.
    # - `summary["next_steps_count"] == len(next_steps)`.
    raise NotImplementedError("Build the plan, save the report, and return a summary.")

def check_metrics(metrics: PlanningSummary) -> bool:
    """Return True when the planning summary has the expected keys."""
    required = {"paper_title", "core_claim", "main_metric", "report_path", "next_steps_count"}
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
    """Run the paper reading and planning scaffold one step at a time."""
    config = ExperimentPlanningConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 13

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    _, ok = report_task_progress("describe_planning_workflow", describe_planning_workflow, config)
    if not count_or_stop(ok):
        return

    paper_info, ok = report_task_progress("choose_paper_or_claim", choose_paper_or_claim, config)
    if not count_or_stop(ok):
        return

    core_claim, ok = report_task_progress("write_core_claim", write_core_claim, paper_info)
    if not count_or_stop(ok):
        return

    plain_summary, ok = report_task_progress("write_plain_language_summary", write_plain_language_summary, paper_info, core_claim)
    if not count_or_stop(ok):
        return

    smallest_test, ok = report_task_progress("write_smallest_testable_version", write_smallest_testable_version, core_claim)
    if not count_or_stop(ok):
        return

    methods, ok = report_task_progress("choose_methods", choose_methods, core_claim)
    if not count_or_stop(ok):
        return

    dataset_plan, ok = report_task_progress("choose_dataset_and_split", choose_dataset_and_split, config, core_claim)
    if not count_or_stop(ok):
        return

    metric_plan, ok = report_task_progress("choose_metric_and_success_rule", choose_metric_and_success_rule, core_claim)
    if not count_or_stop(ok):
        return

    ablations, ok = report_task_progress("plan_ablations", plan_ablations, core_claim)
    if not count_or_stop(ok):
        return

    risks, ok = report_task_progress("list_risks_and_failure_cases", list_risks_and_failure_cases, core_claim)
    if not count_or_stop(ok):
        return

    next_steps, ok = report_task_progress("write_exact_next_steps", write_exact_next_steps, core_claim)
    if not count_or_stop(ok):
        return

    plan = {
        "paper_info": paper_info,
        "core_claim": core_claim,
        "plain_language_summary": plain_summary,
        "smallest_testable_version": smallest_test,
        "methods": methods,
        "dataset_and_split": dataset_plan,
        "metric_and_success_rule": metric_plan,
        "ablations": ablations,
        "risks": risks,
        "next_coding_steps": next_steps,
    }
    _, ok = report_task_progress("save_experiment_plan", save_experiment_plan, config, plan)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_planning_workflow", run_planning_workflow, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(
            passed,
            "experiment planning summary",
            float(int(passed)),
            "planning keys present",
        )
if __name__ == "__main__":
    main()
