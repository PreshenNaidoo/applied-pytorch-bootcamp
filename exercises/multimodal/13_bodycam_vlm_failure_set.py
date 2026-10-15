"""
============================================================
EXERCISE 13: BODYCAM VLM FAILURE SET
============================================================

WHAT YOU ARE BUILDING
You are building a small failure set for bodycam-style VLM evaluation.

The goal is not to make a perfect bodycam model.
The goal is to learn how to test model behavior under hard visual conditions.

Failure conditions:
- low light
- motion blur
- occlusion
- unusual angle
- small object
- ambiguous scene
- not enough information

It should use COCO-30 images as the starting point.
It may use a pretrained CLIP model later.

WHY THIS EXERCISE EXISTS
In safety-sensitive settings, average accuracy is not enough.
You need to know when the model fails and whether it can abstain.

HOW THIS FITS INTO THE INTERVIEW PREP
This is one of the most job-specific exercises.
It helps you talk about low light, blur, occlusion, ambiguity, and "not enough
information" as normal parts of evaluation, not edge cases to ignore.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
---------------------------------------------------------------
bodycam clips                        small COCO-30 image sample
messy real conditions                controlled image transforms
model failure analysis (CLIP or Qwen2.5-VL)    per-condition accuracy table
human review                         visual grid of failures

VISUALS TO CREATE
Save:
- grid of failure examples
- per-condition accuracy bar chart
- abstention rate chart

INPUTS AND OUTPUTS
Input example:
- image tensor
- question
- target answer
- failure condition name

Model output:
- answer
- confidence
- abstention flag

METRICS TO TRACK
- overall accuracy
- accuracy per condition
- worst-condition accuracy
- abstention rate
- examples where the model should say "not enough information"

STEP-BY-STEP TODO
1. Build clean image-question-answer examples.
2. Apply failure transforms such as blur, darkness, and occlusion.
3. Add ambiguous examples where the correct answer is not enough information.
4. Run a tiny model or pretrained CLIP-based model.
5. Convert model scores into answers and confidence.
6. Abstain when confidence is below the threshold.
7. Compute accuracy per condition.
8. Find the worst condition.
9. Save a grid of failure examples.
10. Save a per-condition chart.

DEFINITION OF DONE
You are done when:
- the failure set has named conditions
- model performance is reported per condition
- ambiguous examples can use "not enough information"
- charts show where the model is weak
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict

import torch
import torch.nn as nn

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"


CLIP_ANSWER_CHOICES = [
    "a person",
    "a vehicle",
    "an animal",
    "food",
    "a sports scene",
    "furniture",
    "an indoor scene",
    "an outdoor scene",
    "not enough information",
]


def caption_to_clip_answer(caption: str) -> str:
    """Map one COCO caption to the same answer choices used by CLIP."""
    text = caption.lower()

    if any(word in text for word in ["person", "people", "man", "woman", "boy", "girl", "child"]):
        return "a person"
    if any(word in text for word in ["car", "bus", "truck", "train", "motorcycle", "bike", "bicycle", "boat", "airplane"]):
        return "a vehicle"
    if any(word in text for word in ["dog", "cat", "horse", "sheep", "cow", "bird", "bear", "zebra", "giraffe", "animal"]):
        return "an animal"
    if any(word in text for word in ["food", "pizza", "sandwich", "cake", "banana", "apple", "orange", "broccoli", "donut"]):
        return "food"
    if any(word in text for word in ["tennis", "baseball", "skateboard", "surfboard", "snowboard", "frisbee", "sport"]):
        return "a sports scene"
    if any(word in text for word in ["chair", "couch", "bed", "table", "desk", "bench", "furniture"]):
        return "furniture"
    if any(word in text for word in ["room", "kitchen", "bathroom", "bedroom", "indoor", "inside"]):
        return "an indoor scene"

    return "an outdoor scene"


@dataclass
class FailureSetConfig:
    """Settings for bodycam-style failure testing."""

    image_size: int = 64
    confidence_threshold: float = 0.35
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    hf_dataset_name: str = "sayakpaul/coco-30-val-2014"
    sample_size: int = 128
    model_mode: str = "clip"
    pretrained_clip_name: str = "openai/clip-vit-base-patch32"


class FailureExample(TypedDict):
    """One hard evaluation example."""

    image: torch.Tensor
    question: str
    target_answer: str
    condition: str


class FailurePrediction(TypedDict):
    """One model prediction."""

    answer: str
    confidence: float
    abstained: bool


class FailureSetMetrics(TypedDict):
    """Failure-set summary."""

    overall_accuracy: float
    abstention_rate: float
    worst_condition_accuracy: float
    worst_condition: str
    failure_grid_path: str
    condition_chart_path: str


def build_failure_set(config: FailureSetConfig) -> list[FailureExample]:
    """Build hard image-question examples."""
    # Where this fits:
    # This helper creates the evaluation set.
    # It should be hard on purpose so the model's weak spots are visible.
    #
    # Goal:
    # Create a named failure set from images.
    #
    # Dataset:
    # Start with COCO-30 images from Hugging Face:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    #
    # Apply these fixed transforms:
    # - darken image
    # - blur image
    # - cover part of object
    # - shrink object
    # - rotate or crop image
    # - create ambiguous blank or blocked image
    #
    # Important:
    # The transformed images are still based on real images.
    # Do not create the main examples from drawn shapes.
    #
    # Output:
    # Return a list of dictionaries with:
    # - `"image"`: transformed image tensor shaped `[3, image_size, image_size]`
    # - `"question"`: text question
    # - `"target_answer"`: target answer string
    # - `"condition"`: name such as `"low_light"` or `"blur"`
    #
    # Method:
    # Use `load_dataset` from the Hugging Face `datasets` package:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    # Use this exact transform shape from `torchvision.transforms`:
    # `torchvision.transforms.Compose([torchvision.transforms.Resize((config.image_size, config.image_size)), torchvision.transforms.ToTensor()])`.
    # Load real images first, then create named stress-test versions with the
    # local helper `apply_failure_transform(...)`.
    #
    # Steps:
    # 1. Load the Hugging Face split with `load_dataset(...)`.
    # 2. Inspect one row with `data[0].keys()` if you are unsure of the field names.
    # 3. For each row, read the image and caption or label text.
    # 4. Convert the image to RGB with the PIL method `image.convert("RGB")`.
    # 5. Resize the image and convert it to a tensor.
    # 6. Build a simple question, such as `"what is visible?"`.
    # 7. Pick the target answer from the caption or label text.
    # 8. Loop over the fixed condition names.
    # 9. Apply each condition with `apply_failure_transform(image, condition, config)`.
    # 10. For ambiguous or blocked examples, use target answer `"not enough information"`.
    # 11. Append one dictionary per transformed example.
    #
    # Checks:
    # - Every condition should appear at least once.
    # - Transformed images should keep the same shape as original images.
    raise NotImplementedError("Build bodycam-style failure set.")





def apply_failure_transform(image: torch.Tensor, condition: str, config: FailureSetConfig) -> torch.Tensor:
    """Apply one visual failure condition."""
    # Method:
    # Use tensor operations from PyTorch.
    # - low light: `output = output * 0.25`
    # - occlusion: assign a slice to 0, for example `output[:, y0:y1, x0:x1] = 0`
    # - blur: use `torch.nn.functional.avg_pool2d(...)` on a batched image
    # - clamp: use `output.clamp(0, 1)`
    # You can import `torch.nn.functional as F` if you use `F.avg_pool2d`.
    #
    # Input:
    # - `image`: tensor shaped `[3, image_size, image_size]`
    # - `condition`: one of the condition names from the failure set
    #
    # Output:
    # - transformed image tensor with the same shape as input.
    #
    # Steps:
    # 1. Clone the input image with `output = image.clone()`.
    # 2. If `condition == "low_light"`, multiply by 0.25.
    # 3. If `condition == "occlusion"`, set a center rectangle slice to 0.
    # 4. If `condition == "blur"`, add a batch dimension, apply average pooling, then remove the batch dimension.
    # 5. If `condition == "small_object"`, resize or crop so the visible content is smaller.
    # 6. If `condition == "unusual_angle"`, use a simple crop, flip, or rotation transform.
    # 7. If `condition == "ambiguous"`, hide most of the image or make it mostly blank.
    # 8. Clamp the final tensor with `output = output.clamp(0, 1)`.
    # 9. Return the transformed tensor.
    #
    # Checks:
    # - Shape should not change.
    # - Pixel values should stay between 0 and 1.
    raise NotImplementedError("Apply visual failure transform.")





class TinyFailureSetModel(nn.Module):
    """Small model or wrapper used for failure testing."""

    def __init__(self, config: FailureSetConfig):
        super().__init__()
        # Goal:
        # Use either:
        # - a tiny trained model from earlier exercises
        # - pretrained CLIP when available
        #
        # If pretrained CLIP is not available, use a tiny model trained on real images.
        #
        # Suggested paths:
        # - For CLIP: load a pretrained CLIP model and processor from `transformers`.
        # - For a tiny local model: build a small CNN classifier and use the
        #   labels from the image dataset as possible answers.
        #
        # Method:
        # For CLIP, use Hugging Face Transformers:
        # `CLIPModel.from_pretrained(config.pretrained_clip_name)` and
        # `CLIPProcessor.from_pretrained(config.pretrained_clip_name)`.
        # These classes live in the `transformers` package.
        # For a tiny local model, use PyTorch layers from `torch.nn`, such as
        # `nn.Sequential`, `nn.Conv2d`, `nn.ReLU`, `nn.AdaptiveAvgPool2d`, and `nn.Linear`.
        #
        # Steps:
        # 1. Save `config.confidence_threshold` on `self`.
        # 2. Decide from `config.model_mode` whether to use `"clip"` or `"tiny"`.
        # 3. If using CLIP, load the processor and model with `.from_pretrained(...)`.
        # 4. If using the tiny path, build a small classifier and a list of label names.
        # 5. Move any child model to `device` if needed.
        # 6. Make sure `predict(...)` can access the model and label names.
        #
        # Checks:
        # - The model should accept real image tensors.
        # - `config.confidence_threshold` should be used in `predict(...)`.
        raise NotImplementedError("Build or load model for failure testing.")


    def predict(self, image: torch.Tensor, question: str) -> FailurePrediction:
        """Return answer, confidence, and abstention flag."""
        # Goal:
        # Return one answer plus a confidence score and abstention decision.
        #
        # Input:
        # - `image`: `[3, image_size, image_size]`
        # - `question`: text question for the image
        #
        # Output:
        # - dictionary with:
        #   - `"answer"`: string
        #   - `"confidence"`: float
        #   - `"abstained"`: bool
        #
        # Method:
        # For the tiny model, use `torch.softmax(logits, dim=-1)` to convert logits
        # to probabilities, then use `.max(dim=-1)` to get confidence and label id.
        # For CLIP, use the processor to build model inputs, run the CLIP model,
        # then apply `torch.softmax(...)` to the image-text similarity logits.
        #
        # Steps:
        # 1. Put the wrapped model in eval mode if it has `.eval()`.
        # 2. Add a batch dimension to the image with `image.unsqueeze(0)`.
        # 3. Run the tiny model or CLIP wrapper inside `with torch.no_grad():`.
        # 4. Convert scores to probabilities with `torch.softmax(..., dim=-1)`.
        # 5. Get top confidence and top label id with `.max(dim=-1)`.
        # 6. Convert the label id to an answer string.
        # 7. Set `abstained = confidence < self.confidence_threshold`.
        # 8. Return a dictionary with `"answer"`, `"confidence"`, and `"abstained"`.
        #
        # Check:
        # - Confidence should be between 0 and 1.
        raise NotImplementedError("Predict with confidence and possible abstention.")







def evaluate_failure_set(model: TinyFailureSetModel, examples: list[FailureExample], config: FailureSetConfig) -> FailureSetMetrics:
    """Evaluate accuracy and abstention by condition."""
    # Goal:
    # Measure how the model behaves under each failure condition.
    #
    # Method:
    # Use the model's local `predict(image, question)` method.
    # Use a normal Python dictionary such as `condition_stats` to group counts by condition.
    #
    # Steps:
    # 1. Start counters for total examples, correct predictions, and abstentions.
    # 2. Start a dictionary for per-condition totals and correct counts.
    # 3. Loop over examples.
    # 4. Call `model.predict(example["image"], example["question"])`.
    # 5. Count an answer as correct only when it did not abstain and the answer matches the target.
    # 6. Add abstentions to the abstention counter.
    # 7. Update the per-condition counts.
    # 8. Compute overall accuracy.
    # 9. Compute abstention rate.
    # 10. Compute accuracy for each condition.
    # 11. Return the lowest per-condition accuracy as `worst_condition_accuracy`.
    #
    # Check:
    # - Low worst-condition accuracy is not a code failure here.
    #   It means the failure set found a weak spot.
    raise NotImplementedError("Evaluate failure set by condition.")







def save_failure_visuals(examples: list[FailureExample], predictions: list[FailurePrediction], config: FailureSetConfig) -> tuple[str, str]:
    """Save failure grid and per-condition chart."""
    # Goal:
    # Save visuals that show what the model failed on.
    #
    # Output:
    # - `(failure_grid_path, condition_chart_path)` as strings.
    #
    # Method:
    # Use `config.output_dir.mkdir(parents=True, exist_ok=True)` from `pathlib.Path`.
    # Use Matplotlib: `matplotlib.use("Agg")`, `plt.subplots(...)`,
    # `ax.imshow(...)`, `ax.bar(...)`, `plt.savefig(path)`, and `plt.close()`.
    #
    # Steps:
    # 1. Create `config.output_dir`.
    # 2. Pick a small set of examples and matching predictions.
    # 3. Convert each image from `[3, H, W]` to `[H, W, 3]` with `.permute(1, 2, 0)` for `imshow`.
    # 4. Put condition, target, prediction, confidence, and abstention in each title.
    # 5. Save the failure grid.
    # 6. Group prediction correctness by condition.
    # 7. Save a bar chart of accuracy by condition.
    # 8. Return both saved paths as strings.
    #
    # Check:
    # - The grid should include at least a few wrong or abstained examples.
    raise NotImplementedError("Save failure-set visuals.")









def run_exercise(config: FailureSetConfig) -> FailureSetMetrics:
    """Run the full failure-set workflow."""
    # Main workflow:
    # hard examples -> model predictions -> per-condition metrics -> failure visuals.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_failure_set(config)
    model = TinyFailureSetModel(config).to(device)
    predictions = [model.predict(example["image"], example["question"]) for example in examples]
    metrics = evaluate_failure_set(model, examples, config)
    print(f"Overall accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"Worst condition: {metrics['worst_condition']}")
    print(f"Worst-condition accuracy: {metrics['worst_condition_accuracy']:.3f}")
    print(f"Abstention rate: {metrics['abstention_rate']:.3f}")
    failure_grid_path, condition_chart_path = save_failure_visuals(examples, predictions, config)
    metrics["failure_grid_path"] = failure_grid_path
    metrics["condition_chart_path"] = condition_chart_path
    return metrics


def check_metrics(metrics: FailureSetMetrics) -> bool:
    # This exercise is about honest measurement.
    # Passing does not require high accuracy.
    return metrics["worst_condition_accuracy"] >= 0.0 and metrics["abstention_rate"] >= 0.0


def report_task_progress(task_name: str, task_fn, *args):
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the failure-set scaffold one step at a time."""
    config = FailureSetConfig()
    set_seed(config.random_state)
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

    examples, ok = report_task_progress("build_failure_set", build_failure_set, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("apply_failure_transform", apply_failure_transform, examples[0]["image"], examples[0]["condition"], config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinyFailureSetModel.__init__", TinyFailureSetModel, config)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "failure-set measured", metrics["worst_condition_accuracy"], "per-condition results exist")


if __name__ == "__main__":
    main()
