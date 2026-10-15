"""
============================================================
EXERCISE 11: GROUNDED VLM ANSWER WITH EVIDENCE
============================================================

WHAT YOU ARE BUILDING
You are building a tiny system where an answer must include visible evidence.

The model output is not just:

```text
"red square"
```

It must also include a box:

```text
answer: "red square"
evidence_box: [x_min, y_min, x_max, y_max]
```

WHY THIS EXERCISE EXISTS
For sensitive settings, text alone is not enough.
A human should be able to check what the model used as evidence.

This is very relevant for bodycam or policing-style VLM work.

HOW THIS FITS INTO THE INTERVIEW PREP
This exercise gives you a practical answer to:
`How do you make VLM outputs checkable?`
The key idea is that an answer should point to evidence, not only sound fluent.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
---------------------------------------------------------------
real frame or video                  small CPPE-5 image with object boxes
natural language answer              short answer label
box, point, or timestamp evidence (Molmo or Florence-2)    bounding box evidence
human review                         automatic support check

WHAT IS INCLUDED
- image tensors
- target answers
- target boxes
- predicted answers
- predicted boxes
- IoU check
- supported/unsupported answer check

INPUTS AND OUTPUTS
Image:
- `[3, image_size, image_size]`

Question:
- text such as `"what object is visible?"`

Target answer:
- short answer such as `"red square"`

Target box:
- `[x_min, y_min, x_max, y_max]`
- coordinates are normalized from 0 to 1

Model output:
- answer logits
- predicted box

METRICS TO TRACK
- answer accuracy
- mean IoU
- supported answer rate
- rejected unsupported answer count

STEP-BY-STEP TODO
1. Build real images with object labels and target boxes.
2. Build an answer vocabulary.
3. Build a model with an answer head and box head.
4. Compute answer cross-entropy loss.
5. Compute box coordinate loss.
6. Train both heads together.
7. Compute IoU for predicted boxes.
8. Check whether each answer is supported by its box.
9. Reject unsupported answers.
10. Save target and predicted boxes drawn on images.

DEFINITION OF DONE
You are done when:
- the system predicts an answer and box
- IoU is computed correctly
- unsupported answers can be rejected
- a visualization draws target and predicted boxes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GroundedVLMConfig:
    """Settings for grounded VLM answer checking."""

    image_size: int = 64
    hidden_size: int = 96
    batch_size: int = 16
    epochs: int = 300
    learning_rate: float = 3e-3
    iou_threshold: float = 0.50
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    hf_dataset_name: str = "cppe-5"
    sample_size: int = 256
    pretrained_vision_mode: str = "tiny"


class GroundedExample(TypedDict):
    """One grounded image example."""

    image: torch.Tensor
    question: str
    answer: str
    box: torch.Tensor


class GroundedPrediction(TypedDict):
    """One answer with evidence."""

    answer: str
    box: torch.Tensor
    confidence: float


class GroundingMetrics(TypedDict):
    """Grounding summary."""

    answer_accuracy: float
    mean_iou: float
    supported_answer_rate: float
    rejected_unsupported_count: int
    box_visual_path: str
    loss_curve_path: str


class GroundedDataset(Dataset):
    """Dataset for answer plus evidence-box training."""

    def __init__(self, examples: list[GroundedExample], answer_vocab: dict[str, int]):
        self.examples = examples
        self.answer_vocab = answer_vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        return {
            "image": example["image"],
            "answer_id": torch.tensor(self.answer_vocab[example["answer"]], dtype=torch.long),
            "box": example["box"],
            "question": example["question"],
            "answer": example["answer"],
        }


def build_grounded_dataset(config: GroundedVLMConfig) -> list[GroundedExample]:
    """Build images with answer labels and target boxes."""
    # Where this fits:
    # This is the evidence dataset.
    # Each example must say both what the answer is and where the evidence is.
    #
    # Goal:
    # Return image tensors with object boxes.
    #
    # Dataset:
    # Use CPPE-5 from Hugging Face:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    # CPPE-5 has real images with object boxes.
    #
    # Output:
    # - list of dictionaries with:
    #   - `"image"`: `[3, image_size, image_size]`
    #   - `"question"`: string
    #   - `"answer"`: object label string
    #   - `"box"`: tensor `[x_min, y_min, x_max, y_max]`
    #
    # Method:
    # Use `load_dataset` from the Hugging Face `datasets` package:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    # Use this exact transform shape from `torchvision.transforms`:
    # `torchvision.transforms.Compose([torchvision.transforms.Resize((config.image_size, config.image_size)), torchvision.transforms.ToTensor()])`.
    # Use `torch.tensor(box, dtype=torch.float32)` for the target box.
    #
    # Steps:
    # 1. Load the Hugging Face split with `load_dataset(...)`.
    # 2. Inspect one row with `data[0].keys()` if you are unsure of the field names.
    # 3. For each row, read the image and its object annotations.
    # 4. Skip rows with no boxes.
    # 5. Convert the image to RGB with the PIL method `image.convert("RGB")`.
    # 6. Resize the image and convert it to a tensor.
    # 7. Pick one object box for the first version.
    # 8. Convert the box to `[x_min, y_min, x_max, y_max]` format if needed.
    # 9. Normalize box coordinates by image width and height so values are in 0-1.
    # 10. Convert the object category id into the answer string.
    # 11. Append a dictionary with `"image"`, `"question"`, `"answer"`, and `"box"`.
    #
    # Checks:
    # - Box values should be between 0 and 1.
    # - `x_min < x_max` and `y_min < y_max`.

    raise NotImplementedError("Build grounded image-answer-box data.")







def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute IoU for boxes in normalized xyxy format."""
    # Goal:
    # Compute Intersection over Union.
    #
    # Input:
    # - `boxes_a`: `[N, 4]`
    # - `boxes_b`: `[N, 4]`
    # - each box is `[x_min, y_min, x_max, y_max]`
    # - coordinates are normalized from 0 to 1
    #
    # Output:
    # - IoU tensor with shape `[N]`
    #
    # Formula:
    # `IoU = overlap_area / union_area`
    #
    # Method:
    # Use PyTorch elementwise functions:
    # `torch.maximum(...)`, `torch.minimum(...)`, and `.clamp(min=0)`.
    #
    # Steps:
    # 1. Split each box tensor into left, top, right, and bottom values.
    # 2. Find overlap left edge with `torch.maximum(a_left, b_left)`.
    # 3. Find overlap top edge with `torch.maximum(a_top, b_top)`.
    # 4. Find overlap right edge with `torch.minimum(a_right, b_right)`.
    # 5. Find overlap bottom edge with `torch.minimum(a_bottom, b_bottom)`.
    # 6. Compute overlap width and height, then clamp both at 0.
    # 7. Compute overlap area.
    # 8. Compute each box area.
    # 9. Compute union area.
    # 10. Return `overlap_area / union_area.clamp(min=1e-6)`.
    #
    # Check:
    # - A box compared with itself should have IoU close to 1.

    # Box coordinates use xyxy format:
    #
    # x grows left to right.
    # y grows top to bottom. ie. y = 0.0  top of image, y = 1.0  bottom of image
    #
    #     x=0.0                                      x=1.0
    # y=0.0  +------------------------------------------+
    #        |                                          |
    #        |                                          |
    #        |        box A                             |
    #        |        +-------------------+             |
    #        |        |                   |             |
    #        |        |       overlap     |             |
    #        |        |           +-------|-------+     |
    #        |        |           |       |       |     |
    #        |        +-----------|-------+       |     |
    #        |                    |     box B     |     |
    #        |                    +---------------+     |
    # y=1.0  +------------------------------------------+
    #
    # box A corners:
    # - top-left:     a_left, a_top
    # - bottom-right: a_right, a_bot
    #
    # box B corners:
    # - top-left:     b_left, b_top
    # - bottom-right: b_right, b_bot
    #
    # The overlap box uses:
    # - the larger left edge, because overlap starts after both boxes have started
    # - the larger top edge, because overlap starts below both top edges
    # - the smaller right edge, because overlap ends before either box has ended
    # - the smaller bottom edge, because overlap ends above either bottom edge
    #
    # Concrete example:
    #
    # box A = [0.20, 0.20, 0.70, 0.70]
    #         [left, top,  right, bottom]
    #
    # box B = [0.50, 0.40, 0.90, 0.90]
    #         [left, top,  right, bottom]
    #
    # overlap_left   = max(0.20, 0.50) = 0.50
    # overlap_top    = max(0.20, 0.40) = 0.40
    # overlap_right  = min(0.70, 0.90) = 0.70
    # overlap_bottom = min(0.70, 0.90) = 0.70
    #
    # So the overlap box is [0.50, 0.40, 0.70, 0.70].
    #
    # Why max for left/top:
    # the overlap cannot start until both boxes have started.
    #
    # Why min for right/bottom:
    # the overlap must stop as soon as either box ends.

    #a_left, a_top, a_right, a_bot = boxes_a[0], boxes_a[1], boxes_a[2], boxes_a[3]
    raise NotImplementedError("Compute box IoU.")

    #b_left, b_top, b_right, b_bot = boxes_b[0], boxes_b[1], boxes_b[2], boxes_b[3]






    #iou = overlap_area / union_area   # This can divide by zero if a bad box has zero area.

    # IoU and Dice both measure overlap, but they divide by different things.
    #
    # IoU asks:
    # "How much of the total covered area is shared?"
    #
    # Dice asks:
    # "How much overlap is there compared with the two shapes' sizes?"
    #
    # The key difference is the denominator:
    # - IoU uses the union area: everything covered by either box
    # - Dice uses area A + area B, with 2 * overlap on top
    #
    # Small example:
    # area A = 10
    # area B = 10
    # overlap_area = 5
    #
    # union_area = 10 + 10 - 5 = 15
    # IoU = 5 / 15 = 0.33
    #
    # Dice = 2 * 5 / (10 + 10) = 10 / 20 = 0.50
    #
    # Plain memory rule:
    # - IoU is stricter because it asks how much of the full covered area overlaps.
    # - Dice often feels more like an overlap similarity score. i.e. Dice is often easier to read as "how much the two shapes match."




class TinyGroundedVLM(nn.Module):
    """Tiny model that predicts answer class and evidence box."""

    def __init__(self, num_answers: int, config: GroundedVLMConfig):
        super().__init__()
        # Goal:
        # Build one shared image encoder with two heads:
        # - answer head -> class logits
        # - box head -> 4 normalized box values
        #
        # Suggested layers:
        # - `self.image_encoder = nn.Sequential(...)`
        #   A small CNN can end with one vector shaped `[batch_size, hidden_size]`.
        # - `self.answer_head = nn.Linear(config.hidden_size, num_answers)`
        # - `self.box_head = nn.Linear(config.hidden_size, 4)`
        #
        # Method:
        # Use PyTorch layers from `torch.nn`, usually imported as `nn`.
        # `nn.Sequential(...)` can hold a small CNN.
        # Good layer choices are `nn.Conv2d(...)`, `nn.ReLU()`,
        # `nn.MaxPool2d(...)`, `nn.AdaptiveAvgPool2d((1, 1))`, `nn.Flatten()`,
        # and `nn.Linear(...)`.
        #
        # Steps:
        # 1. Build `self.image_encoder` so it returns a spatial feature map.
        # 2. Build `self.answer_pool` so the answer head gets one summary vector.
        # 3. Build `self.answer_head = nn.Linear(config.hidden_size, num_answers)`.
        # 4. Build `self.box_pool` so the box head keeps a small spatial map.
        # 5. Build `self.box_head` so it predicts 4 box coordinates.
        # 6. Save `config` on `self` if needed.
        #
        # Checks:
        # - Answer logits should have width `num_answers`.
        # - Box predictions should have exactly 4 values per image.

        raise NotImplementedError("Build grounded VLM model.")

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return answer logits and predicted boxes."""
        # Goal:
        # Predict both the answer class and the evidence box from the same image.
        #
        # Output:
        # - answer logits: `[batch_size, num_answers]`
        # - predicted boxes: `[batch_size, 4]`
        #
        # Method:
        # Use `self.image_encoder(images)` for shared spatial features.
        # The answer path pools the image down to one vector because answering
        # "what object is visible?" mostly needs object identity.
        # The box path keeps a 4 by 4 map because predicting a box needs location.
        # Use `torch.sigmoid(...)` from PyTorch on the box head output so
        # predicted coordinates stay in `[0, 1]`.
        #
        # Steps:
        # 1. Run the shared image encoder.
        # 2. Pool and flatten features for the answer head.
        # 3. Pool and flatten features for the box head, but keep a 4 by 4 grid first.
        # 4. Send the answer feature to `self.answer_head`.
        # 5. Send the box feature to `self.box_head`.
        # 6. Apply `torch.sigmoid(...)` to the box output.
        # 7. Return answer logits and predicted boxes.
        #
        # Check:
        # - Do not apply softmax to answer logits.

        raise NotImplementedError("Predict answer logits and boxes.")



def compute_grounded_loss(
    answer_logits: torch.Tensor,
    predicted_boxes: torch.Tensor,
    answer_ids: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """Combine answer loss and box loss."""
    # Goal:
    # Train the answer and evidence box together.
    #
    # Suggested loss:
    # - cross entropy for answer class
    # - L1 loss for box coordinates
    # - total = answer_loss + box_loss
    #
    # Method:
    # Use PyTorch loss classes from `torch.nn`:
    # `nn.CrossEntropyLoss()` for answer class ids and `nn.L1Loss()` for boxes.
    #
    # Steps:
    # 1. Create or call `nn.CrossEntropyLoss()` on `(answer_logits, answer_ids)`.
    # 2. Create or call `nn.L1Loss()` on `(predicted_boxes, target_boxes)`.
    # 3. Add the two losses.
    # 4. Return the scalar total loss.
    #
    # Checks:
    # - `answer_ids` should have dtype `torch.long`.
    # - `predicted_boxes.shape == target_boxes.shape`.

    raise NotImplementedError("Compute answer plus box loss.")
    # L1 loss is mean absolute error.
    # Behind the scenes it is roughly:
    # absolute_errors = torch.abs(predicted_boxes - target_boxes)
    # box_loss = absolute_errors.mean()
    # For boxes, this compares left, top, right, and bottom coordinates one by one.



def is_answer_supported(prediction: GroundedPrediction, target_answer: str, target_box: torch.Tensor, config: GroundedVLMConfig) -> bool:
    """Return True if answer and evidence box support the claim."""
    # A prediction is supported when:
    # - predicted answer equals target answer
    # - IoU(predicted box, target box) >= threshold
    #
    # Method:
    # Use the local `box_iou(...)` helper.
    # Add a batch dimension with `.unsqueeze(0)` because `box_iou` expects `[N, 4]`.
    #
    # Steps:
    # 1. Compare `prediction["answer"]` with `target_answer`.
    # 2. Convert `prediction["box"]` and `target_box` to shape `[1, 4]`.
    # 3. Compute IoU with `box_iou(pred_box, target_box)`.
    # 4. Read the scalar IoU value with `.item()`.
    # 5. Return True only if the answer matches and IoU is at least `config.iou_threshold`.
    #
    # Check:
    # - A correct answer with a bad box should be unsupported.

    raise NotImplementedError("Check whether answer is supported by the evidence box.")



def train_one_epoch(model: TinyGroundedVLM, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train one grounded VLM epoch."""
    # Goal:
    # Train answer and evidence box prediction for one epoch.
    #
    # Method:
    # Use normal PyTorch training calls:
    # `model.train()`, `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.
    #
    # Steps:
    # 1. Call `model.train()` before the loop.
    # 2. Move `batch["image"]`, `batch["answer_id"]`, and `batch["box"]` to `device`.
    # 3. Run `answer_logits, predicted_boxes = model(images)`.
    # 4. Compute loss with `compute_grounded_loss(...)`.
    # 5. Clear old gradients with `optimizer.zero_grad()`.
    # 6. Backprop with `loss.backward()`.
    # 7. Update weights with `optimizer.step()`.
    # 8. Add `loss.item()` to a running total.
    # 9. Return mean loss over examples.
    #
    # Check:
    # - Return mean loss over examples.

    raise NotImplementedError("Train one grounded VLM epoch.")




def evaluate_grounding(model: TinyGroundedVLM, examples: list[GroundedExample], config: GroundedVLMConfig) -> GroundingMetrics:
    """Evaluate answer accuracy, IoU, and support rate."""
    # Goal:
    # Measure whether answers are correct and backed by boxes.
    #
    # Method:
    # Use `model.eval()` and the PyTorch context manager `torch.no_grad()`.
    # Use `logits.argmax(dim=-1)` to get predicted answer ids.
    # Use the local `box_iou(...)` and `is_answer_supported(...)` helpers.
    #
    # Steps:
    # 1. Build `answer_to_id` and `id_to_answer` from the example answers.
    # 2. Put the model in eval mode.
    # 3. Loop over examples inside `with torch.no_grad():`.
    # 4. Add a batch dimension to each image with `.unsqueeze(0)`.
    # 5. Run the model.
    # 6. Convert the top answer id to an answer string.
    # 7. Compute answer correctness.
    # 8. Compute IoU between predicted and target box.
    # 9. Build a `GroundedPrediction` dictionary.
    # 10. Count supported predictions with `is_answer_supported(...)`.
    # 11. Return answer accuracy, mean IoU, and supported answer rate.
    #
    # Check:
    # - Supported answer rate should be no higher than answer accuracy.
    raise NotImplementedError("Evaluate grounded answers.")







def save_box_visualization(model: TinyGroundedVLM, examples: list[GroundedExample], config: GroundedVLMConfig) -> str:
    """Draw target and predicted boxes on example images."""
    # Goal:
    # Save an image grid where predicted evidence can be checked by eye.
    #
    # Method:
    # Use `config.output_dir.mkdir(parents=True, exist_ok=True)` from `pathlib.Path`.
    # Use Matplotlib: `matplotlib.use("Agg")`, `plt.subplots(...)`,
    # `ax.imshow(...)`, `matplotlib.patches.Rectangle(...)`, `ax.add_patch(...)`,
    # `plt.savefig(path)`, and `plt.close()`.
    # Use `torch.no_grad()` when running the model for boxes.
    #
    # Steps:
    # 1. Create `config.output_dir`.
    # 2. Pick a few examples.
    # 3. Run the model on each image and get predicted boxes.
    # 4. Convert normalized boxes back to image pixel positions for drawing.
    # 5. Draw the target box and predicted box in different colors.
    # 6. Put the target and predicted answer in the title.
    # 7. Save the figure and return the path as a string.
    #
    # Check:
    # - The saved image should make unsupported answers easy to spot.
    raise NotImplementedError("Save box visualization.")











def save_loss_curve(losses: list[float], config: GroundedVLMConfig) -> str:
    """Save the training loss curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "grounded_vlm_loss_curve.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = list(range(1, len(losses) + 1))
    ax.plot(epochs, losses, marker="o")
    ax.set_title("Grounded VLM training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return str(output_path)


def run_exercise(config: GroundedVLMConfig) -> GroundingMetrics:
    """Run the grounded VLM workflow."""
    # Main workflow:
    # boxed examples -> answer ids -> model with answer and box heads -> train -> support check.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_grounded_dataset(config)
    answer_vocab = {answer: index for index, answer in enumerate(sorted({example["answer"] for example in examples}))}
    dataset = GroundedDataset(examples, answer_vocab)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TinyGroundedVLM(num_answers=len(answer_vocab), config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses = []
    for i in range(config.epochs):
        loss = train_one_epoch(model, loader, optimizer)
        losses.append(loss)
        print(f"Epoch {i + 1} loss: {loss}")        

    metrics = evaluate_grounding(model, examples, config)
    box_visual_path = save_box_visualization(model, examples, config)
    loss_curve_path = save_loss_curve(losses, config)
    metrics["box_visual_path"] = box_visual_path
    metrics["loss_curve_path"] = loss_curve_path


    print(f"Metrics: {metrics}")

    return metrics


def check_metrics(metrics: GroundingMetrics) -> bool:
    return metrics["answer_accuracy"] > 0.60 and metrics["mean_iou"] > 0.30


def report_task_progress(task_name: str, task_fn, *args):
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the grounded VLM scaffold one step at a time."""
    config = GroundedVLMConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 5

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    examples, ok = report_task_progress("build_grounded_dataset", build_grounded_dataset, config)
    if not count_or_stop(ok):
        return
    answer_vocab = {answer: index for index, answer in enumerate(sorted({example["answer"] for example in examples}))}

    _, ok = report_task_progress("box_iou", box_iou, examples[0]["box"].unsqueeze(0), examples[0]["box"].unsqueeze(0))
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinyGroundedVLM.__init__", TinyGroundedVLM, len(answer_vocab), config)
    if not count_or_stop(ok):
        return

    dummy_prediction: GroundedPrediction = {"answer": examples[0]["answer"], "box": examples[0]["box"], "confidence": 1.0}
    _, ok = report_task_progress("is_answer_supported", is_answer_supported, dummy_prediction, examples[0]["answer"], examples[0]["box"], config)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "grounded support rate", metrics["supported_answer_rate"], "answer and box evidence are useful")


if __name__ == "__main__":
    main()
