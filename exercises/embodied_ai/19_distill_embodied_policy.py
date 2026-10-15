"""
============================================================
EXERCISE 19: DISTILL EMBODIED POLICY
============================================================

WHAT YOU ARE BUILDING
You are distilling a larger embodied policy into a smaller policy.

Plain meaning:

```text
large image-language policy -> soft action targets -> small edge policy
```

Soft action targets means the teacher gives a full probability-like pattern,
not only one correct action.
For example:

```text
teacher logits: [3.0, 1.0, -2.0, 0.5]
teacher mostly prefers action 0, but action 1 is still more reasonable than action 2
```

The student learns that softer pattern.

WHY THIS EXERCISE EXISTS
the related exercise teaches distillation on image classification.
This exercise applies the same idea to an embodied agent, where the final check
is not only action accuracy but also rollout behavior.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
large VLA or VLM policy             larger small image-language policy
small deployable policy             smaller CNN/text policy
device latency                      local CPU latency
task rollout score                  grid-world or 3D rollout success

DATASET AND DOWNLOADS
Default dataset:
- no download
- start with 2D VLA rollouts from `common.embodied_gridworld`
- after the 3D chain works, use `outputs/mini3d_vla_dataset.pt`

Teacher:
- first version: train a larger image-language policy like the related exercise or 129
- later version: load a stronger pretrained vision-language policy if you have one

Student:
- smaller CNN/text policy
- lower parameter count and lower latency than the teacher

INPUTS AND OUTPUTS
- images: `[batch_size, 3, image_size, image_size]`
- instruction ids: `[batch_size, text_length]`
- hard actions: `[batch_size]`
- teacher logits: `[batch_size, num_actions]`
- student logits: `[batch_size, num_actions]`

LOSS FUNCTION

```text
soft_loss = KL(softmax(teacher_logits / temperature),
               softmax(student_logits / temperature))
hard_loss = cross_entropy(student_logits, hard_actions)
total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
```

Plain meaning:
- soft loss teaches the student to copy the teacher's full action pattern
- hard loss keeps the student tied to the real logged action
- `temperature` makes the teacher pattern softer
- `alpha` controls how much the student listens to the teacher

METRICS TO TRACK
- teacher action accuracy
- student action accuracy
- teacher rollout success rate
- student rollout success rate
- success drop: `teacher_success_rate - student_success_rate`
- student size in MB
- student latency in ms
- edge passed: true only if success drop, size, and latency are acceptable

VISUALS TO CREATE
Save `outputs/embodied_distillation/teacher_student_comparison.png`.
It should compare teacher and student accuracy, success rate, size, and latency.

DEFINITION OF DONE
You are done when:
- a teacher policy is trained or loaded
- a smaller student policy learns from teacher logits
- teacher and student action accuracy are compared
- teacher and student rollout success are compared
- size and latency are measured
- the accuracy and success drop are reported
- a teacher-vs-student comparison chart is saved
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn


class EmbodiedDistillationMetrics(TypedDict):
    """Distillation metrics for embodied policies."""

    teacher_action_accuracy: float
    student_action_accuracy: float
    teacher_success_rate: float
    student_success_rate: float
    student_size_mb: float
    student_latency_ms: float
    success_drop: float
    edge_passed: bool


@dataclass
class EmbodiedPolicyDistillationConfig:
    """Settings for embodied policy distillation."""

    image_size: int = 28
    channels: int = 3
    num_actions: int = 4
    temperature: float = 2.0
    alpha: float = 0.7
    max_success_drop: float = 0.10
    max_model_size_mb: float = 2.0
    max_latency_ms: float = 10.0
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    random_state: int = 42


class EmbodiedStudentPolicy(nn.Module):
    """Small policy for edge-style embodied deployment."""

    def __init__(self, config: EmbodiedPolicyDistillationConfig) -> None:
        # Where this fits:
        # This is the smaller model that should be cheaper to run near a camera
        # or inside a real-time agent loop.
        #
        # Goal:
        # Build a smaller policy than the teacher.
        #
        # Suggested model:
        # - small CNN for images
        # - `nn.Embedding(...)` for instruction ids
        # - concatenate image and text features
        # - final `nn.Linear(..., config.num_actions)`
        super().__init__()
        _ = config
        raise NotImplementedError("Build the embodied student policy.")

    def forward(self, images: torch.Tensor, instruction_ids: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        # Input:
        # - `images`: `[batch_size, 3, image_size, image_size]`
        # - `instruction_ids`: `[batch_size, text_length]`
        #
        # Output:
        # - action logits shaped `[batch_size, num_actions]`
        #
        # Check:
        # - Return raw logits. Do not apply softmax here.
        _ = (images, instruction_ids)
        raise NotImplementedError("Run the embodied student policy.")


def compute_policy_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_actions: torch.Tensor,
    config: EmbodiedPolicyDistillationConfig,
) -> torch.Tensor:
    """Combine teacher soft-action loss and hard behavior-cloning loss."""
    # Goal:
    # Use the same idea as the related exercise, but logits are action scores.
    #
    # Formula:
    # `alpha * KL(teacher_actions || student_actions) + (1 - alpha) * CE(student_logits, hard_actions)`
    #
    # Method:
    # 1. Read `temperature = config.temperature`.
    # 2. Compute teacher probabilities:
    #    `torch.softmax(teacher_logits / temperature, dim=-1)`.
    # 3. Compute student log probabilities:
    #    `torch.log_softmax(student_logits / temperature, dim=-1)`.
    # 4. Compute KL loss with `nn.KLDivLoss(reduction="batchmean")`.
    # 5. Multiply the KL loss by `temperature ** 2`.
    # 6. Compute hard loss with `nn.functional.cross_entropy(student_logits, hard_actions)`.
    # 7. Return the weighted sum using `config.alpha`.
    _ = (student_logits, teacher_logits, hard_actions, config)
    raise NotImplementedError("Compute embodied policy distillation loss.")


def train_distilled_embodied_policy(config: EmbodiedPolicyDistillationConfig) -> EmbodiedDistillationMetrics:
    """Train and evaluate a distilled embodied policy."""
    # Goal:
    # Distill the teacher, then test both policies in the environment.
    #
    # Method:
    # 1. Build or load logged VLA data.
    # 2. Train or load the teacher policy.
    # 3. Freeze the teacher with `teacher.eval()`.
    # 4. For each batch:
    #    - get teacher logits inside `with torch.no_grad():`
    #    - get student logits normally
    #    - compute distillation loss
    #    - update only the student
    # 5. Evaluate teacher and student action accuracy.
    # 6. Run both policies in the real environment.
    # 7. Measure student size and latency.
    # 8. Save the teacher-vs-student comparison chart.
    #
    # Checks:
    # - Report rollout success, not only action accuracy.
    # - Report latency and size so this connects back to deployment.
    raise NotImplementedError("Train and evaluate distilled embodied policy.")


def main() -> None:
    """Run the embodied policy distillation scaffold."""
    config = EmbodiedPolicyDistillationConfig()
    try:
        print(train_distilled_embodied_policy(config))
    except NotImplementedError as error:
        print(f"[TODO] train_distilled_embodied_policy: {error}")


if __name__ == "__main__":
    main()
