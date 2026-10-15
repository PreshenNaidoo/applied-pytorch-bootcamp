"""Small metric helpers used across the bootcamp.

These are intentionally lightweight.
They help you measure results without giving away the exercise logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class AverageMeter:
    """Keep a running average of scalar values.

    Example:
        meter = AverageMeter()
        meter.update(0.5, count=32)
        meter.update(0.4, count=32)
        print(meter.average)
    """

    total: float = 0.0
    count: int = 0

    def update(self, value: float, count: int = 1) -> None:
        """Add one more value or a batch of values."""
        self.total += float(value) * count
        self.count += int(count)

    @property
    def average(self) -> float:
        """Return the current average.

        Returns 0.0 if nothing has been added yet.
        """
        if self.count == 0:
            return 0.0
        return self.total / self.count



def binary_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute binary accuracy from raw logits.

    This expects:
    - `logits` to contain raw model outputs
    - `targets` to contain 0/1 labels
    """
    predicted = (logits >= 0).to(targets.dtype)
    correct = (predicted.view_as(targets) == targets).float()
    return float(correct.mean().item())



def binary_predictions_from_logits(logits: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Turn raw binary logits into 0/1 predictions."""
    return (logits >= threshold).to(torch.long)



def binary_accuracy_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute binary accuracy from probabilities in the range [0, 1]."""
    predicted = (probabilities >= threshold).to(targets.dtype)
    correct = (predicted.view_as(targets) == targets).float()
    return float(correct.mean().item())



def multiclass_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute multiclass accuracy from raw class logits."""
    predicted = logits.argmax(dim=1)
    correct = (predicted == targets).float()
    return float(correct.mean().item())



def top1_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy from raw class logits."""
    return multiclass_accuracy_from_logits(logits, targets)



def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return the mean over only the positions where mask is True."""
    masked_values = values * mask.to(values.dtype)
    denominator = mask.to(values.dtype).sum().clamp_min(eps)
    return masked_values.sum() / denominator



def token_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute token accuracy for sequence or language-model logits.

    Expected shapes:
    - `logits`: `[batch_size, sequence_length, vocab_size]`
    - `targets`: `[batch_size, sequence_length]`
    """
    predictions = logits.argmax(dim=-1)
    valid_mask = targets != ignore_index

    if valid_mask.sum().item() == 0:
        return 0.0

    correct = (predictions == targets) & valid_mask
    return float(correct.float().sum().item() / valid_mask.float().sum().item())



def perplexity_from_loss(loss: float | torch.Tensor) -> float:
    """Turn mean cross-entropy loss into perplexity.

    Perplexity is `exp(loss)`.
    Lower is better.
    A perplexity of 1.0 would mean perfect next-token prediction.
    """
    loss_value = to_float(loss)
    return float(math.exp(loss_value))



def _prepare_binary_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make two mask tensors easy to compare."""
    if predictions.ndim == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]

    predictions = predictions.float().reshape(predictions.shape[0], -1)
    targets = targets.float().reshape(targets.shape[0], -1)
    return predictions, targets



def dice_score_from_masks(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute Dice score for binary masks.

    Dice score is between 0 and 1.
    - 1.0 means perfect overlap
    - 0.0 means no overlap
    """
    predictions, targets = _prepare_binary_masks(predictions, targets)
    intersection = (predictions * targets).sum(dim=1)
    totals = predictions.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (totals + eps)
    return float(dice.mean().item())



def iou_score_from_masks(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute IoU score for binary masks.

    IoU means intersection over union.
    """
    predictions, targets = _prepare_binary_masks(predictions, targets)
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())



def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> float:
    """Compute Dice score from raw binary segmentation logits."""
    predictions = (logits >= threshold).float()
    return dice_score_from_masks(predictions, targets)



def iou_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> float:
    """Compute IoU score from raw binary segmentation logits."""
    predictions = (logits >= threshold).float()
    return iou_score_from_masks(predictions, targets)



def to_float(value: float | torch.Tensor) -> float:
    """Convert a scalar tensor or Python number to a float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
