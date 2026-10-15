"""Small data helpers for synthetic vector, vision, and text exercises."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def train_val_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> dict[str, np.ndarray]:
    """Split arrays into train, validation, and test parts.

    The validation size is taken from the non-test portion.
    """
    stratify_labels: Any = labels if stratify else None

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    stratify_train_val: Any = y_train_val if stratify else None

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def numpy_features_to_tensor(features: np.ndarray) -> torch.Tensor:
    """Convert numeric features to a float32 tensor."""
    return torch.tensor(features, dtype=torch.float32)



def numpy_labels_to_long_tensor(labels: np.ndarray) -> torch.Tensor:
    """Convert class labels to an int64 tensor."""
    return torch.tensor(labels, dtype=torch.long)



def numpy_labels_to_float_tensor(labels: np.ndarray) -> torch.Tensor:
    """Convert binary labels to a float32 tensor."""
    return torch.tensor(labels, dtype=torch.float32)



def remap_numpy_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Remap NumPy class ids with a simple lookup table."""
    return np.array([mapping[int(label)] for label in labels], dtype=np.int64)



def remap_tensor_labels(labels: torch.Tensor, mapping: dict[int, int]) -> torch.Tensor:
    """Remap tensor class ids with a simple lookup table."""
    remapped = [mapping[int(label)] for label in labels.detach().cpu().tolist()]
    return torch.tensor(remapped, dtype=labels.dtype)



def build_class_filter_mask(labels: np.ndarray | torch.Tensor, class_ids: list[int] | tuple[int, ...]) -> np.ndarray | torch.Tensor:
    """Return a mask that keeps only selected class ids."""
    if isinstance(labels, np.ndarray):
        return np.isin(labels, list(class_ids))

    mask = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in class_ids:
        mask |= labels == int(class_id)
    return mask



def filter_numpy_arrays_by_class_ids(
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the rows whose label is in class_ids."""
    keep_mask = build_class_filter_mask(labels, class_ids)
    return features[keep_mask], labels[keep_mask]



def filter_tensor_arrays_by_class_ids(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_ids: list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only the rows whose label is in class_ids."""
    keep_mask = build_class_filter_mask(labels, class_ids)
    return features[keep_mask], labels[keep_mask]



def describe_basic_vision_transform_rules() -> list[str]:
    """Return short notes about common vision transform rules."""
    return [
        "Normalize image tensors after you turn them into float tensors.",
        "If you resize images and masks together for segmentation, use bilinear for images and nearest-neighbor for masks.",
        "For BCEWithLogitsLoss, keep the model output as raw logits. Do not apply sigmoid before the loss.",
        "For CrossEntropyLoss, keep the model output as raw logits. Do not apply softmax before the loss.",
        "Images usually use shape [batch_size, channels, height, width].",
    ]



def whitespace_tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Split text into simple whitespace tokens.

    This helper is intentionally simple.
    It is good enough for the beginner text exercises in this repo.
    """
    if lowercase:
        text = text.lower()
    return text.strip().split()



def build_token_to_id(
    token_lists: list[list[str]],
    *,
    min_frequency: int = 1,
    special_tokens: list[str] | None = None,
) -> dict[str, int]:
    """Build a small vocabulary mapping from token lists.

    The ids are assigned in this order:
    1. special tokens first
    2. then normal tokens in alphabetical order
    """
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>"]

    counts: dict[str, int] = {}
    for tokens in token_lists:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

    vocab: dict[str, int] = {}
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    kept_tokens = sorted(token for token, count in counts.items() if count >= min_frequency)
    for token in kept_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab



def encode_tokens(tokens: list[str], token_to_id: dict[str, int], unknown_token: str = "<unk>") -> list[int]:
    """Turn tokens into integer ids."""
    unknown_id = token_to_id[unknown_token]
    return [token_to_id.get(token, unknown_id) for token in tokens]



def pad_or_truncate_ids(
    token_ids: list[int],
    max_length: int,
    pad_id: int,
) -> list[int]:
    """Return a fixed-length token-id list.

    If the sequence is too long, keep only the first `max_length` ids.
    If the sequence is too short, add `pad_id` at the end.
    """
    clipped = token_ids[:max_length]
    if len(clipped) < max_length:
        clipped = clipped + [pad_id] * (max_length - len(clipped))
    return clipped



def lengths_to_padding_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Build a boolean mask from sequence lengths.

    Returns:
    - type: `torch.Tensor`
    - dtype: `torch.bool`
    - shape: `[batch_size, max_length]`

    Meaning:
    - `True` means this position holds a real token
    - `False` means this position is padding
    """
    if max_length is None:
        max_length = int(lengths.max().item())
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)



def make_causal_next_token_pairs(token_ids: list[int], context_length: int) -> tuple[list[list[int]], list[list[int]]]:
    """Create input and target windows for next-token prediction."""
    if len(token_ids) <= context_length:
        return [], []

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    last_start = len(token_ids) - context_length - 1
    for start_index in range(last_start + 1):
        input_window = token_ids[start_index : start_index + context_length]
        target_window = token_ids[start_index + 1 : start_index + context_length + 1]
        inputs.append(input_window)
        targets.append(target_window)
    return inputs, targets



def describe_text_batch_rules() -> list[str]:
    """Return short notes about common text batching rules."""
    return [
        "Token ids usually use dtype torch.long.",
        "For nn.Embedding, the input should be integer token ids, not floats.",
        "For CrossEntropyLoss on class labels, keep the model output as raw logits.",
        "For padded sequence batches, keep track of which positions are real tokens and which are padding.",
        "For language modeling, inputs are the current tokens and targets are the next tokens.",
    ]
