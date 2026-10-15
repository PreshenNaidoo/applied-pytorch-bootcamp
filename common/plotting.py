"""Lightweight plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    *,
    title: str = "Training vs Validation Loss",
    save_path: str | Path | None = None,
) -> None:
    """Plot train and validation loss curves on one figure."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(Path(save_path), dpi=150)

    plt.show()
