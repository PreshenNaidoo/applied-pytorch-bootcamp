"""Shared real ViT and CIFAR-100 code for the distributed exercises.

Students do not need to edit this file. Keeping this code in one place makes
the distributed experiments use the same model, data, and metric rules.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.models.vision_transformer import EncoderBlock, VisionTransformer


@dataclass
class DistributedViTConfig:
    """Small run settings around a real ViT-B/16-shaped model."""

    data_root: str = str(Path(__file__).resolve().parents[2] / "data")
    image_size: int = 224
    patch_size: int = 16
    num_layers: int = 12
    num_heads: int = 12
    hidden_dim: int = 768
    mlp_dim: int = 3072
    num_classes: int = 100
    dropout: float = 0.0
    attention_dropout: float = 0.0
    batch_size_per_worker: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 2
    train_examples: int = 5_000
    validation_examples: int = 1_000
    data_loader_workers: int = 4
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DistributedViTConfig":
        known_fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known_fields})


def seed_everything(seed: int) -> None:
    """Give Python and PyTorch the same starting seed on each worker."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vit(config: DistributedViTConfig) -> VisionTransformer:
    """Build torchvision's real Vision Transformer implementation."""

    return VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        num_classes=config.num_classes,
    )


def get_vit_blocks(model: nn.Module) -> list[EncoderBlock]:
    """Find ViT transformer blocks without naming GPUs or tensor slices."""

    blocks = [module for module in model.modules() if isinstance(module, EncoderBlock)]
    if not blocks:
        raise ValueError("No torchvision EncoderBlock objects were found in the model.")
    return blocks


def _image_transforms(image_size: int, training: bool) -> transforms.Compose:
    operations: list[Any] = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


def _fixed_subset(dataset: Dataset[Any], size: int, seed: int) -> Dataset[Any]:
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_cifar100_datasets(
    config: DistributedViTConfig,
    *,
    download: bool,
) -> tuple[Dataset[Any], Dataset[Any]]:
    """Load real CIFAR-100 train and test images, then take fixed subsets."""

    train_dataset = CIFAR100(
        root=config.data_root,
        train=True,
        transform=_image_transforms(config.image_size, training=True),
        download=download,
    )
    validation_dataset = CIFAR100(
        root=config.data_root,
        train=False,
        transform=_image_transforms(config.image_size, training=False),
        download=download,
    )
    train_dataset = _fixed_subset(train_dataset, config.train_examples, config.seed)
    validation_dataset = _fixed_subset(
        validation_dataset,
        config.validation_examples,
        config.seed + 1,
    )
    return train_dataset, validation_dataset


def build_data_loaders(
    config: DistributedViTConfig,
    *,
    download: bool,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build normal loaders; Ray will add distributed samplers later."""

    train_dataset, validation_dataset = build_cifar100_datasets(
        config,
        download=download,
    )
    loader_kwargs = {
        "batch_size": config.batch_size_per_worker,
        "num_workers": config.data_loader_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.data_loader_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    return train_loader, validation_loader


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def patch_tokens_per_image(config: DistributedViTConfig) -> int:
    patches_per_side = config.image_size // config.patch_size
    return patches_per_side * patches_per_side + 1


def worker_can_download() -> bool:
    """Let local rank 0 download once when workers share a filesystem."""

    return int(os.environ.get("LOCAL_RANK", "0")) == 0
