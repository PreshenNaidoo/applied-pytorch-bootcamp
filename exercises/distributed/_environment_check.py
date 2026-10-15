"""Check the software and GPUs needed by the distributed ViT exercises."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    problems: list[str] = []

    print("Python:", sys.version.split()[0])
    print("RAY_TRAIN_V2_ENABLED:", os.environ.get("RAY_TRAIN_V2_ENABLED", "not set"))
    print("Ray:", package_version("ray"))
    print("PyTorch:", package_version("torch"))
    print("torchvision:", package_version("torchvision"))

    try:
        import torch

        print("CUDA available:", torch.cuda.is_available())
        print("visible CUDA GPUs:", torch.cuda.device_count())
        print("NCCL available:", torch.distributed.is_nccl_available())

        if not torch.cuda.is_available():
            problems.append("PyTorch cannot use CUDA.")
        if torch.cuda.device_count() < 2:
            problems.append("The first six exercises expect at least two visible GPUs.")
        if not torch.distributed.is_nccl_available():
            problems.append("This PyTorch build does not include NCCL.")

        try:
            from torch.distributed.fsdp import fully_shard  # noqa: F401
        except ImportError:
            problems.append(
                "PyTorch does not expose torch.distributed.fsdp.fully_shard. "
                "Install PyTorch 2.8 or newer."
            )
    except ImportError:
        problems.append("PyTorch is not installed.")

    try:
        from ray.train.torch import TorchTrainer  # noqa: F401
    except ImportError:
        problems.append('Ray Train is missing. Install "ray[train]>=2.55.1".')

    if os.environ.get("RAY_TRAIN_V2_ENABLED") != "1":
        problems.append("Set RAY_TRAIN_V2_ENABLED=1 before starting Python.")

    if problems:
        print("\nNot ready:")
        for problem in problems:
            print("-", problem)
        raise SystemExit(1)

    print("\nReady for the distributed ViT exercises.")


if __name__ == "__main__":
    main()
