"""Real-video loading shared by world-model exercises 12 and 14."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
import random
import re
import subprocess
import sys

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
JEPA_COMMIT = "51c59d518fc63c08464af6de585f78ac0c7ed4d5"


def reference_import(root: Path, commit: str, module: str):
    """Load checked reference code without mixing two projects named src."""
    root = root.resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"Missing reference checkout: {root}. Follow the exercise's setup commands.")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError(f"Reference revision is {actual}; this exercise was checked against {commit}.")
    loaded = sys.modules.get("src")
    if loaded is not None:
        locations = [Path(p).resolve() for p in getattr(loaded, "__path__", [])]
        if root / "src" not in locations:
            raise RuntimeError("Another reference project's src is already imported. Run each exercise in a fresh Python process.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(module)


@dataclass(frozen=True)
class VideoRecord:
    path: str
    label: int
    group: str


def ucf_records(video_root: Path, annotations: Path, classes: int = 10,
                train_per_class: int = 12, seed: int = 42) -> tuple[dict[str, list[VideoRecord]], list[str]]:
    """Respect the official test fold and split training by source-video group."""
    train_file, test_file = annotations / "trainlist01.txt", annotations / "testlist01.txt"
    if not video_root.is_dir() or not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Expected UCF-101 videos at {video_root} and official fold lists at {annotations}.")
    train_paths = [line.split()[0] for line in train_file.read_text().splitlines() if line.strip()]
    test_paths = [line.split()[0] for line in test_file.read_text().splitlines() if line.strip()]
    names = sorted({p.split('/')[0] for p in train_paths})
    rng = random.Random(seed); rng.shuffle(names); names = sorted(names[:classes])
    if len(names) != classes or train_per_class < 1:
        raise ValueError("Requested class count or training-video count is invalid.")
    splits: dict[str, list[VideoRecord]] = {name: [] for name in ("train", "validation", "test")}
    for label, name in enumerate(names):
        grouped: dict[str, list[str]] = {}
        for path in train_paths:
            if path.split('/')[0] != name:
                continue
            match = re.search(r"_g(\d+)_c\d+", path)
            if match is None:
                raise ValueError(f"Cannot determine UCF source group: {path}")
            grouped.setdefault(name + ":" + match[1], []).append(path)
        group_ids = sorted(grouped); rng.shuffle(group_ids)
        if len(group_ids) < 2:
            raise ValueError(f"Need at least two source groups for {name}.")
        val_ids = set(group_ids[:max(1, len(group_ids) // 5)])
        for split, chosen in (("train", [g for g in group_ids if g not in val_ids]),
                              ("validation", [g for g in group_ids if g in val_ids])):
            # Interleave groups before limiting videos, so one long group cannot dominate.
            pairs = [(g, p) for g in chosen for p in grouped[g]]
            rng.shuffle(pairs)
            limit = train_per_class if split == "train" else max(2, train_per_class // 3)
            splits[split].extend(VideoRecord(str(video_root / p), label, g) for g, p in pairs[:limit])
        paths = [p for p in test_paths if p.split('/')[0] == name]; rng.shuffle(paths)
        splits["test"].extend(VideoRecord(str(video_root / p), label, name + ":test:" + Path(p).stem)
                              for p in paths[:max(2, train_per_class // 3)])
    for records in splits.values():
        if not records:
            raise ValueError("Empty video split.")
        for record in records:
            if not Path(record.path).is_file():
                raise FileNotFoundError(record.path)
    if {r.group for r in splits["train"]} & {r.group for r in splits["validation"]}:
        raise AssertionError("Source group leaked across splits.")
    return splits, names


def decode_clip(path: str, frames: int, stride: int, training: bool = False) -> Tensor:
    """Decode distinct RGB frames; compress spacing when a clip is short."""
    if frames < 2 or stride < 1:
        raise ValueError("Need at least two frames and positive spacing.")
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot decode {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < frames:
            raise ValueError(f"{path} has {count} frames; need at least {frames} distinct frames.")
        span = min(count, (frames - 1) * stride + 1)
        start = random.randint(0, count - span) if training else (count - span) // 2
        wanted = set(np.linspace(start, start + span - 1, frames).round().astype(int).tolist())
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        selected = []
        for frame_id in range(start, start + span):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Decode failed at frame {frame_id} in {path}")
            if frame_id in wanted:
                selected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return torch.from_numpy(np.stack(selected)).permute(0, 3, 1, 2).float() / 255
    finally:
        capture.release()


def spatial_transform(clip: Tensor, image_size: int, training: bool) -> Tensor:
    """Use the same crop for every frame, preserving apparent motion."""
    if training:
        top, left, height, width = RandomResizedCrop.get_params(clip[0], scale=(0.3, 1.0), ratio=(0.75, 1.35))
        clip = TF.resized_crop(clip, top, left, height, width, [image_size, image_size], antialias=True)
        if torch.rand(()) < 0.5:
            clip = TF.hflip(clip)
    else:
        clip = TF.resize(clip, image_size, antialias=True)
        clip = TF.center_crop(clip, [image_size, image_size])
    return clip


def normalize_video(clip: Tensor) -> Tensor:
    return TF.normalize(clip, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class VideoDataset(Dataset):
    def __init__(self, records: list[VideoRecord], frames: int = 16, stride: int = 4,
                 image_size: int = 224, training: bool = False, normalize: bool = True) -> None:
        self.records, self.frames, self.stride = records, frames, stride
        self.image_size, self.training, self.normalize = image_size, training, normalize

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        clip = spatial_transform(decode_clip(record.path, self.frames, self.stride, self.training),
                                 self.image_size, self.training)
        if self.normalize:
            clip = normalize_video(clip)
        return clip.permute(1, 0, 2, 3), record.label  # C,T,H,W


def manifest(splits: dict[str, list[VideoRecord]]) -> dict:
    return {name: [asdict(record) for record in records] for name, records in splits.items()}
