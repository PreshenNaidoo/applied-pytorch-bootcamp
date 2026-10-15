"""
============================================================
EXERCISE 12: VIDEO FRAME SAMPLING AND EVENT EVAL
============================================================

WHAT YOU ARE BUILDING
You are building a tiny video evaluation setup.

Video is harder than image work because time matters.

An image question asks:

```text
what is visible now?
```

A video question may ask:

```text
what happened, and when did it happen?
```

Use COIN clips.

WHY THIS EXERCISE EXISTS
Bodycam-style data is video, not just still images.
A model can understand sampled frames and still miss a short event between them.

HOW THIS FITS INTO THE INTERVIEW PREP
This helps you explain why video evaluation needs time-aware checks.
It also gives you a concrete way to talk about frame sampling, missed events,
and temporal IoU.

REAL SYSTEM VS THIS EXERCISE

real system                                         this exercise
---------------------------------------------------------------
bodycam video                                       small COIN instructional video sample
event in real time                                  annotated procedure step
frame sampling                                      select frames from the clip
event localization model (VideoMAE or TimeSformer)  predict start and end frame
missed short events                                 step may be missed by sparse sampling

DATASET SETUP
This exercise uses the real COIN dataset, not fake events.

COIN is an instructional video dataset. Each video has real step labels and
real start/end times, such as "remove the tire" from 35.2s to 47.8s.

Download:
- annotations: https://github.com/coin-dataset/annotations
- dataset page: https://coin-dataset.github.io/

Put files here:

```text
data/coin/COIN.json
data/coin/videos/<youtube_id>.mp4
```

Example:

```text
data/coin/videos/LtRSn-ntcLY.mp4
```

The official COIN page links a script for downloading videos from YouTube.
Some YouTube videos may be unavailable now. That is normal for online video
datasets. This exercise skips missing local videos and uses the clips you have.

VISUALS TO CREATE
Save:
- sampled frame grid
- timeline showing target event and predicted event
- bar chart of event detection by sampling rate

INPUTS AND OUTPUTS
Video clip:
- `[clip_length, 3, image_size, image_size]`

Sampled clip:
- `[sampled_frames, 3, image_size, image_size]`

Target event:
- `event_start`
- `event_end`
- both are frame indices

Model output:
- step category
- predicted start and end frame

METRICS TO TRACK
- step classification accuracy
- mean temporal IoU
- missed short event rate

STEP-BY-STEP TODO
1. Load COIN annotations and local COIN video clips.
2. Sample a fixed number of frames from each clip.
3. Show how sampling can miss short events.
4. Build a small frame encoder.
5. Predict which COIN step is happening.
6. Predict approximate start and end frames.
7. Train the model.
8. Compute temporal IoU.
9. Measure missed short-event rate.
10. Save sampled frames and timeline visuals.

DEFINITION OF DONE
You are done when:
- clips are real frame tensors
- frame sampling can miss short events
- temporal IoU is computed
- at least one timeline visualization is saved
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
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
class VideoEventConfig:
    """Settings for tiny video event evaluation."""

    image_size: int = 48
    clip_length: int = 16
    sampled_frames: int = 4
    hidden_size: int = 64
    num_event_classes: int = 12
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 3e-3
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    coin_annotation_path: Path = PROJECT_ROOT / "data" / "coin" / "COIN.json"
    coin_video_root: Path = PROJECT_ROOT / "data" / "coin" / "videos"
    coin_video_size_index_path: Path = PROJECT_ROOT / "data" / "coin" / "video_size_index.json"
    sample_size: int = 10
    download_video_count: int = 128
    context_seconds: float = 4.0


class VideoEventExample(TypedDict):
    """One video clip with an event window."""

    frames: torch.Tensor
    event_start: int
    event_end: int
    event_label: int
    event_text: str
    video_id: str


class VideoEventMetrics(TypedDict):
    """Video event summary."""

    event_accuracy: float
    mean_temporal_iou: float
    missed_short_event_rate: float
    frame_grid_path: str
    timeline_path: str


class VideoEventDataset(Dataset):
    """Dataset of sampled clips and event windows."""

    def __init__(self, examples: list[VideoEventExample], config: VideoEventConfig):
        self.examples = examples
        self.config = config
        self.sample_indices = sample_frame_indices(config.clip_length, config.sampled_frames)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        sampled = example["frames"][self.sample_indices]
        return {
            "sampled_frames": sampled,
            "event_label": torch.tensor(example["event_label"], dtype=torch.long),
            "event_window": torch.tensor([example["event_start"], example["event_end"]], dtype=torch.float32),
        }


def build_coin_video_size_index(config: VideoEventConfig) -> list[dict[str, int | str]]:
    """Write downloaded COIN video ids ordered from smallest file to largest."""
    # Goal:
    # Make loading faster by trying small local videos first.
    #
    # Input:
    # - `config.coin_video_root`: folder with downloaded `.mp4` files
    # - `config.coin_video_size_index_path`: output JSON path
    #
    # Output:
    # - a list like:
    #   `{"video_id": "FTcnD1etGKs", "file_name": "FTcnD1etGKs.mp4", "size_bytes": 75860}`
    #
    # Method:
    # Use `Path.glob("*.mp4")` to find videos.
    # Use `path.stat().st_size` to get each file size in bytes.
    # Sort by `size_bytes` from low to high.
    #
    # Check:
    # - The first item should be the smallest local video file.
    if not config.coin_video_root.exists():
        raise FileNotFoundError(f"COIN video folder not found: {config.coin_video_root}")

    rows = [
        {
            "video_id": path.stem,
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
        }
        for path in config.coin_video_root.glob("*.mp4")
    ]
    rows.sort(key=lambda row: int(row["size_bytes"]))

    config.coin_video_size_index_path.parent.mkdir(parents=True, exist_ok=True)
    config.coin_video_size_index_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def download_coin_videos(config: VideoEventConfig, max_videos: int | None = None) -> list[Path]:
    """Download a small local COIN video set for this exercise."""
    # Where this fits:
    # This helper prepares the real video files used by `build_video_event_dataset`.
    #
    # Goal:
    # Download about 128 COIN videos from the local `COIN.json` file.
    #
    # Important:
    # COIN points to YouTube videos. Some videos may be gone now.
    # That is normal for older online video datasets.
    # This function keeps going when one video fails.
    #
    # Input:
    # - `config.coin_annotation_path`: path to `COIN.json`
    # - `config.coin_video_root`: folder where `.mp4` files are saved
    # - `max_videos`: how many videos to try to keep locally
    #
    # Output:
    # - list of local `.mp4` paths that exist after the download pass
    #
    # Method:
    # Use `yt-dlp` through Python.
    #
    # Setup command:
    # `uv pip install yt-dlp`
    #
    # Steps:
    # 1. Read `COIN.json`.
    # 2. Pick videos from the training split when possible.
    # 3. Prefer videos whose step ids fit the first `config.num_event_classes`
    #    classes used by the exercise.
    # 4. Save videos as `data/coin/videos/<youtube_id>.mp4`.
    # 5. Skip videos that already exist.
    # 6. Continue if one YouTube video is unavailable.
    #
    # Check:
    # - After this runs, `data/coin/videos` should contain `.mp4` files.
    if not config.coin_annotation_path.exists():
        raise FileNotFoundError(f"COIN annotation file not found: {config.coin_annotation_path}")

    if importlib.util.find_spec("yt_dlp") is None:
        raise RuntimeError("Install yt-dlp first. Run: uv pip install yt-dlp")

    target_count = max_videos if max_videos is not None else config.download_video_count
    config.coin_video_root.mkdir(parents=True, exist_ok=True)

    coin_data = json.loads(config.coin_annotation_path.read_text(encoding="utf-8"))
    database = coin_data["database"]
    selected_step_ids: set[str] = set()
    selected_video_ids: list[str] = []

    sorted_items = sorted(database.items(), key=lambda item: item[1].get("subset") != "training")
    for video_id, video_info in sorted_items:
        annotations = video_info.get("annotation", [])
        if not annotations:
            continue

        video_step_ids = {str(annotation["id"]) for annotation in annotations}
        if len(selected_step_ids) < config.num_event_classes:
            remaining_slots = config.num_event_classes - len(selected_step_ids)
            selected_step_ids.update(sorted(video_step_ids)[:remaining_slots])

        if selected_step_ids.isdisjoint(video_step_ids):
            continue

        selected_video_ids.append(video_id)
        if len(selected_video_ids) >= target_count:
            break

    downloaded_paths: list[Path] = []
    for video_id in selected_video_ids:
        output_path = config.coin_video_root / f"{video_id}.mp4"
        if output_path.exists():
            downloaded_paths.append(output_path)
            continue

        url = f"https://www.youtube.com/watch?v={video_id}"
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "best[height<=480][ext=mp4]/best[height<=480]/best[ext=mp4]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            str(config.coin_video_root / f"{video_id}.%(ext)s"),
            url,
        ]
        try:
            result = subprocess.run(command, check=False)
        except FileNotFoundError as error:
            raise RuntimeError("Install yt-dlp first. Run: uv pip install yt-dlp") from error

        if result.returncode == 0 and output_path.exists():
            downloaded_paths.append(output_path)
        else:
            print(f"Skipping unavailable COIN video: {video_id}")

    return downloaded_paths


def build_video_event_dataset(config: VideoEventConfig) -> list[VideoEventExample]:
    """Build video clips with event windows or labels."""
    # Where this fits:
    # This is the video world for the exercise.
    # Later helpers sample from these clips and test whether the event was found.
    #
    # Goal:
    # Return real frame sequences with real step labels and real event windows.
    #
    # Input/output:
    # - `frames`: `[clip_length, 3, image_size, image_size]`
    # - `event_start`: first frame index where the event is active
    # - `event_end`: first frame index after the event ends
    #
    # Dataset:
    # Use COIN annotations and local COIN videos.
    # COIN gives step labels and start/end times, so the event window is real.
    #
    # Output:
    # Return a list of dictionaries with:
    # - `"frames"`: tensor shaped `[clip_length, 3, image_size, image_size]`
    # - `"event_label"`: compact integer id for the COIN step label
    # - `"event_start"` and `"event_end"`: frame indices
    # - `"event_text"`: human-readable COIN step text
    # - `"video_id"`: YouTube id used by COIN
    #
    # Method:
    # Use `json.load(...)` from the Python standard library to read COIN.json.
    # Use `torchvision.io.read_video(...)` to read local `.mp4` files.
    # Use `torchvision.transforms.Resize(...)` for frame resizing.
    #
    # Steps:
    # 1. Read `config.coin_annotation_path`.
    # 2. Read the size-ordered video list from `config.coin_video_size_index_path`.
    # 3. Use the first `config.sample_size` video ids from that list.
    # 4. Look up each video id in the COIN `"database"` field.
    # 5. Look for a local video file named `<video_id>.mp4`.
    # 6. For each annotated step, read a short clip around the step.
    # 7. Convert raw video frames from `[T, H, W, C]` to `[T, C, H, W]`.
    # 8. Convert byte pixels to float pixels in `[0, 1]`.
    # 9. Resize frames to `[3, image_size, image_size]`.
    # 10. Pad, crop, or sample each clip to exactly `config.clip_length` frames.
    # 11. Convert the COIN step start/end seconds to frame indices in the clip.
    # 12. Build compact label ids for the first `config.num_event_classes` step ids.
    #
    # Checks:
    # - Every example must have the same frame tensor shape.
    # - `0 <= event_start <= event_end <= config.clip_length`.

    if not config.coin_annotation_path.exists():
        raise FileNotFoundError(
            "COIN annotations were not found. Download COIN.json from "
            "https://github.com/coin-dataset/annotations and place it at "
            f"{config.coin_annotation_path}."
        )

    if not config.coin_video_root.exists():
        raise FileNotFoundError(
            "COIN videos were not found. Put local .mp4 files in "
            f"{config.coin_video_root}. Each file should be named with its COIN YouTube id, "
            "for example LtRSn-ntcLY.mp4."
        )

    if not config.coin_video_size_index_path.exists():
        build_coin_video_size_index(config)

    resize = torchvision.transforms.Resize((config.image_size, config.image_size), antialias=True)
    coin_data = json.loads(config.coin_annotation_path.read_text(encoding="utf-8"))
    database = coin_data["database"]
    size_rows = json.loads(config.coin_video_size_index_path.read_text(encoding="utf-8"))
    ordered_video_ids = [str(row["video_id"]) for row in size_rows[: config.sample_size]]
    examples: list[VideoEventExample] = []
    label_to_id: dict[str, int] = {}

    def prepare_frames(video: torch.Tensor) -> torch.Tensor:
        """Convert raw video frames to `[clip_length, 3, image_size, image_size]`."""
        frames = torch.as_tensor(video)
        if frames.ndim != 4 or frames.shape[-1] not in (1, 3):
            raise ValueError(f"Expected video frames shaped [T, H, W, C], got {tuple(frames.shape)}.")

        frames = frames.permute(0, 3, 1, 2).float()
        if frames.max() > 1.0:
            frames = frames / 255.0
        if frames.shape[1] == 1:
            frames = frames.repeat(1, 3, 1, 1)

        frames = resize(frames)

        if frames.shape[0] == config.clip_length:
            return frames.contiguous()
        if frames.shape[0] < config.clip_length:
            pad_count = config.clip_length - frames.shape[0]
            padding = frames[-1:].repeat(pad_count, 1, 1, 1)
            return torch.cat([frames, padding], dim=0).contiguous()

        indices = torch.linspace(0, frames.shape[0] - 1, steps=config.clip_length).round().long()
        return frames[indices].contiguous()

    def time_to_frame_index(time_seconds: float, clip_start: float, clip_end: float) -> int:
        """Map one time in seconds to one frame index in the fixed-length clip."""
        if clip_end <= clip_start:
            return 0
        fraction = (time_seconds - clip_start) / (clip_end - clip_start)
        index = round(fraction * config.clip_length)
        return max(0, min(config.clip_length, index))
    

    # Begin
    count = 0
    for video_id in ordered_video_ids:
        if video_id not in database:
            continue

        video_info = database[video_id]
        video_path = config.coin_video_root / f"{video_id}.mp4"
        if not video_path.exists():
            continue
        
        print(f"Loading video :{count}")
        count+=1

        for annotation in video_info.get("annotation", []):
            step_id = str(annotation["id"])
            if step_id not in label_to_id:
                if len(label_to_id) >= config.num_event_classes:
                    continue
                label_to_id[step_id] = len(label_to_id)

            event_start_sec, event_end_sec = annotation["segment"]
            clip_start_sec = max(float(video_info["start"]), float(event_start_sec) - config.context_seconds)
            clip_end_sec = min(float(video_info["end"]), float(event_end_sec) + config.context_seconds)
            if clip_end_sec <= clip_start_sec:
                continue

            video, _, _ = torchvision.io.read_video(
                str(video_path),
                start_pts=clip_start_sec,
                end_pts=clip_end_sec,
                pts_unit="sec",
                output_format="THWC",
            )
            if video.numel() == 0:
                continue

            event_start = time_to_frame_index(float(event_start_sec), clip_start_sec, clip_end_sec)
            event_end = time_to_frame_index(float(event_end_sec), clip_start_sec, clip_end_sec)
            event_end = max(event_start + 1, event_end)
            event_end = min(config.clip_length, event_end)

            examples.append(
                {
                    "frames": prepare_frames(video),
                    "event_label": label_to_id[step_id],
                    "event_start": event_start,
                    "event_end": event_end,
                    "event_text": str(annotation["label"]),
                    "video_id": video_id,
                }
            )

    if not examples:
        raise ValueError(
            "No usable COIN examples were found. Check that COIN.json is present and that "
            "at least a few matching .mp4 files exist in data/coin/videos."
        )

    return examples


def sample_frame_indices(clip_length: int, sampled_frames: int) -> list[int]:
    """Return frame indices to inspect."""
    # Goal:
    # Choose evenly spaced frames.
    #
    # Important:
    # A short event can happen between sampled frames.
    # This is one reason video evaluation is harder than image evaluation.
    #
    # Input:
    # - `clip_length`: total number of frames in the clip.
    # - `sampled_frames`: number of frames the model will see.
    #
    # Output:
    # - list of integer frame indices.
    #
    # Method:
    # Use `torch.linspace` from PyTorch.
    # Exact call shape:
    # `torch.linspace(0, clip_length - 1, steps=sampled_frames).round().long().tolist()`.
    # `torch.linspace(...)` creates evenly spaced values, 
    # `.round()` makes them frame numbers, 
    # `.long()` makes integer tensors, and `.tolist()` returns Python ints.
    #
    # Checks:
    # - The first index should be at least 0.
    # - The last index should be less than `clip_length`.
    # - The list length should equal `sampled_frames`.

    indexes_to_select = torch.linspace(0, clip_length - 1, steps=sampled_frames).round().long().tolist()

    # Example:
    # If `clip_length == 16` and `sampled_frames == 4`,
    # this returns `[0, 5, 10, 15]`.
    return indexes_to_select
    #raise NotImplementedError("Sample frame indices.")


def temporal_iou(pred_start: int, pred_end: int, target_start: int, target_end: int) -> float:
    """Compute temporal IoU for event windows."""
    # Formula:
    # overlap length / union length
    #
    # Input:
    # - predicted start/end frame indices
    # - target start/end frame indices
    #
    # Method:
    # Use normal Python `max(...)` and `min(...)` because these inputs are ints.
    #
    # Steps:
    # 1. Compute `overlap_start = max(pred_start, target_start)`.
    # 2. Compute `overlap_end = min(pred_end, target_end)`.
    # 3. Compute `overlap = max(0, overlap_end - overlap_start)`.
    # 4. Compute `union_start = min(pred_start, target_start)`.
    # 5. Compute `union_end = max(pred_end, target_end)`.
    # 6. Compute `union = max(0, union_end - union_start)`.
    # 7. If `union == 0`, return `0.0`.
    # 8. Return `overlap / union`.
    #
    # Checks:
    # - Same windows should return 1.
    # - Non-overlapping windows should return 0.

    overlap_start = max(pred_start, target_start)
    overlap_end = min(pred_end, target_end)
    overlap = max(0, overlap_end - overlap_start)       # intersection

    union_start = min(pred_start, target_start)
    union_end = max(pred_end, target_end)
    union = max(0, union_end - union_start)             

    if union == 0:
         return 0.0

    # Example:
    # Prediction: frames 2 to 6.
    # Target: frames 4 to 8.
    # The shared part is frames 4 to 6, so `overlap = 2`.
    # The full covered range is frames 2 to 8, so `union = 6`.
    # The temporal IoU is `2 / 6`, which is about `0.33`.
    return overlap/union

    #raise NotImplementedError("Compute temporal IoU.")


class TinyVideoEventModel(nn.Module):
    """Tiny model for event detection from sampled frames."""

    def __init__(self, config: VideoEventConfig):
        super().__init__()
        # Goal:
        # Encode sampled frames and predict:
        # - which COIN step category is happening
        # - approximate start and end frame
        #
        # Suggested layers:
        # - `self.frame_encoder = nn.Sequential(...)`
        #   Apply this to each frame and return one vector per frame.
        # - `self.event_head = nn.Linear(config.hidden_size, config.num_event_classes)`
        #   This predicts one COIN step category.
        # - `self.window_head = nn.Linear(config.hidden_size, 2)`
        #   This predicts start and end frame positions.
        #
        # Method:
        # Use PyTorch layers from `torch.nn`, usually imported as `nn`.
        # `nn.Sequential(...)` can hold the frame CNN.
        # Good layer choices are `nn.Conv2d(...)`, `nn.ReLU()`,
        # `nn.MaxPool2d(...)`, `nn.AdaptiveAvgPool2d((1, 1))`, `nn.Flatten()`,
        # and `nn.Linear(...)`.
        #
        # Steps:
        # 1. Build `self.frame_encoder` so one frame becomes one hidden vector.
        # 2. Build `self.event_head = nn.Linear(config.hidden_size, config.num_event_classes)`.
        # 3. Build `self.window_head = nn.Linear(config.hidden_size, 2)`.
        # 4. Save `config` on `self` if forward needs `clip_length`.
        #
        # Checks:
        # - Event logits should be `[batch_size, config.num_event_classes]`.
        # - Window predictions should be `[batch_size, 2]`.

        self.config = config
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, config.hidden_size),
            nn.ReLU(),
        )
        self.event_head = nn.Linear(config.hidden_size, config.num_event_classes)
        self.window_head = nn.Linear(config.hidden_size, 2)


        #raise NotImplementedError("Build video event model.")

    def forward(self, sampled_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return event logits and predicted event window."""
        # Goal:
        # Predict which step is happening and where it happens in time.
        #
        # Input:
        # - `sampled_frames`: `[batch_size, sampled_frames, 3, image_size, image_size]`
        #
        # Output:
        # - event logits: `[batch_size, config.num_event_classes]`
        # - event window: `[batch_size, 2]`, predicted start and end frame
        #
        # Method:
        # Use `.reshape(batch_size * sampled_count, 3, image_size, image_size)`
        # to flatten batch and time before the frame encoder.
        # Use `.reshape(batch_size, sampled_count, hidden_size)` to restore time.
        # Use `.mean(dim=1)` to pool over sampled frames.
        # Use `torch.sigmoid(...) * config.clip_length` if you want the window
        # head to return frame-index-like values.
        #
        # Steps:
        # 1. Read `batch_size` and `sampled_count` from `sampled_frames.shape`.
        # 2. Flatten batch and time into one frame batch.
        # 3. Encode all frames with `self.frame_encoder`.
        # 4. Reshape frame features back to `[batch_size, sampled_count, hidden_size]`.
        # 5. Average frame features over time.
        # 6. Compute event logits with `self.event_head`.
        # 7. Compute start/end predictions with `self.window_head`.
        # 8. Return event logits and window predictions.
        #
        # Check:
        # - Start/end predictions should be in frame-index units.

        batch_size, sampled_count = sampled_frames.shape[:2]
        flattened_frames = sampled_frames.reshape(batch_size * sampled_count, 3, self.config.image_size, self.config.image_size)
        # this will not work: flattened_frames = sampled_frames.reshape(batch_size * sampled_count, 3, -1, -1)
        # PyTorch only allows one -1 in a reshape. It can infer one missing size, not two.
        # this is valid: flattened_frames = sampled_frames.reshape(batch_size * sampled_count, 3, self.config.image_size, -1)

        encoded_frames = self.frame_encoder(flattened_frames)
        encoded_frames = encoded_frames.reshape(batch_size, sampled_count, self.config.hidden_size)
        video_features = encoded_frames.mean(dim=1)
        event_logits = self.event_head(video_features)
        event_window = torch.sigmoid(self.window_head(video_features)) * self.config.clip_length

        return event_logits, event_window

        #raise NotImplementedError("Predict event class and time window.")


def train_one_epoch(model: TinyVideoEventModel, loader: DataLoader, optimizer: torch.optim.Optimizer, config: VideoEventConfig) -> float:
    """Train one event model epoch."""
    # Goal:
    # Train event classification and event-window prediction together.
    #
    # Method:
    # Use normal PyTorch training calls:
    # `model.train()`, `nn.CrossEntropyLoss()`, `nn.L1Loss()` or `nn.MSELoss()`,
    # `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.
    #
    # Steps:
    # 1. Create `class_loss_fn = nn.CrossEntropyLoss()`.
    # 2. Create `window_loss_fn = nn.L1Loss()` or `nn.MSELoss()`.
    # 3. Call `model.train()` before the loop.
    # 4. Move sampled frames, event labels, and target windows to `device`.
    # 5. Run `event_logits, predicted_windows = model(sampled_frames)`.
    # 6. Compute class loss from event logits and labels.
    # 7. Compute window loss from predicted and target start/end values.
    # 8. Add the two losses.
    # 9. Clear old gradients, backprop, and step the optimizer.
    # 10. Return mean loss as a Python float.
    #
    # Check:
    # - Return mean loss as a Python float.
    class_loss_fn = nn.CrossEntropyLoss()
    window_loss_fn = nn.L1Loss()
    model.train()

    total_loss = 0.0
    example_count = 0

    for batch in loader:
        sampled_frames = batch["sampled_frames"].to(device)
        event_labels = batch["event_label"].to(device)
        target_windows = batch["event_window"].to(device)
        batch_size = sampled_frames.shape[0]

        event_logits, predicted_windows = model(sampled_frames)
        class_loss = class_loss_fn(event_logits, event_labels)
        window_loss = window_loss_fn(predicted_windows, target_windows)
        loss = class_loss + window_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        example_count += batch_size

    if example_count == 0:
        return 0.0

    return total_loss / example_count
    #raise NotImplementedError("Train one video event epoch.")


def evaluate_events(model: TinyVideoEventModel, examples: list[VideoEventExample], config: VideoEventConfig) -> VideoEventMetrics:
    """Evaluate event detection and temporal IoU."""
    # Goal:
    # Measure whether the model finds the event and estimates its time window.
    #
    # Method:
    # Use `model.eval()` and the PyTorch context manager `torch.no_grad()`.
    # Use `sample_frame_indices(...)` for the model-visible frames.
    # Use `logits.argmax(dim=-1)` for event class predictions.
    # Use the local `temporal_iou(...)` helper for start/end windows.
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Loop over examples inside `with torch.no_grad():`.
    # 3. Get frame indices from `sample_frame_indices(config.clip_length, config.sampled_frames)`.
    # 4. Select those frames from `example["frames"]`.
    # 5. Add a batch dimension with `.unsqueeze(0)`.
    # 6. Run the model.
    # 7. Convert class logits to a predicted class id.
    # 8. Compare predicted class id with `example["event_label"]`.
    # 9. Convert predicted start/end tensors to Python numbers.
    # 10. Compute temporal IoU against the target window.
    # 11. Count short events missed by the selected frame indices.
    # 12. Return event accuracy, mean temporal IoU, and missed-short-event rate.
    #
    # Check:
    # - Mean temporal IoU should be between 0 and 1.
    model.eval()
    sample_indices = sample_frame_indices(config.clip_length, config.sampled_frames)
    short_event_threshold = max(1, config.clip_length // config.sampled_frames)

    correct_count = 0
    total_iou = 0.0
    short_event_count = 0
    missed_short_event_count = 0

    with torch.no_grad():
        for example in examples:
            sampled_frames = example["frames"][sample_indices].unsqueeze(0).to(device)
            event_logits, predicted_windows = model(sampled_frames)

            predicted_label = int(event_logits.argmax(dim=-1).item())
            if predicted_label == example["event_label"]:
                correct_count += 1

            predicted_start = int(round(float(predicted_windows[0, 0].item())))
            predicted_end = int(round(float(predicted_windows[0, 1].item())))
            predicted_start = max(0, min(config.clip_length, predicted_start))
            predicted_end = max(0, min(config.clip_length, predicted_end))
            if predicted_end < predicted_start:
                predicted_start, predicted_end = predicted_end, predicted_start

            target_start = example["event_start"]
            target_end = example["event_end"]
            total_iou += temporal_iou(predicted_start, predicted_end, target_start, target_end)

            event_length = target_end - target_start
            if event_length <= short_event_threshold:
                short_event_count += 1
                sampled_event_frames = [index for index in sample_indices if target_start <= index < target_end]
                if len(sampled_event_frames) == 0:
                    missed_short_event_count += 1

    if not examples:
        return {
            "event_accuracy": 0.0,
            "mean_temporal_iou": 0.0,
            "missed_short_event_rate": 0.0,
            "frame_grid_path": "",
            "timeline_path": "",
        }

    missed_short_event_rate = 0.0
    if short_event_count > 0:
        missed_short_event_rate = missed_short_event_count / short_event_count

    return {
        "event_accuracy": correct_count / len(examples),
        "mean_temporal_iou": total_iou / len(examples),
        "missed_short_event_rate": missed_short_event_rate,
        "frame_grid_path": "",
        "timeline_path": "",
    }
    #raise NotImplementedError("Evaluate video event detection.")


def save_video_visuals(examples: list[VideoEventExample], predictions: list[tuple[int, int]], config: VideoEventConfig) -> tuple[str, str]:
    """Save sampled frames and event timeline."""
    # Goal:
    # Save visuals that make frame sampling and event timing easy to inspect.
    #
    # Output:
    # - `(frame_grid_path, timeline_path)` as strings.
    #
    # Method:
    # Use `config.output_dir.mkdir(parents=True, exist_ok=True)` from `pathlib.Path`.
    # Use Matplotlib: `matplotlib.use("Agg")`, `plt.subplots(...)`,
    # `ax.imshow(...)`, `ax.axvspan(...)`, `plt.savefig(path)`, and `plt.close()`.
    # Use the local `sample_frame_indices(...)` helper for the frame grid.
    #
    # Steps:
    # 1. Create `config.output_dir`.
    # 2. Pick one or more examples.
    # 3. Sample frame indices with `sample_frame_indices(...)`.
    # 4. Save a grid of those sampled frames.
    # 5. Build a second figure for a timeline.
    # 6. Draw the target event window and predicted window with different colors.
    # 7. Label the x-axis in frame indices.
    # 8. Return both saved paths as strings.
    #
    # Check:
    # - The timeline should show missed short events clearly.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame_grid_path = config.output_dir / "video_sampled_frames.png"
    timeline_path = config.output_dir / "video_event_timeline.png"

    if not examples:
        return str(frame_grid_path), str(timeline_path)

    sample_indices = sample_frame_indices(config.clip_length, config.sampled_frames)
    first_example = examples[0]
    sampled_frames = first_example["frames"][sample_indices]

    fig, axes = plt.subplots(1, len(sample_indices), figsize=(3 * len(sample_indices), 3))
    if len(sample_indices) == 1:
        axes = [axes]

    for ax, frame, frame_index in zip(axes, sampled_frames, sample_indices):
        image = frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(f"frame {frame_index}")
        ax.axis("off")

    fig.suptitle(f"Sampled frames: {first_example['event_text']}")
    fig.tight_layout()
    fig.savefig(frame_grid_path)
    plt.close(fig)

    shown_examples = examples[: len(predictions)]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(shown_examples))))

    for row_index, (example, prediction) in enumerate(zip(shown_examples, predictions)):
        target_start = example["event_start"]
        target_end = example["event_end"]
        pred_start, pred_end = prediction

        ax.broken_barh([(target_start, target_end - target_start)], (row_index - 0.18, 0.16), facecolors="tab:green")
        ax.broken_barh([(pred_start, pred_end - pred_start)], (row_index + 0.02, 0.16), facecolors="tab:orange")

    ax.set_xlim(0, config.clip_length)
    ax.set_ylim(-0.5, max(0.5, len(shown_examples) - 0.5))
    ax.set_xlabel("frame index")
    ax.set_ylabel("example")
    ax.set_title("Target event window and predicted event window")
    ax.set_yticks(range(len(shown_examples)))
    ax.set_yticklabels([example["event_text"][:40] for example in shown_examples])
    ax.legend(["target", "prediction"], loc="upper right")
    fig.tight_layout()
    fig.savefig(timeline_path)
    plt.close(fig)

    return str(frame_grid_path), str(timeline_path)
    #raise NotImplementedError("Save video frame and timeline visuals.")


def run_exercise(config: VideoEventConfig) -> VideoEventMetrics:
    """Run the video event workflow."""
    # Main workflow:
    # download clips -> sampled frames -> video model -> train -> temporal evaluation -> visuals.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    #download_coin_videos(config)
    examples = build_video_event_dataset(config)
    dataset = VideoEventDataset(examples, config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TinyVideoEventModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for i in range(config.epochs):
        loss = train_one_epoch(model, loader, optimizer, config)
        print(f"Epoch {i + 1} loss: {loss}")   

    metrics = evaluate_events(model, examples, config)
    predictions = [(example["event_start"], example["event_end"]) for example in examples[:8]]
    frame_grid_path, timeline_path = save_video_visuals(examples, predictions, config)
    metrics["frame_grid_path"] = frame_grid_path
    metrics["timeline_path"] = timeline_path

    print(metrics)

    return metrics


def check_metrics(metrics: VideoEventMetrics) -> bool:
    return metrics["event_accuracy"] > 0.60 and metrics["mean_temporal_iou"] > 0.30


def report_task_progress(task_name: str, task_fn, *args):
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the video event scaffold one step at a time."""
    config = VideoEventConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 6

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    # _, ok = report_task_progress("download_coin_videos", download_coin_videos, config)
    # if not count_or_stop(ok):
    #     return

    # examples, ok = report_task_progress("build_video_event_dataset", build_video_event_dataset, config)
    # if not count_or_stop(ok):
    #     return

    #     _, ok = report_task_progress("sample_frame_indices", sample_frame_indices, config.clip_length, config.sampled_frames)
    # if not count_or_stop(ok):
    #     return

    # _, ok = report_task_progress("temporal_iou", temporal_iou, 2, 6, 3, 7)
    # if not count_or_stop(ok):
    #     return

    # _, ok = report_task_progress("TinyVideoEventModel.__init__", TinyVideoEventModel, config)
    # if not count_or_stop(ok):
    #     return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "video event accuracy", metrics["event_accuracy"], "> 0.60 and temporal IoU useful")


if __name__ == "__main__":
    main()
