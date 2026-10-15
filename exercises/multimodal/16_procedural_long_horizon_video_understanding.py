"""
============================================================
EXERCISE 16: LONG-CONTEXT EPIC-KITCHENS ACTION FORECASTING
============================================================

WHAT YOU ARE BUILDING
You are building a small VideoTaskformer-style model for real first-person
kitchen videos.

The model sees four ordered action clips and predicts the next action verb.
It also learns a masked-step task: hide one step feature, look at the steps
around it, and predict the hidden verb.

Examples of verb labels are:
- open
- take
- wash
- put
- close

This exercise uses real EPIC-KITCHENS-100 videos and annotations. It does not
use random videos, generated clips, or a synthetic dataset.

WHY THIS EXERCISE EXISTS
A short-clip model answers:

    what action is visible in this clip?

A long-context model asks:

    given the actions that happened in order, what action is likely next?

That is useful for assistants and agents that must remember more than one
moment. It is also the main idea behind procedure-aware video models: encode
individual steps, then reason over the ordered step sequence.

HOW THIS FITS INTO THE BOOTCAMP
Earlier multimodal exercises taught image-text alignment, grounded answers,
short video sampling, and video representations. Exercise 15 focused on dense
video features. This exercise uses a pretrained video encoder and focuses on
the next level: reasoning across several action clips.

THE REAL PAPER WORKFLOW FIRST
VideoTaskformer uses this broad workflow:

1. turn an instructional video into ordered step clips 
2. encode every clip with a pretrained video model
3. run a transformer across the ordered clip features
4. hide a step and predict its weak text label
5. fine-tune or evaluate the representation on tasks such as step forecasting

ProcedureVRL goes further. It learns from large narrated instructional-video
collections and models several plausible step orders and futures.

This exercise is a scaled-down VideoTaskformer-style exercise. It is not a
full ProcedureVRL reproduction.

WHAT IS FIXED, WHAT IS LEARNED, AND WHAT IS HIDDEN
Fixed:
- the real EPIC-KITCHENS videos and action timestamps
- the fixed train, validation, held-out-kitchen, and test manifest
- a TimeSformer pretrained on Kinetics-400
- cached TimeSformer features after they are extracted once

Learned:
- a small projection from TimeSformer features to the exercise hidden size
- a learned mask token
- a transformer over ordered action-step features
- a classifier over common training verbs

Hidden during masked-step training:
- one step feature in a five-step sequence

Target during training:
- the real annotated verb of the hidden step
- the real annotated verb of the next step

IMPORTANT DATASET WORDING
EPIC-KITCHENS contains unscripted kitchen activity. It has real ordered action
segments, but it does not mark a clean recipe boundary around every sequence.
For that reason, this exercise is called long-context action forecasting, not
full recipe or procedure planning.

Windows must:
- stay inside one video
- use consecutive original annotation rows
- avoid a large time gap between neighboring actions

TINY REAL EXAMPLE
Suppose one real video contains these consecutive annotations:

    order  start   stop    narration       verb
    20     62.1    63.0    open fridge     open
    21     63.4    64.2    take milk       take
    22     64.5    65.1    close fridge    close
    23     66.0    67.4    pour milk       pour
    24     68.0    68.8    put milk down   put

Forecasting example:

    history = [open, take, close, pour]
    target  = put

Masked-step example with mask index 1:

    input   = [open, MASK, close, pour, put]
    target  = take

The model receives video features, not the written verb words. The words above
only make the example easy to read.

REAL SYSTEM TO EXERCISE MAPPING

    real paper system                    this guided exercise
    -------------------------------------------------------------------
    large instructional-video corpus     15 real EPIC-KITCHENS videos
    pretrained TimeSformer               frozen Kinetics-400 TimeSformer
    step clip feature                    cached CLS feature, shape [768]
    long ordered step sequence           five consecutive annotated actions
    masked step modeling                 hide one of five cached step features
    downstream step forecasting          four clips predict the next verb
    large step vocabulary                eight common training verbs
    large step transformer               two-layer student-built transformer

WHY THE PRETRAINED ENCODER IS FROZEN
Ten official training videos are not enough to learn a useful video transformer
from scratch. The papers rely on large-scale pretraining. This exercise keeps
that real workflow by using a pretrained TimeSformer and freezing it.

The expensive video pass happens once:

    real MP4 clips -> frozen TimeSformer -> cached feature vectors

Training then becomes small and practical:

    cached feature sequence -> step transformer -> verb logits

DATASET AND FIXED SPLITS
Before starting, run this from the repository root:

    python download_epic_kitchens_subset.py

It creates:

    data/epic_kitchens/EPIC_100_train.csv
    data/epic_kitchens/EPIC_100_validation.csv
    data/epic_kitchens/video_manifest.csv
    data/epic_kitchens/videos/.../<video_id>.MP4

The manifest has these fixed experiment roles:

    source videos                  experiment role
    --------------------------------------------------
    6 official train videos        train
    2 official train videos        validation
    2 videos from one participant  held-out kitchen
    5 official validation videos   final test

The final test videos are chosen without looking at their verb labels.
Do not choose model settings from test results.

VOCABULARY RULE
Build the verb vocabulary from the six final training videos only.

Keep the eight most common training verbs. A validation or test target outside
that vocabulary is not used as a closed-vocabulary classification example.
Report known-label coverage so this filtering remains visible.

Never delete unknown actions before making sequences. Deleting them would turn
non-neighboring real actions into false neighbors. Keep every action in its
original position, then keep a forecasting window only when its target verb is
in the training vocabulary.

INPUTS AND OUTPUTS
The frozen TimeSformer produces:

    one step feature: [timesformer_feature_dim] = [768]

One dataset batch contains:

    history_features: [batch, 4, 768]        # for forecasting
    masked_features:  [batch, 5, 768]        # for masked-step learning
    mask_index:       [batch]
    next_label:       [batch]                # for forecasting
    masked_label:     [batch]                # for masked-step learning 

The student model returns:

    forecast_logits: [batch, num_classes]
    masked_logits:   [batch, num_classes]

MODEL ORDER
For forecasting:

    four cached features
    -> linear projection
    -> step position embeddings
    -> step transformer
    -> final history token
    -> verb classifier

For masked-step learning:

    five cached features
    -> linear projection
    -> replace one projected feature with a learned mask token
    -> step position embeddings
    -> the same step transformer
    -> output at the masked position
    -> verb classifier

LOSS FUNCTION
Use cross entropy for both real-label tasks:

    forecast_loss = nn.CrossEntropyLoss()(forecast_logits, next_label)
    masked_loss = nn.CrossEntropyLoss()(masked_logits, masked_label)
    total_loss = forecast_loss + masked_loss_weight * masked_loss

`masked_loss_weight` controls how much the masked-step task affects training.
The default is `0.5`, so forecasting remains the main task.

BASELINES
Report two simple comparisons:

1. Majority baseline: always predict the most common training target verb.
2. Last-step baseline: give one linear classifier only the final history clip.

The main model sees four clips. The last-step baseline sees one. This tests
whether ordered context helps more than the most recent clip alone.

METRICS TO TRACK
Report:
- mean cross-entropy loss, lower is better
- top-1 next-verb accuracy, higher is better
- top-3 next-verb accuracy, higher is better
- macro F1, higher is better
- masked-step accuracy, higher is better
- majority-baseline accuracy
- last-step-baseline accuracy
- known-label coverage for every split

Macro F1 gives every verb class equal importance. This matters because common
verbs otherwise dominate normal accuracy.

Known-label coverage is:

    usable target windows / all valid consecutive candidate windows

Example:
- 80 consecutive test windows are found
- 60 targets use verbs present in the training vocabulary
- known-label coverage is `60 / 80 = 0.75`, or 75 percent

How to read results:
- if the main model does not beat the majority baseline, check class imbalance
- if it does not beat the last-step baseline, longer context has not helped yet
- weak held-out-kitchen or test accuracy can be a real generalization result
- do not hide weak results by selecting different test videos

VISUALS TO SAVE
Save to `outputs/epic_action_forecasting/`:
- `training_curves.png`: train and validation loss and accuracy
- `confusion_matrix.png`: true versus predicted verb names
- `forecast_timeline.png`: four real history actions and the true/predicted next action
- `next_verb_probabilities.png`: top predicted verbs for one test window
- `class_distribution.png`: training examples per verb

STEP-BY-STEP TODO
1. Convert EPIC timestamps to seconds.
2. Load the fixed manifest and real action annotations.
3. Build the verb vocabulary from the final training split only.
4. Build real consecutive five-step windows without deleting unknown actions.
5. Sample eight real frames inside every action segment.
6. Load and freeze the pretrained TimeSformer.
7. Extract and cache one feature per action step used by the windows.
8. Build the cached-feature datasets and dataloaders.
9. Build the shared step transformer and its two prediction paths.
10. Train masked-step and forecasting losses together.
11. Train the one-clip linear baseline.
12. Select the best model using validation accuracy only.
13. Evaluate the held-out kitchen and final test once.
14. Save the five required visuals.

LEARNING ORDER
Finish the stubs in the order they appear. Do not start the model until a real
window can be printed with five real narrations and timestamps. Do not start
training until the cached feature shapes have been checked.

DEFINITION OF DONE
You are done when:
- every example comes from a real EPIC-KITCHENS MP4
- the manifest fixes all four splits before vocabulary building
- the vocabulary uses only final training videos
- every window preserves consecutive original annotation order
- TimeSformer is frozen and each required step feature is cached
- training runs on cached features without reopening MP4 files each epoch
- the main model and both baselines are reported
- top-1, top-3, macro F1, masked accuracy, and coverage are printed
- held-out-kitchen and final-test results are separate
- all five visuals are saved with real verb names
- you can explain why pretrained clip features make this small exercise trainable
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TYPE_CHECKING, Literal, TypedDict

import av
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

if TYPE_CHECKING:
    from transformers import BaseImageProcessor, TimesformerModel

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"
SplitName = Literal["train", "val", "unseen_kitchen", "test"]


class ActionStep(TypedDict):
    """One real annotated action segment from one local video."""

    step_key: str
    participant_id: str
    video_id: str
    video_path: str
    official_split: str
    experiment_split: SplitName
    narration: str
    verb_name: str
    verb_class: int
    start_seconds: float
    end_seconds: float
    raw_order_index: int


class ForecastWindow(TypedDict):
    """Five consecutive actions used for masking and next-step forecasting."""

    video_id: str
    sequence_steps: list[ActionStep]
    mask_index: int


class ModelOutputs(TypedDict):
    """The two class predictions returned during joint training."""

    forecast_logits: torch.Tensor
    masked_logits: torch.Tensor


class EpochMetrics(TypedDict):
    """Training measurements averaged across one epoch."""

    mean_loss: float
    forecast_accuracy: float
    masked_accuracy: float
    last_step_accuracy: float


class EvaluationMetrics(TypedDict):
    """Measurements and saved predictions for one held-out split."""

    mean_loss: float
    top1_accuracy: float
    top3_accuracy: float
    macro_f1: float
    masked_accuracy: float
    majority_accuracy: float
    last_step_accuracy: float
    labels: list[int]
    predictions: list[int]
    probabilities: torch.Tensor


class ExperimentResult(TypedDict):
    """Final result returned by the full exercise."""

    validation: EvaluationMetrics
    unseen_kitchen: EvaluationMetrics
    test: EvaluationMetrics
    known_label_coverage: dict[str, float]
    visual_paths: dict[str, str]


@dataclass(frozen=True)
class VerbVocabulary:
    """Map raw EPIC verb classes to the small training-only classifier."""

    verb_class_to_label_id: dict[int, int]
    label_id_to_name: list[str]
    training_counts: list[int]

    @property
    def num_classes(self) -> int:
        """Return the classifier output size."""
        return len(self.label_id_to_name)

    def label_id(self, step: ActionStep) -> int:
        """Return the small class ID for one known-vocabulary step."""
        return self.verb_class_to_label_id[step["verb_class"]]


@dataclass
class WindowBuildResult:
    """Windows and visible coverage information for every split."""

    windows_by_split: dict[SplitName, list[ForecastWindow]]
    known_label_coverage: dict[str, float]
    candidate_counts: dict[str, int]


@dataclass
class ActionForecastingConfig:
    """Settings for the scaled-down VideoTaskformer-style experiment."""

    train_annotation_path: Path = PROJECT_ROOT / "data" / "epic_kitchens" / "EPIC_100_train.csv"
    validation_annotation_path: Path = (
        PROJECT_ROOT / "data" / "epic_kitchens" / "EPIC_100_validation.csv"
    )
    manifest_path: Path = PROJECT_ROOT / "data" / "epic_kitchens" / "video_manifest.csv"
    video_root: Path = PROJECT_ROOT / "data" / "epic_kitchens" / "videos"
    feature_cache_path: Path = (
        PROJECT_ROOT / "data" / "epic_kitchens" / "feature_cache" / "timesformer_k400_8_frames.pt"
    )

    output_dir: Path = PROJECT_ROOT / "outputs" / "epic_action_forecasting"
    timesformer_model_name: str = "facebook/timesformer-base-finetuned-k400"
    timesformer_feature_dim: int = 768
    frames_per_step: int = 8
    history_steps: int = 4
    num_common_verbs: int = 8
    max_gap_seconds: float = 30.0
    hidden_dim: int = 256
    transformer_depth: int = 2
    num_heads: int = 4
    mlp_dim: int = 512
    dropout: float = 0.1
    masked_loss_weight: float = 0.5
    batch_size: int = 16
    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_train_windows: int = 256
    max_eval_windows: int = 128
    random_state: int = 416

    def __post_init__(self) -> None:
        """Catch settings that would break the student model."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must divide evenly by num_heads.")
        if self.frames_per_step != 8:
            raise ValueError("The selected pretrained TimeSformer expects 8 frames.")
        if self.history_steps < 1:
            raise ValueError("history_steps must be at least 1.")
        if self.num_common_verbs < 3:
            raise ValueError("Use at least 3 common verbs for a useful classifier.")


class CachedWindowDataset(Dataset):
    """Return already-extracted real video features for one five-step window."""

    def __init__(
        self,
        windows: list[ForecastWindow],
        feature_by_step_key: dict[str, torch.Tensor],
        vocabulary: VerbVocabulary,
        history_steps: int,
    ) -> None:
        self.windows = windows
        self.feature_by_step_key = feature_by_step_key
        self.vocabulary = vocabulary
        self.history_steps = history_steps

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        sequence_steps = window["sequence_steps"]
        sequence_features = torch.stack(
            [self.feature_by_step_key[step["step_key"]] for step in sequence_steps], dim=0
        )
        mask_index = window["mask_index"]
        return {
            "history_features": sequence_features[: self.history_steps],
            "masked_features": sequence_features,
            "mask_index": torch.tensor(mask_index, dtype=torch.long),
            "next_label": torch.tensor(self.vocabulary.label_id(sequence_steps[-1]), dtype=torch.long),
            "masked_label": torch.tensor(
                self.vocabulary.label_id(sequence_steps[mask_index]), dtype=torch.long
            ),
        }


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert an EPIC timestamp such as `00:01:02.50` to seconds."""
    # Where this fits:
    # EPIC gives action boundaries as clock-like text. PyAV reports decoded
    # frame times in seconds, so both sides must use the same unit.
    #
    # Goal:
    # Convert one `hours:minutes:seconds` timestamp into a float.
    #
    # Input: `timestamp="00:01:02.50"`
    # Output: `62.5`
    #
    # Method and steps:
    # 1. Split the string with `timestamp.split(":")`.
    # 2. Convert hours, minutes, and seconds to floats.
    # 3. Return `hours * 3600 + minutes * 60 + seconds`.
    #
    # Check: `timestamp_to_seconds("01:02:03.5") == 3723.5`

    texts = timestamp.split(':')
    hrs = float(texts[0])
    mins = float(texts[1])
    secs = float(texts[2])    

    return hrs * 3600 + mins * 60 + secs    


def load_manifested_action_steps(config: ActionForecastingConfig) -> list[ActionStep]:
    """Load real action rows only for videos in the fixed manifest."""
    # Where this fits:
    # The next function needs one clean list of actions. Right now, the facts it
    # needs are spread across three places:
    #
    # source                         what it tells us
    # -------------------------------------------------------------------------
    # video_manifest.csv             which 15 videos to use and their split
    # EPIC_100_train/validation.csv  what actions happen and when they happen
    # videos/.../<video_id>.MP4      where the real video file is on disk
    #
    # This function joins those three sources by `video_id`.
    #
    # Goal:
    # Return one `ActionStep` for every annotated action in a manifested video.
    # Keep every action for now, including verbs that may not enter the small
    # vocabulary later. This preserves the true order of events.
    #
    # The two kinds of split are different:
    # - `official_split` tells you which annotation CSV contains the video.
    # - `experiment_split` tells you how this exercise will use the video.
    #
    # For example, this manifest row:
    #
    #     video_id=P03_116, official_split=train, experiment_split=val
    #
    # means: read P03_116 actions from `EPIC_100_train.csv`, but use those
    # actions as this exercise's validation data.
    #
    # Output:
    # Return `list[ActionStep]`. One item should look like this shape:
    #
    #     {
    #         "step_key": annotation_row["narration_id"],
    #         "participant_id": manifest_row["participant_id"],
    #         "video_id": annotation_row["video_id"],
    #         "video_path": str(path_to_that_video),
    #         "official_split": manifest_row["official_split"],
    #         "experiment_split": manifest_row["experiment_split"],
    #         "narration": annotation_row["narration"],
    #         "verb_name": annotation_row["verb"],
    #         "verb_class": int(annotation_row["verb_class"]),
    #         "start_seconds": timestamp_to_seconds(...),
    #         "end_seconds": timestamp_to_seconds(...),
    #         "raw_order_index": position inside this video's sorted rows,
    #     }
    #
    # Method:
    # Open a CSV with `path.open("r", encoding="utf-8", newline="")`, then
    # pass that open file to `csv.DictReader(file)`. Iterating over the reader
    # gives dictionaries whose keys are the CSV column names.
    #
    # Useful containers to build:
    #
    #     manifest_by_video_id: dict[str, dict[str, str]]
    #     video_path_by_id: dict[str, Path]
    #     annotation_rows_by_video_id: dict[str, list[dict[str, str]]]
    #     steps: list[ActionStep]
    #
    # STEP 1: Check the four inputs before doing any work.
    # Use `path.exists()` for the manifest, both annotation CSVs, and the video
    # root. Raise `FileNotFoundError` with the missing path if one is absent.
    #
    # STEP 2: Read and validate the manifest.
    # Build `manifest_by_video_id`, keyed by `row["video_id"]`. Each
    # `experiment_split` must be one of `train`, `val`, `unseen_kitchen`, or
    # `test`. Reject duplicate video IDs because one video must have one role.
    #
    # Also check kitchen separation. Make a set of participant IDs used by
    # `train` and another used by `unseen_kitchen`. Their intersection must be
    # empty. Otherwise the supposedly unseen kitchen was seen during training.
    #
    # STEP 3: Find the local MP4 for each manifested video.
    # `config.video_root.rglob("*")` visits files in all folders below the video
    # root. Keep files where `path.is_file()` and `path.suffix.lower() == ".mp4"`.
    # Key the result by `path.stem`, because the stem of `P03_116.MP4` is
    # `P03_116`. Check that every manifest video ID has exactly one path.
    #
    # STEP 4: Read the two annotation CSVs into groups.
    # Treat each CSV together with its source name:
    #
    #     config.train_annotation_path       -> "train"
    #     config.validation_annotation_path  -> "validation"
    #
    # For each annotation row:
    # 1. Look up `row["video_id"]` in `manifest_by_video_id`.
    # 2. Skip the row if that video is not in the manifest.
    # 3. Compare the manifest's `official_split` with the current CSV's source
    #    name. Raise an error if they disagree; the data came from the wrong CSV.
    # 4. Append the unchanged row to that video's list.
    #
    # STEP 5: Turn each video's annotation rows into ordered `ActionStep`s.
    # Work through the manifest video IDs one video at a time. Sort that video's
    # full row list by `timestamp_to_seconds(row["start_timestamp"])`. Then use
    # `enumerate(sorted_rows)` to get `raw_order_index` values `0, 1, 2, ...`.
    #
    # Convert each row using the output shape shown above. Check that
    # `end_seconds > start_seconds`, then append the new `ActionStep` to `steps`.
    # Do not filter by verb here. The later vocabulary and window functions do
    # that without damaging the original action order.
    #
    # STEP 6: Validate and report the finished list.
    # Every manifest video should contribute actions. Every experiment split
    # should also contribute actions. Print counts for videos, actions,
    # participants, and experiment splits, then return `steps`.
    #
    # Checks:
    # - `len(manifest_by_video_id) == 15`
    # - one P03_116 step uses its real MP4 path and its manifest split
    # - raw order starts at 0 separately for each video
    # - every action has `end_seconds > start_seconds`
    # - all four experiment splits have at least one action
    required_paths = (
        config.manifest_path,
        config.train_annotation_path,
        config.validation_annotation_path,
        config.video_root,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required EPIC-KITCHENS path does not exist: {path}")

    allowed_experiment_splits = {"train", "val", "unseen_kitchen", "test"}
    allowed_official_splits = {"train", "validation"}
    manifest_by_video_id: dict[str, dict[str, str]] = {}

    with config.manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            video_id = row["video_id"]
            if video_id in manifest_by_video_id:
                raise ValueError(f"Duplicate video ID in the manifest: {video_id}")
            if row["experiment_split"] not in allowed_experiment_splits:
                raise ValueError(
                    f"Unknown experiment split for {video_id}: {row['experiment_split']}"
                )
            if row["official_split"] not in allowed_official_splits:
                raise ValueError(
                    f"Unknown official split for {video_id}: {row['official_split']}"
                )
            manifest_by_video_id[video_id] = row

    if not manifest_by_video_id:
        raise ValueError(f"The manifest contains no videos: {config.manifest_path}")

    train_participants = {
        row["participant_id"]
        for row in manifest_by_video_id.values()
        if row["experiment_split"] == "train"
    }
    unseen_participants = {
        row["participant_id"]
        for row in manifest_by_video_id.values()
        if row["experiment_split"] == "unseen_kitchen"
    }
    shared_participants = train_participants & unseen_participants
    if shared_participants:
        raise ValueError(
            "These unseen-kitchen participants also appear in train: "
            f"{sorted(shared_participants)}"
        )

    video_path_by_id: dict[str, Path] = {}
    for path in config.video_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        if path.stem in video_path_by_id:
            raise ValueError(f"More than one MP4 was found for video {path.stem}.")
        video_path_by_id[path.stem] = path.resolve()

    missing_video_ids = sorted(set(manifest_by_video_id) - set(video_path_by_id))
    if missing_video_ids:
        raise FileNotFoundError(
            "No local MP4 was found for these manifest videos: "
            f"{missing_video_ids}"
        )

    rows_by_video_id: dict[str, list[dict[str, str]]] = {
        video_id: [] for video_id in manifest_by_video_id
    }
    annotation_sources = (
        (config.train_annotation_path, "train"),
        (config.validation_annotation_path, "validation"),
    )
    for annotation_path, source_split in annotation_sources:
        with annotation_path.open("r", encoding="utf-8", newline="") as annotation_file:
            for row in csv.DictReader(annotation_file):
                video_id = row["video_id"]
                manifest_row = manifest_by_video_id.get(video_id)
                if manifest_row is None:
                    continue
                if manifest_row["official_split"] != source_split:
                    raise ValueError(
                        f"Video {video_id} was found in the {source_split} annotation CSV, "
                        f"but the manifest says {manifest_row['official_split']}."
                    )
                rows_by_video_id[video_id].append(row)

    steps: list[ActionStep] = []
    for video_id, manifest_row in manifest_by_video_id.items():
        video_rows = rows_by_video_id[video_id]
        if not video_rows:
            raise ValueError(f"Manifest video {video_id} has no annotation rows.")

        video_rows.sort(key=lambda row: timestamp_to_seconds(row["start_timestamp"]))
        for raw_order_index, row in enumerate(video_rows):
            if row["participant_id"] != manifest_row["participant_id"]:
                raise ValueError(
                    f"Participant mismatch for {video_id}: the annotation says "
                    f"{row['participant_id']}, but the manifest says "
                    f"{manifest_row['participant_id']}."
                )

            start_seconds = timestamp_to_seconds(row["start_timestamp"])
            end_seconds = timestamp_to_seconds(row["stop_timestamp"])
            if end_seconds <= start_seconds:
                raise ValueError(
                    f"Action {row['narration_id']} has end time {end_seconds}, "
                    f"which is not after start time {start_seconds}."
                )

            steps.append(
                ActionStep(
                    step_key=row["narration_id"],
                    participant_id=manifest_row["participant_id"],
                    video_id=video_id,
                    video_path=str(video_path_by_id[video_id]),
                    official_split=manifest_row["official_split"],
                    experiment_split=manifest_row["experiment_split"],
                    narration=row["narration"],
                    verb_name=row["verb"],
                    verb_class=int(row["verb_class"]),
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    raw_order_index=raw_order_index,
                )
            )

    action_counts = Counter(step["experiment_split"] for step in steps)
    missing_splits = sorted(allowed_experiment_splits - set(action_counts))
    if missing_splits:
        raise ValueError(f"These experiment splits have no actions: {missing_splits}")

    participant_count = len({step["participant_id"] for step in steps})
    print(
        f"Loaded {len(steps)} actions from {len(manifest_by_video_id)} videos "
        f"and {participant_count} participants."
    )
    for split_name in ("train", "val", "unseen_kitchen", "test"):
        print(f"  {split_name}: {action_counts[split_name]} actions")

    return steps


def build_training_verb_vocabulary(
    steps: list[ActionStep], config: ActionForecastingConfig
) -> VerbVocabulary:
    """Build the classifier vocabulary from final training videos only."""
    # Where this fits:
    # The classifier needs small IDs. This creates them without looking at
    # validation, held-out-kitchen, or test labels.
    #
    # Goal:
    # Keep `config.num_common_verbs` common raw verb classes from `train` only.
    #
    # Output:
    # - `verb_class_to_label_id`, such as `{2: 0, 0: 1, 5: 2}`
    # - `label_id_to_name`, such as `["take", "put", "open"]`
    # - `training_counts` in the same small-ID order
    #
    # Method:
    # Use `collections.Counter` on raw `verb_class` values from train steps.
    #
    # Steps:
    # 1. Select only steps where `experiment_split == "train"`.
    # 2. Count raw verb classes.
    # 3. Sort by descending count, then ascending raw class ID to break ties.
    # 4. Keep exactly `config.num_common_verbs` classes.
    # 5. Assign small IDs from `0` upward and find each readable verb name.
    #
    # Checks:
    # - no held-out split may contribute to counts
    # - `vocabulary.num_classes == config.num_common_verbs`
    # - print every small ID, name, raw ID, and count
    raise NotImplementedError("Build a training-only verb vocabulary.")








def build_consecutive_windows(
    steps: list[ActionStep],
    vocabulary: VerbVocabulary,
    config: ActionForecastingConfig,
) -> WindowBuildResult:
    """Build honest five-step windows without creating false neighbors."""
    # Where this fits:
    # This converts individual annotations into masked-step and forecasting examples.
    #
    # Goal:
    # Build `history_steps + 1` consecutive actions from one video. Keep a
    # window only when its next target is in the training vocabulary. Preserve
    # unknown actions inside history instead of deleting them.
    #
    # Small example:
    # - raw orders `[20, 21, 22, 23, 24]` are valid
    # - raw orders `[20, 21, 23, 24, 25]` must be skipped
    #
    # Steps:
    # 1. Group by `(experiment_split, video_id)` and sort by raw order.
    # 2. Slide a window of `config.history_steps + 1` actions.
    # 3. Require every neighboring raw order to differ by exactly `1`.
    # 4. Reject a neighbor gap larger than `config.max_gap_seconds`.
    # 5. Count every remaining sequence as a candidate window.
    # 6. Keep it only if the final verb is in the vocabulary.
    # 7. Find positions with known verbs and choose one deterministic mask index
    #    using the window start and `config.random_state`.
    # 8. Store all original steps and the mask index.
    # 9. Shuffle each split with `random.Random(config.random_state)`.
    # 10. Cap train with `max_train_windows` and other splits with `max_eval_windows`.
    # 11. Compute `kept / candidates` known-label coverage for every split.
    #
    # Checks:
    # - each window has five real consecutive actions
    # - final and masked verbs are in the vocabulary
    # - print candidates, kept windows, and coverage for every split
    raise NotImplementedError("Build real consecutive action windows.")






def sample_action_frames(step: ActionStep, config: ActionForecastingConfig) -> list[np.ndarray]:
    """Sample eight real RGB frames inside one annotated action segment."""
    # Where this fits:
    # This turns one timestamped action into pixels for TimeSformer.
    #
    # Output:
    # - exactly eight NumPy arrays shaped `[height, width, 3]`
    # - dtype `uint8`, RGB channel order
    #
    # Method:
    # - `av.open(str(video_path))` opens the MP4
    # - `frame.to_ndarray(format="rgb24")` returns RGB pixels
    #
    # Steps:
    # 1. Use `np.linspace(start, end, frames_per_step + 2)[1:-1].tolist()`.
    # 2. Open the video and get `container.streams.video[0]`.
    # 3. Set `video_stream.thread_type = "AUTO"`.
    # 4. Convert the first target to stream units with:
    #    `int(target_seconds[0] / float(video_stream.time_base))`.
    # 5. Seek with `backward=True, any_frame=False, stream=video_stream`.
    # 6. Decode forward and compare `float(frame.time)` with each target.
    # 7. Append real `rgb24` arrays until exactly eight frames are collected.
    #    A `while` loop may repeat one real frame for very close targets.
    #
    # Checks:
    # - never return generated or zero-filled frames
    # - raise a path-and-timestamp error if eight real frames cannot be returned
    raise NotImplementedError("Sample real frames from one EPIC action segment.")







def load_frozen_timesformer(
    config: ActionForecastingConfig,
) -> tuple[BaseImageProcessor, TimesformerModel]:
    """Load the pretrained clip encoder used to create the feature cache."""
    # Where this fits:
    # The papers use strong pretrained clip representations. This loads that
    # fixed visual encoder before the student model is trained.
    #
    # Method:
    # Import inside this function:
    # `from transformers import AutoImageProcessor, TimesformerModel`
    #
    # Exact calls:
    # `processor = AutoImageProcessor.from_pretrained(config.timesformer_model_name)`
    # `model = TimesformerModel.from_pretrained(config.timesformer_model_name)`
    #
    # Steps:
    # 1. Move the model to `device` and call `model.eval()`.
    # 2. Set every `parameter.requires_grad = False`.
    # 3. Check `model.config.hidden_size == config.timesformer_feature_dim`.
    #
    # Checks:
    # - `model.training` is `False`
    # - no model parameter requires gradients
    # - print model name and hidden size

    raise NotImplementedError("Load and freeze the pretrained TimeSformer.")






def extract_and_cache_step_features(
    windows_by_split: dict[SplitName, list[ForecastWindow]],
    processor: BaseImageProcessor,
    timesformer: TimesformerModel,
    config: ActionForecastingConfig,
) -> dict[str, torch.Tensor]:
    """Extract each required real action feature once and save it."""
    # Where this fits:
    # This is the only expensive pass over raw videos. Training later uses CPU
    # feature tensors and does not reopen MP4 files each epoch.
    #
    # Goal:
    # Return `{step_key: feature}` for every unique step used by kept windows.
    # Each feature is a CPU tensor shaped `[768]`.
    #
    # Cache dictionary keys:
    # - `model_name`: string
    # - `frames_per_step`: integer
    # - `features`: `{step_key: CPU tensor}`
    #
    # Method:
    # - `processor(frames, return_tensors="pt")` prepares one eight-frame video
    # - `timesformer(pixel_values=pixel_values)` runs the frozen model
    # - `outputs.last_hidden_state[:, 0]` is the `[1, 768]` CLS feature
    #
    # Steps:
    # 1. Collect unique steps by `step_key` from all windows.
    # 2. If the cache exists, load it with `torch.load(..., weights_only=True)`.
    # 3. Reuse every valid required feature already in the cache. If the cache
    #    is partial, continue from its first missing step.
    # 4. Under `torch.no_grad()`, sample each missing step's real frames.
    # 5. Call the processor and check pixel shape `[1, 8, 3, 224, 224]`.
    # 6. Move pixels to `device`, run TimeSformer, take the CLS feature, detach,
    #    convert to float, and move it to CPU.
    # 7. Save an atomic checkpoint every 10 new features and once more when the
    #    loop finishes or is interrupted. An atomic save keeps the old cache
    #    intact until the new cache file is complete.
    #
    # Checks:
    # - print first-time extraction progress
    # - every required key exists exactly once
    # - every feature is finite and shaped `[768]`
    # - a rerun reuses the cache without decoding again
    raise NotImplementedError("Extract and cache frozen TimeSformer step features.")










def build_cached_dataloaders(
    window_result: WindowBuildResult,
    feature_by_step_key: dict[str, torch.Tensor],
    vocabulary: VerbVocabulary,
    config: ActionForecastingConfig,
) -> dict[SplitName, DataLoader]:
    """Build fast dataloaders from cached real-video features."""
    # Goal:
    # Return `train`, `val`, `unseen_kitchen`, and `test` DataLoaders.
    #
    # Method:
    # Wrap each split with:
    # `CachedWindowDataset(windows, feature_by_step_key, vocabulary, config.history_steps)`
    # Then use:
    # `DataLoader(dataset, batch_size=config.batch_size, shuffle=..., num_workers=0)`
    #
    # Steps:
    # 1. Check every step key needed by every window exists in the cache.
    # 2. Build one dataset per split.
    # 3. Shuffle only training. Use `num_workers=0` for reliable Windows runs.
    #
    # Checks:
    # - print dataset and batch counts
    # - inspect one real batch against the five shapes listed at the top
    # - raise an error when any split has zero usable windows
    raise NotImplementedError("Build cached-feature dataloaders.")







class VideoTaskformerStudent(nn.Module):
    """Reason over ordered pretrained step features for two real-label tasks."""

    def __init__(self, config: ActionForecastingConfig, num_classes: int) -> None:
        super().__init__()
        # Goal:
        # Build one shared step transformer with forecast and masked-step paths.
        #
        # Build:
        # - `nn.Linear(config.timesformer_feature_dim, config.hidden_dim)`
        # - learned mask token `[1, 1, hidden_dim]`
        # - learned position embedding `[1, history_steps + 1, hidden_dim]`
        # - `nn.TransformerEncoder(...)`
        # - `nn.LayerNorm(config.hidden_dim)`
        # - `nn.Linear(config.hidden_dim, num_classes)`
        #
        # Exact encoder layer:
        # `nn.TransformerEncoderLayer(`
        # `    d_model=config.hidden_dim, nhead=config.num_heads,`
        # `    dim_feedforward=config.mlp_dim, dropout=config.dropout,`
        # `    activation="gelu", batch_first=True, norm_first=True,`
        # `)`
        # Pass it to `nn.TransformerEncoder(layer, num_layers=config.transformer_depth)`.
        #
        # Checks:
        # - projection maps `768 -> 256`
        # - position embedding has room for five steps
        # - classifier output size equals `num_classes`
        raise NotImplementedError("Build the step-context transformer.")


        # Fill the learned mask token with small random starting values. The
        # values come from a normal distribution centred on 0. A standard
        # deviation of 0.02 means most values start close to 0, so the token
        # does not create a very large signal before training begins.

        # Give each learned step-position vector small random starting values.
        # This lets positions start slightly different from one another, while
        # keeping their initial effect small. The trailing `_` in `normal_`
        # means PyTorch changes this parameter directly instead of returning a
        # separate tensor.


    def forward(
        self,
        history_features: torch.Tensor,
        masked_features: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> ModelOutputs:
        """Return next-verb and masked-verb logits."""
        # Inputs:
        # - history `[batch, 4, 768]`
        # - masked sequence `[batch, 5, 768]`
        # - mask index `[batch]`, values `0` through `4`
        #
        # Output:
        # - `forecast_logits` and `masked_logits`, both `[batch, classes]`
        #
        # Forecast path:
        # 1. Project history, add first four positions, and run the transformer.
        # 2. Normalize and classify the final history token.
        #
        # Masked path:
        # 1. Project all five features and clone the projected tensor.
        # 2. Build `batch_ids = torch.arange(batch_size, device=masked_features.device)`.
        # 3. Replace `masked_tokens[batch_ids, mask_index]` with the learned mask token.
        # 4. Add all five positions and run the same transformer.
        # 5. Gather the output at each mask index, normalize, and classify it.
        #
        # Checks:
        # - both outputs have the same class dimension
        # - neither output contains `NaN` or infinity
        raise NotImplementedError("Run forecasting and masked-step prediction.")







class LastStepBaseline(nn.Module):
    """A baseline that sees only the most recent clip feature."""

    # This small model checks whether the main model really benefits from its
    # longer action history. It ignores the first three history clips and uses
    # one linear layer to predict the next verb from the latest clip only.
    #
    # If VideoTaskformerStudent beats this baseline, the earlier clips are
    # providing useful context. If their scores are similar, the main model may
    # be relying mostly on the latest clip. This is an exercise-specific
    # baseline; `LastStepBaseline` is not a model name used in the paper.

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, history_features: torch.Tensor) -> torch.Tensor:
        """Classify only the final feature in the history."""
        return self.classifier(history_features[:, -1])


def train_one_epoch(
    model: VideoTaskformerStudent,
    baseline: LastStepBaseline,
    loader: DataLoader,
    model_optimizer: torch.optim.Optimizer,
    baseline_optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: ActionForecastingConfig,
) -> EpochMetrics:
    """Train the context model and one-step baseline for one epoch."""
    # Goal:
    # Update the main model on forecast plus masking loss and update the baseline
    # on next-label loss. TimeSformer is absent because features are cached.
    #
    # Steps per batch:
    # 1. Put both models in train mode and move all five tensors to `device`.
    # 2. Zero the main optimizer, run the main model, and compute both losses.
    # 3. Use `forecast_loss + masked_loss_weight * masked_loss`.
    # 4. Backpropagate and step the main optimizer.
    # 5. Zero the baseline optimizer, run it on history, compute next-label loss,
    #    backpropagate, and step its optimizer.
    # 6. Count examples, correct predictions, and batch-size-weighted loss.
    # 7. Divide totals by the number of examples.
    #
    # Checks:
    # - no optimizer contains a frozen TimeSformer parameter
    # - returned accuracies are between `0.0` and `1.0`
    raise NotImplementedError("Train one cached-feature epoch.")








@torch.no_grad()
def evaluate_models(
    model: VideoTaskformerStudent,
    baseline: LastStepBaseline,
    loader: DataLoader,
    loss_fn: nn.Module,
    majority_label_id: int,
    num_classes: int,
) -> EvaluationMetrics:
    """Evaluate forecasting, masking, and both baselines without learning."""
    # Goal:
    # Return loss, top-1, top-3, macro F1, masked accuracy, baseline scores,
    # labels, predictions, and probability rows.
    #
    # Method:
    # - `torch.softmax(logits, dim=-1)` gives probabilities
    # - `.argmax(dim=-1)` gives top-1 IDs
    # - `.topk(k=min(3, num_classes), dim=-1).indices` gives top-3 IDs
    # - use `f1_score(..., labels=range(num_classes), average="macro", zero_division=0)`
    #
    # Steps:
    # 1. Put both models in eval mode and run every batch without optimizers.
    # 2. Compute forecast loss and accumulate all predictions on CPU.
    # 3. Count top-3, masked, majority, and last-step hits over examples.
    # 4. Concatenate probability batches and compute macro F1.
    #
    # Checks:
    # - every probability row sums to about `1.0`
    # - probability shape is `[len(loader.dataset), num_classes]`
    raise NotImplementedError("Evaluate next-action forecasting and baselines.")









def save_action_forecasting_visuals(
    training_history: list[dict[str, float]],
    test_metrics: EvaluationMetrics,
    example_window: ForecastWindow,
    vocabulary: VerbVocabulary,
    config: ActionForecastingConfig,
) -> dict[str, str]:
    """Save five readable plots with real action and verb names."""
    # Goal:
    # Save the five PNGs listed at the top and return their string paths.
    #
    # Method:
    # - use Matplotlib for every figure
    # - use `confusion_matrix(labels, predictions, labels=range(num_classes))`
    # - use `torch.topk(test_metrics["probabilities"][0], k=min(5, num_classes))`
    #
    # Steps:
    # 1. Create output_dir.
    # 2. Plot train/validation loss and accuracy with labels and legends.
    # 3. Plot the confusion matrix with readable verb names on both axes.
    # 4. Plot five timeline bars from the real timestamps and narrations. Mark
    #    the final true and predicted verbs.
    # 5. Plot the first test row's top probabilities with readable names.
    # 6. Plot training counts with readable names.
    # 7. Tighten, save, and close every figure.
    #
    # Checks:
    # - all returned paths exist and end in `.png`
    # - probability row zero matches `example_window`
    # - never display raw IDs without names
    raise NotImplementedError("Save action-forecasting metrics and real timelines.")














def run_action_forecasting_experiment(config: ActionForecastingConfig) -> ExperimentResult:
    """Run feature preparation, training, model selection, and final evaluation."""
    # Exact workflow:
    # 1. Set the seed, load real actions, build the train-only vocabulary, and
    #    build consecutive windows.
    # 2. Load frozen TimeSformer, extract or reuse features, and build loaders.
    # 3. Build the main model and baseline on `device`.
    # 4. Build separate `torch.optim.AdamW` optimizers with config learning rate
    #    and weight decay, plus `nn.CrossEntropyLoss()`.
    # 5. Find the majority target ID from training windows only.
    # 6. For every epoch, train once and evaluate only validation.
    # 7. Save history and print main plus baseline metrics.
    # 8. When validation top-1 improves, save both states with `deepcopy(...)`.
    # 9. Restore the best validation states after training.
    # 10. Evaluate validation, held-out kitchen, and final test once.
    # 11. Save visuals using the first final-test window and return all results.
    #
    # Checks:
    # - never select settings from held-out-kitchen or test accuracy
    # - print split sizes, coverage, main accuracy, and both baselines together
    # - raise a clear error if no best validation state was saved
    raise NotImplementedError("Run the full VideoTaskformer-style exercise.")















def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether the student has completed it."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the long-context action-forecasting scaffold."""
    config = ActionForecastingConfig()
    result, ok = report_task_progress(
        "run_action_forecasting_experiment", run_action_forecasting_experiment, config
    )
    if ok:
        print(result)


if __name__ == "__main__":
    main()
