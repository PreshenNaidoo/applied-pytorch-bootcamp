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
    training_steps = [step for step in steps if step["experiment_split"] == "train"]
    if not training_steps:
        raise ValueError("Cannot build the vocabulary because there are no training actions.")

    verb_counts = Counter(step["verb_class"] for step in training_steps)
    ordered_verb_classes = sorted(
        verb_counts,
        key=lambda verb_class: (-verb_counts[verb_class], verb_class),
    )
    selected_verb_classes = ordered_verb_classes[: config.num_common_verbs]
    if len(selected_verb_classes) < config.num_common_verbs:
        raise ValueError(
            f"Training data contains only {len(selected_verb_classes)} verb classes, "
            f"but the config asks for {config.num_common_verbs}."
        )

    verb_name_counts_by_class: dict[int, Counter[str]] = {}
    for step in training_steps:
        verb_class = step["verb_class"]
        verb_name = step["verb_name"]
        verb_name_counts_by_class.setdefault(verb_class, Counter())[verb_name] += 1

    verb_name_by_class = {
        verb_class: sorted(
            name_counts,
            key=lambda verb_name: (-name_counts[verb_name], verb_name),
        )[0]
        for verb_class, name_counts in verb_name_counts_by_class.items()
    }

    vocabulary = VerbVocabulary(
        verb_class_to_label_id={
            verb_class: label_id
            for label_id, verb_class in enumerate(selected_verb_classes)
        },
        label_id_to_name=[
            verb_name_by_class[verb_class] for verb_class in selected_verb_classes
        ],
        training_counts=[verb_counts[verb_class] for verb_class in selected_verb_classes],
    )

    print("Training-only verb vocabulary:")
    for label_id, verb_class in enumerate(selected_verb_classes):
        print(
            f"  label {label_id}: {verb_name_by_class[verb_class]} "
            f"(raw class {verb_class}, count {verb_counts[verb_class]})"
        )

    # raise NotImplementedError("Build a training-only verb vocabulary.")
    return vocabulary


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
    split_names: tuple[SplitName, ...] = ("train", "val", "unseen_kitchen", "test")
    steps_by_video: dict[tuple[SplitName, str], list[ActionStep]] = {}
    for step in steps:
        group_key = (step["experiment_split"], step["video_id"])
        steps_by_video.setdefault(group_key, []).append(step)

    windows_by_split: dict[SplitName, list[ForecastWindow]] = {
        split_name: [] for split_name in split_names
    }
    candidate_counts = {split_name: 0 for split_name in split_names}
    known_target_counts = {split_name: 0 for split_name in split_names}
    window_size = config.history_steps + 1

    for (split_name, video_id), video_steps in steps_by_video.items():
        ordered_steps = sorted(video_steps, key=lambda step: step["raw_order_index"])

        for start_index in range(len(ordered_steps) - window_size + 1):
            sequence_steps = ordered_steps[start_index : start_index + window_size]

            has_consecutive_order = all(
                next_step["raw_order_index"] == current_step["raw_order_index"] + 1
                for current_step, next_step in zip(sequence_steps, sequence_steps[1:])
            )
            has_acceptable_gaps = all(
                next_step["start_seconds"] - current_step["end_seconds"]
                <= config.max_gap_seconds
                for current_step, next_step in zip(sequence_steps, sequence_steps[1:])
            )
            if not has_consecutive_order or not has_acceptable_gaps:
                continue

            candidate_counts[split_name] += 1
            if sequence_steps[-1]["verb_class"] not in vocabulary.verb_class_to_label_id:
                continue

            known_positions = [
                position
                for position, step in enumerate(sequence_steps)
                if step["verb_class"] in vocabulary.verb_class_to_label_id
            ]
            if not known_positions:
                continue

            known_target_counts[split_name] += 1
            first_raw_index = sequence_steps[0]["raw_order_index"]
            mask_rng = random.Random(
                f"{config.random_state}:{split_name}:{video_id}:{first_raw_index}"
            )
            mask_index = mask_rng.choice(known_positions)
            windows_by_split[split_name].append(
                ForecastWindow(
                    video_id=video_id,
                    sequence_steps=sequence_steps,
                    mask_index=mask_index,
                )
            )

    known_label_coverage: dict[str, float] = {}
    for split_name in split_names:
        candidate_count = candidate_counts[split_name]
        known_label_coverage[split_name] = (
            known_target_counts[split_name] / candidate_count if candidate_count else 0.0
        )

        split_rng = random.Random(f"{config.random_state}:{split_name}")
        split_rng.shuffle(windows_by_split[split_name])
        maximum_windows = (
            config.max_train_windows if split_name == "train" else config.max_eval_windows
        )
        windows_by_split[split_name] = windows_by_split[split_name][:maximum_windows]

        if not windows_by_split[split_name]:
            raise ValueError(
                f"No usable {split_name} windows were built. Check action counts, "
                "time gaps, and training-vocabulary coverage."
            )

        print(
            f"{split_name}: {candidate_count} candidates, "
            f"{len(windows_by_split[split_name])} kept, "
            f"known-label coverage {known_label_coverage[split_name]:.1%}"
        )

    # raise NotImplementedError("Build real consecutive action windows.")
    return WindowBuildResult(
        windows_by_split=windows_by_split,
        known_label_coverage=known_label_coverage,
        candidate_counts=candidate_counts,
    )


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
    video_path = Path(step["video_path"])
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video for action {step['step_key']} does not exist: {video_path}"
        )
    if step["end_seconds"] <= step["start_seconds"]:
        raise ValueError(
            f"Action {step['step_key']} has an invalid time range: "
            f"{step['start_seconds']:.3f}s to {step['end_seconds']:.3f}s."
        )

    target_seconds = np.linspace(
        step["start_seconds"],
        step["end_seconds"],
        config.frames_per_step + 2,
    )[1:-1].tolist()
    sampled_frames: list[np.ndarray] = []

    try:
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                raise ValueError(f"The file has no video stream: {video_path}")

            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"
            if video_stream.time_base is None:
                raise ValueError(f"The video stream has no time base: {video_path}")

            seek_offset = max(
                0,
                int(target_seconds[0] / float(video_stream.time_base)),
            )
            container.seek(
                seek_offset,
                backward=True,
                any_frame=False,
                stream=video_stream,
            )

            target_index = 0
            for frame in container.decode(video_stream):
                frame_seconds = frame.time
                if frame_seconds is None and frame.pts is not None:
                    frame_seconds = float(frame.pts * video_stream.time_base)
                if frame_seconds is None:
                    continue

                rgb_frame: np.ndarray | None = None
                while (
                    target_index < len(target_seconds)
                    and float(frame_seconds) >= target_seconds[target_index]
                ):
                    if rgb_frame is None:
                        rgb_frame = frame.to_ndarray(format="rgb24")
                    sampled_frames.append(rgb_frame)
                    target_index += 1

                if target_index == len(target_seconds):
                    break
    except (av.FFmpegError, OSError) as error:
        raise RuntimeError(
            f"Could not decode {video_path} for action {step['step_key']} "
            f"between {step['start_seconds']:.3f}s and {step['end_seconds']:.3f}s."
        ) from error

    if len(sampled_frames) != config.frames_per_step:
        raise RuntimeError(
            f"Decoded only {len(sampled_frames)} of {config.frames_per_step} frames "
            f"from {video_path} for action {step['step_key']} between "
            f"{step['start_seconds']:.3f}s and {step['end_seconds']:.3f}s."
        )

    for frame in sampled_frames:
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"Action {step['step_key']} produced an invalid RGB frame with "
                f"shape {frame.shape} and dtype {frame.dtype}."
            )

    # raise NotImplementedError("Sample real frames from one EPIC action segment.")
    return sampled_frames


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

    from transformers import AutoImageProcessor, TimesformerModel
    processor = AutoImageProcessor.from_pretrained(config.timesformer_model_name)
    model = TimesformerModel.from_pretrained(config.timesformer_model_name, use_safetensors=True)

    model = model.to(device)
    model.eval()    

    for param in model.parameters():
        param.requires_grad = False

    if model.config.hidden_size != config.timesformer_feature_dim:
        raise ValueError("The hidden dimensions are not compatible.")    

    return processor, model
    #raise NotImplementedError("Load and freeze the pretrained TimeSformer.")


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
    required_steps: dict[str, ActionStep] = {}
    for windows in windows_by_split.values():
        for window in windows:
            for step in window["sequence_steps"]:
                step_key = step["step_key"]
                if step_key in required_steps:
                    previous_step = required_steps[step_key]
                    if (
                        previous_step["video_id"] != step["video_id"]
                        or previous_step["start_seconds"] != step["start_seconds"]
                        or previous_step["end_seconds"] != step["end_seconds"]
                    ):
                        raise ValueError(f"Step key {step_key!r} refers to two different actions.")
                else:
                    required_steps[step_key] = step

    if not required_steps:
        raise ValueError("No action steps were provided for feature extraction.")

    config.feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache_path = config.feature_cache_path.with_suffix(
        config.feature_cache_path.suffix + ".tmp"
    )

    def save_feature_checkpoint(features_to_save: dict[str, torch.Tensor]) -> None:
        torch.save(
            {
                "model_name": config.timesformer_model_name,
                "frames_per_step": config.frames_per_step,
                "features": features_to_save,
            },
            temporary_cache_path,
        )
        temporary_cache_path.replace(config.feature_cache_path)

    features: dict[str, torch.Tensor] = {}
    if config.feature_cache_path.exists():
        cache = torch.load(
            config.feature_cache_path,
            map_location="cpu",
            weights_only=True,
        )
        settings_match = (
            isinstance(cache, dict)
            and cache.get("model_name") == config.timesformer_model_name
            and cache.get("frames_per_step") == config.frames_per_step
            and isinstance(cache.get("features"), dict)
        )

        if settings_match:
            cached_features = cache["features"]
            for step_key in required_steps:
                cached_feature = cached_features.get(step_key)
                if (
                    isinstance(cached_feature, torch.Tensor)
                    and cached_feature.shape == (config.timesformer_feature_dim,)
                    and torch.isfinite(cached_feature).all().item()
                ):
                    features[step_key] = cached_feature.detach().float().cpu()

            if len(features) == len(required_steps):
                print(
                    f"Reusing {len(required_steps)} step features from "
                    f"{config.feature_cache_path}."
                )
                return features
            if features:
                print(
                    f"Resuming from {len(features)} cached features in "
                    f"{config.feature_cache_path}."
                )
        else:
            print("The existing feature cache does not match this run. Rebuilding it.")

    total_steps = len(required_steps)
    steps_to_extract = [
        (step_key, step)
        for step_key, step in required_steps.items()
        if step_key not in features
    ]
    starting_feature_count = len(features)
    progress = (
        tqdm(
            steps_to_extract,
            desc="extract features",
            initial=starting_feature_count,
            total=total_steps,
            unit="step",
        )
        if tqdm is not None
        else steps_to_extract
    )

    try:
        with torch.no_grad():
            for new_step_number, (step_key, step) in enumerate(progress, start=1):
                if tqdm is None:
                    current_step_number = starting_feature_count + new_step_number
                    print(f"Extracting feature {current_step_number}/{total_steps}: {step_key}")

                frames = sample_action_frames(step, config)
                processed = processor(frames, return_tensors="pt")
                if "pixel_values" not in processed:
                    raise KeyError("The image processor did not return pixel_values.")

                pixel_values = processed["pixel_values"]
                expected_pixel_shape = (1, config.frames_per_step, 3, 224, 224)
                if tuple(pixel_values.shape) != expected_pixel_shape:
                    raise ValueError(
                        f"The processor returned shape {tuple(pixel_values.shape)} for "
                        f"{step_key}; expected {expected_pixel_shape}."
                    )

                outputs = timesformer(pixel_values=pixel_values.to(device))
                feature = outputs.last_hidden_state[:, 0].squeeze(0).detach().float().cpu()
                expected_feature_shape = (config.timesformer_feature_dim,)
                if tuple(feature.shape) != expected_feature_shape:
                    raise ValueError(
                        f"The feature for {step_key} has shape {tuple(feature.shape)}; "
                        f"expected {expected_feature_shape}."
                    )
                if not torch.isfinite(feature).all().item():
                    raise ValueError(f"The feature for {step_key} contains NaN or infinity.")

                features[step_key] = feature
                if new_step_number % 10 == 0:
                    save_feature_checkpoint(features)
                if tqdm is not None:
                    progress.set_postfix(saved=f"{len(features)}/{total_steps}")
    finally:
        if len(features) > starting_feature_count:
            save_feature_checkpoint(features)

    print(f"Saved {len(features)} step features to {config.feature_cache_path}.")

    # raise NotImplementedError("Extract and cache frozen TimeSformer step features.")
    return features


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
    split_names: tuple[SplitName, ...] = ("train", "val", "unseen_kitchen", "test")
    dataloaders: dict[SplitName, DataLoader] = {}

    for split_name in split_names:
        windows = window_result.windows_by_split[split_name]
        if not windows:
            raise ValueError(f"Cannot build the {split_name} DataLoader: it has no windows.")

        required_step_keys = {
            step["step_key"]
            for window in windows
            for step in window["sequence_steps"]
        }
        missing_step_keys = sorted(required_step_keys - set(feature_by_step_key))
        if missing_step_keys:
            raise KeyError(
                f"The feature cache is missing {len(missing_step_keys)} steps needed by "
                f"{split_name}, including {missing_step_keys[:5]}."
            )

        for step_key in required_step_keys:
            feature = feature_by_step_key[step_key]
            expected_shape = (config.timesformer_feature_dim,)
            if tuple(feature.shape) != expected_shape:
                raise ValueError(
                    f"Cached feature {step_key!r} has shape {tuple(feature.shape)}; "
                    f"expected {expected_shape}."
                )
            if not torch.isfinite(feature).all().item():
                raise ValueError(f"Cached feature {step_key!r} contains NaN or infinity.")

        dataset = CachedWindowDataset(
            windows=windows,
            feature_by_step_key=feature_by_step_key,
            vocabulary=vocabulary,
            history_steps=config.history_steps,
        )
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split_name == "train",
            num_workers=0,
        )
        print(
            f"{split_name} DataLoader: {len(dataset)} examples, "
            f"{len(dataloaders[split_name])} batches"
        )

    example_batch = next(iter(dataloaders["val"]))
    expected_trailing_shapes = {
        "history_features": (config.history_steps, config.timesformer_feature_dim),
        "masked_features": (config.history_steps + 1, config.timesformer_feature_dim),
    }
    for key, expected_shape in expected_trailing_shapes.items():
        if tuple(example_batch[key].shape[1:]) != expected_shape:
            raise ValueError(
                f"Batch field {key!r} has shape {tuple(example_batch[key].shape)}; "
                f"expected [batch, {', '.join(str(size) for size in expected_shape)}]."
            )

    for key in ("mask_index", "next_label", "masked_label"):
        if example_batch[key].ndim != 1:
            raise ValueError(
                f"Batch field {key!r} has shape {tuple(example_batch[key].shape)}; "
                "expected [batch]."
            )

    print("Example validation batch shapes:")
    for key, value in example_batch.items():
        print(f"  {key}: {tuple(value.shape)}")

    # raise NotImplementedError("Build cached-feature dataloaders.")
    return dataloaders


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
        self.feature_projection = nn.Linear(
            config.timesformer_feature_dim,
            config.hidden_dim,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, config.history_steps + 1, config.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.step_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_depth,
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, num_classes)

        # Fill the learned mask token with small random starting values. The
        # values come from a normal distribution centred on 0. A standard
        # deviation of 0.02 means most values start close to 0, so the token
        # does not create a very large signal before training begins.
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        # Give each learned step-position vector small random starting values.
        # This lets positions start slightly different from one another, while
        # keeping their initial effect small. The trailing `_` in `normal_`
        # means PyTorch changes this parameter directly instead of returning a
        # separate tensor.
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        # raise NotImplementedError("Build the step-context transformer.")

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
        if history_features.ndim != 3 or masked_features.ndim != 3:
            raise ValueError("History and masked features must both have three dimensions.")
        if history_features.shape[0] != masked_features.shape[0]:
            raise ValueError("History and masked features must have the same batch size.")
        if history_features.shape[2] != self.feature_projection.in_features:
            raise ValueError(
                f"History feature size is {history_features.shape[2]}; "
                f"expected {self.feature_projection.in_features}."
            )
        if masked_features.shape[2] != self.feature_projection.in_features:
            raise ValueError(
                f"Masked feature size is {masked_features.shape[2]}; "
                f"expected {self.feature_projection.in_features}."
            )
        if history_features.shape[1] != self.position_embedding.shape[1] - 1:
            raise ValueError(
                f"History contains {history_features.shape[1]} steps; "
                f"expected {self.position_embedding.shape[1] - 1}."
            )
        if masked_features.shape[1] != self.position_embedding.shape[1]:
            raise ValueError(
                f"Masked input contains {masked_features.shape[1]} steps; "
                f"expected {self.position_embedding.shape[1]}."
            )
        if mask_index.ndim != 1 or mask_index.shape[0] != masked_features.shape[0]:
            raise ValueError("mask_index must have shape [batch].")

        mask_index = mask_index.to(device=masked_features.device, dtype=torch.long)
        if torch.any(mask_index < 0) or torch.any(mask_index >= masked_features.shape[1]):
            raise ValueError("mask_index contains a position outside the masked sequence.")

        history_tokens = self.feature_projection(history_features)
        history_tokens = history_tokens + self.position_embedding[:, : history_tokens.shape[1]]
        history_context = self.step_transformer(history_tokens)
        forecast_vector = self.output_norm(history_context[:, -1])
        forecast_logits = self.classifier(forecast_vector)

        masked_tokens = self.feature_projection(masked_features).clone()
        batch_ids = torch.arange(masked_tokens.shape[0], device=masked_tokens.device)
        masked_tokens[batch_ids, mask_index] = self.mask_token[0, 0]
        masked_tokens = masked_tokens + self.position_embedding[:, : masked_tokens.shape[1]]
        masked_context = self.step_transformer(masked_tokens)
        masked_vector = self.output_norm(masked_context[batch_ids, mask_index])
        masked_logits = self.classifier(masked_vector)

        if not torch.isfinite(forecast_logits).all() or not torch.isfinite(masked_logits).all():
            raise ValueError("The model produced NaN or infinity in its logits.")

        # raise NotImplementedError("Run forecasting and masked-step prediction.")
        return ModelOutputs(
            forecast_logits=forecast_logits,
            masked_logits=masked_logits,
        )


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
    model.train()
    baseline.train()

    total_examples = 0
    total_loss = 0.0
    forecast_correct = 0
    masked_correct = 0
    last_step_correct = 0

    progress = tqdm(loader, desc="train", leave=False) if tqdm is not None else loader

    for step_number, batch in enumerate(progress, start=1):
        history_features = batch["history_features"].to(device)
        masked_features = batch["masked_features"].to(device)
        mask_index = batch["mask_index"].to(device)
        next_label = batch["next_label"].to(device)
        masked_label = batch["masked_label"].to(device)

        model_optimizer.zero_grad(set_to_none=True)
        outputs = model(history_features, masked_features, mask_index)
        forecast_loss = loss_fn(outputs["forecast_logits"], next_label)
        masked_loss = loss_fn(outputs["masked_logits"], masked_label)
        loss = forecast_loss + config.masked_loss_weight * masked_loss
        if not torch.isfinite(loss).item():
            raise ValueError("The main model produced a non-finite training loss.")
        loss.backward()
        model_optimizer.step()

        baseline_optimizer.zero_grad(set_to_none=True)
        last_step_logits = baseline(history_features)
        baseline_loss = loss_fn(last_step_logits, next_label)
        if not torch.isfinite(baseline_loss).item():
            raise ValueError("The last-step baseline produced a non-finite training loss.")
        baseline_loss.backward()
        baseline_optimizer.step()

        batch_size = next_label.shape[0]
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        forecast_correct += (
            outputs["forecast_logits"].argmax(dim=-1) == next_label
        ).sum().item()
        masked_correct += (
            outputs["masked_logits"].argmax(dim=-1) == masked_label
        ).sum().item()
        last_step_correct += (last_step_logits.argmax(dim=-1) == next_label).sum().item()

        if tqdm is not None:
            progress.set_postfix(
                step=f"{step_number}/{len(loader)}",
                loss=f"{loss.item():.4f}",
                forecast=f"{forecast_correct / total_examples:.3f}",
                masked=f"{masked_correct / total_examples:.3f}",
                last_step=f"{last_step_correct / total_examples:.3f}",
            )

    if total_examples == 0:
        raise ValueError("Cannot train on an empty DataLoader.")

    metrics = EpochMetrics(
        mean_loss=total_loss / total_examples,
        forecast_accuracy=forecast_correct / total_examples,
        masked_accuracy=masked_correct / total_examples,
        last_step_accuracy=last_step_correct / total_examples,
    )
    for metric_name in ("forecast_accuracy", "masked_accuracy", "last_step_accuracy"):
        if not 0.0 <= metrics[metric_name] <= 1.0:
            raise ValueError(f"{metric_name} is outside the valid range from 0 to 1.")

    # raise NotImplementedError("Train one cached-feature epoch.")
    return metrics


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
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")
    if not 0 <= majority_label_id < num_classes:
        raise ValueError("majority_label_id must be a valid class ID.")

    model.eval()
    baseline.eval()

    total_examples = 0
    total_loss = 0.0
    forecast_correct = 0
    top3_correct = 0
    masked_correct = 0
    majority_correct = 0
    last_step_correct = 0
    labels: list[int] = []
    predictions: list[int] = []
    probability_batches: list[torch.Tensor] = []

    for batch in loader:
        history_features = batch["history_features"].to(device)
        masked_features = batch["masked_features"].to(device)
        mask_index = batch["mask_index"].to(device)
        next_label = batch["next_label"].to(device)
        masked_label = batch["masked_label"].to(device)

        outputs = model(history_features, masked_features, mask_index)
        forecast_logits = outputs["forecast_logits"]
        masked_logits = outputs["masked_logits"]
        last_step_logits = baseline(history_features)

        expected_logit_shape = (next_label.shape[0], num_classes)
        for name, logits in (
            ("forecast_logits", forecast_logits),
            ("masked_logits", masked_logits),
            ("last_step_logits", last_step_logits),
        ):
            if tuple(logits.shape) != expected_logit_shape:
                raise ValueError(
                    f"{name} has shape {tuple(logits.shape)}; "
                    f"expected {expected_logit_shape}."
                )

        forecast_loss = loss_fn(forecast_logits, next_label)
        probabilities = torch.softmax(forecast_logits, dim=-1)
        forecast_prediction = forecast_logits.argmax(dim=-1)
        masked_prediction = masked_logits.argmax(dim=-1)
        last_step_prediction = last_step_logits.argmax(dim=-1)
        top3_indices = forecast_logits.topk(k=min(3, num_classes), dim=-1).indices

        batch_size = next_label.shape[0]
        total_examples += batch_size
        total_loss += forecast_loss.item() * batch_size
        forecast_correct += (forecast_prediction == next_label).sum().item()
        top3_correct += (
            top3_indices == next_label.unsqueeze(dim=1)
        ).any(dim=1).sum().item()
        masked_correct += (masked_prediction == masked_label).sum().item()
        majority_correct += (next_label == majority_label_id).sum().item()
        last_step_correct += (last_step_prediction == next_label).sum().item()

        labels.extend(next_label.cpu().tolist())
        predictions.extend(forecast_prediction.cpu().tolist())
        probability_batches.append(probabilities.cpu())

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty DataLoader.")

    all_probabilities = torch.cat(probability_batches, dim=0)
    expected_probability_shape = (total_examples, num_classes)
    if tuple(all_probabilities.shape) != expected_probability_shape:
        raise ValueError(
            f"Probability tensor has shape {tuple(all_probabilities.shape)}; "
            f"expected {expected_probability_shape}."
        )
    if not torch.allclose(
        all_probabilities.sum(dim=-1),
        torch.ones(total_examples),
        atol=1e-5,
    ):
        raise ValueError("One or more probability rows do not sum to 1.")

    macro_f1 = f1_score(
        labels,
        predictions,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    # raise NotImplementedError("Evaluate next-action forecasting and baselines.")
    return EvaluationMetrics(
        mean_loss=total_loss / total_examples,
        top1_accuracy=forecast_correct / total_examples,
        top3_accuracy=top3_correct / total_examples,
        macro_f1=float(macro_f1),
        masked_accuracy=masked_correct / total_examples,
        majority_accuracy=majority_correct / total_examples,
        last_step_accuracy=last_step_correct / total_examples,
        labels=labels,
        predictions=predictions,
        probabilities=all_probabilities,
    )


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
    if not training_history:
        raise ValueError("training_history must contain at least one epoch.")

    history_keys = (
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
    )
    for epoch_index, epoch_metrics in enumerate(training_history, start=1):
        missing_keys = [key for key in history_keys if key not in epoch_metrics]
        if missing_keys:
            raise KeyError(f"Training epoch {epoch_index} is missing keys: {missing_keys}")

    num_classes = vocabulary.num_classes
    if num_classes == 0:
        raise ValueError("Cannot draw plots for an empty vocabulary.")
    if len(vocabulary.training_counts) != num_classes:
        raise ValueError("Vocabulary names and training counts must have the same length.")
    if len(test_metrics["labels"]) != len(test_metrics["predictions"]):
        raise ValueError("Test labels and predictions must have the same length.")
    if not test_metrics["labels"]:
        raise ValueError("Test metrics contain no examples to plot.")

    probabilities = test_metrics["probabilities"]
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError(
            f"Test probabilities have shape {tuple(probabilities.shape)}; "
            f"expected [examples, {num_classes}]."
        )
    if probabilities.shape[0] != len(test_metrics["labels"]):
        raise ValueError("Probability rows must match the number of test labels.")

    sequence_steps = example_window["sequence_steps"]
    if len(sequence_steps) != config.history_steps + 1:
        raise ValueError(
            f"The example window has {len(sequence_steps)} steps; "
            f"expected {config.history_steps + 1}."
        )
    true_label_id = vocabulary.label_id(sequence_steps[-1])
    if test_metrics["labels"][0] != true_label_id:
        raise ValueError("Probability row zero does not match the supplied example window.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    visual_paths = {
        "training_curves": config.output_dir / "training_curves.png",
        "confusion_matrix": config.output_dir / "confusion_matrix.png",
        "forecast_timeline": config.output_dir / "forecast_timeline.png",
        "next_verb_probabilities": config.output_dir / "next_verb_probabilities.png",
        "class_distribution": config.output_dir / "class_distribution.png",
    }

    epochs = list(range(1, len(training_history) + 1))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in training_history], label="Train")
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in training_history],
        label="Validation",
    )
    axes[0].set_title("Forecasting loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in training_history],
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["validation_accuracy"] for row in training_history],
        label="Validation",
    )
    axes[1].set_title("Next-verb accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(visual_paths["training_curves"], dpi=150)
    plt.close(figure)

    matrix = confusion_matrix(
        test_metrics["labels"],
        test_metrics["predictions"],
        labels=list(range(num_classes)),
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Next-verb confusion matrix")
    axis.set_xlabel("Predicted verb")
    axis.set_ylabel("True verb")
    axis.set_xticks(range(num_classes), vocabulary.label_id_to_name, rotation=45, ha="right")
    axis.set_yticks(range(num_classes), vocabulary.label_id_to_name)
    for true_id in range(num_classes):
        for predicted_id in range(num_classes):
            axis.text(predicted_id, true_id, str(matrix[true_id, predicted_id]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Number of examples")
    figure.tight_layout()
    figure.savefig(visual_paths["confusion_matrix"], dpi=150)
    plt.close(figure)

    first_start = sequence_steps[0]["start_seconds"]
    relative_starts = [step["start_seconds"] - first_start for step in sequence_steps]
    durations = [step["end_seconds"] - step["start_seconds"] for step in sequence_steps]
    positions = list(range(len(sequence_steps)))
    colors = ["tab:blue"] * config.history_steps + ["tab:orange"]
    predicted_label_id = int(probabilities[0].argmax().item())
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.barh(positions, durations, left=relative_starts, color=colors)
    axis.set_yticks(
        positions,
        [f"{index + 1}. {step['narration']}" for index, step in enumerate(sequence_steps)],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Seconds from the start of this window")
    axis.set_title(
        f"True next verb: {vocabulary.label_id_to_name[true_label_id]} | "
        f"Predicted: {vocabulary.label_id_to_name[predicted_label_id]}"
    )
    figure.tight_layout()
    figure.savefig(visual_paths["forecast_timeline"], dpi=150)
    plt.close(figure)

    top_count = min(5, num_classes)
    top_probabilities, top_label_ids = torch.topk(probabilities[0], k=top_count)
    top_names = [vocabulary.label_id_to_name[label_id] for label_id in top_label_ids.tolist()]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(top_names, top_probabilities.tolist(), color="tab:purple")
    axis.set_title("Top next-verb predictions")
    axis.set_xlabel("Verb")
    axis.set_ylabel("Probability")
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(visual_paths["next_verb_probabilities"], dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(
        vocabulary.label_id_to_name,
        vocabulary.training_counts,
        color="tab:green",
    )
    axis.set_title("Training examples per verb")
    axis.set_xlabel("Verb")
    axis.set_ylabel("Number of actions")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(visual_paths["class_distribution"], dpi=150)
    plt.close(figure)

    for path in visual_paths.values():
        if not path.exists() or path.suffix.lower() != ".png":
            raise RuntimeError(f"Expected visual was not saved: {path}")

    # raise NotImplementedError("Save action-forecasting metrics and real timelines.")
    return {name: str(path) for name, path in visual_paths.items()}


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
    set_seed(config.random_state)

    steps = load_manifested_action_steps(config)
    vocabulary = build_training_verb_vocabulary(steps, config)
    window_result = build_consecutive_windows(steps, vocabulary, config)
    processor, timesformer = load_frozen_timesformer(config)
    feature_by_step_key = extract_and_cache_step_features(
        window_result.windows_by_split,
        processor,
        timesformer,
        config,
    )
    dataloaders = build_cached_dataloaders(
        window_result,
        feature_by_step_key,
        vocabulary,
        config,
    )

    del processor
    del timesformer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = VideoTaskformerStudent(config, vocabulary.num_classes).to(device)
    baseline = LastStepBaseline(
        config.timesformer_feature_dim,
        vocabulary.num_classes,
    ).to(device)
    model_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    baseline_optimizer = torch.optim.AdamW(
        baseline.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    main_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    baseline_parameter_count = sum(
        parameter.numel() for parameter in baseline.parameters() if parameter.requires_grad
    )
    print("\nAction-forecasting training setup")
    print(f"  device: {device}")
    print(f"  verb classes: {vocabulary.num_classes}")
    print(f"  main model trainable parameters: {main_parameter_count:,}")
    print(f"  baseline trainable parameters: {baseline_parameter_count:,}")
    print(
        "  windows: "
        + ", ".join(
            f"{split_name}={len(window_result.windows_by_split[split_name])}"
            for split_name in ("train", "val", "unseen_kitchen", "test")
        )
    )

    training_target_counts = Counter(
        vocabulary.label_id(window["sequence_steps"][-1])
        for window in window_result.windows_by_split["train"]
    )
    if not training_target_counts:
        raise ValueError("Cannot find a majority label because training has no windows.")
    majority_label_id = min(
        training_target_counts,
        key=lambda label_id: (-training_target_counts[label_id], label_id),
    )
    print(
        f"Majority baseline always predicts label {majority_label_id}: "
        f"{vocabulary.label_id_to_name[majority_label_id]}"
    )

    training_history: list[dict[str, float]] = []
    best_validation_accuracy = -1.0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_baseline_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            baseline,
            dataloaders["train"],
            model_optimizer,
            baseline_optimizer,
            loss_fn,
            config,
        )
        validation_metrics = evaluate_models(
            model,
            baseline,
            dataloaders["val"],
            loss_fn,
            majority_label_id,
            vocabulary.num_classes,
        )

        training_history.append(
            {
                "train_loss": train_metrics["mean_loss"],
                "validation_loss": validation_metrics["mean_loss"],
                "train_accuracy": train_metrics["forecast_accuracy"],
                "validation_accuracy": validation_metrics["top1_accuracy"],
            }
        )
        print(f"\nepoch {epoch}/{config.epochs}")
        print(
            f"  train | combined loss {train_metrics['mean_loss']:.4f} | "
            f"forecast accuracy {100.0 * train_metrics['forecast_accuracy']:.2f}% | "
            f"masked accuracy {100.0 * train_metrics['masked_accuracy']:.2f}% | "
            f"last-step accuracy {100.0 * train_metrics['last_step_accuracy']:.2f}%"
        )
        print(
            f"  val   | forecast loss {validation_metrics['mean_loss']:.4f} | "
            f"top-1 {100.0 * validation_metrics['top1_accuracy']:.2f}% | "
            f"top-3 {100.0 * validation_metrics['top3_accuracy']:.2f}% | "
            f"macro F1 {validation_metrics['macro_f1']:.3f} | "
            f"masked {100.0 * validation_metrics['masked_accuracy']:.2f}% | "
            f"last-step {100.0 * validation_metrics['last_step_accuracy']:.2f}%"
        )

        if validation_metrics["top1_accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["top1_accuracy"]
            best_model_state = deepcopy(model.state_dict())
            best_baseline_state = deepcopy(baseline.state_dict())

    if best_model_state is None or best_baseline_state is None:
        raise RuntimeError("Training finished without saving a best validation model.")

    model.load_state_dict(best_model_state)
    baseline.load_state_dict(best_baseline_state)

    validation_metrics = evaluate_models(
        model,
        baseline,
        dataloaders["val"],
        loss_fn,
        majority_label_id,
        vocabulary.num_classes,
    )
    unseen_kitchen_metrics = evaluate_models(
        model,
        baseline,
        dataloaders["unseen_kitchen"],
        loss_fn,
        majority_label_id,
        vocabulary.num_classes,
    )
    test_metrics = evaluate_models(
        model,
        baseline,
        dataloaders["test"],
        loss_fn,
        majority_label_id,
        vocabulary.num_classes,
    )

    final_metrics = {
        "validation": validation_metrics,
        "unseen_kitchen": unseen_kitchen_metrics,
        "test": test_metrics,
    }
    print("\nFinal action-forecasting metrics")
    for split_name, metrics in final_metrics.items():
        coverage_split_name = "val" if split_name == "validation" else split_name
        print(
            f"  {split_name:<14} | loss {metrics['mean_loss']:.4f} | "
            f"top-1 {100.0 * metrics['top1_accuracy']:.2f}% | "
            f"top-3 {100.0 * metrics['top3_accuracy']:.2f}% | "
            f"macro F1 {metrics['macro_f1']:.3f} | "
            f"masked {100.0 * metrics['masked_accuracy']:.2f}%"
        )
        print(
            f"  {'':<14} | majority {100.0 * metrics['majority_accuracy']:.2f}% | "
            f"last-step {100.0 * metrics['last_step_accuracy']:.2f}% | "
            f"known-label coverage "
            f"{100.0 * window_result.known_label_coverage[coverage_split_name]:.2f}%"
        )

    visual_paths = save_action_forecasting_visuals(
        training_history,
        test_metrics,
        window_result.windows_by_split["test"][0],
        vocabulary,
        config,
    )
    print(f"visuals saved in: {config.output_dir}")
    for visual_name, visual_path in visual_paths.items():
        print(f"  {visual_name}: {visual_path}")

    # raise NotImplementedError("Run the full VideoTaskformer-style exercise.")
    return ExperimentResult(
        validation=validation_metrics,
        unseen_kitchen=unseen_kitchen_metrics,
        test=test_metrics,
        known_label_coverage=window_result.known_label_coverage,
        visual_paths=visual_paths,
    )


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
