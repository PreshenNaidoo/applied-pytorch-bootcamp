"""
EXERCISE 09: TRAIN A DINO WORLD MODEL ON PUSHT
==============================================

WHAT YOU ARE BUILDING

PushT is a tabletop task. A round pusher must move a T-shaped block to a goal.
You will train a model that answers this question:

    If the scene looks like this, and the pusher follows these controls,
    what will the scene look like next?

The model does not predict pixels. A frozen DINOv2 image encoder first turns
each image into a grid of useful visual features. Your world model predicts the
next grid of features. Exercise 10 will use those predictions to choose actions.

Reference paper: https://arxiv.org/abs/2411.04983
Reference code: https://github.com/gaoyuezhou/dino_wm

THE FULL WORKFLOW

```text
record real PushT attempts
-> use frozen DINOv2 to describe every image
-> train the world model to predict the next description
-> try possible future actions inside the world model
-> choose an action and use it in the real environment
```

This exercise covers the middle two steps. The recordings and pretrained
DINOv2 model already exist. This file does not train an actor or step the real
environment.

THE DATA

Each row in the public recording contains:

- an image of the pusher and block
- the pusher's current `[x, y]` position
- an action `[x, y]`, meaning the target position requested by the controller
- episode and frame numbers

The block's simulator coordinates are hidden. The model must get information
about the block from the image features.

```text
real object                     name used here
---------------------------------------------------------------------
recorded table                  source
one complete PushT attempt      episode
image description               DINOv2 patch features
pusher position description     learned state embedding
several controls in order       action chunk
next scene description          target observation
model of what happens next      DINOWorldModel
```

WHAT LEARNS

DINOv2 is already trained and stays frozen. It receives a 196 x 196 image and
splits it into a 14 x 14 grid. This gives 196 patch features, each containing
384 numbers. The grid matters because the task depends on where the pusher and
block are.

The state encoder, action encoder, position values, and Transformer start
untrained. They learn PushT from the recordings. The model never turns its
predicted features back into an image in this exercise.

The reference task image size is 224, but its DINO path resizes those images to
196 so 14-pixel patches make a 14 x 14 feature grid. This exercise starts from
the public dataset's 96 x 96 camera image and applies that same 196 x 196 DINO
resize. It also uses the paper's six predictor blocks, 16 attention heads,
64 values per head, 2048-wide MLP, learning rates, and 100 epochs. The
public dataset has 206 episodes, not the paper's much larger PushT dataset. We
use 160 for training and 40 for validation, keeping whole episodes separate.

ONE TRAINING EXAMPLE

The default history is three model steps. One model step covers five recorded
actions. The selected frames and controls line up like this:

```text
frame 0  -- actions 0..4  --> frame 5
frame 5  -- actions 5..9  --> frame 10
frame 10 -- actions 10..14 --> frame 15
```

Frames 0, 5, and 10 are inputs. Frames 5, 10, and 15 are their targets. This is
why the batch stores four frames but only three action chunks.

One action chunk contains five ordered actions and has shape `[5, 2]`. Each
action is one `[x, y]` target position for the pusher. The environment receives
the five actions one at a time, so one chunk covers five environment steps.

For example, one complete action chunk could be:

```text
action 0: [250, 300]
action 1: [255, 302]
action 2: [260, 305]
action 3: [265, 307]
action 4: [270, 310]
```

The action encoder keeps that order and flattens the chunk to:

```text
[250, 300, 255, 302, 260, 305, 265, 307, 270, 310]
```

This means "request `[250, 300]`, then `[255, 302]`, and continue in that
order."

HOW ONE BATCH MOVES THROUGH THE MODEL

`B` is batch size, `H=3` is history length, `S=5` is actions per chunk,
and `P=196` is the number of image patches.

```text
features             [B, 4, P, 384]   DINO descriptions of four images
states               [B, 4, 2]        pusher x/y positions
actions              [B, 3, S, 2]     controls between those images
observation tokens   [B, 4, P, 394]   image features + state embedding
model input          [B, 3, P, 404]   observation + action embedding
predictions          [B, 3, P, 394]   three predicted next observations
targets              [B, 3, P, 394]   three real next observations
```

The same state and action description is attached to every patch in its frame.
The 196 visual patch features stay separate.

RAW VALUES, NORMALIZED VALUES, AND EMBEDDINGS

Do not mix up these three forms:

1. Raw value: an x/y pair from the recording, such as `[250, 300]`.
2. Normalized value: `(raw - training mean) / training standard deviation`.
3. Embedding: learned numbers made by `state_encoder` or `action_encoder`.

The dataset code normalizes raw x/y values. The model turns those normalized
values into embeddings.

WHY THE TRANSFORMER NEEDS A MASK

Frame 5 is the target for the first input, but it is also the second input. If
the first prediction could read frame 5, it could copy the answer. The mask
stops this leak:

```text
prediction for frame 5    may read frame 0
prediction for frame 10   may read frames 0 and 5
prediction for frame 15   may read frames 0, 5, and 10
```

All patches may read one another inside the same input frame. No patch may read
a later frame.

LOSS AND ONE TRAINING UPDATE

```text
loss = mean((prediction - detached target) ** 2)
```

"Detached" means the target numbers stay the same, but gradients cannot change
them just to make the loss smaller. The input path still trains normally.


TRAINING AND IMAGINATION ARE DIFFERENT

During training, every input frame is real. This is often called teacher
forcing. During imagination, the model feeds its own last prediction back in:

```text
[real 0, real 5, real 10]       -> predict 15
[real 5, real 10, predicted 15] -> predict 20
```

Each imagined step also receives a proposed action chunk. Prediction errors can
grow over several steps, so a good one-step loss does not guarantee good plans.

WORK THROUGH THE STUBS IN THIS ORDER

1. `build_window`: line up frames with all actions between them.
2. `frame_causal_mask`: stop predictions from reading future frames.
3. `PredictorAttention`: build and run masked multi-head attention.
4. `PredictorBlock`: add attention and feed-forward residual updates.
5. `PredictorTransformer`: stack all six predictor blocks.
6. `join_conditioning`: attach state and action information to each patch.
7. `predict`: run the masked Transformer.
8. `prediction_loss`: compare predictions with detached targets.
9. `train_batch`: perform one weight update.
10. `imagine`: reuse predictions to make a possible future.

Data loading, DINO feature caching, evaluation, plots, and checkpoint saving are
provided. Each stub explains its purpose, gives a concrete input/output example,
and lists the implementation steps in order.

HOW TO TELL WHETHER IT WORKED

- `mse`: prediction error on real held-out next frames. Lower is better.
- `persistence_mse`: error from guessing that nothing changes.
- `shuffled_action_mse`: error when the model receives the wrong actions.
- `cosine`: agreement in feature direction. Closer to 1 is better.

The model should beat the "nothing changes" guess, and correct actions should
beat shuffled actions. If not, first check frame/action alignment and whether
action embeddings reach the Transformer. Exercise 10 tests whether the learned
predictions are useful for planning.

HOW TO START

Run the file from the repository root. It first prints one data example and
stops at `build_window`. Complete and check one stub at a time. DINO feature
caching starts after the first stub and can take time and several GB of space;
later runs reuse the cache.

```text
python -m exercises.world_models.09_dino_wm_training
```

When completing a stub, keep its original `raise NotImplementedError(...)` line
and comment it out.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from PIL.Image import Image

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Data and frozen image encoder. DINOv2-small fixes visual_dim=384. Its
    # 14-pixel patches turn a 196-pixel side into grid=14, or 196 total patches.
    dataset: str = "lerobot/pusht_image"
    encoder: str = "facebook/dinov2-small"
    image_size: int = 196
    # Time layout. history=3 means three observations enter the predictor.
    # action_chunk=5 means each jump between observations contains five controls.
    # window_stride=5 moves the next example's start by five recorded rows.
    history: int = 3
    action_chunk: int = 5
    grid: int = 14
    visual_dim: int = 384
    condition_dim: int = 10
    # Predictor size. The token width is computed below as 384+10+10=404.
    # Attention temporarily projects it to heads*head_dim=16*64=1024.
    depth: int = 6
    heads: int = 16
    head_dim: int = 64
    mlp_dim: int = 2048
    # Training and caching. cache_batch_size affects only frozen DINO encoding;
    # batch_size affects world-model training and therefore gradient memory.
    batch_size: int = 32
    cache_batch_size: int = 32
    epochs: int = 100
    predictor_lr: float = 5e-5
    action_encoder_lr: float = 5e-4
    train_episodes: int = 160
    val_episodes: int = 40
    window_stride: int = 5
    seed: int = 42
    output: str = str(ROOT / "outputs/dino_wm_pusht")

    def __post_init__(self) -> None:
        # Fail at startup instead of producing confusing empty loaders, invalid
        # tensor shapes, or an optimizer that cannot learn.
        if min(self.history, self.action_chunk, self.batch_size, self.epochs,
               self.train_episodes, self.val_episodes, self.window_stride,
               self.heads, self.head_dim) < 1:
            raise ValueError("History, chunks, batches, epochs and episode counts must be positive.")
        if self.grid != self.image_size // 14 or self.image_size % 14:
            # DINOv2-small uses 14 x 14 pixel patches. With no pooling, the
            # configured grid must exactly match the resized image.
            raise ValueError("DINO patch grid must match image_size / 14; no pooling is used.")
        if min(self.predictor_lr, self.action_encoder_lr) <= 0:
            raise ValueError("Learning rates must be positive.")

    @property
    def width(self) -> int:
        # One predictor input token: 384 visual + 10 state + 10 action = 404.
        return self.visual_dim + 2 * self.condition_dim

    @property
    def observation_dim(self) -> int:
        # One prediction target: 384 visual + 10 state = 394. Actions explain
        # what causes the next observation; they are not part of that target.
        return self.visual_dim + self.condition_dim


def load_source(config: Config) -> HFDataset:
    # Where this fits: this is the first data step. It downloads or reuses the
    # public PushT recordings and checks that later functions can find the
    # columns they need.
    #
    # A Hugging Face Dataset behaves like a table. source[row_id] returns one
    # recorded step as a dictionary. source["action"] returns the full action
    # column. The public dataset calls its only file split "train", but we have
    # not made our own training/validation episode split yet.
    from datasets import load_dataset
    source = load_dataset(config.dataset, split="train")
    required = {"observation.image", "observation.state", "action", "episode_index", "frame_index"}
    if not required.issubset(source.column_names):
        raise ValueError(f"Expected image-row dataset fields {sorted(required)}; got {source.column_names}.")
    row = source[0]
    print("Source:", row["observation.image"].size, row["observation.state"], row["action"])
    return source


def group_episodes(source: HFDataset) -> dict[int, list[int]]:
    # Where this fits: build_window must never join the end of one PushT attempt
    # to the start of another. This function makes one ordered row list for each
    # complete attempt.
    #
    # Output example: {7: [840, 841, 842, ...]}. Here 7 is an episode ID and
    # 840 is a row ID in the full dataset. frame_index is different: it is the
    # time number inside that episode. Sorting by frame_index restores time order.
    # The check catches missing or repeated times before they can corrupt a
    # training window.
    groups: dict[int, list[tuple[int, int]]] = {}
    for row, (episode, frame) in enumerate(zip(source["episode_index"], source["frame_index"])):
        groups.setdefault(int(episode), []).append((int(frame), row))
    result = {}
    for episode, pairs in groups.items():
        pairs.sort()
        if [p[0] for p in pairs] != list(range(pairs[0][0], pairs[0][0] + len(pairs))):
            raise ValueError(f"Missing or duplicate frames in episode {episode}.")
        result[episode] = [p[1] for p in pairs]
    return result


def split_episodes(groups: dict[int, list[int]], config: Config) -> dict[str, list[int]]:
    # Where this fits: choose which complete attempts are used for learning and
    # which are used only for evaluation. The returned lists contain episode
    # IDs, not row IDs.
    #
    # Keep entire episodes together. If we split overlapping windows instead,
    # nearly identical neighbouring frames could appear in both sets and make
    # validation look better than it really is. A local Random object makes the
    # split repeatable without changing random choices elsewhere in the program.
    ids = sorted(groups)
    random.Random(config.seed).shuffle(ids)
    if len(ids) < config.train_episodes + config.val_episodes:
        raise ValueError("Requested more complete episodes than the dataset contains.")
    return {"train": ids[:config.train_episodes],
            "validation": ids[config.train_episodes:config.train_episodes + config.val_episodes]}


def build_window(rows: list[int], start: int, history: int, chunk: int) -> tuple[list[int], list[list[int]]]:
    # Purpose:
    # Select four observations and the three five-action chunks between them
    # from one episode. This fixes which actions caused each next observation.
    #
    # Example input and output:
    # With rows=list(range(20, 36)), start=0, history=3, and chunk=5, return:
    #   frame_ids = [20, 25, 30, 35]
    #   action_chunks = [[20,21,22,23,24],
    #                    [25,26,27,28,29],
    #                    [30,31,32,33,34]]
    # These are dataset row IDs. They are not image tensors or x/y actions.
    #   20..35 = the ordered row IDs available in this example episode
    #   start=0 = begin at position 0 in rows, whose row ID is 20
    #   history=3 = build three transitions and therefore four frame IDs
    #   chunk=5 = place five action-row IDs between consecutive frame IDs
    #
    # Steps:
    # 1. Reject history < 1, chunk < 1, start < 0, or a window whose final
    #    position `start + history * chunk` is outside rows.
    # 2. For each i in range(history + 1), select
    #    `rows[start + i * chunk]` for frame_ids.
    # 3. For each i in range(history), slice rows from
    #    `start + i * chunk` up to, but not including, the next frame position.
    # 4. Return `(frame_ids, action_chunks)`.

    # default values, history=3, action chunk=5, start=0

    frame_ids: list[int] = []
    action_chunks: list[list[int]] = []

    frame_index = 0
    for i in range(history):
        frame_index = start + i * chunk
        frame_ids.append(rows[frame_index])

        action_transitions = []
        for j in range(chunk):
            action_transitions.append(rows[frame_index + j])

        action_chunks.append(action_transitions)

    frame_ids.append(rows[frame_index + chunk])

    return frame_ids, action_chunks

    #raise NotImplementedError("Align frame samples with complete action chunks.")


def prepare_image(image: Image, config: Config) -> Tensor:
    # Where this fits: convert one stored camera image into the exact numeric
    # form expected by DINOv2.
    #
    # Input: a PIL image, normally [96,96,3] uint8 values from 0 to 255.
    # Output: [3,196,196] float tensor. pil_to_tensor puts colour channels first;
    # division by 255 changes pixels to 0..1; resize changes only image size;
    # normalize applies one fixed mean and standard deviation per RGB channel.
    # These RGB statistics are unrelated to the pusher's x/y statistics.
    image = TF.resize(TF.pil_to_tensor(image.convert("RGB")).float() / 255,
                      [config.image_size, config.image_size], antialias=True)
    return TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_encoder(config: Config, device: torch.device) -> nn.Module:
    # Where this fits: load the pretrained image description model used to make
    # fixed targets. It is separate from the trainable DINOWorldModel.
    #
    # eval() turns off training-only behaviour. requires_grad_(False) prevents
    # gradient storage and weight updates. to(device) moves the encoder once so
    # cache_features can process image batches on the GPU.
    from transformers import Dinov2Model
    return Dinov2Model.from_pretrained(config.encoder).eval().requires_grad_(False).to(device)


@torch.no_grad()
def encode_images(encoder: nn.Module, images: Tensor, config: Config) -> Tensor:
    # Where this fits: turn prepared images into the visual values that the world
    # model will predict. @torch.no_grad() avoids building a gradient graph for
    # the frozen encoder.
    #
    # Input: N images [N,3,196,196]. Dinov2Model returns an object; its
    # last_hidden_state is [N,197,384]. Slot 0 is one whole-image class token.
    # Slots 1..196 are the 14 x 14 spatial grid. [:,1:] removes only the class
    # token, leaving [N,196,384]. The shape check catches a wrong resize, encoder,
    # or patch-grid setting immediately.
    patches = encoder(pixel_values=images).last_hidden_state[:, 1:]
    if patches.shape[1:] != (config.grid**2, config.visual_dim):
        raise ValueError(f"Unexpected DINO feature shape: {patches.shape}.")
    return patches.float()


def cache_features(source: HFDataset, rows: list[int], config: Config, device: torch.device) -> dict:
    # Where this fits: DINO is frozen, so a recorded image always produces the
    # same features. Compute them once instead of rerunning DINO every epoch.
    #
    # Input rows may contain repeats or be out of order. sorted(set(rows)) makes
    # one stable cache entry per source row. Result keys:
    #   'identity': source/model/preprocessing settings used to create the cache
    #   'rows': source row IDs in cache order, length N
    #   'features': CPU float16 [N,P,E], converted to float32 when a window loads
    # The identity records every choice that changes the cached values. Its hash
    # gives the file a short unique name. A 196-pixel cache therefore cannot be
    # mistaken for an older 224-pixel cache.
    # Learned state/action embeddings are NOT cached: their weights will change.
    rows = sorted(set(rows))
    identity = {"dataset": config.dataset, "fingerprint": source._fingerprint,
                "encoder": config.encoder, "size": config.image_size, "grid": config.grid,
                "preprocess": "rgb-bilinear-antialias-imagenet-v1", "rows": rows}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(config.output) / f"features_{digest}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # map_location="cpu" avoids filling GPU memory just to inspect a cache.
        # weights_only=True limits loading to plain tensors and safe containers.
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Feature cache identity mismatch.")
        return saved
    encoder = load_encoder(config, device)
    features = []
    for begin in range(0, len(rows), config.cache_batch_size):
        chosen = rows[begin:begin + config.cache_batch_size]
        # Prepare on CPU, stack into [N,3,196,196], then move one batch to the
        # encoder's device. Store float16 on CPU to halve cache memory and disk
        # use. WindowDataset converts selected features to float32 for training.
        images = torch.stack([prepare_image(source[r]["observation.image"], config) for r in chosen])
        features.append(encode_images(encoder, images.to(device), config).half().cpu())
        print(f"DINO cache {min(begin + len(chosen), len(rows))}/{len(rows)}")
    saved = {"identity": identity, "rows": rows, "features": torch.cat(features)}
    torch.save(saved, path)
    # DINO is not needed after its features have been saved. Removing the last
    # reference and clearing unused CUDA blocks leaves room for the world model.
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return saved


class WindowDataset(torch.utils.data.Dataset):
    # Where this fits: build_window returns row IDs. This class turns those IDs
    # into the feature, state, and action tensors used by model.forward.
    # __getitem__ returns one window without a batch axis. DataLoader stacks
    # several windows and creates batch axis B.
    def __init__(self, source: HFDataset, groups: dict[int, list[int]], ids: list[int],
                 cache: dict, stats: dict[str, Tensor], config: Config) -> None:
        self.windows = []
        # Each valid start becomes a separate example. window_stride=5 moves to
        # the next sampled frame instead of making five nearly identical starts.
        for episode in ids:
            rows = groups[episode]
            for start in range(0, len(rows) - config.history * config.action_chunk, config.window_stride):
                self.windows.append(build_window(rows, start, config.history, config.action_chunk))
        if not self.windows:
            raise ValueError("No complete windows; use longer episodes or a shorter history.")
        self.lookup = {row: slot for slot, row in enumerate(cache["rows"])}
        # Cache order is sorted source row order, but it may contain only selected
        # episodes. lookup converts a global source row ID to its cache position.
        self.features = cache["features"]
        self.states = torch.as_tensor(source["observation.state"], dtype=torch.float32)
        self.actions = torch.as_tensor(source["action"], dtype=torch.float32)
        self.stats = stats

    def __len__(self) -> int:
        # DataLoader uses this to know how many windows can be sampled.
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        # Output shapes for the default H=3, S=5, P=196:
        # features [4,196,384], states [4,2], actions [3,5,2].
        # States and actions are normalized with TRAINING statistics. Features
        # are converted back to float32 after being stored as float16 in cache.
        frames, chunks = self.windows[index]
        # Cache slots differ from source row IDs, so visual features need lookup.
        # States/actions still index the full source columns with source IDs.
        # Turn nested chunk IDs into ONE integer tensor [H,S] before indexing;
        # a nested Python list can instead be read as separate axis indices.
        return {"features": self.features[[self.lookup[r] for r in frames]].float(),
                "states": (self.states[frames] - self.stats["state_mean"]) / self.stats["state_std"],
                "actions": (self.actions[torch.as_tensor(chunks, dtype=torch.long)] - self.stats["action_mean"]) / self.stats["action_std"]}


def frame_causal_mask(history: int, patches: int, device: torch.device) -> Tensor:
    # Purpose:
    # Build the boolean attention mask that stops a patch token from reading
    # patches in later frames. Patches may read every patch in their own frame.
    #
    # Example input and output:
    # With the defaults history=3 and patches=196, there are 3*196=588 tokens.
    # Return a bool tensor [588,588]. Rows are query tokens and columns are key
    # tokens. True means blocked. Every query from frame 0 blocks frame 1 and 2
    # keys; every query from frame 1 blocks frame 2 keys; frame 2 blocks none.
    # A small value example with H=2,P=2 is:
    #   [[F,F,T,T],
    #    [F,F,T,T],
    #    [F,F,F,F],
    #    [F,F,F,F]]
    # Here F means allowed and T means blocked.
    #
    # Steps:
    # 1. Make frame IDs with `torch.arange(history, device=device)`.
    # 2. Call `.repeat_interleave(patches)` so the default IDs contain 196
    #    zeros, then 196 ones, then 196 twos: shape [588].
    # 3. Make key frame IDs [1,588] with `.unsqueeze(0)` and query frame IDs
    #    [588,1] with `.unsqueeze(1)`. The 1 is a temporary single row or
    #    column that broadcasting expands during the comparison.
    # 4. Return the comparison that is True only when key_frame > query_frame.

    #defaults:  H=3 history, P=14x14=196 patches
    #final mask shape should be:
    # [H × P, H × P]
    # = [3 × 196, 3 × 196]
    # = [588, 588]

    frame_ids = torch.arange(history, device=device)        # shape: [H]
    frame_ids = frame_ids.repeat_interleave(patches)        # shape: [H*P]
    key_frames = frame_ids.unsqueeze(dim=0)                 # shape: [1, H*P]
    query_frames = frame_ids.unsqueeze(dim=1)               # shape: [H*P, 1]
    mask = key_frames > query_frames

    return mask
    #raise NotImplementedError("Build a frame-causal attention mask.")


class PredictorAttention(nn.Module):
    """Let every patch gather information from allowed patches and frames."""

    def __init__(self, width: int, heads: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        # Purpose:
        # Create the layers used by masked multi-head attention. A patch token
        # enters and leaves with 404 values.
        #
        # Example input and output:
        # With width=404, heads=16, head_dim=64, and dropout=0.1, the temporary
        # width is 16*64=1024. The QKV layer maps [32,588,404] to
        # [32,588,3072]. The output layer maps [32,588,1024] back to
        # [32,588,404].
        #   32   = training windows in the batch
        #   588  = 3 frames * 196 patch tokens per frame
        #   404  = 384 visual + 10 state + 10 action values per token
        #   1024 = 16 attention heads * 64 values per head
        #   3072 = 3 groups (query, key, value) * 1024 values
        #
        # Steps:
        # 1. Save `heads` and `head_dim`; forward needs both for reshaping.
        # 2. Save the attention scale `head_dim ** -0.5`. This is 1/sqrt(head_dim).
        # 3. Compute `inner_width = heads * head_dim`.
        # 4. Make `nn.LayerNorm(width)` for normalizing tokens before attention.
        # 5. Make `nn.Linear(width, 3 * inner_width, bias=False)` for Q, K, and V.
        # 6. Make an output path containing `nn.Linear(inner_width, width)` and
        #    `nn.Dropout(dropout)`. `nn.Sequential(...)` can hold both in order.
        # 7. Make another `nn.Dropout(dropout)` for attention probabilities.

        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_width = heads * head_dim  # 1024

        # N is the total number of patch tokens across all history frames

        self.norm = nn.LayerNorm(width)                                 # [B,N,width] -> [B,N,width]; default [B,588,404]
        self.to_qkv = nn.Linear(width, 3 * inner_width, bias=False)     # default [B,588,404] -> [B,588,3072]
        self.to_output = nn.Sequential(
            nn.Linear(inner_width, width),                  # default [B,588,1024] -> [B,588,404]
            nn.Dropout(dropout),                            # shape stays [B,588,404] by default
        )
        self.attention_dropout = nn.Dropout(dropout)  # default [B,16,588,588] -> same shape

        # raise NotImplementedError("Create the multi-head attention layers.")

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        # Purpose:
        # Let every patch token gather information from allowed patch tokens.
        # The mask prevents it from reading later frames.
        #
        # Example input and output:
        # tokens [32,588,404] contains 588 patch tokens per training window.
        # mask [588,588] is boolean; True means a connection is blocked.
        # Return [32,588,404] in the same token order.
        #   32  = training windows in the batch
        #   588 = 3 frames * 196 patches
        #   404 = 384 visual + 10 state + 10 action values per patch token
        # Each mask row is one query token; each column is one key token.
        #
        # Steps:
        # 1. Read batch_size=32 and num_tokens=588 from tokens.shape.
        # 2. Normalize tokens, then pass them through the QKV linear layer.
        #    Shape: [32,588,404] -> [32,588,3072].
        # 3. Reshape to [32,588,3,16,64]. Then use
        #    `.permute(2,0,3,1,4).unbind(0)` to get queries, keys, and values.
        #    Each resulting tensor has shape [32,16,588,64].
        # 4. Compute query-key scores with matrix multiplication. Transpose only
        #    the final two key axes with `keys.transpose(-2,-1)`, then multiply
        #    by the saved scale. Scores have shape [32,16,588,588].
        # 5. Expand the shared mask to [1,1,588,588] with `mask[None,None]`.
        #    Use `scores.masked_fill(...)` to replace blocked scores with
        #    `torch.finfo(scores.dtype).min`.
        # 6. Apply `torch.softmax(scores, dim=-1)`. The last axis lists the keys
        #    available to one query, so its probabilities must sum to 1 there.
        #    Pass those probabilities through the attention dropout layer.
        # 7. Multiply the probabilities by values: [32,16,588,64].
        # 8. Move N before heads with `.transpose(1,2)`, then reshape the heads
        #    beside one another to [32,588,1024].
        # 9. Pass that tensor through the output path and return [32,588,404].

        batch_size, num_tokens, width = tokens.shape        # [32, 588, 404]
        #default num_heads = 16, head_dim = 64

        x = self.norm(tokens)   # [32, 588, 404]
        x = self.to_qkv(x)      # [32, 588, 3*heads*head_dim] = [B,588,3072]
        x = x.reshape(batch_size, num_tokens, 3, self.heads, self.head_dim)     # [32, 588, 3, 16, 64]
        x = x.permute(2, 0, 3, 1, 4)        # [3, 32, 16, 588, 64]

        q = x[0]    # [B,heads,N,head_dim] = [32, 16, 588, 64]
        k = x[1]    # [32, 16, 588, 64]
        v = x[2]    # [32, 16, 588, 64]

        scores = q @ k.transpose(-2, -1)    # [32, 16, 588, 64] @ [32, 16, 64, 588]
        scores = scores * self.scale        # [32, 16, 588, 588]

        mask = mask.unsqueeze(dim=0).unsqueeze(dim=0)                           # [588, 588] -> [1, 1, 588, 588]
        masked_scores = scores.masked_fill(mask=mask, value=float("-inf"))      # [32, 16, 588, 588]

        att_weights = torch.softmax(masked_scores, dim=-1)      # [32, 16, 588, 588]
        att_weights = self.attention_dropout(att_weights)       # [32, 16, 588, 588]

        output = att_weights @ v    # [32, 16, 588, 588] @ [32, 16, 588, 64] = [32, 16, 588, 64]

        output = output.transpose(1, 2)                                             # [32, 588, 16, 64]
        output = output.reshape(batch_size, num_tokens, self.heads*self.head_dim)   # [32, 588, 1024]

        output = self.to_output(output)     # [32, 588, 404]

        return output
        #raise NotImplementedError("Run masked multi-head attention.")


class PredictorBlock(nn.Module):
    """Mix information between tokens, then update each token separately."""

    def __init__(self, width: int, heads: int, head_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        # Purpose:
        # Create one Transformer block with an attention update followed by a
        # feed-forward update.
        #
        # Example input and output:
        # With width=404, heads=16, head_dim=64, mlp_dim=2048, and dropout=0.1,
        # both paths receive and return [32,588,404]. The feed-forward path
        # temporarily expands each token to [32,588,2048].
        #   32   = training windows in the batch
        #   588  = 3 frames * 196 patches
        #   404  = 384 visual + 10 state + 10 action values per token
        #   2048 = hidden values used inside the feed-forward path
        #
        # Steps:
        # 1. Create PredictorAttention(width, heads, head_dim, dropout).
        # 2. Create an nn.Sequential feed-forward path in this exact order:
        #    LayerNorm(width), Linear(width,mlp_dim), GELU(), Dropout(dropout),
        #    Linear(mlp_dim,width), Dropout(dropout).
        # The feed-forward path changes each token separately. It does not mix
        # patches or frames; the attention layer already does that.

        self.attention = PredictorAttention(width, heads, head_dim, dropout)  # [B,N,404] -> [B,N,404]
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),            # [B,N,404] -> [B,N,404]
            nn.Linear(width, mlp_dim),      # [B,N,404] -> [B,N,2048]
            nn.GELU(),                      # shape stays [B,N,2048]
            nn.Dropout(dropout),            # shape stays [B,N,2048]
            nn.Linear(mlp_dim, width),      # [B,N,2048] -> [B,N,404]
            nn.Dropout(dropout),            # shape stays [B,N,404]
        )

        # raise NotImplementedError("Create one predictor Transformer block.")

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        # Purpose:
        # Apply the attention and feed-forward updates while preserving the
        # original token through two residual additions.
        #
        # Example input and output:
        # tokens [32,588,404] and mask [588,588] return [32,588,404].
        #   32  = training windows in the batch
        #   588 = 3 frames * 196 patches; it is also each mask axis
        #   404 = 384 visual + 10 state + 10 action values per token
        #
        # Steps:
        # 1. Add the attention result to the original tokens.
        # 2. Run that updated tensor through feed_forward and add its result.
        # These additions are residual connections: each path learns a change
        # to the token instead of having to rebuild the whole token.

        residual = tokens
        x = self.attention(tokens, mask)
        x = x + residual

        residual = x
        x = self.feed_forward(x)
        x = x + residual

        return x

        #raise NotImplementedError("Apply attention and feed-forward residual updates.")


class PredictorTransformer(nn.Module):
    """Run the same frame mask through all six predictor blocks."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Purpose:
        # Build the complete Transformer from six separate PredictorBlock
        # objects and one final normalization layer.
        #
        # Example input and output:
        # With config.depth=6 and config.width=404, the stack receives and
        # returns [32,588,404]. Each block owns different trainable weights.
        #   6   = Transformer blocks
        #   32  = training windows in the batch
        #   588 = 3 frames * 196 patches
        #   404 = 384 visual + 10 state + 10 action values per token
        #
        # Steps:
        # 1. Build six PredictorBlock objects with width=404, heads=16,
        #    head_dim=64, mlp_dim=2048, and dropout=0.1.
        # 2. Store them in `nn.ModuleList` so PyTorch registers their weights.
        # 3. Create the final `nn.LayerNorm(config.width)`.

        self.layers = nn.ModuleList([
            PredictorBlock(
                config.width,
                config.heads,
                config.head_dim,
                config.mlp_dim,
                dropout=0.1,
            )
            for _ in range(config.depth)
        ])  # config.depth blocks; each keeps [B,N,404] unchanged

        self.norm = nn.LayerNorm(config.width)  # [B,N,404] -> [B,N,404]

        #raise NotImplementedError("Create the predictor block stack.")

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        # Purpose:
        # Send the tokens through all six blocks using the same frame mask, then
        # normalize the final tokens.
        #
        # Example input and output:
        # tokens [32,588,404] and mask [588,588] return [32,588,404].
        #   32  = training windows in the batch
        #   588 = 3 frames * 196 patches; it is also each mask axis
        #   404 = 384 visual + 10 state + 10 action values per token
        #
        # Steps:
        # 1. Loop over self.layers in order.
        # 2. Replace tokens with layer(tokens, mask) each time. Passing the same
        #    mask to every block prevents every layer from reading future frames.
        # 3. Apply the final normalization and return its result.

        features = tokens
        for layer in self.layers:
            features = layer(features, mask)    # [B,N,404] -> [B,N,404]

        return self.norm(features)              # [B,N,404] -> [B,N,404]

        # raise NotImplementedError("Run all predictor blocks.")


class DINOWorldModel(nn.Module):
    # WHAT THIS MODEL LEARNS
    #
    # The model sees three selected camera observations and the robot controls
    # that follow each one. The numbers below are positions INSIDE one training
    # window. They are not dataset row numbers.
    #
    #   observation 0 + action chunk 0 -> predict observation 1
    #   observation 1 + action chunk 1 -> predict observation 2
    #   observation 2 + action chunk 2 -> predict observation 3
    #
    # With action_chunk=5 and a window starting at dataset row 20, this means:
    #
    #   frame at row 20 + actions from rows 20..24 -> predict frame at row 25
    #   frame at row 25 + actions from rows 25..29 -> predict frame at row 30
    #   frame at row 30 + actions from rows 30..34 -> predict frame at row 35
    #
    # So "observation 0" means the first selected frame in this window. It
    # means recorded frame 0 only when the current window really starts there.
    #
    # It does not predict pixels. DINO has already changed each image into 196
    # patch descriptions, with 384 values per patch. In one forward pass, this
    # model predicts all 196 patch descriptions for each next observation.
    # Each predicted patch contains
    # 384 next-image feature values plus the same 10-value description of the
    # predicted robot state, giving 394 predicted values per patch. Attention
    # lets each patch prediction use every allowed patch from the current and
    # earlier frames.
    #
    # DEFAULT DATA FLOW
    #
    # camera features   [B,4,196,384]  four real frames
    # robot states      [B,4,2]        one x/y state per frame i.e. 4 = four selected frames in each window, 2 = [x, y] position of the pusher in each frame
    # action chunks     [B,3,5,2]      five x/y controls after each input frame
    #
    # observation_tokens adds state information to every patch:
    #   384 visual + 10 state = 394 values
    #   [B,4,196,384] -> [B,4,196,394]
    #
    # predict uses the first three observations and adds action information:
    #   394 observation + 10 action = 404 values
    #   [B,3,196,394] -> [B,3,196,404] -> [B,588,404]           # 3x196=588
    #   Transformer -> [B,588,404] -> predictions [B,3,196,394]
    #
    # The targets are real observations 1, 2, and 3: [B,3,196,394].
    # DINO stays fixed. The state encoder, action encoder, position values, and
    # Transformer learn from the difference between predictions and targets.

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        # state_encoder changes one normalized x/y state into 10 learned values.
        # It handles each frame separately. Linear changes only the final axis,
        # so the batch and time axes stay in place.
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 64),                       # default [32,4,2] -> [32,4,64]
            nn.ReLU(),                              # shape stays [32,4,64]
            nn.Linear(64, config.condition_dim),    # [32,4,64] -> [32,4,10]
        )
        # action_encoder changes one complete action chunk into 10 learned
        # values. Five x/y controls contain 5*2=10 input numbers. The 10 output
        # values are learned features; they are not copies of those inputs.
        # config.action_chunk = 5
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_chunk * 2, 64),     # [32,3,10] -> [32,3,64]
            nn.ReLU(),                                  # shape stays [32,3,64]
            nn.Linear(64, config.condition_dim),        # [32,3,64] -> [32,3,10]
        )
        # Attention does not know where a token came from. This trainable table
        # gives each frame/patch position its own description. Its leading 1
        # means the same position table is shared across every batch example.
        self.position = nn.Parameter(torch.zeros(1, config.history * config.grid**2, config.width))  # default [1,3*196,404] = [1,588,404]
        nn.init.trunc_normal_(self.position, std=0.02)
        # Six Transformer blocks mix information between allowed patch tokens.
        # The frame mask stops earlier frames from reading later frames. Each
        # token enters and leaves the stack with 404 values.
        self.blocks = PredictorTransformer(config)  # [32,588,404] -> [32,588,404]

    def observation_tokens(self, features: Tensor, states: Tensor) -> Tensor:
        # PURPOSE
        # Build a full description of what was observed at each image patch.
        # An observation means visual information plus robot state. The action
        # is deliberately separate because it describes what happens next.
        #
        # INPUT AND OUTPUT
        # features [B,T,P,384]: one DINO description for each image patch.
        # states   [B,T,2]:     one normalized x/y state for each whole frame.
        # output   [B,T,P,394]: 384 visual values + 10 state values per patch.
        # During training, T=4 and P=196, so output is [B,4,196,394].
        #
        # WHY THE STATE IS EXPANDED
        # A frame has 196 visual patches but only one robot state. Every patch
        # token needs that same state. `unsqueeze(2)` makes a one-item patch axis.
        # `expand` shares the state across all 196 patches:
        #
        #   one state description [10] -> [196,10]
        #
        # The visual values still differ between patches. Only the state part
        # is shared. No action information is added in this method.

        # sttes = [32,4,2]
        # B = number of training windows, normally 32
        # 4 = four selected frames in each window
        # 2 = [x, y] position of the pusher in each frame

        state = self.state_encoder(states)                      # default [32,4,2] -> [32,4,10]
        state = state.unsqueeze(2)                              # [32,4,10] -> [32,4,1,10]
        state = state.expand(-1, -1, features.shape[2], -1)     # -> [32,4,196,10]
        return torch.cat([features, state], dim=-1)             # [32,4,196,384] cat [32,4,196,10] -> [32,4,196,394]

    def join_conditioning(self, observations: Tensor, actions: Tensor) -> Tensor:
        # Purpose:
        # Tell the model what it sees now and what the robot does next. The
        # action attached to observation i must be the action chunk immediately
        # after observation i. This pairing teaches cause and effect.
        #
        # Example input and output:
        # observations [32,3,196,394] contains three selected frames. Each patch
        # has 384 DINO values and 10 learned pusher-state values.
        # actions [32,3,5,2] contains one five-control x/y chunk after each frame.
        # Return [32,3,196,404]. One returned patch token contains:
        #   32  = training windows in the batch
        #   3   = selected input frames and their following action chunks
        #   196 = image patches in each frame
        #   5   = recorded controls in one action chunk
        #   2   = x/y target values in one control
        #   394 = 384 visual + 10 state values
        #   404 = 394 observation + 10 action values
        # Those final 404 values are ordered as:
        #   384 values describing one image patch
        #    10 values describing the robot state in that frame
        #    10 values describing the controls that follow that frame
        #
        # Steps:
        # 1. Flatten actions from [32,3,5,2] to [32,3,10]. Do not flatten H:
        #    five ordered controls belong to one transition, not to five frames.
        # 2. Call self.action_encoder(flat_actions) -> [32,3,10]. The input
        #    length and output width happen to both be 10 by default; their
        #    meanings differ. The output is learned, not a copy of the controls.
        # 3. Call action_embedding.unsqueeze(2) -> [32,3,1,10].
        # 4. Expand with .expand(-1, -1, observations.shape[2], -1).
        #    A -1 keeps that axis unchanged; P replaces the singleton patch axis.
        #    The result is [32,3,196,10]. Do not modify this expanded view in place.
        # 5. Concatenate it AFTER observations along dim=-1. Return [32,3,196,404].
        #    Do not detach: the prediction loss must train action_encoder.

        # observations [B,H,P,394] = [32, 3, 196, 394]
        # actions [B,H,S,2] = [32,3,5,2]
        batch_size, history, chunk_size, xy_num = actions.shape
        flattened_actions = actions.reshape(batch_size, history, chunk_size * xy_num)  # [32,3,5,2] -> [32, 3, 10]

        action_embedding = self.action_encoder(flattened_actions)    # [32,3,10] -> [32,3,10]

        action_embedding = action_embedding.unsqueeze(dim=2)         # [32,3,10] -> [32,3,1,10]

        # In expand, -1 means: Keep this dimension at its current size.
        action_embedding =action_embedding.expand(-1, -1, observations.shape[2], -1)    # [32,3,1,10] -> [32,3,196,10]

        joint_embeddings = torch.cat([observations, action_embedding], dim=-1)      # [32, 3, 196, 394] cat [32,3,196,10] = [32, 3, 196, 404]

        return joint_embeddings
        #raise NotImplementedError("Concatenate observation and action embeddings.")

    def predict(self, observations: Tensor, actions: Tensor) -> Tensor:
        # Purpose:
        # Use three current observations and their following action chunks to
        # predict the next three observations. All 196 patches are predicted
        # together. During training the three inputs are real observations.
        #
        # Example input and output:
        # Using the config defaults:
        # observations has shape [32,3,196,394].
        #   32  = training windows in this batch
        #   3   = selected input frames, such as recorded rows 20, 25, and 30
        #   196 = image patches in each frame
        #   394 = 384 DINO features + 10 learned pusher-state features per patch
        # actions has shape [32,3,5,2].
        #   3 = one action chunk after each selected input frame
        #   5 = five recorded controls in each chunk
        #   2 = the normalized x/y target in each control
        # The returned predictions have shape [32,3,196,394]:
        #   input row 20 + actions 20..24 predicts the observation at row 25
        #   input row 25 + actions 25..29 predicts the observation at row 30
        #   input row 30 + actions 30..34 predicts the observation at row 35
        #
        # Steps:
        # 1. Read batch size, history length, and patch count from observations.
        # 2. Call self.join_conditioning(observations, actions). This adds the
        #    action features: [32,3,196,394] -> [32,3,196,404].
        # 3. Flatten only the frame and patch axes with
        #    `.flatten(start_dim=1, end_dim=2)`: [32,3,196,404] -> [32,588,404].
        # 4. Add self.position, shape [1,588,404]. It is shared across the 32
        #    examples and tells the Transformer each token's frame/patch slot.
        # 5. Create the boolean mask with
        #    `frame_causal_mask(history, patches, tokens.device)`. Its default
        #    shape is [588,588], and it blocks tokens from later frames.
        # 6. Call self.blocks(tokens, mask), which returns [32,588,404].
        # 7. Reshape to [32,3,196,404]. Keep the first 394 values of every patch
        #    with `[..., :self.config.observation_dim]` and return
        #    [32,3,196,394]. The final 10 Transformer values are not targets.

        batch_size, history, patches, _ = observations.shape

        tokens = self.join_conditioning(observations, actions)  # [32,3,196,394] and [32,3,5,2] -> [32,3,196,404]
        tokens = tokens.flatten(start_dim=1, end_dim=2)  # [32,3,196,404] -> [32,588,404]
        tokens = tokens + self.position  # [32,588,404] + [1,588,404] -> [32,588,404]

        mask = frame_causal_mask(history, patches, tokens.device)  # [588,588]
        predicted = self.blocks(tokens, mask)  # [32,588,404]
        predicted = predicted.reshape(batch_size, history, patches, self.config.width)  # [32,588,404] -> [32,3,196,404]

        return predicted[..., :self.config.observation_dim]  # [32,3,196,394]

        # raise NotImplementedError("Predict shifted observation embeddings.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        # PURPOSE
        # Build the predictions and matching real targets used by the loss.
        # train_batch calls model(batch), and PyTorch sends that call here.
        #
        # With four real observations [A,B,C,D]:
        # - predict receives [A,B,C] and their three following action chunks
        # - the targets are [B,C,D]
        # - the matched pairs are A->B, B->C, and C->D
        #
        # Observation D is used only as the last target. It is not an input to
        # the prediction that tries to produce D.
        observations = self.observation_tokens(
            batch["features"], batch["states"]
        )  # [32,4,196,384] and [32,4,2] -> [32,4,196,394]
        # predict sees only the first H observations. Its frame mask prevents
        # an earlier output from reading a later input that is also its target.
        # The returned target stays attached here; prediction_loss detaches it.
        # For [A,B,C,D], the input slice is [A,B,C] and the target slice is
        # [B,C,D]. predict returns one guess for each matching target.
        return (
            self.predict(observations[:, :-1], batch["actions"]),  # -> [32,3,196,394]
            observations[:, 1:],  # target: [32,3,196,394]
        )


def prediction_loss(predicted: Tensor, target: Tensor) -> Tensor:
    # Purpose:
    # Measure how far the predicted next observations are from the real ones.
    # Stop gradients through the targets so training changes the prediction
    # path instead of moving the targets to make the loss smaller.
    #
    # Example input and output:
    # predicted and target both have shape [32,3,196,394]. Return one scalar
    # tensor with shape []. For the small value example predicted=[1,3] and
    # target=[2,1], mean squared error is ((1-2)^2 + (3-1)^2) / 2 = 2.5.
    # The real loss averages across all 32*3*196*394 values.
    #   32  = training windows in the batch
    #   3   = predicted next observations per window
    #   196 = image patches in each predicted observation
    #   394 = 384 visual + 10 state values per predicted patch
    # Shape [] means one number rather than a tensor with batch or time axes.
    #
    # Steps:
    # 1. Compare predicted.shape and target.shape; raise ValueError if unequal.
    #    MSE can broadcast shapes, so an unchecked mismatch might hide a bug.
    # 2. Call `target.detach()` to keep its values but cut its gradient path.
    # 3. Return `F.mse_loss(predicted, detached_target, reduction="mean")`.
    #    Keep the result as a tensor so train_batch can call backward on it.
    raise NotImplementedError("Compute the detached observation prediction loss.")


def train_batch(model: DINOWorldModel, batch: dict[str, Tensor], optimizer: torch.optim.Optimizer,
                scaler: torch.amp.GradScaler, device: torch.device) -> float:
    # Purpose:
    # Perform one training update: make predictions, calculate loss, calculate
    # gradients, and update the world model. DINO is not run or updated here.
    #
    # Example input and output:
    # The default CPU batch contains:
    #   features [32,4,196,384]
    #   states   [32,4,2]
    #   actions  [32,3,5,2]
    #   32  = training windows in the batch
    #   4   = real observations: three inputs plus one final target
    #   196 = image patches in each observation
    #   384 = frozen DINO values describing one patch
    #   2   = pusher x/y position in states and x/y target in actions
    #   3   = action chunks, one after each input observation
    #   5   = recorded controls in each action chunk
    # The model is already on device. The function returns one Python float such
    # as 0.42 for logging and changes the model weights once.
    #
    # Steps:
    # 1. Call model.train() to enable training behavior such as dropout. This
    #    does not itself compute a loss or change any weights.
    # 2. Build a device batch using {key: value.to(device) for key, value in
    #    batch.items()}. Every field must move, including states/actions.
    # 3. Call optimizer.zero_grad(set_to_none=True) to clear previous gradients.
    # 4. Enter `torch.autocast(device_type=device.type, dtype=torch.float16,
    #    enabled=device.type == "cuda")`. Run
    #    `predicted, target = model(device_batch)` inside it.
    # 5. Still inside autocast, call prediction_loss(predicted, target).
    #    It returns a scalar Tensor. Do not call .item() before backward.
    # 6. Leave autocast, then call `scaler.scale(loss).backward()`.
    # 7. Call scaler.unscale_(optimizer) to restore the gradients' real size.
    # 8. Call torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).
    #    It reduces the combined gradient size if it exceeds 1; it does not
    #    clamp each weight or compute gradients. A total norm of 2 is scaled
    #    by about 1/2, whereas a norm of 0.5 is left alone.
    # 9. Call scaler.step(optimizer), then scaler.update(). The scaler can skip
    #    an unsafe update if gradients overflow and adjust its scale for later.
    # 10. Return `loss.item()` after the tensor has been used for backward.
    raise NotImplementedError("Train one DINO-WM batch.")


@torch.no_grad()
def imagine(model: DINOWorldModel, observations: Tensor, past_actions: Tensor, future_actions: Tensor) -> Tensor:
    # Purpose:
    # Predict several future observations without stepping the real environment.
    # Feed each new prediction back into the three-observation history before
    # predicting the next step. No weights are updated.
    #
    # Example input and output:
    # With 32 histories and two proposed future action chunks:
    #   observations   [32,3,196,394] = known observations [A,B,C]
    #   past_actions   [32,2,5,2]     = known chunks [aAB,aBC]
    #   future_actions [32,2,5,2]     = proposed chunks [aCD,aDE]
    # Return [32,2,196,394] containing predictions [D,E]. Do not return the
    # starting observations [A,B,C]. The two past chunks exist because three
    # known observations have only two transitions between them.
    #   axis 0 of every tensor: 32 separate histories rolled forward together
    #   observations axis 1: 3 observations kept in the moving history
    #   past_actions axis 1: 2 known transitions connecting those 3 observations
    #   future_actions/output axis 1: 2 proposed future model steps
    #   action axis 2: 5 recorded controls in each action chunk
    #   action axis 3: 2 normalized values, the x/y target of one control
    #   observation axis 2: 196 image patches
    #   observation axis 3: 394 values = 384 visual + 10 learned state values
    #
    # Steps:
    # 1. Call `model.eval()`. Read future_steps=2 from future_actions.shape[1]
    #    and reject values below 1. Start an empty output list. Use local names
    #    for the current observation history and known past actions.
    # 2. For each future step, select `future_actions[:, step:step+1]` so the
    #    proposed chunk keeps shape [32,1,5,2].
    # 3. Join past and proposed chunks along time to make [32,3,5,2].
    # 4. Call model.predict(history, joined_actions), then keep only its last
    #    time slot with `[:, -1:]`, giving [32,1,196,394]. Earlier output slots
    #    re-predict earlier transitions; they are not additional future steps.
    # 5. Append `next_observation[:, 0]`, shape [32,196,394], to outputs.
    # 6. Slide the observation window with `torch.cat([history[:, 1:],
    #    next_observation], dim=1)`. It remains [32,3,196,394].
    # 7. Slide controls with `known=joined_actions[:, 1:]` -> [32,2,5,2].
    #    The just-proposed control is now a past control for the shifted window.
    # 8. Return `torch.stack(outputs, dim=1)`, shape [32,2,196,394].
    raise NotImplementedError("Roll the world model forward through action chunks.")


@torch.no_grad()
def evaluate(model: DINOWorldModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    # Where this fits: run calls this after each training epoch to measure the
    # model on complete episodes that were not used for weight updates.
    # This function only measures: there is no optimizer or backward call.
    #
    # It compares three answers for the same held-out histories:
    # - learned prediction: what the model predicts with the recorded actions
    # - persistence: guess that the visual features do not change
    # - shuffled actions: give the model deliberately mismatched controls
    # Scores use only the first E=384 visual channels, because frozen DINO space
    # is common and fixed. The learned state embedding can move during training.
    # Each prediction starts from recorded history; this is not a long rollout.
    model.eval()
    totals = {key: 0.0 for key in ("mse", "persistence_mse", "shuffled_action_mse", "cosine")}
    count = 0
    for cpu_batch in loader:
        batch = {k: v.to(device) for k, v in cpu_batch.items()}
        predicted, target = model(batch)
        E = model.config.visual_dim
        p, t = predicted[..., :E], target[..., :E]
        # Flatten B and H, move every action chunk one slot, then restore [B,H,S,2].
        # Rolling both axes together means even a final batch with B=1 usually
        # receives different controls as long as H>1.
        wrong = batch["actions"].flatten(0, 1).roll(1, 0).reshape_as(batch["actions"])
        other, _ = model({**batch, "actions": wrong})
        # p and t are [B,H,P,384]. Every metric below becomes one scalar for
        # this batch. Cosine compares each matching patch along its 384 features.
        values = {"mse": F.mse_loss(p, t), "persistence_mse": F.mse_loss(batch["features"][:, :-1], t),
                  "shuffled_action_mse": F.mse_loss(other[..., :E], t),
                  "cosine": F.cosine_similarity(p, t, dim=-1).mean()}
        # Weight by the number of windows so a smaller final batch does not have
        # the same influence as a full batch.
        n = p.shape[0]
        for key, value in values.items():
            totals[key] += value.item() * n
        count += n
    if not count:
        raise ValueError("Empty evaluation loader.")
    return {key: value / count for key, value in totals.items()}


def prepare_data(config: Config, device: torch.device):
    # Where this fits: perform the complete data setup before the model and
    # optimizer are created.
    #
    # Order: source -> ordered episodes -> episode split -> training-only x/y
    # statistics -> frozen DINO cache -> WindowDataset objects. Means and standard
    # deviations use training rows only. Using validation rows would let held-out
    # data influence preprocessing before evaluation.
    source = load_source(config)
    groups = group_episodes(source)
    splits = split_episodes(groups, config)
    # Reach the first student task BEFORE the expensive DINO download/cache.
    print("First aligned window:", build_window(groups[splits["train"][0]], 0, config.history, config.action_chunk))
    # Flatten the selected training episodes into source row IDs. State and
    # action each have two columns, so every saved mean/std tensor has shape [2].
    train_rows = [r for e in splits["train"] for r in groups[e]]
    stats = {}
    for prefix, field in (("state", "observation.state"), ("action", "action")):
        values = torch.as_tensor(source[field], dtype=torch.float32)[train_rows]
        stats[prefix + "_mean"] = values.mean(0)
        stats[prefix + "_std"] = values.std(0, unbiased=False).clamp_min(1e-6)
    # Each mean/std is [2], one value per coordinate. clamp_min avoids division
    # by zero if a coordinate happens to be constant in the selected recordings.
    # Cache both splits because training and validation both need DINO features.
    # Exercise 11 can reuse all rows from these held-out episodes for longer
    # histories and horizons, so cache complete episodes rather than only the
    # windows used by this exercise's current settings.
    needed = []
    for ids in splits.values():
        for episode in ids:
            rows = groups[episode]
            # Include every row so 11 can change history length or horizon later.
            needed.extend(rows)
    cache = cache_features(source, needed, config, device)
    datasets = {name: WindowDataset(source, groups, ids, cache, stats, config) for name, ids in splits.items()}
    return source, groups, splits, stats, cache, datasets


def load_checkpoint(path: Path, device: torch.device) -> tuple[DINOWorldModel, dict]:
    # Where this fits: exercises 10 and 11 need exactly the model and data rules
    # that produced a saved checkpoint.
    #
    # The saved config recreates the architecture before strict=True checks that
    # every weight name and shape matches. The returned saved dictionary also
    # contains training normalization and split IDs. Planning must reuse those
    # values instead of fitting new statistics to its current episode.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "dino-wm-absolute-chunks-v3":
        raise ValueError("Use a current exercise 09 checkpoint; older weights use a different predictor or patch grid.")
    model = DINOWorldModel(Config(**saved["config"]))
    model.load_state_dict(saved["model"], strict=True)
    # Planning and benchmarking only measure/use this model. eval() turns off
    # dropout and requires_grad_(False) prevents accidental further training.
    return model.to(device).eval().requires_grad_(False), saved


def run(config: Config) -> None:
    # Where this fits: this is the full training program that connects the
    # provided setup with the student's train_batch implementation.
    # Keeping one model, optimizer and scaler for the whole run preserves their
    # learned values and optimizer history across batches and epochs.
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source, _, splits, stats, cache, datasets = prepare_data(config, device)
    # Shuffle training windows so batches change each epoch. Validation order
    # does not affect its averages, so leave it stable and easy to reproduce.
    loaders = {name: DataLoader(data, batch_size=config.batch_size, shuffle=name == "train")
               for name, data in datasets.items()}
    model = DINOWorldModel(config).to(device)
    # The paper uses a smaller learning rate for the predictor than for the two
    # small coordinate encoders. Parameter groups let one AdamW optimizer keep
    # those rates while train_batch still performs one normal optimizer step.
    # State/action encoders share the larger action-encoder learning rate. All
    # remaining parameters belong to the predictor group. IDs are used here
    # because a Tensor cannot safely be compared with `in` like a normal value.
    condition_parameters = list(model.state_encoder.parameters()) + list(model.action_encoder.parameters())
    condition_parameter_ids = {id(parameter) for parameter in condition_parameters}
    predictor_parameters = [parameter for parameter in model.parameters()
                            if id(parameter) not in condition_parameter_ids]
    optimizer = torch.optim.AdamW([
        {"params": predictor_parameters, "lr": config.predictor_lr},
        {"params": condition_parameters, "lr": config.action_encoder_lr},
    ], weight_decay=0.01)
    # GradScaler is active only on CUDA. train_batch can therefore use the same
    # calls on GPU and CPU: on CPU the scaler simply forwards them unchanged.
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    # One epoch visits every training window once. Evaluation never changes
    # weights. Saving after every epoch makes the latest complete epoch reusable
    # by planning even if a later run is interrupted.
    for epoch in range(config.epochs):
        losses = [train_batch(model, batch, optimizer, scaler, device) for batch in loaders["train"]]
        metrics = evaluate(model, loaders["validation"], device)
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), **metrics})
        print(history[-1])
        torch.save({"format": "dino-wm-absolute-chunks-v3", "model": model.state_dict(),
                    "config": asdict(config), "stats": stats, "splits": splits,
                    "data_fingerprint": source._fingerprint, "feature_identity": cache["identity"],
                    "epoch": epoch + 1}, output / "predictor.pt")
    # Keep raw numbers for later comparison and draw the three lower-is-better
    # visual MSE curves. Cosine remains in metrics.json but uses a different
    # scale and direction, so putting it on the same y-axis would be misleading.
    (output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    fig, ax = plt.subplots()
    for key in ("mse", "persistence_mse", "shuffled_action_mse"):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
    ax.set(xlabel="epoch", ylabel="held-out visual MSE", title="Does the predictor beat no change?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "prediction_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    # Command-line values override only the two settings people most often lower
    # for a quick run. Every other value comes from Config's documented default.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    defaults = Config()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    args = parser.parse_args()
    try:
        run(Config(epochs=args.epochs, batch_size=args.batch_size))
    except NotImplementedError as error:
        # A new student reaches one TODO at a time and sees its message without
        # a long traceback hiding the exercise instructions above that stub.
        print(f"[TODO] {error}")


if __name__ == "__main__":
    main()
