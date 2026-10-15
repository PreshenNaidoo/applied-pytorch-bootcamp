"""
EXERCISE 14: VIDEO WORLD-MODEL ROBUSTNESS ON SOMETHING-SOMETHING V2

WHAT YOU ARE BUILDING
Compare frozen pretrained V-JEPA 2-L and VideoMAEv2-L on real action videos.
Train a small attention classifier (a probe) to read each encoder's tokens.
Then test whether its decisions survive damaged inputs and reversed motion.
Reference: https://arxiv.org/abs/2605.15618
Sources: https://github.com/facebookresearch/vjepa2
         https://huggingface.co/OpenGVLab/VideoMAEv2-Large

WHY THIS EXERCISE EXISTS / BOOTCAMP CONNECTION
09-11 tested prediction and planning; 12 trained video features; 13 predicted
robot features given motion. Here we test what video features preserve.
An encoder returning one constant vector is perfectly stable but useless.
Measure recognition as well as feature similarity. This is a representation
study, not a learned simulator or a robot-success benchmark.

FULL WORKFLOW / SCOPE
Pretrain encoders (provided) -> freeze -> extract patch tokens -> train probes
on clean training clips -> select learning rate/epoch on clean validation ->
freeze probes -> evaluate the same clean/damaged test videos for both models.
Keep real ViT-L backbones and attention over all space/time tokens. Reduce to
eight classes, 32 training clips/class, 16 frames, one probe query and batch 2.
The paper covers more models, classes, corruptions and probe settings; this
is a smaller study, not its full protocol. V-JEPA uses 256-pixel images and
VideoMAE its published 224-pixel input. Pretraining data also differ, so these
results cannot isolate the effect of the pretraining objective alone.
Run one encoder at a time on an 8 GB GPU. Cache clean train/validation tokens
(about 1.3 GB per encoder). Stream test interventions; never cache them all.

SETUP / REAL DATA
Use exercise 13's pinned reference checkout and optional dependency:
  git clone https://github.com/facebookresearch/vjepa2.git data/reference_vjepa2
  git -C data/reference_vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
  uv pip install --python .venv/Scripts/python.exe timm==1.0.22 einops==0.8.1 easydict==1.13
Obtain SSv2 through its dataset access process and extract the actual videos:
  https://www.qualcomm.com/developer/software/something-something-v-2-dataset
Expected layout (or set --data-root):
  data/ssv2/videos/12345.webm
  data/ssv2/labels/something-something-v2-train.json
  data/ssv2/labels/something-something-v2-validation.json
Split official training clips into probe train/validation by video ID. Use
official validation as final test because public test labels are unavailable.
These are clip-disjoint splits, not guaranteed actor-disjoint splits. Missing
data raises an error; no fake clips or random encoder fallback are provided.
Weights download on the first run. VideoMAE's published loader runs remote
Python code, pinned below to an inspected revision.

MENTAL PICTURE / SHAPES
A clip shows a hand opening a box. Sixteen ordered RGB frames become 2048
V-JEPA tokens or 1568 VideoMAE tokens, each containing 1024 numbers. The probe
query learns which tokens help recognize the action; its head returns eight
scores. Only probe weights change; the labelled action supplies feedback.
Pixels [T,3,H,W] in [0,1] -> encoder input [1,3,T,H,W] -> tokens [1,N,1024]
-> probe logits [1,8]. Cross entropy compares logits [B,8] with labels [B].
Reversed opening is scored against closing. Pretend-opening has no automatic
reverse label. Inspect examples: reverse playback need not be natural motion.

METRICS TO TRACK
Accuracy = correct / total; higher is better; chance here is 1/8 = 12.5%.
Conditional retention = clean-correct AND damaged-correct / clean-correct.
Example: two clips correct when clean, one still correct -> retention 0.5.
This is not damaged accuracy divided by clean accuracy. Report null if its
denominator is zero. Mean-feature cosine similarity is a secondary diagnostic.
For reversal, use only valid opposite-label pairs. The confusion matrix shows
real/pretend mistakes hidden by mean accuracy. No metric measures robot success.

LEARNING ORDER / STEP-BY-STEP TODO
1. perturb: create identical interventions for both models before preprocessing.
2. AttentiveProbe.forward: read tokens with one learned attention query.
3. fit_probe: train using cross entropy; select using clean validation only.
4. paired_metrics: separate useful recognition from feature stability.
Provided: real-video loading, split manifest, pretrained extraction, bounded
cache, paired evaluation and plots. The first stub defines the test inputs.

DEFINITION OF DONE
Save selected settings, video IDs, per-video predictions, robustness curves
and confusion plots for both encoders. Use equal probe search budgets and
three seeds before claiming a ranking. Weak clean accuracy means check labels,
decoded frames, preprocessing and probe training first; increase data/epochs
if those checks pass. Do not draw robustness conclusions from weak probes.
Run: python -m exercises.world_models.14_video_world_model_robustness
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.video_world_model_data import ROOT, normalize_video, spatial_transform

VM_REVISION = "9981a9c8f77118c421e5228e1b219468a4b0238d"
CLASSES = ["Opening something", "Closing something", "Moving something up",
           "Moving something down", "Moving something towards the camera",
           "Moving something away from the camera",
           "Pretending to open something without actually opening it",
           "Pretending to close something without actually closing it"]
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
CONDITIONS = ["clean", "noise_005", "noise_015", "occlude_025", "occlude_050", "hold", "reverse"]


@dataclass
class Config:
    data_root: str = str(ROOT / "data/ssv2")
    output: str = str(ROOT / "outputs/video_robustness")
    reference: str = str(ROOT / "data/reference_vjepa2")
    train_per_class: int = 32
    validation_per_class: int = 8
    test_per_class: int = 12
    epochs: int = 20
    batch_size: int = 2
    seed: int = 42


def load_splits(config: Config) -> dict[str, list[dict]]:
    """Select balanced, disjoint real clips before loading either encoder."""
    root = Path(config.data_root); sources = {}
    for name in ("train", "validation"):
        path = root / "labels" / f"something-something-v2-{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Follow SETUP / REAL DATA above.")
        groups = {label: [] for label in range(len(CLASSES))}
        for row in json.loads(path.read_text(encoding="utf-8")):
            template = row["template"].replace("[", "").replace("]", "")
            if template in CLASSES:
                label = CLASSES.index(template)
                groups[label].append({"id": str(row["id"]), "label": label})
        sources[name] = groups
    rng = random.Random(config.seed)
    splits = {name: [] for name in ("train", "validation", "test")}
    for label in range(len(CLASSES)):
        train, test = sources["train"][label], sources["validation"][label]
        rng.shuffle(train); rng.shuffle(test)
        n, v, t = config.train_per_class, config.validation_per_class, config.test_per_class
        if min(n, v, t) < 1 or len(train) < n + v or len(test) < t:
            raise ValueError(f"Insufficient clips for {CLASSES[label]}.")
        splits["train"].extend(train[:n]); splits["validation"].extend(train[n:n+v]); splits["test"].extend(test[:t])
    ids = [row["id"] for rows in splits.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated video IDs across selected splits.")
    for clip_id in ids:
        path = root / "videos" / f"{clip_id}.webm"
        if not path.exists():
            raise FileNotFoundError(path)
    return splits


def read_video(config: Config, clip_id: str) -> Tensor:
    """Sample 16 ordered real frames, shared by both models."""
    path = Path(config.data_root) / "videos" / f"{clip_id}.webm"
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 16:
            raise ValueError(f"Expected at least 16 frames: {path}")
        indices = set(np.linspace(0, count - 1, 16).round().astype(int).tolist())
        frames = []
        for index in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}: {path}")
            if index in indices:
                frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        return torch.stack(frames).float() / 255
    finally:
        cap.release()


def perturb(clip: Tensor, condition: str, seed: int) -> Tensor:
    # Where this fits: paired interventions before model-specific preprocessing.
    # Input: CPU RGB float [16,3,H,W] in [0,1]. Output: same shape/range, no input mutation.
    # Method: out=clip.clone(); generator=torch.Generator().manual_seed(seed).
    # Steps: clean returns out; reverse uses out.flip(0); noise_005/noise_015 adds
    # 0.05/0.15 * torch.randn(out.shape,generator=generator), then .clamp(0,1).
    # occlude_025/occlude_050 hide a central rectangle covering 25%/50% of AREA:
    # height and width fractions are sqrt(0.25)/sqrt(0.50). Fill with 0.5 in all frames.
    # hold replaces frames 4:12 with frame 3; raise ValueError for unknown conditions.
    # Labels stay separate: paired_metrics handles valid reversed-label pairs.
    # Check: same seed -> same noise; reversed first frame == original last frame.
    raise NotImplementedError("Create reproducible pixel and temporal interventions.")


def load_encoder(name: str, config: Config, device: torch.device) -> nn.Module:
    if name == "vjepa2":
        robot = importlib.import_module("exercises.world_models.13_vjepa2_action_conditioned_robot_dynamics")
        return robot.load_frozen_encoder(robot.Config(reference=config.reference), device)
    if name == "videomaev2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained("OpenGVLab/VideoMAEv2-Large", revision=VM_REVISION,
                                          trust_remote_code=True)
        return model.eval().requires_grad_(False).to(device)
    raise ValueError(name)


@torch.no_grad()
def tokens_for(model: nn.Module, name: str, clip: Tensor, device: torch.device) -> Tensor:
    pixels = normalize_video(spatial_transform(clip, 256 if name == "vjepa2" else 224, training=False))
    pixels = pixels.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if name == "vjepa2":
            tokens = model(pixels)
        else:
            # Published forward pools tokens. Use its blocks before pooling so
            # the probe can read all space/time positions without changing weights.
            backbone = model.model
            tokens = backbone.patch_embed(pixels)
            tokens = backbone.pos_drop(tokens + backbone.pos_embed.to(tokens))
            for block in backbone.blocks:
                tokens = block(tokens)
        tokens = F.layer_norm(tokens.float(), (tokens.shape[-1],))
    expected = 2048 if name == "vjepa2" else 1568
    if tokens.shape != (1, expected, 1024):
        raise ValueError(f"Unexpected {name} token shape: {tuple(tokens.shape)}")
    return tokens.squeeze(0).cpu().half()


class TokenDataset(Dataset):
    def __init__(self, records: list[dict], directory: Path):
        self.records, self.directory = records, directory

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return torch.load(self.directory / f"{row['id']}.pt", weights_only=True).float(), row["label"]


def cache_clean(model, name: str, splits: dict, config: Config, device: torch.device) -> Path:
    records = splits["train"] + splits["validation"]
    identities = []
    for row in records:
        path = Path(config.data_root) / "videos" / f"{row['id']}.webm"
        identities.append((row, hashlib.sha256(path.read_bytes()).hexdigest()))
    identity = {"version": 1, "model": name, "vm_revision": VM_REVISION,
                "vjepa_revision": "204698b45b3712590f06245fbfba32d3be539812", "records": identities,
                "preprocessing": "uniform-16-centre-crop-imagenet-layernorm"}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(config.output) / "cache" / key; directory.mkdir(parents=True, exist_ok=True)
    for row in records:
        path = directory / f"{row['id']}.pt"
        if not path.exists():
            torch.save(tokens_for(model, name, read_video(config, row["id"]), device), path)
    (directory / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return directory


class AttentiveProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, 1024))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(1024)
        self.head = nn.Linear(1024, len(CLASSES))

    def forward(self, tokens: Tensor) -> Tensor:
        # Where this fits: the learned classifier reads frozen video tokens.
        # Input [B,N,1024]; output raw scores [B,8], not probabilities.
        # Method: query=self.query.expand(tokens.shape[0],-1,-1);
        # self.attention(query,tokens,tokens,need_weights=False) returns
        # (pooled [B,1,1024], None). Query asks what to read; keys/values are video tokens.
        # Steps: expand query; attend; .squeeze(1); apply self.norm then self.head.
        # Check: batch 1 still returns [1,8]. Cross entropy handles softmax later.
        raise NotImplementedError("Read tokens with a learned attention query.")


@torch.no_grad()
def probe_accuracy(probe: nn.Module, loader: DataLoader, device: torch.device) -> float:
    probe.eval(); correct = total = 0
    for tokens, labels in loader:
        scores = probe(tokens.to(device))
        correct += (scores.argmax(-1).cpu() == labels).sum().item(); total += len(labels)
    return correct / total


def fit_probe(train: TokenDataset, validation: TokenDataset, config: Config,
              device: torch.device) -> tuple[nn.Module, dict]:
    # Goal: select a probe on clean validation; test clips never enter this function.
    # Output: best probe in eval mode on device, plus JSON-safe lr/epoch/validation score.
    # Method: DataLoader(train,batch_size=config.batch_size,shuffle=True,num_workers=0);
    # DataLoader(validation,batch_size=config.batch_size,shuffle=False,num_workers=0).
    # Steps: for each lr in [1e-4,3e-4], reset torch.manual_seed(config.seed), create
    # AttentiveProbe().to(device), torch.optim.AdamW(probe.parameters(),lr=lr,weight_decay=0.01).
    # For each epoch: probe.train(); move tokens and labels to device;
    # optimizer.zero_grad(set_to_none=True); logits=probe(tokens);
    # loss=F.cross_entropy(logits,labels); loss.backward(); optimizer.step().
    # Call probe_accuracy after each epoch. Track the best score across ALL rates
    # and epochs, copying its state with {k:v.detach().cpu().clone() for k,v in
    # probe.state_dict().items()}. Save settings, not references to changing weights.
    # Load best state into a fresh probe and return it in eval mode with settings.
    # Check: encoder is absent from optimizer; both encoders receive equal search budgets.
    raise NotImplementedError("Train and select the clean-video attention probe.")


def paired_metrics(rows: list[dict], condition: str) -> dict:
    # Goal: summarize paired predictions without mistaking stability for correctness.
    # Input rows: label, clean_prediction, prediction, cosine (Python numbers).
    # Output keys: count, accuracy, clean_correct_count, conditional_retention, mean_cosine.
    # For reverse, keep only labels in OPPOSITE and score against OPPOSITE[label].
    # Otherwise score against label. clean_correct ALWAYS uses the original label.
    # Steps: filter eligible rows; count correct, clean-correct and their intersection;
    # divide by denominators defined above; average cosine over eligible rows.
    # Use None for undefined fractions/means and Python numbers for JSON output.
    # Check: two clean-correct clips, one surviving -> retention 0.5, even if a
    # third originally wrong clip becomes correct. That third clip affects accuracy only.
    raise NotImplementedError("Measure paired recognition and feature similarity.")


@torch.no_grad()
def evaluate(model, name: str, probe: nn.Module, records: list[dict], config: Config,
             device: torch.device) -> tuple[dict, dict]:
    probe.eval(); results = {condition: [] for condition in CONDITIONS}
    for row in records:
        clip = read_video(config, row["id"]); seed = config.seed + int(row["id"])
        clean = tokens_for(model, name, clip, device)
        clean_prediction = probe(clean.float().unsqueeze(0).to(device)).argmax(-1).item()
        for condition in CONDITIONS:
            tokens = clean if condition == "clean" else tokens_for(model, name, perturb(clip, condition, seed), device)
            prediction = probe(tokens.float().unsqueeze(0).to(device)).argmax(-1).item()
            cosine = F.cosine_similarity(clean.float().mean(0), tokens.float().mean(0), dim=0).item()
            results[condition].append({**row, "clean_prediction": clean_prediction, "prediction": prediction, "cosine": cosine})
    return {condition: paired_metrics(rows, condition) for condition, rows in results.items()}, results


def save_plots(name: str, metrics: dict, rows: dict, output: Path) -> None:
    conditions = [c for c in CONDITIONS if c != "reverse"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for key in ("accuracy", "conditional_retention"):
        ax.plot(conditions, [metrics[c][key] for c in conditions], "o-", label=key.replace("_", " "))
    ax.set(ylabel="Fraction (higher is better)", ylim=(0, 1), title=name)
    ax.tick_params(axis="x", rotation=25); ax.legend(); fig.tight_layout()
    fig.savefig(output / f"{name}_robustness.png"); plt.close(fig)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for row in rows["clean"]:
        matrix[row["label"], row["prediction"]] += 1
    fig, ax = plt.subplots(figsize=(12, 9)); ax.imshow(matrix)
    ax.set(xticks=range(8), yticks=range(8), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted action", ylabel="Real action", title=f"{name}: clean confusion counts")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white")
    fig.tight_layout(); fig.savefig(output / f"{name}_confusion.png"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-root", default=Config.data_root)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=["vjepa2", "videomaev2"], default=["vjepa2", "videomaev2"])
    args = parser.parse_args()
    config = Config(data_root=args.data_root, output=args.output, seed=args.seed)
    torch.manual_seed(config.seed); splits = load_splits(config)
    output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({"config": asdict(config), "classes": CLASSES, "splits": splits}, indent=2), encoding="utf-8")
    clip = read_video(config, splits["test"][0]["id"])
    fig, axes = plt.subplots(len(CONDITIONS), 4, figsize=(10, 14))
    for axes_row, condition in zip(axes, CONDITIONS):
        transformed = perturb(clip, condition, config.seed)
        for ax, index in zip(axes_row, [0, 5, 10, 15]):
            ax.imshow(transformed[index].permute(1, 2, 0)); ax.axis("off"); ax.set_title(f"{condition}: frame {index}")
    fig.tight_layout(); fig.savefig(output / "interventions.png"); plt.close(fig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in args.models:
        model = load_encoder(name, config, device)
        directory = cache_clean(model, name, splits, config, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        probe, settings = fit_probe(TokenDataset(splits["train"], directory), TokenDataset(splits["validation"], directory), config, device)
        torch.save({"state": probe.state_dict(), "settings": settings, "config": asdict(config), "classes": CLASSES}, output / f"{name}_probe.pt")
        model = load_encoder(name, config, device)
        metrics, rows = evaluate(model, name, probe, splits["test"], config, device)
        report = {"model": name, "settings": settings, "metrics": metrics, "per_video": rows}
        (output / f"{name}_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_plots(name, metrics, rows, output); print(name, json.dumps(metrics, indent=2))
        del model, probe
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
