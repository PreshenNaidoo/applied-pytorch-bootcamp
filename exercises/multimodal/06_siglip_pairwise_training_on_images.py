"""
============================================================
EXERCISE 06: SigLIP PAIRWISE TRAINING ON IMAGES
============================================================

WHAT YOU ARE BUILDING
You are building a tiny SigLIP-style image-text matching model.

SigLIP is close to CLIP, but the loss asks a different question.

CLIP asks:

```text
Which caption in this batch matches this image?
```

SigLIP asks:

```text
Is this image-text pair a match or not?
```

Dataset path:
- use COCO-30 image-caption pairs

WHY THIS EXERCISE EXISTS
Many modern VLMs use SigLIP-style vision encoders.
This exercise gives you a clean comparison with CLIP:
- CLIP uses a softmax over a batch
- SigLIP uses pairwise match/non-match labels

HOW THIS FITS INTO THE INTERVIEW PREP
This helps you compare two common image-text training losses.
That matters because interviewers may ask why one VLM backbone or training setup
is different from another.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
---------------------------------------------------------------
large web image-text pairs           small COCO-30 sample
many negative pairs                  small wrong image-caption pairs
large vision/text encoders (SigLIP ViT or CLIP ViT)    small CNN and text encoder
pairwise sigmoid loss                pairwise sigmoid loss

VISUALS TO CREATE
Save:
- a match-score table before training
- a match-score table after training
- a loss curve

INPUTS AND OUTPUTS
Image:
- `[3, image_size, image_size]`

Caption:
- text such as `"red square"`

Pair target:
- `1` means the image and caption match
- `0` means they do not match

Model output:
- one raw match logit for each image-text pair

METRICS TO TRACK
- pairwise training loss
- match/non-match accuracy
- average score for matching pairs
- average score for wrong pairs

STEP-BY-STEP TODO
1. Build real image-caption examples from COCO-30.
2. Turn them into matching and mismatching pairs.
3. Build a caption vocabulary.
4. Tokenize captions.
5. Build image and text encoders.
6. Compute one match score for each image-text pair.
7. Train with binary cross entropy on match labels.
8. Evaluate pair accuracy.
9. Save before/after score tables and a loss curve.

DEFINITION OF DONE
You are done when:
- matched pairs score higher than mismatched pairs
- loss goes down
- the score table makes the model behavior visible
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from turtle import forward
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
class SigLIPConfig:
    """Settings for tiny SigLIP training."""

    image_size: int = 48
    embedding_dim: int = 64
    max_text_length: int = 4
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 3e-3
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    hf_dataset_name: str = "sayakpaul/coco-30-val-2014"
    sample_size: int = 128


class PairExample(TypedDict):
    """One image-text pair."""

    image: torch.Tensor
    caption: str
    target: int


class SigLIPMetrics(TypedDict):
    """Training summary."""

    initial_loss: float
    final_loss: float
    pair_accuracy: float
    loss_curve_path: str
    score_table_path: str


# Learning order:
# 1. Build image-caption examples.
# 2. Build matching and mismatching pairs.
# 3. Encode images and text.
# 4. Compute one score per pair.
# 5. Train with pairwise sigmoid loss.
# 6. Visualize score tables.


def build_pair_dataset(config: SigLIPConfig) -> list[PairExample]:
    """Build positive and negative image-text pairs."""
    # Where this fits:
    # SigLIP trains on image-text pairs marked as match or mismatch.
    # This helper creates those pairs from image tensors and captions.
    #
    # Goal:
    # Create image tensors and captions, then build pairs.
    #
    # Input:
    # - `config.hf_dataset_name`: `"sayakpaul/coco-30-val-2014"`
    # - `config.sample_size`: how many image-caption rows to load
    #
    # Output:
    # - list of dictionaries with:
    #   - `"image"`: `[3, image_size, image_size]`
    #   - `"caption"`: text label
    #   - `"target"`: `1` for match, `0` for mismatch
    #
    # Method:
    # Use `datasets.load_dataset` from Hugging Face:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    # Use `torchvision.transforms.Resize((config.image_size, config.image_size))`
    # and `torchvision.transforms.ToTensor()` to turn each PIL image into a tensor.
    #
    # Steps:
    # 1. Load COCO-30 with the exact `load_dataset(...)` call above.
    # 2. Build the transform with
    #    `torchvision.transforms.Compose([torchvision.transforms.Resize((config.image_size, config.image_size)), torchvision.transforms.ToTensor()])`.
    # 3. For each row, convert `row["image"]` to RGB, resize it, and tensorize it.
    # 4. Store the real image-caption match as target `1`.
    # 5. Build a mismatched pair by using the same image with the next caption in the list.
    # 6. Store the mismatched pair as target `0`.
    # 7. Return all positive and negative pairs.
    #
    # Checks:
    # - both target values 0 and 1 should appear
    # - images should be real tensors, not feature vectors
    
    from datasets import load_dataset

    data = load_dataset(path=config.hf_dataset_name, split=f"train[:{config.sample_size}]")

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(config.image_size, config.image_size)),
                                                torchvision.transforms.ToTensor()])

    examples: list[tuple[torch.Tensor, str]] = []
    for row in data:
        image = row["image"].convert("RGB")
        image_tensor = transform(image)

        caption = row["caption"]
        if isinstance(caption, list):
            caption = caption[0]
        examples.append((image_tensor, caption))

    pairs: list[PairExample] = []
    for index, (image_tensor, caption) in enumerate(examples):
        wrong_caption = examples[(index + 1) % len(examples)][1]
        pairs.append({"image": image_tensor, "caption": caption, "target": 1})
        pairs.append({"image": image_tensor, "caption": wrong_caption, "target": 0})

    # raise NotImplementedError("Build SigLIP positive and negative image-text pairs.")
    return pairs


def build_vocabulary(captions: list[str]) -> dict[str, int]:
    """Build a text vocabulary."""
    # Goal:
    # Build a word-to-id dictionary from all captions.
    #
    # Input:
    # - captions such as `"a dog running"` or `"a photo of a truck"`
    #
    # Output:
    # - dictionary with `<pad>` at id 0 and `<unk>` at id 1.
    #
    # Method:
    # Use normal Python string methods:
    # - `caption.lower()` lowercases a caption.
    # - `.split()` splits it into words on whitespace.
    # Use `sorted(...)` so the ids are stable every run.
    #
    # Steps:
    # 1. Start with `{"<pad>": 0, "<unk>": 1}`.
    # 2. Lowercase and split every caption.
    # 3. Sort the unique words.
    # 4. Add each word to the dictionary.
    #
    # Checks:
    # - `token_to_id["<pad>"] == 0`
    # - `token_to_id["<unk>"] == 1`
    # - A real word from the captions should be present.

    all_words = []

    for caption in captions:
        words = caption.lower().split()
        all_words.extend(words)

    vocabulary = sorted(set(all_words))
    vocabulary.insert(0, "<pad>")
    vocabulary.insert(1, "<unk>")

    token_to_id = {w:i for i,w in enumerate(vocabulary)}

    return token_to_id

    #raise NotImplementedError("Build vocabulary from pair captions.")


def tokenize_captions(captions: list[str], token_to_id: dict[str, int], config: SigLIPConfig) -> torch.Tensor:
    """Tokenize captions into padded ids."""
    # Goal:
    # Turn caption strings into a fixed-size tensor of token ids.
    #
    # Input:
    # - `captions`: list[str]
    # - `token_to_id`: output from `build_vocabulary`
    #
    # Output:
    # - tensor shaped `[len(captions), config.max_text_length]`
    # - dtype `torch.long`
    #
    # Method:
    # Use `torch.full((len(captions), config.max_text_length), 0, dtype=torch.long)`.
    # `torch.full` is a PyTorch function that creates a tensor already filled
    # with the pad id.
    #
    # Steps:
    # 1. Lowercase and split each caption.
    # 2. Convert words to ids with `.get(word, unk_id)`.
    # 3. Keep at most `config.max_text_length` ids.
    # 4. Copy ids into the padded tensor.
    #
    # Checks:
    # - Shape is `[number of captions, max_text_length]`.
    # - Padding positions are `0`.

    pad_id = 0
    unk_id = 1

    token_ids_list = []

    for caption in captions:
        words = caption.lower().split()

        token_ids = []
        for word in words:
            token_id = token_to_id.get(word, unk_id)
            token_ids.append(token_id)

        token_ids = token_ids[: config.max_text_length] # truncate if greater than max_text_length

        if len(token_ids) < config.max_text_length:
            token_ids += [pad_id] * (config.max_text_length - len(token_ids))

        token_ids_list.append(torch.tensor(token_ids, dtype=torch.long))

    return torch.stack(token_ids_list, dim=0)

    #raise NotImplementedError("Tokenize captions.")


class SigLIPPairDataset(Dataset):
    """Dataset of image-caption pairs."""

    def __init__(self, pairs: list[PairExample], token_to_id: dict[str, int], config: SigLIPConfig):
        self.pairs = pairs
        self.token_to_id = token_to_id
        self.config = config

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[index]
        token_ids = tokenize_captions([pair["caption"]], self.token_to_id, self.config)[0]
        return {
            "image": pair["image"],
            "caption_token_ids": token_ids,
            "target": torch.tensor(float(pair["target"]), dtype=torch.float32),
        }

class TinyImageEncoder(nn.Module):
    def __init__(self, config: SigLIPConfig):
        super().__init__()

        self.config = config

        self.conv2d_1 = nn.Conv2d(in_channels = 3, out_channels=16, kernel_size=3, stride=1, padding="same")

        self.maxpool2d = nn.MaxPool2d(kernel_size=2) #default stride is kernel_size

        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same")

        self.adaptive_avgpool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.linear = nn.Linear(in_features=32, out_features=config.embedding_dim)

    def forward(self, images) -> torch.tensor:
        
        x = nn.functional.relu(self.conv2d_1(images))   # [B, 16, 48, 48]
        x = self.maxpool2d(x)                           # [B, 16, 24, 24]

        x = nn.functional.relu(self.conv2d_2(x))        # [B, 32, 24, 24]
        x = self.adaptive_avgpool2d(x)                  # [B, 32, 1, 1]

        x = torch.flatten(x, start_dim=1)               # [B, 32]

        embeddings = self.linear(x)  # [B, embedding_dim]

        norm_embeddings = nn.functional.normalize(embeddings, dim=-1)

        return norm_embeddings

    
class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: SigLIPConfig):
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.embedding_dim, padding_idx=pad_id)

    def forward(self, caption_token_ids: torch.tensor) -> torch.tensor:
        
        # caption_token_ids shape is [batch_size, max_text_length]

        embeddings = self.embedding(caption_token_ids)          # [batch_size, max_text_length, embedding_dim]

        mask = caption_token_ids != self.pad_id                 # [batch_size, max_text_length]
        mask = mask.unsqueeze(dim=-1)                           # [batch_size, max_text_length, 1]  for broadcasting in the next line

        masked_embeddings = embeddings * mask                   # [batch_size, max_text_length, embedding_dim]

        summed = torch.sum(masked_embeddings, dim=1)            # [batch_size, embedding_dim]

        count_non_pad = torch.sum(mask, dim=1).clamp(min=1)     # [batch_size, 1]

        text_embeddings = summed/count_non_pad                  # [batch_size, embedding_dim]

        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)  # [batch_size, embedding_dim]

        return text_embeddings


class TinySigLIP(nn.Module):
    """Tiny SigLIP-style model."""

    def __init__(self, vocab_size: int, config: SigLIPConfig):
        super().__init__()
        # Goal:
        # Build image and text encoders that return same-width vectors.
        # You may reuse the same small CNN/text-averaging idea from the related exercise.
        #
        # Input:
        # - `vocab_size`: number of text tokens.
        #
        # Output:
        # - Store layers on `self`, for example:
        #   - `self.image_encoder`
        #   - `self.text_embedding`
        #
        # Method:
        # Use PyTorch layers from `torch.nn`:
        # - `nn.Conv2d(...)` for image feature extraction.
        # - `nn.ReLU()` for the non-linearity.
        # - `nn.MaxPool2d(2)` to shrink the image grid.
        # - `nn.AdaptiveAvgPool2d((1, 1))` to get one vector per image.
        # - `nn.Linear(32, config.embedding_dim)` for the final image projection.
        # - `nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)` for text.
        #
        # Steps:
        # 1. Build `self.image_encoder` with `nn.Sequential(...)`.
        # 2. Build `self.text_embedding`.
        # 3. In `forward`, average non-padding token embeddings.
        # 4. Normalize image and text vectors before scoring.
        #
        # Check:
        # - Both paths must produce `[batch_size, config.embedding_dim]`.

        self.image_encoder = TinyImageEncoder(config)
        self.text_encoder = TinyTextEncoder(vocab_size, 0, config)

        #raise NotImplementedError("Build TinySigLIP encoders.")

    def forward(self, images: torch.Tensor, caption_token_ids: torch.Tensor) -> torch.Tensor:
        """Return one match logit per image-text pair."""
        # Output shape:
        # - `[batch_size]`
        #
        # Method:
        # Use `torch.nn.functional.normalize(..., dim=-1)` to normalize vectors.
        # Use elementwise `*` to multiply matching image/text rows.
        # Use `.sum(dim=-1)` to turn each row into one raw logit.
        #
        # Steps:
        # 1. Encode images to `[batch_size, embedding_dim]`.
        # 2. Encode captions to `[batch_size, embedding_dim]`.
        # 3. Normalize both with `torch.nn.functional.normalize(..., dim=-1)`.
        # 4. Compute one score per pair:
        #    `(image_embeddings * text_embeddings).sum(dim=-1)`.
        #
        # Check:
        # - Output shape should be `[batch_size]`, not `[batch_size, batch_size]`.

        image_embeddings = self.image_encoder(images)               # [batch_size, embedding_dim]

        text_embeddings = self.text_encoder(caption_token_ids)      # [batch_size, embedding_dim]

        scores = image_embeddings * text_embeddings                 # [batch_size, embedding_dim]

        scores = scores.sum(dim=-1)                                 # [batch_size]

        return scores

        #raise NotImplementedError("Return pairwise match logits.")


def compute_siglip_loss(pair_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute pairwise sigmoid loss."""
    # Goal:
    # Train match pairs to high scores and mismatch pairs to low scores.
    #
    # Input:
    # - `pair_logits`: `[batch_size]`
    # - `targets`: `[batch_size]`, where 1 means match and 0 means mismatch
    #
    # Method:
    # Use `nn.BCEWithLogitsLoss()` from `torch.nn`.
    # Call it like:
    # `loss = nn.BCEWithLogitsLoss()(pair_logits, targets)`.
    # Do not apply `torch.sigmoid(...)` before this loss.
    #
    # Steps:
    # 1. Make sure `targets` is float, not long.
    # 2. Build the loss function with `nn.BCEWithLogitsLoss()`.
    # 3. Return `loss_fn(pair_logits, targets)`.

    loss = nn.BCEWithLogitsLoss()

    # Compare each raw pair score with its match label and return the mean loss.
    # `BCEWithLogitsLoss` applies sigmoid inside, so it turns each raw score into
    # a match probability before comparing it with target 0 or 1.

    # `pair_logits` are raw scores, not probabilities.
    # `BCEWithLogitsLoss` first applies sigmoid inside the loss:
    # sigmoid(logit) = 1 / (1 + exp(-logit)).
    # That turns each raw score into a probability from 0 to 1.
    # Then binary cross entropy compares that probability with the target:
    # if target is 1, the loss is `-log(probability)`, so low probability is punished.
    # if target is 0, the loss is `-log(1 - probability)`, so high probability is punished.
    # target 1 pushes the score higher, and target 0 pushes the score lower.
    # PyTorch combines sigmoid and binary cross entropy in one stable function,
    # which avoids numeric problems from calling sigmoid ourselves first.
    return loss(pair_logits, targets)

    #raise NotImplementedError("Compute binary cross entropy with logits.")


def train_one_epoch(model: TinySigLIP, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train one SigLIP epoch."""
    # Goal:
    # Train for one pass over the pair loader and return mean loss.
    #
    # Input:
    # - batches have `"image"`, `"caption_token_ids"`, and `"target"`.
    #
    # Method:
    # Use normal PyTorch training calls:
    # - `model.train()` turns on training mode.
    # - `.to(device)` moves tensors to CPU or GPU.
    # - `optimizer.zero_grad()` clears old gradients.
    # - `loss.backward()` computes new gradients.
    # - `optimizer.step()` updates parameters.
    #
    # Steps:
    # 1. Call `model.train()`.
    # 2. Start `total_loss = 0.0` and `total_examples = 0`.
    # 3. For each batch, move `"image"`, `"caption_token_ids"`, and `"target"` to `device`.
    # 4. Compute logits with `model(images, caption_token_ids)`.
    # 5. Compute loss with `compute_siglip_loss(logits, targets)`.
    # 6. Run zero-grad, backward, and step.
    # 7. Add `loss.item() * batch_size` to `total_loss`.
    # 8. Return `total_loss / total_examples`.
    #
    # Check:
    # - Return a Python float mean loss over all examples.

    model.train()

    total_loss = 0
    total_elements = 0

    for data in loader:
        images = data["image"]
        captions = data["caption_token_ids"]
        targets = data["target"]

        images = images.to(device)
        captions = captions.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        pair_logits = model(images, captions)
        loss = compute_siglip_loss(pair_logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]

        total_loss += loss.item() * batch_size
        total_elements += batch_size

    return total_loss / total_elements

    #raise NotImplementedError("Train one epoch.")

@torch.no_grad()
def evaluate_pairs(model: TinySigLIP, loader: DataLoader) -> float:
    """Return pair classification accuracy."""
    # Goal:
    # Measure how often the model says match for target 1 and mismatch for target 0.
    #
    # Method:
    # Use `model.eval()` and `torch.no_grad()` from PyTorch so evaluation does
    # not build a training graph.
    # Use `torch.sigmoid(logits)` to convert raw logits into probabilities.
    #
    # Steps:
    # 1. Call `model.eval()`.
    # 2. Loop over the loader inside `with torch.no_grad():`.
    # 3. Move batch tensors to `device`.
    # 4. Compute logits with the model.
    # 5. Convert logits with `torch.sigmoid(logits)`.
    # 6. Use `probs >= 0.5` for predicted labels.
    # 7. Compare predictions with targets and return total correct / total examples.
    #
    # Check:
    # - Accuracy should be a float between `0.0` and `1.0`.

    model.eval()

    total_accuracy = 0
    num_batches = 0

    for data in loader:
        images = data["image"]
        captions = data["caption_token_ids"]
        targets = data["target"]

        images = images.to(device)
        captions = captions.to(device)
        targets = targets.to(device)

        pair_logits = model(images, captions)
        
        probabilities = torch.sigmoid(pair_logits)

        pred = (probabilities>=0.5).float()
        correct = (pred == targets).float()
        accuracy = correct.mean().item()

        total_accuracy+=accuracy
        num_batches+=1

    mean_accuracy = total_accuracy / num_batches

    return mean_accuracy
    #raise NotImplementedError("Evaluate match/non-match accuracy.")


def save_siglip_visuals(losses: list[float], model: TinySigLIP, pairs: list[PairExample], token_to_id: dict[str, int], config: SigLIPConfig) -> tuple[str, str]:
    """Save a loss curve and score table."""
    # Goal:
    # Save simple visuals that show whether matching pairs score higher.
    #
    # Output:
    # - `(loss_curve_path, score_table_path)` as strings.
    #
    # Method:
    # Use Matplotlib with the non-GUI backend:
    # `import matplotlib; matplotlib.use("Agg")`.
    # Use `plt.plot(...)` for the loss curve.
    # Use `plt.bar(...)` or `plt.imshow(...)` for the score table.
    #
    # Steps:
    # 1. Create `config.output_dir` with `mkdir(parents=True, exist_ok=True)`.
    # 2. Save `losses` to `siglip_loss_curve.png` with `plt.plot(...)`.
    # 3. Build a small `SigLIPPairDataset` from the first few pairs.
    # 4. Score those pairs with `model.eval()` and `torch.no_grad()`.
    # 5. Save scores and targets as `siglip_score_table.png`.
    # 6. Return both paths as strings.
    #
    # Check:
    # - Both paths should point inside `outputs/multimodal_interview/`.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)

    loss_curve_path = config.output_dir / "siglip_loss_curve.png"
    score_table_path = config.output_dir / "siglip_score_table.png"

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("SigLIP training loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()

    small_pairs = pairs[:8]
    small_dataset = SigLIPPairDataset(small_pairs, token_to_id, config)
    small_loader = DataLoader(small_dataset, batch_size=len(small_dataset), shuffle=False)

    model.eval()
    batch = next(iter(small_loader))
    images = batch["image"].to(device)
    caption_token_ids = batch["caption_token_ids"].to(device)
    targets = batch["target"].to(device)

    with torch.no_grad():
        logits = model(images, caption_token_ids)
        scores = torch.sigmoid(logits).detach().cpu()

    target_values = targets.detach().cpu()
    labels = ["match" if target.item() == 1 else "wrong" for target in target_values]
    colors = ["tab:green" if target.item() == 1 else "tab:red" for target in target_values]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(scores)), scores.tolist(), color=colors)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(range(len(scores)), labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("SigLIP pair match probabilities")
    plt.xlabel("pair target")
    plt.ylabel("predicted match probability")
    plt.tight_layout()
    plt.savefig(score_table_path)
    plt.close()

    # raise NotImplementedError("Save SigLIP training visuals.")
    return str(loss_curve_path), str(score_table_path)


def run_exercise(config: SigLIPConfig) -> SigLIPMetrics:
    """Run full SigLIP training."""
    # Main workflow:
    # pairs -> vocabulary -> loader -> model -> train -> evaluate -> visualize.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pair_dataset(config)
    captions = [pair["caption"] for pair in pairs]
    token_to_id = build_vocabulary(captions)
    dataset = SigLIPPairDataset(pairs, token_to_id, config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TinySigLIP(vocab_size=len(token_to_id), config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    for i in range(config.epochs):
        loss = train_one_epoch(model, loader, optimizer)
        losses.append(loss)
        print(f"Epoch {i+1} -> loss: {loss}")

    pair_accuracy = evaluate_pairs(model, loader)
    loss_curve_path, score_table_path = save_siglip_visuals(losses, model, pairs, token_to_id, config)

    print(f"pair_accuracy: {pair_accuracy}")

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "pair_accuracy": pair_accuracy,
        "loss_curve_path": loss_curve_path,
        "score_table_path": score_table_path,
    }


def check_metrics(metrics: SigLIPMetrics) -> bool:
    """Return True when training improved."""
    return metrics["final_loss"] < metrics["initial_loss"] and metrics["pair_accuracy"] > 0.70


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the SigLIP scaffold one step at a time."""
    config = SigLIPConfig()
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

    pairs, ok = report_task_progress("build_pair_dataset", build_pair_dataset, config)
    if not count_or_stop(ok):
        return
    captions = [pair["caption"] for pair in pairs]

    token_to_id, ok = report_task_progress("build_vocabulary", build_vocabulary, captions)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("tokenize_captions", tokenize_captions, captions[:4], token_to_id, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinySigLIP.__init__", TinySigLIP, len(token_to_id), config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("compute_siglip_loss", compute_siglip_loss, torch.zeros(4), torch.ones(4))
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "siglip pair accuracy", metrics["pair_accuracy"], "> 0.70 and loss improves")


if __name__ == "__main__":
    main()
