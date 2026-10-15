"""
============================================================
EXERCISE 09: BLIP-2 Q-FORMER ON IMAGE CAPTION DATA
============================================================

WHAT YOU ARE BUILDING
You are building the core BLIP-2 bridge idea:
learned query tokens pull useful information from image features.

Plain picture:

```text
image -> image encoder -> image patch features
learned query tokens -> attend to image features
query outputs -> caption or answer head
```

Image patch features come from the image encoder inside this exercise.

WHY THIS EXERCISE EXISTS
BLIP-2 matters because it connects a frozen image encoder to a language model
using a smaller bridge called a Q-Former.

This exercise teaches the bridge idea without needing a large LLM.

HOW THIS FITS INTO THE INTERVIEW PREP
This gives you a second VLM connection pattern after LLaVA.
Instead of projecting every visual token into the language model, BLIP-2 uses
learned queries to pull out a smaller amount of visual information.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
---------------------------------------------------------------
large frozen image encoder (EVA-CLIP or ViT-G)  small CNN patch encoder
Q-Former with learned queries (BLIP-2 Q-Former) small attention block with learned queries
frozen large language model (Flan-T5 or OPT)    small caption/answer head
large caption data                    small COCO-30 image-caption sample

VISUALS TO CREATE
Save:
- loss curve
- query attention heatmap over image patches

INPUTS AND OUTPUTS
Image:
- `[3, image_size, image_size]`

Patch features:
- `[num_patches, vision_hidden_size]`

Learned query outputs:
- `[num_query_tokens, query_hidden_size]`

Model output:
- logits for the small caption or answer label

METRICS TO TRACK
- training loss
- label accuracy
- attention weights from query tokens to image patches

STEP-BY-STEP TODO
1. Build real image-caption examples.
2. Convert captions into small class labels.
3. Encode each image into patch features.
4. Create learned query tokens.
5. Let query tokens attend to patch features.
6. Pool query outputs.
7. Predict the caption label.
8. Train the small bridge.
9. Evaluate label accuracy.
10. Save the loss curve and query attention heatmap.

DEFINITION OF DONE
You are done when:
- query tokens attend to image patch features
- training loss goes down
- attention can be visualized over patches
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
class QFormerConfig:
    """Settings for the tiny Q-Former exercise."""

    image_size: int = 48
    patch_grid_size: int = 6
    vision_hidden_size: int = 64
    query_hidden_size: int = 64
    num_query_tokens: int = 8
    num_attention_heads: int = 4
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 3e-3
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    hf_dataset_name: str = "sayakpaul/coco-30-val-2014"
    sample_size: int = 128


class CaptionExample(TypedDict):
    """One image-caption example."""

    image: torch.Tensor
    caption: str
    label_id: int


class QFormerMetrics(TypedDict):
    """Training summary."""

    initial_loss: float
    final_loss: float
    label_accuracy: float
    loss_curve_path: str
    attention_path: str


class QFormerDataset(Dataset):
    """Dataset for the tiny Q-Former label task."""

    def __init__(self, examples: list[CaptionExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        return {
            "image": example["image"],
            "label_id": torch.tensor(example["label_id"], dtype=torch.long),
            "caption": example["caption"],
        }


# Learning order:
# 1. Build image-caption data.
# 2. Convert captions to small class labels first.
# 3. Build image patch features from pixels.
# 4. Build learned query tokens.
# 5. Let queries attend to patches.
# 6. Train a small head from query outputs.
# 7. Visualize attention.


def build_caption_dataset(config: QFormerConfig) -> list[CaptionExample]:
    """Build real image-caption examples."""
    # Where this fits:
    # BLIP-2 uses a bridge between image features and language.
    # This helper creates the small real image-caption task used to train that bridge.
    #
    # Goal:
    # Return image tensors, captions, and small caption labels.
    #
    # Input:
    # - `config.hf_dataset_name`: `"sayakpaul/coco-30-val-2014"`
    # - `config.sample_size`: how many image-caption rows to load
    #
    # Output:
    # - list of dictionaries with:
    #   - `"image"`: `[3, image_size, image_size]`
    #   - `"caption"`: text string
    #   - `"label_id"`: integer class id
    #
    # Method:
    # Use `load_dataset` from the Hugging Face `datasets` package:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    # Use this exact transform shape from `torchvision.transforms`:
    # `torchvision.transforms.Compose([torchvision.transforms.Resize((config.image_size, config.image_size)), torchvision.transforms.ToTensor()])`.
    # For a small classification target, map captions into keyword labels such as
    # person, vehicle, animal, food, or other.
    #
    # Steps:
    # 1. Load the Hugging Face split with `load_dataset(...)`.
    # 2. Inspect one row with `data[0].keys()` if you are unsure of the field names.
    # 3. For each row, read the image and caption.
    # 4. Convert the image to RGB with the PIL method `image.convert("RGB")`.
    # 5. Resize the image and convert it to a tensor with the torchvision transform.
    # 6. Lowercase the caption.
    # 7. Map caption keywords to one integer `label_id`.
    # 8. Append dictionaries with `"image"`, `"caption"`, and `"label_id"`.
    # 9. Stop once you have `config.sample_size` examples.
    #
    # Checks:
    # - Images are tensors, not precomputed feature vectors.
    # - `label_id` values start at 0 and are valid class ids.


    from datasets import load_dataset

    data = load_dataset(path=config.hf_dataset_name, split=f"train[:{config.sample_size}]")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config.image_size, config.image_size)),
        torchvision.transforms.ToTensor(),
    ])

    label_keywords = {
        0: ("person", "people", "man", "woman", "child", "boy", "girl", "skier", "player"),
        1: ("car", "truck", "bus", "train", "bike", "bicycle", "motorcycle", "vehicle", "boat"),
        2: ("dog", "cat", "horse", "bird", "animal", "elephant", "bear", "zebra", "giraffe"),
        3: ("food", "pizza", "sandwich", "cake", "banana", "apple", "orange", "meal", "plate"),
    }

    examples: list[CaptionExample] = []
    for row in data:
        image = row["image"].convert("RGB")
        image_tensor = transform(image)

        caption = row["caption"]
        if isinstance(caption, list):
            caption = caption[0]
        caption = str(caption)
        caption_lower = caption.lower()

        label_id = 4
        for candidate_label, keywords in label_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                label_id = candidate_label
                break

        examples.append({"image": image_tensor, "caption": caption, "label_id": label_id})

    return examples
    
    #raise NotImplementedError("Build image-caption data for Q-Former training.")


class PatchImageEncoder(nn.Module):
    """Small image encoder that returns patch features."""

    def __init__(self, config: QFormerConfig):
        super().__init__()
        # Goal:
        # Convert images `[B, 3, H, W]` into patch features `[B, num_patches, vision_hidden_size]`.
        #
        # Method:
        # Use `nn.Conv2d` from `torch.nn`.
        # The call shape is:
        # `nn.Conv2d(3, config.vision_hidden_size, kernel_size=patch_size, stride=patch_size)`.
        # A stride equal to patch size means each output grid cell sees one image patch.
        #
        # Steps:
        # 1. Compute `patch_size = config.image_size // config.patch_grid_size`.
        # 2. Check that `config.image_size % config.patch_grid_size == 0`.
        # 3. Create `self.patch_conv` with `kernel_size=patch_size` and `stride=patch_size`.
        #
        # Checks:
        # - `config.image_size` should divide cleanly by `config.patch_grid_size`.
        # - The convolution output grid should be `[patch_grid_size, patch_grid_size]`.

        self.patch_size = config.image_size // config.patch_grid_size

        if config.image_size % config.patch_grid_size != 0:
            raise ValueError("Image_size is not divisible by patch size")

        self.patch_conv = nn.Conv2d(in_channels = 3, out_channels=config.vision_hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        
        #raise NotImplementedError("Build patch image encoder.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch features."""
        # Goal:
        # Turn each image into a sequence of patch vectors.
        #
        # Input:
        # - `images`: `[batch_size, 3, image_size, image_size]`
        #
        # Output:
        # - `[batch_size, num_patches, vision_hidden_size]`
        #
        # Method:
        # Use `self.patch_conv(images)` for the convolution.
        # Use `features.flatten(start_dim=2)` to flatten only height and width.
        # Use `.transpose(1, 2)` to move hidden size after patch index.
        #
        # Steps:
        # 1. Run `features = self.patch_conv(images)`.
        # 2. Check that `features` is `[B, hidden, grid, grid]`.
        # 3. Flatten grid cells with `features = features.flatten(start_dim=2)`.
        # 4. Transpose with `features = features.transpose(1, 2)`.
        # 5. Return features shaped `[B, num_patches, hidden]`.
        #
        # Check:
        # - `num_patches` should equal `patch_grid_size * patch_grid_size`.

        # patch_grid_size is how many patches you want along one side of the image.
        # patch_size is how many pixels wide and tall each patch is.

        # Example:
        # If images are [batch_size, 3, 48, 48], vision_hidden_size is 64,
        # and patch_grid_size is 6, then patch_size is 48 // 6 = 8.
        # The convolution uses 64 filters. Each filter looks at one 8x8 patch.
        # Because stride is also 8, it moves by 8 pixels each time.
        # So the 48x48 image becomes a 6x6 grid of patch outputs.
        # Shape change:
        # [batch_size, 3, 48, 48] -> [batch_size, 64, 6, 6]
        # num_patches = 6x6 = 36 total number of patches

        features = self.patch_conv(images)  # [batch_size, vision_hidden_size, patch_grid_size, patch_grid_size]

        features = features.flatten(start_dim = 2)  # [batch_size, vision_hidden_size, num_patches]

        features = features.transpose(1, 2)  # [batch_size, num_patches, vision_hidden_size]

        return features

        #raise NotImplementedError("Return image patch features.")


class TinyQFormer(nn.Module):
    """Tiny learned-query bridge."""

    def __init__(self, config: QFormerConfig):
        super().__init__()
        # Goal:
        # Build learned query tokens and cross-attention.
        #
        # Suggested layers:
        # - `self.query_tokens = nn.Parameter(torch.randn(config.num_query_tokens, config.query_hidden_size) * 0.02)`
        # - `self.cross_attention = nn.MultiheadAttention(embed_dim=config.query_hidden_size, num_heads=config.num_attention_heads, batch_first=True)`
        #
        # Method:
        # Use `nn.Parameter` from `torch.nn` for learned query tokens.
        # Use `torch.randn(...)` to initialize them with small random values.
        # Use `nn.MultiheadAttention` from `torch.nn` with `batch_first=True`.
        #
        # Steps:
        # 1. Create `self.query_tokens` with shape `[num_query_tokens, query_hidden_size]`.
        # 2. Create `self.cross_attention = nn.MultiheadAttention(...)`.
        # 3. Use `embed_dim=config.query_hidden_size`.
        # 4. Use `num_heads=config.num_attention_heads`.
        #
        # Checks:
        # - Query token shape should be `[num_query_tokens, query_hidden_size]`.
        # - `query_hidden_size` must match the attention `embed_dim`.

        self.query_tokens = nn.Parameter(torch.randn(config.num_query_tokens, config.query_hidden_size) * 0.02)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=config.query_hidden_size, num_heads=config.num_attention_heads, batch_first=True)

        #raise NotImplementedError("Build learned queries and attention.")

    def forward(self, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return query outputs and attention weights."""
        # Goal:
        # Let learned query tokens pull useful information from image patch features.
        #
        # Input:
        # - `image_features`: `[batch_size, num_patches, vision_hidden_size]`
        #
        # Output:
        # - query outputs: `[batch_size, num_query_tokens, query_hidden_size]`
        # - attention weights: useful for the heatmap
        #
        # Method:
        # Expand queries with:
        # `queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)`.
        # Call the attention layer like:
        # `outputs, weights = self.cross_attention(query=queries, key=image_features, value=image_features, need_weights=True)`.
        #
        # Steps:
        # 1. Read `batch_size = image_features.shape[0]`.
        # 2. Expand learned queries across the batch.
        # 3. Pass the expanded queries as `query`.
        # 4. Pass image patch features as both `key` and `value`.
        # 5. Return query outputs and attention weights.
        #
        # Check:
        # - Attention should have one value per image patch for each query.

        batch_size = image_features.shape[0]

        # image_features has shape [batch_size, 36, 64]
        # query_tokens has shape [8, 64]

        queries = self.query_tokens.unsqueeze(dim=0).expand(size=(batch_size, -1, -1))  # [batch_size, 8, 64]

        outputs, weights = self.cross_attention(query=queries, key=image_features, value=image_features, need_weights=True)

        # The query tokens are learned because they are `nn.Parameter`s.
        # That means gradient descent updates them during training, just like layer weights.
        # They start as small random vectors, then learn what kind of image details help the
        # caption-label task.
        #
        # Cross-attention uses the learned queries to look at the image patches:
        # - query = the learned query tokens, asking "what should I pull from the image?"
        # - key = image patch features, used to decide which patches match each query
        # - value = image patch features, used to build the returned query outputs
        #
        # outputs are the updated query vectors after they pulled information from patches.
        # Shape: [batch_size, 8, 64].
        # weights show how strongly each query looked at each image patch.
        # weights are the attention weights after softmax.
        # They are not the raw attention scores.
        # attention scores = raw match numbers before softmax
        # attention weights = normalized numbers after softmax
        #
        # Behind the scenes, attention is roughly:
        #
        # scores = queries @ image_features.transpose(-2, -1)       # [batch_size, 8, 64] @ [batch_size, 64, 36] = [batch_size, 8, 36]
        # weights = torch.softmax(scores, dim=-1)                   # [batch_size, 8, 36]
        # outputs = weights @ image_features                        # [batch_size, 8, 36] @ [batch_size, 36, 64] = [batch_size, 8, 64]
        #
        # In words:
        # - scores are raw match numbers between each query and each image patch
        # - softmax turns those scores into weights that sum to 1 across patches
        # - the weights mix the image patch vectors to make the final query outputs
        #
        # Small example for one query looking at 3 patches:
        # raw attention scores: [2.0, 1.0, 0.0]
        # attention weights after softmax: [0.67, 0.24, 0.09]
        #
        # This means:
        # - this query looked 67% at patch 0
        # - this query looked 24% at patch 1
        # - this query looked 9% at patch 2

        return outputs, weights     # [batch_size, 8, 64], [batch_size, 8, 36]
        #raise NotImplementedError("Let learned queries attend to image patch features.")


class TinyBLIP2Bridge(nn.Module):
    """Image encoder + Q-Former + small label head."""

    def __init__(self, num_labels: int, config: QFormerConfig):
        super().__init__()
        # Goal:
        # Build the image encoder, Q-Former, and label head.
        #
        # Layers:
        # - `PatchImageEncoder(config)`
        # - `TinyQFormer(config)`
        # - a linear head from query hidden size to `num_labels`
        #
        # Method:
        # Use the local classes `PatchImageEncoder(config)` and `TinyQFormer(config)`.
        # Use `nn.Linear(config.query_hidden_size, num_labels)` from `torch.nn`
        # for the label head.
        #
        # Steps:
        # 1. Store `self.image_encoder = PatchImageEncoder(config)`.
        # 2. Store `self.qformer = TinyQFormer(config)`.
        # 3. Store `self.label_head = nn.Linear(config.query_hidden_size, num_labels)`.
        # 4. Save `config` on `self` if needed.
        #
        # Check:
        # - The final logits should have shape `[batch_size, num_labels]`.

        self.config = config

        self.image_encoder = PatchImageEncoder(config)

        self.qformer = TinyQFormer(config)

        self.label_head = nn.Linear(config.query_hidden_size, num_labels)

        #raise NotImplementedError("Build the tiny BLIP-2 bridge model.")

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return label logits and attention weights."""
        # Goal:
        # Run image patches through the Q-Former and predict the caption label.
        #
        # Method:
        # Use `.mean(dim=1)` to average over query tokens.
        # Do not use `softmax` before `nn.CrossEntropyLoss`; return raw logits.
        #
        # Steps:
        # 1. Run `patch_features = self.image_encoder(images)`.
        # 2. Run `query_outputs, attention_weights = self.qformer(patch_features)`.
        # 3. Pool query outputs with `pooled = query_outputs.mean(dim=1)`.
        # 4. Compute `logits = self.label_head(pooled)`.
        # 5. Return `(logits, attention_weights)`.
        #
        # Check:
        # - Logits shape is `[batch_size, num_labels]`.

        #images shape is [B, 3, 48, 48]

        patch_features = self.image_encoder(images)     # [B, 36, 64]

        query_outputs, attention_weights = self.qformer(patch_features)     # [B, 8, 64], [b, 8, 36]

        pooled = query_outputs.mean(dim=1)          # [B, 64]

        logits = self.label_head(pooled)            # [B, num_labels]

        # In a real BLIP-2 style system, these learned query tokens are not the
        # user's text question.
        #
        # The user question is normal text, such as:
        # "What color is the car?"
        #
        # The learned query tokens are internal visual slots. They ask the image
        # patches what useful visual information should be pulled out.
        #
        # A real workflow is roughly:
        # image -> image encoder -> image patch features
        # learned query tokens -> Q-Former -> visual summary tokens
        # user question -> language model text input
        # visual summary tokens + user question -> answer
        #
        # In this toy exercise, there is no user question or language model.
        # The pooled query outputs go straight into a small label head.
        return logits, attention_weights

        #raise NotImplementedError("Run image encoder, Q-Former, and label head.")


def train_one_epoch(model: TinyBLIP2Bridge, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train one epoch."""
    # Goal:
    # Train the Q-Former bridge for one epoch.
    #
    # Method:
    # Use normal PyTorch calls:
    # `model.train()`, `nn.CrossEntropyLoss()`, `optimizer.zero_grad()`,
    # `loss.backward()`, and `optimizer.step()`.
    #
    # Steps:
    # 1. Create `loss_fn = nn.CrossEntropyLoss()`.
    # 2. Call `model.train()` before the loop.
    # 3. Move `batch["image"]` and `batch["label_id"]` to `device`.
    # 4. Run `logits, _ = model(images)`.
    # 5. Compute `loss = loss_fn(logits, label_ids)`.
    # 6. Clear old gradients, backprop, and step the optimizer.
    # 7. Add `loss.item()` to a running total.
    # 8. Return the mean loss as a Python float.
    #
    # Check:
    # - Return mean loss as a Python float.

    loss_fn = nn.CrossEntropyLoss()

    model.train()

    total_loss = 0
    total_examples = 0

    for data in loader:
        images = data["image"].to(device)
        label_ids = data["label_id"].to(device)

        optimizer.zero_grad()

        logits, _ = model(images)
        loss = loss_fn(logits, label_ids)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples
    #raise NotImplementedError("Train one Q-Former epoch.")

@torch.no_grad()
def evaluate_labels(model: TinyBLIP2Bridge, loader: DataLoader) -> float:
    """Return caption-label accuracy."""
    # Goal:
    # Measure how often the model predicts the correct small caption label.
    #
    # Method:
    # Use `model.eval()` and the PyTorch context manager `torch.no_grad()`.
    # Use `logits.argmax(dim=-1)` to get predicted class ids.
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Start `correct = 0` and `total = 0`.
    # 3. Loop over batches inside `with torch.no_grad():`.
    # 4. Move images and labels to `device`.
    # 5. Run `logits, _ = model(images)`.
    # 6. Get predictions with `logits.argmax(dim=-1)`.
    # 7. Count matches with `(predictions == label_ids).sum().item()`.
    # 8. Return `correct / total`.
    #
    # Check:
    # - Accuracy is a float between `0.0` and `1.0`.

    model.eval()

    total_correct = 0
    total_examples = 0

    for data in loader:
        images = data["image"].to(device)
        label_ids = data["label_id"].to(device)

        logits, _ = model(images)

        preds = logits.argmax(dim=-1)
        correct = (preds == label_ids).sum().item()
        

        batch_size = images.shape[0]
        total_correct += correct        
        total_examples += batch_size

    return total_correct / total_examples # examples

    #raise NotImplementedError("Evaluate label accuracy.")


def save_qformer_visuals(losses: list[float], model: TinyBLIP2Bridge, examples: list[CaptionExample], config: QFormerConfig) -> tuple[str, str]:
    """Save loss curve and attention heatmap."""
    # Goal:
    # Save the training curve and show where Q-Former queries attended.
    #
    # Method:
    # Use `config.output_dir.mkdir(parents=True, exist_ok=True)` from `pathlib.Path`.
    # Use Matplotlib: `matplotlib.use("Agg")`, `plt.figure()`, `plt.plot(...)`,
    # `plt.imshow(...)`, `plt.colorbar()`, `plt.savefig(path)`, and `plt.close()`.
    # Use `torch.no_grad()` when running the model for the heatmap.
    #
    # Steps:
    # 1. Create `config.output_dir`.
    # 2. Save `losses` as a line plot.
    # 3. Pick one example image and add a batch dimension with `.unsqueeze(0)`.
    # 4. Run the model inside `with torch.no_grad():`.
    # 5. Average attention over heads and query tokens if needed.
    # 6. Reshape the patch attention to `[patch_grid_size, patch_grid_size]`.
    # 7. Save the heatmap with `plt.imshow(...)`.
    # 8. Return both paths as strings.
    #
    # Check:
    # - The heatmap should have `patch_grid_size` by `patch_grid_size` cells.

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)

    loss_curve_path = config.output_dir / "qformer_loss_curve.png"
    attention_path = config.output_dir / "qformer_attention_heatmap.png"

    plt.figure()
    plt.plot(losses)
    plt.title("Q-Former training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(loss_curve_path)
    plt.close()

    model.eval()
    image = examples[0]["image"].unsqueeze(0).to(device)  # [3, 48, 48] -> [1, 3, 48, 48]

    with torch.no_grad():
        _, attention_weights = model(image)  # [1, 8, 36] for averaged heads

    if attention_weights.dim() == 4:
        patch_attention = attention_weights.mean(dim=(1, 2))[0]  # [1, heads, 8, 36] -> [36]
    else:
        patch_attention = attention_weights.mean(dim=1)[0]  # [1, 8, 36] -> [36]

    heatmap = patch_attention.reshape(config.patch_grid_size, config.patch_grid_size)  # [36] -> [6, 6]
    heatmap = heatmap.detach().cpu()

    plt.figure()
    plt.imshow(heatmap)
    plt.title("Average query attention over image patches")
    plt.colorbar()
    plt.savefig(attention_path)
    plt.close()

    return str(loss_curve_path), str(attention_path)
    #raise NotImplementedError("Save Q-Former visuals.")


def run_exercise(config: QFormerConfig) -> QFormerMetrics:
    """Run full Q-Former training."""
    # Main workflow:
    # image-caption labels -> loader -> image patches -> Q-Former -> train -> evaluate -> visualize.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_caption_dataset(config)
    num_labels = max(example["label_id"] for example in examples) + 1
    dataset = QFormerDataset(examples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TinyBLIP2Bridge(num_labels=num_labels, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    for i in range(config.epochs):
        loss = train_one_epoch(model, loader, optimizer)
        losses.append(loss)
        print(f"Epoch {i+1} -> loss: {loss}")

    label_accuracy = evaluate_labels(model, loader)
    loss_curve_path, attention_path = save_qformer_visuals(losses, model, examples, config)

    print(f"\n label_accuracy: {label_accuracy}")

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "label_accuracy": label_accuracy,
        "loss_curve_path": loss_curve_path,
        "attention_path": attention_path,
    }


def check_metrics(metrics: QFormerMetrics) -> bool:
    return metrics["final_loss"] < metrics["initial_loss"] and metrics["label_accuracy"] > 0.60


def report_task_progress(task_name: str, task_fn, *args):
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the Q-Former scaffold one step at a time."""
    config = QFormerConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 5

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    examples, ok = report_task_progress("build_caption_dataset", build_caption_dataset, config)
    if not count_or_stop(ok):
        return
    num_labels = max(example["label_id"] for example in examples) + 1

    _, ok = report_task_progress("PatchImageEncoder.__init__", PatchImageEncoder, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinyQFormer.__init__", TinyQFormer, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinyBLIP2Bridge.__init__", TinyBLIP2Bridge, num_labels, config)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "q-former label accuracy", metrics["label_accuracy"], "> 0.60 and loss improves")


if __name__ == "__main__":
    main()
