"""
============================================================
EXERCISE 04: VL-JEPA TEXT EMBEDDING PREDICTION
============================================================

WHAT YOU ARE BUILDING
You are building a small version of the main VL-JEPA paper idea.

VL-JEPA means Vision-Language Joint Embedding Predictive Architecture.

The key paper idea:
- do not train the model to write the next text token
- train it to predict a continuous text embedding instead
- a text embedding is a vector that stands for the meaning of text
- decode text only later if a task needs words

Plain meaning:
- the model sees an image
- the caption is turned into a target vector
- the predictor tries to make an image-based vector that matches that target vector
- the loss compares vectors, not pixels and not next-word tokens

WHY THIS EXERCISE EXISTS
Earlier exercises show image-text matching and CLIP-style contrastive training.
This file teaches the VL-JEPA shift:

```text
classic vision-language training: image -> generate text tokens
VL-JEPA-style training:          image -> predict text meaning vector
```

REAL SYSTEM FIRST
In the paper, the larger system:

1. reads visual input
2. predicts target text embeddings in a shared representation space
3. uses a lightweight text decoder only when text must be produced
4. can use the same embedding space for retrieval, classification, and VQA-style checks

This exercise covers the representation-learning part:

1. load real image-caption pairs
2. encode each image into visual tokens
3. encode each caption into a target text embedding
4. predict the target text embedding from the image tokens
5. train with an embedding-space loss
6. use a small retrieval loss so the tiny text space learns useful meaning
7. check image-to-caption retrieval inside a small batch

This exercise skips:
- video input, because short COCO image-caption batches are cheaper
- a large pretrained vision encoder
- a large pretrained text model
- the final text decoder
- VQA and open-vocabulary classification benchmarks

Why the retrieval loss is included:
- the paper can use strong text embeddings from a much larger setup
- this exercise starts with a tiny text encoder
- without one extra learning signal, the text vectors would start as random vectors
- the retrieval loss teaches matching image-caption pairs to be close
- the MSE loss still teaches the VL-JEPA idea: predict the caption embedding itself

WHAT IS SCALED DOWN AND WHY
This exercise should stay close to the paper's main idea, but it must run on
limited GPU memory.

Scaled down:
- COCO-30 sample instead of web-scale image/video-text data
- images instead of videos
- 128 by 128 images instead of high-resolution visual input
- a tiny ViT-style image encoder instead of a large vision backbone
- a tiny text transformer instead of a large text model
- one pooled caption embedding instead of many decoded text targets
- a short training run instead of large-scale pretraining
- a small in-batch retrieval loss instead of a large pretrained text embedding space

Not scaled down:
- real image-caption pairs
- real captions
- vision input predicts text embeddings
- target text embeddings are detached before the prediction loss
- loss happens in representation space
- retrieval-style evaluation checks whether the learned embedding space is useful

Mapping from the real system to this exercise:

    real VL-JEPA system                 this exercise
    ---------------------------------------------------------------
    large image/video-text dataset       COCO-30 image-caption sample
    visual input                         one real image
    large vision encoder                 tiny ViT-style image encoder
    target text embedding                tiny text-transformer caption embedding
    predictor                            tiny MLP over pooled image tokens
    representation-space loss            MSE to detached caption embedding
    learned semantic space               small in-batch retrieval loss
    optional text decoder                skipped to keep the exercise small
    downstream checks                    small image-to-caption retrieval check

INPUTS AND OUTPUTS
Each dataloader batch should contain:

    image:        [batch, 3, image_size, image_size]
    token_ids:    [batch, max_text_length]
    text_lengths: [batch]
    caption_text: Python strings, kept for visuals

With default settings:

    image_size = 128
    patch_size = 16
    patch grid = 8 by 8
    num_patches = 64

The image encoder outputs:

    image_tokens: [batch, 64, image_dim]

The target text encoder outputs:

    text_tokens:      [batch, max_text_length, text_dim]
    text_embedding:   [batch, text_dim]

The predictor outputs:

    predicted_text_embedding: [batch, text_dim]

LOSS FUNCTION
Use latent-space mean squared error for the VL-JEPA part:

    prediction_loss = nn.MSELoss()(
        predicted_text_embedding,
        target_text_embedding.detach(),
    )

Use a small image-caption retrieval loss to keep the tiny target text space useful:

    retrieval_loss = cross_entropy(predicted_to_target_scores, labels)

Use both together:

    loss = prediction_loss + contrastive_loss_weight * retrieval_loss

Plain meaning:
- the target text embedding is the answer for this update
- `.detach()` means gradients do not move the target through the MSE loss
- the predictor and image encoder learn to land near the caption meaning vector
- the retrieval loss keeps matching image-caption pairs close and mismatched pairs farther apart

METRICS TO TRACK
Track:
- train text embedding MSE, lower is better
- validation text embedding MSE, lower is better
- test text embedding MSE, lower is better
- validation cosine similarity, higher is better
- test cosine similarity, higher is better
- image-to-caption retrieval top-1 inside a batch, higher is better

VISUALS TO SAVE
Save to `outputs/vljepa_text_embeddings/`:
- real image-caption examples
- training curve
- image-to-caption similarity heatmap

STEP-BY-STEP TODO
1. Load COCO-30 from Hugging Face.
2. Build a small word vocabulary from training captions.
3. Tokenize captions into fixed-length token ids.
4. Build the tiny ViT-style image encoder.
5. Build the tiny target text transformer.
6. Build the image-to-text-embedding predictor.
7. Train on text embedding MSE plus the small retrieval loss.
8. Report held-out MSE, cosine similarity, and retrieval top-1.
9. Save visual checks.

DEFINITION OF DONE
You are done when:
- the exercise uses real image-caption pairs
- the image branch predicts text embeddings, not image patch embeddings
- the target caption embeddings are detached for the prediction loss
- validation and test metrics are printed
- image-to-caption retrieval top-1 is printed
- visual outputs are saved
- you can explain why VL-JEPA predicts text vectors instead of next tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed


# Learning order:
# 1. Load real COCO image-caption pairs.
# 2. Tokenize captions with a small training-set vocabulary.
# 3. Build the image encoder.
# 4. Build the target text encoder.
# 5. Build the predictor that maps image features to text embeddings.
# 6. Train on text embedding MSE plus a small retrieval loss.
# 7. Evaluate MSE, cosine similarity, and retrieval.
# 8. Save visuals.


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VLJEPAConfig:
    """Settings for the small VL-JEPA-style run."""

    hf_dataset_name: str = "sayakpaul/coco-30-val-2014"
    output_dir: str = str(BOOTCAMP_ROOT / "outputs" / "vljepa_text_embeddings")
    image_size: int = 128
    patch_size: int = 16
    image_channels: int = 3
    max_text_length: int = 16
    min_word_count: int = 1
    image_dim: int = 128
    text_dim: int = 128
    image_encoder_depth: int = 2
    image_encoder_num_heads: int = 4
    image_encoder_mlp_dim: int = 256
    text_encoder_depth: int = 1
    text_encoder_num_heads: int = 4
    text_encoder_mlp_dim: int = 256
    predictor_hidden_dim: int = 256
    contrastive_loss_weight: float = 0.1
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 12
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    random_state: int = 303

    def __post_init__(self) -> None:
        """Catch config values that would break the tiny transformers."""
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must divide evenly by patch_size.")
        if self.image_dim % self.image_encoder_num_heads != 0:
            raise ValueError("image_dim must divide evenly by image_encoder_num_heads.")
        if self.text_dim % self.text_encoder_num_heads != 0:
            raise ValueError("text_dim must divide evenly by text_encoder_num_heads.")
        if self.train_fraction <= 0 or self.val_fraction <= 0:
            raise ValueError("train_fraction and val_fraction must be positive.")
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("train_fraction plus val_fraction must leave room for test data.")

    @property
    def patches_per_side(self) -> int:
        """Return how many patches fit along one image side."""
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Return the total number of image patches."""
        return self.patches_per_side * self.patches_per_side


class COCOCaptionEmbeddingDataset(Dataset):
    """Wrap real image-caption rows for text embedding prediction."""

    def __init__(
        self,
        rows: list[dict[str, object]],
        vocab: dict[str, int],
        image_transform,
        max_text_length: int,
    ) -> None:
        self.rows = rows
        self.vocab = vocab
        self.image_transform = image_transform
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = row["image"].convert("RGB")
        image = self.image_transform(image)
        caption = str(row["caption"])
        token_ids, text_length = tokenize_caption(caption, self.vocab, self.max_text_length)
        return {
            "image": image,
            "token_ids": token_ids,
            "text_lengths": text_length,
            "caption_text": caption,
        }


def simple_word_tokenize(text: str) -> list[str]:
    """Split a caption into simple lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def tokenize_caption(
    caption: str,
    vocab: dict[str, int],
    max_text_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn one caption into padded token ids and a length tensor."""
    words = simple_word_tokenize(caption)[:max_text_length]
    ids = [vocab.get(word, 1) for word in words]
    length = len(ids)
    ids = ids + [0] * (max_text_length - len(ids))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(length, dtype=torch.long)


def load_coco30_vljepa_dataloaders(config: VLJEPAConfig) -> dict[str, object]:
    """Load real COCO-30 image-caption pairs and return dataloaders plus vocab."""
    # Where this fits:
    # This builds the real paired image-text data.
    #
    # Goal:
    # Return a dictionary with:
    # - `train`, `val`, and `test` DataLoaders
    # - `vocab`, the word-to-id dictionary
    #
    # Method:
    # Use Hugging Face datasets and torchvision transforms:
    # - `from datasets import load_dataset`
    # - `from torchvision import transforms`
    # - `dataset = load_dataset(config.hf_dataset_name, split="train")`
    #
    # Steps:
    # 1. Load COCO-30.
    # 2. Keep rows that have an image and a caption.
    # 3. Split rows into train, validation, and test.
    # 4. Build the vocabulary from training captions.
    # 5. Build one transform with:
    #    `transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])`.
    # 6. Wrap each split with `COCOCaptionEmbeddingDataset`.
    # 7. Build DataLoaders.
    #
    # Checks:
    # - captions should come from real COCO rows.
    # - token ids should be fixed length.
    # - if loading fails, do not replace the data with fake pairs.
    raise NotImplementedError("Load real COCO-30 image-caption dataloaders.")


class TinyViTImageEncoder(nn.Module):
    """Tiny ViT-style image encoder that returns one latent vector per patch."""

    def __init__(self, config: VLJEPAConfig) -> None:
        super().__init__()
        # Where this fits:
        # In the paper, visual input is encoded before predicting text embeddings.
        # This tiny encoder is the small version of that visual branch.
        #
        # Goal:
        # Turn `[batch, 3, 128, 128]` images into `[batch, 64, image_dim]`
        # patch tokens.
        #
        # Method:
        # Build these parts:
        # - `self.num_patches = config.num_patches`
        # - `self.image_dim = config.image_dim`
        # - `self.patch_embed = nn.Conv2d(config.image_channels, config.image_dim, kernel_size=config.patch_size, stride=config.patch_size)`
        # - `self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.image_dim))`
        # - `encoder_layer = nn.TransformerEncoderLayer(d_model=config.image_dim, nhead=config.image_encoder_num_heads, dim_feedforward=config.image_encoder_mlp_dim, dropout=0.0, activation="gelu", batch_first=True)`
        # - `self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.image_encoder_depth)`
        # - `self.norm = nn.LayerNorm(config.image_dim)`
        #
        # Checks:
        # - `TinyViTImageEncoder(config)(torch.randn(2, 3, 128, 128)).shape`
        #   should be `[2, 64, config.image_dim]`.
        raise NotImplementedError("Build the tiny VL-JEPA image encoder.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into patch tokens."""
        # Goal:
        # Run the tiny image encoder.
        #
        # Input:
        # - `images`: `[batch, 3, 128, 128]`
        #
        # Output:
        # - `image_tokens`: `[batch, 64, image_dim]`
        #
        # Steps:
        # 1. Run `patch_tokens = self.patch_embed(images)`.
        #    Shape: `[batch, image_dim, 8, 8]`.
        # 2. Flatten the patch grid with:
        #    `patch_tokens = patch_tokens.flatten(2).transpose(1, 2)`.
        # 3. Add `self.pos_embed`.
        # 4. Run `self.transformer(patch_tokens)`.
        # 5. Return `self.norm(patch_tokens)`.
        raise NotImplementedError("Run the tiny VL-JEPA image encoder.")


class TinyTargetTextTransformer(nn.Module):
    """Tiny text encoder that provides target caption embeddings."""

    def __init__(self, vocab_size: int, config: VLJEPAConfig) -> None:
        super().__init__()
        # Where this fits:
        # The paper trains the visual side to predict target text embeddings.
        # This small text transformer creates those target embeddings.
        #
        # Goal:
        # Turn token ids into text token vectors shaped
        # `[batch, max_text_length, text_dim]`.
        #
        # Method:
        # Build:
        # - `self.token_embed = nn.Embedding(vocab_size, config.text_dim, padding_idx=0)`
        # - `self.pos_embed = nn.Parameter(torch.zeros(1, config.max_text_length, config.text_dim))`
        # - `encoder_layer = nn.TransformerEncoderLayer(d_model=config.text_dim, nhead=config.text_encoder_num_heads, dim_feedforward=config.text_encoder_mlp_dim, dropout=0.0, activation="gelu", batch_first=True)`
        # - `self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.text_encoder_depth)`
        # - `self.norm = nn.LayerNorm(config.text_dim)`
        raise NotImplementedError("Build the tiny target text transformer.")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode captions into text token vectors."""
        # Steps:
        # 1. Build `padding_mask = token_ids == 0`.
        # 2. Embed token ids with `self.token_embed(token_ids)`.
        # 3. Add `self.pos_embed[:, : token_ids.shape[1]]`.
        # 4. Run the transformer with `src_key_padding_mask=padding_mask`.
        # 5. Return `self.norm(text_tokens)`.
        raise NotImplementedError("Run the tiny target text transformer.")


def pool_text_embeddings(text_tokens: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Average non-padding text token vectors into one caption embedding."""
    # Where this fits:
    # The paper's target is a text embedding. This helper turns many token
    # vectors into one caption vector for the small exercise.
    #
    # Goal:
    # Convert:
    # - `text_tokens`: `[batch, max_text_length, text_dim]`
    # - `token_ids`: `[batch, max_text_length]`
    #
    # Into:
    # - `caption_embedding`: `[batch, text_dim]`
    #
    # Steps:
    # 1. Build `non_padding = token_ids != 0`.
    # 2. Convert it to float and unsqueeze the last dimension.
    # 3. Compute the weighted sum across text positions.
    # 4. Divide by the number of real tokens, using `.clamp_min(1.0)`.
    # 5. Normalize with `F.normalize(caption_embedding, dim=-1)`.
    raise NotImplementedError("Pool non-padding text tokens into caption embeddings.")


class ImageToTextEmbeddingPredictor(nn.Module):
    """Tiny predictor that maps image tokens to one predicted text embedding."""

    def __init__(self, config: VLJEPAConfig) -> None:
        super().__init__()
        # Where this fits:
        # This is the VL-JEPA-style predictor. It predicts a text meaning vector
        # from visual tokens.
        #
        # Goal:
        # Build a predictor that returns `[batch, text_dim]`.
        #
        # Method:
        # Build these parts:
        # - `self.image_pool_norm = nn.LayerNorm(config.image_dim)`
        # - `self.net = nn.Sequential(nn.Linear(config.image_dim, config.predictor_hidden_dim), nn.GELU(), nn.Linear(config.predictor_hidden_dim, config.text_dim))`
        #
        # Checks:
        # - input image tokens should be `[batch, 64, image_dim]`.
        # - output should be `[batch, text_dim]`.
        raise NotImplementedError("Build the image-to-text embedding predictor.")

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """Predict one caption embedding from image tokens."""
        # Steps:
        # 1. Average image tokens across patches:
        #    `pooled_image = image_tokens.mean(dim=1)`.
        # 2. Apply `self.image_pool_norm(pooled_image)`.
        # 3. Run `self.net(pooled_image)`.
        # 4. Normalize with `F.normalize(predicted_embedding, dim=-1)`.
        raise NotImplementedError("Predict text embeddings from image tokens.")


def build_vljepa_modules(
    config: VLJEPAConfig,
    vocab_size: int,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Build image encoder, target text encoder, and predictor."""
    # Goal:
    # Return three modules:
    # - image encoder: trained by gradients
    # - target text encoder: creates caption target embeddings
    # - predictor: trained by gradients
    #
    # Important:
    # The target text embedding is detached in the loss.
    # That is the small-exercise version of treating text embeddings as the
    # prediction target.
    #
    # Steps:
    # 1. Create `image_encoder = TinyViTImageEncoder(config)`.
    # 2. Create `target_text_encoder = TinyTargetTextTransformer(vocab_size, config)`.
    # 3. Create `predictor = ImageToTextEmbeddingPredictor(config)`.
    # 4. Move all modules to `device`.
    # 5. Return them.
    raise NotImplementedError("Build the VL-JEPA text embedding modules.")


def compute_vljepa_text_embedding_loss(
    predicted_text_embedding: torch.Tensor,
    target_text_embedding: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """Compute the representation-space prediction loss."""
    # Goal:
    # Compare the predicted caption embedding with the target caption embedding.
    #
    # Input:
    # - `predicted_text_embedding`: `[batch, text_dim]`
    # - `target_text_embedding`: `[batch, text_dim]`
    #
    # Method:
    # Use:
    # `loss = loss_fn(predicted_text_embedding, target_text_embedding.detach())`
    #
    # Checks:
    # - both inputs should have the same shape.
    # - do not use cross entropy here because this is not next-token training.
    raise NotImplementedError("Compute VL-JEPA text embedding MSE loss.")


def compute_image_caption_retrieval_loss(
    predicted_text_embedding: torch.Tensor,
    target_text_embedding: torch.Tensor,
) -> torch.Tensor:
    """Compute a small in-batch image-caption retrieval loss."""
    # Where this fits:
    # The real system has a much stronger representation space than this tiny
    # exercise starts with. This helper gives the small text space a useful
    # learning signal.
    #
    # Goal:
    # Make predicted image-based text embeddings match the caption embeddings
    # from the same row.
    #
    # Input:
    # - `predicted_text_embedding`: `[batch, text_dim]`
    # - `target_text_embedding`: `[batch, text_dim]`
    #
    # Method:
    # 1. Normalize both tensors with `F.normalize(..., dim=-1)`.
    # 2. Compute scores with `similarity = predicted @ target.T`.
    # 3. Build labels with `torch.arange(batch_size, device=similarity.device)`.
    # 4. Use `F.cross_entropy(similarity, labels)`.
    #
    # Check:
    # - row `i` should learn to pick column `i`.
    raise NotImplementedError("Compute the small image-caption retrieval loss.")


def retrieval_top1(
    predicted_text_embedding: torch.Tensor,
    target_text_embedding: torch.Tensor,
) -> float:
    """Return image-to-caption retrieval top-1 inside one batch."""
    # Goal:
    # Check whether image `i` retrieves caption `i`.
    #
    # Method:
    # 1. Normalize predicted vectors with `F.normalize(..., dim=-1)`.
    # 2. Normalize target text vectors.
    # 3. Compute `similarity = predicted @ target.T`.
    # 4. The correct caption for row `i` is column `i`.
    # 5. Return the fraction of rows where `argmax` chooses the correct column.
    raise NotImplementedError("Compute image-to-caption retrieval top-1.")


def train_one_epoch(
    image_encoder: nn.Module,
    target_text_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: VLJEPAConfig,
) -> dict[str, float]:
    """Train VL-JEPA-style text embedding prediction for one epoch."""
    # Steps:
    # 1. Put all modules in train mode.
    # 2. Move images, token ids, and text lengths to `device`.
    # 3. Encode images with `image_encoder(images)`.
    # 4. Encode captions with `target_text_encoder(token_ids)`.
    # 5. Pool caption tokens with `pool_text_embeddings(text_tokens, token_ids)`.
    # 6. Predict caption embeddings with `predictor(image_tokens)`.
    # 7. Compute MSE against `target_text_embedding.detach()`.
    # 8. Compute retrieval loss with `compute_image_caption_retrieval_loss(...)`.
    # 9. Combine them:
    #    `loss = prediction_loss + config.contrastive_loss_weight * retrieval_loss`.
    # 10. Backprop and step the optimizer.
    # 11. Track MSE, cosine similarity, and retrieval top-1.
    raise NotImplementedError("Train VL-JEPA text embedding prediction for one epoch.")


@torch.no_grad()
def evaluate(
    image_encoder: nn.Module,
    target_text_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    config: VLJEPAConfig,
) -> dict[str, float]:
    """Evaluate text embedding prediction and retrieval top-1."""
    # Steps:
    # 1. Put all modules in eval mode.
    # 2. Move images and token ids to `device`.
    # 3. Encode images with `image_encoder(images)`.
    # 4. Encode captions with `target_text_encoder(token_ids)`.
    # 5. Pool caption tokens into target text embeddings.
    # 6. Predict caption embeddings from image tokens.
    # 7. Track average MSE and cosine similarity.
    # 8. Compute image-to-caption retrieval top-1 inside each batch.
    raise NotImplementedError("Evaluate VL-JEPA text embedding metrics.")


def save_vljepa_visuals(
    history: list[dict[str, float]],
    sample_batch: dict[str, object],
    image_encoder: nn.Module,
    target_text_encoder: nn.Module,
    predictor: nn.Module,
    config: VLJEPAConfig,
) -> dict[str, str]:
    """Save image-caption examples, curves, and a retrieval heatmap."""
    # Save:
    # - a grid of real images with short captions
    # - train and validation text embedding MSE curves
    # - an image-to-caption similarity heatmap for one batch
    #
    # Checks:
    # - output paths should point inside `outputs/vljepa_text_embeddings/`.
    # - at least one visual should show the actual caption next to its image.
    raise NotImplementedError("Save VL-JEPA text embedding visuals.")


def run_vljepa_experiment(config: VLJEPAConfig) -> dict[str, object]:
    """Run the full VL-JEPA-style guided experiment."""
    # Steps:
    # 1. Load data with `load_coco30_vljepa_dataloaders(config)`.
    # 2. Read `vocab` from the returned dictionary.
    # 3. Build modules with `build_vljepa_modules(config, len(vocab))`.
    # 4. Create an optimizer for image encoder, target text encoder, and predictor.
    # 5. Train for `config.epochs`.
    # 6. Evaluate on validation and test loaders.
    # 7. Save visuals.
    #
    # Return:
    # - `train_text_embedding_mse`
    # - `val_text_embedding_mse`
    # - `test_text_embedding_mse`
    # - `val_cosine_similarity`
    # - `test_cosine_similarity`
    # - `val_retrieval_top1`
    # - `test_retrieval_top1`
    # - `visual_paths`
    raise NotImplementedError("Run the full VL-JEPA text embedding experiment.")


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
    """Run the VL-JEPA scaffold."""
    config = VLJEPAConfig()
    set_seed(config.random_state)
    print("exercise 04 predicts caption embeddings from images, following the main VL-JEPA idea.")
    result, ok = report_task_progress("run_vljepa_experiment", run_vljepa_experiment, config)
    if ok:
        print(result)


if __name__ == "__main__":
    main()
