"""
============================================================
EXERCISE 05: CLIP CONTRASTIVE TRAINING ON IMAGES
============================================================

WHAT YOU ARE BUILDING
You are building a tiny CLIP-style model with real image-caption examples.

CLIP means:
- one image encoder turns an image into a vector
- one text encoder turns a caption into a vector
- matching image-caption pairs should be close
- wrong image-caption pairs should be farther apart

Dataset path:
- use a small sample from `sayakpaul/coco-30-val-2014` on Hugging Face
- each row has a real image and one human-written caption

WHY THIS EXERCISE EXISTS
CLIP is one of the basic ideas behind modern vision-language models.
It is useful because text can act like flexible labels.

This exercise teaches the real workflow:
1. build image-caption training data
2. encode images
3. encode text
4. compute image-text scores
5. train with contrastive loss
6. visualize retrieval before and after training

HOW THIS FITS INTO THE INTERVIEW PREP
This is the starting point for VLM work.
If you can explain this exercise, you can explain how image-text matching works
before moving to chat-style VLMs such as LLaVA, BLIP-2, and Flamingo.

REAL SYSTEM VS THIS EXERCISE

real system                        this exercise
---------------------------------------------------------------
internet-scale image-caption data   small COCO-30 sample
large image encoder (ResNet-50 or ViT-B/16)    small CNN image encoder
large text encoder (Transformer text encoder)  small bag-of-words text encoder
large batch contrastive training    small batch contrastive training
image-text retrieval                small retrieval grid

DATASET
Use COCO-30 only:
- `sayakpaul/coco-30-val-2014`
- load it with `datasets.load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`
- each row should have an image and a caption

INPUTS AND OUTPUTS
Image batch:
- `images`: `[batch_size, 3, image_size, image_size]`

Text batch:
- `caption_token_ids`: `[batch_size, max_text_length]`

Model outputs:
- `image_embeddings`: `[batch_size, embedding_dim]`
- `text_embeddings`: `[batch_size, embedding_dim]`
- `similarity_scores`: `[batch_size, batch_size]`

LOSS FUNCTION
Use CLIP-style contrastive loss:

```text
image_to_text_loss = cross_entropy(similarity_scores, labels)
text_to_image_loss = cross_entropy(similarity_scores.T, labels)
loss = (image_to_text_loss + text_to_image_loss) / 2
```

where:
- `labels = [0, 1, 2, ..., batch_size - 1]`
- row `i` should match column `i`

VISUALS TO CREATE
Save plots in `outputs/multimodal_interview/`.

Create:
- a grid of real training images and captions
- a loss curve
- a retrieval score heatmap after training

METRICS TO TRACK
- training loss
- image-to-text retrieval accuracy
- text-to-image retrieval accuracy if you add the extra check
- average score for correct pairs
- average score for wrong pairs

STEP-BY-STEP TODO
1. Build real image-caption examples from COCO-30.
2. Build a tiny caption vocabulary.
3. Tokenize captions into padded ids.
4. Build a small image encoder.
5. Build a small text encoder.
6. Compute the image-text similarity matrix.
7. Compute CLIP contrastive loss.
8. Train for a few epochs.
9. Evaluate retrieval.
10. Save the loss curve and retrieval heatmap.

DEFINITION OF DONE
You are done when:
- the model trains on image-caption pairs
- loss goes down
- correct image-caption pairs score higher than wrong pairs
- at least one visualization is saved
- you can explain CLIP in plain words
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
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
class CLIPTrainingConfig:
    """Settings for tiny CLIP training."""

    image_size: int = 48
    embedding_dim: int = 64
    max_text_length: int = 32
    batch_size: int = 16
    epochs: int = 60
    learning_rate: float = 3e-3
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    hf_dataset_name: str = "sayakpaul/coco-30-val-2014"
    sample_size: int = 128


class ImageCaptionExample(TypedDict):
    """One image-caption example."""

    image: torch.Tensor
    caption: str


class CLIPMetrics(TypedDict):
    """Training and retrieval summary."""

    initial_loss: float
    final_loss: float
    retrieval_accuracy: float
    loss_curve_path: str
    heatmap_path: str


# Learning order:
# 1. Build real image-caption data.
# 2. Build the tokenizer.
# 3. Build the image and text encoders.
# 4. Compute the similarity matrix.
# 5. Compute CLIP loss.
# 6. Train and visualize retrieval.


def build_real_image_caption_dataset(config: CLIPTrainingConfig) -> list[ImageCaptionExample]:
    """Load real image-caption examples."""
    # Where this fits:
    # This helper loads the image-caption examples used by the CLIP workflow.
    #
    # Goal:
    # Return image tensors and matching captions.
    #
    # Input:
    # - `config.hf_dataset_name`: `"sayakpaul/coco-30-val-2014"`
    # - `config.sample_size`: how many rows to load
    #
    # Output:
    # - `list[ImageCaptionExample]`
    # - Each example has:
    #   - `"image"`: torch.Tensor with shape `[3, image_size, image_size]`
    #   - `"caption"`: a string matched to the image
    #
    # Method:
    # 1. Import `load_dataset` from `datasets`.
    # 2. Call:
    #    `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`
    # 3. Each row has:
    #    - `row["image"]`: a PIL image
    #    - `row["caption"]`: one caption string
    # 4. Use:
    #    `caption = row["caption"]`
    # 5. Resize the PIL image to `(config.image_size, config.image_size)`.
    # 6. Convert it to RGB.
    # 7. Convert it to a tensor in `[0, 1]`.
    # Checks:
    # - `examples[0]["image"].shape == (3, config.image_size, config.image_size)`
    # - `examples[0]["caption"]` is not empty.

    raise NotImplementedError("Load real image-caption examples from COCO-30.")







def build_vocabulary(captions: list[str]) -> dict[str, int]:
    """Build a small text vocabulary from captions."""
    # Goal:
    # Build a word-to-id dictionary for the captions.
    #
    # Input:
    # - captions such as:
    #   - `"A child in a pink dress is climbing stairs."`
    #   - `"a photo of a truck"`
    #
    # Output:
    # - dictionary with `<pad>` at id 0, `<unk>` at id 1, and normal words after it.
    # - Example:
    #   `{"<pad>": 0, "<unk>": 1, "a": 2, "cat": 3, "photo": 4}`
    #
    # Method:
    # Use a tiny whitespace tokenizer:
    # `words = caption.lower().split()`
    #
    # Steps:
    # 1. Start with `token_to_id = {"<pad>": 0, "<unk>": 1}`.
    # 2. Lowercase each caption.
    # 3. Split each caption on spaces.
    # 4. Collect all words in a set.
    # 5. Sort the set so ids are stable every time the script runs.
    # 6. Add each word that is not already in `token_to_id`.
    # 7. Return the dictionary.
    #
    # Checks:
    # - `token_to_id["<pad>"] == 0`
    # - `token_to_id["<unk>"] == 1`
    # - If one caption is `"a photo of a cat"`, `"cat"` should be in the dictionary.   
    
    raise NotImplementedError("Build caption vocabulary.")





def tokenize_captions(captions: list[str], token_to_id: dict[str, int], config: CLIPTrainingConfig) -> torch.Tensor:
    """Turn captions into padded token ids."""
    # Goal:
    # Convert a list of captions to a tensor of token ids.
    #
    # Input:
    # - `captions`: list of strings.
    #   Example: `["a photo of a cat", "a photo of a truck"]`
    # - `token_to_id`: dictionary from `build_vocabulary`.
    #
    # Output:
    # - tensor shaped `[num_captions, config.max_text_length]`
    # - dtype should be `torch.long`
    #
    # Method:
    # Use simple whitespace tokenization.
    # Pad shorter captions with id 0.
    # Use `token_to_id.get(word, token_to_id["<unk>"])` for unknown words.
    #
    # Exact tensor creation:
    # `token_ids = torch.full((len(captions), config.max_text_length), 0, dtype=torch.long)`
    #
    # Steps:
    # 1. Create `token_ids` filled with pad id `0`.
    # 2. For each caption, lowercase it and split it into words.
    # 3. Convert words to ids.
    # 4. Keep at most `config.max_text_length` ids.
    # 5. Copy those ids into the matching row of `token_ids`.
    # 6. Return `token_ids`.
    #
    # Checks:
    # - `token_ids.shape == (len(captions), config.max_text_length)`
    # - `token_ids.dtype == torch.long`
    # - Padding positions should stay at `0`.

    raise NotImplementedError("Tokenize captions into a padded tensor.")





class ShapeCaptionDataset(Dataset):
    """Dataset for tiny CLIP training."""

    def __init__(self, examples: list[ImageCaptionExample], token_to_id: dict[str, int], config: CLIPTrainingConfig):
        # Goal:
        # Store the data and tokenizer objects that `__getitem__` will need.
        #
        # Input:
        # - `examples`: list of dictionaries from `build_real_image_caption_dataset`.
        #   Each one has:
        #   - `"image"`: tensor shaped `[3, image_size, image_size]`
        #   - `"caption"`: string
        # - `token_to_id`: word-to-id dictionary from `build_vocabulary`.
        # - `config`: holds `max_text_length`.
        #
        # Steps:
        # 1. Save `examples` on `self`.
        # 2. Save `token_to_id` on `self`.
        # 3. Save `config` on `self`.
        #
        # Check:
        # - After this method runs, `len(self.examples)` should equal `len(examples)`.

        raise NotImplementedError("Store examples, vocabulary, and config.")


    def __len__(self) -> int:
        # Goal:
        # Tell the DataLoader how many examples are in the dataset.
        #
        # Output:
        # - integer number of examples.
        #
        # Method:
        # Return `len(self.examples)`.

        raise NotImplementedError("Return dataset length.")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        # Goal:
        # Return one image-caption training item.
        #
        # Input:
        # - `index`: integer position in `self.examples`.
        #
        # Output:
        # Return a dictionary with:
        # - `"image"`: image tensor shaped `[3, image_size, image_size]`
        # - `"caption_token_ids"`: token ids shaped `[max_text_length]`
        # - `"caption"`: original caption string
        #
        # Method:
        # 1. Read one example with `example = self.examples[index]`.
        # 2. Tokenize its caption by calling:
        #    `tokenize_captions([example["caption"]], self.token_to_id, self.config)`.
        # 3. The tokenizer returns shape `[1, max_text_length]`.
        # 4. Take `[0]` to get shape `[max_text_length]`.
        # 5. Return the image, token ids, and original caption.
        #
        # Check:
        # - `item["caption_token_ids"].dtype` should be `torch.long`.

        raise NotImplementedError("Return one tokenized image-caption item.")





class TinyImageEncoder(nn.Module):
    """Small CNN image encoder."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        # Goal:
        # Build a small image encoder that maps `[B, 3, H, W]` to `[B, embedding_dim]`.
        #
        # Suggested layers:
        # - Conv2d(3, 16, kernel_size=3, padding=1)
        # - ReLU
        # - MaxPool2d(2)
        # - Conv2d(16, 32, kernel_size=3, padding=1)
        # - ReLU
        # - AdaptiveAvgPool2d((1, 1))
        # - Linear(32, embedding_dim)

        raise NotImplementedError("Build the image encoder layers.")
        





    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized image embeddings."""
        # Goal:
        # Turn each image into one vector that can be compared with a text vector.
        #
        # Input:
        # - `images`: tensor shaped `[batch_size, 3, image_size, image_size]`
        #
        # Output:
        # - tensor shaped `[batch_size, embedding_dim]`
        #
        # Method:
        # 1. Run the CNN layers you built in `__init__`.
        # 2. Flatten the pooled feature map if needed.
        # 3. Run the final linear layer.
        # 4. Normalize the vector with:
        #    `torch.nn.functional.normalize(image_embeddings, dim=-1)`
        #
        # Check:
        # - `image_embeddings.norm(dim=-1)` should be close to `1`.

        raise NotImplementedError("Encode images and normalize the output vectors.")

        



class TinyTextEncoder(nn.Module):
    """Small text encoder for captions."""

    def __init__(self, vocab_size: int, embedding_dim: int, pad_token_id: int = 0):
        super().__init__()
        # Goal:
        # Build a small text encoder that maps `[B, T]` token ids to `[B, embedding_dim]`.
        #
        # Suggested method:
        # - use `nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)`
        # - average non-padding token embeddings
        # - normalize the final vector

        raise NotImplementedError("Build the text encoder layers.")


    def forward(self, caption_token_ids: torch.Tensor) -> torch.Tensor:
        """Return normalized text embeddings."""
        # Goal:
        # Turn each caption into one vector that can be compared with an image vector.
        #
        # Input:
        # - `caption_token_ids`: tensor shaped `[batch_size, max_text_length]`
        # - pad token id is `0`
        #
        # Output:
        # - tensor shaped `[batch_size, embedding_dim]`
        #
        # Method:
        # 1. Look up token vectors with `self.embedding(caption_token_ids)`.
        # 2. Build a mask with `caption_token_ids != 0`.
        # 3. Multiply embeddings by the mask so padding does not count.
        # 4. Sum token embeddings over the text length dimension.
        # 5. Divide by the number of non-padding tokens.
        # 6. Normalize with `torch.nn.functional.normalize(text_embeddings, dim=-1)`.
        #
        # Check:
        # - Clamp the token count with `.clamp(min=1)` so empty captions do not create NaNs.

        raise NotImplementedError("Encode captions and normalize the output vectors.")










class TinyCLIP(nn.Module):
    """Tiny CLIP-style model."""

    def __init__(self, vocab_size: int, config: CLIPTrainingConfig):
        super().__init__()
        self.image_encoder = TinyImageEncoder(config.embedding_dim)
        self.text_encoder = TinyTextEncoder(vocab_size, config.embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor, caption_token_ids: torch.Tensor) -> torch.Tensor:
        """Return image-text similarity scores with shape `[B, B]`."""
        # Goal:
        # Compare every image in the batch with every caption in the batch.
        #
        # Input:
        # - `images`: `[batch_size, 3, image_size, image_size]`
        # - `caption_token_ids`: `[batch_size, max_text_length]`
        #
        # Output:
        # - `similarity_scores`: `[batch_size, batch_size]`
        # - row `i` is image `i`
        # - column `j` is caption `j`
        #
        # Method:
        # 1. Run `self.image_encoder(images)`.
        # 2. Run `self.text_encoder(caption_token_ids)`.
        # 3. Compute scores with:
        #    `similarity_scores = image_embeddings @ text_embeddings.T`
        # 4. Multiply by `self.logit_scale.exp()`.
        #
        # Check:
        # - If batch size is 8, the output shape should be `[8, 8]`.

        raise NotImplementedError("Compute normalized image-text similarity scores.")



        # This scale is the inverse temperature used before cross entropy.
        # It is learned, so the model can choose how sharp the matching scores should be.
        # `.exp()` means e raised to this value, so the real scale is e^logit_scale.
        # Example: logit_scale=0.0 gives e^0.0 = 1.0.
        # Example: logit_scale=2.0 gives e^2.0 = about 7.4.
        # Larger scale means lower temperature, so the softmax becomes more confident.
        # Smaller scale means higher temperature, so the softmax becomes softer.



def compute_clip_loss(similarity_scores: torch.Tensor) -> torch.Tensor:
    """Compute CLIP contrastive loss."""
    # Goal:
    # Make row i match column i.
    #
    # Input:
    # - `similarity_scores`: `[batch_size, batch_size]`
    #
    # Output:
    # - scalar tensor loss
    #
    # Method:
    # - `labels = torch.arange(batch_size, device=similarity_scores.device)`
    # - image-to-text loss uses `similarity_scores`
    # - text-to-image loss uses `similarity_scores.T`
    # - average the two losses

    raise NotImplementedError("Compute CLIP image-to-text and text-to-image loss.")





def train_one_epoch(model: TinyCLIP, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train one epoch and return mean loss."""
    # Goal:
    # Run one training pass over all batches and return the mean loss.
    #
    # Input:
    # - `loader` returns dictionaries with:
    #   - `"image"`: `[batch_size, 3, image_size, image_size]`
    #   - `"caption_token_ids"`: `[batch_size, max_text_length]`
    #
    # Output:
    # - Python float: mean loss for the epoch.
    #
    # Method:
    # Use the normal PyTorch training order:
    # 1. Call `model.train()`.
    # 2. Move images and caption ids to `device`.
    # 3. Run `similarity_scores = model(images, caption_token_ids)`.
    # 4. Run `loss = compute_clip_loss(similarity_scores)`.
    # 5. Clear old gradients with `optimizer.zero_grad()`.
    # 6. Run `loss.backward()`.
    # 7. Run `optimizer.step()`.
    # 8. Add `loss.item() * batch_size` to a running total.
    #
    # Check:
    # - Return `total_loss / total_examples`, not just the final batch loss.

    raise NotImplementedError("Train one CLIP epoch on image-caption batches.")






@torch.no_grad()
def evaluate_retrieval(model: TinyCLIP, examples: list[ImageCaptionExample], token_to_id: dict[str, int], config: CLIPTrainingConfig) -> float:
    """Return image-to-text retrieval accuracy."""
    # Goal:
    # Check whether each image retrieves its matching caption.
    #
    # Input:
    # - `examples`: image-caption pairs in matching order.
    # - The correct caption for image `i` is caption `i`.
    #
    # Output:
    # - accuracy as a float between `0.0` and `1.0`.
    #
    # Method:
    # 1. Call `model.eval()`.
    # 2. Use `torch.no_grad()`.
    # 3. Stack example images into `[num_examples, 3, image_size, image_size]`.
    # 4. Tokenize all captions.
    # 5. Run the model to get a `[num_examples, num_examples]` score matrix.
    # 6. Use `scores.argmax(dim=1)` to pick the best caption for each image.
    # 7. Compare with `torch.arange(num_examples, device=scores.device)`.
    #
    # Check:
    # - Random retrieval over 16 examples is about `1 / 16 = 0.0625`.
    # - A trained tiny model should do better than random on the small set.

    raise NotImplementedError("Evaluate whether each image retrieves its matching caption.")





    




def save_training_visuals(
    losses: list[float],
    model: TinyCLIP,
    examples: list[ImageCaptionExample],
    token_to_id: dict[str, int],
    config: CLIPTrainingConfig,
) -> tuple[str, str]:
    """Save a loss curve and retrieval heatmap."""
    # Goal:
    # Save visual proof that training changed model behavior.
    #
    # Create:
    # - `clip_loss_curve.png`
    # - `clip_retrieval_heatmap.png`
    #
    # Use Matplotlib with a non-GUI backend:
    # `import matplotlib; matplotlib.use("Agg")`
    #
    # Method:
    # 1. Create `config.output_dir` with `mkdir(parents=True, exist_ok=True)`.
    # 2. Plot `losses` and save `clip_loss_curve.png`.
    # 3. Recompute a small similarity matrix for the first few examples.
    # 4. Plot that matrix with `plt.imshow(...)`.
    # 5. Save it as `clip_retrieval_heatmap.png`.
    # 6. Return both file paths as strings.
    #
    # Check:
    # - Both returned paths should point inside `outputs/multimodal_interview/`.

    raise NotImplementedError("Save CLIP loss curve and retrieval heatmap.")










def run_exercise(config: CLIPTrainingConfig) -> CLIPMetrics:
    """Run the full CLIP training workflow."""
    # This is the main workflow the smaller helpers plug into.
    # Keep this order:
    # data -> vocabulary -> loader -> model -> train -> evaluate -> visualize.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_real_image_caption_dataset(config)
    captions = [example["caption"] for example in examples]
    token_to_id = build_vocabulary(captions)
    dataset = ShapeCaptionDataset(examples, token_to_id, config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TinyCLIP(vocab_size=len(token_to_id), config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    for i in range(config.epochs):
        loss = train_one_epoch(model, loader, optimizer)
        losses.append(loss)
        print(f"Epoch {i+1} -> loss: {loss}")

    retrieval_accuracy = evaluate_retrieval(model, examples, token_to_id, config)
    loss_curve_path, heatmap_path = save_training_visuals(losses, model, examples, token_to_id, config)

    print(f"retrieval_accuracy: {retrieval_accuracy}")

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "retrieval_accuracy": retrieval_accuracy,
        "loss_curve_path": loss_curve_path,
        "heatmap_path": heatmap_path,
    }


def check_metrics(metrics: CLIPMetrics) -> bool:
    """Return True when the final loss improved and retrieval is above chance."""
    return metrics["final_loss"] < metrics["initial_loss"] and metrics["retrieval_accuracy"] > 0.25


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
    """Run the CLIP scaffold one step at a time."""
    config = CLIPTrainingConfig()
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

    examples, ok = report_task_progress("build_real_image_caption_dataset", build_real_image_caption_dataset, config)
    if not count_or_stop(ok):
        return
    captions = [example["caption"] for example in examples]

    token_to_id, ok = report_task_progress("build_vocabulary", build_vocabulary, captions)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("tokenize_captions", tokenize_captions, captions[:4], token_to_id, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinyCLIP.__init__", TinyCLIP, len(token_to_id), config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("compute_clip_loss", compute_clip_loss, torch.eye(4))
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "clip retrieval", metrics["retrieval_accuracy"], "> 0.25 and loss improves")


if __name__ == "__main__":
    main()
