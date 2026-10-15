"""
============================================================
EXERCISE 10: FLAMINGO RESAMPLER CROSS ATTENTION ON IMAGES
============================================================

WHAT YOU ARE BUILDING
You are building the two core Flamingo-style ideas:

```text
many visual tokens -> resampler -> fewer visual tokens
text tokens -> cross-attention -> look at visual tokens
```

Image features come from the image encoder inside this exercise.

WHY THIS EXERCISE EXISTS
Video and high-resolution images can create many visual tokens.
Flamingo-style models reduce the number of visual tokens and let text tokens
attend to them through cross-attention.

HOW THIS FITS INTO THE INTERVIEW PREP
This helps you explain how a model can handle many image or video tokens without
putting all of them directly into the language model sequence.
That is useful for video and edge settings where compute matters.

REAL SYSTEM VS THIS EXERCISE

real system                                             this exercise
---------------------------------------------------------------
large frozen vision encoder (NFNet or ViT)              small image patch encoder
Perceiver Resampler (Flamingo-style resampler)          small learned-latent resampler
large frozen language model (Llama-style LM)            tiny text decoder/classifier
gated cross-attention layers (Flamingo-style)           one cross-attention block
interleaved image-text data           small VQA v2      sample image-question-answer data

VISUALS TO CREATE
Save:
- loss curve
- attention map from text token to resampled visual tokens

INPUTS AND OUTPUTS
Image:
- `[3, image_size, image_size]`

Visual tokens before resampling:
- many tokens, one per patch or small image region

Visual tokens after resampling:
- `[num_resampled_tokens, text_hidden_size]`

Question tokens:
- text tokens for the question

Model output:
- answer logits

METRICS TO TRACK
- training loss
- answer accuracy
- attention weights from question tokens to resampled visual tokens

STEP-BY-STEP TODO
1. Build image-question-answer examples.
2. Tokenize questions and answers.
3. Encode each image into many visual tokens.
4. Compress visual tokens with the resampler.
5. Embed question tokens.
6. Let question tokens cross-attend to visual tokens.
7. Predict the answer.
8. Train the model.
9. Evaluate answer accuracy.
10. Save the loss curve and attention map.

DEFINITION OF DONE
You are done when:
- many image tokens are compressed to fewer visual tokens
- text tokens can attend to visual tokens
- attention can be visualized
- training improves answer accuracy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FlamingoTinyConfig:
    """Settings for the tiny Flamingo-style exercise."""

    image_size: int = 48
    visual_hidden_size: int = 64
    text_hidden_size: int = 64
    num_resampled_tokens: int = 8
    num_attention_heads: int = 4
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 3e-3
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    hf_dataset_name: str = "Multimodal-Fatima/VQAv2_sample_train"
    sample_size: int = 256
    lm_mode: str = "tiny"


class FlamingoExample(TypedDict):
    """One image-question-answer example."""

    image: torch.Tensor
    question: str
    answer: str


class FlamingoMetrics(TypedDict):
    """Training summary."""

    initial_loss: float
    final_loss: float
    answer_accuracy: float
    loss_curve_path: str
    attention_path: str


class FlamingoDataset(Dataset):
    """Dataset for the tiny Flamingo-style answer task."""

    def __init__(self, examples: list[FlamingoExample], question_vocab: dict[str, int], answer_vocab: dict[str, int]):
        self.examples = examples
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.max_question_length = max(len(example["question"].lower().split()) for example in examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        question_ids = tokenize_questions([example["question"]], self.question_vocab)[0]
        padded_question_ids = torch.full((self.max_question_length,), self.question_vocab.get("<pad>", 0), dtype=torch.long)
        copy_length = min(question_ids.shape[0], self.max_question_length)
        padded_question_ids[:copy_length] = question_ids[:copy_length]
        return {
            "image": example["image"],
            "question_ids": padded_question_ids,
            "answer_id": torch.tensor(self.answer_vocab[example["answer"]], dtype=torch.long),
            "question": example["question"],
            "answer": example["answer"],
        }


def build_flamingo_dataset(config: FlamingoTinyConfig) -> list[FlamingoExample]:
    """Build real image-question-answer examples."""
    # Where this fits:
    # This creates the real image and question examples used to test the resampler
    # and cross-attention path.
    #
    # Dataset:
    # Use a small parquet VQA v2 sample from Hugging Face:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    #
    # Goal:
    # Return real images with question-answer text.
    #
    # Output:
    # - list of dictionaries with:
    #   - `"image"`: `[3, image_size, image_size]`
    #   - `"question"`: string
    #   - `"answer"`: string
    #
    # Method:
    # Use `load_dataset` from the Hugging Face `datasets` package:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    # Use this exact transform shape from `torchvision.transforms`:
    # `torchvision.transforms.Compose([torchvision.transforms.Resize((config.image_size, config.image_size)), torchvision.transforms.ToTensor()])`.
    # Keep only short answers at first, such as colors, objects, or yes/no.
    #
    # Steps:
    # 1. Load the Hugging Face split with `load_dataset(...)`.
    # 2. Inspect one row with `data[0].keys()` if you are unsure of the field names.
    # 3. For each row, read the PIL image, question text, and answer text.
    #    This dataset has `multiple_choice_answer`, which is already one cleaned answer.
    # 4. Convert the image to RGB with the PIL method `image.convert("RGB")`.
    # 5. Resize the image and convert it to a tensor with the torchvision transform.
    # 6. Lowercase and strip question and answer strings.
    # 7. Skip rows with empty questions or answers.
    # 8. Append dictionaries with `"image"`, `"question"`, and `"answer"`.
    # 9. Stop once you have `config.sample_size` examples.
    #
    # Checks:
    # - Images are real tensors.
    # - Questions and answers are non-empty strings.

    raise NotImplementedError("Build data for Flamingo-style training.")






class VisualTokenEncoder(nn.Module):
    """Encode an image into many visual tokens."""

    def __init__(self, config: FlamingoTinyConfig):
        super().__init__()
        # Goal:
        # Build a small image encoder that returns many visual tokens.
        #
        # Method:
        # Use `nn.Conv2d` from `torch.nn`.
        # A simple call shape is:
        # `nn.Conv2d(3, config.visual_hidden_size, kernel_size=4, stride=4)`.
        # Each output grid cell becomes one visual token.
        #
        # Steps:
        # 1. Create a convolution and store it on `self`.
        # 2. Use 3 input channels because the images are RGB.
        # 3. Use `config.visual_hidden_size` output channels.
        # 4. Use the same `kernel_size` and `stride` so patches do not overlap.
        #
        # Check:
        # - The forward output should be `[batch_size, num_visual_tokens, visual_hidden_size]`.

        raise NotImplementedError("Build image-to-visual-token encoder.")
        # Example with the default config:
        # input images are [batch_size, 3, 48, 48].
        # kernel_size=4 means each output looks at one 4x4 image patch.
        # stride=4 means the convolution moves 4 pixels each time, so patches do not overlap.
        # 48 / 4 = 12, so the output grid is 12 by 12.
        # 12 * 12 = 144 visual tokens.
        # Shape after this convolution: [batch_size, visual_hidden_size, 12, 12].


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return visual tokens with shape `[B, num_visual_tokens, visual_hidden_size]`."""
        # Goal:
        # Convert image pixels into a sequence of visual tokens.
        #
        # Method:
        # Use the convolution created in `__init__`.
        # Use `.flatten(start_dim=2)` to flatten height and width into tokens.
        # Use `.transpose(1, 2)` to move the hidden size after the token index.
        #
        # Steps:
        # 1. Run the convolution to get `[B, hidden, grid_h, grid_w]`.
        # 2. Flatten grid positions with `tokens = features.flatten(start_dim=2)`.
        # 3. Transpose with `tokens = tokens.transpose(1, 2)`.
        # 4. Return `[B, num_visual_tokens, visual_hidden_size]`.
        #
        # Check:
        # - The last dimension should be `config.visual_hidden_size`.

        raise NotImplementedError("Return visual tokens.")



class TinyPerceiverResampler(nn.Module):
    """Compress many visual tokens into a fixed small set."""

    def __init__(self, config: FlamingoTinyConfig):
        super().__init__()
        # Goal:
        # Build learned latent tokens and attention from latents to visual tokens.
        #
        # Output shape:
        # - `[B, num_resampled_tokens, text_hidden_size]`
        #
        # Suggested layers:
        # - `self.latents = nn.Parameter(torch.randn(config.num_resampled_tokens, config.text_hidden_size) * 0.02)`
        # - `self.visual_to_text = nn.Linear(config.visual_hidden_size, config.text_hidden_size)`
        # - `self.attention = nn.MultiheadAttention(embed_dim=config.text_hidden_size, num_heads=config.num_attention_heads, batch_first=True)`
        #
        # Method:
        # Use `nn.Parameter` from `torch.nn` for learned latent tokens.
        # Use `nn.Linear(config.visual_hidden_size, config.text_hidden_size)` to
        # project visual tokens to the text width.
        # Use `nn.MultiheadAttention(embed_dim=config.text_hidden_size, num_heads=config.num_attention_heads, batch_first=True)`.
        #
        # Steps:
        # 1. Create `self.latents` with shape `[num_resampled_tokens, text_hidden_size]`.
        # 2. Create `self.visual_to_text`.
        # 3. Create `self.attention`.
        # 4. Save `config` on `self` if forward needs token counts.
        #
        # Checks:
        # - The resampler always returns exactly `config.num_resampled_tokens`.
        # - The last dimension should be `config.text_hidden_size`.

        raise NotImplementedError("Build resampler layers.")



    def forward(self, visual_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return resampled tokens and attention weights."""
        # Goal:
        # Compress many visual tokens into a fixed smaller number of visual tokens.
        #
        # Input:
        # - `visual_tokens`: `[batch_size, num_visual_tokens, visual_hidden_size]`
        #
        # Output:
        # - resampled tokens: `[batch_size, num_resampled_tokens, text_hidden_size]`
        # - attention weights for visualization
        #
        # Method:
        # Use `self.visual_to_text(visual_tokens)` before attention.
        # Expand latents with:
        # `latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)`.
        # Call attention like:
        # `outputs, weights = self.attention(query=latents, key=visual_tokens, value=visual_tokens, need_weights=True)`.
        #
        # Steps:
        # 1. Read `batch_size = visual_tokens.shape[0]`.
        # 2. Project visual tokens to text width.
        # 3. Expand learned latent tokens across the batch.
        # 4. Use latents as `query`.
        # 5. Use projected visual tokens as `key` and `value`.
        # 6. Return resampled outputs and attention weights.
        #
        # Check:
        # - The number of output tokens should always be `config.num_resampled_tokens`.

        # visual_tokens shape is [b, 144, 64] 
        # where num_visual_tokens = 64, and we want to compress/reduce it down to num_resampled_tokens=8

        raise NotImplementedError("Compress visual tokens.")




        # The latent tokens are learned because they are `nn.Parameter`s.
        # That means gradient descent updates them during training, just like layer weights.
        # They start as small random vectors, then learn what kind of visual summary tokens
        # help the image-question-answer task.
        #
        # Cross-attention uses the learned latents to look at the many image visual tokens:
        # - query = the learned latent tokens, asking "what visual summary should I build?"
        # - key = projected visual tokens, used to decide which image tokens match each latent
        # - value = projected visual tokens, used to build the returned resampled outputs
        #
        # outputs are the updated latent vectors after they pulled information from visual tokens.
        # Shape: [batch_size, 8, 64].
        # weights show how strongly each latent looked at each visual token.
        # weights are the attention weights after softmax.
        # They are not the raw attention scores.
        # attention scores = raw match numbers before softmax
        # attention weights = normalized numbers after softmax
        #
        # Behind the scenes, attention is roughly:
        #
        # scores = latents @ projected_visual_tokens.transpose(-2, -1)  # [B, 8, 64] @ [B, 64, 144] = [B, 8, 144]
        # weights = torch.softmax(scores, dim=-1)                       # [B, 8, 144]
        # outputs = weights @ projected_visual_tokens                    # [B, 8, 144] @ [B, 144, 64] = [B, 8, 64]
        #
        # In words:
        # - scores are raw match numbers between each latent and each visual token
        # - softmax turns those scores into weights that sum to 1 across visual tokens
        # - the weights mix the visual token vectors to make the final resampled outputs
        #
        # Small example for one latent looking at 3 visual tokens:
        # raw attention scores: [2.0, 1.0, 0.0]
        # attention weights after softmax: [0.67, 0.24, 0.09]
        #
        # This means:
        # - this latent looked 67% at visual token 0
        # - this latent looked 24% at visual token 1
        # - this latent looked 9% at visual token 2



class TinyCrossAttentionAnswerModel(nn.Module):
    """Text attends to resampled visual tokens before answering."""

    def __init__(self, question_vocab_size: int, num_answers: int, config: FlamingoTinyConfig):
        super().__init__()
        # Goal:
        # Build:
        # - question embedding
        # - visual encoder
        # - resampler
        # - cross-attention from question tokens to visual tokens
        # - answer head
        #
        # Suggested layers:
        # - `self.question_embedding = nn.Embedding(question_vocab_size, config.text_hidden_size)`
        # - `self.visual_encoder = VisualTokenEncoder(config)`
        # - `self.resampler = TinyPerceiverResampler(config)`
        # - `self.cross_attention = nn.MultiheadAttention(embed_dim=config.text_hidden_size, num_heads=config.num_attention_heads, batch_first=True)`
        # - `self.answer_head = nn.Linear(config.text_hidden_size, num_answers)`
        #
        # Method:
        # Use `nn.Embedding` from `torch.nn` for question token ids.
        # Use the local `VisualTokenEncoder(config)` and `TinyPerceiverResampler(config)`.
        # Use `nn.MultiheadAttention(..., batch_first=True)` for cross-attention.
        # Use `nn.Linear(config.text_hidden_size, num_answers)` for the answer head.
        #
        # Steps:
        # 1. Build `self.question_embedding`.
        # 2. Build `self.visual_encoder`.
        # 3. Build `self.resampler`.
        # 4. Build `self.cross_attention`.
        # 5. Build `self.answer_head`.
        #
        # Checks:
        # - The embedding table has `question_vocab_size` rows.
        # - The answer head output width is `num_answers`.

        raise NotImplementedError("Build Flamingo-style answer model.")






    def forward(self, images: torch.Tensor, question_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return answer logits and attention weights."""
        # Goal:
        # Let question tokens attend to compressed visual tokens, then predict an answer.
        #
        # Input:
        # - `images`: `[batch_size, 3, image_size, image_size]`
        # - `question_ids`: token ids for questions
        #
        # Output:
        # - answer logits: `[batch_size, num_answers]`
        # - attention weights for the visual tokens
        #
        # Method:
        # Use `self.question_embedding(question_ids)` for text embeddings.
        # Call cross-attention like:
        # `attended, weights = self.cross_attention(query=question_embeddings, key=resampled_tokens, value=resampled_tokens, need_weights=True)`.
        # Use `.mean(dim=1)` to pool over question tokens.
        #
        # Steps:
        # 1. Encode images with `self.visual_encoder(images)`.
        # 2. Compress visual tokens with `self.resampler(...)`.
        # 3. Embed question ids.
        # 4. Use question embeddings as `query`.
        # 5. Use resampled visual tokens as `key` and `value`.
        # 6. Pool attended question states with `.mean(dim=1)`.
        # 7. Run the answer head.
        # 8. Return raw logits and attention weights.
        #
        # Check:
        # - Return raw logits, not probabilities.

        # images is shape [b, 3, 48, 48]

        raise NotImplementedError("Run resampler and cross-attention.")








def tokenize_questions(questions: list[str], token_to_id: dict[str, int]) -> torch.Tensor:
    """Tokenize questions."""
    # Goal:
    # Convert question strings into padded token ids.
    #
    # Output:
    # - tensor shaped `[len(questions), max_question_length]`
    #
    # Method:
    # Use Python string methods `question.lower().split()`.
    # Use `torch.full((len(questions), max_length), pad_id, dtype=torch.long)` to
    # create the padded output tensor.
    # Use `token_to_id.get(word, unk_id)` for unknown words.
    #
    # Steps:
    # 1. Set `pad_id = token_to_id.get("<pad>", 0)`.
    # 2. Set `unk_id = token_to_id.get("<unk>", 1)`.
    # 3. Compute `max_length` from the longest split question in this batch.
    # 4. Create the padded tensor with `torch.full(...)`.
    # 5. For each question, map words to ids with `.get(...)`.
    # 6. Copy ids into the matching row and first columns of the tensor.
    #
    # Check:
    # - The dtype should be `torch.long`.

    raise NotImplementedError("Tokenize questions.")




def train_one_epoch(model: TinyCrossAttentionAnswerModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train one epoch."""
    # Goal:
    # Train the Flamingo-style answer model for one epoch.
    #
    # Method:
    # Use normal PyTorch calls:
    # `model.train()`, `nn.CrossEntropyLoss()`, `optimizer.zero_grad()`,
    # `loss.backward()`, and `optimizer.step()`.
    #
    # Steps:
    # 1. Create `loss_fn = nn.CrossEntropyLoss()`.
    # 2. Call `model.train()` before the loop.
    # 3. Move `batch["image"]`, `batch["question_ids"]`, and `batch["answer_id"]` to `device`.
    # 4. Run `logits, _ = model(images, question_ids)`.
    # 5. Compute `loss = loss_fn(logits, answer_ids)`.
    # 6. Clear old gradients, backprop, and step the optimizer.
    # 7. Add `loss.item()` to a running total.
    # 8. Return the mean loss as a Python float.
    #
    # Check:
    # - Return mean loss as a Python float.

    raise NotImplementedError("Train one Flamingo-style epoch.")





def evaluate_answers(model: TinyCrossAttentionAnswerModel, loader: DataLoader) -> float:
    """Return answer accuracy."""
    # Goal:
    # Measure answer accuracy.
    #
    # Method:
    # Use `model.eval()` and the PyTorch context manager `torch.no_grad()`.
    # Use `logits.argmax(dim=-1)` to get predicted answer ids.
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Start `correct = 0` and `total = 0`.
    # 3. Loop over batches inside `with torch.no_grad():`.
    # 4. Move images, question ids, and answer ids to `device`.
    # 5. Run `logits, _ = model(images, question_ids)`.
    # 6. Get predictions with `logits.argmax(dim=-1)`.
    # 7. Count matches with `(predictions == answer_ids).sum().item()`.
    # 8. Return `correct / total`.
    #
    # Check:
    # - Return a float between `0.0` and `1.0`.

    raise NotImplementedError("Evaluate answer accuracy.")




def save_flamingo_visuals(losses: list[float], model: TinyCrossAttentionAnswerModel, examples: list[FlamingoExample], config: FlamingoTinyConfig) -> tuple[str, str]:
    """Save loss curve and attention visualization."""
    # Goal:
    # Save the loss curve and a visual-token attention map.
    #
    # Method:
    # Use `config.output_dir.mkdir(parents=True, exist_ok=True)` from `pathlib.Path`.
    # Use Matplotlib: `matplotlib.use("Agg")`, `plt.figure()`, `plt.plot(...)`,
    # `plt.imshow(...)`, `plt.savefig(path)`, and `plt.close()`.
    # Use `torch.no_grad()` when running the model for attention.
    #
    # Steps:
    # 1. Create `config.output_dir`.
    # 2. Save `losses` as a line plot.
    # 3. Pick one example and add a batch dimension to its image.
    # 4. Reuse the same question-id preparation pattern used by `FlamingoDataset`.
    # 5. Run the model inside `with torch.no_grad():`.
    # 6. Convert attention weights to a 2D tensor for plotting.
    # 7. Save the attention plot with `plt.imshow(...)`.
    # 8. Return both paths as strings.
    #
    # Check:
    # - The attention plot should have one axis for text tokens and one for visual tokens.

    raise NotImplementedError("Save Flamingo-style visuals.")













def run_exercise(config: FlamingoTinyConfig) -> FlamingoMetrics:
    """Run full Flamingo-style training."""
    # Main workflow:
    # VQA data -> question/answer ids -> visual tokens -> resampler -> cross-attention -> train.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_flamingo_dataset(config)
    question_words = sorted({word for example in examples for word in example["question"].lower().split()})
    question_vocab = {"<pad>": 0, **{word: index + 1 for index, word in enumerate(question_words)}}
    answer_vocab = {answer: index for index, answer in enumerate(sorted({example["answer"] for example in examples}))}
    dataset = FlamingoDataset(examples, question_vocab, answer_vocab)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TinyCrossAttentionAnswerModel(
        question_vocab_size=len(question_vocab),
        num_answers=len(answer_vocab),
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    for i in range(config.epochs):
        loss = train_one_epoch(model, loader, optimizer)
        losses.append(loss)
        print(f"Epoch {i+1} -> loss: {loss}")

    answer_accuracy = evaluate_answers(model, loader)
    loss_curve_path, attention_path = save_flamingo_visuals(losses, model, examples, config)

    print(f"\n answer_accuracy: {answer_accuracy}")

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "answer_accuracy": answer_accuracy,
        "loss_curve_path": loss_curve_path,
        "attention_path": attention_path,
    }


def check_metrics(metrics: FlamingoMetrics) -> bool:
    return metrics["final_loss"] < metrics["initial_loss"] and metrics["answer_accuracy"] > 0.60


def report_task_progress(task_name: str, task_fn, *args):
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the Flamingo-style scaffold one step at a time."""
    config = FlamingoTinyConfig()
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

    examples, ok = report_task_progress("build_flamingo_dataset", build_flamingo_dataset, config)
    if not count_or_stop(ok):
        return
    question_words = sorted({word for example in examples for word in example["question"].lower().split()})
    question_vocab = {"<pad>": 0, **{word: index + 1 for index, word in enumerate(question_words)}}
    answer_vocab = {answer: index for index, answer in enumerate(sorted({example["answer"] for example in examples}))}

    _, ok = report_task_progress("tokenize_questions", tokenize_questions, [example["question"] for example in examples[:4]], question_vocab)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("VisualTokenEncoder.__init__", VisualTokenEncoder, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinyPerceiverResampler.__init__", TinyPerceiverResampler, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinyCrossAttentionAnswerModel.__init__", TinyCrossAttentionAnswerModel, len(question_vocab), len(answer_vocab), config)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "flamingo answer accuracy", metrics["answer_accuracy"], "> 0.60 and loss improves")


if __name__ == "__main__":
    main()
