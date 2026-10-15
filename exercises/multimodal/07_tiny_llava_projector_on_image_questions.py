"""
============================================================
EXERCISE 07: TINY LLaVA PROJECTOR ON IMAGE QUESTIONS
============================================================

WHAT YOU ARE BUILDING
You are building the bridge used in a LLaVA-style vision-language model.

LLaVA-style picture:

```text
image -> vision encoder -> visual features -> projector -> language model space
question -> text tokens
visual tokens + question tokens -> answer
```

This exercise trains on real image-question-answer data.
Use a small parquet VQA v2 sample from Hugging Face.

WHY THIS EXERCISE EXISTS
Many VLMs use a vision encoder and a projector to connect vision to a language model.
The projector is small, but it is important:
it changes visual features into vectors with the same width as language-token embeddings.

HOW THIS FITS INTO THE INTERVIEW PREP
This is the simplest LLaVA-style bridge.
It helps you explain how an image encoder can connect to a language model without
pretending that the image tokens are normal words.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
---------------------------------------------------------------
real images                          small VQA v2 sample
large vision encoder (CLIP ViT-L/14)            small CNN vision encoder
large pretrained LLM (Vicuna, Llama, or Qwen)   tiny answer classifier first
projector into LLM hidden size (MLP projector)  projector into tiny text hidden size
visual question answering            color/shape question answering

WHAT IS INCLUDED
- image tensors
- image-question-answer examples
- a vision encoder
- a trainable projector
- an answer head
- training and visual checks

WHAT IS SKIPPED
- full autoregressive answer generation
- large pretrained LLMs

That full next-token version comes in the related exercise.

VISUALS TO CREATE
Save:
- training loss curve
- prediction grid showing image, question, target answer, predicted answer

INPUTS AND OUTPUTS
Input image:
- `[3, image_size, image_size]`

Input question:
- text such as `"what color is the shape?"`

Target answer:
- text class such as `"red"` or `"circle"`

Model output:
- answer logits with shape `[batch_size, num_answers]`

METRICS TO TRACK
- training loss
- answer accuracy
- a few before/after predictions

STEP-BY-STEP TODO
1. Build real image-question-answer examples.
2. Build a question vocabulary.
3. Build an answer vocabulary.
4. Encode questions.
5. Build a small image encoder.
6. Build the projector from vision width to language width.
7. Fuse image and question vectors.
8. Train the answer head.
9. Evaluate answer accuracy.
10. Save the loss curve and prediction grid.

DEFINITION OF DONE
You are done when:
- the model trains on image-question-answer examples
- projector output has the expected shape
- prediction accuracy improves
- saved visuals show model outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict
import string

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
class TinyLLaVAProjectorConfig:
    """Settings for the tiny LLaVA projector exercise."""

    image_size: int = 48
    vision_hidden_size: int = 64
    language_hidden_size: int = 96
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 3e-3
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    hf_dataset_name: str = "Multimodal-Fatima/VQAv2_sample_train"
    sample_size: int = 256
    lm_mode: str = "tiny"


class VQAExample(TypedDict):
    """One image-question-answer example."""

    image: torch.Tensor
    question: str
    answer: str
    color: str
    shape: str


class ProjectorMetrics(TypedDict):
    """Training summary."""

    initial_loss: float
    final_loss: float
    answer_accuracy: float
    visual_grid_path: str
    loss_curve_path: str


# Learning order:
# 1. Build real image-question-answer data.
# 2. Build vocabularies for questions and answers.
# 3. Encode images with a small vision encoder.
# 4. Project visual vectors into language hidden size.
# 5. Fuse visual and question vectors.
# 6. Train an answer classifier.
# 7. Visualize predictions.


def build_vqa_dataset(config: TinyLLaVAProjectorConfig) -> list[VQAExample]:
    """Build real image-question-answer examples."""
    # Where this fits:
    # This creates the small visual question answering world used by the projector.
    # The model should learn from image pixels and question text.
    #
    # Goal:
    # Return examples with image tensors and question-answer labels.
    #
    # Example:
    # - image: red square
    # - question: "what color is the shape?"
    # - answer: "red"
    #
    # Include both question types:
    # - color questions
    # - shape questions
    #
    # Dataset:
    # Use a small parquet VQA v2 sample from Hugging Face:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    #
    # Output:
    # Return a list of dictionaries with:
    # - `"image"`: tensor shaped `[3, image_size, image_size]`
    # - `"question"`: short question string
    # - `"answer"`: short answer string
    # - `"color"` and `"shape"`: empty strings here because real VQA rows do not
    #   give simple color or shape metadata
    #
    # Method:
    # Use `datasets.load_dataset` from Hugging Face:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    # Use this exact transform shape:
    # `torchvision.transforms.Compose([torchvision.transforms.Resize((config.image_size, config.image_size)), torchvision.transforms.ToTensor()])`.
    #
    # Steps:
    # 1. Load a small VQA v2 split with `load_dataset(...)`.
    # 2. Inspect one row's keys if needed with `data[0].keys()`.
    # 3. Read the image field, convert it with `.convert("RGB")`, then apply the transform.
    # 4. Read `row["question"]`.
    # 5. Read `row["multiple_choice_answer"]`.
    #    This dataset has `multiple_choice_answer`, which is already one cleaned answer.
    # 6. Lowercase and strip the answer text.
    # 7. Keep examples with a non-empty question and non-empty answer.
    # 8. Return at most `config.sample_size` usable examples.
    #
    # Checks:
    # - `examples[0]["image"].shape == (3, config.image_size, config.image_size)`.
    # - Questions and answers should be strings, not token ids yet.

    from datasets import load_dataset

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(config.image_size, config.image_size)),
        torchvision.transforms.ToTensor(),
    ])

    data = load_dataset(path=config.hf_dataset_name, split=f"train[:{config.sample_size}]")

    examples: list[VQAExample] = []

    for row in data:
        image = row["image"].convert("RGB")
        image_tensor = transform(image)

        question = str(row["question"]).strip()
        answer = str(row["multiple_choice_answer"]).lower().strip()

        if not question or not answer:
            continue

        examples.append({
            "image": image_tensor,
            "question": question,
            "answer": answer,
            "color": str(row.get("color", "")),
            "shape": str(row.get("shape", "")),
        })

        if len(examples) >= config.sample_size:
            break

    # raise NotImplementedError("Build image-question-answer training data.")
    return examples


def build_text_vocabulary(texts: list[str]) -> dict[str, int]:
    """Build a small word vocabulary."""
    # Goal:
    # Build a word-to-id dictionary for VQA questions.
    #
    # Input:
    # - questions such as `"what color is the object ?"`
    #
    # Output:
    # - dictionary with `<pad>` at id 0 and `<unk>` at id 1.
    #
    # Method:
    # Use normal Python string methods:
    # - `text.lower()` lowercases a question.
    # - `.split()` splits it into words.
    # Use `sorted(...)` to make token ids stable.
    #
    # Steps:
    # 1. Start with `{"<pad>": 0, "<unk>": 1}`.
    # 2. Lowercase and split each question.
    # 3. Sort the unique words.
    # 4. Add each word to the dictionary.
    #
    # Check:
    # - Common words like `"what"` and `"color"` should have ids if they appear.

    all_words = []

    for text in texts:
        words = text.lower().strip(string.punctuation).split()
        all_words.extend(words)

    vocabulary = sorted(set(all_words))
    vocabulary.insert(0, "<pad>")
    vocabulary.insert(1, "<unk>")

    token_to_id = {w:i for i,w in enumerate(vocabulary)}

    return token_to_id
    #raise NotImplementedError("Build question vocabulary.")


def encode_questions(questions: list[str], token_to_id: dict[str, int]) -> torch.Tensor:
    """Encode questions as simple bag-of-words vectors or token ids."""
    # Goal:
    # Turn each question into a simple vector the model can use.
    #
    # Input:
    # - `questions`: list[str]
    # - `token_to_id`: output from `build_text_vocabulary`
    #
    # Output:
    # - tensor shaped `[len(questions), len(token_to_id)]`
    # - each row is a bag-of-words vector
    #
    # Method:
    # Use `torch.zeros(len(questions), len(token_to_id), dtype=torch.float32)`.
    # `torch.zeros` is a PyTorch function that creates the full batch tensor.
    # Use dictionary `.get(word, token_to_id["<unk>"])` for unknown words.
    #
    # Steps:
    # 1. Create `vectors = torch.zeros(len(questions), len(token_to_id))`.
    # 2. Lowercase and split each question.
    # 3. Convert each word to an id with `.get(word, token_to_id["<unk>"])`.
    # 4. Add `1.0` to that position.
    # 5. Return the full `[num_questions, vocab_size]` tensor.
    #
    # Check:
    # - If a question has words, its row sum should be greater than 0.    

    vectors = torch.zeros(len(questions), len(token_to_id), dtype=torch.float32)    
    
    for i, question in enumerate(questions):
        words = question.lower().strip(string.punctuation).split()
        
        for word in words:
            token_id = token_to_id.get(word, token_to_id["<unk>"])
            vectors[i][token_id] += 1.0        

    return vectors
    #raise NotImplementedError("Encode questions.")


def build_answer_vocabulary(answers: list[str]) -> dict[str, int]:
    """Map answer strings to class ids."""
    # Goal:
    # Build the answer-class dictionary.
    #
    # Input:
    # - answers such as `"red"`, `"blue"`, `"cat"`, `"car"`
    #
    # Output:
    # - dictionary mapping answer text to class id.
    # - Example: `{"blue": 0, "red": 1}`
    #
    # Method:
    # Use Python `set(...)` to remove duplicates and `sorted(...)` for stable ids.
    #
    # Steps:
    # 1. Strip each answer.
    # 2. Keep non-empty answers only.
    # 3. Sort unique answers so ids are stable.
    # 4. Return `{answer: index}`.
    #
    # Check:
    # - Every example answer should be in this dictionary before building the dataset.

    vocabulary = sorted({answer.lower().strip() for answer in answers if answer.strip()})
    token_to_id = {answer: i for i, answer in enumerate(vocabulary)}

    return token_to_id
    #raise NotImplementedError("Build answer vocabulary.")


class TinyVQADataset(Dataset):
    """Dataset for projector-based visual question answering."""

    def __init__(self, examples: list[VQAExample], question_vocab: dict[str, int], answer_vocab: dict[str, int]):
        self.examples = examples
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        question_vector = encode_questions([example["question"]], self.question_vocab)[0]
        answer_id = self.answer_vocab[example["answer"].lower().strip()]
        return {
            "image": example["image"],
            "question_vector": question_vector,
            "answer_id": torch.tensor(answer_id, dtype=torch.long),
            "question": example["question"],
            "answer": example["answer"],
        }



class TinyImageEncoder(nn.Module):
    def __init__(self, config: TinyLLaVAProjectorConfig):
        super().__init__()

        self.config = config

        self.conv2d_1 = nn.Conv2d(in_channels = 3, out_channels=16, kernel_size=3, stride=1, padding="same")

        self.maxpool2d = nn.MaxPool2d(kernel_size=2) #default stride is kernel_size

        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same")

        self.adaptive_avgpool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.linear = nn.Linear(in_features=32, out_features=config.vision_hidden_size)

    def forward(self, images) -> torch.tensor:
        
        x = nn.functional.relu(self.conv2d_1(images))   # [B, 16, 48, 48]
        x = self.maxpool2d(x)                           # [B, 16, 24, 24]

        x = nn.functional.relu(self.conv2d_2(x))        # [B, 32, 24, 24]
        x = self.adaptive_avgpool2d(x)                  # [B, 32, 1, 1]

        x = torch.flatten(x, start_dim=1)               # [B, 32]

        features = self.linear(x)  # [B, embedding_dim]        

        return features

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, config: TinyLLaVAProjectorConfig):
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size

        self.linear = nn.Linear(in_features=vocab_size, out_features=config.language_hidden_size)

    def forward(self, question_vectors: torch.Tensor) -> torch.Tensor:
        
        # question_vectors shape is [batch_size, question_vocab_size]

        question_embeddings = self.linear(question_vectors)      # [batch_size, language_hidden_size]        

        return question_embeddings

class TinyLLaVAProjectorModel(nn.Module):
    """Tiny LLaVA-style projector model for answer classification."""

    def __init__(self, question_vocab_size: int, num_answers: int, config: TinyLLaVAProjectorConfig):
        super().__init__()
        # Goal:
        # Build:
        # - small CNN vision encoder -> `[B, vision_hidden_size]`
        # - projector -> `[B, language_hidden_size]`
        # - question encoder -> `[B, language_hidden_size]`
        # - answer head -> `[B, num_answers]`
        #
        # This is not a full LLM yet.
        # It isolates the visual bridge idea before the related exercise.
        #
        # Suggested layers:
        # - `self.vision_encoder = nn.Sequential(...)`
        #   A small CNN can end with one vector shaped `[batch_size, vision_hidden_size]`.
        # - `self.projector = nn.Linear(config.vision_hidden_size, config.language_hidden_size)`
        #   This is the key LLaVA-style bridge.
        # - `self.question_encoder = nn.Linear(question_vocab_size, config.language_hidden_size)`
        #   This works if questions are bag-of-words vectors.
        # - `self.answer_head = nn.Linear(config.language_hidden_size, num_answers)`
        #
        # Method:
        # Use PyTorch layers from `torch.nn`:
        # - `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, and `nn.AdaptiveAvgPool2d`
        #   for the small image encoder.

        # - `nn.Linear(config.vision_hidden_size, config.language_hidden_size)`
        #   for the visual projector.

        # - `nn.Linear(question_vocab_size, config.language_hidden_size)` for the question path.
        # - `nn.Linear(config.language_hidden_size, num_answers)` for the answer head.
        #
        # Steps:
        # 1. Build `self.vision_encoder` so it returns `[B, vision_hidden_size]`.
        # 2. Build `self.projector`.
        # 3. Build `self.question_encoder`.
        # 4. Build `self.answer_head`.
        # 5. In `forward`, add projected image and question vectors before the answer head.
        #
        # Checks:
        # - The projector input width must equal `config.vision_hidden_size`.
        # - The answer head output width must equal `num_answers`.

        self.vision_encoder = TinyImageEncoder(config)

        self.projector = nn.Sequential(nn.Linear(in_features=config.vision_hidden_size, out_features=config.language_hidden_size))
        
        self.question_encoder = QuestionEncoder(question_vocab_size, config)

        self.answer_head = nn.Linear(config.language_hidden_size, num_answers)

        #raise NotImplementedError("Build the tiny LLaVA projector model.")

    def forward(self, images: torch.Tensor, question_vectors: torch.Tensor) -> torch.Tensor:
        """Return answer logits with shape `[batch_size, num_answers]`."""
        # Goal:
        # Use the image and question together to predict an answer class.
        #
        # Input:
        # - `images`: `[batch_size, 3, image_size, image_size]`
        # - `question_vectors`: `[batch_size, question_vocab_size]`
        #
        # Output:
        # - logits shaped `[batch_size, num_answers]`
        #
        # Method:
        # Use the layers created in `__init__`.
        # Use `torch.flatten(x, start_dim=1)` after image pooling if the image
        # encoder returns `[B, hidden, 1, 1]`.
        #
        # Steps:
        # 1. Encode the image with the CNN vision encoder.
        # 2. Flatten if needed to get `[B, vision_hidden_size]`.
        # 3. Project the vision vector to `config.language_hidden_size`.
        # 4. Encode the question vector to `config.language_hidden_size`.
        # 5. Add the two hidden vectors.
        # 6. Run the answer head and return raw logits.
        #
        # Check:
        # - Do not apply softmax. Cross entropy expects raw logits.

        visual_features = self.vision_encoder(images)       # [B, vision_hidden_size]

        projection = self.projector(visual_features)        # [B, language_hidden_size]

        question_embeddings = self.question_encoder(question_vectors)   # [B, language_hidden_size]

        add = projection + question_embeddings

        logits = self.answer_head(add)

        return logits

        #raise NotImplementedError("Encode image and question, then predict answer logits.")


def train_one_epoch(model: TinyLLaVAProjectorModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train one epoch on image-question-answer examples."""
    # Goal:
    # Train the projector model for one full pass over the data.
    #
    # Method:
    # Use standard PyTorch calls:
    # - `model.train()`
    # - `.to(device)`
    # - `nn.CrossEntropyLoss()`
    # - `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
    #
    # Steps:
    # 1. Call `model.train()`.
    # 2. Start running totals for loss and example count.
    # 3. Move image, question vector, and answer id tensors to `device`.
    # 4. Get logits from the model.
    # 5. Use `nn.CrossEntropyLoss()(logits, answer_ids)`.
    # 6. Run zero-grad, backward, and step.
    # 7. Return mean loss over all examples.
    #
    # Check:
    # - Return mean loss as a Python float.

    model.train()

    total_loss = 0
    total_examples = 0
    loss_fn = nn.CrossEntropyLoss()

    for data in loader:
        images = data["image"]
        question_vectors = data["question_vector"]
        answer_ids = data["answer_id"]

        images = images.to(device)
        question_vectors = question_vectors.to(device)
        answer_ids = answer_ids.to(device)

        optimizer.zero_grad()
        logits = model(images, question_vectors)
        loss = loss_fn(logits, answer_ids)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples

    # raise NotImplementedError("Train one projector VQA epoch.")

@torch.no_grad()
def evaluate_answers(model: TinyLLaVAProjectorModel, loader: DataLoader) -> float:
    """Return answer accuracy."""
    # Goal:
    # Measure answer classification accuracy.
    #
    # Method:
    # Use `model.eval()` and `torch.no_grad()` from PyTorch.
    # Use `logits.argmax(dim=-1)` to pick the class id with the largest logit.
    #
    # Steps:
    # 1. Call `model.eval()`.
    # 2. Loop over batches inside `with torch.no_grad():`.
    # 3. Move tensors to `device`.
    # 4. Compute logits.
    # 5. Predict with `argmax(dim=-1)`.
    # 6. Count correct predictions and return correct / total.
    #
    # Check:
    # - Accuracy should be between `0.0` and `1.0`.

    model.eval()

    total_correct = 0
    total_examples = 0    

    for data in loader:
        images = data["image"]
        question_vectors = data["question_vector"]
        answer_ids = data["answer_id"]

        images = images.to(device)
        question_vectors = question_vectors.to(device)
        answer_ids = answer_ids.to(device)

        
        logits = model(images, question_vectors)

        pred = logits.argmax(dim=-1)
        correct = (pred == answer_ids).sum().item()

        batch_size = images.shape[0]
        total_correct += correct
        total_examples += batch_size

    return total_correct / total_examples

    #raise NotImplementedError("Evaluate answer accuracy.")


def save_projector_visuals(losses: list[float], model: TinyLLaVAProjectorModel, examples: list[VQAExample], config: TinyLLaVAProjectorConfig) -> tuple[str, str]:
    """Save loss curve and prediction grid."""
    # Goal:
    # Save visual proof of training and a few image-question predictions.
    #
    # Output:
    # - `(visual_grid_path, loss_curve_path)` as strings.
    #
    # Method:
    # Use Matplotlib with `matplotlib.use("Agg")`.
    # Use `plt.plot(...)` for the loss curve.
    # Use `plt.subplots(...)` and `imshow(...)` for the prediction grid.
    #
    # Steps:
    # 1. Create `config.output_dir` with `mkdir(parents=True, exist_ok=True)`.
    # 2. Save `losses` to `projector_loss_curve.png`.
    # 3. Choose the first few examples.
    # 4. Run the model with `model.eval()` and `torch.no_grad()`.
    # 5. Put image, question, target answer, and predicted answer on the grid.
    # 6. Save `projector_prediction_grid.png`.
    # 7. Return both paths as strings.
    #
    # Check:
    # - The saved grid should be readable without rerunning training.

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)

    loss_curve_path = config.output_dir / "projector_loss_curve.png"
    visual_grid_path = config.output_dir / "projector_prediction_grid.png"

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Projector training loss")
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()

    question_vocab = build_text_vocabulary([example["question"] for example in examples])
    answer_vocab = build_answer_vocabulary([example["answer"] for example in examples])
    id_to_answer = {answer_id: answer for answer, answer_id in answer_vocab.items()}

    sample_examples = examples[: min(6, len(examples))]
    images = torch.stack([example["image"] for example in sample_examples]).to(device)
    question_vectors = encode_questions([example["question"] for example in sample_examples], question_vocab).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(images, question_vectors)
        predicted_ids = logits.argmax(dim=-1).cpu().tolist()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for axis, example, predicted_id in zip(axes, sample_examples, predicted_ids):
        image = example["image"].permute(1, 2, 0).cpu().numpy()
        predicted_answer = id_to_answer[predicted_id]

        axis.imshow(image)
        axis.set_title(
            f"Q: {example['question']}\nTarget: {example['answer']}\nPred: {predicted_answer}",
            fontsize=8,
        )
        axis.axis("off")

    for axis in axes[len(sample_examples):]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(visual_grid_path)
    plt.close(fig)

    return str(visual_grid_path), str(loss_curve_path)

    # raise NotImplementedError("Save projector training visuals.")


def run_exercise(config: TinyLLaVAProjectorConfig) -> ProjectorMetrics:
    """Run the full projector workflow."""
    # Main workflow:
    # VQA data -> vocabularies -> loader -> projector model -> train -> evaluate -> visualize.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_vqa_dataset(config)
    question_vocab = build_text_vocabulary([example["question"] for example in examples])
    answer_vocab = build_answer_vocabulary([example["answer"] for example in examples])
    dataset = TinyVQADataset(examples, question_vocab, answer_vocab)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TinyLLaVAProjectorModel(
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
    visual_grid_path, loss_curve_path = save_projector_visuals(losses, model, examples, config)

    print(f"answer_accuracy: {answer_accuracy}")

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "answer_accuracy": answer_accuracy,
        "visual_grid_path": visual_grid_path,
        "loss_curve_path": loss_curve_path,
    }


def check_metrics(metrics: ProjectorMetrics) -> bool:
    """Return True when answer accuracy is useful."""
    return metrics["final_loss"] < metrics["initial_loss"] and metrics["answer_accuracy"] > 0.70


def report_task_progress(task_name: str, task_fn, *args):
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the LLaVA projector scaffold one step at a time."""
    config = TinyLLaVAProjectorConfig()
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

    examples, ok = report_task_progress("build_vqa_dataset", build_vqa_dataset, config)
    if not count_or_stop(ok):
        return

    question_vocab, ok = report_task_progress("build_text_vocabulary", build_text_vocabulary, [example["question"] for example in examples])
    if not count_or_stop(ok):
        return

    answer_vocab, ok = report_task_progress("build_answer_vocabulary", build_answer_vocabulary, [example["answer"] for example in examples])
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("encode_questions", encode_questions, [example["question"] for example in examples[:4]], question_vocab)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinyLLaVAProjectorModel.__init__", TinyLLaVAProjectorModel, len(question_vocab), len(answer_vocab), config)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "projector answer accuracy", metrics["answer_accuracy"], "> 0.70 and loss improves")


if __name__ == "__main__":
    main()
