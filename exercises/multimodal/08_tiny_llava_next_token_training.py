"""
============================================================
EXERCISE 08: TINY LLaVA NEXT TOKEN TRAINING
============================================================

WHAT YOU ARE BUILDING
You are building a tiny LLaVA-style training loop where an image and question
lead to answer tokens.

This is the next step after the related exercise.

the related exercise used answer classification.
This exercise uses next-token training, closer to a real chat-style VLM.

WHY THIS EXERCISE EXISTS
Generative VLMs usually learn by predicting answer tokens.
The image and question are context.
The answer is what the loss trains on.

HOW THIS FITS INTO THE INTERVIEW PREP
This helps you explain how a VLM is trained to answer questions, not only choose
from fixed classes.
It also teaches why labels often use `-100` for image and prompt positions.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
---------------------------------------------------------------
real image                          small VQA v2 sample image
large vision encoder (CLIP ViT-L/14)            small CNN encoder
projector into LLM hidden size (MLP projector)  projector into tiny decoder hidden size
large pretrained LLM (Vicuna, Llama, or Qwen)   tiny decoder, optional small pretrained LM later
full text answer                    short answer tokens such as "red" or "circle"

LANGUAGE MODEL PATH
Use:
- a tiny tokenizer and tiny decoder trained here
- or, later, a small pretrained model such as `sshleifer/tiny-gpt2`

WHAT THE LOSS SEES
The loss should only train on answer tokens.

Example:

```text
visual tokens: ignored
question tokens: ignored
answer tokens: trained
```

Use `-100` for ignored labels.

VISUALS TO CREATE
Save:
- token loss curve
- sample answers before and after training

INPUTS AND OUTPUTS
Image:
- `[3, image_size, image_size]`

Text input:
- question tokens plus an answer marker

Labels:
- same length as text input
- question positions use `-100`
- answer positions use real token ids

Model output:
- token logits with shape `[batch_size, text_length, vocab_size]`

METRICS TO TRACK
- next-token training loss
- exact answer match
- sample answers before and after training

STEP-BY-STEP TODO
1. Build image-question-answer examples.
2. Build a tiny tokenizer.
3. Encode question and answer text.
4. Set labels to `-100` for prompt tokens.
5. Build an image encoder and projector.
6. Build a tiny decoder.
7. Train with next-token loss.
8. Generate short answers.
9. Evaluate exact match.
10. Save loss and sample-output visuals.

DEFINITION OF DONE
You are done when:
- labels ignore visual and question positions
- answer-token loss goes down
- sample answers improve on simple examples
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from tracemalloc import start
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
class TinyLLaVANextTokenConfig:
    """Settings for tiny next-token VLM training."""

    image_size: int = 48
    hidden_size: int = 96
    num_visual_tokens: int = 4
    max_text_length: int = 12
    batch_size: int = 16
    epochs: int = 400
    learning_rate: float = 3e-3
    max_answer_tokens: int = 4
    ignore_index: int = -100
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "multimodal_interview"
    hf_dataset_name: str = "Multimodal-Fatima/VQAv2_sample_train"
    sample_size: int = 256
    lm_mode: str = "tiny"


class VLMTextExample(TypedDict):
    """One image-question-answer training example."""

    image: torch.Tensor
    question: str
    answer: str


class VLMTokenBatch(TypedDict):
    """Batch for tiny VLM next-token training."""

    images: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor


class NextTokenMetrics(TypedDict):
    """Training summary."""

    initial_loss: float
    final_loss: float
    exact_match_accuracy: float
    loss_curve_path: str
    sample_path: str


# Learning order:
# 1. Build image-question-answer examples.
# 2. Build a tiny tokenizer.
# 3. Build labels that ignore question tokens.
# 4. Build image encoder and projector.
# 5. Build a tiny decoder.
# 6. Train with next-token loss.
# 7. Generate short answers and visualize them.


def build_vlm_text_dataset(config: TinyLLaVANextTokenConfig) -> list[VLMTextExample]:
    """Build image-question-answer examples."""
    # Where this fits:
    # This is the training data for the tiny chat-style VLM.
    # Each row gives the image, the question context, and the answer text to learn.
    #
    # Goal:
    # Return image tensors with text questions and text answers.
    #
    # Example:
    # - question: "what color is the shape ?"
    # - answer: "red"
    #
    # Dataset:
    # Use a small parquet VQA v2 sample from Hugging Face:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    #
    # Output:
    # Return a list of dictionaries with:
    # - `"image"`: tensor shaped `[3, image_size, image_size]`
    # - `"question"`: prompt text
    # - `"answer"`: target answer text
    #
    # Method:
    # Use `load_dataset` from the Hugging Face `datasets` package:
    # `load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")`.
    # Use this exact transform shape:
    # `torchvision.transforms.Compose([torchvision.transforms.Resize((config.image_size, config.image_size)), torchvision.transforms.ToTensor()])`.
    # Token ids and labels are built later, not in this helper.
    #
    # Steps:
    # 1. Load the Hugging Face split with `load_dataset(...)`.
    # 2. Inspect one row with `data[0].keys()` if you are unsure of the field names.
    # 3. For each row, read the PIL image, question text, and answer text.
    #    This dataset has `multiple_choice_answer`, which is already one cleaned answer.
    # 4. Convert the image to RGB with the PIL method `image.convert("RGB")`.
    # 5. Resize the image and convert it to a tensor with the torchvision transform.
    # 6. Lowercase and strip the question and answer strings.
    # 7. Keep short answers first, such as colors, objects, or yes/no.
    # 8. Append dictionaries with keys `"image"`, `"question"`, and `"answer"`.
    # 9. Stop once you have `config.sample_size` examples.
    #
    # Checks:
    # - Images come from the dataset and are stored as tensors.
    # - The answer should be text here, not a class id.

    from datasets import load_dataset

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config.image_size, config.image_size)),
        torchvision.transforms.ToTensor(),
    ])

    data = load_dataset(config.hf_dataset_name, split=f"train[:{config.sample_size}]")

    examples: list[VLMTextExample] = []

    for row in data:
        image = row["image"].convert("RGB")
        image_tensor = transform(image)

        question = str(row["question"]).lower().strip()
        answer = str(row["multiple_choice_answer"]).lower().strip()

        if not question or not answer:
            continue

        examples.append({
            "image": image_tensor,
            "question": question,
            "answer": answer,
        })

        if len(examples) >= config.sample_size:
            break

    return examples

    # raise NotImplementedError("Build VLM text training examples.")


def build_token_vocabulary(examples: list[VLMTextExample]) -> dict[str, int]:
    """Build token ids for questions and answers."""
    # Goal:
    # Build a tiny tokenizer vocabulary from all questions and answers.
    #
    # Input:
    # - examples with `"question"` and `"answer"` strings.
    #
    # Output:
    # - dictionary with special tokens:
    #   - `<pad>` at id 0
    #   - `<unk>` at id 1
    #   - `<answer>` for the point where the answer starts
    #   - `<eos>` for the end of the answer
    #
    # Method:
    # Use normal Python string methods: `text.lower().split()`.
    # Use a Python `set` to collect unique words, then `sorted(words)` so ids are stable.
    #
    # Steps:
    # 1. Create `token_to_id = {"<pad>": 0, "<unk>": 1, "<answer>": 2, "<eos>": 3}`.
    # 2. Make an empty Python `set` for normal words.
    # 3. For each example, split `example["question"].lower()` into words and add them.
    # 4. Do the same for `example["answer"].lower()`.
    # 5. Loop over `sorted(words)`.
    # 6. Add the word to `token_to_id` only if it is not already a special token.
    #
    # Check:
    # - The vocabulary must contain `<answer>` and `<eos>`.

    token_to_id = {"<pad>": 0, "<unk>": 1, "<answer>": 2, "<eos>": 3}
    words: set[str] = set()

    for example in examples:
        words.update(example["question"].lower().split())
        words.update(example["answer"].lower().split())

    for word in sorted(words):
        if word not in token_to_id:
            token_to_id[word] = len(token_to_id)

    return token_to_id

    # raise NotImplementedError("Build tiny tokenizer vocabulary.")


def encode_prompt_and_answer(example: VLMTextExample, token_to_id: dict[str, int], config: TinyLLaVANextTokenConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return input ids and labels for one VLM example."""
    # Where this fits:
    # This is the most important data-format helper in the exercise.
    # It decides which tokens train the model and which tokens are only context.
    #
    # Goal:
    # Build text input ids and next-token labels.
    #
    # Important:
    # - question tokens are context
    # - answer tokens are targets
    # - labels for question tokens should be `config.ignore_index`
    #
    # Output:
    # - `input_ids`: `[max_text_length]`
    # - `labels`: `[max_text_length]`
    #
    # Method:
    # Use a simple format:
    # `question tokens + <answer> + answer tokens + <eos>`.
    # Use `torch.full((config.max_text_length,), config.ignore_index, dtype=torch.long)`
    # for labels because most positions should be ignored.
    # Use `torch.full((config.max_text_length,), pad_id, dtype=torch.long)`
    # for input ids, then copy real ids into the front.
    #
    # Steps:
    # 1. Convert question words to ids with `token_to_id.get(word, token_to_id["<unk>"])`.
    # 2. Convert answer words the same way.
    # 3. Build `full_ids = question_ids + [answer_marker_id] + answer_ids + [eos_id]`.
    # 4. Truncate `full_ids` to `config.max_text_length`.
    # 5. Create `input_ids` filled with the pad id.
    # 6. Copy the truncated ids into the start of `input_ids`.
    # 7. Create `labels` filled with `config.ignore_index`.
    # 8. Find the first answer token position: `len(question_ids) + 1`.
    # 9. Shift the answer targets one position left in `labels`.
    #    The label at `<answer>` should be the first answer token.
    #    The label at the first answer token should be the next answer token or `<eos>`.
    # 10. Leave question and pad positions as `config.ignore_index`.
    #
    # Small example:
    # - question: `"what color"`
    # - answer: `"red"`
    # - input might be `[what, color, <answer>, red, <eos>, <pad>]`
    # - labels might be `[-100, -100, red, <eos>, -100, -100]`
    #
    # Check:
    # - `input_ids.shape == labels.shape == (config.max_text_length,)`

    question = example["question"]
    answer = example["answer"]

    unk_id = token_to_id["<unk>"]
    pad_id = token_to_id["<pad>"]
    answer_markerid = token_to_id["<answer>"]
    eos_id = token_to_id["<eos>"]

    question_token_ids = [token_to_id.get(word, unk_id) for word in question.lower().strip().split()]
    answer_token_ids = [token_to_id.get(word, unk_id) for word in answer.lower().strip().split()]

    full_ids = question_token_ids + [answer_markerid] + answer_token_ids + [eos_id]

    full_ids = full_ids[:config.max_text_length]    # no padding yet
    
    input_ids = torch.full(size=[config.max_text_length,], fill_value=pad_id, dtype=torch.long) #padding
    input_ids[:len(full_ids)] = torch.tensor(full_ids, dtype=torch.long)

    labels = torch.full(size=[config.max_text_length,], fill_value=config.ignore_index, dtype=torch.long) # -100

    answer_start_pos = len(question_token_ids) + 1 # first answer token position
    answer_end_pos = len(full_ids)
    # Shift targets left for next-token training:
    # at `<answer>`, predict the first answer token; at each answer token, predict the next one.
    labels[answer_start_pos - 1 : answer_end_pos - 1] = input_ids[answer_start_pos:answer_end_pos]

    return input_ids, labels
    #raise NotImplementedError("Encode one question-answer pair with ignored prompt labels.")


class TinyVLMNextTokenDataset(Dataset):
    """Dataset for next-token VLM training."""

    def __init__(self, examples: list[VLMTextExample], token_to_id: dict[str, int], config: TinyLLaVANextTokenConfig):
        self.examples = examples
        self.token_to_id = token_to_id
        self.config = config

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> VLMTokenBatch:
        example = self.examples[index]
        input_ids, labels = encode_prompt_and_answer(example, self.token_to_id, self.config)
        return {"images": example["image"], "input_ids": input_ids, "labels": labels}


class TinyVLMImageEncoder(nn.Module):
    """Small CNN that turns one image into one vector."""

    def __init__(self, config: TinyLLaVANextTokenConfig):
        super().__init__()
        # Goal:
        # Build the image part of the tiny VLM.
        # This should turn each image into one vector with width `config.hidden_size`.
        #
        # Input:
        # - image batch shaped `[batch_size, 3, image_size, image_size]`
        #
        # Output:
        # - image vectors shaped `[batch_size, hidden_size]`
        #
        # Method:
        # Use a small CNN with PyTorch layers from `torch.nn`.
        # A simple stack is enough:
        # - `nn.Conv2d(3, 16, kernel_size=3, padding="same")`
        # - `nn.ReLU()`
        # - `nn.MaxPool2d(kernel_size=2)`
        # - `nn.Conv2d(16, 32, kernel_size=3, padding="same")`
        # - `nn.ReLU()`
        # - `nn.AdaptiveAvgPool2d((1, 1))`
        # - flatten
        # - `nn.Linear(32, config.hidden_size)`
        #
        # Steps:
        # 1. Save `config` if you need it in `forward`.
        # 2. Build the CNN layers.
        # 3. Build the final linear layer.
        #
        # Check:
        # - Forward should return `[batch_size, config.hidden_size]`.

        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding="same")
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)

        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.adaptive_avgpool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.linear = nn.Linear(in_features = 32, out_features=config.hidden_size)

        #raise NotImplementedError("Build tiny VLM image encoder.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into one vector each."""
        # Steps:
        # 1. Run the images through the CNN layers.
        # 2. Flatten `[B, 32, 1, 1]` to `[B, 32]` with `torch.flatten(x, start_dim=1)`.
        # 3. Run the final linear layer.
        # 4. Return image vectors shaped `[B, hidden_size]`.

        x = nn.functional.relu(self.conv2d_1(images))   # [B, 16, 48, 48]
        x = self.maxpool2d(x)                           # [B, 16, 24, 24]

        x = nn.functional.relu(self.conv2d_2(x))        # [B, 32, 24, 24]
        x = self.adaptive_avgpool2d(x)                  # [B, 32, 1, 1]

        x = torch.flatten(x, start_dim=1)               # [B, 32]

        features = self.linear(x)                       # [B, hidden_size]

        return features
        #raise NotImplementedError("Encode images.")


class TinyVLMTextEncoder(nn.Module):
    """Token embedding layer for question and answer ids."""

    def __init__(self, vocab_size: int, config: TinyLLaVANextTokenConfig):
        super().__init__()
        # Goal:
        # Build the text input part of the tiny VLM.
        # This turns token ids into vectors the decoder can read.
        #
        # Input:
        # - token ids shaped `[batch_size, text_length]`
        #
        # Output:
        # - token embeddings shaped `[batch_size, text_length, hidden_size]`
        #
        # Method:
        # Use `nn.Embedding(vocab_size, config.hidden_size)`.
        # `vocab_size` is the number of possible token ids.
        # `config.hidden_size` is the width of each token vector.
        #
        # Steps:
        # 1. Save `config` if you need it.
        # 2. Build `self.token_embedding = nn.Embedding(vocab_size, config.hidden_size)`.
        #
        # Check:
        # - Forward should not apply softmax.
        # - It should only return embeddings.

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.hidden_size)

        #raise NotImplementedError("Build tiny VLM text encoder.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids."""
        # Steps:
        # 1. Pass `input_ids` to the embedding layer.
        # 2. Return `[batch_size, text_length, hidden_size]`.

        text_embeddings = self.embedding(input_ids)     # [B, text_length, hidden_size]
        return text_embeddings

        #raise NotImplementedError("Embed input token ids.")


class TinyVLMTransformerDecoder(nn.Module):
    """Small sequence model that reads visual tokens plus text tokens."""

    def __init__(self, config: TinyLLaVANextTokenConfig):
        super().__init__()
        # Goal:
        # Build the tiny decoder that mixes image context with text tokens.
        #
        # Input:
        # - combined embeddings shaped `[batch_size, sequence_length, hidden_size]`
        # - sequence is `visual tokens + text tokens`
        #
        # Output:
        # - decoder outputs shaped `[batch_size, sequence_length, hidden_size]`
        #
        # Method:
        # Use PyTorch `nn.TransformerEncoderLayer(...)` and `nn.TransformerEncoder(...)`
        # as a simple decoder-like block for this exercise.
        # Use:
        # `nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=4, batch_first=True)`
        # Then wrap it with:
        # `nn.TransformerEncoder(encoder_layer, num_layers=2)`
        #
        # Why this is okay here:
        # A real LLaVA uses a causal language model, so it cannot look at future tokens.
        # This toy exercise uses a small transformer block to practice the image-text
        # shape path first. Later exercises can add stricter causal masking.
        #
        # Steps:
        # 1. Build one `nn.TransformerEncoderLayer`.
        # 2. Build `self.transformer = nn.TransformerEncoder(...)`.
        #
        # Check:
        # - Keep `batch_first=True` so tensors stay `[B, sequence_length, hidden_size]`.

        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model = config.hidden_size, nhead=2, batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer=self.transformerEncoderLayer, num_layers=2)

        #raise NotImplementedError("Build tiny VLM transformer decoder.")

    def forward(self, combined_embeddings: torch.Tensor) -> torch.Tensor:
        """Run the combined visual and text embeddings through the transformer."""
        # Steps:
        # 1. Pass `combined_embeddings` through the transformer.
        # 2. Return outputs with the same first two dimensions.

        #combined_embeddings is [B, sequence_length, hidden_size]

        return self.transformerEncoder(combined_embeddings)         # return same shape [B, sequence_length, hidden_size]

        #raise NotImplementedError("Decode visual and text embeddings.")


class TinyLLaVADecoder(nn.Module):
    """Tiny image-conditioned decoder."""

    def __init__(self, vocab_size: int, config: TinyLLaVANextTokenConfig):
        super().__init__()
        # Goal:
        # Build the full tiny VLM by connecting the smaller modules above.
        #
        # Parts:
        # - `TinyVLMImageEncoder`: turns images into one image vector.
        # - `self.visual_projector`: turns one image vector into several visual tokens.
        # - `TinyVLMTextEncoder`: turns input token ids into token embeddings.
        # - `TinyVLMTransformerDecoder`: reads visual tokens plus text tokens.
        # - `self.output_head`: turns decoder outputs into token logits.
        #
        # Suggested attributes:
        # - `self.image_encoder = TinyVLMImageEncoder(config)`
        # - `self.visual_projector = nn.Linear(config.hidden_size, config.num_visual_tokens * config.hidden_size)`
        # - `self.text_encoder = TinyVLMTextEncoder(vocab_size, config)`
        # - `self.decoder = TinyVLMTransformerDecoder(config)`
        # - `self.output_head = nn.Linear(config.hidden_size, vocab_size)`
        #
        # Method:
        # Keep this class small.
        # The helper classes do the image, text, and sequence work.
        # This class wires them together and owns the final output head.
        #
        # Steps:
        # 1. Save `config` on `self`.
        # 2. Build `self.image_encoder`.
        # 3. Build `self.visual_projector` so that vector can become several visual tokens.
        # 4. Build `self.text_encoder`.
        # 5. Build `self.decoder`.
        # 6. Build `self.output_head`.
        #
        # Checks:
        # - The output head width must equal `vocab_size`.
        # - The visual projector must produce `num_visual_tokens * hidden_size` values.

        self.config = config

        self.image_encoder = TinyVLMImageEncoder(config)
        self.visual_projector = nn.Linear(config.hidden_size, config.num_visual_tokens * config.hidden_size)
        self.text_encoder = TinyVLMTextEncoder(vocab_size, config)
        self.decoder = TinyVLMTransformerDecoder(config)
        self.output_head = nn.Linear(config.hidden_size, vocab_size)

        #raise NotImplementedError("Build tiny image-conditioned decoder.")

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for text positions."""
        # Output shape:
        # - `[batch_size, text_length, vocab_size]`
        #
        # The model may internally prepend visual tokens, but return logits aligned
        # only with text token positions so the labels can be compared directly.
        #
        # Method:
        # Use the helper modules from `__init__`.
        # Use `.view(batch_size, config.num_visual_tokens, config.hidden_size)` or
        # `.reshape(...)` to turn the projected vector into visual tokens.
        # Use `torch.cat([visual_tokens, text_embeddings], dim=1)` to prepend image tokens.
        #
        # Steps:
        # 1. Read `batch_size` and `text_length` from `input_ids.shape`.
        # 2. Encode images with `self.image_encoder(images)`.
        # 3. Project that vector with `self.visual_projector`.
        # 4. Reshape projected values to `[batch_size, num_visual_tokens, hidden_size]`.
        # 5. Embed `input_ids` with `self.text_encoder(input_ids)`.
        # 6. Concatenate visual tokens before text embeddings with `torch.cat(..., dim=1)`.
        # 7. Run the combined embeddings through `self.decoder`.
        # 8. Drop the visual-token outputs and keep the last `text_length` positions.
        # 9. Pass those text outputs through `self.output_head`.
        #
        # Check:
        # - Return shape must match labels: `[batch_size, text_length, vocab_size]`.

        batch_size, text_length = input_ids.shape           # input_ids: [batch_size, sequence_length]
        
        visual_features = self.image_encoder(images)        # [batch_size, hidden_size]

        projected_features = self.visual_projector(visual_features)  # [batch_size, num_visual_tokens * hidden_size]
        projected_features = torch.reshape(projected_features,       # [batch_size, num_visual_tokens, hidden_size]
                                            shape=[batch_size, self.config.num_visual_tokens, self.config.hidden_size])

        text_embeddings = self.text_encoder(input_ids)      # [batch_size, sequence_length, hidden_size]

        combined_embeddings = torch.cat([projected_features, text_embeddings], dim=1)  # [batch_size, sequence_length+num_visual_tokens, hidden_size]

        decoder_output = self.decoder(combined_embeddings)
        decoder_output = decoder_output[:, self.config.num_visual_tokens:, :]           # [B, text_length, hidden_size]

        output = self.output_head(decoder_output)                   # [B, text_length, vocab_size]

        return output
        #raise NotImplementedError("Run image-conditioned next-token forward pass.")


def compute_next_token_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Compute next-token cross entropy with ignored labels."""
    # Goal:
    # Compute loss only on answer token positions.
    #
    # Input:
    # - `logits`: `[batch_size, text_length, vocab_size]`
    # - `labels`: `[batch_size, text_length]`
    # - ignored positions contain `ignore_index`, usually `-100`
    #
    # Method:
    # Use `torch.nn.functional.cross_entropy`, usually imported as `F.cross_entropy`,
    # or use the class `nn.CrossEntropyLoss(ignore_index=ignore_index)`.
    # The call shape is:
    # `F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)`.
    #
    # Steps:
    # 1. Read `vocab_size = logits.shape[-1]`.
    # 2. Flatten logits with `logits.reshape(-1, vocab_size)`.
    # 3. Flatten labels with `labels.reshape(-1)`.
    # 4. Pass both flattened tensors to cross entropy.
    # 5. Return the scalar loss tensor.
    #
    # Check:
    # - Question-label positions should not affect the loss.

    batch_size, text_length, vocab_size = logits.shape
    flattened_logits = logits.reshape(shape=[batch_size*text_length, vocab_size])
    flattened_labels = labels.reshape(shape=[batch_size*text_length])

    loss = torch.nn.functional.cross_entropy(flattened_logits, flattened_labels, ignore_index=ignore_index)

    return loss
    #raise NotImplementedError("Compute next-token loss with ignored prompt labels.")


def train_one_epoch(model: TinyLLaVADecoder, loader: DataLoader, optimizer: torch.optim.Optimizer, config: TinyLLaVANextTokenConfig) -> float:
    """Train one epoch."""
    # Goal:
    # Train the image-conditioned decoder for one pass over the data.
    #
    # Method:
    # Use normal PyTorch training calls:
    # `model.train()`, `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.
    #
    # Steps:
    # 1. Call `model.train()` once before the loop.
    # 2. For each batch, move `batch["images"]`, `batch["input_ids"]`, and `batch["labels"]` to `device`.
    # 3. Call `logits = model(images, input_ids)`.
    # 4. Call `loss = compute_next_token_loss(logits, labels, config.ignore_index)`.
    # 5. Clear old gradients with `optimizer.zero_grad()`.
    # 6. Backprop with `loss.backward()`.
    # 7. Update weights with `optimizer.step()`.
    # 8. Add `loss.item()` to a running total.
    # 9. Return the mean loss as a Python float.
    #
    # Check:
    # - Return mean loss as a Python float.

    total_loss = 0
    total_examples = 0

    model.train()

    for data in loader:
        images = data["images"].to(device)
        input_ids = data["input_ids"].to(device)
        labels = data["labels"].to(device)

        optimizer.zero_grad()

        logits = model(images, input_ids)
        loss = compute_next_token_loss(logits, labels, ignore_index=config.ignore_index)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples+=batch_size

    return total_loss/total_examples

    #raise NotImplementedError("Train one next-token VLM epoch.")


def generate_answer(model: TinyLLaVADecoder, image: torch.Tensor, question: str, token_to_id: dict[str, int], config: TinyLLaVANextTokenConfig) -> str:
    """Generate a short answer."""
    # Goal:
    # Generate a small answer from one image and one question.
    #
    # Method:
    # Use `model.eval()` and `torch.no_grad()` because generation should not train.
    # Use `torch.argmax(next_token_logits, dim=-1)` for greedy decoding.
    #
    # Steps:
    # 1. Build the starting token ids from `question.lower().split()` plus `<answer>`.
    # 2. Pad or crop to `config.max_text_length`.
    # 3. Add a batch dimension to the image with `image.unsqueeze(0)`.
    # 4. Add a batch dimension to token ids with `input_ids.unsqueeze(0)`.
    # 5. Run the model and read logits for the current last non-pad position.
    # 6. Pick the next token id with `torch.argmax(...)`.
    # 7. Stop if the token is `<eos>`.
    # 8. Append normal answer tokens to a Python list of words.
    # 9. Repeat until you hit `config.max_answer_tokens`.
    # 10. Join the words with `" ".join(words)`.
    #
    # Check:
    # - The returned value should be a string, not token ids.

    id_to_token = {token_id: token for token, token_id in token_to_id.items()}
    unk_id = token_to_id["<unk>"]
    pad_id = token_to_id["<pad>"]
    answer_marker_id = token_to_id["<answer>"]
    eos_id = token_to_id["<eos>"]

    question_ids = [token_to_id.get(word, unk_id) for word in question.lower().strip().split()]
    question_ids = question_ids[: config.max_text_length - 1]
    token_ids = question_ids + [answer_marker_id]
    generated_words: list[str] = []

    model.eval()
    with torch.no_grad():
        for _ in range(config.max_answer_tokens):
            input_ids = torch.full((config.max_text_length,), pad_id, dtype=torch.long)  # [max_text_length]
            current_ids = token_ids[:config.max_text_length]
            input_ids[:len(current_ids)] = torch.tensor(current_ids, dtype=torch.long)

            batch_images = image.unsqueeze(0).to(device)             # [1, 3, image_size, image_size]
            batch_input_ids = input_ids.unsqueeze(0).to(device)      # [1, max_text_length]

            logits = model(batch_images, batch_input_ids)            # [1, max_text_length, vocab_size]

            #read logits for the current last non-pad position.
            next_position = min(len(current_ids) - 1, config.max_text_length - 1)
            next_token_logits = logits[0, next_position, :]             # [vocab_size]
            next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())

            if next_token_id == eos_id:
                break

            next_token = id_to_token.get(next_token_id, "<unk>")
            if next_token not in {"<pad>", "<unk>", "<answer>", "<eos>"}:
                generated_words.append(next_token)

            if len(token_ids) >= config.max_text_length:
                break
            token_ids.append(next_token_id)

    return " ".join(generated_words)

    # raise NotImplementedError("Generate a short answer from image and question.")


def evaluate_exact_match(model: TinyLLaVADecoder, examples: list[VLMTextExample], token_to_id: dict[str, int], config: TinyLLaVANextTokenConfig) -> float:
    """Return exact answer match accuracy."""
    # Goal:
    # Count how often generated answers exactly match target answers.
    #
    # Method:
    # Use the local `generate_answer(...)` helper.
    # Use Python string methods `.strip().lower()` before comparing strings.
    #
    # Steps:
    # 1. Start `correct = 0`.
    # 2. Loop over the examples.
    # 3. Generate a prediction with `generate_answer(...)`.
    # 4. Normalize prediction and target with `.strip().lower()`.
    # 5. Add 1 to `correct` when the strings match exactly.
    # 6. Return `correct / len(examples)`.
    #
    # Check:
    # - Return a float between `0.0` and `1.0`.

    correct = 0

    for example in examples:
        prediction = generate_answer(model, example["image"], example["question"], token_to_id, config)
        target = example["answer"]

        if prediction.strip().lower() == target.strip().lower():
            correct += 1

    return correct / len(examples)

    # raise NotImplementedError("Evaluate sample answers.")


def save_next_token_visuals(losses: list[float], samples: list[str], config: TinyLLaVANextTokenConfig) -> tuple[str, str]:
    """Save loss curve and sample answers."""
    # Goal:
    # Save training loss and generated answer examples.
    #
    # Output:
    # - `(loss_curve_path, sample_path)` as strings.
    #
    # Method:
    # Use `config.output_dir.mkdir(parents=True, exist_ok=True)` from `pathlib.Path`.
    # Use Matplotlib: `matplotlib.use("Agg")`, `plt.figure()`, `plt.plot(...)`,
    # `plt.savefig(path)`, and `plt.close()`.
    # Use `Path.write_text(...)` to save generated answer samples.
    #
    # Steps:
    # 1. Create `config.output_dir`.
    # 2. Build a path such as `config.output_dir / "203_next_token_loss.png"`.
    # 3. Plot `losses` as a line chart with epoch on the x-axis.
    # 4. Save and close the plot.
    # 5. Build a sample text path.
    # 6. Join `samples` with newline characters.
    # 7. Write the text with `sample_path.write_text(text, encoding="utf-8")`.
    # 8. Return both paths as strings.
    #
    # Check:
    # - The sample file should let you inspect whether answers improved.

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)

    loss_curve_path = config.output_dir / "203_next_token_loss.png"
    sample_path = config.output_dir / "203_next_token_samples.txt"

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Next-token training loss")
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()

    sample_text = "\n".join(samples)
    sample_path.write_text(sample_text, encoding="utf-8")

    return str(loss_curve_path), str(sample_path)

    # raise NotImplementedError("Save next-token training visuals.")


def run_exercise(config: TinyLLaVANextTokenConfig) -> NextTokenMetrics:
    """Run full next-token VLM training."""
    # Main workflow:
    # VQA text data -> token ids and labels -> loader -> decoder -> train -> generate -> visualize.
    set_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_vlm_text_dataset(config)
    token_to_id = build_token_vocabulary(examples)
    dataset = TinyVLMNextTokenDataset(examples, token_to_id, config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TinyLLaVADecoder(vocab_size=len(token_to_id), config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    for i in range(config.epochs):
        loss = train_one_epoch(model, loader, optimizer, config)
        losses.append(loss)
        print(f"Epoch {i+1} -> loss: {loss}")

    exact_match_accuracy = evaluate_exact_match(model, examples, token_to_id, config)
    samples = []
    for example in examples[:8]:
        prediction = generate_answer(model, example["image"], example["question"], token_to_id, config)
        samples.append(
            f"Question: {example['question']}\n"
            f"Target answer: {example['answer']}\n"
            f"Generated answer: {prediction}\n"
        )
    loss_curve_path, sample_path = save_next_token_visuals(losses, samples, config)

    print(f"exact_match_accuracy: {exact_match_accuracy}")

    print(samples)

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "exact_match_accuracy": exact_match_accuracy,
        "loss_curve_path": loss_curve_path,
        "sample_path": sample_path,
    }


def check_metrics(metrics: NextTokenMetrics) -> bool:
    return metrics["final_loss"] < metrics["initial_loss"] and metrics["exact_match_accuracy"] > 0.50


def report_task_progress(task_name: str, task_fn, *args):
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the tiny next-token VLM scaffold one step at a time."""
    config = TinyLLaVANextTokenConfig()
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

    examples, ok = report_task_progress("build_vlm_text_dataset", build_vlm_text_dataset, config)
    if not count_or_stop(ok):
        return

    token_to_id, ok = report_task_progress("build_token_vocabulary", build_token_vocabulary, examples)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("encode_prompt_and_answer", encode_prompt_and_answer, examples[0], token_to_id, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("TinyLLaVADecoder.__init__", TinyLLaVADecoder, len(token_to_id), config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("compute_next_token_loss", compute_next_token_loss, torch.zeros(2, 3, len(token_to_id)), torch.zeros(2, 3, dtype=torch.long), config.ignore_index)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_exercise", run_exercise, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "next-token exact match", metrics["exact_match_accuracy"], "> 0.50 and loss improves")


if __name__ == "__main__":
    main()
