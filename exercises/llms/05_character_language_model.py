"""
============================================================
EXERCISE 05: CHARACTER LANGUAGE MODEL
============================================================

WHAT YOU ARE BUILDING
You are building a small character-level language model.

This file is meant to be easier than the mini GPT file that comes next.
A clear default path is:
- character vocabulary
- embedding layer
- one GRU
- linear language-model head

The model reads a short context window of characters and predicts the next character at each position.
A character vocabulary is just the list of characters the model knows about, like letters, spaces, and punctuation.
An embedding layer turns each character id into a learned vector of numbers.
A GRU is a sequence model that reads one step at a time and keeps a running memory.

Beginner intuition:
- the model reads a short piece of text and tries to guess the next character
- after it sees `h e l l`, a good guess might be `o`
- after it sees `q`, a good guess might be `u`
- the GRU helps because it keeps a memory of what came earlier in the sequence

WHY THIS EXERCISE EXISTS
Language modeling is a very important idea.
It teaches you how next-token prediction works.

This exercise teaches:
- how to split a text corpus into train, val, and test parts
- how to build next-token input and target windows
- what a vocabulary is at the character level
- what perplexity means in plain words

DATASET
Use Tiny Shakespeare or another very small public text corpus.

Keep one clear default split rule:
- first `60%` of the text for train
- next `20%` of the text for val
- final `20%` of the text for test

For language modeling, keeping the text in order is a good simple default.
Do not shuffle the raw text before splitting it.

HOW TO LOAD OR DOWNLOAD THE DATASET
Simple path:
- download a tiny text file if it is missing
- read the whole file as one Python string
- build a character vocabulary from the training text only
- turn characters into integer ids
- create input windows and target windows

INPUTS AND OUTPUTS
Input batch:
- `input_ids`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size, context_length]`
- `target_ids`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size, context_length]`
  - meaning: the next character id at each position

Example batch:
- `input_ids`: `[32, 64]`
- `target_ids`: `[32, 64]`

What those shapes mean:
- `input_ids: [32, 64]` means 32 training examples and 64 character positions in each input window
- `target_ids: [32, 64]` means one next-character target for each of those input positions

Model output:
- type: `torch.Tensor`
- shape: `[batch_size, context_length, vocab_size]`
- meaning: one raw logit vector per time step

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Default path for this easier language model:
- character embeddings
- one GRU
- linear head to vocabulary logits

This is simpler than a transformer decoder.
That is why it comes first.

More plain words:
- a `context window` means the short chunk of text the model is allowed to read at one time
- `logits` are raw prediction scores before converting them into probabilities
- the `language-model head` is the final layer that turns hidden features into one score per possible next character

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

Important:
- return raw logits
- do not apply softmax before the loss
- for language modeling, flatten the logits and targets if that makes the loss call easier

METRICS TO TRACK
Track at least:
- `train_loss`
  - type: `float`
  - meaning: mean training loss over the whole final training epoch
- `val_loss`
  - type: `float`
  - meaning: mean validation loss over the whole validation loader
- `val_perplexity`
  - type: `float`
  - meaning: `exp(val_loss)`
  - this is the mean validation perplexity summary for the whole validation loader

Perplexity, in plain words:
- lower is better
- `1.0` would mean perfect next-character prediction
- a very high perplexity means the model is still very unsure
- you can think of perplexity as "how confused is the model on average?"

Another simple way to think about perplexity:
- it is like asking how many next-character choices the model is juggling in its head
- if perplexity is close to `10`, the model is acting a bit like it has about 10 reasonable choices on average
- if perplexity gets smaller, the model is becoming less confused

If you pass the test loader to `evaluate(...)`:
- `loss` means mean test loss over the whole test loader
- `perplexity` means `exp(test_loss)` for the whole test loader

TARGET RESULT RANGE
On a tiny public corpus with a small GRU, a rough validation loss around `1.5` to `2.5` can already be a useful sign.
That usually means validation perplexity is roughly in the `4.5` to `12.5` range.

You do not need perfect text.
You only need a sensible small-model result.

COMMON MISTAKES
- building targets that are the same as the inputs instead of one step ahead
- using float token ids instead of `torch.long`
- applying softmax before `CrossEntropyLoss`
- mixing up sequence length and vocabulary size
- reporting train loss as if it were validation loss

STEP-BY-STEP TODO
1. Load the text file.
2. Split the raw text into `60 / 20 / 20` train, val, and test sections in order.
3. Build the character vocabulary from the training text only.
4. Turn characters into integer ids.
5. Create next-character input and target windows.
6. Build the GRU language model.
7. Train for several epochs.
8. Evaluate on validation text.
9. Generate a short sample.

What each step is doing in simple words:
1. read the source text
2. keep train, val, and test in text order
3. list the characters the model is allowed to know
4. turn each character into an integer id
5. build input windows and one-step-ahead target windows
6. make a small sequence model that reads the window
7. train it to predict the next character
8. check whether it generalizes to unseen text
9. see what kind of text it can produce

STRETCH GOALS
- compare GRU and LSTM
- compare different context lengths
- sample with different temperatures

DEFINITION OF DONE
You are done when:
- your inputs and targets both have shape `[batch_size, context_length]`
- your model returns raw logits with shape `[batch_size, context_length, vocab_size]`
- you can explain next-token prediction and perplexity in plain words
- the validation loss is in a sensible range
"""

from __future__ import annotations

from dataclasses import dataclass
from os import read
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.metrics import perplexity_from_loss
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"

class CharacterLanguageModelConfig:
    """Settings for the character language-model exercise."""

    data_path: str = str(BOOTCAMP_ROOT / "data" / "tiny_shakespeare.txt")
    context_length: int = 64
    embedding_dim: int = 96
    hidden_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-4 #1e-3
    epochs: int = 8
    random_state: int = 42
    target_val_loss: float = 2.5
    sample_length: int = 200

    vocab_size: int | None = None
    str_to_int: dict | None = None
    int_to_str: dict | None = None




def describe_language_model_shapes(config: CharacterLanguageModelConfig) -> dict[str, object]:
    """Return the expected shape flow for the character language model.

    Return a dict with exactly these keys:
    - `default_split`
    - `input_shape`
    - `target_shape`
    - `logit_shape`
    - `metric_note`
    """
    return {
        "default_split": {"train": 0.60, "val": 0.20, "test": 0.20},
        "input_shape": (config.batch_size, config.context_length),
        "target_shape": (config.batch_size, config.context_length),
        "logit_shape": (config.batch_size, config.context_length, "vocab_size"),
        "metric_note": "val_perplexity means exp(val_loss) over the whole validation loader.",
    }

        #raise NotImplementedError("Run the character language model and return raw logits.")


# Load the raw text corpus.
# Return:
# - type: `str`
# - meaning: the full text corpus as one Python string
# If the file is missing, you may download it first, then return the loaded text.
def load_raw_text(config: CharacterLanguageModelConfig) -> str:

    with open(file=f"{PROJECT_ROOT}/data/input.txt", mode="r", encoding="utf-8") as f:
        text = f.read()

    print("length of dataset in characters: ", len(text))
    #print(text[:100])

    return text

    #raise NotImplementedError("Return the full Tiny Shakespeare text as one Python string.")


# Split the raw text into train, validation, and test text sections.
# Return a dict with exactly these keys:
# - `train_text`
# - `val_text`
# - `test_text`
# Use a contiguous 60% train / 20% val / 20% test split.
# Important beginner rule:
# - split by position in the text, not by random shuffle
# - this means the first 60% of characters go to train, the next 20% go to val, and the last 20% go to test
# Intuition:
# - language is ordered
# - keeping the text in order makes the task feel more natural
# - it also avoids mixing together pieces of text in a confusing way
def split_text_corpus(raw_text: str) -> dict[str, str]:    
    
    num_train, num_val = int(len(raw_text) * 0.6), int(len(raw_text) * 0.2)

    train_text = raw_text[:num_train]
    val_text = raw_text[num_train:num_train+num_val]
    test_text = raw_text[num_train+num_val:]

    return {"train_text": train_text, "val_text": val_text, "test_text": test_text}


    #raise NotImplementedError("Return the raw text split into contiguous train, val, and test sections.")


class CharTokenDataset(torch.utils.data.Dataset):
    def __init__(self, text, context_length, vocab_dict):
        super().__init__()

        self.text = text
        self.text_length = len(text)
        self.context_length = context_length
        self.vocab_dict = vocab_dict
        self.encode = lambda s: [self.vocab_dict[c] for c in s]

    def __len__(self):
        return self.text_length - self.context_length

    def __getitem__(self, index):
        text_portion = self.text[index: index + self.context_length]
        target_portion = self.text[index + 1: index + 1 + self.context_length]

        input_ids = torch.tensor(self.encode(text_portion), dtype=torch.long)
        target_ids = torch.tensor(self.encode(target_portion), dtype = torch.long)

        return {"input_ids":input_ids, "target_ids":target_ids}



@dataclass

# Build the train, validation, and test dataloaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Return type:
# - `dict[str, DataLoader]`
# Each batch should be a dict with exactly these keys:
# - `input_ids`
# - `target_ids`
# Both tensors should have shape `[batch_size, context_length]`.
# Both tensors should use dtype `torch.long`.
# Beginner-friendly window idea:
# - one training example is a short text window
# - `input_ids` is the current window
# - `target_ids` is the same window shifted one character to the left
# - example:
#   - input:  `h e l l`
#   - target: `e l l o`
# Why this setup teaches next-token prediction:
# - every position asks the model to predict the very next character
# - the input and target are almost the same, but shifted by one step
# - that one-step shift is the whole point of language modeling
# A clean way to do this is to make a small Dataset class.
# That class should inherit from `torch.utils.data.Dataset`.
# The Dataset should store the encoded text and the context length.
# `__len__` should return how many short windows can be made from the text.
# `__getitem__` should return one dict with exactly these keys:
# - `"input_ids"`: characters from position i to i + context_length
# - `"target_ids"`: the same text shifted one character forward
# For example, with text "hello" and context length 3:
# - input is  "hel"
# - target is "ell"
def load_dataloaders(config: CharacterLanguageModelConfig) -> dict[str, DataLoader]:

    raw_text = load_raw_text(config)
    text_dict = split_text_corpus(raw_text)

    train_text = text_dict["train_text"]

    vocabulary = sorted(list(set(train_text + text_dict["val_text"])))
    vocab_size = len(vocabulary)

    print(f"VOCAB SIZE: {vocab_size}")
    print(f"Vocabulary: {''.join(vocabulary)}")

    str_to_int = {c:i for i, c in enumerate(vocabulary)}
    int_to_str = {i:c for i ,c in enumerate(vocabulary)}
    # encode = lambda s: [str_to_int[c] for c in s]
    # decode = lambda l: "".join(int_to_str[i] for i in l)

    # print(f"Encoded string: {encode('Hello World!')}")
    # print(f"Decoded string: {decode(encode('Hello World!'))}")

    train_ds = CharTokenDataset(text=train_text, context_length=config.context_length, vocab_dict=str_to_int)
    val_ds = CharTokenDataset(text=text_dict["val_text"], context_length=config.context_length, vocab_dict=str_to_int)
    test_ds = CharTokenDataset(text=text_dict["test_text"], context_length=config.context_length, vocab_dict=str_to_int)

    config.vocab_size = vocab_size
    config.str_to_int = str_to_int
    config.int_to_str = int_to_str

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}



class CharacterLanguageModel(nn.Module):
    """Simple character language-model stub.

    Recommended default path:
    - embedding layer
    - one GRU
    - linear head to vocabulary logits
    """

    def __init__(self, vocab_size: int, config: CharacterLanguageModelConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config
        # Beginner-friendly build order:
        # 1. character embedding layer
        # 2. one GRU
        # 3. linear head to vocabulary logits
        #
        # Simple default objects:
        # - `nn.Embedding(vocab_size, embedding_dim)`
        # - `nn.GRU(embedding_dim, hidden_dim, batch_first=True)`
        # - `nn.Linear(hidden_dim, vocab_size)`
        #
        # Intuition:
        # - the embedding layer changes character ids into vectors the model can work with
        # - the GRU reads those vectors one step at a time and carries forward a memory
        # - the linear head turns each hidden state into one score per possible next character
        #
        # Important sequence idea:
        # - the dataloader gives the model a whole window, like `h e l`
        # - it does not split that into separate examples like `h`, `h e`, and `h e l`
        # - after embedding, the shape is `[batch_size, context_length, embedding_dim]`
        # - because `batch_first=True`, PyTorch reads that as:
        #   - dimension 0: which example in the batch
        #   - dimension 1: which time step in the text window
        #   - dimension 2: the vector numbers for one character
        # - so for the text window `h e l`, the recurrent layer reads:
        #   - time step 0: vector for `h`
        #   - time step 1: vector for `e`, plus memory from `h`
        #   - time step 2: vector for `l`, plus memory from `h e`
        # - that is why output position 0 has seen `h` and should predict `e`
        # - output position 1 has seen `h e` and should predict `l`
        # - output position 2 has seen `h e l` and should predict the next `l`

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.embedding_dim)

        self.lstm = nn.LSTM(input_size = config.embedding_dim, hidden_size=config.hidden_dim, batch_first=True)

        self.classifier_head = nn.Linear(in_features=config.hidden_dim, out_features=vocab_size)

        #raise NotImplementedError("Build the embedding layer, one GRU, and the vocabulary prediction head.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return raw next-character logits.

        Input:
        - `input_ids`: tensor with shape `[batch_size, context_length]`

        Return:
        - type: `torch.Tensor`
        - shape: `[batch_size, context_length, vocab_size]`
        - meaning: raw logits for the next character at every position

        Step-by-step tensor path:
        1. input ids -> embeddings
           shape: `[batch_size, context_length, embedding_dim]`
        2. embeddings -> GRU outputs
           shape: `[batch_size, context_length, hidden_dim]`
        3. GRU outputs -> linear head
           shape: `[batch_size, context_length, vocab_size]`

        What one output position means:
        - at each character position, the model makes one next-character prediction
        - so if the current input window is `h e l l`, the last position should lean toward `o`
        """

        # input_ids shape: # [batch_size, context_length]   (32, 64)

        embeddings = self.embed(input_ids)  # [batch_size, context_length, embedding_dim]  (32, 64, 96)
        lstm_output, (hidden_state, cell_state) = self.lstm(embeddings)  # lstm_output.shape [batch_size, context_length, hidden_dim]
        # lstm_output.shape (32, 64, 128)
        # hidden_state.shape (1, 64, 128)
        # cell_state.shape (1, 64, 128)

        # `lstm_output, (hidden_state, cell_state)` are the values returned by an LSTM layer. 
        # `lstm_output` contains the LSTM output at every time step, so if the input is `"h e l"`, 
        # it has one output after seeing `h`, one after seeing `h e`, and one after seeing `h e l`. 
        # This is useful when you want a prediction at every position, like in a language model. 
        # `hidden_state` is the final hidden memory after the LSTM has read the whole sequence. 
        # For "h e l", it is like the final summary.
        # `cell_state` is another final memory value that the LSTM uses internally to remember 
        # longer-term information. For many beginner tasks, you use `lstm_output` for predictions 
        # and ignore `hidden_state` and `cell_state` unless you specifically need the 
        # final summary of the sequence.

        # For this exercise, you mostly need lstm_output, because it gives you one output for every time step. 
        # That lets your classifier head make one next-character prediction at every position.

        logits = self.classifier_head(lstm_output)  # (32, 64, 65)

        # 32 = batch size
        # 64 = number of character positions in each input window
        # 65 = vocabulary size

        # 32 text windows at once
        # 64 characters per window
        # 65 possible next characters

        # The model returns logits with shape [batch_size, context_length, vocab_size], for example (32, 64, 65). 
        # Read this as: �for 32 text windows, for 64 positions in each window, give 65 next-character scores.� 
        # The model does not give one answer for the whole window. It gives one set of scores at every 
        # character position. If the input is h e l, then the first set of scores is for what should 
        # come after h, the second set is for what should come after h e, and the third set is for what 
        # should come after h e l. The last number, 65, means there are 65 possible characters the model 
        # can choose from. So each position gets 65 scores, one score per possible next character. A bigger 
        # score means �the model thinks this character is more likely.�

        return logits


    #raise NotImplementedError("Return train, val, and test dataloaders for next-character prediction.")


# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.CrossEntropyLoss`.
def build_training_objects(config: CharacterLanguageModelConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    model = CharacterLanguageModel(vocab_size = config.vocab_size, config=config)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())

    return model, loss_fn, optimizer
    #raise NotImplementedError("Return the character language model, CrossEntropyLoss, and optimizer.")


# Run one full training epoch.
# Return a dict with exactly these keys:
# - `loss`
# - `perplexity`
# Return type:
# - `dict[str, float]`
# Both values must be Python floats.
# Metric meaning:
# - `loss` is the mean train loss over the whole training epoch
# - `perplexity` is `exp(loss)` for that full training epoch
# Important:
# - move the model, `input_ids`, and `target_ids` to the same device here
# - a simple loss path is to flatten:
#   - logits from `[batch_size, context_length, vocab_size]` to `[batch_size * context_length, vocab_size]`
#   - targets from `[batch_size, context_length]` to `[batch_size * context_length]`
# - do not return one value per batch
# - return one final epoch summary dict
# Why flattening is okay:
# - the loss only needs a big list of predictions and a matching big list of correct answers
# - flattening does not change which prediction matches which target
# - it just makes the loss call easier to write
#
# CrossEntropyLoss is easiest to use when predictions have shape:
# - `[number_of_items, number_of_classes]`
# and targets have shape:
# - `[number_of_items]`
#
# In this exercise, one "item" means one character position.
# One "class" means one possible next character.
#
# The model gives logits with shape:
# - `[batch_size, context_length, vocab_size]`
# Example:
# - `[32, 64, 65]`
#
# This means:
# - 32 text windows
# - 64 character positions in each window
# - 65 next-character scores at each position
#
# The loss function does not need to know which text window a position came from.
# It only needs each prediction matched with the correct next character.
#
# So we flatten the first two dimensions:
# - logits:  `[32, 64, 65] -> [32 * 64, 65] -> [2048, 65]`
# - targets: `[32, 64]     -> [32 * 64]     -> [2048]`
#
# Now the loss gets a simple list:
# - logits[0] should match targets[0]
# - logits[1] should match targets[1]
# - logits[2] should match targets[2]
# - and so on
#
# Flattening does not change what each prediction means.
# It only changes the shape into the format CrossEntropyLoss expects.

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:


    model.train()

    losses = []
    perplexities = []

    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        # input_ids shape is (32, 64)
        # target_ids shape is (32, 64)    

        logits = model(input_ids) # output shape is (32, 64, 65)

        flatten_logits = torch.flatten(logits, start_dim=0, end_dim=1) # output shape is (2048, 65)
        flatten_targets = torch.flatten(target_ids) # output shape is (2048)

        loss = loss_fn(flatten_logits, flatten_targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        perplexity = torch.exp(loss)
        perplexities.append(perplexity.item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_perp = torch.tensor(perplexities).mean().item()

    return {"loss": mean_loss, "perplexity": mean_perp}

    #raise NotImplementedError("Return the mean train loss and train perplexity for one full epoch.")


# Run one full evaluation pass.
# Return a dict with exactly these keys:
# - `loss`
# - `perplexity`
# Return type:
# - `dict[str, float]`
# Both values must be Python floats.
# If you pass the validation loader:
# - `loss` means mean validation loss over the whole validation loader
# - `perplexity` means `exp(loss)` for the whole validation loader
# If you pass the test loader:
# - `loss` means mean test loss over the whole test loader
# - `perplexity` means `exp(loss)` for the whole test loader
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    model.eval()

    losses = []
    perplexities = []

    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)

        flatten_logits = torch.flatten(logits, start_dim=0, end_dim=1)
        flatten_targets = torch.flatten(target_ids)

        loss = loss_fn(flatten_logits, flatten_targets)

        losses.append(loss.item())

        perplexity = torch.exp(loss)
        perplexities.append(perplexity.item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_perp = torch.tensor(perplexities).mean().item()

    return {"loss": mean_loss, "perplexity": mean_perp}

    #raise NotImplementedError("Return the mean loss and perplexity for the loader you evaluate.")


# Generate a short text sample from a finished model.
# Return:
# - type: `str`
# - meaning: one generated character string
# A simple default path is greedy decoding or simple sampling.
# Small temporary step that is allowed while you debug:
# - you may return `start_text` unchanged first
# - then replace that with real generation once the rest of the file works
def generate_text(model: nn.Module, start_text: str, config: CharacterLanguageModelConfig) -> str:
    # Beginner-friendly generation idea:
    # - turn `start_text` into character ids
    # - feed the current ids into the model
    # - read the logits at the last time step only
    # - pick one next character id
    # - append that character
    # - repeat until you reach the target sample length
    # Intuition:
    # - generation is just prediction used in a loop
    # - the model guesses one next character
    # - then that new character becomes part of the context for the next guess

    if not start_text:
        start_text = 'hello'

    encode = lambda s: [config.str_to_int[c] for c in s]
    decode = lambda l: "".join(config.int_to_str[i] for i in l)

    context_text = start_text[-config.context_length:]
    # `start_text[-config.context_length:]` takes only the last `context_length` characters 
    # from `start_text`. This is useful during generation because the model was trained to use 
    # a limited text window, not the full text history forever. If `start_text` is longer 
    # than `context_length`, this keeps the most recent characters, which are usually the most 
    # useful for predicting the next character. If `start_text` is shorter than `context_length`, 
    # Python just returns the whole string, so it still works. For example, 
    # if `context_length` is `4` and `start_text` is `"hello"`, the model sees `"ello"`. 
    # If `start_text` is `"hi"`, the model sees `"hi"`.
    # Also In Python, slicing is forgiving. If you ask for a start position that is before 
    # the beginning of the string, Python just starts at the beginning.
    # Example:
    # start_text = "hi"
    # context_length = 4
    # start_text[-context_length:]
    # This means:
    # start_text[-4:]
    # start 4 characters from the end, then go to the end

    # model.parameters() gives an iterator next(...) gets the first parameter, 
    # and .device tells you where the model is, like CPU or GPU.
    model_device = next(model.parameters()).device 

    character_ids = torch.tensor(encode(context_text), dtype=torch.long)        # shape = (5)
    character_ids = character_ids.unsqueeze(dim=0)                              # shape = (1, 5)  i.e. 1 sample with 5 characters
    character_ids = character_ids.to(model_device)

    generated = []
    for i in range(config.sample_length):
        logits = model(character_ids)   # output shape is (1, 5, 65)        
        last_vec = logits[0, -1, :]           # output shape is (65)
        next_char = torch.argmax(last_vec).long()

        next_char = next_char.to(model_device)

        generated.append(next_char.item())

        next_char = next_char.reshape(1, 1) # shape = (1, 1)
        character_ids = torch.concat([character_ids, next_char], dim=1)
        character_ids = character_ids[:,-config.context_length:]

    decoded_text = start_text + decode(generated)

    return decoded_text

    #raise NotImplementedError("Generate and return one short character sample.")


# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `val_loss`
# - `val_perplexity`
# Return type:
# - `dict[str, float]`
# This is the final epoch summary, not a history list.
def run_training(config: CharacterLanguageModelConfig) -> dict[str, float]:

    torch.manual_seed(config.random_state)

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    for i in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i+1} -> loss: {train_dict['loss']}, perp: {train_dict['perplexity']}, val_loss: {val_dict['loss']}, val_perp: {val_dict['perplexity']}")

    test_dict = evaluate(model, loaders["test"], loss_fn)
    print(f"Test results -> loss: {test_dict['loss']}, perp: {test_dict['perplexity']}")

    print(f"\n GENERATED: \n {generate_text(model=model, start_text = 'To be', config=config)}")

    return {"train_loss": train_dict["loss"], 
            "val_loss": val_dict["loss"], "val_perplexity": val_dict["perplexity"]}

    #raise NotImplementedError("Run the full character language-model training flow and return the final metrics.")



def check_metrics(metrics: dict[str, float], target: float) -> bool:
    """Return True when validation loss is at or below the target."""
    return float(metrics.get("val_loss", float("inf"))) <= float(target)



def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
        print(f"[DONE] {task_name}")
        return result, True
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False



def main() -> None:
    """Run the character language-model scaffold one step at a time."""
    config = CharacterLanguageModelConfig()
    set_seed(config.random_state)
    print("Running exercise 05 starter scaffold...")
    print("Main goal: build a simple next-character language model before the mini GPT file.")
    print(f"Device note: keep the model and all token tensors on the same device. Using {device}.")

    completed_steps = 0
    total_steps = 11

    raw_text, ok = report_task_progress("load_raw_text", load_raw_text, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("split_text_corpus", split_text_corpus, raw_text)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_language_model_shapes", describe_language_model_shapes, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("CharacterLanguageModel.__init__", CharacterLanguageModel, 65, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_input_ids = torch.zeros(config.batch_size, config.context_length, dtype=torch.long)
    _, ok = report_task_progress("CharacterLanguageModel.forward", model, sample_input_ids)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return    

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    training_model, loss_fn, optimizer = training_objects

    _, ok = report_task_progress("train_one_epoch", train_one_epoch, training_model, loaders["train"], loss_fn, optimizer)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("evaluate", evaluate, training_model, loaders["val"], loss_fn)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("generate_text", generate_text, model, "To be", config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics, config.target_val_loss)
        print_pass_or_try_again(
            passed,
            "val_loss",
            float(metrics["val_loss"]),
            f"<= {config.target_val_loss:.2f}",
        )
        print(f"Validation perplexity summary: {float(metrics['val_perplexity']):.4f}")


if __name__ == "__main__":
    main()
