"""
============================================================
EXERCISE 10: DIFFUSION MODEL BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny diffusion-style denoising exercise.
This is a mini teaching version.
It is not a full image generator.
It is not meant to match large modern diffusion systems.
It is meant to help you understand the main training idea very clearly first.
You will build:
- a tiny real dataset
- a noise schedule
- a function that adds noise to clean data
- a small denoising model
- a noise-prediction training loop
- a simple reverse sampling loop

Plain-language meaning:
- `diffusion training` means we take clean data and slowly corrupt it with noise
- `noise schedule` means a list of values that tells us how much noise to use at each step
- `denoising model` means a model that looks at a noisy sample and tries to tell us what noise was added
- `sampling loop` means start from random noise and clean it up a little at a time

WHY THIS EXERCISE EXISTS
Diffusion models are important for image and video generation.
This exercise keeps the setup tiny so the core idea is clear.
This exercise teaches:
- noise schedule
- noisy inputs
- denoising model
- noise prediction loss
- simple sampling loop

HOW THIS FITS INTO THE BOOTCAMP
Earlier generative exercises focused on autoregressive sequence models.
This file introduces another generative idea:
- learn to remove noise step by step
- instead of predicting the next token

Beginner intuition:
- training says: "here is a clean digit vector after noise was added; can you tell me exactly what noise was added?"
- if the model can predict the noise well, then we can subtract that noise back out
- sampling says: "start from random noise and slowly clean it up"
- many small denoising steps can turn random noise into something structured

Small example:
- imagine one clean flattened digit vector with 64 numbers
- a tiny slice of that vector might look like `[0.0, 0.25, 0.75, 1.0]`
- pick a time step, for example `t = 40`
- add random noise so the vector becomes messier
- the model sees the noisy vector and the time step
- the model tries to predict the noise that was added
- later, during sampling, we start from random noise and repeatedly remove predicted noise
Why predict noise instead of predicting the clean vector directly?
- because the standard simple diffusion setup is written that way
- it often makes the training target easier to define
- once you know the noise, you can work backward toward the clean data

DATASET
Use one clear default dataset:
- a tiny real batch from `sklearn.datasets.load_digits`
Recommended beginner choice:
- use the flattened digit images from `load_digits`
- treat each example as one clean vector
- keep the first version very small so the diffusion math stays readable
Why use `load_digits` first?
- it is a real dataset
- it is small
- the vectors are still short enough to inspect
- it is much easier than starting with large images

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- load `load_digits`
- flatten the image data if needed
- keep the clean data in one tensor of shape `[num_samples, data_dim]`
- wrap that tensor in a `DataLoader`
- each batch should return clean samples only
You do not need labels for this exercise.
The training target is the added noise, not a class label.

INPUTS AND OUTPUTS
Clean batch:
- `clean_x`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, data_dim]`
Time step batch:
- `timesteps`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`
  - meaning: each sample can be noised at a different step
Noisy batch:
- `noisy_x`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, data_dim]`
True noise:
- `noise`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, data_dim]`

Model output:
- `predicted_noise`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, data_dim]`

What those shapes mean:
- `[64, 64]` means 64 digit examples in the batch and each digit has 64 numbers
- `[64]` for timesteps means one noise step per digit example
- the model output must match the true noise shape because it is trying to predict that noise directly

MODEL ARCHITECTURE (HIGH LEVEL ONLY)

Default path:
- input clean vector is first turned into a noisy vector
- the denoiser receives:
  - the noisy vector
  - the time step
- the denoiser predicts the noise that was added
Simple model idea:
- take the noisy digit vector
- turn the time step into a small learned vector or simple numeric feature
- combine digit-vector information and time-step information
- pass them through a small MLP
- output 64 numbers for the predicted noise
Mental picture:
- the model is not asked "what is the clean digit vector?"
- it is asked "what random push was added to this vector at this step?"
- once it learns that, we can reverse the corruption process

LOSS FUNCTION
Use `nn.MSELoss`.
Important:
- compare `predicted_noise` to the true sampled noise
- do not compare directly to the clean vector in this basic version
- lower loss means the model is getting better at telling which noise was added

METRICS TO TRACK
Track at least:
- `train_loss`
  - type: `float`
  - meaning: mean noise-prediction loss over the epoch
- `sample_shape`
  - type: `tuple`
  - meaning: the shape of generated samples after reverse denoising
- `diffusion_note`
  - type: `str`
  - meaning: your own plain-language summary of what the model learned
Simple qualitative check:
- print a few clean digit-vector slices
- print noisy versions of those same slices
- print sampled vector slices after denoising
- check whether sampled vectors look less random than pure random noise

TARGET RESULT RANGE
Because this is a tiny toy diffusion exercise, focus on understanding, not on a magic score.

Useful signs:
- training loss goes down over epochs
- the reverse sampling loop runs without shape errors
- sampled vectors look less random than the starting noise
- you can explain the full process in plain words

COMMON MISTAKES
- not specifying one clear default dataset
- forgetting that each sample can have a different time step
- mixing up `clean_x`, `noisy_x`, and `noise`
- returning model output with the wrong shape
- trying to predict clean vectors when the exercise target is noise
- forgetting to move schedule tensors to the same device as the batch
- making the reverse loop run in the wrong time order
What "forward diffusion" means in plain words:
- start with clean data
- add noise
- add more noise
- after many steps, the sample becomes mostly random noise
What "reverse diffusion" means in plain words:
- start from random noise
- predict what noise is present
- remove some of it
- repeat many times
- hope to end with a structured sample
Another small concrete example:
- clean vector slice: `[0.0, 0.25, 0.75, 1.0]`
- true noise slice: `[-0.5, 0.1, 0.3, -0.2]`
- noisy vector: clean vector after mixing in that noise
- model input: noisy vector plus time step
- model target: the full true noise vector, with the same shape as the clean vector
- if the model predicts something close to that target, it is learning the denoising task correctly

STEP-BY-STEP TODO
1. Create a simple real-vector dataset.
2. Build a noise schedule.
3. Add noise at random time steps.
4. Build a tiny denoising model.
5. Train the model to predict noise.
6. Run a short sampling loop.
7. Compare original, noisy, and denoised samples.
8. Write a short note about the denoising objective.

What each step is doing in simple words:
1. load clean real data
2. decide how noise strength changes over time
3. corrupt clean vectors in a controlled way
4. build a model that sees noisy vectors and time steps
5. train the model to recover the added noise
6. reverse the process from noise back toward structure
7. check whether denoising is doing anything sensible
8. explain the whole idea in your own plain words

STRETCH GOALS
- plot clean digit vectors and sampled vectors as 8 by 8 images
- compare an early noisy step and a late noisy step
- try a slightly deeper denoiser
- try predicting clean vectors only after you finish the main version and compare the difference

DEFINITION OF DONE
You are done when:
- noise can be added at a chosen time step
- the model predicts noise
- denoising loss is reported
- a sampling loop runs
- you can explain diffusion in plain words
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


DiffusionProcessDescription = dict[str, tuple[int, int] | int | str]
NoiseSchedule = dict[str, torch.Tensor]
NoisedBatch = dict[str, torch.Tensor]
TrainingMetrics = dict[str, float]
DiffusionDemoSummary = dict[str, float | tuple[int, ...] | str]
# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DiffusionConfig:
    """Settings for diffusion basics."""
    num_samples: int = 1024
    data_dim: int = 64
    hidden_dim: int = 128
    num_timesteps: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 5
    random_state: int = 42

def describe_diffusion_process(config: DiffusionConfig) -> DiffusionProcessDescription:
    """Return the main diffusion pieces."""
    return {
        "data_shape": (config.batch_size, config.data_dim),
        "num_timesteps": config.num_timesteps,
        "training_target": "predict added noise",
    }

# Build the diffusion dataset.
# Return one dataloader of clean samples only.
# The training target is the added noise, not a class label.
def build_toy_dataset(config: DiffusionConfig) -> DataLoader:
    """Return simple real data for denoising practice."""
    # Goal:
    # - Return a dataloader of clean digit vectors.
    # - The batch should contain clean data only. Labels are not needed.
    #
    # Input:
    # - `config.batch_size`: number of digit vectors per batch.
    # - `config.num_samples`: maximum number of digit examples to keep.
    # - `config.data_dim`: expected feature size. For `load_digits`, this should be 64.
    #
    # Output:
    # - Return a PyTorch `DataLoader`.
    # - Each batch should be one tensor:
    #   `clean_x`
    # - `clean_x` shape: `[batch_size, config.data_dim]`
    # - `clean_x` dtype: `torch.float32`
    # - Small example shape for digits:
    #   `[64, 64]` means 64 digit images, each flattened to 64 numbers.
    #
    # Method:
    # - Use scikit-learn and PyTorch:
    #   `from sklearn.datasets import load_digits`
    #   `from torch.utils.data import TensorDataset, DataLoader`
    # - Load data with:
    #   `digits = load_digits()`
    # - `digits.data` has shape `[num_examples, 64]`.
    # - Scale pixel values from `[0, 16]` to roughly `[-1, 1]`:
    #   `x = torch.tensor(digits.data, dtype=torch.float32) / 16.0`
    #   `x = x * 2.0 - 1.0`
    #
    # Steps:
    # 1. Load the digits dataset.
    # 2. Take at most `config.num_samples` rows.
    # 3. Convert the feature matrix to `torch.float32`.
    # 4. Scale values to `[-1, 1]`.
    # 5. Check that `x.shape[1] == config.data_dim`.
    # 6. Wrap the tensor with `TensorDataset(x)`.
    # 7. Return `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`.
    #
    # Checks:
    # - `batch = next(iter(loader))[0]`
    # - `batch.shape[1] == config.data_dim`
    # - `batch.dtype == torch.float32`
    # - `batch.min() >= -1.0` and `batch.max() <= 1.0`
    raise NotImplementedError("Return a dataloader of simple clean real samples.")

# Build the diffusion noise schedule.
# Return at least:
# - beta values
# - alpha values
# - cumulative alpha values
# These control how much noise is present at each time step.
def build_noise_schedule(config: DiffusionConfig) -> NoiseSchedule:
    """Return beta, alpha, and cumulative alpha values."""
    # Goal:
    # - Build the values that control how much noise is added at each time step.
    #
    # Input:
    # - `config.num_timesteps`: number of diffusion steps.
    #
    # Output:
    # - Return `NoiseSchedule`.
    # - Exact keys:
    #   - `"beta"`: tensor with shape `[num_timesteps]`
    #   - `"alpha"`: tensor with shape `[num_timesteps]`
    #   - `"alpha_bar"`: tensor with shape `[num_timesteps]`
    # - All tensors should be `torch.float32`.
    #
    # Method:
    # - Use a simple linear beta schedule:
    #   `beta = torch.linspace(1e-4, 0.02, config.num_timesteps, dtype=torch.float32)`
    # - Meanings:
    #   - `beta[t]` is the amount of new noise added at step `t`
    #   - `alpha[t] = 1 - beta[t]` is the clean-data amount kept at step `t`
    #   - `alpha_bar[t]` is the cumulative product of alphas up to step `t`
    # - Exact formulas:
    #   `alpha = 1.0 - beta`
    #   `alpha_bar = torch.cumprod(alpha, dim=0)`
    #
    # Steps:
    # 1. Create `beta`.
    # 2. Compute `alpha`.
    # 3. Compute `alpha_bar`.
    # 4. Return the three tensors in a dictionary.
    #
    # Checks:
    # - `set(schedule) == {"beta", "alpha", "alpha_bar"}`
    # - `schedule["beta"].shape == (config.num_timesteps,)`
    # - `schedule["alpha_bar"][0] > schedule["alpha_bar"][-1]`
    raise NotImplementedError("Build a simple diffusion noise schedule.")

# Add noise to clean samples at chosen time steps.
# Return:
# - `noisy_x`
# - `noise`
# The model sees `noisy_x` and tries to predict `noise`.
def add_noise(
    clean_x: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: NoiseSchedule,
) -> NoisedBatch:
    """Add noise to clean samples at chosen timesteps."""
    # Goal:
    # - Add the right amount of noise to each clean sample for its timestep.
    # - Return both the noisy sample and the exact noise used.
    #
    # Input:
    # - `clean_x`: `torch.float32` tensor with shape `[batch_size, data_dim]`.
    # - `timesteps`: `torch.long` tensor with shape `[batch_size]`.
    # - `schedule["alpha_bar"]`: tensor with shape `[num_timesteps]`.
    #
    # Output:
    # - Return `NoisedBatch`.
    # - Exact keys:
    #   - `"noisy_x"`: tensor with shape `[batch_size, data_dim]`
    #   - `"noise"`: tensor with shape `[batch_size, data_dim]`
    #
    # Method:
    # - Sample noise with:
    #   `noise = torch.randn_like(clean_x)`
    # - Get the schedule value for each sample:
    #   `alpha_bar_t = schedule["alpha_bar"][timesteps].unsqueeze(1).to(clean_x.device)`
    # - Formula:
    #   `noisy_x = sqrt(alpha_bar_t) * clean_x + sqrt(1 - alpha_bar_t) * noise`
    # - Meaning:
    #   - high `alpha_bar_t` keeps more clean data
    #   - low `alpha_bar_t` mixes in more noise
    #
    # Steps:
    # 1. Check that `clean_x.ndim == 2`.
    # 2. Check that `timesteps.shape == (clean_x.shape[0],)`.
    # 3. Move schedule tensors to the same device as `clean_x`.
    # 4. Sample noise with the same shape as `clean_x`.
    # 5. Gather `alpha_bar_t` for each sample.
    # 6. Apply the formula above.
    # 7. Return `{"noisy_x": noisy_x, "noise": noise}`.
    #
    # Checks:
    # - `result["noisy_x"].shape == clean_x.shape`
    # - `result["noise"].shape == clean_x.shape`
    # - `result["noise"]` is the training target, not `clean_x`.
    raise NotImplementedError("Return noisy samples and the true noise.")


# Beginner hints for this file:
# - use one default dataset first: flattened digit vectors from `load_digits`
# - the model is not trying to predict the clean sample directly in this basic version
# - it is trying to predict the exact noise that was added
# - training adds noise; sampling removes noise a little at a time
# - keep the first version tiny so the main idea is easy to inspect
# - if the schedule values are confusing, print a few early and late ones to see the pattern
class TinyDenoiser(nn.Module):
    """Tiny model that predicts noise."""
    def __init__(self, config: DiffusionConfig) -> None:
        # Goal:
        # - Build a small MLP that predicts the noise added to a noisy digit vector.
        # - The model must use both the noisy vector and the timestep.
        #
        # Input:
        # - `config.data_dim`: input and output vector size, 64 for flattened digits.
        # - `config.hidden_dim`: hidden layer size.
        # - `config.num_timesteps`: number of possible timestep ids.
        #
        # Output:
        # - Return `None`, but create:
        #   - a timestep embedding
        #   - an MLP that returns predicted noise
        #
        # Method:
        # - Use these exact modules:
        #   `self.time_embedding = nn.Embedding(config.num_timesteps, config.hidden_dim)`
        #   `self.net = nn.Sequential(`
        #   `    nn.Linear(config.data_dim + config.hidden_dim, config.hidden_dim),`
        #   `    nn.SiLU(),`
        #   `    nn.Linear(config.hidden_dim, config.hidden_dim),`
        #   `    nn.SiLU(),`
        #   `    nn.Linear(config.hidden_dim, config.data_dim),`
        #   `)`
        # - `nn.SiLU` is a smooth activation often used in diffusion models.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Save `self.config = config` if useful.
        # 3. Create the timestep embedding.
        # 4. Create the MLP above.
        #
        # Checks:
        # - `self.time_embedding.num_embeddings == config.num_timesteps`
        # - `self.net(torch.randn(4, config.data_dim + config.hidden_dim)).shape == (4, config.data_dim)`
        #
        super().__init__()
        raise NotImplementedError("Build a tiny denoising model that also uses the time step.")
    def forward(self, noisy_x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return predicted noise with the same shape as `noisy_x`."""
        # Goal:
        # - Predict the noise that was added to `noisy_x`.
        # - The returned tensor must match the true noise shape exactly.
        #
        # Input:
        # - `noisy_x`: `torch.float32` tensor with shape `[batch_size, data_dim]`.
        # - `timesteps`: `torch.long` tensor with shape `[batch_size]`.
        #
        # Output:
        # - Return `predicted_noise` with shape `[batch_size, data_dim]`.
        # - Do not return a clean image prediction in this basic version.
        #
        # Method:
        # - Embed timesteps:
        #   `time_features = self.time_embedding(timesteps)`
        # - Concatenate noisy data and time features:
        #   `model_input = torch.cat([noisy_x, time_features], dim=1)`
        # - Predict noise:
        #   `predicted_noise = self.net(model_input)`
        #
        # Steps:
        # 1. Check that `noisy_x.ndim == 2`.
        # 2. Check that `timesteps.shape == (noisy_x.shape[0],)`.
        # 3. Get time embeddings.
        # 4. Concatenate along feature dimension.
        # 5. Run the MLP.
        # 6. Return predicted noise.
        #
        # Checks:
        # - `predicted_noise.shape == noisy_x.shape`
        # - Do not apply softmax. Noise values are real numbers, not class probabilities.
        #
        raise NotImplementedError("Predict the noise added to the clean sample.")

def build_training_objects(config: DiffusionConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return denoiser, loss function, and optimizer."""
    # Goal:
    # - Build the denoiser, noise-prediction loss, and optimizer.
    #
    # Input:
    # - `config.learning_rate`: optimizer learning rate.
    #
    # Output:
    # - Return type: tuple[nn.Module, nn.Module, torch.optim.Optimizer]
    # - Tuple order must be:
    #   `(model, loss_fn, optimizer)`
    #
    # Method:
    # - Use these exact calls:
    #   `model = TinyDenoiser(config).to(device)`
    #   `loss_fn = nn.MSELoss()`
    #   `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    # - `MSELoss` compares predicted noise to true noise with the same shape.
    #
    # Steps:
    # 1. Build and move the model to `device`.
    # 2. Create `nn.MSELoss`.
    # 3. Create `torch.optim.AdamW`.
    # 4. Return the tuple in the exact order listed above.
    #
    # Checks:
    # - `isinstance(loss_fn, nn.MSELoss)`
    # - `next(model.parameters()).device.type` should match `device`.
    #
    raise NotImplementedError("Return model, MSELoss, and optimizer.")

# Train the denoiser for one epoch.
# Return:
# - mean noise-prediction loss
# Each batch should:
# - choose time steps
# - add noise
# - predict that noise
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: NoiseSchedule,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DiffusionConfig,
) -> TrainingMetrics:
    """Train the denoiser for one epoch."""
    # Goal:
    # - Train for one epoch by asking the model to predict the noise added to clean samples.
    #
    # Input:
    # - `model`: `TinyDenoiser`.
    # - `loader`: yields clean batches from `build_toy_dataset`.
    # - `schedule`: result from `build_noise_schedule`.
    # - `loss_fn`: `nn.MSELoss`.
    # - `optimizer`: AdamW optimizer.
    # - `config.num_timesteps`: max timestep id is `config.num_timesteps - 1`.
    #
    # Output:
    # - Return `TrainingMetrics`.
    # - Exact key:
    #   - `"train_loss"`: mean MSE noise-prediction loss over the epoch
    #
    # Method:
    # - Each batch from `TensorDataset(x)` may arrive as a tuple `(clean_x,)`.
    # - Pick random timesteps with:
    #   `timesteps = torch.randint(0, config.num_timesteps, (clean_x.shape[0],), device=clean_x.device, dtype=torch.long)`
    # - Add noise with:
    #   `noised = add_noise(clean_x, timesteps, schedule)`
    # - Predict and train:
    #   `predicted_noise = model(noised["noisy_x"], timesteps)`
    #   `loss = loss_fn(predicted_noise, noised["noise"])`
    #
    # Steps:
    # 1. Put the model in train mode.
    # 2. For each batch, get `clean_x` and move it to `device`.
    # 3. Sample one timestep per item.
    # 4. Add noise with `add_noise`.
    # 5. Predict the true noise.
    # 6. Compute MSE loss against `noised["noise"]`.
    # 7. Clear gradients, backpropagate, and step the optimizer.
    # 8. Accumulate `loss.item() * batch_size`.
    # 9. Return the average loss over all examples.
    #
    # Checks:
    # - `timesteps.shape == (clean_x.shape[0],)`
    # - `predicted_noise.shape == clean_x.shape`
    # - `metrics["train_loss"] >= 0.0`
    raise NotImplementedError("Return mean noise prediction loss.")

# Run the reverse denoising loop.
# In plain words:
# - start from random noise
# - predict noise
# - remove some of it
# - repeat many times
def sample_from_noise(model: nn.Module, schedule: NoiseSchedule, config: DiffusionConfig) -> torch.Tensor:
    """Run a simple reverse denoising loop."""
    # Goal:
    # - Start from random noise and repeatedly remove predicted noise.
    # - This is a simple teaching sampler, not a full DDPM sampler.
    #
    # Input:
    # - `model`: trained denoiser.
    # - `schedule`: result from `build_noise_schedule`.
    # - `config.batch_size`, `config.data_dim`, and `config.num_timesteps`: sample shape and loop length.
    #
    # Output:
    # - Return samples with shape `[config.batch_size, config.data_dim]`.
    # - Use `torch.float32`.
    #
    # Method:
    # - Start with:
    #   `x = torch.randn(config.batch_size, config.data_dim, device=device)`
    # - Loop backward:
    #   `for t in reversed(range(config.num_timesteps)):`
    # - Build timestep tensor:
    #   `timesteps = torch.full((config.batch_size,), t, device=device, dtype=torch.long)`
    # - Predict noise:
    #   `predicted_noise = model(x, timesteps)`
    # - Simple update:
    #   `beta_t = schedule["beta"][t].to(device)`
    #   `x = x - beta_t * predicted_noise`
    # - This simple update teaches the idea without all full DDPM terms.
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Start from random noise with the output shape.
    # 3. Use `torch.no_grad()`.
    # 4. Loop from the last timestep down to 0.
    # 5. Predict noise at the current step.
    # 6. Move the sample a small step opposite the predicted noise.
    # 7. Return the final sample tensor.
    #
    # Checks:
    # - `samples.shape == (config.batch_size, config.data_dim)`
    # - The loop must run in reverse timestep order.
    raise NotImplementedError("Return generated samples after denoising from noise.")

def write_diffusion_note(metrics: TrainingMetrics) -> str:
    """Write a short note about denoising."""
    # Goal:
    # - Explain the diffusion objective in plain words.
    # - Mention that the model learns to predict added noise, not class labels.
    #
    # Input:
    # - `metrics`: training metrics with key `"train_loss"`.
    #
    # Output:
    # - Return one short string.
    # - Include the train loss value if it is available.
    #
    # Method:
    # - Use an f-string, for example:
    #   `f"The denoiser trained to predict sampled noise. Final train loss was {loss:.4f}."`
    #
    # Steps:
    # 1. Read `metrics["train_loss"]`.
    # 2. Write one or two direct sentences.
    # 3. Say that forward diffusion adds noise.
    # 4. Say that reverse sampling removes predicted noise step by step.
    # 5. Return the note.
    #
    # Checks:
    # - `isinstance(note, str)`
    # - The note should mention `"noise"` and `"denoise"` or `"remove"`.
    raise NotImplementedError("Explain the diffusion objective in plain words.")

def run_diffusion_demo(config: DiffusionConfig) -> DiffusionDemoSummary:
    """Run the full diffusion demo."""
    # Goal:
    # - Run the full tiny diffusion workflow and return the main metrics.
    #
    # Input:
    # - `config`: dataset, schedule, model, and training settings.
    #
    # Output:
    # - Return `DiffusionDemoSummary`.
    # - Exact keys:
    #   - `"train_loss"`: final epoch mean noise-prediction loss
    #   - `"sample_shape"`: tuple shape of samples from `sample_from_noise`
    #   - `"diffusion_note"`: note from `write_diffusion_note`
    #
    # Method:
    # - Call the helpers in this order:
    #   `loader = build_toy_dataset(config)`
    #   `schedule = build_noise_schedule(config)`
    #   `model, loss_fn, optimizer = build_training_objects(config)`
    #   `train_one_epoch(...)` for each epoch
    #   `samples = sample_from_noise(model, schedule, config)`
    #   `note = write_diffusion_note(metrics)`
    #
    # Steps:
    # 1. Seed the run with `set_seed(config.random_state)`.
    # 2. Build the clean dataset.
    # 3. Build the noise schedule.
    # 4. Build model, MSE loss, and optimizer.
    # 5. Train for `config.epochs` epochs.
    # 6. Sample from random noise.
    # 7. Write the note.
    # 8. Return the three exact keys listed above.
    #
    # Checks:
    # - `summary["sample_shape"] == (config.batch_size, config.data_dim)`
    # - `summary["train_loss"] >= 0.0`
    # - `check_metrics(summary)` should return `True`.
    raise NotImplementedError("Train denoiser, sample, and return metrics.")

def check_metrics(metrics: DiffusionDemoSummary) -> bool:
    """Return True when diffusion outputs are present."""
    required = {"train_loss", "sample_shape", "diffusion_note"}
    return required.issubset(metrics)

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
    """Run the diffusion scaffold."""
    config = DiffusionConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 10
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    loader, ok = report_task_progress("build_toy_dataset", build_toy_dataset, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("describe_diffusion_process", describe_diffusion_process, config)
    if not count_or_stop(ok):
        return
    schedule, ok = report_task_progress("build_noise_schedule", build_noise_schedule, config)
    if not count_or_stop(ok):
        return
    denoiser, ok = report_task_progress("TinyDenoiser.__init__", TinyDenoiser, config)
    if not count_or_stop(ok):
        return
    sample_x = torch.zeros(config.batch_size, config.data_dim)
    sample_t = torch.zeros(config.batch_size, dtype=torch.long)
    _, ok = report_task_progress("TinyDenoiser.forward", denoiser, sample_x, sample_t)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("add_noise", add_noise, sample_x, sample_t, schedule)
    if not count_or_stop(ok):
        return
    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return
    training_model, loss_fn, optimizer = training_objects
    _, ok = report_task_progress("train_one_epoch", train_one_epoch, training_model, loader, schedule, loss_fn, optimizer, config)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("sample_from_noise", sample_from_noise, training_model, schedule, config)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_diffusion_demo", run_diffusion_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "diffusion summary", float(int(passed)), "diffusion keys present")
if __name__ == "__main__":
    main()
