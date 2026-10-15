"""
============================================================
EXERCISE 15: DIFFUSION MINI CAPSTONE DIGITS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny conditional diffusion capstone on digit images.

This exercise combines:
- basic denoising from the related exercise
- noise schedules from the related exercise
- conditional guidance from the related exercise
- latent and reconstruction ideas from the related exercise

The goal is not photo-quality generation.
The goal is to build one full small loop:
train a denoiser -> sample digits -> compare results.

WHY THIS EXERCISE EXISTS
The previous diffusion exercises teach pieces.
This exercise gives you one place to connect those pieces in a working training and sampling flow.

This exercise teaches:
- how to train a conditional noise-prediction model
- how labels can guide generation
- how to sample by slowly removing noise
- how to inspect generated digit-like outputs
- how to write a short capstone report

DATASET
Use `sklearn.datasets.load_digits`.

Why:
- it is real image data
- images are tiny `8 x 8`
- training stays fast enough for a learning exercise

INPUTS AND OUTPUTS
Clean image vectors:
- shape: `[batch_size, 64]`

Labels:
- shape: `[batch_size]`

Noisy image vectors:
- shape: `[batch_size, 64]`

Predicted noise:
- shape: `[batch_size, 64]`

LOSS FUNCTION
Use `nn.MSELoss`.

The model predicts the noise that was added.
The target is the true noise.

METRICS TO TRACK
Track at least:
- train denoising loss
- validation denoising loss
- sample grid path
- one note about sample quality
- one note about what conditioning changed

STEP-BY-STEP TODO
1. Load and scale the digits dataset.
2. Split into train, validation, and test.
3. Build a beta noise schedule.
4. Add noise to clean digits at random timesteps.
5. Build a conditional denoising model.
6. Train the model to predict noise.
7. Evaluate validation denoising loss.
8. Sample new digits from random noise.
9. Compare samples for different requested labels.
10. Save a small sample grid.
11. Write a short capstone report.

DEFINITION OF DONE
You are done when:
- the denoiser trains without shape errors
- validation denoising loss is returned
- samples are generated for requested labels
- a sample grid is saved or returned
- the report explains what worked and what did not
"""

from __future__ import annotations

from collections.abc import Callable
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


DataloaderSplits = dict[str, DataLoader]
NoiseSchedule = dict[str, torch.Tensor]
CapstoneMetrics = dict[str, float | str]
CapstoneShapes = dict[str, tuple[int, ...]]


@dataclass
class DiffusionCapstoneConfig:
    """Settings for the diffusion mini capstone."""

    data_dim: int = 64
    num_classes: int = 10
    hidden_dim: int = 128
    num_timesteps: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sample_count: int = 10
    random_state: int = 42
    sample_grid_path: str = str(BOOTCAMP_ROOT / "outputs" / "diffusion_digit_capstone_samples.png")


def describe_capstone_shapes(config: DiffusionCapstoneConfig) -> CapstoneShapes:
    """Return the main expected tensor shapes."""
    return {
        "clean_batch": (config.batch_size, config.data_dim),
        "labels": (config.batch_size,),
        "timesteps": (config.batch_size,),
        "noisy_batch": (config.batch_size, config.data_dim),
        "predicted_noise": (config.batch_size, config.data_dim),
    }


def load_dataloaders(config: DiffusionCapstoneConfig) -> DataloaderSplits:
    """Return train, validation, and test dataloaders."""
    # Goal:
    # - Load `sklearn` digits, scale pixel values, split the data, and return three dataloaders.
    # - Each batch must contain both the clean digit vector and its label.
    #
    # Input:
    # - `config.data_dim`: expected flattened image size, 64 for `8 x 8` digits.
    # - `config.batch_size`: batch size for all loaders.
    # - `config.random_state`: seed used for reproducible splits.
    #
    # Output:
    # - Return `DataloaderSplits`.
    # - Exact keys:
    #   - `"train"`
    #   - `"val"`
    #   - `"test"`
    # - Each loader yields `(x, labels)`.
    # - `x` shape: `[batch_size, config.data_dim]`, dtype `torch.float32`.
    # - `labels` shape: `[batch_size]`, dtype `torch.long`.
    # - Small example:
    #   `x.shape == (64, 64)` and `labels[:4] == tensor([0, 7, 2, 1])`.
    #
    # Method:
    # - Use these imports inside the function:
    #   `from sklearn.datasets import load_digits`
    #   `from sklearn.model_selection import train_test_split`
    #   `from torch.utils.data import TensorDataset, DataLoader`
    # - Load the data:
    #   `digits = load_digits()`
    # - `digits.data` has shape `[num_examples, 64]`.
    # - `digits.target` has shape `[num_examples]`.
    # - Scale pixels from `[0, 16]` to `[-1, 1]`:
    #   `x = torch.tensor(digits.data, dtype=torch.float32) / 16.0`
    #   `x = x * 2.0 - 1.0`
    # - Convert labels:
    #   `y = torch.tensor(digits.target, dtype=torch.long)`
    # - Split twice:
    #   `x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=config.random_state, stratify=y)`
    #   `x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=config.random_state, stratify=y_temp)`
    # - `stratify=y` keeps all digit classes balanced across splits.
    #
    # Steps:
    # 1. Load the digits dataset.
    # 2. Convert features to `torch.float32` and labels to `torch.long`.
    # 3. Scale features to `[-1, 1]`.
    # 4. Check that `x.shape[1] == config.data_dim`.
    # 5. Create train, validation, and test splits.
    # 6. Wrap each split with `TensorDataset(x_split, y_split)`.
    # 7. Return dataloaders with the exact keys listed above.
    #
    # Checks:
    # - `set(loaders) == {"train", "val", "test"}`
    # - `x_batch, y_batch = next(iter(loaders["train"]))`
    # - `x_batch.shape[1] == config.data_dim`
    # - `y_batch.dtype == torch.long`
    # - `x_batch.min() >= -1.0` and `x_batch.max() <= 1.0`
    raise NotImplementedError("Load sklearn digits, scale values, split data, and return dataloaders.")


def build_noise_schedule(config: DiffusionCapstoneConfig) -> NoiseSchedule:
    """Return beta, alpha, and cumulative alpha values."""
    # Goal:
    # - Build the noise schedule used by training and sampling.
    #
    # Input:
    # - `config.num_timesteps`: number of diffusion steps.
    # - `config.beta_start`: first beta value.
    # - `config.beta_end`: last beta value.
    #
    # Output:
    # - Return `NoiseSchedule`.
    # - Exact keys:
    #   - `"betas"`: shape `[num_timesteps]`
    #   - `"alphas"`: shape `[num_timesteps]`
    #   - `"alpha_bars"`: shape `[num_timesteps]`
    # - All tensors should be `torch.float32`.
    #
    # Method:
    # - Use:
    #   `betas = torch.linspace(config.beta_start, config.beta_end, config.num_timesteps, dtype=torch.float32)`
    #   `alphas = 1.0 - betas`
    #   `alpha_bars = torch.cumprod(alphas, dim=0)`
    # - `beta[t]` is new noise added at step `t`.
    # - `alpha[t]` is clean signal kept at step `t`.
    # - `alpha_bars[t]` is clean signal kept after all steps up to `t`.
    #
    # Steps:
    # 1. Create `betas`.
    # 2. Compute `alphas`.
    # 3. Compute `alpha_bars`.
    # 4. Return the three tensors in a dictionary.
    #
    # Checks:
    # - `set(schedule) == {"betas", "alphas", "alpha_bars"}`
    # - `schedule["betas"].shape == (config.num_timesteps,)`
    # - `schedule["alpha_bars"][0] > schedule["alpha_bars"][-1]`
    raise NotImplementedError("Build a linear beta schedule and cumulative alpha products.")


def add_noise(clean_x: torch.Tensor, timesteps: torch.Tensor, schedule: NoiseSchedule) -> tuple[torch.Tensor, torch.Tensor]:
    """Return noisy data and the true noise that was added."""
    # Goal:
    # - Add the right amount of noise to each clean digit vector.
    # - Return both the noisy input and the exact noise used.
    #
    # Input:
    # - `clean_x`: `torch.float32` tensor with shape `[batch_size, 64]`.
    # - `timesteps`: `torch.long` tensor with shape `[batch_size]`.
    # - `schedule["alpha_bars"]`: tensor with shape `[num_timesteps]`.
    #
    # Output:
    # - Return `(noisy_x, noise)`.
    # - `noisy_x` shape: `[batch_size, 64]`.
    # - `noise` shape: `[batch_size, 64]`.
    # - `noise` is the target the model learns to predict.
    #
    # Method:
    # - Sample noise:
    #   `noise = torch.randn_like(clean_x)`
    # - Gather the schedule value for each row:
    #   `alpha_bar_t = schedule["alpha_bars"].to(clean_x.device)[timesteps].unsqueeze(1)`
    # - Formula:
    #   `noisy_x = torch.sqrt(alpha_bar_t) * clean_x + torch.sqrt(1.0 - alpha_bar_t) * noise`
    # - `.unsqueeze(1)` changes `[batch_size]` to `[batch_size, 1]`.
    # - This lets each row use its own timestep across all 64 pixel values.
    #
    # Steps:
    # 1. Check that `clean_x.ndim == 2`.
    # 2. Check that `timesteps.shape == (clean_x.shape[0],)`.
    # 3. Sample random noise with the same shape as `clean_x`.
    # 4. Gather `alpha_bar_t`.
    # 5. Apply the noising formula.
    # 6. Return `(noisy_x, noise)`.
    #
    # Checks:
    # - `noisy_x.shape == clean_x.shape`
    # - `noise.shape == clean_x.shape`
    # - Later timesteps should usually change `clean_x` more than early timesteps.
    raise NotImplementedError("Sample noise and mix it with clean data using the selected timesteps.")


class ConditionalDenoiser(nn.Module):
    """Tiny conditional denoising model."""

    def __init__(self, config: DiffusionCapstoneConfig) -> None:
        # Goal:
        # - Build a small model that predicts noise from three inputs:
        #   noisy digit vector, timestep, and digit label.
        #
        # Input:
        # - `config.data_dim`: input and output size, 64 for flattened digits.
        # - `config.hidden_dim`: hidden feature size.
        # - `config.num_timesteps`: number of possible timestep ids.
        # - `config.num_classes`: number of digit labels, 10 for digits 0 to 9.
        #
        # Output:
        # - Return `None`, but create:
        #   - `self.time_embedding`
        #   - `self.label_embedding`
        #   - `self.net`
        #
        # Method:
        # - Use these exact modules:
        #   `self.time_embedding = nn.Embedding(config.num_timesteps, config.hidden_dim)`
        #   `self.label_embedding = nn.Embedding(config.num_classes, config.hidden_dim)`
        #   `self.net = nn.Sequential(`
        #   `    nn.Linear(config.data_dim + 2 * config.hidden_dim, config.hidden_dim),`
        #   `    nn.SiLU(),`
        #   `    nn.Linear(config.hidden_dim, config.hidden_dim),`
        #   `    nn.SiLU(),`
        #   `    nn.Linear(config.hidden_dim, config.data_dim),`
        #   `)`
        # - The model output is predicted noise, not class probabilities.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Save `self.config = config`.
        # 3. Create the timestep embedding.
        # 4. Create the label embedding.
        # 5. Create the MLP above.
        #
        # Checks:
        # - `self.time_embedding(torch.zeros(4, dtype=torch.long)).shape == (4, config.hidden_dim)`
        # - `self.label_embedding(torch.zeros(4, dtype=torch.long)).shape == (4, config.hidden_dim)`
        # - `self.net(torch.randn(4, config.data_dim + 2 * config.hidden_dim)).shape == (4, config.data_dim)`
        super().__init__()
        raise NotImplementedError("Build timestep, label, and noisy-input layers.")

    def forward(self, noisy_x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return predicted noise with shape `[batch, data_dim]`."""
        # Goal:
        # - Predict the noise that was added to each noisy digit vector.
        # - Use both timestep and label information.
        #
        # Input:
        # - `noisy_x`: `torch.float32` tensor with shape `[batch_size, 64]`.
        # - `timesteps`: `torch.long` tensor with shape `[batch_size]`.
        # - `labels`: `torch.long` tensor with shape `[batch_size]`.
        #
        # Output:
        # - Return `predicted_noise`.
        # - Shape: `[batch_size, 64]`, same as `noisy_x`.
        #
        # Method:
        # - Embed timesteps:
        #   `time_features = self.time_embedding(timesteps)`
        # - Embed labels:
        #   `label_features = self.label_embedding(labels)`
        # - Concatenate all inputs:
        #   `features = torch.cat([noisy_x, time_features, label_features], dim=1)`
        # - Predict noise:
        #   `predicted_noise = self.net(features)`
        #
        # Steps:
        # 1. Check that `noisy_x.ndim == 2`.
        # 2. Check that `timesteps.shape == labels.shape == (noisy_x.shape[0],)`.
        # 3. Create timestep features.
        # 4. Create label features.
        # 5. Concatenate along feature dimension.
        # 6. Run `self.net`.
        # 7. Return predicted noise.
        #
        # Checks:
        # - `predicted_noise.shape == noisy_x.shape`
        # - Do not apply softmax. Noise values are real numbers.
        raise NotImplementedError("Predict the noise added to each noisy digit.")


def build_training_objects(config: DiffusionCapstoneConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return denoiser, MSE loss, and optimizer."""
    # Goal:
    # - Build the conditional denoiser, the noise-prediction loss, and the optimizer.
    #
    # Input:
    # - `config.learning_rate`: optimizer learning rate.
    #
    # Output:
    # - Return `(model, loss_fn, optimizer)`.
    # - `model`: `ConditionalDenoiser`.
    # - `loss_fn`: `nn.MSELoss`.
    # - `optimizer`: `torch.optim.AdamW`.
    #
    # Method:
    # - Use exact calls:
    #   `model = ConditionalDenoiser(config)`
    #   `loss_fn = nn.MSELoss()`
    #   `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    # - MSE compares predicted noise to true noise with the same shape.
    #
    # Steps:
    # 1. Build the model.
    # 2. Create `nn.MSELoss()`.
    # 3. Create `torch.optim.AdamW(...)`.
    # 4. Return the tuple in the exact order above.
    #
    # Checks:
    # - `isinstance(loss_fn, nn.MSELoss)`
    # - `len(optimizer.param_groups) == 1`
    raise NotImplementedError("Build the conditional denoiser, MSE loss, and Adam optimizer.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: NoiseSchedule,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Return mean train denoising loss."""
    # Goal:
    # - Train for one epoch by predicting the exact noise added to clean digits.
    #
    # Input:
    # - `model`: conditional denoiser.
    # - `loader`: yields `(clean_x, labels)`.
    # - `schedule`: dictionary from `build_noise_schedule`.
    # - `loss_fn`: `nn.MSELoss`.
    # - `optimizer`: AdamW optimizer.
    #
    # Output:
    # - Return one Python `float`: mean training loss over the whole loader.
    # - The loss is MSE between `predicted_noise` and true `noise`.
    #
    # Method:
    # - Put the model in training mode:
    #   `model.train()`
    # - For each batch:
    #   `clean_x, labels = batch`
    #   `timesteps = torch.randint(0, schedule["betas"].numel(), (clean_x.shape[0],), dtype=torch.long)`
    #   `noisy_x, noise = add_noise(clean_x, timesteps, schedule)`
    #   `predicted_noise = model(noisy_x, timesteps, labels)`
    #   `loss = loss_fn(predicted_noise, noise)`
    #
    # Steps:
    # 1. Set `total_loss = 0.0` and `total_examples = 0`.
    # 2. Put the model in train mode.
    # 3. Loop over `(clean_x, labels)`.
    # 4. Sample one random timestep per row.
    # 5. Add noise with `add_noise`.
    # 6. Predict the noise with the model.
    # 7. Compute MSE loss against the true noise.
    # 8. Clear gradients, backpropagate, and step the optimizer.
    # 9. Add `loss.item() * clean_x.shape[0]` to `total_loss`.
    # 10. Return `total_loss / total_examples`.
    #
    # Checks:
    # - `timesteps.shape == labels.shape`
    # - `predicted_noise.shape == clean_x.shape`
    # - Returned loss should be `>= 0.0`.
    raise NotImplementedError("Train one epoch by predicting added noise.")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, schedule: NoiseSchedule, loss_fn: nn.Module) -> float:
    """Return mean validation denoising loss."""
    # Goal:
    # - Measure denoising loss without updating model weights.
    #
    # Input:
    # - Same batch format as `train_one_epoch`: `(clean_x, labels)`.
    # - Use the same `add_noise` and model call as training.
    #
    # Output:
    # - Return one Python `float`: mean validation MSE loss.
    #
    # Method:
    # - Use:
    #   `model.eval()`
    # - This function already has `@torch.no_grad()`, so PyTorch will not store gradients.
    # - Keep the same timestep sampling shape:
    #   `timesteps = torch.randint(0, schedule["betas"].numel(), (clean_x.shape[0],), dtype=torch.long)`
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Loop over `(clean_x, labels)`.
    # 3. Sample timesteps.
    # 4. Add noise.
    # 5. Predict noise.
    # 6. Compute loss against true noise.
    # 7. Return the weighted average loss over all validation examples.
    #
    # Checks:
    # - `predicted_noise.shape == clean_x.shape`
    # - Returned validation loss should be `>= 0.0`.
    raise NotImplementedError("Evaluate denoising loss without updating weights.")


@torch.no_grad()
def sample_digits(model: nn.Module, labels: torch.Tensor, schedule: NoiseSchedule, config: DiffusionCapstoneConfig) -> torch.Tensor:
    """Return sampled digit vectors for the requested labels."""
    # Goal:
    # - Start from random noise and repeatedly remove predicted noise for requested labels.
    # - This is a simple teaching sampler, not a full production DDPM sampler.
    #
    # Input:
    # - `model`: trained conditional denoiser.
    # - `labels`: `torch.long` tensor with shape `[sample_count]`.
    # - `schedule["betas"]`: tensor with shape `[num_timesteps]`.
    # - `config.data_dim`: output vector size, 64.
    #
    # Output:
    # - Return `samples`.
    # - Shape: `[labels.shape[0], config.data_dim]`.
    # - Values may be clamped to `[-1, 1]` at the end because training data used that range.
    #
    # Method:
    # - Start with:
    #   `x = torch.randn(labels.shape[0], config.data_dim)`
    # - Loop backward:
    #   `for t in reversed(range(config.num_timesteps)):`
    # - Build timestep tensor:
    #   `timesteps = torch.full((labels.shape[0],), t, dtype=torch.long)`
    # - Predict noise:
    #   `predicted_noise = model(x, timesteps, labels)`
    # - Simple update:
    #   `beta_t = schedule["betas"][t]`
    #   `x = x - beta_t * predicted_noise`
    # - This simple update teaches the reverse direction without all full DDPM terms.
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Create random starting noise `x`.
    # 3. Loop from the last timestep down to 0.
    # 4. Predict noise for the current `x`, timestep, and label.
    # 5. Move `x` a small step opposite the predicted noise.
    # 6. Optionally clamp final values to `[-1, 1]`.
    # 7. Return `x`.
    #
    # Checks:
    # - `samples.shape == (labels.shape[0], config.data_dim)`
    # - The loop must run in reverse timestep order.
    raise NotImplementedError("Start from noise and repeatedly remove predicted noise.")


def save_sample_grid(samples: torch.Tensor, labels: torch.Tensor, config: DiffusionCapstoneConfig) -> str:
    """Save or return a small sample grid path."""
    # Goal:
    # - Save generated `8 x 8` digit samples in one image file.
    # - Return the path as a string.
    #
    # Input:
    # - `samples`: `torch.float32` tensor with shape `[sample_count, 64]`.
    # - `labels`: `torch.long` tensor with shape `[sample_count]`.
    # - `config.sample_grid_path`: output image path.
    #
    # Output:
    # - Return `config.sample_grid_path` as a string.
    # - The parent folder should exist after this function runs.
    #
    # Method:
    # - Use a non-GUI Matplotlib backend inside the function:
    #   `import matplotlib`
    #   `matplotlib.use("Agg")`
    #   `import matplotlib.pyplot as plt`
    # - Create the folder:
    #   `output_path = Path(config.sample_grid_path)`
    #   `output_path.parent.mkdir(parents=True, exist_ok=True)`
    # - Convert vectors back to images:
    #   `images = samples.detach().cpu().view(-1, 8, 8)`
    # - If samples are in `[-1, 1]`, convert them to `[0, 1]` for display:
    #   `images = (images + 1.0) / 2.0`
    # - Save with:
    #   `fig.savefig(output_path, bbox_inches="tight")`
    #   `plt.close(fig)`
    #
    # Steps:
    # 1. Check that `samples.shape[1] == config.data_dim`.
    # 2. Create the output folder.
    # 3. Reshape samples to `[sample_count, 8, 8]`.
    # 4. Plot one small grayscale image per sample.
    # 5. Put the requested label in each subplot title.
    # 6. Save the figure.
    # 7. Close the figure and return the path string.
    #
    # Checks:
    # - `Path(returned_path).exists()` should be true.
    # - `samples.view(-1, 8, 8).shape[1:] == (8, 8)`.
    raise NotImplementedError("Save generated 8x8 digit samples in a small grid.")


def write_capstone_report(metrics: CapstoneMetrics) -> str:
    """Return a short plain-language report."""
    # Goal:
    # - Explain the final result in plain language.
    # - Mention loss, sample quality, and what labels were meant to control.
    #
    # Input:
    # - `metrics["train_loss"]`: final train denoising loss.
    # - `metrics["val_loss"]`: validation denoising loss.
    # - `metrics["sample_grid_path"]`: path to the saved sample image.
    #
    # Output:
    # - Return one short string.
    # - The report should be honest if samples are rough.
    #
    # Method:
    # - Use an f-string, for example:
    #   `f"Final train loss was {train_loss:.4f} and validation loss was {val_loss:.4f}. ..."`
    # - Say that the model learned to predict added noise.
    # - Say that labels were given to the denoiser so samples could be aimed at requested digits.
    #
    # Steps:
    # 1. Read the three metric values listed above.
    # 2. Write two or three direct sentences.
    # 3. Mention what the loss measures.
    # 4. Mention where the sample grid was saved.
    # 5. Return the report string.
    #
    # Checks:
    # - `isinstance(report, str)`
    # - The report should mention `"loss"` and `"sample"`.
    raise NotImplementedError("Explain loss, sample quality, and what conditioning changed.")


def run_capstone(config: DiffusionCapstoneConfig) -> CapstoneMetrics:
    """Run the full diffusion mini capstone."""
    # Goal:
    # - Run the full capstone workflow from data loading to report.
    #
    # Input:
    # - `config`: all dataset, schedule, model, training, and sampling settings.
    #
    # Output:
    # - Return `CapstoneMetrics`.
    # - Exact keys:
    #   - `"train_loss"`: final train loss
    #   - `"val_loss"`: validation loss
    #   - `"sample_grid_path"`: saved image path
    #   - `"report"`: plain-language report
    #
    # Method:
    # - Call helpers in this order:
    #   `loaders = load_dataloaders(config)`
    #   `schedule = build_noise_schedule(config)`
    #   `model, loss_fn, optimizer = build_training_objects(config)`
    #   `train_loss = train_one_epoch(model, loaders["train"], schedule, loss_fn, optimizer)`
    #   `val_loss = evaluate(model, loaders["val"], schedule, loss_fn)`
    #   `labels = torch.arange(config.sample_count, dtype=torch.long) % config.num_classes`
    #   `samples = sample_digits(model, labels, schedule, config)`
    #   `sample_grid_path = save_sample_grid(samples, labels, config)`
    #   `report = write_capstone_report(metrics)`
    #
    # Steps:
    # 1. Seed the run with `set_seed(config.random_state)`.
    # 2. Load dataloaders.
    # 3. Build the schedule.
    # 4. Build model, loss, and optimizer.
    # 5. Train for `config.epochs` epochs, keeping the last train loss.
    # 6. Evaluate validation loss.
    # 7. Build requested labels for sampling.
    # 8. Sample digit vectors.
    # 9. Save the sample grid.
    # 10. Build the metrics dictionary.
    # 11. Write the report and store it in the metrics dictionary.
    # 12. Return the metrics.
    #
    # Checks:
    # - `check_metrics(metrics)` should return `True`.
    # - `metrics["train_loss"] >= 0.0`
    # - `metrics["val_loss"] >= 0.0`
    raise NotImplementedError("Train, evaluate, sample digits, save a grid, and return a report.")


def check_metrics(metrics: CapstoneMetrics) -> bool:
    """Return True when the capstone returned the expected fields."""
    return (
        "train_loss" in metrics
        and "val_loss" in metrics
        and "sample_grid_path" in metrics
        and "report" in metrics
    )


def report_task_progress(task_name: str, task_fn: Callable, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the diffusion capstone scaffold."""
    config = DiffusionCapstoneConfig()
    set_seed(config.random_state)
    metrics, ok = report_task_progress("run_capstone", run_capstone, config)
    if ok and metrics is not None:
        print_pass_or_try_again(check_metrics(metrics), "diffusion mini capstone", float(int(check_metrics(metrics))), "training, sampling, and report returned expected fields")


if __name__ == "__main__":
    main()
