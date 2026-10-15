"""
============================================================
EXERCISE 22: TEMPORAL MEMORY FOR 3D AGENTS
============================================================

WHAT YOU ARE BUILDING
You are building a first-person policy that remembers earlier views.

The reactive policy from the earlier exercise learns:

```text
current image + instruction -> next action
```

This exercise learns:

```text
recent images + recent actions + instruction -> next action
```

The second form can solve tasks where the current image is not enough.

REAL SYSTEM FIRST
In a large embodied agent, memory may contain:

- recent camera frames
- recent actions
- objects seen earlier
- rooms already visited
- failed attempts
- a compact spatial map
- the current subgoal

The agent uses the current observation to ask which old information matters
now. Large systems may use attention over a long memory store.

This exercise starts with a real, smaller version:

1. render a cue in a first-person 3D task
2. hide the cue after the agent moves away
3. keep an ordered history of real rendered frames and actions
4. encode each frame with a CNN
5. combine each frame with the action taken after it
6. pass the ordered sequence through `nn.GRU`
7. use the final valid GRU state to choose the next action
8. compare it with a policy that sees only the current frame

In PyTorch, `GRU` means gated recurrent unit.
It is a neural network that reads a sequence in order and carries a learned
summary forward.

WHY THIS EXERCISE EXISTS
The main 3D path says that the agent should remember useful things, but a
single-frame policy cannot do that.

A normal navigation scene can also hide this problem. If the target remains
visible, the current frame may be enough. This exercise therefore requires a
memory task.

MEMORY-REQUIRED TASK
Add a `memory_corridor` task variant to the small 3D environment:

```text
start:
the agent sees a red or blue cue

middle:
the agent walks through a corridor where the cue is no longer visible

decision point:
the camera view is the same for both cue colours

correct action:
red cue  -> turn left
blue cue -> turn right
```

At the decision point, two episodes can have the same current image but need
different actions. A current-frame policy must guess. A memory policy can use
the earlier cue.

REAL SYSTEM VS THIS EXERCISE

real embodied agent                 this exercise
---------------------------------------------------------------
long video and action history       last `history_length` frames and actions
large visual backbone               small CNN
attention or learned memory bank    one `nn.GRU`
many memory-dependent tasks         red-left / blue-right corridor
long natural-language goal          short instruction
large benchmark suite               memory and normal navigation splits

WHY THIS SMALLER VERSION EXISTS
The goal is to prove that memory changes what the policy can do.
Training a large video-language memory model would hide that idea behind
model size and download cost.

WHAT IS INCLUDED

- real rendered frames from the local 3D environment
- ordered episode windows
- previous-action embeddings
- padded history masks
- a GRU memory state
- a current-frame baseline
- memory-required and normal navigation evaluation

WHAT IS SKIPPED

- a million-frame memory store
- retrieval over many past episodes
- a learned 3D scene graph
- a large pretrained VLM
- online updates to model weights during evaluation

DATASET
Use `outputs/mini3d_vla_dataset.pt` from `embodied_ai/11`.

Before collecting the memory episodes, extend the logger so each step also
stores:

```text
task_variant: "navigation" or "memory_corridor"
layout_seed: same value for paired red and blue corridor geometry
cue_id: 0 for red or 1 for blue
cue_visible: True only while the cue appears in the current frame
memory_required: True only at the hidden-cue decision point
```

Do not put frames from one episode into different data splits.
Also keep paired red and blue episodes with the same `layout_seed` in the same
split. Split by layout seed first, then collect whole episodes.

Use scripted successful demonstrations for policy training.
At the hidden decision point, red episodes must be labelled `turn left` and
blue episodes must be labelled `turn right`.
Random rollouts are useful for testing the logger, but random action labels
cannot teach the memory rule.

INPUTS AND OUTPUTS

```text
images:
[batch, history_length, 3, image_size, image_size]

previous_actions:
[batch, history_length]

history_mask:
[batch, history_length]
1 means this is a real step
0 means this is right padding

instruction_ids:
[batch, text_length]

target_actions:
[batch]

episode_ids and layout_seeds:
[batch]

action_logits:
[batch, num_actions]
```

TINY CONCRETE EXAMPLE

```text
history_length = 4

episode history:
step 0: see red cue, action=forward
step 1: cue leaves view, action=forward
step 2: blank corridor, action=forward
step 3: decision view, choose an action

target action at step 3:
turn left
```

If only three steps exist, place them first and pad on the right:

```text
history_mask = [1, 1, 1, 0]
```

`pack_padded_sequence(...)` expects the real timesteps first.
The final position is padding and must not change the memory state used for
the decision.

LOSS FUNCTION
Use cross entropy:

```python
loss = nn.functional.cross_entropy(action_logits, target_actions)
```

METRICS TO TRACK

- normal navigation action accuracy: higher is better
- memory-required action accuracy: higher is better
- current-frame memory-task accuracy: should be near chance
- history-policy memory-task accuracy: should beat the current-frame policy
- memory-task rollout success: higher is better
- unseen-cue-layout success: higher is better

For the red-left / blue-right decision with two possible answers, chance is
about `0.5`.

ABLATIONS
An ablation removes one part to check whether that part helped.

Run:

```text
current image only
history of images, but no previous actions
history of images and previous actions
history with the order randomly shuffled
```

If shuffled history works just as well, the model may not be using time in the
way you expect.

VISUALS TO SAVE
Save to `outputs/temporal_memory_3d/`:

- `memory_ablation.png`
- `memory_episode_strip.png`
- `accuracy_by_history_length.png`

COMMON MISTAKES

- shuffling steps before building ordered windows
- splitting frames instead of splitting whole episodes
- using the current action as an input when predicting that same action
- treating padded frames as real history
- testing memory in a task that can be solved from the current image
- reporting action accuracy without running the policy in the environment

STEP-BY-STEP TODO

1. Add the memory-corridor variant to the 3D environment.
2. Log ordered episodes with cue and decision-point metadata.
3. split whole layout groups and episodes into train, validation, and test sets.
4. build the instruction vocabulary from training episodes only.
5. build fixed-length history windows with right padding.
6. train the current-frame baseline.
7. build the CNN, action embedding, GRU, text embedding, and action head.
8. train the history-aware policy.
9. run all four ablations.
10. evaluate full episodes in seen and unseen layouts.
11. save the three visuals.

LEARNING ORDER

Complete the stubs in file order.
The first stub loads and checks episode order.
The second builds the training-only vocabulary.
The third builds the exact tensor windows used by every later stub.

DEFINITION OF DONE

You are done when:

- the test task truly needs information from an earlier frame
- dataset splits are made by episode
- the history tensor and mask shapes are checked
- the current-frame and history policies use the same training episodes
- the history policy beats the current-frame policy on memory-required steps
- a shuffled-history ablation is reported
- real rollout success is reported
- the memory episode and metric plots are saved
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn


class MemoryRolloutStep(TypedDict):
    """One ordered step from a memory or normal navigation episode."""

    image: torch.Tensor
    instruction: str
    action: int
    episode_id: int
    step_id: int
    split: str
    layout_seed: int
    task_variant: str
    cue_id: int
    cue_visible: bool
    memory_required: bool


class TemporalExamples(TypedDict):
    """All tensors needed to train the sequence policy."""

    images: torch.Tensor
    previous_actions: torch.Tensor
    history_mask: torch.Tensor
    instruction_ids: torch.Tensor
    target_actions: torch.Tensor
    memory_required: torch.Tensor
    episode_ids: torch.Tensor
    layout_seeds: torch.Tensor


class MemoryAblationMetrics(TypedDict):
    """Results used to prove whether memory helped."""

    current_frame_accuracy: float
    image_history_accuracy: float
    image_action_history_accuracy: float
    shuffled_history_accuracy: float
    memory_rollout_success: float
    unseen_layout_success: float
    memory_gain: float


@dataclass
class TemporalMemoryConfig:
    """Settings for the history-aware 3D policy."""

    dataset_path: Path = Path("outputs/mini3d_vla_dataset.pt")
    output_dir: Path = Path("outputs/temporal_memory_3d")
    image_size: int = 64
    history_length: int = 8
    text_length: int = 12
    vocab_size: int = 256
    num_actions: int = 6
    image_dim: int = 96
    action_dim: int = 16
    text_dim: int = 32
    memory_dim: int = 128
    batch_size: int = 32
    epochs: int = 8
    learning_rate: float = 3e-4
    random_state: int = 42


def load_memory_rollouts(config: TemporalMemoryConfig) -> list[list[MemoryRolloutStep]]:
    """Load ordered episodes and check that memory metadata exists."""
    # Where this fits:
    # This is the first data step. Every later tensor must preserve the order
    # found here.
    #
    # Goal:
    # Load `config.dataset_path` and return nested ordered episodes.
    #
    # Method:
    # Use:
    # `torch.load(config.dataset_path, weights_only=False)`
    #
    # Steps:
    # 1. Check that the file exists.
    # 2. Check that the loaded value is a list of episodes.
    # 3. Check that each memory episode contains `task_variant`, `cue_id`, and
    #    `memory_required`.
    # 4. Sort each episode by `step_id`.
    # 5. Confirm that every episode belongs to only one split.
    # 6. Confirm that every layout seed belongs to only one split.
    #
    # Checks:
    # - At least one step must have `memory_required=True`.
    # - Raise a clear error if the logger has not collected memory tasks yet.
    # - Pair red and blue decision frames with the same `layout_seed` and check them
    #   with `torch.allclose(...)`. If the pixels reveal the cue, the task does
    #   not prove memory helped.
    _ = config
    raise NotImplementedError("Load and validate ordered memory rollouts.")


def build_instruction_vocabulary(
    episodes: list[list[MemoryRolloutStep]],
    config: TemporalMemoryConfig,
) -> dict[str, int]:
    """Build text ids from training instructions only."""
    # Where this fits:
    # The text embedding receives integer word ids, so the instructions need
    # one stable mapping before temporal windows are built.
    #
    # Goal:
    # Return a dictionary with:
    # - `"<pad>": 0`
    # - `"<unk>": 1`
    # - common lower-case training words starting at id `2`
    #
    # Method:
    # 1. Read only episodes whose steps have `split == "train"`.
    # 2. Lower-case instructions and split them on whitespace.
    # 3. Count words with `collections.Counter`.
    # 4. Keep at most `config.vocab_size - 2` words.
    # 5. Sort equal-frequency words alphabetically so the result is repeatable.
    #
    # Checks:
    # - Do not build the vocabulary from validation or test instructions.
    # - Save this vocabulary with the trained checkpoint.
    _ = (episodes, config)
    raise NotImplementedError("Build the training instruction vocabulary.")


def build_temporal_examples(
    episodes: list[list[MemoryRolloutStep]],
    vocabulary: dict[str, int],
    config: TemporalMemoryConfig,
) -> TemporalExamples:
    """Turn ordered episodes into padded history windows."""
    # Where this fits:
    # The GRU receives fixed-size batches, so variable-length episode history
    # must be turned into fixed `history_length` windows.
    #
    # Goal:
    # Build one training example for every target action without crossing an
    # episode boundary.
    #
    # Input example:
    # - current step index `t=2`
    # - available images are steps `[0, 1, 2]`
    # - `history_length=5`
    #
    # Output example:
    # - images contain frames `[0, 1, 2]` followed by two zero frames
    # - history mask is `[1, 1, 1, 0, 0]`
    # - previous actions align with the images
    # - the current target action is stored separately
    #
    # Method:
    # 1. Loop over episodes, then over target step index `t`.
    # 2. Slice at most `history_length` images ending at `t`.
    # 3. Use the action from the step before each image as its previous action.
    # 4. Reserve action id `config.num_actions` as the start/padding action.
    # 5. Right-pad images, actions, and masks. This is required because
    #    `pack_padded_sequence(...)` reads the first `length` positions.
    # 6. Tokenize the instruction with `vocabulary`; use `0` for padding and
    #    `1` for unknown words.
    # 7. Store `episode_id` and `layout_seed` for split and paired-frame checks.
    # 8. Stack every field into the shapes shown at the top of this file.
    #
    # Checks:
    # - `target_actions` must not appear inside the input action history.
    # - no window may contain two episode ids
    # - `history_mask.sum(dim=1)` must always be at least 1
    _ = (episodes, vocabulary, config)
    raise NotImplementedError("Build padded temporal policy examples.")


class HistoryAwarePolicy(nn.Module):
    """Choose an action from ordered frames, previous actions, and text."""

    def __init__(self, config: TemporalMemoryConfig) -> None:
        # Where this fits:
        # This is the small learned memory system.
        #
        # Goal:
        # Build five named parts:
        #
        # 1. Image encoder:
        #    `nn.Sequential(`
        #    `nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),`
        #    `nn.ReLU(),`
        #    `nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),`
        #    `nn.ReLU(),`
        #    `nn.AdaptiveAvgPool2d((1, 1)),`
        #    `nn.Flatten(),`
        #    `nn.Linear(64, config.image_dim),`
        #    `)`
        #
        # 2. Previous-action embedding:
        #    `nn.Embedding(config.num_actions + 1, config.action_dim)`
        #    The extra id is the start/padding action.
        #
        # 3. Ordered memory:
        #    `nn.GRU(`
        #    `input_size=config.image_dim + config.action_dim,`
        #    `hidden_size=config.memory_dim,`
        #    `batch_first=True,`
        #    `)`
        #
        # 4. Text embedding:
        #    `nn.Embedding(config.vocab_size, config.text_dim, padding_idx=0)`
        #
        # 5. Action head:
        #    concatenate memory and text, then use
        #    `nn.Linear(config.memory_dim + config.text_dim, config.num_actions)`
        super().__init__()
        _ = config
        raise NotImplementedError("Build the history-aware action policy.")

    def forward(
        self,
        images: torch.Tensor,
        previous_actions: torch.Tensor,
        history_mask: torch.Tensor,
        instruction_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return action logits shaped `[batch, num_actions]`."""
        # Goal:
        # Encode the ordered history and return raw scores for the next action.
        #
        # Input:
        # - `images`: `[B, T, 3, H, W]`
        # - `previous_actions`: `[B, T]`
        # - `history_mask`: `[B, T]`
        # - `instruction_ids`: `[B, L]`
        #
        # Steps:
        # 1. Reshape images to `[B * T, 3, H, W]`.
        # 2. Encode them and reshape to `[B, T, image_dim]`.
        # 3. Embed previous actions to `[B, T, action_dim]`.
        # 4. Concatenate image and action features on the last dimension.
        # 5. Compute lengths with `history_mask.sum(dim=1)`.
        # 6. Use `nn.utils.rnn.pack_padded_sequence(...)` with
        #    `batch_first=True` and `enforce_sorted=False`.
        # 7. Pass the packed sequence to the GRU.
        # 8. Use the final GRU hidden state shaped `[B, memory_dim]`.
        # 9. Average only non-padding instruction embeddings.
        # 10. Concatenate memory and text, then return action logits.
        #
        # Checks:
        # - Do not apply softmax before cross entropy.
        # - Valid history must come before right-padding.
        # - If you use left-padding here, the packed GRU will read the padding
        #   and drop recent real frames.
        _ = (images, previous_actions, history_mask, instruction_ids)
        raise NotImplementedError("Run the history-aware action policy.")


def train_temporal_policy(
    examples: TemporalExamples,
    config: TemporalMemoryConfig,
    *,
    use_image_history: bool,
    use_action_history: bool,
    shuffle_history_order: bool = False,
) -> tuple[HistoryAwarePolicy, dict[str, list[float]]]:
    """Train one memory ablation with episode-safe splits."""
    # Goal:
    # Train one of the ablations listed at the top of the file.
    #
    # Method:
    # 1. Select rows by their preassigned layout-safe episode split, not with a
    #    random frame split.
    # 2. Wrap tensors with `TensorDataset`.
    # 3. Use `DataLoader(..., batch_size=config.batch_size, shuffle=True)`.
    # 4. Build `HistoryAwarePolicy(config)`.
    # 5. Use `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 6. Compute:
    #    `nn.functional.cross_entropy(logits, target_actions)`.
    # 7. Call `optimizer.zero_grad()`, `loss.backward()`, and
    #    `optimizer.step()`.
    # 8. Record training loss and validation memory-step accuracy.
    #
    # Ablation rules:
    # - current frame: gather the final valid image from each row, move it to
    #   position `0`, and set the packed length to `1`
    # - image history: replace previous actions with the start-action id
    # - image and action history: use both inputs
    # - shuffled history: shuffle valid history positions inside each row
    #
    # Checks:
    # - Use the same episode splits and random seed for every ablation.
    _ = (
        examples,
        config,
        use_image_history,
        use_action_history,
        shuffle_history_order,
    )
    raise NotImplementedError("Train one temporal-memory ablation.")


def evaluate_memory_ablations(
    examples: TemporalExamples,
    config: TemporalMemoryConfig,
) -> MemoryAblationMetrics:
    """Compare current-frame, history, and shuffled-history policies."""
    # Goal:
    # Produce the evidence that ordered memory helps.
    #
    # Steps:
    # 1. Train all four ablations with `train_temporal_policy(...)`.
    # 2. Evaluate only test rows where `memory_required=True`.
    # 3. Run the best history policy in full memory-corridor episodes.
    # 4. Repeat on unseen cue and corridor layouts.
    # 5. Compute:
    #    `memory_gain = image_action_history_accuracy - current_frame_accuracy`.
    #
    # Checks:
    # - A useful model should beat the current-frame policy.
    # - Ordered history should beat shuffled history.
    # - If not, inspect data leakage and whether the task truly hides the cue.
    _ = (examples, config)
    raise NotImplementedError("Evaluate temporal-memory ablations.")


def save_memory_visuals(
    metrics: MemoryAblationMetrics,
    episode: list[MemoryRolloutStep],
    history_results: dict[int, float],
    config: TemporalMemoryConfig,
) -> dict[str, str]:
    """Save the ablation, episode-strip, and history-length plots."""
    # Goal:
    # Make the memory result easy to inspect.
    #
    # Method:
    # Use Matplotlib with the `Agg` backend.
    #
    # Plot 1:
    # - one bar per ablation
    # - y-axis is memory-required action accuracy
    # - include a horizontal chance line at `0.5`
    #
    # Plot 2:
    # - ordered real frames from one memory episode
    # - mark where the cue is visible and where the decision happens
    #
    # Plot 3:
    # - x-axis is history length
    # - y-axis is memory-task accuracy
    #
    # Return:
    # - a dictionary mapping plot name to saved path
    _ = (metrics, episode, history_results, config)
    raise NotImplementedError("Save temporal-memory visuals.")


def main() -> None:
    """Run the temporal-memory exercise scaffold."""
    config = TemporalMemoryConfig()
    try:
        episodes = load_memory_rollouts(config)
        vocabulary = build_instruction_vocabulary(episodes, config)
        examples = build_temporal_examples(episodes, vocabulary, config)
        metrics = evaluate_memory_ablations(examples, config)
        print(metrics)
    except NotImplementedError as error:
        print(f"[TODO] temporal_memory_for_3d_agents: {error}")


if __name__ == "__main__":
    main()
