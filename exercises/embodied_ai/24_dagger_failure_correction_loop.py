"""
============================================================
EXERCISE 24: DAGGER FAILURE CORRECTION LOOP
============================================================

WHAT YOU ARE BUILDING
You are improving an embodied policy by correcting it where it fails.

DAgger stands for Dataset Aggregation.
The name means:

```text
run the current policy
-> ask an expert for the right action in states the policy visits
-> add those corrections to the dataset
-> train again
```

This is supervised imitation learning on states visited by the policy.
It does not learn from a reward-only trial-and-error loop.

REAL SYSTEM FIRST
A behavior-cloning policy trains on expert demonstrations.
At deployment, one small mistake can move it into a state that was rare or
missing in those demonstrations.
The next action is then harder, which can cause another mistake.

DAgger fixes this data problem by collecting examples from the policy's own
states.

Modern robot-learning tools use a practical human version:

```text
the policy acts
a person watches
the person takes over when the policy begins to fail
the corrected frames and actions are saved
the policy is fine-tuned on the combined data
```

Keep two collection modes separate:

```text
DAgger mode:
the policy action is executed
the expert labels the state
the policy therefore visits its own mistakes

safe takeover mode:
the expert action can replace an unsafe policy action
the replacement is recorded as an intervention
```

Use DAgger mode in the safe local simulator first.
Use takeover mode later when allowing every policy error would be unsafe.

FULL LOOP

```text
1. train the starting behavior-cloning policy
2. freeze one evaluation set of episode seeds
3. run the current policy in new training episodes
4. ask the expert for an action at every visited state
5. save disagreements and optional short correction segments
6. mix old demonstrations with new corrections
7. give correction rows enough sampling weight
8. fine-tune the policy
9. evaluate on the unchanged evaluation seeds
10. repeat for a small number of rounds
```

REAL SYSTEM VS THIS EXERCISE

real correction workflow            this exercise
---------------------------------------------------------------
person takes over a robot            scripted expert or keyboard takeover
many camera and robot streams        local first-person 3D frame
continuous action chunks             one discrete action at a time
large replay store                    saved local correction rows
many training rounds                 two or three small DAgger rounds
real safety supervisor                cost threshold and stop action

WHY THIS SMALLER VERSION EXISTS
Human correction is easiest to learn in a safe simulator.
The same dataset logic can later be used with LeRobot's rollout and DAgger
workflow.

DATA
Starting demonstrations:

- `outputs/mini3d_vla_dataset.pt` from `embodied_ai/11`

Starting policy:

- behavior-cloning policy from `embodied_ai/12`
- or the memory policy from `embodied_ai/22`
- save and reuse the same instruction vocabulary and maximum text length

If you use the memory policy, wrap it so `predict_action(...)` updates its
internal frame history. Clear that history in `reset_episode()` and pass the
action actually executed to `observe_action(...)`.

Correction data:

```text
outputs/dagger_3d/round_00_corrections.pt
outputs/dagger_3d/round_01_corrections.pt
```

Each correction row must include:

```text
image
instruction
policy_action
expert_action
executed_action
intervention
episode_id
step_id
round_id
reward
cost
done
```

`intervention=True` means the expert action replaced the policy action.
`policy_action != expert_action` means the two controllers disagreed.
Those are not the same thing in normal DAgger mode.

WHO IS THE EXPERT?
Use one of these in order:

1. a scripted shortest-path controller with full map access
2. keyboard input from a person
3. a stronger policy checkpoint

The expert may use hidden map state to create a label.
The student policy must not receive that hidden state.

TINY CONCRETE EXAMPLE

```text
policy sees a wall and predicts:
move forward

expert predicts:
turn left

saved safe-takeover correction:
policy_action=0
expert_action=1
executed_action=1
intervention=True
```

In default DAgger mode, the environment executes `move forward`.
That lets the next row show the state caused by the policy's mistake.
The saved row would have `executed_action=0` and `intervention=False`.

In optional safe-takeover mode, the environment executes `turn left` and sets
`intervention=True`.

MIXING OLD AND NEW DATA
Do not train only on the latest corrections.
That can make the policy forget normal behavior.

Start with:

```text
75% original demonstrations
25% correction rows
```

Also report a correction-heavy run:

```text
50% original demonstrations
50% correction rows
```

Use `torch.utils.data.WeightedRandomSampler` when the two sources have very
different sizes.

LOSS FUNCTION
Use the same behavior-cloning cross entropy:

```python
loss = nn.functional.cross_entropy(action_logits, expert_actions)
```

For non-intervention rows, `expert_actions` still contains the expert label.

METRICS TO TRACK

- task success before and after each round: higher is better
- safe-takeover intervention rate: lower should become better over rounds
- policy-expert disagreement rate: lower is better
- collision cost: lower is better
- recovery success after intervention: higher is better
- success on the original validation set: should not collapse
- success on unseen rooms: higher is better
- number of new correction rows

RECOVERY METRIC
After an intervention, give the policy control again.

`recovery success` means:

```text
the agent returns to useful behavior and finishes the task
```

This is stricter than action accuracy.

VISUALS TO SAVE
Save to `outputs/dagger_3d/`:

- `success_by_round.png`
- `intervention_rate_by_round.png`
- `original_vs_correction_accuracy.png`
- `correction_episode_strip.png`

COMMON MISTAKES

- collecting new rows but forgetting to add them to the next training set
- marking every disagreement as an intervention
- always executing the expert and never visiting policy-caused failure states
- training only on correction rows and forgetting normal behavior
- changing evaluation seeds between rounds
- counting expert-controlled steps as autonomous policy success
- saving only the expert action and losing the policy disagreement
- choosing correction rows from the final test set

STEP-BY-STEP TODO

1. Load the starting demonstrations and completed policy.
2. Build the expert-action interface.
3. Run policy-controlled training episodes.
4. Save policy and expert actions at every visited state.
5. In optional safe-takeover runs, mark and execute interventions.
6. save one correction dataset per round.
7. build a weighted mixed dataloader.
8. fine-tune the policy.
9. evaluate before and after on fixed untouched episodes.
10. plot success, interventions, and forgetting checks.

LEARNING ORDER
Complete the correction collector before the mixed dataloader.
Inspect one saved correction episode before training on it.

DEFINITION OF DONE

You are done when:

- corrections come from states visited by the current policy
- policy and expert actions are both saved
- the executed action is recorded clearly
- original and correction data are mixed
- evaluation episodes stay unchanged across rounds
- intervention rate and task success are reported for every round
- original validation performance is checked for forgetting
- at least one correction episode can be replayed visually
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class CorrectionStep(TypedDict):
    """One state visited by the policy and labelled by the expert."""

    image: torch.Tensor
    instruction: str
    policy_action: int
    expert_action: int
    executed_action: int
    intervention: bool
    episode_id: int
    step_id: int
    round_id: int
    reward: float
    cost: float
    done: bool


class DAggerRoundMetrics(TypedDict):
    """Measurements saved after one correction and fine-tuning round."""

    round_id: int
    num_correction_rows: int
    success_rate: float
    intervention_rate: float
    disagreement_rate: float
    mean_cost: float
    recovery_success_rate: float
    original_validation_accuracy: float
    unseen_room_success_rate: float


class ActionPolicy(Protocol):
    """Policy interface used during data collection."""

    def reset_episode(self) -> None:
        """Clear any recurrent or frame history before a new episode."""

    def predict_action(self, image: torch.Tensor, instruction: str) -> int:
        """Return one action id without learning."""

    def observe_action(self, action: int) -> None:
        """Record the action actually executed; reactive policies can do nothing."""


class ExpertController(Protocol):
    """Expert interface used only to label and correct visited states."""

    def get_action(
        self,
        image: torch.Tensor,
        instruction: str,
        evaluation_info: dict[str, object],
    ) -> int:
        """Return the expert action for the current state."""


class CorrectionEnvironment(Protocol):
    """Environment interface needed by the DAgger collector."""

    def reset(self, *, seed: int) -> tuple[torch.Tensor, str, dict[str, object]]:
        """Return first image, instruction, and evaluation-only information."""

    def step(
        self,
        action: int,
    ) -> tuple[torch.Tensor, float, float, bool, dict[str, object]]:
        """Execute one action and return next image and outcome fields."""


class CorrectionEnvironmentFactory(Protocol):
    """Callable that builds a fresh correction environment for one seed."""

    def __call__(self, *, seed: int) -> CorrectionEnvironment:
        """Return one environment instance."""


@dataclass
class DAggerConfig:
    """Settings for the local correction loop."""

    original_dataset_path: Path = Path("outputs/mini3d_vla_dataset.pt")
    output_dir: Path = Path("outputs/dagger_3d")
    num_rounds: int = 3
    episodes_per_round: int = 24
    max_steps: int = 50
    original_data_fraction: float = 0.75
    max_instruction_length: int = 12
    batch_size: int = 32
    fine_tune_epochs: int = 3
    learning_rate: float = 1e-4
    collection_mode: Literal["dagger", "safe_takeover"] = "dagger"
    disagreement_only: bool = False
    return_control_after_intervention: bool = True
    random_state: int = 42


def collect_dagger_round(
    environment_factory: CorrectionEnvironmentFactory,
    policy: ActionPolicy,
    expert: ExpertController,
    config: DAggerConfig,
    *,
    round_id: int,
    training_seeds: list[int],
) -> list[list[CorrectionStep]]:
    """Collect expert labels from states reached by the current policy."""
    # Where this fits:
    # This is the key DAgger difference. The starting demonstrations came from
    # expert states. These rows come from states reached by the learned policy.
    #
    # Goal:
    # Return ordered correction episodes.
    #
    # Steps:
    # 1. Build one environment per seed.
    # 2. Call `policy.reset_episode()`, then reset the environment and get the
    #    first real rendered frame.
    # 3. Ask `policy.predict_action(image, instruction)`.
    # 4. Ask `expert.get_action(image, instruction, evaluation_info)`.
    # 5. Set `disagreement = policy_action != expert_action`.
    # 6. If `collection_mode == "dagger"`:
    #    - execute the policy action
    #    - set `intervention=False`
    # 7. If `collection_mode == "safe_takeover"` and the actions disagree:
    #    - execute the expert action
    #    - set `intervention=True`
    #    Otherwise execute the policy action.
    # 8. Step the environment, call `policy.observe_action(executed_action)`,
    #    and store every `CorrectionStep` field.
    # 9. Stop on `done` or `config.max_steps`.
    #
    # If `config.disagreement_only` is true:
    # - save only rows where `policy_action != expert_action`
    # - still keep episode and step ids from the full rollout
    #
    # Checks:
    # - The expert may read `evaluation_info`.
    # - The policy must receive only the image and instruction.
    # - Never collect from final evaluation seeds.
    # - Default DAgger mode must execute policy actions so later rows include
    #   states caused by policy mistakes.
    _ = (
        environment_factory,
        policy,
        expert,
        config,
        round_id,
        training_seeds,
    )
    raise NotImplementedError("Collect one DAgger correction round.")


def save_correction_round(
    episodes: list[list[CorrectionStep]],
    config: DAggerConfig,
    *,
    round_id: int,
) -> Path:
    """Save one ordered correction dataset without overwriting older rounds."""
    # Goal:
    # Save to:
    # `config.output_dir / f"round_{round_id:02d}_corrections.pt"`
    #
    # Method:
    # 1. Create `config.output_dir` with `mkdir(parents=True, exist_ok=True)`.
    # 2. Use `torch.save(episodes, output_path)`.
    # 3. Return the output `Path`.
    #
    # Check:
    # - Do not merge or delete older rounds here.
    _ = (episodes, config, round_id)
    raise NotImplementedError("Save one DAgger correction round.")


class MixedDemonstrationDataset(Dataset):
    """Expose original demonstrations and new corrections in one format."""

    def __init__(
        self,
        original_rows: list[dict[str, object]],
        correction_rows: list[CorrectionStep],
        vocabulary: dict[str, int],
        config: DAggerConfig,
    ) -> None:
        # Goal:
        # Keep both data sources and return:
        # `(image, instruction_ids, expert_action, source_id)`.
        #
        # `source_id`:
        # - `0` means original demonstration
        # - `1` means DAgger correction
        #
        # Checks:
        # - Original rows use their logged action as `expert_action`.
        # - Correction rows use `row["expert_action"]`.
        # - Reuse the exact vocabulary from the starting policy.
        # - Use vocabulary id `0` for padding and `1` for unknown words.
        # - Pad or trim every instruction to `config.max_instruction_length`.
        self.original_rows = original_rows
        self.correction_rows = correction_rows
        self.vocabulary = vocabulary
        self.config = config
        raise NotImplementedError("Build the mixed demonstration dataset.")

    def __len__(self) -> int:
        """Return total original plus correction rows."""
        raise NotImplementedError("Return mixed dataset length.")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one policy-training row and its source id."""
        raise NotImplementedError("Return one mixed demonstration row.")


def build_weighted_dagger_loader(
    dataset: MixedDemonstrationDataset,
    config: DAggerConfig,
) -> DataLoader:
    """Sample the requested mix of original and correction rows."""
    # Goal:
    # Make a small correction dataset visible during fine-tuning without
    # discarding the larger original dataset.
    #
    # Method:
    # 1. Count original and correction rows.
    # 2. Give all original rows a per-row weight whose total is
    #    `config.original_data_fraction`.
    # 3. Give all correction rows a per-row weight whose total is
    #    `1.0 - config.original_data_fraction`.
    # 4. Build:
    #    `WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)`.
    # 5. Return:
    #    `DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)`.
    #
    # Checks:
    # - Raise a clear error when either source is empty.
    # - Do not also set `shuffle=True` when a sampler is provided.
    _ = (dataset, config, WeightedRandomSampler)
    raise NotImplementedError("Build the weighted DAgger dataloader.")


def fine_tune_on_corrections(
    policy: torch.nn.Module,
    loader: DataLoader,
    config: DAggerConfig,
) -> list[float]:
    """Fine-tune the policy with expert action labels."""
    # Goal:
    # Update the same action policy used for collection.
    #
    # Method:
    # 1. Use `torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)`.
    # 2. For each batch, compute policy action logits.
    # 3. Compute:
    #    `torch.nn.functional.cross_entropy(logits, expert_actions)`.
    # 4. Call `optimizer.zero_grad()`, `loss.backward()`, and
    #    `optimizer.step()`.
    # 5. Return mean loss for each epoch.
    #
    # Checks:
    # - Train on `expert_actions`, never `policy_actions`.
    # - Keep the original validation set untouched.
    _ = (policy, loader, config)
    raise NotImplementedError("Fine-tune the policy on mixed demonstrations.")


def evaluate_dagger_round(
    policy: ActionPolicy,
    expert: ExpertController,
    environment_factory: CorrectionEnvironmentFactory,
    correction_episodes: list[list[CorrectionStep]],
    original_validation_rows: list[dict[str, object]],
    config: DAggerConfig,
    *,
    round_id: int,
    fixed_evaluation_seeds: list[int],
) -> DAggerRoundMetrics:
    """Measure improvement and forgetting on fixed evaluation episodes."""
    # Goal:
    # Return every field in `DAggerRoundMetrics`.
    #
    # Method:
    # - autonomous run: execute only policy actions and measure success and cost
    # - disagreement check: ask the expert for labels on those same visited
    #   states without changing the executed action
    # - separate safe-takeover run: replace unsafe disagreements with expert
    #   actions and measure interventions / visited rows
    # - recovery: in that safe-takeover run, return control after an
    #   intervention and check task success
    # - original accuracy: evaluate the unchanged starting validation rows
    # - unseen success: use the fixed unseen-room seeds
    #
    # Checks:
    # - Do not count an expert-controlled episode as autonomous success.
    # - Use the exact same evaluation seeds after every round.
    # - Keep autonomous and safe-takeover metric rows separate internally.
    # - Use `float("nan")` for recovery when no intervention occurred.
    _ = (
        policy,
        expert,
        environment_factory,
        correction_episodes,
        original_validation_rows,
        config,
        round_id,
        fixed_evaluation_seeds,
    )
    raise NotImplementedError("Evaluate one DAgger round.")


def run_dagger_loop(
    environment_factory: CorrectionEnvironmentFactory,
    policy: torch.nn.Module,
    expert: ExpertController,
    vocabulary: dict[str, int],
    config: DAggerConfig,
) -> list[DAggerRoundMetrics]:
    """Collect, mix, fine-tune, and evaluate for several rounds."""
    # Goal:
    # Run the complete improvement loop.
    #
    # Steps:
    # 1. Load original demonstrations and make episode-safe splits.
    #    Also load the exact instruction vocabulary used by `policy`.
    # 2. Create fixed training and evaluation seed lists.
    # 3. Evaluate round 0 before collecting new corrections.
    # 4. For each DAgger round:
    #    a. collect policy-state corrections
    #    b. save that round
    #    c. combine corrections from all rounds so far
    #    d. build the weighted mixed dataloader
    #    e. fine-tune
    #    f. evaluate on fixed seeds
    # 5. Return the metric row for every round.
    #
    # Check:
    # - Later rounds must include earlier corrections, not only the newest file.
    _ = (environment_factory, policy, expert, vocabulary, config)
    raise NotImplementedError("Run the complete DAgger correction loop.")


def save_dagger_report(
    metrics: list[DAggerRoundMetrics],
    example_episode: list[CorrectionStep],
    config: DAggerConfig,
) -> dict[str, str]:
    """Save success, intervention, forgetting, and episode plots."""
    # Save:
    # - success by round
    # - intervention rate by round
    # - original validation accuracy beside correction accuracy
    # - an ordered strip with policy and expert actions
    #
    # Use Matplotlib's `Agg` backend so this works from the terminal.
    _ = (metrics, example_episode, config)
    raise NotImplementedError("Save the DAgger report.")


def main() -> None:
    """Run the DAgger exercise scaffold."""
    config = DAggerConfig()
    print(
        "Complete the earlier 3D policy, then run "
        f"{config.num_rounds} correction rounds."
    )


if __name__ == "__main__":
    main()
