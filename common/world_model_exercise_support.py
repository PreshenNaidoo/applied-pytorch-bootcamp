"""Small, tested helpers shared by the guided world-model exercises."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Sequence

import torch

from common.embodied_gridworld import LoggedGridStep


@dataclass(frozen=True)
class GridDataAudit:
    """Facts a learner should see before training a world model."""

    episodes: int
    transitions: int
    action_counts: dict[int, int]
    cost_events: int
    terminations: int
    truncations: int
    changed_observations: int

    @property
    def cost_event_fraction(self) -> float:
        return self.cost_events / max(self.transitions, 1)


def split_complete_episodes(
    episodes: Sequence[list[LoggedGridStep]], validation_fraction: float, seed: int
) -> tuple[list[list[LoggedGridStep]], list[list[LoggedGridStep]]]:
    """Split whole episodes so nearby frames cannot leak between splits."""

    if len(episodes) < 2:
        raise ValueError("Need at least two episodes for train and validation data.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def audit_grid_episodes(episodes: Sequence[Sequence[LoggedGridStep]]) -> GridDataAudit:
    """Count action coverage and rare outcomes before model training starts."""

    steps = [step for episode in episodes for step in episode]
    if not steps:
        raise ValueError("The dataset contains no transitions.")

    action_counts = Counter(int(step["action"]) for step in steps)
    return GridDataAudit(
        episodes=len(episodes),
        transitions=len(steps),
        action_counts=dict(sorted(action_counts.items())),
        cost_events=sum(float(step["cost"]) > 0.0 for step in steps),
        terminations=sum(bool(step["terminated"]) for step in steps),
        truncations=sum(bool(step["truncated"]) for step in steps),
        changed_observations=sum(
            not torch.equal(step["image"], step["next_image"]) for step in steps
        ),
    )


def print_grid_data_audit(audit: GridDataAudit) -> None:
    """Print the small set of counts that can reveal a broken dataset."""

    print(f"episodes: {audit.episodes}")
    print(f"transitions: {audit.transitions}")
    print(f"actions: {audit.action_counts}")
    print(
        "events: "
        f"cost={audit.cost_events}, terminated={audit.terminations}, "
        f"truncated={audit.truncations}"
    )
    print(f"changed observations: {audit.changed_observations}/{audit.transitions}")


def count_conflicting_targets(
    episodes: Sequence[Sequence[LoggedGridStep]], *, input_key: str
) -> int:
    """Count identical state-action inputs that have different real outcomes.

    A positive result means a deterministic model was not given enough
    information for every row. The rendered ``image`` and ``full_state`` are
    suitable keys. The older six-number ``state`` is expected to fail this
    check on a large enough collection because it omits walls.
    """

    outcomes: defaultdict[tuple[bytes, int], set[tuple[bytes, float, float, bool]]] = (
        defaultdict(set)
    )
    next_key = "next_image" if input_key == "image" else "next_full_state"
    for episode in episodes:
        for step in episode:
            value = step[input_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{input_key} must contain tensors.")
            key = (value.detach().cpu().contiguous().numpy().tobytes(), int(step["action"]))
            next_value = step[next_key]
            outcome = (
                next_value.detach().cpu().contiguous().numpy().tobytes(),
                float(step["reward"]),
                float(step["cost"]),
                bool(step["terminated"]),
            )
            outcomes[key].add(outcome)
    return sum(len(values) > 1 for values in outcomes.values())


def save_model_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    model_config: Any,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save model weights and the settings needed to rebuild the model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    config_values = asdict(model_config) if hasattr(model_config, "__dataclass_fields__") else model_config
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_values,
            "extra_state": extra_state or {},
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load one trusted local exercise checkpoint with a clear missing-file error."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has the wrong format: {path}")
    return checkpoint
