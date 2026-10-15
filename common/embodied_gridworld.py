"""Small visual grid-world data for embodied-agent exercises."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import TypedDict

import torch


class LoggedGridStep(TypedDict):
    """One saved step from the visual grid-world."""

    image: torch.Tensor
    instruction: str
    state: torch.Tensor
    full_state: torch.Tensor
    action: int
    reward: float
    cost: float
    next_image: torch.Tensor
    next_state: torch.Tensor
    next_full_state: torch.Tensor
    terminated: bool
    truncated: bool
    done: bool


@dataclass
class VisualGridWorldConfig:
    """Settings for the small embodied-agent world."""

    grid_size: int = 7
    image_size: int = 28
    num_episodes: int = 64
    max_steps: int = 18
    random_state: int = 42
    wall_count: int = 5
    hazard_count: int = 2
    step_penalty: float = 0.01
    wall_cost: float = 0.2
    hazard_cost: float = 1.0
    goal_reward: float = 1.0
    explore_probability: float = 0.15


ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1, 0),  # up
    1: (1, 0),  # down
    2: (0, -1),  # left
    3: (0, 1),  # right
}

ACTION_NAMES: dict[int, str] = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}


def render_grid_image(
    agent: tuple[int, int],
    goal: tuple[int, int],
    hazards: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    config: VisualGridWorldConfig,
) -> torch.Tensor:
    """Render one grid state as a small RGB image."""

    image = torch.ones(3, config.image_size, config.image_size, dtype=torch.float32)
    cell = config.image_size // config.grid_size
    colors = {
        "wall": torch.tensor([0.45, 0.45, 0.45]),
        "hazard": torch.tensor([1.0, 0.0, 0.0]),
        "goal": torch.tensor([0.0, 0.85, 0.0]),
        "agent": torch.tensor([0.0, 0.25, 1.0]),
    }

    def paint(position: tuple[int, int], color: torch.Tensor) -> None:
        row, col = position
        row_start = row * cell
        col_start = col * cell
        image[:, row_start : row_start + cell, col_start : col_start + cell] = color[:, None, None]

    for wall in walls:
        paint(wall, colors["wall"])
    for hazard in hazards:
        paint(hazard, colors["hazard"])
    paint(goal, colors["goal"])
    paint(agent, colors["agent"])
    return image


def encode_grid_state(
    agent: tuple[int, int],
    goal: tuple[int, int],
    hazards: set[tuple[int, int]],
    config: VisualGridWorldConfig,
) -> torch.Tensor:
    """Encode the older compact state used by earlier exercises.

    This value is not the full environment state. It leaves out every wall and
    all but the first hazard. New dynamics exercises should use ``full_state``
    or the rendered image instead.
    """

    first_hazard = sorted(hazards)[0]
    scale = float(config.grid_size - 1)
    values = [
        agent[0] / scale,
        agent[1] / scale,
        goal[0] / scale,
        goal[1] / scale,
        first_hazard[0] / scale,
        first_hazard[1] / scale,
    ]
    return torch.tensor(values, dtype=torch.float32)


def encode_full_grid_state(
    current_position: tuple[int, int],
    goal_position: tuple[int, int],
    hazard_positions: set[tuple[int, int]],
    wall_positions: set[tuple[int, int]],
    config: VisualGridWorldConfig,
) -> torch.Tensor:
    """Encode every grid object as four binary channels."""

    full_state = torch.zeros(4, config.grid_size, config.grid_size, dtype=torch.float32)

    full_state[0, current_position[0], current_position[1]] = 1.0
    full_state[1, goal_position[0], goal_position[1]] = 1.0

    for row, col in hazard_positions:
        full_state[2, row, col] = 1.0

    for row, col in wall_positions:
        full_state[3, row, col] = 1.0

    return full_state


def make_grid_scene(
    config: VisualGridWorldConfig,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]:
    """Create one reachable grid-world scene."""

    cells = [(row, col) for row in range(config.grid_size) for col in range(config.grid_size)]

    while True:
        rng.shuffle(cells)
        agent = cells[0]
        goal = cells[1]
        hazards = set(cells[2 : 2 + config.hazard_count])
        wall_start = 2 + config.hazard_count
        walls = set(cells[wall_start : wall_start + config.wall_count])
        blocked = hazards | walls
        if shortest_path_action(agent, goal, walls, config) is not None and goal not in blocked and agent not in blocked:
            return agent, goal, hazards, walls


def step_grid_world(
    agent: tuple[int, int],
    action: int,
    goal: tuple[int, int],
    hazards: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    config: VisualGridWorldConfig,
) -> tuple[tuple[int, int], float, float, bool]:
    """Step the real grid-world once."""

    row_delta, col_delta = ACTION_DELTAS[int(action)]
    candidate = (agent[0] + row_delta, agent[1] + col_delta)
    inside = 0 <= candidate[0] < config.grid_size and 0 <= candidate[1] < config.grid_size

    reward = -config.step_penalty
    cost = 0.0
    next_agent = agent

    if not inside or candidate in walls:
        cost += config.wall_cost
    else:
        next_agent = candidate

    done = False
    if next_agent in hazards:
        cost += config.hazard_cost
    if next_agent == goal:
        reward += config.goal_reward
        done = True

    return next_agent, reward, cost, done


def shortest_path_action(
    start: tuple[int, int],
    goal: tuple[int, int],
    walls: set[tuple[int, int]],
    config: VisualGridWorldConfig,
) -> int | None:
    """Return the first action on a shortest wall-avoiding path."""

    queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start, [])])
    visited = {start}

    while queue:
        position, actions = queue.popleft()
        if position == goal:
            return actions[0] if actions else None
        for action, (row_delta, col_delta) in ACTION_DELTAS.items():
            next_position = (position[0] + row_delta, position[1] + col_delta)
            inside = 0 <= next_position[0] < config.grid_size and 0 <= next_position[1] < config.grid_size
            if not inside or next_position in walls or next_position in visited:
                continue
            visited.add(next_position)
            queue.append((next_position, actions + [action]))
    return None


def choose_scripted_action(
    agent: tuple[int, int],
    goal: tuple[int, int],
    hazards: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    config: VisualGridWorldConfig,
    rng: random.Random,
    explore_probability: float = 0.15,
) -> int:
    """Choose an action for data collection."""

    if rng.random() < explore_probability:
        return rng.randrange(len(ACTION_DELTAS))

    safe_walls = walls | hazards
    action = shortest_path_action(agent, goal, safe_walls, config)
    if action is not None:
        return action

    action = shortest_path_action(agent, goal, walls, config)
    if action is not None:
        return action

    return rng.randrange(len(ACTION_DELTAS))


def collect_visual_gridworld_episodes(config: VisualGridWorldConfig) -> list[list[LoggedGridStep]]:
    """Collect real logged rollouts from the visual grid-world."""

    rng = random.Random(config.random_state)
    episodes: list[list[LoggedGridStep]] = []
    instruction = "go to the green goal and avoid red hazards"

    for _ in range(config.num_episodes):
        agent, goal, hazards, walls = make_grid_scene(config, rng)
        episode: list[LoggedGridStep] = []

        for step_index in range(config.max_steps):
            image = render_grid_image(agent, goal, hazards, walls, config)
            state = encode_grid_state(agent, goal, hazards, config)
            full_state = encode_full_grid_state(agent, goal, hazards, walls, config)
            action = choose_scripted_action(
                agent,
                goal,
                hazards,
                walls,
                config,
                rng,
                explore_probability=config.explore_probability,
            )
            next_agent, reward, cost, terminated = step_grid_world(
                agent, action, goal, hazards, walls, config
            )
            next_image = render_grid_image(next_agent, goal, hazards, walls, config)
            next_state = encode_grid_state(next_agent, goal, hazards, config)
            next_full_state = encode_full_grid_state(next_agent, goal, hazards, walls, config)
            truncated = step_index == config.max_steps - 1 and not terminated
            done = terminated or truncated

            episode.append(
                {
                    "image": image,
                    "instruction": instruction,
                    "state": state,
                    "full_state": full_state,
                    "action": int(action),
                    "reward": float(reward),
                    "cost": float(cost),
                    "next_image": next_image,
                    "next_state": next_state,
                    "next_full_state": next_full_state,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": bool(done),
                }
            )

            agent = next_agent
            if done:
                break

        episodes.append(episode)

    return episodes


def flatten_logged_episodes(episodes: list[list[LoggedGridStep]]) -> list[LoggedGridStep]:
    """Flatten episodes into single-step training examples."""

    return [step for episode in episodes for step in episode]
