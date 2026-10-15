"""
============================================================
EXERCISE 10: MINIWORLD FIRST PERSON NAVIGATION
============================================================

WHAT YOU ARE BUILDING
You are building the first small 3D environment for the embodied-agent path.

The agent sees a first-person camera frame, receives a language goal, and
chooses keyboard-like actions.

This is the first step from the 2D visual grid-world toward a generalist
visual-action agent.

FULL WORKFLOW THIS EXERCISE STARTS

```text
embodied_ai/10: build a small first-person 3D environment
embodied_ai/11: save rollouts from that environment
embodied_ai/12: train a behavior-cloning policy from saved rollouts
embodied_ai/13: train a latent world model from saved rollouts
embodied_ai/14: plan with the learned world model and test in the real environment
embodied_ai/22: test whether a policy remembers a cue that is no longer visible
```

This exercise covers the environment step.
It does not train a neural network yet.
It gives later exercises the world they will collect data from.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
large 3D game or simulator          small 3D room or MiniWorld task
rich first-person frames            small RGB camera frames
keyboard and mouse actions          move, turn, strafe, stop
natural language goals              short goals like "go to the red cube"
long evaluation tasks               short navigation episodes

WHY THIS TOY EXISTS
The real systems use large 3D games or simulators.
That is too much for the first exercise.
This file uses a tiny local 3D-style room so you can learn the agent loop first:

```text
see a first-person frame -> choose a keyboard action -> get reward and cost
```

After this local version works, you can replace the room with MiniWorld,
AI2-THOR, Habitat, or another simulator while keeping the same return fields.

WHY THIS EXERCISE EXISTS
Systems such as SIMA act from pixels inside 3D virtual worlds.
Before training a policy or world model, you need an environment that can reset,
step, render frames, and report success, cost, and path length.

DATASET AND DOWNLOADS
Default dataset:
- no download
- generate rollouts locally by stepping this exercise's environment
- later exercises save those rollouts to `outputs/mini3d_vla_dataset.pt`

Build two task types:
- `navigation`: move to a visible target
- `memory_corridor`: see a red or blue cue, walk until it is hidden, then
  turn left for red or right for blue

The memory corridor matters because the final camera view can be identical for
both cues. A current-frame-only policy cannot solve that decision reliably.

Optional later simulator:
- MiniWorld can replace the local room after the interface works
- if you use MiniWorld, install it in your project environment first
- keep the same action ids, image shape, reward, cost, and done fields

Do not make the first version depend on internet downloads.
The first version should run from local code only.

MAIN LOOP IN PLAIN WORDS

```text
reset the 3D room
get a camera frame and instruction
choose an action
step the environment
get next camera frame, reward, cost, and done
repeat until success or timeout
```

WHAT THE AGENT KNOWS
- the current first-person image
- the instruction string
- its current pose if you include it for debugging

WHAT IS HIDDEN
- the full map
- the target position except what can be seen from the camera
- the hazard positions except what can be seen from the camera

The environment keeps that hidden state so it can score the task.
A scripted expert may also read it to create training labels.
The learned policy in later exercises must still act from images and language.

WHAT CHANGES DURING THE LOOP
- the agent pose changes when the action moves or turns
- the camera image changes after the pose changes
- reward, cost, and done come from the environment rule

TINY CONCRETE EXAMPLE

```text
instruction: "go to the red cube"
pose before action: x=1.0, z=2.0, heading=0 degrees
action: 0, move forward
pose after action: x=1.0, z=1.8, heading=0 degrees
reward: -0.01 because the agent used one step
cost: 0.0 because it did not hit a wall or hazard
done: False because it did not reach the cube yet
```

INPUTS AND OUTPUTS
- camera frame: `[3, image_size, image_size]`
- instruction: string
- action: int
- reward: float
- cost: float
- done: bool

METRICS TO TRACK
- `success`: `1.0` if the agent reaches the target, else `0.0`
- `total_reward`: sum of all rewards in the episode; higher is better
- `total_cost`: sum of wall, collision, or hazard costs; lower is better
- `path_length`: number of environment steps; lower is better after success
- `frame_strip_path`: saved image path showing a few frames from the rollout
- memory-corridor success: whether the final turn matches the earlier cue

VISUALS TO CREATE
Save a frame strip to `outputs/mini3d_navigation/random_rollout_strip.png`.
It should show the first-person view over time.
This helps you check whether actions really change what the agent sees.

TARGET RESULT RANGE
This exercise does not train a model.
A random agent can fail often and still be fine.

Useful signs:
- the frame strip is not blank
- turning changes the view
- moving changes the view when there is open space
- wall or hazard contact increases cost
- reaching the target gives success `1.0`

COMMON MISTAKES
- making the agent see the full map instead of a first-person view
- returning random images that do not depend on the agent pose
- changing the pose but forgetting to render a new image
- treating `stop` as success even when the agent is far from the target
- using reward only and forgetting cost
- making `run_random_3d_rollout(...)` learn; it should only measure

STEP-BY-STEP TODO
1. Define a tiny room with walls, one target cube, and one hazard.
2. Store the agent pose as `[x, z, heading]`.
3. Render a simple first-person RGB frame from the pose.
4. Implement the six action ids.
5. Return reward, cost, done, and the next frame after every action.
6. Run a random rollout.
7. Add the memory-corridor variant and mark its hidden-cue decision point.
8. Save a small frame strip.
9. Print success, total reward, total cost, and path length.

DEFINITION OF DONE
You are done when:
- the environment can reset and step
- a random rollout saves frames
- each episode reports success, cost, and path length
- the same interface can feed embodied exercises 11 to 14 and 22
- a memory-corridor episode hides its cue before the final decision
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class Mini3DState(TypedDict):
    """Everything needed to render and continue one small 3D episode."""

    image: torch.Tensor
    instruction: str
    agent_pose: torch.Tensor
    target_position: torch.Tensor
    hazard_position: torch.Tensor
    room_size: float
    step_count: int
    layout_seed: int
    task_variant: str
    cue_id: int
    cue_visible: bool
    memory_required: bool


class FirstPersonStep(TypedDict):
    """One transition returned after an environment action."""

    image: torch.Tensor
    instruction: str
    action: int
    reward: float
    cost: float
    next_image: torch.Tensor
    done: bool
    agent_pose: torch.Tensor
    success: bool
    layout_seed: int
    task_variant: str
    cue_id: int
    cue_visible: bool
    memory_required: bool
    next_state: Mini3DState


@dataclass
class Mini3DNavigationConfig:
    """Settings for the small 3D navigation environment."""

    image_size: int = 64
    num_actions: int = 6
    max_steps: int = 40
    memory_task_fraction: float = 0.5
    random_state: int = 42
    output_dir: str = "outputs/mini3d_navigation"


def render_first_person_frame(
    state: Mini3DState,
    config: Mini3DNavigationConfig,
) -> torch.Tensor:
    """Render the agent's current first-person view."""
    # Where this fits:
    # Reset and step both need this helper. Keeping rendering in one function
    # makes it easier to debug the environment.
    #
    # Goal:
    # Return an RGB frame that depends on the agent pose and nearby objects.
    #
    # Input:
    # - `state["agent_pose"]`: tensor shaped `[3]`, meaning `[x, z, heading]`.
    # - `state["target_position"]`: tensor shaped `[2]`.
    # - `state["hazard_position"]`: tensor shaped `[2]`.
    #
    # Output:
    # - frame tensor shaped `[3, image_size, image_size]`.
    #
    # Method:
    # First version:
    # 1. Start with a light background using `torch.ones(...)`.
    # 2. Draw a colored target block if it is roughly in front of the agent.
    # 3. Draw a hazard block if it is roughly in front of the agent.
    # 4. Use the heading to shift objects left or right in the image.
    # 5. For `task_variant="memory_corridor"`, draw the red or blue cue only
    #    while `state["cue_visible"]` is true. Do not leave a color hint in
    #    later corridor frames.
    #
    # Check:
    # - If the agent turns, the target should move across the image or disappear.
    _ = (state, config)
    raise NotImplementedError("Render one first-person frame.")


def reset_mini_3d_environment(
    config: Mini3DNavigationConfig,
    *,
    seed: int | None = None,
    cue_id: int | None = None,
) -> Mini3DState:
    """Start a new first-person navigation episode."""
    # Where this fits:
    # This is the first call in every episode. Later exercises call this before
    # collecting data, training a policy, or testing a planner.
    #
    # Goal:
    # Return the first camera frame, the instruction, and the starting pose.
    #
    # Input:
    # - `config.image_size`: the height and width of the returned square image.
    # - `seed`: controls this episode's layout.
    # - when `seed` is `None`, use `config.random_state`.
    # - `cue_id`: optionally force red `0` or blue `1` so the logger can make
    #   paired episodes with identical geometry. Passing either value also
    #   forces `task_variant="memory_corridor"`.
    #
    # Method:
    # For the first version, use a small local room, not an external simulator.
    # You can represent the room with normal Python data:
    # - room size, such as `4.0` by `4.0`
    # - agent pose tensor `[x, z, heading]`
    # - target position tensor `[x, z]`
    # - hazard position tensor `[x, z]`
    #
    # Use `torch.zeros(3, config.image_size, config.image_size)` for the image
    # at first, then draw colored blocks into it. This can be simple. The point
    # is to create a frame that changes when the agent moves.
    #
    # Output:
    # Return a dictionary with at least:
    # - `"image"`: float tensor shaped `[3, image_size, image_size]`
    # - `"instruction"`: string such as `"go to the red cube"`
    # - `"agent_pose"`: tensor shaped `[3]`, meaning `[x, z, heading]`
    # - `"target_position"`: tensor shaped `[2]`
    # - `"hazard_position"`: tensor shaped `[2]`
    # - `"room_size"`: width and depth of the square room
    # - `"step_count"`: `0` at reset
    # - `"layout_seed"`: the seed used to build geometry
    # - `"task_variant"`: `"navigation"` or `"memory_corridor"`
    # - `"cue_id"`: `0` for red, `1` for blue, or `-1` for normal navigation
    # - `"cue_visible"`: whether the cue should appear in the current frame
    # - `"memory_required"`: true only at the hidden-cue decision point
    #
    # For a memory corridor:
    # 1. Use the requested `cue_id`, or randomly choose 0 or 1 when it is
    #    `None`.
    # 2. Show the cue near the start.
    # 3. Hide it after the agent moves down the corridor.
    # 4. Make the later corridor view the same for both cue ids.
    # 5. Reward left at the junction for red and right for blue.
    #
    # Checks:
    # - The image should be a float tensor.
    # - Pixel values should be in `[0, 1]`.
    # - The pose should not start inside a wall or hazard.
    # - The same seed should reproduce the same layout.
    # - Different collection seeds should produce varied valid layouts.
    # - The same seed with cue ids 0 and 1 should keep geometry identical.
    # - A provided cue id must create a memory-corridor episode.
    _ = (config, seed, cue_id)
    raise NotImplementedError("Reset the mini 3D navigation environment.")


def step_mini_3d_environment(
    state: Mini3DState,
    action: int,
    config: Mini3DNavigationConfig,
) -> FirstPersonStep:
    """Take one keyboard-like action in the 3D environment."""
    # Where this fits:
    # This is the environment step. It is the 3D version of Gymnasium's
    # `env.step(action)`: action in, next observation and scores out.
    #
    # Goal:
    # Apply one action and return the transition in the shared embodied format.
    #
    # Input:
    # - `state`: the dictionary returned by `reset_mini_3d_environment(...)`
    #   or the updated state after a previous step.
    # - `action`: one integer from `0` to `5`.
    #
    # Actions:
    # - 0: move forward
    # - 1: turn left
    # - 2: turn right
    # - 3: strafe left
    # - 4: strafe right
    # - 5: stop
    #
    # Scoring rule:
    # - start with `reward = -0.01` as a small step penalty
    # - add `+1.0` reward if the agent reaches the target
    # - add cost, such as `+1.0`, if the agent hits a wall or hazard
    # - set `done=True` when the target is reached, stop is chosen at the
    #   target, or the episode times out
    #
    # Output:
    # Return a FirstPersonStep with:
    # - `image`: current frame before the action
    # - `next_image`: frame after the action
    # - `agent_pose`: new pose tensor shaped `[3]`
    # - `success`: true only when the target was reached
    # - `layout_seed`, `task_variant`, `cue_id`, `cue_visible`, and
    #   `memory_required`
    # - `next_state`: the full updated `Mini3DState` used by the next call
    #
    # `cue_visible` and `memory_required` describe the current frame and the
    # action chosen from that frame. Their next values live in `next_state`.
    #
    # Checks:
    # - Reward should increase when the goal is reached.
    # - Cost should increase for collisions or hazards.
    # - The returned `next_image` shape should match `image.shape`.
    # - `next_state["image"]` should equal the returned `next_image`.
    # - At a memory decision point, `cue_visible` must be false.
    # - Changing only `cue_id` must not change the decision-point frame.
    _ = (state, action, config)
    raise NotImplementedError("Step the mini 3D navigation environment.")


def choose_scripted_3d_action(
    state: Mini3DState,
    config: Mini3DNavigationConfig,
) -> int:
    """Choose an expert action from the environment's hidden state."""
    # Where this fits:
    # Exercise 11 needs correct actions for behavior-cloning labels.
    # Exercises 23 and 24 reuse the same idea for oracle skills and DAgger
    # corrections.
    #
    # This helper is an expert, not the learned policy.
    # It may read target positions, the cue id, and other hidden environment
    # state. Do not give those fields to the policy being trained.
    #
    # Goal:
    # Return one action id from `0` to `5` that moves toward task success.
    #
    # Method for normal navigation:
    # 1. Subtract the agent's `[x, z]` position from the target position.
    # 2. Use `torch.atan2(...)` to get the desired heading.
    # 3. Wrap the heading error to the range `[-pi, pi]`.
    # 4. Turn left or right while the error is larger than a small threshold.
    # 5. Otherwise move forward.
    # 6. Stop only when the agent is inside the success radius.
    #
    # Method for the memory corridor:
    # 1. Follow the fixed corridor while the junction is still ahead.
    # 2. At the hidden-cue decision point, choose left for `cue_id=0`.
    # 3. Choose right for `cue_id=1`.
    # 4. Continue toward the goal and stop only after reaching it.
    #
    # Checks:
    # - Validate the returned id with `0 <= action < config.num_actions`.
    # - Run this expert on many seeds and report its success rate.
    # - If it does not solve almost every valid local room, fix the expert
    #   before using its actions as labels.
    _ = (state, config)
    raise NotImplementedError("Choose one hidden-state scripted expert action.")


def run_random_3d_rollout(config: Mini3DNavigationConfig) -> dict[str, float]:
    """Run a random policy and report basic episode metrics."""
    # Where this fits:
    # This is the first final check. It proves the environment can run before
    # any learning code uses it.
    #
    # Goal:
    # Make sure the environment works before training any model.
    #
    # Method:
    # 1. Call `reset_mini_3d_environment(config)`.
    # 2. For at most `config.max_steps`:
    #    - sample `action = torch.randint(0, config.num_actions, ()).item()`
    #    - call `step_mini_3d_environment(state, action, config)`
    #    - add reward and cost to running totals
    #    - save a few frames for a frame strip
    #    - set `state = transition["next_state"]`
    #    - stop if `done` is true
    # 3. Save a simple frame strip with Matplotlib.
    #    Use `matplotlib.use("Agg")` before importing `pyplot`.
    #
    # Output:
    # - `"success"`: 0.0 or 1.0
    # - `"total_reward"`: total episode reward
    # - `"total_cost"`: total episode cost
    # - `"path_length"`: number of steps taken
    # - `"frame_strip_saved"`: 1.0 if the visual was saved
    #
    # Checks:
    # - This function only measures. It does not learn.
    # - A random agent does not need to score well. It only needs to produce a
    #   valid rollout and interpretable metrics.
    raise NotImplementedError("Run and score a random first-person 3D rollout.")


def save_rollout_frame_strip(frames: list[torch.Tensor], config: Mini3DNavigationConfig) -> str:
    """Save a strip of frames from one rollout."""
    # Where this fits:
    # This is the visual proof that the environment is producing useful frames.
    #
    # Goal:
    # Save several frames side by side.
    #
    # Input:
    # - `frames`: list of tensors shaped `[3, image_size, image_size]`.
    #
    # Output:
    # - path string for `outputs/mini3d_navigation/random_rollout_strip.png`.
    #
    # Method:
    # 1. Use `matplotlib.use("Agg")`.
    # 2. Create `config.output_dir`.
    # 3. Plot up to 6 frames.
    # 4. Convert each frame to `[height, width, 3]` with `.permute(1, 2, 0)`.
    # 5. Save and return the path.
    #
    # Check:
    # - The saved strip should show changes over time.
    _ = (frames, config)
    raise NotImplementedError("Save a random rollout frame strip.")


def main() -> None:
    """Run the 3D navigation environment scaffold."""
    config = Mini3DNavigationConfig()
    try:
        print(run_random_3d_rollout(config))
    except NotImplementedError as error:
        print(f"[TODO] run_random_3d_rollout: {error}")


if __name__ == "__main__":
    main()
