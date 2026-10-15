"""
============================================================
EXERCISE 23: HIERARCHICAL VLM SUBGOAL AGENT
============================================================

WHAT YOU ARE BUILDING
You are adding a slow visual-language reasoner above a faster action policy.

```text
slow planner:
image + goal + memory summary -> subgoal

fast controller:
image + subgoal -> immediate keyboard action
```

The slow model does not press every key.
It chooses a small skill such as:

```text
look_for_target
approach_target
avoid_hazard
go_around_obstacle
stop_at_target
```

The fast controller executes actions for a short time.
The slow planner then looks again and may choose a new subgoal.

REAL SYSTEM FIRST
Newer embodied systems often separate expensive reasoning from fast control.

The high-level model handles:

- the language goal
- scene meaning
- task progress
- choosing a subgoal
- deciding when to replan

The low-level policy handles:

- movement
- turning
- avoiding immediate obstacles
- producing actions quickly

This is useful because a large VLM can be too slow for every environment
step, while a small action policy may be too weak for long instructions.

FULL LOOP

```text
1. reset the environment
2. read the overall goal
3. look at the current frame and compact memory summary
4. ask the frozen VLM for one allowed subgoal
5. validate the VLM output
6. turn the subgoal into the text input used by the low-level policy
7. run the low-level policy for at most `skill_horizon` steps
8. stop early if the task ends, progress stalls, or danger rises
9. update the memory summary
10. ask for another subgoal
11. repeat until success or the step budget ends
```

REAL SYSTEM VS THIS EXERCISE

real generalist embodied agent      this exercise
---------------------------------------------------------------
large multimodal reasoner           small frozen open VLM
large learned skill library         five fixed navigation skills
complex learned memory              short structured memory summary
continuous keyboard and mouse       six discrete actions
many games and tasks                local first-person 3D rooms
learned progress verifier           simple progress and safety signals

WHY THIS SMALLER VERSION EXISTS
The important lesson is the boundary between reasoning and control.
You do not need to fine-tune a large VLM to learn that system shape.

The VLM stays frozen.
Only the earlier low-level policy is trained.

DATA AND MODELS
Environment:

- use the local 3D environment from `embodied_ai/10`
- use logged or live first-person frames
- test seen and unseen rooms

Low-level policy:

- reuse the behavior-cloning architecture from `embodied_ai/12`
- collect new rows labelled with the five allowed skill instructions
- train or fine-tune the controller on those skill-labelled rows
- optionally add the memory architecture from `embodied_ai/22`
- expose one method:
  `predict_action(image, instruction, recent_steps) -> int`

Do not use an old whole-task checkpoint without skill-labelled fine-tuning.
A controller trained only on `"go to the red cube"` has not automatically
learned what `"go around the left side of the wall"` means.

High-level VLM:

- default small option:
  `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
- stronger upgrade when memory allows:
  `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- load it with `transformers.AutoProcessor.from_pretrained(...)`
- load the model with
  `transformers.AutoModelForImageTextToText.from_pretrained(...)`
- use `dtype=torch.bfloat16` on supported GPUs
- if the model does not fit, run subgoal generation offline and save the
  returned JSON beside each evaluation frame

Do not use a text-only model here.
The planner must receive the current image.

ALLOWED SUBGOAL OUTPUT
Require one JSON object:

```json
{
  "skill": "go_around_obstacle",
  "target": "red cube",
  "reason": "the wall blocks the direct path",
  "replan_after_steps": 4
}
```

Only `skill` and `target` are given to the action policy.
`reason` is saved for inspection.
It is not treated as proof that the model reasoned correctly.

INPUTS AND OUTPUTS

VLM input:

```text
current RGB frame: [3, H, W]
overall goal: string
memory summary: string
previous subgoal: string or "none"
recent progress: float
recent cost: float
```

VLM output:

```text
one validated Subgoal dictionary
```

Low-level policy input:

```text
current RGB frame: [3, H, W]
skill instruction: string
optional sequence memory
```

Low-level policy output:

```text
one action id from 0 to 5
```

TINY CONCRETE EXAMPLE

```text
overall goal:
"reach the red cube without touching the hazard"

current frame:
red cube is visible behind a wall

VLM subgoal:
{"skill": "go_around_obstacle", "target": "left side of wall", ...}

low-level policy actions:
turn left, move forward, move forward

new frame:
the cube is now reachable

next VLM subgoal:
{"skill": "approach_target", "target": "red cube", ...}
```

ORACLE SUBGOAL BASELINE
`oracle` means the environment provides the correct high-level subgoal.
The action policy still has to execute it.

This gives an upper-bound check:

```text
low oracle success:
the low-level skills are weak

high oracle success but low VLM success:
the high-level planner or prompt is weak
```

This is much more useful than reporting one final success number.

METRICS TO TRACK

- low-level action accuracy and rollout success for each allowed skill
- oracle-subgoal success rate: higher is better
- VLM-subgoal success rate: higher is better
- subgoal accuracy against oracle labels: higher is better
- invalid JSON rate: lower is better
- high-level replans per successful episode: lower is usually better
- unnecessary replan rate: lower is better
- recovery rate after a bad subgoal: higher is better
- collision cost: lower is better
- mean VLM latency in milliseconds: lower is better
- low-level action latency in milliseconds: lower is better

VISUALS TO SAVE
Save to `outputs/hierarchical_vlm_agent/`:

- a timeline of frames, subgoals, actions, reward, and cost
- oracle versus VLM success bars
- latency split between the planner and controller
- a failure gallery with the returned JSON

COMMON MISTAKES

- letting the VLM choose arbitrary text that the controller never saw
- calling the large VLM at every action step without measuring latency
- allowing invalid JSON to crash the episode
- using hidden map information in the learned VLM prompt
- comparing oracle and VLM planners with different low-level policies
- judging the agent from the VLM's written reason instead of the rollout
- hiding failed or invalid subgoals from the final report

STEP-BY-STEP TODO

1. Define the closed skill list and `Subgoal` schema.
2. write the exact image-and-text VLM prompt.
3. parse and validate one JSON response.
4. wrap the completed low-level policy behind `LowLevelPolicy`.
5. collect skill-labelled low-level demonstrations.
6. train and test the skill-conditioned controller.
7. build the compact visible-history summary.
8. build an oracle subgoal planner from evaluation-only environment state.
9. load the frozen VLM and generate one subgoal.
10. run the hierarchical loop with fixed replan intervals.
11. add early replanning after stalls, danger, or invalid actions.
12. compare oracle and VLM subgoals on the same episodes.
13. save timelines, latency, and failures.

LEARNING ORDER
Complete the schema and parser before loading a model.
The parser defines the safe boundary between open text generation and the
fixed action system.

DEFINITION OF DONE

You are done when:

- the low-level controller was trained and tested on every allowed skill
- a frozen VLM receives both the image and language goal
- only validated skills reach the low-level controller
- the controller is the same for oracle and VLM evaluation
- the VLM is not called on every low-level action
- oracle and learned-subgoal success are reported separately
- invalid output and recovery are measured
- high-level and low-level latency are measured separately
- a full episode timeline and failure gallery are saved
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, TypedDict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin


SkillName = Literal[
    "look_for_target",
    "approach_target",
    "avoid_hazard",
    "go_around_obstacle",
    "stop_at_target",
]


class Subgoal(TypedDict):
    """One validated high-level instruction."""

    skill: SkillName
    target: str
    reason: str
    replan_after_steps: int


class HierarchicalStep(TypedDict):
    """One low-level step plus the high-level decision that controlled it."""

    image: torch.Tensor
    overall_goal: str
    subgoal: Subgoal
    action: int
    reward: float
    cost: float
    done: bool
    success: bool
    planner_latency_ms: float
    action_latency_ms: float
    replan_reason: str


class HierarchicalMetrics(TypedDict):
    """Metrics that separate high-level planning from low-level control."""

    oracle_success_rate: float
    vlm_success_rate: float
    subgoal_accuracy: float
    invalid_output_rate: float
    mean_replans_per_success: float
    unnecessary_replan_rate: float
    recovery_rate: float
    mean_cost: float
    mean_planner_latency_ms: float
    mean_action_latency_ms: float


class SkillPolicyExample(TypedDict):
    """One low-level action labelled with an oracle skill instruction."""

    image: torch.Tensor
    skill_instruction: str
    expert_action: int
    episode_id: int
    split: str


class LowLevelPolicy(Protocol):
    """Small interface around the completed action policy."""

    def predict_action(
        self,
        image: torch.Tensor,
        instruction: str,
        recent_steps: list[HierarchicalStep],
    ) -> int:
        """Return one environment action id."""


class HierarchicalEnvironment(Protocol):
    """Environment interface needed by the hierarchical loop."""

    def reset(self, *, seed: int) -> tuple[torch.Tensor, dict[str, object]]:
        """Return the first image and evaluation-only information."""

    def step(
        self,
        action: int,
    ) -> tuple[torch.Tensor, float, float, bool, dict[str, object]]:
        """Apply one action and return image, reward, cost, done, and info."""


class HierarchicalEnvironmentFactory(Protocol):
    """Callable that builds a fresh environment for one fixed seed."""

    def __call__(self, *, seed: int) -> HierarchicalEnvironment:
        """Return one resettable evaluation environment."""


@dataclass
class HierarchicalAgentConfig:
    """Settings for the slow-planner and fast-controller experiment."""

    vlm_model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    output_dir: Path = Path("outputs/hierarchical_vlm_agent")
    allowed_skills: tuple[SkillName, ...] = (
        "look_for_target",
        "approach_target",
        "avoid_hazard",
        "go_around_obstacle",
        "stop_at_target",
    )
    default_skill_horizon: int = 4
    max_skill_horizon: int = 8
    max_episode_steps: int = 60
    max_replans: int = 20
    stall_window: int = 4
    danger_replan_threshold: float = 0.5
    max_new_tokens: int = 96
    skill_training_episodes: int = 96
    skill_validation_episodes: int = 24
    skill_batch_size: int = 32
    skill_epochs: int = 6
    random_state: int = 42


def collect_skill_policy_examples(
    environment_factory: HierarchicalEnvironmentFactory,
    config: HierarchicalAgentConfig,
) -> list[SkillPolicyExample]:
    """Collect low-level demonstrations labelled with allowed skills."""
    # Where this fits:
    # The VLM can only choose useful subgoals if the controller was trained to
    # follow the same skill words.
    #
    # Goal:
    # Build real rendered state-action rows for all allowed skills.
    #
    # Method:
    # 1. Use separate fixed episode seeds for train and validation.
    # 2. At each state, read evaluation-only environment information.
    # 3. Call `choose_oracle_subgoal(...)`.
    # 4. Turn it into stable text with `subgoal_to_policy_instruction(...)`.
    # 5. Ask the scripted shortest-path controller for the expert low-level
    #    action. The scripted controller may read hidden state.
    # 6. Save the current real image, skill text, expert action, episode id,
    #    and split.
    # 7. Execute the expert action and continue until the episode ends.
    #
    # Checks:
    # - Print row counts for every allowed skill.
    # - Keep whole episodes in one split.
    # - The learned controller receives only image and skill text.
    # - Hidden state creates labels but is never a controller input.
    _ = (environment_factory, config)
    raise NotImplementedError("Collect skill-labelled controller examples.")


def train_skill_conditioned_controller(
    examples: list[SkillPolicyExample],
    config: HierarchicalAgentConfig,
) -> tuple[LowLevelPolicy, dict[str, float]]:
    """Train the fast controller and report accuracy for each skill."""
    # Goal:
    # Reuse the CNN-and-text policy shape from `embodied_ai/12`, but train it
    # on the exact skill instructions produced in this file.
    #
    # Steps:
    # 1. Build the word vocabulary from training rows only.
    # 2. Tokenize each `skill_instruction` with fixed padding.
    # 3. Build train and validation dataloaders by episode split.
    # 4. Train with action cross entropy.
    # 5. Report validation action accuracy for each of the five skills.
    # 6. Save the model and vocabulary together.
    #
    # Checks:
    # - Do not move on to VLM evaluation if one required skill has no rows.
    # - Run oracle-subgoal rollouts. Classification accuracy alone is not
    #   enough to prove the controller can finish a skill.
    _ = (examples, config)
    raise NotImplementedError("Train the skill-conditioned low-level policy.")


def build_subgoal_prompt(
    overall_goal: str,
    memory_summary: str,
    previous_subgoal: Subgoal | None,
    recent_progress: float,
    recent_cost: float,
    config: HierarchicalAgentConfig,
) -> str:
    """Return the exact prompt that constrains the VLM to the skill schema."""
    # Where this fits:
    # The prompt is the contract between open language generation and the
    # closed skill library.
    #
    # Goal:
    # Ask for exactly one JSON object and list every allowed skill.
    #
    # Include:
    # - the overall goal
    # - the memory summary
    # - the previous subgoal or `"none"`
    # - recent progress and cost
    # - the allowed skill names
    # - the four required JSON keys
    # - `1 <= replan_after_steps <= config.max_skill_horizon`
    #
    # Check:
    # - Do not include target position, shortest path, or other hidden state.
    _ = (
        overall_goal,
        memory_summary,
        previous_subgoal,
        recent_progress,
        recent_cost,
        config,
    )
    raise NotImplementedError("Build the constrained VLM subgoal prompt.")


def parse_subgoal_json(raw_text: str, config: HierarchicalAgentConfig) -> Subgoal:
    """Parse one VLM response and reject values outside the fixed schema."""
    # Where this fits:
    # Generated text must not directly control the environment.
    #
    # Goal:
    # Turn valid JSON into a `Subgoal` and raise `ValueError` for bad output.
    #
    # Method:
    # 1. Use `json.loads(raw_text)`.
    # 2. Check that the result is a dictionary.
    # 3. Check that the exact required keys exist.
    # 4. Check that `skill` is in `config.allowed_skills`.
    # 5. Check that `target` and `reason` are strings.
    # 6. Check that `replan_after_steps` is an integer from 1 through
    #    `config.max_skill_horizon`.
    # 7. Return a `Subgoal`.
    #
    # Do not:
    # - use `eval(...)`
    # - silently map unknown skills to arbitrary actions
    _ = (raw_text, config, json)
    raise NotImplementedError("Parse and validate one VLM subgoal.")


class FrozenVLMSubgoalPlanner:
    """Use a frozen image-and-text model to choose one allowed subgoal."""

    def __init__(
        self,
        processor: ProcessorMixin,
        model: PreTrainedModel,
        config: HierarchicalAgentConfig,
    ) -> None:
        self.processor = processor
        self.model = model
        self.config = config

    @classmethod
    def from_pretrained(cls, config: HierarchicalAgentConfig) -> FrozenVLMSubgoalPlanner:
        """Load the processor and frozen image-to-text model."""
        # Goal:
        # Load a real pretrained VLM without training it.
        #
        # Method:
        # Import inside this method:
        # `from transformers import AutoModelForImageTextToText, AutoProcessor`
        #
        # Exact calls:
        # `if torch.cuda.is_available():`
        # `    model_dtype = (`
        # `        torch.bfloat16`
        # `        if torch.cuda.is_bf16_supported()`
        # `        else torch.float16`
        # `    )`
        # `else:`
        # `    model_dtype = torch.float32`
        #
        # `processor = AutoProcessor.from_pretrained(config.vlm_model_id)`
        # `model = AutoModelForImageTextToText.from_pretrained(`
        # `    config.vlm_model_id,`
        # `    dtype=model_dtype,`
        # `    device_map="auto" if torch.cuda.is_available() else None,`
        # `)`
        #
        # Then:
        # - call `model.eval()`
        # - set `parameter.requires_grad_(False)` for every parameter
        # - return `cls(processor, model, config)`
        _ = config
        raise NotImplementedError("Load the frozen visual-language planner.")

    def choose_subgoal(
        self,
        image: torch.Tensor,
        prompt: str,
    ) -> tuple[Subgoal, str, float]:
        """Generate, validate, and time one subgoal."""
        # Goal:
        # Return `(validated_subgoal, raw_text, latency_ms)`.
        #
        # Method:
        # 1. Convert `[3, H, W]` float image values from `[0, 1]` to a PIL image
        #    with `torchvision.transforms.functional.to_pil_image(image.cpu())`.
        # 2. Build one chat message:
        #    `{"role": "user", "content": [`
        #    `    {"type": "image", "image": pil_image},`
        #    `    {"type": "text", "text": prompt},`
        #    `]}`
        # 3. Call:
        #    `processor.apply_chat_template(`
        #    `    messages,`
        #    `    add_generation_prompt=True,`
        #    `    tokenize=True,`
        #    `    return_dict=True,`
        #    `    return_tensors="pt",`
        #    `)`.
        # 4. Move every returned tensor to the model device.
        # 5. Save `input_length = inputs["input_ids"].shape[1]`.
        # 6. Time `model.generate(**inputs, max_new_tokens=...)` with
        #    `perf_counter()`.
        # 7. Slice generated ids with `generated_ids[:, input_length:]` so the
        #    prompt is not included in the decoded answer.
        # 8. Decode with
        #    `processor.batch_decode(new_ids, skip_special_tokens=True)`.
        # 9. Call `parse_subgoal_json(...)`.
        #
        # Checks:
        # - Wrap generation in `torch.inference_mode()`.
        # - Record invalid responses before applying a safe fallback.
        _ = (image, prompt, perf_counter)
        raise NotImplementedError("Generate one VLM subgoal.")


def subgoal_to_policy_instruction(subgoal: Subgoal) -> str:
    """Turn a validated skill into text for the low-level policy."""
    # Goal:
    # Use one stable wording for each skill.
    #
    # Example:
    # `{"skill": "approach_target", "target": "red cube", ...}`
    # becomes:
    # `"approach the red cube"`
    #
    # Check:
    # - Do not pass the free-form `reason` to the low-level controller.
    _ = subgoal
    raise NotImplementedError("Turn a subgoal into a policy instruction.")


def build_memory_summary(
    recent_steps: list[HierarchicalStep],
    *,
    max_steps: int = 6,
) -> str:
    """Summarize recent visible actions and outcomes for the next VLM call."""
    # Where this fits:
    # The slow planner is not called every step, but its next decision should
    # know what the controller just tried.
    #
    # Goal:
    # Return short plain text containing only information available to the
    # acting agent.
    #
    # Include for the most recent `max_steps` rows:
    # - validated skill name
    # - chosen low-level action id
    # - reward
    # - cost
    # - whether the task ended
    #
    # Do not include:
    # - target coordinates
    # - shortest-path direction
    # - hidden object identities
    # - oracle subgoals
    #
    # Check:
    # - Return `"no recent history"` when the list is empty.
    _ = (recent_steps, max_steps)
    raise NotImplementedError("Build the visible recent-history summary.")


def choose_oracle_subgoal(
    evaluation_info: dict[str, object],
    config: HierarchicalAgentConfig,
) -> Subgoal:
    """Choose the correct high-level skill using evaluation-only state."""
    # Where this fits:
    # This is the high-level upper bound. It tests whether the low-level skill
    # policy can solve the task when the subgoal is correct.
    #
    # Goal:
    # Map the local environment's hidden state to one allowed `Subgoal`.
    #
    # Input examples:
    # - target visible or hidden
    # - obstacle blocks direct route
    # - hazard is near the planned path
    # - agent is already at the target
    #
    # Method:
    # Use a small ordered rule set:
    # 1. at target -> `stop_at_target`
    # 2. immediate hazard -> `avoid_hazard`
    # 3. blocking obstacle -> `go_around_obstacle`
    # 4. target visible -> `approach_target`
    # 5. otherwise -> `look_for_target`
    #
    # Check:
    # - Call this only for oracle labels and oracle evaluation.
    # - Never put `evaluation_info` into the learned VLM prompt.
    _ = (evaluation_info, config)
    raise NotImplementedError("Choose one evaluation-only oracle subgoal.")


def should_replan(
    recent_rewards: list[float],
    recent_costs: list[float],
    steps_on_subgoal: int,
    subgoal: Subgoal,
    config: HierarchicalAgentConfig,
) -> tuple[bool, str]:
    """Decide whether the slow planner should run again."""
    # Goal:
    # Replan for clear, measurable reasons.
    #
    # Return `True` when:
    # - `steps_on_subgoal >= subgoal["replan_after_steps"]`
    # - the last `stall_window` rewards show no progress
    # - recent cost crosses `danger_replan_threshold`
    #
    # Return a reason such as:
    # - `"scheduled"`
    # - `"stalled"`
    # - `"danger"`
    # - `"continue"`
    _ = (recent_rewards, recent_costs, steps_on_subgoal, subgoal, config)
    raise NotImplementedError("Decide whether to request a new subgoal.")


def run_hierarchical_episode(
    environment: HierarchicalEnvironment,
    low_level_policy: LowLevelPolicy,
    planner: FrozenVLMSubgoalPlanner,
    overall_goal: str,
    config: HierarchicalAgentConfig,
    *,
    seed: int,
) -> list[HierarchicalStep]:
    """Run one real closed-loop episode with slow and fast model calls."""
    # Goal:
    # Implement the full loop listed at the top of this file.
    #
    # Steps:
    # 1. Reset the environment.
    # 2. Start an empty recent-step list and no previous subgoal.
    # 3. Call `build_memory_summary(...)`, build the prompt, and request the
    #    first subgoal.
    # 4. Convert it to the low-level policy instruction.
    # 5. Time `low_level_policy.predict_action(...)`.
    # 6. Step the environment.
    # 7. Store a `HierarchicalStep`.
    # 8. Update recent reward, cost, memory, and step counters.
    # 9. Use `should_replan(...)`.
    # 10. Stop at success, done, max steps, or max replans.
    #
    # Checks:
    # - Hidden environment state may create oracle labels for evaluation, but
    #   it must not enter the learned planner prompt.
    # - Use a safe fixed subgoal after invalid JSON and record that failure.
    _ = (environment, low_level_policy, planner, overall_goal, config, seed)
    raise NotImplementedError("Run one hierarchical embodied episode.")


def evaluate_oracle_and_vlm_subgoals(
    environment_factory: HierarchicalEnvironmentFactory,
    low_level_policy: LowLevelPolicy,
    planner: FrozenVLMSubgoalPlanner,
    config: HierarchicalAgentConfig,
) -> tuple[HierarchicalMetrics, list[list[HierarchicalStep]]]:
    """Compare oracle and learned subgoals with the same action policy."""
    # Goal:
    # Separate high-level planning quality from low-level skill quality.
    #
    # Method:
    # 1. Prepare a fixed list of seen and unseen episode seeds.
    # 2. Run the low-level policy with environment-provided oracle subgoals.
    # 3. Run the same policy with VLM subgoals on the same seeds.
    # 4. Compare subgoals at the same decision points.
    # 5. Aggregate all metrics listed in `HierarchicalMetrics`.
    #
    # Checks:
    # - Use the same controller checkpoint and episode seeds.
    # - Count invalid VLM responses as failures, even if a fallback recovers.
    _ = (environment_factory, low_level_policy, planner, config)
    raise NotImplementedError("Evaluate oracle and VLM subgoal planners.")


def save_hierarchical_agent_report(
    metrics: HierarchicalMetrics,
    episodes: list[list[HierarchicalStep]],
    config: HierarchicalAgentConfig,
) -> dict[str, str]:
    """Save subgoal timelines, metric bars, latency, and failures."""
    # Goal:
    # Make it possible to see why the agent succeeded or failed.
    #
    # Save:
    # - `oracle_vs_vlm_success.png`
    # - `planner_controller_latency.png`
    # - `episode_timeline.png`
    # - `failure_gallery.png`
    #
    # Each failure should show:
    # - frame
    # - overall goal
    # - raw VLM JSON
    # - validated or fallback subgoal
    # - actions taken
    # - final outcome
    _ = (metrics, episodes, config)
    raise NotImplementedError("Save the hierarchical-agent report.")


def main() -> None:
    """Run the hierarchical VLM-agent scaffold."""
    config = HierarchicalAgentConfig()
    try:
        planner = FrozenVLMSubgoalPlanner.from_pretrained(config)
        print(f"Loaded planner: {planner.config.vlm_model_id}")
    except NotImplementedError as error:
        print(f"[TODO] hierarchical_vlm_subgoal_agent: {error}")


if __name__ == "__main__":
    main()
