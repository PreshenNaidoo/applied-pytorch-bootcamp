"""
EXERCISE 10: DINO-WM GOAL-IMAGE PLANNING IN GYMNASIUM PUSHT

WHAT YOU ARE BUILDING
Use exercise 09's frozen dynamics model to choose continuous PushT controls.
Reference: https://arxiv.org/abs/2411.04983
Planner reference: https://github.com/gaoyuezhou/dino_wm/tree/main/planning
Environment: https://github.com/huggingface/gym-pusht (project pins 0.1.6).

FULL SYSTEM
recorded data -> train world model (09) -> search actions -> execute -> observe
This file covers the last three steps. No model weights change during planning.
The planner imagines candidate action sequences; Gymnasium supplies the actual
outcomes only after the chosen actions are executed.

HOW THIS BUILDS ON 08 AND 09
08 trained an actor to choose actions. Here there is no actor network: the
controller searches possible action sequences whenever it needs a new plan.
09 learned what an action chunk does to image/state features. Now freeze that
model and ask it: 'Which of these possible futures ends nearest the goal image?'
Changing the proposed actions changes the prediction, without changing weights.

MAIN OBJECTS
Object                   What it represents                    Does it change?
-----------------------------------------------------------------------------
env                      real PushT simulation                 after env.step
encoder                  pretrained DINO image encoder         frozen
model                    dynamics learned in exercise 09       frozen
saved['stats']           training coordinate means/stds         fixed
goal                     patch features of a target image      fixed
History                  recent real observations/controls     after a full chunk
candidate                one proposed sequence of controls     each search round
mean, std                where/how widely CEM samples actions   each search round
Plan                     best scored candidate and prediction  each planning call

The simulator knows the block's true position and rotation. The controller
gets a picture and the pusher's own x/y position, not those hidden block values.
The environment supplies coverage/success for reporting only. The planner's
feedback during search is the world model's predicted goal-feature distance.

WHAT AN ACTION MEANS HERE
An action [250,300] requests an absolute pusher target in the simulator's
0-to-512 coordinate system. It is not 'move 250 right and 300 up'. The
simulator applies its controller/physics; the pusher need not reach that target
in one step. The 96 x 96 camera image does not change the action coordinates.
Keep raw actions for env.step and Plan.actions. Normalize them with 09's saved
action statistics only when giving them to the learned world model.

RESEARCH METHOD AND LOCAL CHANGES
Keep goal-image feature costs, CEM search and receding-horizon control. CEM means
Cross-Entropy Method: sample action sequences, keep the best, then sample near
them. The reference supports CEM and gradient planning; we implement CEM.
Our checkpoint uses the public dataset's absolute x/y controls, not the paper's
relative controls. See 09 for the exact model/data differences. Do not load an
upstream checkpoint through this adapter; compatible weights are required.

CEM IS SEARCH, NOT ANOTHER NETWORK TRAINING LOOP
For every future control coordinate, CEM keeps a mean and standard deviation.
The mean is the current centre of the search. The standard deviation controls
how far candidate actions spread around it. These are tensors of action values,
not optimizer state or parameters learned through backpropagation.

One search round:
  sample 64 complete plans around mean/std
  -> imagine all plans from the SAME current history
  -> score their final visual features against the goal
  -> keep the 8 lowest-cost plans (the 'elites')
  -> fit a new mean/std from those plans

Repeat four rounds by default. The best candidate found is returned for actual
execution. No candidate is tested by secretly stepping and resetting env.
Despite CEM's name, this exercise does not use nn.CrossEntropyLoss, labels or
loss.backward(). The update is elite selection followed by averages/spreads.

The search distribution has separate means/stds for each [chunk,control,x/y]
entry; it does not fit a full covariance between control times. Each complete
sequence is still ranked as a unit, because its controls jointly cause a future.
Keep a minimum standard deviation so the search does not collapse to one point.
The mean itself may never have been scored; return the best evaluated candidate,
not an assumed-good average of different plans.

ACTION TIMING MATTERS
One prediction covers S=5 environment actions. With horizon K=5, a candidate
contains [5,5,2] values and describes 25 actual environment steps.
MPC executes one CHUNK of five actions, gets a new frame, then replans.
Open-loop CEM executes all five chunks before replanning. Calling env.step once
per chunk would silently change the learned dynamics and is wrong.
At reset we repeat the first observation H=3 times and use position-hold actions
for the missing history. This startup convention is a documented approximation.

MPC means Model Predictive Control. Receding horizon means that after executing
part of a plan, we start a new K-chunk prediction from the latest real history.
The horizon stays K; we do not simply count down an old plan to zero.

Concrete K=3,S=5 timeline from environment step 0:
  both modes search controls for steps 0..14
  MPC:       execute 0..4, observe, search again for steps 5..19
  open_loop: execute 0..14, observe, search again for steps 15..29

Here 'open_loop' means no replanning WITHIN those K chunks; it can replan after
the whole block. Both modes still record real feedback and stop immediately on
episode end. Recording feedback is not the same as changing the queued controls.
An action chunk is S successive controls that may differ, not one control repeated S times.

WHAT A WARM START DOES
After MPC executes chunk A from [A,B,C], the next search begins near [B,C,C].
Keep the unexecuted part and repeat the last chunk to restore length K. This
reuses useful work, but the new search scores candidates from the new REAL
history. It does not blindly trust the old predicted state. After executing
all three chunks in open_loop, the same convention starts from [C,C,C].

WHAT THE PLANNER KNOWS
- RGB observations, pusher position, past controls and a supplied goal image.
- Frozen DINOv2, the trained dynamics model and training normalization values.
It does not read block pose, contacts or reward while scoring candidate actions.
The provided goal-scene setup uses simulator state only to create a real target
image, like giving the controller a photograph of the desired result.
The target image also contains the pusher at the provided reference position.
A whole-image feature cost can care about that appearance as well as the block.
It is a proxy for the task, not an exact expression of simulator coverage.

LOOP IN PLAIN WORDS
env.reset(seed=seed): start a test episode.
encode the observed history; propose several futures; keep the best sequence.
env.step(action): execute ONE absolute x/y control and get the actual result.
After S actions, update history from the real observation. MPC plans again;
open_loop continues the selected plan until its K chunks are complete.
Stop on terminated OR truncated. Success comes from info['is_success'], not
from treating a timeout as a solved task. env.close() releases the renderer.

WHAT GYMNASIUM RETURNS
  observation, info = env.reset(seed=seed)
  observation, reward, terminated, truncated, info = env.step(action)
observation is a dictionary: 'pixels' is uint8 RGB [96,96,3]; 'agent_pos' is
the pusher's two coordinates. encode_observation turns these into model tokens.
terminated means the task has ended; in this pinned PushT version it reports
success. truncated means an outside limit, such as maximum episode steps, ended
the attempt. Use info['is_success'] as the success flag, never bool(terminated
or truncated). Once either flag is true, do not step again before reset.

TWO DIFFERENT USES OF PREDICTIONS
During search, imagined states feed later imagined states, as in 09's imagine.
After execution, real images replace the newest entry in the OBSERVED history.
For H=3, History stores [A,B,C] and controls [A->B,B->C]. After the complete
chunk C->D, encode the real D and store [B,C,D] with controls [B->C,C->D].
This corrects prediction errors before the next search. Do not append the
imagined D where the controller has an actual camera observation available.

SHAPES AND COST
B=candidates, K=chunks in plan, S=controls per chunk, P=patches, E=visual width.
raw candidates [B,K,S,2] -> normalized controls -> imagined [B,K,P,394]
cost = mean((imagined_final_visual - goal_visual)**2), over P and E.
Keep the feature scale used in training. State embeddings are not goal targets.
For costs [0.8,0.2,0.4,0.9], two elites are candidates 1 and 2. Fit each control's
new mean and standard deviation from those two; clamp std to prevent no search.
Use the best sequence seen across all updates, not merely the final mean.

FOLLOW ONE DEFAULT SEARCH BATCH
  sampled raw plans           [64,5,5,2]      64 candidates, 25 controls each
  one memory-sized group      [4,5,5,2]       rollout_batch=4
  repeated starting history   [4,3,256,394]   same history, four candidate futures
  imagined observations       [4,5,256,394]   one prediction per chunk
  final visual features       [4,256,384]     last chunk, visual channels only
  goal visual features        [256,384]       same target for all candidates
  per-candidate costs         [4]             one number for ranking each plan
  all costs after joining     [64]            original candidate order

The batch axis here means proposed futures for ONE real episode, not different
environments. rollout_batch changes memory use, not how many plans are searched.
Dropping state-embedding channels from the goal cost does not remove their use
inside the model: state conditioning still helps it predict visual motion.

GOAL COST AND PREDICTION GAP ANSWER DIFFERENT QUESTIONS
Goal cost: predicted final features versus desired goal features. Lower means
the model expects the plan to reach the desired appearance.
Prediction gap: predicted features versus the actual features AFTER execution.
Lower means the model predicted that outcome more accurately. Compare the same
time point: MPC executes one chunk, so its gap needs a one-chunk prediction,
not Plan.final_prediction from K chunks ahead. Make that prediction from the
history BEFORE acting; using the updated history would compare different times.

LEARNING ORDER
1. goal_cost: connect imagined visual features to the desired observation.
2. update_distribution: select elites and fit the next search distribution.
3. cem_plan: search with batched, memory-bounded model rollouts.
4. advance_history: replace imagination with the actual observed transition.
5. run_episode: execute the plan with the correct action timing and stop rules.
The runner first gives you a real goal image, then reaches goal_cost before
trying a full episode. Environment setup, encoding, random baseline, reporting
and saved videos are provided.

EVALUATION / DEFINITION OF DONE
Compare random, open-loop CEM and MPC on the same seeds. Report actual coverage,
success rate, environment steps and planning seconds/action. Reward in the pinned
environment is clipped coverage divided by its success threshold; use
info['coverage'] for raw coverage. Higher coverage/success is better.
Save predicted/actual feature gaps, search curves and videos. Low search cost
with poor actual coverage can mean the planner exploits model errors. Check
09/11's held-out rollouts before increasing the search budget. A two-episode
smoke run is not a reliable success estimate; the default is ten episodes.

METRICS TO READ TOGETHER
- success_rate: fraction of episodes solved, 0..1, higher is better. 0.7 means
  7 of 10 attempts succeeded, not that the block covered 70% of the goal.
- max_coverage/final_coverage: mean per-episode best/final overlap fraction,
  0..1, higher is better. The pinned task succeeds above 0.95 coverage.
- steps: number of actual env.step calls, not chunks or internal physics steps.
- seconds_per_action: total CEM search wall time divided by executed controls;
  lower is faster. This excludes rendering, encoding and gap diagnostics, so it
  is not total end-to-end controller latency. CUDA timing must wait for the GPU.
- prediction_mse: mean completed-plan/chunk prediction gap, lower is better.
  MPC and open_loop gaps cover different horizons; compare them with that in mind.
  A gap is omitted when an episode ends before the relevant endpoint is reached.

The random baseline samples legal actions and measures outcomes. It does not
learn, call an optimizer or use a learned policy. Matched reset seeds give all
three modes the same starting situations, though their later paths differ.
Search cost falling is not proof of task success: model errors can look like
shortcuts to the planner. Inspect saved real videos and actual coverage too.

HOW TO START / DEFINITION OF DONE
Complete 09's stubs and train its checkpoint first; --checkpoint can name an
alternative compatible file. This exercise also calls 09's imagine stub.
Run from the repository root in the bootcamp environment. The runner loads
weights, creates/saves goal.png and stops at goal_cost before attempting episodes.
Use each stub's small checks before paying for a complete search/evaluation.
Finish by comparing all three modes on the same seeds. Inspect metrics.json,
success.png and every saved episode video. First-plan search costs are stored
in metrics.json; save_results does not draw a separate search-curve figure.
Keep each original NotImplementedError commented when you solve that stub.
Run: python -m exercises.world_models.10_dino_wm_planning --episodes 10
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import gymnasium as gym

training = importlib.import_module("exercises.world_models.09_dino_wm_training")
# File names begin with numbers, so importlib loads 09 by its module string.
# Its Config, checkpoint loader, encoding helpers and imagine are reused here.
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PlanningConfig:
    # Search size and execution timing are separate:
    # horizon = future CHUNKS per plan, candidates = complete plans per CEM round,
    # elites = winning candidates used to fit the next round, updates = rounds.
    # rollout_batch limits GPU memory when scoring, not the total candidate count.
    # initial_std/minimum_std are in raw 0..512 control-coordinate units.
    # max_steps limits actual env.step calls per episode, not search iterations.
    checkpoint: str = str(ROOT / "outputs/dino_wm_pusht/predictor.pt")
    horizon: int = 5
    candidates: int = 64
    elites: int = 8
    updates: int = 4
    rollout_batch: int = 4
    minimum_std: float = 5.0
    initial_std: float = 100.0
    episodes: int = 10
    max_steps: int = 300
    seed: int = 10000
    output: str = str(ROOT / "outputs/dino_wm_planning")

    def __post_init__(self) -> None:
        if not 1 <= self.elites <= self.candidates:
            raise ValueError("Elites must be between 1 and candidate count.")
        if min(self.horizon, self.updates, self.rollout_batch, self.episodes, self.max_steps,
               self.minimum_std, self.initial_std) <= 0:
            raise ValueError("Planning lengths and search scales must be positive.")


@dataclass
class History:
    # Observed context for ONE real environment. The first axis is 1 even when
    # CEM will imagine 64 futures from it. H observations have H-1 connecting
    # chunks; a proposed future chunk will complete the inputs to 09's predict.
    # These fields contain embeddings/normalized controls, not image pixels.
    observations: Tensor  # [1,H,P,394]
    actions: Tensor       # [1,H-1,S,2], normalized


@dataclass
class Plan:
    # Search output: raw actions can go to env; final_prediction is only a model
    # guess. costs records successive search rounds, not per-environment-step rewards.
    actions: Tensor       # [K,S,2], raw absolute controls
    costs: list[float]    # best-so-far cost per CEM update
    final_prediction: Tensor  # [P,384]


def make_environment(max_steps: int) -> gym.Env:
    # Provided: pixels_agent_pos exposes camera RGB plus the pusher's own x/y.
    # rgb_array makes rendering return arrays for saved videos. max_episode_steps
    # installs Gymnasium's time limit, which reports truncated when reached.
    import gymnasium as gym
    import gym_pusht  # Registers gym_pusht/PushT-v0.
    return gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=max_steps)


def make_goal_image() -> np.ndarray:
    # Provided task setup: use a SEPARATE scene to obtain a real success image.
    # env.unwrapped bypasses wrappers only here so we can position this target.
    # The planner receives the resulting RGB photograph, not world.goal_pose.
    # A hold-position step asks the real simulator to verify coverage/success.
    # This is task setup, not a controller oracle. No candidate rollout calls it.
    # Set angle before position to avoid legacy reset's centre-of-mass shift.
    env = make_environment(5)
    try:
        env.reset(seed=0)
        world = env.unwrapped
        world.block.angle = float(world.goal_pose[2])
        world.block.position = tuple(world.goal_pose[:2])
        world.block.velocity = (0, 0)
        world.block.angular_velocity = 0
        world.agent.position = (100, 100)
        observation, _, _, _, info = env.step(np.array([100, 100], dtype=np.float32))
        if not info["is_success"] or info["coverage"] < 0.95:
            raise RuntimeError("Goal-scene setup did not produce a successful real observation.")
        return observation["pixels"].copy()
    finally:
        env.close()


def normalize_actions(actions: Tensor, saved: dict) -> Tensor:
    # Provided: raw controls -> world-model inputs using 09's TRAINING statistics.
    # Each statistic has shape [2] and broadcasts over any leading axes, such
    # as [B,K,S,2] or [S,2]. Output has the input shape/device. Do not overwrite
    # raw Plan.actions: the simulator still needs raw absolute coordinates.
    # Example for one coordinate: raw=250, training_mean=200, std=100 -> 0.5.
    return (actions - saved["stats"]["action_mean"].to(actions.device)) / saved["stats"]["action_std"].to(actions.device)


@torch.no_grad()
def encode_observation(observation: dict, model, encoder, saved: dict, device: torch.device) -> Tensor:
    # Provided: one REAL observation -> [1,1,P,394]. Image.fromarray converts
    # uint8 RGB [96,96,3] to PIL; 09's prepare_image returns [3,224,224].
    # DINO produces [1,P,384]; unsqueeze(1) adds the one-frame time axis.
    # Normalize pusher coordinates with state statistics, not action statistics.
    # observation_tokens joins the learned 10-value state to the visual grid.
    # Do not call this on imagined features; it expects real camera pixels.
    image = training.prepare_image(Image.fromarray(observation["pixels"]), model.config)
    visual = training.encode_images(encoder, image.unsqueeze(0).to(device), model.config).unsqueeze(1)
    state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32, device=device)
    state = (state - saved["stats"]["state_mean"].to(device)) / saved["stats"]["state_std"].to(device)
    return model.observation_tokens(visual, state.reshape(1, 1, 2))


def initial_history(observation: dict, model, encoder, saved: dict, device: torch.device) -> History:
    # Provided startup convention: reset gives one frame but 09 expects H.
    # Repeat it H times and fill H-1 missing chunks with position-hold controls.
    # These are assumed past observations/controls, not steps actually executed.
    # Real completed chunks replace this padding as the episode proceeds.
    # A hold control is the CURRENT absolute x/y, not raw [0,0], which requests
    # the corner of the table. H=1 naturally gives [1,0,S,2] past actions.
    H, S = model.config.history, model.config.action_chunk
    tokens = encode_observation(observation, model, encoder, saved, device).repeat(1, H, 1, 1)
    hold = torch.as_tensor(observation["agent_pos"], dtype=torch.float32, device=device)
    actions = normalize_actions(hold, saved).reshape(1, 1, 1, 2).repeat(1, H - 1, S, 1)
    return History(tokens, actions)


def goal_cost(predicted_visual: Tensor, goal_visual: Tensor) -> Tensor:
    # Where this fits:
    # score_candidates extracts the last imagined visual grid for each plan.
    # This helper tells CEM how far each predicted outcome is from the goal.
    # It never inspects the environment, reward or hidden block coordinates.
    #
    # Goal:
    # Return one distance per candidate, preserving the candidate axis so the
    # search can rank plans. Lower cost means a closer predicted appearance.
    #
    # Input:
    # predicted_visual [B,P,E], default [64,256,384]; goal_visual [P,E].
    # Both are on the same device and in the same DINO feature space. The 10
    # learned state channels have already been removed by score_candidates.
    #
    # Output:
    # costs [B], floating-point Tensor. Even B=1 must return [1], not a scalar.
    # It is a feature distance, not a probability, simulator reward or pixel error.
    #
    # Formula:
    # cost[b] = sum over patches/features of (prediction[b,p,e]-goal[p,e])**2
    #           divided by P*E. Here b identifies one candidate, p one patch,
    #           and e one feature entry. Do not average over b.
    # Example with one patch and two features: goal=[1,2], candidates
    # [1,2] and [3,2] have costs 0 and ((3-1)**2+(2-2)**2)/2 = 2.
    #
    # Method:
    # Tensor.unsqueeze(0) changes goal [P,E] to [1,P,E]. Subtraction broadcasts
    # it across B without making separate goals. Tensor.square() squares each
    # difference; Tensor.mean(dim=(1,2)) averages only patch/feature axes.
    # A plain .mean() or F.mse_loss with its default reduction would combine
    # all candidates into one number and destroy the ranking.
    #
    # Steps:
    # 1. Check predicted_visual.ndim == 3, goal_visual.ndim == 2 and
    #    predicted_visual.shape[1:] == goal_visual.shape; otherwise raise ValueError.
    #    Do not allow accidental broadcasting along a mismatched patch axis.
    # 2. Add the singleton goal batch axis with goal_visual.unsqueeze(0).
    # 3. Subtract it from predicted_visual, then square the difference.
    # 4. Average over dim=(1,2) and return the resulting [B] tensor.
    #    Preserve the trained feature scale; do not add unit normalization.
    #
    # Checks:
    # - The two-candidate example returns tensor([0.,2.]).
    # - Identical grids give zero, costs are nonnegative, and shape is [B].
    # - Permuting candidate order permutes costs in exactly the same way.
    raise NotImplementedError("Score imagined final features against the goal image.")


def update_distribution(actions: Tensor, costs: Tensor, config: PlanningConfig) -> tuple[Tensor, Tensor]:
    # Where this fits:
    # One CEM round has scored every candidate. This helper moves the next
    # round's sampling distribution toward the best complete sequences.
    #
    # Goal:
    # Select config.elites lowest-cost candidates, then compute a separate
    # mean and standard deviation for each future x/y control coordinate.
    # The model is fixed; these are action-search statistics, not model updates.
    #
    # Input:
    # actions [B,K,S,2]: RAW absolute controls used for those predictions.
    # costs [B]: finite goal costs in the same candidate order.
    # config.elites is between 1 and B; config.minimum_std uses raw action units.
    #
    # Output:
    # (mean,std), each [K,S,2], on actions.device and in raw action units.
    # The B axis disappears, but every time/control/coordinate position stays.
    #
    # Method:
    # torch.topk(costs, k=config.elites, largest=False) returns values AND indices.
    # Use its .indices to gather elite_actions = actions[indices]. Setting
    # largest=False matters: low distance is good. Index the whole sequences,
    # not individual controls selected from unrelated plans.
    # elite_actions.mean(dim=0) and elite_actions.std(dim=0, unbiased=False)
    # fit statistics over the elite-candidate axis only. unbiased=False divides
    # variance by the elite count, so one elite gives zero std instead of NaN.
    # Tensor.clamp_min(config.minimum_std) then preserves some exploration.
    #
    # Steps:
    # 1. Check actions has four axes ending in 2, costs.shape == (len(actions),),
    #    and 1 <= config.elites <= len(actions); raise ValueError on mismatch.
    # 2. Require torch.isfinite(costs).all() and torch.isfinite(actions).all().
    #    Raise ValueError for NaN/inf, rather than fitting a broken distribution.
    # 3. Select lowest-cost indices with topk, then gather [elites,K,S,2].
    # 4. Average elites along dim=0 -> mean [K,S,2].
    # 5. Compute population std along dim=0 and clamp to the minimum std.
    # 6. Return (mean,std). Do not normalize controls in this helper.
    #
    # Concrete check for ONE coordinate across four complete candidates:
    # costs=[0.8,0.2,0.4,0.9], x-values=[100,200,220,400], elites=2.
    # Candidates 1 and 2 survive: x mean=210, population std=10.
    # This same calculation runs independently at every [K,S,x/y] position.
    # If both elite x-values are 200, std becomes minimum_std, not zero.
    # Check: one elite is valid, outputs are finite, every std >= minimum_std.
    raise NotImplementedError("Refit the CEM action distribution from elites.")


@torch.no_grad()
def score_candidates(actions: Tensor, history: History, model, saved: dict, goal: Tensor,
                     config: PlanningConfig) -> tuple[Tensor, Tensor]:
    # Provided: raw candidate controls [B,K,S,2] -> (costs [B], finals [B,P,384]).
    # The actual K comes from actions, so this also supports MPC's K=1 diagnostic
    # even when config.horizon is larger. Each group starts from the same history.
    # expand(n,-1,-1,-1) shares that history across n imagined futures; -1 keeps
    # each other axis unchanged. 09's imagine must not modify expanded inputs.
    # Normalization happens HERE, once, before model rollouts. The final time
    # slice removes K; slicing :visual_dim removes the learned state channels.
    # Concatenating group results keeps the original candidate order, including
    # a final group smaller than rollout_batch. No real environment is involved.
    costs, finals = [], []
    for start in range(0, len(actions), config.rollout_batch):
        part = normalize_actions(actions[start:start + config.rollout_batch], saved)
        n = len(part)
        imagined = training.imagine(model, history.observations.expand(n, -1, -1, -1),
                                    history.actions.expand(n, -1, -1, -1), part)
        final = imagined[:, -1, :, :model.config.visual_dim]
        costs.append(goal_cost(final, goal))
        finals.append(final)
    return torch.cat(costs), torch.cat(finals)


@torch.no_grad()
def cem_plan(history: History, model, saved: dict, goal: Tensor, position: Tensor,
             config: PlanningConfig, generator: torch.Generator, warm_start: Tensor | None = None) -> Plan:
    # Where this fits:
    # run_episode calls this each time it needs a plan. All candidates start
    # from the same observed history. The real environment does not advance
    # during any of these search rounds, and model weights never change.
    #
    # Goal:
    # Search raw action sequences and return the best actually evaluated one,
    # its final visual prediction and its best-so-far cost after every round.
    #
    # Input:
    # - history: observations [1,H,P,394], past normalized actions [1,H-1,S,2].
    # - model/saved: frozen 09 dynamics and checkpoint normalization metadata.
    # - goal [P,384]: fixed goal-image features on the model device.
    # - position [2]: current RAW pusher x/y, float32 on goal.device.
    # - generator: torch.Generator on that device, seeded once by run_episode.
    # - warm_start: None or RAW [K,S,2] controls, already shifted by the caller.
    # K=config.horizon; S=model.config.action_chunk. Do not hard-code S=5.
    #
    # Output:
    # Plan(actions=[K,S,2], costs=list of config.updates Python floats,
    #      final_prediction=[P,384]). Tensors stay on the model device.
    # Plan.final_prediction must belong to Plan.actions, not a different elite.
    #
    # Method:
    # Draw independent standard-normal noise with torch.randn and transform:
    # candidate = mean + std * noise. With mean=250,std=100,noise=-0.5, that
    # coordinate is 200. Clamp sampled coordinates to the legal interval [0,512].
    # The resulting clipped samples need not retain an exact Gaussian shape.
    # score_candidates handles normalization, memory-sized groups, 09's imagine
    # and goal_cost. Use update_distribution to fit the next search round.
    # @torch.no_grad() already disables gradient recording for this helper.
    #
    # Steps:
    # 1. Call model.eval(). Set shape=(config.horizon,model.config.action_chunk,2).
    #    If no warm start exists, position.expand(shape).clone() makes each
    #    control initially request the current pusher location. This is only
    #    the sampling centre; sampled candidates will move elsewhere.
    # 2. Otherwise require warm_start.shape == shape and start with its clone.
    #    Keep mean float32 on goal.device. Do not mutate the caller's warm start.
    #    Build std with torch.full_like(mean, config.initial_std).
    # 3. Set best_cost=float('inf'), best_actions=None, best_final=None and
    #    best_costs=[]. These describe this planning call, not the whole episode.
    # 4. For each round in range(config.updates), draw
    #    torch.randn((config.candidates,*shape), device=goal.device,
    #                dtype=mean.dtype, generator=generator).
    #    Mean/std [K,S,2] broadcast across candidates. Multiply noise by std,
    #    add mean and call .clamp(0.0,512.0). Keep these raw controls for execution.
    # 5. If a best_actions already exists, copy it into actions[0]. Keeping this
    #    previously good candidate lets it participate in the next elite set.
    # 6. Call score_candidates(actions,history,model,saved,goal,config).
    #    It returns costs [B] and finals [B,P,384] in matching order. Reject
    #    non-finite costs before ranking so NaN cannot become a winning plan.
    # 7. Find index=costs.argmin().item() and candidate_cost=costs[index].item().
    #    If candidate_cost < best_cost, save it and clone actions[index] and
    #    finals[index]. Clones prevent later sampling from changing saved results.
    # 8. Append best_cost (a float) to best_costs. This tracks the best found
    #    across ALL rounds, even if this round's other candidates were worse.
    # 9. Call update_distribution(actions,costs,config) to replace mean/std.
    #    Do not reseed the generator inside the loop; advance its random stream.
    # 10. After all rounds, return Plan(best_actions,best_costs,best_final).
    #     config.updates >= 1 guarantees a winner if all costs are finite.
    #
    # Checks:
    # - len(plan.costs) == config.updates and costs never increase.
    # - Controls have shape [K,S,2], are finite, and lie in [0,512].
    # - Rescoring plan.actions.unsqueeze(0) from the SAME history gives the
    #   last best cost and matching final features, up to numerical tolerance.
    # - Use fresh generators with the same seed to check repeatability. Reusing
    #   an already-advanced generator intentionally draws different candidates.
    # - There is no env.step, hidden-state access, optimizer or backward call.
    #   Legal actions may still be unfamiliar to the model; low cost is a guess.
    raise NotImplementedError("Search action chunks with CEM and a warm start.")


def advance_history(history: History, real_tokens: Tensor, executed: Tensor, saved: dict) -> History:
    # Where this fits:
    # A complete S-control chunk has run in the REAL environment. The newest
    # real image/state is encoded. Update observed history before the next plan.
    # Open-loop execution also calls this after each full chunk, but keeps
    # executing its queued plan until that plan ends.
    #
    # Goal:
    # Drop the oldest observed frame and its outgoing controls, then append
    # the real next observation and the controls that actually led to it.
    #
    # Input:
    # history.observations [1,H,P,394]; history.actions [1,H-1,S,2] normalized.
    # real_tokens [1,1,P,394] from encode_observation, NOT model imagination.
    # executed [S,2]: all raw controls actually sent to env, on history's device.
    # saved contains the training action means/stds. Infer H/S from history;
    # even for H=1, the empty actions tensor retains its S dimension.
    #
    # Output:
    # A NEW History with exactly the same shapes as the input history. Leave
    # its original tensors unchanged, so earlier predictions retain their context.
    #
    # Method:
    # normalize_actions(executed,saved) -> [S,2] uses the training statistics.
    # Tensor.reshape(1,1,S,2) adds episode and chunk axes. torch.cat with dim=1
    # joins time/chunk axes; it does not create an extra batch dimension.
    #
    # Steps:
    # 1. Read H=history.observations.shape[1], S=history.actions.shape[2].
    #    Require executed.shape == (S,2) and real_tokens has exactly one time
    #    slot and matching batch/patch/feature sizes. Reject incomplete chunks.
    # 2. Form new_observations=torch.cat([history.observations[:,1:],
    #    real_tokens], dim=1). Removing one old and adding one new keeps H.
    # 3. Normalize executed once and reshape to [1,1,S,2]. These controls already
    #    happened; do not use unexecuted controls from the rest of the plan.
    # 4. Append this normalized chunk to history.actions, then slice [:,1:]
    #    to remove the oldest chunk. Appending BEFORE slicing also handles H=1:
    #    zero old chunks -> one appended chunk -> zero retained past chunks.
    # 5. Return History(new_observations,new_actions).
    #
    # Concrete check:
    # [A,B,C] with [aAB,aBC], after raw aCD and real D, becomes
    # [B,C,D] with [aBC,normalize(aCD)]. It does not become [B,C,imagined_D].
    # For H=1, retain only real D and an empty [1,0,S,2] action history.
    # If an episode ends after 2 of S=5 controls, the caller must stop without
    # inserting that partial chunk. Its duration would not match 09's model step.
    raise NotImplementedError("Update history from the actual completed action chunk.")


def run_episode(env: gym.Env, seed: int, mode: str, model, encoder, saved: dict,
                goal: Tensor, config: PlanningConfig) -> dict:
    # Where this fits:
    # main calls this for each matched-seed MPC/open_loop attempt. All earlier
    # stubs meet here: choose a plan, act in the real environment, observe the
    # result, update history and repeat. This measures behavior; it never learns.
    #
    # Goal:
    # Execute the requested mode with correct action timing, ending rules and
    # diagnostics. A CEM update, model chunk and env.step are DIFFERENT counters.
    #
    # Input:
    # env: one Gymnasium PushT environment, reused but reset for each episode.
    # seed: selects the initial situation and seeds a separate CEM generator.
    # mode: exactly 'mpc' or 'open_loop'; raise ValueError otherwise.
    # model/encoder/saved: frozen dynamics, frozen DINO and training metadata.
    # goal: [P,384] goal tokens; goal.device is the model device.
    # config: search settings and maximum number of real controls to execute.
    # The caller owns env and closes it in finally; do not close it in this helper.
    #
    # Output:
    # Return exactly these keys, consumed by save_results:
    #   'success': bool, whether info['is_success'] became true
    #   'coverage': list[float], one actual overlap fraction per env.step
    #   'steps': int, actual env.step count
    #   'planning_seconds': float, sum of CEM search wall times
    #   'frames': list[np.ndarray], reset image then one real RGB image per step
    #   'search_costs': list[float], best-so-far costs from the FIRST planning call
    #   'prediction_gaps': list[float], same-endpoint predicted/actual visual MSE
    # Empty prediction_gaps is valid if the episode ends before a measured endpoint.
    # Do not put CUDA tensors in this dictionary: metrics.json needs Python values.
    #
    # Method / provided helpers:
    # observation,info = env.reset(seed=seed) starts one episode.
    # initial_history(observation,model,encoder,saved,goal.device) makes History.
    # cem_plan(history,model,saved,goal,position,config,generator,warm_start)
    # returns a Plan. Its controls are RAW [K,S,2], ready for the simulator.
    # observation,reward,terminated,truncated,info = env.step(raw_action)
    # returns the next real observation and separate ending flags, not one done.
    # encode_observation(observation,model,encoder,saved,goal.device) makes
    # [1,1,P,394] tokens. advance_history adds them after a full executed chunk.
    #
    # Loop structure before writing code:
    #   while episode can continue:
    #       plan from current real history
    #       for each chunk this mode will execute:
    #           for each raw control in that chunk:
    #               step env, record feedback, check ending
    #           if chunk is complete: encode real endpoint, update history
    #           if episode ended: leave the chunk loop
    #       if measurement endpoint reached: record prediction gap
    #       if still running: shift warm start, repeat planning loop
    # A break inside the innermost loop does not automatically exit the others.
    # Carry one done flag through all levels and check it before further steps.
    #
    # Steps A: initialize one attempt.
    # 1. Validate mode; call model.eval() and encoder.eval(). Unpack reset's
    #    (observation,info), then call initial_history with that observation.
    # 2. Make generator=torch.Generator(device=goal.device).manual_seed(seed).
    #    Seed once here, not once per planning call, so successive calls explore
    #    new samples reproducibly. The environment's RNG is separate from this.
    # 3. Set warm_start=None, done=False, success=False, steps=0,
    #    planning_seconds=0.0; make empty coverage/search_costs/prediction_gaps
    #    lists and frames=[observation['pixels'].copy()]. Set S=model.config.action_chunk.
    #
    # Steps B: plan and preserve the prediction made BEFORE acting.
    # 4. Start an outer loop while not done and steps < config.max_steps.
    #    Save planning_history=history. advance_history returns NEW tensors, so
    #    this name keeps the old context. Convert observation['agent_pos'] with
    #    torch.as_tensor(observation['agent_pos'],dtype=torch.float32,device=goal.device).
    # 5. If goal.device.type == 'cuda', call torch.cuda.synchronize(goal.device)
    #    BEFORE starting time.perf_counter(). Call cem_plan with these inputs;
    #    synchronize again after it, then add the elapsed perf_counter time.
    #    GPU work is asynchronous; without waiting, timing can measure only
    #    queued work. Do not include later encoding/gap checks in this timer.
    # 6. If search_costs is still empty, copy plan.costs with list(plan.costs).
    #    Set chunks_to_execute = 1 for MPC, or config.horizon for open_loop.
    # 7. Choose expected_visual at this execution block's endpoint:
    #    - open_loop: plan.final_prediction already predicts after K chunks.
    #    - MPC: call score_candidates(plan.actions[:1].unsqueeze(0),
    #      planning_history,model,saved,goal,config). Input shape is [1,1,S,2]:
    #      ONE candidate, ONE chunk. Unpack (costs,finals); use finals[0].
    #    Do this BEFORE updating history. Giving the helper an already-updated
    #    history would predict another chunk ahead of the observation we compare.
    #    Both expected_visual tensors have shape [P,384].
    #
    # Steps C: execute selected controls, checking every real step.
    # 8. Start completed_chunks=0. Iterate the first chunks_to_execute entries
    #    of plan.actions. For each chunk, start executed_controls=[].
    # 9. Iterate its S controls in order. A control is Tensor [2]. Convert with
    #    raw_action.detach().cpu().numpy().astype(np.float32), then call env.step.
    #    Do NOT normalize before env.step, and do NOT pass a whole [S,2] chunk.
    # 10. After each call, append the raw control to executed_controls, increment
    #     steps, copy observation['pixels'] into frames and append
    #     float(info['coverage']) to coverage. Accumulate success with
    #     success = success or bool(info['is_success']). Reward is not the cost.
    # 11. Set done = bool(terminated or truncated or steps >= config.max_steps).
    #     If done, break the control loop immediately. Never execute a sixth
    #     control after a termination on the fifth, or keep stepping on timeout.
    # 12. If len(executed_controls) == S, encode this final real observation.
    #     Stack controls with torch.stack(executed_controls,dim=0) -> [S,2].
    #     Call advance_history(history,real_tokens,executed_chunk,saved) and
    #     increment completed_chunks. Do this even if the last control ended
    #     the episode: the complete endpoint can still supply a diagnostic.
    #     If fewer than S controls ran, do not pad them or update history.
    # 13. If done, break the chunk loop. Otherwise open_loop executes its next
    #     queued chunk; MPC has only one selected chunk and leaves this loop.
    #
    # Steps D: record a fair gap and prepare another search.
    # 14. Only if completed_chunks == chunks_to_execute, compare expected_visual
    #     with real_tokens[0,0,:,:model.config.visual_dim]. Both are [P,384].
    #     Compute (expected_visual-actual_visual).square().mean().item() and
    #     append it to prediction_gaps. These must be the SAME time point.
    #     An early end partway through an open-loop plan has no K-chunk endpoint;
    #     skip its gap, even if one or more earlier chunks completed.
    # 15. If done, leave the outer loop. Otherwise the selected block completed:
    #     let n=completed_chunks, retain plan.actions[n:], and make tail with
    #     plan.actions[-1:].repeat(n,1,1). Concatenate both with torch.cat(
    #     [remaining,tail],dim=0) to get warm_start [K,S,2] for the next call.
    #     This shifts chunks, not individual controls; do not normalize it.
    # 16. Return the dictionary with the seven output keys above. Do not infer
    #     success from low goal cost, a done flag, or reaching max_steps.
    #
    # Checks / concrete timing:
    # - With K=3,S=5 and no ending: MPC takes 5 steps per plan; open_loop takes 15.
    # - If the environment ends on control 2 of the first chunk, return steps=2,
    #   len(frames)=3, len(coverage)=2, no gap, and no second planning call.
    # - If MPC ends exactly on control 5, its one-chunk gap is valid. An open-loop
    #   K=3 prediction still has no matching 15-step outcome, so its gap is absent.
    # - Always len(frames)==steps+1 and len(coverage)==steps.
    # - Same seeds reset each mode to the same situations. All saved frames are
    #   real observations. No optimizer, backward call or hidden state is used.
    raise NotImplementedError("Run matched real-environment planning evaluation.")


def random_episode(env: gym.Env, seed: int) -> dict:
    # Provided baseline: sample a raw legal [2] action at every environment step.
    # Reset and action_space have separate seeds. This function only measures;
    # it never calls the world model, a learned policy, or an optimizer.
    # Its output keys match run_episode so reporting treats every mode alike.
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    frames, coverage = [observation["pixels"]], []
    while True:
        observation, _, terminated, truncated, info = env.step(env.action_space.sample())
        frames.append(observation["pixels"])
        coverage.append(float(info["coverage"]))
        if terminated or truncated:
            break
    return {"success": bool(info["is_success"]), "coverage": coverage, "steps": len(coverage),
            "planning_seconds": 0.0, "frames": frames, "search_costs": [], "prediction_gaps": []}


def save_results(runs: dict[str, list[dict]], config: PlanningConfig) -> dict:
    # Provided: runs maps each mode to episode dictionaries. Aggregate actual
    # task scores, save ALL real episode videos, and write raw metrics to JSON.
    # Max coverage is first computed within an episode, then averaged across
    # episodes; one fortunate frame across all runs is not the reported score.
    # No prediction gaps (random/early endings) -> None, not a misleading zero.
    # Video pixels use RGB; OpenCV's writer expects BGR, hence cvtColor below.
    # The chosen 10-fps playback rate is for inspection, not a runtime benchmark.
    import cv2
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for mode, episodes in runs.items():
        gaps = [g for r in episodes for g in r["prediction_gaps"]]
        summary[mode] = {"success_rate": float(np.mean([r["success"] for r in episodes])),
                         "max_coverage": float(np.mean([max(r["coverage"]) for r in episodes])),
                         "final_coverage": float(np.mean([r["coverage"][-1] for r in episodes])),
                         "seconds_per_action": sum(r["planning_seconds"] for r in episodes) / sum(r["steps"] for r in episodes),
                         "prediction_mse": float(np.mean(gaps)) if gaps else None}
        # Save each episode, so a video cannot be selected just because it succeeded.
        for index, run in enumerate(episodes):
            frames = run["frames"]
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(str(output / f"{mode}_{index:02}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Could not open the MP4 video writer.")
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
    serializable = {mode: [{k: v for k, v in run.items() if k != "frames"} for run in episodes]
                    for mode, episodes in runs.items()}
    (output / "metrics.json").write_text(json.dumps({"summary": summary, "episodes": serializable}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots()
    ax.bar(list(summary), [v["success_rate"] for v in summary.values()])
    ax.set(ylabel="success fraction", ylim=(0, 1), title="Matched PushT evaluation")
    fig.tight_layout(); fig.savefig(output / "success.png", dpi=150); plt.close(fig)
    return summary


def main() -> None:
    # Provided setup and matched evaluation. The goal scene has its own env;
    # each controller mode also gets a separate test env. Every mode uses the
    # same list of reset seeds. Only run_episode executes the learned planner;
    # there is no online training or checkpoint update in this script.
    parser = argparse.ArgumentParser(description="Plan with the exercise 09 checkpoint.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--checkpoint", default=PlanningConfig.checkpoint)
    args = parser.parse_args()
    config = PlanningConfig(episodes=args.episodes, checkpoint=args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, saved = training.load_checkpoint(Path(config.checkpoint), device)
        encoder = training.load_encoder(model.config, device)
        goal_pixels = make_goal_image()
        output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(goal_pixels).save(output / "goal.png")
        goal = training.encode_images(encoder, training.prepare_image(Image.fromarray(goal_pixels), model.config).unsqueeze(0).to(device), model.config)[0]
        print("Goal self-distance:", goal_cost(goal.unsqueeze(0), goal))
        runs = {mode: [] for mode in ("random", "open_loop", "mpc")}
        for mode in runs:
            env = make_environment(config.max_steps)
            try:
                for index in range(config.episodes):
                    seed = config.seed + index
                    result = random_episode(env, seed) if mode == "random" else run_episode(env, seed, mode, model, encoder, saved, goal, config)
                    runs[mode].append(result)
            finally:
                env.close()
        print(save_results(runs, config))
    except NotImplementedError as error:
        print(f"[TODO] {error}")


if __name__ == "__main__":
    main()
