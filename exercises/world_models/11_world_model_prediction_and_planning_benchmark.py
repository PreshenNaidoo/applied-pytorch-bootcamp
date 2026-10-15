"""
EXERCISE 11: DOES BETTER PREDICTION PRODUCE BETTER PLANNING?

WHAT YOU ARE BUILDING
A controlled benchmark for the actual PushT checkpoints from 09 and planner
from 10. Start with one trained model against no-change and wrong-action
baselines; then compare checkpoints trained for different budgets or histories.
Pong and PushT are different tasks. This exercise does not rank their latent
errors against each other or assume either has a nonexistent safety head.

References:
https://arxiv.org/abs/2411.04983 (DINO-WM)
https://github.com/facebookresearch/jepa-wms (physical planning evaluation)
This is our benchmark using their evaluation principles, not a reproduction of
every experiment in either paper. Native paper checkpoints need their own
matching environment/data adapter; this runner accepts the 09 format only.

FULL WORKFLOW
Train model -> hold out complete episodes -> test prediction -> test control
-> compare changes on the same data and seeds. This exercise does not train.
The frozen encoder supplies observed targets, the world model supplies imagined
futures, and the live environment supplies task success. Those are different tests.

HOW THIS BUILDS ON 09 AND 10
09 trained a predictor. 10 used one predictor to search actions in the simulator.
Here the question is whether changing a model or its planning horizon helps.
An error curve and a task-success score answer different questions: predicting
recorded movements accurately does not guarantee accurate predictions for the
new controls a planner searches. This is an evaluation exercise, not another
training loop. No optimizer or parameter update belongs in these stubs.

MAIN OBJECTS
Object                    Role in the experiment
--------------------------------------------------------------------------
checkpoint                trained weights, settings, splits and normalization
source                    real recorded PushT images, states and actions
cache                     frozen DINO descriptions of those images
anchor                    last real observation before prediction begins
window                    history, future controls and matching real targets
model                     frozen dynamics used to predict possible futures
goal image                desired appearance for the separate live control test
environment               source of actual live coverage and success
bootstrap sample          resampled episode outcomes, not new environment runs

The model knows its initial history and the supplied controls. The offline
evaluator can see recorded future images to SCORE predictions, but the open-loop
model cannot read those images as inputs. The observed-history comparison is
explicitly allowed to read each new real observation before predicting again.

WHAT MAKES A COMPARISON FAIR?
Use the same DINO encoder, image preprocessing, action-chunk duration, real
prediction targets, and live reset seeds. Use each model's own saved coordinate
statistics and learned state encoder. Their learned state spaces can differ;
compare only the common frozen VISUAL feature space, not those state channels.

Hold out whole episodes, not random nearby frames from one episode. A recording
used to train ANY compared checkpoint cannot be an evaluation recording here.
common_protocol finds the intersection of validation IDs and removes all
training IDs. The runner rejects incompatible feature spaces instead of trying
to make their numbers comparable by rescaling them.

These are the validation episodes stored by 09. If you already used their scores
to select checkpoints or settings, this is a development comparison, not a fresh
final test. Repeatedly choosing winners on this same split can overfit it.

SAME ANCHOR, DIFFERENT HISTORY LENGTHS
Suppose the last real observation is episode frame 20 and S=5:
  H=3 model receives frames [10,15,20]
  H=2 model receives frames [15,20]
  BOTH predict frames [25,30,...] using the SAME future controls
Starting both models at frame 10 instead would give them different last observed
frames and different prediction targets. That would not isolate history length.
The runner chooses anchors that support the largest history and largest horizon
in the comparison, then works backward to each model's own starting offset.

THREE PREDICTION CONDITIONS
1. Open loop: after the initial history, use only predicted observation tokens.
2. Observed-history prediction: insert the true latest observation before each
   next prediction. This measures individual transitions, not imagination.
3. Persistence: copy the final observed visual grid to every future time.
Also repeat open-loop prediction with the same window's action chunks shuffled.
No true future is ever inserted into condition 1.

ONE TWO-STEP COMPARISON
Call the final real history frame C, and the next real frames D and E:
  open loop:        [A,B,C] -> guessed D; [B,C,guessed D] -> guessed E
  observed history:[A,B,C] -> guessed D; [B,C,REAL D]    -> guessed E
  persistence:     use C's visual features as the guesses for BOTH D and E
The target for the first output is real D; for the second it is real E in all
conditions. Observed-history is sometimes called teacher-forced evaluation:
it asks how well each transition is predicted when prior mistakes are corrected.
It must not be presented as a successful two-step imagined rollout.

For the shuffled-control check, permute complete future chunks while preserving
the order of controls inside each chunk. Keep targets and initial history fixed.
This deliberately mismatches controls and outcomes; it does not provide the real
counterfactual outcome of executing those different controls. A positive action
gap supports sensitivity to aligned controls, not proof of causal understanding.
With K=1 there is nothing to permute. A random permutation can also be unchanged,
or different chunks may contain almost identical controls. A zero gap in those
cases says nothing about whether the model uses actions.

EXAMPLE / SHAPES
With H=3,S=5,K=2, frames [0,5,10] initialize the model. Actions [10:15] predict
frame 15; actions [15:20] predict frame 20. Open loop uses predicted frame-15
features in the second call; observed-history prediction uses real frame 15.
history [1,H,P,394], past actions [1,H-1,S,2], future actions [1,K,S,2],
target visual features [1,K,P,384]. K counts chunks; K*S counts env steps.

WHY THE WINDOW NEEDS H+K OBSERVATIONS
The H starting observations include the anchor. Each of K transitions then
adds one target observation. Therefore we need H+K frames and H+K-1 connecting
chunks. For H=3,K=2,S=5:
  selected frames:  [0,5,10,15,20]       five observations
  selected chunks: [0:5,5:10,10:15,15:20] four transitions
  past chunks:     [0:5,5:10]           H-1 chunks before the anchor
  future chunks:   [10:15,15:20]        K chunks after the anchor
  target frames:   [15,20]              K future outcomes
Python slices exclude their right endpoint. Row 15's control belongs to the
15->20 chunk, not the 10->15 chunk. Dataset row IDs may differ from frame numbers.

The default runner evaluates one window at a time, so B=1. A window stores
all_observations for the observed-history condition, but its separate observations
field contains ONLY the initial H frames for open-loop imagination. Keep those
two dictionary fields separate when calling 09's imagine.

METRICS
MSE(k) = mean((prediction_k - target_k)**2), across patch and feature axes.
Cosine(k) averages the angle similarity of corresponding patches; higher is
better. MSE is lower-is-better and is comparable here only because all models
use the SAME DINO feature space and preprocessing. Normalizing unrelated feature
spaces does not make their error numbers a fair ranking.
Action gap = shuffled-action MSE minus correct-action MSE. Positive is useful;
a small gap can mean weak action use or nearly identical shuffled actions.
Feedback gap = open-loop MSE minus observed-history MSE. Large values reveal
errors that build up as predictions become new inputs.
Real success fraction: proportion of fresh episodes that solve PushT. Report a
bootstrap interval by resampling whole EPISODES, not correlated video frames.

READING THE NUMBERS
If open-loop MSE at a horizon is 0.06, observed-history MSE is 0.02, and
shuffled-action MSE is 0.09, feedback gap is 0.04 and action gap is 0.03.
These are squared-feature units, not metres or pixel displacement. The gaps
are differences between comparable curves; the runner saves the curves, not
separate gap fields. Either gap can be negative on a finite evaluation set.

Cosine similarity compares vector direction, not size. If a target patch is
[1,0] and a prediction is [2,0], cosine is 1 but MSE is 0.5. Neither metric alone
proves good control. Near-zero vectors also make cosine less informative.

AVERAGE WITHOUT LETTING LONG EPISODES DOMINATE
The runner samples up to eight windows per eligible held-out episode. It first
averages windows WITHIN each episode, then averages those episode means. Thus
an episode with eight windows and one with two windows get equal final weight.
Overlapping windows share frames; they are not eight independent experiments.

WHAT THE SUCCESS INTERVAL DOES
Ten live attempts produce ten booleans, for example seven successes and three
failures. Their measured mean is 0.7. Bootstrap resampling repeatedly draws ten
entries WITH replacement from those ten outcomes and recomputes the mean.
The middle 95% of 2,000 resampled means gives the reported percentile interval.
It describes sampling uncertainty, not a guaranteed range for future performance.
It does not run 2,000 new episodes or turn ten attempts into a large test set.
All failures yield [0,0], and all successes [1,1]: the resampling cannot invent
an outcome absent from the observed data. More independent episodes are needed.
Intervals for two models are not a paired significance test; matched seeds help
comparison, but overlapping/non-overlapping intervals alone do not settle it.

PREDICTION HORIZON VERSUS PLANNING HORIZON
With --horizons 1 3 5 10, offline prediction computes every horizon 1..10 in one
rollout and saves the complete curves. Live evaluation runs MPC with each of
the four requested look-ahead lengths. MPC still executes only ONE chunk before
replanning. A ten-chunk look-ahead is not ten chunks of open-loop execution.
The live test compares random control and MPC; 10 separately compares open_loop.
Candidate count and CEM rounds stay fixed across horizons, but longer rollouts
cost more computation and search more control values. Compare planning time too;
this is not an experiment with equal wall-clock budgets.

LEARNING ORDER
1. assemble_window: align history, action chunks and all future targets.
2. observed_history_rollout: make the feedback comparison without confusing it
   with open-loop prediction.
3. horizon_metrics: produce one value per prediction horizon.
4. bootstrap_success: describe uncertainty across actual evaluation episodes.
The provided runner picks identical rows for every checkpoint, runs all
conditions, plots curves, then invokes 10's real controller on matching seeds.

DEFINITION OF DONE
- Use only episode IDs held out by EVERY compared checkpoint.
- Record checkpoint paths, exact row IDs, horizons and evaluation seeds.
- Report all three curves and the action shuffle comparison.
- Run at least ten live episodes/model; read the uncertainty interval too.
- Explain one case where good prediction did or did not imply good control.
- Compare different history lengths by training separate 09 models; do not
  silently resize a trained positional embedding at evaluation time.
Run: python -m exercises.world_models.11_world_model_prediction_and_planning_benchmark
     --checkpoints outputs/dino_wm_pusht/predictor.pt --horizons 1 3 5 10
--offline-only is a debugging mode; it does not finish the planning comparison.

HOW TO START / WHAT TO INSPECT
Complete the model/rollout stubs in 09 and the planner stubs in 10 first. Supply
trained 09 checkpoints; this file does not create weights. Keep copies from
different runs before 09 overwrites predictor.pt. Comparing a path with itself
is useful for checking the runner, but is not a model improvement experiment.
The provided setup loads the dataset and DINO cache before assemble_window.
Follow the four stub checks before running the full benchmark. No new model
training should happen here, including inside observed_history_rollout.

prediction.json stores checkpoint paths, sampled anchors, per-window curves
and episode-weighted means; model_N_drift.png shows visual-error curves.
planning.json stores live summaries and success intervals. The model/horizon
subfolders contain 10's episode metrics and real videos. Read success, coverage
and planning time together. Weak success after a short 09 training run is a
result to diagnose, not a reason to replace evaluation with training loss.
Keep each original NotImplementedError commented when solving its stub.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

training = importlib.import_module("exercises.world_models.09_dino_wm_training")
planning = importlib.import_module("exercises.world_models.10_dino_wm_planning")
ROOT = Path(__file__).resolve().parents[2]


def assemble_window(model, source, rows: list[int], start: int, horizon: int,
                    cache: dict, saved: dict, device: torch.device) -> dict[str, Tensor]:
    # Where this fits:
    # benchmark_predictions chooses a common anchor for all models, then calls
    # this helper with each model's own start offset. All conditions for that
    # model receive this same window, so only the prediction rule changes.
    #
    # Goal:
    # Gather aligned history, controls and future targets from ONE held-out
    # episode. Keep future targets available to evaluation but separate from
    # the starting observations passed to an imagined rollout.
    #
    # Input:
    # - model: loaded 09 world model with H=config.history, S=config.action_chunk.
    # - source: Hugging Face Dataset with observation.state/action columns.
    # - rows: episode's ordered SOURCE row IDs; start is a position in this list.
    # - horizon: K future model steps, each containing S actual controls.
    # - cache: 'rows' source IDs and 'features' CPU [N,P,E] frozen DINO tokens.
    # - saved: this model's checkpoint, including state/action means and stds.
    # - device: where the frozen world model lives.
    # Example: H=3,K=2,S=5 requires five frame IDs and four action chunks.
    #
    # Output:
    # A dictionary of tensors on device, with these EXACT keys:
    # 'observations':     [1,H,P,394]       initial observations only
    # 'past_actions':     [1,H-1,S,2]       normalized controls before the anchor
    # 'future_actions':   [1,K,S,2]         normalized controls after the anchor
    # 'all_observations': [1,H+K,P,394]     real observations, including future
    # 'targets':          [1,K,P,384]       real future VISUAL features only
    # The leading 1 is the batch axis for one window, not a candidate axis.
    # Use model.config dimensions rather than hard-coding 394/384.
    #
    # Method:
    # training.build_window(rows,start,H+K-1,S) returns (frame_ids,chunk_ids).
    # Its third argument counts TRANSITIONS, so H+K-1 yields H+K observations.
    # The returned IDs index the source table, not necessarily cache positions.
    # torch.as_tensor(source['observation.state'],dtype=torch.float32) makes
    # a full [source_rows,2] tensor; index it with frame_ids for this window.
    # Normalize with (values-mean)/std using this checkpoint's saved statistics.
    #
    # Steps:
    # 1. Read H/S from model.config and K=horizon; reject K < 1. Call build_window
    #    with H+K-1 transitions. That helper checks the full window fits in rows.
    # 2. Make lookup={row_id:slot for slot,row_id in enumerate(cache['rows'])}.
    #    Use [lookup[row_id] for row_id in frame_ids] to gather cached features.
    #    A missing row is an error; do not substitute a nearby cached frame.
    # 3. Convert gathered features with .float().unsqueeze(0).to(device).
    #    Shape: [H+K,P,E] -> [1,H+K,P,E]. The cache stores float16 for space;
    #    the model/evaluation use float32 feature values here.
    # 4. Gather raw states at frame_ids as [H+K,2], move to device, subtract
    #    saved['stats']['state_mean'].to(device), divide by its state_std on
    #    the same device, then unsqueeze(0) -> [1,H+K,2]. Do not refit statistics.
    # 5. Within torch.no_grad(), call model.observation_tokens(features,states).
    #    This returns all_observations [1,H+K,P,E+condition_dim]. It uses this
    #    model's learned state encoder; do not reuse another model's state tokens.
    # 6. Convert actions with torch.as_tensor(source['action'],dtype=torch.float32).
    #    Index it with torch.as_tensor(chunk_ids,dtype=torch.long), giving
    #    [H+K-1,S,2]. Use ONE tensor of IDs: a nested Python list can be read
    #    as indices for different axes instead of one two-dimensional index.
    # 7. Move actions to device, call planning.normalize_actions(actions,saved),
    #    then unsqueeze(0) -> [1,H+K-1,S,2]. Normalize once, not again in imagine.
    # 8. Return observations=all_observations[:,:H], past_actions=actions[:,:H-1],
    #    future_actions=actions[:,H-1:], all_observations unchanged, and
    #    targets=features[:,H:]. Return dictionary keys, not a positional tuple.
    #
    # Checks:
    # - H=3,K=2: 5 observations, 2 past chunks, 2 future chunks, 2 visual targets.
    # - H=1: past_actions is [1,0,S,2], not None or a fabricated hold chunk.
    # - The last future action precedes the last target; it is not that target's
    #   outgoing action. Verify the frame/chunk IDs before running a large model.
    # - Open-loop callers use 'observations'; only the feedback comparison may
    #   read later entries of 'all_observations'. Targets never choose actions.
    raise NotImplementedError("Assemble one held-out multi-step benchmark window.")


@torch.no_grad()
def observed_history_rollout(model, window: dict[str, Tensor]) -> Tensor:
    # Where this fits:
    # This is the real-feedback comparison beside 09's imagine. It asks whether
    # errors shrink when each prediction starts from actual recorded history.
    # It does not learn from those observations or act in a live environment.
    #
    # Goal:
    # Predict K next observations, each using the latest H REAL observations.
    # Never feed a prediction back into this particular condition.
    #
    # Input:
    # window: assemble_window's dictionary. In particular, all_observations is
    # [1,H+K,P,394], past_actions [1,H-1,S,2], future_actions [1,K,S,2].
    # All controls are already normalized; all tensors are on the model device.
    #
    # Output:
    # Predicted visual features [1,K,P,384], ordered to match window['targets'].
    # It contains guesses, not copies of those target features.
    #
    # Method:
    # torch.cat([window['past_actions'],window['future_actions']],dim=1) makes
    # all_actions [1,H+K-1,S,2]. Every action chunk follows the matching real
    # observation in all_observations. model.predict accepts exactly H inputs
    # and H following chunks; use a sliding H-slot slice of both tensors.
    # @torch.no_grad() disables gradient recording; model.eval() also disables
    # dropout. These do different jobs and both matter for repeatable comparisons.
    #
    # Steps:
    # 1. Call model.eval(); read H=model.config.history, K from future_actions
    #    axis 1, and E=model.config.visual_dim. Start predictions=[].
    # 2. Concatenate past and future controls along dim=1 as described above.
    # 3. For k in range(K), take real_history=all_observations[:,k:k+H] and
    #    controls=all_actions[:,k:k+H]. Their last input is at position k+H-1;
    #    the unknown next observation is at position k+H, OUTSIDE that slice.
    # 4. Call model.predict(real_history,controls) -> [1,H,P,394]. Keep
    #    predicted[:,-1,:,:E] -> [1,P,E] and append it to predictions.
    #    Earlier output slots re-predict earlier transitions; they are not
    #    additional future targets for this benchmark step.
    # 5. Return torch.stack(predictions,dim=1). stack creates the K axis;
    #    concatenating along dim=1 would accidentally join patch axes instead.
    #
    # Concrete H=3,K=2 check:
    # all_observations=[A,B,C,D,E], all_actions=[aAB,aBC,aCD,aDE].
    # k=0: [A,B,C] plus [aAB,aBC,aCD] predicts D.
    # k=1: [B,C,D] plus [aBC,aCD,aDE] predicts E; D here is REAL.
    # Neither call sees the real observation it is currently asked to predict.
    #
    # Checks:
    # - The first output should match open-loop imagine's first output under
    #   eval mode: they start from identical history and controls.
    # - Later outputs may differ because their histories differ.
    # - Label this curve 'observed_history', never 'open_loop'. Low error here
    #   alone does not show the model can sustain an imagined trajectory.
    raise NotImplementedError("Measure next-step prediction with observed histories.")


def horizon_metrics(predictions: Tensor, targets: Tensor) -> dict[str, list[float]]:
    # Where this fits:
    # Every condition predicts the same real targets. This helper keeps a
    # separate score at each future time, so growing rollout error stays visible.
    # The runner later averages windows within episodes, then across episodes.
    #
    # Goal:
    # Measure visual error/directional similarity without averaging away K.
    # This is evaluation, not a loss that will be used for backward().
    #
    # Input:
    # predictions and targets: matching floating [B,K,P,E] on the same device.
    # B=windows in this call (normally 1), K=future chunks, P=patches,
    # E=DINO feature width. Learned state/action channels are already excluded.
    #
    # Output:
    # {'mse': [K Python floats], 'cosine': [K Python floats]}.
    # Entry 0 measures ONE chunk into the future, not the initial observed frame.
    # Return lists rather than CUDA tensors so the result can be written to JSON.
    #
    # Formula / method:
    # MSE[k] = sum((pred[b,k,p,e]-target[b,k,p,e])**2) / (B*P*E).
    # Tensor.square().mean(dim=(0,2,3)) removes B,P,E but preserves axis 1 (K).
    # F.cosine_similarity(predictions,targets,dim=-1,eps=1e-8) compares each
    # patch vector along its feature axis, returning [B,K,P]. F is the imported
    # alias for torch.nn.functional. Average this tensor over dim=(0,2).
    # Cosine divides vector dot products by their lengths, with eps protecting
    # near-zero lengths. Do not normalize the inputs before computing MSE.
    #
    # Steps:
    # 1. Require both tensors to have four axes, exactly matching shapes and
    #    nonempty dimensions. Raise ValueError otherwise; accidental broadcasting
    #    could hide a wrong target slice. Reject non-finite values too.
    # 2. Subtract targets from predictions, square, average over (0,2,3).
    # 3. Compute patch cosine similarities, then average over (0,2).
    # 4. For each [K] result, call .detach().cpu().tolist(). Return both lists
    #    under the exact keys 'mse' and 'cosine'. Do not call .item() on K values.
    #
    # Checks:
    # - Every prediction matching its nonzero target gives MSE=0 and cosine~1.
    # - With one patch per horizon, targets [[1,0],[1,0]] and predictions
    #   [[1,0],[2,0]], expected lists are mse=[0,0.5], cosine=[1,1].
    # - Identical zero vectors give MSE=0 but cosine=0 with this epsilon rule;
    #   do not claim all zero-error inputs must have cosine 1.
    # - Reordering K reorders the scores; K must not disappear into one mean.
    raise NotImplementedError("Compute visual prediction metrics at every horizon.")


def bootstrap_success(successes: list[bool], seed: int = 42) -> dict[str, float]:
    # Where this fits:
    # benchmark_planning has finished live MPC episodes and kept one success
    # flag per episode. This helper describes uncertainty from that finite sample.
    # It does not run the environment or train/evaluate any model again.
    #
    # Goal:
    # Report the observed success fraction and a 95% bootstrap percentile
    # interval by resampling WHOLE episode outcomes with replacement.
    #
    # Input:
    # successes: nonempty list[bool], one value per separately reset episode.
    # Example [True,False,True,True] becomes [1,0,1,1], with measured mean 0.75.
    # seed controls the NumPy resampling generator, not the live episode seeds.
    #
    # Output:
    # {'mean': float, 'lower': float, 'upper': float}, values in [0,1].
    # 'mean' is the ORIGINAL observed mean, not an average of bootstrap means.
    # lower/upper are the 2.5th and 97.5th percentiles of the resampled means.
    #
    # Method:
    # rng=np.random.default_rng(seed) makes a local NumPy random generator.
    # values=np.asarray(successes,dtype=float) -> [N] values, N=episode count.
    # rng.integers(0,N,size=(2000,N)) returns integer indices; upper bound N is
    # excluded. Every row is an independently resampled set of N episode IDs.
    # Duplicates are allowed: [2,2,0,1] is a valid four-episode bootstrap draw.
    # values[indices] gives [2000,N] outcomes; .mean(axis=1) gives 2000 means.
    # np.quantile(means,[0.025,0.975]) returns two interval endpoints.
    #
    # Steps:
    # 1. Raise ValueError for an empty list; there is no success rate to estimate.
    # 2. Convert flags to a NumPy float array and make the seeded generator.
    # 3. Draw the [2000,N] index array WITH replacement using integers above.
    #    Do not use one permutation: that keeps the mean identical in every draw.
    # 4. Gather outcomes and average each row over its N episode entries.
    # 5. Compute the two percentiles with np.quantile. Convert them with float().
    # 6. Return the original values.mean() as a Python float alongside the bounds.
    #
    # Why episodes, not frames?
    # A 300-frame failed attempt is one failure, not 300 independent failures.
    # Resampling its correlated frames would pretend we had much more evidence
    # and produce misleadingly narrow uncertainty estimates.
    #
    # Checks / limits:
    # - Same flags and seed give the same result, with 0 <= lower <= upper <= 1.
    # - All false -> mean=lower=upper=0; all true -> all three are 1.
    #   These degenerate intervals do NOT prove perfect certainty. Bootstrap
    #   cannot discover unseen outcomes; ten successes are not proof of 100%.
    # - A one-episode sample is computable but not a useful performance claim.
    # - Never replace this with mean loss or one success flag per video frame.
    raise NotImplementedError("Estimate uncertainty by resampling evaluation episodes.")


def common_protocol(saved_models: list[dict]) -> list[int]:
    # Provided fairness gate, called before any model comparison. The shared
    # encoder/cache identity makes visual MSE comparable; equal chunk length
    # makes horizon k represent the same duration. Different H values are allowed.
    # Intersect validation sets and remove every model's training IDs. It is
    # not enough for an episode to be held out by only the first checkpoint.
    identities = [s["feature_identity"] for s in saved_models]
    for key in ("dataset", "fingerprint", "encoder", "size", "grid", "preprocess"):
        if len({str(i[key]) for i in identities}) != 1:
            raise ValueError(f"Models differ in {key}; raw latent errors are not comparable.")
    if len({s["config"]["action_chunk"] for s in saved_models}) != 1:
        raise ValueError("All models must use the same action chunk length for this comparison.")
    common = set(saved_models[0]["splits"]["validation"])
    for saved in saved_models:
        common &= set(saved["splits"]["validation"])
        common -= set(saved["splits"]["train"])
    if not common:
        raise ValueError("No episode is held out by every model.")
    return sorted(common)


@torch.no_grad()
def benchmark_predictions(paths: list[Path], horizons: list[int], output: Path, device: torch.device) -> dict:
    # Provided offline runner. Load checkpoint metadata first, check compatibility,
    # then cache real held-out image features once. Load one dynamics model at a
    # time; its state embeddings and action normalization remain model-specific.
    # K=max(horizons) is used for one continuous curve covering 1..K. The requested
    # horizons list is also recorded so the live planner can test those settings.
    if not horizons or min(horizons) < 1:
        raise ValueError("Horizons must be positive action-chunk counts.")
    saved_models = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    for saved in saved_models:
        if saved.get("format") != "dino-wm-absolute-chunks-v1":
            raise ValueError("Every model must be an exercise 09 checkpoint.")
    ids = common_protocol(saved_models)
    config = training.Config(**saved_models[0]["config"])
    source = training.load_source(config)
    if source._fingerprint != saved_models[0]["data_fingerprint"]:
        raise ValueError("Dataset changed since training; do not silently reuse episode row IDs.")
    groups = training.group_episodes(source)
    cache = training.cache_features(source, [r for e in ids for r in groups[e]], config, device)
    max_history = max(s["config"]["history"] for s in saved_models)
    K, S = max(horizons), config.action_chunk
    anchors = []
    # An anchor is the final REAL history frame, shared even if H differs.
    # end is a POSITION inside an episode's rows list, not a source row ID.
    # Earliest anchor leaves enough past for max_history; latest leaves K*S
    # controls plus the final target. range excludes its right endpoint.
    # Choose at most eight windows per episode, deterministically, and use the
    # same anchors for all models. Episodes with no valid window contribute none.
    for episode in ids:
        options = list(range((max_history - 1) * S, len(groups[episode]) - K * S, S))
        random.Random(config.seed + episode).shuffle(options)
        anchors.extend((episode, end) for end in sorted(options[:8]))
    if not anchors:
        raise ValueError("Held-out episodes are too short for this history and horizon.")
    report = {"horizons": horizons, "environment_steps": [h * S for h in horizons],
              "anchors": anchors, "models": {}}
    for model_index, path in enumerate(paths):
        model, saved = training.load_checkpoint(path, device)
        records = []
        for episode, end in anchors:
            # Work back from the common anchor by this model's own H-1 chunks.
            # Different input histories still lead to identical future targets.
            window = assemble_window(model, source, groups[episode], end - (model.config.history - 1) * S,
                                     K, cache, saved, device)
            pred = training.imagine(model, window["observations"], window["past_actions"], window["future_actions"])[..., :config.visual_dim]
            # Shuffle chunks with a fixed permutation for fair model comparisons.
            # Leave controls WITHIN each chunk intact. The same permutation is
            # reused for every checkpoint; no target/image ordering is changed.
            # It can be the identity, especially for K=1; interpret the gap with
            # that limitation rather than assuming every window has wrong controls.
            order = torch.randperm(K, generator=torch.Generator().manual_seed(episode + end))
            wrong = training.imagine(model, window["observations"], window["past_actions"], window["future_actions"][:, order])[..., :config.visual_dim]
            conditions = {"open_loop": pred, "observed_history": observed_history_rollout(model, window),
                          "persistence": window["observations"][:, -1:, :, :config.visual_dim].expand(-1, K, -1, -1),
                          "shuffled_actions": wrong}
            # Persistence shares the last REAL visual grid across K time slots;
            # expand does not update it with each new real target. That would
            # measure a different, easier one-transition baseline.
            records.append({"episode": episode, "anchor_row": groups[episode][end],
                            "conditions": {name: horizon_metrics(p, window["targets"]) for name, p in conditions.items()}})
        # Average windows within episode first, then give each episode equal weight.
        # horizon_metrics has preserved K, so every averaging operation below
        # still produces a complete curve rather than a single error number.
        means = {}
        for condition in records[0]["conditions"]:
            means[condition] = {}
            for metric in ("mse", "cosine"):
                episode_means = [np.mean([r["conditions"][condition][metric] for r in records if r["episode"] == e], axis=0)
                                 for e in sorted({r["episode"] for r in records})]
                means[condition][metric] = np.mean(episode_means, axis=0).tolist()
        key = f"model_{model_index}"
        report["models"][key] = {"checkpoint": str(path), "history": model.config.history,
                                  "means": means, "windows": records}
        fig, ax = plt.subplots()
        for name, values in means.items():
            ax.plot(np.arange(1, K + 1) * S, values["mse"], label=name)
        ax.set(xlabel="actual environment steps into future", ylabel="visual MSE", title=key)
        ax.legend(); fig.tight_layout(); fig.savefig(output / f"{key}_drift.png", dpi=150); plt.close(fig)
        del model
    return report


def benchmark_planning(paths: list[Path], horizons: list[int], output: Path, device: torch.device,
                       episodes: int) -> dict:
    # Provided live test, not replay evaluation. For each model/horizon, reset
    # random control and MPC to the same seeds (PlanningConfig.seed + episode).
    # The same env instance is safe to reuse because each episode helper resets
    # it; the random trajectory is never the starting state for the MPC attempt.
    # Coverage/success come from env, never from the model's predicted goal cost.
    # Search settings other than horizon use 10's defaults. Compare time as well
    # as success because larger K increases computation/search difficulty.
    pixels = planning.make_goal_image()
    report = {}
    for index, path in enumerate(paths):
        model, saved = training.load_checkpoint(path, device)
        encoder = training.load_encoder(model.config, device)
        goal = training.encode_images(encoder, training.prepare_image(Image.fromarray(pixels), model.config).unsqueeze(0).to(device), model.config)[0]
        for horizon in horizons:
            config = planning.PlanningConfig(checkpoint=str(path), horizon=horizon, episodes=episodes,
                                              output=str(output / f"model_{index}_h{horizon}"))
            env = planning.make_environment(config.max_steps)
            runs = {"random": [], "mpc": []}
            try:
                for episode in range(episodes):
                    seed = config.seed + episode
                    runs["random"].append(planning.random_episode(env, seed))
                    runs["mpc"].append(planning.run_episode(env, seed, "mpc", model, encoder, saved, goal, config))
            finally:
                env.close()
            result = planning.save_results(runs, config)
            result["success_interval"] = bootstrap_success([r["success"] for r in runs["mpc"]])
            # One outcome per real MPC episode enters the bootstrap. The interval
            # is for this model/horizon's success fraction, not the difference
            # from random or another model. A paired comparison needs paired
            # episode differences and is outside the four tasks in this file.
            report[f"model_{index}_h{horizon}"] = result
        del encoder, model
    return report


def main() -> None:
    # Provided entry point: offline checks always run first. --offline-only
    # skips live control for debugging, so it cannot answer the exercise's title.
    # Run from the repository root; use distinct saved checkpoint paths when
    # comparing training settings. This script never edits those checkpoints.
    parser = argparse.ArgumentParser(description="Compare prediction and planning on matched PushT tasks.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=[ROOT / "outputs/dino_wm_pusht/predictor.pt"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "outputs/world_model_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        report = benchmark_predictions(args.checkpoints, args.horizons, output, device)
        (output / "prediction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.offline_only:
            result = benchmark_planning(args.checkpoints, args.horizons, output, device, args.episodes)
            (output / "planning.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except NotImplementedError as error:
        print(f"[TODO] {error}")


if __name__ == "__main__":
    main()
