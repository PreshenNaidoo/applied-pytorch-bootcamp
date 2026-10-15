"""
EXERCISE 13: V-JEPA 2-AC ROBOT DYNAMICS ON REAL DROID TRAJECTORIES

WHAT YOU ARE BUILDING
Freeze a pretrained V-JEPA 2 encoder and train the official action-conditioned
predictor on a real DROID subset. It predicts future spatial features from image
features, Cartesian robot state and motion. This is the action-conditioned
post-training stage, not another image autoencoder or a grid-world replacement.
Paper: https://arxiv.org/abs/2506.09985
Reference: https://github.com/facebookresearch/vjepa2
Read app/vjepa_droid/train.py, app/vjepa_droid/droid.py,
src/models/ac_predictor.py and configs/train/vitg16/droid-256px-8f.yaml.

SETUP
git clone https://github.com/facebookresearch/vjepa2.git data/reference_vjepa2
git -C data/reference_vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
uv pip install --python .venv/Scripts/python.exe timm==1.0.22 einops==0.8.1
The pinned classes are imported directly. The pretrained ViT-L encoder is
downloaded from the official checkpoint URL on the first feature-cache run.

Use the official 100-episode RLDS sample, not the multi-terabyte raw release:
  gsutil -m cp -r gs://gresearch/robotics/droid_100 data/droid_rlds
This needs Google's gsutil and about 2 GB of source data. The actual TFDS builder
directory is the downloaded directory containing dataset_info.json. Find it with:
  Get-ChildItem data/droid_rlds -Recurse -Filter dataset_info.json
Install the optional export dependencies in an isolated environment:
  uv venv .venv-droid --python 3.12
  uv pip install --python .venv-droid/Scripts/python.exe tensorflow-cpu tensorflow-datasets numpy opencv-python
Export with that environment (no PyTorch needed for this command):
  .venv-droid/Scripts/python.exe common/export_droid_world_model.py --builder-dir PATH_FROM_ABOVE
Then run this exercise in the normal PyTorch environment. The exporter keeps
RGB, Cartesian pose and gripper readings, and records source episode IDs.
Dataset/schema: https://droid-dataset.github.io/droid/the-droid-dataset

WHAT IS PRESERVED / WHAT IS REDUCED
Preserve the official action/state tokens, spatial patch grid, frame-causal
attention, rotary positions, teacher-forced plus autoregressive L1 losses, and
frame-by-frame frozen encoding. The reference repeats each frame twice for the
encoder's two-frame tubelet; it does NOT encode an entire future clip to obtain
the current input. We do the same, preventing future-image leakage.
Use pretrained ViT-L (1024 features) instead of ViT-G, and a 384-wide, six-layer
predictor instead of the reference's 1024-wide, 24-layer model. Both are real
upstream model classes. Keep 256-pixel images, 256 patches and 8-frame sequences.
Batch 1, a subset of episodes and cached frozen features fit the 8 GB budget.
The RLDS sample supplies lower-resolution images than the raw camera release.
These changes prevent direct loading of the published giant AC predictor;
the student checkpoint is a smaller post-training run, not published performance.

THE ACTION FIELD TRAP
Do NOT use RLDS step['action'] merely because it has seven numbers. It stores a
different control convention. Match the reference loader's poses_to_diffs:
  state = [x,y,z,roll,pitch,yaw,gripper]
  action = [change in xyz, relative rotation xyz angles, change in gripper]
Positions are metres, rotations radians. Relative rotation is R_next @ R_now.T;
subtracting Euler angles is generally wrong. scipy Rotation handles conversion.
These are achieved pose changes from the demonstration, not commanded motor
velocities. Offline accuracy under these conditions is not a robot success rate.

FULL WORKFLOW / SCOPE
pretrain video encoder (provided) -> freeze it -> prepare robot trajectories
-> train action-conditioned predictor -> evaluate held-out motion prediction
-> deploy a planner in a compatible robot environment (NOT covered here).
We do not pretend that a PushT simulator can evaluate a DROID robot checkpoint.

HOW THIS BUILDS ON 09-12
09 learned dynamics in a frozen image-feature space; 10 used those predictions
to choose controls; 11 separated prediction quality from real task success.
12 showed how masked video features are pretrained. Here we start AFTER that
large pretraining stage, using published V-JEPA 2 weights rather than 12's small
local run. We now teach a new predictor how robot motion relates to visual change.

MAIN OBJECTS / WHAT CHANGES
Object                       Role                                  Updated by
-----------------------------------------------------------------------------
exported DROID episode        real RGB, robot poses and gripper     fixed data
pretrained video encoder      describes each recorded image         frozen
cached features              spatial image descriptions            fixed data
states                       absolute recorded robot readings      fixed data
actions                      achieved changes between readings     fixed data
AC predictor                 predicts next visual patch features    optimizer
teacher-forced targets       real features one step later          fixed data
autoregressive history       earlier predicted visual features     predictor calls

Cartesian position means the end effector's x/y/z location, as opposed to the
angles of the robot's individual joints. Pose combines that position with its
orientation. AC means action-conditioned: the predictor receives robot state
and motion as well as image features. It does not choose the controls itself.
The robot's object contact and object pose are not supplied as separate labels;
their visual effects must be represented by the image features.

STATE AND ACTION ARE TWO DIFFERENT SEVEN-NUMBER OBJECTS
state_t = absolute [x,y,z,roll,pitch,yaw,gripper] measured at image t.
action_t = achieved motion FROM that state TO the next sampled state.
The gripper entry keeps the dataset's scalar reading; do not reinterpret it as
an angle or a position in metres. A difference of gripper readings describes
change in that scalar. Inputs keep the reference's units; do not reuse 09's
PushT coordinate means/stds or invent normalization for a different robot.

For a pure translation from x=0.40 to x=0.42, action x is +0.02 metres.
It is a displacement, not a velocity: do not divide by elapsed time. For
orientation, use relative rotation matrices rather than subtracting three
Euler angles. Euler angles describe orientation using three ordered rotations;
3D rotations generally depend on their order. The stub gives the exact recipe.

TIME IN THE EXPORTED DATA
The provided exporter keeps every fourth observation from the 15-Hz RLDS
sample: source steps [0,4,8,...] become exported frames [0,1,2,...]. The sampled
rate is 3.75 Hz, so one model transition spans about 0.267 seconds. An eight-frame
window has seven transitions, spanning about 1.87 seconds from first to last.
pose_differences operates on these sampled states, so actions describe the net
achieved motion BETWEEN sampled images. It does not output the four original
motor commands, and this exercise does not use 09's five-control action chunks.

The exporter writes one .npz file per episode. Relevant arrays are:
  images [episode_frames,3,256,256], uint8 RGB
  states [episode_frames,7], float32 pose/gripper readings
  source_frame_indices [episode_frames], original RLDS step numbers
  episode_id, source_path, sampled_fps: identity and timing metadata
It uses one exterior camera and resizes images to 256 pixels. It does not add
camera extrinsics (camera pose) as model inputs or transform the robot poses
into a newly invented coordinate frame. These are local data-adapter choices.

SHAPES / CONCRETE EXAMPLE
One sequence: 8 recorded images and states, 7 aligned motion vectors.
images [T,3,256,256] -> frozen features [T,256,1024]
batch z [B,T,P,E], state [B,T,7], actions [B,T-1,7].
If x moves from 0.40 to 0.42 metres with unchanged orientation/gripper, action
starts [0.02,0,0,...]. It predicts the feature change caused by that transition.
Predictor receives z[:,:-1].flatten(1,2), actions and states[:,:-1].
It returns [B,(T-1)*P,E]; reshape this to [B,T-1,P,E].

WHY REPEAT EACH IMAGE TWICE?
The pretrained encoder begins with a two-frame tubelet: its first layer expects
two frames to make one temporal slice of patch features. Encoding the WHOLE
eight-frame sequence would let its video attention mix later images into an
earlier representation. A causal predictor could then receive the future answer
inside its supposedly current features, even if its own attention mask is correct.

Instead, take one real image A, make the two-frame input [A,A], encode it, and
keep its 16x16 spatial grid. Then independently encode [B,B], and so on.
This is the reference AC feature-extraction recipe. The duplication adapts the
video encoder's input shape; it is not two independently observed moments or a
claim that the robot stopped moving. No later real image can enter A's features.

  image A [3,256,256]
    -> [1,3,2,256,256], the same image in both temporal slots
    -> encoder [1,256,1024], one time slice of 16x16 patches
    -> layer norm across 1024 features -> cached A [256,1024]

NORMALIZATION AND PREDICTION MEANING
The cache contains layer-normalized image features, not pixels or labels.
Predictor outputs are also layer-normalized across feature width, following
the reference normalize_reps recipe. This differs from 12, where you did not
add normalization to the predictor output. Layer norm is not unit-vector norm.
The predictor returns the full NEXT feature vector, not a feature difference;
do not add its output to the current image features as a residual prediction.

WHAT THE PREDICTOR DOES INSIDE
For each input time, it maps the 256 visual tokens from width 1024 to 384 and
adds two separate 384-value tokens: one for action and one for robot state.
The internal sequence is [action_t,state_t,patch_0,...,patch_255] for each time.
Its frame-causal attention permits the current and earlier frames, not later
ones. Rotary positions, also called RoPE, mark time and spatial location by
rotating attention query/key components according to position.
The module removes action/state tokens from its OUTPUT and projects visual
tokens back to width 1024. The student passes raw state/action vectors; the
reference module handles their embeddings, interleaving and attention mask.

LOSS / WHAT IS KNOWN DURING ROLLOUT
Teacher-forced L1 compares predictions from real image histories with z[:,1:].
Autoregressive L1 compares repeated predictions with the same frozen targets.
total = teacher_forced_L1 + autoregressive_L1 (first two imagined steps by default).
Like the reference training code, both paths condition on recorded robot states.
Thus autoregressive here means no future IMAGES enter the rollout. Future robot
poses ARE supplied. Every report labels this as state-conditioned prediction.
It must not be described as a fully autonomous robot rollout.

FOLLOW BOTH TRAINING PATHS
Let z0,z1,z2 be features of real images, s0,s1 their recorded robot states,
and a0,a1 achieved motions between those states:

  teacher forcing:
    [z0,s0,a0]                  -> guess for z1
    [z0,s0,a0; real z1,s1,a1]   -> guess for z2
  autoregressive:
    [z0,s0,a0]                  -> guessed z1
    [z0,s0,a0; guessed z1,s1,a1]-> guessed z2

Teacher forcing evaluates all seven transitions in one causal call. Autoregressive
training starts at the first frame and predicts two steps by default; validation
tests up to seven. It never receives real z1/z2 as new inputs, but it DOES receive
recorded s1 and a1 for its second prediction. It does not predict robot state.
The action itself was derived from recorded future pose change. These are
conditional offline predictions, not forecasts of what arbitrary robot commands
will achieve. A deployment planner would need compatible proposed motions/states.

WHY KEEP GRADIENTS THROUGH IMAGINED FRAMES?
The two-step loss should teach the model to make a first prediction that supports
a good second prediction. If guessed z1 is detached before predicting z2, that
learning path is cut. Keep autoregressive differentiable during training; the
provided evaluation wrapper disables gradients only when measuring.
This differs from 09's imagine helper, which was built for no-gradient planning.

ONE TRAINING UPDATE, IN ORDER
1. Sample complete cached windows from training episodes, adding a batch axis.
2. Move features/states/actions to device; clear old predictor gradients.
3. Teacher-forced path predicts all real next frames using real histories.
4. Autoregressive path predicts its short future using earlier visual guesses.
5. Compare each path with its aligned frozen targets using mean absolute error.
6. ADD the two scalar means, backpropagate, clip gradients and step AdamW.
7. Repeat. At epoch end, measure longer state-conditioned rollouts on validation.

The two losses have equal coefficients, but cover different numbers of frames.
Do not concatenate all errors and average: that would give the seven-step
teacher-forced path more weight than the two-step autoregressive path. Unlike
12 there is no EMA encoder update: this pretrained encoder stays fixed throughout.

LEARNING ORDER
1. pose_differences: build the reference's motion meaning from measured poses.
2. encode_frames: use the pretrained video encoder without seeing future frames.
3. build_predictor: construct the actual AC predictor at a smaller width/depth.
4. teacher_forced: predict every real transition with aligned conditioning.
5. autoregressive: feed predictions back while preserving the stated pose conditioning.
6. dynamics_loss: combine the two reference losses without changing the encoder.
Data export, episode splits, cache files, optimization, plots and checkpoints
are provided. The first stub operates on actual robot poses before model downloads.

DEFINITION OF DONE
Train/validation/test use separate complete episodes. Report L1/cosine at each
horizon, no-change baseline, and shuffled-action results. Lower L1 and higher
cosine are better; correct controls should usually beat shuffled controls.
Save real frame strips and error heatmaps, plus model/config/split metadata.
Do not claim task success from offline loss or use a random encoder fallback.

READING THE METRICS AND PLOTS
prediction_l1[k-1] is mean absolute feature error k sampled transitions after
the starting image; lower is better. It has feature units, not metres or radians.
persistence_l1 keeps the first visual grid unchanged for every future time.
shuffled_action_l1 shifts the recorded actions by one time slot, keeping visual
targets AND recorded robot states fixed. Repeated actions may barely change.
Because states contain related motion information, this is a conditioning
ablation, not an isolated proof that actions cause successful robot behavior.
cosine is patch-wise vector-direction similarity; nearer 1 is better, but it
does not by itself show low absolute error or task completion.

Average windows within each episode, then give episodes equal weight. Overlapping
or related frames from one attempt are not independent evidence. The splits
separate source episodes, not necessarily robot sites, scenes or task families.
If loss stays poor, first check pose/action alignment, rotation convention,
frozen encoder loading and target normalization. Longer-horizon failure can
persist even after the two-step training loss improves; inspect rollout.png.

real_frames_and_feature_error.png shows actual target images above 16x16 grids
of feature error. These heatmaps are neither reconstructed images nor object
segmentation masks. Large errors identify patch representations the model missed.
predictor.pt saves weights/config/split paths, not an optimizer resume state;
metrics.json records training history and state-conditioned test results.

HOW TO START
Follow the export commands before running: the script requires at least five
complete real episode files and does not make substitute data. With exports
present, it reaches pose_differences before downloading the pretrained encoder.
The frozen feature cache then makes later predictor runs cheaper. Use a fresh
Python process so the pinned reference src package cannot conflict with 12's.
Work through the six stub checks in order; keep each original NotImplementedError
line commented when solving it. A full run trains only the AC predictor.
Run: python -m exercises.world_models.13_vjepa2_action_conditioned_robot_dynamics
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation

from common.video_world_model_data import ROOT, normalize_video, reference_import

REFERENCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"


@dataclass
class Config:
    # frames counts exported observations, not original 15-Hz RLDS steps.
    # patches=(256/16)**2 and feature_dim=1024 are fixed by the chosen encoder.
    # predictor_dim/depth/heads size the trainable dynamics model.
    # autoregressive_steps is the short TRAINING rollout; evaluation uses frames-1.
    # windows_per_episode limits cached windows while keeping episode splits intact.
    reference: str = str(ROOT / "data/reference_vjepa2")
    episodes_dir: str = str(ROOT / "data/droid_world_model")
    output: str = str(ROOT / "outputs/vjepa2_robot_dynamics")
    frames: int = 8
    image_size: int = 256
    patches: int = 256
    feature_dim: int = 1024
    predictor_dim: int = 384
    predictor_depth: int = 6
    predictor_heads: int = 6
    autoregressive_steps: int = 2
    batch_size: int = 1
    epochs: int = 20
    windows_per_episode: int = 4
    lr: float = 1e-4
    seed: int = 42


def pose_differences(states: Tensor) -> Tensor:
    # Where this fits:
    # cached_sequences calls this on a complete exported episode before slicing
    # windows. It defines the physical meaning of every later action token.
    #
    # Goal:
    # Convert adjacent recorded poses into achieved translation, relative
    # rotation and gripper change, matching the reference poses_to_diffs method.
    #
    # Input:
    # CPU float32 states [T,7], T>=2. Each row is
    # [x,y,z,roll,pitch,yaw,gripper]: metres, xyz Euler radians, recorded gripper scalar.
    # Rows are already temporally subsampled; these are not motor-command vectors.
    #
    # Output:
    # CPU float32 [T-1,7]. Row t describes the motion from states[t] to states[t+1].
    # Order: [delta_x,delta_y,delta_z,relative_roll,relative_pitch,relative_yaw,
    #         delta_gripper]. No padding row or batch dimension is added.
    #
    # Method:
    # Rotation is imported from scipy.spatial.transform. Use the full calls:
    # Rotation.from_euler('xyz',values[:,3:6],degrees=False).as_matrix()
    # Rotation.from_matrix(relative).as_euler('xyz',degrees=False)
    # Lowercase 'xyz' specifies SciPy's extrinsic (fixed-axis) xyz convention, as in the
    # reference; changing order/case changes the orientation convention.
    # The first call returns matrices [T,3,3], the second angles [T-1,3].
    # Relative rotation is R_next @ R_now.T, not elementwise angle subtraction.
    #
    # Steps:
    # 1. Require a finite [T,7] tensor with T>=2; raise ValueError otherwise.
    #    Convert using values=states.detach().cpu().numpy(). This preprocessing
    #    has no trainable parameters and needs no autograd path.
    # 2. Compute values[1:,:3]-values[:-1,:3] -> translations [T-1,3].
    # 3. Convert all values[:,3:6] to rotation matrices using from_euler above.
    # 4. Compute relative=matrices[1:] @ matrices[:-1].transpose(0,2,1).
    #    NumPy @ multiplies corresponding 3x3 matrices. transpose(0,2,1)
    #    swaps the LAST two axes; matrices.T would also reverse the time axis.
    # 5. Convert relative matrices to xyz Euler radians using from_matrix above.
    # 6. Compute gripper change with values[1:,6:7]-values[:-1,6:7]. The slice
    #    6:7 preserves [T-1,1]; using just 6 removes that needed column axis.
    # 7. Join [translations,relative_angles,gripper_change] using
    #    np.concatenate([translations,relative_angles,gripper_change],axis=1).
    #    Return torch.from_numpy(motions).float(); SciPy may have produced float64.
    #    Do not divide displacement by elapsed time or apply PushT statistics.
    #
    # Checks:
    # - Repeated states -> all zeros. A pure x move .40 -> .42 produces .02.
    # - A gripper reading .2 -> .5 produces .3 in the final column.
    # - For pure yaw, +179 degrees -> -179 degrees is a small +2-degree
    #   relative rotation, not a -358-degree turn. Convert test angles to radians.
    # - For general orientations, compare reconstructed rotation MATRICES when
    #   testing: multiple Euler triples can represent the same rotation, and
    #   near singular orientations SciPy can choose a different equivalent triple.
    raise NotImplementedError("Convert recorded Cartesian poses into aligned relative motions.")


def load_frozen_encoder(config: Config, device: torch.device) -> nn.Module:
    # Provided: construct the official ViT-L architecture, then strictly load
    # published weights. num_frames=64 matches its nominal video setup; rotary
    # positions let the actual AC input be one duplicated two-frame image.
    # Only the unused fixed pos_embed entry is omitted, as in the upstream loader.
    # eval() controls module behavior; requires_grad_(False) freezes parameters.
    # This encoder is used only for cache creation, never placed in the optimizer.
    module = reference_import(Path(config.reference), REFERENCE_COMMIT, "src.models.vision_transformer")
    encoder = module.vit_large(img_size=(256, 256), patch_size=16, num_frames=64,
                               tubelet_size=2, use_sdpa=True, use_silu=False, wide_silu=True,
                               uniform_power=False, use_rope=True)
    saved = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
                                              map_location="cpu", weights_only=True)
    state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in saved["target_encoder"].items()}
    # This pretrained checkpoint has fixed positions; the official loader uses
    # rotary positions and omits only this unused entry. All learned weights match.
    state.pop("pos_embed", None)
    encoder.load_state_dict(state, strict=True)
    del saved, state
    return encoder.eval().requires_grad_(False).to(device)


@torch.no_grad()
def encode_frames(encoder: nn.Module, images: Tensor, config: Config, device: torch.device) -> Tensor:
    # Where this fits:
    # Before predictor training, cached_sequences encodes real images once.
    # These fixed features serve both as predictor inputs and as later targets.
    #
    # Goal:
    # Make one spatial feature grid per image without seeing any later image.
    # Run the frozen video encoder on TWO COPIES of one frame at a time.
    #
    # Input:
    # images: CPU uint8 RGB [T,3,256,256], already resized by the exporter.
    # encoder: official pretrained ViT-L, frozen/eval on device. config defines
    # image_size=256, patches=256 and feature_dim=1024 for this encoder.
    #
    # Output:
    # CPU float16 [T,256,1024], in the same time order as images. There is no
    # batch axis in the cache. No patch pooling or class-token removal is needed.
    #
    # Method:
    # normalize_video(images.float()/255.0) applies the encoder's RGB channel
    # means/stds to [T,3,H,W]. This is image preprocessing, not robot pose scaling.
    # For frame [3,H,W], frame.unsqueeze(0) -> [1,3,H,W]; unsqueeze(2) then
    # inserts time -> [1,3,1,H,W]; .repeat(1,1,2,1,1) -> [1,3,2,H,W].
    # encoder(clip) returns Tensor [1,256,1024] directly, not an output dictionary.
    # F.layer_norm(tokens,(tokens.shape[-1],)) normalizes per patch across width.
    #
    # Steps:
    # 1. Verify the input image shape/dtype, call encoder.eval(), convert pixels
    #    to float [0,1] and call normalize_video once. Start an empty features list.
    # 2. Loop through normalized frames in order. Duplicate each frame using the
    #    operations above and move that small two-frame clip to device.
    # 3. In torch.autocast(device_type=device.type,dtype=torch.bfloat16,
    #    enabled=device.type=='cuda'), run tokens=encoder(clip).
    #    The provided @torch.no_grad() already disables gradient recording.
    # 4. Convert tokens to float32 and layer-normalize its last dimension. Do
    #    this before half-precision storage, rather than normalizing quantized
    #    cache values later. Check [1,config.patches,config.feature_dim].
    # 5. Remove only batch with .squeeze(0), then .cpu().half() and append it.
    #    Float16 is a disk/CPU cache choice; training converts features to float32.
    # 6. Return torch.stack(features,dim=0) -> [T,P,E]. Do not concatenate patches
    #    from different frames into one unlabelled axis at this stage.
    #
    # Checks:
    # - Output dtype=torch.float16, device.type='cpu', requires_grad=False.
    # - Encoding identical frames independently gives matching feature grids.
    # - Changing image t+1 cannot change cached features for image t. Calling
    #   the encoder on the whole real sequence would violate this requirement.
    # - No decoder or pixel reconstruction is involved; values are features.
    raise NotImplementedError("Cache frame-wise pretrained V-JEPA 2 features.")


def build_predictor(config: Config, device: torch.device) -> nn.Module:
    # Where this fits:
    # Once real feature caches exist, construct the only model that will learn.
    # Its internal state/action embedding layers are trained with the predictor.
    #
    # Goal:
    # Instantiate the actual reference AC Transformer at the smaller local size.
    # Do not reuse 12's masked-feature predictor: that has a different input API.
    #
    # Input:
    # config gives reference path and architecture sizes; device holds training.
    # No pretrained AC predictor weights are loaded; this predictor starts random.
    # Output: trainable nn.Module on device; the provided run creates AdamW later.
    #
    # Method / full call:
    # module=reference_import(Path(config.reference),REFERENCE_COMMIT,
    #                         'src.models.ac_predictor')
    # module.vit_ac_predictor(
    #     img_size=config.image_size,patch_size=16,num_frames=config.frames,
    #     tubelet_size=1,embed_dim=config.feature_dim,
    #     predictor_embed_dim=config.predictor_dim,depth=config.predictor_depth,
    #     num_heads=config.predictor_heads,action_embed_dim=7,
    #     is_frame_causal=True,use_extrinsics=False,use_rope=True,
    #     use_silu=False,use_activation_checkpointing=True)
    #
    # Argument meanings:
    # - img_size=256 and patch_size=16 tell the module its spatial grid is 16x16.
    # - num_frames=8 allocates a mask large enough for the input history. Calls
    #   with shorter prefixes are supported; keep histories within this capacity.
    # - tubelet_size=1 counts already-encoded frames here. It does NOT change the
    #   video encoder's two-frame tubelets or justify encoding real future images.
    # - embed_dim=1024 is input/output feature width; predictor_embed_dim=384 is
    #   internal width. The module learns both projections between these widths.
    # - depth=6 stacks six attention blocks; num_heads=6 splits attention into
    #   six groups. Keep compatible widths/heads when changing the architecture.
    # - action_embed_dim=7 is the RAW vector length for BOTH action and state
    #   encoders. It is not the final action-token embedding width, which is 384.
    # - is_frame_causal=True blocks future frame groups, preventing a teacher-
    #   forced prediction from reading the later real image it must predict.
    # - use_extrinsics=False excludes camera-pose conditioning, as in this recipe.
    # - use_rope=True enables reference time/spatial rotary position handling.
    # - use_silu=False selects the reference GELU feed-forward activation.
    # - use_activation_checkpointing=True recomputes activations in backward
    #   to save GPU memory. It does not save a training checkpoint to disk.
    #
    # Steps:
    # 1. Import the pinned module using reference_import above.
    # 2. Construct the predictor with the complete call and move it via .to(device).
    # 3. Return it; do not freeze its learned layers or create an encoder inside it.
    #
    # Checks:
    # - For L input frames, call predictor(x,actions,states) with
    #   x [B,L*256,1024], actions [B,L,7], states [B,L,7]. Output is [B,L*256,1024].
    # - Internally the attention length is L*(256+2), including state/action
    #   tokens. Those extra tokens are removed before the return value.
    # - The upstream boolean mask uses True for ALLOWED attention; do not pass
    #   09's True-means-blocked TransformerEncoder mask into this module.
    raise NotImplementedError("Build the official action-conditioned predictor.")


def teacher_forced(predictor: nn.Module, features: Tensor, actions: Tensor, states: Tensor) -> Tensor:
    # Where this fits:
    # This makes the first prediction path in dynamics_loss. It predicts all
    # T-1 real transitions in one call, with recorded images as history inputs.
    #
    # Goal:
    # Return aligned next-frame feature guesses for every input time. Teacher
    # forcing means later time slots use REAL prior images, not earlier guesses.
    #
    # Input:
    # features [B,T,P,E]: fixed, normalized features of T real images.
    # actions [B,T-1,7]: achieved motions linking them; states [B,T,7]: real poses.
    # Default B=1,T=8,P=256,E=1024. All tensors already float32 on the model device;
    # run's autocast controls suitable operation dtypes during training.
    #
    # Output:
    # [B,T-1,P,E] predictions, where output[:,i] targets features[:,i+1].
    # Gradients remain attached to predictor weights; no loss or optimizer here.
    #
    # Method:
    # Tensor.flatten(start_dim=1,end_dim=2) joins TIME and PATCH axes, leaving B/E.
    # predictor(features[:,:-1].flatten(1,2),actions,states[:,:-1]) returns
    # [B,(T-1)*P,E]. The module internally attaches state/action tokens to each
    # frame group, applies causal attention and removes conditioning tokens.
    # F.layer_norm(output,(E,)) normalizes its predicted feature vectors.
    #
    # Steps:
    # 1. Unpack B,T,P,E=features.shape; require T>=2 and aligned states/actions.
    #    There must be T states and exactly T-1 motions, not T padded motions.
    # 2. Select features[:,:-1] -> [B,T-1,P,E]; flatten axes 1/2 to [B,(T-1)*P,E].
    #    Select states[:,:-1] -> [B,T-1,7]; all actions already have this length.
    # 3. Call predictor(input_tokens,actions,input_states). Do not pass a future
    #    state at input slot i: that slot gets states[:,i], not states[:,i+1].
    # 4. Apply F.layer_norm to its last width E and reshape to [B,T-1,P,E].
    # 5. Return predictions. Do not detach or decorate this helper with no_grad;
    #    later teacher-forced loss must train the predictor through this path.
    #
    # Concrete T=3 mapping:
    # slot 0 uses (z0,s0,a0) and predicts z1;
    # slot 1 sees (z0,s0,a0) and (real z1,s1,a1), then predicts z2.
    # z1 occurs both as a target and a later input. The FRAME-CAUSAL attention
    # mask is what prevents slot 0 from reading that later input and cheating.
    #
    # Checks:
    # - Default output shape [1,7,256,1024], aligned with features[:,1:].
    # - In predictor.eval(), changing a later input frame/state/action cannot
    #   change an earlier output slot beyond numerical tolerance.
    # - The final real frame and final state do not enter the input call.
    raise NotImplementedError("Predict aligned teacher-forced robot transitions.")


def autoregressive(predictor: nn.Module, first: Tensor, actions: Tensor, states: Tensor, steps: int) -> Tensor:
    # Where this fits:
    # dynamics_loss uses this for a short training rollout; evaluate uses it
    # for all seven transitions without gradients. The SAME helper serves both.
    #
    # Goal:
    # Start from one real image's features and feed each visual prediction back
    # as history. Continue using the supplied recorded robot states and motions.
    #
    # Input:
    # first [B,1,P,E]: the ONLY real visual input, with time axis preserved.
    # actions [B,T-1,7], states [B,T,7]: known demonstration conditioning.
    # steps: number of visual predictions, 1 <= steps <= T-1 and no greater
    # than the predictor's configured history capacity. Training defaults to 2.
    #
    # Output:
    # [B,steps,P,E], excluding the real starting frame. Keep the computation
    # graph during training. No robot-state prediction is returned or computed.
    #
    # Method:
    # predictor(history.flatten(1,2),actions[:,:k+1],states[:,:k+1]) uses
    # a growing history of k+1 frames on iteration k (zero-based).
    # Output includes a next-frame guess for EVERY input frame; only its last
    # P tokens are the next new step. Earlier outputs re-predict older steps.
    # torch.cat([history,next_frame],dim=1) appends one temporal slot.
    # Unlike 09's fixed sliding H-frame rollout, this short sequence grows from
    # one input frame; the reference accepts these prefixes within its capacity.
    #
    # Steps:
    # 1. Read B,one,P,E=first.shape; require one==1, valid positive steps and
    #    enough aligned action/state entries. Set history=first.
    #    Do not call predictor.eval() here: the outer caller chooses train/eval.
    # 2. For k in range(steps), history has [B,k+1,P,E]. Flatten time/patch to
    #    [B,(k+1)*P,E], keeping the features of all earlier predicted frames.
    # 3. Pass that tensor, actions[:,:k+1] and states[:,:k+1] to predictor.
    #    Both conditioning tensors have [B,k+1,7], one pair per history frame.
    # 4. Select output[:,-P:,:] -> [B,P,E]. Apply F.layer_norm(last,(E,)), then
    #    .unsqueeze(1) -> [B,1,P,E] for the new frame's time axis.
    # 5. Append it with torch.cat([history,next_frame],dim=1). Do NOT detach,
    #    convert to NumPy, enter no_grad, or fetch a real next image here.
    # 6. After the loop return history[:,1:], dropping the initial real frame.
    #    Do not stack the whole history again or include re-predicted old outputs.
    #
    # Two-step trace:
    # k=0: history=[real z0], actions=[a0], states=[s0] -> predicted z1
    # k=1: history=[real z0,predicted z1], actions=[a0,a1], states=[s0,s1]
    #      -> predicted z2. Return [predicted z1,predicted z2].
    # s1 is recorded future robot state relative to the initial image; it is
    # intentionally supplied. Do not describe this as fully open-loop robot control.
    #
    # Checks:
    # - steps=1 returns [B,1,P,E]; steps=2 performs two predictor calls.
    # - With dropout disabled, step 1 agrees with teacher_forced's first output,
    #   because both see the same initial image/state/motion and causal mask.
    # - A loss on step 2 can send gradients through predicted z1. If inspecting
    #   that intermediate tensor's .grad in a test, call .retain_grad() first:
    #   predicted z1 is not a leaf tensor.
    # - No input contains future real image features. The provided states/actions
    #   still contain information about the recorded motion, as declared above.
    raise NotImplementedError("Roll predicted visual features forward with robot conditioning.")


def dynamics_loss(predictor: nn.Module, batch: dict[str, Tensor], config: Config) -> Tensor:
    # Where this fits:
    # run moves a cached batch to device and calls this inside autocast. This
    # helper builds both training paths and returns the objective for backward.
    #
    # Goal:
    # Train next-frame prediction from real histories AND short visual rollouts
    # from earlier guesses, using the same predictor weights for both paths.
    #
    # Input:
    # batch dictionary: 'features' [B,T,P,E], 'states' [B,T,7],
    # 'actions' [B,T-1,7], all already on device. S=config.autoregressive_steps
    # controls training rollout length; S=2 by default, T=8. S is NOT an action
    # chunk size here. The feature encoder is frozen and absent from this helper.
    #
    # Output:
    # One scalar Tensor, shape [], attached to both prediction branches.
    # Do not return a Python float or call optimizer.step/backward here.
    #
    # Formula / method:
    # tf_loss = F.l1_loss(tf_predictions,features[:,1:].detach(),reduction='mean')
    # ar_loss = F.l1_loss(ar_predictions,features[:,1:1+S].detach(),reduction='mean')
    # total = tf_loss + ar_loss
    # F is torch.nn.functional; L1 means average ABSOLUTE error, not squared error.
    # Each mean covers its own B/time/P/E axes. For tf_loss=.3 and ar_loss=.5,
    # total is .8, not .4. The first transition contributes to both objectives,
    # as in the reference; do not remove it from AR targets as a 'duplicate'.
    #
    # Steps:
    # 1. Read features=batch['features'], states=batch['states'],
    #    actions=batch['actions']; require 1 <= S <= features.shape[1]-1.
    # 2. Call teacher_forced(predictor,features,actions,states) -> [B,T-1,P,E].
    # 3. Call autoregressive(predictor,features[:,:1],actions,states,S)
    #    -> [B,S,P,E]. [:,:1] preserves the initial one-frame time axis.
    # 4. Select both target slices above and verify shapes match predictions
    #    exactly. Do not let loss broadcasting conceal a shifted time axis.
    # 5. Compute each mean L1 with detached targets, then ADD the two scalars.
    # 6. Return that tensor. Keep both prediction graphs attached. The runner
    #    clears gradients, backpropagates, clips and updates parameters later.
    #
    # Checks:
    # - For defaults, teacher targets are frames 1..7; AR targets frames 1..2.
    # - Frozen cache features need no gradients. Predictor parameters receive
    #   gradients from both losses, including the repeated-prediction path.
    # - Zero errors on both paths give total zero. Equal per-entry errors do
    #   not weight teacher forcing 7/2 times more heavily because it has more frames.
    # - Reference code reuses teacher forcing's first output to start its AR
    #   path. This helper recomputes it through autoregressive for a simpler
    #   reusable API; with this zero-dropout predictor it is the same calculation
    #   up to numerical precision, with some extra compute.
    raise NotImplementedError("Combine one-step and repeated-prediction L1 losses.")


def split_paths(config: Config) -> dict[str, list[Path]]:
    # Provided: approximately 60/20/20 split of whole exported episodes, selected
    # reproducibly. A source trajectory's windows must all belong to one split.
    # cached_sequences also checks original episode IDs, catching renamed copies.
    paths = sorted(Path(config.episodes_dir).glob("episode_*.npz"))
    if len(paths) < 5:
        raise FileNotFoundError("Export at least five complete real DROID episodes with common/export_droid_world_model.py first.")
    random.Random(config.seed).shuffle(paths)
    holdout = max(1, len(paths) // 5)
    return {"train": paths[2 * holdout:], "validation": paths[:holdout], "test": paths[holdout:2 * holdout]}


def cached_sequences(config: Config, splits: dict[str, list[Path]], device: torch.device) -> dict[str, list[dict]]:
    # Provided: read exported RGB/state arrays, derive motions, choose up to four
    # non-overlapping eight-frame windows per episode, and cache frozen features.
    # Return {'train':[sequence,...],'validation':[...],'test':[...]}. Each sequence
    # has features [T,P,E], states [T,7], actions [T-1,7], original images and
    # episode/start metadata. No batch axis yet; FeatureDataset/DataLoader add it.
    # Cache identity includes source content hash, selected windows and encoding
    # recipe. It must not silently reuse features made from different source data.
    output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
    encoder = None
    results = {name: [] for name in splits}
    try:
        for split, paths in splits.items():
            for path in paths:
                with np.load(path, allow_pickle=False) as source:
                    states = torch.from_numpy(source["states"].copy()).float()
                    images = torch.from_numpy(source["images"].copy())
                    episode_id = str(source["episode_id"].item())
                if states.ndim != 2 or states.shape[1] != 7 or images.shape != (len(states), 3, 256, 256):
                    raise ValueError(f"Unexpected DROID export shapes in {path}.")
                motions = pose_differences(states)  # first student task, before downloading encoder
                # Compute motion BEFORE window slicing. A window starting at j
                # needs states/images j..j+T-1 and motions j..j+T-2, all aligned.
                starts = list(range(0, len(states) - config.frames + 1, config.frames))
                random.Random(config.seed + int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16)).shuffle(starts)
                starts = sorted(starts[:config.windows_per_episode])
                if not starts:
                    raise ValueError(f"Episode too short: {path}")
                identity = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "starts": starts,
                            "frames": config.frames, "encoder": "official-vjepa2-vitl", "size": 256,
                            "reference": REFERENCE_COMMIT, "encoding": "repeat-each-frame-twice-layernorm-v1"}
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
                cache_path = output / f"features_{digest}.pt"
                if cache_path.exists():
                    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                    if cached["identity"] != identity:
                        raise ValueError("Feature cache identity mismatch.")
                    sequences = cached["sequences"]
                else:
                    if encoder is None:
                        encoder = load_frozen_encoder(config, device)
                    sequences = []
                    for start in starts:
                        features = encode_frames(encoder, images[start:start + config.frames], config, device)
                        sequences.append({"features": features, "states": states[start:start + config.frames],
                                          "actions": motions[start:start + config.frames - 1],
                                          "episode": episode_id, "start": start,
                                          "images": images[start:start + config.frames]})
                    torch.save({"identity": identity, "sequences": sequences}, cache_path)
                results[split].extend(sequences)
    finally:
        if encoder is not None:
            del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # The export can be copied/renamed; detect leakage using original episode IDs too.
    id_sets = {name: {s["episode"] for s in sequences} for name, sequences in results.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if id_sets[a] & id_sets[b]:
            raise ValueError(f"Duplicate source episode across {a} and {b}.")
    return results


class FeatureDataset(Dataset):
    # Provided training view of the cached sequences. Keep metadata/images for
    # reporting outside the batch; expose only the three predictor/loss tensors.
    # .float() restores float32 from compact half-precision feature storage.
    def __init__(self, sequences: list[dict]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {key: self.sequences[index][key].float() for key in ("features", "states", "actions")}


@torch.no_grad()
def evaluate(predictor, sequences: list[dict], config: Config, device: torch.device) -> dict:
    # Provided measurement only. The decorator turns OFF rollout gradients here,
    # while the same autoregressive helper must keep them during training.
    # Evaluate frames-1 transitions even if training only used two AR steps.
    predictor.eval()
    records = []
    for sequence in sequences:
        b = {k: sequence[k].float().unsqueeze(0).to(device) for k in ("features", "actions", "states")}
        K = config.frames - 1
        target = b["features"][:, 1:]
        predicted = autoregressive(predictor, b["features"][:, :1], b["actions"], b["states"], K)
        wrong = autoregressive(predictor, b["features"][:, :1], b["actions"].roll(1, 1), b["states"], K)
        # roll(1,1) cyclically shifts motion chunks along TIME, not the seven
        # coordinate fields. States remain recorded, so this tests mismatched
        # action conditioning, not total removal of robot-motion information.
        persistence = b["features"][:, :1].expand_as(target)
        record = {"episode": sequence["episode"], "start": sequence["start"]}
        for name, value in (("prediction", predicted), ("persistence", persistence), ("shuffled_action", wrong)):
            # Reduce batch/patch/feature axes, preserving the transition axis K.
            # Index 0 is one transition AFTER the initial observed image.
            record[name + "_l1"] = (value - target).abs().mean((0, 2, 3)).cpu().tolist()
        record["cosine"] = F.cosine_similarity(predicted, target, dim=-1).mean((0, 2)).cpu().tolist()
        records.append(record)
    # Equal episode weight, even if episode lengths yield different window counts.
    means = {}
    ids = sorted({r["episode"] for r in records})
    for key in ("prediction_l1", "persistence_l1", "shuffled_action_l1", "cosine"):
        means[key] = np.mean([np.mean([r[key] for r in records if r["episode"] == episode], axis=0) for episode in ids], axis=0).tolist()
    return {"conditioning": "recorded Cartesian robot states; no future images", "means": means, "windows": records}


@torch.no_grad()
def save_visuals(predictor, sequence: dict, config: Config, device: torch.device, output: Path) -> None:
    # Provided qualitative check: real target frames above per-patch feature L1.
    # Mean over E leaves 256 patch errors, reshaped to their actual 16x16 grid.
    # One shared color scale permits comparisons across time in this figure.
    # There is no pixel decoder: the lower row is not generated robot imagery.
    b = {k: sequence[k].float().unsqueeze(0).to(device) for k in ("features", "actions", "states")}
    predicted = autoregressive(predictor, b["features"][:, :1], b["actions"], b["states"], config.frames - 1)
    error = (predicted - b["features"][:, 1:]).abs().mean(-1)[0].cpu().reshape(-1, 16, 16)
    fig, axes = plt.subplots(2, config.frames - 1, figsize=(2.5 * (config.frames - 1), 5))
    for k in range(config.frames - 1):
        axes[0, k].imshow(sequence["images"][k + 1].permute(1, 2, 0)); axes[0, k].set_title(f"real frame {k+1}")
        axes[1, k].imshow(error[k], vmin=0, vmax=float(error.max()), cmap="magma")
        axes[1, k].set_title("feature L1 error")
        axes[0, k].axis("off"); axes[1, k].axis("off")
    fig.tight_layout(); fig.savefig(output / "real_frames_and_feature_error.png", dpi=150); plt.close(fig)


def run(config: Config) -> None:
    # Provided order: episode split -> frozen feature cache -> train predictor
    # -> validation each epoch -> final held-out test and figures. The encoder
    # is released after caching to leave GPU memory for predictor gradients.
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = split_paths(config)
    sequences = cached_sequences(config, splits, device)
    predictor = build_predictor(config, device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config.lr, weight_decay=0.04)
    # Only predictor weights change, including its state/action embedding layers.
    # No encoder EMA or representation pretraining happens in this exercise.
    loader = DataLoader(FeatureDataset(sequences["train"]), batch_size=config.batch_size, shuffle=True)
    output = Path(config.output)
    history = []
    for epoch in range(config.epochs):
        predictor.train(); losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                # bfloat16 autocast reduces memory for suitable operations. Cached
                # batches stay float32; unlike 09's float16 loop, no scaler is used.
                loss = dynamics_loss(predictor, batch, config)
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step(); losses.append(loss.item())
        validation = evaluate(predictor, sequences["validation"], config, device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": validation["means"]})
        print(history[-1])
        torch.save({"format": "vjepa2-ac-vitl-local-v1", "config": asdict(config),
                    "reference_commit": REFERENCE_COMMIT, "predictor": predictor.state_dict(),
                    "splits": {k: [str(p) for p in v] for k, v in splits.items()}, "epoch": epoch + 1}, output / "predictor.pt")
    test = evaluate(predictor, sequences["test"], config, device)
    (output / "metrics.json").write_text(json.dumps({"history": history, "test": test}, indent=2), encoding="utf-8")
    save_visuals(predictor, sequences["test"][0], config, device, output)
    fig, ax = plt.subplots()
    for key in ("prediction_l1", "persistence_l1", "shuffled_action_l1"):
        ax.plot(range(1, config.frames), test["means"][key], label=key)
    ax.set(xlabel="sampled robot transitions", ylabel="feature L1", title="Held-out state-conditioned prediction")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "rollout.png", dpi=150); plt.close(fig)


def main() -> None:
    # Provided CLI. --episodes-dir selects actual exported .npz files; it is not
    # the raw TFDS builder directory. The export command uses a separate environment.
    parser = argparse.ArgumentParser(description="Train the official AC predictor on real robot features.")
    parser.add_argument("--episodes-dir", default=Config.episodes_dir)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    try:
        run(Config(episodes_dir=args.episodes_dir, epochs=args.epochs))
    except NotImplementedError as error:
        print(f"[TODO] {error}")


if __name__ == "__main__":
    main()
