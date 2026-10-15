"""
EXERCISE 12: V-JEPA VIDEO PRETRAINING WITH THE OFFICIAL MODEL COMPONENTS

WHAT YOU ARE BUILDING
V-JEPA learns visual representations by predicting hidden video features. The
online encoder sees only visible space-time tokens. A predictor receives those
tokens and the positions of missing tokens. A slowly updated target encoder
sees the clean video and supplies the answers. No pixel decoder is trained.
Paper: https://arxiv.org/abs/2404.08471
Reference: https://github.com/facebookresearch/jepa
Recipe: configs/pretrain/vitl16.yaml and app/vjepa/train.py at the revision below.

SETUP (FROM THE REPOSITORY ROOT)
git clone https://github.com/facebookresearch/jepa.git data/reference_jepa
git -C data/reference_jepa checkout 51c59d518fc63c08464af6de585f78ac0c7ed4d5
If the checkout already exists, only the second command is needed. Its source
and license remain in that folder; this exercise imports its actual classes.
Put the real UCF101 videos in data/ucf101/UCF-101 and its official split files in
data/ucf101/ucfTrainTestlist. These are the bootcamp's existing UCF101 locations.
Data source: https://www.crcv.ucf.edu/data/UCF101.php
Normally sample 16 frames spaced four source frames apart. For a video shorter
than that 61-frame span, sample 16 distinct frames evenly across its full length.
This keeps short real clips without repeating frames; their time spacing differs.
Videos with fewer than 16 frames are rejected. Validation sampling is fixed.

HOW THIS FITS
09 used a pretrained image encoder; here you learn how a VIDEO encoder is trained.
13 uses strong pretrained V-JEPA 2 weights for robot dynamics. Your short run
here teaches pretraining; it is not a replacement for those large pretrained weights.

FULL WORKFLOW / WHAT IS KNOWN AT THE START
Real video -> sample/crop a clip -> choose hidden regions -> predict their
representations -> update the encoder/predictor -> slowly update the target.
After training: freeze the encoder -> extract clean clip features -> train a
small labelled classifier -> measure whether those features help recognition.

Unlike 09, the small video encoder starts with random weights. There is no
already-trained teacher supplying correct class labels. The target encoder
starts as a COPY of the online encoder and follows it slowly during learning.
It supplies a feature-learning target, not a guarantee of semantic correctness.
The pretrained ViT-L is loaded only for the final reference comparison.

Object                     Role                              How it changes
-----------------------------------------------------------------------------
UCF101 video               real visual experience             fixed source data
online encoder             describes visible tubelets        optimizer gradients
predictor                  guesses hidden tubelet features   optimizer gradients
target encoder             describes the complete clip       moving average only
context_ids/target_ids     positions used by each branch     sampled for each batch
linear probe               reads a frozen clip description   separate labelled fit

Online means the encoder being directly trained; it does not mean internet
access or live environment interaction. Context means the visible part of the
video. A representation/embedding is a learned feature vector, not RGB values.
The probe is an nn.Linear classifier, not the feature predictor used in pretraining.

THIS IS MASKED VIDEO LEARNING, NOT CAUSAL NEXT-FRAME ROLLOUT
09 predicts future observations from past observations and actions. Here the
online encoder can attend across visible regions throughout the sampled clip,
including later times. It learns to infer hidden content from available context.
There is no action input, actor, reward, causal attention mask or environment
step in this exercise. Do not reuse 09's frame-causal mask here.
13 later uses a pretrained video encoder as part of action-conditioned dynamics.

ONE TRAINING EXAMPLE IN PLAIN WORDS
Imagine a clip of a person swinging a golf club. Hide blocks of the clip's
space/time grid. The online encoder describes the remaining regions. The
predictor receives those descriptions and the POSITIONS it must predict.
Separately, the target encoder sees the complete clip and describes those
hidden positions. Compare the guesses with those target vectors, then update
the online encoder/predictor. No 'golf swing' label enters this update.

SAME METHOD, SMALLER TRAINING BUDGET
Use the reference's actual VisionTransformer, predictor and multi-block masking
classes. Keep 224-pixel clips, 16 frames, 16x16 spatial patches, two-frame tubelets,
3D positions, removed context tokens, learned predictor mask tokens, L1 loss,
and an EMA target. Use both short-range and long-range reference mask settings.
The official ViT-S constructor (384-wide, 12 layers) replaces ViT-L to fit 8 GB.
The predictor is 192-wide with 6 layers instead of 384-wide with 12 layers.
Batch=1 with eight-batch gradient accumulation replaces distributed large batches.
UCF101 subset, short run, linear probe and local schedules replace web-scale
data, long training and the paper's attentive evaluation. The frozen pretrained
ViT-L comparison makes the scale gap visible; it is NOT a matched-capacity study.
There is no added visible-token loss, deep loss, or covariance penalty in this
original V-JEPA recipe. Those would be different methods, not free improvements.

SHAPES / EXAMPLE
clip [B,3,16,224,224] -> 8 x 14 x 14 = 1568 tokens of width 384.
A tubelet is a 2-frame,16x16-pixel block. For a tiny indexing example with four
positions, context IDs [0,3] and target IDs [1,2] mean only positions 0 and 3
enter online attention. Targets 1 and 2 enter the predictor as learned mask
tokens, not clean features. The actual clips always come from real videos.
Context IDs [B,N_visible], target IDs [B,N_hidden] are indices to KEEP in their
respective branches, despite the upstream API calling them masks.

FOLLOW THE TWO BRANCHES
For E=384 encoder features, D=192 predictor features, and N=1568 positions:
  clean clip -> target encoder -> [B,N,E] -> layer norm -> gather target IDs
                                                    -> answers [B,N_hidden,E]
  same clip -> patch embedding -> KEEP context IDs -> online attention
                                                  -> visible [B,N_visible,E]
  visible features + learned placeholders at target IDs -> predictor
                                                       -> guesses [B,N_hidden,E]
  guesses versus answers -> L1 loss -> online/predictor gradients only

The online encoder's patch embedding uses non-overlapping tubelets. It may
compute local embeddings for all tubelets, but removes hidden tokens BEFORE
Transformer attention can mix them into visible tokens. Encoding the full clip
with attention and then slicing would leak hidden image content into context.
Black rectangles in mask_preview.png are for display only. Feeding a blacked-out
full video through attention is not the token-removal method implemented here.

POSITION IDS ARE NOT BOOLEAN ATTENTION MASKS
For four token positions [0,1,2,3], context=[[0,3]] selects TWO visible tokens.
target=[[1,2]] selects TWO answers. Both tensors have dtype torch.long and shape
[B,N_selected], not a [B,N,N] matrix. The position IDs tell the predictor where
each visible/missing token belongs in the original space/time grid.
Unlike DINO in 09, this reference encoder returns no extra class token: keep
all N clean output tokens. Removing index 0 would drop a real video tubelet.

WHY TWO MASK RECIPES?
Recipe 0 uses eight smaller spatial blocks, each roughly 15% of frame area;
recipe 1 uses two larger blocks, each roughly 70%. Blocks can overlap, so these
numbers are not the final total hidden percentages. Both extend across the
full clip's time axis. 'Short-range/long-range' here describes smaller/larger
spatial prediction regions, not prediction horizons in environment steps.
The collator ensures visible context remains and returns one index tensor per
recipe. Within a batch it trims selected IDs to compatible lengths; do not
assume a fixed hidden-token count or that every token must appear in an output.

context_ids=[context_for_recipe_0, context_for_recipe_1]
target_ids =[targets_for_recipe_0, targets_for_recipe_1]
Different recipes can have different N_visible/N_hidden, so predictions and
targets are LISTS of tensors, not a single stacked mask axis. Process one recipe
per predictor call, passing one-element index lists to its reference API.

LOSS AND UPDATE
target = layer_norm(target_encoder(clean_video)), with no gradients.
loss = mean(abs(prediction - target_at_hidden_positions)).
Average this over the two mask recipes. Layer normalization works across the
feature width; it is different from forcing vectors to unit length.
After a successful optimizer step:
target_weight = momentum*target_weight + (1-momentum)*online_weight.
Momentum rises from 0.998 toward 1. Higher means the target changes more slowly.

WHY THE TARGET IS FROZEN BUT STILL CHANGES
Frozen here means no gradients and no optimizer step. EMA means exponential
moving average: manually blend each target parameter toward the latest online
parameter. If target=2, online=4 and momentum=0.9, new target is 2.2.
The target changes slowly while the online network learns from its targets.
Using target=online would make them the same object, defeating this design.
Freezing the target and using torch.no_grad() on its forward pass are explicit
protections; they do not disable learning through the online/predictor branch.

NORMALIZATION AND LOSS ARE DIFFERENT OPERATIONS
F.layer_norm(clean,(E,)) centres/scales each target token across its E features.
It does not average across videos or make every vector have length one. The
predictor already has its own internal normalization; do not add another
normalization to its final predictions to make matching easier.
L1 measures absolute feature error, not squared error or classification error.
Compute one mean L1 per recipe, then average those two scalars equally. A recipe
with more hidden tokens does not get a larger weight merely for that reason.

ONE OPTIMIZER UPDATE, IN ORDER
1. The DataLoader gives a real clip batch plus context/target index lists.
2. Target branch encodes the full clip without gradients and supplies answers.
3. For each recipe, online encodes visible tokens and predictor guesses targets.
4. Compute mean L1 over recipes and backpropagate through online/predictor.
5. Accumulate gradients across eight batches by default, then clip and step AdamW.
6. Update the target ONCE with EMA after that successful optimizer step.
7. Repeat. At epoch end, evaluate fixed validation clips/masks without updates.

Accumulation lets eight one-clip batches contribute to one update without
keeping eight clips' graphs in memory together. Divide each loss by the actual
number of batches in its accumulation group. If the last group has three
batches, divide by three, not eight. The provided runner handles that tail.
It uses bfloat16 autocast on CUDA; unlike 09's float16 path it has no GradScaler.
Non-finite gradients raise an error before optimizer/target updates.

WHY WE TRAIN A SECOND, MUCH SMALLER CLASSIFIER
Low feature-prediction loss alone is not proof of useful representations. If
every video received the same feature vector, matching could be easy but action
recognition would fail. This loss has no explicit class-separation term and the
EMA design does not justify assuming collapse is impossible; measure usefulness.

After pretraining, read each clean video's encoder tokens and average them into
one vector [E]. Freeze those vectors and train a fresh nn.Linear(E,classes) probe
using the real UCF labels. Cross entropy teaches ONLY that classifier. The
encoder does not receive labels or change during the probe stage. This tests
whether useful class information is easy to read from the learned features.
Run the same 100-step probe recipe on random, trained and pretrained encoders.
The trained/random pair has matched architecture; the larger pretrained model
is a scale reference, not evidence that the small pretraining procedure failed.

WHAT YOU IMPLEMENT / LEARNING ORDER
1. build_models: construct the reference encoder, predictor and frozen target.
2. masked_forward: connect both branches without leaking hidden pixels.
3. jepa_loss: compute the recipe's hidden-feature L1 loss.
4. update_target: update the target after a successful optimizer step.
5. fit_probe: train only a linear classifier on frozen features.
Downloading, real-video decoding, consistent clip crops, grouped data splits,
optimizer bookkeeping, caching probe features, figures and checkpoints are provided.
The first stub is model construction, before any large download or training.

METRICS / DEFINITION OF DONE
- Held-out hidden-feature L1 falls; lower is better but is not proof of usefulness.
- Clean target features vary across videos; mean feature std near zero warns of collapse.
- Frozen probe accuracy exceeds chance (1/number_of_classes); higher is better.
- Compare a random encoder, your trained encoder and pretrained V-JEPA ViT-L.
- Train/validation are split by UCF source group; official fold 1 supplies test.
- Save model/settings/splits, training curves, mask preview and probe results.
- Explain why a short local pretraining run can lose to pretrained weights.
Run: python -m exercises.world_models.12_vjepa_video_pretraining --epochs 20
--no-compare-pretrained skips the large reference download for debugging only.

HOW TO READ THE SAVED RESULTS
Accuracy is a fraction: 0.6 means 60% of clips classified correctly. For ten
roughly balanced classes, chance is about 0.1. Validation/test labels are used
only for probe measurement, not its gradient updates. This fixed recipe does
not select the best probe epoch; use validation, never test, if experimenting.
feature_std averages feature-wise variation across validation clips. Near zero
warns of nearly constant pooled features, but scale differs across encoders:
it is a diagnostic, not a model-quality ranking by itself.
The target representation changes during training, so falling L1 is not an
absolute score on a permanently fixed task. Read it alongside the frozen probe.

manifest.json records classes and split videos. mask_preview.png shows one
recipe on a real clip. vjepa.pt stores the trained models/config/optimizer;
random_encoder.pt preserves the untrained target for the baseline. metrics.json
contains losses and probe results; training.png plots the loss curves.
The script saves checkpoints but has no resume option. Complete each stub and
its small checks before a full run. Keep its original NotImplementedError line
commented when solving it. Run each reference-based exercise in its own Python
process: the upstream projects reuse the package name src.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.video_world_model_data import (ROOT, JEPA_COMMIT, VideoDataset, manifest,
                                           reference_import, ucf_records)


@dataclass
class Config:
    # frames/stride describe sampled source frames; patch/tubelet describe model
    # tokenization AFTER sampling. 16 sampled frames spaced four source frames
    # apart still make only eight two-frame tubelets, not 32 temporal tokens.
    # classes/train_per_class limit real data; labels do not enter JEPA loss.
    # accumulation counts microbatches per optimizer update, not masking recipes.
    reference: str = str(ROOT / "data/reference_jepa")
    videos: str = str(ROOT / "data/ucf101/UCF-101")
    annotations: str = str(ROOT / "data/ucf101/ucfTrainTestlist")
    output: str = str(ROOT / "outputs/vjepa_pretraining")
    image_size: int = 224
    frames: int = 16
    stride: int = 4
    patch: int = 16
    tubelet: int = 2
    classes: int = 10
    train_per_class: int = 12
    batch_size: int = 1
    accumulation: int = 8
    epochs: int = 20
    lr: float = 3e-4
    seed: int = 42


def reference_modules(config: Config):
    # Provided: verify the pinned git revision, then return actual upstream
    # module objects. The shared loader also rejects mixing two different
    # reference repositories called src in one Python process.
    root = Path(config.reference)
    vision = reference_import(root, JEPA_COMMIT, "src.models.vision_transformer")
    predictor = reference_import(root, JEPA_COMMIT, "src.models.predictor")
    masking = reference_import(root, JEPA_COMMIT, "src.masks.multiblock3d")
    return vision, predictor, masking


def build_models(config: Config, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module]:
    # Where this fits:
    # run calls this before loading videos or creating an optimizer. Construct
    # the actual reference modules that masked_forward will connect later.
    #
    # Goal:
    # Create a trainable online video encoder, a trainable predictor, and a
    # separate initially-identical target encoder that learns ONLY through EMA.
    #
    # Input:
    # config supplies image/clip/tubelet sizes and pinned reference checkout.
    # device is torch.device('cuda') or torch.device('cpu'). Nothing here needs
    # labels, image examples or a pretrained checkpoint download.
    #
    # Output:
    # (online,target,predictor), three nn.Module objects on device, in that order.
    # online/target output width is 384; predictor uses width 192 internally
    # but projects its predictions back to 384 to match the target features.
    #
    # Method / exact constructors:
    # vision,prediction,_ = reference_modules(config) returns imported modules;
    # these constructors belong to those modules, not torch.nn or timm.
    # vision.vit_small(img_size=config.image_size,patch_size=config.patch,
    #                  num_frames=config.frames,tubelet_size=config.tubelet,
    #                  uniform_power=True)
    # prediction.VisionTransformerPredictor(
    #     img_size=config.image_size,patch_size=config.patch,
    #     num_frames=config.frames,tubelet_size=config.tubelet,
    #     embed_dim=384,predictor_embed_dim=192,depth=6,num_heads=6,
    #     uniform_power=True,use_mask_tokens=True,num_mask_tokens=2,
    #     zero_init_mask_tokens=True,norm_layer=partial(nn.LayerNorm,eps=1e-6))
    #
    # Constructor meanings:
    # - img_size=224 and patch_size=16 give a 14x14 spatial token grid.
    # - num_frames=16 and tubelet_size=2 give eight temporal token positions.
    #   Both networks must use the SAME grid to interpret mask position IDs.
    # - vit_small fixes encoder width=384, depth=12 and attention heads=6.
    # - embed_dim=384 matches that encoder. predictor_embed_dim=192 makes the
    #   prediction network narrower; depth=6 stacks six attention blocks, and
    #   num_heads=6 splits its internal attention width into six groups.
    # - uniform_power=True assigns equal positional-encoding width to time,
    #   vertical position and horizontal position for these chosen dimensions.
    # - use_mask_tokens=True supplies learned placeholders at hidden positions.
    #   False selects a different reference path that reads noisy target features.
    # - num_mask_tokens=2 creates two learned placeholder vectors, one per recipe,
    #   not one vector for each hidden position. Position encodings distinguish
    #   positions sharing the same placeholder. zero_init_mask_tokens starts
    #   those parameters at zero; it does NOT freeze them.
    # - functools.partial(nn.LayerNorm,eps=1e-6) supplies a constructor with eps
    #   preset. The reference later passes each layer's width to that constructor.
    #   eps is a small stability value used when dividing by feature spread.
    #
    # Steps:
    # 1. Load reference_modules and construct online using vit_small above.
    # 2. Construct predictor with the complete constructor above. Do not invent
    #    a smaller MLP replacement or use the defaults for use_mask_tokens.
    # 3. Make target=copy.deepcopy(online), using the imported copy module.
    #    target=online would alias the same object; shallow copy is wrong too.
    # 4. Call target.eval() and target.requires_grad_(False). eval controls
    #    layer behavior; requires_grad_ prevents target parameter gradients.
    #    Online/predictor remain trainable (fixed positional parameters are
    #    intentionally non-trainable in these upstream modules).
    # 5. Move all three modules with .to(device) and return (online,target,predictor).
    #    The provided run function owns the optimizer and later mode changes.
    #
    # Checks:
    # - target is not online. Corresponding parameters start torch.equal but
    #   have different .data_ptr() addresses, so updating one cannot alias the other.
    # - Every target parameter has requires_grad=False; predictor mask tokens
    #   and the online encoder's learned weights still require gradients.
    # - predictor maps visible width 384 -> internal 192 -> output 384.
    raise NotImplementedError("Build the official V-JEPA model components.")


def make_mask_collator(config: Config):
    # Provided DataLoader adapter: collator receives a Python list of dataset
    # items [(clip_tensor,label), ...]. It stacks clips/labels and builds indices.
    # It returns ((clips,labels),context_ids,target_ids), not just a tensor batch.
    # clips is [B,3,T,H,W]; labels [B]; both index outputs are lists of two tensors.
    # aspect_ratio controls block height/width; spatial_scale is area fraction
    # per block; temporal_scale=(1,1) spans all tubelet times. num_blocks sets
    # how many blocks are combined. max_temporal_keep=1 allows context across
    # the full clip, not just past frames. No causal time mask is constructed.
    _, _, module = reference_modules(config)
    recipes = [dict(aspect_ratio=(0.75, 1.5), spatial_scale=(scale, scale),
                    temporal_scale=(1.0, 1.0), num_blocks=blocks, max_temporal_keep=1.0)
               for scale, blocks in ((0.15, 8), (0.7, 2))]
    return module.MaskCollator(cfgs_mask=recipes, crop_size=config.image_size,
                               num_frames=config.frames, patch_size=config.patch,
                               tubelet_size=config.tubelet)


def masked_forward(online: nn.Module, target: nn.Module, predictor: nn.Module, clips: Tensor,
                   context_ids: list[Tensor], target_ids: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    # Where this fits:
    # The DataLoader has made one real clip batch and two masking recipes.
    # This helper connects the target branch and the trainable prediction branch.
    # jepa_loss will compare the two returned lists, one entry per recipe.
    #
    # Goal:
    # Predict each recipe's hidden feature vectors without leaking those hidden
    # pixels into online attention or backpropagating through the target branch.
    #
    # Input:
    # - clips: normalized RGB float [B,3,T,H,W], default [B,3,16,224,224].
    # - context_ids: two torch.long tensors [B,N_visible_i].
    # - target_ids: two torch.long tensors [B,N_hidden_i].
    # i indexes the mask recipe, not an example in the batch. Lengths can differ
    # between recipes. IDs refer to the original N=1568 token grid and are already
    # on clips.device. The runner sets train/eval modes and the autocast context.
    #
    # Output:
    # (predictions,targets), two lists of length 2. Corresponding elements have
    # identical shape [B,N_hidden_i,384]. Prediction tensors keep their gradient
    # paths during training; every target tensor has requires_grad=False.
    #
    # Method:
    # target(clips) returns a Tensor [B,N,384], not a Hugging Face output object.
    # F.layer_norm(clean,(clean.shape[-1],)) normalizes each complete token vector.
    # clean.gather(dim=1,index=gather_ids) selects target positions independently
    # for each batch example. gather_ids must include the FEATURE axis too.
    # online(clips,masks=[context]) returns [B,N_visible_i,384] directly.
    # predictor(visible,hidden,[context],[ids],mask_index=i) returns hidden
    # predictions [B,N_hidden_i,384]. Each index-list argument has ONE entry here.
    #
    # Why does predictor accept hidden answers as an argument?
    # The reference forward signature supports two methods. With the required
    # use_mask_tokens=True, it ignores the hidden tensor's CONTENT and inserts
    # learned mask placeholders instead. With that flag False it reads noisy
    # target features, a different method. Keep the explicit constructor setting.
    # Do not concatenate hidden target vectors with visible online vectors yourself.
    #
    # Steps:
    # 1. Check the two index lists are paired and both contain two recipes, so
    #    zip cannot silently discard an extra recipe. Start empty output lists.
    # 2. Inside torch.no_grad(), run clean=target(clips), then normalize clean
    #    with F.layer_norm(clean,(clean.shape[-1],)). Encode the complete video
    #    ONCE, since both recipes ask for targets from this same full clip.
    # 3. Leave that no_grad block before doing online/predictor work. Iterate
    #    enumerate(zip(context_ids,target_ids)) to obtain i,context,ids.
    # 4. Build gather_ids=ids.unsqueeze(-1).expand(-1,-1,clean.shape[-1]):
    #    [B,N_hidden] -> [B,N_hidden,1] -> [B,N_hidden,384]. -1 keeps an axis size.
    #    Gather hidden=clean.gather(dim=1,index=gather_ids). Repeating the ID over
    #    features copies the whole selected token, not just one feature entry.
    # 5. Run visible=online(clips,masks=[context]). The brackets are important:
    #    they adapt one recipe to the reference API, not a boolean attention mask.
    #    Do not run online(clips) and slice AFTER full attention; that leaks targets.
    # 6. Run predictor(visible,hidden,[context],[ids],mask_index=i). It adds position
    #    encodings, inserts placeholders, and returns only requested hidden outputs.
    # 7. Append this prediction and hidden to their respective lists in the same
    #    order. Do not detach predictions or stack differently sized recipes.
    # 8. Return (predictions,targets). Do not update EMA or the optimizer here;
    #    this helper is also called for validation, which must not change weights.
    #
    # Tiny gather check (illustrative token values, not exercise training data):
    # clean=[[[10,11],[20,21],[30,31],[40,41]]], ids=[[1,2]] gives
    # hidden=[[[20,21],[30,31]]]. The feature axis remains length 2.
    #
    # Checks:
    # - Every prediction shape matches its target; targets have no gradient path.
    # - Under eval mode, with fixed IDs and already-preprocessed clips, altering
    #   ONLY hidden tubelets leaves online visible outputs unchanged. Target
    #   outputs may change, because the target intentionally sees the full clip.
    # - During training, jepa_loss can backpropagate to online and predictor.
    raise NotImplementedError("Connect visible context to hidden target-feature prediction.")


def jepa_loss(predictions: list[Tensor], targets: list[Tensor]) -> Tensor:
    # Where this fits:
    # masked_forward has made aligned hidden predictions/answers for each recipe.
    # This helper turns their differences into the scalar training objective.
    #
    # Goal:
    # Average absolute feature errors within each recipe, then weight the two
    # recipes equally. Preserve gradients through predictions, not targets.
    #
    # Input:
    # Equal nonempty lists; pair i has shape [B,N_hidden_i,E] on the same device.
    # E=384 here; N_hidden can differ between pairs. Targets were already
    # layer-normalized by masked_forward. Predictions are its unmodified outputs.
    #
    # Output:
    # One scalar Tensor (shape []), attached to the predictions' training graph.
    # The runner divides by accumulation group size separately before backward.
    #
    # Formula:
    # recipe_loss_i = mean(abs(prediction_i - detached_target_i))
    # total_loss = (recipe_loss_0 + recipe_loss_1) / 2
    # Mean covers batch, hidden tokens and feature entries WITHIN each recipe.
    # Example: errors [1,3] in recipe 0 give 2; four errors [1,1,1,1] in recipe
    # 1 give 1; total=(2+1)/2=1.5. Concatenating all six errors gives 8/6,
    # which incorrectly weights the larger recipe more heavily.
    #
    # Method:
    # F.l1_loss(prediction,target.detach(),reduction='mean') is from
    # torch.nn.functional. It uses absolute differences, not F.mse_loss's squares.
    # torch.stack(losses).mean() averages scalar tensors while retaining gradients.
    # torch.tensor(losses) would construct a new tensor and lose those graph paths.
    #
    # Steps:
    # 1. Reject empty or unequal list lengths with ValueError. For each pair,
    #    require matching three-dimensional, nonempty shapes; do not let loss
    #    broadcasting hide a wrong gather or feature dimension.
    # 2. Compute the mean L1 for each pair with the detached target.
    # 3. Stack those scalar losses, average and return the Tensor. Do not call
    #    .item(), divide by accumulation count, or detach the predictions here.
    # 4. Add no visible-token loss, RGB reconstruction, covariance penalty or
    #    final prediction normalization; those would alter this recipe.
    #
    # Checks:
    # - Equal predictions/targets give zero; the unequal-size example gives 1.5.
    # - Backward reaches both recipe predictions. Separate leaf targets with
    #   requires_grad=True should still receive no gradient from this helper.
    raise NotImplementedError("Compute the two-mask V-JEPA L1 objective.")


@torch.no_grad()
def update_target(online: nn.Module, target: nn.Module, momentum: float) -> None:
    # Where this fits:
    # run calls this ONCE after each optimizer update, after any accumulated
    # batches have contributed gradients. It never runs during validation/probing.
    #
    # Goal:
    # Move the target encoder slowly toward the newly updated online encoder.
    # Do not update the predictor here, and never copy target back into online.
    #
    # Input:
    # online and target have the same architecture and parameter ordering;
    # target was made with deepcopy. momentum is a finite number in [0,1].
    # The runner schedules it from 0.998 toward 1 over OPTIMIZER updates.
    #
    # Output:
    # None. Target parameters change IN PLACE; online parameters stay unchanged.
    # Target stays frozen and in eval mode. This is not an optimizer step.
    #
    # Formula:
    # new_target = momentum * old_target + (1-momentum) * current_online
    # For old_target=2, online=4, momentum=0.9: 1.8+0.4=2.2.
    # At momentum=0.998, only 0.2% of the new value comes from current online.
    # A larger momentum means a SLOWER target, not a larger learning rate.
    #
    # Method:
    # Iterate zip(online.parameters(),target.parameters()) as (source,dest).
    # dest.mul_(momentum).add_(source,alpha=1-momentum) implements the blend.
    # Underscores mean in-place tensor updates; add_'s alpha scales source.
    # The provided @torch.no_grad() makes these manual updates outside autograd.
    # Do not use dest = expression alone: that only rebinds a local Python name.
    # Do not use target.load_state_dict(online.state_dict()) for every update;
    # that is a hard copy, equivalent to momentum=0, not the requested EMA.
    #
    # Steps:
    # 1. Reject values outside [0,1] (including NaN) with ValueError, for example
    #    by checking not (0.0 <= momentum <= 1.0).
    # 2. Iterate matching parameters and apply the in-place blend above. The
    #    architecture/deepcopy guarantee matching order and shapes; do not pair
    #    online parameters with predictor parameters by mistake.
    # 3. Leave target eval/frozen; return None. Do not add optimizer.step or
    #    backward. These reference encoders have no BatchNorm running statistics
    #    to update separately; do not assume every other model shares that property.
    #
    # Checks:
    # - momentum=0 copies parameter values; momentum=1 leaves target unchanged.
    # - A scalar example gives 2.2, and online remains 4.
    # - Target has no .grad tensors or shared parameter storage with online.
    # - Do not call per microbatch: doing eight EMA blends per accumulated update
    #   would change the intended target-update rate.
    raise NotImplementedError("Update the target encoder through EMA.")


def fit_probe(train_x: Tensor, train_y: Tensor, val_x: Tensor, val_y: Tensor,
              test_x: Tensor, test_y: Tensor, classes: int, device: torch.device) -> dict[str, float]:
    # Where this fits:
    # evaluate_representation already ran a FROZEN encoder on clean clips and
    # cached one mean-pooled feature vector per video. Only now do class labels
    # become learning targets, for a fresh small classifier called a linear probe.
    #
    # Goal:
    # Measure whether action classes are easy to separate using frozen features.
    # Train the classifier on train only, then measure validation and test once.
    # This is supervised probe training, not fine-tuning the video encoder.
    #
    # Input:
    # train_x [N_train,D], val_x [N_val,D], test_x [N_test,D]: CPU feature tensors.
    # Corresponding train_y/val_y/test_y: [N_split] integer labels in 0..classes-1.
    # D=384 for random/trained ViT-S and 1024 for the pretrained ViT-L reference.
    # classes=10 by default. The encoder itself is not an argument to this helper.
    #
    # Output:
    # {'validation_accuracy': Python float, 'test_accuracy': Python float}.
    # Accuracy = correct clips / total clips, in [0,1]. Do not return percentages
    # or losses under these names. Example: 24/40 correct -> accuracy 0.6.
    #
    # Method:
    # nn.Linear(train_x.shape[1],classes).to(device) creates a new classifier:
    # input [N,D] -> logits [N,classes]. It has no hidden layer or encoder update.
    # torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=0.01) optimizes
    # only its weights/bias. lr is update size; weight_decay discourages large weights.
    # F.cross_entropy(logits,train_y) expects raw scores and integer class IDs;
    # it handles log-softmax internally. Do not pass softmax probabilities or
    # one-hot labels. logits.argmax(dim=-1) returns predicted class IDs [N].
    #
    # Steps:
    # 1. Check each feature matrix has two axes, matching feature width D and
    #    as many rows as its nonempty label vector. No split should be empty.
    # 2. Detach feature tensors and move them as float32, for example
    #    train_x=train_x.detach().to(device=device,dtype=torch.float32).
    #    Do the same for val_x/test_x. Move all labels using
    #    train_y=train_y.to(device=device,dtype=torch.long), and likewise for others.
    # 3. Use torch.manual_seed(42) for a fixed probe initialization recipe. Build
    #    a FRESH nn.Linear and AdamW with the calls above for each encoder run.
    #    Do not reuse a classifier already fitted to another feature space.
    # 4. Call probe.train(). For 100 full-batch steps, call
    #    optimizer.zero_grad(set_to_none=True), logits=probe(train_x),
    #    loss=F.cross_entropy(logits,train_y), loss.backward(), optimizer.step().
    #    Full batch means all cached training vectors per step; no videos or
    #    encoder graphs are held in GPU memory. This is small for the chosen subset.
    # 5. Call probe.eval(). Inside torch.no_grad(), compute val_predictions=
    #    probe(val_x).argmax(dim=-1). Compare with val_y, convert the boolean
    #    vector using .float(), average using .mean(), and use .item() for JSON.
    # 6. Repeat the same measurement on test_x/test_y, without any gradient step.
    #    This fixed 100-step recipe does not pick an epoch based on validation.
    #    If you later tune hyperparameters, use validation and reserve test for
    #    the final evaluation; do not optimize settings against test accuracy.
    # 7. Return the two floats under the exact output keys above.
    #
    # Checks:
    # - Only probe parameters belong to its optimizer; input features stay fixed.
    # - logits shape is [N,classes], predictions and labels both [N].
    # - Chance for ten balanced classes is about 0.1; weak results warrant checking
    #   label alignment, class balance and pretraining before blaming the probe.
    # - Use the identical probe budget/optimizer for all three encoders. The
    #   pretrained encoder is larger, so that comparison still includes a scale gap.
    raise NotImplementedError("Fit and evaluate a frozen-feature linear probe.")


@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    # Provided probe-data stage. Clean unmasked encoder output is [B,N,D], with
    # no class token. mean(1) pools all space/time tokens -> one [B,D] clip vector.
    # It does not average the batch axis. CPU outputs [all_clips,D] and labels
    # [all_clips] stay in matching loader order. This pooling is only for probing;
    # masked pretraining compares separate hidden tokens without mean pooling.
    encoder.eval()
    features, labels = [], []
    for clips, y in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens = encoder(clips.to(device))
        features.append(tokens.float().mean(1).cpu()); labels.append(y)
    return torch.cat(features), torch.cat(labels)


def evaluate_representation(encoder: nn.Module, loaders: dict[str, DataLoader], classes: int,
                            device: torch.device) -> dict[str, float]:
    # Provided: extract each split with the SAME frozen encoder, then fit a
    # separate linear probe. feature_std takes std across validation CLIPS for
    # each feature coordinate, then averages coordinates. It does not measure
    # variation across tokens inside one video.
    data = {name: extract_features(encoder, loader, device) for name, loader in loaders.items()}
    results = fit_probe(*data["train"], *data["validation"], *data["test"], classes, device)
    results["feature_std"] = data["validation"][0].std(0, unbiased=False).mean().item()
    return results


@torch.no_grad()
@torch.random.fork_rng(devices=[])
def validation_loss(online, target, predictor, dataset: VideoDataset, config: Config, device: torch.device) -> float:
    # Provided measurement only: fixed clip crops and repeatable mask draws make
    # epoch comparisons less noisy. No optimizer or EMA update occurs here.
    # The outer decorator disables gradients; fork_rng preserves CPU RNG state.
    online.eval(); predictor.eval(); target.eval()
    # Mask positions also use the global CPU generator. Reset it temporarily;
    # fork_rng restores training's random sequence when validation finishes.
    torch.random.default_generator.manual_seed(config.seed)
    collator = make_mask_collator(config)
    loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collator)
    total, count = 0.0, 0
    for (clips, _), context, hidden in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p, t = masked_forward(online, target, predictor, clips.to(device),
                                   [x.to(device) for x in context], [x.to(device) for x in hidden])
            loss = jepa_loss(p, t)
        total += loss.item() * len(clips); count += len(clips)
    return total / count


def save_mask_preview(dataset: VideoDataset, collator, config: Config, output: Path) -> None:
    # Provided visual check for recipe 0. Reverse image normalization, expand
    # hidden tubelet IDs to their pixel cells, and black them out FOR DISPLAY.
    # Training removes tokens before attention instead of feeding this preview.
    # Rows show the clean clip and its hidden regions at four sampled times.
    (clips, _), _, targets = collator([dataset[0]])
    clip = clips[0].permute(1, 0, 2, 3)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixels = (clip * std + mean).clamp(0, 1)
    grid = config.image_size // config.patch
    mask = torch.zeros(config.frames // config.tubelet * grid * grid, dtype=torch.bool)
    mask[targets[0][0]] = True
    mask = mask.reshape(config.frames // config.tubelet, grid, grid)
    mask = mask.repeat_interleave(config.tubelet, 0).repeat_interleave(config.patch, 1).repeat_interleave(config.patch, 2)
    preview = pixels.masked_fill(mask.unsqueeze(1), 0)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for column, frame in enumerate(torch.linspace(0, config.frames - 1, 4).long()):
        axes[0, column].imshow(pixels[frame].permute(1, 2, 0)); axes[1, column].imshow(preview[frame].permute(1, 2, 0))
        axes[0, column].set_title(f"frame {frame.item()}")
        axes[0, column].axis("off"); axes[1, column].axis("off")
    fig.suptitle("Real clip and hidden tubelets (display only)")
    fig.tight_layout(); fig.savefig(output / "mask_preview.png", dpi=150); plt.close(fig)


def run(config: Config, compare_pretrained: bool = True) -> None:
    # Provided orchestration: the student functions define model construction,
    # gradient paths, objective, EMA and probe fitting. This loop owns their order.
    random.seed(config.seed); torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online, target, predictor = build_models(config, device)
    splits, names = ucf_records(Path(config.videos), Path(config.annotations), config.classes, config.train_per_class, config.seed)
    # UCF clips with the same class/source group stay together for train/validation;
    # official fold 1 defines test. Default selected sizes: up to 120/40/40 clips.
    # Probe loaders use fixed centre crops; JEPA train_data below uses random but
    # temporally consistent crops/flips across every frame of each clip.
    datasets = {name: VideoDataset(records, config.frames, config.stride, config.image_size) for name, records in splits.items()}
    probe_loaders = {name: DataLoader(data, batch_size=1) for name, data in datasets.items()}
    output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({"classes": names, "splits": manifest(splits)}, indent=2), encoding="utf-8")
    # Baseline before any self-supervised update.
    torch.save(target.state_dict(), output / "random_encoder.pt")
    train_data = VideoDataset(splits["train"], config.frames, config.stride, config.image_size, training=True)
    collator = make_mask_collator(config)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    save_mask_preview(datasets["validation"], make_mask_collator(config), config, output)
    optimizer = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=config.lr, weight_decay=0.04)
    # Target parameters are deliberately absent. AdamW sees online + predictor;
    # EMA sees online + target. Do not swap those two pairs.
    total_updates = config.epochs * math.ceil(len(loader) / config.accumulation)
    update, history = 0, []
    for epoch in range(config.epochs):
        online.train(); predictor.train(); target.eval()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, ((clips, _), context, hidden) in enumerate(loader):
            # _ discards labels: the pretraining target comes from video features.
            # Divide by the actual number of microbatches in THIS update group,
            # including a shorter group at the end of the epoch.
            group_start = step // config.accumulation * config.accumulation
            group_size = min(config.accumulation, len(loader) - group_start)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                p, t = masked_forward(online, target, predictor, clips.to(device),
                                       [x.to(device) for x in context], [x.to(device) for x in hidden])
                loss = jepa_loss(p, t)
            (loss / group_size).backward(); losses.append(loss.item())
            if (step + 1) % config.accumulation == 0 or step + 1 == len(loader):
                # Schedules use optimizer-update count, not clip count. Warmup
                # ramps learning rate before cosine decay; weight decay grows
                # from 0.04 to 0.4. These are compact local training schedules.
                warmup = max(1, total_updates // 10)
                progress = max(0, update - warmup) / max(1, total_updates - warmup)
                lr = config.lr * min(1, (update + 1) / warmup) * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)
                for group in optimizer.param_groups:
                    group["lr"] = lr; group["weight_decay"] = 0.04 + 0.36 * update / max(1, total_updates - 1)
                nn.utils.clip_grad_norm_(list(online.parameters()) + list(predictor.parameters()), 10.0, error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                update_target(online, target, 0.998 + 0.002 * update / max(1, total_updates - 1))
                # Clip-grad raises on non-finite gradients before optimizer.step;
                # EMA is reached only after the optimizer update succeeds.
                update += 1
        val = validation_loss(online, target, predictor, datasets["validation"], config, device)
        history.append({"epoch": epoch + 1, "train_l1": sum(losses) / len(losses), "validation_l1": val})
        print(history[-1])
        torch.save({"config": asdict(config), "reference_commit": JEPA_COMMIT,
                    "online": online.state_dict(), "target": target.state_dict(),
                    "predictor": predictor.state_dict(), "optimizer": optimizer.state_dict(),
                    "splits": manifest(splits), "classes": names, "epoch": epoch + 1}, output / "vjepa.pt")
    del online, predictor, optimizer
    results = {"trained": evaluate_representation(target, probe_loaders, config.classes, device)}
    # Evaluate the trained EMA target, then restore its saved INITIAL weights
    # to evaluate a matched-architecture random baseline with a fresh probe.
    # The trained checkpoint is already saved; restoring here does not overwrite it.
    target.load_state_dict(torch.load(output / "random_encoder.pt", map_location=device, weights_only=True))
    results["random"] = evaluate_representation(target, probe_loaders, config.classes, device)
    del target
    if compare_pretrained:
        # Separate reference: larger pretrained ViT-L, loaded strictly so a
        # mismatched checkpoint fails rather than silently leaving random layers.
        vision, _, _ = reference_modules(config)
        model = vision.vit_large(img_size=224, patch_size=16, num_frames=16, tubelet_size=2, uniform_power=True)
        weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar", map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in weights["target_encoder"].items()}
        model.load_state_dict(state, strict=True); del weights, state
        results["pretrained_vitl"] = evaluate_representation(model.to(device).requires_grad_(False), probe_loaders, config.classes, device)
    (output / "metrics.json").write_text(json.dumps({"history": history, "probes": results}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots()
    for key in ("train_l1", "validation_l1"):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
    ax.set(xlabel="epoch", ylabel="hidden-feature L1", title="V-JEPA learning"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "training.png", dpi=150); plt.close(fig)


def main() -> None:
    # Provided CLI. --no-compare-pretrained only skips the final large download;
    # it does not replace real UCF data or make a tiny run a paper reproduction.
    parser = argparse.ArgumentParser(description="Train with the official V-JEPA components.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--compare-pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    try:
        run(Config(epochs=args.epochs), args.compare_pretrained)
    except NotImplementedError as error:
        print(f"[TODO] {error}")


if __name__ == "__main__":
    main()
