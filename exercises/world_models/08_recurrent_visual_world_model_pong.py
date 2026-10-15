"""
EXERCISE 08: A SMALL ONLINE DREAMERV3 AGENT FOR ATARI PONG
==========================================================

WHAT YOU ARE BUILDING

You are building a small DreamerV3-style agent that learns from the real Atari
Pong environment. The agent sees only 64 x 64 RGB images. It cannot read the
emulator's hidden ball or paddle positions. It chooses one of six discrete
actions.

Dreamer repeats this loop:

```text
actor uses the real environment
-> new experience enters the replay buffer
-> world model learns from replay
-> actor and critic train in the updated world model
-> repeat
```

The run first collects random actions so replay is not empty. After that, the
current actor collects new experience during every round.

MAIN OBJECTS

```text
Pong environment       the real game and source of true rewards
PongReplayBuffer       stores real images, actions, rewards, and endings
TinyDreamerWorldModel  learns a compact state and predicts what happens next
TinyDreamerActor       chooses actions from the configured action distribution
TinyDreamerCritic      estimates future reward
```

ONE REPLAY BATCH

Dreamer stores a posterior RSSM state beside each real image in replay. It is a
posterior state because the model made it after seeing that real image. When
training samples a sequence from the middle of a game, the stored posterior
state provides its starting GRU memory and latent state. The model does not
need ignored warm-up steps.

```text
T = number of transitions that contribute training loss

images:                [B, T + 1, 3, 64, 64]
actions:               [B, T, 6]
rewards:               [B, T, 1]
continues:             [B, T, 1]
initial_deterministic: [B, deterministic_dim]       h from stored posterior state
initial_stochastic:    [B, variables, classes]      z from stored posterior state
initial_probabilities: [B, variables, classes]      posterior probabilities for z
```

There is one more image than action because every transition ends at another
image:

```text
image_0 --action_0--> image_1
image_1 --action_1--> image_2
```

THE RSSM

The recurrent state-space model, or RSSM, stores two kinds of state:

```text
h_t  deterministic GRU memory
z_t  sampled categorical latent state
```

For every real transition:

```text
previous h and z + action -> new GRU memory h_t
h_t                       -> prior prediction z_hat_t
h_t + real next image     -> posterior state z_t
```

The prior is the belief before seeing the new image. The posterior is the
updated belief after seeing it. Training makes the prior similar to the
posterior, teaching the model to predict what the real image would reveal.

The world model reconstructs images and predicts rewards and continuation. The
actor and critic then practise on futures produced by the prior alone.

BEHAVIOUR LEARNING

```text
real replay sequence -> posterior states
actor chooses action -> RSSM imagines next state
world model          -> predicted reward and continuation probability
critic               -> value prediction
lambda return        -> critic training target and actor learning signal
```

===============================================================================
FROM REAL REPLAY TO VALUE LEARNING
===============================================================================

The meaning of value is easier to understand after following the data from the
real game into an imagined future.

First, the agent collects real Pong experience. Replay stores the images the
agent saw, the actions it took, the rewards returned by Pong, whether each
transition continued, and posterior RSSM states inferred from the real images.
A posterior state is the world model's compact description of the game after
it has seen the matching real image... z_t = q(z_t | h_t, x_t).

The default training batch contains 16 replay sequences, each with 64
transitions/steps:

```text
B = 16 sequences
T = 64 transitions in each sequence
B * T = 1,024 real replay positions
```

The world model first trains on these real sequences. It learns to reconstruct
the next image, predict the reward, predict whether play continues, and make
its prior prediction agree with the posterior state that saw the real image.

Behaviour learning now passes the sampled sequences through the world model
again. Each sequence begins with its stored posterior state. The model then
takes the first recorded action, sees the next real image, and produces the
posterior state for that next image. It repeats this for all 64 recorded
actions. Because the posterior is allowed to see the real image, these states
are descriptions of situations that really occurred in replay, rather than
world-model guesses about unseen future images.

For each of the 16 replay sequences, this gives 65 posterior states: the stored
starting state plus(iitial..) one new posterior state after each of the 64 recorded
actions. The stored starting state is needed to process the sequence, but it is
not used as one of the 64 new imagination starting states. The code selects the
64 posterior states reached after the recorded actions. Across all 16 replay
sequences, these selected states have this shape:

```text
[16 replay sequences, 64 positions in each sequence, state contents]
```
The code flattens only the first two axes:
```text
[16, 64, state contents] -> [16 * 64, state contents]
                         -> [1,024, state contents]
```

We now have 1,024 posterior states from different steps in the sampled replay
sequences. Each one represents the model's state at a particular point in a
real game. Some may come from near the beginning of a game, some from the
middle, and some from later situations. They may also come from different
games. The reshape keeps every posterior state separate; it only puts all
1,024 states onto one batch axis. We then use each of these states as the
starting point for its own 15-step imagined trajectory. This means the actor
practises imagined futures beginning from many different real game situations
in the same training update.

These starting states represent many different Pong situations. One might
describe the ball moving toward the agent, another might describe the opponent
about to return it, and another might describe a point that has just begun.
They can come from different times in different games. The important fact is
that each state is grounded in a real replay image before imagination begins.

From each of the 1,024 starting states, the actor and world model produce a
15-step imagined trajectory:

```text
real posterior starting state
    -> actor samples action_0
    -> world model imagines state_1 and predicts reward_0
    -> actor samples action_1
    -> world model imagines state_2 and predicts reward_1
    -> ...
    -> world model imagines state_15
```

No real Pong images are used inside these 15 steps. The actor chooses actions,
and the world model's prior predicts what would happen. This produces 1,024
possible futures from 1,024 real starting situations.

A trajectory is one path through time. It records a starting state followed by
the actions that are taken and what happens after each action. One imagined
trajectory in this exercise looks like this:

```text
starting state
    -> action_0 -> imagined state_1, predicted reward_0, continuation probability_0
    -> action_1 -> imagined state_2, predicted reward_1, continuation probability_1
    -> ...
    -> action_14 -> imagined state_15, predicted reward_14, continuation probability_14
```

The numbers after the underscores are step positions. They are not reward or
probability values. For example, `reward_0` means the reward predicted for the
first imagined transition; it does not mean that the reward equals zero.

The trajectory contains 15 imagined transitions because the actor takes 15
actions. It contains 16 states because it also includes the real posterior
state from which imagination started:

```text
1 starting state + 15 imagined next states = 16 states
```

Therefore, a trajectory is not one state and it is not one prediction. It is
the complete ordered path produced from one starting state: the states visited,
the actions chosen, the predicted rewards, and the predicted continuation
probabilities. Each of the 1,024 starting states produces its own separate
trajectory.

WHAT VALUE MEANS

Value belongs to one state, not to a whole trajectory. For a state `s_t`, the
critic answers this question:

```text
Starting from this state, what return does the critic expect the current actor
to obtain?
```

The ideal value is the expected discounted sum of future rewards. The standard
formula is:

```text
V(s_t) = E[r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ...]
```

`V(s_t)` is the value of state `s_t`. `E[...]` means the expected value. The
square brackets after `E` contain one return:

```text
r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ...
```

A return is this discounted sum of future rewards along one trajectory. `E`
belongs to the mathematical definition of value: value is defined as the
expected return under the current actor. The result is one value, `V(s_t)`, for
the one state `s_t`.

For critic training, keep the input and target separate:

```text
critic input:   the model state features made from h_t and z_t
critic training target:  the lambda return calculated for that state
```

The critic receives `h_t` and `z_t` joined into one feature vector. From this
input, it predicts the value of the state. The lambda return is the value that
the prediction is trained to match. It is calculated using the predicted
rewards, continuation probabilities, and later value predictions along the one
imagined trajectory.

Therefore, the model state features are the critic input, while the lambda
return is the critic training target. As the critic trains on many such pairs
over many updates, it learns to estimate value. `E` describes what that learned
value should represent.

`r_t` is the reward received after acting from `s_t`. `r_(t+1)` is the reward
one step later, and so on. `gamma` is the discount. It makes a nearby reward
count more than the same reward much farther in the future.

In the imagined trajectory, `r_t` is predicted by the world model. The exact
order for the first imagined transition is:

```text
posterior starting state s_0
-> actor chooses action a_0 from s_0
-> world model prior predicts imagined next state s_1
-> world model reward head predicts r_0 from s_1
```

Therefore, the first imagined reward is predicted from the prior state reached after
taking the first imagined action. The same rule is used at every later step:

```text
s_t -> action a_t -> prior next state s_(t+1) -> predicted reward r_t
```

The reward head learned how to make these predictions during world-model
training. During that training, it predicts rewards from posterior states that
saw real images, and the real rewards stored in replay are the targets. During
imagination, there are no new real images, so the same reward head predicts
rewards from prior imagined states.

The actor samples one action at every step of the one imagined trajectory.
Those sampled actions and the world model's predictions provide the rewards
used to calculate the lambda return. The critic uses the lambda return to
improve its value prediction for the states along that trajectory.

For Pong, a higher value means the state looks better for the agent. For
example, a state from which the actor is likely to win the next point should
usually have a higher value than a state from which it is likely to lose it.
The exact number is an expected return, not a percentage of the game completed
and not a measure of image quality.

The critic predicts a value for every state in an imagined trajectory:

```text
V(start), V(imagined state 1), ..., V(imagined state 15)
```

There are 16 value predictions for each trajectory: one for the real posterior
starting state and one for each of the 15 imagined states. With 1,024
trajectories, the value predictions have shape `[1024, 16]`.

VALUE PREDICTION AND LAMBDA RETURN ARE DIFFERENT

The critic's value prediction is its current estimate. The imagined trajectory
is used to calculate a lambda return for each action-taking state. These two
values should not be mixed together:

```text
value prediction   the critic's current estimate of the state's expected return
lambda return      the value that the critic's prediction is trained to match
```

Suppose an imagined future predicts rewards like this:

```text
starting state -> 0 -> 0 -> +1 -> 0 -> ...
```

Those predicted rewards are used to calculate the lambda return for the
starting state and the lambda returns for later action-taking states. The
critic trains so its value predictions move toward those lambda returns. The
value prediction is attached to one state. The lambda return is calculated
from the one sampled imagined trajectory and is used to train that prediction.

WHY A 15-STEP TRAJECTORY CAN REPRESENT A LONGER FUTURE

Imagination stops after 15 steps, but value does not mean "reward during only
the next 15 steps." At the end of the trajectory, the critic estimates what
may happen after the final imagined state. The code calculates each lambda
return by working backward through the trajectory:

```text
G_t = r_t + gamma * c_t * ((1 - lambda) * V(s_(t+1)) + lambda * G_(t+1))
```

`G_t` is the lambda return for `s_t`. `r_t` is the predicted reward after
taking `a_t`. `c_t` is the predicted continuation probability. `gamma` is the
discount. `V(s_(t+1))` is the critic's value prediction for the next state.
`G_(t+1)` is the lambda return for the next state. `lambda` controls the blend
between the next value prediction and the next lambda return.

For the final imagined transition, there is no later lambda return already
inside the trajectory. The calculation therefore starts with the critic's
value prediction for the final imagined state. Using a value prediction for
the future beyond the trajectory is called bootstrapping. It lets the 15-step
trajectory include an estimate of rewards that may happen later.

The lambda return blends short and long reward paths. Some parts depend more on
nearby predicted rewards, while other parts look farther along the trajectory
before using a later value prediction. `return_lambda` controls that blend. The
result is one lambda return for each of the 15 action-taking states in every
trajectory, so the lambda return tensor has shape `[1024, 15]`.

The actor and critic use the lambda returns differently:

```text
critic: learn to predict the lambda return from each state
actor:  make actions that lead to returns better than the critic expected
```

The difference between the lambda return and the critic's value prediction is
called the advantage:

```text
advantage = lambda return - current value prediction
```

A positive advantage means the sampled imagined path turned out better than
the critic expected from that state. A negative advantage means it turned out
worse. This comparison gives the actor a learning signal for its sampled
actions.

The full flow is therefore:

```text
real replay images
-> posterior states grounded in those images
-> 1,024 separate imagination starting states
-> 15 imagined actions and transitions from every starting state
-> predicted rewards, continuation probabilities, and value predictions
-> lambda returns
-> critic learns what states are worth
-> actor learns which actions lead to better-than-expected futures
```

The critic learns from both imagined futures and posterior states inferred from
real replay observations. Stored posterior RSSM states initialize sequences
sampled from the middle of real experience.

The imagined lambda return is a training signal, not the final task score. The
important final metric is mean episode return from real Pong. The random-policy
return gives a simple baseline.

WHAT IS SCALED DOWN

This keeps the real environment, replay loop, RSSM, categorical latent state,
balanced KL loss, imagined actor-critic learning, lambda returns, two-hot
critic, and slow critic. The network is smaller to fit the available GPU.
The default run uses the Atari100k interaction budget and a training ratio
of 256 replay transitions per collected action. A smaller network does not
mean a smaller training ratio. This can still take a long time to run and
does not guarantee a strong Pong score.

LEARNING ORDER

1. Read the replay buffer to see how real transitions become sequences.
2. Read the categorical sampling helpers.
3. Read the world model and its prior/posterior steps.
4. Read the world-model loss and update.
5. Read the actor, critic, lambda returns, and behaviour loss.
6. Read real environment collection.
7. Read the main online loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Literal, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTCAMP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTCAMP_ROOT))

from common.reproducibility import set_seed

pong_video_utils = importlib.import_module("08_recurrent_visual_world_model_pong_utils")
record_random_pong_video = pong_video_utils.record_random_pong_video
save_pong_video = pong_video_utils.save_pong_video
save_pong_training_plots = pong_video_utils.save_pong_training_plots


device = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceBatch(TypedDict):
    """One batch of consecutive transitions sampled from replay."""

    images: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    initial_deterministic: torch.Tensor
    initial_stochastic: torch.Tensor
    initial_probabilities: torch.Tensor
    replay_segment_indices: torch.Tensor
    replay_start_indices: torch.Tensor


class RSSMState(TypedDict):
    """One recurrent state used by the world model."""

    deterministic: torch.Tensor         # h_t
    stochastic: torch.Tensor            # z_t or z_hat_t
    probabilities: torch.Tensor         # probabilities for z_t or z_hat_t


class WorldModelOutputs(TypedDict):
    """Predictions and distributions for a full observed sequence."""

    prior_probabilities: torch.Tensor
    posterior_probabilities: torch.Tensor
    posterior_deterministic: torch.Tensor
    posterior_stochastic: torch.Tensor
    posterior_images: torch.Tensor
    reward_logits: torch.Tensor
    continue_logits: torch.Tensor


class WorldModelLosses(TypedDict):
    """The total world-model loss and every part used to build it."""

    total: torch.Tensor
    image: torch.Tensor
    reward: torch.Tensor
    continuation: torch.Tensor
    dynamics_kl: torch.Tensor
    representation_kl: torch.Tensor
    reward_mae: torch.Tensor
    real_nonzero_reward_fraction: torch.Tensor
    predicted_reward_mean: torch.Tensor
    predicted_continue_mean: torch.Tensor
    gradient_norm: torch.Tensor


class ImaginedTrajectory(TypedDict):
    """One batch of prior-only futures used by the actor and critic."""

    features: torch.Tensor           # [B, H+1, F]: starting state is from h_t and z_t posterior, then followed by H prior states
    log_probabilities: torch.Tensor  # log-probability of each sampled action: [B, H]
    entropies: torch.Tensor          # uncertainty of the actor's action probabilities: [B, H]
    rewards: torch.Tensor            # reward predicted after each imagined action: [B, H]
    continues: torch.Tensor          # probability the episode continues after each action: [B, H]


class BehaviorLosses(TypedDict):
    """Losses and learning signals from one imagined trajectory batch."""

    actor: torch.Tensor
    critic: torch.Tensor
    critic_prediction: torch.Tensor
    replay_critic: torch.Tensor
    slow_critic_regularization: torch.Tensor
    mean_lambda_return: torch.Tensor
    actor_entropy: torch.Tensor
    imagined_reward_mean: torch.Tensor
    imagined_continue_mean: torch.Tensor
    actor_gradient_norm: torch.Tensor
    critic_gradient_norm: torch.Tensor


@dataclass
class DreamerWorldModelConfig:
    """Atari100k-style Pong settings with a smaller teaching model."""

    # =========================================================================
    # ENVIRONMENT INPUT AND REPLAY-BATCH SHAPES
    # =========================================================================
    # Atari100k uses 64 x 64 RGB images, action repeat 4, no sticky actions,
    # and only the actions needed by each game.
    image_size: int = 64
    # `action_dim` means the number of choices for a discrete environment or
    # the vector width for a continuous environment. Pong uses six choices and
    # passes its selected action into the RSSM as a six-entry one-hot vector.
    action_dim: int = 6

    # =========================================================================
    # ACTOR ACTION DISTRIBUTION
    # =========================================================================
    action_distribution: Literal["categorical", "bounded_normal"] = "categorical"
    continuous_action_min_std: float = 0.1
    continuous_action_max_std: float = 1.0
    actor_output_scale: float = 0.01

    # =========================================================================
    # REPLAY-BATCH SHAPES
    # =========================================================================
    sequence_length: int = 64
    batch_size: int = 16

    # =========================================================================
    # WORLD MODEL
    # =========================================================================
    # The paper's model is much larger. These widths keep the same main parts
    # while making the exercise practical on a normal GPU.
    learning_rate: float = 1e-4
    encoder_dim: int = 256
    deterministic_dim: int = 256
    # Eight groups each update 32 of the 256 memory values. The memory width
    # must divide evenly by this count; the groups still share input information.
    recurrent_groups: int = 8
    stochastic_variables: int = 8
    categorical_classes: int = 8
    hidden_dim: int = 256
    unimix_ratio: float = 0.01
    free_nats: float = 1.0
    dynamics_kl_weight: float = 1.0
    representation_kl_weight: float = 0.1
    reward_loss_weight: float = 1.0
    continuation_loss_weight: float = 1.0
    # DreamerV3 predicts a probability distribution over 255 reward bins.
    # The bin locations cover symlog values from -20 to 20, then `symexp`
    # changes them back into normal reward values.
    reward_bins: int = 255
    reward_symlog_limit: float = 20.0

    # =========================================================================
    # SHARED ACTOR-CRITIC IMAGINATION AND RETURN SETTINGS
    # =========================================================================
    imagination_horizon: int = 15
    behavior_hidden_dim: int = 256

    # =========================================================================
    # ACTOR AND CRITIC OPTIMIZERS
    # =========================================================================
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4

    # These settings build the return targets used by both actor and critic.
    discount: float = 0.997
    return_lambda: float = 0.95

    # =========================================================================
    # ACTOR TRAINING
    # =========================================================================
    entropy_scale: float = 3e-4

    # =========================================================================
    # CRITIC TRAINING
    # =========================================================================
    critic_bins: int = 255
    critic_symlog_limit: float = 20.0
    replay_critic_loss_weight: float = 0.3
    slow_critic_decay: float = 0.98
    slow_critic_regularization_weight: float = 1.0

    # The actor uses this tracker to normalize its advantage values.
    return_scale_decay: float = 0.99

    # =========================================================================
    # DATA COLLECTION, UPDATE COUNTS, AND EVALUATION
    # =========================================================================
    # Data budget with the current 95 online iterations:
    # `4,096 + 95 * 1,024 = 101,376` chosen actions. With action repeat 4,
    # that is 405,504 emulator frames, close to the Atari100k budget.
    # Increase `online_iterations` when running a larger training budget.
    environment_name: str = "ALE/Pong-v5"
    replay_capacity: int = 50_000
    # Start with four blocks of random experience so replay contains more Pong
    # situations and scoring events before learning begins.
    seed_steps: int = 4_096
    online_iterations: int = 95
    collect_steps_per_iteration: int = 1_024
    # With batch size 16 and sequence length 64, one update reads 1,024
    # training transitions. Running 256 updates per 1,024 collected actions
    # gives a world-model replay ratio of 256 with these settings.
    world_model_updates_per_iteration: int = 256
    behavior_updates_per_iteration: int = 256
    # 256 updates × (16 × 64 transitions) ÷ 1,024 collected actions = replay ratio 256
    # Behaviour updates also start from replay, then train on imagined futures.

    # With action repeat 4, one call to `env.step(action)` automatically applies
    # that same action for four internal Atari frames. The agent does not choose
    # another action between those frames. Afterward, `env.step(action)` returns
    # one processed observation made by taking the pixel-wise maximum of the
    # third and fourth frames, and the rewards summed across all four frames.
    #
    # Why frame skipping helps: repeating each chosen action for four emulator
    # frames gives the agent roughly 75% fewer decision points and stored
    # observations than storing every frame. Consecutive Atari frames often
    # look almost identical, so this removes many redundant tracking frames.
    # Important events, such as a `+1` or `-1` point in Pong, therefore make up
    # a larger share of the shorter replay timeline. Gymnasium implements this
    # here through `AtariPreprocessing`, rather than a separate `FrameSkip`
    # wrapper. It returns a max-pooled image from the last two repeated frames,
    # not simply the unchanged fourth frame.
    #
    # For DreamerV3's standard Atari setup:
    # 200 million emulator frames / 4 frames per chosen action
    # = about 50 million calls to the wrapped environment's `step()`.
    action_repeat: int = 4
    sticky_action_probability: float = 0.0  #Atari100k benchmark: DreamerV3 uses 0.0, Standard Atari benchmark: DreamerV3 uses sticky actions with probability 0.25.
    evaluation_episodes: int = 5
    # Atari commonly allows 108,000 emulator frames per episode.
    # With action repeat 4, that is 27,000 chosen actions.
    max_episode_steps: int = 27_000
    random_state: int = 42
    output_dir: Path = BOOTCAMP_ROOT / "outputs" / "dreamerv3_pong"

    @property
    def stochastic_dim(self) -> int:
        """Return the flattened size of the categorical latent sample."""
        return self.stochastic_variables * self.categorical_classes


@dataclass
class PongReplaySegment:
    """One consecutive piece of Pong experience that never crosses a reset."""

    # There is one more image than action because each action leads to the next
    # image: `image[t] -> action[t] -> image[t + 1]`.
    # Each stored posterior RSSM state lines up with the real image at the same
    # list position. It is a posterior state because the model saw that image
    # when it made the state. This lets a sampled sequence start from remembered
    # model state instead of rebuilding it with a separate GRU warm-up section.
    images: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    dones: list[bool]
    deterministic_states: list[np.ndarray]
    stochastic_states: list[np.ndarray]
    probability_states: list[np.ndarray]


class PongReplayBuffer:
    """Store real Pong experience and sample consecutive training sequences."""

    def __init__(self, capacity: int, sequence_length: int, random_state: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive.")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.segments: list[PongReplaySegment] = []
        self.transition_count = 0
        self.random = random.Random(random_state)

        # The paper uses each fresh, non-overlapping replay sequence before
        # filling the rest of a batch with uniform replay samples. This
        # exercise trains the world model and behaviour in separate loops, so
        # each loop has its own queue of references to the same new data. The
        # image arrays are not copied.
        self.online_queues: dict[str, deque[tuple[PongReplaySegment, int]]] = {"world_model": deque(), "behavior": deque()}

    @property
    def stored_ending_fraction(self) -> float:
        """Fraction of stored transitions whose done flag is True."""
        # add_segment guarantees that only the last transition can end a segment.
        # An ending here includes a game ending or a time-limit reset, matching
        # the existing done flags and continuation targets.
        endings = sum(bool(segment.dones[-1]) for segment in self.segments)
        return endings / self.transition_count if self.transition_count else 0.0

    def _remove_segment_from_online_queues(self, segment: PongReplaySegment) -> None:
        """Drop queued references to a segment that is leaving replay."""
        for name, queue in self.online_queues.items():
            self.online_queues[name] = deque(
                (queued_segment, start)
                for queued_segment, start in queue
                if queued_segment is not segment
            )

    def add_segment(self, segment: PongReplaySegment) -> None:
        """Add one real segment and remove the oldest data when full."""
        if len(segment.images) != len(segment.actions) + 1:
            raise ValueError("A replay segment needs one more image than action.")
        if not (len(segment.actions) == len(segment.rewards) == len(segment.dones)):
            raise ValueError("Replay actions, rewards, and done flags must align.")
        if any(segment.dones[:-1]):
            raise ValueError("A replay segment cannot continue after an episode has ended.")
        if not (
            len(segment.deterministic_states)
            == len(segment.stochastic_states)
            == len(segment.probability_states)
            == len(segment.images)
        ):
            raise ValueError("Replay needs one stored posterior RSSM state for every image.")
        if not segment.actions:
            return

        self.segments.append(segment)

        self.transition_count += len(segment.actions)
        while self.transition_count > self.capacity:
            overflow = self.transition_count - self.capacity
            oldest = self.segments[0]
            self._remove_segment_from_online_queues(oldest)
            if len(oldest.actions) <= overflow:
                self.segments.pop(0)
                self.transition_count -= len(oldest.actions)
                continue

            # Remove only the oldest transitions needed to reach capacity.
            # Removing N transitions also removes the first N images, leaving
            # one starting image for the first action that remains.
            oldest.images = oldest.images[overflow:]
            oldest.actions = oldest.actions[overflow:]
            oldest.rewards = oldest.rewards[overflow:]
            oldest.dones = oldest.dones[overflow:]
            oldest.deterministic_states = oldest.deterministic_states[overflow:]
            oldest.stochastic_states = oldest.stochastic_states[overflow:]
            oldest.probability_states = oldest.probability_states[overflow:]
            self.transition_count -= overflow

        # Queue each complete, non-overlapping sequence from the new segment.
        # Any short remainder still stays in normal replay; it is simply too
        # short to form an online sequence by itself.
        if any(stored_segment is segment for stored_segment in self.segments):
            online_sequences = [
                (segment, start)
                for start in range(
                    0,
                    len(segment.actions) - self.sequence_length + 1,
                    self.sequence_length,
                )
            ]
            for queue in self.online_queues.values():
                queue.extend(online_sequences)

    def sample_batch(
        self,
        batch_size: int,
        sequence_length: int,
        action_dim: int,
        online_queue: Literal["world_model", "behavior"] | None = None,
    ) -> SequenceBatch:
        """Sample fresh data, then replay with an occasional ending sequence."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if sequence_length != self.sequence_length:
            raise ValueError("sample_batch sequence_length must match the replay buffer's sequence_length.")

        # Collection ends a segment whenever Pong ends. Therefore, a sampled
        # sequence can use every transition directly: it cannot cross a reset
        # and does not need an episode-boundary mask.
        valid = [
            (
                segment_index,
                segment,
                len(segment.actions) - sequence_length,
            )
            for segment_index, segment in enumerate(self.segments)
            if len(segment.actions) >= sequence_length
        ]
        if not valid:
            raise RuntimeError(f"Replay needs a segment with at least {sequence_length} transitions.")

        image_batches: list[np.ndarray] = []
        action_batches: list[list[int]] = []
        reward_batches: list[list[float]] = []
        continue_batches: list[list[float]] = []
        initial_deterministic: list[np.ndarray] = []
        initial_stochastic: list[np.ndarray] = []
        initial_probabilities: list[np.ndarray] = []
        segment_indices: list[int] = []
        start_indices: list[int] = []

        selections: list[tuple[int, PongReplaySegment, int]] = []
        if online_queue is not None:
            queue = self.online_queues[online_queue]
            while queue and len(selections) < batch_size:
                segment, training_start = queue.popleft()
                segment_index = next(
                    (
                        index
                        for index, stored_segment in enumerate(self.segments)
                        if stored_segment is segment
                    ),
                    None,
                )
                if (segment_index is not None and training_start + sequence_length <= len(segment.actions)):
                    selections.append((segment_index, segment, training_start))

        # Choose uniformly across all valid sequence starts, rather than first
        # choosing a segment uniformly. Otherwise a short segment and a long
        # segment would receive the same total probability, which would make
        # each position in the short segment much more likely.
        uniform_weights = [latest_start + 1 for _, _, latest_start in valid]
        ending_windows = [
            (segment_index, segment, latest_start)
            for segment_index, segment, latest_start in valid
            if segment.dones[-1]
        ]
        remaining = batch_size - len(selections)
        if remaining and ending_windows:
            # Why add an ending window? An interior transition can occur in 64
            # different length-64 windows, but an ending occurs in only one.
            # Choosing window starts uniformly therefore rarely trains on endings.
            # We occasionally use the final 64 transitions of a segment instead.
            # All actions, images, rewards, and states stay consecutive; the
            # batch format and every loss formula stay the same.
            #
            # Target: give this batch the stored ending frequency on average.
            # Example: 25 endings / 50,000 transitions * (16*64) = 0.512 endings.
            # Each eligible ending belongs to exactly one uniform window, so
            # q below is the chance that ONE uniform window contains an ending.
            # Count endings already supplied by the fresh-data queue as well.
            q = len(ending_windows) / sum(uniform_weights)
            queued_endings = sum(
                bool(segment.dones[start + sequence_length - 1])
                for _, segment, start in selections
            )
            target_endings = batch_size * sequence_length * self.stored_ending_fraction
            expected_endings = queued_endings + remaining * q
            # Replacing one uniform draw with an ending window adds 1-q expected
            # endings. Thus p*(1-q) must equal the shortfall in expected endings.
            # In the example above, p is about 0.504: roughly every other batch.
            # Cap at one targeted window per batch. When all slots are fresh,
            # keep them; do not discard or silently consume queued sequences.
            # If q=1, every ordinary window already includes an ending.
            p = min(1.0, max(0.0, (target_endings - expected_endings) / (1.0 - q))) if q < 1.0 else 0.0
            if self.random.random() < p:
                selections.append(self.random.choice(ending_windows))
            # This is a sampling adjustment, not perfectly uniform transition
            # sampling: steps just before endings also become more common.
            # Segments shorter than sequence_length still cannot be sampled.
            # The cap and fresh-data batches can prevent an exact frequency
            # match, so compare stored and sampled ending metrics over time.
        while len(selections) < batch_size:
            segment_index, segment, latest_start = self.random.choices(valid, weights=uniform_weights, k=1)[0]
            training_start = self.random.randint(0, latest_start)
            selections.append((segment_index, segment, training_start))

        for segment_index, segment, training_start in selections:
            stop = training_start + sequence_length
            image_batches.append(np.stack(segment.images[training_start : stop + 1]))
            action_batches.append(segment.actions[training_start:stop])
            reward_batches.append(segment.rewards[training_start:stop])
            continue_batches.append([0.0 if done else 1.0  for done in segment.dones[training_start:stop] ])
            initial_deterministic.append(segment.deterministic_states[training_start])
            initial_stochastic.append(segment.stochastic_states[training_start])
            initial_probabilities.append(segment.probability_states[training_start])
            segment_indices.append(segment_index)
            start_indices.append(training_start)

        # Raw replay images are `[B,T+1,H,W,3]`. Move RGB first for PyTorch.
        images = torch.from_numpy(np.stack(image_batches)).permute(0, 1, 4, 2, 3)
        images = images.float() / 255.0
        action_ids = torch.tensor(action_batches, dtype=torch.long)  # `[B,T]`
        actions = F.one_hot(action_ids, num_classes=action_dim).float()  # `[B,T,A]`
        rewards = torch.tensor(reward_batches, dtype=torch.float32).unsqueeze(-1)
        continues = torch.tensor(continue_batches, dtype=torch.float32).unsqueeze(-1)

        return SequenceBatch(
            images=images,
            actions=actions,
            rewards=rewards,
            continues=continues,
            initial_deterministic=torch.from_numpy(np.stack(initial_deterministic)).float(),
            initial_stochastic=torch.from_numpy(np.stack(initial_stochastic)).float(),
            initial_probabilities=torch.from_numpy(np.stack(initial_probabilities)).float(),
            replay_segment_indices=torch.tensor(segment_indices, dtype=torch.long),
            replay_start_indices=torch.tensor(start_indices, dtype=torch.long),
        )

    @torch.no_grad()
    def refresh_states(self, batch: SequenceBatch, outputs: WorldModelOutputs) -> None:
        """Write freshly inferred posterior states back into sampled replay positions."""
        # `outputs[:, 0]` is the state after processing the first transition,
        # so it belongs beside `image[start + 1]`. The supplied initial state
        # already belongs beside `image[start]`; this is why `+ 1` appears
        # below. Repeated samples gradually replace states made by older model
        # parameters with states made by the current model.
        segment_indices = batch["replay_segment_indices"].detach().cpu().tolist()
        start_indices = batch["replay_start_indices"].detach().cpu().tolist()
        deterministic = outputs["posterior_deterministic"].detach().cpu().numpy()
        stochastic = outputs["posterior_stochastic"].detach().cpu().numpy()
        probabilities = outputs["posterior_probabilities"].detach().cpu().numpy()

        for batch_index, (segment_index, start_index) in enumerate(zip(segment_indices, start_indices)):
            segment = self.segments[segment_index]
            for time_index in range(deterministic.shape[1]):
                replay_index = start_index + time_index + 1
                segment.deterministic_states[replay_index] = deterministic[batch_index, time_index].copy()
                segment.stochastic_states[replay_index] = stochastic[batch_index, time_index].copy()
                segment.probability_states[replay_index] = probabilities[batch_index, time_index].copy()


def unimix_categorical(logits: torch.Tensor, ratio: float) -> torch.Tensor:
    """Return categorical probabilities mixed with a little uniform probability."""
    # Where this fits:
    # The prior, posterior, and discrete actor produce raw category scores.
    # This turns those scores into stable probabilities before sampling.
    #
    # Goal:
    # Keep most of the learned probability distribution while reserving a tiny
    # amount for every category.
    #
    # Input:
    # - logits shaped `[..., categorical_classes]`, normally `[B,8,8]`
    # - `ratio=0.01` means 1% uniform and 99% learned probabilities
    #
    # Output:
    # - probabilities of the same shape; every final axis sums to 1
    #
    # Method:
    # Use `torch.softmax(..., dim=-1)` because the last axis contains the
    # competing categories of one stochastic variable.
    #
    # Formula:
    # `probs = (1 - ratio) * softmax(logits) + ratio / num_classes`
    #
    # Steps:
    # 1. Check `0 <= ratio <= 1`.
    # 2. Call `torch.softmax(logits, dim=-1)`.
    # 3. Use `logits.shape[-1]` as the number of classes.
    # 4. Apply the formula and return the result.
    #
    # Why: the 1% uniform part stops any category probability becoming exactly
    # zero and keeps KL calculations well behaved.
    #
    # Small example:
    # With four classes and `ratio=0.01`, each class receives at least
    # `0.01 / 4 = 0.0025` from the uniform part.
    #
    # Check:
    # `torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]))`
    # should be true, and every returned value should be greater than zero.

    raise NotImplementedError("Apply categorical unimix.")

    # This stops the model from becoming completely certain that some classes
    # are impossible. For four classes, imagine softmax returns:
    #
    # learned probabilities: [1.0,    0.0,    0.0,    0.0]
    # uniform probabilities: [0.25,   0.25,   0.25,   0.25]
    # ratio = 0.01
    # final probabilities:   [0.9925, 0.0025, 0.0025, 0.0025]
    #
    # Every class keeps a small chance. This avoids zero probabilities, keeps
    # `log(probability)` calculations stable, and makes it easier for the model
    # to use a class again later. This function does not sample a class. It only
    # returns the safer probabilities used by later code.


def sample_straight_through(probabilities: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    """Make a hard one-hot sample with a soft probability gradient."""
    # Where this fits:
    # The RSSM passes a discrete latent state into the GRU and decoder. Ordinary
    # discrete sampling blocks gradients, so Dreamer uses a straight-through
    # sample that is hard during the forward pass but trainable during backward.
    #
    # Goal:
    # Choose one category for each stochastic variable without losing the
    # gradient path back to its probabilities.
    #
    # Input:
    # - probabilities shaped `[..., classes]`, normally `[B,8,8]`
    # - `deterministic=False` while training and `True` during evaluation
    #
    # Output:
    # - one-hot values of the same shape, still connected to probabilities for
    #   gradient calculation
    #
    # Steps:
    # 1. Choose one category id.
    #    - evaluation: `ids = probabilities.argmax(dim=-1)`
    #    - training: use
    #      `torch.distributions.Categorical(probs=probabilities).sample()`
    # 2. Turn each chosen id into a one-hot vector:
    #    `hard = F.one_hot(ids, num_classes=probabilities.shape[-1]).float()`
    # 3. Return:
    #    `hard + probabilities - probabilities.detach()`
    #
    # Why step 3 works:
    # `detach()` keeps the same numbers but removes their gradient path.
    # For `[0.1, 0.7, 0.2]`, suppose `hard` is `[0, 1, 0]`:
    #
    # `[0, 1, 0] + [0.1, 0.7, 0.2] - [0.1, 0.7, 0.2] = [0, 1, 0]`
    #
    # The two soft vectors cancel in the returned numbers, so the model receives
    # the hard one-hot vector. Only the non-detached `probabilities` term carries
    # gradients backward, so the network can still learn.
    #
    # Check:
    # Every final-axis sum should equal 1, every forward value should be 0 or 1,
    # and the result should require gradients when probabilities do.

    # `probabilities` shape: `[B, 8, 8]`
    raise NotImplementedError("Sample a straight-through categorical latent.")

    # Example with `deterministic=False`:
    # `probabilities = [[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]]` shape `[2, 3]`
    # One possible random sample is `ids = [1, 2]` shape `[2]`.
    # A different call may sample different ids.
    # With `deterministic=True`, argmax always gives `ids = [1, 0]`.
    #
    # Example output for deterministic=False:
    # `hard = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]` shape `[2, 3]`




class RMSNorm(nn.Module):
    """Keep feature sizes steady without using statistics from other examples."""

    # Where this fits: hidden layers use Linear -> RMSNorm -> SiLU. RMS means
    # root mean square: square the numbers, average them, then take a square root.
    # For features [3, 4], RMS is sqrt((9 + 16) / 2), about 3.54.
    # Dividing by it gives about [0.85, 1.13]. A learned scale then lets the
    # network choose the useful size of each feature. Unlike LayerNorm, this
    # does not subtract the mean. This follows the current official RMS norm.
    def __init__(self, features: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.eps = eps  # Prevent division by zero for an all-zero feature vector.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input/output: [..., features]. Normalize only the last dimension.
        # Use float32 for the squares even when other layers use lower precision.
        values = x.float()
        mean_square = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(mean_square + self.eps) * self.scale.float()
        return normalized.to(x.dtype)


class ChannelRMSNorm(nn.Module):
    """Normalize the channels separately at each image location."""

    # Why this layer exists:
    # A convolution produces feature maps: channels of learned signals that
    # describe patterns such as edges or shapes. These are hidden features,
    # not necessarily the red, green, and blue channels of the original image.
    # At one location, C channels give C different feature values.
    #
    # As convolution weights change during training, these values can become
    # much larger or smaller. That changes the inputs to SiLU and the next
    # layer, even when the relative pattern of features stays similar.
    # Normalizing their shared size gives later layers more consistent input
    # sizes and can make learning easier. It does not guarantee stable training.
    #
    # Example at one location, with two channels (ignoring epsilon):
    # [3, 4]   / RMS([3, 4])   = [0.85, 1.13]
    # [30, 40] / RMS([30, 40]) = [0.85, 1.13]
    # The tenfold change in size disappears, but channel 2 is still stronger.
    # A learned scale for each channel then lets the network adjust its size.
    # The trade-off is that the original overall size at this location is lost
    # approximately; the learned scale cannot recover that per-location size.
    #
    # Why normalize across channels? Together they describe one location.
    # Each location gets its own divisor, without mixing nearby locations or
    # other images. No batch averages are needed, so this calculation works
    # the same when training on a batch or acting on one Pong image.
    # We use Conv -> ChannelRMSNorm -> SiLU in hidden image layers. Keeping the
    # inputs to SiLU at a steadier size also limits changes caused just by the
    # convolution's overall scale. The final image output is not normalized
    # this way because it needs to predict actual pixel values.

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch images are [B,C,H,W]. Move channels last so RMSNorm sees the
        # C features at one pixel, not the image width or the whole batch.
        # [B,C,H,W] -> [B,H,W,C] -> normalize C -> [B,C,H,W].
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GroupedLinear(nn.Module):
    """Apply a separate learned matrix to each group of features."""

    # Connection to nn.Linear(in_features, out_features):
    # - inputs_per_group means in_features for EACH group's linear calculation.
    # - outputs_per_group means out_features for EACH group's linear calculation.
    # - groups says how many separate sets of weights and biases we have.
    # These counts describe feature values, not images or training examples.
    #
    # GroupedLinear(groups=8, inputs_per_group=32, outputs_per_group=96)
    # is like eight separate nn.Linear(in_features=32, out_features=96) layers.
    # Layer 0 reads group 0's 32 features and creates 96 new features;
    # layer 1 does the same for group 1 with different weights, and so on.
    # Input shape:  [B, 8, 32] -- 8 groups, each containing 32 feature values.
    # Output shape: [B, 8, 96] -- 8 groups, each containing 96 feature values.
    # B is the batch size and stays unchanged. If we flatten the groups,
    # there are 8*32 = 256 input features and 8*96 = 768 output features.
    # The layer returns the grouped shape; it does not flatten it for us.
    #
    # nn.Linear(32, 96) applied to [B, 8, 32] uses the SAME weights and bias
    # across all 8 groups, and across all batch examples.
    #
    # nn.Linear(32, 96):
    #     one shared transformation
    #     [B, 8, 32] -> [B, 8, 96]
    #
    # GroupedLinear(8, 32, 96):
    #     eight separate learned transformations, one per group
    #     [B, 8, 32] -> [B, 8, 96]
    #
    # Start with an ordinary Linear layer: each output is a weighted sum of
    # all its input values, plus a bias. Training learns the weights and bias.
    # GroupedLinear does that same calculation separately for several groups.
    # Each group has its OWN weights and bias; they are not shared across groups.
    #
    # Small example: GroupedLinear(2, 2, 1).
    # This is like TWO separate nn.Linear(in_features=2, out_features=1) layers.
    # Each layer reads two feature values and calculates one new feature value.
    # With one batch example, the input is [[[2, 3], [4, 5]]], shape [1,2,2].
    # The outer brackets hold the batch; [2,3] is group 0 and [4,5] is group 1.
    # These numbers are feature VALUES, not feature counts or weights.
    #
    # To make the arithmetic visible, suppose the learned weights currently are:
    # group 0: weights 10 and 1, bias 0
    # group 1: weights  1 and 2, bias 0
    # These are example parameter values, not values fixed by GroupedLinear.
    # In training, the optimizer changes them.
    #
    # Group 0 reads [2,3]: multiply each input by its weight, then add the bias.
    #     2*10 + 3*1 + 0 = 23
    # Group 1 reads [4,5] and uses its OWN weights:
    #     4*1 + 5*2 + 0 = 14
    # Output: [[[23], [14]]], shape [1,2,1]. The two groups are still separate;
    #
    # Compare with ONE nn.Linear(2,1) applied to that same [1,2,2] input.
    # If its weights are [10,1] and bias is 0, BOTH groups use those weights:
    # group 0: 2*10 + 3*1 = 23
    # group 1: 4*10 + 5*1 = 45
    # Output: [[[23], [45]]]. Same output shape, but shared weights.
    #
    # Why use this in our GRU?
    # The GRU stores 256 memory values, arranged into 8 groups of 32. Its gate
    # layer produces 96 scores per group: 32 reset, 32 candidate, and 32 update
    # scores. These scores control how that group's memory changes.
    # GroupedLinear(8, 32, 96) uses 8*32*96 = 24,576 weights. (96 output features per group × 8 groups = 768 scores)
    # A fully connected Linear(256, 768) uses 256*768 = 196,608 weights.
    # Both produce 768 scores, but grouping uses eight times fewer weights
    # for this layer, with fewer multiplications and less parameter storage.
    # That does not mean the whole model runs eight times faster.
    #
    # The trade-off is fewer direct connections between memory groups.
    # Groups are not assigned fixed jobs such as "ball" or "paddle"; training
    # learns what their values represent. They are also not isolated in this
    # GRU: the earlier hidden layer gives every group a shared vector built
    # from the WHOLE old memory, the latent state, and the action. Its input
    # is [B,8,800]: 32 local memory values plus 768 shared values per group.
    # Normalization across the hidden values also links their scales.

    def __init__(self, groups: int, inputs_per_group: int, outputs_per_group: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, inputs_per_group, outputs_per_group))
        self.bias = nn.Parameter(torch.zeros(groups, outputs_per_group))
        # Match the official block layer's truncated-normal initialization.
        # Each output receives inputs_per_group numbers, not all groups' inputs.
        # Why 1.1368? We sample a bell-shaped normal distribution, but reject
        # values outside two standard deviations (its typical spread) from zero.
        # Removing these extremes reduces the spread to about 0.8796 of the
        # original. Multiplying by 1/0.8796, about 1.1368, compensates for that.
        # The target spread after truncation is 1/sqrt(inputs_per_group).
        # For 32 inputs: target about 0.1768; sampling std about 0.2010;
        # after truncation, the distribution's std is about 0.1768 again.
        # More inputs mean more terms in each weighted sum, so smaller initial
        # weights help keep the outputs from growing just because width grows.
        # This constant sets the initial weights; it is not learned or tuned.
        std = 1.1368 / inputs_per_group ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input [B,G,I], weights [G,I,O], output [B,G,O]. G is the group count,
        # I is each group's input width, and O is each group's output width.
        # B is the number of examples. This layer expects groups already made;
        # the GRU's reshape calls arrange the features into those groups.
        # For example b, group g, and output o, the calculation is:
        # y[b,g,o] = sum over i of x[b,g,i] * weight[g,i,o] + bias[g,o].
        # In "bgi,gio->bgo", i disappears because we sum over the input values.
        # g stays because each group keeps its own output and weights.
        # Equivalent for one group: x[:, g, :] @ weight[g] + bias[g].
        # einsum performs this for all groups and examples in one call.
        return torch.einsum("bgi,gio->bgo", x, self.weight) + self.bias


class GroupedDreamerGRU(nn.Module):
    """Update one step of memory using the current official Dreamer core layout."""

    # Where this fits: both real observation steps and imagined steps use this
    # same memory update. It receives previous memory h, sampled state z, and
    # action a. It never sees the next image; the posterior handles that later.
    #
    #       Official part                         This smaller exercise
    # separate h, z, a input networks       three 256-wide networks
    # grouped recurrent memory              256 values split into 8 groups of 32
    # grouped hidden and gate layers        one hidden layer, then three gate scores
    # RMS normalization and gates           same calculations, at smaller widths
    #
    # This keeps the core structure while reducing memory use for a small GPU.
    # The image encoder/decoder remain the exercise's smaller convolution nets;
    def __init__(self, config: DreamerWorldModelConfig) -> None:
        super().__init__()
        self.groups = config.recurrent_groups           # 8
        self.memory_dim = config.deterministic_dim      # 256
        if self.groups < 1 or self.memory_dim < 1 or self.memory_dim % self.groups:
            raise ValueError("deterministic_dim must be positive and divisible by recurrent_groups.")
        self.group_width = self.memory_dim // self.groups   # 32
        hidden = config.hidden_dim                          # 256

        # Each input gets its own projection before we join their information.
        self.memory_input = nn.Sequential(nn.Linear(self.memory_dim, hidden), RMSNorm(hidden), nn.SiLU())           # (256 -> 256)
        self.stochastic_input = nn.Sequential(nn.Linear(config.stochastic_dim, hidden), RMSNorm(hidden), nn.SiLU()) # (64 -> 256)
        self.action_input = nn.Sequential(nn.Linear(config.action_dim, hidden), RMSNorm(hidden), nn.SiLU())         # (6 -> 256)
        for network in (self.memory_input, self.stochastic_input, self.action_input):
            # Same truncation correction explained in GroupedLinear: 1.1368
            # restores the spread lost by rejecting the normal's extremes.
            std = 1.1368 / network[0].in_features ** 0.5
            nn.init.trunc_normal_(network[0].weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(network[0].bias)

        # Each group reads its own old memory plus the same shared input vector.
        # Defaults: [B,8,32 + 3*256] -> [B,8,32].
        self.hidden = GroupedLinear(self.groups, self.group_width + 3 * hidden, self.group_width)   # (32 + 768 -> 32)
        # Official normalization runs over all 256 hidden values after the
        # grouped linear layer, not separately over each set of 32 values.
        self.hidden_norm = RMSNorm(self.memory_dim) # (256)
        self.gates = GroupedLinear(self.groups, self.group_width, 3 * self.group_width)     # (32 -> 96)

    def forward(self, memory: torch.Tensor, stochastic: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Inputs: h [B,256], flattened z [B,64], a [B,6]. Output: h_next [B,256].
        # Keep action magnitudes at most 1. Detach the divisor, as in the
        # official core. This leaves Pong's one-hot actions unchanged.
        action = action / action.abs().clamp_min(1.0).detach()
        shared = torch.cat((self.memory_input(memory), self.stochastic_input(stochastic), self.action_input(action)), dim=-1) # [B,768]
        batch_size = memory.shape[0]
        grouped_memory = memory.reshape(batch_size, self.groups, self.group_width)  # [B,256] -> [B,8,32]
        shared = shared.unsqueeze(1).expand(-1, self.groups, -1)  # [B,8,768]
        group_inputs = torch.cat((grouped_memory, shared), dim=-1)  # [B,8,800]

        # Groups have separate weights, but are not isolated: shared includes
        # information from the whole previous memory, z, and action.
        hidden = self.hidden(group_inputs).flatten(start_dim=1)  # [B,8,32] flatten -> [B,256]
        hidden = F.silu(self.hidden_norm(hidden))
        gate_scores = self.gates(hidden.reshape(batch_size, self.groups, self.group_width)) # [B,8,32] -> [B,8,96]
        # Split WITHIN each group: [B,8,96] -> three tensors [B,8,32].
        # Flattening before splitting would mix up which scores control which memory.
        reset_scores, candidate_scores, update_scores = gate_scores.chunk(3, dim=-1)
        # Why these three parts exist:
        # The GRU carries 256 memory values from one step to the next. It needs
        # to keep useful information while allowing new information to replace
        # it. Each memory value gets its own reset, candidate, and update value;
        # after flattening, each tensor below has shape [B,256]. The network
        # calculates them again at every step, rather than using fixed settings.
        #
        # CANDIDATE = a proposed new memory value, not yet the stored value.
        # candidate_scores contains raw proposals calculated from the old memory,
        # latent state, and action. Reset scales these scores, then tanh puts
        # the proposals between -1 and 1. They do not have to be written fully.
        #
        # RESET = a control between 0 and 1 for forming each candidate here.
        # Sigmoid turns a raw reset score into this control. Near 0 suppresses
        # the candidate score toward zero; near 1 lets more of it through.
        # In THIS implementation it scales the whole candidate score. It does
        # not directly erase old memory or reset the Pong episode. Clearing
        # the memory at an episode boundary is handled separately.
        #
        # UPDATE = how much of the candidate to write into each memory value.
        # In this code, near 0 means keep mostly old memory; near 1 means use
        # mostly the candidate. The final mixture is element by element:
        # new_memory = update*candidate + (1-update)*old_memory.
        #
        # One memory value, with rounded numbers:
        # old_memory=0.8, candidate_score=0.4, reset=0.5
        # candidate = tanh(0.5*0.4) = tanh(0.2), about 0.197
        # if update=0.25, new_memory = 0.25*0.197 + 0.75*0.8, about 0.649.
        # We keep most of the old value and move partway toward the proposal.
        # A small reset alone would not erase 0.8: update still decides how
        # much of the resulting near-zero candidate replaces that old value.
        reset = torch.sigmoid(reset_scores).flatten(start_dim=1)
        candidate = torch.tanh(reset * candidate_scores.flatten(start_dim=1))
        update = torch.sigmoid(update_scores.flatten(start_dim=1) - 1.0)

        # reset controls the candidate; update chooses how much of it to write.
        # h_next = update * candidate + (1 - update) * old_memory.
        # Example: old=0.8, candidate=0.2, update=0.25 gives next=0.65.
        # The -1 above encourages keeping old memory when gate scores are near 0.
        return update * candidate + (1.0 - update) * memory


class TinyDreamerWorldModel(nn.Module):
    """Convolutional encoder/decoder and categorical recurrent state-space model."""

    # DREAMER V3 COMPONENTS AND THEIR NAMES IN THIS CLASS
    #
    # `h_t` = the GRU memory, `z_t` = the latent state found using the real
    # observation, `z_hat_t` = the latent state predicted without that
    # observation, `a_t` = the action, and `x_t` = the observation.
    #
    # Sequence model: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`
    # In this class: `self.gru`, used by `observe_step` and `imagine_step`.
    #
    # Encoder: `z_t ~ q(z_t | h_t, x_t)`
    # In this class: `self.image_encoder`, `self.observation_encoder`, and
    # `self.posterior_net`.
    #
    # Dynamics predictor: `z_hat_t ~ p(z_hat_t | h_t)`
    # In this class: `self.prior_net`.
    #
    # Reward predictor: `r_t ~ p(r_t | h_(t+1), z_(t+1))`
    # Here, r_t is the reward after taking a_t from s_t. The reward head reads
    # the state reached by that action, matching the notation at the top.
    # In this class: `self.reward_head`.
    #
    # Continue predictor: `c_t ~ p(c_t | h_(t+1), z_(t+1))`
    # In this class: `self.continue_head`. It predicts whether the Pong episode
    # continues after the current transition.
    #
    # Decoder: `x_t ~ p(x_t | h_t, z_t)`
    # In this class: `self.image_decoder_input` and `self.image_decoder`.

    # The RSSM is a recurrent state-space model that predicts the next state based on the current state and action.
    # This is the parts of this class that implement the RSSM.
    # RSSM:
    # - self.gru             → updates deterministic memory h
    # - self.prior_net       → predicts stochastic state z without seeing observation
    # - self.posterior_net   → chooses z after seeing observation
    # - RSSMState            → stores h and z

    def __init__(self, config: DreamerWorldModelConfig) -> None:
        # The image encoder turns a real frame into one vector.
        # The GRU updates the deterministic memory h_t.
        # The prior predicts z_hat_t without seeing the next frame.
        # The posterior produces z_t after seeing the next frame.
        # The decoder reconstructs the frame from h_t and z_t.
        # The reward and continuation heads predict what follows the action.
        super().__init__()

        raise NotImplementedError("Build the small DreamerV3 world-model modules.")

        # IMAGE ENCODER
        # Normalize hidden channels before each activation. RMSNorm uses each
        # example's own features, so collection with batch size 1 uses the same
        # calculation as training. It has no BatchNorm running averages.

        # OBSERVATION ENCODER

        # RECURRENT DETERMINISTIC STATE
        # One stochastic sample has shape `[B, variables, classes]`. Flattening
        # it gives `[B, config.stochastic_dim]`. The grouped core processes this
        # sample, the action, and the previous memory separately before joining
        # them. It advances one step so we can form a prior and then a posterior.
        # The old plain GRU did not include the official input networks, groups,
        # normalization, or gate calculations:
        # self.gru = nn.GRUCell(input_size=config.stochastic_dim + config.action_dim, hidden_size=config.deterministic_dim)

        # PRIOR AND POSTERIOR DISTRIBUTIONS
        # - prior = prediction before seeing the next observation
        # - posterior = corrected belief after seeing the next observation
        # Both networks output one logit for every category of every stochastic variable.
        # Their output size is therefore:

        # The posterior reads recurrent memory(h_t) plus the encoded real observation(x_t). This is Z_t

        # The prior only reads recurrent memory(h_t). This is Z_hat_t

        # IMAGE DECODER
        # A complete latent feature joins deterministic memory with the flattened stochastic sample:
        # 256 + 64

        # First expand that feature back to a small spatial map:

        # `Sigmoid` keeps reconstructed pixels between 0 and 1, matching replay.
        # Normalize hidden features only. The final pixel layer keeps its own
        # brightness scale; reward/action/value output layers keep raw scores.

        # REWARD AND CONTINUATION HEADS
        # Do not put a sigmoid inside the continuation head. The later
        # `binary_cross_entropy_with_logits(...)` call expects raw logits and
        # applies the stable sigmoid calculation itself.

        # A randomly initialized reward head can predict large rewards before it
        # has learned anything. Those false rewards can delay the start of useful
        # learning. Dreamer avoids this by setting the output weights and bias to
        # zero. Every reward bin then starts with the same score, so the first
        # decoded reward is close to zero and early learning is more stable.

        # for the reward head, the bin centers are fixed and not learned. They are used to convert the categorical output into a scalar reward value.


    def encode_observation(self, image: torch.Tensor) -> torch.Tensor:
        """Turn a Pong image into one observation embedding."""
        # The posterior uses this encoded real image to correct the prior.
        # `[B,3,64,64] -> [B,4096] -> [B,encoder_dim]`
        #
        # Dreamer applies `symlog` to vector observations, such as a robot's
        # position, speed, or joint measurements. One very large input value
        # could otherwise create a large reconstruction gradient and overpower
        # the representation loss. `symlog` compresses large positive and
        # negative values, which keeps that trade-off more stable.
        #
        # This exercise receives Pong images instead of vector observations, so
        # it sends normalized pixels through the convolutional image encoder and
        # does not apply `symlog` here.

        raise NotImplementedError("Encode and join one real observation.")


    def distribution_from_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape flat logits and make one categorical latent sample."""
        # Where this fits:
        # `prior_net` and `posterior_net` end with flat linear outputs. RSSM code
        # needs those values arranged as separate categorical variables.
        #
        # Goal:
        # Turn flat network output into stable category probabilities and a
        # usable discrete latent sample.
        #
        # Input:
        # - flat logits `[B, variables * classes]`, which is `[B,64]` by default
        #
        # Output:
        # - probabilities `[B,variables,classes]`
        # - straight-through one-hot sample with the same shape
        #
        # Method:
        # Use Tensor `.view(...)`, then the two student helpers directly above
        # this class. Do not apply a second softmax after `unimix_categorical`.
        #
        # Steps:
        # 1. Use `.view(...)` to reshape the logits to
        #    `[B, config.stochastic_variables, config.categorical_classes]`.
        # 2. Call `unimix_categorical(shaped, config.unimix_ratio)`.
        # 3. Call `sample_straight_through(probabilities,
        #    deterministic=not self.training)`.
        #    Training samples categories. Evaluation chooses the largest one so
        #    repeated evaluation runs are stable.
        # 4. Return probabilities first and the sample second.
        #
        # Check: both output tensors are `[B,8,8]` with default settings.

        raise NotImplementedError("Build a categorical stochastic state from logits.")




    def initial_posterior(self, image: torch.Tensor) -> RSSMState:
        """Use the first real observation to begin the recurrent state."""
        # Where this fits:
        # A sequence needs a starting belief before its first replay action is
        # applied. There is no earlier memory, so deterministic memory starts at
        # zero and the first observation supplies the stochastic state.
        #
        # Goal:
        # Create the corrected latent belief for time 0 without inventing an
        # earlier action or earlier recurrent memory.
        #
        # Input: first image `[B,3,64,64]`.
        #
        # Output:
        # - one `RSSMState` dictionary
        #
        # Method:
        # Use `torch.zeros(...)`, `self.encode_observation(...)`,
        # `torch.cat(..., dim=-1)`, `self.posterior_net(...)`, and
        # `self.distribution_from_logits(...)` in that order.
        #
        # Steps:
        # 1. Read `batch_size = image.shape[0]`.
        # 2. Create zeros shaped `[B,config.deterministic_dim]` on `image.device`.
        # 3. Call `self.encode_observation(image)`.
        # 4. Concatenate zero memory and the embedding along the final axis.
        # 5. Run `self.posterior_net(...)`.
        # 6. Call `self.distribution_from_logits(...)`.
        # 7. Return keys `deterministic`, `stochastic`, and `probabilities`.
        #
        # Do not call the GRU here. No action has happened yet.
        #
        # Check:
        # All three returned tensors keep batch size `B`; probabilities and the
        # stochastic sample are `[B,8,8]` with the default configuration.

        raise NotImplementedError("Build the initial posterior from the first observation.")





    def observe_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        next_image: torch.Tensor,
    ) -> tuple[RSSMState, RSSMState]:
        """Return the prior prediction and posterior correction for one step."""
        # Where this fits:
        # This is one real-data RSSM transition inside `forward_sequence(...)`.
        # It teaches dynamics and representation learning at the same time.
        #
        # Goal:
        # First predict the next latent from old belief plus action, then correct
        # it using the real next observation from replay.
        #
        # Input:
        # - previous deterministic `[B, deterministic_dim]`
        # - previous stochastic `[B, variables, classes]`
        # - one-hot action `[B,6]` and next image `[B,3,64,64]`
        #
        # Output:
        # - `(prior, posterior)`, two `RSSMState` dictionaries
        # - both deterministic states are the same `[B,deterministic_dim]`
        # - their probabilities and samples are separate `[B,variables,classes]`
        #
        # Method:
        # Use Tensor `.flatten(start_dim=1)`, `torch.cat(..., dim=-1)`, the
        # local `GroupedDreamerGRU` stored in `self.gru`, and the encoder and
        # distribution helpers. Do not use `torch.no_grad()` here; both paths
        # are trained by the later losses.
        #
        # Steps:
        # 1. Use `previous["stochastic"].flatten(start_dim=1)` to get `[B,64]`.
        # 2. Call `self.gru(previous["deterministic"], flat_stochastic, action)`.
        #    The core processes these three inputs and returns new memory [B,256].
        # 3. Pass the new deterministic state through `self.prior_net`, then
        #    through `self.distribution_from_logits(...)`.
        # 4. Encode the real next image with `self.encode_observation(...)`.
        # 5. Join the new deterministic state and observation embedding with
        #    `torch.cat(..., dim=-1)`, then pass them through
        #    `self.posterior_net` and the same distribution helper.
        # 6. Build two RSSMState dictionaries. They share the new deterministic
        #    state but have their own probabilities and stochastic samples.
        # 7. Return `(prior, posterior)`.
        #
        # The prior never receives the next observation. The posterior does.
        #
        # Check:
        # `torch.equal(prior["deterministic"], posterior["deterministic"])`
        # should be true. Their categorical probabilities usually differ.

        raise NotImplementedError("Implement one RSSM observation step.")
        # Process previous memory, sampled state, and action separately. Both
        # observation and imagination use this same core; it does not see images.


        # The next image shows the Pong screen after the action.


        # Both states use `deterministic = h_t`, the new GRU memory.
        # observation_embedding is the encoded version of x_t
        # `prior` uses `stochastic = z_hat_t`, predicted without the observation.
        # `posterior` uses `stochastic = z_t`, corrected using the observation.
        # Dynamics predictor is the network: self.prior_net.  z_hat_t is the latent state produced by that network.
        # Encoder/posterior is the network that produces the latent state. z_t is the latent state it produces

    def imagine_step(self, previous: RSSMState, action: torch.Tensor) -> RSSMState:
        """Predict one next state without reading the real next observation."""
        # Where this fits:
        # World-model training uses `observe_step(...)` to read real replay images.
        # Actor and critic training uses this method to imagine future states
        # without new images. Real Pong evaluation uses `observe_step(...)`
        # to correct the agent's state with each new screen.
        #
        # Goal:
        # Perform the prior half of `observe_step(...)` using only the current
        # latent state and proposed action.
        #
        # Input:
        # - previous RSSM state with deterministic `[B,D]` and stochastic
        #   `[B,variables,classes]`
        # - one-hot discrete Pong action `[B,6]`
        #
        # Output:
        # - one prior `RSSMState`; no real image or posterior is involved
        #
        # Method:
        # Use the same `.flatten(start_dim=1)`, `torch.cat(..., dim=-1)`,
        # `self.gru(...)`, `self.prior_net(...)`, and
        # `self.distribution_from_logits(...)` calls as the prior half of
        # `observe_step(...)`.
        #
        # Steps:
        # 1. Flatten `previous["stochastic"]` to [B,64]; keep the action separate.
        # 2. Call `self.gru(previous["deterministic"], flat_stochastic, action)`.
        # 3. Produce flat prior logits with `self.prior_net(...)`.
        # 4. Convert logits into prior probabilities and a sample.
        # 5. Return all three fields in an `RSSMState`. There is no posterior
        #    because no future observation is available.
        #
        # Check:
        # The method has no image argument and never calls
        # `self.encode_observation(...)` or `self.posterior_net(...)`.

        raise NotImplementedError("Implement one prior-only imagination step.")
        # The core also applies the same action limit as in observation steps.




    def state_features(self, state: RSSMState) -> torch.Tensor:
        flat_stochastic = state["stochastic"].flatten(start_dim=1)
        return torch.cat((state["deterministic"], flat_stochastic), dim=-1) # shape: `[B, deterministic_dim + stochastic_dim]`  i.e. [B, 320]

    def decode(self, state: RSSMState) -> torch.Tensor:
        features = self.state_features(state)
        image_hidden = self.image_decoder_input(features)
        return self.image_decoder(image_hidden)  # `[B,3,64,64]` with default settings

    def predict_reward_logits(self, state: RSSMState) -> torch.Tensor:
        """Return one score for every possible reward bin."""
        features = self.state_features(state)
        return self.reward_head(features)   # `[B, reward_bins]` i.e. [B, 255] with default settings

    def predict_signals(self, state: RSSMState) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one decoded reward and one raw continuation logit."""
        features = self.state_features(state)                           # `[B, deterministic_dim + stochastic_dim]`  i.e. [B, 320]
        reward_logits = self.reward_head(features)                      # `[B,reward_bins]`
        reward_probabilities = torch.softmax(reward_logits, dim=-1)     # `[B,reward_bins]`
        reward = weighted_bin_average(reward_probabilities, self.reward_bin_centers)    # `[B]`
        return reward, self.continue_head(features)                     # `[B]`, `[B,1]`

    def forward_sequence(self, batch: SequenceBatch) -> WorldModelOutputs:
        """Run the prior/posterior update across one real sequence."""
        # Where this fits:
        # This is the main RSSM training loop. It shows exactly when the action
        # is applied, when the prior is made, and when the real next observation
        # is allowed to correct that prior.
        #
        # Goal:
        # Apply the one-step RSSM update in the correct order for every action
        # and collect aligned tensors for loss calculation and evaluation.
        #
        # Input:
        # - images contain `T+1` observations
        # - actions contain the `T` transitions between those observations
        # - the three `initial_*` tensors contain the posterior RSSM state
        #   stored after the model saw the first observation
        #
        # Output:
        # - every field in `WorldModelOutputs`, stacked along time as `[B,T,...]`
        #
        # Method:
        # Use the class helpers rather than repeating their internal layer calls.
        # Use Python lists during the loop and `torch.stack(..., dim=1)` once at
        # the end so axis 0 remains batch and axis 1 becomes time.
        #
        # Setup steps:
        # 1. Read images and actions from the batch.
        # 2. Build `current` from the stored posterior state supplied by replay.
        # 3. Make lists for the distributions and predictions from each step.
        #
        # At each `time_index` from 0 through `T-1`:
        # 1. Call `self.observe_step(...)` with:
        #    - the current posterior
        #    - `actions[:, time_index]`
        #    - `images[:, time_index + 1]`
        # 2. Decode the posterior. This reconstruction may use the real next
        #    observation because the posterior has seen it.
        # 3. Save the prior probabilities for the KL loss. The prior does not
        #    need a separate image decode during training.
        # 4. Predict reward and continuation from the posterior.
        # 5. Append the results to their matching lists.
        # 6. Set `current = posterior`, so the next step starts from the belief
        #    corrected by the real frame.
        #
        # Return steps:
        # Stack every list with `torch.stack(list_name, dim=1)` and return these
        # exact keys:
        # - `prior_probabilities`
        # - `posterior_probabilities`
        # - `posterior_deterministic`
        # - `posterior_stochastic`
        # - `posterior_images`
        # - `reward_logits`
        # - `continue_logits`
        #
        # Shape check with default settings:
        # `posterior_images` is `[B,64,3,64,64]` and both probability tensors
        # are `[B,64,8,8]` with the default 64 training steps.
        # Also check that every output's time size equals `actions.shape[1]`.

        raise NotImplementedError("Run the RSSM across one observed sequence.")




        # stack on dim=1 because each list contains one tensor per time step
        # V = number of stochastic variables. Default: 8
        # C = number of categories each stochastic variable can choose from. Default: 8



def categorical_kl(left_probabilities: torch.Tensor, right_probabilities: torch.Tensor) -> torch.Tensor:
    """Return KL(left || right), summed over variables and classes."""
    # This measures how different two categorical probability distributions are.
    # Here, it usually compares the posterior with the prior:
    # - posterior: belief after seeing the real observation
    # - prior: prediction made before seeing the real observation
    #
    # Both inputs have shape `[B, T, V, C]`:
    # - `B` = batch size
    # - `T` = time steps, default 64
    # - `V` = stochastic variables, default 8
    # - `C` = categories for each variable, deafult 8
    #
    # Formula for one variable:
    # `KL(left || right) = sum(left * (log(left) - log(right)))`
    #This is from the KL divergence formula
    #
    # Example:
    # `left  = [0.75, 0.25]`
    # `right = [0.50, 0.50]`
    # `KL(left || right)` is about `0.13`.
    # A result of `0` means they match. Larger means they disagree more.

    # Avoid `log(0)`, which is not a finite number.
    # Shapes stay `[B, T, V, C]`.
    left = left_probabilities.clamp_min(1e-8)
    right = right_probabilities.clamp_min(1e-8)

    # Calculate the contribution from each category.
    # Shape: `[B, T, V, C]`.
    per_class = left * (left.log() - right.log())

    # Sum over categories `C`, then variables `V`.
    # Output shape: `[B, T]`, one KL value per batch item and time step.
    return per_class.sum(dim=-1).sum(dim=-1)


def balanced_kl_loss(posterior_probabilities: torch.Tensor, prior_probabilities: torch.Tensor, free_nats: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Dreamer-style dynamics and representation KL losses."""
    # Where this fits:
    # `forward_sequence(...)` produces a prior and posterior at every time step.
    # These losses teach the two distributions to agree without letting either
    # side dominate both updates.
    #
    # Goal:
    # Train the prior to predict encoded reality and train the posterior to stay
    # predictable from dynamics, while ignoring KL values already below the
    # configured free-bits threshold.
    #
    # Input:
    # - posterior and prior probabilities `[B,T,variables,classes]`
    # - `free_nats`, normally `1.0`; this is the floor applied to each `[B,T]`
    #   KL value before averaging
    #
    # Output:
    # - `(dynamics_kl, representation_kl)`, two scalar tensors
    #
    # Method:
    # Use the provided `categorical_kl(left, right)` helper, Tensor `.detach()`,
    # `.clamp_min(free_nats)`, and `.mean()`.
    #
    # Steps:
    # 1. Dynamics KL trains the prior while treating the posterior as fixed:
    #    `categorical_kl(posterior.detach(), prior)`.
    # 2. Representation KL trains the posterior while treating the prior as fixed:
    #    `categorical_kl(posterior, prior.detach())`.
    # 3. Clamp each `[B,T]` result with `.clamp_min(free_nats)`.
    # 4. Average each result and return both scalars.
    #
    # Do not detach both sides. Each loss must still train one distribution.
    #
    # Plain meaning:
    # - dynamics KL updates the model that predicts without the next image
    # - representation KL updates the model that encodes the next image
    #
    # Check:
    # Both outputs are zero-dimensional tensors, are at least `free_nats`, and
    # still have `requires_grad == True` during training.

    raise NotImplementedError("Compute balanced categorical KL with free bits.")




    # `free_nats=1.0` means: reduce KL when it is above 1, but stop trying to
    # reduce it after it falls below 1.
    # It creates a point where training stops caring about making KL smaller.
    #
    # Before clamping: `KL = [2.0, 0.6, 0.1]`
    # After clamping:  `KL = [2.0, 1.0, 1.0]`
    #
    # The `2.0` value still has a gradient, so training pushes it down.
    # Values below 1 are on the flat, clamped part and have zero gradient.
    # Clamping does not push `0.6` or `0.1` up to 1. It only tells training to
    # stop spending effort on making those already-small KL values even smaller.
    # This leaves the posterior free to keep useful information from the image.
    #
    # How this affects gradients when `free_nats=1.0`:
    #
    # `y = x.clamp_min(1.0)` means:
    # - when `x < 1`, `y = 1`
    # - when `x > 1`, `y = x`
    #
    # The function is continuous because both sides meet at 1. It has a kink at
    # exactly 1, so the mathematical gradient is not unique at that one point.
    # PyTorch chooses a usable gradient there. This is similar to ReLU at 0.
    #
    # Away from that exact point:
    # - when `x < 1`, the gradient is 0 because the output stays flat at 1
    # - when `x > 1`, the gradient is 1 before the mean is applied
    #
    # Example:
    # `x = [2.0, 0.6, 0.1]`
    # `x.clamp_min(1.0) = [2.0, 1.0, 1.0]`
    # `x.clamp_min(1.0).mean() = 1.333`
    #
    # The gradients with respect to the original values are `[1/3, 0, 0]`.
    # Only `2.0` is pushed down. The values below 1 receive no gradient.
    # "Pushed down" means the optimizer changes the model weights so the next
    # KL value should be smaller. Gradient descent follows:
    # `new value = old value - learning_rate * gradient`.
    # For this simple example, with learning rate `0.1`:
    # `2.0 - 0.1 * (1/3) = 1.97` approximately.
    # In the real model, the optimizer changes network weights, not the KL value
    # directly. Those weight changes make the prior and posterior agree more.



def compute_world_model_loss(outputs: WorldModelOutputs, batch: SequenceBatch, config: DreamerWorldModelConfig) -> WorldModelLosses:
    """Compute reconstruction, prediction, and balanced KL losses."""
    # Where this fits:
    # This function turns the model outputs and the original batch into one
    # number that `backward()` can minimize. It is called once per training batch.
    #
    # Goal:
    # Teach the posterior to keep useful visual and task information, while the
    # KL terms teach the prior to predict that representation from actions.
    #
    # Input:
    # - all `[B,T,...]` tensors in `WorldModelOutputs`
    # - the original `SequenceBatch`
    # - configured weights and free-nats value
    #
    # Targets:
    # - next images are `batch["images"][:,1:]`
    # - rewards and continuation targets already have `[B,T,1]`
    #
    # Output:
    # - scalar image, reward, continuation, dynamics KL, representation KL,
    #   and total losses
    #
    # Method and steps:
    # 1. Use `F.mse_loss` for posterior image reconstruction.
    # 2. Turn each reward into a two-hot target over DreamerV3's 255
    #    symexp-spaced reward bins. Train the reward logits with soft-target
    #    cross-entropy.
    # 3. Use `F.binary_cross_entropy_with_logits` for continuation. Do not apply
    #    sigmoid before this loss.
    # 4. Call `balanced_kl_loss(...)`.
    # 5. Build the total exactly as:
    #
    #    `image_loss`
    #    `+ config.reward_loss_weight * reward_loss`
    #    `+ config.continuation_loss_weight * continuation_loss`
    #    `+ config.dynamics_kl_weight * dynamics_kl`
    #    `+ config.representation_kl_weight * representation_kl`
    #
    #    Image loss has an implicit weight of 1.
    # 6. Return the scalar tensors in `WorldModelLosses`.
    #
    # Checks:
    # - every returned tensor has shape `[]`
    # - `total.requires_grad` is true during training

    # DreamerV3 sums the squared errors across every channel and pixel in one
    # frame. The mean below then averages those per-frame losses across batch
    # and time. Using a pixel mean here would divide the reconstruction signal
    # by `3 * 64 * 64 = 12,288`, making it far too weak compared with the other
    # world-model losses.
    raise NotImplementedError("Compute the complete Dreamer world-model loss.")

    # Pong has sparse rewards: most actions receive reward `0`, and a non-zero
    # reward appears only when either player scores a point. Sparse means useful
    # feedback happens rarely.
    # The reward head must learn which visual states and actions lead toward
    # those uncommon `+1` and `-1` outcomes.
    # batch["rewards"] has shape `[B,T,1]`, this is ground truth.
    # convert this scalar reward into a two-hot target over the 255 bins. This is the target for the reward head loss.


    # This belongs in `torch.no_grad()` because it is only for logging and evaluation. It does not contribute to the loss or gradients.






def move_batch(batch: SequenceBatch, target_device: str) -> SequenceBatch:
    """Move every sequence tensor to the selected device."""
    return {key: value.to(target_device) for key, value in batch.items()}  # type: ignore[return-value]


def train_one_batch(
    model: TinyDreamerWorldModel,
    replay: PongReplayBuffer,
    batch: SequenceBatch,
    optimizer: torch.optim.Optimizer,
    config: DreamerWorldModelConfig,
) -> WorldModelLosses:
    """Perform one ordinary PyTorch update of the world model."""
    # Where this fits:
    # The model and loss only describe a computation. This helper performs the
    # actual learning update and is repeated by the main training loop.
    #
    # Goal:
    # Run one forward pass, backpropagate the total loss, protect the recurrent
    # model from very large gradients, and update its parameters once.
    #
    # Input:
    # - initialized model
    # - replay buffer whose sampled states will be refreshed
    # - one CPU `SequenceBatch` sampled from replay
    # - an existing `torch.optim.Optimizer`
    # - exercise config
    #
    # Output:
    # - the loss tensors from this batch; callers detach them when logging
    #
    # Method:
    # Use the normal PyTorch order: `zero_grad`, forward, loss, `backward`,
    # gradient clipping, then `step`. The optimizer is created outside this
    # helper so it keeps its running Adam state across batches.
    #
    # Steps:
    # 1. Call `model.train()` and move the batch with `move_batch(batch, device)`.
    # 2. Call `optimizer.zero_grad()`.
    # 3. Run `outputs = model.forward_sequence(batch)`.
    # 4. Compute losses with `compute_world_model_loss(...)`.
    # 5. Write the freshly inferred posterior states back into replay.
    # 6. Call `losses["total"].backward()`.
    # 7. Clip gradients with
    #    `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)`.
    # 8. Call `optimizer.step()` and return the loss dictionary.
    #
    # Checks:
    # - `losses["total"]` is finite: use `torch.isfinite(...)`
    # - the returned total requires gradients
    # - the returned loss can be tracked across normal training epochs

    raise NotImplementedError("Complete one world-model training update.")





    # Stored posterior states can become stale as the model learns. Refresh
    # every sampled position with posteriors inferred by the current model.


    # Computes one combined norm across all parameter gradients.
    # If that norm is above 100, scales every gradient down by the same factor.
    # Changes each param.grad in place.
    # Example:
    # combined gradient norm = 200
    # maximum norm = 100
    # scale factor = 100 / 200 = 0.5
    # Every gradient is multiplied by 0.5
    # If the norm is 80, nothing changes.
    # It does not:
    # Calculate the gradients—that is done by .backward().
    # Update model parameters—that is done by optimizer.step().




class TinyDreamerActor(nn.Module):
    """Policy for either discrete actions or bounded continuous actions."""

    def __init__(self, config: DreamerWorldModelConfig) -> None:
        super().__init__()

        self.config = config
        feature_dim = config.deterministic_dim + config.stochastic_dim  # = 256 + 64 = 320

        # Normalize hidden features before SiLU, just as in the world model.
        # Leave the final action scores unnormalized so the actor can change
        # how strongly it prefers one action. Keep the small output initialization.
        output_dim = (
            config.action_dim
            if config.action_distribution == "categorical"
            else 2 * config.action_dim
        )

        self.network = nn.Sequential(
            nn.Linear(feature_dim, config.behavior_hidden_dim),
            RMSNorm(config.behavior_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.behavior_hidden_dim, config.behavior_hidden_dim),
            RMSNorm(config.behavior_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.behavior_hidden_dim, output_dim),
        )
        with torch.no_grad():
            self.network[-1].weight.mul_(config.actor_output_scale)
            self.network[-1].bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return parameters for the configured action distribution."""
        return self.network(features)

    def sample_action(self, features: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action vectors, their log-probabilities, and entropy.

        `bounded_normal` bounds the predicted mean with `tanh`. A random sample
        can still pass -1 or 1. The environment adapter should clip that sample,
        while the RSSM limits it inside `observe_step` and `imagine_step`.
        """

        # features shape: `[B, deterministic_dim + stochastic_dim]`  i.e. [B, 320]
        parameters = self(features) # calls forward() to get the logits for the action distribution
        # parameters shape: `[B, action_dim]` for categorical or `[B, 2*action_dim]` for continuous

        if self.config.action_distribution == "categorical":
            probabilities = unimix_categorical(parameters, self.config.unimix_ratio)                # shape: `[B, action_dim]`
            distribution = torch.distributions.Categorical(probs=probabilities)
            action_ids = (probabilities.argmax(dim=-1) if deterministic else distribution.sample()) # shape: `[B]`
            actions = F.one_hot(action_ids, num_classes=self.config.action_dim).float()             # shape: `[B, action_dim]`
            return actions, distribution.log_prob(action_ids), distribution.entropy()

        mean_scores, standard_deviation_scores = parameters.chunk(2, dim=-1)
        means = torch.tanh(mean_scores)
        minimum = self.config.continuous_action_min_std
        maximum = self.config.continuous_action_max_std
        standard_deviations = minimum + (maximum - minimum) * torch.sigmoid(standard_deviation_scores + 2.0)
        distribution = torch.distributions.Independent(torch.distributions.Normal(means, standard_deviations), 1)

        if deterministic:
            actions = means
            log_probabilities = distribution.log_prob(actions)
        else:
            actions = distribution.sample()
            log_probabilities = distribution.log_prob(actions)

        return actions, log_probabilities, distribution.entropy() #


class TinyDreamerCritic(nn.Module):
    """Distributional value model over symlog-spaced return bins."""

    def __init__(self, config: DreamerWorldModelConfig) -> None:
        # Where this fits:
        # An imagined rollout is deliberately short. The critic estimates the
        # reward likely to come after that short horizon and gives the actor a
        # baseline for judging whether a sampled action was better than expected.
        #
        # Goal:
        # Build a network that outputs one logit for every possible return bin.
        # This is a scaled version of DreamerV3's distributional critic.
        #
        # Method and exact modules:
        # 1. Call `super().__init__()` and save `config`.
        # 2. Compute the same `feature_dim` used by the actor.
        # 3. Build `self.network = nn.Sequential(...)` with two hidden
        #    `nn.Linear` + local `RMSNorm` + `nn.SiLU` groups and a final
        #    `nn.Linear(config.behavior_hidden_dim, config.critic_bins)`.
        # 4. Make evenly spaced symlog bin positions with:
        #    `torch.linspace(-config.critic_symlog_limit,`
        #    `               config.critic_symlog_limit, config.critic_bins)`.
        # 5. Apply `symexp(...)` and store the raw return locations with
        #    `self.register_buffer("bin_centers", bin_centers)`.
        #    A buffer moves with the model but is not trained by the optimizer.
        # 6. DreamerV3 avoids wild initial value guesses by zero-initializing
        #    the critic output layer. Use:
        #    `nn.init.zeros_(self.network[-1].weight)` and
        #    `nn.init.zeros_(self.network[-1].bias)`.
        #
        # Check: default logits end in 255 values and `bin_centers` is not in
        # `dict(critic.named_parameters())`.
        super().__init__()

        raise NotImplementedError("Build the distributional return critic.")



        # Future returns can have several likely values and can differ greatly
        # in size across environments. Dreamer makes critic learning faster and
        # more stable by predicting a categorical distribution over return bins.
        # These bins are evenly spaced on the compressed symlog scale. After
        # `symexp` changes them back to the normal return scale, their spacing
        # grows roughly exponentially as they move away from zero.
        # register_buffer moves the tensor with the model but does not train it.

        # A randomly initialized critic can predict large future returns before
        # it has learned anything. Those false values can delay useful learning.
        # Dreamer sets the output weights and bias to zero so every return bin
        # begins with the same score. This gives a neutral initial value estimate
        # and makes the beginning of training more stable.


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Goal: return critic logits `[..., critic_bins]` from RSSM features.
        # Method: call the `nn.Sequential` network created in the constructor.
        # Check: `[4,320]` default features produce `[4,255]` logits.
        raise NotImplementedError("Predict critic return-bin logits.")

    def value(self, features: torch.Tensor) -> torch.Tensor:
        """Read one continuous return estimate from the critic distribution."""
        logits = self(features)  # shape: `[B,H+1,critic_bins]`, for example `[1024,16,255]`
        probabilities = torch.softmax(logits, dim=-1)   # shape: `[B,H+1,critic_bins]`, for example `[1024,16,255]`
        return weighted_bin_average(probabilities, self.bin_centers).squeeze(-1)    # shape: `[B,H+1]`, for example `[1024,16]`


class ReturnScaleTracker:
    """Smooth DreamerV3's robust 5th-to-95th percentile return range."""

    # This keeps actor updates stable when returns are very large or very small.
    # The actor later uses:
    # `normalized_advantage = advantage / return_scale`
    #
    # `update(...)` first flattens all returns into one list. It finds their
    # normal range with:
    # `spread = 95th percentile - 5th percentile`
    # - 5th percentile: 5% of returns are at or below this value.
    # - 95th percentile: 95% of returns are at or below this value.
    # The distance between them covers the middle 90% of the returns.
    # Using percentiles means a few extreme returns (outliers) have less effect.
    #
    # Example:
    # `5th percentile = 10`, `95th percentile = 110`
    # `spread = 110 - 10 = 100`
    #
    # The tracker changes its stored scale slowly:
    # `new_scale = decay * old_scale + (1 - decay) * new_spread`
    # With `decay=0.99`, `old_scale=100`, and `new_spread=60`:
    # `new_scale = 0.99 * 100 + 0.01 * 60 = 99.6`
    #
    # `max(1.0, self.value)` stops the scale going below 1. For example:
    # `advantage=50`, `return_scale=100`, normalized advantage `= 0.5`.
    # The actor receives `0.5` instead of `50`, so its update is more stable.

    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.value: float | None = None

    @torch.no_grad()
    def update(self, returns: torch.Tensor) -> float:
        flat = returns.detach().float().flatten()
        spread = float(torch.quantile(flat, 0.95) - torch.quantile(flat, 0.05))
        self.value = spread if self.value is None else (
            self.decay * self.value + (1.0 - self.decay) * spread
        )
        return max(1.0, self.value)


def validate_behavior_config(config: DreamerWorldModelConfig) -> None:
    """Fail early when a behaviour setting would make its math invalid."""
    if config.imagination_horizon < 1:
        raise ValueError("imagination_horizon must be at least 1.")
    if not 0.0 <= config.discount <= 1.0:
        raise ValueError("discount must be between 0 and 1.")
    if not 0.0 <= config.return_lambda <= 1.0:
        raise ValueError("return_lambda must be between 0 and 1.")
    if config.critic_bins < 3 or config.critic_symlog_limit <= 0.0:
        raise ValueError("The critic needs at least three bins and a positive limit.")
    if config.reward_bins < 3 or config.reward_symlog_limit <= 0.0:
        raise ValueError("The reward head needs at least three bins and a positive limit.")
    if config.action_distribution not in {"categorical", "bounded_normal"}:
        raise ValueError(
            "action_distribution must be 'categorical' or 'bounded_normal'."
        )
    if not (
        0.0 < config.continuous_action_min_std
        <= config.continuous_action_max_std
    ):
        raise ValueError("Continuous action standard deviations are invalid.")
    if config.actor_output_scale < 0.0:
        raise ValueError("actor_output_scale cannot be negative.")
    if config.replay_critic_loss_weight < 0.0:
        raise ValueError("replay_critic_loss_weight cannot be negative.")
    if not 0.0 <= config.slow_critic_decay < 1.0:
        raise ValueError("slow_critic_decay must be at least 0 and below 1.")
    if not 0.0 <= config.return_scale_decay < 1.0:
        raise ValueError("return_scale_decay must be at least 0 and below 1.")


# Observation values, rewards, and returns can have very different scales in
# different environments. Squared error on a large target can create a huge
# gradient and make training diverge. Absolute error and Huber loss limit that
# effect, but DreamerV3 found that they could cause learning to stop improving.
# Normalizing with running statistics has another problem: the statistics keep
# changing, so the same raw target gets a moving training target.
#
# The paper suggests symlog squared error as a simple answer. `symlog` compresses
# large positive and negative targets before squared error is calculated, while
# preserving their sign. This Pong exercise uses the same compression idea for
# rewards and returns through symlog-spaced categorical bins. Its image decoder
# still reconstructs normalized pixels directly.
#
# Why this helps across tasks: rewards can have very different sizes. Symlog
# compresses large numbers while preserving their sign:
#
# symlog(x) = sign(x) * ln(abs(x) + 1)
#
# For example, 1000 becomes about 6.91. Pong's occasional +1 and -1 rewards
# are sparse, not large. Symlog does not solve sparse rewards or guarantee
# that gradients cannot explode. Here, it defines the spacing of reward and
# value bins; we do not directly apply symlog to each training reward.
#
# WHERE SYMLOG AND SYMEXP ARE USED IN THIS FILE
#
# `symlog(...)` is not called directly on a training tensor below. It defines
# the compressed scale used to choose the bin positions.
#
# `make_symexp_bins(...)` first makes evenly spaced positions from `-20` to
# `+20` on that compressed scale. It then calls `symexp(...)` to turn those
# positions into bin centres on the normal reward and return scale.
#
# Those bin centres are used in two places:
# 1. The world model's reward head learns which bins contain each real Pong
#    reward from replay, such as `-1`, `0`, or `+1`.
# 2. The critic learns which bins contain each calculated lambda-return.
#
# The image observations, actions, continuation targets, and RSSM states do not
# use `symlog(...)` or `symexp(...)`.

# the symlog is not called anywhere in this file. The -20 to +20 default range
# used to create bin centres is already on a compressed symlog scale.
# The `symexp(...)` function is used to turn those compressed coordinates back
# into normal reward and return values.
def symlog(values: torch.Tensor) -> torch.Tensor:
    """Compress positive and negative magnitudes while preserving their sign."""
    # Example:
    # values              = [-100.0, -1.0, 0.0, 1.0, 100.0]
    # values.abs()        = [ 100.0,  1.0, 0.0, 1.0, 100.0]
    # log1p(values.abs()) = [  4.615, 0.693, 0.0, 0.693, 4.615]
    # result              = [ -4.615, -0.693, 0.0, 0.693, 4.615]
    # `torch.sign(values)` puts the original negative signs back at the end.
    return torch.sign(values) * torch.log1p(values.abs())


def symexp(values: torch.Tensor) -> torch.Tensor:
    """Undo `symlog`."""
    # SYMEXP EXAMPLE: REVERSE THE SYMLOG EXAMPLE ABOVE
    # values              = [-4.615, -0.693, 0.0, 0.693, 4.615]
    # values.abs()        = [ 4.615,  0.693, 0.0, 0.693, 4.615]
    # expm1(values.abs()) = [ 100.0,    1.0, 0.0,   1.0, 100.0]
    # result              = [-100.0,   -1.0, 0.0,   1.0, 100.0]
    # `torch.sign(values)` puts the negative signs back after expansion.
    return torch.sign(values) * torch.expm1(values.abs()) # expm1(x) = exp(x) - 1, which is more accurate for small x than exp(x) - 1


def make_symexp_bins(num_bins: int, symlog_limit: float, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Build DreamerV3's exponentially spaced bins in normal value space."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    if symlog_limit <= 0.0:
        raise ValueError("symlog_limit must be positive.")

    # Choose evenly spaced coordinates in the compressed symlog domain.
    # These coordinates are created directly, rather than by transforming raw
    # values with `symlog(...)`. `symexp(...)` maps them to bin centres in the
    # original reward or return domain.
    #
    # Example:
    # compressed coordinates: [-2, -1, 0, 1, 2]
    # raw-value bin centres:   [-6.39, -1.72, 0, 1.72, 6.39]
    compressed_bin_positions = torch.linspace(
        -symlog_limit,
        symlog_limit,
        num_bins,
        device=device,
        dtype=dtype,
    )
    return symexp(compressed_bin_positions)


def weighted_bin_average(probabilities: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode bin probabilities into one value using stable summation."""
    # WHAT THIS FUNCTION CALCULATES
    #
    # The network predicts one probability for each fixed bin. Multiplying
    # each probability by its bin value and adding the results gives one
    # ordinary reward or return prediction:
    #
    # bins          = [-100.0, -1.0, 0.0, 1.0, 100.0]
    # probabilities = [   0.10, 0.20, 0.40, 0.20,   0.10]
    # weighted bins = [ -10.00, -0.20, 0.00, 0.20,  10.00]
    # prediction    = -10.00 - 0.20 + 0.00 + 0.20 + 10.00 = 0.00
    #
    # WHY THE OLD SINGLE SUM DID NOT WORK RELIABLY
    #
    # The real 255 bins begin as compressed coordinates from `-20` to `+20`.
    # `symexp(...)` turns their endpoints into raw values of approximately
    # `-485,165,184` and `+485,165,184`. At initialization, softmax gives every
    # bin almost the same probability. The weighted terms therefore include
    # many very large negative and positive numbers that should cancel.
    #
    # The old code asked float32 to mix all of them in one reduction:
    #
    # `return (probabilities * bins).sum(dim=-1, keepdim=True)`
    #
    # The final layers of the reward head and critic are initialized with zero
    # weights and zero biases. They therefore begin by producing 255 logits
    # that are all `0.0`. Softmax turns equal logits into equal probabilities,
    # so every bin receives `1 / 255`, or approximately `0.00392157`. These
    # probabilities are correct and add up to `1.0`. The error happened only
    # when the old code multiplied them by the 255 large positive and negative
    # bin values and summed every result at once. That float32 sum returned
    # about `0.3086` instead of the mathematically correct prediction of `0.0`.
    #
    # Float32 keeps limited precision. When numbers hundreds of millions apart
    # are reduced together, some low digits are rounded away. The positive and
    # negative rounding errors do not have to be identical, even though the
    # mathematical values are symmetric. This was a floating-point rounding error.
    #
    # Full old-code example:
    #
    # bins          = [-100.0, -1.0, 0.0, 1.0, 100.0]
    # probabilities = [   0.10, 0.20, 0.40, 0.20,   0.10]
    # weighted bins = [ -10.00, -0.20, 0.00, 0.20,  10.00]
    # old operation = sum([-10.00, -0.20, 0.00, 0.20, 10.00])
    # exact result  = 0.00
    #
    # These small example numbers are exactly representable enough to give the
    # right result. The problem becomes visible with the real bins. For uniform
    # probabilities over this file's 255 bins, the exact prediction is `0.0`,
    # but the old float32 sum returned about `0.3086` on the CPU in a direct
    # test. A freshly initialized reward head therefore appeared to predict a
    # positive reward instead of zero.
    #
    # HOW THE NEW SUM FIXES IT
    #
    # Match every negative weighted bin with its equally distant positive
    # weighted bin before doing the final sum:
    #
    # negative weighted bins         = [-10.00, -0.20]
    # reverse the negative side      = [ -0.20, -10.00]
    # positive weighted bins         = [  0.20,  10.00]
    # matched pairs                  = [  0.00,   0.00]
    # centre weighted bin            = 0.40 * 0.00 = 0.00
    # new result                     = 0.00 + sum([0.00, 0.00]) = 0.00
    #
    # With the real uniform prediction, the roughly `-1,902,608` contribution
    # from the first bin is paired directly with the roughly `+1,902,608`
    # contribution from the last bin. Each pair cancels before the final sum,
    # so large same-sign subtotals are never created. This is the symmetric-sum
    # method used by the public DreamerV3 implementation.
    #
    # The code also supports an even number of bins. Odd bin counts, such as
    # the 255 used here, have one unpaired centre bin. Even counts do not.
    number_of_bins = bins.shape[-1]
    middle = number_of_bins // 2

    negative_probabilities = probabilities[..., :middle]
    negative_bins = bins[:middle]

    if number_of_bins % 2 == 1:
        centre_probability = probabilities[..., middle : middle + 1]
        centre_bin = bins[middle : middle + 1]
        positive_probabilities = probabilities[..., middle + 1 :]
        positive_bins = bins[middle + 1 :]
        centre_contribution = centre_probability * centre_bin
    else:
        positive_probabilities = probabilities[..., middle:]
        positive_bins = bins[middle:]
        centre_contribution = torch.zeros_like(probabilities[..., :1])

    negative_contributions = negative_probabilities * negative_bins
    positive_contributions = positive_probabilities * positive_bins
    matched_pairs = negative_contributions.flip(dims=(-1,)) + positive_contributions
    return centre_contribution + matched_pairs.sum(dim=-1, keepdim=True)


def two_hot_targets(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Share each value between the neighbouring bin centres surrounding it."""
    # How this fits into Dreamer:
    # The reward head predicts the reward for a state, and the critic predicts
    # how much future reward a state may lead to. Instead of predicting each
    # value as one raw number, they predict probabilities across fixed bins.
    #
    # A two-hot target puts its weight on the two bins surrounding the real
    # value. It is similar to a one-hot label, but it can represent a value that
    # lies between two bins.
    #
    # Small example:
    #
    # `bins = [0.0, 1.0, 2.0]`
    # `value = 1.25`
    # `target = [0.0, 0.75, 0.25]`
    #
    # The value 1.25 is 25% of the way from 1.0 to 2.0. Therefore, bin 1.0
    # receives weight 0.75 and bin 2.0 receives weight 0.25. The weights in
    # each target always add up to 1.
    #
    # More examples using the same bins:
    #
    # ```text
    # value 0.00 -> [1.00, 0.00, 0.00]
    # value 0.50 -> [0.50, 0.50, 0.00]
    # value 1.00 -> [0.00, 1.00, 0.00]
    # value 1.25 -> [0.00, 0.75, 0.25]
    # value 2.00 -> [0.00, 0.00, 1.00]
    # ```
    #
    # The main steps below are:
    # 1. Keep each value inside the smallest and largest bin.
    # 2. Use `torch.searchsorted(...)` to find the first bin at or above it.
    # 3. The bin immediately before that is the lower bin.
    # 4. Measure how far the value lies between the lower and upper bins.
    # 5. Put the two weights into one target vector with `F.one_hot(...)`.
    #
    # If `values` has shape `[B,H]` and there are `K` bins, the returned target
    # has shape `[B,H,K]`. `B` is batch size and `H` is the number of imagined
    # steps. The final `K` values are used as a soft target for cross-entropy.
    if bins.ndim != 1 or bins.numel() < 2:
        raise ValueError("bins must be one-dimensional with at least two entries.")
    if not bool(torch.all(bins[1:] > bins[:-1])):
        raise ValueError("bins must be in strictly increasing order.")

    num_bins = bins.shape[0]
    clipped_values = values.clamp(min=bins[0], max=bins[-1])
    upper_indices = torch.searchsorted(sorted_sequence=bins, input=clipped_values.contiguous(), right=False).clamp(min=1, max=num_bins - 1)
    lower_indices = upper_indices - 1

    lower_bins = bins[lower_indices]
    upper_bins = bins[upper_indices]
    upper_weights = (clipped_values - lower_bins) / (upper_bins - lower_bins)
    lower_weights = 1.0 - upper_weights

    lower_one_hot = F.one_hot(lower_indices, num_classes=num_bins).to(values.dtype)
    upper_one_hot = F.one_hot(upper_indices, num_classes=num_bins).to(values.dtype)

    return (
        lower_one_hot * lower_weights.unsqueeze(-1)
        + upper_one_hot * upper_weights.unsqueeze(-1)
    )


@torch.no_grad()
def posterior_replay_states(model: TinyDreamerWorldModel, batch: SequenceBatch) -> tuple[torch.Tensor, RSSMState]:
    # Where this fits:
    # Dreamer does not begin imagination from a made-up random tensor. It begins
    # from beliefs inferred from real replay observations (we pass the real image observation below into observe_step).
    # This helper creates those beliefs before the actor starts choosing new imagined actions
    # later on in function imagine_actor_critic_trajectory().
    #
    # Goal:
    # Observe the complete real batch sequence. Return features for the replay
    # critic loss and one imagination start for every training transition.
    #
    # Input:
    # - one device-ready batch with observations `[B,T+1,...]`, action vectors
    #   `[B,T,A]`, and a stored starting posterior RSSM state
    #
    # Output:
    # - replay features `[B,T+1,F]`
    # - flattened imagination starts `[B*T,...]`
    # - every returned tensor is detached
    #
    # Method and steps:
    # 1. Build the starting posterior state from the three `initial_*` tensors.
    # 2. Loop over all `T` actions with `model.observe_step(...)`.
    # 3. Save every posterior feature for the replay critic.
    # 4. Select posterior states reached by the T training transitions and
    #    flatten batch and time into one imagination-start axis.
    #
    # Check: the start state has batch size `B*T` and requires no gradients.

    raise NotImplementedError("Infer posterior states where imagination begins.")





    # Dreamer deliberately turns the states along each real replay sequence
    # into separate starting points for new imagined futures. For example, a
    # replay sequence may look like this:
    #
    # image_0 --action_0--> image_1 --action_1--> image_2
    #
    # `state_steps` contains the posterior belief after each real image:
    # `[state_0, state_1, state_2, ...]`. We use `state_steps[1:]`, then treat it as:
    #
    # state_1 -> one new imagined future chosen by the current actor (--new actor action--> imagined state --> imagined state ...)
    # state_2 -> another new imagined future chosen by the current actor (--new actor action--> imagined state --> imagined state ...)
    # state_3 -> another new imagined future chosen by the current actor(--new actor action--> imagined state --> imagined state ...)
    # Each separate imagined future runs up to the configured imagination horizon H
    # 1,024 separate imagined trajectories, 15 imagined transitions per trajectory, 1,024 × 15 = 15,360 imagined transitions
    #
    # Dreamer is not continuing the original action sequence here. The real
    # sequence was used to infer realistic states. From each of those states,
    # the current actor tries new actions and the world model predicts what
    # would happen. This gives the actor practice from many real situations in
    # one update instead of practising only from the final replay state.
    #
    # Real replay:
    # one connected sequence used to infer realistic states
    # Actor training:
    # each inferred state becomes an independent starting point
    # for a new imagined future
    #
    # Stacking first gives `[B,T,...]`. `flatten(0, 1)` turns that into
    # `[B*T,...]`, so a batch with B replay sequences and T transitions gives
    # B*T separate imagination starting states. With the default B=16 and
    # T=64, this gives 16*64=1,024 starting states. The actor uses each one to
    # begin a separate imagined future.



def imagine_actor_critic_trajectory(world_model: TinyDreamerWorldModel, actor: TinyDreamerActor, start: RSSMState, config: DreamerWorldModelConfig,) -> ImaginedTrajectory:
    # Where this fits:
    # This is Dreamer's inner practice world. No environment call and no real
    # future observation is allowed inside this loop.
    #
    # Goal:
    # Let the actor choose H actions and use only the frozen RSSM prior to
    # predict the states, rewards, and continuation weights that follow.
    #
    # Shapes with batch B and horizon H:
    # - features `[B,H+1,feature_dim]`
    # - action vectors `[B,H,A]`
    # - log probabilities, entropies, rewards, continues `[B,H]`
    #
    # H is the imagination horizon: the number of future steps Dreamer imagines.
    #
    # Method:
    # Use `world_model.state_features`, `actor.sample_action`,
    # `world_model.imagine_step`, `world_model.predict_signals`, and
    # `torch.stack(..., dim=1)`. Wrap only the world-model calls in
    # `torch.no_grad()`. Actor log-probabilities must keep their gradient.
    #
    # Steps for each imagined time:
    # 1. Save the current state features.
    # 2. Call `actor.sample_action(features, deterministic=False)`.
    # 3. Under `torch.no_grad()`, pass `action.detach()` to
    #    `world_model.imagine_step(...)`, then predict reward and continuation.
    # 4. Flatten the final size-1 reward and continuation axes.
    # 5. Use `continue_probability = continue_logit.sigmoid()`.
    # 6. Save log-probability, entropy, reward, and continuation.
    # 7. Move current to the imagined next state and repeat.
    # 8. After the loop, append the final state feature. Stack and return every
    #    exact key in `ImaginedTrajectory`.
    #
    # Checks:
    # - only `features` has H+1 time entries
    # - actor log-probabilities require gradients
    # - rewards and features from the frozen model do not require gradients

    raise NotImplementedError("Imagine an actor-critic training trajectory.")

    # current is intiliased to start state, which is the posterior state inferred from the real replay sequence/observation.
    # Then starting from this real posterior state, we will imagine a trajectory of H steps using the actor and the world model.
    # When imagining using imagine_step, we save the prior state features




def lambda_returns(rewards: torch.Tensor, continues: torch.Tensor, next_values: torch.Tensor, discount: float, lambda_: float) -> torch.Tensor:
    # Where this fits:
    # An H-step rollout does not contain the agent's entire future. Lambda
    # returns combine predicted rewards with critic values beyond each step.
    # They become the critic targets and the main actor learning signal.
    #
    # H is the imagination horizon: the number of future steps Dreamer imagines.
    #
    # Input and output:
    # - rewards, continues, and next_values all `[B,H]`
    # - output lambda returns `[B,H]`
    # - `next_values[:,t]` estimates the state reached after action t
    #
    # Formula, working backward:
    # `G_t = r_t + discount * c_t * ((1-lambda) * V_{t+1} + lambda * G_{t+1})`
    # The final `G` after the horizon starts from `next_values[:,-1]`.
    #
    # Method and steps:
    # 1. Validate matching rank-2 shapes and values of discount/lambda in [0,1].
    # 2. Set `running = next_values[:, -1]` and make an empty Python list.
    # 3. Loop backward over the H axis.
    # 4. Apply the formula using `next_values[:, time_index]`.
    # 5. Append each result, reverse the list, and stack with `dim=1`.
    #
    # Small example with lambda=0 uses only reward plus the next critic value.
    # With lambda=1 it follows all imagined rewards to the final bootstrap.
    # Check: output shape exactly matches rewards.
    raise NotImplementedError("Compute backward bootstrapped lambda returns.")

    # This loop calculates one lambda-return target for every imagined step. It
    # works backward from the end of the imagined trajectory. For example:
    #
    # step 0 -> step 1 -> step 2
    #
    # The loop visits:
    #
    # step 2 -> step 1 -> step 0
    #
    # This is necessary because the return at an earlier step depends on the
    # return already calculated for the following step.
    #
    # `running` starts as the critic's estimated value after the final imagined
    # action. This is the value used beyond the imagination horizon.
    #
    # At each step, the calculation is:
    #
    # return now = reward now + discounted future estimate
    #
    # The future estimate mixes two choices:
    #
    # `(1.0 - lambda_) * next_values[:, time_index]`
    #     the critic's estimate of all remaining rewards from the next state
    #
    # `lambda_ * running`
    #     the return already calculated using later rewards and a critic value
    #
    # Both choices include distant rewards. The difference is how many rewards
    # are used explicitly before relying on a critic value estimate.
    #
    # `lambda_` controls the mixture:
    #
    # lambda_ = 0.00 -> use only the critic's next-state value
    # lambda_ = 1.00 -> use the longer return from later steps
    # lambda_ = 0.95 -> use 5% next-state value and 95% longer return
    #
    # `discount * continues[:, time_index]` controls how much the future counts:
    #
    # continues = 1.0 -> count the discounted future normally
    # continues = 0.0 -> the episode ended, so ignore future value
    # continues = 0.5 -> count half as much future value

    # Start with the critic's estimated value of the state reached after the final
    # imagined action. This value estimates what may happen beyond the imagination
    # horizon.
    # `next_values` received `all_values[:, 1:]`, so its last column contains the
    # critic's value estimate for the state reached after the final imagined action.


    # The terms are grouped like this:
    # current reward
    # +
    # discount × continuation probability
    # ×
    # (
    #     (1 - lambda) * next-state value
    #     +
    #     lambda * return already calculated from later steps
    # )

    # The loop works backward because each earlier return uses `running`, which
    # already contains the return calculated for the following step. This means
    # the list is currently ordered `[last_step, ..., first_step]`.
    # Reverse it back to normal time order, then stack the B-value tensors into
    # `[B,H]`. The result contains one lambda-return target for every imagined
    # step in every trajectory.


def compute_behavior_losses(
    trajectory: ImaginedTrajectory,
    replay_features: torch.Tensor,
    replay_batch: SequenceBatch,
    critic: TinyDreamerCritic,
    slow_critic: TinyDreamerCritic,
    return_scale: ReturnScaleTracker,
    config: DreamerWorldModelConfig,
) -> BehaviorLosses:
    # Where this fits:
    # The imagined rollout supplies experience. This function turns that
    # experience into one critic objective and one actor objective.
    #
    # Goal:
    # Train the critic toward two-hot lambda-return targets and train the actor
    # with normalized REINFORCE advantages plus entropy.
    #
    # Inputs:
    # - trajectory features `[N,H+1,F]`; all other trajectory fields `[N,H,...]`
    # - current critic is trainable; slow critic is an EMA target and is frozen
    #
    # H is the imagination horizon: the number of future steps Dreamer imagines.
    #
    # Output:
    # Return the scalar tensors named by `BehaviorLosses`.
    #
    # Steps:
    # 1. Under `torch.no_grad()`, evaluate `critic.value(...)` on all features.
    #    The current critic builds DreamerV3's returns; stopping gradients keeps
    #    those self-made targets fixed during this update. Pass values for
    #    times 1 through H to:
    #    `lambda_returns(trajectory["rewards"], trajectory["continues"],`
    #    `               values[:, 1:], config.discount,`
    #    `               config.return_lambda)`.
    # 2. Build cumulative weights from discount and continuation. A later step
    #    counts less when it is far away or the episode may have ended.
    # 3. Use only features at times 0 through H-1 as critic inputs.
    # 4. Get current critic logits `[N,H,K]` and make two-hot targets over
    #    `critic.bin_centers`.
    # 5. Compute soft-target cross-entropy at each step, multiply it by the
    #    cumulative weight, and average it.
    # 6. Under `torch.no_grad()`, call `slow_critic.value(current_features)`
    #    to get one predicted value per state, then call
    #    `two_hot_targets(slow_values, critic.bin_centers)` to encode those
    #    values as bin targets. Their cross-entropy against the current critic's
    #    predictions is the slow-critic regularizer.
    # 7. Combine the imagined prediction loss, slow-critic regularizer, and
    #    weighted replay critic loss as shown below to get total critic loss.
    # 8. Under `torch.no_grad()`, read current critic values for current states.
    #    Compute `advantage = lambda_targets - current_values` and divide it by
    #    `return_scale.update(lambda_targets)`.
    # 9. Actor loss is the weighted negative mean of:
    #    `advantage.detach() * trajectory["log_probabilities"]`
    #    `+ config.entropy_scale * trajectory["entropies"]`.
    # 10. Return scalar losses: `actor`, `critic`, `critic_prediction`,
    #    `replay_critic`, and `slow_critic_regularization`. Keep their gradients.
    #    Also return detached logging values: `mean_lambda_return`,
    #    `actor_entropy`, `imagined_reward_mean`, and `imagined_continue_mean`.
    #    `actor_gradient_norm` and `critic_gradient_norm` start as zero here;
    #    the behaviour update fills them after backward and gradient clipping.
    #
    # Checks: all returned tensors are scalars; actor loss reaches actor
    # parameters, while critic loss reaches critic parameters.

    # B = number of replay sequences, here 16
    # T = real transitions per replay sequence, here 64
    # N = B*T imagined trajectories, here 1024
    # H = number of imagined future steps
    # F = feature size
    # K = number of critic value bins, here 255

    # INPUTS EXPLAINED CLEARLY:
    #
    # `trajectory` comes from `imagine_actor_critic_trajectory(...)`. It contains
    # imagined features, action log-probabilities, action-distribution entropies,
    # predicted rewards, and predicted continuation probabilities.
    #
    # Each imagined trajectory begins from a posterior RSSM state inferred from
    # real replay images. After that starting point, the actor chooses new
    # actions and the world model's prior net predicts the following states.
    #
    # `trajectory["features"]` has shape `[N,H+1,F]` i.e. [1024, 16, 320]. It contains the starting
    # state's features followed by the features of all H imagined future states.
    # Each feature vector joins the deterministic RSSM state (256) with the flattened
    # stochastic RSSM state(8x8=64), giving a total feature size of F (320).
    #
    # `trajectory["log_probabilities"]` has shape `[N,H]` i.e.[1024, 15]. Each value is the log
    # probability that the actor assigned to the action sampled at that imagined
    # step. For a continuous action distribution, it is a log probability
    # density instead. For now, for pong, we are concerned with categorical actions,
    # so it is a log probability.
    #
    # Example: if action 1 is sampled with probability 0.90, the stored value is
    # `log(0.90)`, which is about `-0.105`. This is a log-probability, not a
    # negative log-likelihood. Its negative, `-log(0.90) = 0.105`, would be the
    # negative log-likelihood. The actor later uses
    # `-advantage * log_probability`, so this is an advantage-weighted policy
    # gradient loss rather than an ordinary negative log-likelihood loss. A
    # positive advantage increases the sampled action's probability; a negative
    # advantage decreases it.
    #
    # `trajectory["entropies"]` has shape `[N,H]` i.e. [1024, 15]. Each value measures how
    # uncertain the actor's action distribution was at that step.
    # Example with three possible actions:
    #
    # confident actor probabilities = [0.90, 0.05, 0.05]
    # uncertain actor probabilities = [0.34, 0.33, 0.33]
    #
    # Entropy is:
    # `-sum(probability * log(probability))`
    #
    # confident entropy:
    # `-(0.90*log(0.90) + 0.05*log(0.05) + 0.05*log(0.05))`
    # `= about 0.39`
    # less surpise gives lower entropy so the actor is more certain about which action to choose
    #
    # uncertain entropy:
    # `-(0.34*log(0.34) + 0.33*log(0.33) + 0.33*log(0.33))`
    # `= about 1.10`
    # more surprise gives higher entropy so the actor is less certain about which action to choose
    #
    # Lower entropy means the actor strongly prefers certain actions.
    # Higher entropy means the probabilities are spread more evenly, so the actor
    # is less certain about which action to choose.
    # log_probability -> concerns the one action that was sampled
    # entropy         -> concerns the complete action probability array
    #
    # Unlike `log_probabilities`, entropy uses the probabilities of every possible
    # action. It does not use only the probability of the sampled action.
    #
    # The entropy bonus encourages exploration instead of letting the actor
    # become certain too early.
    #
    # `trajectory["rewards"]` has shape `[N,H]`. These rewards are predicted by
    # the world model for the imagined future states. They do not come directly
    # from the real Pong environment.
    #
    # `trajectory["continues"]` has shape `[N,H]` i.e. [1024, 15]. Each value is the world
    # model's predicted probability that the episode continues after that
    # imagined step. A value near 1 means "probably continues"; a value near 0
    # means "probably ends here."
    ######
    # `replay_features` comes from `posterior_replay_states(...)`. It contains the
    # features of the posterior RSSM states inferred from the real replay images.
    # replay feature = deterministic state h_t + posterior stochastic state z_t
    # It has shape `[B,T+1,F]` i.e. [16, 65, 320]: one feature vector for the initial replay state and
    # one for each of the T states reached after the T real replay actions.
    #
    # The states after the actions are also flattened separately and used as the
    # B*T starting points for imagined trajectories.
    # posterior states from real replay
    #   ├── features kept as [B,T+1,F] for the replay critic loss
    #   └── states 1 through T flattened to [B*T,...] for imagination starts (not used here, used in imagination)
    #
    # `replay_batch` is the original real replay batch. This function uses its real
    # rewards and continuation values to calculate return targets for the replay
    # critic loss. The replay critic learns to predict these targets from
    # `replay_features`.

    raise NotImplementedError("Compute imagined actor and critic losses.")

        # The shape stays `[N,H]`, but the discount values are shifted one
        # position to the right. For one trajectory, suppose:
        #
        # `discounts = [d0, d1, d2, d3]`
        #
        # `discounts[:, :-1]` removes the final value:
        #
        # `[d0, d1, d2]`
        #
        # Joining `first_weight` to the front gives:
        #
        # `[1, d0, d1, d2]`
        #
        # `torch.cumprod(..., dim=1)` then produces:
        #
        # `weight_0 = 1`
        # `weight_1 = d0`
        # `weight_2 = d0 * d1`
        # `weight_3 = d0 * d1 * d2`
        #
        # Step 0 has weight 1 because nothing must happen before reaching it.
        # Step 1 counts only if the trajectory continued through step 0. Step 2
        # counts only if it continued through steps 0 and 1, and so on. The
        # final discount `d3` would weight a later step 4, but this four-step
        # example has no step 4 loss, so that final discount is not needed.
        # Therefore, this operation keeps the same shape but changes alignment:
        #
        # before: `[d0, d1, d2, d3]`
        # after:  `[1,  d0, d1, d2]`
        #
        # Why use cumulative products? A loss at a later step should count only
        # if the imagined episode is expected to reach that step. Reaching step
        # 3 means the episode must continue after step 0, step 1, and step 2.
        # The probability of all three events is their product.
        #
        # The loss at step 0 gets full weight. Later losses get smaller weights
        # because they are farther away and the episode may end before reaching
        # them. If a continuation probability is 0, every later weight becomes
        # 0, so actor and critic losses after that predicted ending are ignored.
        #
        # Example 1: the episode may continue through every step.
        # Use `discount = 0.9` and the same loss of 2.0 at every step:
        #
        # continues              = `[1.0, 0.5, 0.8, 1.0]`
        # discount * continues   = `[0.90, 0.45, 0.72, 0.90]`
        # add 1 and shift right   = `[1.0, 0.90, 0.45, 0.72]`
        # weights after cumprod  = `[1.0, 0.90, 0.405, 0.2916]`
        # Then used later
        # loss at each step      = `[2.0, 2.0, 2.0, 2.0]`
        # weighted losses        = `[2.0, 1.80, 0.810, 0.5832]`
        #
        # Example 2: the episode is predicted to end after step 1.
        # Again use `discount = 0.9` and a loss of 2.0 at every step:
        #
        # continues              = `[1.0, 0.0, 1.0, 1.0]`
        # discount * continues   = `[0.90, 0.00, 0.90, 0.90]`
        # add 1 and shift right   = `[1.0, 0.90, 0.00, 0.90]`
        # weights after cumprod  = `[1.0, 0.90, 0.00, 0.00]`
        # Then used later
        # loss at each step      = `[2.0, 2.0, 2.0, 2.0]`
        # weighted losses        = `[2.0, 1.80, 0.00, 0.00]`
        #
        # The zero at step 1 makes every weight after step 1 zero. Later
        # continuation predictions cannot restart an episode that already ended.
        # The losses at steps 2 and 3 are ignored because the imagined episode
        # is not expected to reach them.

    # From the step above, we have:
    # lambda_targets: what return the critic should predict
    # weights: how much each imagined step’s loss should count


    # -------------------------------------------------------------------------
    # CRITIC LOSS COMPUTATION STARTS HERE
    # -------------------------------------------------------------------------

    # This is the critic's trainable forward pass. It runs outside
    # `torch.no_grad()`, so PyTorch records it in the computation graph.

    # take scalar lambda return targets and convert them into two-hot targets over the bin centers


    # The paper calls this maximum-likelihood training. The critic should place
    # as much probability as possible on the return target. The code performs
    # the equivalent operation by minimizing negative log-likelihood.
    #
    # Maximizing likelihood, maximizing log-likelihood, and minimizing negative
    # log-likelihood all train the probabilities in the same direction.
    # Because the target is soft and two-hot rather than one hard bin, the code
    # implements this objective as soft-target cross-entropy:
    #
    # Regular cross-entropy uses a one-hot target. For example:
    #
    # target probabilities    = `[0.00, 1.00, 0.00]`  # from one-hot-encoded true label
    # predicted probabilities = `[0.10, 0.60, 0.30]`
    # loss = `-(1.00 * log(0.60))`
    #
    # Only one target entry is 1, so only the predicted log-probability for that
    # one bin contributes to the loss. The entries multiplied by 0 disappear.
    #
    # Two-hot cross-entropy shares the target between two nearby bins:
    #
    # target probabilities    = `[0.00, 0.75, 0.25]`
    # predicted probabilities = `[0.10, 0.60, 0.30]`
    # loss = `-(0.75 * log(0.60) + 0.25 * log(0.30))`
    #
    # Here two non-zero target numbers are multiplied by the two matching
    # predicted log-probabilities. This lets the target represent a return that
    # lies between two critic bins instead of forcing it into only one bin.
    # `loss = -sum(target_k * log(predicted_probability_k))`
    #
    # Example for one imagined step with three return bins:
    # `critic_targets = [0.00, 0.75, 0.25]`
    # `critic probabilities = [0.10, 0.60, 0.30]`
    # `loss = -(0.00*log(0.10) + 0.75*log(0.60) + 0.25*log(0.30))`
    # `loss` is about `0.68`.
    # Summing over K removes the return-bin axis. This leaves one loss for every
    # imagined step in every trajectory, with shape `[N,H]`.

    # Multiply each step's loss by its cumulative weight. Early, reachable
    # steps count more; steps after a likely episode ending count little or
    # nothing. Then average all B*H weighted losses into one scalar that can be
    # used by `.backward()`.

    # Now, we have calculated `critic_prediction_loss`, the main
    # maximum-likelihood loss on imagined return targets. Two supporting parts
    # still need to be calculated:
    #
    # complete critic loss
    # = imagined-target prediction loss
    # + weighted slow-critic regularization
    # + weighted replay critic loss
    #
    # The next block calculates slow-critic regularization. The replay section
    # after it calculates the replay critic loss. Near the end, the function
    # combines all three parts into `critic_loss`.

    # The critic learns from return targets that partly use the critic's own
    # value predictions. `lambda_targets` partly depend on the critic's own
    # next-state value predictions from `all_values[:, 1:]`. As the critic
    # changes, these return targets also change, which can make training unstable.
    # This moving target can make learning unstable. Dreamer therefore also
    # trains the critic to stay close to a slow copy of itself.
    # `slow_critic` uses an exponential moving average (EMA) of the current
    # critic's parameters, so its predictions change more smoothly over time.
    # Old approach: copy every probability from the slow critic.
    # This is wrong for DreamerV3's slow-critic loss: it asks the current critic
    # to copy the whole distribution, including its spread, instead of using
    # the slow critic's predicted value as a target. Copying a distribution is
    # a valid loss for other purposes, but it is a different training goal.
    # with torch.no_grad():
    #     slow_critic_logits = slow_critic(current_features)
    #     slow_critic_probabilities = torch.softmax(slow_critic_logits, dim=-1)
    # slow_critic_regularization_per_step = -(slow_critic_probabilities * critic_log_probabilities).sum(dim=-1)

    # New approach: probabilities -> one predicted value -> two-hot target.
    # Our local `.value(current_features)` method averages the slow critic's
    # bin values using their probabilities. It returns one value per state.
    # `two_hot_targets(slow_values, critic.bin_centers)` then places target
    # probability on the two neighbouring bins that represent each value.
    #
    # Example with small illustrative bins [-10, 0, 10]:
    # slow probabilities [0.5, 0, 0.5] give value 0.5*(-10) + 0.5*10 = 0.
    # The old target was [0.5, 0, 0.5]. The new target is [0, 1, 0].
    # The current critic should learn the value 0, not preserve both peaks.
    # If a value falls between bins, two_hot_targets shares the target between
    # those neighbours so their weighted average equals that value.
    #
    # `no_grad()` keeps this target fixed during backpropagation. Only the
    # current critic learns from this loss; update_slow_critic() separately
    # moves the slow critic's parameters towards the current critic's parameters.

    # Cross-entropy: -sum(target_probability * log(predicted_probability)).
    # This increases the current critic's probability on the target bins.
    # Sum over the K bins to get one loss per state, then apply the existing
    # weights so imagined steps after a likely episode ending count less.

    # Replay part of the loss:
    # Also train the critic on the real replay sequence. Each real next state
    # gets the first imagined return from the rollout that started there. This
    # is DreamerV3's replay value loss, called `repval` in the official config.




    # Old loss used the full slow distribution, which is the wrong target here.
    # replay_slow_per_step = -(replay_slow_probabilities * replay_log_probabilities).sum(dim=-1)
    # Compare the current critic against the two-hot encoding of the slow value.
    # Summing over K bins leaves one supporting loss per replay state: [B,T].

    # Complete critic loss:
    # 1. Learn the imagined lambda-return targets.
    # 2. Stay close to the slowly changing critic.
    # 3. Also learn value targets from the real replay sequence.

    # Side note: how PyTorch autograd handles these critic forward passes
    #
    # The current critic has two trainable forward passes above:
    #
    # 1. `critic(current_features)` is the forward pass for the imagined
    #    states. It creates `critic_log_probabilities`, which are used by both
    #    `critic_prediction_loss` and `slow_critic_regularization`.
    # 2. `critic(replay_current_features)` is the forward pass for the real
    #    replay states. It creates `replay_log_probabilities`, which are used
    #    by `replay_critic_loss`.
    #
    # The slow critic is a separate network. Its forward passes run inside
    # `torch.no_grad()`, so backward does not update its parameters. Its scalar
    # value predictions are converted into fixed two-hot targets; we do not
    # copy its full probability distributions. Slow-critic regularization updates
    # the current critic because the current critic's log-probabilities remain
    # connected to the computation graph.
    #
    # PyTorch keeps the two trainable current-critic forward passes as separate
    # graph branches. Both branches point back to the same current-critic
    # parameters/weights. Each loss is reduced to a scalar with `.mean()`, and the code
    # above combines those scalar losses with a weighted sum.
    #
    # One call to `critic_loss.backward()` follows both trainable branches. For
    # every current-critic parameter, `parameter.grad` receives the sum of:
    #
    # gradient from `critic_prediction_loss`
    # + weighted gradient from `slow_critic_regularization`
    # + weighted gradient from `replay_critic_loss`
    #
    # This assumes the optimizer cleared old gradients before backward.

    # -------------------------------------------------------------------------
    # ACTOR LOSS COMPUTATION STARTS HERE
    # -------------------------------------------------------------------------

    # The first term is the REINFORCE, or score-function policy-gradient,
    # objective. It multiplies the log-probability of each sampled action by
    # that action's normalized advantage. A positive advantage increases the
    # sampled action's probability. A negative advantage decreases it.
    # the REINFORCE term encourages the actor to increase the probability
    # of actions that led to better-than-expected returns and decrease the
    # probability of actions that led to worse-than-expected returns.
    # `.detach()` keeps the advantage fixed while training the actor, so actor
    # gradients flow through the action log-probabilities and not through the
    # value calculations used to produce the advantage, i.e. The advantage is
    # detached so the actor does not backpropagate through the critic.
    #
    # The second term is the entropy bonus. Maximizing it encourages the actor
    # to keep exploring instead of becoming certain too early. The two terms
    # are objectives to maximize. PyTorch optimizers minimize a loss, so the
    # code negates their sum. `weights` reduces the effect of later steps when
    # the discounted continuation probability is small.



@torch.no_grad()
def update_slow_critic(critic: TinyDreamerCritic, slow_critic: TinyDreamerCritic, decay: float) -> None:
    """Move slow critic parameters a small distance toward the current critic."""
    for current, slow in zip(critic.parameters(), slow_critic.parameters()):
        slow.mul_(decay).add_(current, alpha=1.0 - decay)
    for current, slow in zip(critic.buffers(), slow_critic.buffers()):
        slow.copy_(current)


def train_actor_critic_one_batch(
    world_model: TinyDreamerWorldModel,
    actor: TinyDreamerActor,
    critic: TinyDreamerCritic,
    slow_critic: TinyDreamerCritic,
    raw_batch: SequenceBatch,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    return_scale: ReturnScaleTracker,
    config: DreamerWorldModelConfig,
) -> BehaviorLosses:
    # Where this fits:
    # This is one complete behaviour update. It begins from real replay,
    # practices only in imagination, and changes actor and critic weights.
    #
    # Goal:
    # Perform one safe actor-critic optimizer step without changing the trained
    # world model.
    #
    # Method and steps:
    # 1. Move the batch with `move_batch`.
    # 2. Call `zero_grad()` on both optimizers.
    # 3. Infer replay posterior states with `posterior_replay_states(...)`.
    # 4. Call `imagine_actor_critic_trajectory(...)`.
    # 5. Compute behavior losses and call
    #    `(losses["actor"] + losses["critic"]).backward()`.
    # 6. Clip actor and critic gradients separately with
    #    `torch.nn.utils.clip_grad_norm_(..., max_norm=100.0)`.
    # 7. Step both optimizers.
    # 8. Call `update_slow_critic(...)` using the configured decay.
    # 9. Return the loss dictionary.
    #
    # Checks:
    # - world-model parameters all have `grad is None`
    # - actor and critic losses are finite
    raise NotImplementedError("Complete one imagined actor-critic update.")


    # replay_features contains the features for the critic loss, and start is the initial RSSM state for imagination.
    # replay_features is the model state features combining the deterministic and stochastic parts of the posteriorRSSM state (h_t and z_t),
    # `start` contains the RSSM states reached after each real action in the
    # replay batch. Dreamer deliberately uses every one as a separate starting
    # point for a new imagined future. It does not continue the stored replay
    # actions. The current actor chooses new actions from each starting point.
    # The `[B,T,...]` batch and replay-time axes are combined
    # into `[B*T,...]`, so the actor treats all B*T states as separate starting
    # points for imagined futures. The parts inside each RSSM state are not
    # flattened or mixed together. With B=16 and T=64, `start` contains 1,024
    # separate RSSM states.
    # trajectory.features has shape `[B*T, H+1, F]` and contains the imagined features for the critic loss.
    # trajectory.log_probabilities, trajectory.entropies, trajectory.rewards, and trajectory.continues all
    # have shape `[B*T, H]` and contain the imagined log probabilities, entropies, rewards, and
    # continuation probabilities for the actor loss.






def save_metrics(metrics: dict[str, Any], config: DreamerWorldModelConfig) -> str:
    path = config.output_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


@dataclass
class PongEvaluationEpisode:
    """Return and frames from one real Pong evaluation episode."""

    total_reward: float
    steps: int
    frames: list[np.ndarray]


def make_pong_environment(config: DreamerWorldModelConfig) -> Any:
    """Create Atari Pong with image processing for this small model."""
    try:
        import ale_py
        import gymnasium as gym
        from gymnasium.wrappers import AtariPreprocessing
    except ImportError as error:
        raise ImportError("Pong needs the Atari dependency. Run: uv sync") from error

    gym.register_envs(ale_py)
    # ALE advances one emulator frame here. AtariPreprocessing repeats each
    # chosen action four times, max-pools frames, and resizes them to 64x64.
    base_env = gym.make(
        config.environment_name,
        frameskip=1,
        repeat_action_probability=config.sticky_action_probability,
        full_action_space=False,
        render_mode="rgb_array",
    )
    env = AtariPreprocessing(
        base_env,
        noop_max=30,
        frame_skip=config.action_repeat,
        screen_size=config.image_size,
        terminal_on_life_loss=False,
        grayscale_obs=False,
        scale_obs=False,
    )
    return gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)


def validate_pong_config(config: DreamerWorldModelConfig) -> None:
    """Catch online settings that cannot produce a replay training batch."""
    if config.image_size != 64:
        raise ValueError("This small encoder and decoder expect 64x64 Pong images.")
    if config.action_dim != 6:
        raise ValueError("The minimal Pong action set contains six actions.")
    if config.action_distribution != "categorical":
        raise ValueError("Pong needs action_distribution='categorical'.")
    if config.sequence_length < 1 or config.batch_size < 1:
        raise ValueError("Sequence length and batch size must be positive.")
    if config.seed_steps < config.sequence_length:
        raise ValueError("seed_steps must be at least sequence_length.")
    if config.replay_capacity < config.sequence_length:
        raise ValueError("Replay capacity must hold at least one sequence.")
    if config.online_iterations < 1 or config.collect_steps_per_iteration < 1:
        raise ValueError("Online iterations and collection steps must be positive.")
    if (
        config.world_model_updates_per_iteration < 1
        or config.behavior_updates_per_iteration < 1
    ):
        raise ValueError("Both update counts must be positive.")
    if config.action_repeat < 1:
        raise ValueError("action_repeat must be positive.")
    if not 0.0 <= config.sticky_action_probability <= 1.0:
        raise ValueError("sticky_action_probability must be between 0 and 1.")
    if config.evaluation_episodes < 1 or config.max_episode_steps < 1:
        raise ValueError("Pong evaluation needs at least one episode and one step.")


def pong_frame(observation: Any, config: DreamerWorldModelConfig) -> np.ndarray:
    """Return one Pong observation as a `uint8 [64,64,3]` replay frame."""
    frame = np.asarray(observation)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    expected = (config.image_size, config.image_size, 3)
    if frame.shape != expected:
        raise ValueError(f"Expected Pong image {expected}, received {frame.shape}.")
    return np.asarray(frame, dtype=np.uint8).copy()


def pong_frame_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert one replay frame to `[1,3,64,64]` in the range 0 to 1."""
    return (
        torch.from_numpy(frame)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
        .to(device)
    )


def rssm_state_to_numpy(state: RSSMState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy one batch-size-one RSSM state into replay-friendly NumPy arrays."""
    if state["deterministic"].shape[0] != 1:
        raise ValueError("Collection expects exactly one RSSM state.")
    return (
        state["deterministic"].squeeze(0).detach().cpu().numpy().copy(),
        state["stochastic"].squeeze(0).detach().cpu().numpy().copy(),
        state["probabilities"].squeeze(0).detach().cpu().numpy().copy(),
    )


@torch.no_grad()
def collect_pong_experience(
    world_model: TinyDreamerWorldModel,
    actor: TinyDreamerActor,
    replay: PongReplayBuffer,
    config: DreamerWorldModelConfig,
    number_of_steps: int,
    seed: int,
    random_actions: bool,
    env: Any | None = None,
    starting_observation: Any | None = None,
    starting_state: RSSMState | None = None,
) -> tuple[Any, RSSMState]:
    """Use real Pong and add each transition and aligned RSSM state to replay."""
    if (starting_observation is None) != (starting_state is None):
        raise ValueError(
            "Pass both starting_observation and starting_state, or pass neither."
        )

    owns_env = env is None
    if env is None:
        env = make_pong_environment(config)

    try:
        if int(env.action_space.n) != config.action_dim:
            raise ValueError(f"Pong exposes {env.action_space.n} actions but action_dim is {config.action_dim}.")
        if starting_observation is None:
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
        else:
            observation = starting_observation

        current_observation = observation
        first_frame = pong_frame(observation, config)
        images = [first_frame]
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []
        world_model.eval()
        actor.eval()

        if starting_observation is None:
            current = world_model.initial_posterior(pong_frame_tensor(first_frame))
        else:
            # Dreamer keeps the recurrent state across collection blocks. It may
            # be slightly stale after model updates, just like stored replay
            # posterior states, and replay training later refreshes sampled
            # posterior states.
            current = starting_state

        first_deterministic, first_stochastic, first_probabilities = (rssm_state_to_numpy(current))
        deterministic_states = [first_deterministic]
        stochastic_states = [first_stochastic]
        probability_states = [first_probabilities]

        for _ in range(number_of_steps):
            if random_actions:
                action_id = int(env.action_space.sample())
            else:
                features = world_model.state_features(current)
                one_hot_action, _, _ = actor.sample_action(features, deterministic=False)
                action_id = int(one_hot_action.argmax(dim=-1).item())

            next_observation, reward, terminated, truncated, _ = env.step(action_id)
            current_observation = next_observation
            done = bool(terminated or truncated)
            next_frame = pong_frame(next_observation, config)
            images.append(next_frame)
            actions.append(action_id)
            rewards.append(float(reward))
            dones.append(done)
            action = F.one_hot(torch.tensor([action_id], device=device), num_classes=config.action_dim,).float()
            _, current = world_model.observe_step(current, action, pong_frame_tensor(next_frame)) # current is the posterior state after observing the next frame
            next_deterministic, next_stochastic, next_probabilities = (rssm_state_to_numpy(current))
            deterministic_states.append(next_deterministic)
            stochastic_states.append(next_stochastic)
            probability_states.append(next_probabilities)

            if done:
                replay.add_segment(PongReplaySegment(images, actions, rewards, dones, deterministic_states, stochastic_states, probability_states))
                observation, _ = env.reset()
                current_observation = observation
                first_frame = pong_frame(observation, config)
                images, actions, rewards, dones = [first_frame], [], [], []
                current = world_model.initial_posterior(pong_frame_tensor(first_frame))
                first_deterministic, first_stochastic, first_probabilities = ( rssm_state_to_numpy(current))
                deterministic_states = [first_deterministic]
                stochastic_states = [first_stochastic]
                probability_states = [first_probabilities]

        replay.add_segment( PongReplaySegment(images, actions, rewards, dones, deterministic_states, stochastic_states, probability_states))
        return current_observation, current
    finally:
        if owns_env:
            env.close()


@torch.no_grad()
def evaluate_pong_actor(
    world_model: TinyDreamerWorldModel,
    actor: TinyDreamerActor,
    config: DreamerWorldModelConfig,
    random_actions: bool = False,
    number_of_episodes: int | None = None,
) -> list[PongEvaluationEpisode]:
    """Run the learned actor or a random baseline without learning."""
    # Pong reward:
    # - the agent scores one point     -> reward `+1`
    # - the opponent scores one point -> reward `-1`
    #
    # Episode return is the sum of those rewards for one complete game.
    # For example, losing 21 to 0 gives:
    # `0 agent points - 21 opponent points = -21 episode return`.
    # Losing 21 to 5 gives `5 - 21 = -16`.
    # Higher is better. A return of `+21` means winning 21 to 0.
    world_model.eval()
    actor.eval()
    results: list[PongEvaluationEpisode] = []
    episode_count = config.evaluation_episodes if number_of_episodes is None else number_of_episodes
    if episode_count < 1:
        raise ValueError("number_of_episodes must be at least 1.")
    for episode_index in range(episode_count):
        env = make_pong_environment(config)
        try:
            seed = config.random_state + 50_000 + episode_index
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            frame = pong_frame(observation, config)
            frames = [frame]
            current = world_model.initial_posterior(pong_frame_tensor(frame))
            total_reward = 0.0
            steps = 0
            for _ in range(config.max_episode_steps):
                if random_actions:
                    action_id = int(env.action_space.sample())
                else:
                    features = world_model.state_features(current)
                    one_hot_action, _, _ = actor.sample_action(features, deterministic=True)
                    action_id = int(one_hot_action.argmax(dim=-1).item())
                next_observation, reward, terminated, truncated, _ = env.step(action_id)
                next_frame = pong_frame(next_observation, config)
                action = F.one_hot(torch.tensor([action_id], device=device), num_classes=config.action_dim).float()
                _, current = world_model.observe_step(current, action, pong_frame_tensor(next_frame))
                total_reward += float(reward)
                steps += 1
                if steps % 4 == 0: # steps % 25 == 0:
                    frames.append(next_frame)
                if terminated or truncated:
                    break
            results.append(PongEvaluationEpisode(total_reward, steps, frames))
        finally:
            env.close()
    return results


def run_online_pong_exercise(config: DreamerWorldModelConfig) -> dict[str, Any]:
    """Collect data, update the world model, and then update behaviour."""
    # perf_counter measures elapsed seconds without being affected by clock
    # changes. Total run time includes setup, collection, training, evaluation,
    # checkpoints, and plots, up to writing the final metrics file.
    run_started = time.perf_counter()

    # -------------------------------------------------------------------------
    # STAGE 1: CHECK SETTINGS AND PREPARE THE OUTPUT FOLDER
    # -------------------------------------------------------------------------
    set_seed(config.random_state)
    validate_behavior_config(config)
    validate_pong_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 2: CHECK THE REAL PONG ENVIRONMENT
    # -------------------------------------------------------------------------
    # This confirms the image shape and the six actions before training starts.
    contract_env = make_pong_environment(config)
    try:
        observation, _ = contract_env.reset(seed=config.random_state)
        print(
            "Pong contract:",
            {
                "image": tuple(np.asarray(observation).shape),
                "actions": int(contract_env.action_space.n),
                "meanings": contract_env.unwrapped.get_action_meanings(),
            },
        )
    finally:
        contract_env.close()

    # Keep this quick environment check separate from the larger replay seed.
    video_path = record_random_pong_video(None, config, number_of_steps=1_024)

    # -------------------------------------------------------------------------
    # STAGE 3: CREATE THE MODELS, REPLAY BUFFER, AND OPTIMIZERS
    # -------------------------------------------------------------------------
    world_model = TinyDreamerWorldModel(config).to(device)
    actor = TinyDreamerActor(config).to(device)
    critic = TinyDreamerCritic(config).to(device)
    slow_critic = TinyDreamerCritic(config).to(device)
    slow_critic.load_state_dict(critic.state_dict())
    slow_critic.requires_grad_(False)

    # Count every learned number in each network. The trainable total matches
    # the usual meaning of model size in papers. The slow critic is shown
    # separately because it is a stored copy updated by EMA, not by an optimizer.
    world_model_parameters = sum(parameter.numel() for parameter in world_model.parameters())
    actor_parameters = sum(parameter.numel() for parameter in actor.parameters())
    critic_parameters = sum(parameter.numel() for parameter in critic.parameters())
    slow_critic_parameters = sum(parameter.numel() for parameter in slow_critic.parameters())
    trainable_parameters = world_model_parameters + actor_parameters + critic_parameters
    print(
        "Model parameters:",
        {
            "world_model": f"{world_model_parameters:,}",
            "actor": f"{actor_parameters:,}",
            "critic": f"{critic_parameters:,}",
            "trainable_total": f"{trainable_parameters:,}",
            "slow_critic_copy": f"{slow_critic_parameters:,}",
            "stored_total_including_slow_critic": f"{trainable_parameters + slow_critic_parameters:,}",
        },
    )

    replay = PongReplayBuffer(config.replay_capacity, config.sequence_length, config.random_state)
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=config.learning_rate)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)
    return_scale = ReturnScaleTracker(config.return_scale_decay)

    # -------------------------------------------------------------------------
    # STAGE 4: COLLECT RANDOM EXPERIENCE TO START THE REPLAY BUFFER
    # -------------------------------------------------------------------------
    # The actor cannot train before replay contains real Pong transitions.
    print(f"Collecting {config.seed_steps} random seed steps.")
    collection_env = make_pong_environment(config)      # It is created once, but reset whenever an episode ends.
    current_observation, collection_state = collect_pong_experience(
        world_model,
        actor,
        replay,
        config,
        config.seed_steps,
        config.random_state,
        random_actions=True,
        env=collection_env,
    )

    # -------------------------------------------------------------------------
    # STAGE 5: REPEAT THE MAIN ONLINE DREAMER LOOP
    # -------------------------------------------------------------------------
    history: list[dict[str, float | int | None]] = []
    # Wait for GPU setup work before timing the loop. Its average includes
    # collection, model updates, periodic evaluations, and checkpoint saves.
    if device == "cuda":
        torch.cuda.synchronize()
    iterations_started = time.perf_counter()
    for iteration in range(config.online_iterations):
        # ---------------------------------------------------------------------
        # STAGE 5A: ACT IN REAL PONG AND ADD THE NEW EXPERIENCE TO REPLAY
        # ---------------------------------------------------------------------
        current_observation, collection_state = collect_pong_experience(
            world_model,
            actor,
            replay,
            config,
            config.collect_steps_per_iteration,
            config.random_state + iteration + 1,
            random_actions=False,
            env=collection_env,
            starting_observation=current_observation,
            starting_state=collection_state,
        )

        # ---------------------------------------------------------------------
        # STAGE 5B: TRAIN THE WORLD MODEL ON REAL EXPERIENCE FROM REPLAY
        # ---------------------------------------------------------------------
        world_model.requires_grad_(True)
        world_model.train()
        world_model_loss_total = 0.0
        image_loss_total = 0.0
        reward_loss_total = 0.0
        reward_mae_total = 0.0
        continuation_loss_total = 0.0
        dynamics_kl_total = 0.0
        representation_kl_total = 0.0
        real_nonzero_reward_fraction_total = 0.0
        predicted_reward_mean_total = 0.0
        predicted_continue_mean_total = 0.0
        world_model_gradient_norm_total = 0.0
        world_model_sampled_endings = 0
        world_model_sampled_transitions = 0
        for _ in range(config.world_model_updates_per_iteration):
            batch = replay.sample_batch(
                config.batch_size,
                config.sequence_length,
                config.action_dim,
                online_queue="world_model",
            )
            # Count the actual sampled targets on CPU, before moving the batch
            # to the GPU. An ending has continue=0, regardless of its reward.
            world_model_sampled_endings += int((batch["continues"] == 0).sum().item())
            world_model_sampled_transitions += batch["continues"].numel()
            losses = train_one_batch(world_model, replay, batch, world_optimizer, config)
            world_model_loss_total += float(losses["total"].detach().cpu())
            image_loss_total += float(losses["image"].detach().cpu())
            reward_loss_total += float(losses["reward"].detach().cpu())
            reward_mae_total += float(losses["reward_mae"].cpu())
            continuation_loss_total += float(losses["continuation"].detach().cpu())
            dynamics_kl_total += float(losses["dynamics_kl"].detach().cpu())
            representation_kl_total += float(losses["representation_kl"].detach().cpu())
            real_nonzero_reward_fraction_total += float(losses["real_nonzero_reward_fraction"].cpu())
            predicted_reward_mean_total += float(losses["predicted_reward_mean"].cpu())
            predicted_continue_mean_total += float(losses["predicted_continue_mean"].cpu())
            world_model_gradient_norm_total += float(losses["gradient_norm"].cpu())

        # ---------------------------------------------------------------------
        # STAGE 5C: TRAIN THE ACTOR AND CRITIC IN THE UPDATED WORLD MODEL
        # ---------------------------------------------------------------------
        # The world model is frozen while it creates imagined futures.
        world_model.zero_grad(set_to_none=True)
        world_model.requires_grad_(False)
        # Keep sampling latent states during imagination. Freezing the weights
        # stops parameter updates; training mode still controls latent sampling.
        # In this model, eval mode picks the most likely latent category instead.
        # RMS normalization behaves the same in both modes.
        world_model.train()
        actor_loss_total = 0.0
        critic_loss_total = 0.0
        replay_critic_loss_total = 0.0
        imagined_return_total = 0.0
        actor_entropy_total = 0.0
        critic_prediction_loss_total = 0.0
        slow_critic_regularization_total = 0.0
        imagined_reward_mean_total = 0.0
        imagined_continue_mean_total = 0.0
        actor_gradient_norm_total = 0.0
        critic_gradient_norm_total = 0.0
        behavior_sampled_endings = 0
        behavior_sampled_transitions = 0
        for _ in range(config.behavior_updates_per_iteration):
            batch = replay.sample_batch(
                config.batch_size,
                config.sequence_length,
                config.action_dim,
                online_queue="behavior",
            )
            behavior_sampled_endings += int((batch["continues"] == 0).sum().item())
            behavior_sampled_transitions += batch["continues"].numel()
            losses = train_actor_critic_one_batch(
                world_model,
                actor,
                critic,
                slow_critic,
                batch,
                actor_optimizer,
                critic_optimizer,
                return_scale,
                config,
            )
            actor_loss_total += float(losses["actor"].detach().cpu())
            critic_loss_total += float(losses["critic"].detach().cpu())
            replay_critic_loss_total += float(losses["replay_critic"].detach().cpu())
            imagined_return_total += float(losses["mean_lambda_return"].detach().cpu())
            actor_entropy_total += float(losses["actor_entropy"].cpu())
            critic_prediction_loss_total += float(losses["critic_prediction"].detach().cpu())
            slow_critic_regularization_total += float(losses["slow_critic_regularization"].detach().cpu())
            imagined_reward_mean_total += float(losses["imagined_reward_mean"].cpu())
            imagined_continue_mean_total += float(losses["imagined_continue_mean"].cpu())
            actor_gradient_norm_total += float(losses["actor_gradient_norm"].cpu())
            critic_gradient_norm_total += float(losses["critic_gradient_norm"].cpu())

        # ---------------------------------------------------------------------
        # STAGE 5D: RECORD THIS ONLINE ITERATION'S TRAINING VALUES
        # ---------------------------------------------------------------------
        environment_actions = (config.seed_steps + (iteration + 1) * config.collect_steps_per_iteration)
        completed_iterations = iteration + 1
        is_tenth_iteration = completed_iterations % 10 == 0
        is_last_iteration = completed_iterations == config.online_iterations
        evaluation_mean_return: float | None = None
        if is_tenth_iteration:
            evaluation_episodes = evaluate_pong_actor(world_model, actor, config, number_of_episodes=3)
            evaluation_mean_return = float(np.mean([episode.total_reward for episode in evaluation_episodes]))

        world_updates = config.world_model_updates_per_iteration
        behavior_updates = config.behavior_updates_per_iteration
        record = {
            "iteration": iteration + 1,
            "environment_actions": environment_actions,
            "emulator_frames": config.action_repeat * environment_actions,
            "replay_size": replay.transition_count,
            # Fractions of real transitions, not percentages or reward rates.
            # Example: 0.0005 means 0.05%, or one ending per 2,000 transitions.
            # Higher is not automatically better: sampled fractions should be
            # near the stored fraction over many batches. A persistent shortfall
            # can mean short ending segments or the one-window cap limits coverage.
            "replay_stored_ending_fraction": replay.stored_ending_fraction,
            "world_model_sampled_ending_fraction": world_model_sampled_endings / world_model_sampled_transitions,
            "behavior_sampled_ending_fraction": behavior_sampled_endings / behavior_sampled_transitions,
            "world_model_loss": world_model_loss_total / world_updates,
            "image_loss": image_loss_total / world_updates,
            "reward_loss": reward_loss_total / world_updates,
            "reward_mae": reward_mae_total / world_updates,
            "continuation_loss": continuation_loss_total / world_updates,
            "dynamics_kl": dynamics_kl_total / world_updates,
            "representation_kl": representation_kl_total / world_updates,
            "real_nonzero_reward_fraction": real_nonzero_reward_fraction_total / world_updates,
            "predicted_reward_mean": predicted_reward_mean_total / world_updates,
            "predicted_continue_mean": predicted_continue_mean_total / world_updates,
            "world_model_gradient_norm": world_model_gradient_norm_total / world_updates,
            "actor_loss": actor_loss_total / behavior_updates,
            "actor_entropy": actor_entropy_total / behavior_updates,
            "actor_gradient_norm": actor_gradient_norm_total / behavior_updates,
            "critic_loss": critic_loss_total / behavior_updates,
            "critic_prediction_loss": critic_prediction_loss_total / behavior_updates,
            "replay_critic_loss": replay_critic_loss_total / behavior_updates,
            "slow_critic_regularization": slow_critic_regularization_total / behavior_updates,
            "critic_gradient_norm": critic_gradient_norm_total / behavior_updates,
            "imagined_return": imagined_return_total / behavior_updates,
            "imagined_reward_mean": imagined_reward_mean_total / behavior_updates,
            "imagined_continue_mean": imagined_continue_mean_total / behavior_updates,
            "evaluation_mean_return": evaluation_mean_return,
        }
        history.append(record)
        print(
            f"iteration {iteration + 1:02d}/{config.online_iterations} "
            f"actions={record['environment_actions']} "
            f"replay={record['replay_size']} "
            f"model={record['world_model_loss']:.4f} "
            f"reward_mae={record['reward_mae']:.4f} "
            f"nonzero_reward_frac={record['real_nonzero_reward_fraction']:.4f} "
            f"endings(stored/wm/behavior)="
            f"{record['replay_stored_ending_fraction']:.6f}/"
            f"{record['world_model_sampled_ending_fraction']:.6f}/"
            f"{record['behavior_sampled_ending_fraction']:.6f} "
            f"actor={record['actor_loss']:.4f} "
            f"entropy={record['actor_entropy']:.4f} "
            f"eval={record['evaluation_mean_return']}"
        )

        # Save after iterations 10, 20, 30, and so on, and always save the last
        # iteration. The replay data and optimizer states are needed to continue
        # training rather than merely run the trained models.
        if is_tenth_iteration or is_last_iteration:
            checkpoint_path = (config.output_dir / f"checkpoint_iteration_{completed_iterations:04d}.pt")
            torch.save(
                {
                    "completed_iterations": completed_iterations,
                    "world_model": world_model.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "slow_critic": slow_critic.state_dict(),
                    "world_optimizer": world_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "replay_segments": replay.segments,
                    "replay_transition_count": replay.transition_count,
                    "return_scale": return_scale.value,
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to: {checkpoint_path}")

    #END OF MAIN LOOP
    # GPU calls can return before their work finishes. Include that work before
    # stopping the timer. Divide seconds by 60 to report minutes.
    if device == "cuda":
        torch.cuda.synchronize()
    average_iteration_time_minutes = (
        (time.perf_counter() - iterations_started) / len(history) / 60.0
    )

    # -------------------------------------------------------------------------
    # STAGE 6: TEST THE TRAINED ACTOR AND A RANDOM ACTOR IN REAL PONG
    # -------------------------------------------------------------------------
    # No learning happens during this stage.
    collection_env.close()
    episodes = evaluate_pong_actor(world_model, actor, config)
    random_episodes = evaluate_pong_actor(world_model, actor, config, random_actions=True)
    episode_returns = [episode.total_reward for episode in episodes]
    random_returns = [episode.total_reward for episode in random_episodes]

    for episode_number, episode in enumerate(episodes, start=1):
        video_path = save_pong_video(episode.frames, config.output_dir / f"evaluated_pong_episode_{episode_number:02d}.mp4")
        print(f"Saved episode video to: {video_path}")

    # -------------------------------------------------------------------------
    # STAGE 7: SAVE THE FINAL METRICS AND ONE REAL PONG ROLLOUT
    # -------------------------------------------------------------------------
    # `mean_episode_return` averages the Pong score difference across games.
    # Example: returns `[-21, -16, -20]` mean the agent lost by 21, 16, and 20
    # points. Their mean is `(-21 - 16 - 20) / 3 = -19`.
    metrics: dict[str, Any] = {
        "history": history,
        "real_control": {
            "episode_returns": episode_returns,
            "mean_episode_return": float(np.mean(episode_returns)),
            "random_episode_returns": random_returns,
            "random_mean_episode_return": float(np.mean(random_returns)),
        },
    }
    metrics["training_plots"] = save_pong_training_plots(metrics, config.output_dir)
    if device == "cuda":
        torch.cuda.synchronize()
    metrics["total_training_time_minutes"] = (time.perf_counter() - run_started) / 60.0
    metrics["average_iteration_time_minutes"] = average_iteration_time_minutes
    metrics["metrics_path"] = save_metrics(metrics, config)
    return metrics


def report_task_progress(task_name: str, task_fn, *args):
    """Run the exercise while keeping unfinished student work easy to locate."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    config = DreamerWorldModelConfig()
    metrics, complete = report_task_progress(
        "DreamerV3 online Pong exercise", run_online_pong_exercise, config
    )
    if complete:
        print(json.dumps(metrics, indent=2))
        print(f"\nDetailed metrics saved to: {metrics['metrics_path']}")


if __name__ == "__main__":
    main()
