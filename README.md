# Applied PyTorch Bootcamp

A hands-on set of PyTorch exercises covering model training, computer vision, language models, reinforcement learning,
world models, multimodal learning, embodied AI, and distributed training.

You learn by completing the missing code in each exercise and running its checks.

## Requirements

- Python 3.12 or newer
- A CPU is enough for the early exercises
- A CUDA GPU is recommended for larger vision, language model, and world model exercises

## Setup with uv

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
```

Run the first exercise:

```bash
uv run python exercises/basics/01_pytorch_tensors_basics.py
```

## Setup with pip

Create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python exercises/basics/01_pytorch_tensors_basics.py
```

The distributed exercises use a separate environment. Follow the instructions in
[`exercises/distributed/README.md`](exercises/distributed/README.md) before running them.

## How the Exercises Work

Each exercise is a Python file with explanations, starter code, and checks.

1. Read the introduction at the top of the file.
2. Complete the stubs in order.
3. Run the file after each change.
4. Check the printed shapes, values, and metrics before moving on.

An active `raise NotImplementedError(...)` marks code that you still need to write. The exercises do not include finished
answers.

Exercise numbers restart in each folder. For example, `exercises/vision/01_...` and `exercises/rl/01_...` are separate
exercises.

## Topics

| Folder | What it covers |
| --- | --- |
| [`basics`](exercises/basics) | Tensors, autograd, datasets, training loops, checkpoints, and profiling |
| [`vision`](exercises/vision) | Classification, transfer learning, segmentation, detection, and diffusion |
| [`llms`](exercises/llms) | Tokenization, transformers, fine-tuning, LoRA, retrieval, and evaluation |
| [`multimodal`](exercises/multimodal) | Models that connect images, video, and text |
| [`rl`](exercises/rl) | Gymnasium, Q-learning, DQN, policy gradients, actor-critic methods, and PPO |
| [`world_models`](exercises/world_models) | Learned dynamics, latent prediction, model rollouts, and planning |
| [`embodied_ai`](exercises/embodied_ai) | Robot data, policies, imitation learning, navigation, and action models |
| [`distributed`](exercises/distributed) | Multi-GPU PyTorch training |

Files inside each folder are numbered in a suggested learning order. If you are new to PyTorch, begin with `basics`.

## Data and Outputs

Some exercises download public datasets or pretrained models when they first run. Other exercises explain how to obtain
the data they need.

- Put local datasets in `data/`.
- Generated plots, checkpoints, and other results go in `outputs/`.
- Do not commit downloaded data or generated outputs.

## Answers

Finished solutions are intentionally left out. Read the stub comments, work through one step at a time, and use the
checks in each file to confirm your work.
