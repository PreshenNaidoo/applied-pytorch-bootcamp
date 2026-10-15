# Applied PyTorch Bootcamp

A practical collection of PyTorch exercises for learning how modern AI systems are built and trained.

The course starts with tensors, datasets, and training loops. It then moves into computer vision, language models,
reinforcement learning, multimodal models, world models, embodied AI, and distributed training.

## Getting Started

You need Python 3.12 or newer. A CPU is enough for the early exercises, while some later exercises are much faster on a
CUDA GPU.

Clone the repo and install the dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/PreshenNaidoo/applied-pytorch-bootcamp.git
cd applied-pytorch-bootcamp
uv sync
```

Run the first exercise:

```bash
uv run python exercises/basics/01_pytorch_tensors_basics.py
```

If you use `pip` instead of `uv`:

```bash
python -m pip install -r requirements.txt
python exercises/basics/01_pytorch_tensors_basics.py
```

## How to Use the Exercises

Each exercise is a Python file with an explanation, code to complete, and checks that help you see whether your work is
correct.

1. Read the introduction at the top of the file.
2. Work through the file in order.
3. Run it after each change.
4. Check the printed shapes, values, and metrics before moving on.

If you are new to PyTorch, begin with the `basics` folder. If you already know the basics, choose the topic you want to
study and follow its numbered files in order.

Exercise numbers restart inside each folder. For example, `vision/01` and `rl/01` are different exercises.

## Topics

| Folder | What you will learn |
| --- | --- |
| [`basics`](exercises/basics) | Tensors, autograd, datasets, training loops, checkpoints, and profiling |
| [`vision`](exercises/vision) | Image classification, transfer learning, segmentation, detection, and diffusion |
| [`llms`](exercises/llms) | Tokenization, transformers, fine-tuning, LoRA, retrieval, and evaluation |
| [`multimodal`](exercises/multimodal) | Models that connect images, video, and text |
| [`rl`](exercises/rl) | Gymnasium, Q-learning, DQN, policy gradients, actor-critic methods, and PPO |
| [`world_models`](exercises/world_models) | Learned dynamics, latent prediction, model rollouts, and planning |
| [`embodied_ai`](exercises/embodied_ai) | Robot data, imitation learning, navigation, and action models |
| [`distributed`](exercises/distributed) | Multi-GPU training with PyTorch and Ray |

Some advanced exercises need extra datasets, pretrained models, or more GPU memory. The introduction at the top of each
exercise explains what it needs.

The distributed exercises use a separate environment. Follow
[`exercises/distributed/README.md`](exercises/distributed/README.md) before running them.
