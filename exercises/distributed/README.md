# Distributed ViT exercises

This folder teaches Ray Train and PyTorch FSDP2 with a real Vision Transformer
(ViT). The exercises use torchvision's `VisionTransformer` and real CIFAR-100
images. The default model is a ViT-B/16 shape, but the run is kept short so the
focus stays on distributed training.

There is no single-GPU baseline in this folder. Exercise 1 starts with two Ray
workers and two GPUs.

## What “automatic sharding” means here

The code finds every transformer block in the ViT and calls FSDP2 on those
blocks. FSDP2 then decides which part of each parameter, gradient, and optimizer
state belongs to each worker.

You do not write rules such as “put block 0 on GPU 0” or “split this tensor at
this index.” The same sharding function works when the worker count changes.

There are still two choices in your code:

1. which module type is one communication group; these exercises choose a ViT
   `EncoderBlock`
2. which workers form the FSDP mesh; Ray supplies the worker count

After those choices, default FSDP2 splits each parameter along dimension 0 and
handles the local slices. It also shards gradients and optimizer state. You do
not calculate slice boundaries.

FSDP2 does not split the arithmetic inside one matrix multiplication. That is
tensor parallelism, which is a different topic. These exercises focus on the
automatic memory sharding that lets a model use the combined memory of several
GPUs.

These exercises also build the teaching-size model normally on CPU before
sharding it. A model that cannot fit in one worker's CPU memory needs a later
large-model initialization step, usually construction on PyTorch's `meta`
device followed by shard-aware materialization. That topic is outside this
folder; FSDP2 sharding alone does not make initial model construction free.

## Real system and scaled exercise

| Real training system | This folder |
| --- | --- |
| A large ViT | torchvision `VisionTransformer` |
| A very large image dataset | real CIFAR-100 images resized to 224 x 224 |
| Long multi-node training | short Ray Train jobs |
| Many transformer blocks | 12 blocks by default |
| Automatic parameter sharding | PyTorch FSDP2 `fully_shard(...)` |
| Cluster workers | Ray Train workers; they can run on one node or many nodes |

Only the dataset size, epoch count, and number of workers are reduced. The
training path still uses a real image dataset, a real ViT, Ray Train, FSDP2,
distributed metrics, and distributed checkpoints.

## Learning order

1. `01_ray_train_distributed_vit.py` — run a ViT on Ray Train workers.
2. `02_automatic_fsdp2_vit_sharding.py` — let FSDP2 shard every ViT block.
3. `03_fsdp2_vit_memory_controls.py` — compare bfloat16, activation
   checkpointing, and CPU offload.
4. `04_fsdp2_vit_checkpointing.py` — save and restore a sharded model with
   Distributed Checkpoint.
5. `05_ray_train_fsdp2_failure_recovery.py` — restart after one planned worker
   failure.
6. `06_fsdp2_worker_count_resharding.py` — save with one worker count and resume
   with another.
7. `07_ray_train_fsdp2_scaling_study.py` — measure speed and memory with
   1, 2, 4, and 8 workers.

Exercise 1 uses Ray's normal distributed data parallel path so you can first see
what Ray owns: worker startup, ranks, devices, metrics, and results. Exercise 2
changes the model wrapping to FSDP2. It is not a separate plain PyTorch
baseline.

## Hardware and software

- Linux is the normal environment for multi-GPU NCCL training.
- Exercises 1 through 6 need at least 2 CUDA GPUs for their default settings.
- Exercise 7 can record only the worker counts your cluster can provide.
- Use PyTorch 2.8 or newer. The repository's older CUDA 12.1 lock may install
  PyTorch 2.5, which does not expose the public FSDP2 import used here.
- Install the dedicated dependencies from this folder instead of changing the
  main bootcamp environment.
- Enable the current Ray Train API before Python starts with
  `export RAY_TRAIN_V2_ENABLED=1`.

One clean setup is:

```bash
cd exercises/distributed
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export RAY_TRAIN_V2_ENABLED=1
python _environment_check.py
```

Choose the PyTorch CUDA wheel that matches the NVIDIA driver on the machines.
The small requirements file gives minimum versions but does not force one CUDA
wheel index.

CIFAR-100 is downloaded to the repository's `data/` folder. On a multi-node
cluster, point `data_root` at storage that every worker can read, or copy the
dataset to every node before starting the job.

Ray checkpoints also need storage that survives worker loss. A normal local
folder is enough for one machine. Multi-node runs should set Ray's
`RunConfig(storage_path=...)` to shared or object storage.

## Files students should edit

Each numbered file contains guided stubs and active
`raise NotImplementedError(...)` lines. `_shared_vit.py` is provided code. It
keeps dataset and model setup the same across all experiments, so results are
easier to compare. `_environment_check.py` checks versions, CUDA, GPU count,
NCCL, Ray Train, and the public FSDP2 import without starting a training job.

## When only one machine is available

Use two or more GPUs in that machine. Ray still starts one process per GPU, and
the FSDP2 code is the same as it is on several machines. Anyscale is not needed
to complete the local multi-GPU exercises. It becomes useful when you want Ray
to create and manage cloud machines, shared storage, and job services for you.

Useful official references:

- [Ray Train PyTorch guide](https://docs.ray.io/en/latest/train/getting-started-pytorch.html)
- [Ray Train FSDP example](https://docs.ray.io/en/latest/train/examples/pytorch/pytorch-fsdp.html)
- [PyTorch FSDP2 documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [PyTorch Distributed Checkpoint documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
