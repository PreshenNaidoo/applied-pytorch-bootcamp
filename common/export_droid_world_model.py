"""Export real DROID RLDS observations for world-model exercise 13.

Run in the optional TensorFlow environment described at the top of exercise 13.
The export needs no PyTorch and never substitutes joint controls for Cartesian poses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def export(builder_dir: Path, output: Path, episodes: int) -> None:
    import cv2
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # TensorFlow is only decoding data; leave the GPU free for the PyTorch run.
    tf.config.set_visible_devices([], "GPU")
    if not (builder_dir / "dataset_info.json").exists():
        raise FileNotFoundError("--builder-dir must name the directory containing dataset_info.json.")
    builder = tfds.builder_from_directory(str(builder_dir))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, episode in enumerate(dataset.take(episodes)):
        images, states, indices = [], [], []
        for step_index, step in enumerate(tfds.as_numpy(episode["steps"])):
            # DROID RLDS is 15 Hz. ceil(15/4)=4 matches the reference sampling rule:
            # actual sampled rate is 3.75 Hz; save it, do not claim exact 4 Hz.
            if step_index % 4:
                continue
            obs = step["observation"]
            pose = np.asarray(obs["cartesian_position"], dtype=np.float32).reshape(-1)
            gripper = np.asarray(obs["gripper_position"], dtype=np.float32).reshape(-1)
            if pose.shape != (6,) or gripper.shape != (1,):
                raise ValueError("Expected Cartesian xyz + Euler angles and one gripper position.")
            image = np.asarray(obs["exterior_image_1_left"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("Expected uint8 RGB exterior camera images.")
            images.append(cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA).transpose(2, 0, 1))
            states.append(np.concatenate([pose, gripper])); indices.append(step_index)
        if len(images) < 8:
            print(f"Skipping source episode {index}: fewer than eight sampled frames.")
            continue
        metadata = episode["episode_metadata"]
        raw_id = metadata["file_path"].numpy().decode("utf-8")
        episode_id = hashlib.sha256(raw_id.encode()).hexdigest()
        path = output / f"episode_{index:04}.npz"
        np.savez_compressed(path, images=np.stack(images), states=np.stack(states),
                            source_frame_indices=np.asarray(indices), episode_id=np.asarray(episode_id),
                            source_path=np.asarray(raw_id), sampled_fps=np.asarray(3.75))
        records.append({"file": path.name, "source_path": raw_id, "episode_id": episode_id,
                        "sampled_frames": len(images), "sampled_fps": 3.75})
        print(f"Exported {path.name}: {len(images)} actual sampled frames")
    (output / "export_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/droid_world_model")
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()
    if args.episodes < 5:
        parser.error("Export at least five episodes for separate train/validation/test groups.")
    export(args.builder_dir, args.output, args.episodes)


if __name__ == "__main__":
    main()
