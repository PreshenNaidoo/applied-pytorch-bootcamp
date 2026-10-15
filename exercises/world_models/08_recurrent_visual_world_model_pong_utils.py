"""Small video helpers for the recurrent Pong world-model exercise."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_pong_training_plots(
    metrics: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    """Save the main Pong training diagnostics as separate PNG files."""
    history = metrics.get("history", [])
    if not history:
        raise ValueError("Training history is needed to create plots.")

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    environment_steps = np.asarray(
        [record["environment_actions"] for record in history], dtype=np.float64
    )
    saved_paths: dict[str, str] = {}

    def save_figure(name: str, figure: plt.Figure) -> None:
        path = plot_dir / f"{name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        saved_paths[name] = str(path)

    # Real Pong performance. Evaluation is only run every ten iterations, so
    # records from the other iterations contain `None` and are left out.
    evaluation_by_step = {
        record["environment_actions"]: record["evaluation_mean_return"]
        for record in history
        if record.get("evaluation_mean_return") is not None
    }
    # The training loop also evaluates the finished actor after its last
    # iteration. Add that result so short runs still have one evaluation point.
    final_return = metrics.get("real_control", {}).get("mean_episode_return")
    if final_return is not None:
        evaluation_by_step[history[-1]["environment_actions"]] = final_return
    evaluation_points = sorted(evaluation_by_step.items())
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if evaluation_points:
        evaluation_frames, evaluation_returns = zip(*evaluation_points)
        axis.plot(evaluation_frames, evaluation_returns, marker="o", label="Learned actor")
    random_return = metrics.get("real_control", {}).get("random_mean_episode_return")
    if random_return is not None:
        axis.axhline(random_return, color="tab:gray", linestyle="--", label="Random actor")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Real Pong evaluation", xlabel="Environment steps", ylabel="Mean episode return")
    if evaluation_points or random_return is not None:
        axis.legend()
    axis.grid(alpha=0.25)
    save_figure("pong_evaluation_return", figure)

    # Keep each loss in its own panel because their useful numeric ranges can
    # differ greatly.
    loss_names = [
        ("world_model_loss", "Total world-model loss"),
        ("image_loss", "Image loss"),
        ("reward_loss", "Reward loss"),
        ("continuation_loss", "Continuation loss"),
        ("dynamics_kl", "Dynamics KL"),
        ("representation_kl", "Representation KL"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    for axis, (key, title) in zip(axes.flat, loss_names):
        axis.plot(environment_steps, [record[key] for record in history])
        axis.set_title(title)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Environment steps")
    save_figure("pong_world_model_losses", figure)

    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(
        environment_steps,
        [record["real_nonzero_reward_fraction"] for record in history],
    )
    axes[0].set(title="Non-zero rewards in sampled replay", ylabel="Fraction")
    axes[1].plot(environment_steps, [record["reward_mae"] for record in history])
    axes[1].set(title="Reward prediction error", xlabel="Environment steps", ylabel="MAE")
    for axis in axes:
        axis.grid(alpha=0.25)
    save_figure("pong_reward_diagnostics", figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(environment_steps, [record["actor_entropy"] for record in history])
    axis.axhline(np.log(6.0), color="tab:gray", linestyle="--", label="Maximum for 6 actions")
    axis.set(title="Actor entropy", xlabel="Environment steps", ylabel="Entropy")
    axis.legend()
    axis.grid(alpha=0.25)
    save_figure("pong_actor_entropy", figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        environment_steps,
        [record["imagined_return"] for record in history],
        label="Mean lambda-return",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Mean return in imagined rollouts", xlabel="Environment steps", ylabel="Lambda-return")
    axis.legend()
    axis.grid(alpha=0.25)
    save_figure("pong_imagined_return", figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for key, label in (
        ("world_model_gradient_norm", "World model"),
        ("actor_gradient_norm", "Actor"),
        ("critic_gradient_norm", "Critic"),
    ):
        axis.plot(environment_steps, [record[key] for record in history], label=label)
    axis.set_yscale("symlog", linthresh=1e-3)
    axis.set(title="Gradient norms before clipping", xlabel="Environment steps", ylabel="Gradient norm")
    axis.legend()
    axis.grid(alpha=0.25)
    save_figure("pong_gradient_norms", figure)

    return saved_paths


def save_pong_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    frames_per_second: int = 15,
    display_scale: int = 4,
) -> str:
    """Save RGB observations returned by Pong as an MP4 video."""
    if not frames:
        raise ValueError("At least one frame is needed to create a video.")
    if frames_per_second < 1 or display_scale < 1:
        raise ValueError("frames_per_second and display_scale must be positive.")

    try:
        import av
    except ImportError as error:
        raise ImportError("Saving a Pong video needs PyAV. Run: uv sync") from error

    first_frame = np.asarray(frames[0], dtype=np.uint8)
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError("Each video frame must have shape [height, width, 3].")

    height, width, _ = first_frame.shape
    video_path = Path(output_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=frames_per_second)
        stream.width = width * display_scale
        stream.height = height * display_scale
        stream.pix_fmt = "yuv420p"

        for frame_index, frame in enumerate(frames):
            rgb_frame = np.asarray(frame, dtype=np.uint8)
            if rgb_frame.shape != first_frame.shape:
                raise ValueError(
                    f"Frame {frame_index} has shape {rgb_frame.shape}; "
                    f"expected {first_frame.shape}."
                )
            # The observations are only 64x64. Repeating the pixels makes the
            # saved video easier to see without changing what happened in Pong.
            enlarged_frame = np.repeat(
                np.repeat(rgb_frame, display_scale, axis=0),
                display_scale,
                axis=1,
            )
            video_frame = av.VideoFrame.from_ndarray(enlarged_frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        # Encoding keeps a few frames temporarily. This writes those last
        # frames before the video file is closed.
        for packet in stream.encode():
            container.mux(packet)

    return str(video_path)


def record_random_pong_video(
    env: Any | None,
    config: Any,
    output_path: str | Path | None = None,
    number_of_steps: int | None = None,
) -> str:
    """Play Pong with random actions and save the returned observations."""
    steps = config.seed_steps if number_of_steps is None else number_of_steps
    if steps < 1:
        raise ValueError("number_of_steps must be positive.")

    owns_env = env is None
    if env is None:
        try:
            import ale_py
            import gymnasium as gym
            from gymnasium.wrappers import AtariPreprocessing
        except ImportError as error:
            raise ImportError("Pong needs the Atari dependency. Run: uv sync") from error

        gym.register_envs(ale_py)
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
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)

    try:
        seed = config.random_state + 1
        env.action_space.seed(seed)
        observation, _ = env.reset(seed=seed)
        frames = [np.asarray(observation, dtype=np.uint8).copy()]

        for _ in range(steps):
            action_id = int(env.action_space.sample())
            observation, _, terminated, truncated, _ = env.step(action_id)
            frames.append(np.asarray(observation, dtype=np.uint8).copy())

            if terminated or truncated:
                break

        if output_path is None:
            output_path = config.output_dir / "random_pong_game.mp4"

        video_path = save_pong_video(frames, output_path)
        print(f"Saved video to: {video_path}")
        return video_path
    finally:
        if owns_env:
            env.close()
