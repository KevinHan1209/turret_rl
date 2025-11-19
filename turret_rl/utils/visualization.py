"""Visualization utilities for rendering and video recording.

This module provides helper functions for creating videos from environment
frames and managing visualization tasks.
"""

import os
import numpy as np
import imageio
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path


def save_video(
    frames: List[np.ndarray],
    filepath: str,
    fps: int = 20,
    quality: Optional[int] = 8
) -> None:
    """Save a list of frames as a video file.

    Args:
        frames: List of RGB frames (H, W, 3) as numpy arrays
        filepath: Path where to save the video
        fps: Frames per second for the video
        quality: Video quality (1-10, higher is better), used for codec settings
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine format from extension
    extension = os.path.splitext(filepath)[1].lower()

    if extension == '.gif':
        # Save as GIF
        imageio.mimsave(filepath, frames, fps=fps, loop=0)
    elif extension in ['.mp4', '.avi', '.mov']:
        # Save as video using ffmpeg
        writer = imageio.get_writer(
            filepath,
            fps=fps,
            codec='libx264' if extension == '.mp4' else None,
            quality=quality,
            pixelformat='yuv420p'  # For compatibility
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    else:
        # Default to mp4
        filepath = filepath.replace(extension, '.mp4')
        writer = imageio.get_writer(
            filepath,
            fps=fps,
            codec='libx264',
            quality=quality,
            pixelformat='yuv420p'
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    print(f"Video saved to: {filepath}")


def create_episode_summary_frame(
    episode_stats: dict,
    figsize: tuple = (10, 6)
) -> np.ndarray:
    """Create a summary frame showing episode statistics.

    Args:
        episode_stats: Dictionary containing episode statistics
        figsize: Size of the figure (width, height)

    Returns:
        RGB array of the summary frame
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Episode Summary', fontsize=16, fontweight='bold')

    # Episode info
    ax = axes[0, 0]
    ax.axis('off')
    info_text = f"Episode Length: {episode_stats.get('length', 0)} steps\n"
    info_text += f"Total Reward: {episode_stats.get('total_reward', 0):.3f}\n"
    info_text += f"Result: {'HIT' if episode_stats.get('hit', False) else 'MISS'}\n"
    info_text += f"Shots Fired: {episode_stats.get('shots_fired', 0)}\n"
    info_text += f"Drone Speed: {episode_stats.get('drone_speed', 0):.1f} m/s"

    ax.text(
        0.1, 0.5, info_text,
        fontsize=12,
        verticalalignment='center',
        fontweight='bold' if episode_stats.get('hit', False) else 'normal'
    )

    # Reward over time
    ax = axes[0, 1]
    if 'rewards' in episode_stats and len(episode_stats['rewards']) > 0:
        ax.plot(episode_stats['rewards'], 'b-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward over Time')
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')

    # Distance to drone over time
    ax = axes[1, 0]
    if 'distances' in episode_stats and len(episode_stats['distances']) > 0:
        ax.plot(episode_stats['distances'], 'r-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Distance (m)')
        ax.set_title('Distance to Drone')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Min approach dist')
        ax.legend()
    else:
        ax.axis('off')

    # Shooting accuracy
    ax = axes[1, 1]
    if episode_stats.get('shots_fired', 0) > 0:
        hit_rate = 1.0 if episode_stats.get('hit', False) else 0.0
        colors = ['red', 'green'] if hit_rate > 0 else ['red']
        sizes = [1 - hit_rate, hit_rate] if hit_rate > 0 else [1.0]
        labels = ['Miss', 'Hit'] if hit_rate > 0 else ['Miss']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Shooting Accuracy\n({episode_stats.get('shots_fired', 0)} shots)")
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No Shots Fired', ha='center', va='center', fontsize=14)

    plt.tight_layout()

    # Convert to RGB array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    return img[:, :, :3]  # Remove alpha channel


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: tuple = (10, 10),
    font_size: int = 20,
    color: tuple = (255, 255, 255),
    background_color: Optional[tuple] = (0, 0, 0),
    alpha: float = 0.7
) -> np.ndarray:
    """Add text overlay to a frame.

    This is a simple implementation. For better text rendering,
    consider using cv2.putText if OpenCV is available.

    Args:
        frame: RGB frame to add text to
        text: Text to overlay
        position: (x, y) position for text (top-left corner)
        font_size: Size of the font
        color: RGB color for text
        background_color: RGB color for background (None for transparent)
        alpha: Background transparency (0=transparent, 1=opaque)

    Returns:
        Frame with text overlay
    """
    # This is a placeholder implementation
    # In practice, you might want to use PIL or OpenCV for better text rendering
    # For now, we'll just return the frame as-is with a note
    # that text overlay would be added here

    # TODO: Implement actual text overlay using PIL or matplotlib
    # For now, just return the original frame
    return frame.copy()


def create_comparison_video(
    episodes: List[List[np.ndarray]],
    labels: List[str],
    output_path: str,
    fps: int = 20,
    arrangement: str = 'horizontal'
) -> None:
    """Create a comparison video showing multiple episodes side by side.

    Args:
        episodes: List of episode frame lists
        labels: Labels for each episode
        output_path: Where to save the comparison video
        fps: Frames per second
        arrangement: How to arrange videos ('horizontal', 'vertical', or 'grid')
    """
    if len(episodes) == 0:
        raise ValueError("No episodes provided for comparison")

    # Ensure all episodes have the same number of frames
    max_frames = max(len(ep) for ep in episodes)
    padded_episodes = []

    for ep in episodes:
        if len(ep) < max_frames:
            # Pad with last frame
            last_frame = ep[-1] if len(ep) > 0 else np.zeros_like(episodes[0][0])
            padded_ep = ep + [last_frame] * (max_frames - len(ep))
            padded_episodes.append(padded_ep)
        else:
            padded_episodes.append(ep)

    # Combine frames based on arrangement
    combined_frames = []
    for i in range(max_frames):
        frames_at_i = [ep[i] for ep in padded_episodes]

        if arrangement == 'horizontal':
            combined = np.hstack(frames_at_i)
        elif arrangement == 'vertical':
            combined = np.vstack(frames_at_i)
        elif arrangement == 'grid':
            # Simple 2x2 grid for up to 4 episodes
            n = len(frames_at_i)
            if n <= 2:
                combined = np.hstack(frames_at_i) if n == 2 else frames_at_i[0]
            elif n <= 4:
                top = np.hstack(frames_at_i[:2])
                if n == 3:
                    frames_at_i.append(np.zeros_like(frames_at_i[0]))
                bottom = np.hstack(frames_at_i[2:4])
                combined = np.vstack([top, bottom])
            else:
                # For more than 4, just use horizontal
                combined = np.hstack(frames_at_i[:4])
        else:
            raise ValueError(f"Unknown arrangement: {arrangement}")

        combined_frames.append(combined)

    # Save the combined video
    save_video(combined_frames, output_path, fps=fps)


def plot_training_curves(
    log_path: str,
    output_path: Optional[str] = None,
    metrics: List[str] = ['ep_rew_mean', 'ep_len_mean']
) -> None:
    """Plot training curves from Stable-Baselines3 logs.

    Args:
        log_path: Path to the CSV log file
        output_path: Where to save the plot (None to display)
        metrics: List of metrics to plot
    """
    import pandas as pd

    # Read the log file
    df = pd.read_csv(log_path)

    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax = axes[i]
            ax.plot(df['time/total_timesteps'], df[metric], 'b-', linewidth=2)
            ax.set_xlabel('Total Timesteps')
            ax.set_ylabel(metric)
            ax.set_title(f'Training Progress: {metric}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Training curves saved to: {output_path}")
    else:
        plt.show()

    plt.close()


class VideoRecorder:
    """Helper class for recording videos during training or evaluation."""

    def __init__(
        self,
        output_dir: str = 'videos',
        fps: int = 20,
        prefix: str = 'episode'
    ):
        """Initialize the video recorder.

        Args:
            output_dir: Directory to save videos
            fps: Frames per second for videos
            prefix: Prefix for video filenames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.prefix = prefix
        self.frames = []
        self.episode_count = 0

    def reset(self) -> None:
        """Reset the recorder for a new episode."""
        self.frames = []

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the current episode recording.

        Args:
            frame: RGB frame to add
        """
        self.frames.append(frame)

    def save_episode(
        self,
        suffix: Optional[str] = None,
        episode_stats: Optional[dict] = None
    ) -> str:
        """Save the current episode as a video.

        Args:
            suffix: Optional suffix for the filename
            episode_stats: Optional statistics to include as summary frame

        Returns:
            Path to the saved video
        """
        if len(self.frames) == 0:
            print("Warning: No frames to save")
            return ""

        # Add summary frame if stats provided
        if episode_stats:
            summary_frame = create_episode_summary_frame(episode_stats)
            # Show summary for a second
            for _ in range(self.fps):
                self.frames.append(summary_frame)

        # Generate filename
        self.episode_count += 1
        if suffix:
            filename = f"{self.prefix}_{self.episode_count:03d}_{suffix}.mp4"
        else:
            filename = f"{self.prefix}_{self.episode_count:03d}.mp4"

        filepath = str(self.output_dir / filename)

        # Save video
        save_video(self.frames, filepath, fps=self.fps)

        # Reset for next episode
        self.reset()

        return filepath