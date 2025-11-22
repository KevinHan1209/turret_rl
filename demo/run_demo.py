#!/usr/bin/env python3
"""Demo script to run the trained Turret vs Drone SAC agent.

This script loads the trained model and runs episodes, saves them as video,
and displays statistics demonstrating the agent's learned behavior.

Usage:
    python demo/run_demo.py                    # Run 5 episodes, save as video
    python demo/run_demo.py --episodes 10      # Run 10 episodes
    python demo/run_demo.py --video-path out.mp4  # Custom video output path
    python demo/run_demo.py --no-video         # Disable video saving
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from stable_baselines3 import SAC

from turret_rl.envs.turret_env import TurretEnv
from turret_rl.config.config import WorldConfig, RewardConfig


def run_demo(
    model_path: str,
    n_episodes: int = 5,
    render: bool = True,
    save_video: bool = True,
    video_path: str = "demo/demo_video.mp4",
    deterministic: bool = True,
    verbose: bool = True
):
    """Run demonstration episodes with the trained model.

    Args:
        model_path: Path to the trained model (.zip file)
        n_episodes: Number of episodes to run
        render: Whether to render episodes visually
        save_video: Whether to save episodes as video
        video_path: Path to save video file
        deterministic: Use deterministic actions (no exploration)
        verbose: Print episode statistics
    """
    # Load the trained model
    if verbose:
        print("=" * 60)
        print("Turret vs Drone - SAC Agent Demo")
        print("=" * 60)
        print(f"\nLoading model from: {model_path}")

    model = SAC.load(model_path)

    # Create environment with appropriate render mode
    render_mode = "rgb_array" if save_video else ("human" if render else None)
    env = TurretEnv(
        world_config=WorldConfig(),
        reward_config=RewardConfig(),
        render_mode=render_mode
    )

    if verbose:
        print(f"Environment created with render_mode='{render_mode}'")
        print(f"\nRunning {n_episodes} episodes...")
        print("-" * 60)

    # Video recording setup
    frames = []

    # Statistics tracking
    hits = 0
    misses = 0
    no_shots = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step = 0

        shot_fired = False

        while not (done or truncated):
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=deterministic)

            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            # Track if shot was fired
            if info.get('shots_fired', 0) > 0:
                shot_fired = True

            # Collect frames for video
            if save_video and render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            elif render and render_mode == "human":
                env.render()

        # Simulate bullet trajectory after episode ends (for visualization)
        if shot_fired and save_video and render_mode == "rgb_array":
            post_shot_frames = env.simulate_post_shot(n_steps=200)
            frames.extend(post_shot_frames)

        # Track statistics
        total_rewards.append(episode_reward)
        episode_lengths.append(step)

        if info.get('hit', False):
            hits += 1
            result = "HIT"
        elif info.get('miss', False):
            misses += 1
            result = "MISS"
        else:
            no_shots += 1
            result = "NO SHOT"

        if verbose:
            print(f"Episode {episode + 1:3d}: {result:8s} | "
                  f"Reward: {episode_reward:7.3f} | "
                  f"Steps: {step:3d}")

    # Print summary statistics
    if verbose:
        print("-" * 60)
        print("\nSummary Statistics:")
        print(f"  Total Episodes:    {n_episodes}")
        print(f"  Hits:              {hits} ({100*hits/n_episodes:.1f}%)")
        print(f"  Misses:            {misses} ({100*misses/n_episodes:.1f}%)")
        print(f"  No Shots:          {no_shots} ({100*no_shots/n_episodes:.1f}%)")
        print(f"  Hit Rate:          {100*hits/n_episodes:.1f}%")
        print(f"  Avg Reward:        {np.mean(total_rewards):.3f}")
        print(f"  Avg Episode Length: {np.mean(episode_lengths):.1f} steps")
        print("=" * 60)

    # Save video if requested
    if save_video and frames:
        save_frames_as_video(frames, video_path, fps=20)
        if verbose:
            print(f"\nVideo saved to: {video_path}")

    env.close()

    return {
        'hits': hits,
        'misses': misses,
        'no_shots': no_shots,
        'hit_rate': hits / n_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_length': np.mean(episode_lengths)
    }


def save_frames_as_video(frames, output_path, fps=20):
    """Save frames as an MP4 video file.

    Args:
        frames: List of RGB frames (numpy arrays)
        output_path: Path to save the video
        fps: Frames per second
    """
    import imageio

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write video
    with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)


def get_default_model_path():
    """Get the appropriate default model based on platform."""
    import platform
    import os
    
    system = platform.system()
    mac_model = "turret_rl/models/turret_sac/turret_sac_mac_clean.zip"
    linux_model = "turret_rl/models/turret_sac/turret_sac_final.zip"
    
    # Check if Mac model exists when on macOS
    if system == "Darwin" and os.path.exists(mac_model):
        print(f"Detected macOS - using Mac-compatible model: {mac_model}")
        return mac_model
    
    # Default to Linux/original model
    return linux_model

def main():
    parser = argparse.ArgumentParser(
        description="Run demo of trained Turret vs Drone SAC agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        default=get_default_model_path(),
        help="Path to trained model"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable visualization"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video saving"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="demo/demo_output.mp4",
        help="Path to save video"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (with exploration noise)"
    )

    args = parser.parse_args()

    run_demo(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render,
        save_video=not args.no_video,
        video_path=args.video_path,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
