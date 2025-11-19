"""Evaluation and video recording script for trained RL agents.

This module loads trained models (PPO or SAC), evaluates their performance over
multiple episodes, and records videos of the agent's behavior.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime
import zipfile

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from turret_rl.envs.turret_env import TurretEnv
from turret_rl.config.config import (
    WorldConfig,
    RewardConfig,
    EvaluationConfig,
    DEFAULT_EVAL_CONFIG
)
from turret_rl.utils.visualization import VideoRecorder, save_video


def _detect_algorithm(model_path: str) -> str:
    """Detect which algorithm was used to train the model.

    Args:
        model_path: Path to the trained model

    Returns:
        Algorithm name: 'PPO' or 'SAC'
    """
    # Try to detect from the model file itself
    try:
        with zipfile.ZipFile(model_path, 'r') as archive:
            # Check for algorithm-specific files
            namelist = archive.namelist()
            if 'policy.optimizer.pth' in namelist:
                # Read the data to check structure
                data = archive.read('data')
                if b'SACPolicy' in data or b'sac' in data.lower():
                    return 'SAC'
                elif b'ActorCriticPolicy' in data or b'ppo' in data.lower():
                    return 'PPO'
    except Exception:
        pass

    # Fallback: check filename
    model_name = Path(model_path).stem.lower()
    if 'sac' in model_name:
        return 'SAC'
    elif 'ppo' in model_name:
        return 'PPO'

    # Default to SAC (new default algorithm)
    print("Warning: Could not detect algorithm from model file. Defaulting to SAC.")
    return 'SAC'


def evaluate_agent(
    model_path: str,
    world_config: Optional[WorldConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    eval_config: Optional[EvaluationConfig] = None,
    vec_normalize_path: Optional[str] = None,
    seed: Optional[int] = 42,
    deterministic: bool = True,
    verbose: int = 1,
    algorithm: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate a trained agent and record videos.

    Args:
        model_path: Path to the trained model
        world_config: World configuration
        reward_config: Reward configuration
        eval_config: Evaluation configuration
        vec_normalize_path: Path to VecNormalize statistics (if used in training)
        seed: Random seed for reproducibility
        deterministic: Whether to use deterministic actions
        verbose: Verbosity level
        algorithm: Algorithm used ('PPO' or 'SAC'). Auto-detected if None.

    Returns:
        Dictionary containing evaluation statistics
    """
    # Use default configs if not provided
    eval_config = eval_config or DEFAULT_EVAL_CONFIG
    world_config = world_config or WorldConfig()
    reward_config = reward_config or RewardConfig()

    # Detect algorithm if not specified
    if algorithm is None:
        algorithm = _detect_algorithm(model_path)

    print("=" * 50)
    print("Evaluating Trained Turret Agent")
    print("=" * 50)
    print(f"\nAlgorithm: {algorithm}")
    print(f"Model path: {model_path}")
    print(f"Episodes to evaluate: {eval_config.n_eval_episodes}")
    print(f"Deterministic actions: {deterministic}")
    print(f"Recording videos: {eval_config.record_video}")
    if eval_config.record_video:
        print(f"Video output folder: {eval_config.video_folder}")
    print("=" * 50 + "\n")

    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"Loading {algorithm} model...")
    if algorithm.upper() == 'SAC':
        model = SAC.load(model_path)
    elif algorithm.upper() == 'PPO':
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'PPO' or 'SAC'.")

    # Create evaluation environment
    render_mode = "rgb_array" if eval_config.record_video else None
    env = TurretEnv(
        world_config=world_config,
        reward_config=reward_config,
        render_mode=render_mode
    )

    # Wrap with Monitor for logging
    env = Monitor(env)

    # Apply VecNormalize if it was used during training
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"Loading normalization statistics from: {vec_normalize_path}")
        # Wrap in DummyVecEnv first
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False  # Set to evaluation mode
        vec_env.norm_reward = False  # Don't normalize rewards during evaluation
    else:
        # Just use the environment as-is
        vec_env = None

    # Initialize video recorder if needed
    if eval_config.record_video:
        video_dir = Path(eval_config.video_folder)
        video_dir.mkdir(parents=True, exist_ok=True)
        video_recorder = VideoRecorder(
            output_dir=str(video_dir),
            fps=eval_config.video_fps,
            prefix=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print(f"Video recorder initialized. Output directory: {video_dir}\n")

    # Evaluation statistics
    all_rewards = []
    all_lengths = []
    all_hits = []
    all_shots_fired = []
    all_drone_speeds = []

    # Run evaluation episodes
    for episode_idx in range(eval_config.n_eval_episodes):
        print(f"\nEpisode {episode_idx + 1}/{eval_config.n_eval_episodes}")
        print("-" * 30)

        # Reset environment
        if vec_env:
            obs = vec_env.reset()
            current_env = vec_env.envs[0].env  # Access the underlying environment
        else:
            obs, info = env.reset(seed=seed + episode_idx if seed else None)
            current_env = env

        # Episode tracking
        episode_reward = 0
        episode_length = 0
        episode_frames = []
        episode_rewards = []
        episode_distances = []
        shots_fired = 0
        drone_speed = info.get('drone_speed', 0) if not vec_env else 0

        # Run episode
        done = False
        while not done:
            # Get action from model
            if vec_env:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = vec_env.step(action)
                # Extract scalar values
                reward = reward[0]
                done = done[0]
                info = info[0]
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Track statistics
            episode_reward += reward
            episode_length += 1
            episode_rewards.append(reward)

            if 'drone_distance' in info:
                episode_distances.append(info['drone_distance'])

            if info.get('shots_fired', 0) > 0:
                shots_fired += info['shots_fired']

            # Capture frame for video
            if eval_config.record_video:
                # Get frame from the actual environment
                frame = current_env.render()
                if frame is not None:
                    episode_frames.append(frame)
                    video_recorder.add_frame(frame)

            # Check for hit
            if done and info.get('hit', False):
                hit = True
            elif done:
                hit = False

        # Get drone speed from final info if not already captured
        if drone_speed == 0 and 'drone_speed' in info:
            drone_speed = info['drone_speed']

        # Episode complete
        print(f"  Result: {'HIT ✓' if hit else 'MISS ✗'}")
        print(f"  Episode length: {episode_length} steps")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Shots fired: {shots_fired}")
        if drone_speed > 0:
            print(f"  Drone speed: {drone_speed:.1f} m/s")

        # Store statistics
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        all_hits.append(hit)
        all_shots_fired.append(shots_fired)
        if drone_speed > 0:
            all_drone_speeds.append(drone_speed)

        # Save video for this episode
        if eval_config.record_video and len(episode_frames) > 0:
            # Create episode statistics for summary
            episode_stats = {
                'length': episode_length,
                'total_reward': episode_reward,
                'hit': hit,
                'shots_fired': shots_fired,
                'drone_speed': drone_speed,
                'rewards': episode_rewards,
                'distances': episode_distances
            }

            # Save the video
            suffix = "hit" if hit else "miss"
            video_path = video_recorder.save_episode(
                suffix=suffix,
                episode_stats=episode_stats if eval_config.show_info else None
            )
            if video_path:
                print(f"  Video saved: {video_path}")

    # Calculate aggregate statistics
    hit_rate = sum(all_hits) / len(all_hits) if all_hits else 0
    avg_reward = np.mean(all_rewards) if all_rewards else 0
    std_reward = np.std(all_rewards) if all_rewards else 0
    avg_length = np.mean(all_lengths) if all_lengths else 0
    avg_shots = np.mean(all_shots_fired) if all_shots_fired else 0
    avg_drone_speed = np.mean(all_drone_speeds) if all_drone_speeds else 0

    # Print summary statistics
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Total episodes: {eval_config.n_eval_episodes}")
    print(f"Hit rate: {hit_rate:.1%} ({sum(all_hits)}/{len(all_hits)})")
    print(f"Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"Average episode length: {avg_length:.1f} steps")
    print(f"Average shots fired: {avg_shots:.1f}")
    if avg_drone_speed > 0:
        print(f"Average drone speed: {avg_drone_speed:.1f} m/s")

    # Performance breakdown
    if len(all_hits) > 1:
        print("\nPerformance by outcome:")
        hit_rewards = [r for r, h in zip(all_rewards, all_hits) if h]
        miss_rewards = [r for r, h in zip(all_rewards, all_hits) if not h]
        if hit_rewards:
            print(f"  Hits: avg reward = {np.mean(hit_rewards):.3f}")
        if miss_rewards:
            print(f"  Misses: avg reward = {np.mean(miss_rewards):.3f}")

    # Clean up
    env.close()
    if vec_env:
        vec_env.close()

    # Return evaluation statistics
    return {
        'n_episodes': eval_config.n_eval_episodes,
        'hit_rate': hit_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'avg_shots': avg_shots,
        'avg_drone_speed': avg_drone_speed,
        'all_rewards': all_rewards,
        'all_hits': all_hits
    }


def compare_models(
    model_paths: List[str],
    labels: Optional[List[str]] = None,
    n_episodes: int = 10,
    seed: Optional[int] = 42
) -> None:
    """Compare multiple trained models.

    Args:
        model_paths: List of paths to models to compare
        labels: Labels for each model (optional)
        n_episodes: Number of episodes to evaluate each model
        seed: Random seed
    """
    if not labels:
        labels = [f"Model {i+1}" for i in range(len(model_paths))]

    print("=" * 50)
    print("Model Comparison")
    print("=" * 50)

    results = []
    for path, label in zip(model_paths, labels):
        print(f"\nEvaluating: {label}")
        print(f"Path: {path}")

        eval_config = EvaluationConfig(
            n_eval_episodes=n_episodes,
            record_video=False  # Disable video for comparison
        )

        stats = evaluate_agent(
            model_path=path,
            eval_config=eval_config,
            seed=seed,
            verbose=0
        )
        stats['label'] = label
        results.append(stats)

    # Print comparison table
    print("\n" + "=" * 50)
    print("Comparison Results")
    print("=" * 50)
    print(f"{'Model':<20} {'Hit Rate':<12} {'Avg Reward':<12} {'Avg Length':<12}")
    print("-" * 56)

    for r in results:
        print(f"{r['label']:<20} "
              f"{r['hit_rate']:.1%}{'':^7} "
              f"{r['avg_reward']:.3f}{'':^7} "
              f"{r['avg_length']:.1f}")


def main():
    """Main entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained RL agent (PPO or SAC)")
    parser.add_argument(
        "--model",
        type=str,
        default="turret_rl/models/turret_sac/turret_sac.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=['PPO', 'SAC', 'ppo', 'sac'],
        default=None,
        help="Algorithm used to train the model (auto-detected if not specified)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="turret_rl/videos",
        help="Directory to save videos"
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to VecNormalize statistics file"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Compare multiple models (provide paths)"
    )

    args = parser.parse_args()

    if args.compare:
        # Run model comparison
        compare_models(
            model_paths=args.compare,
            n_episodes=args.episodes,
            seed=args.seed
        )
    else:
        # Check if vec_normalize file exists in default location
        vec_norm_path = args.vec_normalize
        if vec_norm_path is None:
            # Try to find it in the same directory as the model
            model_dir = Path(args.model).parent
            default_vec_norm = model_dir / "vec_normalize.pkl"
            if default_vec_norm.exists():
                vec_norm_path = str(default_vec_norm)
                print(f"Found VecNormalize file: {vec_norm_path}")

        # Create evaluation config
        eval_config = EvaluationConfig(
            n_eval_episodes=args.episodes,
            record_video=not args.no_video,
            video_folder=args.video_dir
        )

        # Run evaluation
        evaluate_agent(
            model_path=args.model,
            eval_config=eval_config,
            vec_normalize_path=vec_norm_path,
            deterministic=not args.stochastic,
            seed=args.seed,
            algorithm=args.algorithm.upper() if args.algorithm else None
        )


if __name__ == "__main__":
    main()