"""Custom callbacks for training monitoring and video recording.

This module provides callbacks for Stable-Baselines3 training that record
evaluation videos at regular intervals during training.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize, sync_envs_normalization

from ..envs.turret_env import TurretEnv
from ..utils.visualization import save_video, create_episode_summary_frame
from ..config.config import WorldConfig, RewardConfig


class VideoRecorderCallback(BaseCallback):
    """Records evaluation videos at regular intervals during training.

    This callback creates a separate evaluation environment with video recording
    enabled and runs evaluation episodes at specified intervals. Videos are saved
    with metadata about training progress.

    Example:
        callback = VideoRecorderCallback(
            eval_freq=10000,  # Record every 10k timesteps
            n_eval_episodes=3,
            video_folder="training_videos"
        )
        model.learn(total_timesteps=1000000, callback=callback)
    """

    def __init__(
        self,
        eval_freq: int = 10000,
        n_eval_episodes: int = 1,
        video_folder: str = "turret_rl/videos/training",
        video_length: Optional[int] = None,
        deterministic: bool = True,
        world_config: Optional[WorldConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        verbose: int = 1,
        render_fps: int = 20,
        save_video_on_best: bool = True,
        name_prefix: str = "training_eval"
    ):
        """Initialize the video recorder callback.

        Args:
            eval_freq: Evaluate and record video every n timesteps
            n_eval_episodes: Number of episodes to record at each interval
            video_folder: Directory to save videos
            video_length: Maximum length of each episode (None for unlimited)
            deterministic: Whether to use deterministic actions during evaluation
            world_config: World configuration for eval environment
            reward_config: Reward configuration for eval environment
            verbose: Verbosity level (0: none, 1: info)
            render_fps: FPS for saved videos
            save_video_on_best: Save additional video when best model is achieved
            name_prefix: Prefix for video filenames
        """
        super().__init__(verbose)

        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.video_folder = Path(video_folder)
        self.video_length = video_length
        self.deterministic = deterministic
        self.world_config = world_config or WorldConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_fps = render_fps
        self.save_video_on_best = save_video_on_best
        self.name_prefix = name_prefix

        # Create video directory
        self.video_folder.mkdir(parents=True, exist_ok=True)

        # Tracking variables
        self.last_eval_timestep = 0
        self.eval_count = 0
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []

        # Evaluation environment (created on first use)
        self.eval_env = None

    def _init_eval_env(self) -> None:
        """Initialize the evaluation environment with rendering enabled."""
        if self.eval_env is None:
            # Store the raw environment directly - we'll handle VecNormalize manually
            self.raw_env = TurretEnv(
                world_config=self.world_config,
                reward_config=self.reward_config,
                render_mode="rgb_array"
            )

            # Keep VecNormalize stats if training uses it
            self.vec_normalize = self.model.get_vec_normalize_env()
            self.eval_env = self.raw_env  # Use raw env directly

            if self.verbose > 0:
                print(f"Initialized evaluation environment for video recording")

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.verbose > 0:
            print(f"VideoRecorderCallback: Will record videos every {self.eval_freq} timesteps")
            print(f"Videos will be saved to: {self.video_folder}")

    def _on_step(self) -> bool:
        """Called at each step during training.

        Returns:
            True to continue training
        """
        # Check if it's time to record a video
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self._record_evaluation_videos()
            self.last_eval_timestep = self.num_timesteps

        return True

    def _record_evaluation_videos(self) -> None:
        """Record evaluation episodes and save as videos."""
        self._init_eval_env()
        self.eval_count += 1

        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Recording Evaluation Videos (Eval #{self.eval_count})")
            print(f"Timestep: {self.num_timesteps:,}")
            print(f"{'='*60}")

        episode_rewards_current = []
        episode_lengths_current = []
        episode_hits = []

        # Record n_eval_episodes
        for episode_idx in range(self.n_eval_episodes):
            frames = []
            episode_reward = 0.0
            episode_length = 0
            episode_info = {
                'rewards': [],
                'distances': [],
                'drone_speed': 0
            }
            hit = False
            shot_fired = False

            # Reset the RAW environment
            obs, info = self.raw_env.reset()

            # If training uses VecNormalize, normalize the observation
            if self.vec_normalize is not None:
                obs = self.vec_normalize.normalize_obs(obs)

            # Capture initial frame from RAW environment
            frame = self.raw_env.render()
            if frame is not None:
                frames.append(frame)

            # Run episode
            done = False
            while not done and (self.video_length is None or episode_length < self.video_length):
                # Get action from model (model expects normalized obs if using VecNormalize)
                action, _ = self.model.predict(obs, deterministic=self.deterministic)

                # Step the RAW environment
                obs, reward, terminated, truncated, step_info = self.raw_env.step(action)
                done = terminated or truncated

                # If training uses VecNormalize, normalize the observation for next prediction
                if self.vec_normalize is not None:
                    obs = self.vec_normalize.normalize_obs(obs)

                # Track statistics
                episode_reward += reward
                episode_length += 1
                episode_info['rewards'].append(reward)

                if 'drone_distance' in step_info:
                    episode_info['distances'].append(step_info['drone_distance'])

                if step_info.get('hit', False):
                    hit = True

                if step_info.get('shot_taken', False):
                    shot_fired = True

                # Capture frame from RAW environment
                frame = self.raw_env.render()
                if frame is not None:
                    frames.append(frame)

            # If shot was fired, continue simulating for visualization
            # (shows bullet trajectory and full drone trajectory)
            if shot_fired:
                post_shot_frames = self.raw_env.simulate_post_shot(n_steps=200)
                frames.extend(post_shot_frames)

            # Store episode statistics
            episode_rewards_current.append(episode_reward)
            episode_lengths_current.append(episode_length)
            episode_hits.append(hit)

            # Create episode summary
            episode_stats = {
                'length': episode_length,
                'total_reward': episode_reward,
                'hit': hit,
                'shots_fired': step_info.get('active_bullets', 0),
                'drone_speed': episode_info['drone_speed'],
                'rewards': episode_info['rewards'],
                'distances': episode_info['distances']
            }

            # Add summary frame at the end
            summary_frame = create_episode_summary_frame(episode_stats)
            # Show summary for 1 second
            for _ in range(self.render_fps):
                frames.append(summary_frame)

            # Save video
            result_str = "hit" if hit else "miss"
            video_filename = (
                f"{self.name_prefix}_"
                f"step{self.num_timesteps:09d}_"
                f"eval{self.eval_count:03d}_"
                f"ep{episode_idx+1:02d}_"
                f"{result_str}.mp4"
            )
            video_path = self.video_folder / video_filename

            save_video(frames, str(video_path), fps=self.render_fps)

            if self.verbose > 0:
                print(f"  Episode {episode_idx+1}/{self.n_eval_episodes}: "
                      f"Reward={episode_reward:.3f}, "
                      f"Length={episode_length}, "
                      f"Result={'HIT ✓' if hit else 'MISS ✗'}")

        # Calculate statistics
        mean_reward = np.mean(episode_rewards_current)
        std_reward = np.std(episode_rewards_current)
        mean_length = np.mean(episode_lengths_current)
        hit_rate = np.mean(episode_hits)

        # Store for later analysis
        self.episode_rewards.extend(episode_rewards_current)
        self.episode_lengths.extend(episode_lengths_current)

        # Log to tensorboard if available
        if self.logger is not None:
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_length", mean_length)
            self.logger.record("eval/hit_rate", hit_rate)

        if self.verbose > 0:
            print(f"\nEvaluation Results:")
            print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"  Mean Length: {mean_length:.1f} steps")
            print(f"  Hit Rate: {hit_rate:.1%}")
            print(f"  Videos saved to: {self.video_folder}")
            print(f"{'='*60}\n")

        # Check if this is the best performance
        if self.save_video_on_best and mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print(f"🏆 New best mean reward: {mean_reward:.3f}!")
                print(f"   (Previous best: {self.best_mean_reward:.3f})")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.eval_env is not None:
            self.eval_env.close()

        if self.verbose > 0:
            print(f"\nVideoRecorderCallback Summary:")
            print(f"  Total evaluations: {self.eval_count}")
            print(f"  Videos saved: {len(self.episode_rewards)}")
            print(f"  Best mean reward: {self.best_mean_reward:.3f}")
            print(f"  All videos saved to: {self.video_folder}")


class TrainingProgressCallback(BaseCallback):
    """Enhanced progress callback with hit rate tracking.

    Tracks and logs detailed training statistics including hit rate,
    episode lengths, and rewards. Can be used alongside VideoRecorderCallback.
    """

    def __init__(self, verbose: int = 1, log_freq: int = 100):
        """Initialize the callback.

        Args:
            verbose: Verbosity level
            log_freq: Log progress every n episodes
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.n_episodes = 0
        self.n_hits = 0
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        """Called at each step of the environment.

        Returns:
            True to continue training
        """
        # Check for completed episodes
        if len(self.locals.get('dones', [])) > 0:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    # Episode completed
                    self.n_episodes += 1

                    # Get episode info
                    info = self.locals.get('infos', [{}])[i]
                    if info.get('hit', False):
                        self.n_hits += 1

                    # Log progress periodically
                    if self.n_episodes % self.log_freq == 0 and self.verbose > 0:
                        hit_rate = self.n_hits / max(1, self.n_episodes)
                        print(f"\n[Training Progress] "
                              f"Episodes: {self.n_episodes} | "
                              f"Hit Rate: {hit_rate:.2%} ({self.n_hits}/{self.n_episodes}) | "
                              f"Timesteps: {self.num_timesteps:,}")

                        # Log to tensorboard
                        if self.logger is not None:
                            self.logger.record("train/episodes", self.n_episodes)
                            self.logger.record("train/hit_rate", hit_rate)
                            self.logger.record("train/n_hits", self.n_hits)

        return True


def create_video_callback(
    total_timesteps: int,
    n_videos: int = 10,
    n_eval_episodes_per_video: int = 3,
    video_folder: str = "turret_rl/videos/training",
    **kwargs
) -> VideoRecorderCallback:
    """Helper function to create a video callback with evenly distributed recordings.

    Args:
        total_timesteps: Total number of training timesteps
        n_videos: Number of videos to record throughout training
        n_eval_episodes_per_video: Episodes to record at each interval
        video_folder: Where to save videos
        **kwargs: Additional arguments for VideoRecorderCallback

    Returns:
        Configured VideoRecorderCallback

    Example:
        # Record 10 videos during 1M timestep training
        callback = create_video_callback(
            total_timesteps=1_000_000,
            n_videos=10,
            n_eval_episodes_per_video=3
        )
        model.learn(total_timesteps=1_000_000, callback=callback)
    """
    # Calculate frequency to distribute videos evenly
    eval_freq = total_timesteps // n_videos

    print(f"Configuring video recording:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Videos to record: {n_videos}")
    print(f"  Recording frequency: every {eval_freq:,} timesteps")
    print(f"  Episodes per video: {n_eval_episodes_per_video}")
    print(f"  Output folder: {video_folder}")

    return VideoRecorderCallback(
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes_per_video,
        video_folder=video_folder,
        **kwargs
    )