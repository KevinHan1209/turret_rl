"""Weights & Biases integration for training monitoring and visualization.

This module provides a callback that logs training metrics, videos, and other
artifacts to Weights & Biases for experiment tracking.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize

from ..envs.turret_env import TurretEnv
from ..config.config import WorldConfig, RewardConfig

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbCallback(BaseCallback):
    """Callback for logging training progress to Weights & Biases.

    Logs:
    - Training metrics (rewards, episode length, hit rate)
    - Episode videos at regular intervals
    - Model checkpoints
    - Hyperparameters and config

    Example:
        wandb_callback = WandbCallback(
            project="turret-rl",
            name="single-shot-ppo",
            log_freq=100,
            video_freq=10000
        )
        model.learn(total_timesteps=1000000, callback=wandb_callback)
    """

    def __init__(
        self,
        project: str = "turret-rl",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        log_freq: int = 100,  # Log metrics every N episodes
        video_freq: int = 10000,  # Record video every N timesteps
        n_video_episodes: int = 3,  # Number of episodes per video recording
        world_config: Optional[WorldConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        verbose: int = 1,
        sync_tensorboard: bool = True,
        save_code: bool = True,
    ):
        """Initialize the wandb callback.

        Args:
            project: W&B project name
            entity: W&B entity (username or team name)
            name: Run name (auto-generated if None)
            tags: List of tags for the run
            config: Configuration dict to log
            log_freq: Log metrics every N episodes
            video_freq: Record video every N timesteps
            n_video_episodes: Number of episodes to record per checkpoint
            world_config: World configuration
            reward_config: Reward configuration
            verbose: Verbosity level
            sync_tensorboard: Sync TensorBoard logs to W&B
            save_code: Save code to W&B
        """
        super().__init__(verbose)

        if not WANDB_AVAILABLE:
            warnings.warn(
                "wandb not installed. Install with: pip install wandb\n"
                "Callback will be disabled.",
                UserWarning
            )
            self.enabled = False
            return

        self.enabled = True
        self.project = project
        self.entity = entity
        self.run_name = name
        self.tags = tags or ["single-shot", "ppo"]
        self.config = config or {}
        self.log_freq = log_freq
        self.video_freq = video_freq
        self.n_video_episodes = n_video_episodes
        self.world_config = world_config or WorldConfig()
        self.reward_config = reward_config or RewardConfig()
        self.sync_tensorboard = sync_tensorboard
        self.save_code = save_code

        # Tracking variables
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_hits = []
        self.last_video_timestep = 0

        # Wandb run (initialized in _on_training_start)
        self.run = None

        # Evaluation environment for videos
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

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if not self.enabled:
            return

        # Initialize wandb run
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            tags=self.tags,
            config=self.config,
            sync_tensorboard=self.sync_tensorboard,
            save_code=self.save_code,
            reinit=True
        )

        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Weights & Biases logging initialized")
            print(f"Project: {self.project}")
            print(f"Run: {self.run.name}")
            print(f"URL: {self.run.url}")
            print(f"{'='*60}\n")

    def _on_step(self) -> bool:
        """Called at each step during training."""
        if not self.enabled:
            return True

        # Check for completed episodes
        if len(self.locals.get('dones', [])) > 0:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    # Episode completed
                    self.episode_count += 1

                    # Get episode info
                    info = self.locals.get('infos', [{}])[i]

                    # Track episode statistics
                    episode_reward = info.get('episode', {}).get('r', 0)
                    episode_length = info.get('episode', {}).get('l', 0)
                    hit = info.get('hit', False)

                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episode_hits.append(1 if hit else 0)

                    # Log metrics periodically
                    if self.episode_count % self.log_freq == 0:
                        self._log_metrics()

        # Record videos periodically
        if self.num_timesteps - self.last_video_timestep >= self.video_freq:
            self._record_videos()
            self.last_video_timestep = self.num_timesteps

        return True

    def _log_metrics(self) -> None:
        """Log accumulated metrics to wandb."""
        if not self.enabled or len(self.episode_rewards) == 0:
            return

        # Calculate statistics
        mean_reward = np.mean(self.episode_rewards[-self.log_freq:])
        mean_length = np.mean(self.episode_lengths[-self.log_freq:])
        hit_rate = np.mean(self.episode_hits[-self.log_freq:])

        # Log to wandb
        wandb.log({
            "episode/mean_reward": mean_reward,
            "episode/mean_length": mean_length,
            "episode/hit_rate": hit_rate,
            "episode/total_episodes": self.episode_count,
            "episode/total_hits": sum(self.episode_hits),
        }, step=self.num_timesteps)

        if self.verbose > 0:
            total_hits = sum(self.episode_hits)
            overall_hit_rate = total_hits / self.episode_count if self.episode_count > 0 else 0.0
            print(f"Episodes: {self.episode_count} | "
                  f"Hit Rate: {hit_rate:.2%} (overall: {overall_hit_rate:.2%}) | "
                  f"Mean Reward: {mean_reward:.3f} | "
                  f"Mean Length: {mean_length:.1f} | "
                  f"Timesteps: {self.num_timesteps}")

    def _record_videos(self) -> None:
        """Record evaluation episodes and log videos to wandb."""
        if not self.enabled:
            return

        self._init_eval_env()

        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Recording Videos for W&B (Timestep: {self.num_timesteps:,})")
            print(f"{'='*60}")

        videos_to_log = []

        # Record multiple episodes
        for ep_idx in range(self.n_video_episodes):
            frames = []
            episode_reward = 0.0
            hit = False
            shot_fired = False
            shot_step = None

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
            step_count = 0

            while not done and step_count < 500:
                # Get action from model (model expects normalized obs if using VecNormalize)
                action, _ = self.model.predict(obs, deterministic=True)

                # Step the RAW environment
                obs, reward, terminated, truncated, info = self.raw_env.step(action)
                done = terminated or truncated

                # If training uses VecNormalize, normalize the observation for next prediction
                if self.vec_normalize is not None:
                    obs = self.vec_normalize.normalize_obs(obs)

                episode_reward += reward
                step_count += 1

                if info.get('hit', False):
                    hit = True

                if info.get('shot_taken', False):
                    shot_fired = True
                    shot_step = step_count

                # Capture frame from RAW environment
                frame = self.raw_env.render()
                if frame is not None:
                    frames.append(frame)

            # Debug: Report episode result
            if self.verbose > 0:
                if not shot_fired:
                    print(f"    [WARNING] Episode ended without firing!")
                else:
                    print(f"    Shot fired at step {shot_step}/{step_count}")

            # If shot was fired, continue simulating for visualization
            # (shows bullet trajectory and full drone trajectory)
            if shot_fired:
                post_shot_frames = self.raw_env.simulate_post_shot(n_steps=200)
                frames.extend(post_shot_frames)

            # Convert frames to video
            if frames:
                # Stack frames into numpy array (T, H, W, C)
                video_array = np.array(frames)

                # Create caption
                result = "HIT" if hit else "MISS"
                caption = f"Ep {ep_idx+1} | {result} | Reward: {episode_reward:.2f} | Steps: {step_count}"

                videos_to_log.append({
                    'video': video_array,
                    'caption': caption,
                    'hit': hit
                })

                if self.verbose > 0:
                    shot_info = f"shot@step{shot_step}" if shot_fired else "no_shot"
                    print(f"  Episode {ep_idx+1}/{self.n_video_episodes}: {result} "
                          f"(reward={episode_reward:.2f}, steps={step_count}, {shot_info}, frames={len(frames)})")

        # Log videos to wandb and save locally
        if videos_to_log:
            from ..utils.visualization import save_video

            # Create local video directory
            local_video_dir = Path("turret_rl/videos/training")
            local_video_dir.mkdir(parents=True, exist_ok=True)

            for i, video_data in enumerate(videos_to_log):
                # Convert video array to list of frames for save_video
                frames_list = [video_data['video'][j] for j in range(len(video_data['video']))]

                # Save locally
                local_video_path = local_video_dir / f"timestep_{self.num_timesteps}_ep_{i+1}.mp4"
                save_video(
                    frames_list,
                    str(local_video_path),
                    fps=20
                )

                if self.verbose > 0:
                    print(f"    Saved {len(frames_list)} frames to {local_video_path}")

                # Upload to wandb
                wandb.log({
                    f"video/episode_{i+1}": wandb.Video(
                        str(local_video_path),
                        caption=video_data['caption'],
                        format="mp4"
                    )
                }, step=self.num_timesteps)

            # Also log hit rate for this batch
            batch_hit_rate = np.mean([v['hit'] for v in videos_to_log])
            wandb.log({
                "video/batch_hit_rate": batch_hit_rate
            }, step=self.num_timesteps)

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if not self.enabled:
            return

        # Log final metrics
        if len(self.episode_rewards) > 0:
            wandb.log({
                "final/total_episodes": self.episode_count,
                "final/total_hits": sum(self.episode_hits),
                "final/overall_hit_rate": np.mean(self.episode_hits),
                "final/mean_reward": np.mean(self.episode_rewards),
            }, step=self.num_timesteps)

        # Finish the run
        if self.run is not None:
            self.run.finish()

        if self.verbose > 0:
            print(f"\n{'='*60}")
            print("Weights & Biases logging finished")
            print(f"{'='*60}\n")
