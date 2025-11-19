"""PPO training script for the Turret vs Drone environment.

This module implements training using Proximal Policy Optimization (PPO)
from Stable-Baselines3, with configurable hyperparameters and automatic
model saving.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch.nn as nn

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from turret_rl.envs.turret_env import TurretEnv
from turret_rl.config.config import (
    TrainingConfig,
    WorldConfig,
    RewardConfig,
    DEFAULT_TRAINING_CONFIG
)
from turret_rl.agents.callbacks import (
    VideoRecorderCallback,
    create_video_callback
)
from turret_rl.agents.wandb_callback import WandbCallback


class TrainingProgressCallback(BaseCallback):
    """Custom callback for logging training progress."""

    def __init__(self, verbose: int = 1):
        """Initialize the callback.

        Args:
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.n_episodes = 0
        self.n_hits = 0

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
                    if self.n_episodes % 100 == 0 and self.verbose > 0:
                        hit_rate = self.n_hits / max(1, self.n_episodes)
                        print(f"\nEpisodes: {self.n_episodes} | "
                              f"Hit Rate: {hit_rate:.2%} | "
                              f"Timesteps: {self.num_timesteps}")

        return True


def create_env(
    world_config: Optional[WorldConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    seed: Optional[int] = None
) -> TurretEnv:
    """Create a single Turret environment instance.

    Args:
        world_config: World configuration
        reward_config: Reward configuration
        seed: Random seed

    Returns:
        Configured environment instance
    """
    env = TurretEnv(
        world_config=world_config,
        reward_config=reward_config,
        render_mode=None  # No rendering during training
    )

    # Wrap with Monitor for logging
    env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)

    return env


def train_ppo(
    training_config: Optional[TrainingConfig] = None,
    world_config: Optional[WorldConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    seed: Optional[int] = 42,
    load_checkpoint: Optional[str] = None,
    record_videos: bool = True,
    n_videos: int = 10,
    n_eval_episodes_per_video: int = 3,
    use_wandb: bool = True,
    wandb_project: str = "turret-rl-single-shot",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[list] = None
) -> PPO:
    """Train a PPO agent on the Turret environment.

    Args:
        training_config: Training configuration
        world_config: World configuration
        reward_config: Reward configuration
        seed: Random seed for reproducibility
        load_checkpoint: Path to checkpoint to resume from
        record_videos: Whether to record evaluation videos during training
        n_videos: Number of videos to record (distributed evenly across training)
        n_eval_episodes_per_video: Number of episodes to record at each interval

    Returns:
        Trained PPO model
    """
    # Use default configs if not provided
    training_config = training_config or DEFAULT_TRAINING_CONFIG
    world_config = world_config or WorldConfig()
    reward_config = reward_config or RewardConfig()

    print("=" * 50)
    print("PPO Training for Turret vs Drone Environment")
    print("=" * 50)
    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {training_config.total_timesteps:,}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  N steps: {training_config.n_steps}")
    print(f"  N environments: {training_config.n_envs}")
    print(f"  Normalize observations: {training_config.normalize_observations}")
    print(f"\nWorld Configuration:")
    print(f"  World size: {world_config.world_size}m x {world_config.world_size}m")
    print(f"  Drone speed: {world_config.drone_speed_min}-{world_config.drone_speed_max} m/s")
    print(f"  Bullet speed: {world_config.bullet_speed} m/s")
    print(f"  Max steps: {world_config.max_steps}")
    print(f"\nReward Configuration:")
    print(f"  Hit reward: {reward_config.hit_reward}")
    print(f"  Step penalty: {reward_config.step_penalty}")
    print(f"  Shot penalty: {reward_config.shot_penalty}")
    print("=" * 50)

    # Create vectorized environments
    print("\nCreating training environments...")

    def env_fn():
        return create_env(world_config, reward_config, seed)

    # Create multiple parallel environments
    vec_env = make_vec_env(
        env_fn,
        n_envs=training_config.n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv
    )

    # Optionally normalize observations and rewards
    if training_config.normalize_observations or training_config.normalize_rewards:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=training_config.normalize_observations,
            norm_reward=training_config.normalize_rewards,
            clip_obs=10.0,  # Clip observations to reasonable range
            clip_reward=10.0,  # Clip rewards to reasonable range
            gamma=training_config.gamma
        )
        print(f"  Applied normalization - Obs: {training_config.normalize_observations}, "
              f"Reward: {training_config.normalize_rewards}")

    # Set up model save directory
    model_dir = Path(training_config.model_save_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set up tensorboard logging
    log_dir = model_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logger
    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])

    # Create or load PPO model
    if load_checkpoint:
        print(f"\nLoading model from checkpoint: {load_checkpoint}")
        model = PPO.load(
            load_checkpoint,
            env=vec_env,
            tensorboard_log=str(log_dir)
        )
        model.set_logger(new_logger)
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=training_config.learning_rate,
            n_steps=training_config.n_steps,
            batch_size=training_config.batch_size,
            n_epochs=training_config.n_epochs,
            gamma=training_config.gamma,
            gae_lambda=training_config.gae_lambda,
            clip_range=training_config.clip_range,
            ent_coef=training_config.ent_coef,
            vf_coef=training_config.vf_coef,
            max_grad_norm=training_config.max_grad_norm,
            policy_kwargs={
                "net_arch": training_config.net_arch,
                "activation_fn": nn.ReLU  # Using ReLU activation
            } if training_config.net_arch else None,
            tensorboard_log=str(log_dir),
            verbose=training_config.verbose,
            seed=seed
        )
        model.set_logger(new_logger)

    # Create callbacks
    callbacks = []

    # Weights & Biases callback (if enabled)
    if use_wandb:
        wandb_config = {
            "algorithm": "PPO",
            "total_timesteps": training_config.total_timesteps,
            "learning_rate": training_config.learning_rate,
            "n_steps": training_config.n_steps,
            "batch_size": training_config.batch_size,
            "n_epochs": training_config.n_epochs,
            "gamma": training_config.gamma,
            "gae_lambda": training_config.gae_lambda,
            "clip_range": training_config.clip_range,
            "ent_coef": training_config.ent_coef,
            "vf_coef": training_config.vf_coef,
            "n_envs": training_config.n_envs,
            "normalize_obs": training_config.normalize_observations,
            "normalize_rewards": training_config.normalize_rewards,
            "world_size": world_config.world_size,
            "drone_speed_min": world_config.drone_speed_min,
            "drone_speed_max": world_config.drone_speed_max,
            "bullet_speed": world_config.bullet_speed,
            "max_steps": world_config.max_steps,
            "hit_reward": reward_config.hit_reward,
            "miss_penalty": reward_config.miss_penalty,
            "step_penalty": reward_config.step_penalty,
        }

        wandb_callback = WandbCallback(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            tags=wandb_tags,
            config=wandb_config,
            log_freq=100,  # Log every 100 episodes
            video_freq=max(training_config.total_timesteps // 10, 10000),  # 10 videos
            n_video_episodes=n_eval_episodes_per_video,
            world_config=world_config,
            reward_config=reward_config,
            verbose=training_config.verbose,
            sync_tensorboard=True,
            save_code=True
        )
        callbacks.append(wandb_callback)
        print("\n🔗 Weights & Biases logging enabled")
        print(f"   Project: {wandb_project}")
        if wandb_entity:
            print(f"   Entity: {wandb_entity}")

    # Checkpoint callback for periodic saves
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.save_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="turret_ppo_checkpoint",
        save_vecnormalize=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # Progress logging callback
    progress_callback = TrainingProgressCallback(verbose=training_config.verbose)
    callbacks.append(progress_callback)

    # Video recording callback (if enabled and not using wandb)
    # If using wandb, videos are handled by WandbCallback
    if record_videos and not use_wandb:
        video_callback = create_video_callback(
            total_timesteps=training_config.total_timesteps,
            n_videos=n_videos,
            n_eval_episodes_per_video=n_eval_episodes_per_video,
            video_folder=str(model_dir / "training_videos"),
            world_config=world_config,
            reward_config=reward_config,
            verbose=training_config.verbose,
            deterministic=True
        )
        callbacks.append(video_callback)
        print("\n📹 Video recording enabled:")
        print(f"   {n_videos} videos will be recorded during training")
        print(f"   {n_eval_episodes_per_video} episodes per recording")
        print(f"   Saved to: {model_dir / 'training_videos'}")

    # Optional: Evaluation callback
    # Uncomment to enable periodic evaluation during training
    # eval_env = create_env(world_config, reward_config, seed=seed+1000)
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=str(model_dir / "best_model"),
    #     log_path=str(model_dir / "eval_logs"),
    #     eval_freq=10000,
    #     deterministic=True,
    #     render=False
    # )
    # callbacks.append(eval_callback)

    # Combine all callbacks
    callback = CallbackList(callbacks)

    # Train the model
    print(f"\nStarting training for {training_config.total_timesteps:,} timesteps...")
    print("You can monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")
    print("\n" + "=" * 50)

    try:
        model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=callback,
            log_interval=training_config.log_interval,
            progress_bar=True,
            reset_num_timesteps=False if load_checkpoint else True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        # Save the final model
        final_model_path = str(model_dir / training_config.model_name)
        print(f"\nSaving final model to: {final_model_path}")
        model.save(final_model_path)

        # Save normalization statistics if using VecNormalize
        if isinstance(vec_env, VecNormalize):
            vec_norm_path = str(model_dir / "vec_normalize.pkl")
            vec_env.save(vec_norm_path)
            print(f"Saved normalization statistics to: {vec_norm_path}")

    print("\nTraining completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Total episodes completed: {progress_callback.n_episodes}")
    if progress_callback.n_episodes > 0:
        final_hit_rate = progress_callback.n_hits / progress_callback.n_episodes
        print(f"Final hit rate: {final_hit_rate:.2%}")

    return model


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO agent for Turret vs Drone")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total number of training timesteps"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable observation and reward normalization"
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable video recording during training"
    )
    parser.add_argument(
        "--n-videos",
        type=int,
        default=10,
        help="Number of evaluation videos to record during training (default: 10)"
    )
    parser.add_argument(
        "--episodes-per-video",
        type=int,
        default=3,
        help="Number of episodes to record at each video checkpoint (default: 3)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="turret-rl-single-shot",
        help="W&B project name (default: turret-rl-single-shot)"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (username or team)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not specified)"
    )

    args = parser.parse_args()

    # Create custom training config
    training_config = TrainingConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        normalize_observations=not args.no_normalize,
        normalize_rewards=not args.no_normalize
    )

    # Run training
    train_ppo(
        training_config=training_config,
        seed=args.seed,
        load_checkpoint=args.checkpoint,
        record_videos=not args.no_videos,
        n_videos=args.n_videos,
        n_eval_episodes_per_video=args.episodes_per_video,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name
    )


if __name__ == "__main__":
    main()