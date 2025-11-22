"""SAC training script for the Turret vs Drone environment.

This module implements training using Soft Actor-Critic (SAC) from Stable-Baselines3,
optimized for the single-shot turret environment. SAC is an off-policy algorithm that
excels with continuous action spaces and short-horizon episodes.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch.nn as nn

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import SAC
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


def train_sac(
    training_config: Optional[TrainingConfig] = None,
    world_config: Optional[WorldConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    seed: Optional[int] = 42,
    load_checkpoint: Optional[str] = None,
    record_videos: bool = True,
    n_videos: int = 10,
    n_eval_episodes_per_video: int = 3,
    use_wandb: bool = False,
    wandb_project: str = "turret-rl-sac",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[list] = None,
    # SAC-specific hyperparameters
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    buffer_size: int = 200000,
    tau: float = 0.005,
    gamma: float = 0.99,
    train_freq: int = 1,
    gradient_steps: int = 1,
    ent_coef: str = "auto",
    learning_starts: int = 1000,
    use_normalize: bool = False
) -> SAC:
    """Train a SAC agent on the Turret environment.

    SAC (Soft Actor-Critic) is an off-policy algorithm that is particularly
    well-suited for:
    - Continuous action spaces (turret angle + fire gate)
    - Short-horizon episodes (single-shot decision)
    - Sample efficiency (learns from replay buffer)
    - Superior exploration (entropy-regularized policy)

    Args:
        training_config: Training configuration
        world_config: World configuration
        reward_config: Reward configuration
        seed: Random seed for reproducibility
        load_checkpoint: Path to checkpoint to resume from
        record_videos: Whether to record evaluation videos during training
        n_videos: Number of videos to record (distributed evenly across training)
        n_eval_episodes_per_video: Number of episodes to record at each interval
        use_wandb: Whether to enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity (username or team)
        wandb_run_name: W&B run name
        wandb_tags: W&B tags for this run
        learning_rate: Learning rate for actor and critic networks
        batch_size: Minibatch size for gradient updates
        buffer_size: Size of replay buffer
        tau: Soft update coefficient (for target networks)
        gamma: Discount factor
        train_freq: Update the model every train_freq steps
        gradient_steps: How many gradient steps to do after each rollout
        ent_coef: Entropy coefficient ('auto' for automatic tuning)
        learning_starts: How many steps before training starts
        use_normalize: Whether to use VecNormalize wrapper

    Returns:
        Trained SAC model
    """
    # Use default configs if not provided
    training_config = training_config or DEFAULT_TRAINING_CONFIG
    world_config = world_config or WorldConfig()
    reward_config = reward_config or RewardConfig()

    print("=" * 60)
    print("SAC Training for Turret vs Drone Environment")
    print("=" * 60)
    print(f"\nAlgorithm: Soft Actor-Critic (SAC)")
    print(f"  - Off-policy reinforcement learning")
    print(f"  - Optimized for continuous actions")
    print(f"  - Superior sample efficiency")
    print(f"  - Entropy-regularized for exploration")
    print(f"\nSAC Hyperparameters:")
    print(f"  Total timesteps: {training_config.total_timesteps:,}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Buffer size: {buffer_size:,}")
    print(f"  Tau (soft update): {tau}")
    print(f"  Gamma (discount): {gamma}")
    print(f"  Train frequency: {train_freq}")
    print(f"  Gradient steps: {gradient_steps}")
    print(f"  Entropy coefficient: {ent_coef}")
    print(f"  Learning starts: {learning_starts:,}")
    print(f"  Normalize observations: {use_normalize}")
    print(f"\nWorld Configuration:")
    print(f"  World size: {world_config.world_size}m x {world_config.world_size}m")
    print(f"  Drone speed: {world_config.drone_speed_min}-{world_config.drone_speed_max} m/s")
    print(f"  Bullet speed: {world_config.bullet_speed} m/s")
    print(f"  Max steps: {world_config.max_steps}")
    print(f"\nReward Configuration:")
    print(f"  Hit reward: {reward_config.hit_reward}")
    print(f"  Miss penalty: {reward_config.miss_penalty}")
    print(f"  No-shot penalty: {reward_config.no_shot_penalty}")
    print(f"  Step penalty: {reward_config.step_penalty}")
    print("=" * 60)

    # Create environment
    print("\nCreating training environment...")

    def make_env():
        """Factory function to create environment."""
        return create_env(world_config, reward_config, seed)

    # SAC requires vectorized environment
    env = DummyVecEnv([make_env])

    # Optionally normalize observations (generally not required for SAC)
    if use_normalize:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=False,  # Don't normalize rewards for SAC
            clip_obs=10.0,
            gamma=gamma
        )
        print(f"  Applied VecNormalize - Obs: True, Reward: False")

    # Set up model save directory
    model_dir = Path("turret_rl/models/turret_sac")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set up tensorboard logging
    log_dir = model_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logger
    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])

    # Create or load SAC model
    if load_checkpoint:
        print(f"\nLoading model from checkpoint: {load_checkpoint}")
        model = SAC.load(
            load_checkpoint,
            env=env,
            tensorboard_log=str(log_dir)
        )
        model.set_logger(new_logger)
    else:
        print("\nCreating new SAC model...")
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            device="auto",  # Automatically use GPU if available
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=seed
        )
        model.set_logger(new_logger)

    # Create callbacks
    callbacks = []

    # Weights & Biases callback (if enabled)
    if use_wandb:
        wandb_config = {
            "algorithm": "SAC",
            "total_timesteps": training_config.total_timesteps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "tau": tau,
            "gamma": gamma,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "ent_coef": ent_coef,
            "learning_starts": learning_starts,
            "normalize_obs": use_normalize,
            "world_size": world_config.world_size,
            "drone_speed_min": world_config.drone_speed_min,
            "drone_speed_max": world_config.drone_speed_max,
            "bullet_speed": world_config.bullet_speed,
            "max_steps": world_config.max_steps,
            "hit_reward": reward_config.hit_reward,
            "miss_penalty": reward_config.miss_penalty,
            "no_shot_penalty": reward_config.no_shot_penalty,
            "step_penalty": reward_config.step_penalty,
        }

        wandb_callback = WandbCallback(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            tags=wandb_tags,
            config=wandb_config,
            log_freq=100,
            video_freq=max(training_config.total_timesteps // 10, 10000),
            n_video_episodes=n_eval_episodes_per_video,
            world_config=world_config,
            reward_config=reward_config,
            verbose=1,
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
        save_freq=max(10000, training_config.total_timesteps // 20),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="turret_sac_checkpoint",
        save_vecnormalize=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # Progress logging callback
    progress_callback = TrainingProgressCallback(verbose=1)
    callbacks.append(progress_callback)

    # Video recording callback (if enabled and not using wandb)
    if record_videos and not use_wandb:
        video_callback = create_video_callback(
            total_timesteps=training_config.total_timesteps,
            n_videos=n_videos,
            n_eval_episodes_per_video=n_eval_episodes_per_video,
            video_folder=str(model_dir / "training_videos"),
            world_config=world_config,
            reward_config=reward_config,
            verbose=1,
            deterministic=True
        )
        callbacks.append(video_callback)
        print("\n📹 Video recording enabled:")
        print(f"   {n_videos} videos will be recorded during training")
        print(f"   {n_eval_episodes_per_video} episodes per recording")
        print(f"   Saved to: {model_dir / 'training_videos'}")

    # Combine all callbacks
    callback = CallbackList(callbacks)

    # Train the model
    print(f"\nStarting SAC training for {training_config.total_timesteps:,} timesteps...")
    print("You can monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")
    print("\n" + "=" * 60)

    try:
        model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=callback,
            log_interval=4,
            progress_bar=True,
            reset_num_timesteps=False if load_checkpoint else True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        # Save the final model
        final_model_path = str(model_dir / "turret_sac.zip")
        print(f"\nSaving final model to: {final_model_path}")
        model.save(final_model_path)

        # Save normalization statistics if using VecNormalize
        if isinstance(env, VecNormalize):
            vec_norm_path = str(model_dir / "vec_normalize.pkl")
            env.save(vec_norm_path)
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

    parser = argparse.ArgumentParser(
        description="Train SAC agent for Turret vs Drone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total number of training timesteps"
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

    # SAC hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for actor and critic networks"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size for gradient updates"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=200000,
        help="Size of the replay buffer"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Soft update coefficient for target networks"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Update the model every train_freq steps"
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="How many gradient steps after each rollout"
    )
    parser.add_argument(
        "--ent-coef",
        type=str,
        default="auto",
        help="Entropy coefficient (use 'auto' for automatic tuning)"
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="How many steps before training starts"
    )

    # Environment wrappers
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable observation normalization (optional for SAC)"
    )

    # Video recording
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable video recording during training"
    )
    parser.add_argument(
        "--n-videos",
        type=int,
        default=10,
        help="Number of evaluation videos to record during training"
    )
    parser.add_argument(
        "--episodes-per-video",
        type=int,
        default=3,
        help="Number of episodes to record at each video checkpoint"
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="turret-rl-sac",
        help="W&B project name"
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
    )

    # Run training
    train_sac(
        training_config=training_config,
        seed=args.seed,
        load_checkpoint=args.checkpoint,
        record_videos=not args.no_videos,
        n_videos=args.n_videos,
        n_eval_episodes_per_video=args.episodes_per_video,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        # SAC hyperparameters
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        learning_starts=args.learning_starts,
        use_normalize=args.normalize
    )


if __name__ == "__main__":
    main()
