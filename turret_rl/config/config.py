"""Configuration module for the Turret-vs-Drone RL environment.

This module contains all configurable parameters for the simulation,
training, and evaluation, organized in dataclasses for clean access.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WorldConfig:
    """Configuration parameters for the simulation world (single-shot version)."""

    # World dimensions
    world_size: float = 300.0  # meters, square world extends from -150 to 150

    # Time settings
    dt: float = 0.05  # seconds, simulation timestep
    max_steps: int = 200  # maximum steps per episode (reduced for single-shot problem)

    # Drone parameters
    drone_radius: float = 2.0  # meters, collision radius
    drone_speed_min: float = 20.0  # m/s, minimum drone speed
    drone_speed_max: float = 80.0  # m/s, maximum drone speed
    min_approach_distance: float = 100.0  # meters, min distance from origin that drone path must pass

    # Bullet parameters
    bullet_speed: float = 100.0  # m/s
    bullet_max_range: float = 100.0  # meters, bullet disappears after this distance
    bullet_radius: float = 2.0  # meters, for visualization and collision (increased for visibility)

    # Turret parameters
    turret_cooldown: float = 0.1  # seconds, minimum time between shots
    turret_radius: float = 3.0  # meters, for visualization only

    @property
    def bullet_lifetime(self) -> float:
        """Calculate bullet lifetime based on speed and max range."""
        return self.bullet_max_range / self.bullet_speed

    @property
    def world_half_size(self) -> float:
        """Half the world size, useful for normalization."""
        return self.world_size / 2.0


@dataclass
class TrainingConfig:
    """Configuration for PPO training (single-shot version).

    Since episodes are shorter (single-shot decision), we can increase
    total timesteps to ensure sufficient learning.
    """

    # Training duration (increased since episodes are shorter)
    total_timesteps: int = 500_000  # Can increase to 1M if needed

    # PPO hyperparameters
    n_steps: int = 1024  # Number of steps to run for each environment per update
    batch_size: int = 64  # Minibatch size
    n_epochs: int = 10  # Number of epochs when optimizing the surrogate loss
    learning_rate: float = 3e-4  # Learning rate
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    clip_range: float = 0.2  # Clipping parameter for PPO
    ent_coef: float = 0.01  # Entropy coefficient for exploration
    vf_coef: float = 0.5  # Value function coefficient in the loss
    max_grad_norm: float = 0.5  # Maximum norm for gradient clipping

    # Network architecture
    net_arch: list = None  # If None, uses default [64, 64]

    # Environment settings
    n_envs: int = 4  # Number of parallel environments for training
    normalize_observations: bool = True  # Whether to use VecNormalize
    normalize_rewards: bool = True  # Whether to normalize rewards

    # Logging and saving
    save_freq: int = 10000  # Save model every n steps
    log_interval: int = 10  # Log training stats every n updates
    verbose: int = 1  # Verbosity level (0: none, 1: info, 2: debug)

    # Model paths
    model_save_path: str = "turret_rl/models/turret_ppo"
    model_name: str = "turret_ppo.zip"

    def __post_init__(self):
        """Initialize network architecture if not provided."""
        if self.net_arch is None:
            self.net_arch = [dict(pi=[64, 64], vf=[64, 64])]


@dataclass
class RewardConfig:
    """Configuration for the reward function (single-shot version).

    For the single-shot problem:
    - Agent gets +1.0 for hitting the drone when firing
    - Agent gets -1.0 for missing when firing (or never firing)
    - Small step penalty encourages timely decision-making
    """

    # Main rewards
    hit_reward: float = 1.0  # Reward for successfully hitting the drone when firing
    miss_penalty: float = -1.0  # Penalty for missing when firing

    # Step penalties to encourage efficiency and discourage "never fire" strategy
    # IMPORTANT: step_penalty should be VERY SMALL to avoid incentivizing immediate firing
    # With -0.001, agent learns to fire immediately to avoid accumulated penalties
    # Recommended: 0.0 (no penalty) or -0.00001 (extremely small)
    step_penalty: float = 0.0  # Penalty per timestep (set to 0 to allow agent to wait for good shots)
    shot_penalty: float = 0.0  # Not used in single-shot version (no longer penalize firing)

    # Penalty for never firing (when episode ends without shot)
    no_shot_penalty: float = -5.0  # SEVERE penalty - never firing must be much worse than missing

    # Note: The large difference between miss_penalty (-1.0) and no_shot_penalty (-5.0)
    # ensures the agent always attempts a shot, even if uncertain

    # Optional shaping rewards (set to 0 to disable)
    distance_reward_scale: float = 0.0  # Scale for distance-based shaping
    near_miss_reward: float = 0.0  # Reward for bullets passing close to drone


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and video recording."""

    n_eval_episodes: int = 5  # Number of episodes to evaluate
    record_video: bool = True  # Whether to record videos
    video_fps: int = 20  # Frames per second for video recording
    video_folder: str = "turret_rl/videos"  # Folder to save videos
    show_trajectory: bool = True  # Whether to show drone trajectory in rendering
    show_info: bool = True  # Whether to overlay info text on frames


# Default configuration instances
DEFAULT_WORLD_CONFIG = WorldConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_EVAL_CONFIG = EvaluationConfig()


def get_default_config():
    """Get all default configurations as a single object."""
    return {
        'world': DEFAULT_WORLD_CONFIG,
        'training': DEFAULT_TRAINING_CONFIG,
        'reward': DEFAULT_REWARD_CONFIG,
        'evaluation': DEFAULT_EVAL_CONFIG
    }