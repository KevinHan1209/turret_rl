"""Turret vs Drone Reinforcement Learning Environment.

A 2D environment for training RL agents to intercept moving targets
using projectile physics.
"""

__version__ = "1.0.0"
__author__ = "RL Engineer"

# Make key components easily importable
from .envs.turret_env import TurretEnv
from .config.config import (
    WorldConfig,
    TrainingConfig,
    RewardConfig,
    EvaluationConfig,
    get_default_config
)

__all__ = [
    "TurretEnv",
    "WorldConfig",
    "TrainingConfig",
    "RewardConfig",
    "EvaluationConfig",
    "get_default_config"
]