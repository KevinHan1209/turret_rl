# Turret vs Drone - Single-Shot RL Environment

A reinforcement learning environment where a stationary turret must intercept a moving drone with a single shot. Trained with Soft Actor-Critic (SAC).

## Quick Start

### Linux
```bash
# Setup environment
./setup.sh
conda activate turret_rl

# Run demo with pre-trained model (saves video to demo/demo_output.mp4)
python demo/run_demo.py

# Train new model
python -m turret_rl.agents.train_sac --timesteps 1000000
```

### macOS
```bash
# Setup environment (Mac-specific script)
./setup_mac.sh
conda activate turret_rl

# Run demo (auto-detects Mac model)
python demo/run_demo.py --no-video  # Use --no-video if rendering hangs

# Train new model
python -m turret_rl.agents.train_sac --timesteps 1000000
```

---

## Problem Formulation

The agent controls a turret at the origin that must fire once to intercept a drone flying across the battlefield. The agent observes position and velocity, then decides when and where to shoot.

---

## Observation Space (4D)

The agent receives a **4-dimensional continuous observation vector** in turret-centric Cartesian coordinates:

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0 | `x_d_norm` | Drone x position | / 150 (world half-size) |
| 1 | `y_d_norm` | Drone y position | / 150 (world half-size) |
| 2 | `vx_d_norm` | Drone x velocity | / 80 (max speed) |
| 3 | `vy_d_norm` | Drone y velocity | / 80 (max speed) |

**Details:**
- Turret is at origin `(0, 0)`
- World bounds: `[-150, 150] × [-150, 150]` meters
- All values normalized to approximately `[-1, 1]` range
- No bullet information in observation (bullets are for visualization only)

---

## Action Space (2D)

The agent outputs a **2-dimensional continuous action vector**:

| Index | Action | Range | Interpretation |
|-------|--------|-------|----------------|
| 0 | Firing angle | `[-1, 1]` | Maps to azimuth `[-π, π]` radians |
| 1 | Fire gate | `[-1, 1]` | `> 0` to fire, `≤ 0` to wait |

**Behavior:**
- If `action[1] > 0` and `shot_taken == False`:
  - Agent fires at angle `action[0] * π`
  - Environment evaluates hit/miss analytically
  - Episode terminates immediately
- If `action[1] ≤ 0` or shot already taken:
  - No firing occurs
  - Drone continues moving
  - Agent gets next observation

---

## Reward Structure

Carefully designed reward structure to encourage the agent to learn firing behavior:

| Event | Reward | When |
|-------|--------|------|
| **Hit** | `+1.0` | Agent fires and shot intercepts drone |
| **Miss** | `-1.0` | Agent fires but shot misses drone |
| **No shot (escaped/timeout)** | `-5.0` | Drone escapes or max steps without agent firing |
| **Step penalty** | `-0.001` | Every timestep (encourages timely decision) |

**Total episode reward examples:**
- Fire at step 50 and hit: `1.0 - 0.001 * 50 = +0.95`
- Fire at step 10 and miss: `-1.0 - 0.001 * 10 = -1.01`
- Never fire, timeout at step 200: `-5.0 - 0.001 * 200 = -5.2`

**Reward design rationale:**
- Missing (`-1.01`) is MUCH better than never trying (`-5.2`) to strongly encourage exploration
- Severe no-shot penalty (`-5.0`) ensures agent learns to always take a shot
- Step penalty (`-0.001`) is small and encourages timely decisions
- Hitting early yields highest rewards, encouraging decisive action

---

## Episode Termination

Episodes terminate in the following cases:

1. **Agent fires** (`action[1] > 0` and `shot_taken == False`):
   - Hit/miss evaluated analytically
   - Episode ends immediately with appropriate reward

2. **Drone escapes** (position outside `[-165, 165]²`):
   - If no shot taken: reward = `-1.0 + step_penalty * steps`
   - Episode terminates

3. **Max steps reached** (default: 200 steps):
   - If no shot taken: reward = `-1.0 + step_penalty * steps`
   - Episode truncated

---

## Analytic Hit/Miss Evaluation

When the agent fires at angle `θ`, the environment solves for intercept time `τ` by solving the quadratic equation:

```
|| p_d + (v_d - v_b * u) * τ ||² = r²
```

Where `p_d` is drone position, `v_d` is drone velocity, `u = [cos(θ), sin(θ)]` is bullet direction, `v_b = 100 m/s` is bullet speed, and `r = 2.0 m` is drone radius.

**Implementation:** See `TurretEnv._will_hit_drone()` in [turret_env.py:263](turret_rl/envs/turret_env.py#L263)

---

## Installation

### Linux/Ubuntu (Recommended)

```bash
# Use setup script (faster - uses pip instead of conda for packages)
./setup.sh
conda activate turret_rl
```

### macOS

```bash
# Use the Mac-specific setup script (handles NumPy/PyTorch compatibility)
./setup_mac.sh
conda activate turret_rl

# Alternative: Manual installation
conda create -n turret_rl python=3.10
conda activate turret_rl
pip install "numpy<2.0"  # Use NumPy 1.x for compatibility
pip install -r requirements.txt
```

### Alternative Installation Methods

```bash
# Pure conda (slower)
conda env create -f environment.yml
conda activate turret_rl

# Or pip only (if you already have Python 3.10+)
pip install -r requirements.txt
```

### Platform Compatibility Notes

- **Linux**: Full compatibility with all features
- **macOS**: Use the Mac-compatible model (`turret_sac_mac_clean.zip`) for demos
- **NumPy Version**: The environment works with both NumPy 1.x and 2.x, but pre-trained models may require conversion between versions

---

## Usage

### Training

Train a **SAC (Soft Actor-Critic)** agent on the single-shot turret environment:

```bash
# Basic training (500k timesteps, default SAC config)
python -m turret_rl.agents.train_sac

# Custom training duration
python -m turret_rl.agents.train_sac --timesteps 1000000

# Adjust SAC hyperparameters
python -m turret_rl.agents.train_sac \
    --lr 3e-4 \
    --batch-size 256 \
    --buffer-size 200000 \
    --tau 0.005

# Resume from checkpoint
python -m turret_rl.agents.train_sac --checkpoint turret_rl/models/turret_sac/checkpoints/turret_sac_checkpoint_100000_steps.zip

# Enable observation normalization (optional for SAC)
python -m turret_rl.agents.train_sac --normalize

# Control video recording
python -m turret_rl.agents.train_sac --n-videos 10 --episodes-per-video 5
python -m turret_rl.agents.train_sac --no-videos
```

**Why SAC?**

SAC (Soft Actor-Critic) is the **preferred algorithm** for this environment because:
- **Off-policy learning**: More sample-efficient than on-policy methods like PPO
- **Continuous action spaces**: Naturally suited for turret angle + fire gate control
- **Superior exploration**: Entropy-regularized policy encourages diverse strategies
- **Short-horizon episodes**: Handles single-shot decisions more effectively
- **Replay buffer**: Learns from all past experiences, not just recent ones

**Training outputs:**
- Model checkpoints: `turret_rl/models/turret_sac/checkpoints/`
- Final model: `turret_rl/models/turret_sac/turret_sac.zip`
- TensorBoard logs: `turret_rl/models/turret_sac/logs/`
- Training videos: `turret_rl/models/turret_sac/training_videos/`

**SAC Hyperparameters Explained:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 3e-4 | Learning rate for actor and critic networks |
| `--batch-size` | 256 | Minibatch size for gradient updates |
| `--buffer-size` | 200000 | Replay buffer capacity (stores past experiences) |
| `--tau` | 0.005 | Soft update coefficient for target networks |
| `--gamma` | 0.99 | Discount factor for future rewards |
| `--train-freq` | 1 | Update model every N environment steps |
| `--gradient-steps` | 1 | Gradient steps per environment step |
| `--ent-coef` | auto | Entropy coefficient (auto-tuned by default) |
| `--learning-starts` | 1000 | Random exploration steps before training |

**Training with Weights & Biases:**
```bash
# Enable W&B logging (requires: pip install wandb)
python -m turret_rl.agents.train_sac --wandb --timesteps 1000000

# Specify project and entity
python -m turret_rl.agents.train_sac --wandb \
    --wandb-project my-turret-project \
    --wandb-entity my-username

# Custom run name
python -m turret_rl.agents.train_sac --wandb \
    --wandb-run-name "sac-single-shot-v1"
```

**Monitor training:**
```bash
# TensorBoard
tensorboard --logdir turret_rl/models/turret_sac/logs/

# Or view on Weights & Biases dashboard (if --wandb enabled)
# URL will be printed at start of training
```

---

### Training with PPO (Alternative)

The environment also supports PPO training (legacy approach):

```bash
# Basic PPO training
python -m turret_rl.agents.train_ppo --timesteps 1000000

# PPO with custom hyperparameters
python -m turret_rl.agents.train_ppo --n-envs 8 --lr 1e-4 --no-normalize
```

**Note:** SAC is recommended over PPO for this environment due to better sample efficiency and exploration.

### Demo

```bash
# Run demo with default model (Linux/original)
python demo/run_demo.py                          # Run demo, save video
python demo/run_demo.py --episodes 10            # More episodes
python demo/run_demo.py --no-video               # Stats only

# macOS users: Use Mac-compatible model
python demo/run_demo.py --model turret_rl/models/turret_sac/turret_sac_mac_clean.zip
python demo/run_demo.py --model turret_rl/models/turret_sac/turret_sac_mac_clean.zip --episodes 10

# Note: If video rendering hangs on macOS, use --no-video flag
python demo/run_demo.py --no-video --episodes 10
```

**Pre-trained models:**
- **Linux/Original**: `turret_rl/models/turret_sac/turret_sac_final.zip` (trained with NumPy 2.3.5)
- **macOS Compatible**: `turret_rl/models/turret_sac/turret_sac_mac_clean.zip` (works with NumPy 1.x)

**Converting Models Between Platforms:**

If you trained a model on Linux and need to run it on macOS (or vice versa), use the conversion script:

```bash
# Convert Linux model to Mac-compatible version
python convert_weights_only.py

# This will create turret_sac_mac_clean.zip from turret_sac_final.zip
```

### Interactive Testing

Test the environment directly in Python:

```python
import gymnasium as gym
from turret_rl.envs.turret_env import TurretEnv

# Create environment
env = TurretEnv(render_mode='rgb_array')

# Run random agent
obs, info = env.reset()
for _ in range(100):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {info['step']}: reward={reward:.3f}, shot_taken={info['shot_taken']}")

    if terminated or truncated:
        print(f"Episode ended: {info['shot_result']}")
        break

env.close()
```

---

## Configuration

All parameters are centralized in [config/config.py](turret_rl/config/config.py).

### WorldConfig

```python
world_size: float = 300.0          # Battlefield size (meters)
dt: float = 0.05                   # Simulation timestep (seconds)
max_steps: int = 200               # Max steps per episode

drone_radius: float = 2.0          # Drone collision radius (meters)
drone_speed_min: float = 20.0      # Min drone speed (m/s)
drone_speed_max: float = 80.0      # Max drone speed (m/s)

bullet_speed: float = 100.0        # Bullet speed (m/s)
bullet_max_range: float = 100.0    # Bullet max travel distance (meters)
```

### RewardConfig

```python
hit_reward: float = 1.0            # Reward for hitting drone
miss_penalty: float = -1.0         # Penalty for missing
no_shot_penalty: float = -5.0      # Severe penalty for not firing (much worse than missing)
step_penalty: float = 0.0          # Per-timestep penalty (set to 0 to avoid immediate firing)
```

**Critical Design Note:** The `step_penalty` is set to **0.0** (no penalty per step).

**Why?** With a non-zero step penalty (e.g., -0.001), the agent learns to **fire immediately** to minimize accumulated penalties, even if waiting would improve accuracy:
- Fire at step 1: reward = +1.0 - 0.001×1 = +0.999
- Wait until step 50 for perfect shot: reward = +1.0 - 0.001×50 = +0.950

Setting `step_penalty = 0.0` allows the agent to learn optimal timing based purely on hit/miss feedback.

### TrainingConfig

**SAC Hyperparameters** (configured via CLI arguments):

```python
learning_rate: float = 3e-4        # Actor/critic learning rate
batch_size: int = 256              # Minibatch size (larger than PPO)
buffer_size: int = 200_000         # Replay buffer size
tau: float = 0.005                 # Soft update coefficient
gamma: float = 0.99                # Discount factor
train_freq: int = 1                # Model update frequency
gradient_steps: int = 1            # Gradient steps per rollout
ent_coef: str = "auto"             # Entropy coefficient (auto-tuned)
learning_starts: int = 1000        # Random steps before training
```

**PPO Hyperparameters** (in config.py, for legacy training):

```python
total_timesteps: int = 500_000     # Training duration
n_envs: int = 4                    # Parallel environments
learning_rate: float = 3e-4        # PPO learning rate
batch_size: int = 64               # Minibatch size
n_epochs: int = 10                 # PPO epochs per update
```

---

## File Structure

```
9Mothers_TakeHome/
├── turret_rl/
│   ├── envs/
│   │   └── turret_env.py          # Core Gymnasium environment (single-shot version)
│   ├── config/
│   │   └── config.py              # Centralized configuration
│   ├── agents/
│   │   ├── train_sac.py           # SAC training script (recommended)
│   │   ├── train_ppo.py           # PPO training script (legacy)
│   │   ├── callbacks.py           # Custom training callbacks
│   │   └── wandb_callback.py      # Weights & Biases integration
│   ├── scripts/
│   │   └── evaluate_and_record.py # Evaluation and video recording (SAC/PPO)
│   ├── utils/
│   │   └── visualization.py       # Video/rendering utilities
│   ├── models/
│   │   ├── turret_sac/            # SAC models and checkpoints
│   │   └── turret_ppo/            # PPO models and checkpoints (legacy)
│   └── videos/                    # Recorded videos (created during training/eval)
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## Key Implementation Details

**Single-Shot Mechanism:** When `action[1] > 0` and `shot_taken == False`, the environment sets `shot_taken = True`, calls `_will_hit_drone()` to analytically evaluate intercept, and terminates the episode with appropriate reward. See [turret_env.py:354](turret_rl/envs/turret_env.py#L354).

**Bullet Visualization:** Bullets are created for rendering but **not included in the observation vector**. This allows realistic visualization while keeping the observation space minimal (4D).

**Drone Trajectory Sampling:** Drones spawn on the world boundary and fly in a straight line, guaranteed to pass within 100m of origin with speed sampled from `[20, 80]` m/s. See [turret_env.py:123](turret_rl/envs/turret_env.py#L123).

---

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [SAC Paper](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018) - Soft Actor-Critic
- [PPO Paper](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) - Proximal Policy Optimization
