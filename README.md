# Turret vs Drone - Single-Shot RL Environment

A reinforcement learning environment where a stationary turret must make a **single decisive shot** to intercept a moving drone. Implemented using Gymnasium and trained with **Soft Actor-Critic (SAC)**.

## Problem Formulation

The agent controls a stationary turret at the origin `(0, 0)`. A drone flies across the battlefield in a straight line at constant velocity. The agent observes the drone's position and velocity and must decide:
1. **When to shoot** (timing)
2. **Where to shoot** (aiming angle)

**Key constraint**: The agent can fire **at most once per episode**. When the agent fires, the environment immediately determines whether the shot will intercept the drone using analytic physics, and the episode terminates with a reward based on hit/miss.

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

```bash
# Clone the repository
git clone <repository-url>
cd 9Mothers_TakeHome

# Install dependencies
pip install -r requirements.txt

# Or manually install core dependencies:
pip install gymnasium numpy stable-baselines3 matplotlib imageio imageio-ffmpeg

# Optional: Install wandb for experiment tracking
pip install wandb
```

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

### Evaluation

Evaluate a trained SAC model and record videos:

```bash
# Basic evaluation (5 episodes, deterministic)
python -m turret_rl.scripts.evaluate_and_record \
    --model turret_rl/models/turret_sac/turret_sac.zip \
    --episodes 10

# Specify algorithm explicitly (auto-detected by default)
python -m turret_rl.scripts.evaluate_and_record \
    --model turret_rl/models/turret_sac/turret_sac.zip \
    --algorithm SAC \
    --episodes 20

# With VecNormalize statistics (if used during training)
python -m turret_rl.scripts.evaluate_and_record \
    --model turret_rl/models/turret_sac/turret_sac.zip \
    --vec-normalize turret_rl/models/turret_sac/vec_normalize.pkl \
    --episodes 20 \
    --video-dir turret_rl/videos/eval_custom/

# Stochastic policy (for diversity)
python -m turret_rl.scripts.evaluate_and_record \
    --model turret_rl/models/turret_sac/turret_sac.zip \
    --stochastic \
    --episodes 50

# Evaluate PPO model (legacy)
python -m turret_rl.scripts.evaluate_and_record \
    --model turret_rl/models/turret_ppo/turret_ppo.zip \
    --algorithm PPO \
    --episodes 10

# Compare multiple models
python -m turret_rl.scripts.evaluate_and_record \
    --compare \
    turret_sac.zip turret_ppo.zip model3.zip
```

**Note:** The evaluation script automatically detects whether a model was trained with SAC or PPO based on the model file contents and filename.

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

## Expected Training Performance

### SAC Performance Milestones

| Timesteps | Hit Rate | Behavior |
|-----------|----------|----------|
| 0-10k | 0-5% | Random exploration (filling replay buffer) |
| 10k-50k | 5-25% | Learning basic aiming patterns |
| 50k-150k | 25-50% | Improving timing and interception |
| 150k-300k | 50-75% | Consistent hits, good decision-making |
| 300k-500k | 75-90% | Near-optimal performance |

**SAC advantages:**
- Faster initial learning due to replay buffer
- More stable training (off-policy updates)
- Better exploration through entropy regularization
- Superior sample efficiency (learns from all experiences)

### PPO Performance (Legacy)

| Timesteps | Hit Rate | Behavior |
|-----------|----------|----------|
| 0-50k | 0-10% | Random firing, poor timing |
| 50k-150k | 10-40% | Learning to wait, improving aim |
| 150k-300k | 40-70% | Consistent interception, good timing |
| 300k+ | 70-90% | Near-optimal performance |

**Monitor training with TensorBoard:**
```bash
# For SAC
tensorboard --logdir turret_rl/models/turret_sac/logs/

# For PPO
tensorboard --logdir turret_rl/models/turret_ppo/logs/
```

**Key metrics to monitor:**
- `rollout/ep_rew_mean`: Average episode reward
- `train/actor_loss`: Actor network loss (SAC)
- `train/critic_loss`: Critic network loss (SAC)
- `train/ent_coef`: Entropy coefficient (SAC, if auto-tuned)
- `train/entropy_loss`: Policy entropy (PPO)
- Custom callback metrics: `hit_rate`, `episode_length`

---

## Troubleshooting

### Low Hit Rate (<20% after 200k steps)

**For SAC:**
- Learning rate too high/low → Try `--lr 1e-4` or `--lr 5e-4`
- Small replay buffer → Increase `--buffer-size 500000`
- Not enough learning → Increase `--gradient-steps 2` or `--train-freq 2`
- Insufficient exploration → Check `train/ent_coef` in logs; try manual `--ent-coef 0.2`
- Training starting too late → Reduce `--learning-starts 500`

**For PPO (legacy):**
- Learning rate too high/low → Try `--lr 1e-4` or `--lr 1e-3`
- Insufficient exploration → Increase `ent_coef` in config
- Normalization disabled → Ensure VecNormalize is active
- Too few environments → Increase `--n-envs` to 8 or 16

### Agent Never Fires

**Symptoms:** `shot_result = 'no_shot_timeout'` in most episodes

**Fixes (SAC):**
- SAC should explore naturally; check that `train/ent_coef` > 0
- Verify reward config: `no_shot_penalty` should be very negative (-5.0)
- Check action output: run eval with `--stochastic` to verify action[1] distribution
- Increase exploration: set `--ent-coef 0.2` (if using manual tuning)

**Fixes (PPO):**
- Increase step penalty magnitude: `step_penalty = -0.01`
- Reduce miss penalty to encourage exploration: `miss_penalty = -0.5`
- Check action space: ensure `action[1]` can exceed 0

### Agent Fires Immediately Every Episode

**Symptoms:** Fires at step 0-1 every episode, even with low hit rate

**Root Cause:** The `step_penalty` creates a perverse incentive. If `step_penalty = -0.001`, waiting for a better shot is punished:
```
Fire immediately:  +1.0 - 0.001×1  = +0.999
Wait 50 steps:     +1.0 - 0.001×50 = +0.950  (worse!)
```

**Fixes:**
1. **Set `step_penalty = 0.0`** in [config.py](turret_rl/config/config.py) (recommended)
2. Make step penalty extremely small: `step_penalty = -0.00001`
3. Use reward shaping based on drone proximity instead
4. Retrain the model after changing the reward structure

**Note:** This is the **intended behavior** with the original `step_penalty = -0.001`. The agent correctly learned to minimize accumulated penalties by firing immediately. Change the config to fix this.

### "VecNormalize statistics not found" Warning

**Note:** SAC training does **not require** VecNormalize by default (unlike PPO). Only provide normalization stats if you trained with `--normalize`:

```bash
# Only if you used --normalize during training
python -m turret_rl.scripts.evaluate_and_record \
    --model turret_rl/models/turret_sac/turret_sac.zip \
    --vec-normalize turret_rl/models/turret_sac/vec_normalize.pkl
```

For standard SAC training without normalization, simply run:
```bash
python -m turret_rl.scripts.evaluate_and_record \
    --model turret_rl/models/turret_sac/turret_sac.zip
```

---

## Algorithm Comparison: SAC vs PPO

| Feature | SAC (Recommended) | PPO |
|---------|-------------------|-----|
| **Policy Type** | Off-policy | On-policy |
| **Sample Efficiency** | High (replay buffer) | Moderate |
| **Training Stability** | Very stable | Stable |
| **Exploration** | Entropy-regularized | Entropy bonus |
| **Memory Usage** | Higher (replay buffer) | Lower |
| **Continuous Actions** | Excellent | Good |
| **Short Episodes** | Excellent | Good |
| **Parallelization** | Single env sufficient | Benefits from many envs |
| **Hyperparameter Tuning** | Easier (auto-tuning) | More sensitive |

**Recommendation:** Use **SAC** for this environment due to:
1. Superior sample efficiency with the replay buffer
2. Better handling of continuous action spaces
3. More effective exploration through entropy maximization
4. Ideal for single-shot, short-horizon episodes

---

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [SAC Paper](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018) - Soft Actor-Critic
- [PPO Paper](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) - Proximal Policy Optimization
