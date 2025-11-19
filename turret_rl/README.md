# Turret vs Drone RL Environment

A 2D reinforcement learning environment where a stationary turret learns to intercept drones using projectile physics. The agent uses Proximal Policy Optimization (PPO) to develop an interception strategy.

## Project Overview

This environment simulates a defensive scenario where a turret positioned at the origin must shoot down drones flying across the battlefield. The turret can instantly rotate to any angle and fire bullets that travel at constant velocity. The challenge is to predict the drone's trajectory and time shots appropriately to achieve successful interceptions.

**Key Features:**
- Clean, modular codebase with type hints and comprehensive documentation
- Fully observable state space (drone position and velocity) - designed for easy modification to partial observability
- Continuous action space for precise turret control
- Configurable world physics and reward structure
- Integrated video recording for evaluation and visualization
- Production-ready training pipeline using Stable-Baselines3

## Installation

### Prerequisites
- Python 3.10 or higher (or Conda/Miniconda)
- FFmpeg (for video recording) - automatically installed with conda, or install via system package manager for pip

### Setup Instructions

#### Option 1: Using Conda (Recommended)

Conda automatically handles FFmpeg and other system dependencies.

1. Navigate to the project directory:
```bash
cd turret_rl
```

2. Create and activate the conda environment:
```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate turret_rl
```

3. Verify installation:
```bash
python -c "import gymnasium; import stable_baselines3; print('Setup successful!')"
```

**Note:** If you don't have a GPU or CUDA installed, edit `environment.yml` and:
- Remove the line: `- pytorch-cuda=11.8`
- Or replace it with: `- cpuonly`

To update an existing environment:
```bash
conda env update -f environment.yml --prune
```

To remove the environment:
```bash
conda deactivate
conda env remove -n turret_rl
```

#### Option 2: Using pip and venv

1. Navigate to the project directory:
```bash
cd turret_rl
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg manually:
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from https://ffmpeg.org/

## Training

To train a PPO agent with default settings:
```bash
python -m turret_rl.agents.train_ppo
```

### Training Options

```bash
python -m turret_rl.agents.train_ppo --timesteps 2000000 --n-envs 8 --lr 0.0003 --seed 42
```

Available arguments:
- `--timesteps`: Total training timesteps (default: 1,000,000)
- `--n-envs`: Number of parallel environments (default: 4)
- `--lr`: Learning rate (default: 3e-4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--checkpoint`: Path to checkpoint to resume training
- `--no-normalize`: Disable observation/reward normalization
- `--n-videos`: Number of evaluation videos to record during training (default: 10)
- `--episodes-per-video`: Episodes per video checkpoint (default: 3)
- `--no-videos`: Disable video recording for faster training

### Video Recording During Training

**NEW!** By default, the training process now records evaluation videos at regular intervals, allowing you to visualize learning progression:

```bash
# Default: 10 videos distributed evenly across training
python -m turret_rl.agents.train_ppo

# Customize: 20 videos with 5 episodes each
python -m turret_rl.agents.train_ppo --n-videos 20 --episodes-per-video 5

# Disable for faster training
python -m turret_rl.agents.train_ppo --no-videos
```

Videos are saved to `turret_rl/models/training_videos/` and show the agent's behavior at different training stages (early random behavior → learned interception strategies).

### Monitoring Training

Training progress is logged to TensorBoard. To visualize:
```bash
tensorboard --logdir turret_rl/models/logs
```

Then open http://localhost:6006 in your browser.

## Evaluation & Video Recording

To evaluate a trained model and record videos:
```bash
python -m turret_rl.scripts.evaluate_and_record
```

### Evaluation Options

```bash
python -m turret_rl.scripts.evaluate_and_record \
    --model turret_rl/models/turret_ppo.zip \
    --episodes 10 \
    --video-dir turret_rl/videos
```

Available arguments:
- `--model`: Path to trained model (default: turret_rl/models/turret_ppo.zip)
- `--episodes`: Number of episodes to evaluate (default: 5)
- `--no-video`: Disable video recording
- `--video-dir`: Directory for video output (default: turret_rl/videos)
- `--stochastic`: Use stochastic actions instead of deterministic
- `--seed`: Random seed (default: 42)

### Model Comparison

To compare multiple trained models:
```bash
python -m turret_rl.scripts.evaluate_and_record \
    --compare model1.zip model2.zip model3.zip \
    --episodes 20
```

## Environment Details

### Observation Space

The agent observes a continuous vector containing (all values normalized):

| Index | Description | Range | Normalization |
|-------|-------------|-------|---------------|
| 0-1 | Drone position (x, y) | [-1, 1] | Divided by world half-size (150m) |
| 2-3 | Drone velocity (vx, vy) | [-1, 1] | Divided by max drone speed (80 m/s) |
| 4 | Distance to drone | [0, 1] | Divided by max diagonal distance |
| 5-6 | Direction to drone (cos θ, sin θ) | [-1, 1] | Unit vector components |
| 7+ | Bullet states (4 values per bullet) | [0, 1] | Active flag, distance traveled, direction |

**Note:** This version provides full observability including drone velocity. The architecture is designed to easily remove velocity components for partial observability experiments.

### Action Space

2D continuous action vector:

| Index | Description | Range | Physical Mapping |
|-------|-------------|-------|------------------|
| 0 | Turret azimuth | [-1, 1] | Maps to [-π, π] radians |
| 1 | Fire command | [-1, 1] | Fire if > 0 and cooldown expired |

### Reward Function

The reward structure encourages efficient drone interception:

- **+1.0**: Successfully hitting the drone (episode terminates)
- **-0.001**: Per timestep (encourages faster solutions)
- **-0.01**: Per bullet fired (encourages accuracy)
- **0.0**: Terminal penalty if drone escapes

These values are configurable in `config/config.py`.

## Configuration

All environment parameters are centralized in `turret_rl/config/config.py`:

### World Configuration
- `world_size`: 300m × 300m battlefield
- `dt`: 0.05 second simulation timestep
- `drone_speed_min/max`: 20-80 m/s drone velocity range
- `bullet_speed`: 100 m/s projectile velocity
- `drone_radius`: 2m collision radius
- `min_approach_distance`: 100m minimum pass distance from origin

### Training Configuration
- PPO hyperparameters (learning rate, batch size, etc.)
- Network architecture
- Normalization settings
- Save frequency and logging options

### Customization

To modify parameters, either:
1. Edit the default values in `config/config.py`
2. Pass custom config objects to the training/evaluation scripts
3. Create configuration files and load them programmatically

## Project Structure

```
turret_rl/
├── envs/
│   └── turret_env.py          # Main Gymnasium environment
├── agents/
│   └── train_ppo.py           # PPO training implementation
├── scripts/
│   └── evaluate_and_record.py # Evaluation and video recording
├── config/
│   └── config.py              # Centralized configuration
├── utils/
│   └── visualization.py       # Video recording utilities
├── models/                    # Saved models directory
├── videos/                    # Recorded videos directory
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Development Notes

### Code Quality
- Full type hints throughout the codebase
- Comprehensive docstrings following Google style
- Modular design for easy extension
- Clear separation of concerns

### Future Enhancements

This codebase is structured to facilitate several extensions:

1. **Partial Observability**: Remove velocity from observation space, requiring velocity estimation from position history
2. **Advanced Rewards**: Add shaping based on bullet-drone distance for smoother learning
3. **Multiple Drones**: Extend to simultaneous multi-target scenarios
4. **Limited Ammunition**: Add strategic resource management
5. **Curved Trajectories**: Implement non-linear drone paths for increased difficulty

### Performance Considerations

- Environment uses vectorization for efficient parallel training
- Observation/reward normalization improves training stability
- Configurable simulation timestep allows speed/accuracy tradeoffs
- Video recording is optional to minimize evaluation overhead

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg for video recording:
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from https://ffmpeg.org/

2. **CUDA/GPU errors**: The environment runs on CPU by default. For GPU training, ensure PyTorch is properly configured for your system.

3. **Memory issues**: Reduce `n_envs` in training configuration or decrease `n_steps` for smaller batches.

## Citation

If you use this environment in your research, please cite:
```
@software{turret_rl_2024,
  title = {Turret vs Drone RL Environment},
  year = {2024},
  description = {A 2D reinforcement learning environment for projectile interception}
}
```

## License

This project is provided as educational/research code. Feel free to modify and extend for your needs.