#!/bin/bash

# Setup script for macOS systems
# This handles NumPy/PyTorch compatibility issues on Mac

echo "=================================================="
echo "Turret RL Environment Setup for macOS"
echo "=================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment with Python 3.10
echo ""
echo "Creating conda environment 'turret_rl' with Python 3.10..."
conda create -n turret_rl python=3.10 -y

# Get conda base directory
CONDA_BASE=$(conda info --base)

# Activate the environment in this script
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate turret_rl

echo ""
echo "Installing packages with pip (this may take a few minutes)..."

# Install packages with specific versions that work on macOS
pip install --upgrade pip

# Install NumPy 1.x for compatibility
pip install "numpy<2.0"

# Install PyTorch (CPU version for Mac)
pip install torch==2.2.2 torchvision --index-url https://download.pytorch.org/whl/cpu

# Install stable-baselines3 and dependencies
pip install "stable-baselines3==2.2.1"

# Install other required packages
pip install gymnasium matplotlib imageio imageio-ffmpeg tqdm rich

# Install optional packages
pip install tensorboard wandb

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To activate the environment, run:"
echo "    conda activate turret_rl"
echo ""
echo "To run the demo:"
echo "    conda activate turret_rl"
echo "    python demo/run_demo.py"
echo ""
echo "The demo will automatically use the Mac-compatible model."
echo ""
echo "To train a new model:"
echo "    python -m turret_rl.agents.train_sac --timesteps 100000"
echo ""