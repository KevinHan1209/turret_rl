#!/bin/bash
# Setup script for Turret vs Drone RL environment
# Creates conda environment with Python and installs packages via pip (faster than conda)

set -e  # Exit on error

ENV_NAME="turret_rl"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "Setting up Turret RL Environment"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing '${ENV_NAME}' environment..."
    conda env remove -n ${ENV_NAME} -y
fi

# Create new environment with just Python
echo ""
echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install PyTorch (with CUDA support if available)
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected - installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected - installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install \
    gymnasium==0.29.1 \
    stable-baselines3==2.2.1 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.8 \
    tqdm>=4.65.0

# Install optional dependencies
echo ""
echo "Installing optional dependencies..."
pip install \
    tensorboard>=2.14.0 \
    wandb>=0.15.0

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run the demo:"
echo "  python demo/run_demo.py"
echo ""
echo "To train a new model:"
echo "  python -m turret_rl.agents.train_sac --timesteps 1000000"
echo ""
