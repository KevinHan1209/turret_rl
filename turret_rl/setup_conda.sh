#!/bin/bash
# Quick setup script for conda environment

echo "=========================================="
echo "Turret RL - Conda Environment Setup"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "  - Anaconda: https://www.anaconda.com/download"
    exit 1
fi

echo "Conda found: $(conda --version)"
echo ""

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "Error: environment.yml not found in current directory"
    echo "Please run this script from the turret_rl directory"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^turret_rl "; then
    echo "Environment 'turret_rl' already exists."
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating environment..."
        conda env update -f environment.yml --prune
    else
        echo "Skipping update."
    fi
else
    echo "Creating new conda environment 'turret_rl'..."
    conda env create -f environment.yml
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate turret_rl"
echo ""
echo "To train an agent:"
echo "  python -m turret_rl.agents.train_ppo"
echo ""
echo "To evaluate and record videos:"
echo "  python -m turret_rl.scripts.evaluate_and_record"
echo ""
echo "For more information, see README.md"
echo ""