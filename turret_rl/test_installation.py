#!/usr/bin/env python
"""Quick test script to verify installation is correct."""

import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports when running from turret_rl directory
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False

    try:
        import gymnasium
        print("✓ gymnasium")
    except ImportError as e:
        print(f"✗ gymnasium: {e}")
        return False

    try:
        import stable_baselines3
        print("✓ stable_baselines3")
    except ImportError as e:
        print(f"✗ stable_baselines3: {e}")
        return False

    try:
        import torch
        print(f"✓ torch (version {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("  Running on CPU")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False

    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False

    try:
        import imageio
        print("✓ imageio")
    except ImportError as e:
        print(f"✗ imageio: {e}")
        return False

    print("\nAll required packages imported successfully!")
    return True


def test_environment():
    """Test that the environment can be created and used."""
    print("\nTesting environment...")

    try:
        from turret_rl.envs.turret_env import TurretEnv
        from turret_rl.config.config import WorldConfig

        # Create environment
        env = TurretEnv()
        print("✓ Environment created")

        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✓ Environment reset (observation shape: {obs.shape})")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step executed (reward: {reward:.4f})")

        # Test a few more steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        print("✓ Multiple steps executed")

        env.close()
        print("✓ Environment closed")

        print("\nEnvironment test passed!")
        return True

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_imports():
    """Test that training modules can be imported."""
    print("\nTesting training modules...")

    try:
        from turret_rl.agents.train_ppo import train_ppo, create_env
        print("✓ Training module imported")

        from turret_rl.scripts.evaluate_and_record import evaluate_agent
        print("✓ Evaluation module imported")

        from turret_rl.utils.visualization import VideoRecorder
        print("✓ Visualization utilities imported")

        print("\nAll training modules imported successfully!")
        return True

    except Exception as e:
        print(f"✗ Training module import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Turret RL Installation Test")
    print("=" * 50)
    print()

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test environment
    if not test_environment():
        all_passed = False

    # Test training imports
    if not test_training_imports():
        all_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests PASSED!")
        print("=" * 50)
        print("\nYou're ready to start training!")
        print("Run: python -m turret_rl.agents.train_ppo")
        return 0
    else:
        print("✗ Some tests FAILED")
        print("=" * 50)
        print("\nPlease check the error messages above and ensure")
        print("all dependencies are installed correctly.")
        print("\nFor conda: conda env create -f environment.yml")
        print("For pip: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())