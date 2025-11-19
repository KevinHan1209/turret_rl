@echo off
REM Quick setup script for conda environment (Windows)

echo ==========================================
echo Turret RL - Conda Environment Setup
echo ==========================================
echo.

REM Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first:
    echo   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo   - Anaconda: https://www.anaconda.com/download
    pause
    exit /b 1
)

conda --version
echo.

REM Check if environment.yml exists
if not exist "environment.yml" (
    echo Error: environment.yml not found in current directory
    echo Please run this script from the turret_rl directory
    pause
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr /C:"turret_rl" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Environment 'turret_rl' already exists.
    set /p UPDATE="Do you want to update it? (y/n): "
    if /i "%UPDATE%"=="y" (
        echo Updating environment...
        conda env update -f environment.yml --prune
    ) else (
        echo Skipping update.
    )
) else (
    echo Creating new conda environment 'turret_rl'...
    conda env create -f environment.yml
)

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo To activate the environment, run:
echo   conda activate turret_rl
echo.
echo To train an agent:
echo   python -m turret_rl.agents.train_ppo
echo.
echo To evaluate and record videos:
echo   python -m turret_rl.scripts.evaluate_and_record
echo.
echo For more information, see README.md
echo.
pause