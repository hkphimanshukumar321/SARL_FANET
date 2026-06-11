@echo off
REM SARL on MARL Environment - Parallel Experiment Runner Script
REM Logs into wandb and runs the experiments in parallel.

echo ========================================================
echo   Starting SARL Parallel Experiment Pipeline
echo ========================================================

REM Install required packages if not already installed
echo Checking dependencies...
python -m pip install -q psutil tqdm optuna wandb
if %ERRORLEVEL% neq 0 (
    echo Failed to install dependencies. Please check your python/pip setup.
    exit /b %ERRORLEVEL%
)

REM Login to Wandb with the provided API key
echo Logging into wandb...
python -m wandb login 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
if %ERRORLEVEL% neq 0 (
    echo Wandb login failed. Proceeding without wandb...
    set WANDB_FLAG=
) else (
    echo Wandb login successful.
    set WANDB_FLAG=--wandb
)

REM Execute the parallel runner
echo Launching parallel experiments...
python experiments\parallel_runner.py %WANDB_FLAG% %*

echo ========================================================
echo   Experiment Pipeline Complete
echo ========================================================
pause
