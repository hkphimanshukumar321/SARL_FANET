#!/bin/bash
# SARL on MARL Environment - Parallel Experiment Runner Script
# Logs into wandb and runs the experiments in parallel.

echo "========================================================"
echo "  Starting SARL Parallel Experiment Pipeline"
echo "========================================================"

# Install required packages if not already installed
echo "Checking dependencies..."
python3 -m pip install -q psutil tqdm optuna wandb
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Please check your python/pip setup."
    exit 1
fi

# Login to Wandb with the provided API key
echo "Logging into wandb..."
python3 -m wandb login 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
if [ $? -ne 0 ]; then
    echo "Wandb login failed. Proceeding without wandb..."
    WANDB_FLAG=""
else
    echo "Wandb login successful."
    WANDB_FLAG="--wandb"
fi

# Execute the parallel runner using nohup
echo "Launching parallel experiments in the background..."
nohup python3 experiments/parallel_runner.py $WANDB_FLAG "$@" > experiments_runner.log 2>&1 &
echo "Process started with PID $!"
echo "Check experiments_runner.log for overall progress and results/ for algorithm-specific logs."

echo "========================================================"
echo "  Experiment Pipeline Kicked Off"
echo "========================================================"
