import os
import sys
import time
import subprocess
import argparse
from tqdm import tqdm

try:
    import psutil
except ImportError:
    print("Please install psutil: pip install psutil")
    sys.exit(1)

# List of algorithms from run_sarl_comparison.py
ALGORITHMS = ["Tabular", "DQN", "PPO", "A2C", "MCA-D3QN", "MCA-PPO"]

def check_memory(limit_percent=85.0):
    """Returns True if memory usage is below the limit."""
    return psutil.virtual_memory().percent < limit_percent

import datetime

# Create a shared output directory for this parallel run
SHARED_OUT_DIR = os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "results", 
    f"sarl_comparison_parallel_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

def run_experiment(algo, wandb_flag, seed, timesteps):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_sarl_comparison.py"),
        "--algo", algo,
        "--seed", str(seed),
        "--out-dir", SHARED_OUT_DIR,
        "--skip-baselines",
        "--skip-plots"
    ]
    if timesteps is not None:
        cmd.extend(["--timesteps", str(timesteps)])
    if os.path.exists("best_weights_and_params.json"):
        import json
        try:
            with open("best_weights_and_params.json", "r") as f:
                all_params = json.load(f)
            # Find the params for this specific algorithm
            algo_key = algo.lower().replace("-", "_")
            if algo_key in all_params:
                params = all_params[algo_key]
                if "lr" in params:
                    cmd.extend(["--lr", str(params["lr"])])
                if "batch_size" in params:
                    cmd.extend(["--batch-size", str(params["batch_size"])])
                print(f"  [Optuna] Injected tuned hyperparameters for {algo}: {params}")
        except Exception as e:
            print(f"  [Optuna] Failed to load tuned params: {e}")
        
    # Do not redirect stdout/stderr so that tqdm progress shows up in the main console
    process = subprocess.Popen(cmd)
    return algo, process

def optuna_objective(trial, algo):
    """
    Optuna implementation: tunes learning rate and batch size for the given algorithm.
    """
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_sarl_comparison.py"),
        "--algo", algo,
        "--seed", "42",
        "--out-dir", SHARED_OUT_DIR,
        "--skip-baselines",
        "--skip-plots",
        "--skip-eval",
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--timesteps", "10000" # Full tune length
    ]
    process = subprocess.Popen(cmd)
    process.wait()
    
    import pandas as pd
    try:
        csv_path = os.path.join(SHARED_OUT_DIR, "csv", f"{algo}_training_rewards.csv")
        df = pd.read_csv(csv_path)
        return df['reward'].tail(max(1, len(df)//10)).mean()
    except Exception:
        return -1000.0

def main():
    parser = argparse.ArgumentParser(description="Parallel SARL Runner with OOM Guardrails")
    parser.add_argument("--wandb", action="store_true", help="Enable Wandb logging")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() or 4, help="Max concurrent processes")
    parser.add_argument("--mem-limit", type=float, default=85.0, help="Pause starting new jobs if memory exceeds this %")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the runs")
    parser.add_argument("--timesteps", type=int, default=None, help="Training timesteps")
    parser.add_argument("--use-optuna", action="store_true", help="Run hyperparameter sweep using Optuna")
    args = parser.parse_args()

    if args.use_optuna:
        import optuna
        import json
        from functools import partial
        
        algos_to_tune = ["dqn", "ppo", "a2c", "mca_d3qn", "mca_ppo"]
        print(f"[Optuna Mode] Running Optuna sweep for algorithms: {algos_to_tune}")
        
        master_params = {}
        try:
            for algo in algos_to_tune:
                print(f"\n[Optuna] Starting tuning for {algo.upper()} (16 trials x 10,000 steps)...")
                study = optuna.create_study(direction="maximize")
                objective_with_algo = partial(optuna_objective, algo=algo)
                study.optimize(objective_with_algo, n_trials=16)
                print(f"[{algo.upper()}] Best params: {study.best_params}")
                master_params[algo] = study.best_params
                
            with open("best_weights_and_params.json", "w") as f:
                json.dump(master_params, f, indent=4)
            print("\n[Optuna] Tuning Complete! Wrote all best parameters to best_weights_and_params.json")
        except ImportError:
            print("Optuna not installed. Run: pip install optuna")
        return

    print(f"Shared Output Directory: {SHARED_OUT_DIR}")
    os.makedirs(SHARED_OUT_DIR, exist_ok=True)
    
    print("\n--- Running Baseline Sweep (Once) ---")
    subprocess.run([
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_sarl_comparison.py"),
        "--algo", "none",
        "--seed", str(args.seed),
        "--out-dir", SHARED_OUT_DIR,
        "--skip-plots"
    ])

    print(f"\n--- Starting Parallel Execution of {len(ALGORITHMS)} Algorithms ---")
    print(f"Max Workers: {args.max_workers}, Memory Limit: {args.mem_limit}%")

    active_processes = []
    pending_algos = ALGORITHMS.copy()
    completed = 0

    try:
        while pending_algos or active_processes:
            # Check finished processes
            for ap in list(active_processes):
                algo, proc = ap
                if proc.poll() is not None:  # Process finished
                    active_processes.remove(ap)
                    completed += 1
                    if proc.returncode != 0:
                        print(f"\n[ERROR] Algorithm {algo} failed with code {proc.returncode}.")

            # Start new processes if we have capacity and memory
            while pending_algos and len(active_processes) < args.max_workers:
                if not check_memory(args.mem_limit):
                    break

                algo = pending_algos.pop(0)
                print(f"--> Launching Algorithm: {algo}")
                algo, proc = run_experiment(algo, args.wandb, args.seed, args.timesteps)
                active_processes.append((algo, proc))

            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted! Terminating active processes...")
        for algo, proc in active_processes:
            proc.terminate()
    finally:
        print(f"\nFinished. {completed}/{len(ALGORITHMS)} completed.")
        
        print("\n--- Aggregating Results & Generating Plots ---")
        import pandas as pd
        csv_dir = os.path.join(SHARED_OUT_DIR, "csv")
        csv_files = [f for f in os.listdir(csv_dir) if f.startswith("sarl_evaluation_results_") and f.endswith(".csv")]
        
        all_dfs = []
        for f in csv_files:
            all_dfs.append(pd.read_csv(os.path.join(csv_dir, f)))
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(os.path.join(csv_dir, "sarl_evaluation_results.csv"), index=False)
            print("Combined evaluation CSV generated.")
            
            # Run the plot generation
            subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), "run_sarl_comparison.py"),
                "--algo", "none",
                "--out-dir", SHARED_OUT_DIR,
                "--skip-baselines"
            ])
            print(f"Done! Find results in {SHARED_OUT_DIR}")
        else:
            print("No evaluation results found to aggregate.")

if __name__ == "__main__":
    main()
