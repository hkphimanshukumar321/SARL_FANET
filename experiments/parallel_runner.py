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

def run_experiment(algo, wandb_flag, seed, timesteps):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_sarl_comparison.py"),
        "--algo", algo,
        "--seed", str(seed)
    ]
    if timesteps is not None:
        cmd.extend(["--timesteps", str(timesteps)])
    if wandb_flag:
        cmd.append("--wandb")
        
    # We will pipe stdout/stderr or just let them go to their own logs
    # To keep console clean for tqdm, we redirect output to files
    log_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{algo}_run.log")
    
    with open(log_path, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return algo, process

def optuna_objective(trial):
    """
    Optuna Skeleton implementation.
    If you wish to do hyperparameter tuning later, modify this block.
    """
    # Example hyperparameters:
    # lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # You would then pass these to your subprocess as command line arguments
    # and read them in run_sarl_comparison.py
    # cmd = [..., "--lr", str(lr), "--batch-size", str(batch_size)]
    # process = subprocess.Popen(cmd)
    # process.wait()
    # return the metric you want to maximize (e.g. throughput or avg_reward)
    return 0.0

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
        print("[Optuna Mode] Please define your search space in optuna_objective().")
        try:
            import optuna
            study = optuna.create_study(direction="maximize")
            study.optimize(optuna_objective, n_trials=10)
            print("Best params:", study.best_params)
        except ImportError:
            print("Optuna not installed. Run: pip install optuna")
        return

    print(f"Starting Parallel Execution of {len(ALGORITHMS)} Algorithms")
    print(f"Max Workers: {args.max_workers}, Memory Limit: {args.mem_limit}%")

    active_processes = []
    pending_algos = ALGORITHMS.copy()
    completed = 0

    pbar = tqdm(total=len(ALGORITHMS), desc="Overall Progress")

    try:
        while pending_algos or active_processes:
            # Check finished processes
            for ap in list(active_processes):
                algo, proc = ap
                if proc.poll() is not None:  # Process finished
                    active_processes.remove(ap)
                    completed += 1
                    pbar.update(1)
                    if proc.returncode != 0:
                        print(f"\n[ERROR] Algorithm {algo} failed with code {proc.returncode}. Check results/{algo}_run.log")

            # Start new processes if we have capacity and memory
            while pending_algos and len(active_processes) < args.max_workers:
                if not check_memory(args.mem_limit):
                    # We have hit the memory limit
                    # Break to wait for a process to finish and free up memory
                    break

                algo = pending_algos.pop(0)
                algo, proc = run_experiment(algo, args.wandb, args.seed, args.timesteps)
                active_processes.append((algo, proc))

            # Small sleep to prevent tight loop
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted! Terminating active processes...")
        for algo, proc in active_processes:
            proc.terminate()
    finally:
        pbar.close()
        print(f"\nFinished. {completed}/{len(ALGORITHMS)} completed.")

if __name__ == "__main__":
    main()
