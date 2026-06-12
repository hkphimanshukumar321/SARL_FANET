import optuna
import time
import os
from tqdm import tqdm
from datetime import timedelta

def monitor_sweep():
    """
    Connects to the Optuna SQLite databases to track the global progress.
    Shows real-time status even before any trials complete.
    """
    
    # Target trials per algorithm
    targets = {
        "full_v100_sweep_magat_d3qn": 20,
        "full_v100_sweep_qmix": 20,
        "full_v100_sweep_vdn": 20,
        "full_v100_sweep_iql": 20,
        "full_v100_sweep_dqn": 48,
        "full_v100_sweep_mca_d3qn": 48,
        "full_v100_sweep_ppo": 48,
        "full_v100_sweep_a2c": 48
    }

    # Workers per algo (for wall-clock ETA calculation)
    workers_count = {
        "magat_d3qn": 4,
        "qmix": 4,
        "vdn": 4,
        "iql": 4,
        "dqn": 2,
        "mca_d3qn": 2,
        "ppo": 2,
        "a2c": 2
    }

    # Resolve project root from this script's location (utils/monitor_sweep.py -> project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_dir = os.path.join(project_root, "results", "optuna")

    print("=" * 60)
    print("  LIVE OPTUNA SWEEP TRACKER (Ctrl+C to exit)")
    print("  Refreshes every 30 seconds")
    print("=" * 60)
    
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print("=" * 60)
            print("  OPTUNA SWEEP STATUS")
            print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
            print("=" * 60)
            
            max_eta_seconds = 0
            total_completed = 0
            total_target = sum(targets.values())
            any_db_found = False

            for study_name, target in targets.items():
                algo_name = study_name.replace("full_v100_sweep_", "").upper()
                db_path = os.path.join(base_dir, study_name, f"{study_name}.db")
                
                if not os.path.exists(db_path):
                    print(f"  {algo_name:<12} | DB not created yet (waiting to start...)")
                    continue
                
                any_db_found = True
                
                try:
                    optuna.logging.set_verbosity(optuna.logging.ERROR)
                    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
                    
                    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
                    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                    
                    n_done = len(completed)
                    n_run = len(running)
                    n_fail = len(failed)
                    total_completed += n_done
                    
                    # Progress bar using simple characters
                    bar_len = 20
                    filled = min(bar_len, int(bar_len * n_done / target)) if target > 0 else 0
                    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                    
                    status_parts = [f"{n_done}/{target} done"]
                    if n_run > 0:
                        run_ids = [str(t.number) for t in running]
                        status_parts.append(f"{n_run} running (T#{','.join(run_ids)})")
                    if n_fail > 0:
                        status_parts.append(f"{n_fail} failed")
                    
                    # ETA calculation
                    algo_key = study_name.replace("full_v100_sweep_", "")
                    remaining = target - n_done
                    eta_str = ""
                    if remaining > 0 and n_done > 0:
                        recent = completed[-5:]
                        avg_s = sum((t.datetime_complete - t.datetime_start).total_seconds() for t in recent) / len(recent)
                        n_jobs = workers_count.get(algo_key, 1)
                        study_eta = (remaining / n_jobs) * avg_s
                        if study_eta > max_eta_seconds:
                            max_eta_seconds = study_eta
                        eta_td = timedelta(seconds=int(study_eta))
                        eta_str = f" | ETA: {eta_td}"
                    elif remaining > 0 and n_run > 0:
                        eta_str = " | ETA: calculating..."
                    elif remaining == 0:
                        eta_str = " | \u2605 DONE!"
                    
                    print(f"  {algo_key.upper():<12} |{bar}| {', '.join(status_parts)}{eta_str}")
                    
                except Exception as e:
                    print(f"  {algo_name:<12} | Error reading DB: {str(e)[:40]}")
            
            # Global summary
            print("-" * 60)
            pct = (total_completed / total_target * 100) if total_target > 0 else 0
            print(f"  TOTAL PROGRESS: {total_completed}/{total_target} trials ({pct:.1f}%)")
            
            if max_eta_seconds > 0:
                g = timedelta(seconds=int(max_eta_seconds))
                days = g.days
                hours, rem = divmod(g.seconds, 3600)
                minutes, _ = divmod(rem, 60)
                print(f"  NET TIME LEFT:  {days} Days, {hours} Hours, {minutes} Minutes")
            elif total_completed == total_target and any_db_found:
                print("  \u2605 ALL TRIALS COMPLETED! \u2605")
            elif not any_db_found:
                print("  Waiting for first database to be created...")
                print(f"  Looking in: {os.path.abspath(base_dir)}")
            else:
                print("  NET TIME LEFT:  Calculating after first trial completes...")
            
            print("=" * 60)
            print("  Press Ctrl+C to exit (sweep continues in background)")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\nExiting monitor. Background sweep still running safely!")

if __name__ == "__main__":
    monitor_sweep()
