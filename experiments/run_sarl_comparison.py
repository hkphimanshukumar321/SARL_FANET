"""
run_sarl_comparison.py -- SARL on MARL Experiment Runner
=============================================================
Trains and evaluates Single-Agent RL (SARL) agents on a Multi-Agent
environment (MARLMacEnv), using a unified wrapper (MARLtoSARLWrapper).
This allows central control over distributed UAV agents.

Training Architecture:
  ALL SARL agents → Monitor(MARLtoSARLWrapper(...)) + DummyVecEnv

Models:
  - Tabular Q-Learning  (MARLtoSARLWrapper → Discrete(2))
  - SB3 DQN             (MARLtoSARLWrapper → Discrete(2))
  - SB3 PPO             (MARLtoSARLWrapper → Discrete(2))
  - SB3 A2C             (MARLtoSARLWrapper → Discrete(2))
  - MCA-D3QN (Ours)     (MARLtoSARLWrapper → Discrete(2), SB3 DuelingDQN)

Usage:
    python experiments/run_sarl_comparison.py
    python experiments/run_sarl_comparison.py --dry-run
    python experiments/run_sarl_comparison.py --timesteps 50000
"""

import os
import sys
import json
import argparse
import datetime
import time
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

from tqdm import tqdm

# Project path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs import config as params
from configs.cluster_config import ClusterConfig as CC
from configs.sarl_config import RLConfig
from envs.burst_scheduler import VALID_RHO_LEVELS, encode_action
from utils.rich_logger import log_step, print_banner

# =====================================================================
# Constants
# =====================================================================
MAC_NAMES = {0: "TDMA", 1: "CSMA_CA"}


def default_fixed_action(mac_mode: int) -> int:
    rho_index = len(VALID_RHO_LEVELS) // 2
    return encode_action(mac_mode, rho_index)


def make_output_dir():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(project_root, "results", f"sarl_comparison_{ts}")
    for sub in ["images", "csv", "checkpoints", "logs"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base


def aggregate_cluster_infos(raw_infos):
    if not raw_infos:
        return {"throughput_mbps": 0.0, "delay_ms": 0.0, "drops": 0.0, "collisions": 0.0}
    alive_infos = [info for info in raw_infos.values() if info.get("alive", False)]
    if not alive_infos:
        return {"throughput_mbps": 0.0, "delay_ms": 0.0, "drops": 0.0, "collisions": 0.0}
    return {
        "throughput_mbps": float(sum(i.get("throughput_mbps", 0.0) for i in alive_infos)),
        "delay_ms": float(np.mean([i.get("delay_ms", 0.0) for i in alive_infos])),
        "drops": float(sum(i.get("drops", 0.0) for i in alive_infos)),
        "collisions": float(sum(i.get("collisions", 0.0) for i in alive_infos)),
        "jains_fairness": float(alive_infos[0].get("jains_fairness", 1.0)),
        "leader_health": float(np.mean([i.get("leader_health", 1.0) for i in alive_infos])),
    }


# =====================================================================
# Step 1: Baseline MAC Simulations
# =====================================================================
def step1_baselines(pps_list, seed, log):
    from envs.marl_mac_env import MARLMacEnv

    log("\n" + "=" * 60)
    log(" STEP 1: Baseline MAC Simulations (MARL Env)")
    log("=" * 60)

    results = []
    protocol_map = {
        default_fixed_action(0): "TDMA",
        default_fixed_action(1): "CSMA",
    }

    for idx, pps in enumerate(tqdm(pps_list, desc="Baseline Sweep", unit="load")):
        params.SWEEP_MAX_PPS = int(pps)
        setattr(params, "OFFERED_PPS", int(pps))
        row = {"Offered_Load_pps": int(pps)}

        for fixed_action, proto_name in protocol_map.items():
            env = MARLMacEnv(seed=seed + idx + fixed_action * 10000)
            obs, _ = env.reset(seed=seed + idx + fixed_action * 10000)

            ep_thr, ep_del = [], []
            ep_drops, ep_col = 0, 0
            ep_fairness, ep_health = [], []

            while env.agents:
                actions = {a: fixed_action for a in env.possible_agents}
                obs, rewards, terms, truncs, infos = env.step(actions)
                agg = aggregate_cluster_infos(infos)
                ep_thr.append(agg["throughput_mbps"])
                ep_del.append(agg["delay_ms"])
                ep_drops += agg["drops"]
                ep_col += agg["collisions"]
                ep_fairness.append(agg["jains_fairness"])
                ep_health.append(agg["leader_health"])

            n_steps = max(len(ep_thr), 1)
            row[f"{proto_name}_Throughput_Mbps"] = round(sum(ep_thr) / n_steps, 6)
            row[f"{proto_name}_Delay_s"] = round(sum(ep_del) / n_steps / 1000.0, 8)
            row[f"{proto_name}_Drops"] = ep_drops
            row[f"{proto_name}_Collisions"] = ep_col
            row[f"{proto_name}_Fairness"] = round(sum(ep_fairness) / n_steps, 4)
            row[f"{proto_name}_Health"] = round(sum(ep_health) / n_steps, 4)

        results.append(row)
    return pd.DataFrame(results)


# =====================================================================
# Step 2: Unified Training — Workers
# =====================================================================
def _train_sarl_worker(kwargs):
    """
    Worker function to train a single SARL algorithm.
    Runs entirely isolated to prevent multiprocess GPU locking.
    """
    algo = kwargs['algo']
    timesteps = kwargs['timesteps']
    cp_dir = kwargs['cp_dir']
    csv_dir = kwargs['csv_dir']
    seed = kwargs['seed']
    force_retrain = kwargs['force_retrain']
    
    # Extract tuning kwargs cleanly
    tuning_kwargs = {}
    if 'lr' in kwargs and kwargs['lr'] is not None:
        tuning_kwargs['learning_rate'] = kwargs['lr']
    if 'batch_size' in kwargs and kwargs['batch_size'] is not None:
        tuning_kwargs['batch_size'] = kwargs['batch_size']

    from envs.marl_sarl_wrapper import MARLtoSARLWrapper
    from utils.device_manager import resolve_device

    pid = os.getpid()
    train_device = resolve_device("train")
    print(f"  [Worker {pid}] SARL training: {algo.upper()} on device: {train_device}")

    if algo == 'tabular':
        from algorithms.rl.tabular_qlearning import TabularQLearning
        save_path = os.path.join(cp_dir, "unified_tabular_model.json")
        if (not force_retrain) and os.path.exists(save_path):
            return f"{algo.upper()} skipped (checkpoint)"

        env = MARLtoSARLWrapper(seed=seed)
        model = TabularQLearning(seed=seed)
        obs, _ = env.reset(seed=seed)
        rewards_log, steps_log = [], []
        ep_reward = 0.0

        pbar = tqdm(total=timesteps, desc=f"TABULAR", unit="step", leave=True)
        for step in range(timesteps):
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            model.learn(obs, action, reward, next_obs, done=(terminated or truncated))
            obs = next_obs
            ep_reward += reward
            pbar.update(1)
            if terminated or truncated:
                obs, _ = env.reset()
                rewards_log.append(ep_reward)
                steps_log.append(step + 1)
                pbar.set_postfix({"R": f"{ep_reward:.2f}"})
                ep_reward = 0.0
        pbar.close()
        model.save(save_path)
        if steps_log:
            df = pd.DataFrame({"step": steps_log, "reward": rewards_log})
            df.to_csv(os.path.join(csv_dir, "tabular_training_rewards.csv"), index=False)
            df.to_csv(os.path.join(cp_dir, "tabular_training_rewards.csv"), index=False)
        return f"{algo.upper()} trained."

    elif algo in ['dqn', 'ppo', 'a2c']:
        from algorithms.rl.sb3_baselines import create_sb3_baseline
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        save_path = os.path.join(cp_dir, f"unified_{algo}_model")
        if (not force_retrain) and os.path.exists(save_path + ".zip"):
            return f"{algo.upper()} skipped (checkpoint)"

        class RewardLogger(BaseCallback):
            def __init__(self, total):
                super().__init__(verbose=0)
                self.rewards, self.steps = [], []
                self.pbar = tqdm(total=total, desc=algo.upper(), unit="step", leave=True)
            def _on_step(self):
                self.pbar.update(1)
                if "episode" in self.locals.get("infos", [{}])[0]:
                    ep = self.locals["infos"][0]["episode"]
                    self.rewards.append(ep["r"])
                    self.steps.append(self.num_timesteps)
                    self.pbar.set_postfix({"R": f"{ep['r']:.2f}"})
                return True
            def _on_training_end(self):
                self.pbar.close()

        env = Monitor(MARLtoSARLWrapper(seed=seed))
        vec_env = DummyVecEnv([lambda: env])
        model = create_sb3_baseline(vec_env, algo_name=algo, seed=seed, **tuning_kwargs)
        cb = RewardLogger(timesteps)
        model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
        model.save(save_path)
        if cb.steps:
            df = pd.DataFrame({"step": cb.steps, "reward": cb.rewards})
            df.to_csv(os.path.join(csv_dir, f"{algo}_training_rewards.csv"), index=False)
            df.to_csv(os.path.join(cp_dir, f"{algo}_training_rewards.csv"), index=False)
        return f"{algo.upper()} trained."

    elif algo == 'mca_d3qn':
        from algorithms.rl.custom_mca_d3qn import create_mca_d3qn
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        save_path = os.path.join(cp_dir, "unified_mca_d3qn_model")
        if (not force_retrain) and os.path.exists(save_path + ".zip"):
            return f"MCA-D3QN skipped (checkpoint)"

        class RewardLogger(BaseCallback):
            def __init__(self, total):
                super().__init__(verbose=0)
                self.rewards, self.steps = [], []
                self.pbar = tqdm(total=total, desc="MCA-D3QN", unit="step", leave=True)
            def _on_step(self):
                self.pbar.update(1)
                if "episode" in self.locals.get("infos", [{}])[0]:
                    ep = self.locals["infos"][0]["episode"]
                    self.rewards.append(ep["r"])
                    self.steps.append(self.num_timesteps)
                    self.pbar.set_postfix({"R": f"{ep['r']:.2f}"})
                return True
            def _on_training_end(self):
                self.pbar.close()

        env = Monitor(MARLtoSARLWrapper(seed=seed))
        vec_env = DummyVecEnv([lambda: env])
        model = create_mca_d3qn(vec_env, seed=seed, **tuning_kwargs)
        cb = RewardLogger(timesteps)
        model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
        model.save(save_path)
        if cb.steps:
            df = pd.DataFrame({"step": cb.steps, "reward": cb.rewards})
            df.to_csv(os.path.join(csv_dir, "mca_d3qn_training_rewards.csv"), index=False)
            df.to_csv(os.path.join(cp_dir, "mca_d3qn_training_rewards.csv"), index=False)
        return f"MCA-D3QN trained."

    elif algo == 'mca_ppo':
        from algorithms.rl.custom_mca_ppo import create_mca_ppo
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        save_path = os.path.join(cp_dir, "unified_mca_ppo_model")
        if (not force_retrain) and os.path.exists(save_path + ".zip"):
            return f"MCA-PPO skipped (checkpoint)"

        class RewardLogger(BaseCallback):
            def __init__(self, total):
                super().__init__(verbose=0)
                self.rewards, self.steps = [], []
                self.pbar = tqdm(total=total, desc="MCA-PPO", unit="step", leave=True)
            def _on_step(self):
                self.pbar.update(1)
                if "episode" in self.locals.get("infos", [{}])[0]:
                    ep = self.locals["infos"][0]["episode"]
                    self.rewards.append(ep["r"])
                    self.steps.append(self.num_timesteps)
                    self.pbar.set_postfix({"R": f"{ep['r']:.2f}"})
                return True
            def _on_training_end(self):
                self.pbar.close()

        env = Monitor(MARLtoSARLWrapper(seed=seed))
        vec_env = DummyVecEnv([lambda: env])
        model = create_mca_ppo(vec_env, seed=seed, **tuning_kwargs)
        cb = RewardLogger(timesteps)
        model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
        model.save(save_path)
        if cb.steps:
            df = pd.DataFrame({"step": cb.steps, "reward": cb.rewards})
            df.to_csv(os.path.join(csv_dir, "mca_ppo_training_rewards.csv"), index=False)
            df.to_csv(os.path.join(cp_dir, "mca_ppo_training_rewards.csv"), index=False)
        return f"MCA-PPO trained."

    return f"Unknown SARL algo: {algo}"


def _dispatch_training(kw):
    """Module-level dispatch for multiprocessing (must be picklable)."""
    return _train_sarl_worker(kw)


def step2_train(out_dir, sarl_timesteps, log,
                force_retrain=False, use_checkpoints_only=False, **kwargs):
    """Launch all training jobs across processes."""
    log("\n" + "=" * 60)
    log(" STEP 2: Training SARL agents on MARL env")
    log("=" * 60)

    cp_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(cp_dir, exist_ok=True)
    csv_dir = os.path.join(out_dir, "csv")

    # Build task lists
    sarl_tasks = []
    if getattr(params, "RUN_TABULAR_QLEARNING", True): sarl_tasks.append('tabular')
    if getattr(params, "RUN_DQN", True): sarl_tasks.append('dqn')
    if getattr(params, "RUN_PPO", True): sarl_tasks.append('ppo')
    if getattr(params, "RUN_A2C", True): sarl_tasks.append('a2c')
    if getattr(params, "RUN_CUSTOM_RL", True): 
        sarl_tasks.append('mca_d3qn')
        sarl_tasks.append('mca_ppo')

    all_kwargs = []
    for algo in sarl_tasks:
        kw = {
            'algo': algo, 'timesteps': sarl_timesteps,
            'cp_dir': cp_dir, 'csv_dir': csv_dir, 'seed': params.SEED,
            'force_retrain': force_retrain,
        }
        kw.update(kwargs) # inject tuning args
        all_kwargs.append(kw)

    n_workers = min(os.cpu_count() or 1, len(all_kwargs))
    log(f"  Launching {len(all_kwargs)} training jobs on {n_workers} workers")
    log(f"  SARL: {sarl_tasks} ({sarl_timesteps} timesteps each)")

    if use_checkpoints_only and not force_retrain:
        log("  Checkpoint policy: USE_CHECKPOINTS_ONLY (training skipped)")
        return cp_dir

    with multiprocessing.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_dispatch_training, all_kwargs):
            log(f"    --> {result}")

    log("  All training complete.")
    return cp_dir


# =====================================================================
# Step 3: Evaluate all Models
# =====================================================================
def step3_evaluate(pps_list, cp_dir, out_dir, seed, log, deterministic_eval=True):
    """Load all models and evaluate across traffic sweep on MARL env."""
    log("\n" + "=" * 60)
    log(" STEP 3: Evaluation Sweep")
    log("=" * 60)

    import torch
    from envs.marl_mac_env import MARLMacEnv
    from envs.marl_sarl_wrapper import MARLtoSARLWrapper

    models = {}  # name → model

    # --- Load SARL models ---
    def _try_load_sb3(name, cls, filename):
        path = os.path.join(cp_dir, filename)
        if os.path.exists(path + ".zip"):
            models[name] = cls.load(path)
            log(f"  Loaded SARL: {name}")

    from stable_baselines3 import DQN, PPO, A2C
    if getattr(params, "RUN_CUSTOM_RL", True):
        _try_load_sb3("MCA-D3QN", DQN, "unified_mca_d3qn_model")
        _try_load_sb3("MCA-PPO", PPO, "unified_mca_ppo_model")
    if getattr(params, "RUN_DQN", True):
        _try_load_sb3("DQN", DQN, "unified_dqn_model")
    if getattr(params, "RUN_PPO", True):
        _try_load_sb3("PPO", PPO, "unified_ppo_model")
    if getattr(params, "RUN_A2C", True):
        _try_load_sb3("A2C", A2C, "unified_a2c_model")

    if getattr(params, "RUN_TABULAR_QLEARNING", True):
        tab_path = os.path.join(cp_dir, "unified_tabular_model.json")
        if os.path.exists(tab_path):
            from algorithms.rl.tabular_qlearning import TabularQLearning
            tab = TabularQLearning()
            tab.load(tab_path)
            tab.epsilon = 0.0
            models["Tabular"] = tab
            log(f"  Loaded SARL: Tabular")

    if not models:
        log("  WARNING: No models found. Skipping evaluation.")
        return pd.DataFrame()

    log(f"  Evaluating {len(models)} models across {len(pps_list)} load points")

    # --- Evaluation loop ---
    sarl_wrapper = MARLtoSARLWrapper()
    results = []

    for pps in tqdm(pps_list, desc="Eval Sweep", unit="load"):
        params.SWEEP_MAX_PPS = pps
        setattr(params, "OFFERED_PPS", int(pps))

        for model_name, model in models.items():
            total_thr, total_delay, total_drops, total_collisions, steps = 0, 0, 0, 0, 0
            total_fairness, total_health = 0.0, 0.0
            mac_choices = []
            avg_thr, avg_delay, dom_mac = 0.0, 0.0, "CSMA"
            total_inf_time = 0.0
            inf_steps = 0

            obs, _ = sarl_wrapper.reset()
            done = False

            while not done:
                t0 = time.perf_counter()
                action, _ = model.predict(obs, deterministic=deterministic_eval)
                t1 = time.perf_counter()
                total_inf_time += (t1 - t0)
                inf_steps += 1

                obs, reward, terminated, truncated, info = sarl_wrapper.step(action)
                done = terminated or truncated
                mac_choices.append(int(action))

                raw = info.get("raw_infos", {})
                if raw:
                    agg = aggregate_cluster_infos(raw)
                    total_thr += agg["throughput_mbps"]
                    total_delay += agg["delay_ms"]
                    total_drops += agg["drops"]
                    total_collisions += agg["collisions"]
                    total_fairness += agg["jains_fairness"]
                    total_health += agg["leader_health"]
                steps += 1

            avg_thr = total_thr / max(steps, 1)
            avg_delay = total_delay / max(steps, 1)
            avg_fairness = total_fairness / max(steps, 1)
            avg_health = total_health / max(steps, 1)
            dom_mac = "TDMA" if mac_choices.count(0) > mac_choices.count(1) else "CSMA"

            tdma_count = int(mac_choices.count(0))
            csma_count = int(mac_choices.count(1))
            total_count = max(tdma_count + csma_count, 1)
            avg_inf_ms = (total_inf_time / max(inf_steps, 1)) * 1000.0

            results.append({
                'Model': model_name,
                'Offered_Load_pps': pps,
                'Throughput_Mbps': round(avg_thr, 4),
                'Delay_ms': round(avg_delay, 4),
                'Drops': total_drops,
                'Collisions': total_collisions,
                'Fairness': round(avg_fairness, 4),
                'Health': round(avg_health, 4),
                'Dominant_MAC': dom_mac,
                'TDMA_Share': round(tdma_count / total_count, 4),
                'CSMA_Share': round(csma_count / total_count, 4),
                'Avg_Inference_ms': round(avg_inf_ms, 4),
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "csv", "unified_eval_sweep.csv")
    df.to_csv(csv_path, index=False)
    log(f"  Saved: {csv_path}")
    return df


# =====================================================================
# Step 4: Generate Comparison Plots
# =====================================================================
def step4_plots(baseline_df, eval_df, out_dir, log):
    log("\n" + "=" * 60)
    log(" STEP 4: Generating Comparison Plots")
    log("=" * 60)

    img_dir = os.path.join(out_dir, "images")

    if eval_df.empty:
        log("  No evaluation data. Skipping.")
        return

    # --- Throughput comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Throughput_Mbps'],
                'b--', label='TDMA (Baseline)', alpha=0.5, linewidth=1.5)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Throughput_Mbps'],
                'r--', label='CSMA/CA (Baseline)', alpha=0.5, linewidth=1.5)

    rl_models = eval_df['Model'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rl_models), 1)))
    for i, model_name in enumerate(rl_models):
        sub = eval_df[eval_df['Model'] == model_name]
        ax.plot(sub['Offered_Load_pps'], sub['Throughput_Mbps'],
                linestyle='-', marker='o', color=colors[i],
                label=f"SARL: {model_name}", linewidth=2, markersize=4)

    ax.set_xlabel("Offered Load (pps)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("SARL Comparison — Throughput")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "unified_throughput_comparison.png"), dpi=150)
    plt.close()

    # --- Delay comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Delay_s'] * 1000,
                'b--', label='TDMA (Baseline)', alpha=0.5, linewidth=1.5)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Delay_s'] * 1000,
                'r--', label='CSMA/CA (Baseline)', alpha=0.5, linewidth=1.5)

    for i, model_name in enumerate(rl_models):
        sub = eval_df[eval_df['Model'] == model_name]
        ax.plot(sub['Offered_Load_pps'], sub['Delay_ms'],
                linestyle='-', marker='o', color=colors[i],
                label=f"SARL: {model_name}", linewidth=2, markersize=4)

    ax.set_xlabel("Offered Load (pps)")
    ax.set_ylabel("Delay (ms)")
    ax.set_title("SARL Comparison — Delay")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "unified_delay_comparison.png"), dpi=150)
    plt.close()

    # --- MAC Selection per model ---
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = eval_df['Model'].unique()
    tdma_pcts, csma_pcts = [], []
    for m in model_names:
        sub = eval_df[eval_df['Model'] == m]
        tdma_pct = (sub['Dominant_MAC'] == 'TDMA').mean() * 100
        tdma_pcts.append(tdma_pct)
        csma_pcts.append(100 - tdma_pct)

    y_pos = np.arange(len(model_names))
    ax.barh(y_pos, tdma_pcts, color='tab:green', alpha=0.9, label='TDMA')
    ax.barh(y_pos, csma_pcts, left=tdma_pcts, color='tab:purple', alpha=0.75, label='CSMA/CA')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Selection share (%)")
    ax.set_title("MAC Selection Preference per Agent")
    ax.set_xlim(0, 100)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "mac_selection_preference.png"), dpi=150)
    plt.close()

    # --- Summary Bar Chart ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics = ["Throughput_Mbps", "Delay_ms", "Drops", "Collisions"]
    titles = ["Avg Throughput (Mbps)", "Avg Delay (ms)", "Total Drops", "Total Collisions"]

    for ax, metric, title in zip(axes, metrics, titles):
        agg_fn = "mean" if metric in ("Throughput_Mbps", "Delay_ms") else "sum"
        model_vals = eval_df.groupby("Model")[metric].agg(agg_fn)
        bars = ax.bar(model_vals.index, model_vals.values, color=colors[:len(model_vals)])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("SARL Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "summary_bar_chart.png"), dpi=150)
    plt.close()

    log(f"  Saved plots to {img_dir}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="SARL Experiment on MARL Environment")
    parser.add_argument("--algo", type=str, default="all", help="Algorithm to train (for parallel runner)")
    parser.add_argument("--dry-run", action="store_true", help="Sanity check without training")
    parser.add_argument("--timesteps", type=int, default=RLConfig.TOTAL_TIMESTEPS, help="SARL training timesteps")
    parser.add_argument("--seed", type=int, default=params.SEED, help="Random seed")
    parser.add_argument("--sweep-steps", type=int, default=params.SWEEP_STEPS, help="Number of load points")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (use existing checkpoints)")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline MAC simulation")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation phase")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (creates new if None)")
    parser.add_argument("--stochastic-eval", action="store_true", help="Use stochastic evaluation")
    parser.add_argument("--wandb", action="store_true", help="Use WandB")
    
    # Tuning / Profiling Args
    parser.add_argument("--profile", action="store_true", help="Run with cProfile")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (tuning)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (tuning)")
    args = parser.parse_args()

    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        main_execution(args)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.dump_stats(os.path.join(args.out_dir if args.out_dir else "results", f"sarl_profile_{args.algo}.prof"))
    else:
        main_execution(args)


def main_execution(args):
    out_dir = args.out_dir if args.out_dir else make_output_dir()
    for sub in ["images", "csv", "checkpoints", "logs"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    log_file = open(os.path.join(out_dir, "logs", "experiment.log"), "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    print_banner(
        "SARL EXPERIMENT RUNNER",
        f"SARL agents on MARL Environment | {datetime.datetime.now().isoformat()}",
    )

    log("=" * 60)
    log(f" Timestamp: {datetime.datetime.now().isoformat()}")
    log(f" Output: {out_dir}")
    log(f" SARL Timesteps: {args.timesteps}")
    log(f" Seed: {args.seed}")
    log(f" N={params.N}, PHY={params.PHY_RATE_BPS/1e6:.0f}Mbps, Fading={params.FADING_MODEL}")
    log("=" * 60)

    pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, args.sweep_steps).astype(int)

    if args.dry_run:
        log("\n  [DRY RUN] Verifying imports and environment setup...")
        from envs.marl_mac_env import MARLMacEnv
        from envs.marl_sarl_wrapper import MARLtoSARLWrapper

        env1 = MARLtoSARLWrapper(seed=args.seed)
        obs1, _ = env1.reset()
        log(f"    MARLtoSARLWrapper: obs scalars shape = {obs1['scalars'].shape}")
        log(f"    Action space: {env1.action_space}")

        log("  [DRY RUN] All imports and environments OK.")
        log_file.close()
        return

    # Step 1: Baseline
    if not args.skip_baselines:
        baseline_df = step1_baselines(pps_list, args.seed, log)
        baseline_df.to_csv(os.path.join(out_dir, "csv", "baseline_results.csv"), index=False)
    else:
        baseline_path = os.path.join(out_dir, "csv", "baseline_results.csv")
        if os.path.exists(baseline_path):
            baseline_df = pd.read_csv(baseline_path)
        else:
            baseline_df = pd.DataFrame()

    # Step 2: Train
    if args.dry_run:
        sarl_ts = 100
    else:
        sarl_ts = args.timesteps

    cp_dir = step2_train(
        out_dir, sarl_ts, log,
        force_retrain=args.force_retrain,
        use_checkpoints_only=args.skip_training,
        lr=args.lr,
        batch_size=args.batch_size
    )

    # Step 3: Evaluate
    if not args.skip_eval:
        eval_df = step3_evaluate(
            pps_list, cp_dir, out_dir, args.seed, log,
            deterministic_eval=(not args.stochastic_eval),
        )
    else:
        eval_df = pd.DataFrame()

    # Step 4: Plots
    if not args.skip_plots and not baseline_df.empty and not eval_df.empty:
        step4_plots(baseline_df, eval_df, out_dir, log)

    # Save experiment metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sarl_timesteps": sarl_ts,
        "seed": args.seed,
        "n_nodes": params.N,
        "phy_rate_bps": params.PHY_RATE_BPS,
        "fading_model": params.FADING_MODEL,
        "mobility_model": params.MOBILITY_MODEL,
        "models_evaluated": list(eval_df['Model'].unique()) if not eval_df.empty else [],
        "load_points": len(pps_list),
    }
    with open(os.path.join(out_dir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log("\n" + "=" * 60)
    log(" EXPERIMENT COMPLETE")
    log(f" Results: {out_dir}")
    log("=" * 60)
    log_file.close()


if __name__ == "__main__":
    main()
