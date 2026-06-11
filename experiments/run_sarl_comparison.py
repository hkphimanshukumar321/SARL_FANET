"""
run_sarl_comparison.py -- Unified SARL Experiment Runner
=========================================================
Trains and evaluates ALL SARL agents on the MARL environment (MARLMacEnv),
then generates comparison plots and CSV outputs.

Models:
  - Tabular Q-Learning  (MARLtoSARLWrapper → Discrete(2))
  - SB3 DQN             (MARLtoSARLWrapper → Discrete(2))
  - SB3 PPO             (SARLCentralEnv → MultiDiscrete)
  - SB3 A2C             (SARLCentralEnv → MultiDiscrete)
  - MCA-D3QN (Ours)     (SARLCentralEnv → MultiDiscrete branching)
  - MCA-PPO  (Ours)     (SARLCentralEnv → MultiDiscrete branching)

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

# =====================================================================
# Constants
# =====================================================================
MAC_NAMES = {0: "TDMA", 1: "CSMA_CA"}

SARL_MODELS = {
    "Tabular":   {"wrapper": "sarl_wrapper",  "type": "tabular"},
    "DQN":       {"wrapper": "sarl_wrapper",  "type": "sb3", "algo": "dqn"},
    "PPO":       {"wrapper": "sarl_central",  "type": "sb3", "algo": "ppo"},
    "A2C":       {"wrapper": "sarl_central",  "type": "sb3", "algo": "a2c"},
    "MCA-D3QN":  {"wrapper": "sarl_central",  "type": "mca_d3qn"},
    "MCA-PPO":   {"wrapper": "sarl_central",  "type": "mca_ppo"},
}


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

            while env.agents:
                actions = {a: fixed_action for a in env.possible_agents}
                obs, rewards, terms, truncs, infos = env.step(actions)
                agg = aggregate_cluster_infos(infos)
                ep_thr.append(agg["throughput_mbps"])
                ep_del.append(agg["delay_ms"])
                ep_drops += agg["drops"]
                ep_col += agg["collisions"]

            n_steps = max(len(ep_thr), 1)
            row[f"{proto_name}_Throughput_Mbps"] = round(sum(ep_thr) / n_steps, 6)
            row[f"{proto_name}_Delay_s"] = round(sum(ep_del) / n_steps / 1000.0, 8)
            row[f"{proto_name}_Drops"] = ep_drops
            row[f"{proto_name}_Collisions"] = ep_col

        results.append(row)
    return pd.DataFrame(results)


# =====================================================================
# Step 2: Train SARL Models
# =====================================================================
def step2_train(timesteps, cp_dir, csv_dir, seed, log):
    from envs.sarl_central_env import SARLCentralEnv
    from envs.marl_sarl_wrapper import MARLtoSARLWrapper
    from utils.device_manager import resolve_device

    log("\n" + "=" * 60)
    log(f" STEP 2: Training SARL Models ({timesteps} timesteps)")
    log("=" * 60)

    device = resolve_device("train")
    trained_models = {}

    # --- Tabular Q-Learning ---
    if getattr(params, "RUN_TABULAR_QLEARNING", True):
        log("  Training: Tabular Q-Learning (MARLtoSARLWrapper)")
        from algorithms.rl.tabular_qlearning import TabularQLearning

        env = MARLtoSARLWrapper(seed=seed)
        agent = TabularQLearning(action_dim=2, num_scalar_features=14, seed=seed)
        obs, _ = env.reset()
        rewards_log = []

        for step in tqdm(range(timesteps), desc="Tabular", unit="step"):
            action, _ = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            agent.learn(obs, int(action), reward, next_obs, bool(terminated or truncated))
            obs = next_obs
            rewards_log.append(reward)
            if terminated or truncated:
                obs, _ = env.reset()

        save_path = os.path.join(cp_dir, "tabular_model.json")
        agent.save(save_path)
        trained_models["Tabular"] = ("tabular", agent)
        log(f"    Saved: {save_path}")

    # --- SB3 DQN (Discrete wrapper) ---
    if getattr(params, "RUN_DQN", True):
        log("  Training: DQN (MARLtoSARLWrapper)")
        from stable_baselines3 import DQN
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(MARLtoSARLWrapper(seed=seed))
        model = DQN("MultiInputPolicy", env, learning_rate=1e-3, buffer_size=100000,
                     batch_size=64, gamma=0.99, device=device, seed=seed, verbose=0)
        model.learn(total_timesteps=timesteps, progress_bar=True)
        save_path = os.path.join(cp_dir, "dqn_model")
        model.save(save_path)
        trained_models["DQN"] = ("sb3", model)
        log(f"    Saved: {save_path}")

    # --- SB3 PPO (SARLCentralEnv) ---
    if getattr(params, "RUN_PPO", True):
        log("  Training: PPO (SARLCentralEnv)")
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(SARLCentralEnv(seed=seed))
        model = PPO("MultiInputPolicy", env, learning_rate=3e-4, n_steps=128,
                     batch_size=64, n_epochs=10, gamma=0.99, device=device, seed=seed, verbose=0)
        model.learn(total_timesteps=timesteps, progress_bar=True)
        save_path = os.path.join(cp_dir, "ppo_model")
        model.save(save_path)
        trained_models["PPO"] = ("sb3", model)
        log(f"    Saved: {save_path}")

    # --- SB3 A2C (SARLCentralEnv) ---
    if getattr(params, "RUN_A2C", True):
        log("  Training: A2C (SARLCentralEnv)")
        from stable_baselines3 import A2C
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(SARLCentralEnv(seed=seed))
        model = A2C("MultiInputPolicy", env, learning_rate=7e-4, n_steps=5,
                     gamma=0.99, device=device, seed=seed, verbose=0)
        model.learn(total_timesteps=timesteps, progress_bar=True)
        save_path = os.path.join(cp_dir, "a2c_model")
        model.save(save_path)
        trained_models["A2C"] = ("sb3", model)
        log(f"    Saved: {save_path}")

    # --- MCA-D3QN (SARLCentralEnv) ---
    if getattr(params, "RUN_CUSTOM_RL", True):
        log("  Training: MCA-D3QN (SARLCentralEnv)")
        from algorithms.rl.custom_mca_d3qn import create_mca_d3qn

        env = SARLCentralEnv(seed=seed)
        model = create_mca_d3qn(env, seed=seed, device=device)
        model.learn(total_timesteps=timesteps)
        save_path = os.path.join(cp_dir, "mca_d3qn_model")
        model.save(save_path)
        trained_models["MCA-D3QN"] = ("mca_d3qn", model)
        log(f"    Saved: {save_path}")

    # --- MCA-PPO (SARLCentralEnv) ---
    if getattr(params, "RUN_MCA_PPO", True):
        log("  Training: MCA-PPO (SARLCentralEnv)")
        from algorithms.rl.custom_mca_ppo import create_mca_ppo

        env = SARLCentralEnv(seed=seed)
        model = create_mca_ppo(env, seed=seed, device=device)
        model.learn(total_timesteps=timesteps)
        save_path = os.path.join(cp_dir, "mca_ppo_model")
        model.save(save_path)
        trained_models["MCA-PPO"] = ("mca_ppo", model)
        log(f"    Saved: {save_path}")

    log(f"  Trained {len(trained_models)} models total.")
    return trained_models


# =====================================================================
# Step 3: Evaluate all SARL Models
# =====================================================================
def step3_evaluate(trained_models, pps_list, seed, log):
    from envs.marl_mac_env import MARLMacEnv
    from envs.sarl_central_env import SARLCentralEnv
    from envs.marl_sarl_wrapper import MARLtoSARLWrapper

    log("\n" + "=" * 60)
    log(" STEP 3: Evaluating SARL Models")
    log("=" * 60)

    all_results = []

    for model_name, (model_type, model) in trained_models.items():
        log(f"  Evaluating: {model_name}")

        for idx, pps in enumerate(tqdm(pps_list, desc=model_name, unit="load")):
            params.SWEEP_MAX_PPS = int(pps)
            setattr(params, "OFFERED_PPS", int(pps))

            # Select appropriate env
            uses_wrapper = model_type in ("tabular", "sb3") and model_name in ("Tabular", "DQN")

            if uses_wrapper:
                env = MARLtoSARLWrapper(seed=seed + idx)
            else:
                env = SARLCentralEnv(seed=seed + idx)

            obs, _ = env.reset(seed=seed + idx)
            ep_thr, ep_del = [], []
            ep_drops, ep_col = 0, 0
            ep_rewards = []

            max_steps = 200
            for step_i in range(max_steps):
                if model_type == "tabular":
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                elif model_type == "sb3":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_rewards.append(reward)

                if uses_wrapper:
                    raw_infos = info.get("raw_infos", {})
                    agg = aggregate_cluster_infos(raw_infos)
                else:
                    agg = {
                        "throughput_mbps": info.get("global_th", 0.0),
                        "delay_ms": 0.0,
                        "drops": info.get("global_drops", 0.0),
                        "collisions": info.get("global_collisions", 0.0),
                    }
                    raw_infos = info.get("raw_infos", {})
                    if raw_infos:
                        alive = [i for i in raw_infos.values() if i.get("alive", False)]
                        if alive:
                            agg["delay_ms"] = float(np.mean([i.get("delay_ms", 0.0) for i in alive]))

                ep_thr.append(agg["throughput_mbps"])
                ep_del.append(agg["delay_ms"])
                ep_drops += agg["drops"]
                ep_col += agg["collisions"]

                if terminated or truncated:
                    break

            n_steps = max(len(ep_thr), 1)
            all_results.append({
                "Model": model_name,
                "Offered_Load_pps": int(pps),
                "Throughput_Mbps": round(sum(ep_thr) / n_steps, 6),
                "Delay_ms": round(sum(ep_del) / n_steps, 4),
                "Drops": int(ep_drops),
                "Collisions": int(ep_col),
                "Avg_Reward": round(sum(ep_rewards) / n_steps, 6),
            })

    return pd.DataFrame(all_results)


# =====================================================================
# Step 4: Generate Comparison Plots
# =====================================================================
def step4_plots(baseline_df, eval_df, out_dir, log):
    log("\n" + "=" * 60)
    log(" STEP 4: Generating Comparison Plots")
    log("=" * 60)

    img_dir = os.path.join(out_dir, "images")
    models = eval_df["Model"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models) + 2))

    # --- Throughput vs Load ---
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(baseline_df["Offered_Load_pps"], baseline_df["TDMA_Throughput_Mbps"],
            "--", color="gray", alpha=0.6, label="TDMA (Fixed)", linewidth=2)
    ax.plot(baseline_df["Offered_Load_pps"], baseline_df["CSMA_Throughput_Mbps"],
            "--", color="lightcoral", alpha=0.6, label="CSMA (Fixed)", linewidth=2)

    for i, model_name in enumerate(models):
        mdf = eval_df[eval_df["Model"] == model_name]
        ax.plot(mdf["Offered_Load_pps"], mdf["Throughput_Mbps"],
                "-o", color=colors[i], label=model_name, linewidth=2, markersize=5)

    ax.set_xlabel("Offered Load (pps)", fontsize=13)
    ax.set_ylabel("Throughput (Mbps)", fontsize=13)
    ax.set_title("SARL Model Comparison: Throughput vs. Traffic Load", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "throughput_comparison.png"), dpi=150)
    plt.close()

    # --- Delay vs Load ---
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(baseline_df["Offered_Load_pps"], baseline_df["TDMA_Delay_s"] * 1000,
            "--", color="gray", alpha=0.6, label="TDMA (Fixed)", linewidth=2)
    ax.plot(baseline_df["Offered_Load_pps"], baseline_df["CSMA_Delay_s"] * 1000,
            "--", color="lightcoral", alpha=0.6, label="CSMA (Fixed)", linewidth=2)

    for i, model_name in enumerate(models):
        mdf = eval_df[eval_df["Model"] == model_name]
        ax.plot(mdf["Offered_Load_pps"], mdf["Delay_ms"],
                "-o", color=colors[i], label=model_name, linewidth=2, markersize=5)

    ax.set_xlabel("Offered Load (pps)", fontsize=13)
    ax.set_ylabel("Delay (ms)", fontsize=13)
    ax.set_title("SARL Model Comparison: Delay vs. Traffic Load", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "delay_comparison.png"), dpi=150)
    plt.close()

    # --- Drops vs Load ---
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(baseline_df["Offered_Load_pps"], baseline_df["TDMA_Drops"],
            "--", color="gray", alpha=0.6, label="TDMA (Fixed)", linewidth=2)
    ax.plot(baseline_df["Offered_Load_pps"], baseline_df["CSMA_Drops"],
            "--", color="lightcoral", alpha=0.6, label="CSMA (Fixed)", linewidth=2)

    for i, model_name in enumerate(models):
        mdf = eval_df[eval_df["Model"] == model_name]
        ax.plot(mdf["Offered_Load_pps"], mdf["Drops"],
                "-o", color=colors[i], label=model_name, linewidth=2, markersize=5)

    ax.set_xlabel("Offered Load (pps)", fontsize=13)
    ax.set_ylabel("Packet Drops", fontsize=13)
    ax.set_title("SARL Model Comparison: Drops vs. Traffic Load", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "drops_comparison.png"), dpi=150)
    plt.close()

    # --- Collisions vs Load ---
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(baseline_df["Offered_Load_pps"], baseline_df["TDMA_Collisions"],
            "--", color="gray", alpha=0.6, label="TDMA (Fixed)", linewidth=2)
    ax.plot(baseline_df["Offered_Load_pps"], baseline_df["CSMA_Collisions"],
            "--", color="lightcoral", alpha=0.6, label="CSMA (Fixed)", linewidth=2)

    for i, model_name in enumerate(models):
        mdf = eval_df[eval_df["Model"] == model_name]
        ax.plot(mdf["Offered_Load_pps"], mdf["Collisions"],
                "-o", color=colors[i], label=model_name, linewidth=2, markersize=5)

    ax.set_xlabel("Offered Load (pps)", fontsize=13)
    ax.set_ylabel("Collisions", fontsize=13)
    ax.set_title("SARL Model Comparison: Collisions vs. Traffic Load", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "collisions_comparison.png"), dpi=150)
    plt.close()

    # --- Summary Bar Chart (avg metrics across all loads) ---
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

    plt.suptitle("SARL Model Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "summary_bar_chart.png"), dpi=150)
    plt.close()

    log(f"  Saved plots to {img_dir}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="SARL Model Comparison on MARL Environment")
    parser.add_argument("--dry-run", action="store_true", help="Sanity check without training")
    parser.add_argument("--timesteps", type=int, default=RLConfig.TOTAL_TIMESTEPS, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=params.SEED, help="Random seed")
    parser.add_argument("--sweep-steps", type=int, default=params.SWEEP_STEPS, help="Number of load points")
    args = parser.parse_args()

    out_dir = make_output_dir()
    log_file = open(os.path.join(out_dir, "logs", "experiment.log"), "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 60)
    log(" SARL Models on MARL Environment — Comparison Experiment")
    log(f" Timestamp: {datetime.datetime.now().isoformat()}")
    log(f" Output: {out_dir}")
    log(f" Timesteps: {args.timesteps}")
    log(f" Seed: {args.seed}")
    log(f" N={params.N}, PHY={params.PHY_RATE_BPS/1e6:.0f}Mbps, Fading={params.FADING_MODEL}")
    log("=" * 60)

    pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, args.sweep_steps).astype(int)

    if args.dry_run:
        log("\n  [DRY RUN] Verifying imports and environment setup...")
        from envs.marl_mac_env import MARLMacEnv
        from envs.sarl_central_env import SARLCentralEnv
        from envs.marl_sarl_wrapper import MARLtoSARLWrapper

        env1 = MARLtoSARLWrapper(seed=args.seed)
        obs1, _ = env1.reset()
        log(f"    MARLtoSARLWrapper: obs scalars shape = {obs1['scalars'].shape}")
        log(f"    Action space: {env1.action_space}")

        env2 = SARLCentralEnv(seed=args.seed)
        obs2, _ = env2.reset()
        log(f"    SARLCentralEnv: obs scalars shape = {obs2['scalars'].shape}")
        log(f"    Action space: {env2.action_space}")

        log("  [DRY RUN] All imports and environments OK. Exiting.")
        log_file.close()
        return

    # Step 1: Baseline
    baseline_df = step1_baselines(pps_list, args.seed, log)
    baseline_df.to_csv(os.path.join(out_dir, "csv", "baseline_results.csv"), index=False)

    # Step 2: Train
    cp_dir = os.path.join(out_dir, "checkpoints")
    csv_dir = os.path.join(out_dir, "csv")
    trained_models = step2_train(args.timesteps, cp_dir, csv_dir, args.seed, log)

    # Step 3: Evaluate
    eval_df = step3_evaluate(trained_models, pps_list, args.seed, log)
    eval_df.to_csv(os.path.join(out_dir, "csv", "sarl_evaluation_results.csv"), index=False)

    # Step 4: Plots
    step4_plots(baseline_df, eval_df, out_dir, log)

    # Save experiment metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "timesteps": args.timesteps,
        "seed": args.seed,
        "n_nodes": params.N,
        "phy_rate_bps": params.PHY_RATE_BPS,
        "fading_model": params.FADING_MODEL,
        "mobility_model": params.MOBILITY_MODEL,
        "models_trained": list(trained_models.keys()),
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
