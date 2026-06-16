"""
run_inter_intra_analysis.py -- Inter/Intra Cluster Delay & Throughput Analysis
================================================================================
Loads pretrained SARL checkpoints and TDMA/CSMA baselines, evaluates across
20 load points (50-1000 pps), and records inter-cluster vs intra-cluster
delay and throughput separately for each model.

Load Classification:
    Low:    50 - 350 pps
    Medium: 351 - 700 pps
    High:   701 - 1000 pps

Usage:
    python experiments/run_inter_intra_analysis.py
    python experiments/run_inter_intra_analysis.py --checkpoint-dir results/sarl_comparison_parallel_20260613_132212/checkpoints
"""

import os
import sys
import json
import time
import argparse
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

from tqdm import tqdm

# -- Project path setup --
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs import config as params
from configs.cluster_config import ClusterConfig as CC
from configs.sarl_config import RLConfig
from envs.burst_scheduler import VALID_RHO_LEVELS, encode_action, decode_action

# =====================================================================
# Constants
# =====================================================================
LOAD_CLASSIFICATION = {
    "Low":    (50, 350),
    "Medium": (351, 700),
    "High":   (701, 1000),
}

MODEL_STYLES = {
    "MCA-D3QN": {"color": "#e63946", "marker": "D", "ls": "-"},
    "MCA-PPO":  {"color": "#457b9d", "marker": "s", "ls": "-"},
    "DQN":      {"color": "#2a9d8f", "marker": "^", "ls": "--"},
    "PPO":      {"color": "#e9c46a", "marker": "v", "ls": "--"},
    "A2C":      {"color": "#f4a261", "marker": "o", "ls": "--"},
    "Tabular":  {"color": "#264653", "marker": "P", "ls": ":"},
}
BASELINE_STYLES = {
    "TDMA": {"color": "#adb5bd", "ls": "--", "lw": 1.5, "marker": "x"},
    "CSMA": {"color": "#6c757d", "ls": "-.", "lw": 1.5, "marker": "+"},
}


def _get_style(model_name):
    if model_name in MODEL_STYLES:
        return MODEL_STYLES[model_name]
    if model_name in BASELINE_STYLES:
        return BASELINE_STYLES[model_name]
    return {"color": "#888888", "marker": "x", "ls": "-"}


def classify_load(pps):
    """Classify a load value into Low/Medium/High."""
    for regime, (lo, hi) in LOAD_CLASSIFICATION.items():
        if lo <= pps <= hi:
            return regime
    if pps < 50:
        return "Low"
    return "High"


# =====================================================================
# Helpers
# =====================================================================
def normalize_weights(weights_dict, keys):
    total = sum(weights_dict.get(k, 0.0) for k in keys)
    if total > 0:
        return {k: weights_dict.get(k, 0.0) / total for k in keys}
    n = len(keys)
    return {k: 1.0 / n for k in keys}


def apply_best_weights(algo_params):
    """Apply Optuna-tuned best weights to global configs."""
    rw = normalize_weights(algo_params,
                           ["w_throughput", "w_delay", "w_failures", "w_jitter"])
    RLConfig.REWARD_W_THROUGHPUT = rw["w_throughput"]
    RLConfig.REWARD_W_DELAY     = rw["w_delay"]
    RLConfig.REWARD_W_FAILURES  = rw["w_failures"]
    RLConfig.REWARD_W_JITTER    = rw["w_jitter"]

    cw = normalize_weights(algo_params,
                           ["c_w_dist", "c_w_sinr", "c_w_mob", "c_w_load"])
    CC.W_DIST = cw["c_w_dist"]
    CC.W_SINR = cw["c_w_sinr"]
    CC.W_MOB  = cw["c_w_mob"]
    CC.W_LOAD = cw["c_w_load"]

    ca = normalize_weights(algo_params,
                           ["c_a_energy", "c_a_degree", "c_a_mobstab",
                            "c_a_queue", "c_a_risk"])
    CC.A_ENERGY  = ca["c_a_energy"]
    CC.A_DEGREE  = ca["c_a_degree"]
    CC.A_MOBSTAB = ca["c_a_mobstab"]
    CC.A_QUEUE   = ca["c_a_queue"]
    CC.A_RISK    = ca["c_a_risk"]


def aggregate_inter_intra(raw_infos):
    """
    Aggregate per-agent infos into inter/intra cluster-level metrics.

    Returns dict with keys:
        intra_throughput_mbps, inter_throughput_mbps,
        intra_delay_ms, inter_delay_ms,
        drops, collisions
    """
    if not raw_infos:
        return {
            "intra_throughput_mbps": 0.0, "inter_throughput_mbps": 0.0,
            "intra_delay_ms": 0.0, "inter_delay_ms": 0.0,
            "drops": 0.0, "collisions": 0.0,
        }
    alive_infos = [info for info in raw_infos.values() if info.get("alive", False)]
    if not alive_infos:
        return {
            "intra_throughput_mbps": 0.0, "inter_throughput_mbps": 0.0,
            "intra_delay_ms": 0.0, "inter_delay_ms": 0.0,
            "drops": 0.0, "collisions": 0.0,
        }
    return {
        "intra_throughput_mbps": float(sum(i.get("throughput_mbps", 0.0) for i in alive_infos)),
        "inter_throughput_mbps": float(sum(i.get("inter_throughput_mbps", 0.0) for i in alive_infos)),
        "intra_delay_ms": float(np.mean([i.get("delay_ms", 0.0) for i in alive_infos])),
        "inter_delay_ms": float(np.mean([i.get("inter_delay_ms", 0.0) for i in alive_infos])),
        "drops": float(sum(i.get("drops", 0.0) for i in alive_infos)),
        "collisions": float(sum(i.get("collisions", 0.0) for i in alive_infos)),
    }


# =====================================================================
# Model Loading
# =====================================================================
def load_all_models(cp_dir, log):
    """Load all trained model checkpoints."""
    from stable_baselines3 import DQN, PPO, A2C

    models = {}

    def _try_load_sb3(display_name, cls, filename, json_key):
        path = os.path.join(cp_dir, filename)
        if os.path.exists(path + ".zip"):
            models[display_name] = (cls.load(path), json_key)
            log(f"  [OK] Loaded: {display_name}")
        else:
            log(f"  [X]  Not found: {path}.zip")

    _try_load_sb3("MCA-D3QN", DQN, "unified_mca_d3qn_model", "mca_d3qn")
    _try_load_sb3("MCA-PPO",  PPO, "unified_mca_ppo_model",  "mca_ppo")
    _try_load_sb3("DQN",      DQN, "unified_dqn_model",      "dqn")
    _try_load_sb3("PPO",      PPO, "unified_ppo_model",       "ppo")
    _try_load_sb3("A2C",      A2C, "unified_a2c_model",       "a2c")

    # Tabular
    tab_path = os.path.join(cp_dir, "unified_tabular_model")
    if os.path.exists(tab_path):
        from algorithms.rl.tabular_qlearning import TabularQLearning
        tab = TabularQLearning()
        tab.load(tab_path)
        tab.epsilon = 0.0
        models["Tabular"] = (tab, None)
        log(f"  [OK] Loaded: Tabular")
    elif os.path.exists(tab_path + ".pkl"):
        from algorithms.rl.tabular_qlearning import TabularQLearning
        tab = TabularQLearning()
        tab.load(tab_path + ".pkl")
        tab.epsilon = 0.0
        models["Tabular"] = (tab, None)
        log(f"  [OK] Loaded: Tabular (.pkl)")
    else:
        log(f"  [X]  Not found: {tab_path}")

    return models


# =====================================================================
# Baseline Evaluation (TDMA / CSMA with inter/intra extraction)
# =====================================================================
def evaluate_baselines(pps_list, seed, log):
    """Evaluate TDMA and CSMA baselines, extracting inter/intra metrics."""
    from envs.marl_mac_env import MARLMacEnv

    log("\n" + "=" * 70)
    log("  BASELINE EVALUATION (TDMA / CSMA)")
    log("=" * 70)

    results = []
    protocol_map = {
        encode_action(0, len(VALID_RHO_LEVELS) // 2): "TDMA",
        encode_action(1, len(VALID_RHO_LEVELS) // 2): "CSMA",
    }

    for idx, pps in enumerate(tqdm(pps_list, desc="Baseline Sweep", unit="load")):
        params.SWEEP_MAX_PPS = int(pps)
        setattr(params, "OFFERED_PPS", int(pps))

        for fixed_action, proto_name in protocol_map.items():
            env = MARLMacEnv(seed=seed + idx + fixed_action * 10000)
            obs, _ = env.reset(seed=seed + idx + fixed_action * 10000)

            total_intra_thr, total_inter_thr = 0.0, 0.0
            total_intra_del, total_inter_del = 0.0, 0.0
            total_drops, total_col = 0.0, 0.0
            steps = 0

            while env.agents:
                actions = {a: fixed_action for a in env.possible_agents}
                obs, rewards, terms, truncs, infos = env.step(actions)

                agg = aggregate_inter_intra(infos)
                total_intra_thr += agg["intra_throughput_mbps"]
                total_inter_thr += agg["inter_throughput_mbps"]
                total_intra_del += agg["intra_delay_ms"]
                total_inter_del += agg["inter_delay_ms"]
                total_drops += agg["drops"]
                total_col += agg["collisions"]
                steps += 1

            n_steps = max(steps, 1)
            results.append({
                "Model": proto_name,
                "Offered_Load_pps": int(pps),
                "Load_Regime": classify_load(int(pps)),
                "Intra_Throughput_Mbps": round(total_intra_thr / n_steps, 6),
                "Inter_Throughput_Mbps": round(total_inter_thr / n_steps, 6),
                "Intra_Delay_ms": round(total_intra_del / n_steps, 4),
                "Inter_Delay_ms": round(total_inter_del / n_steps, 4),
                "Total_Throughput_Mbps": round((total_intra_thr + total_inter_thr) / n_steps, 6),
                "Total_Delay_ms": round((total_intra_del + total_inter_del) / n_steps, 4),
                "Drops": total_drops,
                "Collisions": total_col,
            })

    log(f"  Baseline evaluation complete: {len(results)} records")
    return pd.DataFrame(results)


# =====================================================================
# RL Model Evaluation (with inter/intra extraction)
# =====================================================================
def evaluate_rl_models(models, best_params, pps_list, seed, log):
    """
    Evaluate each RL model across the traffic sweep,
    recording inter/intra cluster metrics separately.
    """
    from envs.marl_sarl_wrapper import MARLtoSARLWrapper

    results = []

    for model_name, (model, json_key) in models.items():
        # Apply per-algorithm best weights
        if json_key and json_key in best_params:
            apply_best_weights(best_params[json_key])
            log(f"\n  [{model_name}] Applied Optuna best weights")
        else:
            log(f"\n  [{model_name}] Using default config weights")

        sarl_wrapper = MARLtoSARLWrapper()

        for pps in tqdm(pps_list, desc=f"Eval {model_name}", unit="load"):
            params.SWEEP_MAX_PPS = pps
            setattr(params, "OFFERED_PPS", int(pps))

            total_intra_thr, total_inter_thr = 0.0, 0.0
            total_intra_del, total_inter_del = 0.0, 0.0
            total_drops, total_col = 0.0, 0.0
            steps = 0

            obs, _ = sarl_wrapper.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = sarl_wrapper.step(action)
                done = terminated or truncated

                raw = info.get("raw_infos", {})
                if raw:
                    agg = aggregate_inter_intra(raw)
                    total_intra_thr += agg["intra_throughput_mbps"]
                    total_inter_thr += agg["inter_throughput_mbps"]
                    total_intra_del += agg["intra_delay_ms"]
                    total_inter_del += agg["inter_delay_ms"]
                    total_drops += agg["drops"]
                    total_col += agg["collisions"]
                steps += 1

            n_steps = max(steps, 1)
            results.append({
                "Model": model_name,
                "Offered_Load_pps": int(pps),
                "Load_Regime": classify_load(int(pps)),
                "Intra_Throughput_Mbps": round(total_intra_thr / n_steps, 6),
                "Inter_Throughput_Mbps": round(total_inter_thr / n_steps, 6),
                "Intra_Delay_ms": round(total_intra_del / n_steps, 4),
                "Inter_Delay_ms": round(total_inter_del / n_steps, 4),
                "Total_Throughput_Mbps": round((total_intra_thr + total_inter_thr) / n_steps, 6),
                "Total_Delay_ms": round((total_intra_del + total_inter_del) / n_steps, 4),
                "Drops": total_drops,
                "Collisions": total_col,
            })

    return pd.DataFrame(results)


# =====================================================================
# CSV Generation
# =====================================================================
def generate_csvs(df, out_dir, log):
    """Generate the 3 output CSVs."""
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # 1. Detailed CSV
    detail_path = os.path.join(csv_dir, "inter_intra_detailed.csv")
    df.to_csv(detail_path, index=False)
    log(f"  [OK] {detail_path} ({len(df)} rows)")

    # 2. By-regime aggregation
    regime_df = df.groupby(["Model", "Load_Regime"]).agg({
        "Intra_Throughput_Mbps": "mean",
        "Inter_Throughput_Mbps": "mean",
        "Intra_Delay_ms": "mean",
        "Inter_Delay_ms": "mean",
        "Total_Throughput_Mbps": "mean",
        "Total_Delay_ms": "mean",
        "Drops": "sum",
        "Collisions": "sum",
    }).round(4).reset_index()

    # Ensure regime ordering
    regime_order = pd.CategoricalDtype(["Low", "Medium", "High"], ordered=True)
    regime_df["Load_Regime"] = regime_df["Load_Regime"].astype(regime_order)
    regime_df = regime_df.sort_values(["Model", "Load_Regime"]).reset_index(drop=True)

    regime_path = os.path.join(csv_dir, "inter_intra_by_regime.csv")
    regime_df.to_csv(regime_path, index=False)
    log(f"  [OK] {regime_path} ({len(regime_df)} rows)")

    # 3. Summary table
    summary_df = df.groupby("Model").agg({
        "Intra_Throughput_Mbps": "mean",
        "Inter_Throughput_Mbps": "mean",
        "Intra_Delay_ms": "mean",
        "Inter_Delay_ms": "mean",
        "Total_Throughput_Mbps": "mean",
        "Total_Delay_ms": "mean",
        "Drops": "sum",
        "Collisions": "sum",
    }).round(4).reset_index()
    summary_df.columns = [
        "Model", "Avg Intra Thr (Mbps)", "Avg Inter Thr (Mbps)",
        "Avg Intra Delay (ms)", "Avg Inter Delay (ms)",
        "Avg Total Thr (Mbps)", "Avg Total Delay (ms)",
        "Total Drops", "Total Collisions",
    ]
    summary_path = os.path.join(csv_dir, "summary_table.csv")
    summary_df.to_csv(summary_path, index=False)
    log(f"  [OK] {summary_path} ({len(summary_df)} rows)")

    return regime_df, summary_df


# =====================================================================
# Plot Generation (10 publication-quality plots)
# =====================================================================
def generate_all_plots(df, out_dir, log):
    """Generate 10 comprehensive inter/intra comparison plots."""
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    if df.empty:
        log("  No data -- skipping plot generation.")
        return

    all_models = df["Model"].unique()
    rl_models = [m for m in all_models if m not in ("TDMA", "CSMA")]
    baseline_models = [m for m in all_models if m in ("TDMA", "CSMA")]

    # ---- Plot 1: Intra-Cluster Throughput vs Load ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for mn in baseline_models:
        s = BASELINE_STYLES.get(mn, _get_style(mn))
        sub = df[df["Model"] == mn].sort_values("Offered_Load_pps")
        ax.plot(sub["Offered_Load_pps"], sub["Intra_Throughput_Mbps"],
                color=s["color"], ls=s["ls"], lw=s.get("lw", 1.5),
                marker=s.get("marker", "x"), label=f"{mn} (Baseline)",
                alpha=0.7, markersize=5)
    for mn in rl_models:
        s = _get_style(mn)
        sub = df[df["Model"] == mn].sort_values("Offered_Load_pps")
        ax.plot(sub["Offered_Load_pps"], sub["Intra_Throughput_Mbps"],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=f"SARL: {mn}", linewidth=2, markersize=5)
    _add_load_regime_bands(ax)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_ylabel("Intra-Cluster Throughput (Mbps)", fontsize=12)
    ax.set_title("Intra-Cluster Throughput vs Offered Load", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "01_intra_throughput_vs_load.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log("    [OK] 01_intra_throughput_vs_load.png")

    # ---- Plot 2: Inter-Cluster Throughput vs Load ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for mn in baseline_models:
        s = BASELINE_STYLES.get(mn, _get_style(mn))
        sub = df[df["Model"] == mn].sort_values("Offered_Load_pps")
        ax.plot(sub["Offered_Load_pps"], sub["Inter_Throughput_Mbps"],
                color=s["color"], ls=s["ls"], lw=s.get("lw", 1.5),
                marker=s.get("marker", "x"), label=f"{mn} (Baseline)",
                alpha=0.7, markersize=5)
    for mn in rl_models:
        s = _get_style(mn)
        sub = df[df["Model"] == mn].sort_values("Offered_Load_pps")
        ax.plot(sub["Offered_Load_pps"], sub["Inter_Throughput_Mbps"],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=f"SARL: {mn}", linewidth=2, markersize=5)
    _add_load_regime_bands(ax)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_ylabel("Inter-Cluster Throughput (Mbps)", fontsize=12)
    ax.set_title("Inter-Cluster Throughput vs Offered Load", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "02_inter_throughput_vs_load.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log("    [OK] 02_inter_throughput_vs_load.png")

    # ---- Plot 3: Intra-Cluster Delay vs Load ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for mn in baseline_models:
        s = BASELINE_STYLES.get(mn, _get_style(mn))
        sub = df[df["Model"] == mn].sort_values("Offered_Load_pps")
        ax.plot(sub["Offered_Load_pps"], sub["Intra_Delay_ms"],
                color=s["color"], ls=s["ls"], lw=s.get("lw", 1.5),
                marker=s.get("marker", "x"), label=f"{mn} (Baseline)",
                alpha=0.7, markersize=5)
    for mn in rl_models:
        s = _get_style(mn)
        sub = df[df["Model"] == mn].sort_values("Offered_Load_pps")
        ax.plot(sub["Offered_Load_pps"], sub["Intra_Delay_ms"],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=f"SARL: {mn}", linewidth=2, markersize=5)
    _add_load_regime_bands(ax)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_ylabel("Intra-Cluster Delay (ms)", fontsize=12)
    ax.set_title("Intra-Cluster Delay vs Offered Load", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "03_intra_delay_vs_load.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log("    [OK] 03_intra_delay_vs_load.png")

    # ---- Plot 4: Inter-Cluster Delay vs Load ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for mn in baseline_models:
        s = BASELINE_STYLES.get(mn, _get_style(mn))
        sub = df[df["Model"] == mn].sort_values("Offered_Load_pps")
        ax.plot(sub["Offered_Load_pps"], sub["Inter_Delay_ms"],
                color=s["color"], ls=s["ls"], lw=s.get("lw", 1.5),
                marker=s.get("marker", "x"), label=f"{mn} (Baseline)",
                alpha=0.7, markersize=5)
    for mn in rl_models:
        s = _get_style(mn)
        sub = df[df["Model"] == mn].sort_values("Offered_Load_pps")
        ax.plot(sub["Offered_Load_pps"], sub["Inter_Delay_ms"],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=f"SARL: {mn}", linewidth=2, markersize=5)
    _add_load_regime_bands(ax)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_ylabel("Inter-Cluster Delay (ms)", fontsize=12)
    ax.set_title("Inter-Cluster Delay vs Offered Load", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "04_inter_delay_vs_load.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log("    [OK] 04_inter_delay_vs_load.png")

    # ---- Plot 5: Grouped Bar - Intra vs Inter Delay per Model by Regime ----
    _plot_grouped_bar_by_regime(
        df, all_models, "Intra_Delay_ms", "Inter_Delay_ms",
        "Intra Delay (ms)", "Inter Delay (ms)",
        "Intra vs Inter Cluster Delay by Load Regime",
        os.path.join(img_dir, "05_delay_grouped_bar.png"), log,
    )
    log("    [OK] 05_delay_grouped_bar.png")

    # ---- Plot 6: Grouped Bar - Intra vs Inter Throughput per Model by Regime ----
    _plot_grouped_bar_by_regime(
        df, all_models, "Intra_Throughput_Mbps", "Inter_Throughput_Mbps",
        "Intra Thr (Mbps)", "Inter Thr (Mbps)",
        "Intra vs Inter Cluster Throughput by Load Regime",
        os.path.join(img_dir, "06_throughput_grouped_bar.png"), log,
    )
    log("    [OK] 06_throughput_grouped_bar.png")

    # ---- Plot 7: Stacked Area - Intra + Inter Throughput Contribution ----
    fig, axes = plt.subplots(1, len(all_models), figsize=(4 * len(all_models), 5), sharey=True)
    if len(all_models) == 1:
        axes = [axes]
    for ax, mn in zip(axes, all_models):
        sub = df[df["Model"] == mn].sort_values("Offered_Load_pps")
        ax.fill_between(sub["Offered_Load_pps"], 0, sub["Intra_Throughput_Mbps"],
                        alpha=0.6, color="#2a9d8f", label="Intra")
        ax.fill_between(sub["Offered_Load_pps"],
                        sub["Intra_Throughput_Mbps"],
                        sub["Intra_Throughput_Mbps"] + sub["Inter_Throughput_Mbps"],
                        alpha=0.6, color="#e76f51", label="Inter")
        ax.set_title(mn, fontsize=10, fontweight="bold")
        ax.set_xlabel("Load (pps)", fontsize=9)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Throughput (Mbps)", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
    plt.suptitle("Throughput Composition: Intra + Inter Cluster",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(img_dir, "07_throughput_stacked_area.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log("    [OK] 07_throughput_stacked_area.png")

    # ---- Plot 8: Heatmap - Delay Breakdown by Model x Regime ----
    _plot_delay_heatmap(df, all_models, img_dir, log)
    log("    [OK] 08_delay_heatmap.png")

    # ---- Plot 9: Radar Chart ----
    _plot_radar_chart(df, all_models, img_dir, log)
    log("    [OK] 09_radar_chart.png")

    # ---- Plot 10: Summary Table Image ----
    _plot_summary_table(df, all_models, img_dir, log)
    log("    [OK] 10_summary_table.png")

    log(f"\n  All plots saved to: {img_dir}")


def _add_load_regime_bands(ax):
    """Add colored vertical bands for Low/Medium/High load regimes."""
    ax.axvspan(50, 350, alpha=0.06, color="green", label="_Low")
    ax.axvspan(351, 700, alpha=0.06, color="orange", label="_Medium")
    ax.axvspan(701, 1000, alpha=0.06, color="red", label="_High")
    # Text labels at the top
    ylim = ax.get_ylim()
    y_top = ylim[1] * 0.97
    ax.text(200, y_top, "Low", ha="center", fontsize=8, color="green", alpha=0.7, fontweight="bold")
    ax.text(525, y_top, "Medium", ha="center", fontsize=8, color="orange", alpha=0.7, fontweight="bold")
    ax.text(850, y_top, "High", ha="center", fontsize=8, color="red", alpha=0.7, fontweight="bold")


def _plot_grouped_bar_by_regime(df, all_models, col_intra, col_inter,
                                 label_intra, label_inter, title, save_path, log):
    """Create a grouped bar chart showing intra vs inter metrics per model, grouped by regime."""
    regimes = ["Low", "Medium", "High"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, regime in zip(axes, regimes):
        sub = df[df["Load_Regime"] == regime]
        if sub.empty:
            ax.set_title(f"{regime} Load", fontsize=12, fontweight="bold")
            continue

        model_means_intra = sub.groupby("Model")[col_intra].mean()
        model_means_inter = sub.groupby("Model")[col_inter].mean()

        # Ensure consistent model ordering
        present_models = [m for m in all_models if m in model_means_intra.index]
        x = np.arange(len(present_models))
        width = 0.35

        intra_vals = [model_means_intra.get(m, 0.0) for m in present_models]
        inter_vals = [model_means_inter.get(m, 0.0) for m in present_models]

        bars1 = ax.bar(x - width / 2, intra_vals, width, label=label_intra,
                       color="#2a9d8f", alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width / 2, inter_vals, width, label=label_inter,
                       color="#e76f51", alpha=0.85, edgecolor="white")

        # Value labels
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01 * max(intra_vals + inter_vals + [1]),
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01 * max(intra_vals + inter_vals + [1]),
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(present_models, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{regime} Load", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if ax == axes[0]:
            ax.legend(fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_delay_heatmap(df, all_models, img_dir, log):
    """Create a 2-panel heatmap: intra delay and inter delay by model × regime."""
    regimes = ["Low", "Medium", "High"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, title_suffix in [
        (ax1, "Intra_Delay_ms", "Intra-Cluster"),
        (ax2, "Inter_Delay_ms", "Inter-Cluster"),
    ]:
        pivot = df.pivot_table(values=metric, index="Model", columns="Load_Regime", aggfunc="mean")
        # Reorder columns and rows
        pivot = pivot.reindex(columns=[r for r in regimes if r in pivot.columns])
        pivot = pivot.reindex([m for m in all_models if m in pivot.index])

        if pivot.empty:
            continue

        cmap = "YlOrRd"
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                text_color = "white" if val > pivot.values.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

        ax.set_title(f"{title_suffix} Delay (ms)", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Delay (ms)", shrink=0.8)

    plt.suptitle("Delay Breakdown: Model × Load Regime", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(img_dir, "08_delay_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_radar_chart(df, all_models, img_dir, log):
    """Multi-metric radar chart comparing all models."""
    # Metrics: High intra thr, High inter thr, Low intra delay, Low inter delay, Low drops
    labels = ["Intra Thr", "Inter Thr", "Low Intra Delay", "Low Inter Delay", "Low Drops"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for mn in all_models:
        sub = df[df["Model"] == mn]
        vals = []

        # Intra throughput (higher is better)
        v = sub["Intra_Throughput_Mbps"].mean()
        col_max = df["Intra_Throughput_Mbps"].max()
        vals.append(v / col_max if col_max > 0 else 0)

        # Inter throughput (higher is better)
        v = sub["Inter_Throughput_Mbps"].mean()
        col_max = df["Inter_Throughput_Mbps"].max()
        vals.append(v / col_max if col_max > 0 else 0)

        # Intra delay (lower is better → invert)
        v = sub["Intra_Delay_ms"].mean()
        col_max = df["Intra_Delay_ms"].max()
        vals.append(1.0 - (v / col_max) if col_max > 0 else 1.0)

        # Inter delay (lower is better → invert)
        v = sub["Inter_Delay_ms"].mean()
        col_max = df["Inter_Delay_ms"].max()
        vals.append(1.0 - (v / col_max) if col_max > 0 else 1.0)

        # Drops (lower is better → invert)
        v = sub["Drops"].mean()
        col_max = df["Drops"].max()
        vals.append(1.0 - (v / col_max) if col_max > 0 else 1.0)

        vals += vals[:1]
        s = _get_style(mn)
        ax.plot(angles, vals, color=s["color"], linewidth=2, label=mn)
        ax.fill(angles, vals, color=s["color"], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Inter/Intra Performance Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "09_radar_chart.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_summary_table(df, all_models, img_dir, log):
    """Render a summary table as an image."""
    summary_rows = []
    for mn in all_models:
        sub = df[df["Model"] == mn]
        summary_rows.append({
            "Model": mn,
            "Intra Thr (Mbps)": round(sub["Intra_Throughput_Mbps"].mean(), 4),
            "Inter Thr (Mbps)": round(sub["Inter_Throughput_Mbps"].mean(), 4),
            "Intra Delay (ms)": round(sub["Intra_Delay_ms"].mean(), 2),
            "Inter Delay (ms)": round(sub["Inter_Delay_ms"].mean(), 2),
            "Total Drops": int(sub["Drops"].sum()),
        })
    summary_df = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(16, 2 + 0.45 * len(summary_rows)))
    ax.axis("off")
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc="center", loc="center",
                     colColours=["#f0f0f0"] * len(summary_df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # Highlight best values
    for col_idx, col in enumerate(summary_df.columns):
        if col == "Model":
            continue
        vals = summary_df[col].values
        if col in ("Intra Delay (ms)", "Inter Delay (ms)", "Total Drops"):
            best_idx = np.argmin(vals)
        else:
            best_idx = np.argmax(vals)
        table[best_idx + 1, col_idx].set_facecolor("#d4edda")

    ax.set_title("Inter/Intra Cluster Analysis Summary", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "10_summary_table.png"), dpi=200, bbox_inches="tight")
    plt.close()


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Inter/Intra Cluster Delay & Throughput Analysis")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to checkpoints directory (auto-detected if None)")
    parser.add_argument("--best-weights", type=str, default=None,
                        help="Path to best_weights_and_params JSON file")
    parser.add_argument("--seed", type=int, default=params.SEED, help="Random seed")
    parser.add_argument("--sweep-steps", type=int, default=params.SWEEP_STEPS,
                        help="Number of load points (default: 20)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline evaluation")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    # Auto-detect paths
    results_root = os.path.join(project_root, "results")

    if args.checkpoint_dir is None:
        candidates = sorted(
            [d for d in os.listdir(results_root)
             if d.startswith("sarl_comparison_parallel_") and
             os.path.isdir(os.path.join(results_root, d))],
            reverse=True
        )
        if candidates:
            latest = candidates[0]
            args.checkpoint_dir = os.path.join(results_root, latest, "checkpoints")
        else:
            print("ERROR: No sarl_comparison_parallel_* directory found.")
            sys.exit(1)

    if args.best_weights is None:
        candidates_json = [
            os.path.join(project_root, "best_weights_and_params (1).json"),
            os.path.join(project_root, "best_weights_and_params.json"),
        ]
        for cj in candidates_json:
            if os.path.exists(cj):
                args.best_weights = cj
                break
        if args.best_weights is None:
            print("ERROR: No best_weights_and_params JSON found.")
            sys.exit(1)

    # Create output directory (separate from other experiments)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_root, f"inter_intra_analysis_{ts}")
    for sub in ["images", "csv", "logs"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # Logger
    log_file = open(os.path.join(out_dir, "logs", "analysis.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 70)
    log("  INTER/INTRA CLUSTER DELAY & THROUGHPUT ANALYSIS")
    log("=" * 70)
    log(f"  Timestamp:      {datetime.datetime.now().isoformat()}")
    log(f"  Checkpoints:    {args.checkpoint_dir}")
    log(f"  Best Weights:   {args.best_weights}")
    log(f"  Output Dir:     {out_dir}")
    log(f"  Seed:           {args.seed}")
    log(f"  Sweep Steps:    {args.sweep_steps}")
    log(f"  N={params.N}, PHY={params.PHY_RATE_BPS/1e6:.0f}Mbps, Fading={params.FADING_MODEL}")
    log(f"  Load Regimes:   Low=50-350, Medium=351-700, High=701-1000 pps")
    log("=" * 70)

    # Load best weights
    with open(args.best_weights, "r") as f:
        best_params = json.load(f)
    log(f"\n  Loaded best weights for algorithms: {list(best_params.keys())}")

    # Traffic sweep: 20 load points from 50 to 1000
    pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS,
                           args.sweep_steps).astype(int)
    log(f"  Load points ({len(pps_list)}): {list(pps_list)}")
    log(f"  Load classification: " +
        ", ".join(f"{p}pps={classify_load(p)}" for p in pps_list))

    # ---- Phase 1: Baseline Evaluation ----
    if not args.skip_baselines:
        baseline_df = evaluate_baselines(pps_list, args.seed, log)
    else:
        baseline_df = pd.DataFrame()
        log("\n  Skipping baseline evaluation")

    # ---- Phase 2: Load Models ----
    log("\n" + "=" * 70)
    log("  LOADING MODEL CHECKPOINTS")
    log("=" * 70)
    models = load_all_models(args.checkpoint_dir, log)
    log(f"  Loaded {len(models)} models: {list(models.keys())}")

    if not models:
        log("  ERROR: No models could be loaded. Aborting.")
        log_file.close()
        sys.exit(1)

    # ---- Phase 3: RL Model Evaluation ----
    log("\n" + "=" * 70)
    log("  RL MODEL EVALUATION SWEEP")
    log("=" * 70)
    rl_df = evaluate_rl_models(models, best_params, pps_list, args.seed, log)
    log(f"\n  RL evaluation complete: {len(rl_df)} records")

    # ---- Combine results ----
    all_df = pd.concat([baseline_df, rl_df], ignore_index=True)
    log(f"\n  Total records: {len(all_df)}")

    # ---- Phase 4: Generate CSVs ----
    log("\n" + "=" * 70)
    log("  GENERATING CSVs")
    log("=" * 70)
    regime_df, summary_df = generate_csvs(all_df, out_dir, log)

    # Print summary to log
    log("\n  === SUMMARY TABLE ===")
    log(summary_df.to_string(index=False))

    # ---- Phase 5: Generate Plots ----
    if not args.skip_plots:
        log("\n" + "=" * 70)
        log("  GENERATING PLOTS")
        log("=" * 70)
        generate_all_plots(all_df, out_dir, log)

    # Save metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_type": "inter_intra_cluster_analysis",
        "checkpoint_source": args.checkpoint_dir,
        "best_weights_source": args.best_weights,
        "seed": args.seed,
        "n_nodes": params.N,
        "phy_rate_bps": params.PHY_RATE_BPS,
        "fading_model": params.FADING_MODEL,
        "mobility_model": params.MOBILITY_MODEL,
        "sweep_steps": args.sweep_steps,
        "load_points": list(map(int, pps_list)),
        "load_classification": {
            "Low": "50-350 pps",
            "Medium": "351-700 pps",
            "High": "701-1000 pps",
        },
        "models_evaluated": list(all_df["Model"].unique()),
        "total_records": len(all_df),
    }
    with open(os.path.join(out_dir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log("\n" + "=" * 70)
    log("  EXPERIMENT COMPLETE")
    log(f"  Results: {out_dir}")
    log("=" * 70)
    log_file.close()


if __name__ == "__main__":
    main()
