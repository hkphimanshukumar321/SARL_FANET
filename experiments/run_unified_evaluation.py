"""
run_unified_evaluation.py -- Step 6: Unified Evaluation & Plot Generation
=========================================================================
Loads trained checkpoints, applies per-algorithm Optuna-tuned best weights,
evaluates every model across a traffic sweep, and generates publication-
quality comparison plots in a unified results folder.

Usage:
    python experiments/run_unified_evaluation.py
    python experiments/run_unified_evaluation.py --checkpoint-dir results/sarl_comparison_parallel_20260613_132212/checkpoints
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

# =====================================================================
# Helpers
# =====================================================================
MAC_NAMES = {0: "TDMA", 1: "CSMA_CA"}


def aggregate_cluster_infos(raw_infos):
    """Aggregate per-agent infos into cluster-level metrics."""
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


def normalize_weights(weights_dict, keys):
    """Normalize a set of weight keys so they sum to 1."""
    total = sum(weights_dict.get(k, 0.0) for k in keys)
    if total > 0:
        return {k: weights_dict.get(k, 0.0) / total for k in keys}
    n = len(keys)
    return {k: 1.0 / n for k in keys}


def apply_best_weights(algo_params):
    """Apply an algorithm's Optuna-tuned best weights to global configs."""
    # Reward weights
    rw = normalize_weights(algo_params,
                           ["w_throughput", "w_delay", "w_failures", "w_jitter"])
    RLConfig.REWARD_W_THROUGHPUT = rw["w_throughput"]
    RLConfig.REWARD_W_DELAY     = rw["w_delay"]
    RLConfig.REWARD_W_FAILURES  = rw["w_failures"]
    RLConfig.REWARD_W_JITTER    = rw["w_jitter"]

    # Clustering membership weights
    cw = normalize_weights(algo_params,
                           ["c_w_dist", "c_w_sinr", "c_w_mob", "c_w_load"])
    CC.W_DIST = cw["c_w_dist"]
    CC.W_SINR = cw["c_w_sinr"]
    CC.W_MOB  = cw["c_w_mob"]
    CC.W_LOAD = cw["c_w_load"]

    # CH suitability weights
    ca = normalize_weights(algo_params,
                           ["c_a_energy", "c_a_degree", "c_a_mobstab",
                            "c_a_queue", "c_a_risk"])
    CC.A_ENERGY  = ca["c_a_energy"]
    CC.A_DEGREE  = ca["c_a_degree"]
    CC.A_MOBSTAB = ca["c_a_mobstab"]
    CC.A_QUEUE   = ca["c_a_queue"]
    CC.A_RISK    = ca["c_a_risk"]


# =====================================================================
# Step A: Load Models
# =====================================================================
def load_all_models(cp_dir, log):
    """Load all trained model checkpoints."""
    from stable_baselines3 import DQN, PPO, A2C

    models = {}  # name -> (model, algo_key_in_json)

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
# Step B: Evaluate All Models (Per-Algorithm Best Weights)
# =====================================================================
def evaluate_all_models(models, best_params, pps_list, seed, log):
    """
    Evaluate each model across the traffic sweep.
    For each RL algorithm, apply its own best clustering/reward weights
    before running the evaluation episodes.
    """
    from envs.marl_sarl_wrapper import MARLtoSARLWrapper

    results = []

    for model_name, (model, json_key) in models.items():
        # Apply per-algorithm best weights
        if json_key and json_key in best_params:
            apply_best_weights(best_params[json_key])
            log(f"\n  [{model_name}] Applied best weights from Optuna:")
            p = best_params[json_key]
            log(f"    lr={p.get('lr','N/A')}, batch={p.get('batch_size','N/A')}")
            log(f"    Reward: T={RLConfig.REWARD_W_THROUGHPUT:.3f} D={RLConfig.REWARD_W_DELAY:.3f} "
                f"F={RLConfig.REWARD_W_FAILURES:.3f} J={RLConfig.REWARD_W_JITTER:.3f}")
            log(f"    Cluster-Membership: dist={CC.W_DIST:.3f} sinr={CC.W_SINR:.3f} "
                f"mob={CC.W_MOB:.3f} load={CC.W_LOAD:.3f}")
            log(f"    CH-Suitability: energy={CC.A_ENERGY:.3f} degree={CC.A_DEGREE:.3f} "
                f"mobstab={CC.A_MOBSTAB:.3f} queue={CC.A_QUEUE:.3f} risk={CC.A_RISK:.3f}")
        else:
            log(f"\n  [{model_name}] Using default config weights (no Optuna params)")

        # Evaluate across traffic loads
        sarl_wrapper = MARLtoSARLWrapper()

        for pps in tqdm(pps_list, desc=f"Eval {model_name}", unit="load"):
            params.SWEEP_MAX_PPS = pps
            setattr(params, "OFFERED_PPS", int(pps))

            total_thr, total_delay, total_drops, total_collisions, steps = 0, 0, 0, 0, 0
            total_fairness, total_health = 0.0, 0.0
            mac_choices = []
            total_inf_time = 0.0
            inf_steps = 0

            obs, _ = sarl_wrapper.reset()
            done = False

            while not done:
                t0 = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
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

    return pd.DataFrame(results)


# =====================================================================
# Step C: Generate Publication-Quality Plots
# =====================================================================

# Consistent color + marker scheme across all plots
MODEL_STYLES = {
    "MCA-D3QN": {"color": "#e63946", "marker": "D", "ls": "-"},
    "MCA-PPO":  {"color": "#457b9d", "marker": "s", "ls": "-"},
    "DQN":      {"color": "#2a9d8f", "marker": "^", "ls": "--"},
    "PPO":      {"color": "#e9c46a", "marker": "v", "ls": "--"},
    "A2C":      {"color": "#f4a261", "marker": "o", "ls": "--"},
    "Tabular":  {"color": "#264653", "marker": "P", "ls": ":"},
}
BASELINE_STYLES = {
    "TDMA": {"color": "#adb5bd", "ls": "--", "lw": 1.5},
    "CSMA": {"color": "#6c757d", "ls": "-.", "lw": 1.5},
}


def _get_style(model_name):
    return MODEL_STYLES.get(model_name,
                            {"color": "#888888", "marker": "x", "ls": "-"})


def generate_all_plots(baseline_df, eval_df, out_dir, log):
    """Generate comprehensive comparison plots."""
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    if eval_df.empty:
        log("  No evaluation data -- skipping plot generation.")
        return

    rl_models = eval_df['Model'].unique()
    log(f"  Generating plots for models: {list(rl_models)}")

    # ---- Plot 1: Throughput vs Offered Load ----
    fig, ax = plt.subplots(figsize=(12, 6))
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Throughput_Mbps'],
                color=BASELINE_STYLES["TDMA"]["color"], ls=BASELINE_STYLES["TDMA"]["ls"],
                lw=BASELINE_STYLES["TDMA"]["lw"], label='TDMA (Baseline)', alpha=0.7)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Throughput_Mbps'],
                color=BASELINE_STYLES["CSMA"]["color"], ls=BASELINE_STYLES["CSMA"]["ls"],
                lw=BASELINE_STYLES["CSMA"]["lw"], label='CSMA/CA (Baseline)', alpha=0.7)
    for model_name in rl_models:
        s = _get_style(model_name)
        sub = eval_df[eval_df['Model'] == model_name]
        ax.plot(sub['Offered_Load_pps'], sub['Throughput_Mbps'],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=f"SARL: {model_name}", linewidth=2, markersize=5)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_ylabel("Throughput (Mbps)", fontsize=12)
    ax.set_title("SARL Comparison - Throughput vs Offered Load", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "01_throughput_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 01_throughput_comparison.png")

    # ---- Plot 2: Delay vs Offered Load ----
    fig, ax = plt.subplots(figsize=(12, 6))
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Delay_s'] * 1000,
                color=BASELINE_STYLES["TDMA"]["color"], ls=BASELINE_STYLES["TDMA"]["ls"],
                lw=BASELINE_STYLES["TDMA"]["lw"], label='TDMA (Baseline)', alpha=0.7)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Delay_s'] * 1000,
                color=BASELINE_STYLES["CSMA"]["color"], ls=BASELINE_STYLES["CSMA"]["ls"],
                lw=BASELINE_STYLES["CSMA"]["lw"], label='CSMA/CA (Baseline)', alpha=0.7)
    for model_name in rl_models:
        s = _get_style(model_name)
        sub = eval_df[eval_df['Model'] == model_name]
        ax.plot(sub['Offered_Load_pps'], sub['Delay_ms'],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=f"SARL: {model_name}", linewidth=2, markersize=5)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_ylabel("Delay (ms)", fontsize=12)
    ax.set_title("SARL Comparison - Delay vs Offered Load", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "02_delay_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 02_delay_comparison.png")

    # ---- Plot 3: Packet Drops vs Offered Load ----
    fig, ax = plt.subplots(figsize=(12, 6))
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Drops'],
                color=BASELINE_STYLES["TDMA"]["color"], ls=BASELINE_STYLES["TDMA"]["ls"],
                lw=BASELINE_STYLES["TDMA"]["lw"], label='TDMA (Baseline)', alpha=0.7)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Drops'],
                color=BASELINE_STYLES["CSMA"]["color"], ls=BASELINE_STYLES["CSMA"]["ls"],
                lw=BASELINE_STYLES["CSMA"]["lw"], label='CSMA/CA (Baseline)', alpha=0.7)
    for model_name in rl_models:
        s = _get_style(model_name)
        sub = eval_df[eval_df['Model'] == model_name]
        ax.plot(sub['Offered_Load_pps'], sub['Drops'],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=f"SARL: {model_name}", linewidth=2, markersize=5)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_ylabel("Packet Drops", fontsize=12)
    ax.set_title("SARL Comparison - Packet Drops vs Offered Load", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "03_drops_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 03_drops_comparison.png")

    # ---- Plot 4: Jain's Fairness Index vs Offered Load ----
    fig, ax = plt.subplots(figsize=(12, 6))
    if not baseline_df.empty and 'TDMA_Fairness' in baseline_df.columns:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Fairness'],
                color=BASELINE_STYLES["TDMA"]["color"], ls=BASELINE_STYLES["TDMA"]["ls"],
                lw=BASELINE_STYLES["TDMA"]["lw"], label='TDMA (Baseline)', alpha=0.7)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Fairness'],
                color=BASELINE_STYLES["CSMA"]["color"], ls=BASELINE_STYLES["CSMA"]["ls"],
                lw=BASELINE_STYLES["CSMA"]["lw"], label='CSMA/CA (Baseline)', alpha=0.7)
    for model_name in rl_models:
        s = _get_style(model_name)
        sub = eval_df[eval_df['Model'] == model_name]
        ax.plot(sub['Offered_Load_pps'], sub['Fairness'],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=f"SARL: {model_name}", linewidth=2, markersize=5)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax.set_title("SARL Comparison - Fairness vs Offered Load", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "04_fairness_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 04_fairness_comparison.png")

    # ---- Plot 5: Cluster-Head Health vs Offered Load ----
    fig, ax = plt.subplots(figsize=(12, 6))
    if not baseline_df.empty and 'TDMA_Health' in baseline_df.columns:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Health'],
                color=BASELINE_STYLES["TDMA"]["color"], ls=BASELINE_STYLES["TDMA"]["ls"],
                lw=BASELINE_STYLES["TDMA"]["lw"], label='TDMA (Baseline)', alpha=0.7)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Health'],
                color=BASELINE_STYLES["CSMA"]["color"], ls=BASELINE_STYLES["CSMA"]["ls"],
                lw=BASELINE_STYLES["CSMA"]["lw"], label='CSMA/CA (Baseline)', alpha=0.7)
    for model_name in rl_models:
        s = _get_style(model_name)
        sub = eval_df[eval_df['Model'] == model_name]
        ax.plot(sub['Offered_Load_pps'], sub['Health'],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                label=f"SARL: {model_name}", linewidth=2, markersize=5)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_ylabel("Cluster-Head Health", fontsize=12)
    ax.set_title("SARL Comparison - CH Health vs Offered Load", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "05_health_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 05_health_comparison.png")

    # ---- Plot 6: MAC Selection Preference (Horizontal Stacked Bar) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names_sorted = sorted(rl_models)
    tdma_pcts, csma_pcts = [], []
    for m in model_names_sorted:
        sub = eval_df[eval_df['Model'] == m]
        mean_tdma = sub['TDMA_Share'].mean() * 100
        tdma_pcts.append(mean_tdma)
        csma_pcts.append(100 - mean_tdma)
    y_pos = np.arange(len(model_names_sorted))
    ax.barh(y_pos, tdma_pcts, color='#2a9d8f', alpha=0.9, label='TDMA')
    ax.barh(y_pos, csma_pcts, left=tdma_pcts, color='#e76f51', alpha=0.85, label='CSMA/CA')
    for i, (t, c) in enumerate(zip(tdma_pcts, csma_pcts)):
        if t > 8:
            ax.text(t / 2, i, f"{t:.0f}%", ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if c > 8:
            ax.text(t + c / 2, i, f"{c:.0f}%", ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names_sorted, fontsize=11)
    ax.set_xlabel("Selection Share (%)", fontsize=12)
    ax.set_title("MAC Protocol Selection Preference per Agent", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "06_mac_selection_preference.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 06_mac_selection_preference.png")

    # ---- Plot 7: Summary Bar Chart (4 metrics) ----
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    metrics = ["Throughput_Mbps", "Delay_ms", "Drops", "Collisions"]
    titles = ["Avg Throughput (Mbps)", "Avg Delay (ms)", "Total Drops", "Total Collisions"]
    better_dir = ["higher", "lower", "lower", "lower"]
    for ax, metric, title, direction in zip(axes, metrics, titles, better_dir):
        agg_fn = "mean" if metric in ("Throughput_Mbps", "Delay_ms") else "sum"
        model_vals = eval_df.groupby("Model")[metric].agg(agg_fn).reindex(rl_models)
        colors = [_get_style(m)["color"] for m in rl_models]
        bars = ax.bar(rl_models, model_vals.values, color=colors, edgecolor='white', linewidth=0.8)
        if direction == "higher":
            best_idx = np.argmax(model_vals.values)
        else:
            best_idx = np.argmin(model_vals.values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("SARL Summary - Key Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "07_summary_bar_chart.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 07_summary_bar_chart.png")

    # ---- Plot 8: Inference Latency Comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))
    inf_means = eval_df.groupby("Model")["Avg_Inference_ms"].mean().reindex(rl_models)
    colors = [_get_style(m)["color"] for m in rl_models]
    bars = ax.bar(rl_models, inf_means.values, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, inf_means.values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02,
                f"{val:.2f}ms", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel("Avg Inference Latency (ms)", fontsize=12)
    ax.set_title("Per-Model Inference Latency", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "08_inference_latency.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 08_inference_latency.png")

    # ---- Plot 9: TDMA Share Heatmap (Model x Load) ----
    fig, ax = plt.subplots(figsize=(14, 5))
    pivot = eval_df.pivot_table(values='TDMA_Share', index='Model', columns='Offered_Load_pps')
    pivot = pivot.reindex(rl_models)
    cmap = sns.color_palette("RdYlGn", as_cmap=True) if _HAS_SEABORN else "RdYlGn"
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(c) for c in pivot.columns], rotation=45, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Offered Load (pps)", fontsize=12)
    ax.set_title("TDMA Selection Share (Green=TDMA, Red=CSMA)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="TDMA Share")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "09_tdma_share_heatmap.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 09_tdma_share_heatmap.png")

    # ---- Plot 10: Radar/Spider Chart ----
    radar_metrics = ["Throughput_Mbps", "Fairness", "Health"]
    inv_metrics = ["Delay_ms", "Drops"]
    labels = ["Throughput", "Fairness", "Health", "Low Delay", "Low Drops"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for model_name in rl_models:
        sub = eval_df[eval_df['Model'] == model_name]
        vals = []
        for m in radar_metrics:
            v = sub[m].mean()
            col_max = eval_df[m].max()
            vals.append(v / col_max if col_max > 0 else 0)
        for m in inv_metrics:
            v = sub[m].mean()
            col_max = eval_df[m].max()
            vals.append(1.0 - (v / col_max) if col_max > 0 else 1.0)
        vals += vals[:1]
        s = _get_style(model_name)
        ax.plot(angles, vals, color=s["color"], linewidth=2, label=model_name)
        ax.fill(angles, vals, color=s["color"], alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Multi-Metric Radar - Normalized Performance", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "10_radar_chart.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 10_radar_chart.png")

    # ---- Plot 11: Combined 2x2 Overview ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Throughput
    ax = axes[0, 0]
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Throughput_Mbps'],
                color=BASELINE_STYLES["TDMA"]["color"], ls=BASELINE_STYLES["TDMA"]["ls"],
                lw=1.5, label='TDMA', alpha=0.6)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Throughput_Mbps'],
                color=BASELINE_STYLES["CSMA"]["color"], ls=BASELINE_STYLES["CSMA"]["ls"],
                lw=1.5, label='CSMA', alpha=0.6)
    for mn in rl_models:
        s = _get_style(mn)
        sub = eval_df[eval_df['Model'] == mn]
        ax.plot(sub['Offered_Load_pps'], sub['Throughput_Mbps'],
                color=s["color"], marker=s["marker"], ls=s["ls"], lw=2, ms=4, label=mn)
    ax.set_xlabel("Load (pps)"); ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Throughput", fontweight="bold"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')
    # Delay
    ax = axes[0, 1]
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Delay_s'] * 1000,
                color=BASELINE_STYLES["TDMA"]["color"], ls=BASELINE_STYLES["TDMA"]["ls"],
                lw=1.5, label='TDMA', alpha=0.6)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Delay_s'] * 1000,
                color=BASELINE_STYLES["CSMA"]["color"], ls=BASELINE_STYLES["CSMA"]["ls"],
                lw=1.5, label='CSMA', alpha=0.6)
    for mn in rl_models:
        s = _get_style(mn)
        sub = eval_df[eval_df['Model'] == mn]
        ax.plot(sub['Offered_Load_pps'], sub['Delay_ms'],
                color=s["color"], marker=s["marker"], ls=s["ls"], lw=2, ms=4, label=mn)
    ax.set_xlabel("Load (pps)"); ax.set_ylabel("Delay (ms)")
    ax.set_title("Delay", fontweight="bold"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')
    # Drops
    ax = axes[1, 0]
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Drops'],
                color=BASELINE_STYLES["TDMA"]["color"], ls=BASELINE_STYLES["TDMA"]["ls"],
                lw=1.5, label='TDMA', alpha=0.6)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Drops'],
                color=BASELINE_STYLES["CSMA"]["color"], ls=BASELINE_STYLES["CSMA"]["ls"],
                lw=1.5, label='CSMA', alpha=0.6)
    for mn in rl_models:
        s = _get_style(mn)
        sub = eval_df[eval_df['Model'] == mn]
        ax.plot(sub['Offered_Load_pps'], sub['Drops'],
                color=s["color"], marker=s["marker"], ls=s["ls"], lw=2, ms=4, label=mn)
    ax.set_xlabel("Load (pps)"); ax.set_ylabel("Drops")
    ax.set_title("Packet Drops", fontweight="bold"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')
    # Fairness
    ax = axes[1, 1]
    if not baseline_df.empty and 'TDMA_Fairness' in baseline_df.columns:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Fairness'],
                color=BASELINE_STYLES["TDMA"]["color"], ls=BASELINE_STYLES["TDMA"]["ls"],
                lw=1.5, label='TDMA', alpha=0.6)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Fairness'],
                color=BASELINE_STYLES["CSMA"]["color"], ls=BASELINE_STYLES["CSMA"]["ls"],
                lw=1.5, label='CSMA', alpha=0.6)
    for mn in rl_models:
        s = _get_style(mn)
        sub = eval_df[eval_df['Model'] == mn]
        ax.plot(sub['Offered_Load_pps'], sub['Fairness'],
                color=s["color"], marker=s["marker"], ls=s["ls"], lw=2, ms=4, label=mn)
    ax.set_xlabel("Load (pps)"); ax.set_ylabel("Fairness Index")
    ax.set_title("Jain's Fairness", fontweight="bold"); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='lower left')
    plt.suptitle(f"SARL Unified Evaluation Overview  (N={params.N}, PHY={params.PHY_RATE_BPS/1e6:.0f}Mbps)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(img_dir, "11_combined_overview.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 11_combined_overview.png")

    # ---- Plot 12: Summary Table (CSV + image) ----
    summary_rows = []
    for mn in rl_models:
        sub = eval_df[eval_df['Model'] == mn]
        summary_rows.append({
            'Model': mn,
            'Avg Throughput (Mbps)': round(sub['Throughput_Mbps'].mean(), 4),
            'Avg Delay (ms)': round(sub['Delay_ms'].mean(), 2),
            'Total Drops': int(sub['Drops'].sum()),
            'Avg Fairness': round(sub['Fairness'].mean(), 4),
            'Avg Health': round(sub['Health'].mean(), 4),
            'Avg TDMA %': round(sub['TDMA_Share'].mean() * 100, 1),
            'Avg Inference (ms)': round(sub['Avg_Inference_ms'].mean(), 3),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "csv", "summary_table.csv"), index=False)
    log(f"    [OK] summary_table.csv")

    fig, ax = plt.subplots(figsize=(16, 2 + 0.4 * len(summary_rows)))
    ax.axis('off')
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center', loc='center',
                     colColours=['#f0f0f0'] * len(summary_df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    for col_idx, col in enumerate(summary_df.columns):
        if col == 'Model':
            continue
        vals = summary_df[col].values
        if col in ('Avg Delay (ms)', 'Total Drops'):
            best_idx = np.argmin(vals)
        else:
            best_idx = np.argmax(vals)
        table[best_idx + 1, col_idx].set_facecolor('#d4edda')
    ax.set_title("SARL Evaluation Summary", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "12_summary_table.png"), dpi=200, bbox_inches='tight')
    plt.close()
    log("    [OK] 12_summary_table.png")

    log(f"\n  All plots saved to: {img_dir}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Unified Evaluation & Plot Generation (Best Weights)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to checkpoints directory (auto-detected if None)")
    parser.add_argument("--best-weights", type=str, default=None,
                        help="Path to best_weights_and_params JSON file")
    parser.add_argument("--baseline-csv", type=str, default=None,
                        help="Path to baseline_results.csv (auto-detected if None)")
    parser.add_argument("--seed", type=int, default=params.SEED, help="Random seed")
    parser.add_argument("--sweep-steps", type=int, default=params.SWEEP_STEPS,
                        help="Number of load points")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, just regenerate plots from existing CSV")
    args = parser.parse_args()

    # Auto-detect paths
    results_root = os.path.join(project_root, "results")

    # Find latest parallel results directory
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
            parent_dir = os.path.join(results_root, latest)
        else:
            print("ERROR: No sarl_comparison_parallel_* directory found.")
            sys.exit(1)
    else:
        parent_dir = os.path.dirname(args.checkpoint_dir)

    # Best weights JSON
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

    # Baseline CSV
    if args.baseline_csv is None:
        args.baseline_csv = os.path.join(parent_dir, "csv", "baseline_results.csv")

    # Create unified output directory
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_root, f"unified_evaluation_{ts}")
    for sub in ["images", "csv", "logs"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # Logger
    log_file = open(os.path.join(out_dir, "logs", "evaluation.log"), "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 70)
    log("  STEP 6: UNIFIED EVALUATION & PLOT GENERATION")
    log("=" * 70)
    log(f"  Timestamp:      {datetime.datetime.now().isoformat()}")
    log(f"  Checkpoints:    {args.checkpoint_dir}")
    log(f"  Best Weights:   {args.best_weights}")
    log(f"  Baseline CSV:   {args.baseline_csv}")
    log(f"  Output Dir:     {out_dir}")
    log(f"  Seed:           {args.seed}")
    log(f"  Sweep Steps:    {args.sweep_steps}")
    log(f"  N={params.N}, PHY={params.PHY_RATE_BPS/1e6:.0f}Mbps, Fading={params.FADING_MODEL}")
    log("=" * 70)

    # Load best weights
    with open(args.best_weights, "r") as f:
        best_params = json.load(f)
    log(f"\n  Loaded best weights for algorithms: {list(best_params.keys())}")

    # Load baseline
    if os.path.exists(args.baseline_csv):
        baseline_df = pd.read_csv(args.baseline_csv)
        log(f"  Loaded baseline CSV: {len(baseline_df)} rows")
    else:
        baseline_df = pd.DataFrame()
        log("  WARNING: No baseline CSV found")

    # Load models
    log("\n  Loading model checkpoints...")
    models = load_all_models(args.checkpoint_dir, log)
    log(f"  Loaded {len(models)} models: {list(models.keys())}")

    if not models:
        log("  ERROR: No models could be loaded. Aborting.")
        log_file.close()
        sys.exit(1)

    # Traffic sweep
    pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS,
                           args.sweep_steps).astype(int)

    # Evaluate
    if not args.skip_eval:
        log("\n" + "=" * 70)
        log("  RUNNING EVALUATION SWEEP")
        log("=" * 70)
        eval_df = evaluate_all_models(models, best_params, pps_list, args.seed, log)
        eval_csv_path = os.path.join(out_dir, "csv", "unified_eval_sweep.csv")
        eval_df.to_csv(eval_csv_path, index=False)
        log(f"\n  Saved evaluation CSV: {eval_csv_path}")
    else:
        existing_csv = os.path.join(parent_dir, "csv", "unified_eval_sweep.csv")
        if os.path.exists(existing_csv):
            eval_df = pd.read_csv(existing_csv)
            eval_df.to_csv(os.path.join(out_dir, "csv", "unified_eval_sweep.csv"), index=False)
            log(f"  Loaded existing evaluation CSV: {existing_csv}")
        else:
            log("  ERROR: --skip-eval but no existing CSV found. Run without --skip-eval.")
            log_file.close()
            sys.exit(1)

    # Copy baseline to unified dir
    if not baseline_df.empty:
        baseline_df.to_csv(os.path.join(out_dir, "csv", "baseline_results.csv"), index=False)

    # Generate Plots
    log("\n" + "=" * 70)
    log("  GENERATING PLOTS")
    log("=" * 70)
    generate_all_plots(baseline_df, eval_df, out_dir, log)

    # Save metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint_source": args.checkpoint_dir,
        "best_weights_source": args.best_weights,
        "seed": args.seed,
        "n_nodes": params.N,
        "phy_rate_bps": params.PHY_RATE_BPS,
        "fading_model": params.FADING_MODEL,
        "mobility_model": params.MOBILITY_MODEL,
        "models_evaluated": list(eval_df['Model'].unique()) if not eval_df.empty else [],
        "load_points": len(pps_list),
        "best_params_used": best_params,
    }
    with open(os.path.join(out_dir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log("\n" + "=" * 70)
    log("  STEP 6 COMPLETE")
    log(f"  Results: {out_dir}")
    log("=" * 70)
    log_file.close()

    print(f"\nDone! All results in: {out_dir}")


if __name__ == "__main__":
    main()
