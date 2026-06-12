"""
Generate training reward plots from existing CSV artifacts.

Usage examples:
  python experiments/plot_training_rewards.py --csv-dir results/N150_PHY5M_Q70_RTSCTS_ACK_enabled/trial_UNIFIED_20260316_200811/csv
  python experiments/plot_training_rewards.py --trial-dir results/N150_PHY5M_Q70_RTSCTS_ACK_enabled/trial_UNIFIED_20260316_200811
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

SARL_MODELS = ["tabular", "dqn", "ppo", "a2c", "mca_d3qn"]
MARL_MODELS = ["iql", "vdn", "qmix", "magat_d3qn"]


def _rolling(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=max(window, 1), min_periods=1).mean()


def _load_training_csvs(csv_dir: str) -> Dict[str, pd.DataFrame]:
    data = {}
    for name in SARL_MODELS + MARL_MODELS:
        path = os.path.join(csv_dir, f"{name}_training_rewards.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "reward" in df.columns and len(df) > 0:
                data[name] = df
    return data


def _to_step_axis(df: pd.DataFrame) -> np.ndarray:
    if "step" in df.columns:
        return df["step"].to_numpy()
    if "episode" in df.columns:
        ep = df["episode"].to_numpy()
        return ep + 1
    return np.arange(1, len(df) + 1)


def _to_episode_axis(df: pd.DataFrame) -> np.ndarray:
    if "episode" in df.columns:
        return df["episode"].to_numpy()
    return np.arange(1, len(df) + 1)


def _plot_group(
    out_path: str,
    title: str,
    x_label: str,
    x_builder,
    series: List[Tuple[str, pd.DataFrame]],
) -> bool:
    if not series:
        return False

    plt.figure(figsize=(12, 6), dpi=150)
    palette = sns.color_palette("husl", len(series)) if _HAS_SEABORN else None
    for idx, (model_name, df) in enumerate(series):
        x = x_builder(df)
        y = _rolling(df["reward"], window=max(1, len(df) // 20))
        color = palette[idx] if palette else None
        plt.plot(x, y, linewidth=2, label=model_name.upper(), color=color)

    plt.xlabel(x_label)
    plt.ylabel("Reward (smoothed)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def generate_plots(csv_dir: str, images_dir: str) -> List[str]:
    os.makedirs(images_dir, exist_ok=True)
    data = _load_training_csvs(csv_dir)

    sarl_series = [(k, data[k]) for k in SARL_MODELS if k in data]
    marl_series = [(k, data[k]) for k in MARL_MODELS if k in data]

    written = []

    if _plot_group(
        out_path=os.path.join(images_dir, "sarl_reward_vs_steps.png"),
        title="SARL Training Rewards vs Steps",
        x_label="Steps",
        x_builder=_to_step_axis,
        series=sarl_series,
    ):
        written.append("sarl_reward_vs_steps.png")

    if _plot_group(
        out_path=os.path.join(images_dir, "marl_reward_vs_episodes.png"),
        title="MARL Training Rewards vs Episodes",
        x_label="Episodes",
        x_builder=_to_episode_axis,
        series=marl_series,
    ):
        written.append("marl_reward_vs_episodes.png")

    return written


def main():
    parser = argparse.ArgumentParser(description="Generate SARL/MARL reward plots from training CSVs")
    parser.add_argument("--csv-dir", default=None, help="Path to csv folder containing *_training_rewards.csv")
    parser.add_argument("--trial-dir", default=None, help="Path to trial folder with csv/ and images/")
    args = parser.parse_args()

    if args.csv_dir:
        csv_dir = args.csv_dir
        trial_dir = os.path.dirname(csv_dir.rstrip("/\\"))
    elif args.trial_dir:
        trial_dir = args.trial_dir
        csv_dir = os.path.join(trial_dir, "csv")
    else:
        raise ValueError("Provide either --csv-dir or --trial-dir")

    images_dir = os.path.join(trial_dir, "images")
    written = generate_plots(csv_dir=csv_dir, images_dir=images_dir)

    if written:
        print("Generated plots:")
        for name in written:
            print(f"  - {os.path.join(images_dir, name)}")
    else:
        print("No training reward CSVs found. No plots generated.")


if __name__ == "__main__":
    main()
