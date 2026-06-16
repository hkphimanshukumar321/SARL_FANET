import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Apply seaborn style
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)

MODEL_STYLES = {
    "MCA-D3QN": {"color": "#e63946", "ls": "-"},
    "MCA-PPO":  {"color": "#457b9d", "ls": "-"},
    "DQN":      {"color": "#2a9d8f", "ls": "--"},
    "PPO":      {"color": "#e9c46a", "ls": "--"},
    "A2C":      {"color": "#f4a261", "ls": "--"},
    "Tabular":  {"color": "#264653", "ls": ":"},
}

def plot_training_curves(csv_dir, out_path, window=50):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Mapping CSV files to model names
    file_mapping = {
        "mca_d3qn_training_rewards.csv": "MCA-D3QN",
        "mca_ppo_training_rewards.csv": "MCA-PPO",
        "dqn_training_rewards.csv": "DQN",
        "ppo_training_rewards.csv": "PPO",
        "a2c_training_rewards.csv": "A2C",
        "tabular_training_rewards.csv": "Tabular"
    }

    found_any = False

    for csv_file, model_name in file_mapping.items():
        path = os.path.join(csv_dir, csv_file)
        if os.path.exists(path):
            found_any = True
            df = pd.read_csv(path)
            
            # Smooth the rewards using a rolling window
            if "reward" in df.columns:
                df["smooth_reward"] = df["reward"].rolling(window=window, min_periods=1).mean()
                
                style = MODEL_STYLES.get(model_name, {"color": "black", "ls": "-"})
                
                # Plot smoothed line
                ax.plot(df["step"], df["smooth_reward"], label=model_name, 
                        color=style["color"], linestyle=style["ls"], linewidth=2)
                
                # Plot faded original line (optional, maybe too noisy)
                ax.plot(df["step"], df["reward"], color=style["color"], alpha=0.15)
                
    if not found_any:
        print(f"Error: No training reward CSVs found in {csv_dir}")
        return

    ax.set_title("Training Convergence (Reward vs Timesteps)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episodic Reward", fontsize=12)
    ax.legend(title="RL Agents", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved training curve plot to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot training curves from CSV files.")
    parser.add_argument("--csv-dir", type=str, required=True, help="Directory containing the training reward CSV files.")
    parser.add_argument("--out-path", type=str, required=True, help="Path to save the output plot PNG.")
    args = parser.parse_args()
    
    plot_training_curves(args.csv_dir, args.out_path)
