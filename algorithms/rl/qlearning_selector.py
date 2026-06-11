import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              ORACLE MAC SELECTOR  (Upper-Bound Benchmark)       ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  This agent has FULL HINDSIGHT: it sees metrics for BOTH TDMA   ║
    ║  and CSMA/CA at each traffic point BEFORE making its decision.  ║
    ║  It represents the THEORETICAL UPPER BOUND on MAC selection     ║
    ║  performance — no online RL agent can outperform this.          ║
    ║                                                                  ║
    ║  Usage in journal:                                               ║
    ║    • Plot as dark-black dashed line labelled "Oracle (Upper Bound)" ║
    ║    • The closer an RL agent's curve is to the Oracle, the        ║
    ║      better that agent has learned the optimal MAC policy.        ║
    ╚══════════════════════════════════════════════════════════════════╝

    Actions: 0 = TDMA,  1 = CSMA/CA

    NOTE: Offline full-information updates — at each logged point we
    already have metrics for BOTH MACs, so we update Q(s,0) and Q(s,1)
    every step (no exploration needed). This is NOT a fair online agent.
    """

    def __init__(self, alpha: float, gamma: float, epsilon: float, n_bins: int):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.n_bins = int(max(1, n_bins))

        # Q-table: state -> [Q_TDMA, Q_CSMA]
        self.Q = {}
        # visit counts (offline: updated for both actions)
        self.N_sa = {}

    def _ensure_state(self, state: int):
        if state not in self.Q:
            self.Q[state] = [0.0, 0.0]
            self.N_sa[state] = [0, 0]

    def get_state(self, traffic_rate: float, min_rate: float, max_rate: float) -> int:
        """
        Discretize traffic_rate into n_bins states.
        """
        if self.n_bins <= 1 or max_rate <= min_rate:
            return 0

        # Clamp
        if traffic_rate <= min_rate:
            return 0
        if traffic_rate >= max_rate:
            return self.n_bins - 1

        bin_size = (max_rate - min_rate) / self.n_bins
        state = int((traffic_rate - min_rate) / bin_size)
        return int(min(max(state, 0), self.n_bins - 1))

    def get_q_values(self, state: int):
        self._ensure_state(state)
        return self.Q[state]

    def select_action(self, state: int) -> int:
        """
        Epsilon-greedy selection with RANDOM tie-break (no TDMA bias).
        """
        q_vals = self.get_q_values(state)

        # Explore
        if np.random.rand() < self.epsilon:
            return int(np.random.choice([0, 1]))

        # Exploit with random tie-break
        if q_vals[0] == q_vals[1]:
            return int(np.random.choice([0, 1]))
        return int(np.argmax(q_vals))

    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Standard Q-learning update.
        With gamma=0 this becomes a contextual bandit update:
          Q(s,a) <- (1-a)Q(s,a) + a * reward
        """
        self._ensure_state(state)
        self._ensure_state(next_state)

        best_next_q = float(np.max(self.Q[next_state]))
        td_target = float(reward) + self.gamma * best_next_q
        td_error = td_target - self.Q[state][action]

        self.Q[state][action] += self.alpha * td_error
        self.N_sa[state][action] += 1


def run_rl_postprocess(out_dir, df: pd.DataFrame, params, log_print):
    """
    Consumes existing simulation results (TDMA + CSMA/CA metrics per point)
    and produces:
      - csv/qlearning_mac_selection.csv
      - images/qlearning_selected_mac_vs_traffic_rate.png
      - images/delay_curves_with_qlearning_choice.png
      - images/throughput_curves_with_qlearning_choice.png

    IMPORTANT: This is a post-processing step only. It must NOT alter existing
    simulation outputs; it only adds RL artifacts.
    """

    # -----------------------------
    # Hyperparameters (configurable)
    # -----------------------------
    alpha = float(getattr(params, "RL_ALPHA", 0.1))
    # Contextual bandit behavior is correct here (choose best MAC at each point)
    gamma = float(getattr(params, "RL_GAMMA", 0.0))
    epsilon = float(getattr(params, "RL_EPSILON", 0.0))  # set 0.0 for deterministic selection
    wT = float(getattr(params, "RL_WT", 0.5))
    wD = float(getattr(params, "RL_WD", 0.5))
    n_bins_cfg = int(getattr(params, "RL_TRAFFIC_BINS", 10))

    # -----------------------------
    # Sanity checks / required cols
    # -----------------------------
    required_cols = [
        "Offered_Load_pps",
        "TDMA_Throughput_Mbps",
        "CSMA_Throughput_Mbps",
        "TDMA_Delay_s",
        "CSMA_Delay_s",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"run_rl_postprocess: missing required column: {c}")

    # Ensure output subdirs exist
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    # -----------------------------
    # Binning (avoid single-visit states)
    # -----------------------------
    unique_rates = int(df["Offered_Load_pps"].nunique())
    # Make bins coarse enough so states are revisited (prevents "first-visit" bias)
    # This is still useful even though we do full-information updates.
    n_bins = min(n_bins_cfg, max(1, unique_rates // 2))

    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, n_bins=n_bins)

    min_rate = float(df["Offered_Load_pps"].min())
    max_rate = float(df["Offered_Load_pps"].max())
    if max_rate == min_rate:
        max_rate = min_rate + 1.0  # prevent divide-by-zero in binning

    eps_val = 1e-12
    MAC_NAMES = {0: "TDMA", 1: "CSMA_CA"}

    results = []

    # -----------------------------
    # Reward definition (LOCAL per-point normalization)
    # -----------------------------
    def local_reward(t_tdma, d_tdma, t_csma, d_csma, action_t, action_d):
        """
        Compute reward for a single MAC at this point using LOCAL normalization
        based on the two MACs at the same traffic rate.

        Throughput: higher is better -> normalize by max(T)
        Delay: lower is better -> normalize via min/max scaling:
              D_norm = (D - D_min)/(D_max - D_min + eps)
        Reward: r = wT * T_norm - wD * D_norm
        """
        T_max = max(t_tdma, t_csma) + eps_val
        D_min = min(d_tdma, d_csma)
        D_max = max(d_tdma, d_csma)

        T_norm = action_t / T_max
        D_norm = (action_d - D_min) / (D_max - D_min + eps_val)

        return (wT * T_norm) - (wD * D_norm)

    # Penalize "fake zero delay" when throughput is basically zero (no deliveries)
    # This prevents CSMA/CA (or TDMA) looking perfect because delay=0 but nothing delivered.
    def apply_no_delivery_penalty(reward, throughput_mbps, offered_pps):
        if offered_pps > 0 and throughput_mbps <= 1e-9:
            return -1e6
        return reward

    # -----------------------------
    # Main loop
    # -----------------------------
    for i in range(len(df)):
        row = df.iloc[i]
        traffic_rate = float(row["Offered_Load_pps"])
        state = agent.get_state(traffic_rate, min_rate, max_rate)

        # Read metrics
        T_TDMA = float(row["TDMA_Throughput_Mbps"])
        T_CSMA = float(row["CSMA_Throughput_Mbps"])
        D_TDMA = float(row["TDMA_Delay_s"])
        D_CSMA = float(row["CSMA_Delay_s"])

        # Compute rewards for BOTH actions (full-information)
        reward_tdma = local_reward(T_TDMA, D_TDMA, T_CSMA, D_CSMA, T_TDMA, D_TDMA)
        reward_csma = local_reward(T_TDMA, D_TDMA, T_CSMA, D_CSMA, T_CSMA, D_CSMA)

        reward_tdma = apply_no_delivery_penalty(reward_tdma, T_TDMA, traffic_rate)
        reward_csma = apply_no_delivery_penalty(reward_csma, T_CSMA, traffic_rate)

        # Next state (for completeness; gamma usually 0 here)
        if i + 1 < len(df):
            next_traffic_rate = float(df.iloc[i + 1]["Offered_Load_pps"])
            next_state = agent.get_state(next_traffic_rate, min_rate, max_rate)
        else:
            next_state = state

        # Snapshot Q before update (for logging)
        q_before = agent.get_q_values(state).copy()

        # OFFLINE FULL-INFORMATION UPDATES: update both actions every point
        agent.update(state, 0, reward_tdma, next_state)
        agent.update(state, 1, reward_csma, next_state)

        # Select action after updating (or use greedy on rewards)
        action = agent.select_action(state)
        selected_mac_name = MAC_NAMES[action]
        reward_selected = reward_tdma if action == 0 else reward_csma

        q_after = agent.get_q_values(state).copy()

        results.append(
            {
                "timestamp": i,
                "state": state,
                "traffic_rate": traffic_rate,
                "traffic_rate_unit": "pps",
                "tdma_delay": D_TDMA,
                "csma_delay": D_CSMA,
                "delay_unit": "s",
                "tdma_throughput": T_TDMA,
                "csma_throughput": T_CSMA,
                "throughput_unit": "Mbps",
                "qlearning_selected_mac": selected_mac_name,
                "reward_tdma": reward_tdma,
                "reward_csma": reward_csma,
                "q_tdma_before": q_before[0],
                "q_csma_before": q_before[1],
                "q_tdma_after": q_after[0],
                "q_csma_after": q_after[1],
                "reward_selected": reward_selected,
                "epsilon": epsilon,
                "alpha": alpha,
                "gamma": gamma,
                "n_bins": n_bins,
            }
        )

    # -----------------------------
    # Save RL CSV
    # -----------------------------
    out_csv_path = os.path.join(out_dir, "csv", "qlearning_mac_selection.csv")
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv_path, index=False)
    log_print(f"Saved Q-learning selection data to {out_csv_path}")

    # -----------------------------
    # Plot 1: Selected MAC vs Traffic Rate
    # -----------------------------
    plt.figure(figsize=(10, 6))
    y_vals = [0 if m == "TDMA" else 1 for m in res_df["qlearning_selected_mac"]]
    plt.scatter(
        res_df["traffic_rate"],
        y_vals,
        c=y_vals,
        cmap="coolwarm",
        s=100,
        edgecolor="k",
    )
    plt.yticks([0, 1], ["TDMA", "CSMA/CA"])
    plt.xlabel("Traffic Rate (pps)")
    plt.ylabel("Selected MAC")
    plt.title(
        f"RL Selected MAC vs Traffic Rate\n"
        f"(N={getattr(params, 'N', 'NA')}, RTS/CTS={'ON' if getattr(params, 'RTS_CTS_ENABLED', False) else 'OFF'}, "
        f"ACK={'ON' if getattr(params, 'ACK_ENABLED', False) else 'OFF'})"
    )
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "qlearning_selected_mac_vs_traffic_rate.png"))
    plt.close()

    # -----------------------------
    # Plot 2: Delay curves + RL choices overlay
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        df["Offered_Load_pps"],
        df["TDMA_Delay_s"],
        label="TDMA (Base)",
        color="blue",
        alpha=0.5,
        linestyle="--",
    )
    plt.plot(
        df["Offered_Load_pps"],
        df["CSMA_Delay_s"],
        label="CSMA/CA (Base)",
        color="red",
        alpha=0.5,
        linestyle="--",
    )

    tdma_choices = res_df[res_df["qlearning_selected_mac"] == "TDMA"]
    csma_choices = res_df[res_df["qlearning_selected_mac"] == "CSMA_CA"]

    plt.scatter(
        tdma_choices["traffic_rate"],
        tdma_choices["tdma_delay"],
        color="blue",
        s=100,
        label="Selected TDMA",
        marker="o",
        edgecolor="k",
    )
    plt.scatter(
        csma_choices["traffic_rate"],
        csma_choices["csma_delay"],
        color="red",
        s=100,
        label="Selected CSMA/CA",
        marker="^",
        edgecolor="k",
    )

    plt.xlabel("Traffic Rate (pps)")
    plt.ylabel("End-to-End Delay (s)")
    plt.title("Delay Curves with RL MAC Selection Overlay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "delay_curves_with_qlearning_choice.png"))
    plt.close()

    # -----------------------------
    # Plot 3: Throughput curves + RL choices overlay
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        df["Offered_Load_pps"],
        df["TDMA_Throughput_Mbps"],
        label="TDMA (Base)",
        color="blue",
        alpha=0.5,
        linestyle="--",
    )
    plt.plot(
        df["Offered_Load_pps"],
        df["CSMA_Throughput_Mbps"],
        label="CSMA/CA (Base)",
        color="red",
        alpha=0.5,
        linestyle="--",
    )

    plt.scatter(
        tdma_choices["traffic_rate"],
        tdma_choices["tdma_throughput"],
        color="blue",
        s=100,
        label="Selected TDMA",
        marker="o",
        edgecolor="k",
    )
    plt.scatter(
        csma_choices["traffic_rate"],
        csma_choices["csma_throughput"],
        color="red",
        s=100,
        label="Selected CSMA/CA",
        marker="^",
        edgecolor="k",
    )

    plt.xlabel("Traffic Rate (pps)")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Throughput Curves with RL MAC Selection Overlay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "throughput_curves_with_qlearning_choice.png"))
    plt.close()
    
    return res_df
