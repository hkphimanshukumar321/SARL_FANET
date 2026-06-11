"""
rewards.py -- Decentralized reward functions for cluster-based FANET control.
"""

from __future__ import annotations

import numpy as np

from configs.cluster_config import ClusterConfig as CC


def _clip01(value: float) -> float:
    return float(max(0.0, min(float(value), 1.0)))


def compute_cluster_reward(
    local_throughput_bps: float,
    inter_throughput_bps: float,
    local_delay_ms: float,
    inter_delay_ms: float,
    collisions: int,
    energy_cost: float,
    interference_penalty: float,
    coordination_failure_ratio: float,
    queue_overflow_ratio: float,
    handover_flag: bool,
) -> float:
    """
    Compute local reward for one cluster-head.

    The burst split rho_k(t) should improve the balance between:
    - intra-cluster service performance driven by T1_k
    - inter-cluster coordination quality driven by T2_k
    """
    t_local_norm = _clip01((local_throughput_bps / 1e6) / CC.MAX_THROUGHPUT_MBPS)
    t_inter_norm = _clip01((inter_throughput_bps / 1e6) / CC.MAX_THROUGHPUT_MBPS)
    d_local_norm = _clip01(local_delay_ms / CC.MAX_DELAY_MS)
    d_inter_norm = _clip01(inter_delay_ms / CC.MAX_DELAY_MS)
    c_norm = _clip01(collisions / CC.MAX_COLLISIONS)
    e_norm = _clip01(energy_cost / CC.MAX_ENERGY_COST)
    i_norm = _clip01(interference_penalty)
    f_norm = _clip01(coordination_failure_ratio / CC.MAX_COORD_FAILURE_RATIO)
    q_norm = _clip01(queue_overflow_ratio / CC.MAX_QUEUE_OVERFLOW_RATIO)
    h_norm = 1.0 if handover_flag else 0.0

    return (
        CC.ALPHA_LOCAL_THROUGHPUT * t_local_norm
        + CC.ALPHA_INTER_THROUGHPUT * t_inter_norm
        - CC.BETA_LOCAL_DELAY * d_local_norm
        - CC.BETA_INTER_DELAY * d_inter_norm
        - CC.GAMMA_COLLISION * c_norm
        - CC.DELTA_ENERGY * e_norm
        - CC.ETA_INTERFERENCE * i_norm
        - CC.PSI_COORD_FAILURE * f_norm
        - CC.PHI_QUEUE_OVERFLOW * q_norm
        - CC.ZETA_HANDOVER * h_norm
    )


def compute_team_reward(
    cluster_rewards: list[float],
    cluster_throughputs: list[float],
) -> float:
    """
    Cooperative team reward used by centralized baselines and optional CTDE terms.
    """
    if not cluster_rewards:
        return 0.0

    base_reward = float(np.mean(cluster_rewards))
    th_array = np.asarray(cluster_throughputs, dtype=np.float64)
    sum_th = float(np.sum(th_array))
    if sum_th > 1e-9:
        fairness = (sum_th ** 2) / (len(th_array) * float(np.sum(th_array ** 2)))
    else:
        fairness = 1.0
    return base_reward + CC.LAMBDA_FAIRNESS * float(fairness)
