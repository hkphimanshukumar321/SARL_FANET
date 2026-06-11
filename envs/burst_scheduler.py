"""
burst_scheduler.py -- Helpers for discrete burst-split control.

Encodes the local cluster-head action a_k(t) = (m_k(t), rho_k(t)) and
provides timing / overlap helpers used by the decentralized environment.
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.cluster_config import ClusterConfig as CC


@dataclass(frozen=True)
class DecodedBurstAction:
    action_id: int
    mac_mode: int
    rho_index: int
    rho: float
    t1_time: float
    t2_time: float


def valid_rho_levels() -> tuple[float, ...]:
    """Return discrete rho levels satisfying minimum T1/T2 constraints."""
    levels = []
    for rho in CC.RHO_ACTION_LEVELS:
        t1 = rho * CC.BURST_TOTAL_TIME
        t2 = (1.0 - rho) * CC.BURST_TOTAL_TIME
        if t1 >= CC.T1_MIN and t2 >= CC.T2_MIN:
            levels.append(float(rho))
    if not levels:
        raise ValueError("No valid rho levels satisfy T1/T2 minimum constraints.")
    return tuple(levels)


VALID_RHO_LEVELS = valid_rho_levels()


def decode_action(action_id: int) -> DecodedBurstAction:
    """
    Decode the joint discrete action id into MAC mode and burst split ratio.
    """
    aid = int(action_id) % CC.NUM_ACTIONS
    rho_index = aid % len(VALID_RHO_LEVELS)
    mac_mode = (aid // len(VALID_RHO_LEVELS)) % CC.NUM_MAC_MODES
    rho = VALID_RHO_LEVELS[rho_index]
    t1_time = rho * CC.BURST_TOTAL_TIME
    t2_time = (1.0 - rho) * CC.BURST_TOTAL_TIME
    return DecodedBurstAction(
        action_id=aid,
        mac_mode=mac_mode,
        rho_index=rho_index,
        rho=rho,
        t1_time=t1_time,
        t2_time=t2_time,
    )


def encode_action(mac_mode: int, rho_index: int) -> int:
    mac_id = int(mac_mode) % CC.NUM_MAC_MODES
    rid = int(rho_index) % len(VALID_RHO_LEVELS)
    return mac_id * len(VALID_RHO_LEVELS) + rid


def coordination_overlap(t1_a: float, t1_b: float) -> float:
    """Tail-aligned coordination overlap between two neighboring clusters."""
    if CC.COORD_SYNC_MODE != "tail_aligned":
        raise ValueError(f"Unsupported coordination sync mode: {CC.COORD_SYNC_MODE}")
    overlap = CC.BURST_TOTAL_TIME - max(t1_a, t1_b) - CC.COORD_GUARD_TIME
    return max(0.0, overlap)


def coordination_success_ratio(t1_a: float, t2_a: float, t1_b: float, t2_b: float) -> float:
    """Pairwise coordination success ratio in [0, 1]."""
    overlap = coordination_overlap(t1_a, t1_b)
    denom = max(min(t2_a, t2_b) - CC.COORD_GUARD_TIME, 1e-9)
    return max(0.0, min(overlap / denom, 1.0))
