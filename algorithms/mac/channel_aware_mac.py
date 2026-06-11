# channel_aware_mac.py — Channel-Aware TDMA & CSMA/CA Simulations
# These variants accept precomputed link schedules from mobility traces.
# Used for controlled A/B experiments validating mobility + path-loss impact.
# baseline.py originals are UNTOUCHED — Case A uses those for pure baseline.

import numpy as np
import math

# Re-use Config and Logger from baseline
from algorithms.mac.baseline import Config, Logger


def _slot_to_mob_step(slot_idx, slot_time_s, mobility_dt):
    """Map a MAC slot index to the corresponding mobility time step."""
    t = slot_idx * slot_time_s
    return min(int(t / mobility_dt), 999999)  # capped for safety


# ======================================================================
# TDMA — Channel-Aware
# ======================================================================
def simulate_tdma_aware(cfg, offered_total_pps, log,
                        link_up_schedule, success_prob_schedule=None,
                        mobility_dt=0.1):
    """
    TDMA with link-state awareness.

    Args:
        cfg: Config object (same as baseline).
        offered_total_pps: Total offered traffic in packets/sec.
        log: Logger object.
        link_up_schedule: np.array of shape (N, T_mob_steps), binary 0/1.
        success_prob_schedule: np.array of shape (N, T_mob_steps), float [0,1].
                              None = always succeed when link is up.
        mobility_dt: seconds per mobility step.
    """
    rng = np.random.default_rng(cfg.seed)
    lam_node = offered_total_pps / float(cfg.N)
    if lam_node <= 0:
        return

    N = cfg.N
    q = [[] for _ in range(N)]
    current_q_sum = 0
    channel_busy = 0
    log.total_slots = cfg.total_slots
    max_mob_step = link_up_schedule.shape[1] - 1

    next_arrival = rng.exponential(1.0 / lam_node, size=N)
    next_global_arrival = np.min(next_arrival)
    max_cap = N * cfg.QMAX

    # Counters for link stats
    attempted_pkts = 0
    delivered_pkts = 0
    link_blocked_pkts = 0
    pathloss_failed_pkts = 0

    for s in range(cfg.total_slots):
        if s > 0 and s % cfg.log_interval_slots == 0:
            log.record_queue_time_series(s, max_cap, current_q_sum)

        log.sum_q_len += current_q_sum
        t = s * cfg.slot_time_s
        mob_step = min(int(t / mobility_dt), max_mob_step)

        # 1. Arrivals (always — packets arrive regardless of link)
        if t >= next_global_arrival:
            arriving_nodes = np.where(t >= next_arrival)[0]
            for i in arriving_nodes:
                log.pkts_generated += 1
                if len(q[i]) < cfg.QMAX:
                    q[i].append(t)
                    current_q_sum += 1
                    log.ts_enq += 1
                else:
                    log.pkts_dropped_qfull += 1
                    log.ts_drops += 1
                next_arrival[i] += rng.exponential(1.0 / lam_node)
            next_global_arrival = np.min(next_arrival)

        # 2. Channel busy
        if channel_busy > 0:
            channel_busy -= 1
            log.channel_busy_slots += 1
            continue

        # 3. TDMA slot assignment
        if s % cfg.TDMA_SLOT_TOTAL_SLOTS == 0:
            owner = (s // cfg.TDMA_SLOT_TOTAL_SLOTS) % N
            if len(q[owner]) > 0:
                attempted_pkts += 1
                log.tx_attempts += 1

                # Check link state
                if link_up_schedule[owner, mob_step] == 0:
                    # Link down — slot wasted, packet stays in queue
                    link_blocked_pkts += 1
                    channel_busy = cfg.TDMA_SLOT_TOTAL_SLOTS - 1
                    continue

                # Check path-loss success
                if success_prob_schedule is not None:
                    p_succ = success_prob_schedule[owner, mob_step]
                    if rng.random() > p_succ:
                        # Packet lost to path loss — drop from queue
                        q[owner].pop(0)
                        current_q_sum -= 1
                        log.pkts_dropped_mac += 1
                        log.ts_drops += 1
                        pathloss_failed_pkts += 1
                        channel_busy = cfg.TDMA_SLOT_TOTAL_SLOTS - 1
                        log.channel_busy_slots += cfg.TX_slots
                        continue

                # Success
                arrivalTime = q[owner].pop(0)
                log.pkts_success += 1
                log.payload_bits_tx += cfg.payload_bits
                log.sum_end_to_end_delay_s += ((s + cfg.TX_slots) * cfg.slot_time_s) - arrivalTime
                current_q_sum -= 1
                log.ts_deq += 1
                delivered_pkts += 1

                channel_busy = cfg.TDMA_SLOT_TOTAL_SLOTS - 1
                log.channel_busy_slots += cfg.TX_slots

    # Store link stats on logger
    log._link_stats = {
        "attempted": attempted_pkts,
        "delivered": delivered_pkts,
        "link_blocked": link_blocked_pkts,
        "pathloss_failed": pathloss_failed_pkts,
    }


# ======================================================================
# CSMA/CA — Channel-Aware
# ======================================================================
def simulate_csma_aware(cfg, offered_total_pps, log,
                        link_up_schedule, success_prob_schedule=None,
                        mobility_dt=0.1):
    """
    CSMA/CA with link-state awareness.

    Nodes with link_down skip contention (don't enter backoff).
    If a node wins the channel but path-loss check fails, it's treated
    like a failed transmission → triggers retry/backoff.
    """
    rng = np.random.default_rng(cfg.seed)
    lam_node = offered_total_pps / float(cfg.N)
    if lam_node <= 0:
        return

    N = cfg.N
    q = [[] for _ in range(N)]
    current_q_sum = 0
    cw = np.full(N, cfg.cw_min, dtype=int)
    backoff = np.full(N, -1, dtype=int)
    retry = np.zeros(N, dtype=int)
    channel_busy = 0
    idle_since = 0
    log.total_slots = cfg.total_slots
    max_mob_step = link_up_schedule.shape[1] - 1

    # CSMA durations
    if cfg.rts_cts_enabled and cfg.ack_enabled:
        busy_success = cfg.rts_slots + cfg.sifs_slots + cfg.cts_slots + cfg.sifs_slots + cfg.TX_slots + cfg.sifs_slots + cfg.ack_slots
        busy_collision = cfg.rts_slots + cfg.ack_timeout_slots
    elif cfg.ack_enabled:
        busy_success = cfg.TX_slots + cfg.sifs_slots + cfg.ack_slots
        busy_collision = cfg.TX_slots + cfg.ack_timeout_slots
    else:
        busy_success = cfg.TX_slots
        busy_collision = cfg.TX_slots

    next_arrival = rng.exponential(1.0 / lam_node, size=N)
    next_global_arrival = np.min(next_arrival)
    max_cap = N * cfg.QMAX

    attempted_pkts = 0
    delivered_pkts = 0
    link_blocked_pkts = 0
    pathloss_failed_pkts = 0

    for s in range(cfg.total_slots):
        if s > 0 and s % cfg.log_interval_slots == 0:
            log.record_queue_time_series(s, max_cap, current_q_sum)

        log.sum_q_len += current_q_sum
        t = s * cfg.slot_time_s
        mob_step = min(int(t / mobility_dt), max_mob_step)

        # 1. Arrivals
        if t >= next_global_arrival:
            arriving_nodes = np.where(t >= next_arrival)[0]
            for i in arriving_nodes:
                log.pkts_generated += 1
                if len(q[i]) < cfg.QMAX:
                    q[i].append(t)
                    current_q_sum += 1
                    log.ts_enq += 1
                else:
                    log.pkts_dropped_qfull += 1
                    log.ts_drops += 1
                next_arrival[i] += rng.exponential(1.0 / lam_node)
            next_global_arrival = np.min(next_arrival)

        # 2. Channel busy
        if channel_busy > 0:
            channel_busy -= 1
            log.channel_busy_slots += 1
            if channel_busy == 0:
                idle_since = 0
            continue

        idle_since += 1

        if current_q_sum > 0:
            if idle_since <= cfg.difs_slots:
                continue

            # Only nodes with packets AND link up participate
            has_pkt = np.array([len(q[i]) > 0 for i in range(N)])
            has_link = np.array([link_up_schedule[i, mob_step] == 1 for i in range(N)])
            eligible = has_pkt & has_link

            need_bo = eligible & (backoff < 0)
            if need_bo.any():
                for i in np.where(need_bo)[0]:
                    backoff[i] = rng.integers(0, cw[i] + 1)

            tx_nodes = np.where(eligible & (backoff == 0))[0]
            if tx_nodes.size == 0:
                counting = eligible & (backoff > 0)
                backoff[counting] -= 1
                continue

            idle_since = 0
            log.tx_attempts += tx_nodes.size
            attempted_pkts += tx_nodes.size

            if tx_nodes.size == 1:
                i = tx_nodes[0]

                # Path-loss check
                pl_fail = False
                if success_prob_schedule is not None:
                    p_succ = success_prob_schedule[i, mob_step]
                    if rng.random() > p_succ:
                        pl_fail = True

                if pl_fail:
                    # Path-loss failure — treat like failed TX
                    pathloss_failed_pkts += 1
                    retry[i] += 1
                    if retry[i] > cfg.max_retry:
                        if len(q[i]) > 0:
                            q[i].pop(0)
                            current_q_sum -= 1
                            log.pkts_dropped_mac += 1
                            log.ts_drops += 1
                        retry[i] = 0
                        cw[i] = cfg.cw_min
                    else:
                        cw[i] = min(2 * cw[i] + 1, cfg.cw_max)
                    backoff[i] = -1
                    channel_busy = busy_collision - 1
                    log.channel_busy_slots += 1
                else:
                    # Success
                    arrivalTime = q[i].pop(0)
                    log.pkts_success += 1
                    log.payload_bits_tx += cfg.payload_bits
                    log.sum_end_to_end_delay_s += ((s + busy_success) * cfg.slot_time_s) - arrivalTime
                    current_q_sum -= 1
                    log.ts_deq += 1
                    delivered_pkts += 1
                    cw[i] = cfg.cw_min
                    retry[i] = 0
                    backoff[i] = -1
                    channel_busy = busy_success - 1
                    log.channel_busy_slots += 1
            else:
                # Collision
                log.collision_events += 1
                for i in tx_nodes:
                    retry[i] += 1
                    if retry[i] > cfg.max_retry:
                        if len(q[i]) > 0:
                            q[i].pop(0)
                            current_q_sum -= 1
                            log.pkts_dropped_mac += 1
                            log.ts_drops += 1
                        retry[i] = 0
                        cw[i] = cfg.cw_min
                    else:
                        cw[i] = min(2 * cw[i] + 1, cfg.cw_max)
                    backoff[i] = -1
                channel_busy = busy_collision - 1
                log.channel_busy_slots += 1

    log._link_stats = {
        "attempted": attempted_pkts,
        "delivered": delivered_pkts,
        "link_blocked": link_blocked_pkts,
        "pathloss_failed": pathloss_failed_pkts,
    }
