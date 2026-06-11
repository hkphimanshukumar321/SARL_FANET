"""
marl_mac_env.py -- Decentralized MARL environment for cluster-head control.

Each active cluster-head is a decentralized agent. Inside every burst of
duration t_burst, the agent selects a local MAC mode and a burst split ratio:

    a_k(t) = (m_k(t), rho_k(t))

which induces

    T1_k(t) = rho_k(t) * t_burst
    T2_k(t) = (1 - rho_k(t)) * t_burst

T1 affects intra-cluster service; T2 affects inter-cluster coordination over
the dynamic cluster interference graph.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from algorithms.mac.baseline import Config, Logger, simulate_csma_ca, simulate_tdma
from algorithms.mac.channel_aware_mac import simulate_tdma_aware, simulate_csma_aware
from algorithms.mobility.models import create_mobility_model
from algorithms.mobility.speed import SpeedEngine
from algorithms.rl.rewards import compute_cluster_reward
from configs import config as params
from configs.cluster_config import ClusterConfig as CC
from envs.burst_scheduler import coordination_overlap, coordination_success_ratio, decode_action
from envs.clustering import ClusterManager


try:
    from algorithms.channel.fading import (
        AWGNChannel,
        BERCalculator,
        NakagamiChannel,
        RayleighChannel,
        RicianChannel,
    )

    _HAS_FADING = True
except ImportError:
    _HAS_FADING = False


class MARLMacEnv(ParallelEnv):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, seed=None):
        super().__init__()
        self.N_uavs = params.N

        self.possible_agents = [f"cluster_{k}" for k in range(CC.C_MAX)]
        self.agents = self.possible_agents[:]

        self.action_spaces = {a: spaces.Discrete(CC.NUM_ACTIONS) for a in self.possible_agents}
        self.observation_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(CC.OBS_DIM_CLUSTER,), dtype=np.float32)
            for a in self.possible_agents
        }

        self.seed_val = seed if seed is not None else params.SEED
        self.rng = np.random.default_rng(self.seed_val)
        self.current_step = 0
        self.max_steps = getattr(params, "MAX_STEPS_PER_EP", 200)

        self.speed_engine = None
        self.mobility_model = None
        self.cluster_manager = ClusterManager(self.N_uavs, self.rng)

        self.simulated_queues = np.zeros(self.N_uavs, dtype=np.float64)
        self.recent_collisions = np.zeros(self.N_uavs, dtype=np.float64)
        self.true_edge_index = np.empty((2, 0), dtype=np.int64)
        self.obs_edge_index = np.empty((2, 0), dtype=np.int64)
        self.true_active_cids = []
        self.obs_active_cids = []
        self.obs_adjacency = np.zeros((CC.C_MAX, CC.C_MAX), dtype=bool)
        self.static_obs_adjacency = None

        self.reset_options = {}
        self.failure_schedule = []
        self.failure_events_triggered = []
        self.observation_history = deque(maxlen=64)
        self.graph_history = deque(maxlen=64)
        self.last_step_records = []
        self.last_step_summary = {}
        self.current_offered_pps = getattr(params, "OFFERED_PPS", getattr(params, "SWEEP_MAX_PPS", 400))

        self._init_fading()

    def _init_fading(self):
        if not _HAS_FADING or not getattr(params, "ENABLE_FADING", False):
            self.fading_channel = None
            self.ber_calc = None
            return
        self.ber_calc = BERCalculator(modulation=getattr(params, "MODULATION", "BPSK"))
        model = getattr(params, "FADING_MODEL", "awgn").lower()
        if model == "rayleigh":
            self.fading_channel = RayleighChannel()
        elif model == "rician":
            self.fading_channel = RicianChannel(K=getattr(params, "RICIAN_K", 3.0))
        elif model == "nakagami":
            self.fading_channel = NakagamiChannel(
                m=getattr(params, "NAKAGAMI_M", 2.0),
                omega=getattr(params, "NAKAGAMI_OMEGA", 1.0),
            )
        else:
            self.fading_channel = AWGNChannel()

    def _compute_fading_schedule(self, members: list[int], sim_time: float, cfg) -> tuple[np.ndarray, np.ndarray]:
        """
        Build link_up_schedule and success_prob_schedule arrays for channel-aware MAC.

        Uses current UAV positions, path-loss, fading channel, and BER calculator
        to produce per-node, per-mobility-step packet success probabilities.

        Returns:
            link_up:       (N_cluster, T_mob_steps) binary array (always 1)
            success_prob:  (N_cluster, T_mob_steps) float array in [0, 1]
        """
        n = len(members)
        mobility_dt = getattr(params, "MOBILITY_DT", 0.1)
        t_mob_steps = max(1, int(np.ceil(sim_time / mobility_dt)))

        link_up = np.ones((n, t_mob_steps), dtype=np.int32)
        success_prob = np.ones((n, t_mob_steps), dtype=np.float64)

        if self.fading_channel is None or self.ber_calc is None:
            return link_up, success_prob

        pos = self.mobility_model.positions  # (N_uavs, 3)
        tx_power_dbm = getattr(params, "TX_POWER_DBM", 20.0)
        noise_dbm = getattr(params, "NOISE_POWER_DBM", -80.0)
        eta = getattr(params, "PATHLOSS_ETA", 2.0)
        payload_bits = cfg.payload_bits
        d0 = 1.0  # reference distance in meters
        pl0 = 46.4  # free-space PL at d0=1m, 2.4 GHz

        # Leader is first member (cluster head)
        leader_idx = members[0]
        leader_pos = pos[leader_idx]

        for local_idx, uav_idx in enumerate(members):
            d = np.linalg.norm(pos[uav_idx] - leader_pos)
            d = max(d, d0)  # avoid log(0)

            # Path loss in dB
            pl_db = pl0 + 10.0 * eta * np.log10(d / d0)
            avg_snr_db = tx_power_dbm - pl_db - noise_dbm
            avg_snr_linear = 10.0 ** (avg_snr_db / 10.0)

            for t_step in range(t_mob_steps):
                # Sample fading gain for this link at this time step
                gain = self.fading_channel.sample_gain(1, self.rng)[0]
                inst_snr = avg_snr_linear * gain
                ber = float(self.ber_calc.compute_ber(np.array([inst_snr]))[0])
                # Packet success rate: (1 - BER)^payload_bits
                psr = (1.0 - ber) ** payload_bits if ber < 1.0 else 0.0
                success_prob[local_idx, t_step] = np.clip(psr, 0.0, 1.0)

        return link_up, success_prob

    def _clone_obs_dict(self, obs_dict):
        return {k: np.array(v, dtype=np.float32, copy=True) for k, v in obs_dict.items()}

    def _resolve_option(self, key, default):
        return self.reset_options.get(key, getattr(params, key.upper(), default))

    def _build_topology_positions(self, preset: str) -> np.ndarray:
        if preset == "default":
            return self.mobility_model.positions

        bounds = np.array([params.AREA_X, params.AREA_Y, params.AREA_Z], dtype=np.float64)
        n = self.N_uavs
        positions = np.zeros((n, 3), dtype=np.float64)

        if preset == "compact_dense":
            centers = np.array(
                [
                    [0.25 * bounds[0], 0.30 * bounds[1], 0.50 * bounds[2]],
                    [0.55 * bounds[0], 0.42 * bounds[1], 0.55 * bounds[2]],
                    [0.72 * bounds[0], 0.68 * bounds[1], 0.48 * bounds[2]],
                ],
                dtype=np.float64,
            )
            spread_xy = 0.08 * min(bounds[0], bounds[1])
            spread_z = 0.08 * bounds[2]
        elif preset == "sparse_separated":
            centers = np.array(
                [
                    [0.15 * bounds[0], 0.20 * bounds[1], 0.35 * bounds[2]],
                    [0.80 * bounds[0], 0.18 * bounds[1], 0.70 * bounds[2]],
                    [0.22 * bounds[0], 0.82 * bounds[1], 0.55 * bounds[2]],
                    [0.80 * bounds[0], 0.78 * bounds[1], 0.45 * bounds[2]],
                ],
                dtype=np.float64,
            )
            spread_xy = 0.04 * min(bounds[0], bounds[1])
            spread_z = 0.05 * bounds[2]
        elif preset == "asymmetric_hotspot":
            centers = np.array(
                [
                    [0.30 * bounds[0], 0.35 * bounds[1], 0.50 * bounds[2]],
                    [0.78 * bounds[0], 0.25 * bounds[1], 0.65 * bounds[2]],
                    [0.70 * bounds[0], 0.78 * bounds[1], 0.40 * bounds[2]],
                ],
                dtype=np.float64,
            )
            spread_xy = 0.06 * min(bounds[0], bounds[1])
            spread_z = 0.06 * bounds[2]
        else:
            raise ValueError(f"Unknown topology preset: {preset}")

        if preset == "asymmetric_hotspot":
            counts = [max(1, int(round(0.6 * n))), max(1, int(round(0.2 * n)))]
            counts.append(max(0, n - sum(counts)))
        else:
            base = n // len(centers)
            counts = [base] * len(centers)
            for idx in range(n - base * len(centers)):
                counts[idx] += 1

        cursor = 0
        for center, count in zip(centers, counts):
            if count <= 0:
                continue
            noise = np.column_stack(
                [
                    self.rng.normal(0.0, spread_xy, size=count),
                    self.rng.normal(0.0, spread_xy, size=count),
                    self.rng.normal(0.0, spread_z, size=count),
                ]
            )
            positions[cursor:cursor + count] = center + noise
            cursor += count

        positions = np.clip(positions, [0.0, 0.0, 0.0], bounds)
        return positions

    def _apply_reset_scenario(self):
        topology_preset = self.reset_options.get("topology_preset", getattr(params, "TOPOLOGY_PRESET", "default"))
        speed_scale = float(self.reset_options.get("speed_scale", getattr(params, "SPEED_SCALE", 1.0)))

        if topology_preset != "default":
            self.mobility_model.positions = self._build_topology_positions(topology_preset)

        if "positions" in self.reset_options:
            self.mobility_model.positions = np.asarray(self.reset_options["positions"], dtype=np.float64).copy()

        if "velocities" in self.reset_options:
            self.mobility_model.velocities = np.asarray(self.reset_options["velocities"], dtype=np.float64).copy()
        elif speed_scale != 1.0:
            self.mobility_model.velocities = self.mobility_model.velocities * speed_scale

        if "offered_pps" in self.reset_options:
            self.current_offered_pps = float(self.reset_options["offered_pps"])

    def _interference_scale(self) -> float:
        return float(self.reset_options.get("interference_scale", getattr(params, "INTERFERENCE_SCALE", 1.0)))

    def _coordination_capacity_scale(self) -> float:
        return float(self.reset_options.get("coordination_capacity_scale", getattr(params, "COORDINATION_CAPACITY_SCALE", 1.0)))

    def _sample_offered_pps(self, base_pps: float) -> float:
        traffic_profile = self.reset_options.get("traffic_profile", getattr(params, "TRAFFIC_PROFILE", "smooth"))
        base_pps = float(base_pps)
        if traffic_profile == "smooth":
            return base_pps
        if traffic_profile == "bursty_on_off":
            on_prob = float(self.reset_options.get("traffic_burst_on_prob", getattr(params, "TRAFFIC_BURST_ON_PROB", 0.30)))
            burst_mult = float(self.reset_options.get("traffic_burst_multiplier", getattr(params, "TRAFFIC_BURST_MULTIPLIER", 2.0)))
            return base_pps * (burst_mult if self.rng.random() < on_prob else 0.35)
        if traffic_profile == "heavy_tail":
            shape = float(self.reset_options.get("traffic_heavy_tail_shape", getattr(params, "TRAFFIC_HEAVY_TAIL_SHAPE", 1.8)))
            scale = float(self.reset_options.get("traffic_heavy_tail_scale", getattr(params, "TRAFFIC_HEAVY_TAIL_SCALE", 1.0)))
            pareto = (self.rng.pareto(shape) + 1.0) * scale
            return np.clip(base_pps * pareto, 0.2 * base_pps, 4.0 * base_pps)
        raise ValueError(f"Unknown traffic profile: {traffic_profile}")

    def _edge_index_from_adjacency(self, adjacency: np.ndarray, active_cids: list[int]) -> np.ndarray:
        cid_to_local = {cid: idx for idx, cid in enumerate(active_cids)}
        rows, cols = [], []
        for i_idx, cid_a in enumerate(active_cids):
            for j_idx in range(i_idx + 1, len(active_cids)):
                cid_b = active_cids[j_idx]
                if adjacency[cid_a % CC.C_MAX, cid_b % CC.C_MAX]:
                    li = cid_to_local[cid_a]
                    lj = cid_to_local[cid_b]
                    rows.extend([li, lj])
                    cols.extend([lj, li])
        if rows:
            return np.array([rows, cols], dtype=np.int64)
        return np.empty((2, 0), dtype=np.int64)

    def _shuffle_adjacency(self, adjacency: np.ndarray, active_cids: list[int]) -> np.ndarray:
        shuffled = np.zeros_like(adjacency, dtype=bool)
        if len(active_cids) <= 1:
            return shuffled
        perm = list(active_cids)
        self.rng.shuffle(perm)
        edge_pairs = []
        for i_idx, cid_a in enumerate(active_cids):
            for j_idx in range(i_idx + 1, len(active_cids)):
                cid_b = active_cids[j_idx]
                if adjacency[cid_a % CC.C_MAX, cid_b % CC.C_MAX]:
                    edge_pairs.append((perm[i_idx], perm[j_idx]))
        for u, v in edge_pairs:
            if u == v:
                continue
            shuffled[u % CC.C_MAX, v % CC.C_MAX] = True
            shuffled[v % CC.C_MAX, u % CC.C_MAX] = True
        return shuffled

    def _corrupt_adjacency(self, adjacency: np.ndarray, active_cids: list[int]) -> np.ndarray:
        graph_mode = self.reset_options.get("graph_mode", getattr(params, "GRAPH_MODE", "dynamic"))
        if graph_mode == "none":
            return np.zeros_like(adjacency, dtype=bool)
        if graph_mode == "static" and self.static_obs_adjacency is not None:
            working = self.static_obs_adjacency.copy()
        elif graph_mode == "shuffled":
            working = self._shuffle_adjacency(adjacency, active_cids)
        else:
            working = adjacency.copy()

        miss_prob = float(self.reset_options.get("graph_missing_edge_prob", getattr(params, "GRAPH_MISSING_EDGE_PROB", 0.0)))
        false_prob = float(self.reset_options.get("graph_false_edge_prob", getattr(params, "GRAPH_FALSE_EDGE_PROB", 0.0)))
        for i_idx, cid_a in enumerate(active_cids):
            for j_idx in range(i_idx + 1, len(active_cids)):
                cid_b = active_cids[j_idx]
                has_edge = working[cid_a % CC.C_MAX, cid_b % CC.C_MAX]
                if has_edge and miss_prob > 0.0 and self.rng.random() < miss_prob:
                    working[cid_a % CC.C_MAX, cid_b % CC.C_MAX] = False
                    working[cid_b % CC.C_MAX, cid_a % CC.C_MAX] = False
                elif (not has_edge) and false_prob > 0.0 and self.rng.random() < false_prob:
                    working[cid_a % CC.C_MAX, cid_b % CC.C_MAX] = True
                    working[cid_b % CC.C_MAX, cid_a % CC.C_MAX] = True
        return working

    def _refresh_graph_views(self):
        self.true_edge_index, self.true_active_cids = self.cluster_manager.get_interference_graph()
        base_adj = self.cluster_manager._adj.copy()
        obs_adj = self._corrupt_adjacency(base_adj, self.true_active_cids)
        self.obs_adjacency = obs_adj
        self.obs_edge_index = self._edge_index_from_adjacency(obs_adj, self.true_active_cids)
        self.obs_active_cids = list(self.true_active_cids)
        self.graph_history.append(self.obs_edge_index.copy())

    def _build_clean_observations(self):
        pos = self.mobility_model.positions
        vel = self.mobility_model.velocities
        obs = {}
        for k in range(CC.C_MAX):
            cluster_obs = self.cluster_manager.get_cluster_obs(
                k,
                pos,
                vel,
                self.simulated_queues,
                self.recent_collisions,
                adjacency_override=self.obs_adjacency,
            )
            if not getattr(self, "use_burst_history", True):
                cluster_obs[21:24] = 0.0
            obs[f"cluster_{k}"] = cluster_obs
        return obs

    def _apply_observation_effects(self, clean_obs, mutate_history: bool = True):
        if mutate_history:
            self.observation_history.append(self._clone_obs_dict(clean_obs))
        obs_staleness = int(self.reset_options.get("obs_staleness_steps", getattr(params, "OBS_STALENESS_STEPS", 0)))
        handover_staleness = int(
            self.reset_options.get(
                "handover_info_staleness_steps",
                getattr(params, "HANDOVER_INFO_STALENESS_STEPS", 0),
            )
        )
        noise_std = float(self.reset_options.get("obs_noise_std", getattr(params, "OBS_NOISE_STD", 0.0)))

        source_obs = clean_obs
        if obs_staleness > 0 and len(self.observation_history) > obs_staleness:
            source_obs = self.observation_history[-(obs_staleness + 1)]

        obs = self._clone_obs_dict(source_obs)
        if handover_staleness > 0 and len(self.observation_history) > handover_staleness:
            handover_src = self.observation_history[-(handover_staleness + 1)]
            for name in obs:
                obs[name][14:16] = handover_src[name][14:16]

        if noise_std > 0.0:
            for name in obs:
                obs[name] = obs[name] + self.rng.normal(0.0, noise_std, size=obs[name].shape).astype(np.float32)
        return obs

    def _resolve_graph_for_policy(self):
        graph_staleness = int(self.reset_options.get("graph_staleness_steps", getattr(params, "GRAPH_STALENESS_STEPS", 0)))
        if graph_staleness > 0 and len(self.graph_history) > graph_staleness:
            return self.graph_history[-(graph_staleness + 1)]
        return self.obs_edge_index

    def _resolve_failure_target(self, event, active_cids):
        if not active_cids:
            return None
        if "cluster_id" in event and event["cluster_id"] in active_cids:
            return int(event["cluster_id"])
        target = event.get("target", "random")
        if target == "random":
            return int(self.rng.choice(active_cids))
        if target == "max_backlog":
            return max(active_cids, key=lambda cid: float(self.simulated_queues[self.cluster_manager.get_members(cid)].sum()))
        if target == "max_degree":
            degree_map = self._degree_map(self.true_active_cids, self.true_edge_index)
            return max(active_cids, key=lambda cid: degree_map.get(cid, 0))
        return active_cids[0]

    def _apply_failure_schedule(self):
        if not self.failure_schedule:
            return []
        triggered = []
        active_cids = list(self.cluster_manager.get_active_cluster_ids())
        remaining = []
        for event in self.failure_schedule:
            if int(event.get("step", -1)) != self.current_step:
                remaining.append(event)
                continue
            target_cid = self._resolve_failure_target(event, active_cids)
            if target_cid is None:
                continue
            self.cluster_manager.trigger_leader_failure(
                target_cid,
                self.mobility_model.positions,
                self.mobility_model.velocities,
            )
            triggered.append({"step": self.current_step, "cluster_id": int(target_cid), "mode": event.get("target", "explicit")})
        self.failure_schedule = remaining
        self.failure_events_triggered = triggered
        return triggered

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(self.seed_val)
        self.reset_options = dict(options or {})
        self.failure_schedule = [dict(event) for event in self.reset_options.get("failure_schedule", getattr(params, "FAILURE_SCHEDULE", ()))]
        self.failure_events_triggered = []
        self.observation_history.clear()
        self.graph_history.clear()
        self.last_step_records = []
        self.last_step_summary = {}
        self.use_burst_history = bool(self.reset_options.get("use_burst_history", True))

        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.simulated_queues[:] = 0.0
        self.recent_collisions[:] = 0.0

        speed_scale = float(self.reset_options.get("speed_scale", getattr(params, "SPEED_SCALE", 1.0)))
        v_min = params.V_MIN * speed_scale
        v_max = params.V_MAX * speed_scale
        v_mean = params.V_MEAN * speed_scale
        v_std = params.V_STD * speed_scale

        self.speed_engine = SpeedEngine(
            n_nodes=self.N_uavs,
            v_min=v_min,
            v_max=v_max,
            mode=params.SPEED_MODE,
            v_mean=v_mean,
            v_std=v_std,
            update_interval=params.SPEED_UPDATE_INTERVAL,
            rng=self.rng,
        )
        self.mobility_model = create_mobility_model(
            name=self.reset_options.get("mobility_model", params.MOBILITY_MODEL),
            n_nodes=self.N_uavs,
            bounds=(params.AREA_X, params.AREA_Y, params.AREA_Z),
            speed_engine=self.speed_engine,
            rng=self.rng,
            gm_alpha=params.GM_ALPHA,
        )

        self.current_offered_pps = float(self.reset_options.get("offered_pps", getattr(params, "OFFERED_PPS", getattr(params, "SWEEP_MAX_PPS", 400))))
        self._apply_reset_scenario()
        pos = self.mobility_model.positions
        vel = self.mobility_model.velocities
        original_r_i = CC.R_I
        CC.R_I = original_r_i * self._interference_scale()
        try:
            self.cluster_manager.reset(pos, vel)
            self.cluster_manager._build_interference_graph(pos)
        finally:
            CC.R_I = original_r_i
        self.static_obs_adjacency = self.cluster_manager._adj.copy()
        self._refresh_graph_views()

        obs = self._get_observations()
        infos = {a: {"alive": (a in self._get_active_agent_names())} for a in self.possible_agents}
        return obs, infos

    def _get_active_agent_names(self):
        return [f"cluster_{cid}" for cid in self.cluster_manager.get_active_cluster_ids()]

    def _get_observations(self, mutate_history: bool = True):
        clean_obs = self._build_clean_observations()
        return self._apply_observation_effects(clean_obs, mutate_history=mutate_history)

    def _degree_map(self, active_cids: list[int], edge_index: np.ndarray) -> dict[int, int]:
        degree = {cid: 0 for cid in active_cids}
        for src_local, dst_local in edge_index.T:
            if src_local >= len(active_cids) or dst_local >= len(active_cids):
                continue
            degree[active_cids[src_local]] += 1
        return degree

    def _prepare_cluster_demands(self, cid: int, members: list[int], degree: int) -> tuple[float, float, float]:
        cs = self.cluster_manager.clusters[cid]
        subordinate_count = max(len(members) - 1, 0)
        queue_norm = min(float(self.simulated_queues[members].sum()) / max(len(members) * params.QMAX, 1), 1.0)
        local_ctrl_demand = min(
            cs.local_ctrl_demand
            + CC.LOCAL_CTRL_DEMAND_INCREMENT * max(subordinate_count, 1)
            + 0.2 * queue_norm,
            CC.MAX_LOCAL_CTRL_DEMAND,
        )
        coord_backlog = min(
            cs.coord_backlog
            + CC.COORD_BACKLOG_INCREMENT * degree
            + 0.25 * queue_norm
            + (0.4 if cs.handover_flag else 0.0),
            CC.MAX_COORD_BACKLOG,
        )
        relay_demand = min(
            cs.relay_demand
            + CC.RELAY_DEMAND_INCREMENT * max(degree - 1, 0)
            + (0.3 if cs.handover_flag else 0.0),
            CC.MAX_RELAY_DEMAND,
        )
        return local_ctrl_demand, coord_backlog, relay_demand

    def _build_subenv_config(self, n_in_cluster: int, sim_time_s: float, cid: int) -> Config:
        return Config(
            N=n_in_cluster,
            sim_time_s=max(sim_time_s, 1e-6),
            slot_time_s=params.SLOT_TIME_S,
            phy_rate_bps=params.PHY_RATE_BPS,
            payload_bytes=params.PAYLOAD_BYTES,
            QMAX=params.QMAX,
            seed=self.seed_val + self.current_step + cid,
            cw_min=params.CW_MIN,
            cw_max=params.CW_MAX,
            difs_slots=params.DIFS_SLOTS,
            sifs_slots=params.SIFS_SLOTS,
            ack_slots=params.ACK_SLOTS,
            ack_timeout_slots=params.ACK_TIMEOUT_SLOTS,
            max_retry=params.MAX_RETRY,
            log_interval_slots=params.LOG_INTERVAL_SLOTS,
            rts_cts_enabled=params.RTS_CTS_ENABLED,
            ack_enabled=params.ACK_ENABLED,
            rts_slots=params.RTS_SLOTS,
            cts_slots=params.CTS_SLOTS,
            tdma_guard_time_s=params.TDMA_GUARD_TIME_S,
        )

    def step(self, actions):
        # One MARL step corresponds to one burst, so mobility should advance by burst time.
        self.mobility_model.update(CC.BURST_TOTAL_TIME)
        pos = self.mobility_model.positions
        vel = self.mobility_model.velocities

        original_r_i = CC.R_I
        CC.R_I = original_r_i * self._interference_scale()
        try:
            self.cluster_manager.update(pos, vel, self.simulated_queues, self.current_step)
        finally:
            CC.R_I = original_r_i
        self._apply_failure_schedule()
        original_r_i = CC.R_I
        CC.R_I = original_r_i * self._interference_scale()
        try:
            self.cluster_manager._build_interference_graph(pos)
        finally:
            CC.R_I = original_r_i
        self._refresh_graph_views()
        active_cids = list(self.true_active_cids)
        active_set = set(active_cids)
        degree_map = self._degree_map(active_cids, self.true_edge_index)

        base_pps = self.reset_options.get("offered_pps", getattr(params, "OFFERED_PPS", getattr(params, "SWEEP_MAX_PPS", 400)))
        offered_pps = self._sample_offered_pps(base_pps)
        self.current_offered_pps = float(offered_pps)
        self.recent_collisions[:] = 0.0

        cluster_logs = {cid: None for cid in range(CC.C_MAX)}
        action_decisions = {}
        metrics = {}

        for cid in range(CC.C_MAX):
            agent_name = f"cluster_{cid}"
            members = self.cluster_manager.get_members(cid)
            if cid not in active_set or not members:
                metrics[cid] = {
                    "local_bits": 0.0,
                    "local_delay_ms": 0.0,
                    "local_drops": 0,
                    "queue_overflow_ratio": 0.0,
                    "local_ctrl_demand": 0.0,
                    "coord_backlog": 0.0,
                    "relay_demand": 0.0,
                    "served_local_ctrl": 0.0,
                    "t1_util": 0.0,
                    "inter_bits": 0.0,
                    "inter_delay_ms": 0.0,
                    "coord_success_ratio": 0.0,
                    "coord_failure_ratio": 0.0,
                    "action": decode_action(0),
                }
                continue

            cs = self.cluster_manager.clusters[cid]
            degree = degree_map.get(cid, 0)
            local_ctrl_demand, coord_backlog, relay_demand = self._prepare_cluster_demands(cid, members, degree)

            decoded = decode_action(actions.get(agent_name, 0))
            action_decisions[cid] = decoded

            local_ctrl_capacity = decoded.t1_time * CC.LOCAL_CTRL_CAPACITY_PER_SEC
            served_local_ctrl = min(local_ctrl_demand, local_ctrl_capacity)
            remaining_local_ctrl = max(local_ctrl_demand - served_local_ctrl, 0.0)

            n_in_cluster = len(members)
            cfg = self._build_subenv_config(n_in_cluster, decoded.t1_time, cid)
            log = Logger(load_pps=offered_pps, protocol_name="TDMA" if decoded.mac_mode == 0 else "CSMA_CA")
            sub_load = offered_pps * (n_in_cluster / max(self.N_uavs, 1))

            if self.fading_channel is not None and self.ber_calc is not None:
                link_up, success_prob = self._compute_fading_schedule(members, decoded.t1_time, cfg)
                mob_dt = getattr(params, "MOBILITY_DT", 0.1)
                if decoded.mac_mode == 0:
                    simulate_tdma_aware(cfg, sub_load, log, link_up, success_prob, mobility_dt=mob_dt)
                else:
                    simulate_csma_aware(cfg, sub_load, log, link_up, success_prob, mobility_dt=mob_dt)
            else:
                if decoded.mac_mode == 0:
                    simulate_tdma(cfg, sub_load, log)
                else:
                    simulate_csma_ca(cfg, sub_load, log)

            cluster_logs[cid] = log

            local_bits = float(log.payload_bits_tx)
            local_delay_ms = float(log.get_avg_end_to_end_delay_s() * 1000.0)
            local_drops = int(log.pkts_dropped_qfull + log.pkts_dropped_mac)
            arrivals_total = float(sub_load * decoded.t1_time)
            served_total = float(log.pkts_success)
            per_member_arrivals = arrivals_total / max(n_in_cluster, 1)
            per_member_service = served_total / max(n_in_cluster, 1)
            control_backpressure = remaining_local_ctrl / max(n_in_cluster, 1)

            overflow_events = 0
            avg_collision_ratio = log.collision_events / max(log.tx_attempts, 1)
            for m in members:
                new_q = self.simulated_queues[m] + per_member_arrivals - per_member_service
                if m != cs.leader_idx:
                    new_q += control_backpressure
                if new_q > params.QMAX:
                    overflow_events += 1
                self.simulated_queues[m] = np.clip(new_q, 0.0, float(params.QMAX))
                self.recent_collisions[m] = avg_collision_ratio

            queue_overflow_ratio = overflow_events / max(n_in_cluster, 1)
            demand_total = arrivals_total + local_ctrl_demand
            served_total_with_ctrl = served_total + served_local_ctrl
            t1_util = min(served_total_with_ctrl / max(demand_total, 1e-9), 1.0) if demand_total > 0 else 0.0

            agg_queue = float(self.simulated_queues[members].sum())
            local_effective_bps = local_bits / CC.BURST_TOTAL_TIME
            self.cluster_manager.update_cluster_runtime_state(
                cid,
                agg_queue=agg_queue,
                agg_throughput=local_effective_bps,
                agg_delay=local_delay_ms,
                agg_collisions=log.collision_events,
                agg_drops=local_drops,
                local_ctrl_demand=remaining_local_ctrl,
                coord_backlog=coord_backlog,
                relay_demand=relay_demand,
                recent_rho=decoded.rho,
                recent_t1_util=t1_util,
            )

            metrics[cid] = {
                "local_bits": local_bits,
                "local_delay_ms": local_delay_ms,
                "local_drops": local_drops,
                "queue_overflow_ratio": queue_overflow_ratio,
                "local_ctrl_demand": remaining_local_ctrl,
                "coord_backlog": coord_backlog,
                "relay_demand": relay_demand,
                "served_local_ctrl": served_local_ctrl,
                "t1_util": t1_util,
                "inter_bits": 0.0,
                "inter_delay_ms": 0.0,
                "coord_success_ratio": 0.0,
                "coord_failure_ratio": 0.0,
                "action": decoded,
            }

        interference_penalties = defaultdict(float)
        edge_seen = set()
        coord_success_sums = defaultdict(float)
        coord_counts = defaultdict(int)
        t2_capacity_sums = defaultdict(float)
        t2_served_sums = defaultdict(float)

        for src_local, dst_local in self.true_edge_index.T:
            if src_local >= len(active_cids) or dst_local >= len(active_cids):
                continue
            u = active_cids[src_local]
            v = active_cids[dst_local]
            pair = tuple(sorted((u, v)))
            if u == v or pair in edge_seen:
                continue
            edge_seen.add(pair)

            action_u = metrics[u]["action"]
            action_v = metrics[v]["action"]
            overlap = coordination_overlap(action_u.t1_time, action_v.t1_time)
            success = coordination_success_ratio(
                action_u.t1_time,
                action_u.t2_time,
                action_v.t1_time,
                action_v.t2_time,
            )

            deg_u = max(degree_map.get(u, 0), 1)
            deg_v = max(degree_map.get(v, 0), 1)
            demand_u = metrics[u]["coord_backlog"] / deg_u + 0.5 * metrics[u]["relay_demand"] / deg_u
            demand_v = metrics[v]["coord_backlog"] / deg_v + 0.5 * metrics[v]["relay_demand"] / deg_v
            coord_scale = self._coordination_capacity_scale()
            coord_capacity = overlap * CC.COORD_CAPACITY_PER_SEC * coord_scale
            relay_capacity = overlap * CC.RELAY_SERVICE_CAPACITY_PER_SEC * coord_scale
            served_u = min(demand_u, coord_capacity + relay_capacity)
            served_v = min(demand_v, coord_capacity + relay_capacity)

            coord_bits_u = served_u * params.PAYLOAD_BYTES * 8 * 0.25
            coord_bits_v = served_v * params.PAYLOAD_BYTES * 8 * 0.25
            metrics[u]["inter_bits"] += coord_bits_u
            metrics[v]["inter_bits"] += coord_bits_v

            metrics[u]["coord_backlog"] = max(metrics[u]["coord_backlog"] - served_u, 0.0)
            metrics[v]["coord_backlog"] = max(metrics[v]["coord_backlog"] - served_v, 0.0)
            metrics[u]["relay_demand"] = max(metrics[u]["relay_demand"] - min(relay_capacity, served_u), 0.0)
            metrics[v]["relay_demand"] = max(metrics[v]["relay_demand"] - min(relay_capacity, served_v), 0.0)

            delay_penalty_ms = (1.0 - success) * 0.5 * CC.MAX_DELAY_MS
            if overlap <= 0.0 and (demand_u > 0.0 or demand_v > 0.0):
                delay_penalty_ms += 0.25 * CC.MAX_DELAY_MS
            metrics[u]["inter_delay_ms"] += delay_penalty_ms
            metrics[v]["inter_delay_ms"] += delay_penalty_ms

            coord_success_sums[u] += success
            coord_success_sums[v] += success
            coord_counts[u] += 1
            coord_counts[v] += 1
            t2_capacity_sums[u] += max(coord_capacity + relay_capacity, 0.0)
            t2_capacity_sums[v] += max(coord_capacity + relay_capacity, 0.0)
            t2_served_sums[u] += served_u
            t2_served_sums[v] += served_v

            if action_u.mac_mode == 1 and action_v.mac_mode == 1:
                base_penalty = 0.15
            elif action_u.mac_mode == action_v.mac_mode:
                base_penalty = 0.06
            else:
                base_penalty = 0.02
            penalty = base_penalty * (1.25 - success)
            interference_penalties[u] += penalty
            interference_penalties[v] += penalty

        rewards = {}
        throughput_per_cluster = {}
        infos = {}

        for cid in range(CC.C_MAX):
            agent_name = f"cluster_{cid}"
            if cid not in active_set or cluster_logs[cid] is None:
                rewards[agent_name] = 0.0
                infos[agent_name] = {"alive": False, "throughput_mbps": 0.0, "global_th": 0.0}
                continue

            cs = self.cluster_manager.clusters[cid]
            log = cluster_logs[cid]
            m = metrics[cid]
            action = m["action"]

            coord_success = (
                coord_success_sums[cid] / coord_counts[cid] if coord_counts[cid] > 0 else 0.0
            )
            coord_failure_ratio = (
                max(0.0, 1.0 - coord_success)
                if (m["coord_backlog"] > 0.0 or m["relay_demand"] > 0.0 or degree_map.get(cid, 0) > 0)
                else 0.0
            )
            t2_util = (
                min(t2_served_sums[cid] / max(t2_capacity_sums[cid], 1e-9), 1.0)
                if t2_capacity_sums[cid] > 0
                else 0.0
            )
            self.cluster_manager.update_cluster_runtime_state(
                cid,
                coord_backlog=m["coord_backlog"],
                relay_demand=m["relay_demand"],
                recent_t2_util=t2_util,
                recent_coord_success=coord_success,
            )

            local_effective_bps = m["local_bits"] / CC.BURST_TOTAL_TIME
            inter_effective_bps = m["inter_bits"] / CC.BURST_TOTAL_TIME
            throughput_per_cluster[agent_name] = local_effective_bps

            energy_cost = (
                log.tx_attempts * CC.E_TX_COST
                + len(cs.member_indices) * CC.E_IDLE_COST
                + 0.01 * t2_served_sums[cid]
            )

            reward = compute_cluster_reward(
                local_throughput_bps=local_effective_bps,
                inter_throughput_bps=inter_effective_bps,
                local_delay_ms=m["local_delay_ms"],
                inter_delay_ms=m["inter_delay_ms"],
                collisions=log.collision_events,
                energy_cost=energy_cost,
                interference_penalty=interference_penalties[cid],
                coordination_failure_ratio=coord_failure_ratio,
                queue_overflow_ratio=m["queue_overflow_ratio"],
                handover_flag=cs.handover_flag,
            )
            rewards[agent_name] = reward

            infos[agent_name] = {
                "alive": True,
                "action": action.action_id,
                "chosen_mac": action.mac_mode,
                "rho": action.rho,
                "t1_time": action.t1_time,
                "t2_time": action.t2_time,
                "throughput": local_effective_bps / 1e6,
                "throughput_mbps": local_effective_bps / 1e6,
                "inter_throughput_mbps": inter_effective_bps / 1e6,
                "delay_ms": m["local_delay_ms"],
                "inter_delay_ms": m["inter_delay_ms"],
                "drops": m["local_drops"],
                "collisions": log.collision_events,
                "inter_coord_success": coord_success,
                "queue_overflow_ratio": m["queue_overflow_ratio"],
                "coord_backlog": m["coord_backlog"],
                "cluster_size": len(cs.member_indices),
                "graph_degree": degree_map.get(cid, 0),
                "local_backlog": float(self.simulated_queues[cs.member_indices].sum()),
                "inter_backlog": m["coord_backlog"] + m["relay_demand"],
                "leader_health": self.cluster_manager._leader_health(
                    cs.leader_idx,
                    cs,
                    pos,
                    vel,
                    self.simulated_queues,
                ),
                "handover_flag": float(cs.handover_flag),
                "traffic_profile": self.reset_options.get("traffic_profile", getattr(params, "TRAFFIC_PROFILE", "smooth")),
            }

        global_th = sum(throughput_per_cluster.values()) / 1e6
        global_drops = sum(metrics[cid]["local_drops"] for cid in active_set)
        global_collisions = sum(
            cluster_logs[cid].collision_events for cid in active_set if cluster_logs[cid] is not None
        )
        active_coord_success = [
            infos[f"cluster_{cid}"].get("inter_coord_success", 0.0) for cid in active_set
        ]
        global_coord_success = float(np.mean(active_coord_success)) if active_coord_success else 0.0

        for agent_name, info in infos.items():
            info["global_th"] = global_th
            info["global_drops"] = global_drops
            info["global_collisions"] = global_collisions
            info["global_coord_success"] = global_coord_success
            info["current_offered_pps"] = float(self.current_offered_pps)

        diagnostics = self.cluster_manager.get_diagnostics()
        summary = {
            "step": int(self.current_step),
            "offered_pps": float(self.current_offered_pps),
            "throughput_mbps": float(global_th),
            "drops": float(global_drops),
            "collisions": float(global_collisions),
            "coord_success": float(global_coord_success),
            "num_clusters": int(diagnostics["num_clusters"]),
            "avg_cluster_size": float(diagnostics["mean_cluster_size"]),
            "cluster_size_std": float(diagnostics["cluster_size_std"]),
            "avg_graph_degree": float(diagnostics["avg_graph_degree"]),
            "graph_density": float(diagnostics["graph_density"]),
            "reassociations": int(diagnostics["reassociations"]),
            "splits": int(diagnostics["splits"]),
            "merges": int(diagnostics["merges"]),
            "handovers": int(diagnostics["handovers"]),
            "failure_events": len(self.failure_events_triggered),
        }

        step_records = []
        for cid in active_cids:
            agent_name = f"cluster_{cid}"
            info = infos.get(agent_name, {})
            step_records.append(
                {
                    "step": int(self.current_step),
                    "cluster_id": int(cid),
                    "alive": bool(info.get("alive", False)),
                    "chosen_mac": int(info.get("chosen_mac", 0)),
                    "rho": float(info.get("rho", CC.DEFAULT_RHO)),
                    "t1_time": float(info.get("t1_time", 0.0)),
                    "t2_time": float(info.get("t2_time", 0.0)),
                    "throughput_mbps": float(info.get("throughput_mbps", 0.0)),
                    "inter_throughput_mbps": float(info.get("inter_throughput_mbps", 0.0)),
                    "delay_ms": float(info.get("delay_ms", 0.0)),
                    "inter_delay_ms": float(info.get("inter_delay_ms", 0.0)),
                    "drops": float(info.get("drops", 0.0)),
                    "collisions": float(info.get("collisions", 0.0)),
                    "coord_success": float(info.get("inter_coord_success", 0.0)),
                    "cluster_size": int(info.get("cluster_size", 0)),
                    "graph_degree": int(info.get("graph_degree", 0)),
                    "local_backlog": float(info.get("local_backlog", 0.0)),
                    "inter_backlog": float(info.get("inter_backlog", 0.0)),
                    "leader_health": float(info.get("leader_health", 0.0)),
                    "handover_flag": float(info.get("handover_flag", 0.0)),
                    "traffic_profile": info.get("traffic_profile", "smooth"),
                    "offered_pps": float(self.current_offered_pps),
                    "failure_event": int(any(event["cluster_id"] == cid for event in self.failure_events_triggered)),
                    "num_clusters": int(diagnostics["num_clusters"]),
                    "avg_graph_degree": float(diagnostics["avg_graph_degree"]),
                    "graph_density": float(diagnostics["graph_density"]),
                }
            )

        self.last_step_records = step_records
        self.last_step_summary = summary

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncations = {a: False for a in self.possible_agents}
        terminations = {a: terminated for a in self.possible_agents}
        if terminated:
            self.agents = []

        obs = self._get_observations()
        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_global_graph_state(self):
        """
        Returns:
            node_features: (C_MAX, OBS_DIM_CLUSTER)
            edge_index:    (2, E) in global cluster-id space
            alive_mask:    (C_MAX,)
        """
        obs_dict = self._get_observations(mutate_history=False)
        x = np.stack([obs_dict[a] for a in self.possible_agents])

        active_cids = set(self.cluster_manager.get_active_cluster_ids())
        alive_mask = np.array([1 if k in active_cids else 0 for k in range(CC.C_MAX)], dtype=np.bool_)

        mapped_rows = []
        mapped_cols = []
        policy_edge_index = self._resolve_graph_for_policy()
        if policy_edge_index.shape[1] > 0 and active_cids:
            active_list = self.obs_active_cids if self.obs_active_cids else self.cluster_manager.get_active_cluster_ids()
            for col_idx in range(policy_edge_index.shape[1]):
                u_local = policy_edge_index[0, col_idx]
                v_local = policy_edge_index[1, col_idx]
                if u_local < len(active_list) and v_local < len(active_list):
                    mapped_rows.append(active_list[u_local])
                    mapped_cols.append(active_list[v_local])

        if mapped_rows:
            env_edge_index = np.stack([mapped_rows, mapped_cols], axis=0).astype(np.int64)
        else:
            env_edge_index = np.empty((2, 0), dtype=np.int64)

        return x, env_edge_index, alive_mask

    def get_last_step_records(self):
        return [dict(record) for record in self.last_step_records]

    def get_last_step_summary(self):
        return dict(self.last_step_summary)

    def estimate_neighbor_summary_overhead_bytes(self) -> float:
        summary_scalars = CC.NEIGHBOR_SUMMARY_DIM + 5
        diagnostics = self.cluster_manager.get_diagnostics()
        avg_degree = diagnostics.get("avg_graph_degree", 0.0)
        num_clusters = diagnostics.get("num_clusters", 0)
        return float(num_clusters * avg_degree * summary_scalars * 4.0)
