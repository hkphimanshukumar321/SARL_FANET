import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algorithms.mac.baseline import Config, Logger
from algorithms.mac.channel_aware_mac import simulate_tdma_aware, simulate_csma_aware
from algorithms.mobility.speed import SpeedEngine
from algorithms.mobility.models import create_mobility_model
from algorithms.mobility.link import (
    compute_distances, compute_link_up, compute_pathloss_success_prob, compute_fading_success_prob
)
from algorithms.channel.fading import (
    BERCalculator, AWGNChannel, RayleighChannel, RicianChannel, NakagamiChannel
)
from configs import config as global_cfg
from configs.sarl_config import RLConfig

class AdaptiveMacEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The agent selects the MAC protocol (TDMA vs CSMA/CA) every DECISION_INTERVAL_S.
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        super(AdaptiveMacEnv, self).__init__()
        self.render_mode = render_mode
        self.dt = global_cfg.MOBILITY_DT

        # Action: 0 -> TDMA, 1 -> CSMA_CA
        self.action_space = spaces.Discrete(2)

        # Observation Space (Dict: scalars + history)
        # Scalars: 14 features, normalized to [0, 1] approximately.
        self.num_scalars = RLConfig.NUM_SCALAR_FEATURES
        self.history_len = RLConfig.HISTORY_WINDOW_STEPS

        self.observation_space = spaces.Dict({
            "scalars": spaces.Box(low=0.0, high=1.0, shape=(self.num_scalars,), dtype=np.float32),
            "history": spaces.Box(low=0.0, high=1.0, shape=(self.history_len, self.num_scalars), dtype=np.float32)
        })

        self.sim_time_s = RLConfig.TRAIN_SIM_TIME_S
        self.decision_interval = RLConfig.DECISION_INTERVAL_S
        self.max_steps = int(self.sim_time_s / self.decision_interval)

        # State vars
        self.current_step = 0
        self.global_sim_time = 0.0
        self.mobility_model = None
        self.rng = None
        self.fading_channel = None
        self.ber_calc = None
        
        # Initialize fading models if enabled
        if getattr(global_cfg, 'ENABLE_FADING', False):
            self.ber_calc = BERCalculator(modulation=global_cfg.MODULATION)
            
            f_model = global_cfg.FADING_MODEL.lower()
            if f_model == "awgn":
                self.fading_channel = AWGNChannel()
            elif f_model == "rayleigh":
                self.fading_channel = RayleighChannel()
            elif f_model == "rician":
                self.fading_channel = RicianChannel(K=global_cfg.RICIAN_K)
            elif f_model == "nakagami":
                self.fading_channel = NakagamiChannel(m=global_cfg.NAKAGAMI_M, omega=global_cfg.NAKAGAMI_OMEGA)
            else:
                self.fading_channel = AWGNChannel() # Fallback
        
        # Performance history buffers
        self.history_buffer = np.zeros((self.history_len, self.num_scalars), dtype=np.float32)
        self.prev_delay = 0.0

    def _get_obs(self, metrics, mobility_stats):
        """Constructs the observation dict."""
        # 1. Traffic Load (normalized)
        load_norm = min(global_cfg.OFFERED_PPS / 2000.0, 1.0) if hasattr(global_cfg, 'OFFERED_PPS') else 0.5
        
        # 2. Number of UAVs
        n_norm = min(global_cfg.N / 150.0, 1.0)
        
        # 3. Queue Occupancy
        q_norm = min(metrics.get("avg_queue", 0.0) / RLConfig.MAX_QUEUE_OCCUPANCY, 1.0)
        
        # 4. Backlog
        b_norm = min(metrics.get("backlog", 0.0) / RLConfig.MAX_BACKLOG, 1.0)
        
        # 5. Throughput Est
        thr_norm = min(metrics.get("throughput_mbps", 0.0) / RLConfig.MAX_THROUGHPUT_MBPS, 1.0)
        
        # 6. Delay Est
        del_norm = min((metrics.get("delay_ms", 0.0)) / RLConfig.MAX_DELAY_MS, 1.0)
        
        # 7. Collisions (per step)
        col_norm = min(metrics.get("collisions", 0) / 5000.0, 1.0)
        
        # 8. Drops (per step)
        drop_norm = min(metrics.get("drops", 0) / 1000.0, 1.0)
        
        # 9. Avg Distance
        d_norm = min(mobility_stats.get("avg_dist", 0.0) / 2000.0, 1.0)
        
        # 10. Link Up Ratio
        lu_norm = mobility_stats.get("link_up_ratio", 0.0)
        
        # 11. Mean Speed
        v_norm = min(mobility_stats.get("mean_speed", 0.0) / RLConfig.MAX_SPEED_MPS, 1.0)
        
        # 12. Speed Variance (approx)
        v_var_norm = 0.5  # Fixed for Gaussian unless tracked
        
        # 13. Mean Path Loss (not strictly tracked, use P_succ instead as proxy here)
        sp_norm = mobility_stats.get("avg_succ_prob", 0.0)
        
        # 14. Action feedback (prev step MAC)
        mac_norm = float(metrics.get("prev_action", 1))

        scalar_obs = np.array([
            load_norm, n_norm, q_norm, b_norm, 
            thr_norm, del_norm, col_norm, drop_norm,
            d_norm, lu_norm, v_norm, v_var_norm, sp_norm, mac_norm
        ], dtype=np.float32)

        # Update sliding history
        self.history_buffer = np.roll(self.history_buffer, shift=-1, axis=0)
        self.history_buffer[-1] = scalar_obs

        return {
            "scalars": scalar_obs,
            "history": self.history_buffer.copy()
        }

    def _compute_reward(self, metrics):
        """Rt = w_T*T^ - w_D*D^ - w_F*F^ - w_J*J^"""
        # Normalize terms
        T_hat = min(metrics.get("throughput_mbps", 0.0) / RLConfig.MAX_THROUGHPUT_MBPS, 1.0)
        D_hat = min(metrics.get("delay_ms", 0.0) / RLConfig.MAX_DELAY_MS, 1.0)
        
        pkts_gen = max(metrics.get("pkts_generated", 1), 1)
        F_hat = min(metrics.get("drops", 0) / pkts_gen, 1.0)
        
        curr_delay = metrics.get("delay_ms", 0.0)
        J_hat = min(abs(curr_delay - self.prev_delay) / RLConfig.MAX_DELAY_MS, 1.0)
        self.prev_delay = curr_delay

        r = (RLConfig.REWARD_W_THROUGHPUT * T_hat - 
             RLConfig.REWARD_W_DELAY * D_hat - 
             RLConfig.REWARD_W_FAILURES * F_hat - 
             RLConfig.REWARD_W_JITTER * J_hat)
        return float(r)

    def _step_mobility(self):
        """Advances mobility for DECISION_INTERVAL_S and returns traces."""
        n_steps = int(np.ceil(self.decision_interval / self.dt))
        N = global_cfg.N
        sink_pos = np.array([global_cfg.SINK_X, global_cfg.SINK_Y, global_cfg.SINK_Z])
        
        lu_sched = np.zeros((N, n_steps), dtype=int)
        sp_sched = np.ones((N, n_steps), dtype=float)
        all_speeds = np.zeros((N, n_steps))
        all_dists = np.zeros((N, n_steps))

        for i in range(n_steps):
            pos, vel = self.mobility_model.update(self.dt)
            
            if getattr(global_cfg, 'DECENTRALIZED_COMM', False):
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dist_sq = np.sum(diff ** 2, axis=-1)
                np.fill_diagonal(dist_sq, np.inf)  # Ignore self-distance
                distances = np.sqrt(np.min(dist_sq, axis=1)) # Distance to nearest active peer
            else:
                distances = compute_distances(pos, sink_pos)
            
            lu_sched[:, i] = compute_link_up(distances, global_cfg.COMM_RANGE_R)
            
            if getattr(global_cfg, 'ENABLE_FADING', False) and self.fading_channel is not None:
                payload_bits = getattr(global_cfg, 'PAYLOAD_BYTES', 1500) * 8
                sp_sched[:, i] = compute_fading_success_prob(
                    distances, 
                    self.fading_channel, 
                    self.ber_calc, 
                    global_cfg.TX_POWER_DBM, 
                    global_cfg.NOISE_POWER_DBM, 
                    payload_bits, 
                    self.rng
                )
            elif global_cfg.ENABLE_PATHLOSS:
                sp_sched[:, i] = compute_pathloss_success_prob(distances, k=global_cfg.PATHLOSS_K, eta=global_cfg.PATHLOSS_ETA)
            
            all_speeds[:, i] = np.linalg.norm(vel, axis=1)
            all_dists[:, i] = distances

        stats = {
            "avg_dist": float(np.mean(all_dists)),
            "link_up_ratio": float(np.mean(lu_sched)),
            "mean_speed": float(np.mean(all_speeds)),
            "avg_succ_prob": float(np.mean(sp_sched)),
        }
        return lu_sched, sp_sched, stats

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.global_sim_time = 0.0
        self.history_buffer.fill(0.0)
        self.prev_delay = 0.0

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(global_cfg.SEED)

        # Initialize mobility
        se = SpeedEngine(
            n_nodes=global_cfg.N, v_min=global_cfg.V_MIN, v_max=global_cfg.V_MAX,
            mode=global_cfg.SPEED_MODE, rng=self.rng,
            update_interval=global_cfg.SPEED_UPDATE_INTERVAL
        )
        self.mobility_model = create_mobility_model(
            name=global_cfg.MOBILITY_MODEL, n_nodes=global_cfg.N,
            bounds=(global_cfg.AREA_X, global_cfg.AREA_Y, global_cfg.AREA_Z),
            speed_engine=se, rng=self.rng, gm_alpha=global_cfg.GM_ALPHA,
            rwp_pause_time=global_cfg.RWP_PAUSE_TIME
        )

        # Initial zero-observation
        obs = self._get_obs({}, {})
        info = {}
        return obs, info

    def step(self, action):
        """
        Takes one RL step (e.g., 1.0 seconds of simulation time).
        Action: 0 (TDMA), 1 (CSMA/CA)
        """
        mac_prot = "TDMA" if action == 0 else "CSMA_CA"
        
        # 1. Advance mobility to get traces for this interval
        lu_sched, sp_sched, mob_stats = self._step_mobility()

        # 2. Setup MAC simulation config
        offered_pps = getattr(global_cfg, 'OFFERED_PPS', 400) # Fallback if not set
        cfg = Config(
            N=global_cfg.N, sim_time_s=self.decision_interval, slot_time_s=global_cfg.SLOT_TIME_S,
            phy_rate_bps=global_cfg.PHY_RATE_BPS, payload_bytes=global_cfg.PAYLOAD_BYTES,
            QMAX=global_cfg.QMAX, seed=int(self.rng.integers(0, 10000)),
            cw_min=global_cfg.CW_MIN, cw_max=global_cfg.CW_MAX,
            difs_slots=global_cfg.DIFS_SLOTS, sifs_slots=global_cfg.SIFS_SLOTS,
            ack_slots=global_cfg.ACK_SLOTS, ack_timeout_slots=global_cfg.ACK_TIMEOUT_SLOTS,
            max_retry=global_cfg.MAX_RETRY, log_interval_slots=global_cfg.LOG_INTERVAL_SLOTS,
            rts_cts_enabled=global_cfg.RTS_CTS_ENABLED, ack_enabled=global_cfg.ACK_ENABLED,
            rts_slots=global_cfg.RTS_SLOTS, cts_slots=global_cfg.CTS_SLOTS,
            tdma_guard_time_s=global_cfg.TDMA_GUARD_TIME_S,
        )
        log = Logger(load_pps=offered_pps, protocol_name=mac_prot)

        # 3. Simulate MAC over this interval
        sp_arr = sp_sched if global_cfg.ENABLE_PATHLOSS else None
        if action == 0:
            simulate_tdma_aware(cfg, offered_pps, log, lu_sched, sp_arr, self.dt)
        else:
            simulate_csma_aware(cfg, offered_pps, log, lu_sched, sp_arr, self.dt)

        # 4. Extract metrics
        metrics = {
            "throughput_mbps": log.get_throughput_bps(self.decision_interval) / 1e6,
            "delay_ms": log.get_avg_end_to_end_delay_s() * 1000,
            "drops": log.pkts_dropped_qfull + log.pkts_dropped_mac,
            "collisions": log.collision_events,
            "pkts_generated": log.pkts_generated,
            "avg_queue": 0.0, # Approximate unless tracked
            "backlog": 0.0,
            "prev_action": action
        }

        # 5. Compute Reward & Obs
        reward = self._compute_reward(metrics)
        obs = self._get_obs(metrics, mob_stats)

        self.current_step += 1
        self.global_sim_time += self.decision_interval
        
        terminated = bool(self.current_step >= self.max_steps)
        truncated = False
        info = {"metrics": metrics, "mobility": mob_stats, "reward": reward}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step} | Time: {self.global_sim_time:.1f}s")
