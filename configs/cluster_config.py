# cluster_config.py
# Configuration for Dynamic Cluster-Based Decentralized Control
#
# Mathematical reference:
#   S_{i,k}(t) = w_d f_d + w_s f_s + w_m f_m + w_q f_q
#   cluster(i,t) = argmax_k S_{i,k}(t)  subject to hysteresis

import os


class ClusterConfig:
    """All tunable constants for the clustering subsystem."""

    # --------------------------------------------------
    # Cluster count bounds
    # --------------------------------------------------
    C_MIN = 3
    C_MAX = 10
    C_INIT = 5

    # --------------------------------------------------
    # Cluster size bounds (triggers split / merge)
    # --------------------------------------------------
    N_MIN = 2   # N_{merge}^{min} in math (triggers merge if below)
    N_MAX = 12  # N_{split}^{max} in math (triggers split if above)

    # --------------------------------------------------
    # Spatial radii (meters)
    # --------------------------------------------------
    R_C = 100.0  # R_c in math: cluster-association radius
    R_I = 250.0  # R_I in math: inter-cluster interaction radius

    # --------------------------------------------------
    # Interference graph SINR threshold
    # --------------------------------------------------
    TAU_I = 10.0  # \tau_I in math: interference graph threshold

    # --------------------------------------------------
    # Membership score weights
    # --------------------------------------------------
    # Formula: S_{i,k}(t) = w_d*f_d + w_s*f_s + w_m*f_m + w_q*f_q
    # where:
    #   - f_d = max(1.0 - d / R_c, 0.0) : proximity score ensuring spatial locality within cluster radius
    #   - f_s = max(1.0 - d / R_I, 0.0) : SINR quality proxy penalizing distances relative to interference range
    #   - f_m = 1.0 / (1.0 + ||v_i - v_leader||) : mobility match rewarding similar velocity vectors
    #   - f_q = 1.0 / (1.0 + |C_k| / N_MAX) : load balance penalizing joining near-capacity clusters
    # --------------------------------------------------
    W_DIST = 0.40  # w_d: proximity weight (sign: +)
    W_SINR = 0.25  # w_s: communication quality proxy weight (sign: +)
    W_MOB = 0.20   # w_m: mobility stability compatibility weight (sign: +)
    W_LOAD = 0.15  # w_q: cluster load balancing weight (sign: +)

    # --------------------------------------------------
    # Hysteresis thresholds (anti-oscillation)
    # --------------------------------------------------
    # Rule: Move to cluster M if Score(M) > THETA_JOIN AND Score(Current) < THETA_LEAVE
    # This prevents the ping-pong effect of UAVs rapidly swapping clusters on the edge.
    THETA_JOIN = 0.65  # Minimum score an alternative cluster must offer to justify joining
    THETA_LEAVE = 0.35 # Maximum score the current cluster can provide to permit leaving

    # --------------------------------------------------
    # Cluster-head health & handover
    # --------------------------------------------------
    HEALTH_THRESHOLD = 0.20  # Handover trigger threshold (elect a new leader if health < threshold)
    
    # Formula: H_u(t) = w_1*e_u + w_2*deg_u + w_3*m_u - w_4*q_u - w_5*risk_u
    # where:
    #   - e_u = Energy / E_INIT : normalized residual battery energy
    #   - deg_u = Degree within R_I / max(Clusters - 1, 1) : normalized connectivity to other cluster heads
    #   - m_u = 1.0 / (1.0 + ||v_u|| / 30.0) : mobility stability based on the leader's absolute speed
    #   - q_u = Queue Size / 100.0 : normalized service burden / packet backlog
    #   - risk_u = max(1.0 - e_u, 0.0) : active penalization for fatally low energy levels
    A_ENERGY = 0.30  # w_1: residual-energy weight (sign: +)
    A_DEGREE = 0.25  # w_2: connectivity degree weight (sign: +)
    A_MOBSTAB = 0.20 # w_3: mobility-stability weight (sign: +)
    A_QUEUE = 0.15   # w_4: service burden / queue penalty weight (sign: - in formula)
    A_RISK = 0.10    # w_5: supplementary low-energy risk penalization factor (sign: -)

    # --------------------------------------------------
    # Energy model (simple linear drain)
    # --------------------------------------------------
    E_INIT = 100.0
    E_TX_COST = 0.01
    E_IDLE_COST = 0.001

    # --------------------------------------------------
    # Timing
    # --------------------------------------------------
    T_CLUSTER = 5

    # --------------------------------------------------
    # Observation space
    # --------------------------------------------------
    OBS_DIM_CLUSTER = 24
    NEIGHBOR_SUMMARY_DIM = 3
    MAX_NEIGHBORS = 6

    # --------------------------------------------------
    # Burst split control
    # --------------------------------------------------
    BURST_TOTAL_TIME = 0.5  # t_{burst}: total burst duration
    T1_MIN = 0.05           # min for T_{1,k}(t): intra-cluster phase time
    T2_MIN = 0.05           # min for T_{2,k}(t): head-head coordination phase time
    RHO_ACTION_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # \rho_k(t): burst-allocation ratio
    DEFAULT_RHO = 0.5
    COORD_SYNC_MODE = "tail_aligned"
    COORD_GUARD_TIME = 0.05
    INTRA_CLUSTER_PAYLOAD_TYPES = ("data", "control", "retransmission")
    INTER_CLUSTER_COORD_PAYLOAD_TYPES = (
        "schedule_exchange",
        "relay_control",
        "topology_update",
        "handover_notice",
    )
    COORD_CAPACITY_PER_SEC = 4.0
    LOCAL_CTRL_CAPACITY_PER_SEC = 6.0
    RELAY_SERVICE_CAPACITY_PER_SEC = 3.0
    COORD_BACKLOG_INCREMENT = 0.25
    RELAY_DEMAND_INCREMENT = 0.20
    LOCAL_CTRL_DEMAND_INCREMENT = 0.15
    MAX_COORD_BACKLOG = 10.0
    MAX_RELAY_DEMAND = 10.0
    MAX_LOCAL_CTRL_DEMAND = 10.0

    # --------------------------------------------------
    # Action space
    # --------------------------------------------------
    NUM_MAC_MODES = 2
    NUM_CHANNELS = 1
    NUM_CW_CLASSES = 1
    NUM_RHO_LEVELS = len(RHO_ACTION_LEVELS)

    #   |A| = NUM_MAC_MODES * NUM_CHANNELS * NUM_CW_CLASSES * NUM_RHO_LEVELS
    NUM_ACTIONS = NUM_MAC_MODES * NUM_CHANNELS * NUM_CW_CLASSES * NUM_RHO_LEVELS

    # --------------------------------------------------
    # Reward coefficients
    # --------------------------------------------------
    ALPHA_LOCAL_THROUGHPUT = 0.22  # \alpha_1: intra-cluster throughput (sign: +)
    ALPHA_INTER_THROUGHPUT = 0.13  # \alpha_2: inter-cluster throughput (sign: +)
    BETA_LOCAL_DELAY = 0.10        # \beta_1: intra-cluster delay penalty (sign: -)
    BETA_INTER_DELAY = 0.05        # \beta_2: inter-cluster delay penalty (sign: -)
    GAMMA_COLLISION = 0.20         # \eta_1: collision penalty (sign: -)
    DELTA_ENERGY = 0.05            # \eta_4: energy expenditure penalty (sign: -)
    ETA_INTERFERENCE = 0.12        # \eta_2: interference penalty (sign: -)
    PSI_COORD_FAILURE = 0.08       # part of \eta_5: robustness disruption (sign: -)
    PHI_QUEUE_OVERFLOW = 0.10      # \eta_3: queue stress / overflow (sign: -)
    XI_AOI = 0.00                  # (Unmapped age of information)
    ZETA_HANDOVER = 0.05           # part of \eta_5: handover disruption (sign: -)
    LAMBDA_FAIRNESS = 0.10         # \lambda_J: Jain's fairness index weight (sign: +)

    # --------------------------------------------------
    # Normalization ceilings
    # --------------------------------------------------
    MAX_THROUGHPUT_MBPS = 5.0
    MAX_DELAY_MS = 200.0
    MAX_COLLISIONS = 500
    MAX_ENERGY_COST = 1.0
    MAX_QUEUE_OVERFLOW_RATIO = 1.0
    MAX_COORD_FAILURE_RATIO = 1.0

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    @staticmethod
    def get_cluster_log_dir():
        base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "results", "cluster_logs")
        )
        os.makedirs(base, exist_ok=True)
        return base
