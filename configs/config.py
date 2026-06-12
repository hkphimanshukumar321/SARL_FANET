# config.py
# Centralized Configuration for FANET Simulation Experiments
# (SARL-on-MARL Independent Project)

# ==========================================
# Global Simulation & Node Parameters
# ==========================================
MAC_SELECTION = ["TDMA", "CSMA_CA", "TABULAR", "DQN", "PPO", "A2C", "MCA_D3QN", "MCA_PPO"]
N = 50                                   # Number of senders / nodes
SIM_TIME_S = 15                               # Simulation duration (seconds)
SLOT_TIME_S = 9e-6                            # Discrete time slot granularity
PHY_RATE_BPS = 5e6                            # Data rate (channel/link capacity in bps)
PAYLOAD_BYTES = 1000                      # Packet payload size
QMAX = 100                                   # Queue/buffer capacity per node
SEED = 42                                     # Random seed baseline

# ==========================================
# Experiment Sweep Settings (Load)
# ==========================================
SWEEP_MIN_PPS = 50                           # Minimum total traffic load (packets/sec)
SWEEP_MAX_PPS = 1000                          # Maximum total traffic load (packets/sec)
SWEEP_STEPS = 20                               # Number of granular steps in the sweep

# ==========================================
# Traffic Generation Model
# ==========================================
TRAFFIC_PROFILE = "smooth"   # "smooth" | "bursty_on_off" | "heavy_tail"
TRAFFIC_BURST_ON_PROB = 0.30
TRAFFIC_BURST_MULTIPLIER = 2.0
TRAFFIC_HEAVY_TAIL_SHAPE = 1.8
TRAFFIC_HEAVY_TAIL_SCALE = 1.0

# ==========================================
# Evaluation Regime / Robustness Controls
# ==========================================
TOPOLOGY_PRESET = "default"  # "default" | "compact_dense" | "sparse_separated" | "asymmetric_hotspot"
OBS_STALENESS_STEPS = 0
HANDOVER_INFO_STALENESS_STEPS = 0
OBS_NOISE_STD = 0.0
GRAPH_MODE = "dynamic"       # "dynamic" | "none" | "static" | "shuffled"
GRAPH_STALENESS_STEPS = 0
GRAPH_MISSING_EDGE_PROB = 0.0
GRAPH_FALSE_EDGE_PROB = 0.0
FAILURE_SCHEDULE = ()

# Generalization knobs
SPEED_SCALE = 1.0
INTERFERENCE_SCALE = 1.0
COORDINATION_CAPACITY_SCALE = 1.0

# ==========================================
# Logging Settings
# ==========================================
LOG_INTERVAL_SLOTS = 1000                     # Snapshot interval for buffer tracking

# ==========================================
# CSMA/CA Tunable Parameters
# ==========================================
CW_MIN = 15
CW_MAX = 1023
DIFS_SLOTS = 4
SIFS_SLOTS = 2
ACK_TIMEOUT_SLOTS = 10
MAX_RETRY = 1

# Features
RTS_CTS_ENABLED = True
ACK_ENABLED = True

# Control Frames (in slots - approximations)
RTS_SLOTS = 3
CTS_SLOTS = 3
ACK_SLOTS = 2

# ==========================================
# TDMA Tunable Parameters
# ==========================================
TDMA_GUARD_TIME_S = 1e-6                      # Gap between assigned node slots
TDMA_GUARD_TIME_UNIT = "s"                    # Time unit for guard

# ==========================================
# Q-Learning RL Selector Parameters
# ==========================================
ENABLE_RL_SELECTOR = True
RL_DECISION_INTERVAL_S = 1.0
RL_ALPHA = 1.0
RL_GAMMA = 0.0
RL_EPSILON = 0.0
RL_WT = 0.35
RL_WD = 0.15
RL_STATE_MODE = "traffic_rate_bin"
RL_TRAFFIC_BINS = 20

# ==========================================
# 3D Mobility Parameters
# ==========================================
ENABLE_MOBILITY = True

# Bounding cube (meters)
AREA_X = 200   # A
AREA_Y = 200   # B
AREA_Z = 50    # C

# Sink/base station position (fixed)
SINK_X = 200
SINK_Y = 200
SINK_Z = 25

# Mobility model: "gauss_markov" | "random_waypoint" | "random_walk" | "circular"
MOBILITY_MODEL = "random_walk"

# Gauss-Markov specific
GM_ALPHA = 0.5

# Random Waypoint specific
RWP_PAUSE_TIME = 1.0

# Circular / Spiral specific
CIRC_RADIUS = 100.0
CIRC_OMEGA_MEAN = 0.1
CIRC_OMEGA_STD = 0.02
CIRC_CLIMB_RATE = 0.5

# Speed configuration
SPEED_MODE = "uniform"
V_MIN = 5.0
V_MAX = 35
V_MEAN = 15.0
V_STD = 5.0
SPEED_UPDATE_INTERVAL = 5.0

# Link model
COMM_RANGE_R = 200

# Path-loss
ENABLE_PATHLOSS = True
PATHLOSS_K = 0.00001
PATHLOSS_ETA = 2.0

# Propagation delay
ENABLE_PROP_DELAY = True

# ==========================================
# Fade Model Configuration
# ==========================================
ENABLE_FADING = True
FADING_MODEL = "nakagami"
NAKAGAMI_M = 2.0
NAKAGAMI_OMEGA = 1.0
RICIAN_K = 3.0
TX_POWER_DBM = 20.0
NOISE_POWER_DBM = -80
BER_THRESHOLD = 1e-3
MODULATION = "QPSK"

# Mobility time step resolution (seconds)
MOBILITY_DT = 0.1

# ==========================================
# SARL Experiment Execution Controls
# ==========================================
RUN_TABULAR_QLEARNING = True
RUN_QLEARNING_SELECTOR = True
RUN_CUSTOM_RL = True   # MCA-D3QN
RUN_MCA_PPO = True     # MCA-PPO (new)
RUN_DQN = True
RUN_PPO = True
RUN_A2C = True

# ==========================================
# (MARL algorithms removed to preserve SARL independence)
# ==========================================

# Decentralized communication (needed by MARL env substrate)
DECENTRALIZED_COMM = True

# ==========================================
# Results / Logging Controls
# ==========================================
RESULTS_ROOT = "results"
AUTO_CREATE_RUN_FOLDER = True
SAVE_IMAGES = True
SAVE_CSV = True
SAVE_LOGS = True
SAVE_METADATA = True
SAVE_CHECKPOINTS = True

# ==========================================
# Multiprocessing & Parallel Execution
# ==========================================
ENABLE_MULTIPROCESSING = True
NUM_CPU_WORKERS = "auto"
CPU_UTILIZATION_FRACTION = 0.8
PARALLELIZE_OVER = ["loads"]

# ==========================================
# Hardware / Device Execution
# ==========================================
ENABLE_GPU = True
GPU_DEVICE_ID = 0
FORCE_CPU = False
TRAIN_ON_GPU = True
EVAL_ON_GPU = True
ENABLE_RESOURCE_LOGGING = True
RESULTS_PER_WORKER = True
