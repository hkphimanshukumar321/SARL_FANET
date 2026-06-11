# SARL Models on MARL Environment

**Single-Agent Reinforcement Learning baselines evaluated on a Multi-Agent cluster-based FANET MAC environment.**

This project demonstrates that SARL models can operate on the same decentralized MARL environment — using identical physics (mobility, fading, clustering, MAC simulation) — and compares their performance against each other.

## Models

| Model | Type | Action Space | Environment Wrapper |
|---|---|---|---|
| Tabular Q-Learning | Classical | Discrete(2) | `MARLtoSARLWrapper` |
| DQN (SB3) | Deep RL | Discrete(2) | `MARLtoSARLWrapper` |
| PPO (SB3) | Deep RL | MultiDiscrete | `SARLCentralEnv` |
| A2C (SB3) | Deep RL | MultiDiscrete | `SARLCentralEnv` |
| MCA-D3QN (Ours) | Branching DQN | MultiDiscrete | `SARLCentralEnv` |
| MCA-PPO (Ours) | Branching PPO | MultiDiscrete | `SARLCentralEnv` |

## Environment Architecture

```
MARLMacEnv (PettingZoo Parallel)
    ├── MARLtoSARLWrapper  → Discrete(2) SARL interface
    │       └── Used by: Tabular, DQN
    └── SARLCentralEnv     → MultiDiscrete centralized interface
            └── Used by: PPO, A2C, MCA-D3QN, MCA-PPO
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke tests
python -m pytest tests/ -v --tb=short

# Run full comparison experiment
python experiments/run_sarl_comparison.py

# Dry run (verify setup without training)
python experiments/run_sarl_comparison.py --dry-run
```

## Project Structure

```
sarl_on_marl_env/
├── algorithms/          # Simulation substrate (MAC, mobility, fading, RL models)
├── configs/             # Configuration files
├── envs/                # MARL env + SARL wrappers
├── experiments/         # Experiment runner
├── tests/               # Smoke & integration tests
├── utils/               # Device management, experiment tracking
└── results/             # Output directory
```

## Key Design Decisions

1. **Same Environment**: All SARL models run on `MARLMacEnv` — the same environment used by MARL agents — ensuring a fair comparison of the *learning paradigm* (SARL vs MARL), not the environment.

2. **Two Wrapper Paths**: Models that need `Discrete(2)` actions (Tabular, DQN) use `MARLtoSARLWrapper` which broadcasts a single MAC decision to all cluster-heads. Models that support `MultiDiscrete` (PPO, A2C, MCA-D3QN, MCA-PPO) use `SARLCentralEnv` which allows per-cluster-head decisions from a centralized agent.

3. **Self-Sustaining**: This project contains complete copies of all required infrastructure. It can be moved to any location and run independently.
