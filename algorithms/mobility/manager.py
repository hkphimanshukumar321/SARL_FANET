# manager.py — MobilityManager: Orchestrator for 3D UAV Mobility Simulation
# Creates models, runs time-stepped simulation, stores trajectory history,
# exports CSVs, and delegates plot generation.

import os
import numpy as np
import pandas as pd

from .speed import SpeedEngine
from .models import create_mobility_model
from .link import compute_distances, compute_link_up, compute_pathloss_success_prob, compute_propagation_delay


class MobilityManager:
    """
    Central orchestrator for mobility simulation.

    Usage:
        mgr = MobilityManager(...)
        mgr.run(sim_time_s)           # run full mobility trace
        mgr.export_csv(out_dir)       # save CSVs
        mgr.generate_plots(out_dir)   # save plots
    """

    def __init__(self, n_nodes, bounds, sink_pos,
                 mobility_model="gauss_markov",
                 speed_mode="uniform", v_min=5.0, v_max=30.0,
                 v_mean=None, v_std=None, speed_update_interval=5.0,
                 comm_range=500.0,
                 enable_pathloss=False, pathloss_k=0.001, pathloss_eta=2.0,
                 enable_prop_delay=False,
                 dt=0.1, seed=42,
                 # Model-specific kwargs
                 gm_alpha=0.5,
                 rwp_pause_time=1.0,
                 circ_radius=100.0, circ_omega_mean=0.1, circ_omega_std=0.02,
                 circ_climb_rate=0.5):

        self.n = n_nodes
        self.bounds = (float(bounds[0]), float(bounds[1]), float(bounds[2]))
        self.sink_pos = np.array([float(sink_pos[0]), float(sink_pos[1]), float(sink_pos[2])])
        self.dt = float(dt)
        self.comm_range = float(comm_range)
        self.enable_pathloss = enable_pathloss
        self.pathloss_k = float(pathloss_k)
        self.pathloss_eta = float(pathloss_eta)
        self.enable_prop_delay = enable_prop_delay

        self.rng = np.random.default_rng(seed)

        # Create speed engine
        self.speed_engine = SpeedEngine(
            n_nodes=n_nodes, v_min=v_min, v_max=v_max,
            mode=speed_mode, v_mean=v_mean, v_std=v_std,
            update_interval=speed_update_interval, rng=self.rng,
        )

        # Create mobility model
        self.model = create_mobility_model(
            name=mobility_model, n_nodes=n_nodes, bounds=self.bounds,
            speed_engine=self.speed_engine, rng=self.rng,
            gm_alpha=gm_alpha, rwp_pause_time=rwp_pause_time,
            circ_radius=circ_radius, circ_omega_mean=circ_omega_mean,
            circ_omega_std=circ_omega_std, circ_climb_rate=circ_climb_rate,
        )

        # ---------- History storage ----------
        self.history_positions = []   # list of (N,3) arrays
        self.history_velocities = []  # list of (N,3) arrays
        self.history_distances = []   # list of (N,) arrays
        self.history_link_up = []     # list of (N,) arrays
        self.history_times = []       # list of float timestamps

    # ------------------------------------------------------------------
    def run(self, sim_time_s):
        """Run the full mobility trace for sim_time_s seconds."""
        t = 0.0
        steps = int(np.ceil(sim_time_s / self.dt))

        for _ in range(steps):
            # Record current state *before* update at t=0, then after each step
            if t == 0.0:
                pos = self.model.positions.copy()
                vel = self.model.velocities.copy()
            else:
                pos, vel = self.model.update(self.dt)

            distances = compute_distances(pos, self.sink_pos)
            link_up = compute_link_up(distances, self.comm_range)

            self.history_times.append(t)
            self.history_positions.append(pos)
            self.history_velocities.append(vel)
            self.history_distances.append(distances)
            self.history_link_up.append(link_up)

            t += self.dt

    # ------------------------------------------------------------------
    # Public accessors (clean interface for MAC integration)
    # ------------------------------------------------------------------
    def get_positions_at(self, step_idx):
        """Return (N,3) positions at time step index."""
        return self.history_positions[step_idx]

    def get_link_up_at(self, step_idx):
        """Return (N,) link states at time step index."""
        return self.history_link_up[step_idx]

    @property
    def total_steps(self):
        return len(self.history_times)

    # ------------------------------------------------------------------
    # CSV Export
    # ------------------------------------------------------------------
    def export_csv(self, out_dir):
        """Write mobility_positions.csv and uav_sink_distance.csv."""
        csv_dir = os.path.join(out_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        # ----- mobility_positions.csv -----
        rows_pos = []
        for step_idx, t in enumerate(self.history_times):
            pos = self.history_positions[step_idx]
            vel = self.history_velocities[step_idx]
            speeds = np.linalg.norm(vel, axis=1)
            for i in range(self.n):
                rows_pos.append({
                    "timestamp": round(t, 6),
                    "uav_id": i,
                    "x": round(pos[i, 0], 4),
                    "y": round(pos[i, 1], 4),
                    "z": round(pos[i, 2], 4),
                    "vx": round(vel[i, 0], 4),
                    "vy": round(vel[i, 1], 4),
                    "vz": round(vel[i, 2], 4),
                    "speed": round(speeds[i], 4),
                })
        df_pos = pd.DataFrame(rows_pos)
        df_pos.to_csv(os.path.join(csv_dir, "mobility_positions.csv"), index=False)

        # ----- uav_sink_distance.csv -----
        rows_dist = []
        for step_idx, t in enumerate(self.history_times):
            dists = self.history_distances[step_idx]
            link = self.history_link_up[step_idx]
            for i in range(self.n):
                rows_dist.append({
                    "timestamp": round(t, 6),
                    "uav_id": i,
                    "d_to_sink": round(dists[i], 4),
                    "link_up": int(link[i]),
                })
        df_dist = pd.DataFrame(rows_dist)
        df_dist.to_csv(os.path.join(csv_dir, "uav_sink_distance.csv"), index=False)

        return df_pos, df_dist

    # ------------------------------------------------------------------
    # Plot Generation
    # ------------------------------------------------------------------
    def generate_plots(self, out_dir, top_k=5):
        """Generate mobility-specific plots. Delegates to plotting module."""
        from .plotting import plot_trajectories_3d, plot_distance_vs_time, plot_link_up_ratio
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        plot_trajectories_3d(
            self.history_positions, self.history_times, self.n,
            self.bounds, self.sink_pos, img_dir, top_k=top_k,
        )
        plot_distance_vs_time(
            self.history_distances, self.history_times, self.n,
            self.comm_range, img_dir, top_k=top_k,
        )
        plot_link_up_ratio(
            self.history_link_up, self.history_times, self.n, img_dir,
        )
