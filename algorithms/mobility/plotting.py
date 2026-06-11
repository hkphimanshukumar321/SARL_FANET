# plotting.py — Mobility-specific Plot Generation
# Generates: 3D trajectories, distance-to-sink time series, link-up ratio

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for 3D projection


def plot_trajectories_3d(history_positions, history_times, n_nodes,
                         bounds, sink_pos, img_dir, top_k=5):
    """
    Plot 3D UAV trajectories for the first top_k UAVs + sink marker.
    Saves: uav_trajectories_3d.png
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    k = min(top_k, n_nodes)
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for uav_id in range(k):
        xs = [pos[uav_id, 0] for pos in history_positions]
        ys = [pos[uav_id, 1] for pos in history_positions]
        zs = [pos[uav_id, 2] for pos in history_positions]
        ax.plot(xs, ys, zs, color=colors[uav_id], alpha=0.7, linewidth=0.8,
                label=f"UAV {uav_id}")
        # Start and end markers
        ax.scatter(xs[0], ys[0], zs[0], color=colors[uav_id], marker='o', s=40)
        ax.scatter(xs[-1], ys[-1], zs[-1], color=colors[uav_id], marker='x', s=60)

    # Plot sink
    ax.scatter(*sink_pos, color='red', marker='*', s=200, label='Sink/BS', zorder=5)

    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_zlim(0, bounds[2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D UAV Trajectories (top {k} of {n_nodes} UAVs)\n"
                 f"Bounds: {bounds[0]}×{bounds[1]}×{bounds[2]} m, "
                 f"Duration: {history_times[-1]:.1f}s")
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "uav_trajectories_3d.png"), dpi=150)
    plt.close(fig)


def plot_distance_vs_time(history_distances, history_times, n_nodes,
                          comm_range, img_dir, top_k=5):
    """
    Plot UAV-to-sink distance over time for top_k representative UAVs.
    Saves: distance_to_sink_vs_time_uavK.png
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    k = min(top_k, n_nodes)
    times = np.array(history_times)
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for uav_id in range(k):
        dists = np.array([d[uav_id] for d in history_distances])
        ax.plot(times, dists, color=colors[uav_id], alpha=0.8, linewidth=0.9,
                label=f"UAV {uav_id}")

    # Communication range threshold line
    ax.axhline(y=comm_range, color='red', linestyle='--', linewidth=1.5,
               label=f"Comm Range R={comm_range}m")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance to Sink (m)")
    ax.set_title(f"UAV–Sink Distance vs Time (top {k} UAVs)")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "distance_to_sink_vs_time_uavK.png"), dpi=150)
    plt.close(fig)


def plot_link_up_ratio(history_link_up, history_times, n_nodes, img_dir):
    """
    Plot fraction of UAVs connected (link_up) over time.
    Saves: link_up_ratio_vs_time.png
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    times = np.array(history_times)
    ratios = np.array([lu.sum() / n_nodes for lu in history_link_up])

    ax.plot(times, ratios, color='steelblue', linewidth=1.0)
    ax.fill_between(times, ratios, alpha=0.15, color='steelblue')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction of UAVs Connected")
    ax.set_title(f"Link-Up Ratio vs Time (N={n_nodes})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "link_up_ratio_vs_time.png"), dpi=150)
    plt.close(fig)
