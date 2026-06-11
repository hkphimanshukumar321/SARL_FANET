# speed.py — Variable Speed Engine for 3D UAV Mobility
# Supports 4 speed modes: uniform, gaussian, per_node_uniform, piecewise

import numpy as np


class SpeedEngine:
    """
    Generates time-varying speeds for N UAVs.
    All returned speeds are clamped to [v_min, v_max].
    """

    def __init__(self, n_nodes, v_min, v_max, mode="uniform",
                 v_mean=None, v_std=None, update_interval=5.0, rng=None):
        self.n = n_nodes
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.mode = mode
        self.v_mean = float(v_mean if v_mean is not None else (v_min + v_max) / 2)
        self.v_std = float(v_std if v_std is not None else (v_max - v_min) / 4)
        self.update_interval = float(update_interval)
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Current speeds (initialised on first call)
        self.speeds = self._sample_initial()
        self._time_since_update = np.zeros(n_nodes)

        # Per-node bias for per_node_uniform mode
        if self.mode == "per_node_uniform":
            self._node_bias = self.rng.uniform(self.v_min, self.v_max, size=n_nodes)

        # Piecewise phase tracking
        if self.mode == "piecewise":
            self._phase = self.rng.integers(0, 3, size=n_nodes)  # 0=cruise, 1=hover, 2=sprint

    # ------------------------------------------------------------------
    def _clamp(self, s):
        return np.clip(s, self.v_min, self.v_max)

    def _sample_initial(self):
        if self.mode == "gaussian":
            return self._clamp(self.rng.normal(self.v_mean, self.v_std, size=self.n))
        return self.rng.uniform(self.v_min, self.v_max, size=self.n)

    # ------------------------------------------------------------------
    def sample(self, dt):
        """Return (N,) array of current speeds, then advance internal clock."""
        self._time_since_update += dt
        mask = self._time_since_update >= self.update_interval

        if mask.any():
            if self.mode == "uniform":
                self.speeds[mask] = self.rng.uniform(self.v_min, self.v_max, size=mask.sum())

            elif self.mode == "gaussian":
                self.speeds[mask] = self._clamp(
                    self.rng.normal(self.v_mean, self.v_std, size=mask.sum()))

            elif self.mode == "per_node_uniform":
                jitter_range = (self.v_max - self.v_min) * 0.15
                jitter = self.rng.uniform(-jitter_range, jitter_range, size=mask.sum())
                self.speeds[mask] = self._clamp(self._node_bias[mask] + jitter)

            elif self.mode == "piecewise":
                # Rotate to next phase
                self._phase[mask] = (self._phase[mask] + 1) % 3
                cruise_speed = self.v_mean
                hover_speed = self.v_min
                sprint_speed = self.v_max
                for idx in np.where(mask)[0]:
                    p = self._phase[idx]
                    if p == 0:
                        self.speeds[idx] = cruise_speed
                    elif p == 1:
                        self.speeds[idx] = hover_speed
                    else:
                        self.speeds[idx] = sprint_speed

            self._time_since_update[mask] = 0.0

        return self.speeds.copy()
