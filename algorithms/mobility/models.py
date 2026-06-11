# models.py — 3D Mobility Models for UAV FANET Simulation
# Implements: Gauss-Markov, Random Waypoint, Random Walk, Circular/Spiral
# All models enforce boundary clamping to [0,A]×[0,B]×[0,C]

import numpy as np


class _BaseMobilityModel:
    """Common interface for all 3D mobility models."""

    def __init__(self, n_nodes, bounds, speed_engine, rng=None):
        """
        Args:
            n_nodes: Number of UAVs.
            bounds: (A, B, C) — cube dimensions in meters.
            speed_engine: SpeedEngine instance for variable speed.
            rng: numpy random generator.
        """
        self.n = n_nodes
        self.A, self.B, self.C = float(bounds[0]), float(bounds[1]), float(bounds[2])
        self.speed_engine = speed_engine
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Positions initialised uniformly in the cube
        self.positions = np.column_stack([
            self.rng.uniform(0, self.A, size=n_nodes),
            self.rng.uniform(0, self.B, size=n_nodes),
            self.rng.uniform(0, self.C, size=n_nodes),
        ])
        # Velocities (m/s) — initialised to zero, set by first update
        self.velocities = np.zeros((n_nodes, 3))

    def _clamp_positions(self):
        """Reflect/clamp positions to stay inside the cube."""
        for dim, limit in enumerate([self.A, self.B, self.C]):
            # Reflect off lower boundary
            below = self.positions[:, dim] < 0
            self.positions[below, dim] = -self.positions[below, dim]
            self.velocities[below, dim] = abs(self.velocities[below, dim])
            # Reflect off upper boundary
            above = self.positions[:, dim] > limit
            self.positions[above, dim] = 2 * limit - self.positions[above, dim]
            self.velocities[above, dim] = -abs(self.velocities[above, dim])
            # Final hard clamp (safety)
            self.positions[:, dim] = np.clip(self.positions[:, dim], 0, limit)

    def update(self, dt):
        """Advance by dt seconds. Returns (positions, velocities) — both (N,3)."""
        raise NotImplementedError


# ======================================================================
# 1) Gauss-Markov 3D
# ======================================================================
class GaussMarkov3D(_BaseMobilityModel):
    """Gauss-Markov mobility: velocity is a correlated random process."""

    def __init__(self, n_nodes, bounds, speed_engine, alpha=0.5, rng=None):
        super().__init__(n_nodes, bounds, speed_engine, rng)
        self.alpha = float(alpha)
        # Initial speed & direction
        speeds = self.speed_engine.sample(0)
        theta = self.rng.uniform(0, 2 * np.pi, size=n_nodes)  # azimuth
        phi = self.rng.uniform(-np.pi / 2, np.pi / 2, size=n_nodes)  # elevation
        self.velocities = np.column_stack([
            speeds * np.cos(phi) * np.cos(theta),
            speeds * np.cos(phi) * np.sin(theta),
            speeds * np.sin(phi),
        ])
        self._mean_speed = (speed_engine.v_min + speed_engine.v_max) / 2

    def update(self, dt):
        a = self.alpha
        speeds = self.speed_engine.sample(dt)
        n = self.n

        # Gaussian noise
        noise = self.rng.normal(0, 1, size=(n, 3))

        # Correlated velocity update per component
        mean_v = self._mean_speed / np.sqrt(3)  # spread mean across 3 axes
        for d in range(3):
            self.velocities[:, d] = (
                a * self.velocities[:, d]
                + (1 - a) * mean_v
                + np.sqrt(1 - a ** 2) * noise[:, d] * (speeds / np.sqrt(3))
            )

        # Re-scale so the speed magnitude matches the desired speed
        current_speed = np.linalg.norm(self.velocities, axis=1)
        current_speed = np.maximum(current_speed, 1e-9)
        scale = speeds / current_speed
        self.velocities *= scale[:, np.newaxis]

        self.positions += self.velocities * dt
        self._clamp_positions()
        return self.positions.copy(), self.velocities.copy()


# ======================================================================
# 2) Random Waypoint 3D
# ======================================================================
class RandomWaypoint3D(_BaseMobilityModel):
    """Random Waypoint: fly to random destination, pause, pick new one."""

    def __init__(self, n_nodes, bounds, speed_engine, pause_time=1.0, rng=None):
        super().__init__(n_nodes, bounds, speed_engine, rng)
        self.pause_time = float(pause_time)
        # Generate initial waypoints
        self._waypoints = np.column_stack([
            self.rng.uniform(0, self.A, size=n_nodes),
            self.rng.uniform(0, self.B, size=n_nodes),
            self.rng.uniform(0, self.C, size=n_nodes),
        ])
        self._speeds = self.speed_engine.sample(0)
        self._pausing = np.zeros(n_nodes, dtype=bool)
        self._pause_remaining = np.zeros(n_nodes)
        self._update_directions()

    def _update_directions(self):
        diff = self._waypoints - self.positions
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-9)
        self._directions = diff / dist

    def update(self, dt):
        for i in range(self.n):
            if self._pausing[i]:
                self._pause_remaining[i] -= dt
                if self._pause_remaining[i] <= 0:
                    self._pausing[i] = False
                    # Pick new waypoint and speed
                    self._waypoints[i] = [
                        self.rng.uniform(0, self.A),
                        self.rng.uniform(0, self.B),
                        self.rng.uniform(0, self.C),
                    ]
                    self._speeds[i] = self.speed_engine.sample(dt)[i]
                    diff = self._waypoints[i] - self.positions[i]
                    dist = np.linalg.norm(diff)
                    self._directions[i] = diff / max(dist, 1e-9)
                continue

            # Move toward waypoint
            step = self._directions[i] * self._speeds[i] * dt
            remaining_dist = np.linalg.norm(self._waypoints[i] - self.positions[i])

            if np.linalg.norm(step) >= remaining_dist:
                # Arrived at waypoint
                self.positions[i] = self._waypoints[i].copy()
                self._pausing[i] = True
                self._pause_remaining[i] = self.pause_time
                self.velocities[i] = 0.0
            else:
                self.positions[i] += step
                self.velocities[i] = self._directions[i] * self._speeds[i]

        self._clamp_positions()
        return self.positions.copy(), self.velocities.copy()


# ======================================================================
# 3) Random Walk 3D
# ======================================================================
class RandomWalk3D(_BaseMobilityModel):
    """Random Walk: random direction changes per step interval."""

    def __init__(self, n_nodes, bounds, speed_engine, rng=None):
        super().__init__(n_nodes, bounds, speed_engine, rng)
        self._pick_new_directions()

    def _pick_new_directions(self):
        theta = self.rng.uniform(0, 2 * np.pi, size=self.n)
        phi = self.rng.uniform(-np.pi / 2, np.pi / 2, size=self.n)
        self._directions = np.column_stack([
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi),
        ])

    def update(self, dt):
        speeds = self.speed_engine.sample(dt)
        # Whenever the speed engine triggers an update, also change direction
        # We proxy this by re-picking directions when speeds change significantly
        self._pick_new_directions()

        self.velocities = self._directions * speeds[:, np.newaxis]
        self.positions += self.velocities * dt
        self._clamp_positions()
        return self.positions.copy(), self.velocities.copy()


# ======================================================================
# 4) Circular / Spiral 3D
# ======================================================================
class CircularSpiral3D(_BaseMobilityModel):
    """Circular loiter with variable angular velocity and optional climb."""

    def __init__(self, n_nodes, bounds, speed_engine,
                 radius=100.0, omega_mean=0.1, omega_std=0.02,
                 climb_rate=0.5, rng=None):
        super().__init__(n_nodes, bounds, speed_engine, rng)
        self.radius = float(radius)
        self.omega_mean = float(omega_mean)
        self.omega_std = float(omega_std)
        self.climb_rate = float(climb_rate)

        # Each UAV has its own centre and phase
        self._centres = self.positions.copy()
        self._phase = self.rng.uniform(0, 2 * np.pi, size=n_nodes)
        self._omega = self.rng.normal(omega_mean, omega_std, size=n_nodes)
        self._climb_dir = self.rng.choice([-1, 1], size=n_nodes).astype(float)

        # Set initial positions on circle around centre
        self.positions[:, 0] = self._centres[:, 0] + self.radius * np.cos(self._phase)
        self.positions[:, 1] = self._centres[:, 1] + self.radius * np.sin(self._phase)
        self._clamp_positions()

    def update(self, dt):
        speeds = self.speed_engine.sample(dt)

        # Vary omega based on speed (ω = v / r)
        self._omega = speeds / max(self.radius, 1e-9)
        # Add small noise
        self._omega += self.rng.normal(0, self.omega_std, size=self.n)

        self._phase += self._omega * dt

        # Horizontal circular motion around each centre
        self.positions[:, 0] = self._centres[:, 0] + self.radius * np.cos(self._phase)
        self.positions[:, 1] = self._centres[:, 1] + self.radius * np.sin(self._phase)

        # Vertical climb / descent (spiral)
        self.positions[:, 2] += self._climb_dir * self.climb_rate * dt
        # Reverse climb direction at boundaries
        at_top = self.positions[:, 2] >= self.C
        at_bot = self.positions[:, 2] <= 0
        self._climb_dir[at_top] = -1
        self._climb_dir[at_bot] = 1

        # Compute velocities (dx/dt, dy/dt, dz/dt)
        self.velocities[:, 0] = -self.radius * self._omega * np.sin(self._phase)
        self.velocities[:, 1] = self.radius * self._omega * np.cos(self._phase)
        self.velocities[:, 2] = self._climb_dir * self.climb_rate

        self._clamp_positions()
        return self.positions.copy(), self.velocities.copy()


# ======================================================================
# 5) Static (no movement)
# ======================================================================
class StaticModel(_BaseMobilityModel):
    """Static model: positions fixed at initial placement, velocities zero."""

    def __init__(self, n_nodes, bounds, speed_engine, rng=None):
        super().__init__(n_nodes, bounds, speed_engine, rng)
        self.velocities = np.zeros((n_nodes, 3))

    def update(self, dt):
        # No movement — return unchanged positions and zero velocities
        return self.positions.copy(), self.velocities.copy()


# ======================================================================
# Factory
# ======================================================================
def create_mobility_model(name, n_nodes, bounds, speed_engine, rng=None, **kwargs):
    """Instantiate a mobility model by name string from config."""
    name = name.lower().strip()
    if name == "gauss_markov":
        alpha = kwargs.get("gm_alpha", 0.5)
        return GaussMarkov3D(n_nodes, bounds, speed_engine, alpha=alpha, rng=rng)
    elif name == "random_waypoint":
        pause = kwargs.get("rwp_pause_time", 1.0)
        return RandomWaypoint3D(n_nodes, bounds, speed_engine, pause_time=pause, rng=rng)
    elif name == "random_walk":
        return RandomWalk3D(n_nodes, bounds, speed_engine, rng=rng)
    elif name == "circular":
        return CircularSpiral3D(
            n_nodes, bounds, speed_engine,
            radius=kwargs.get("circ_radius", 100.0),
            omega_mean=kwargs.get("circ_omega_mean", 0.1),
            omega_std=kwargs.get("circ_omega_std", 0.02),
            climb_rate=kwargs.get("circ_climb_rate", 0.5),
            rng=rng,
        )
    elif name == "static":
        return StaticModel(n_nodes, bounds, speed_engine, rng=rng)
    else:
        raise ValueError(f"Unknown mobility model: {name!r}. "
                         f"Choose from: gauss_markov, random_waypoint, random_walk, circular, static")
