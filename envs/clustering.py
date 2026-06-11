"""
clustering.py — Dynamic Cluster Manager for Decentralized FANET Control
=========================================================================

Implements the complete clustering subsystem as specified in the mathematical
formulation:

  1. K-Means initialization of c(t) clusters
  2. Membership score  S_{i,k}(t) = Σ w_j f_j(i,k,t)
  3. Hysteresis-gated reassociation
  4. Split (|C_k| > N_max) and Merge (|C_k| < N_min)
  5. Cluster-head election and fault-tolerant handover
  6. Dynamic interference graph G_t = (V_t, E_t)
  7. Per-cluster observation and summary construction

All arrays are sized to C_MAX and use an alive_mask to handle variable c(t).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from configs.cluster_config import ClusterConfig as CC


# ======================================================================
# Data Structures
# ======================================================================
@dataclass
class ClusterState:
    """Mutable state for a single cluster."""
    cluster_id: int
    leader_idx: int                          # UAV index of the cluster-head
    member_indices: List[int] = field(default_factory=list)
    agg_queue: float = 0.0                   # Aggregated queue occupancy
    agg_throughput: float = 0.0              # Last-step throughput
    agg_delay: float = 0.0                   # Last-step avg delay (ms)
    agg_collisions: int = 0                  # Last-step collisions
    agg_drops: int = 0                       # Last-step drops
    local_ctrl_demand: float = 0.0           # Pending intra-cluster control demand
    coord_backlog: float = 0.0               # Pending inter-cluster coordination demand
    relay_demand: float = 0.0                # Pending relay/forwarding demand
    recent_rho: float = CC.DEFAULT_RHO       # Previous burst split
    recent_t1_util: float = 0.0              # Intra-cluster phase utilization
    recent_t2_util: float = 0.0              # Inter-cluster phase utilization
    recent_coord_success: float = 0.0        # Coordination success ratio
    handover_flag: bool = False              # True if a handover happened recently
    handover_cooldown: int = 0               # Steps since last handover
    alive: bool = True


class ClusterManager:
    """
    Manages dynamic clustering of n UAVs into c(t) clusters.

    Public interface used by MARLMacEnv:
        reset(positions, velocities)  → initializes clusters
        update(positions, velocities, energies, queues, step)
                                      → re-evaluates membership, split/merge
        get_interference_graph()      → returns adjacency for GNN edge_index
        get_cluster_obs(k, ...)       → returns observation vector for head k
        get_active_cluster_ids()      → list of alive cluster IDs
        get_members(k)                → UAV indices in cluster k
        get_leader(k)                 → UAV index of head k
        trigger_leader_failure(k)     → simulate failure of head k
    """

    def __init__(self, n_uavs: int, rng: np.random.Generator):
        self.n = n_uavs
        self.rng = rng

        # Assignment array: assignment[i] = cluster_id for UAV i
        self.assignment = np.full(n_uavs, -1, dtype=int)

        # Energy tracking
        self.energy = np.full(n_uavs, CC.E_INIT, dtype=np.float64)

        # Cluster states, indexed 0..C_MAX-1 (sparse, use alive flag)
        self.clusters: Dict[int, ClusterState] = {}
        self._next_cluster_id = 0

        # Interference adjacency matrix (C_MAX x C_MAX)
        self._adj = np.zeros((CC.C_MAX, CC.C_MAX), dtype=bool)

        # Score cache for hysteresis
        self._score_cache = np.zeros((n_uavs, CC.C_MAX), dtype=np.float64)

        # Logging / diagnostics
        self.split_count = 0
        self.merge_count = 0
        self.reassoc_count = 0
        self.handover_count = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def reset(self, positions: np.ndarray, velocities: np.ndarray):
        """
        Initialize clusters using K-Means on spatial positions.

        Args:
            positions:  (n, 3) array of UAV positions
            velocities: (n, 3) array of UAV velocities
        """
        self.energy[:] = CC.E_INIT
        self.clusters.clear()
        self._next_cluster_id = 0
        self.split_count = 0
        self.merge_count = 0
        self.reassoc_count = 0
        self.handover_count = 0

        c_init = min(CC.C_INIT, self.n)
        c_init = max(c_init, CC.C_MIN)

        # Simple K-Means (5 iterations sufficient for initialization)
        centroids = positions[self.rng.choice(self.n, size=c_init, replace=False)]
        for _ in range(5):
            dists = np.linalg.norm(
                positions[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
            )  # (n, c_init)
            labels = np.argmin(dists, axis=1)
            for k in range(c_init):
                members = np.where(labels == k)[0]
                if len(members) > 0:
                    centroids[k] = positions[members].mean(axis=0)

        # Final assignment
        dists = np.linalg.norm(
            positions[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        labels = np.argmin(dists, axis=1)

        # Create cluster objects
        for k in range(c_init):
            members = np.where(labels == k)[0].tolist()
            if len(members) == 0:
                continue
            cid = self._alloc_cluster_id()
            leader = self._elect_leader(members, positions, velocities)
            cs = ClusterState(cluster_id=cid, leader_idx=leader, member_indices=members)
            self.clusters[cid] = cs
            for i in members:
                self.assignment[i] = cid

        # Sanity: orphan check
        self._fix_orphans(positions, velocities)

    # ------------------------------------------------------------------
    # Periodic Update (call every T_CLUSTER steps)
    # ------------------------------------------------------------------
    def update(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        queues: np.ndarray,
        step: int,
    ):
        """
        Full clustering update cycle:
          1. Recompute membership scores
          2. Hysteresis-gated reassociation
          3. Split oversized clusters
          4. Merge undersized clusters
          5. Update leader health, trigger handover if needed
          6. Rebuild interference graph
        """
        if step % CC.T_CLUSTER != 0:
            return

        # Drain energy
        self._drain_energy()

        # 1–2: Reassociation
        self._reassociate(positions, velocities, queues)

        # 2b: Cull clusters emptied by reassociation
        self._cull_empty_clusters()

        # 3: Split
        self._try_splits(positions, velocities)

        # 4: Merge
        self._try_merges(positions, velocities)

        # 5: Health & handover
        self._check_leader_health(positions, velocities, queues)

        # Fix any orphans created during split/merge
        self._fix_orphans(positions, velocities)

        # 5b: Ensure C_MIN is respected (spawn clusters if needed)
        self._enforce_c_min(positions, velocities)

        # 6: Rebuild interference graph
        self._build_interference_graph(positions)

        # Decrement handover cooldowns
        for cs in self.clusters.values():
            if cs.handover_cooldown > 0:
                cs.handover_cooldown -= 1
                if cs.handover_cooldown == 0:
                    cs.handover_flag = False

    # ------------------------------------------------------------------
    # Membership Score
    # ------------------------------------------------------------------
    def _membership_score(
        self,
        i: int,
        cid: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        queues: np.ndarray,
    ) -> float:
        """
        S_{i,k}(t) = w_d·f_d + w_s·f_s + w_m·f_m + w_q·f_q

        f_d: proximity score  (1 - d/R_c), clipped to [0,1]
        f_s: SINR proxy       (1 - d/R_I), clipped to [0,1]
        f_m: mobility match   1 / (1 + ||v_i - v_leader||)
        f_q: load balance     1 / (1 + |C_k| / N_MAX)
        """
        cs = self.clusters.get(cid)
        if cs is None or not cs.alive:
            return -np.inf

        leader = cs.leader_idx
        d = np.linalg.norm(positions[i] - positions[leader])

        # f_d: proximity
        f_d = max(1.0 - d / CC.R_C, 0.0)

        # f_s: SINR proxy (uses interference radius)
        f_s = max(1.0 - d / CC.R_I, 0.0)

        # f_m: velocity similarity
        dv = np.linalg.norm(velocities[i] - velocities[leader])
        f_m = 1.0 / (1.0 + dv)

        # f_q: load balancing
        cluster_size = len(cs.member_indices)
        f_q = 1.0 / (1.0 + cluster_size / CC.N_MAX)

        score = (CC.W_DIST * f_d +
                 CC.W_SINR * f_s +
                 CC.W_MOB  * f_m +
                 CC.W_LOAD * f_q)
        return score

    # ------------------------------------------------------------------
    # Reassociation with Hysteresis
    # ------------------------------------------------------------------
    def _reassociate(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        queues: np.ndarray,
    ):
        """
        For each UAV, check if a better cluster exists.
        Only move if: score_current < THETA_LEAVE AND score_best > THETA_JOIN.
        After each move, cull empties so C_MIN checks are accurate.
        """
        active_cids = self.get_active_cluster_ids()
        if len(active_cids) <= 1:
            return

        for i in range(self.n):
            # Refresh active list (cluster may have died from previous move)
            active_cids = self.get_active_cluster_ids()
            if len(active_cids) <= 1:
                break

            current_cid = self.assignment[i]
            # Skip if current cluster is dead (will be fixed by _fix_orphans)
            if current_cid not in self.clusters or not self.clusters[current_cid].alive:
                continue

            scores = {}
            for cid in active_cids:
                scores[cid] = self._membership_score(i, cid, positions, velocities, queues)

            current_score = scores.get(current_cid, -np.inf)
            best_cid = max(scores, key=scores.get)
            best_score = scores[best_cid]

            if best_cid != current_cid:
                if current_score < CC.THETA_LEAVE and best_score > CC.THETA_JOIN:
                    # Guard: don't empty a cluster if it would violate C_MIN
                    src_cs = self.clusters.get(current_cid)
                    if src_cs and len(src_cs.member_indices) <= 1:
                        if self.num_active_clusters() <= CC.C_MIN:
                            continue  # Skip — would violate C_MIN
                    self._move_uav(i, current_cid, best_cid)
                    self.reassoc_count += 1
                    # Immediately cull the source if it's now empty
                    if src_cs and len(src_cs.member_indices) == 0:
                        src_cs.alive = False

    def _move_uav(self, uav_idx: int, from_cid: int, to_cid: int):
        """Move a UAV from one cluster to another."""
        if from_cid in self.clusters:
            cs_from = self.clusters[from_cid]
            if uav_idx in cs_from.member_indices:
                cs_from.member_indices.remove(uav_idx)
            # If we moved the leader and members remain, trigger emergency handover
            if cs_from.leader_idx == uav_idx and len(cs_from.member_indices) > 0:
                cs_from.leader_idx = cs_from.member_indices[0]
                cs_from.handover_flag = True
                cs_from.handover_cooldown = CC.T_CLUSTER
                self.handover_count += 1

        if to_cid in self.clusters:
            self.clusters[to_cid].member_indices.append(uav_idx)
        self.assignment[uav_idx] = to_cid

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    def _try_splits(self, positions: np.ndarray, velocities: np.ndarray):
        """Split clusters that exceed N_MAX members."""
        if self.num_active_clusters() >= CC.C_MAX:
            return

        to_split = [
            cid for cid, cs in self.clusters.items()
            if cs.alive and len(cs.member_indices) > CC.N_MAX
        ]

        for cid in to_split:
            if self.num_active_clusters() >= CC.C_MAX:
                break
            cs = self.clusters[cid]
            members = cs.member_indices
            leader_pos = positions[cs.leader_idx]

            # Sort members by distance to leader
            dists = [np.linalg.norm(positions[m] - leader_pos) for m in members]
            sorted_members = [m for _, m in sorted(zip(dists, members))]

            # Keep close half, split far half
            mid = len(sorted_members) // 2
            keep = sorted_members[:mid]
            split_off = sorted_members[mid:]

            if len(split_off) < CC.N_MIN or len(keep) < CC.N_MIN:
                continue  # Would create undersized cluster

            # Create new cluster for the far members
            new_cid = self._alloc_cluster_id()
            new_leader = self._elect_leader(split_off, positions, velocities)
            new_cs = ClusterState(
                cluster_id=new_cid, leader_idx=new_leader,
                member_indices=split_off
            )
            self.clusters[new_cid] = new_cs

            # Shrink original
            cs.member_indices = keep
            if cs.leader_idx not in keep:
                cs.leader_idx = self._elect_leader(keep, positions, velocities)

            for m in split_off:
                self.assignment[m] = new_cid

            self.split_count += 1

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def _try_merges(self, positions: np.ndarray, velocities: np.ndarray):
        """Merge undersized clusters with their nearest neighbor."""
        if self.num_active_clusters() <= CC.C_MIN:
            return

        # Re-cull in case split created empties
        self._cull_empty_clusters()

        undersized = [
            cid for cid, cs in self.clusters.items()
            if cs.alive and len(cs.member_indices) < CC.N_MIN
        ]

        for cid in undersized:
            # Strict guard: never merge below C_MIN
            if self.num_active_clusters() <= CC.C_MIN:
                break
            cs = self.clusters.get(cid)
            if cs is None or not cs.alive:
                continue

            # Find nearest alive cluster by leader distance
            best_target = None
            best_dist = np.inf
            for other_cid, other_cs in self.clusters.items():
                if other_cid == cid or not other_cs.alive:
                    continue
                d = np.linalg.norm(
                    positions[cs.leader_idx] - positions[other_cs.leader_idx]
                )
                if d < best_dist:
                    best_dist = d
                    best_target = other_cid

            if best_target is None:
                continue

            # Merge: absorb members into target
            target_cs = self.clusters[best_target]
            target_cs.member_indices.extend(cs.member_indices)
            for m in cs.member_indices:
                self.assignment[m] = best_target

            # Kill the old cluster
            cs.alive = False
            cs.member_indices = []

            self.merge_count += 1

    # ------------------------------------------------------------------
    # Leader Election & Health
    # ------------------------------------------------------------------
    def _elect_leader(
        self,
        members: List[int],
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> int:
        """
        Elect leader by maximizing suitability H_u:
          H_u = a_E*E_u + a_G*centrality_u + a_M*mobility_stability
                - a_Q*queue_proxy - a_R*risk

        For initialization (no queue/risk data), use energy + centrality.
        """
        if len(members) == 1:
            return members[0]

        member_positions = positions[members]
        centroid = member_positions.mean(axis=0)

        best_score = -np.inf
        best_idx = members[0]

        for m in members:
            # Energy
            e = self.energy[m] / CC.E_INIT

            # Centrality: inverse distance to centroid
            d = np.linalg.norm(positions[m] - centroid)
            centrality = 1.0 / (1.0 + d / CC.R_C)

            # Mobility stability: inverse speed
            speed = np.linalg.norm(velocities[m])
            mob_stab = 1.0 / (1.0 + speed / 30.0)

            score = (CC.A_ENERGY * e +
                     CC.A_DEGREE * centrality +
                     CC.A_MOBSTAB * mob_stab)

            if score > best_score:
                best_score = score
                best_idx = m

        return best_idx

    def _check_leader_health(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        queues: np.ndarray,
    ):
        """Check all leaders; trigger handover if health < HEALTH_THRESHOLD."""
        for cid, cs in list(self.clusters.items()):
            if not cs.alive or len(cs.member_indices) <= 1:
                continue

            leader = cs.leader_idx
            health = self._leader_health(leader, cs, positions, velocities, queues)

            if health < CC.HEALTH_THRESHOLD:
                # Elect a new leader excluding the failing one
                candidates = [m for m in cs.member_indices if m != leader]
                if candidates:
                    new_leader = self._elect_leader(candidates, positions, velocities)
                    cs.leader_idx = new_leader
                    cs.handover_flag = True
                    cs.handover_cooldown = CC.T_CLUSTER
                    self.handover_count += 1

    def _leader_health(
        self,
        leader: int,
        cs: ClusterState,
        positions: np.ndarray,
        velocities: np.ndarray,
        queues: np.ndarray,
    ) -> float:
        """
        Compute health score for a cluster-head.
        H = a_E*E + a_G*deg + a_M*M - a_Q*q - a_R*risk
        """
        e = self.energy[leader] / CC.E_INIT
        speed = np.linalg.norm(velocities[leader])
        mob_stab = 1.0 / (1.0 + speed / 30.0)
        q = queues[leader] / 100.0 if queues is not None else 0.0

        # Degree: count of neighboring cluster leaders within R_I
        deg = 0
        for other_cid, other_cs in self.clusters.items():
            if other_cid == cs.cluster_id or not other_cs.alive:
                continue
            d = np.linalg.norm(positions[leader] - positions[other_cs.leader_idx])
            if d <= CC.R_I:
                deg += 1
        deg_norm = deg / max(self.num_active_clusters() - 1, 1)

        # Risk: low energy is risky
        risk = max(1.0 - e, 0.0)

        health = (CC.A_ENERGY * e +
                  CC.A_DEGREE * deg_norm +
                  CC.A_MOBSTAB * mob_stab -
                  CC.A_QUEUE * q -
                  CC.A_RISK * risk)
        return health

    # ------------------------------------------------------------------
    # Interference Graph
    # ------------------------------------------------------------------
    def _build_interference_graph(self, positions: np.ndarray):
        """
        Build G_t = (V_t, E_t) where V_t = alive clusters.
        Edge (k,m) exists iff ||p_{L_k} - p_{L_m}|| <= R_I
        """
        self._adj[:] = False
        active = self.get_active_cluster_ids()

        for i_idx, cid_a in enumerate(active):
            for j_idx in range(i_idx + 1, len(active)):
                cid_b = active[j_idx]
                leader_a = self.clusters[cid_a].leader_idx
                leader_b = self.clusters[cid_b].leader_idx
                d = np.linalg.norm(positions[leader_a] - positions[leader_b])
                if d <= CC.R_I:
                    # Use cluster_id as index up to C_MAX
                    self._adj[cid_a % CC.C_MAX, cid_b % CC.C_MAX] = True
                    self._adj[cid_b % CC.C_MAX, cid_a % CC.C_MAX] = True

    def get_interference_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns edge_index (2, E) in COO format for PyG compatibility.
        Node indices are remapped to [0, c-1] for the alive clusters.
        Also returns the alive_cid_list for mapping back.
        """
        active = self.get_active_cluster_ids()
        # Build a local-index adjacency
        c = len(active)
        cid_to_local = {cid: idx for idx, cid in enumerate(active)}

        rows, cols = [], []
        for i_idx, cid_a in enumerate(active):
            for j_idx in range(i_idx + 1, len(active)):
                cid_b = active[j_idx]
                if self._adj[cid_a % CC.C_MAX, cid_b % CC.C_MAX]:
                    li = cid_to_local[cid_a]
                    lj = cid_to_local[cid_b]
                    rows.extend([li, lj])
                    cols.extend([lj, li])

        if rows:
            edge_index = np.array([rows, cols], dtype=np.int64)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)

        return edge_index, active

    # ------------------------------------------------------------------
    # Observation Builder
    # ------------------------------------------------------------------
    def get_cluster_obs(
        self,
        cid: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        queues: np.ndarray,
        collision_ratios: np.ndarray,
        adjacency_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build local observation o_k(t) for cluster-head k.

        Features (OBS_DIM_CLUSTER = 24):
          0: cluster_size / N_MAX
          1: agg_queue (sum of member queues, normalized)
          2: avg delay / MAX_DELAY_MS
          3: local offered load (cluster_size / n)
          4: local SINR proxy (1 - avg_dist_to_leader / R_C)
          5: local collision pressure
          6: leader energy (normalized)
          7: leader speed (normalized)
          8: graph degree (normalized)
          9-11: neighbor summary (avg size, avg queue, avg collision)
          12: leader position x / area_x
          13: leader position y / area_y
          14: leader health
          15: handover flag (0 or 1)
          16: intra-cluster pressure score
          17: inter-cluster pressure score
          18: coordination backlog
          19: subordinate service demand ratio
          20: coordination-needy neighbor ratio
          21: previous rho
          22: recent T1 utilization
          23: recent T2 utilization
        """
        cs = self.clusters.get(cid)
        if cs is None or not cs.alive:
            return np.zeros(CC.OBS_DIM_CLUSTER, dtype=np.float32)

        members = cs.member_indices
        leader = cs.leader_idx
        n_members = len(members)

        # Feature 0: cluster size
        f0 = n_members / CC.N_MAX

        # Feature 1: aggregated queue
        agg_q = float(queues[members].sum()) if len(members) > 0 else float(cs.agg_queue)
        f1 = min(agg_q / max(CC.N_MAX * 100.0, 1.0), 1.0)

        # Feature 2: average delay proxy / runtime delay
        if cs.agg_delay > 0.0:
            f2 = min(cs.agg_delay / CC.MAX_DELAY_MS, 1.0)
        else:
            f2 = float(np.mean(collision_ratios[members])) if len(members) > 0 else 0.0

        # Feature 3: local offered load fraction
        f3 = n_members / max(self.n, 1)

        # Feature 4: SINR proxy (avg distance to leader)
        if n_members > 1:
            member_dists = np.linalg.norm(
                positions[members] - positions[leader], axis=1
            )
            avg_dist = np.mean(member_dists)
            f4 = max(1.0 - avg_dist / CC.R_C, 0.0)
        else:
            f4 = 1.0

        # Feature 5: cluster collision pressure
        if cs.agg_collisions > 0:
            f5 = min(cs.agg_collisions / max(CC.MAX_COLLISIONS, 1), 1.0)
        else:
            f5 = float(np.mean(collision_ratios[members])) if len(members) > 0 else 0.0

        # Feature 6: leader energy
        f6 = self.energy[leader] / CC.E_INIT

        # Feature 7: leader speed
        f7 = min(np.linalg.norm(velocities[leader]) / 30.0, 1.0)

        # Feature 8: graph degree
        active = self.get_active_cluster_ids()
        adj = adjacency_override if adjacency_override is not None else self._adj
        deg = 0
        for other_cid in active:
            if other_cid != cid and adj[cid % CC.C_MAX, other_cid % CC.C_MAX]:
                deg += 1
        f8 = deg / max(len(active) - 1, 1)

        # Features 9-11: neighbor summary
        neighbor_sizes = []
        neighbor_queues = []
        neighbor_collisions = []
        coord_needy_neighbors = 0
        for other_cid in active:
            if other_cid == cid:
                continue
            if adj[cid % CC.C_MAX, other_cid % CC.C_MAX]:
                other_cs = self.clusters[other_cid]
                neighbor_sizes.append(len(other_cs.member_indices) / CC.N_MAX)
                n_members_other = other_cs.member_indices
                if n_members_other:
                    neighbor_queues.append(
                        min(queues[n_members_other].sum() / (CC.N_MAX * 100.0), 1.0)
                    )
                    if other_cs.agg_collisions > 0:
                        neighbor_collisions.append(
                            min(other_cs.agg_collisions / max(CC.MAX_COLLISIONS, 1), 1.0)
                        )
                    else:
                        neighbor_collisions.append(
                            float(np.mean(collision_ratios[n_members_other]))
                        )
                else:
                    neighbor_queues.append(0.0)
                    neighbor_collisions.append(0.0)
                if other_cs.coord_backlog > 0.0 or other_cs.relay_demand > 0.0 or other_cs.handover_flag:
                    coord_needy_neighbors += 1

        f9 = float(np.mean(neighbor_sizes)) if neighbor_sizes else 0.0
        f10 = float(np.mean(neighbor_queues)) if neighbor_queues else 0.0
        f11 = float(np.mean(neighbor_collisions)) if neighbor_collisions else 0.0

        # Features 12-13: leader spatial position (normalized)
        from configs import config as params
        f12 = positions[leader][0] / max(params.AREA_X, 1)
        f13 = positions[leader][1] / max(params.AREA_Y, 1)

        # Feature 14: leader health
        f14 = self._leader_health(leader, cs, positions, velocities, queues)

        # Feature 15: handover flag
        f15 = 1.0 if cs.handover_flag else 0.0

        subordinate_count = max(n_members - 1, 1)
        subordinate_queues = [m for m in members if m != leader]
        if subordinate_queues:
            service_need_ratio = sum(1 for m in subordinate_queues if queues[m] > 1e-6) / subordinate_count
        else:
            service_need_ratio = 0.0

        coord_backlog_norm = min(cs.coord_backlog / max(CC.MAX_COORD_BACKLOG, 1e-9), 1.0)
        relay_norm = min(cs.relay_demand / max(CC.MAX_RELAY_DEMAND, 1e-9), 1.0)
        ctrl_norm = min(cs.local_ctrl_demand / max(CC.MAX_LOCAL_CTRL_DEMAND, 1e-9), 1.0)
        coord_neighbor_ratio = coord_needy_neighbors / max(deg, 1) if deg > 0 else 0.0

        intra_pressure = min(
            0.30 * f1 +
            0.20 * service_need_ratio +
            0.20 * f2 +
            0.15 * f5 +
            0.10 * (1.0 - f4) +
            0.05 * ctrl_norm,
            1.0,
        )
        inter_pressure = min(
            0.30 * coord_backlog_norm +
            0.20 * coord_neighbor_ratio +
            0.20 * f8 +
            0.15 * relay_norm +
            0.10 * f11 +
            0.05 * f15,
            1.0,
        )

        obs = np.array([
            f0, f1, f2, f3, f4, f5, f6, f7, f8,
            f9, f10, f11, f12, f13, f14, f15,
            intra_pressure, inter_pressure, coord_backlog_norm,
            service_need_ratio, coord_neighbor_ratio,
            min(cs.recent_rho, 1.0),
            min(cs.recent_t1_util, 1.0),
            min(cs.recent_t2_util, 1.0),
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    # External Failure Injection
    # ------------------------------------------------------------------
    def trigger_leader_failure(self, cid: int, positions: np.ndarray,
                                velocities: np.ndarray):
        """Simulate leader failure for cluster cid. Elect a new leader."""
        cs = self.clusters.get(cid)
        if cs is None or not cs.alive:
            return

        failed_leader = cs.leader_idx
        candidates = [m for m in cs.member_indices if m != failed_leader]
        if candidates:
            cs.leader_idx = self._elect_leader(candidates, positions, velocities)
            cs.handover_flag = True
            cs.handover_cooldown = CC.T_CLUSTER
            self.handover_count += 1
        else:
            # Last member failed — kill cluster
            cs.alive = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_active_cluster_ids(self) -> List[int]:
        """Return sorted list of alive cluster IDs."""
        return sorted([cid for cid, cs in self.clusters.items() if cs.alive])

    def num_active_clusters(self) -> int:
        return sum(1 for cs in self.clusters.values() if cs.alive)

    def get_members(self, cid: int) -> List[int]:
        cs = self.clusters.get(cid)
        return cs.member_indices if cs and cs.alive else []

    def get_leader(self, cid: int) -> int:
        cs = self.clusters.get(cid)
        return cs.leader_idx if cs and cs.alive else -1

    def update_cluster_runtime_state(
        self,
        cid: int,
        *,
        agg_queue: Optional[float] = None,
        agg_throughput: Optional[float] = None,
        agg_delay: Optional[float] = None,
        agg_collisions: Optional[int] = None,
        agg_drops: Optional[int] = None,
        local_ctrl_demand: Optional[float] = None,
        coord_backlog: Optional[float] = None,
        relay_demand: Optional[float] = None,
        recent_rho: Optional[float] = None,
        recent_t1_util: Optional[float] = None,
        recent_t2_util: Optional[float] = None,
        recent_coord_success: Optional[float] = None,
    ):
        """Update runtime execution state for a live cluster."""
        cs = self.clusters.get(cid)
        if cs is None or not cs.alive:
            return
        if agg_queue is not None:
            cs.agg_queue = float(agg_queue)
        if agg_throughput is not None:
            cs.agg_throughput = float(agg_throughput)
        if agg_delay is not None:
            cs.agg_delay = float(agg_delay)
        if agg_collisions is not None:
            cs.agg_collisions = int(agg_collisions)
        if agg_drops is not None:
            cs.agg_drops = int(agg_drops)
        if local_ctrl_demand is not None:
            cs.local_ctrl_demand = float(local_ctrl_demand)
        if coord_backlog is not None:
            cs.coord_backlog = float(coord_backlog)
        if relay_demand is not None:
            cs.relay_demand = float(relay_demand)
        if recent_rho is not None:
            cs.recent_rho = float(recent_rho)
        if recent_t1_util is not None:
            cs.recent_t1_util = float(recent_t1_util)
        if recent_t2_util is not None:
            cs.recent_t2_util = float(recent_t2_util)
        if recent_coord_success is not None:
            cs.recent_coord_success = float(recent_coord_success)

    def _alloc_cluster_id(self) -> int:
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        return cid

    def _drain_energy(self):
        """Simple linear energy drain per step."""
        self.energy -= CC.E_IDLE_COST
        self.energy = np.clip(self.energy, 0.0, CC.E_INIT)

    def _cull_empty_clusters(self):
        """Mark clusters with 0 members as dead.
        This prevents ghost clusters from distorting merge/split counts."""
        for cid, cs in self.clusters.items():
            if cs.alive and len(cs.member_indices) == 0:
                cs.alive = False

    def _fix_orphans(self, positions: np.ndarray, velocities: np.ndarray):
        """Assign any unassigned UAV to the nearest alive cluster."""
        active = self.get_active_cluster_ids()
        if not active:
            return

        for i in range(self.n):
            cid = self.assignment[i]
            if cid not in self.clusters or not self.clusters[cid].alive:
                # Orphan — assign to nearest leader
                best_cid = active[0]
                best_dist = np.inf
                for candidate_cid in active:
                    leader = self.clusters[candidate_cid].leader_idx
                    d = np.linalg.norm(positions[i] - positions[leader])
                    if d < best_dist:
                        best_dist = d
                        best_cid = candidate_cid
                self.assignment[i] = best_cid
                if i not in self.clusters[best_cid].member_indices:
                    self.clusters[best_cid].member_indices.append(i)

    def _enforce_c_min(self, positions: np.ndarray, velocities: np.ndarray):
        """
        If active cluster count < C_MIN, split the largest cluster(s)
        until we reach C_MIN or run out of splittable clusters.
        This makes the system self-healing.
        """
        while self.num_active_clusters() < CC.C_MIN:
            # Find the largest alive cluster
            active = self.get_active_cluster_ids()
            if not active:
                break
            largest_cid = max(active, key=lambda c: len(self.clusters[c].member_indices))
            largest_cs = self.clusters[largest_cid]

            # Need at least 2*N_MIN members to split (or at least 4 to be safe)
            if len(largest_cs.member_indices) < max(2 * CC.N_MIN, 4):
                break  # Can't split further

            members = largest_cs.member_indices
            leader_pos = positions[largest_cs.leader_idx]
            dists = [np.linalg.norm(positions[m] - leader_pos) for m in members]
            sorted_members = [m for _, m in sorted(zip(dists, members))]

            mid = len(sorted_members) // 2
            keep = sorted_members[:mid]
            split_off = sorted_members[mid:]

            if len(keep) < CC.N_MIN or len(split_off) < CC.N_MIN:
                break

            new_cid = self._alloc_cluster_id()
            new_leader = self._elect_leader(split_off, positions, velocities)
            new_cs = ClusterState(
                cluster_id=new_cid, leader_idx=new_leader,
                member_indices=split_off
            )
            self.clusters[new_cid] = new_cs

            largest_cs.member_indices = keep
            if largest_cs.leader_idx not in keep:
                largest_cs.leader_idx = self._elect_leader(keep, positions, velocities)

            for m in split_off:
                self.assignment[m] = new_cid

            self.split_count += 1

    def get_diagnostics(self) -> dict:
        """Return a dict of clustering health metrics for logging."""
        active = self.get_active_cluster_ids()
        sizes = [len(self.clusters[cid].member_indices) for cid in active]
        degrees = {cid: 0 for cid in active}
        edge_count = 0
        for i_idx, cid_a in enumerate(active):
            for j_idx in range(i_idx + 1, len(active)):
                cid_b = active[j_idx]
                if self._adj[cid_a % CC.C_MAX, cid_b % CC.C_MAX]:
                    degrees[cid_a] += 1
                    degrees[cid_b] += 1
                    edge_count += 1
        if len(active) > 1:
            graph_density = (2.0 * edge_count) / (len(active) * (len(active) - 1))
        else:
            graph_density = 0.0
        return {
            "num_clusters": len(active),
            "cluster_sizes": sizes,
            "min_cluster_size": min(sizes) if sizes else 0,
            "max_cluster_size": max(sizes) if sizes else 0,
            "mean_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
            "cluster_size_std": float(np.std(sizes)) if sizes else 0.0,
            "splits": self.split_count,
            "merges": self.merge_count,
            "reassociations": self.reassoc_count,
            "handovers": self.handover_count,
            "avg_graph_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
            "graph_density": float(graph_density),
            "mean_energy": float(np.mean(self.energy)),
            "total_uavs_assigned": sum(sizes),
        }

    def verify_invariants(self) -> List[str]:
        """
        Sanity-check all clustering invariants. Returns list of violations.
        Empty list = all OK.
        """
        errors = []

        # 1. Every UAV belongs to exactly one alive cluster
        assigned_counts = np.zeros(self.n, dtype=int)
        for cid, cs in self.clusters.items():
            if not cs.alive:
                continue
            for m in cs.member_indices:
                if m < 0 or m >= self.n:
                    errors.append(f"Cluster {cid}: invalid member index {m}")
                else:
                    assigned_counts[m] += 1

        orphans = np.where(assigned_counts == 0)[0]
        if len(orphans) > 0:
            errors.append(f"Orphan UAVs (no cluster): {orphans.tolist()}")

        duplicates = np.where(assigned_counts > 1)[0]
        if len(duplicates) > 0:
            errors.append(f"UAVs in multiple clusters: {duplicates.tolist()}")

        # 2. assignment array consistency
        for i in range(self.n):
            cid = self.assignment[i]
            cs = self.clusters.get(cid)
            if cs is None or not cs.alive:
                errors.append(f"UAV {i} assigned to dead/missing cluster {cid}")
            elif i not in cs.member_indices:
                errors.append(f"UAV {i} assigned to cluster {cid} but not in member list")

        # 3. Each alive cluster has a valid leader
        for cid, cs in self.clusters.items():
            if not cs.alive:
                continue
            if cs.leader_idx not in cs.member_indices:
                errors.append(f"Cluster {cid}: leader {cs.leader_idx} not in members")

        # 4. Cluster count bounds
        c = self.num_active_clusters()
        if c < CC.C_MIN:
            errors.append(f"Too few clusters: {c} < C_MIN={CC.C_MIN}")
        if c > CC.C_MAX:
            errors.append(f"Too many clusters: {c} > C_MAX={CC.C_MAX}")

        # 5. Total assigned == n
        total = sum(
            len(cs.member_indices) for cs in self.clusters.values() if cs.alive
        )
        if total != self.n:
            errors.append(f"Total assigned {total} != n={self.n}")

        return errors
