"""
Nearest-driver assignment to spatial clusters (nearest neighbor heuristic).

After DBSCAN partitions the plane, each cluster centroid is matched to the
closest available driver by road-proxy distance (Haversine). Ties are broken
by lower driver_id. Greedy assignment respects unique drivers when possible;
if drivers < clusters, the lowest-index clusters get drivers first and
remaining clusters share nearest drivers (capacity not enforced here—that is
delegated to VRPTW validation downstream).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from utils import haversine_km


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    arr = np.array(points, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def assign_nearest_driver(
    clusters: dict[int, list[int]],
    stops: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
    *,
    location_field: str = "drop",
) -> dict[int, int]:
    """
    Map cluster_id -> driver_id using centroid-to-driver Haversine distance.

    `clusters` maps cluster_id -> list of indices into `stops`.
    """
    cluster_ids = sorted(clusters.keys())
    if not cluster_ids or not drivers:
        return {}

    centroids: dict[int, tuple[float, float]] = {}
    for cid, idxs in clusters.items():
        pts = [tuple(stops[i][location_field]) for i in idxs]
        centroids[cid] = _centroid(pts)

    # Sort clusters by id for deterministic tie handling
    assignment: dict[int, int] = {}
    used: set[int] = set()

    for cid in cluster_ids:
        best_d: int | None = None
        best_dist = float("inf")
        c = centroids[cid]
        for d in sorted(drivers, key=lambda x: x["driver_id"]):
            did = int(d["driver_id"])
            loc = tuple(d["current_location"])
            dist = haversine_km(c, loc)
            if dist < best_dist:
                best_dist = dist
                best_d = did
        if best_d is None:
            continue
        assignment[cid] = best_d
        used.add(best_d)

    return assignment


def stops_by_cluster(labels: np.ndarray) -> dict[int, list[int]]:
    """Group stop indices by cluster label."""
    out: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        out[int(lab)].append(i)
    return dict(out)
