"""
Spatial clustering of delivery stops with DBSCAN.

DBSCAN (Ester et al., 1996) groups points that are density-reachable in an
epsilon neighborhood. Using the Haversine metric treats the Earth as a sphere,
which is appropriate for city-scale routing.

Complexity (sklearn DBSCAN, BallTree + haversine):
- Standard analysis: roughly O(n log n) for low-dimensional spatial data with
  neighborhood indexing; worst-case can degrade toward O(n^2) for adversarial data.
- Memory: typically O(n) for the tree and label arrays.

 eps is specified in **radians** when metric='haversine'; we convert from km.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from utils import EARTH_RADIUS_KM


def cluster_deliveries_dbscan(
    stops: list[dict[str, Any]],
    *,
    eps_km: float = 1.2,
    min_samples: int = 2,
    coordinate_field: str = "drop",
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Cluster stops by coordinates (default: drop location).

    Returns:
      - labels: numpy array of shape (n_stops,), -1 = noise in DBSCAN terms
      - stop_index -> cluster_id mapping (noise mapped to singleton clusters starting at max_label+1)

    For routing we remap noise labels to unique cluster ids so each point still belongs
    to a route partition.
    """
    if not stops:
        return np.array([], dtype=int), {}

    coords = np.array([tuple(s[coordinate_field]) for s in stops], dtype=float)
    # sklearn expects radians for haversine: shape (n, 2) with lat, lon order.
    coords_rad = np.radians(coords)
    eps_rad = float(eps_km) / EARTH_RADIUS_KM

    clustering = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = clustering.fit_predict(coords_rad)

    max_lab = int(labels.max()) if labels.size else -1
    next_id = max_lab + 1
    remapped = labels.copy()
    for i in range(len(remapped)):
        if remapped[i] == -1:
            remapped[i] = next_id
            next_id += 1

    idx_to_cluster = {i: int(remapped[i]) for i in range(len(stops))}
    return remapped, idx_to_cluster
