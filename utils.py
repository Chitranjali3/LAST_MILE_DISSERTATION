"""
Shared utilities for the delivery optimization engine.

Includes geodesic distance, synthetic data generation, and plotting helpers.
"""

from __future__ import annotations

import math
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Earth mean radius in km (WGS84 approximation for Haversine).
EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two WGS84 lat/lon points in kilometers."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def route_length_km(points: list[tuple[float, float]]) -> float:
    """Sum of consecutive Haversine segments; empty or single point -> 0."""
    if len(points) < 2:
        return 0.0
    return sum(haversine_km(points[i], points[i + 1]) for i in range(len(points) - 1))


def minutes_from_hhmm(hhmm: int) -> int:
    """Convert integer HHMM (e.g. 930) to minutes from midnight."""
    h = hhmm // 100
    m = hhmm % 100
    return h * 60 + m


def synthetic_orders(
    n: int = 24,
    seed: int | None = 42,
    region_center: tuple[float, float] = (12.97, 77.59),  # Bangalore-scale spread
    spread_deg: float = 0.08,
) -> list[dict[str, Any]]:
    """
    Generate synthetic orders with pickup/drop, time windows, and weights.

    Intentionally places duplicate drops and collinear triplets for experiments.
    """
    rng = random.Random(seed)
    orders: list[dict[str, Any]] = []
    # Duplicated building delivery (same drop for two users).
    dup_drop = (region_center[0] + 0.01, region_center[1] + 0.01)
    orders.append(
        {
            "order_id": 1,
            "user_id": 101,
            "pickup": (region_center[0], region_center[1]),
            "drop": dup_drop,
            "time_window": [minutes_from_hhmm(900), minutes_from_hhmm(1200)],
            "parcel_weight": 2.0,
        }
    )
    orders.append(
        {
            "order_id": 2,
            "user_id": 102,
            "pickup": (region_center[0], region_center[1]),
            "drop": dup_drop,
            "time_window": [minutes_from_hhmm(920), minutes_from_hhmm(1230)],
            "parcel_weight": 1.5,
        }
    )

    next_id = 3
    while len(orders) < n:
        drop = (
            region_center[0] + rng.uniform(-spread_deg, spread_deg),
            region_center[1] + rng.uniform(-spread_deg, spread_deg),
        )
        pickup = (
            region_center[0] + rng.uniform(-0.02, 0.02),
            region_center[1] + rng.uniform(-0.02, 0.02),
        )
        tw_start = rng.randint(9 * 60, 14 * 60)
        tw_end = tw_start + rng.randint(60, 240)
        orders.append(
            {
                "order_id": next_id,
                "user_id": 200 + next_id,
                "pickup": pickup,
                "drop": drop,
                "time_window": [tw_start, tw_end],
                "parcel_weight": round(rng.uniform(0.5, 8.0), 2),
            }
        )
        next_id += 1

    # Collinear-style triple (A, C, B along a rough line) for Rule 2 demos.
    a = (region_center[0] + 0.02, region_center[1] + 0.00)
    c = (region_center[0] + 0.035, region_center[1] + 0.005)
    b = (region_center[0] + 0.05, region_center[1] + 0.01)
    for oid, pt in [(next_id, a), (next_id + 1, c), (next_id + 2, b)]:
        if len(orders) >= n:
            break
        orders.append(
            {
                "order_id": oid,
                "user_id": 300 + oid,
                "pickup": (region_center[0], region_center[1]),
                "drop": pt,
                "time_window": [10 * 60 + oid, 16 * 60],
                "parcel_weight": 1.0,
            }
        )

    return orders[:n]


def synthetic_drivers(
    k: int = 4,
    seed: int | None = 42,
    region_center: tuple[float, float] = (12.97, 77.59),
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    drivers = []
    for i in range(k):
        drivers.append(
            {
                "driver_id": i + 1,
                "current_location": (
                    region_center[0] + rng.uniform(-0.03, 0.03),
                    region_center[1] + rng.uniform(-0.03, 0.03),
                ),
                "capacity": rng.choice([20, 25, 30, 35]),
            }
        )
    return drivers


def plot_clusters_and_routes(
    stop_points: list[tuple[float, float]],
    cluster_labels: np.ndarray,
    routes: list[list[tuple[float, float]]],
    title: str,
    save_path: str | None = "output_clusters_routes.png",
) -> None:
    """Scatter stops colored by DBSCAN cluster; overlay polylines for routes."""
    plt.figure(figsize=(10, 8))
    labs = np.array(cluster_labels)
    plt.scatter(
        [p[1] for p in stop_points],
        [p[0] for p in stop_points],
        c=labs,
        cmap="tab10",
        s=40,
        edgecolors="k",
        linewidths=0.3,
    )
    for ri, poly in enumerate(routes):
        if len(poly) < 2:
            continue
        lons = [p[1] for p in poly]
        lats = [p[0] for p in poly]
        plt.plot(lons, lats, "-o", linewidth=1.5, markersize=4, label=f"Route {ri+1}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if routes:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_before_after(
    naive_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    optimized_routes: list[list[tuple[float, float]]],
    save_path: str | None = "output_before_after.png",
) -> None:
    """
    Naive: many thin segments (direct out-and-back style).
    Optimized: longer continuous chains per driver.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for a, b in naive_segments:
        axes[0].plot([a[1], b[1]], [a[0], b[0]], "r-", alpha=0.5, linewidth=1)
        axes[0].scatter([a[1], b[1]], [a[0], b[0]], c="k", s=15)
    axes[0].set_title("Naive (independent legs)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(True, alpha=0.3)

    for poly in optimized_routes:
        if len(poly) < 2:
            continue
        axes[1].plot([p[1] for p in poly], [p[0] for p in poly], "-o", linewidth=1.5, markersize=4)
    axes[1].set_title("Optimized multi-stop routes")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
