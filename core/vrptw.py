"""
Vehicle Routing Problem with Time Windows (VRPTW) — feasibility validation.

Given a fixed stop sequence for a vehicle, we simulate forward in time:
travel time from Haversine / average speed, service time at each stop, and
hard time-window constraints. Capacity is cumulative parcel weight.

This module does not solve VRPTW to optimality (that would require column
generation or metaheuristics at city scale); it **validates** proposed routes.

Typical exact VRPTW algorithms are exponential in the worst case; this checker
is O(k) in the number of stops k on the route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils import haversine_km


@dataclass
class VRPTWConfig:
    avg_speed_kmh: float = 25.0
    service_time_min: float = 5.0
    # Relative to the earliest window start across the route (anchors the simulation clock).
    max_route_duration_min: float = 720.0  # 12 hours of operational slack for research demos


def travel_time_min(a: tuple[float, float], b: tuple[float, float], speed_kmh: float) -> float:
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be positive")
    dist_km = haversine_km(a, b)
    return (dist_km / speed_kmh) * 60.0


def validate_route_feasibility(
    driver_location: tuple[float, float],
    drop_coords_in_order: list[tuple[float, float]],
    stop_meta: list[dict[str, Any]],
    vehicle_capacity: float,
    cfg: VRPTWConfig | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check time windows, capacity, and max duration for a single vehicle route.

    The vehicle starts at `driver_location` and visits `drop_coords_in_order`.
    `stop_meta` has one entry per drop (same order), each with:
      - time_window: [start_min, end_min] (minutes)
      - parcel_weight: float
    """
    cfg = cfg or VRPTWConfig()
    if len(drop_coords_in_order) != len(stop_meta):
        raise ValueError("drop_coords_in_order and stop_meta length mismatch")

    # Anchor all windows to the earliest feasible minute so feasibility uses relative time.
    anchor = min(float(m["time_window"][0]) for m in stop_meta)

    load = 0.0
    time_min = 0.0  # minutes since `anchor`
    violations: list[str] = []
    prev = driver_location

    for i, (coord, meta) in enumerate(zip(drop_coords_in_order, stop_meta)):
        dt = travel_time_min(prev, coord, cfg.avg_speed_kmh)
        time_min += dt
        prev = coord

        load += float(meta["parcel_weight"])
        if load > vehicle_capacity + 1e-6:
            violations.append(f"capacity_exceeded@{i}")
        tw = meta["time_window"]
        earliest = float(tw[0]) - anchor
        latest = float(tw[1]) - anchor
        arrival = time_min
        if arrival < earliest:
            time_min = earliest
            arrival = earliest
        if arrival > latest:
            violations.append(f"time_window_late@{i}")
        time_min += cfg.service_time_min

    if time_min > cfg.max_route_duration_min:
        violations.append("max_duration")

    ok = len(violations) == 0
    return ok, {
        "end_time_min": time_min,
        "max_load": load,
        "violations": violations,
    }


def slack_time_windows(
    stops: list[dict[str, Any]],
    *,
    pad_min: float = 30.0,
) -> None:
    """Mutate stops by widening windows (research aid when intersection tightened windows)."""
    for s in stops:
        lo, hi = float(s["time_window"][0]), float(s["time_window"][1])
        s["time_window"] = [lo - pad_min, hi + pad_min]
