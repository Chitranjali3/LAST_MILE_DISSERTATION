"""
Shared utilities for the delivery optimization engine.

Includes geodesic distance, synthetic data generation, and plotting helpers.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from map_basemap import pad_lonlat_extent, try_osm_basemap

# Earth mean radius in km (WGS84 approximation for Haversine).
EARTH_RADIUS_KM = 6371.0088

# Bhubaneswar-area defaults for synthetic data and the visual OSRM app.
# Matches Geofabrik's ``eastern-zone-latest`` OSRM extract (Odisha lies in that zone).
ODISHA_REGION_CENTER: tuple[float, float] = (20.2961, 85.8245)


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
    region_center: tuple[float, float] = ODISHA_REGION_CENTER,
    spread_deg: float = 0.08,
) -> list[dict[str, Any]]:
    """
    Generate synthetic orders with pickup/drop, time windows, and weights.

    Intentionally places duplicate drops and collinear triplets for experiments.
    Defaults cluster around ``ODISHA_REGION_CENTER`` (Bhubaneswar-scale) so points
    lie inside Geofabrik's ``eastern-zone-latest`` OSRM extract.
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
    region_center: tuple[float, float] = ODISHA_REGION_CENTER,
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


def _strip_header(name: str) -> str:
    return name.lstrip("\ufeff").strip()


def _parse_csv_cell_float(key: str, raw: str, *, path: Path, row_num: int) -> float:
    s = (raw or "").strip()
    if not s:
        raise ValueError(f"{path!s} row {row_num}: empty value for {key!r}")
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"{path!s} row {row_num}: invalid float for {key!r}: {raw!r}") from e


def _parse_csv_cell_int(key: str, raw: str, *, path: Path, row_num: int) -> int:
    s = (raw or "").strip()
    if not s:
        raise ValueError(f"{path!s} row {row_num}: empty value for {key!r}")
    try:
        return int(float(s))
    except ValueError as e:
        raise ValueError(f"{path!s} row {row_num}: invalid int for {key!r}: {raw!r}") from e


def load_orders_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load orders from CSV into the dict shape used by merge/routing.

    Required columns:
      order_id, user_id, pickup_lat, pickup_lon, drop_lat, drop_lon,
      tw_start_min, tw_end_min, parcel_weight

    Optional column (exactly nine core columns OR add as tenth column):
      preferred_minute — minutes from midnight (VRPTW ordering); leave blank to omit.

    Coordinates are (lat, lon) in WGS84; sample files cluster around ``ODISHA_REGION_CENTER``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"orders CSV not found: {p}")

    expected_core = (
        "order_id",
        "user_id",
        "pickup_lat",
        "pickup_lon",
        "drop_lat",
        "drop_lon",
        "tw_start_min",
        "tw_end_min",
        "parcel_weight",
    )
    pref_name = "preferred_minute"
    orders: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{p}: missing header row")
        headers = [_strip_header(h) for h in reader.fieldnames if h is not None]
        has_pref_col: bool
        if headers == list(expected_core):
            has_pref_col = False
        elif headers == list(expected_core) + [pref_name]:
            has_pref_col = True
        else:
            raise ValueError(
                f"{p}: expected header {list(expected_core)!r} "
                f"or {list(expected_core) + [pref_name]!r}, got {headers!r}"
            )

        for row_num, raw in enumerate(reader, start=2):
            row = {_strip_header(k): (v.strip() if v else "") for k, v in raw.items() if k is not None}
            if not row or all(not v for v in row.values()):
                continue
            oid = _parse_csv_cell_int("order_id", row.get("order_id", ""), path=p, row_num=row_num)
            if oid in seen_ids:
                raise ValueError(f"{p} row {row_num}: duplicate order_id {oid}")
            seen_ids.add(oid)
            uid = _parse_csv_cell_int("user_id", row.get("user_id", ""), path=p, row_num=row_num)
            plat = _parse_csv_cell_float("pickup_lat", row.get("pickup_lat", ""), path=p, row_num=row_num)
            plon = _parse_csv_cell_float("pickup_lon", row.get("pickup_lon", ""), path=p, row_num=row_num)
            dlat = _parse_csv_cell_float("drop_lat", row.get("drop_lat", ""), path=p, row_num=row_num)
            dlon = _parse_csv_cell_float("drop_lon", row.get("drop_lon", ""), path=p, row_num=row_num)
            tw0 = _parse_csv_cell_int("tw_start_min", row.get("tw_start_min", ""), path=p, row_num=row_num)
            tw1 = _parse_csv_cell_int("tw_end_min", row.get("tw_end_min", ""), path=p, row_num=row_num)
            if tw0 > tw1:
                raise ValueError(
                    f"{p} row {row_num}: tw_start_min ({tw0}) must be <= tw_end_min ({tw1})"
                )
            weight = _parse_csv_cell_float(
                "parcel_weight", row.get("parcel_weight", ""), path=p, row_num=row_num
            )
            rec: dict[str, Any] = {
                "order_id": oid,
                "user_id": uid,
                "pickup": (plat, plon),
                "drop": (dlat, dlon),
                "time_window": [float(tw0), float(tw1)],
                "parcel_weight": weight,
            }
            if has_pref_col:
                pref_raw = row.get(pref_name, "")
                if pref_raw:
                    pm = _parse_csv_cell_int(pref_name, pref_raw, path=p, row_num=row_num)
                    if not (0 <= pm < 24 * 60):
                        raise ValueError(
                            f"{p} row {row_num}: preferred_minute must be in [0, 1439], got {pm}"
                        )
                    rec["preferred_minute"] = float(pm)
            orders.append(rec)

    if not orders:
        raise ValueError(f"{p}: no data rows after header")
    orders.sort(key=lambda o: int(o["order_id"]))
    return orders


def load_drivers_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load drivers from CSV: driver_id, current_lat, current_lon, capacity."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"drivers CSV not found: {p}")

    expected = ("driver_id", "current_lat", "current_lon", "capacity")
    drivers: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{p}: missing header row")
        headers = [_strip_header(h) for h in reader.fieldnames]
        if headers != list(expected):
            raise ValueError(f"{p}: expected header {list(expected)!r}, got {headers!r}")

        for row_num, raw in enumerate(reader, start=2):
            row = {_strip_header(k): (v.strip() if v else "") for k, v in raw.items() if k is not None}
            if not row or all(not v for v in row.values()):
                continue
            did = _parse_csv_cell_int("driver_id", row.get("driver_id", ""), path=p, row_num=row_num)
            if did in seen_ids:
                raise ValueError(f"{p} row {row_num}: duplicate driver_id {did}")
            seen_ids.add(did)
            lat = _parse_csv_cell_float("current_lat", row.get("current_lat", ""), path=p, row_num=row_num)
            lon = _parse_csv_cell_float("current_lon", row.get("current_lon", ""), path=p, row_num=row_num)
            cap = _parse_csv_cell_float("capacity", row.get("capacity", ""), path=p, row_num=row_num)
            if cap <= 0:
                raise ValueError(f"{p} row {row_num}: capacity must be positive, got {cap}")
            drivers.append(
                {
                    "driver_id": did,
                    "current_location": (lat, lon),
                    "capacity": cap,
                }
            )

    if not drivers:
        raise ValueError(f"{p}: no data rows after header")
    drivers.sort(key=lambda d: int(d["driver_id"]))
    return drivers


def plot_clusters_and_routes(
    stop_points: list[tuple[float, float]],
    cluster_labels: np.ndarray,
    routes: list[list[tuple[float, float]]],
    title: str,
    save_path: str | None = "output_clusters_routes.png",
    *,
    viz_mode: str = "map",
) -> None:
    """Scatter stops colored by DBSCAN cluster; overlay polylines for routes."""
    fig, ax = plt.subplots(figsize=(10, 8))
    labs = np.array(cluster_labels)
    lons_all = [p[1] for p in stop_points]
    lats_all = [p[0] for p in stop_points]
    for poly in routes:
        for p in poly:
            lons_all.append(p[1])
            lats_all.append(p[0])
    xlim, ylim = pad_lonlat_extent(lons_all, lats_all, pad_deg=0.012)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    had_bm = try_osm_basemap(ax, viz_mode=viz_mode)
    ax.scatter(
        [p[1] for p in stop_points],
        [p[0] for p in stop_points],
        c=labs,
        cmap="tab10",
        s=40,
        edgecolors="k",
        linewidths=0.3,
        zorder=5,
    )
    for ri, poly in enumerate(routes):
        if len(poly) < 2:
            continue
        lons = [p[1] for p in poly]
        lats = [p[0] for p in poly]
        ax.plot(lons, lats, "-o", linewidth=1.5, markersize=4, label=f"Route {ri+1}", zorder=5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    if not had_bm:
        ax.grid(True, alpha=0.3)
    if routes:
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_before_after(
    naive_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    optimized_routes: list[list[tuple[float, float]]],
    save_path: str | None = "output_before_after.png",
    *,
    viz_mode: str = "map",
) -> None:
    """
    Naive: many thin segments (direct out-and-back style).
    Optimized: longer continuous chains per driver.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    naive_lons: list[float] = []
    naive_lats: list[float] = []
    for a, b in naive_segments:
        for p in (a, b):
            naive_lats.append(p[0])
            naive_lons.append(p[1])
    xlim0, ylim0 = pad_lonlat_extent(naive_lons, naive_lats, pad_deg=0.012)
    axes[0].set_xlim(xlim0)
    axes[0].set_ylim(ylim0)
    axes[0].set_aspect("equal", adjustable="box")
    had_bm0 = try_osm_basemap(axes[0], viz_mode=viz_mode)
    for a, b in naive_segments:
        axes[0].plot([a[1], b[1]], [a[0], b[0]], "r-", alpha=0.5, linewidth=1, zorder=5)
        axes[0].scatter([a[1], b[1]], [a[0], b[0]], c="k", s=15, zorder=6)
    axes[0].set_title("Naive (independent legs)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    if not had_bm0:
        axes[0].grid(True, alpha=0.3)

    opt_lons: list[float] = []
    opt_lats: list[float] = []
    for poly in optimized_routes:
        for p in poly:
            opt_lats.append(p[0])
            opt_lons.append(p[1])
    xlim1, ylim1 = pad_lonlat_extent(opt_lons, opt_lats, pad_deg=0.012)
    axes[1].set_xlim(xlim1)
    axes[1].set_ylim(ylim1)
    axes[1].set_aspect("equal", adjustable="box")
    had_bm1 = try_osm_basemap(axes[1], viz_mode=viz_mode)
    for poly in optimized_routes:
        if len(poly) < 2:
            continue
        axes[1].plot([p[1] for p in poly], [p[0] for p in poly], "-o", linewidth=1.5, markersize=4, zorder=5)
    axes[1].set_title("Optimized multi-stop routes")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    if not had_bm1:
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
