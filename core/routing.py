"""
End-to-end routing orchestration: naive baselines vs. the research pipeline.

Pipeline (system flow):
  merged same-location stops  →  greedy batching (dynamic pre-grouping)
  → DBSCAN clusters  →  nearest-driver assignment
  → GA within each cluster  →  graph legs on the cluster kNN geographic graph
       · Dijkstra: authoritative shortest-path km per leg (summed for the tour)
       · A*: goal-directed pathfinding; leg km feeds quick-route ETA per stop
  → VRPTW feasibility check per route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.batching import greedy_dynamic_batch, merge_same_location_orders
from core.clustering import cluster_deliveries_dbscan
from core.driver_assignment import assign_nearest_driver, stops_by_cluster
from core.ga_optimizer import optimize_route_ga, tour_distance_km
from core.graph_search import astar_shortest_path, build_geographic_graph, dijkstra_shortest_path
from core.osrm_client import OsrmClient
from core.vrptw import VRPTWConfig, slack_time_windows, validate_route_feasibility
from utils import haversine_km


# Average city driving speed (km/h) used as a fallback when OSRM duration is
# unavailable. Matches the proxy speed assumed inside the GA optimizer so the
# reported "effective_duration_min" stays consistent with the planning math.
_DEFAULT_FALLBACK_SPEED_KMH = 25.0


@dataclass
class RouteResult:
    cluster_id: int
    driver_id: int
    stop_order_local: list[int]
    drop_coords_ordered: list[tuple[float, float]]
    metas_ordered: list[dict[str, Any]]
    ga_tour_km: float
    dijkstra_graph_km: float
    astar_leg_km: float
    eta_arrival_min: list[float]
    dijkstra_star_equal: bool
    vrptw_ok: bool
    vrptw_detail: dict[str, Any]
    osrm_road_km: float | None = None
    osrm_duration_min: float | None = None
    osrm_geometry: list[tuple[float, float]] | None = None
    osrm_status: str = "not_requested"
    # Additive M2 fields: a single source of truth for "what should I report or
    # plot for this route?". Existing OSRM/A*/GA fields are untouched, callers
    # that only want the best-available metric can read these directly.
    effective_distance_km: float = 0.0
    effective_duration_min: float | None = None
    effective_distance_source: str = "dijkstra"  # "osrm" | "dijkstra" | "ga_proxy"
    effective_time_source: str = "proxy"  # "osrm" | "proxy"


def select_route_polyline(
    result: "RouteResult",
    driver_start: tuple[float, float],
) -> list[tuple[float, float]]:
    """Pick a safe drawable polyline for `result`, deterministically.

    Order of preference (M2 plotting rule):
      1. OSRM road geometry, when it exists, has at least two points, and is
         not a degenerate near-zero-length artifact (which OSRM occasionally
         emits when waypoints snap to the same edge).
      2. The straight-line "driver_start -> ordered drops" fallback, which is
         always available because A*/GA already validated the stops.
    """
    fallback = [driver_start] + result.drop_coords_ordered
    osrm_poly = result.osrm_geometry or []
    if len(osrm_poly) < 2:
        return fallback

    start_lat, start_lon = osrm_poly[0]
    end_lat, end_lon = osrm_poly[-1]
    if abs(start_lat - end_lat) < 1e-6 and abs(start_lon - end_lon) < 1e-6:
        return fallback
    return osrm_poly


def _effective_metrics_for_route(
    *,
    osrm_road_km: float | None,
    osrm_duration_min: float | None,
    dijkstra_graph_km: float,
    ga_tour_km: float,
) -> tuple[float, float | None, str, str]:
    """Pick the best-available distance/duration source for a single route.

    Fallback priority (per checklist M2):
      1. OSRM road metrics when a numeric value is present.
      2. Dijkstra shortest-path km on the cluster graph (always computed).
      3. GA fallback for geometry-only edge cases (zero-length graph result).
    """
    if osrm_road_km is not None and osrm_road_km > 0.0:
        dist = float(osrm_road_km)
        dist_source = "osrm"
    elif dijkstra_graph_km > 0.0:
        dist = float(dijkstra_graph_km)
        dist_source = "dijkstra"
    else:
        dist = float(ga_tour_km)
        dist_source = "ga_proxy"

    if osrm_duration_min is not None and osrm_duration_min > 0.0:
        return dist, float(osrm_duration_min), dist_source, "osrm"

    proxy_minutes = (dist / _DEFAULT_FALLBACK_SPEED_KMH) * 60.0 if dist > 0.0 else 0.0
    return dist, proxy_minutes, dist_source, "proxy"


def _driver_by_id(drivers: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(d["driver_id"]): d for d in drivers}


def naive_independent_legs_km(
    stops: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
) -> float:
    """
    Baseline: each physical stop is served by its closest driver in isolation
    (no multi-stop chaining). Distance = sum of driver→drop one-way legs.
    """
    total = 0.0
    by_id = _driver_by_id(drivers)
    for s in stops:
        drop = tuple(s["drop"])
        best = float("inf")
        for d in drivers:
            best = min(best, haversine_km(tuple(d["current_location"]), drop))
        total += best
    return total


def cluster_route_graph_metrics(
    driver_loc: tuple[float, float],
    ordered_drops: list[tuple[float, float]],
    *,
    knn: int | None = 4,
    vr_cfg: VRPTWConfig | None = None,
) -> tuple[float, float, list[tuple[float, float, float]], list[float]]:
    """
    On the DBSCAN cluster's kNN geographic graph (driver + ordered drops):

    - Dijkstra per leg → summed ``dijkstra_graph_km`` (shortest-path distance).
    - A* per leg → summed ``astar_leg_km`` and used with ``vr_cfg`` speed to build
      ETA (minutes from departure until arrival at each drop, before service).

    Returns ``(dijkstra_km, astar_km, leg_diag, eta_arrival_min)`` where leg_diag is
    ``(dijkstra_km, astar_km, abs(delta))`` per leg.
    """
    cfg = vr_cfg or VRPTWConfig()
    if not ordered_drops:
        return 0.0, 0.0, [], []

    nodes = [driver_loc] + ordered_drops
    G = build_geographic_graph(nodes, knn=knn if len(nodes) > 5 else None)
    dijkstra_sum = 0.0
    astar_sum = 0.0
    diag: list[tuple[float, float, float]] = []
    eta_arrival: list[float] = []
    clock = 0.0
    prev = driver_loc
    speed = cfg.avg_speed_kmh
    service = cfg.service_time_min

    for nxt in ordered_drops:
        _, dk = dijkstra_shortest_path(G, prev, nxt)
        _, ak = astar_shortest_path(G, prev, nxt)
        dijkstra_sum += dk
        astar_sum += ak
        diag.append((dk, ak, abs(dk - ak)))

        travel_min = (ak / speed) * 60.0 if speed > 0 else 0.0
        clock += travel_min
        eta_arrival.append(clock)
        clock += service

        prev = nxt

    return dijkstra_sum, astar_sum, diag, eta_arrival


def astar_tour_length(
    driver_loc: tuple[float, float],
    ordered_drops: list[tuple[float, float]],
    *,
    knn: int | None = 4,
) -> tuple[float, list[tuple[float, float, float]]]:
    """Backward-compatible: returns (A* tour km, leg diagnostics) only."""
    _dij, ak, diag, _eta = cluster_route_graph_metrics(
        driver_loc, ordered_drops, knn=knn, vr_cfg=VRPTWConfig()
    )
    return ak, diag


def run_optimized_routes(
    orders: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
    *,
    dbscan_eps_km: float = 1.2,
    slack_windows: bool = True,
    use_osrm: bool = False,
    osrm_base_url: str = "http://localhost:5000",
) -> tuple[list[RouteResult], dict[str, Any]]:
    """
    Execute the full optimization stack on synthetic/real dict inputs.

    Returns per-cluster `RouteResult` rows and a summary dictionary.

    OSRM is treated as an *optional enrichment layer*: even when `use_osrm=True`
    the core flow always completes with GA + Dijkstra/A* graph legs + VRPTW metrics. A fast preflight
    decides whether OSRM gets invoked at all and the per-route circuit breaker
    short-circuits remaining calls if the server collapses mid-run.
    """
    merged, merge_map = merge_same_location_orders(orders)
    if slack_windows:
        slack_time_windows(merged, pad_min=45.0)

    _ = greedy_dynamic_batch(orders)  # greedy wave partitioning (dynamic insertion semantics)
    labels, _ = cluster_deliveries_dbscan(merged, eps_km=dbscan_eps_km, min_samples=2)
    clusters = stops_by_cluster(labels)
    assignment = assign_nearest_driver(clusters, merged, drivers)
    dmap = _driver_by_id(drivers)

    # M1: preflight gate. Build the client only when requested, then probe before
    # the route loop. A failing probe immediately drops us into "core_only" mode
    # so no per-cluster OSRM call is ever attempted.
    osrm_client: OsrmClient | None = None
    osrm_connected = False
    osrm_reason = "not_requested"
    osrm_disabled_status: str | None = None
    if use_osrm:
        osrm_client = OsrmClient(osrm_base_url)
        health = osrm_client.health_check()
        osrm_connected = health.ok
        osrm_reason = health.reason
        if not health.ok:
            detail = f": {health.detail}" if health.detail else ""
            osrm_disabled_status = f"{health.reason}{detail}"
            osrm_client = None  # core_only path: never issue route calls

    results: list[RouteResult] = []
    osrm_status_counts: dict[str, int] = {}
    enriched_routes = 0

    for cid, idxs in sorted(clusters.items()):
        d_id = int(assignment.get(cid, drivers[0]["driver_id"]))
        drv = dmap[d_id]
        cluster_coords = [tuple(merged[i]["drop"]) for i in idxs]
        perm, _ga_fit = optimize_route_ga(cluster_coords, tuple(drv["current_location"]))
        ordered_drops = [cluster_coords[k] for k in perm]
        metas = [merged[idxs[k]] for k in perm]

        ga_km = tour_distance_km(perm, cluster_coords, tuple(drv["current_location"]))
        vr_cfg = VRPTWConfig()
        dij_km, ast_km, leg_diag, eta_min = cluster_route_graph_metrics(
            tuple(drv["current_location"]), ordered_drops, vr_cfg=vr_cfg
        )
        leg_equal = all(abs(d[2]) <= 1e-9 for d in leg_diag)

        ok, detail = validate_route_feasibility(
            tuple(drv["current_location"]),
            ordered_drops,
            metas,
            float(drv["capacity"]),
            vr_cfg,
        )

        osrm_road_km: float | None = None
        osrm_duration_min: float | None = None
        osrm_geometry: list[tuple[float, float]] | None = None
        if not use_osrm:
            osrm_status = "not_requested"
        elif osrm_client is None:
            # Either preflight failed or a previous route tripped the breaker.
            osrm_status = osrm_disabled_status or osrm_reason
        else:
            osrm_route = osrm_client.route([tuple(drv["current_location"])] + ordered_drops)
            osrm_road_km = osrm_route.road_km
            osrm_duration_min = osrm_route.duration_min
            osrm_geometry = osrm_route.geometry
            if osrm_route.ok:
                osrm_status = "ok"
                enriched_routes += 1
            else:
                detail_txt = osrm_route.message or osrm_route.code
                osrm_status = f"{osrm_route.status}: {detail_txt}" if detail_txt else osrm_route.status
            # M5: one-way circuit breaker — any transport-level failure stops
            # further OSRM calls and propagates the reason cleanly to remaining
            # clusters via osrm_disabled_status.
            if osrm_route.status in {"unavailable", "http_error"}:
                osrm_disabled_status = osrm_status
                osrm_client = None
        osrm_status_counts[osrm_status] = osrm_status_counts.get(osrm_status, 0) + 1

        eff_km, eff_min, eff_dist_src, eff_time_src = _effective_metrics_for_route(
            osrm_road_km=osrm_road_km,
            osrm_duration_min=osrm_duration_min,
            dijkstra_graph_km=dij_km,
            ga_tour_km=ga_km,
        )

        results.append(
            RouteResult(
                cluster_id=cid,
                driver_id=d_id,
                stop_order_local=perm,
                drop_coords_ordered=ordered_drops,
                metas_ordered=metas,
                ga_tour_km=ga_km,
                dijkstra_graph_km=dij_km,
                astar_leg_km=ast_km,
                eta_arrival_min=eta_min,
                dijkstra_star_equal=leg_equal,
                vrptw_ok=ok,
                vrptw_detail=detail,
                osrm_road_km=osrm_road_km,
                osrm_duration_min=osrm_duration_min,
                osrm_geometry=osrm_geometry,
                osrm_status=osrm_status,
                effective_distance_km=eff_km,
                effective_duration_min=eff_min,
                effective_distance_source=eff_dist_src,
                effective_time_source=eff_time_src,
            )
        )

    total_routes = len(results)
    if not use_osrm:
        mode = "core_only"
    elif enriched_routes > 0 and osrm_connected:
        mode = "core_plus_osrm"
    else:
        mode = "core_only"

    summary = {
        "merged_stop_count": len(merged),
        "clusters": len(clusters),
        "merge_map": merge_map,
        "osrm": {
            "requested": use_osrm,
            "base_url": osrm_base_url if use_osrm else None,
            "connected": bool(osrm_connected and use_osrm),
            "mode": mode,
            "reason": osrm_reason,
            "status_counts": osrm_status_counts,
            "enriched_routes": enriched_routes,
            "total_routes": total_routes,
        },
    }
    return results, summary


def summarize_savings(
    orders: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
    route_results: list[RouteResult],
) -> dict[str, float]:
    """Compare naive independent legs vs. optimized chained tours (GA + graph legs)."""
    merged, _ = merge_same_location_orders(orders)
    naive_km = naive_independent_legs_km(merged, drivers)
    opt_ga_km = sum(r.ga_tour_km for r in route_results)
    opt_dij_km = sum(r.dijkstra_graph_km for r in route_results)
    opt_ast_km = sum(r.astar_leg_km for r in route_results)
    osrm_values = [r.osrm_road_km for r in route_results if r.osrm_road_km is not None]
    opt_osrm_km = sum(osrm_values) if len(osrm_values) == len(route_results) and route_results else 0.0
    return {
        "naive_sum_legs_km": naive_km,
        "optimized_ga_open_tour_km": opt_ga_km,
        "optimized_dijkstra_graph_km": opt_dij_km,
        "optimized_astar_quick_km": opt_ast_km,
        # Historic key: graph-chain km (now Dijkstra SP on cluster graph; equals A* when legs match).
        "optimized_astar_graph_km": opt_dij_km,
        "optimized_osrm_road_km": opt_osrm_km,
        "saved_km_vs_naive_ga": max(0.0, naive_km - opt_ga_km),
        "saved_km_vs_naive_dijkstra": max(0.0, naive_km - opt_dij_km),
        "saved_km_vs_naive_astar": max(0.0, naive_km - opt_ast_km),
        "saved_km_vs_naive_osrm": max(0.0, naive_km - opt_osrm_km) if opt_osrm_km else 0.0,
    }
