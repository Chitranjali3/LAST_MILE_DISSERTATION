#!/usr/bin/env python3
"""
Entry point: run the full optimization engine on synthetic inputs, emit
research-friendly metrics, and save matplotlib figures (clusters & before/after).

Visual walkthrough:

    python main.py --present [--pause SECONDS]

Set Last-Mile_HEADLESS=1 to save step PNGs without blocking on windows (CI / SSH).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.batching import merge_same_location_orders
from core.clustering import cluster_deliveries_dbscan
from core.driver_assignment import assign_nearest_driver, stops_by_cluster
from core.routing import run_optimized_routes, summarize_savings
from simulator import run_all as run_simulations
from utils import (
    haversine_km,
    plot_before_after,
    plot_clusters_and_routes,
    synthetic_drivers,
    synthetic_orders,
)


def _format_grouping(merged: list[dict]) -> list[dict]:
    rows = []
    for m in merged:
        rows.append(
            {
                "representative_order_id": m["order_id"],
                "merged_order_ids": m.get("merged_order_ids", [m["order_id"]]),
                "weight_kg": m["parcel_weight"],
                "drop": m["drop"],
            }
        )
    return rows


def haversine_proxy(d: dict, drop: tuple[float, float]) -> float:
    return haversine_km(tuple(d["current_location"]), drop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delivery optimization engine runner")
    parser.add_argument(
        "--present",
        action="store_true",
        help="Show step-by-step Matplotlib visualization of the pipeline, then emit JSON.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Pause between presenter slides when a display is active (default: 2)",
    )
    parser.add_argument(
        "--use-osrm",
        action="store_true",
        help="Evaluate optimized routes with a local OSRM server and plot road geometry when available.",
    )
    parser.add_argument(
        "--osrm-url",
        default="http://localhost:5000",
        help="Base URL for OSRM, used with --use-osrm or --visual-input.",
    )
    parser.add_argument(
        "--visual-input",
        action="store_true",
        help="Open a click-based graphical input/process/output app backed by OSRM when available.",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    if args.visual_input:
        from visual_osrm_app import run_visual_osrm_app

        run_visual_osrm_app(osrm_base_url=args.osrm_url, out_dir=out_dir)
        return

    orders = synthetic_orders(26, seed=42)
    drivers = synthetic_drivers(5, seed=42)

    if args.present:
        from visual_presenter import present_full_pipeline

        results, savings, merged, labels, assignment, pipeline_info = present_full_pipeline(
            orders,
            drivers,
            dbscan_eps_km=1.4,
            pause_seconds=args.pause,
            out_dir=out_dir,
            use_osrm=args.use_osrm,
            osrm_base_url=args.osrm_url,
        )
    else:
        merged, _ = merge_same_location_orders(orders)
        labels, _ = cluster_deliveries_dbscan(merged, eps_km=1.4, min_samples=2)
        clusters = stops_by_cluster(labels)
        assignment = assign_nearest_driver(clusters, merged, drivers)
        results, pipeline_info = run_optimized_routes(
            orders,
            drivers,
            dbscan_eps_km=1.4,
            use_osrm=args.use_osrm,
            osrm_base_url=args.osrm_url,
        )
        savings = summarize_savings(orders, drivers, results)

    sim_pack = run_simulations()

    routes_for_plot: list[list[tuple[float, float]]] = []
    dmap = {int(d["driver_id"]): d for d in drivers}
    for r in results:
        drv = dmap[r.driver_id]
        poly = r.osrm_geometry if args.use_osrm and r.osrm_geometry else [tuple(drv["current_location"])] + r.drop_coords_ordered
        routes_for_plot.append(poly)

    drops = [tuple(s["drop"]) for s in merged]
    plot_clusters_and_routes(
        drops,
        labels,
        routes_for_plot,
        title="DBSCAN clusters and optimized driver routes",
        save_path=str(out_dir / "output_clusters_routes.png"),
    )

    naive_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for s in merged:
        drop = tuple(s["drop"])
        best_drv = min(drivers, key=lambda d: haversine_proxy(d, drop))
        naive_segments.append((tuple(best_drv["current_location"]), drop))

    plot_before_after(
        naive_segments,
        routes_for_plot,
        save_path=str(out_dir / "output_before_after.png"),
    )

    report = {
        "clustered_deliveries": {
            "n_merged_stops": len(merged),
            "cluster_labels": labels.tolist(),
            "driver_assignment_by_cluster": {str(k): v for k, v in assignment.items()},
        },
        "optimized_routes": [
            {
                "cluster_id": r.cluster_id,
                "driver_id": r.driver_id,
                "stop_sequence": r.stop_order_local,
                "ga_open_tour_km": r.ga_tour_km,
                "astar_graph_km": r.astar_leg_km,
                "osrm_road_km": r.osrm_road_km,
                "osrm_duration_min": r.osrm_duration_min,
                "osrm_status": r.osrm_status,
                "dijkstra_equals_astar_legs": r.dijkstra_star_equal,
                "vrptw_ok": r.vrptw_ok,
                "vrptw": r.vrptw_detail,
            }
            for r in results
        ],
        "totals": savings,
        "delivery_grouping": _format_grouping(merged),
        "comparison_naive_vs_optimized": {
            "naive_sum_independent_legs_km": savings["naive_sum_legs_km"],
            "optimized_ga_km": savings["optimized_ga_open_tour_km"],
            "optimized_astar_km": savings["optimized_astar_graph_km"],
            "optimized_osrm_road_km": savings["optimized_osrm_road_km"],
            "km_saved_ga": savings["saved_km_vs_naive_ga"],
            "km_saved_astar": savings["saved_km_vs_naive_astar"],
            "km_saved_osrm": savings["saved_km_vs_naive_osrm"],
        },
        "simulations": sim_pack,
        "pipeline": pipeline_info,
    }

    print(json.dumps(report, indent=2, default=str))
    print("\nFigures written:", out_dir / "output_clusters_routes.png", out_dir / "output_before_after.png")
    if args.present:
        slides = sorted(out_dir.glob("output_pipeline_*.png"))
        print(f"Presenter step frames: {len(slides)} files (output_pipeline_##_*.png)")


if __name__ == "__main__":
    main()
