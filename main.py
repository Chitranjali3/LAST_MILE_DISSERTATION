#!/usr/bin/env python3
"""
Entry point: run the full optimization engine on synthetic inputs, emit
research-friendly metrics, and save matplotlib figures (clusters & before/after).

Synthetic orders/drivers default to an Odisha / Bhubaneswar region (``utils.ODISHA_REGION_CENTER``),
aligned with the Geofabrik ``eastern-zone-latest`` OSRM extract.

Visual walkthrough:

    python main.py --present [--pause SECONDS] [--viz-mode graph|map]

CSV inputs (orders + drivers together; see ``data/sample_orders.csv`` and ``data/sample_drivers.csv``):

    python main.py --orders-csv data/sample_orders.csv --drivers-csv data/sample_drivers.csv
    python main.py --visual-input --orders-csv data/sample_orders.csv --drivers-csv data/sample_drivers.csv

Set Last-Mile_HEADLESS=1 to save step PNGs without blocking on windows (CI / SSH).
CHITRA_VIZ_MODE=graph|map sets default for --viz-mode when omitted.

Each completed CLI or visual pipeline writes timestamped mates under ``output/csv``, ``output/json``,
and ``output/images`` (figures share the same ``<stamp>`` as CSV/JSON on CLI runs).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from map_basemap import normalize_viz_mode

from pipeline_csv_log import (
    build_pipeline_report_dict,
    new_run_stamp_utc,
    output_csv_dir,
    output_images_dir,
    output_json_dir,
    save_pipeline_report_json,
    write_pipeline_run_csv,
)

from core.batching import merge_same_location_orders
from core.clustering import cluster_deliveries_dbscan
from core.driver_assignment import assign_nearest_driver, stops_by_cluster
from core.routing import run_optimized_routes, select_route_polyline, summarize_savings
from simulator import run_all as run_simulations
from utils import (
    haversine_km,
    load_drivers_csv,
    load_orders_csv,
    plot_before_after,
    plot_clusters_and_routes,
    synthetic_drivers,
    synthetic_orders,
)


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
    parser.add_argument(
        "--use-vrptw",
        action="store_true",
        help=(
            "Enable time-window-aware GA so the visit order respects each order's preferred "
            "delivery time (driver visits the earlier-windowed customer first even when a later "
            "one is geographically closer). Off by default — system runs as before."
        ),
    )
    parser.add_argument(
        "--viz-mode",
        type=str,
        choices=("graph", "map"),
        default=normalize_viz_mode(os.environ.get("CHITRA_VIZ_MODE")),
        help='Plots: "graph" = grid-only lon/lat; "map" = OSM tiles when possible (default from CHITRA_VIZ_MODE or map).',
    )
    parser.add_argument(
        "--orders-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Load orders from CSV (requires --drivers-csv). Schema: data/sample_orders.csv",
    )
    parser.add_argument(
        "--drivers-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Load drivers from CSV (requires --orders-csv). Schema: data/sample_drivers.csv",
    )
    args = parser.parse_args()

    use_csv = args.orders_csv is not None or args.drivers_csv is not None
    if use_csv and (args.orders_csv is None or args.drivers_csv is None):
        parser.error("--orders-csv and --drivers-csv must be given together")

    csv_orders: list | None = None
    csv_drivers: list | None = None
    if use_csv:
        try:
            csv_orders = load_orders_csv(args.orders_csv)
            csv_drivers = load_drivers_csv(args.drivers_csv)
        except (OSError, ValueError) as e:
            parser.error(str(e))

    out_dir = Path(__file__).resolve().parent
    if args.visual_input:
        from visual_osrm_app import run_visual_osrm_app

        run_visual_osrm_app(
            osrm_base_url=args.osrm_url,
            out_dir=out_dir,
            viz_mode=args.viz_mode,
            initial_use_vrptw=args.use_vrptw,
            initial_orders=csv_orders,
            initial_drivers=csv_drivers,
        )
        return

    if use_csv:
        orders = csv_orders
        drivers = csv_drivers
    else:
        orders = synthetic_orders(26, seed=42)
        drivers = synthetic_drivers(5, seed=42)

    if args.use_osrm:
        print(f"OSRM: requested at {args.osrm_url} (preflight pending...)")
    else:
        print("OSRM: not requested (pass --use-osrm to enable road enrichment)")

    dbscan_eps_km = 1.4
    run_stamp = new_run_stamp_utc()
    imgs_dir = output_images_dir(out_dir)

    if args.present:
        from visual_presenter import present_full_pipeline

        results, savings, merged, labels, assignment, pipeline_info = present_full_pipeline(
            orders,
            drivers,
            dbscan_eps_km=dbscan_eps_km,
            pause_seconds=args.pause,
            out_dir=out_dir,
            use_osrm=args.use_osrm,
            osrm_base_url=args.osrm_url,
            viz_mode=args.viz_mode,
            use_vrptw=args.use_vrptw,
            artifact_stamp=run_stamp,
        )
    else:
        merged, _ = merge_same_location_orders(orders)
        labels, _ = cluster_deliveries_dbscan(merged, eps_km=dbscan_eps_km, min_samples=2)
        clusters = stops_by_cluster(labels)
        assignment = assign_nearest_driver(clusters, merged, drivers)
        results, pipeline_info = run_optimized_routes(
            orders,
            drivers,
            dbscan_eps_km=dbscan_eps_km,
            use_osrm=args.use_osrm,
            osrm_base_url=args.osrm_url,
            use_vrptw=args.use_vrptw,
        )
        savings = summarize_savings(orders, drivers, results)

    csv_written = write_pipeline_run_csv(
        output_csv_dir(out_dir) / f"pipeline_run_{run_stamp}.csv",
        source="cli_main",
        orders=orders,
        drivers=drivers,
        merged=merged,
        labels=labels,
        assignment=assignment,
        results=results,
        savings=savings,
        pipeline_info=pipeline_info,
        dbscan_eps_km=dbscan_eps_km,
        use_osrm=args.use_osrm,
    )
    print(f"Pipeline CSV written: {csv_written}")

    osrm_info = pipeline_info.get("osrm", {})
    if args.use_osrm:
        if osrm_info.get("connected"):
            print(f"OSRM: connected (mode={osrm_info.get('mode', 'core_plus_osrm')})")
        else:
            reason = osrm_info.get("reason") or "unknown"
            print(f"OSRM: not connected ({reason}), using core routing")

    sim_pack = run_simulations()

    routes_for_plot: list[list[tuple[float, float]]] = []
    dmap = {int(d["driver_id"]): d for d in drivers}
    for r in results:
        drv = dmap[r.driver_id]
        routes_for_plot.append(select_route_polyline(r, tuple(drv["current_location"])))

    drops = [tuple(s["drop"]) for s in merged]
    clusters_png = imgs_dir / f"clusters_routes_{run_stamp}.png"
    plot_clusters_and_routes(
        drops,
        labels,
        routes_for_plot,
        title="DBSCAN clusters and optimized driver routes",
        save_path=str(clusters_png),
        viz_mode=args.viz_mode,
    )

    naive_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for s in merged:
        drop = tuple(s["drop"])
        best_drv = min(drivers, key=lambda d: haversine_proxy(d, drop))
        naive_segments.append((tuple(best_drv["current_location"]), drop))

    before_after_png = imgs_dir / f"before_after_{run_stamp}.png"
    plot_before_after(
        naive_segments,
        routes_for_plot,
        save_path=str(before_after_png),
        viz_mode=args.viz_mode,
    )

    report = build_pipeline_report_dict(
        merged=merged,
        labels=labels,
        assignment=assignment,
        results=results,
        savings=savings,
        pipeline_info=pipeline_info,
        run_stamp=run_stamp,
        run_source="cli_main",
        simulations=sim_pack,
    )
    json_written = output_json_dir(out_dir) / f"pipeline_run_{run_stamp}.json"
    save_pipeline_report_json(json_written, report)

    print(json.dumps(report, indent=2, default=str))
    print("\nPipeline JSON written:", json_written)
    print("\nFigures written:", clusters_png, before_after_png)
    if args.use_osrm:
        enriched = osrm_info.get("enriched_routes", 0)
        total = osrm_info.get("total_routes", len(results))
        print(f"OSRM enriched routes: {enriched}/{total}")
    if args.present:
        slides = sorted(imgs_dir.glob(f"pipeline_present_{run_stamp}_*.png"))
        print(f"Presenter step frames: {len(slides)} files under output/images (pipeline_present_{run_stamp}_##_*.png)")


if __name__ == "__main__":
    main()
