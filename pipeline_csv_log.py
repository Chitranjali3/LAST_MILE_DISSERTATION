"""
Write per-run pipeline exports under ``output/csv`` and ``output/json``.

Each execution uses a shared UTC timestamp stamp so the CSV and JSON for that
run pair up without overwriting prior runs.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.routing import RouteResult

# Stable column order for readable exports.
FIELDNAMES: tuple[str, ...] = (
    "run_timestamp_utc",
    "run_source",
    "use_osrm",
    "osrm_connected",
    "osrm_pipeline_mode",
    "dbscan_eps_km",
    "vrptw_enabled",
    "n_orders",
    "n_drivers",
    "n_merged_stops",
    "dbscan_cluster_labels",
    "driver_assignment_json",
    "naive_sum_legs_km",
    "optimized_ga_open_tour_km",
    "optimized_dijkstra_graph_km",
    "optimized_astar_quick_km",
    "optimized_astar_graph_km",
    "optimized_osrm_road_km",
    "saved_km_vs_naive_ga",
    "saved_km_vs_naive_dijkstra",
    "saved_km_vs_naive_astar",
    "saved_km_vs_naive_osrm",
    "route_index",
    "cluster_id",
    "driver_id",
    "n_stops_on_route",
    "visit_sequence_rep_order_ids",
    "merged_order_ids_per_stop",
    "ga_open_tour_km",
    "dijkstra_graph_km",
    "astar_route_km",
    "astar_route_min",
    "effective_distance_km",
    "effective_duration_min",
    "effective_distance_source",
    "effective_time_source",
    "osrm_road_km",
    "osrm_duration_min",
    "osrm_status",
    "vrptw_ok",
    "vrptw_optimized",
    "eta_arrival_min_chain",
    "vrptw_detail_json",
)


def new_run_stamp_utc() -> str:
    """Filesystem-safe UTC stamp (microsecond resolution)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def output_csv_dir(project_root: Path) -> Path:
    d = project_root / "output" / "csv"
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_json_dir(project_root: Path) -> Path:
    d = project_root / "output" / "json"
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_images_dir(project_root: Path) -> Path:
    d = project_root / "output" / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt_float(x: float | None, nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _json_compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str, ensure_ascii=False)


def _labels_to_csv(labels: np.ndarray) -> str:
    return ",".join(str(int(x)) for x in labels.tolist())


def _delivery_grouping_json(merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def build_pipeline_report_dict(
    *,
    merged: list[dict[str, Any]],
    labels: np.ndarray,
    assignment: dict[int, int],
    results: list[RouteResult],
    savings: dict[str, Any],
    pipeline_info: dict[str, Any],
    run_stamp: str,
    run_source: str,
    simulations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shape matches the CLI JSON report (``simulations`` omitted when None)."""
    payload: dict[str, Any] = {
        "run_stamp_utc": run_stamp,
        "run_source": run_source,
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
                "dijkstra_graph_km": r.dijkstra_graph_km,
                "astar_fastest_route_km": r.astar_leg_km,
                "astar_fastest_route_min": r.astar_leg_min,
                "eta_arrival_min_from_departure": r.eta_arrival_min,
                "osrm_road_km": r.osrm_road_km,
                "osrm_duration_min": r.osrm_duration_min,
                "osrm_status": r.osrm_status,
                "effective_distance_km": r.effective_distance_km,
                "effective_duration_min": r.effective_duration_min,
                "effective_distance_source": r.effective_distance_source,
                "effective_time_source": r.effective_time_source,
                "dijkstra_equals_astar_distance_legs": r.dijkstra_star_equal,
                "vrptw_ok": r.vrptw_ok,
                "vrptw": r.vrptw_detail,
            }
            for r in results
        ],
        "totals": savings,
        "delivery_grouping": _delivery_grouping_json(merged),
        "comparison_naive_vs_optimized": {
            "naive_sum_independent_legs_km": savings["naive_sum_legs_km"],
            "optimized_ga_km": savings["optimized_ga_open_tour_km"],
            "optimized_dijkstra_graph_km": savings["optimized_dijkstra_graph_km"],
            "optimized_astar_quick_km": savings["optimized_astar_quick_km"],
            "optimized_astar_km": savings["optimized_astar_graph_km"],
            "optimized_osrm_road_km": savings["optimized_osrm_road_km"],
            "km_saved_ga": savings["saved_km_vs_naive_ga"],
            "km_saved_dijkstra": savings["saved_km_vs_naive_dijkstra"],
            "km_saved_astar": savings["saved_km_vs_naive_astar"],
            "km_saved_osrm": savings["saved_km_vs_naive_osrm"],
        },
        "pipeline": pipeline_info,
    }
    payload["simulations"] = simulations
    return payload


def save_pipeline_report_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, default=str, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _build_csv_rows(
    *,
    source: str,
    orders: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    labels: np.ndarray,
    assignment: dict[int, int],
    results: list[RouteResult],
    savings: dict[str, Any],
    pipeline_info: dict[str, Any],
    dbscan_eps_km: float,
    use_osrm: bool,
) -> list[dict[str, str]]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    osrm_blk = pipeline_info.get("osrm") or {}
    vrptw_blk = pipeline_info.get("vrptw") or {}
    vrptw_on = bool(vrptw_blk.get("enabled", False))

    base_run: dict[str, str] = {
        "run_timestamp_utc": ts,
        "run_source": source,
        "use_osrm": str(bool(use_osrm)),
        "osrm_connected": str(bool(osrm_blk.get("connected"))),
        "osrm_pipeline_mode": str(osrm_blk.get("mode", "")),
        "dbscan_eps_km": _fmt_float(dbscan_eps_km, nd=6),
        "vrptw_enabled": str(vrptw_on),
        "n_orders": str(len(orders)),
        "n_drivers": str(len(drivers)),
        "n_merged_stops": str(len(merged)),
        "dbscan_cluster_labels": _labels_to_csv(labels) if len(labels) else "",
        "driver_assignment_json": _json_compact({str(k): int(v) for k, v in sorted(assignment.items())}),
        "naive_sum_legs_km": _fmt_float(savings.get("naive_sum_legs_km")),
        "optimized_ga_open_tour_km": _fmt_float(savings.get("optimized_ga_open_tour_km")),
        "optimized_dijkstra_graph_km": _fmt_float(savings.get("optimized_dijkstra_graph_km")),
        "optimized_astar_quick_km": _fmt_float(savings.get("optimized_astar_quick_km")),
        "optimized_astar_graph_km": _fmt_float(savings.get("optimized_astar_graph_km")),
        "optimized_osrm_road_km": _fmt_float(savings.get("optimized_osrm_road_km")),
        "saved_km_vs_naive_ga": _fmt_float(savings.get("saved_km_vs_naive_ga")),
        "saved_km_vs_naive_dijkstra": _fmt_float(savings.get("saved_km_vs_naive_dijkstra")),
        "saved_km_vs_naive_astar": _fmt_float(savings.get("saved_km_vs_naive_astar")),
        "saved_km_vs_naive_osrm": _fmt_float(savings.get("saved_km_vs_naive_osrm")),
    }

    def row_for_route(idx: int, r: RouteResult) -> dict[str, str]:
        rep_ids = [int(m["order_id"]) for m in r.metas_ordered]
        merged_ids_per = []
        for m in r.metas_ordered:
            mids = m.get("merged_order_ids")
            if isinstance(mids, list) and mids:
                merged_ids_per.append("|".join(str(int(x)) for x in mids))
            else:
                merged_ids_per.append(str(int(m["order_id"])))
        eta_chain = ";".join(_fmt_float(x, nd=2) for x in r.eta_arrival_min)
        out = dict(base_run)
        out.update(
            {
                "route_index": str(idx),
                "cluster_id": str(int(r.cluster_id)),
                "driver_id": str(int(r.driver_id)),
                "n_stops_on_route": str(len(r.metas_ordered)),
                "visit_sequence_rep_order_ids": "|".join(str(x) for x in rep_ids),
                "merged_order_ids_per_stop": ";".join(merged_ids_per),
                "ga_open_tour_km": _fmt_float(r.ga_tour_km),
                "dijkstra_graph_km": _fmt_float(r.dijkstra_graph_km),
                "astar_route_km": _fmt_float(r.astar_leg_km),
                "astar_route_min": _fmt_float(r.astar_leg_min),
                "effective_distance_km": _fmt_float(r.effective_distance_km),
                "effective_duration_min": _fmt_float(r.effective_duration_min, nd=2),
                "effective_distance_source": str(r.effective_distance_source),
                "effective_time_source": str(r.effective_time_source),
                "osrm_road_km": _fmt_float(r.osrm_road_km) if r.osrm_road_km is not None else "",
                "osrm_duration_min": _fmt_float(r.osrm_duration_min, nd=2)
                if r.osrm_duration_min is not None
                else "",
                "osrm_status": str(r.osrm_status).replace("\n", " ").strip(),
                "vrptw_ok": str(bool(r.vrptw_ok)),
                "vrptw_optimized": str(bool(getattr(r, "vrptw_optimized", False))),
                "eta_arrival_min_chain": eta_chain,
                "vrptw_detail_json": _json_compact(r.vrptw_detail),
            }
        )
        return out

    if results:
        return [row_for_route(i, r) for i, r in enumerate(results)]
    empty = dict(base_run)
    empty.update(
        {
            "route_index": "",
            "cluster_id": "",
            "driver_id": "",
            "n_stops_on_route": "0",
            "visit_sequence_rep_order_ids": "",
            "merged_order_ids_per_stop": "",
            "ga_open_tour_km": "",
            "dijkstra_graph_km": "",
            "astar_route_km": "",
            "astar_route_min": "",
            "effective_distance_km": "",
            "effective_duration_min": "",
            "effective_distance_source": "",
            "effective_time_source": "",
            "osrm_road_km": "",
            "osrm_duration_min": "",
            "osrm_status": "",
            "vrptw_ok": "",
            "vrptw_optimized": str(vrptw_on),
            "eta_arrival_min_chain": "",
            "vrptw_detail_json": "",
        }
    )
    return [empty]


def write_pipeline_run_csv(
    path: Path,
    *,
    source: str,
    orders: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    labels: np.ndarray,
    assignment: dict[int, int],
    results: list[RouteResult],
    savings: dict[str, Any],
    pipeline_info: dict[str, Any],
    dbscan_eps_km: float,
    use_osrm: bool,
) -> Path:
    """Write a fresh CSV file (header + rows) under ``output/csv``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _build_csv_rows(
        source=source,
        orders=orders,
        drivers=drivers,
        merged=merged,
        labels=labels,
        assignment=assignment,
        results=results,
        savings=savings,
        pipeline_info=pipeline_info,
        dbscan_eps_km=dbscan_eps_km,
        use_osrm=use_osrm,
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
