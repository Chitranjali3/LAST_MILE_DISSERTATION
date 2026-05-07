"""
Interactive Matplotlib slideshow of the optimization pipeline.

Shows each major processing stage (batching/clustering/assignment/GA-A*/VRPTW/outcome),
then leaves a final metrics panel. Intended for demos and dissertation walkthroughs.

Non-interactive / headless: set environment variable Last-Mile_HEADLESS=1 to skip blocking
plt.show() waits and instead write output_pipeline_steps.png frames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np

from core.batching import greedy_dynamic_batch, merge_same_location_orders
from core.clustering import cluster_deliveries_dbscan
from core.driver_assignment import assign_nearest_driver, stops_by_cluster
from core.routing import RouteResult, run_optimized_routes, summarize_savings
from utils import haversine_km, synthetic_drivers, synthetic_orders


@dataclass
class PresenterConfig:
    pause_seconds: float = 2.0
    out_dir: Path | None = None
    headless: bool = False


def _is_headless() -> bool:
    return os.environ.get("Last-Mile_HEADLESS", "").strip().lower() in ("1", "true", "yes")


class PipelineVisualizer:
    def __init__(self, cfg: PresenterConfig) -> None:
        self.cfg = cfg
        self.out_dir = cfg.out_dir or Path(__file__).resolve().parent
        self.frames: list[np.ndarray | None] = []
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("ggplot")
        if not cfg.headless and not _is_headless():
            plt.ion()
        self._fig = plt.figure(figsize=(12.5, 7.2), constrained_layout=False)
        self._frames_saved = 0

    def _step_header(self, step: str, subtitle: str) -> None:
        self._fig.clf()
        self._fig.suptitle(step, fontsize=15, fontweight="bold", y=0.98)
        self._fig.text(
            0.5,
            0.94,
            subtitle,
            ha="center",
            fontsize=10,
            style="italic",
            color="#333333",
        )

    def _scatter_map(
        self,
        lons: list[float],
        lats: list[float],
        c: Any | None = None,
        labels: list[str] | None = None,
        marker_size: float = 55,
    ) -> Any:
        ax = self._fig.add_axes([0.08, 0.08, 0.72, 0.78])
        sc = ax.scatter(
            lons,
            lats,
            c=c if c is not None else "#1f77b4",
            cmap="tab20" if c is not None else None,
            s=marker_size,
            edgecolors="#222222",
            linewidths=0.4,
            zorder=5,
        )
        if labels:
            for lon, lat, lab in zip(lons, lats, labels, strict=False):
                ax.annotate(str(lab), (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        return ax, sc

    def _annotate_flow(self, lines: list[str]) -> None:
        ax_txt = self._fig.add_axes([0.82, 0.12, 0.17, 0.74])
        ax_txt.axis("off")
        body = "\n".join(lines)
        ax_txt.text(
            0,
            1.0,
            "Pipeline snapshot",
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        ax_txt.text(
            0,
            0.92,
            body,
            fontsize=8.8,
            va="top",
            family="monospace",
            linespacing=1.35,
        )

    def _flush(self, tag: str) -> None:
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        path = self.out_dir / f"output_pipeline_{self._frames_saved:02d}_{tag}.png"
        self._fig.savefig(path, dpi=140, bbox_inches="tight")
        self._frames_saved += 1
        if self.cfg.headless or _is_headless():
            plt.pause(0.05)
        else:
            plt.pause(self.cfg.pause_seconds)

    # --- slides ----------------------------------------------------------

    def slide_01_orders(self, orders: list[dict], drivers: list[dict]) -> None:
        self._step_header(
            "Step 1 / 8 · Input orders & fleet",
            "Synthetic orders (drops) and driver home locations before optimization.",
        )
        dlons = [o["drop"][1] for o in orders]
        dlats = [o["drop"][0] for o in orders]
        plons = [o["pickup"][1] for o in orders]
        plats = [o["pickup"][0] for o in orders]
        ax = self._fig.add_axes([0.08, 0.08, 0.72, 0.78])
        ax.scatter(plons, plats, c="#888888", s=22, marker="x", linewidths=0.8, label="Pickup ref.", zorder=4)
        ax.scatter(dlons, dlats, c="#d62728", s=42, marker="o", edgecolors="#222222", linewidths=0.4, label="Delivery drop", zorder=5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        hlons = [d["current_location"][1] for d in drivers]
        hlats = [d["current_location"][0] for d in drivers]
        ax.scatter(hlons, hlats, c="#2ca02c", s=140, marker="*", edgecolors="#000", linewidths=0.5, zorder=6, label="Drivers")
        ax.legend(loc="upper left", fontsize=8)

        batches = greedy_dynamic_batch(orders)
        self._annotate_flow(
            [
                f"Orders      : {len(orders)}",
                f"Drivers     : {len(drivers)}",
                f"Greedy waves: {len(batches)} batches",
                "",
                "Next: merge same-building drops",
            ]
        )
        self._flush("input")

    def slide_02_merge_rule1(self, orders: list[dict], merged: list[dict]) -> None:
        self._step_header(
            "Step 2 / 8 · Same-location merging (Business Rule 1)",
            "Parcels sharing a drop coordinate are merged into one stop serviced by one agent.",
        )
        raw_lons = [o["drop"][1] for o in orders]
        raw_lats = [o["drop"][0] for o in orders]
        ax, _ = self._scatter_map(raw_lons, raw_lats, marker_size=38)
        ax.scatter([], [], label="Overlaid merges below", alpha=0)
        mer_lons = [m["drop"][1] for m in merged]
        mer_lats = [m["drop"][0] for m in merged]
        mx = max(merged, key=lambda m: len(m.get("merged_order_ids", [m["order_id"]])))

        ax.add_patch(
            Circle((mx["drop"][1], mx["drop"][0]), 0.015, fill=False, color="blue", linewidth=2, zorder=2)
        )
        ax.scatter(mer_lons, mer_lats, facecolors="none", edgecolors="#1f77b4", s=280, linewidths=1.2, label="Merged stop hull")
        ax.legend(loc="upper left", fontsize=8)

        merged_groups = sum(1 for m in merged if len(m.get("merged_order_ids", [m["order_id"]])) > 1)
        self._annotate_flow(
            [
                f"Raw drops   : {len(orders)}",
                f"Merged stops: {len(merged)}",
                f"Multi-order : {merged_groups}",
                "",
                "Next: DBSCAN clustering",
            ]
        )
        self._flush("merge")

    def slide_03_clusters(self, merged: list[dict], labels: np.ndarray) -> None:
        self._step_header(
            "Step 3 / 8 · DBSCAN spatial clustering",
            "Density-based clustering (Haversine metric) forms delivery neighborhoods.",
        )
        lons = [m["drop"][1] for m in merged]
        lats = [m["drop"][0] for m in merged]
        ax, sc = self._scatter_map(lons, lats, c=labels, marker_size=50)
        self._fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, label="Cluster label")
        uniq, counts = np.unique(labels, return_counts=True)
        top = sorted(zip(counts.tolist(), uniq.tolist()), reverse=True)[:4]
        top_s = "; ".join(f"c{lab}: n={cnt}" for cnt, lab in top)
        self._annotate_flow(
            [
                "Algorithm : DBSCAN",
                "Metric    : Haversine (radial ε)",
                f"Clusters  : {len(uniq)}",
                f"largest   : {top_s[:60]}",
                "",
                "Next: nearest-driver assignment",
            ]
        )
        self._flush("dbscan")

    def slide_04_assignment(self, merged: list[dict], labels: np.ndarray, assignment: dict[int, int], drivers: list[dict]) -> None:
        self._step_header(
            "Step 4 / 8 · Nearest-driver heuristic",
            "Each cluster centroid is matched to the closest available driver (Haversine).",
        )
        clusters = stops_by_cluster(labels)
        lons = [m["drop"][1] for m in merged]
        lats = [m["drop"][0] for m in merged]
        ax, _ = self._scatter_map(lons, lats, c=labels, marker_size=44)
        dmap = {int(d["driver_id"]): d for d in drivers}
        for cid, idxs in clusters.items():
            pts = np.array([[merged[i]["drop"][1], merged[i]["drop"][0]] for i in idxs])
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            did = assignment.get(cid)
            if did is None:
                continue
            drv = dmap[did]
            dx, dy = drv["current_location"][1], drv["current_location"][0]
            ax.plot([cx, dx], [cy, dy], "k--", alpha=0.35, linewidth=1)
            ax.scatter([cx], [cy], c="black", s=28, marker="+", zorder=7)
        hlons = [d["current_location"][1] for d in drivers]
        hlats = [d["current_location"][0] for d in drivers]
        ax.scatter(hlons, hlats, c="#2ca02c", s=120, marker="*", edgecolors="#000", linewidths=0.4, zorder=8)
        self._annotate_flow(
            [
                f"Clusters      : {len(clusters)}",
                f"Assigned pairs: {len(assignment)}",
                "Heuristic     : min distance",
                "              centroid → driver",
                "",
                "Next: GA route order per cluster",
            ]
        )
        self._flush("assign")

    def slide_05_ga_intro(self, n_clusters_with_multi: int) -> None:
        self._step_header(
            "Step 5 / 8 · Genetic Algorithm (within-cluster sequencing) — methodology",
            "Already executed for all clusters behind the scenes. Encoding: permutation of stops; operators: PMX, shuffle mutation; selection: tournament; fitness weights distance/time/fuel.",
        )
        ax_txt = self._fig.add_axes([0.06, 0.10, 0.88, 0.78])
        ax_txt.axis("off")
        algo = """Chromosome representation
  └─ permutation of intra-cluster stop indices (open tour from assigned driver).

Operators
  • Crossover : Partially Mapped Crossover (PMX), p≈0.85
  • Mutation  : shuffle indices, stochastic swap
  • Selection : tournament (k = 3)
  • Elitism   : preserve best individual each generation

Fitness (composite minimization)
  distance + λ₁·travel_time_proxy + λ₂·fuel_proxy
"""
        ax_txt.text(0, 1, algo, fontsize=10.8, va="top", family="monospace", linespacing=1.4)
        self._annotate_flow(
            [
                "GA phase      : DONE",
                f"multi-drop CL : ~{n_clusters_with_multi}",
                "See methodology →",
                "",
                "Next: A* + VRPTW rationale",
            ]
        )
        self._flush("ga")

    def slide_06_navigation_vrptw(self, routes_equal_leg_benchmark: bool, osrm_requested: bool = False) -> None:
        self._step_header(
            "Step 6 / 8 · Navigation · OSRM/A* · VRPTW validation",
            "GA picks stop order; OSRM road routes are used when available, with A* as the offline graph baseline.",
        )
        ax_txt = self._fig.add_axes([0.06, 0.12, 0.88, 0.74])
        ax_txt.axis("off")
        body = """A*
  f(n) = g(n) + h(n); h = great-circle miles to goal (research proxy heuristic).

Dijkstra (benchmark)
  Same nonnegative graph → shortest-path cost equals A* result on every leg tested.

VRPTW (feasibility, not solver)
  • Cumulative parcel weight vs vehicle capacity
  • Delivery windows anchored to earliest stop in route
  • Fixed service dwell + route duration ceiling
"""
        ax_txt.text(0, 1, body, fontsize=10.8, va="top", family="monospace", linespacing=1.35)
        self._annotate_flow(
            [
                "Dijkstra == A* (legs):" + (" yes" if routes_equal_leg_benchmark else " check per route"),
                "OSRM road geometry    :" + (" requested" if osrm_requested else " off"),
                "",
                "Next: naive vs optimized map",
            ]
        )
        self._flush("astar_vrptw")

    def slide_07_before_after(self, merged: list[dict], drivers: list[dict], routes_for_plot: list[list[tuple[float, float]]]) -> None:
        self._step_header(
            "Step 7 / 8 · Before vs after (visual)",
            "Left: naive independent legs. Right: optimized multi-stop chaining.",
        )
        ax1 = self._fig.add_axes([0.06, 0.10, 0.42, 0.78])
        ax2 = self._fig.add_axes([0.52, 0.10, 0.42, 0.78])
        naive_segments = []
        for s in merged:
            drop = tuple(s["drop"])
            best_drv = min(drivers, key=lambda d: haversine_km(tuple(d["current_location"]), drop))
            naive_segments.append((tuple(best_drv["current_location"]), drop))
        for a, b in naive_segments:
            ax1.plot([a[1], b[1]], [a[0], b[0]], color="#cc6666", alpha=0.55, lw=1.1)
            ax1.scatter([a[1], b[1]], [a[0], b[0]], c="k", s=14, zorder=4)
        ax1.set_title("Naive routing", fontsize=11)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal", adjustable="box")

        for poly in routes_for_plot:
            if len(poly) < 2:
                continue
            ax2.plot([p[1] for p in poly], [p[0] for p in poly], "-o", lw=1.45, markersize=4)
        ax2.set_title("Optimized chaining", fontsize=11)
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal", adjustable="box")

        self._annotate_flow(
            [
                "Compare geometry",
                "of independent",
                "legs vs tours.",
                "",
                "Next: metrics",
            ]
        )
        self._flush("maps")

    def slide_08_summary(self, savings: dict[str, float], results: list[RouteResult]) -> None:
        self._step_header(
            "Step 8 / 8 · Output metrics",
            "Quantitative gains vs naive baseline.",
        )
        ax_bar = self._fig.add_axes([0.07, 0.38, 0.48, 0.42])
        names = ["Naive\n(legs Σ)", "GA open\ntour", "A*\n(graph)"]
        vals = [
            savings["naive_sum_legs_km"],
            savings["optimized_ga_open_tour_km"],
            savings["optimized_astar_graph_km"],
        ]
        cols = ["#8c564b", "#1f77b4", "#ff7f0e"]
        if savings.get("optimized_osrm_road_km", 0.0) > 0:
            names.append("OSRM\n(road)")
            vals.append(savings["optimized_osrm_road_km"])
            cols.append("#2ca02c")
        bars = ax_bar.bar(names, vals, color=cols, edgecolor="#333", linewidth=0.6)
        ax_bar.set_ylabel("Distance (km)")
        ax_bar.set_title("Aggregate distance comparison")
        for b, v in zip(bars, vals, strict=False):
            ax_bar.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}", ha="center", fontsize=9)

        ax_txt = self._fig.add_axes([0.58, 0.30, 0.38, 0.54])
        ax_txt.axis("off")
        vr_ok = sum(1 for r in results if r.vrptw_ok)
        vr_txt = (
            f"VRPTW routes OK : {vr_ok}/{len(results)}\n"
            f"km saved (GA)  : {savings['saved_km_vs_naive_ga']:.2f}\n"
            f"km saved (A*)  : {savings['saved_km_vs_naive_astar']:.2f}\n"
            f"km saved (OSRM): {savings.get('saved_km_vs_naive_osrm', 0.0):.2f}\n"
            f"Fleet routes   : {len(results)}\n"
        )
        ax_txt.text(0, 1, vr_txt, va="top", fontsize=11, family="monospace")

        self._annotate_flow(
            [
                "DONE",
                "JSON report prints",
                "to stdout next.",
                "",
                "(Close window)",
            ]
        )
        self._flush("summary")


def present_full_pipeline(
    orders: list[dict],
    drivers: list[dict],
    *,
    dbscan_eps_km: float = 1.4,
    pause_seconds: float = 2.0,
    out_dir: Path | None = None,
    headless: bool | None = None,
    use_osrm: bool = False,
    osrm_base_url: str = "http://localhost:5000",
) -> tuple[list[RouteResult], dict[str, float], list[dict[str, Any]], np.ndarray, dict[int, int], dict[str, Any]]:
    """Run slideshow + optimization once; returns (results, savings, merged, labels, assignment, grouping meta)."""
    cfg = PresenterConfig(
        pause_seconds=pause_seconds,
        out_dir=out_dir,
        headless=headless if headless is not None else _is_headless(),
    )
    viz = PipelineVisualizer(cfg)

    merged, _ = merge_same_location_orders(orders)
    labels, _ = cluster_deliveries_dbscan(merged, eps_km=dbscan_eps_km, min_samples=2)
    clusters = stops_by_cluster(labels)
    assignment = assign_nearest_driver(clusters, merged, drivers)

    viz.slide_01_orders(orders, drivers)
    viz.slide_02_merge_rule1(orders, merged)
    viz.slide_03_clusters(merged, labels)
    viz.slide_04_assignment(merged, labels, assignment, drivers)

    results, pipeline_info = run_optimized_routes(
        orders,
        drivers,
        dbscan_eps_km=dbscan_eps_km,
        use_osrm=use_osrm,
        osrm_base_url=osrm_base_url,
    )
    savings = summarize_savings(orders, drivers, results)
    multi = sum(1 for _c, idxs in clusters.items() if len(idxs) > 1)

    all_legs_equal = all(r.dijkstra_star_equal for r in results)

    viz.slide_05_ga_intro(multi)
    viz.slide_06_navigation_vrptw(all_legs_equal, osrm_requested=use_osrm)

    dmap = {int(d["driver_id"]): d for d in drivers}
    routes_plot: list[list[tuple[float, float]]] = []
    for r in results:
        drv = dmap[r.driver_id]
        routes_plot.append(r.osrm_geometry if use_osrm and r.osrm_geometry else [tuple(drv["current_location"])] + r.drop_coords_ordered)

    viz.slide_07_before_after(merged, drivers, routes_plot)
    viz.slide_08_summary(savings, results)

    if not cfg.headless and not _is_headless():
        plt.ioff()
        plt.show()

    return results, savings, merged, labels, assignment, pipeline_info


def quick_demo() -> None:
    orders = synthetic_orders(26, seed=42)
    drivers = synthetic_drivers(5, seed=42)
    present_full_pipeline(orders, drivers)


if __name__ == "__main__":
    quick_demo()
