"""
Click-based visual input/process/output app for Last-Mile + OSRM.

Run with:
    python main.py --visual-input [--viz-mode graph|map]

Controls:
    - Click "Driver mode", then click the map to add driver start points.
    - Click "Order mode", then click the map to add delivery drops.
    - Click "Run pipeline" to see input, clustering/assignment, and route output.
    - Use "View: OSM" / "View: grid" to switch map tiles vs grid-only (also ``--viz-mode``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from core.batching import merge_same_location_orders
from core.clustering import cluster_deliveries_dbscan
from core.driver_assignment import assign_nearest_driver, stops_by_cluster
from core.routing import RouteResult, run_optimized_routes, select_route_polyline, summarize_savings
from map_basemap import normalize_viz_mode, pad_lonlat_extent, try_osm_basemap
from utils import ODISHA_REGION_CENTER, synthetic_drivers, synthetic_orders


def _fmt_clock(mins_since_midnight: float) -> str:
    total = max(0, int(round(mins_since_midnight)))
    hh = (total // 60) % 24
    mm = total % 60
    return f"{hh:02d}:{mm:02d}"


def _osrm_status_label(osrm_info: dict[str, Any]) -> str:
    """Compact human-readable summary of the OSRM pipeline state."""
    if not osrm_info.get("requested"):
        return "OSRM: not requested"
    enriched = int(osrm_info.get("enriched_routes", 0))
    total = int(osrm_info.get("total_routes", 0))
    if not osrm_info.get("connected"):
        reason = osrm_info.get("reason") or "unavailable"
        return f"OSRM: not connected ({reason})"
    if total > 0 and 0 < enriched < total:
        return f"OSRM: partial enrichment {enriched}/{total}"
    if total > 0 and enriched == total:
        return f"OSRM: connected ({enriched}/{total} enriched)"
    return "OSRM: connected"


class VisualOsrmApp:
    def __init__(self, osrm_base_url: str, out_dir: Path | None = None, *, viz_mode: str = "map") -> None:
        self.osrm_base_url = osrm_base_url
        self.out_dir = out_dir or Path(__file__).resolve().parent
        self.viz_mode: str = normalize_viz_mode(viz_mode)
        self.mode = "order"
        self.orders: list[dict[str, Any]] = []
        self.drivers: list[dict[str, Any]] = []
        self._next_order_id = 1
        self._next_driver_id = 1
        self._last_pipeline: tuple[Any, ...] | None = None
        self._output_fig: plt.Figure | None = None
        self._last_osrm_status: str = "OSRM: idle (run pipeline to probe)"

        self.fig = plt.figure(figsize=(12.5, 7.2))
        self.ax = self.fig.add_axes([0.06, 0.14, 0.70, 0.78])
        self.info_ax = self.fig.add_axes([0.78, 0.14, 0.20, 0.78])
        self.info_ax.axis("off")

        self._buttons: list[Button] = []
        self._add_button([0.06, 0.04, 0.13, 0.055], "Order mode", lambda _event: self._set_mode("order"))
        self._add_button([0.205, 0.04, 0.13, 0.055], "Driver mode", lambda _event: self._set_mode("driver"))
        self._add_button([0.35, 0.04, 0.13, 0.055], "Load sample", lambda _event: self._load_sample())
        self._add_button([0.495, 0.04, 0.13, 0.055], "Run pipeline", lambda _event: self._run_pipeline())
        self._add_button([0.64, 0.04, 0.095, 0.055], "Reset", lambda _event: self._reset())
        ax_viz = self.fig.add_axes([0.738, 0.04, 0.125, 0.055])
        self._viz_mode_btn = Button(ax_viz, "")
        self._viz_mode_btn.on_clicked(self._toggle_viz_mode)
        self._buttons.append(self._viz_mode_btn)
        self._sync_viz_button_label()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._redraw_input()

    def _sync_viz_button_label(self) -> None:
        self._viz_mode_btn.label.set_text("View: OSM" if self.viz_mode == "map" else "View: grid")

    def _toggle_viz_mode(self, _event: Any = None) -> None:
        self.viz_mode = "graph" if self.viz_mode == "map" else "map"
        self._sync_viz_button_label()
        self._redraw_input()
        if self._last_pipeline is not None:
            self._render_process_output(*self._last_pipeline)

    def show(self) -> None:
        plt.show()

    def _add_button(self, rect: list[float], label: str, callback: Any) -> None:
        ax_btn = self.fig.add_axes(rect)
        btn = Button(ax_btn, label)
        btn.on_clicked(callback)
        self._buttons.append(btn)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._redraw_input()

    def _load_sample(self) -> None:
        self.orders = synthetic_orders(18, seed=8)
        self.drivers = synthetic_drivers(4, seed=8)
        self._next_order_id = max(int(o["order_id"]) for o in self.orders) + 1
        self._next_driver_id = max(int(d["driver_id"]) for d in self.drivers) + 1
        self._redraw_input()

    def _reset(self) -> None:
        self.orders.clear()
        self.drivers.clear()
        self._next_order_id = 1
        self._next_driver_id = 1
        self._last_pipeline = None
        if self._output_fig is not None:
            plt.close(self._output_fig)
            self._output_fig = None
        self._redraw_input()

    def _on_click(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        lat, lon = float(event.ydata), float(event.xdata)
        if self.mode == "driver":
            self.drivers.append(
                {
                    "driver_id": self._next_driver_id,
                    "current_location": (lat, lon),
                    "capacity": 30.0,
                }
            )
            self._next_driver_id += 1
        else:
            self.orders.append(
                {
                    "order_id": self._next_order_id,
                    "user_id": 1000 + self._next_order_id,
                    "pickup": ODISHA_REGION_CENTER,
                    "drop": (lat, lon),
                    "time_window": [9 * 60, 18 * 60],
                    "parcel_weight": 1.0,
                }
            )
            self._next_order_id += 1
        self._redraw_input()

    def _redraw_input(self) -> None:
        self.ax.clear()
        self.ax.set_title("Graphical input: click to add orders or drivers")
        self.ax.set_aspect("equal", adjustable="box")

        if not self.orders and not self.drivers:
            # Default window: Bhubaneswar / wider Odisha coast-inland area
            lat_c, lon_c = ODISHA_REGION_CENTER
            self.ax.set_xlim(lon_c - 0.10, lon_c + 0.10)
            self.ax.set_ylim(lat_c - 0.10, lat_c + 0.10)
        else:
            all_points = [tuple(o["drop"]) for o in self.orders] + [tuple(d["current_location"]) for d in self.drivers]
            lats = [p[0] for p in all_points]
            lons = [p[1] for p in all_points]
            (min_lon, max_lon), (min_lat, max_lat) = pad_lonlat_extent(lons, lats, pad_deg=0.012)
            self.ax.set_xlim(min_lon, max_lon)
            self.ax.set_ylim(min_lat, max_lat)

        had_basemap = try_osm_basemap(self.ax, viz_mode=self.viz_mode)
        if not had_basemap:
            self.ax.grid(True, alpha=0.25)

        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")

        if self.orders:
            drops = [tuple(o["drop"]) for o in self.orders]
            self.ax.scatter([p[1] for p in drops], [p[0] for p in drops], c="#d62728", s=48, label="Orders", zorder=5, edgecolors="#fff", linewidths=0.35)
            for order in self.orders:
                lat, lon = order["drop"]
                self.ax.annotate(str(order["order_id"]), (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=8, zorder=6)
        if self.drivers:
            starts = [tuple(d["current_location"]) for d in self.drivers]
            self.ax.scatter([p[1] for p in starts], [p[0] for p in starts], c="#2ca02c", marker="*", s=160, label="Drivers", zorder=5, edgecolors="#fff", linewidths=0.35)
            for driver in self.drivers:
                lat, lon = driver["current_location"]
                self.ax.annotate(f"D{driver['driver_id']}", (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=8, zorder=6)

        if self.orders or self.drivers:
            self.ax.legend(loc="upper left", fontsize=8)

        self._write_info(
            [
                "Last-Mile + OSRM visual app",
                "",
                f"Mode       : {self.mode}",
                f"Viz mode   : {self.viz_mode}",
                f"Orders     : {len(self.orders)}",
                f"Drivers    : {len(self.drivers)}",
                f"OSRM URL   : {self.osrm_base_url}",
                "",
                self._last_osrm_status,
                "",
                "Tiles/grid: View button",
                "bottom-right.",
                "",
                "Click map to add points.",
                "Run needs one driver",
                "and one order.",
            ]
        )
        self.fig.canvas.draw_idle()

    def _write_info(self, lines: list[str]) -> None:
        self.info_ax.clear()
        self.info_ax.axis("off")
        self.info_ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9.5)

    def _run_pipeline(self) -> None:
        if not self.orders or not self.drivers:
            self._write_info(["Add at least one order", "and one driver first."])
            self.fig.canvas.draw_idle()
            return

        merged, _ = merge_same_location_orders(self.orders)
        labels, _ = cluster_deliveries_dbscan(merged, eps_km=1.4, min_samples=2)
        clusters = stops_by_cluster(labels)
        assignment = assign_nearest_driver(clusters, merged, self.drivers)
        results, pipeline_info = run_optimized_routes(
            self.orders,
            self.drivers,
            dbscan_eps_km=1.4,
            use_osrm=True,
            osrm_base_url=self.osrm_base_url,
        )
        savings = summarize_savings(self.orders, self.drivers, results)
        self._last_osrm_status = _osrm_status_label(pipeline_info.get("osrm", {}))
        self._last_pipeline = (merged, labels, assignment, results, savings, pipeline_info)
        self._render_process_output(merged, labels, assignment, results, savings, pipeline_info)

    def _render_process_output(
        self,
        merged: list[dict[str, Any]],
        labels: np.ndarray,
        assignment: dict[int, int],
        results: list[RouteResult],
        savings: dict[str, float],
        pipeline_info: dict[str, Any],
    ) -> None:
        if self._output_fig is not None:
            plt.close(self._output_fig)
            self._output_fig = None
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
        self._output_fig = fig
        fig.suptitle("Last-Mile OSRM visual input -> process -> output", fontsize=14, fontweight="bold")

        self._draw_input_panel(axes[0])
        self._draw_process_panel(axes[1], merged, labels, assignment)
        self._draw_output_panel(axes[2], results)

        osrm_info = pipeline_info.get("osrm", {})
        status_label = _osrm_status_label(osrm_info)
        status_counts = osrm_info.get("status_counts", {})
        summary = (
            f"Routes: {len(results)} | "
            f"GA: {savings['optimized_ga_open_tour_km']:.2f} km | "
            f"Dijkstra: {savings['optimized_dijkstra_graph_km']:.2f} km | "
            f"A* quick: {savings['optimized_astar_quick_km']:.2f} km | "
            f"OSRM: {savings['optimized_osrm_road_km']:.2f} km | "
            f"{status_label} | counts: {status_counts}"
        )
        fig.suptitle(
            f"Last-Mile OSRM visual input -> process -> output  ·  {status_label}",
            fontsize=13,
            fontweight="bold",
        )
        fig.text(0.5, 0.02, summary, ha="center", family="monospace", fontsize=9)
        path = self.out_dir / "output_osrm_visual_process.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.show(block=False)
        eta_lines: list[str] = ["ETA per order:"]
        for route in results:
            if not route.metas_ordered:
                continue
            anchor_min = min(float(m["time_window"][0]) for m in route.metas_ordered)
            eta_lines.append(f"D{route.driver_id}/C{route.cluster_id}")
            for meta, eta_rel in zip(route.metas_ordered, route.eta_arrival_min, strict=False):
                order_id = int(meta.get("order_id", -1))
                eta_abs = anchor_min + float(eta_rel)
                eta_lines.append(f"  O{order_id}: {_fmt_clock(eta_abs)} (+{eta_rel:.1f}m)")
        self._write_info(
            [
                "Pipeline complete.",
                "",
                status_label,
                "",
                f"Saved: {path.name}",
                "",
                summary,
                "",
                *eta_lines,
            ]
        )
        self.fig.canvas.draw_idle()

    def _draw_input_panel(self, ax: Any) -> None:
        ax.set_title("1. Input")
        ax.set_aspect("equal", adjustable="box")
        lons: list[float] = []
        lats: list[float] = []
        if self.orders:
            for o in self.orders:
                lat, lon = o["drop"]
                lats.append(lat)
                lons.append(lon)
        if self.drivers:
            for d in self.drivers:
                lat, lon = d["current_location"]
                lats.append(lat)
                lons.append(lon)
        xlim, ylim = pad_lonlat_extent(lons, lats, pad_deg=0.012)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        had_bm = try_osm_basemap(ax, viz_mode=self.viz_mode)
        if self.orders:
            drops = [tuple(o["drop"]) for o in self.orders]
            ax.scatter([p[1] for p in drops], [p[0] for p in drops], c="#d62728", s=42, label="Orders", zorder=5, edgecolors="#fff", linewidths=0.3)
        if self.drivers:
            starts = [tuple(d["current_location"]) for d in self.drivers]
            ax.scatter([p[1] for p in starts], [p[0] for p in starts], c="#2ca02c", marker="*", s=130, label="Drivers", zorder=5, edgecolors="#fff", linewidths=0.3)
        self._finish_map_axis(ax, grid=not had_bm)

    def _draw_process_panel(self, ax: Any, merged: list[dict[str, Any]], labels: np.ndarray, assignment: dict[int, int]) -> None:
        ax.set_title("2. Process: clusters + assignment")
        ax.set_aspect("equal", adjustable="box")
        drops = [tuple(m["drop"]) for m in merged]
        lons = [p[1] for p in drops]
        lats = [p[0] for p in drops]
        for d in self.drivers:
            lat, lon = d["current_location"]
            lats.append(lat)
            lons.append(lon)
        xlim, ylim = pad_lonlat_extent(lons, lats, pad_deg=0.012)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        had_bm = try_osm_basemap(ax, viz_mode=self.viz_mode)
        sc = ax.scatter(
            [p[1] for p in drops],
            [p[0] for p in drops],
            c=labels,
            cmap="tab20",
            s=48,
            edgecolors="k",
            linewidths=0.3,
            zorder=5,
        )
        clusters = stops_by_cluster(labels)
        dmap = {int(d["driver_id"]): d for d in self.drivers}
        for cid, idxs in clusters.items():
            pts = np.array([[merged[i]["drop"][1], merged[i]["drop"][0]] for i in idxs])
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            did = assignment.get(cid)
            if did is None:
                continue
            drv = dmap[did]
            dx, dy = drv["current_location"][1], drv["current_location"][0]
            ax.plot([cx, dx], [cy, dy], "k--", alpha=0.35, linewidth=1, zorder=4)
        self._finish_map_axis(ax, grid=not had_bm)
        plt.colorbar(sc, ax=ax, shrink=0.72, label="Cluster")

    def _draw_output_panel(self, ax: Any, results: list[RouteResult]) -> None:
        ax.set_title("3. Output: optimized road routes")
        ax.set_aspect("equal", adjustable="box")
        dmap = {int(d["driver_id"]): d for d in self.drivers}
        lons: list[float] = []
        lats: list[float] = []
        polylines: list[list[tuple[float, float]]] = []
        for route in results:
            driver = dmap[route.driver_id]
            poly = select_route_polyline(route, tuple(driver["current_location"]))
            polylines.append(poly)
            for lat, lon in poly:
                lats.append(lat)
                lons.append(lon)
        xlim, ylim = pad_lonlat_extent(lons, lats, pad_deg=0.012)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        had_bm = try_osm_basemap(ax, viz_mode=self.viz_mode)
        for route, poly in zip(results, polylines):
            ax.plot(
                [p[1] for p in poly],
                [p[0] for p in poly],
                "-o",
                lw=1.6,
                markersize=3.5,
                label=f"D{route.driver_id}/C{route.cluster_id}",
                zorder=5,
            )
            if route.metas_ordered and route.eta_arrival_min:
                anchor_min = min(float(m["time_window"][0]) for m in route.metas_ordered)
                for stop_idx, (coord, meta, eta_rel) in enumerate(
                    zip(route.drop_coords_ordered, route.metas_ordered, route.eta_arrival_min, strict=False)
                ):
                    lat, lon = coord
                    order_id = int(meta.get("order_id", stop_idx + 1))
                    eta_abs = anchor_min + float(eta_rel)
                    ax.annotate(
                        f"O{order_id} ETA {_fmt_clock(eta_abs)}",
                        (lon, lat),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=7.5,
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75, "ec": "none"},
                        zorder=7,
                    )
        self._finish_map_axis(ax, grid=not had_bm)

    @staticmethod
    def _finish_map_axis(ax: Any, *, grid: bool = True) -> None:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if grid:
            ax.grid(True, alpha=0.25)
        handles, labels_leg = ax.get_legend_handles_labels()
        if handles and labels_leg:
            ax.legend(loc="best", fontsize=8)


def run_visual_osrm_app(
    osrm_base_url: str = "http://localhost:5000",
    out_dir: Path | None = None,
    *,
    viz_mode: str = "map",
) -> None:
    app = VisualOsrmApp(osrm_base_url=osrm_base_url, out_dir=out_dir, viz_mode=viz_mode)
    app.show()


if __name__ == "__main__":
    run_visual_osrm_app()
