"""
Click-based visual input/process/output app for Last-Mile + OSRM.

Run with:
    python main.py --visual-input --use-osrm

Controls:
    - Click "Driver mode", then click the map to add driver start points.
    - Click "Order mode", then click the map to add delivery drops.
    - Click "Run pipeline" to see input, clustering/assignment, and route output.
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
from core.routing import RouteResult, run_optimized_routes, summarize_savings
from utils import synthetic_drivers, synthetic_orders


class VisualOsrmApp:
    def __init__(self, osrm_base_url: str, out_dir: Path | None = None) -> None:
        self.osrm_base_url = osrm_base_url
        self.out_dir = out_dir or Path(__file__).resolve().parent
        self.mode = "order"
        self.orders: list[dict[str, Any]] = []
        self.drivers: list[dict[str, Any]] = []
        self._next_order_id = 1
        self._next_driver_id = 1

        self.fig = plt.figure(figsize=(12.5, 7.2))
        self.ax = self.fig.add_axes([0.06, 0.14, 0.70, 0.78])
        self.info_ax = self.fig.add_axes([0.78, 0.14, 0.20, 0.78])
        self.info_ax.axis("off")

        self._buttons: list[Button] = []
        self._add_button([0.06, 0.04, 0.13, 0.055], "Order mode", lambda _event: self._set_mode("order"))
        self._add_button([0.205, 0.04, 0.13, 0.055], "Driver mode", lambda _event: self._set_mode("driver"))
        self._add_button([0.35, 0.04, 0.13, 0.055], "Load sample", lambda _event: self._load_sample())
        self._add_button([0.495, 0.04, 0.13, 0.055], "Run pipeline", lambda _event: self._run_pipeline())
        self._add_button([0.64, 0.04, 0.10, 0.055], "Reset", lambda _event: self._reset())

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._redraw_input()

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
                    "pickup": (12.97, 77.59),
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
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.grid(True, alpha=0.25)
        self.ax.set_aspect("equal", adjustable="box")

        if self.orders:
            drops = [tuple(o["drop"]) for o in self.orders]
            self.ax.scatter([p[1] for p in drops], [p[0] for p in drops], c="#d62728", s=48, label="Orders")
            for order in self.orders:
                lat, lon = order["drop"]
                self.ax.annotate(str(order["order_id"]), (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=8)
        if self.drivers:
            starts = [tuple(d["current_location"]) for d in self.drivers]
            self.ax.scatter([p[1] for p in starts], [p[0] for p in starts], c="#2ca02c", marker="*", s=160, label="Drivers")
            for driver in self.drivers:
                lat, lon = driver["current_location"]
                self.ax.annotate(f"D{driver['driver_id']}", (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=8)

        if not self.orders and not self.drivers:
            self.ax.set_xlim(77.50, 77.68)
            self.ax.set_ylim(12.89, 13.05)
        else:
            all_points = [tuple(o["drop"]) for o in self.orders] + [tuple(d["current_location"]) for d in self.drivers]
            lats = [p[0] for p in all_points]
            lons = [p[1] for p in all_points]
            pad = 0.012
            self.ax.set_xlim(min(lons) - pad, max(lons) + pad)
            self.ax.set_ylim(min(lats) - pad, max(lats) + pad)
        if self.orders or self.drivers:
            self.ax.legend(loc="upper left", fontsize=8)

        self._write_info(
            [
                "Last-Mile + OSRM visual app",
                "",
                f"Mode       : {self.mode}",
                f"Orders     : {len(self.orders)}",
                f"Drivers    : {len(self.drivers)}",
                f"OSRM URL   : {self.osrm_base_url}",
                "",
                "Click map to add points.",
                "Run needs at least one",
                "driver and one order.",
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
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
        fig.suptitle("Last-Mile OSRM visual input -> process -> output", fontsize=14, fontweight="bold")

        self._draw_input_panel(axes[0])
        self._draw_process_panel(axes[1], merged, labels, assignment)
        self._draw_output_panel(axes[2], results)

        osrm_status = pipeline_info.get("osrm", {}).get("status_counts", {})
        summary = (
            f"Routes: {len(results)} | "
            f"GA: {savings['optimized_ga_open_tour_km']:.2f} km | "
            f"A*: {savings['optimized_astar_graph_km']:.2f} km | "
            f"OSRM: {savings['optimized_osrm_road_km']:.2f} km | "
            f"OSRM status: {osrm_status}"
        )
        fig.text(0.5, 0.02, summary, ha="center", family="monospace", fontsize=9)
        path = self.out_dir / "output_osrm_visual_process.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.show(block=False)
        self._write_info(["Pipeline complete.", "", f"Saved: {path.name}", "", summary])
        self.fig.canvas.draw_idle()

    def _draw_input_panel(self, ax: Any) -> None:
        ax.set_title("1. Input")
        if self.orders:
            drops = [tuple(o["drop"]) for o in self.orders]
            ax.scatter([p[1] for p in drops], [p[0] for p in drops], c="#d62728", s=42, label="Orders")
        if self.drivers:
            starts = [tuple(d["current_location"]) for d in self.drivers]
            ax.scatter([p[1] for p in starts], [p[0] for p in starts], c="#2ca02c", marker="*", s=130, label="Drivers")
        self._finish_map_axis(ax)

    def _draw_process_panel(self, ax: Any, merged: list[dict[str, Any]], labels: np.ndarray, assignment: dict[int, int]) -> None:
        ax.set_title("2. Process: clusters + assignment")
        drops = [tuple(m["drop"]) for m in merged]
        sc = ax.scatter([p[1] for p in drops], [p[0] for p in drops], c=labels, cmap="tab20", s=48, edgecolors="k", linewidths=0.3)
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
            ax.plot([cx, dx], [cy, dy], "k--", alpha=0.35, linewidth=1)
        self._finish_map_axis(ax)
        plt.colorbar(sc, ax=ax, shrink=0.72, label="Cluster")

    def _draw_output_panel(self, ax: Any, results: list[RouteResult]) -> None:
        ax.set_title("3. Output: optimized road routes")
        dmap = {int(d["driver_id"]): d for d in self.drivers}
        for route in results:
            driver = dmap[route.driver_id]
            poly = route.osrm_geometry if route.osrm_geometry and route.osrm_road_km is not None else [tuple(driver["current_location"])] + route.drop_coords_ordered
            ax.plot([p[1] for p in poly], [p[0] for p in poly], "-o", lw=1.6, markersize=3.5, label=f"D{route.driver_id}/C{route.cluster_id}")
        self._finish_map_axis(ax)

    @staticmethod
    def _finish_map_axis(ax: Any) -> None:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="best", fontsize=8)


def run_visual_osrm_app(osrm_base_url: str = "http://localhost:5000", out_dir: Path | None = None) -> None:
    app = VisualOsrmApp(osrm_base_url=osrm_base_url, out_dir=out_dir)
    app.show()


if __name__ == "__main__":
    run_visual_osrm_app()
