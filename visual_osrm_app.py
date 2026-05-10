"""
Click-based visual input/process/output app for Last-Mile + OSRM.

Run with:
    python main.py --visual-input [--viz-mode graph|map] [--use-vrptw]
    python main.py --visual-input --orders-csv data/sample_orders.csv --drivers-csv data/sample_drivers.csv

Controls:
    - Click "Driver mode", then click the map to add driver start points.
    - Click "Order mode", then click the map to add delivery drops.
    - Toggle "VRPTW: off/on" to opt into time-window aware optimization. When
      ON, every new order prompts for a preferred delivery time (HH:MM) and
      the GA orders stops by their windows even when a later-windowed customer
      is geographically closer to the driver.
    - Click "Run pipeline" to see input, clustering/assignment, and route output
      (writes ``output/csv/``, ``output/json/``, and ``output/images/osrm_visual_process_<stamp>.png``).
    - Click "Load sample" — uses ``data/sample_orders.csv`` when VRPTW is off, or
      ``data/sample_orders_vrptw.csv`` (preferred times + tight windows) when VRPTW is on;
      drivers always come from ``data/sample_drivers.csv``.
    - Use "View: OSM" / "View: grid" to switch map tiles vs grid-only (also ``--viz-mode``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

from core.batching import merge_same_location_orders
from core.clustering import cluster_deliveries_dbscan
from core.driver_assignment import assign_nearest_driver, stops_by_cluster
from core.routing import RouteResult, run_optimized_routes, select_route_polyline, summarize_savings
from pipeline_csv_log import (
    build_pipeline_report_dict,
    new_run_stamp_utc,
    output_csv_dir,
    output_images_dir,
    output_json_dir,
    save_pipeline_report_json,
    write_pipeline_run_csv,
)
from map_basemap import normalize_viz_mode, pad_lonlat_extent, try_osm_basemap
from utils import (
    ODISHA_REGION_CENTER,
    hydrate_synthetic_orders_for_vrptw_visual,
    load_drivers_csv,
    load_orders_csv,
    synthetic_drivers,
    synthetic_orders,
)


# ±30 min default tolerance around the user's preferred delivery time. Tight
# enough to make ordering pressure visible in the GA; wide enough that small
# travel-time mis-estimates don't immediately mark the route infeasible.
_VRPTW_WINDOW_HALF_MIN: float = 30.0

# Keep in sync with ``main.py`` / ``run_optimized_routes`` clustering defaults.
_DBSCAN_EPS_KM: float = 1.4


def _parse_clock_to_minutes(text: str | None) -> int | None:
    """Parse '13:00', '1:30PM', '0930', '9' into minutes-since-midnight.

    Returns ``None`` for empty input or anything that can't be coerced; callers
    fall back to a default in that case.
    """
    if text is None:
        return None
    s = text.strip().upper().replace(" ", "")
    if not s:
        return None
    suffix = ""
    if s.endswith("PM"):
        suffix = "PM"
        s = s[:-2]
    elif s.endswith("AM"):
        suffix = "AM"
        s = s[:-2]
    h: int
    m: int = 0
    if ":" in s:
        parts = s.split(":")
        try:
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        except ValueError:
            return None
    else:
        try:
            n = int(s)
        except ValueError:
            return None
        if n >= 100:
            h = n // 100
            m = n % 100
        else:
            h = n
    if suffix == "PM" and h < 12:
        h += 12
    if suffix == "AM" and h == 12:
        h = 0
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return h * 60 + m


# NOTE: We deliberately do NOT spawn a second matplotlib Figure with a nested
# event loop to ask for the preferred time. On macOS the `macosx` backend
# segfaults when a second figure is shown from inside another figure's
# mouse-event callback. Instead, ``VisualOsrmApp`` renders an inline overlay
# (TextBox + OK/Cancel buttons) on the main figure and tracks a pending order
# coordinate while the user types. See ``_start_pending_order``.


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
    def __init__(
        self,
        osrm_base_url: str,
        out_dir: Path | None = None,
        *,
        viz_mode: str = "map",
        initial_use_vrptw: bool = False,
        initial_orders: list[dict[str, Any]] | None = None,
        initial_drivers: list[dict[str, Any]] | None = None,
    ) -> None:
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
        self._manual_view_xlim: tuple[float, float] | None = None
        self._manual_view_ylim: tuple[float, float] | None = None
        self._drag_press_data: tuple[float, float, tuple[float, float], tuple[float, float]] | None = None
        self._dragging = False
        # VRPTW toggle state. When ON, every newly added order is asked for a
        # preferred time and the routing pipeline runs the time-window-aware
        # GA. When OFF, the system behaves exactly as before.
        self.use_vrptw: bool = bool(initial_use_vrptw)
        # Coordinates of an order whose preferred time is still being typed by
        # the user. While not None, map clicks are ignored and the inline
        # overlay is visible.
        self._pending_coord: tuple[float, float] | None = None

        self.fig = plt.figure(figsize=(12.5, 7.2))
        self.ax = self.fig.add_axes([0.06, 0.14, 0.70, 0.78])
        self.info_ax = self.fig.add_axes([0.78, 0.14, 0.20, 0.78])
        self.info_ax.axis("off")

        self._buttons: list[Button] = []
        self._add_button([0.030, 0.04, 0.105, 0.055], "Order mode", lambda _event: self._set_mode("order"))
        self._add_button([0.140, 0.04, 0.105, 0.055], "Driver mode", lambda _event: self._set_mode("driver"))
        self._add_button([0.250, 0.04, 0.105, 0.055], "Load sample", lambda _event: self._load_sample())
        ax_vrptw = self.fig.add_axes([0.360, 0.04, 0.115, 0.055])
        self._vrptw_btn = Button(ax_vrptw, "")
        self._vrptw_btn.on_clicked(self._toggle_vrptw)
        self._buttons.append(self._vrptw_btn)
        self._sync_vrptw_button_label()
        self._add_button([0.480, 0.04, 0.110, 0.055], "Run pipeline", lambda _event: self._run_pipeline())
        self._add_button([0.595, 0.04, 0.080, 0.055], "Reset", lambda _event: self._reset())
        ax_viz = self.fig.add_axes([0.680, 0.04, 0.110, 0.055])
        self._viz_mode_btn = Button(ax_viz, "")
        self._viz_mode_btn.on_clicked(self._toggle_viz_mode)
        self._buttons.append(self._viz_mode_btn)
        self._sync_viz_button_label()

        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)

        # Inline VRPTW dialog overlay. We build it once and keep it hidden
        # until a VRPTW order click arrives. Using axes on the same figure
        # avoids the macOS `macosx`-backend segfault that triggers when a
        # second figure's event loop is started from inside a callback.
        self._build_pending_overlay()

        if initial_orders is None and initial_drivers is None:
            pass
        elif initial_orders is not None and initial_drivers is not None:
            self._set_orders_and_drivers(initial_orders, initial_drivers)
        else:
            raise ValueError("initial_orders and initial_drivers must both be provided or both omitted")

        self._redraw_input()

    def _sync_viz_button_label(self) -> None:
        self._viz_mode_btn.label.set_text("View: OSM" if self.viz_mode == "map" else "View: grid")

    def _sync_vrptw_button_label(self) -> None:
        self._vrptw_btn.label.set_text("VRPTW: on" if self.use_vrptw else "VRPTW: off")
        self._vrptw_btn.color = "#cce5cc" if self.use_vrptw else "0.85"
        self._vrptw_btn.hovercolor = "#b6dab6" if self.use_vrptw else "0.95"

    def _toggle_vrptw(self, _event: Any = None) -> None:
        self.use_vrptw = not self.use_vrptw
        self._sync_vrptw_button_label()
        # Toggling VRPTW always discards a pending order so we never end up
        # with an "asking for time" overlay while VRPTW is off.
        if self._pending_coord is not None:
            self._pending_coord = None
            self._set_overlay_visible(False)
        self._redraw_input()

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

    def _build_pending_overlay(self) -> None:
        """Create the inline 'preferred time' dialog over the main figure.

        The overlay lives on the same figure so we never have to spawn a
        nested matplotlib event loop (which segfaults on macOS).
        """
        # Card background — covers a small area in the centre of the figure.
        self._overlay_card_ax = self.fig.add_axes([0.24, 0.30, 0.46, 0.30])
        self._overlay_card_ax.set_xticks([])
        self._overlay_card_ax.set_yticks([])
        self._overlay_card_ax.set_facecolor("#ffffff")
        for spine in self._overlay_card_ax.spines.values():
            spine.set_edgecolor("#888")
            spine.set_linewidth(1.4)

        self._overlay_title = self._overlay_card_ax.text(
            0.5,
            0.86,
            "Preferred delivery time",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            transform=self._overlay_card_ax.transAxes,
        )
        self._overlay_hint = self._overlay_card_ax.text(
            0.5,
            0.66,
            "e.g. 13:00  ·  1:30PM  ·  9:00AM  ·  0930",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#555",
            transform=self._overlay_card_ax.transAxes,
        )

        # TextBox + buttons — separate axes layered above the card. Their
        # rectangles are absolute (figure coordinates) so they sit cleanly on
        # top of the card.
        self._overlay_tb_ax = self.fig.add_axes([0.30, 0.435, 0.34, 0.045])
        self._overlay_textbox = TextBox(
            self._overlay_tb_ax,
            "Time:",
            initial="",
            textalignment="center",
        )
        self._overlay_textbox.on_submit(self._on_pending_text_submit)

        self._overlay_ok_ax = self.fig.add_axes([0.30, 0.345, 0.16, 0.055])
        self._overlay_ok_btn = Button(
            self._overlay_ok_ax,
            "OK",
            color="#cce5cc",
            hovercolor="#b6dab6",
        )
        self._overlay_ok_btn.on_clicked(self._submit_pending_time)

        self._overlay_cancel_ax = self.fig.add_axes([0.48, 0.345, 0.16, 0.055])
        self._overlay_cancel_btn = Button(self._overlay_cancel_ax, "Cancel")
        self._overlay_cancel_btn.on_clicked(self._cancel_pending)

        self._overlay_axes = [
            self._overlay_card_ax,
            self._overlay_tb_ax,
            self._overlay_ok_ax,
            self._overlay_cancel_ax,
        ]
        self._set_overlay_visible(False)

    def _set_overlay_visible(self, visible: bool) -> None:
        for ax in self._overlay_axes:
            ax.set_visible(visible)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _start_pending_order(self, lat: float, lon: float) -> None:
        """Stash the click coordinate and surface the inline time prompt."""
        self._pending_coord = (lat, lon)
        self._overlay_textbox.set_val("")
        self._overlay_title.set_text(f"Preferred time for Order O{self._next_order_id}")
        self._overlay_hint.set_text("e.g. 13:00  ·  1:30PM  ·  9:00AM  ·  0930")
        self._overlay_hint.set_color("#555")
        self._set_overlay_visible(True)

    def _on_pending_text_submit(self, _text: str) -> None:
        # Pressing Enter inside the textbox is equivalent to clicking OK.
        self._submit_pending_time()

    def _submit_pending_time(self, _event: Any = None) -> None:
        if self._pending_coord is None:
            return
        parsed = _parse_clock_to_minutes(self._overlay_textbox.text)
        if parsed is None:
            self._overlay_hint.set_text("Couldn't parse — try 13:00, 1:30PM, 9, or 0930")
            self._overlay_hint.set_color("#b00020")
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass
            return
        lat, lon = self._pending_coord
        order: dict[str, Any] = {
            "order_id": self._next_order_id,
            "user_id": 1000 + self._next_order_id,
            "pickup": ODISHA_REGION_CENTER,
            "drop": (lat, lon),
            "preferred_minute": float(parsed),
            "time_window": [
                parsed - _VRPTW_WINDOW_HALF_MIN,
                parsed + _VRPTW_WINDOW_HALF_MIN,
            ],
            "parcel_weight": 1.0,
        }
        self.orders.append(order)
        self._next_order_id += 1
        self._pending_coord = None
        self._set_overlay_visible(False)
        self._redraw_input()

    def _cancel_pending(self, _event: Any = None) -> None:
        if self._pending_coord is None:
            return
        skipped_id = self._next_order_id
        self._pending_coord = None
        self._set_overlay_visible(False)
        self._last_osrm_status = (
            f"Order O{skipped_id} skipped: no preferred time entered."
        )
        self._redraw_input()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._redraw_input()

    def _set_orders_and_drivers(
        self,
        orders: list[dict[str, Any]],
        drivers: list[dict[str, Any]],
    ) -> None:
        """Replace in-memory orders/drivers and reset viewport / pending overlays."""
        self.orders = list(orders)
        self.drivers = list(drivers)
        self._next_order_id = (
            max(int(o["order_id"]) for o in self.orders) + 1 if self.orders else 1
        )
        self._next_driver_id = (
            max(int(d["driver_id"]) for d in self.drivers) + 1 if self.drivers else 1
        )
        self._manual_view_xlim = None
        self._manual_view_ylim = None
        if self._pending_coord is not None:
            self._pending_coord = None
            self._set_overlay_visible(False)

    def _load_sample(self) -> None:
        """Load ``data/sample_orders*.csv`` + ``sample_drivers.csv`` (VRPTW vs plain by toggle)."""
        root = Path(__file__).resolve().parent
        data_dir = root / "data"
        try:
            drivers = load_drivers_csv(data_dir / "sample_drivers.csv")
            if self.use_vrptw:
                orders = load_orders_csv(data_dir / "sample_orders_vrptw.csv")
            else:
                orders = load_orders_csv(data_dir / "sample_orders.csv")
            self._last_osrm_status = (
                "Sample: data/"
                + ("sample_orders_vrptw.csv" if self.use_vrptw else "sample_orders.csv")
                + " + sample_drivers.csv"
            )
        except (OSError, ValueError) as e:
            drivers = synthetic_drivers(4, seed=8)
            orders = synthetic_orders(18, seed=8)
            if self.use_vrptw:
                orders = hydrate_synthetic_orders_for_vrptw_visual(
                    orders, window_half_min=_VRPTW_WINDOW_HALF_MIN
                )
            self._last_osrm_status = f"Sample CSV unavailable ({e}); using synthetic fallback."

        self._set_orders_and_drivers(orders, drivers)
        self._redraw_input()

    def _reset(self) -> None:
        self.orders.clear()
        self.drivers.clear()
        self._next_order_id = 1
        self._next_driver_id = 1
        self._last_pipeline = None
        self._manual_view_xlim = None
        self._manual_view_ylim = None
        self._drag_press_data = None
        self._dragging = False
        if self._output_fig is not None:
            plt.close(self._output_fig)
            self._output_fig = None
        if self._pending_coord is not None:
            self._pending_coord = None
            self._set_overlay_visible(False)
        self._redraw_input()

    def _on_mouse_press(self, event: Any) -> None:
        if self._pending_coord is not None:
            # An order is waiting for its preferred time — ignore map clicks
            # so the user must finish or cancel the inline dialog first.
            return
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        self._drag_press_data = (
            float(event.xdata),
            float(event.ydata),
            tuple(self.ax.get_xlim()),
            tuple(self.ax.get_ylim()),
        )
        self._dragging = False

    def _on_mouse_move(self, event: Any) -> None:
        if self._pending_coord is not None:
            return
        if self._drag_press_data is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        start_x, start_y, press_xlim, press_ylim = self._drag_press_data
        dx = float(event.xdata) - start_x
        dy = float(event.ydata) - start_y
        # Small threshold avoids turning ordinary clicks into drags.
        if not self._dragging and (abs(dx) > 0.0008 or abs(dy) > 0.0008):
            self._dragging = True
        if not self._dragging:
            return
        new_xlim = (press_xlim[0] - dx, press_xlim[1] - dx)
        new_ylim = (press_ylim[0] - dy, press_ylim[1] - dy)
        self._manual_view_xlim = new_xlim
        self._manual_view_ylim = new_ylim
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()

    def _on_mouse_release(self, event: Any) -> None:
        if self._pending_coord is not None:
            self._drag_press_data = None
            self._dragging = False
            return
        if self._drag_press_data is None or event.button != 1:
            return
        was_dragging = self._dragging
        self._drag_press_data = None
        self._dragging = False
        if was_dragging:
            return
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
            self._redraw_input()
            return

        if self.use_vrptw:
            # Don't add immediately. Park the click coordinate and surface the
            # inline overlay; the order is only created when the user confirms
            # a parseable preferred time.
            self._start_pending_order(lat, lon)
            return

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

        if self._manual_view_xlim is not None and self._manual_view_ylim is not None:
            self.ax.set_xlim(self._manual_view_xlim)
            self.ax.set_ylim(self._manual_view_ylim)
        elif not self.orders and not self.drivers:
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
            self.ax.scatter(
                [p[1] for p in drops],
                [p[0] for p in drops],
                c="#d62728",
                s=48,
                label="Orders",
                zorder=5,
                edgecolors="#fff",
                linewidths=0.35,
            )
            for order in self.orders:
                lat, lon = order["drop"]
                pref = order.get("preferred_minute")
                if self.use_vrptw and pref is not None:
                    label = f"O{order['order_id']} @ {_fmt_clock(float(pref))}"
                else:
                    label = f"O{order['order_id']}"
                self.ax.annotate(
                    label,
                    (lon, lat),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#3a0e0e",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white", "alpha": 0.85, "ec": "none"},
                    zorder=6,
                )
        if self.drivers:
            starts = [tuple(d["current_location"]) for d in self.drivers]
            self.ax.scatter(
                [p[1] for p in starts],
                [p[0] for p in starts],
                c="#2ca02c",
                marker="*",
                s=160,
                label="Drivers",
                zorder=5,
                edgecolors="#fff",
                linewidths=0.35,
            )
            for driver in self.drivers:
                lat, lon = driver["current_location"]
                self.ax.annotate(
                    f"D{driver['driver_id']}",
                    (lon, lat),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#0e3a0e",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white", "alpha": 0.85, "ec": "none"},
                    zorder=6,
                )

        if self.orders or self.drivers:
            self.ax.legend(loc="upper left", fontsize=8)

        vrptw_lines: list[str] = []
        if self.use_vrptw:
            vrptw_lines.append("VRPTW       : ON")
            vrptw_lines.append("Each new order")
            vrptw_lines.append("prompts for a")
            vrptw_lines.append("preferred time.")
            vrptw_lines.append("Window = pref ± 30m.")
        else:
            vrptw_lines.append("VRPTW       : OFF")
            vrptw_lines.append("Toggle to require")
            vrptw_lines.append("preferred times.")
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
                *vrptw_lines,
                "",
                self._last_osrm_status,
                "",
                "Tiles/grid: View button",
                "bottom-right.",
                "",
                "Click to add points.",
                "Drag to pan map view.",
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
        if self._pending_coord is not None:
            self._write_info(
                [
                    "Finish entering the",
                    "preferred time first,",
                    "or click Cancel.",
                ]
            )
            self.fig.canvas.draw_idle()
            return
        if not self.orders or not self.drivers:
            self._write_info(["Add at least one order", "and one driver first."])
            self.fig.canvas.draw_idle()
            return

        merged, _ = merge_same_location_orders(self.orders)
        labels, _ = cluster_deliveries_dbscan(merged, eps_km=_DBSCAN_EPS_KM, min_samples=2)
        clusters = stops_by_cluster(labels)
        assignment = assign_nearest_driver(clusters, merged, self.drivers)
        results, pipeline_info = run_optimized_routes(
            self.orders,
            self.drivers,
            dbscan_eps_km=_DBSCAN_EPS_KM,
            use_osrm=True,
            osrm_base_url=self.osrm_base_url,
            use_vrptw=self.use_vrptw,
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
        vrptw_info = pipeline_info.get("vrptw", {})
        if vrptw_info.get("enabled"):
            vrptw_label = (
                f"VRPTW: on ({vrptw_info.get('feasible_routes', 0)}/"
                f"{vrptw_info.get('total_routes', len(results))} feasible)"
            )
        else:
            vrptw_label = "VRPTW: off"
        summary = (
            f"Routes: {len(results)} | "
            f"Dijkstra: {savings['optimized_dijkstra_graph_km']:.2f} km | "
            f"A* fastest: {savings['optimized_astar_quick_km']:.2f} km | "
            f"OSRM: {savings['optimized_osrm_road_km']:.2f} km | "
            f"{status_label} | {vrptw_label} | counts: {status_counts}"
        )
        fig.suptitle(
            f"Last-Mile OSRM visual input -> process -> output  ·  {status_label}  ·  {vrptw_label}",
            fontsize=13,
            fontweight="bold",
        )
        fig.text(0.5, 0.02, summary, ha="center", family="monospace", fontsize=9)
        run_stamp = new_run_stamp_utc()
        path = output_images_dir(self.out_dir) / f"osrm_visual_process_{run_stamp}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.show(block=False)

        csv_path = output_csv_dir(self.out_dir) / f"pipeline_run_{run_stamp}.csv"
        write_pipeline_run_csv(
            csv_path,
            source="visual_osrm_app",
            orders=list(self.orders),
            drivers=list(self.drivers),
            merged=merged,
            labels=labels,
            assignment=assignment,
            results=results,
            savings=savings,
            pipeline_info=pipeline_info,
            dbscan_eps_km=_DBSCAN_EPS_KM,
            use_osrm=True,
        )
        json_path = output_json_dir(self.out_dir) / f"pipeline_run_{run_stamp}.json"
        report = build_pipeline_report_dict(
            merged=merged,
            labels=labels,
            assignment=assignment,
            results=results,
            savings=savings,
            pipeline_info=pipeline_info,
            run_stamp=run_stamp,
            run_source="visual_osrm_app",
            simulations=None,
        )
        save_pipeline_report_json(json_path, report)

        eta_lines: list[str] = ["ETA per order:"]
        on_time = 0
        late = 0
        for route in results:
            if not route.metas_ordered:
                continue
            anchor_min = min(float(m["time_window"][0]) for m in route.metas_ordered)
            eta_lines.append(f"D{route.driver_id}/C{route.cluster_id}")
            for meta, eta_rel in zip(route.metas_ordered, route.eta_arrival_min, strict=False):
                order_id = int(meta.get("order_id", -1))
                eta_abs = anchor_min + float(eta_rel)
                pref = meta.get("preferred_minute")
                if self.use_vrptw and pref is not None:
                    delta = eta_abs - float(pref)
                    tag = "OK " if abs(delta) <= _VRPTW_WINDOW_HALF_MIN else "LATE"
                    if tag == "OK ":
                        on_time += 1
                    else:
                        late += 1
                    eta_lines.append(
                        f"  O{order_id}: ETA {_fmt_clock(eta_abs)} pref {_fmt_clock(float(pref))} ({delta:+.0f}m) {tag.strip()}"
                    )
                else:
                    eta_lines.append(f"  O{order_id}: {_fmt_clock(eta_abs)} (+{eta_rel:.1f}m)")
        if self.use_vrptw:
            vrptw_summary_lines = [
                "VRPTW summary",
                f"  on-time stops: {on_time}",
                f"  late stops   : {late}",
                "",
            ]
        else:
            vrptw_summary_lines = []
        self._write_info(
            [
                "Pipeline complete.",
                "",
                status_label,
                "",
                f"Saved: {path.name}",
                f"CSV: output/csv/{csv_path.name}",
                f"JSON: output/json/{json_path.name}",
                "",
                summary,
                "",
                *vrptw_summary_lines,
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
            ax.scatter(
                [p[1] for p in drops],
                [p[0] for p in drops],
                c="#d62728",
                s=42,
                label="Orders",
                zorder=5,
                edgecolors="#fff",
                linewidths=0.3,
            )
            for order in self.orders:
                lat, lon = order["drop"]
                pref = order.get("preferred_minute")
                if self.use_vrptw and pref is not None:
                    text = f"O{order['order_id']} @ {_fmt_clock(float(pref))}"
                else:
                    text = f"O{order['order_id']}"
                ax.annotate(
                    text,
                    (lon, lat),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#3a0e0e",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white", "alpha": 0.85, "ec": "none"},
                    zorder=7,
                )
        if self.drivers:
            starts = [tuple(d["current_location"]) for d in self.drivers]
            ax.scatter(
                [p[1] for p in starts],
                [p[0] for p in starts],
                c="#2ca02c",
                marker="*",
                s=160,
                label="Drivers",
                zorder=5,
                edgecolors="#fff",
                linewidths=0.3,
            )
            for driver in self.drivers:
                lat, lon = driver["current_location"]
                ax.annotate(
                    f"D{driver['driver_id']}",
                    (lon, lat),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#0e3a0e",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white", "alpha": 0.85, "ec": "none"},
                    zorder=7,
                )
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
        # Per-stop cluster id label, e.g. "c0" / "c1" / "c-1" (noise) so the
        # reader can tell which order belongs to which cluster at a glance.
        for stop, cid_val in zip(merged, labels.tolist(), strict=False):
            slat, slon = stop["drop"]
            ax.annotate(
                f"c{int(cid_val)}",
                (slon, slat),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold",
                color="#222",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "alpha": 0.85, "ec": "none"},
                zorder=7,
            )
        clusters = stops_by_cluster(labels)
        dmap = {int(d["driver_id"]): d for d in self.drivers}
        # Plot drivers and tag each with D{id} so the assignment lines have
        # named anchors on both ends.
        if self.drivers:
            ax.scatter(
                [d["current_location"][1] for d in self.drivers],
                [d["current_location"][0] for d in self.drivers],
                c="#2ca02c",
                marker="*",
                s=130,
                edgecolors="#fff",
                linewidths=0.3,
                zorder=6,
                label="Drivers",
            )
            for driver in self.drivers:
                dlat, dlon = driver["current_location"]
                ax.annotate(
                    f"D{driver['driver_id']}",
                    (dlon, dlat),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#0e3a0e",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white", "alpha": 0.85, "ec": "none"},
                    zorder=8,
                )
        # Centroid -> assigned driver dashed link plus a "c{id}\u2192D{id}" tag
        # at the centroid so the cluster -> driver mapping is unambiguous.
        for cid, idxs in clusters.items():
            pts = np.array([[merged[i]["drop"][1], merged[i]["drop"][0]] for i in idxs])
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            did = assignment.get(cid)
            if did is None:
                continue
            drv = dmap[did]
            dx, dy = drv["current_location"][1], drv["current_location"][0]
            ax.plot([cx, dx], [cy, dy], "k--", alpha=0.35, linewidth=1, zorder=4)
            ax.annotate(
                f"c{int(cid)}\u2192D{int(did)}",
                (cx, cy),
                xytext=(0, -10),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                color="#444",
                bbox={"boxstyle": "round,pad=0.18", "fc": "#f7f1d7", "alpha": 0.9, "ec": "none"},
                zorder=8,
            )
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
                    pref = meta.get("preferred_minute")
                    if self.use_vrptw and pref is not None:
                        delta = eta_abs - float(pref)
                        bbox_color = "#d8efd8" if abs(delta) <= _VRPTW_WINDOW_HALF_MIN else "#f4d3d3"
                        text = (
                            f"O{order_id} ETA {_fmt_clock(eta_abs)}\n"
                            f"pref {_fmt_clock(float(pref))} ({delta:+.0f}m)"
                        )
                    else:
                        bbox_color = "white"
                        text = f"O{order_id} ETA {_fmt_clock(eta_abs)}"
                    ax.annotate(
                        text,
                        (lon, lat),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=7.5,
                        bbox={"boxstyle": "round,pad=0.2", "fc": bbox_color, "alpha": 0.85, "ec": "none"},
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
    initial_use_vrptw: bool = False,
    initial_orders: list[dict[str, Any]] | None = None,
    initial_drivers: list[dict[str, Any]] | None = None,
) -> None:
    app = VisualOsrmApp(
        osrm_base_url=osrm_base_url,
        out_dir=out_dir,
        viz_mode=viz_mode,
        initial_use_vrptw=initial_use_vrptw,
        initial_orders=initial_orders,
        initial_drivers=initial_drivers,
    )
    app.show()


if __name__ == "__main__":
    run_visual_osrm_app()
