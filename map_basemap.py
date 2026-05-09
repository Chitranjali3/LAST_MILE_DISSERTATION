"""
Optional OpenStreetMap tile basemap under Matplotlib lon/lat axes (EPSG:4326).

Uses Contextily; fails softly when the package is missing or tiles cannot be fetched.
Set CHITRA_NO_BASEMAP=1 to skip network tile requests (tests / offline).

Visualization modes:
  - ``map`` — OSM tiles under lon/lat data (when available).
  - ``graph`` — classic grid-only plot (no tile fetch).

Optional env ``CHITRA_VIZ_MODE=graph|map`` used as default when CLI does not pass ``--viz-mode``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Literal

VizMode = Literal["graph", "map"]


def basemap_disabled() -> bool:
    return os.environ.get("CHITRA_NO_BASEMAP", "").strip().lower() in ("1", "true", "yes")


def pad_lonlat_extent(
    lons: Iterable[float],
    lats: Iterable[float],
    *,
    pad_deg: float = 0.004,
    default_lon: tuple[float, float] = (77.50, 77.68),
    default_lat: tuple[float, float] = (12.89, 13.05),
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ((min_lon, max_lon), (min_lat, max_lat)) with symmetric padding."""
    lon_list = list(lons)
    lat_list = list(lats)
    if not lon_list or not lat_list:
        return default_lon, default_lat
    min_lon, max_lon = min(lon_list), max(lon_list)
    min_lat, max_lat = min(lat_list), max(lat_list)
    # Avoid zero-area extent when all points coincide.
    if max_lon - min_lon < 1e-9:
        min_lon -= pad_deg
        max_lon += pad_deg
    if max_lat - min_lat < 1e-9:
        min_lat -= pad_deg
        max_lat += pad_deg
    return (
        (min_lon - pad_deg, max_lon + pad_deg),
        (min_lat - pad_deg, max_lat + pad_deg),
    )


def normalize_viz_mode(mode: str | None, *, default: VizMode = "map") -> VizMode:
    """Return ``graph`` or ``map``; invalid values fall back to *default*."""
    if mode is None:
        return default
    m = str(mode).strip().lower()
    if m in ("graph", "plain", "grid"):
        return "graph"
    if m in ("map", "osm", "tiles", "basemap"):
        return "map"
    return default


def try_osm_basemap(
    ax: Any,
    *,
    viz_mode: str = "map",
    alpha: float = 0.78,
    zorder: int = 0,
    disable_grid_on_success: bool = True,
) -> bool:
    """Like ``add_osm_basemap`` but honors *viz_mode* (no tiles in ``graph`` mode)."""
    if normalize_viz_mode(viz_mode) != "map":
        return False
    return add_osm_basemap(
        ax,
        alpha=alpha,
        zorder=zorder,
        disable_grid_on_success=disable_grid_on_success,
    )


def add_osm_basemap(
    ax: Any,
    *,
    alpha: float = 0.78,
    zorder: int = 0,
    disable_grid_on_success: bool = True,
) -> bool:
    """
    Draw OSM raster tiles for the current axis limits (lon on x, lat on y).

    Call after ``set_xlim`` / ``set_ylim``. Vector layers should use zorder > zorder.
    """
    if basemap_disabled():
        return False
    try:
        import contextily as ctx
    except ImportError:
        return False
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=ctx.providers.OpenStreetMap.Mapnik,
            zorder=zorder,
            alpha=alpha,
        )
        if disable_grid_on_success:
            ax.grid(False)
        return True
    except Exception:
        return False
