"""
Small OSRM HTTP client used by Last-Mile's routing pipeline.

OSRM expects coordinates as lon,lat, while the rest of this project stores
points as lat,lon tuples. This module keeps that conversion in one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


LatLon = tuple[float, float]


@dataclass(frozen=True)
class OsrmRoute:
    road_km: float | None
    duration_min: float | None
    geometry: list[LatLon]
    status: str
    code: str | None = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.road_km is not None


class OsrmClient:
    def __init__(self, base_url: str = "http://localhost:5000", timeout_seconds: float = 4.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def route(self, points: list[LatLon]) -> OsrmRoute:
        """Route through the supplied lat,lon waypoints in order."""
        if len(points) < 2:
            return OsrmRoute(0.0, 0.0, points[:], "ok", code="Ok")

        coords = ";".join(f"{lon:.9f},{lat:.9f}" for lat, lon in points)
        query = urlencode(
            {
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
                "alternatives": "false",
            }
        )
        url = f"{self.base_url}/route/v1/driving/{coords}?{query}"

        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return OsrmRoute(None, None, points[:], "http_error", message=f"HTTP {exc.code}")
        except (TimeoutError, URLError, OSError, json.JSONDecodeError) as exc:
            return OsrmRoute(None, None, points[:], "unavailable", message=str(exc))

        code = str(payload.get("code", ""))
        if code != "Ok":
            return OsrmRoute(None, None, points[:], "osrm_error", code=code, message=_message(payload))

        routes = payload.get("routes") or []
        if not routes:
            return OsrmRoute(None, None, points[:], "osrm_error", code=code, message="OSRM returned no routes")

        route = routes[0]
        raw_coords = ((route.get("geometry") or {}).get("coordinates") or [])
        geometry = [(float(lat), float(lon)) for lon, lat in raw_coords] or points[:]
        road_km = float(route.get("distance", 0.0)) / 1000.0
        duration_min = float(route.get("duration", 0.0)) / 60.0
        return OsrmRoute(road_km, duration_min, geometry, "ok", code=code)


def _message(payload: dict[str, Any]) -> str | None:
    msg = payload.get("message")
    return str(msg) if msg is not None else None
