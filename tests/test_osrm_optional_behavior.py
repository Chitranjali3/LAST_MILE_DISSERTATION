"""
Validation matrix for the optional OSRM enrichment layer.

The contract these tests defend (see ``OSRM_OPTIONAL_UPGRADE_CHECKLIST.md``):

1. Core flow runs identically without OSRM.
2. ``--use-osrm`` with a healthy server delivers full enrichment.
3. ``--use-osrm`` with the server down preflights to ``core_only`` immediately
   and never blocks per-route on transport timeouts.
4. ``--use-osrm`` with mid-run failure produces partial enrichment + clean
   continuation (the circuit breaker trips once and stops calling out).
5. The ``visual_osrm_app`` helper still produces a sensible status label in all
   of the above conditions.

Tests stub the OSRM HTTP client so they run offline and finish in <1s.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.osrm_client import OsrmHealth, OsrmRoute
from core.routing import RouteResult, run_optimized_routes, select_route_polyline
from utils import ODISHA_REGION_CENTER, synthetic_drivers, synthetic_orders
from visual_osrm_app import _osrm_status_label


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def small_problem() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Small synthetic dataset that exercises >1 cluster (so per-route OSRM
    counts are meaningful) and stays fast enough to run repeatedly."""
    return synthetic_orders(10, seed=7), synthetic_drivers(2, seed=7)


def _fake_route_ok(points: list[tuple[float, float]]) -> OsrmRoute:
    """Plausible OSRM success: small road km, slow city duration, 3-pt geometry."""
    if len(points) < 2:
        return OsrmRoute(0.0, 0.0, points[:], "ok", code="Ok")
    start = points[0]
    end = points[-1]
    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    return OsrmRoute(2.5, 7.5, [start, midpoint, end], "ok", code="Ok")


# ---------------------------------------------------------------------------
# Scenario 1: --use-osrm off (baseline core-only)
# ---------------------------------------------------------------------------


def test_use_osrm_off_runs_core_only(small_problem):
    orders, drivers = small_problem
    results, info = run_optimized_routes(orders, drivers, dbscan_eps_km=1.4, use_osrm=False)

    assert info["osrm"]["requested"] is False
    assert info["osrm"]["connected"] is False
    assert info["osrm"]["mode"] == "core_only"
    assert info["osrm"]["reason"] == "not_requested"
    assert info["osrm"]["enriched_routes"] == 0
    assert info["osrm"]["total_routes"] == len(results)

    assert len(results) > 0
    for r in results:
        assert r.osrm_status == "not_requested"
        assert r.osrm_road_km is None
        assert r.effective_distance_source in {"astar", "ga_proxy"}
        assert r.effective_time_source == "proxy"
        assert r.effective_distance_km > 0.0
        assert r.effective_duration_min is not None and r.effective_duration_min > 0.0


# ---------------------------------------------------------------------------
# Scenario 2: --use-osrm on, server up
# ---------------------------------------------------------------------------


def test_use_osrm_on_with_server_up_full_enrichment(small_problem, monkeypatch):
    orders, drivers = small_problem

    monkeypatch.setattr(
        "core.routing.OsrmClient.health_check",
        lambda self, **_: OsrmHealth(True, "connected", "Ok"),
    )
    monkeypatch.setattr("core.routing.OsrmClient.route", lambda self, points: _fake_route_ok(points))

    results, info = run_optimized_routes(orders, drivers, dbscan_eps_km=1.4, use_osrm=True)

    assert info["osrm"]["requested"] is True
    assert info["osrm"]["connected"] is True
    assert info["osrm"]["mode"] == "core_plus_osrm"
    assert info["osrm"]["reason"] == "connected"
    assert info["osrm"]["enriched_routes"] == len(results)
    assert info["osrm"]["status_counts"] == {"ok": len(results)}

    for r in results:
        assert r.osrm_status == "ok"
        assert r.osrm_road_km == pytest.approx(2.5)
        assert r.osrm_duration_min == pytest.approx(7.5)
        assert r.effective_distance_source == "osrm"
        assert r.effective_time_source == "osrm"
        assert r.effective_distance_km == pytest.approx(2.5)
        assert r.effective_duration_min == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Scenario 3: --use-osrm on, server down at start
# ---------------------------------------------------------------------------


def test_use_osrm_on_with_server_down_at_start(small_problem, monkeypatch):
    orders, drivers = small_problem

    monkeypatch.setattr(
        "core.routing.OsrmClient.health_check",
        lambda self, **_: OsrmHealth(False, "unavailable", "Connection refused"),
    )

    # If the orchestrator ever called .route() despite a failed preflight, that
    # would be a regression — fail loudly here.
    def _explode(self, points):  # pragma: no cover - defensive
        raise AssertionError("OsrmClient.route called after failed preflight")

    monkeypatch.setattr("core.routing.OsrmClient.route", _explode)

    results, info = run_optimized_routes(orders, drivers, dbscan_eps_km=1.4, use_osrm=True)

    assert info["osrm"]["requested"] is True
    assert info["osrm"]["connected"] is False
    assert info["osrm"]["mode"] == "core_only"
    assert info["osrm"]["reason"] == "unavailable"
    assert info["osrm"]["enriched_routes"] == 0
    assert info["osrm"]["total_routes"] == len(results)

    assert len(results) > 0
    for r in results:
        assert r.osrm_status.startswith("unavailable")
        assert r.osrm_road_km is None
        assert r.effective_distance_source in {"astar", "ga_proxy"}
        assert r.effective_time_source == "proxy"


# ---------------------------------------------------------------------------
# Scenario 4: --use-osrm on, server fails mid-run (circuit breaker trips)
# ---------------------------------------------------------------------------


def test_use_osrm_mid_run_failure_partial_enrichment(small_problem, monkeypatch):
    orders, drivers = small_problem

    monkeypatch.setattr(
        "core.routing.OsrmClient.health_check",
        lambda self, **_: OsrmHealth(True, "connected", "Ok"),
    )

    state = {"calls": 0}

    def _flaky_route(self, points):
        state["calls"] += 1
        if state["calls"] == 1:
            return _fake_route_ok(points)
        # Second call simulates the OSRM container crashing.
        return OsrmRoute(None, None, points[:], "unavailable", message="Connection reset")

    monkeypatch.setattr("core.routing.OsrmClient.route", _flaky_route)

    results, info = run_optimized_routes(orders, drivers, dbscan_eps_km=1.4, use_osrm=True)

    assert len(results) >= 2, "need >= 2 routes to exercise mid-run failure"
    # Exactly one HTTP call goes out: success on route #1, then breaker trips.
    assert state["calls"] == 2

    assert info["osrm"]["requested"] is True
    assert info["osrm"]["connected"] is True
    assert info["osrm"]["mode"] == "core_plus_osrm"
    assert info["osrm"]["enriched_routes"] == 1
    assert info["osrm"]["total_routes"] == len(results)

    ok_routes = [r for r in results if r.osrm_status == "ok"]
    fallback_routes = [r for r in results if r.osrm_status != "ok"]
    assert len(ok_routes) == 1
    assert len(fallback_routes) == len(results) - 1
    for r in fallback_routes:
        assert r.osrm_status.startswith("unavailable")
        assert r.effective_distance_source in {"astar", "ga_proxy"}


# ---------------------------------------------------------------------------
# Cross-cutting assertions
# ---------------------------------------------------------------------------


def test_pipeline_block_always_carries_osrm_status(small_problem):
    orders, drivers = small_problem
    _, info = run_optimized_routes(orders, drivers, dbscan_eps_km=1.4, use_osrm=False)
    osrm = info["osrm"]
    for key in ("requested", "connected", "mode", "reason", "status_counts", "enriched_routes", "total_routes"):
        assert key in osrm, f"pipeline.osrm missing '{key}'"


def test_route_count_unaffected_by_osrm_state(small_problem, monkeypatch):
    orders, drivers = small_problem
    baseline, _ = run_optimized_routes(orders, drivers, dbscan_eps_km=1.4, use_osrm=False)

    monkeypatch.setattr(
        "core.routing.OsrmClient.health_check",
        lambda self, **_: OsrmHealth(False, "unavailable", "down"),
    )
    fallback, _ = run_optimized_routes(orders, drivers, dbscan_eps_km=1.4, use_osrm=True)

    monkeypatch.setattr(
        "core.routing.OsrmClient.health_check",
        lambda self, **_: OsrmHealth(True, "connected", "Ok"),
    )
    monkeypatch.setattr("core.routing.OsrmClient.route", lambda self, points: _fake_route_ok(points))
    enriched, _ = run_optimized_routes(orders, drivers, dbscan_eps_km=1.4, use_osrm=True)

    assert len(baseline) == len(fallback) == len(enriched)


def test_no_unhandled_exception_when_osrm_explodes(small_problem, monkeypatch):
    """The orchestrator should never let an OSRM transport exception bubble up."""
    orders, drivers = small_problem

    # The client itself catches transport errors and returns OsrmHealth/OsrmRoute,
    # so we simulate that contract being honored.
    def _boom_health(self, **_):
        return OsrmHealth(False, "unavailable", "boom")

    def _boom_route(self, _points):  # pragma: no cover - guarded by preflight
        raise RuntimeError("transport bug")

    monkeypatch.setattr("core.routing.OsrmClient.health_check", _boom_health)
    monkeypatch.setattr("core.routing.OsrmClient.route", _boom_route)

    # Should complete without raising, because the failed preflight prevents
    # any .route() call from being issued.
    results, info = run_optimized_routes(orders, drivers, dbscan_eps_km=1.4, use_osrm=True)
    assert len(results) > 0
    assert info["osrm"]["mode"] == "core_only"


# ---------------------------------------------------------------------------
# Visual surfacing helpers
# ---------------------------------------------------------------------------


def test_osrm_status_label_covers_all_modes():
    assert _osrm_status_label({"requested": False}) == "OSRM: not requested"
    assert _osrm_status_label(
        {"requested": True, "connected": False, "reason": "unavailable"}
    ) == "OSRM: not connected (unavailable)"
    assert _osrm_status_label(
        {"requested": True, "connected": True, "enriched_routes": 3, "total_routes": 3}
    ) == "OSRM: connected (3/3 enriched)"
    assert _osrm_status_label(
        {"requested": True, "connected": True, "enriched_routes": 1, "total_routes": 3}
    ) == "OSRM: partial enrichment 1/3"


def test_select_route_polyline_falls_back_when_osrm_geometry_degenerate():
    driver_start = ODISHA_REGION_CENTER
    lat0, lon0 = driver_start
    lat0, lon0 = driver_start
    drops = [(lat0 + 0.01, lon0 + 0.01), (lat0 + 0.02, lon0 + 0.02)]
    fallback = [driver_start] + drops

    no_geometry = RouteResult(
        cluster_id=0, driver_id=1, stop_order_local=[0, 1], drop_coords_ordered=drops,
        metas_ordered=[], ga_tour_km=1.0, astar_leg_km=1.0, dijkstra_star_equal=True,
        vrptw_ok=True, vrptw_detail={}, osrm_geometry=None,
    )
    assert select_route_polyline(no_geometry, driver_start) == fallback

    degenerate_geometry = RouteResult(
        cluster_id=0, driver_id=1, stop_order_local=[0, 1], drop_coords_ordered=drops,
        metas_ordered=[], ga_tour_km=1.0, astar_leg_km=1.0, dijkstra_star_equal=True,
        vrptw_ok=True, vrptw_detail={},
        osrm_geometry=[driver_start, driver_start],  # zero-length artifact
    )
    assert select_route_polyline(degenerate_geometry, driver_start) == fallback

    real_geometry = [
        driver_start,
        (lat0 + 0.01, lon0 + 0.01),
        (lat0 + 0.02, lon0 + 0.02),
    ]
    valid = RouteResult(
        cluster_id=0, driver_id=1, stop_order_local=[0, 1], drop_coords_ordered=drops,
        metas_ordered=[], ga_tour_km=1.0, astar_leg_km=1.0, dijkstra_star_equal=True,
        vrptw_ok=True, vrptw_detail={}, osrm_geometry=real_geometry,
    )
    assert select_route_polyline(valid, driver_start) == real_geometry
