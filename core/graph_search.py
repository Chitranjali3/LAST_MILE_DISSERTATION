"""
Graph-based shortest paths for synthetic road networks.

We construct an undirected graph whose nodes are landmark coordinates (e.g.
delivery drops and the driver origin). Edge weights approximate road distance
with the Haversine metric on the **chord**; for research demos, this proxies
"shortest path" routing between geolocated nodes.

**Dijkstra (1959)** solves single-source shortest paths on graphs with
 nonnegative edge weights.

- Time complexity: O((V + E) log V) with a binary heap (NetworkX default).

**A* (Hart, Nilsson & Raphael, 1968)** is goal-directed: f = g + h with
 admissible heuristic h. For nonnegative costs, A* never expands more nodes
than necessary when h is consistent; with our Haversine-to-goal heuristic on
a geographic graph, the first pop of the goal is optimal when h is admissible.

- Time complexity: O(E) in the best case for many road networks; same worst
  as Dijkstra in adversarial graphs (O((V+E) log V) with a good heap).

**Benchmark note:** Dijkstra (km weights) and A* (minutes weights) can diverge
once edge travel speeds vary, even on the same graph topology.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable

import networkx as nx

from utils import EARTH_RADIUS_KM, haversine_km


def build_geographic_graph(
    nodes: list[tuple[float, float]],
    *,
    knn: int | None = None,
    with_waypoints: bool = True,
    waypoints_per_edge: int = 2,
    waypoint_offset_frac: float = 0.22,
) -> nx.Graph:
    """
    Create a weighted graph over point nodes.

    If `knn` is set, each node connects to its `knn` nearest neighbors (plus mutual);
    otherwise the graph is complete (suitable only for small |V|).

    When ``with_waypoints`` is True, synthetic waypoints are injected
    perpendicular to each direct edge so alternative paths exist between every
    pair of original nodes. This lets fastest-time routing diverge from
    shortest-distance routing on otherwise trivial 2-point graphs.

    Node keys are tuple (lat, lon) rounded to 9 decimals for hashing stability.
    """
    n = len(nodes)
    if n == 0:
        raise ValueError("nodes must be non-empty")
    keyed = [_key(p) for p in nodes]
    G = nx.Graph()
    for k in keyed:
        G.add_node(k)

    def _add_weighted_edge(
        p: tuple[float, float],
        q: tuple[float, float],
        *,
        edge_kind: str = "direct",
    ) -> None:
        kp, kq = _key(p), _key(q)
        if kp == kq:
            return
        if kp not in G:
            G.add_node(kp)
        if kq not in G:
            G.add_node(kq)
        w = haversine_km(p, q)
        if edge_kind == "via_waypoint":
            # Alternate "main-road" segment: faster cruise speed and
            # consistently lower traffic delay than the surface-street direct
            # edge. Distance is still real haversine km.
            speed_kmh = 42.0
            cruise_minutes = (w / speed_kmh) * 60.0 if speed_kmh > 0 else float("inf")
            minutes = cruise_minutes + _waypoint_edge_delay_min(p, q, w)
        else:
            speed_kmh = _edge_speed_kmh(p, q, w)
            cruise_minutes = (w / speed_kmh) * 60.0 if speed_kmh > 0 else float("inf")
            minutes = cruise_minutes + _edge_delay_min(p, q, w)
        if G.has_edge(kp, kq):
            existing = G[kp][kq]
            if w < float(existing.get("km", float("inf"))):
                existing.update(weight=w, km=w, minutes=minutes, speed_kmh=speed_kmh)
            elif minutes < float(existing.get("minutes", float("inf"))):
                existing["minutes"] = minutes
            return
        G.add_edge(kp, kq, weight=w, km=w, minutes=minutes, speed_kmh=speed_kmh)

    direct_pairs: list[tuple[int, int]] = []
    if knn is not None and knn > 0 and n > 1:
        seen: set[tuple[int, int]] = set()
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                dists.append((haversine_km(nodes[i], nodes[j]), j))
            dists.sort(key=lambda x: x[0])
            for _, j in dists[: min(knn, len(dists))]:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in seen:
                    seen.add((a, b))
                    _add_weighted_edge(nodes[a], nodes[b])
                    direct_pairs.append((a, b))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                _add_weighted_edge(nodes[i], nodes[j])
                direct_pairs.append((i, j))

    if with_waypoints and waypoints_per_edge > 0:
        for i, j in direct_pairs:
            a, b = nodes[i], nodes[j]
            d_lat = b[0] - a[0]
            d_lon = b[1] - a[1]
            length = math.hypot(d_lat, d_lon)
            if length < 1e-9:
                continue
            # Perpendicular unit vector in degrees (small-angle planar approx).
            p_lat = -d_lon / length
            p_lon = d_lat / length
            mid_lat = (a[0] + b[0]) / 2.0
            mid_lon = (a[1] + b[1]) / 2.0
            for k in range(waypoints_per_edge):
                sign = 1.0 if (k % 2 == 0) else -1.0
                step = (k // 2) + 1
                off = sign * step * waypoint_offset_frac * length
                wp = (mid_lat + p_lat * off, mid_lon + p_lon * off)
                _add_weighted_edge(a, wp, edge_kind="via_waypoint")
                _add_weighted_edge(wp, b, edge_kind="via_waypoint")

    return G


def _edge_speed_kmh(a: tuple[float, float], b: tuple[float, float], km: float) -> float:
    """Synthetic road-speed profile per edge for time-weighted routing demos."""
    mid_lat = (a[0] + b[0]) / 2.0
    mid_lon = (a[1] + b[1]) / 2.0
    # Deterministic variability: spatial signal + mild distance effect.
    signal = math.sin(mid_lat * 17.0) + math.cos(mid_lon * 19.0)
    base = 30.0 + 8.0 * signal
    length_bias = min(10.0, km * 2.4)
    speed = base + length_bias
    return float(min(55.0, max(18.0, speed)))


def _waypoint_edge_delay_min(a: tuple[float, float], b: tuple[float, float], km: float) -> float:
    """Light, low-variance traffic delay applied to alternate ('main-road') legs."""
    a_lat, a_lon = round(a[0], 4), round(a[1], 4)
    b_lat, b_lon = round(b[0], 4), round(b[1], 4)
    sig = a_lat * 8237.0 + a_lon * 4133.0 + b_lat * 6781.0 + b_lon * 2393.0
    noise = (math.sin(sig) + 1.0) * 0.5  # [0, 1]
    return float(0.20 + 0.9 * noise + min(0.5, km * 0.04))


def _edge_delay_min(a: tuple[float, float], b: tuple[float, float], km: float) -> float:
    """Deterministic congestion delay to separate fastest-time from shortest-km."""
    mid_lat = (a[0] + b[0]) / 2.0
    mid_lon = (a[1] + b[1]) / 2.0
    # Spatial hotspot near Bhubaneswar-like lat/lon; smooth decay with distance.
    dlat = mid_lat - 20.30
    dlon = mid_lon - 85.84
    hotspot = math.exp(-((dlat * dlat) / 0.00045 + (dlon * dlon) / 0.00045))
    # Orientation penalty: east-west arterials get heavier peak traffic.
    bearing_mix = abs(a[0] - b[0]) / max(1e-9, abs(a[1] - b[1]) + abs(a[0] - b[0]))
    ew_penalty = 1.0 - bearing_mix
    # Edge-specific deterministic delay: acts like traffic-light / junction drag.
    # This term is intentionally not tied to distance so time-optimal paths can
    # differ from distance-optimal paths.
    a_lat, a_lon = round(a[0], 4), round(a[1], 4)
    b_lat, b_lon = round(b[0], 4), round(b[1], 4)
    lo = (min(a_lat, b_lat), min(a_lon, b_lon))
    hi = (max(a_lat, b_lat), max(a_lon, b_lon))
    sig = (
        lo[0] * 9283.0
        + lo[1] * 6151.0
        + hi[0] * 3571.0
        + hi[1] * 1217.0
    )
    edge_noise = (math.sin(sig) + 1.0) * 0.5  # [0, 1]
    edge_penalty = 6.0 * edge_noise
    # Fixed-delay style term introduces non-distance effects (signals/turn delays).
    delay = 0.10 + edge_penalty + (5.5 * hotspot * ew_penalty) + min(0.7, km * 0.06)
    return float(max(0.0, delay))


def _key(p: tuple[float, float]) -> tuple[float, float]:
    return (round(p[0], 9), round(p[1], 9))


def dijkstra_shortest_path(
    G: nx.Graph,
    source: tuple[float, float],
    target: tuple[float, float],
) -> tuple[list[tuple[float, float]], float]:
    """Return node coordinate path and total km using Dijkstra (NetworkX)."""
    s, t = _key(source), _key(target)
    if s not in G or t not in G:
        raise nx.NodeNotFound("source or target not in graph")
    length, path_keys = nx.single_source_dijkstra(G, s, t, weight="km")
    path_pts = [(k[0], k[1]) for k in path_keys]
    return path_pts, float(length)


def _astar_manual(
    G: nx.Graph,
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    weight_attr: str = "km",
    heuristic: Callable[[tuple[float, float], tuple[float, float]], float] | None = None,
) -> tuple[list[tuple[float, float]], float]:
    """
    Classic A* on an undirected weighted graph (km on edges).

    Supports customizable edge weight and heuristic.
    """
    s, t = _key(source), _key(target)
    open_heap: list[tuple[float, tuple[float, float]]] = []
    heapq.heappush(open_heap, (0.0, s))
    g_score: dict[tuple[float, float], float] = {s: 0.0}
    came_from: dict[tuple[float, float], tuple[float, float] | None] = {s: None}

    def h(u: tuple[float, float]) -> float:
        if heuristic is None:
            pt_u = (u[0], u[1])
            pt_t = (t[0], t[1])
            return haversine_km(pt_u, pt_t)
        return float(heuristic(u, t))

    while open_heap:
        _f, u = heapq.heappop(open_heap)
        if u == t:
            # Reconstruct path
            rev: list[tuple[float, float]] = []
            cur: tuple[float, float] | None = u
            while cur is not None:
                rev.append((cur[0], cur[1]))
                cur = came_from.get(cur)  # type: ignore[assignment]
            rev.reverse()
            return rev, g_score[u]

        for v in G.neighbors(u):
            edge_cost = float(G[u][v].get(weight_attr, G[u][v].get("weight", 0.0)))
            tentative = g_score[u] + edge_cost
            if tentative < g_score.get(v, float("inf")):
                g_score[v] = tentative
                came_from[v] = u
                f = tentative + h(v)
                heapq.heappush(open_heap, (f, v))

    raise nx.NetworkXNoPath("A*: no path between source and target")


def astar_shortest_path(
    G: nx.Graph,
    source: tuple[float, float],
    target: tuple[float, float],
) -> tuple[list[tuple[float, float]], float]:
    """A* shortest path with Haversine heuristic; returns path vertices and km."""
    return _astar_manual(G, source, target)


def astar_fastest_path(
    G: nx.Graph,
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    max_speed_kmh: float = 55.0,
) -> tuple[list[tuple[float, float]], float, float]:
    """A* fastest path on ``minutes`` edge weights.

    Returns ``(path, km_along_path, total_minutes)``.
    """
    bounded_max_speed = max(1.0, float(max_speed_kmh))

    def _minutes_lb(u: tuple[float, float], t: tuple[float, float]) -> float:
        straight_km = haversine_km((u[0], u[1]), (t[0], t[1]))
        return (straight_km / bounded_max_speed) * 60.0

    path, minutes = _astar_manual(
        G,
        source,
        target,
        weight_attr="minutes",
        heuristic=_minutes_lb,
    )
    km_total = 0.0
    for a, b in zip(path, path[1:], strict=False):
        km_total += float(G[a][b].get("km", G[a][b].get("weight", 0.0)))
    return path, km_total, float(minutes)


def benchmark_dijkstra_vs_astar(
    G: nx.Graph,
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Return (km_dijkstra, km_on_astar_fastest_path) per pair."""
    out: list[tuple[float, float]] = []
    for a, b in pairs:
        _, dk = dijkstra_shortest_path(G, a, b)
        _, ak, _amin = astar_fastest_path(G, a, b)
        out.append((dk, ak))
    return out
