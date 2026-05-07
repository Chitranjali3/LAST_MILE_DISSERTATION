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

**Benchmark note:** On the same deterministic nonnegative graph, optimal
distances match between Dijkstra and A*.
"""

from __future__ import annotations

import heapq

import networkx as nx

from utils import EARTH_RADIUS_KM, haversine_km


def build_geographic_graph(
    nodes: list[tuple[float, float]],
    *,
    knn: int | None = None,
) -> nx.Graph:
    """
    Create a weighted graph over point nodes.

    If `knn` is set, each node connects to its `knn` nearest neighbors (plus mutual);
    otherwise the graph is complete (suitable only for small |V|).

    Node keys are tuple (lat, lon) rounded to 9 decimals for hashing stability.
    """
    n = len(nodes)
    if n == 0:
        raise ValueError("nodes must be non-empty")
    keyed = [_key(p) for p in nodes]
    G = nx.Graph()
    for k in keyed:
        G.add_node(k)

    def add_edge(i: int, j: int) -> None:
        if i == j:
            return
        a, b = nodes[i], nodes[j]
        w = haversine_km(a, b)
        G.add_edge(keyed[i], keyed[j], weight=w, km=w)

    if knn is not None and knn > 0 and n > 1:
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                dists.append((haversine_km(nodes[i], nodes[j]), j))
            dists.sort(key=lambda x: x[0])
            for _, j in dists[: min(knn, len(dists))]:
                add_edge(i, j)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                add_edge(i, j)

    return G


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
) -> tuple[list[tuple[float, float]], float]:
    """
    Classic A* on an undirected weighted graph (km on edges).

    Heuristic: straight-line Haversine km to target (admissible for triangle inequality on metric
    embedding, used here as a research proxy).
    """
    s, t = _key(source), _key(target)
    open_heap: list[tuple[float, tuple[float, float]]] = []
    heapq.heappush(open_heap, (0.0, s))
    g_score: dict[tuple[float, float], float] = {s: 0.0}
    came_from: dict[tuple[float, float], tuple[float, float] | None] = {s: None}

    def h(u: tuple[float, float]) -> float:
        pt_u = (u[0], u[1])
        pt_t = (t[0], t[1])
        return haversine_km(pt_u, pt_t)

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
            edge_km = float(G[u][v].get("km", G[u][v].get("weight", 0.0)))
            tentative = g_score[u] + edge_km
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


def benchmark_dijkstra_vs_astar(
    G: nx.Graph,
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Return (km_dijkstra, km_astar) per pair; distances match on nonnegative fixed graphs."""
    out: list[tuple[float, float]] = []
    for a, b in pairs:
        _, dk = dijkstra_shortest_path(G, a, b)
        _, ak = _astar_manual(G, a, b)
        out.append((dk, ak))
    return out
