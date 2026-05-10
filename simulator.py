"""
Targeted simulation scenarios for research reporting.

Demonstrates:
  1) Duplicate drop consolidation (Rule 1)
  2) Intermediate-stop tour shape (Rule 2) vs. fragmented legs
  3) Dynamic order insertion waves (Rule 5) with distance deltas

These are **not** production Monte Carlo experiments, but compact reproducible
runs suitable for methodology chapters and algorithmic ablations.
"""

from __future__ import annotations

from typing import Any

from core.batching import merge_same_location_orders
from core.ga_optimizer import optimize_route_ga, tour_distance_km
from core.routing import cluster_route_graph_metrics, naive_independent_legs_km
from utils import ODISHA_REGION_CENTER, haversine_km, synthetic_drivers, synthetic_orders


def scenario_duplicate_building() -> dict[str, Any]:
    """
    Two customers share Building X: merged stop count should drop by one
    versus raw order list length.
    """
    orders = synthetic_orders(12, seed=1)
    merged, rep_map = merge_same_location_orders(orders)
    dup_users = [o["user_id"] for o in orders if tuple(o["drop"]) == tuple(orders[0]["drop"])]
    return {
        "raw_orders": len(orders),
        "merged_stops": len(merged),
        "same_building_user_ids": dup_users[:2],
        "merge_mapping_sample": dict(list(rep_map.items())[:5]),
    }


def scenario_collinear_intermediate() -> dict[str, Any]:
    """
    Construct A, C, B roughly collinear and compare chained tour vs. two isolated legs.
    The GA should prefer a single chain A -> C -> B (order may permute) over
    separate excursions when measured in kilometers.
    """
    center = ODISHA_REGION_CENTER
    A = (center[0] + 0.02, center[1] + 0.00)
    C = (center[0] + 0.035, center[1] + 0.005)
    B = (center[0] + 0.05, center[1] + 0.01)
    driver = (center[0] - 0.01, center[1] - 0.01)

    coords = [A, C, B]
    perm, _fit = optimize_route_ga(coords, driver, generations=90, pop_size=120, seed=3)
    chained = tour_distance_km(perm, coords, driver)
    fragmented = haversine_km(driver, A) + haversine_km(driver, C) + haversine_km(driver, B)
    ordered = [coords[i] for i in perm]
    dij_km, ast_km, _, _ = cluster_route_graph_metrics(driver, ordered)
    return {
        "best_perm_indices": perm,
        "open_tour_km": chained,
        "dijkstra_chain_km": dij_km,
        "astar_chain_km": ast_km,
        "sum_independent_one_way_km": fragmented,
        "improvement_km": max(0.0, fragmented - chained),
    }


def scenario_dynamic_insertion_wave() -> dict[str, Any]:
    """
    Simulate two arrival waves: baseline routes on wave-1, then re-optimize
    after inserting wave-2 orders (simple re-merge + distance accounting).
    """
    wave1 = synthetic_orders(10, seed=4)[:8]
    wave2 = synthetic_orders(16, seed=4)[8:16]
    drivers = synthetic_drivers(3, seed=2)

    m1, _ = merge_same_location_orders(wave1)
    naive1 = naive_independent_legs_km(m1, drivers)

    combined = wave1 + wave2
    m2, _ = merge_same_location_orders(combined)
    naive2 = naive_independent_legs_km(m2, drivers)

    return {
        "wave1_size": len(wave1),
        "wave2_size": len(wave2),
        "merged_after_wave1": len(m1),
        "merged_after_combine": len(m2),
        "naive_legs_km_wave1": naive1,
        "naive_legs_km_all_orders": naive2,
        "delta_naive_km_after_insertions": naive2 - naive1,
    }


def run_all() -> dict[str, Any]:
    return {
        "duplicate_location": scenario_duplicate_building(),
        "intermediate_stop": scenario_collinear_intermediate(),
        "dynamic_insertion": scenario_dynamic_insertion_wave(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_all(), indent=2))
