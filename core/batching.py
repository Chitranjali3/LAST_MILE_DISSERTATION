"""
Greedy dynamic batching for streaming order intake.

Greedy batching iterates orders in arrival order and adds an order to the current
batch when it is geographically and temporally compatible with the batch anchor.
This supports Rule 3 (nearby grouping) and Rule 5 (dynamic insertion) at a
lightweight level before full spatial clustering.

Time complexity: O(n * B) per wave where B is mean batch size (dominated by
pairwise checks within a batch); typically O(n^2) worst case if batches grow large.
Space: O(n) for batches and anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils import haversine_km


@dataclass
class GreedyBatchConfig:
    """Hyperparameters for greedy compatibility checks."""

    max_batch_km: float = 2.5
    max_time_gap_min: float = 90.0
    max_orders_per_batch: int = 12


@dataclass
class Batch:
    """Container for a batch of order identifiers sharing a soft anchor."""

    order_ids: list[int] = field(default_factory=list)
    anchor_drop: tuple[float, float] | None = None
    tw_lo: float | None = None
    tw_hi: float | None = None


def _merge_tw(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> tuple[float, float] | None:
    """Intersection of two closed intervals; None if empty."""
    lo = max(lo_a, lo_b)
    hi = min(hi_a, hi_b)
    if lo > hi:
        return None
    return lo, hi


def greedy_dynamic_batch(
    orders: list[dict[str, Any]],
    *,
    config: GreedyBatchConfig | None = None,
) -> list[list[int]]:
    """
    Partition orders (in input order = dynamic arrival order) into batches.

    An order joins the current batch when:
    - drop is within `max_batch_km` of the batch anchor (first drop), and
    - time windows intersect with a slack expansion of `max_time_gap_min`, and
    - batch capacity not exceeded.

    If incompatible, a new batch starts at this order.
    """
    cfg = config or GreedyBatchConfig()
    if not orders:
        return []
    batches: list[Batch] = []
    current = Batch(
        order_ids=[orders[0]["order_id"]],
        anchor_drop=tuple(orders[0]["drop"]),
        tw_lo=float(orders[0]["time_window"][0]),
        tw_hi=float(orders[0]["time_window"][1]),
    )
    batches.append(current)

    for o in orders[1:]:
        oid = o["order_id"]
        drop = tuple(o["drop"])
        lo, hi = float(o["time_window"][0]), float(o["time_window"][1])
        anchor = current.anchor_drop or drop
        dist_ok = haversine_km(drop, anchor) <= cfg.max_batch_km
        tw_merged = _merge_tw(current.tw_lo or lo, current.tw_hi or hi, lo, hi)

        slack_ok = False
        if tw_merged is not None and current.tw_lo is not None and current.tw_hi is not None:
            slack_lo = current.tw_lo - cfg.max_time_gap_min
            slack_hi = current.tw_hi + cfg.max_time_gap_min
            slack_merged = _merge_tw(slack_lo, slack_hi, lo, hi)
            slack_ok = slack_merged is not None
        elif tw_merged is not None:
            slack_ok = True

        size_ok = len(current.order_ids) < cfg.max_orders_per_batch

        if dist_ok and slack_ok and size_ok and tw_merged is not None:
            current.order_ids.append(oid)
            current.tw_lo, current.tw_hi = tw_merged
            # Keep anchor at first drop for stability (Rule 1 handled separately).
        else:
            current = Batch(order_ids=[oid], anchor_drop=drop, tw_lo=lo, tw_hi=hi)
            batches.append(current)

    return [b.order_ids for b in batches]


def merge_same_location_orders(
    orders: list[dict[str, Any]],
    *,
    key_decimals: int = 5,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """
    Enforce Rule 1: orders with identical rounded drop coordinates merge into one stop.

    Returns aggregated pseudo-orders (one row per physical drop) and a mapping
    from original order_id -> representative order_id.
    """
    buckets: dict[tuple[float, float], list[dict[str, Any]]] = {}
    key_fn = lambda d: (round(d[0], key_decimals), round(d[1], key_decimals))
    order_to_rep: dict[int, int] = {}

    for o in orders:
        k = key_fn(tuple(o["drop"]))
        buckets.setdefault(k, []).append(o)

    merged: list[dict[str, Any]] = []
    for _k, group in buckets.items():
        rep = min(group, key=lambda x: x["order_id"])
        # Intersection of delivery windows (feasibility checked later in VRPTW).
        tw_lo = max(float(x["time_window"][0]) for x in group)
        tw_hi = min(float(x["time_window"][1]) for x in group)
        if tw_lo > tw_hi:
            # No overlap: relax to union hull so downstream can report conflict.
            tw_lo = min(float(x["time_window"][0]) for x in group)
            tw_hi = max(float(x["time_window"][1]) for x in group)
        total_w = sum(float(x["parcel_weight"]) for x in group)
        merged.append(
            {
                "order_id": int(rep["order_id"]),
                "merged_order_ids": [int(x["order_id"]) for x in group],
                "user_ids": [int(x["user_id"]) for x in group],
                "pickup": tuple(rep["pickup"]),
                "drop": tuple(rep["drop"]),
                "time_window": [tw_lo, tw_hi],
                "parcel_weight": total_w,
            }
        )
        for x in group:
            order_to_rep[int(x["order_id"])] = int(rep["order_id"])

    merged.sort(key=lambda r: r["order_id"])
    return merged, order_to_rep
