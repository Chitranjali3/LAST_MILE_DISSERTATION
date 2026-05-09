# Cost Module Implementation Plan

## Goal

Add a configurable **monetary cost module** to the routing pipeline so the "main model" can consume explicit cost inputs and produce route outputs optimized on business cost (not only distance/time proxies).

This plan is designed to be implemented incrementally without breaking existing behavior.

## Current State (Baseline)

- Route optimization already computes distance/time/fuel-like proxy fitness in `core/ga_optimizer.py`.
- Pipeline reporting in `core/routing.py` and `main.py` exposes km/time metrics.
- No explicit money-based cost components are modeled (fuel price, labor rate, fixed trip cost, penalties).

## Target State

- Introduce a first-class `CostConfig` input object.
- Compute per-route and total **currency-denominated** costs.
- Keep old metrics for compatibility, but add new fields for cost-aware optimization/reporting.
- Allow toggling between:
  - distance-proxy objective (current)
  - cost objective (new)
  - hybrid weighted objective (optional later phase)

---

## Phase 1: Cost Model Specification

### 1.1 Define required cost inputs

Create a configuration schema (single source of truth) with defaults:

- `currency`: string (e.g., `"INR"`)
- `fuel_price_per_liter`: float
- `vehicle_km_per_liter`: float
- `driver_hourly_wage`: float
- `service_time_min_per_stop`: float
- `fixed_cost_per_route`: float
- `maintenance_cost_per_km`: float
- `late_delivery_penalty_per_min`: float
- `failed_window_penalty`: float
- `co2_cost_per_km` (optional, can default 0)

### 1.2 Define cost formulas

For each route:

- `fuel_cost = (route_km / vehicle_km_per_liter) * fuel_price_per_liter`
- `labor_cost = (route_time_min / 60) * driver_hourly_wage`
- `maintenance_cost = route_km * maintenance_cost_per_km`
- `service_cost = n_stops * service_time_min_per_stop * (driver_hourly_wage / 60)` (or fold into labor)
- `time_window_penalty = late_minutes * late_delivery_penalty_per_min`
- `hard_violation_penalty = failed_window_penalty if infeasible else 0`
- `total_route_cost = fixed_cost_per_route + fuel_cost + labor_cost + maintenance_cost + penalties (+ co2_cost)`

### 1.3 Decide distance/time source policy

Priority for route_km and route_time:

1. OSRM (`osrm_road_km`, `osrm_duration_min`) if available and valid
2. A* graph estimate (`astar_leg_km`) + derived time proxy
3. GA geometry fallback (`ga_tour_km`) + derived time proxy

Document this fallback policy clearly so cost output is reproducible.

---

## Phase 2: Data Contracts and Config Plumbing

### 2.1 Add config object

Add `CostConfig` dataclass in a new module, e.g. `core/cost_model.py`.

### 2.2 Extend pipeline function signatures

Update:

- `run_optimized_routes(..., cost_config: CostConfig | None = None, optimize_for: str = "distance_proxy")`
- `summarize_savings(..., cost_config: CostConfig | None = None)`

### 2.3 CLI input support in `main.py`

Add options:

- `--optimize-for {distance_proxy,cost,hybrid}`
- `--cost-config path/to/cost_config.json`

If no file is provided, use conservative defaults and print them in output metadata.

---

## Phase 3: Cost Engine Implementation

### 3.1 New `core/cost_model.py`

Include:

- `CostConfig` dataclass
- `RouteCostBreakdown` dataclass
- `compute_route_cost(...) -> RouteCostBreakdown`
- `compute_total_cost(...) -> dict`

### 3.2 Enrich `RouteResult`

Add fields:

- `route_time_min_effective`
- `route_km_effective`
- `cost_breakdown` (serializable dict)
- `total_route_cost`

### 3.3 GA objective integration

In `core/ga_optimizer.py`, allow evaluator strategy:

- existing proxy evaluator (default for backward compatibility)
- cost evaluator using `CostConfig` + estimated duration

Important: do not remove current evaluator; keep both switchable.

---

## Phase 4: Reporting and Main-Model Input Contract

### 4.1 Extend JSON report in `main.py`

Add sections:

- `cost_config_used`
- `optimized_routes[].cost_breakdown`
- `optimized_routes[].total_route_cost`
- `totals.total_cost`
- `comparison_naive_vs_optimized.cost_saved`

### 4.2 Stable input payload for main model

Produce a compact payload block in report:

- `model_input.cost_features`:
  - route-level: km, duration, stops, capacity utilization, penalties, total cost
  - aggregate-level: total_cost, avg_cost_per_stop, cost_per_km

This should be versioned:

- `model_input_schema_version: "cost_v1"`

### 4.3 Backward compatibility

Keep existing keys unchanged. Add new keys only.

---

## Phase 5: Validation and Tests

### 5.1 Unit tests

Add tests for `core/cost_model.py`:

- formula correctness
- fallback source selection (OSRM/A*/GA)
- penalty behavior for violations

### 5.2 Integration tests

End-to-end run with:

- `--use-osrm` on
- `--use-osrm` off
- cost config custom file

Verify totals and schema fields exist and are numeric.

### 5.3 Sensitivity sanity checks

Confirm monotonic behavior:

- Higher fuel price -> higher total cost
- Higher wage -> higher labor share
- More late minutes -> higher penalties

---

## Phase 6: Rollout Strategy

1. Land config + cost engine without changing optimizer objective.
2. Add reporting fields and verify no regressions.
3. Enable optional cost objective behind CLI flag.
4. Compare distance-proxy vs cost objective outcomes on same seed.
5. Make cost objective default only after validation.

---

## Suggested Milestone Breakdown (PR-ready)

- **M1:** `core/cost_model.py` + unit tests
- **M2:** `RouteResult` and reporting schema additions
- **M3:** CLI config loading + `--optimize-for` support
- **M4:** GA evaluator strategy switch (distance/cost)
- **M5:** Docs update in `README.md` with examples

---

## Risks and Mitigations

- **Risk:** Mixing estimated vs real travel time sources can bias cost.
  - **Mitigation:** Log metric source per route (`osrm`, `astar`, `ga_proxy`).
- **Risk:** Over-penalization causes unstable GA behavior.
  - **Mitigation:** Normalize/clip penalties and tune weights with fixed seeds.
- **Risk:** Breaking downstream consumers of report JSON.
  - **Mitigation:** additive schema changes only + explicit schema version.

---

## Definition of Done

- Cost module can be configured via file/CLI.
- Output JSON includes route-level and total monetary costs.
- Main-model input includes versioned cost feature block.
- Existing distance/time outputs remain unchanged and valid.
- Unit + integration tests pass for cost and non-cost runs.

---

## Example `cost_config.json` (starter)

```json
{
  "currency": "INR",
  "fuel_price_per_liter": 102.0,
  "vehicle_km_per_liter": 32.0,
  "driver_hourly_wage": 120.0,
  "service_time_min_per_stop": 4.0,
  "fixed_cost_per_route": 25.0,
  "maintenance_cost_per_km": 1.8,
  "late_delivery_penalty_per_min": 3.0,
  "failed_window_penalty": 150.0,
  "co2_cost_per_km": 0.0
}
```

