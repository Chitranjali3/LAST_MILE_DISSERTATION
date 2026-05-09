# OSRM Optional Upgrade - Implementation Checklist

## Objective

Integrate OSRM as a **best-use enhancement layer** (real-road enrichment) while guaranteeing:

- Core system always works without OSRM
- OSRM failure never blocks or crashes core flow
- Users always see clear status: connected vs not connected

---

## Scope Guardrails

- Do **not** change existing core optimization logic as a dependency on OSRM.
- Do **not** remove current GA/A*/VRPTW outputs.
- Do **not** introduce breaking JSON schema changes.
- Additive upgrade only: status, enrichment, and visibility.

---

## Milestone Plan (Execution Order)

## M1 - Runtime Status Model and Health Gate

### Files to touch

- `core/routing.py`
- `core/osrm_client.py` (only if health helper is needed)
- `main.py`

### Tasks

- Add a pipeline-level OSRM state model:
  - requested (`bool`)
  - connected (`bool`)
  - mode (`core_only` | `core_plus_osrm`)
  - reason (`not_requested` | `connected` | `unavailable` | `http_error` | `osrm_error`)
- Run a fast preflight connectivity check before route loop when OSRM is requested.
- If preflight fails:
  - set mode to `core_only`
  - skip OSRM route calls
  - continue normal route computation.
- Preserve existing per-route status counts.

### Acceptance criteria

- Running with no OSRM server does not fail.
- Output includes explicit OSRM connected/not connected status.
- Route results still exist with core metrics only.

---

## M2 - Standardize Effective Metric Source

### Files to touch

- `core/routing.py`
- `main.py`

### Tasks

- For each route, derive effective values with fallback priority:
  1. OSRM road metrics when valid
  2. A* graph metrics fallback
  3. GA fallback for geometry-only cases
- Add explicit source tag fields:
  - `effective_distance_source` (`osrm` | `astar` | `ga_proxy`)
  - `effective_time_source` (`osrm` | `proxy`)
- Ensure plotting selects geometry source deterministically and safely.

### Acceptance criteria

- Every route has distance/time source tags.
- Mixed runs (partial OSRM success) remain valid and understandable.
- No changes to existing legacy keys (only additions).

---

## M3 - CLI and Console UX Clarity

### Files to touch

- `main.py`
- (optional) `visual_presenter.py`

### Tasks

- At startup, print OSRM requested URL + status intent.
- After preflight, print:
  - `OSRM connected` (enrichment enabled), or
  - `OSRM not connected, using core routing`
- At completion, print enrichment coverage summary:
  - e.g. `OSRM enriched routes: X/Y`

### Acceptance criteria

- User can understand OSRM state from terminal logs alone.
- No ambiguity between requested and actually connected.

---

## M4 - Visual Status Surfacing

### Files to touch

- `visual_osrm_app.py`
- `visual_presenter.py`

### Tasks

- Add visible OSRM status text in info panel / slide annotations:
  - connected
  - not connected (fallback)
  - partial route enrichment
- Keep route rendering fallback unchanged:
  - OSRM geometry if valid
  - current internal polyline otherwise.

### Acceptance criteria

- Visual screens always show OSRM state clearly.
- Graph rendering never breaks due to OSRM outage.

---

## M5 - Reliability Controls (Timeouts + Circuit Breaker)

### Files to touch

- `core/osrm_client.py`
- `core/routing.py`

### Tasks

- Keep short request timeout (already present).
- Ensure one-way circuit breaker behavior:
  - once transport-level failure occurs, skip remaining OSRM calls in run.
- Ensure status reason is propagated to remaining routes cleanly.

### Acceptance criteria

- No repeated long waits when OSRM is down.
- Mid-run OSRM failure degrades once and proceeds quickly.

---

## M6 - Tests and Validation Matrix

### Files to add/touch

- `tests/test_osrm_optional_behavior.py` (new)
- Optional updates in existing test modules

### Scenarios (must pass)

- `--use-osrm` off: core-only normal run.
- `--use-osrm` on + server up: full enrichment.
- `--use-osrm` on + server down at start: immediate fallback.
- `--use-osrm` on + server failure mid-run: partial enrichment + continuation.
- Visual mode still runs in all above conditions.

### Assertions

- Output schema contains OSRM status block every time.
- Route count unaffected by OSRM availability.
- No unhandled exception from OSRM transport/API errors.

---

## M7 - Documentation Update

### Files to touch

- `README.md`

### Tasks

- Add "OSRM Optional Mode" section:
  - what it does
  - what it does not do
  - fallback behavior
  - example outputs (`connected` vs `not connected`)
- Update CLI examples showing optional usage and expected logs.

### Acceptance criteria

- New contributor can understand OSRM behavior without reading code.

---

## Field-Level Additions (Non-Breaking Contract)

## Pipeline block additions

- `pipeline.osrm.requested` (existing)
- `pipeline.osrm.base_url` (existing)
- `pipeline.osrm.connected` (new)
- `pipeline.osrm.mode` (new)
- `pipeline.osrm.reason` (new)
- `pipeline.osrm.status_counts` (existing)
- `pipeline.osrm.enriched_routes` (new)
- `pipeline.osrm.total_routes` (new)

## Route-level additions

- `effective_distance_km` (new)
- `effective_duration_min` (new)
- `effective_distance_source` (new)
- `effective_time_source` (new)

---

## Suggested PR Split

- **PR 1:** M1 + M3 (status model, preflight, CLI messaging)
- **PR 2:** M2 + M5 (effective metric source + resilience hardening)
- **PR 3:** M4 (visual surfacing)
- **PR 4:** M6 + M7 (tests + docs)

---

## Done Checklist

- [ ] Core flow works identically with OSRM absent.
- [ ] OSRM status is explicit in logs, JSON, and visuals.
- [ ] Enrichment happens only when connected.
- [ ] Fallback is automatic and fast.
- [ ] Existing consumers are not broken by schema changes.
- [ ] Tests cover connected, disconnected, and mid-run failure.

