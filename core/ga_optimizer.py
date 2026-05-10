"""
Genetic Algorithm for stop sequencing inside a delivery cluster.

We encode a **permutation** of customer visits as a chromosome. The fitness
function aggregates:
  - total tour length (Haversine legs from driver start through ordered stops)
  - travel time proxy (distance / speed)
  - light penalty term toward avoiding redundant detours (fuel proxy)

**DEAP** implements selection, crossover (PMX), and mutation (shuffle).

Complexity (sketch):
  - Per generation: O(P * n) for fitness on P individuals if tour length is O(n).
  - With G generations: **O(G * P * n)** for this encoding.
  - PMX crossover is typically O(n) per operation.

For dissertation reporting: GA does not guarantee global optimality; it explores
the permutation space under stochastic operators to approximate the minimum-weight
open TSP-style tour from a fixed start.
"""

from __future__ import annotations

import random

from deap import base, creator, tools

from utils import haversine_km

# Safe re-execution (notebooks / repeated imports).
try:
    creator.create("FitnessMinVRPTW", base.Fitness, weights=(-1.0,))
except RuntimeError:
    pass

try:
    creator.create("IndividualPermGA", list, fitness=creator.FitnessMinVRPTW)
except RuntimeError:
    pass


def _open_tour_distance(
    perm: list[int],
    coords: list[tuple[float, float]],
    start: tuple[float, float],
) -> float:
    """Driver at `start`, visit coords[perm[0]] ... coords[perm[-1]] in order (no return leg)."""
    if not perm:
        return 0.0
    d = haversine_km(start, coords[perm[0]])
    for i in range(len(perm) - 1):
        d += haversine_km(coords[perm[i]], coords[perm[i + 1]])
    return d


def _evaluate_individual(
    perm: list[int],
    coords: list[tuple[float, float]],
    start: tuple[float, float],
    speed_kmh: float,
) -> tuple[float]:
    dist = _open_tour_distance(perm, coords, start)
    time_pen = dist / max(speed_kmh, 1e-6)
    fuel = 0.35 * dist
    # Composite scalar fitness (single objective minimization).
    fit = dist + 0.25 * time_pen + 0.05 * fuel
    return (fit,)


def optimize_route_ga(
    cluster_coords: list[tuple[float, float]],
    driver_start: tuple[float, float],
    *,
    pop_size: int = 100,
    generations: int = 70,
    cx_prob: float = 0.85,
    mut_prob: float = 0.35,
    speed_kmh: float = 25.0,
    seed: int | None = 11,
) -> tuple[list[int], float]:
    """
    Optimize visit order over `cluster_coords` (local list, fixed positions).

    Returns (best_permutation_of_indices, best_fitness).
    """
    rnd = random.Random(seed)
    n = len(cluster_coords)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], _evaluate_individual([0], cluster_coords, driver_start, speed_kmh)[0]

    toolbox = base.Toolbox()

    def init_ind():
        base_perm = list(range(n))
        rnd.shuffle(base_perm)
        return creator.IndividualPermGA(base_perm)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(ind) -> tuple[float]:
        return _evaluate_individual(list(ind), cluster_coords, driver_start, speed_kmh)

    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    for _ in range(generations):
        offspring = toolbox.select(pop, len(pop) - 1)
        offspring = [toolbox.clone(x) for x in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rnd.random() < cx_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for m in offspring:
            if rnd.random() < mut_prob:
                toolbox.mutate(m)
                del m.fitness.values

        invalid = [x for x in offspring if not x.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        elite = tools.selBest(pop, 1)[0]
        offspring.append(toolbox.clone(elite))
        pop[:] = offspring

    best = tools.selBest(pop, 1)[0]
    return list(best), float(best.fitness.values[0])


def tour_distance_km(
    perm: list[int],
    cluster_coords: list[tuple[float, float]],
    driver_start: tuple[float, float],
) -> float:
    """Physical open-tour length for reporting (same geometry as GA fitness)."""
    return _open_tour_distance(perm, cluster_coords, driver_start)


def _evaluate_individual_vrptw(
    perm: list[int],
    coords: list[tuple[float, float]],
    start: tuple[float, float],
    time_windows: list[tuple[float, float]],
    *,
    speed_kmh: float,
    service_time_min: float,
    late_penalty_per_min: float,
    early_wait_weight: float,
) -> tuple[float]:
    """VRPTW-aware fitness.

    Simulates the tour forward in time using the same anchor convention as
    ``core.vrptw.validate_route_feasibility`` (clock starts at 0 == earliest
    window start across the route). Lateness is penalised heavily so the GA
    naturally orders stops by their delivery windows; idle waiting is
    discouraged but not fatal because some waiting is unavoidable in tight
    schedules.
    """
    if not perm:
        return (0.0,)
    anchor = min(tw[0] for tw in time_windows)
    clock = 0.0
    prev = start
    total_dist = 0.0
    late_pen = 0.0
    wait_pen = 0.0
    for idx in perm:
        seg = haversine_km(prev, coords[idx])
        total_dist += seg
        travel_min = (seg / max(speed_kmh, 1e-6)) * 60.0
        clock += travel_min
        earliest = float(time_windows[idx][0]) - anchor
        latest = float(time_windows[idx][1]) - anchor
        if clock < earliest:
            wait_pen += earliest - clock
            clock = earliest
        if clock > latest:
            late_pen += clock - latest
        clock += service_time_min
        prev = coords[idx]
    time_proxy = total_dist / max(speed_kmh, 1e-6)
    fuel = 0.35 * total_dist
    fit = (
        total_dist
        + 0.25 * time_proxy
        + 0.05 * fuel
        + late_penalty_per_min * late_pen
        + early_wait_weight * wait_pen
    )
    return (float(fit),)


def optimize_route_ga_vrptw(
    cluster_coords: list[tuple[float, float]],
    driver_start: tuple[float, float],
    time_windows: list[tuple[float, float]],
    *,
    pop_size: int = 100,
    generations: int = 100,
    cx_prob: float = 0.85,
    mut_prob: float = 0.35,
    speed_kmh: float = 25.0,
    service_time_min: float = 5.0,
    late_penalty_per_min: float = 50.0,
    early_wait_weight: float = 0.05,
    seed: int | None = 11,
) -> tuple[list[int], float]:
    """Time-window-aware GA: penalises lateness so visit order respects windows.

    Distance is still optimised, but a stop whose latest delivery time would be
    missed contributes a large additive penalty. As a result the algorithm
    prefers a slightly longer route that visits the earlier-windowed customer
    first, even when a later-windowed customer happens to be geographically
    closer to the driver.
    """
    if len(cluster_coords) != len(time_windows):
        raise ValueError("cluster_coords and time_windows length mismatch")
    rnd = random.Random(seed)
    n = len(cluster_coords)
    if n == 0:
        return [], 0.0

    def evaluate_perm(perm: list[int]) -> tuple[float]:
        return _evaluate_individual_vrptw(
            perm,
            cluster_coords,
            driver_start,
            time_windows,
            speed_kmh=speed_kmh,
            service_time_min=service_time_min,
            late_penalty_per_min=late_penalty_per_min,
            early_wait_weight=early_wait_weight,
        )

    if n == 1:
        return [0], evaluate_perm([0])[0]

    toolbox = base.Toolbox()
    earliest_first = sorted(range(n), key=lambda i: time_windows[i][0])

    def init_ind():
        # Seed ~30% of the population with the earliest-first heuristic so the
        # GA starts close to a feasible solution; the rest stay random for
        # exploration.
        if rnd.random() < 0.3:
            base_perm = list(earliest_first)
        else:
            base_perm = list(range(n))
            rnd.shuffle(base_perm)
        return creator.IndividualPermGA(base_perm)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(ind) -> tuple[float]:
        return evaluate_perm(list(ind))

    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    for _ in range(generations):
        offspring = toolbox.select(pop, len(pop) - 1)
        offspring = [toolbox.clone(x) for x in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rnd.random() < cx_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for m in offspring:
            if rnd.random() < mut_prob:
                toolbox.mutate(m)
                del m.fitness.values

        invalid = [x for x in offspring if not x.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        elite = tools.selBest(pop, 1)[0]
        offspring.append(toolbox.clone(elite))
        pop[:] = offspring

    best = tools.selBest(pop, 1)[0]
    return list(best), float(best.fitness.values[0])
