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
