"""
Particle Swarm Optimization (PSO) — baseline, parallel fitness, and generic engine.

Stage 1: sequential PSO with timed baseline.
Stage 2: parallel fitness evaluation via joblib.
Stage 3: run_pso(objective_function, ...) accepts any objective from outside.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from joblib import Parallel, delayed


# Test objective (Sphere) with artificial delay
def sphere_function(position: Sequence[float]) -> float:
    """Sphere: sum(x_i^2), global minimum at origin. Simulates heavy compute."""
    arr = np.asarray(position, dtype=float)
    return float(np.sum(arr ** 2))



# Particle
@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray = field(init=False)
    best_fitness: float = field(init=False)

    def __post_init__(self) -> None:
        self.best_position = self.position.copy()
        self.best_fitness = float("inf")



# Stage 3: generic PSO engine (supports parallel fitness)
def run_pso(
    objective_function: Callable[[Sequence[float]], float],
    *,
    n_particles: int = 50,
    n_iterations: int = 50,
    dimensions: int = 2,
    bounds: tuple[float, float] = (-10.0, 10.0),
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    parallel: bool = False,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple[np.ndarray, float, list[Particle]]:
    """
    Minimize objective_function using PSO.

    Returns (global_best_position, global_best_fitness, swarm).
    """
    low, high = bounds
    rng = np.random.default_rng(42)

    swarm: list[Particle] = []
    for _ in range(n_particles):
        pos = rng.uniform(low, high, size=dimensions)
        vel = rng.uniform(-1.0, 1.0, size=dimensions)
        swarm.append(Particle(position=pos, velocity=vel))

    global_best_position = swarm[0].position.copy()
    global_best_fitness = float("inf")

    def evaluate_swarm_sequential() -> list[float]:
        # Stage 1: вузьке місце — послідовний обхід рою
        fitness_results: list[float] = []
        for particle in swarm:
            fitness = objective_function(particle.position)
            fitness_results.append(fitness)
        return fitness_results

    def evaluate_swarm_parallel() -> list[float]:
        # Stage 2: той самий цикл, але фітнес на всіх ядрах
        fitness_results = Parallel(n_jobs=n_jobs)(
            delayed(objective_function)(p.position) for p in swarm
        )
        return fitness_results

    evaluate = evaluate_swarm_parallel if parallel else evaluate_swarm_sequential

    for iteration in range(n_iterations):
        fitness_values = evaluate()

        for particle, fitness in zip(swarm, fitness_values):
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        for particle in swarm:
            r1 = rng.random(dimensions)
            r2 = rng.random(dimensions)
            cognitive = c1 * r1 * (particle.best_position - particle.position)
            social = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive + social
            particle.position = particle.position + particle.velocity
            particle.position = np.clip(particle.position, low, high)

        if verbose and (iteration + 1) % 10 == 0:
            print(
                f"  iter {iteration + 1:3d} | "
                f"global best fitness = {global_best_fitness:.6f} | "
                f"position ≈ {global_best_position}"
            )

    return global_best_position, global_best_fitness, swarm


# ---------------------------------------------------------------------------
# Stage 1 & 2: benchmarks (same parameters for fair comparison)
# ---------------------------------------------------------------------------

def _run_benchmark(label: str, parallel: bool) -> tuple[float, np.ndarray, float]:
    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    best_pos, best_fit, _ = run_pso(
        sphere_function,
        n_particles=20,
        n_iterations=50,
        dimensions=2,
        parallel=parallel,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nElapsed: {elapsed:.2f} s")
    print(f"Best fitness: {best_fit:.8f}")
    print(f"Best position:  {best_pos}")
    return elapsed, best_pos, best_fit


if __name__ == "__main__":
    cores = os.cpu_count() or 1
    print("PSO on Sphere (50 particles, 50 iterations, sleep 0.01 per eval)")
    print(f"CPU cores (for expected speedup): {cores}")
    print("Expected minimum: fitness → 0, position → [0, 0]\n")

    t_seq, pos_seq, fit_seq = _run_benchmark(
        "Stage 1 — Sequential (Baseline)", parallel=False
    )
    t_par, pos_par, fit_par = _run_benchmark(
        "Stage 2 — Parallel fitness (joblib, n_jobs=-1)", parallel=True
    )

    print(f"\n{'=' * 60}")
    print("Comparison")
    print(f"{'=' * 60}")
    print(f"Baseline (sequential): {t_seq:.2f} s")
    print(f"Parallel:              {t_par:.2f} s")
    if t_par > 0:
        speedup = t_seq / t_par
        print(f"Speedup:                 {speedup:.2f}x (ideal ≈ {cores}x on fitness eval)")

    ok_seq = fit_seq < 0.1 and np.linalg.norm(pos_seq) < 1.0
    ok_par = fit_par < 0.1 and np.linalg.norm(pos_par) < 1.0
    print(f"\nSequential found minimum: {ok_seq}")
    print(f"Parallel found minimum:   {ok_par}")

    print("\nStage 3 — run_pso accepts any external objective:")
    def custom_quadratic(x: Sequence[float]) -> float:
        time.sleep(0.01)
        a = np.asarray(x, dtype=float)
        return float(np.sum((a - 1.0) ** 2))  # minimum at [1, 1]

    pos, fit, _ = run_pso(
        custom_quadratic,
        n_particles=30,
        n_iterations=30,
        parallel=True,
        verbose=False,
    )
    print(f"  custom_quadratic → fitness={fit:.6f}, position={pos}")
