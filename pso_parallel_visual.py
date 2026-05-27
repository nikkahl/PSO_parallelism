"""
Parallel Particle Swarm Optimization (PSO) with visualization (gbest topology).

This module implements a clean, SRP-oriented PSO from scratch (no pyswarms, etc.):
- `objective_function` is a *global* function to stay picklable on Windows.
- `Particle` stores only agent state (Single Responsibility).
- `SwarmOptimizer` controls the algorithm:
  - parallel fitness evaluation via `concurrent.futures.ProcessPoolExecutor`
  - vectorized velocity/position updates (no per-particle loops for the PSO step)
  - history and trajectory collection for plotting/animation
- `Visualizer` renders convergence plot and swarm animation (GIF).

The default objective is a one-dimensional parabola f(x) = x^2 (minimization).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Callable, Iterable

import numpy as np


def objective_function(x: float | np.ndarray) -> float:
    """
    Objective function to minimize.

    Accepts:
    - scalar `x` (float/int)
    - vector `x` (numpy array-like)

    Returns a Python float. For vectors, returns sum(x_i^2).
    """
    arr = np.asarray(x, dtype=float)
    return float(np.sum(arr**2))


def _is_default_objective(func: Callable[[float | np.ndarray], float]) -> bool:
    return getattr(func, "__name__", None) == "objective_function" and getattr(func, "__module__", None) == __name__


def _objective_values_1d(
    func: Callable[[float | np.ndarray], float], xs: np.ndarray
) -> np.ndarray:
    """
    Compute objective values for multiple 1D points.

    Returns values with the same order as `xs` and shape (len(xs),).
    """
    xs = np.asarray(xs, dtype=float)
    if _is_default_objective(func):
        return xs**2
    return np.array([func(float(x)) for x in xs], dtype=float)


def _evaluate_batch(obj_func: Callable[[float | np.ndarray], float], positions: np.ndarray) -> np.ndarray:
    """Evaluate the objective on a batch of particle positions."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        return (positions.astype(float) ** 2).astype(float)
    if _is_default_objective(obj_func):
        return np.sum(positions**2, axis=1).astype(float)
    return np.apply_along_axis(lambda v: float(obj_func(v)), 1, positions).astype(float)


@dataclass(slots=True)
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    pbest_position: np.ndarray
    score: float
    pbest_score: float

    @classmethod
    def create(cls, bounds: tuple[float, float], dimensions: int, rng: np.random.Generator) -> "Particle":
        low, high = bounds
        position = rng.uniform(low, high, size=(dimensions,))
        velocity = rng.uniform(-1.0, 1.0, size=(dimensions,))
        return cls(
            position=position,
            velocity=velocity,
            pbest_position=np.copy(position),
            score=float("inf"),
            pbest_score=float("inf"),
        )


class SwarmOptimizer:
    def __init__(
        self,
        obj_func: Callable[[float | np.ndarray], float],
        n_particles: int,
        bounds: tuple[float, float],
        max_iter: int,
        w: float,
        c1: float,
        c2: float,
        *,
        dimensions: int = 1,
        random_seed: int = 42,
        n_workers: int | None = None,
    ) -> None:
        self.obj_func = obj_func
        self.n_particles = int(n_particles)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.max_iter = int(max_iter)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.dimensions = int(dimensions)
        self.random_seed = int(random_seed)
        self.n_workers = n_workers

        rng = np.random.default_rng(self.random_seed)
        self.particles: list[Particle] = [
            Particle.create(self.bounds, self.dimensions, rng) for _ in range(self.n_particles)
        ]

        self.gbest_position = np.copy(self.particles[0].position)
        self.gbest_score = float("inf")

        self.history: list[float] = []
        self.iter_seconds: list[float] = []
        self.trajectory: list[np.ndarray] = []

        self._positions = np.stack([p.position for p in self.particles], axis=0)
        self._velocities = np.stack([p.velocity for p in self.particles], axis=0)
        self._pbest_positions = np.stack([p.pbest_position for p in self.particles], axis=0)
        self._scores = np.full((self.n_particles,), float("inf"), dtype=float)
        self._pbest_scores = np.full((self.n_particles,), float("inf"), dtype=float)

    def optimize(self) -> tuple[np.ndarray, float]:
        """Run the PSO optimization loop and return (gbest_position, gbest_score)."""
        from concurrent.futures import ProcessPoolExecutor

        low, high = self.bounds
        rng = np.random.default_rng(self.random_seed + 1)
        n_workers = int(self.n_workers or (os.cpu_count() or 1))
        n_workers = max(1, min(n_workers, self.n_particles))

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for _iter in range(self.max_iter):
                t0 = time.perf_counter()
                batches = np.array_split(self._positions, max(1, n_workers))

                results: Iterable[np.ndarray] = ex.map(
                    _evaluate_batch,
                    (self.obj_func for _ in batches),
                    batches,
                )
                fitness = np.concatenate(list(results), axis=0)

                self._scores = fitness

                improved = self._scores < self._pbest_scores
                if np.any(improved):
                    self._pbest_scores = np.where(improved, self._scores, self._pbest_scores)
                    self._pbest_positions[improved] = np.copy(self._positions[improved])

                best_idx = int(np.argmin(self._scores))
                best_score = float(self._scores[best_idx])
                if best_score < self.gbest_score:
                    self.gbest_score = best_score
                    self.gbest_position = np.copy(self._positions[best_idx])

                self.history.append(float(self.gbest_score))
                self.trajectory.append(np.copy(self._positions))

                r1 = rng.random(size=(self.n_particles, self.dimensions))
                r2 = rng.random(size=(self.n_particles, self.dimensions))
                cognitive = self.c1 * r1 * (self._pbest_positions - self._positions)
                social = self.c2 * r2 * (self.gbest_position - self._positions)
                self._velocities = self.w * self._velocities + cognitive + social
                self._positions = self._positions + self._velocities
                self._positions = np.clip(self._positions, low, high)
                self.iter_seconds.append(time.perf_counter() - t0)

        for i, p in enumerate(self.particles):
            p.position = np.copy(self._positions[i])
            p.velocity = np.copy(self._velocities[i])
            p.pbest_position = np.copy(self._pbest_positions[i])
            p.score = float(self._scores[i])
            p.pbest_score = float(self._pbest_scores[i])

        return np.copy(self.gbest_position), float(self.gbest_score)


class Visualizer:
    @staticmethod
    def plot_convergence(history: list[float]) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(history, linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Iteration")
        plt.ylabel("gbest_score")
        plt.title("PSO Convergence Plot")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_timing(iter_seconds: list[float]) -> None:
        """Plot per-iteration runtime and cumulative runtime."""
        import matplotlib.pyplot as plt

        it = np.arange(1, len(iter_seconds) + 1, dtype=int)
        per_iter = np.asarray(iter_seconds, dtype=float)
        cumulative = np.cumsum(per_iter)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(it, per_iter, linewidth=2)
        ax1.set_ylabel("Seconds / iteration")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("PSO Runtime")

        ax2.plot(it, cumulative, linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Cumulative seconds")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def animate_swarm(
        trajectory: list[np.ndarray],
        obj_func: Callable[[float | np.ndarray], float],
        bounds: tuple[float, float],
        *,
        out_gif: str = "pso_swarm.gif",
        fps: int = 20,
        show: bool = True,
    ) -> str:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        low, high = bounds
        xs = np.linspace(low, high, 400)
        ys = _objective_values_1d(obj_func, xs)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, color="black", linewidth=2, alpha=0.7)
        ax.set_xlim(low, high)
        ax.set_ylim(float(np.min(ys)), float(np.max(ys)))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title("Swarm Animation (PSO)")

        scat = ax.scatter([], [], c="red", s=30, alpha=0.8)

        def init():
            scat.set_offsets(np.zeros((0, 2)))
            return (scat,)

        def update(frame_idx: int):
            pos = trajectory[frame_idx].reshape(-1)
            vals = _objective_values_1d(obj_func, pos)
            scat.set_offsets(np.column_stack([pos, vals]))
            ax.set_title(f"Swarm Animation (PSO) — iter {frame_idx + 1}/{len(trajectory)}")
            return (scat,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(trajectory),
            init_func=init,
            interval=int(1000 / max(1, fps)),
            blit=True,
        )

        writer = animation.PillowWriter(fps=fps)
        ani.save(out_gif, writer=writer)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_gif


def main() -> None:
    bounds = (-10.0, 10.0)
    n_particles = 30
    max_iter = 50
    w, c1, c2 = 0.7, 1.5, 1.5

    t0 = time.perf_counter()
    optimizer = SwarmOptimizer(
        objective_function,
        n_particles=n_particles,
        bounds=bounds,
        max_iter=max_iter,
        w=w,
        c1=c1,
        c2=c2,
        dimensions=1,
        n_workers=None,
    )

    best_pos, best_score = optimizer.optimize()
    total_s = time.perf_counter() - t0
    print("Best solution found:")
    print(f"  x* = {best_pos.reshape(-1)[0]:.6f}")
    print(f"  f(x*) = {best_score:.10f}")
    print(f"  total_time = {total_s:.3f} s")

    Visualizer.plot_convergence(optimizer.history)
    Visualizer.plot_timing(optimizer.iter_seconds)
    gif_path = Visualizer.animate_swarm(
        optimizer.trajectory,
        objective_function,
        bounds,
        out_gif="pso_swarm.gif",
        show=True,
    )
    print(f"Swarm animation saved to: {gif_path}")


if __name__ == "__main__":
    main()

