"""Legacy PSO implementation (kept for older demos)."""

from __future__ import annotations

import numpy as np

from particle import Particle


class PSO:
    """A classical PSO engine for the legacy `Particle` class."""

    def __init__(self, w: float = 0.5, c1: float = 0.4, c2: float = 0.4) -> None:
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.g_best_position: np.ndarray | None = None
        self.g_best_value = float("inf")

    def calc_velocity(self, particle: Particle) -> None:
        """Compute and update the particle velocity."""
        if self.g_best_position is None:
            raise ValueError("Global best position is not initialized.")

        r1 = np.random.rand(len(particle.position))
        r2 = np.random.rand(len(particle.position))

        cognitive = self.c1 * r1 * (particle.p_best_position - particle.position)
        social = self.c2 * r2 * (self.g_best_position - particle.position)
        particle.velocity = self.w * particle.velocity + cognitive + social

    def update_global_best(self, particle: Particle) -> None:
        """Update the global best record if the particle is better."""
        if particle.fitness_value < self.g_best_value:
            self.g_best_value = float(particle.fitness_value)
            self.g_best_position = particle.position.copy()