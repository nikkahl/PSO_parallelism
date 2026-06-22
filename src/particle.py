"""Particle state for legacy PSO implementations."""

from __future__ import annotations
import numpy as np

class Particle:
    """A particle used by older PSO demos."""

    def __init__(self, num_dimensions: int, bounds: tuple[float, float]):
        low, high = bounds
        self.position = np.random.uniform(low, high, num_dimensions)
        self.velocity = np.random.uniform(-1, 1, num_dimensions)

        self.p_best_position = self.position.copy()
        self.p_best_value = float("inf")
        self.fitness_value = float("inf")

    def update_position(self) -> None:
        """Update the position using the current velocity."""
        self.position = self.position + self.velocity

    def update_personal_best(self) -> None:
        """Update the personal best based on the current fitness value."""
        if self.fitness_value < self.p_best_value:
            self.p_best_value = float(self.fitness_value)
            self.p_best_position = self.position.copy()