import numpy as np
from particle import Particle


class PSO:
    def __init__(
            self,
            w: float = 0.5,  # Коефіцієнт інерції (наскільки частинка зберігає свій напрямок)
            c1: float = 0.4,  # Когнітивний коефіцієнт (тяжіння до власного досвіду)
            c2: float = 0.4  # Соціальний коефіцієнт (тяжіння до досвіду групи)
    ):
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Глобальний рекорд усього рою
        self.g_best_position = None
        self.g_best_value = float('inf')

    def calc_velocity(self, particle: Particle):
        """Рахує нову швидкість за класичною формулою PSO."""
        r1 = np.random.rand(len(particle.position))
        r2 = np.random.rand(len(particle.position))

        cognitive_velocity = self.c1 * r1 * (particle.p_best_position - particle.position)
        social_velocity = self.c2 * r2 * (self.g_best_position - particle.position)

        # Оновлюємо швидкість частинки
        particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity

    def update_global_best(self, particle: Particle):
        """Перевіряє, чи не побила частинка глобальний рекорд."""
        if particle.fitness_value < self.g_best_value:
            self.g_best_value = particle.fitness_value
            self.g_best_position = particle.position.copy()