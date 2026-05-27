import numpy as np


class Particle:
    def __init__(self, num_dimensions: int, bounds: tuple):
        # Ініціалізуємо випадкову позицію в межах bounds (наприклад, від -10 до 10)
        min_bound, max_bound = bounds
        self.position = np.random.uniform(min_bound, max_bound, num_dimensions)

        # Ініціалізуємо випадкову початкову швидкість
        self.velocity = np.random.uniform(-1, 1, num_dimensions)

        # Пам'ять частинки (персональний рекорд)
        self.p_best_position = self.position.copy()
        self.p_best_value = float('inf')  # Шукаємо мінімум, тому старт з нескінченності

        # Поточне значення функції (фітнес)
        self.fitness_value = float('inf')

    def update_position(self):
        """Оновлює координати частинки на основі її швидкості."""
        self.position += self.velocity

    def update_personal_best(self):
        """Оновлює персональний рекорд, якщо нова позиція краща."""
        if self.fitness_value < self.p_best_value:
            self.p_best_value = self.fitness_value
            self.p_best_position = self.position.copy()