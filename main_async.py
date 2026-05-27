import os

# --- ЗАПОБІЖНИК ВІД ПЕРЕВАНТАЖЕННЯ ПАМ'ЯТІ ---
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
from particle import Particle
from pso import PSO


def objective_function(position):
    x, y = position
    return x + 2 + y - 2


# --- ФУНКЦІЯ ЧАСТИНКИ (ПРАЦЮЄ В ОКРЕМОМУ ПРОЦЕСІ) ---
def particle_worker(particle_id, bounds, w, c1, c2, max_iterations, shared_gbest_pos, shared_gbest_val,
                    shared_positions, shared_status, lock):
    particle = Particle(num_dimensions=2, bounds=bounds)
    pso = PSO(w=w, c1=c1, c2=c2)

    for _ in range(max_iterations):
        # 1. Оцінка
        particle.fitness_value = objective_function(particle.position)
        particle.update_personal_best()

        # 2. Оновлення глобального рекорду
        with lock:
            if particle.fitness_value < shared_gbest_val.value:
                shared_gbest_val.value = particle.fitness_value
                shared_gbest_pos[0] = particle.position[0]
                shared_gbest_pos[1] = particle.position[1]
            local_gbest_pos = np.array([shared_gbest_pos[0], shared_gbest_pos[1]])

        # 3. Рух
        pso.g_best_position = local_gbest_pos
        pso.calc_velocity(particle)
        particle.update_position()

        # 4. ЗАПИСУЄМО ПОТОЧНУ ПОЗИЦІЮ ДЛЯ АНІМАЦІЇ
        shared_positions[particle_id] = [particle.position[0], particle.position[1]]

        # Штучна затримка! Без неї процесор прорахує 50 кроків за 0.001 сек
        # і ви не встигнете побачити анімацію. Збільште число, щоб було повільніше.
        time.sleep(0.1)

        # Кажемо головному потоку, що ця частинка закінчила роботу
    shared_status[particle_id] = False


def main():
    num_particles = 30
    bounds = (-10.0, 10.0)
    max_iterations = 50
    w, c1, c2 = 0.7, 1.5, 1.5

    print("Запуск Асинхронного PSO з живою анімацією...")

    with mp.Manager() as manager:
        # Спільна пам'ять
        shared_gbest_val = manager.Value('d', float('inf'))
        shared_gbest_pos = manager.list([0.0, 0.0])
        lock = manager.Lock()

        # Нова пам'ять спеціально для малювання
        shared_positions = manager.list([[0.0, 0.0] for _ in range(num_particles)])
        shared_status = manager.list([True for _ in range(num_particles)])  # Стан роботи частинок

        processes = []

        # Запускаємо частинки
        for i in range(num_particles):
            p = mp.Process(
                target=particle_worker,
                args=(i, bounds, w, c1, c2, max_iterations, shared_gbest_pos, shared_gbest_val, shared_positions,
                      shared_status, lock)
            )
            processes.append(p)
            p.start()

        # --- БЛОК МАЛЮВАННЯ (Кінопроектор) ---
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))

        # Фон
        x_vals = np.linspace(bounds[0], bounds[1], 100)
        y_vals = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = X ** 2 + Y ** 2

        frame = 0
        # Малюємо, поки хоча б одна частинка ще має статус True (працює)
        while any(shared_status):
            # Робимо копію координат з пам'яті для малювання
            current_positions = list(shared_positions)
            xs = [p[0] for p in current_positions]
            ys = [p[1] for p in current_positions]

            ax.clear()
            ax.contour(X, Y, Z, levels=20, alpha=0.5, cmap='viridis')

            # Малюємо поточний стан частинок
            ax.scatter(xs, ys, color='blue', label='Частинки', zorder=3)

            # Малюємо лідера, якщо він вже є
            gbest_x, gbest_y = shared_gbest_pos[0], shared_gbest_pos[1]
            if gbest_x != 0.0 or gbest_y != 0.0:
                ax.scatter(gbest_x, gbest_y, color='red', marker='*', s=300, label='Глобальний рекорд', zorder=4)

            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[0], bounds[1])
            ax.set_title(f"Асинхронний PSO | Кадр: {frame} | Фітнес: {shared_gbest_val.value:.4f}")
            ax.legend()

            plt.pause(0.5)  # Швидкість оновлення кадрів екрану
            frame += 1

        # Коли анімація завершилась, закриваємо процеси коректно
        for p in processes:
            p.join()

        plt.ioff()
        print(f"Пошук завершено! Мінімум: {shared_gbest_val.value:.6f}")
        plt.show()


if __name__ == "__main__":
    main()