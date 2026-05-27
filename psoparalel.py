import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


# ---------------------------------------------------------
# 1. Наша Цільова Функція
# ---------------------------------------------------------
def objective_function(x, y):
    """Функція Сфери: x^2 + y^2. Мінімум у точці (0,0)"""
    time.sleep(0.01)  # Штучна затримка (імітуємо важкі обчислення)
    return x ** 2 + y ** 2


# ---------------------------------------------------------
# 2. Життя одного Агента (Працює як окремий процес)
# ---------------------------------------------------------
def worker_agent(agent_id, shared_positions, shared_global, lock, stop_event):
    """Ця функція крутиться паралельно для кожного агента"""
    np.random.seed()  # Щоб кожен процес мав свої унікальні випадкові числа
    bounds = [-10.0, 10.0]

    # Початкова випадкова позиція та швидкість
    pos = np.random.uniform(bounds[0], bounds[1], 2)
    vel = np.random.uniform(-1.0, 1.0, 2)

    # Власні рекорди агента
    p_best_pos = pos.copy()
    p_best_fit = float('inf')

    # Параметри PSO
    w, c1, c2 = 0.7, 1.5, 1.5

    # Агент живе і працює, поки ми не закриємо вікно з графіком
    while not stop_event.is_set():
        # 1. Рахуємо фітнес
        fit = objective_function(pos[0], pos[1])

        # 2. Оновлюємо власний рекорд
        if fit < p_best_fit:
            p_best_fit = fit
            p_best_pos = pos.copy()

            # 3. Конкуренція за Глобальний рекорд (використовуємо lock для безпеки)
            with lock:
                if fit < shared_global['fit']:
                    shared_global['fit'] = fit
                    shared_global['pos'] = pos.copy()

        # 4. Читаємо поточний глобальний рекорд
        with lock:
            g_best_pos = shared_global['pos'].copy()

        # 5. Оновлюємо швидкість та рухаємося далі
        r1, r2 = np.random.rand(2), np.random.rand(2)
        vel = w * vel + c1 * r1 * (p_best_pos - pos) + c2 * r2 * (g_best_pos - pos)
        pos = pos + vel

        # Обмежуємо рух агента межами екрану
        pos = np.clip(pos, bounds[0], bounds[1])

        # 6. Записуємо свою нову позицію у спільну пам'ять для ВІЗУАЛІЗАЦІЇ
        shared_positions[agent_id] = pos.tolist()

        # Невелика пауза, щоб ми встигли побачити рух очима (інакше злетяться миттєво)
        time.sleep(0.03)


# ---------------------------------------------------------
# 3. Головний блок: Запуск процесів та малювання
# ---------------------------------------------------------
if __name__ == '__main__':
    num_agents = 30  # Кількість частинок

    # Створюємо Менеджер багатопроцесорності для спільної пам'яті
    manager = multiprocessing.Manager()

    # Спільний список позицій (заповнюємо нулями спочатку)
    shared_positions = manager.list([[0.0, 0.0] for _ in range(num_agents)])
    # Спільний словник для глобального рекорду
    shared_global = manager.dict({'fit': float('inf'), 'pos': np.array([0.0, 0.0])})

    lock = manager.Lock()
    stop_event = multiprocessing.Event()  # Сигнал зупинки для агентів

    print(f"Запускаємо {num_agents} паралельних процесів-агентів...")

    # Запускаємо наших агентів
    processes = []
    for i in range(num_agents):
        p = multiprocessing.Process(
            target=worker_agent,
            args=(i, shared_positions, shared_global, lock, stop_event)
        )
        p.start()
        processes.append(p)

    # --- НАЛАШТУВАННЯ ВІЗУАЛІЗАЦІЇ (MATPLOTLIB) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Паралельний Асинхронний PSO (Live Візуалізація)")

    # Малюємо ціль (червоний хрестик у центрі)
    ax.scatter([0], [0], c='red', marker='x', s=100, label='Ідеальний Мінімум')
    # Створюємо об'єкт для точок-агентів
    scat = ax.scatter([], [], c='blue', s=50, alpha=0.6, edgecolors='black')
    ax.legend()


    # Функція, яка малює новий кадр кожні N мілісекунд
    def update(frame):
        # 1. Читаємо свіжі позиції зі спільної пам'яті
        current_positions = np.array(shared_positions)
        scat.set_offsets(current_positions)

        # 2. Оновлюємо заголовок з найкращим фітнесом
        with lock:
            best_fit = shared_global['fit']
        ax.set_title(f"Паралельний PSO | Найкращий фітнес: {best_fit:.4f}")
        return scat,


    # Що робити при закритті вікна
    def on_close(event):
        print("\nВікно закрито. Даємо команду агентам зупинитися...")
        stop_event.set()  # Відправляємо сигнал


    fig.canvas.mpl_connect('close_event', on_close)

    # Запускаємо анімацію!
    ani = animation.FuncAnimation(fig, update, interval=40, blit=False)
    plt.show()

    # Після того, як користувач закрив графік, чекаємо коректного завершення всіх процесів
    for p in processes:
        p.join()

    print("Всі процеси успішно завершено! Фінальний результат:", shared_global['fit'])