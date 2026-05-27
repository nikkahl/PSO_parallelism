# PSO (Particle Swarm Optimization) — Parallel + Visualization

Цей проект реалізує **Particle Swarm Optimization (PSO)** з топологією **Global Best (gbest)**:

- **Без готових бібліотек оптимізації** (типу `pyswarms`)
- **Паралельна** оцінка `fitness` через `concurrent.futures.ProcessPoolExecutor`
- **Векторизовані** оновлення швидкостей/позицій (без per-particle циклів для кроків PSO)
- **Візуалізація**:
  - графік збіжності `gbest_score`
  - анімація руху рою вздовж цільової функції (GIF)

За замовчуванням мінімізує 1D функцію \(f(x)=x^2\) у межах \([-10, 10]\).

## Структура (Single Responsibility)

- `pso_parallel_visual.py`
  - `objective_function(x)` — **глобальна** цільова функція (picklable для Windows multiprocessing)
  - `Particle` — **тільки стан** частинки
  - `SwarmOptimizer` — **контролер алгоритму** (оптимізація, паралельний fitness, history/trajectory)
  - `Visualizer` — **тільки візуалізація** (convergence plot + GIF animation)

## Встановлення

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

```bash
python main.py
```

Після завершення:

- у консолі буде вивід найкращого знайденого \(x^\*\) та \(f(x^\*)\)
- відкриється графік збіжності
- відкриється графік часу виконання (сек/ітерацію та накопичувально)
- буде збережено GIF `pso_swarm.gif` у корені проекту

## Налаштування

У `main()` файлу `pso_parallel_visual.py` можна змінити:

- `bounds`
- `n_particles`
- `max_iter`
- `w, c1, c2`
- `n_workers` (кількість процесів для `ProcessPoolExecutor`)

