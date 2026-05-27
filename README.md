# PSO (Particle Swarm Optimization) — Parallel + Visualization

This project implements **Particle Swarm Optimization (PSO)** with **Global Best (gbest)** topology:

- **No high-level optimization libraries** (no `pyswarms`, etc.)
- **Parallel** fitness evaluation via `concurrent.futures.ProcessPoolExecutor`
- **Vectorized** velocity/position updates (no per-particle Python loops in the PSO step)
- **Visualization**:
  - convergence plot of `gbest_score`
  - swarm animation along the objective function (GIF)

By default, the algorithm minimizes the one-dimensional function \(f(x) = x^2\) on the interval \([-10, 10]\).

## Structure (Single Responsibility)

- `pso_parallel_visual.py`
  - `objective_function(x)` — global, picklable objective function (safe for Windows multiprocessing)
  - `Particle` — particle state only
  - `SwarmOptimizer` — optimization controller (gbest PSO, parallel fitness, history/trajectory)
  - `Visualizer` — analytics and plotting (convergence, timing, GIF animation)

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

After completion you will get:

- console output with the best found \(x^\*\) and \(f(x^\*)\) and total runtime
- a convergence plot
- a timing plot (seconds per iteration and cumulative)
- a GIF file `pso_swarm.gif` in the project root

## Configuration

In the `main()` function inside `pso_parallel_visual.py` you can change:

- `bounds`
- `n_particles`
- `max_iter`
- `w, c1, c2`
- `n_workers` (number of worker processes for `ProcessPoolExecutor`)

