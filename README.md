

This project is a clean, modular implementation of the Particle Swarm Optimization (PSO) algorithm. It features beautiful, academic-style 2D and 3D animations that visualize how the swarm converges on an optimal solution.

### Key Features

* **No external optimization libraries** (built with pure Python and `numpy`).
* **Modular Structure**: Strict separation of algorithm logic, benchmark math, and visualizations.
* **Academic Visualization**: Generates 2D and 3D GIFs showing particle trajectories and the global best path.

## Project Structure

* `main.py` — The entry point and main configuration block.
* `pso.py` — The core PSO engine.
* `particle.py` — The `Particle` class (position, velocity, personal best).
* `functions.py` — Objective function library (e.g., Sphere, Rastrigin).
* `visualizer.py` — Handles 2D/3D rendering and GIF creation via `matplotlib`.

## How to Run

1. **Setup Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install numpy matplotlib

```
2. **Run the Algorithm**:
```bash
python main.py

```
Upon completion, the best found coordinates and runtime will be printed in the console. Two visualization files (`pso_swarm_2d.gif` and `pso_swarm_3d.gif`) will be saved in your project folder.

## Configuration

You can adjust the algorithm parameters directly in the `main()` function inside `main.py`:

* `target_function` (e.g., `rastrigin_function` or `sphere_function`)
* `search_bounds`
* `n_particles`
* `max_iter`
* PSO coefficients: `w`, `c1`, `c2`