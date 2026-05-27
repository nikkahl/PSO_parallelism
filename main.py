"""
main.py
Asynchronous Parallel execution script using Legacy PSO and Particle classes.
"""
from __future__ import annotations

import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from particle import Particle
from pso import PSO
from functions import sphere_function, rastrigin_function
from visualizer import Visualizer


def _async_worker_task(
    p_idx: int,
    particle: Particle,
    pso_engine: PSO,
    obj_func,
    bounds: tuple[float, float],
    max_iter: int,
    v_max: float,
    shared_gbest_pos,
    shared_gbest_score,
    lock,
    seed: int
) -> tuple[int, Particle, list[np.ndarray]]:
    
    np.random.seed(seed)

    my_trajectory = [particle.position.copy()]
    
    for _ in range(max_iter):
        particle.fitness_value = float(obj_func(particle.position))
        particle.update_personal_best()
        
        with lock:
            if particle.fitness_value < shared_gbest_score.value:
                shared_gbest_score.value = particle.fitness_value
                for d in range(len(bounds)):
                    shared_gbest_pos[d] = particle.position[d]
            
            current_gbest_pos = np.array(shared_gbest_pos[:])

        pso_engine.g_best_position = current_gbest_pos
        pso_engine.calc_velocity(particle)
        
        particle.velocity = np.clip(particle.velocity, -v_max, v_max)
        particle.update_position()
        particle.position = np.clip(particle.position, bounds[0], bounds[1])
        
        my_trajectory.append(particle.position.copy())

    return p_idx, particle, my_trajectory


def main() -> None:
    target_function = rastrigin_function
    func_title = r"Rastrigin Function (Async Parallel PSO)"
    search_bounds = (-5.12, 5.12)
    known_optimum = (0.0, 0.0)
    dimensions = 2
    
    n_particles = 30 
    max_iter = 100
    w, c1, c2 = 0.7, 1.5, 1.5
    v_max = (search_bounds[1] - search_bounds[0]) * 0.15 
    
    n_workers = max(1, min(os.cpu_count() or 1, n_particles))

    print(f"Starting Asynchronous PSO ({func_title}) with {n_workers} parallel workers...")
    t0 = time.perf_counter()

    pso_engine = PSO(w=w, c1=c1, c2=c2)
    particles = [Particle(num_dimensions=dimensions, bounds=search_bounds) for _ in range(n_particles)]

    # спільна пам'ять для процесів
    manager = multiprocessing.Manager()
    shared_gbest_score = manager.Value('d', float("inf"))
    shared_gbest_pos = manager.Array('d', [0.0] * dimensions)
    lock = manager.Lock()

    #  паралельно
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, p in enumerate(particles):
            seed = 42 + i
            futures.append(
                executor.submit(
                    _async_worker_task,
                    i, p, pso_engine, target_function, search_bounds, max_iter, v_max,
                    shared_gbest_pos, shared_gbest_score, lock, seed
                )
            )
        
        results = [future.result() for future in as_completed(futures)]

    # сорт результати за індексом (as_completed)
    results.sort(key=lambda x: x[0])
    all_trajectories = [res[2] for res in results]

    trajectory = []
    gbest_trajectory = []
    
    # Відновлюємо історію крок за кроком для візуалізатора
    current_best_val = float("inf")
    current_best_pos = np.zeros(dimensions)

    for step in range(max_iter + 1):
        step_positions = np.zeros((n_particles, dimensions))
        
        for i in range(n_particles):
            pos = all_trajectories[i][step]
            step_positions[i] = pos
            
            val = target_function(pos)
            if val < current_best_val:
                current_best_val = val
                current_best_pos = pos.copy()
                
        trajectory.append(step_positions)
        gbest_trajectory.append(current_best_pos.copy())

    total_s = time.perf_counter() - t0

    print("\nOptimization Complete")
    print(f"  x* = {list(shared_gbest_pos)}")
    print(f"  Best Function Value = {shared_gbest_score.value:.4f}")
    print(f"  Total time = {total_s:.3f} s\n")
    
    print("Generating visualizations.")
    
    Visualizer.animate_swarm_2d(
        trajectory, 
        bounds=search_bounds, 
        max_iter=max_iter,
        func_name=func_title, 
        optimum_pos=known_optimum, 
        out_gif="pso_swarm_2d.gif"
    )
    
    Visualizer.animate_swarm_3d(
        trajectory, 
        bounds=search_bounds, 
        max_iter=max_iter, 
        obj_func=target_function,
        func_name=func_title, 
        optimum_pos=known_optimum, 
        out_gif="pso_swarm_3d.gif"
    )
    
    print("\nAll done")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()