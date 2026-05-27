"""
main.py
Synchronous execution script using Legacy PSO and Particle classes.
"""
from __future__ import annotations

import time
import numpy as np
from particle import Particle
from pso import PSO
from functions import sphere_function, rastrigin_function
from visualizer import Visualizer

def main() -> None:
    target_function = rastrigin_function
    func_title = r"Rastrigin Function (Legacy PSO)"
    search_bounds = (-5.12, 5.12)
    known_optimum = (0.0, 0.0)
    dimensions = 2
    
    n_particles = 30 
    max_iter = 100
    w, c1, c2 = 0.7, 1.5, 1.5
    
    v_max = (search_bounds[1] - search_bounds[0]) * 0.15 

    print(f"Starting Synchronous Legacy PSO ({func_title}) with {n_particles} particles...")
    t0 = time.perf_counter()

    pso_engine = PSO(w=w, c1=c1, c2=c2)
    particles = [Particle(num_dimensions=dimensions, bounds=search_bounds) for _ in range(n_particles)]

    trajectory = []
    gbest_trajectory = []

    for step in range(max_iter + 1):
        step_positions = np.zeros((n_particles, dimensions))

        for i, p in enumerate(particles):
            p.fitness_value = target_function(p.position)
            p.update_personal_best()
            pso_engine.update_global_best(p)

            step_positions[i] = p.position.copy()

        trajectory.append(step_positions)
        gbest_trajectory.append(pso_engine.g_best_position.copy())

        if step < max_iter:
            for p in particles:
                pso_engine.calc_velocity(p)
                p.velocity = np.clip(p.velocity, -v_max, v_max)
                p.update_position()
                p.position = np.clip(p.position, search_bounds[0], search_bounds[1])

    total_s = time.perf_counter() - t0

    print("\n Optimization Complete")
    print(f"  x* = {pso_engine.g_best_position.tolist()}")
    print(f"  Best Function Value = {pso_engine.g_best_value:.4f}")
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
    
    print("\n All done")

if __name__ == "__main__":
    main()