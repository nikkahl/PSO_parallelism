import multiprocessing as mp
import numpy as np
import time
from typing import Callable, Tuple, Any
from src.particle import Particle
from src.pso import PSO

def island_worker(island_id: int, num_particles: int, target_function: Callable, bounds: Tuple[float, float], 
                  dimensions: int, max_iter: int, pipe_in: mp.connection.Connection, 
                  pipe_out: mp.connection.Connection, result_queue: mp.Queue, 
                  migration_interval: int) -> None: 
        
    swarm = [Particle(num_dimensions=dimensions, bounds=bounds) for _ in range(num_particles)]
    island_pso = PSO(w=0.7 , c1=1.5, c2=1.5)
    history = [] 

    for particle in swarm:
        particle.fitness_value = target_function(particle.position)
        particle.best_fitness_value = float(particle.fitness_value)
        particle.p_best_position = particle.position.copy()
        island_pso.update_global_best(particle)
    
    for step in range(max_iter):
        for particle in swarm:
            island_pso.calc_velocity(particle)
            particle.update_position()
            
            particle.fitness_value = target_function(particle.position)
            
            if particle.fitness_value < particle.best_fitness_value:
                particle.best_fitness_value = float(particle.fitness_value)
                particle.p_best_position = particle.position.copy()
                
            island_pso.update_global_best(particle)
        
        history.append(island_pso.g_best_value)
        
        if step > 0 and step % migration_interval == 0:
            pipe_out.send((island_pso.g_best_value, island_pso.g_best_position.copy()))
            
            if pipe_in.poll(0.01): 
                alien_score, alien_pos = pipe_in.recv()
                
                if alien_score < island_pso.g_best_value:
                    island_pso.g_best_value = float(alien_score)
                    island_pso.g_best_position = alien_pos.copy()
                    
                    worst_idx = max(range(len(swarm)), key=lambda i: swarm[i].fitness_value)
                    swarm[worst_idx].position = alien_pos.copy()
                    swarm[worst_idx].p_best_position = alien_pos.copy()
                    swarm[worst_idx].best_fitness_value = float(alien_score)
                    swarm[worst_idx].fitness_value = float(alien_score)
                    swarm[worst_idx].velocity.fill(0.0)

    result_queue.put((island_id, island_pso.g_best_value, island_pso.g_best_position, history))

def run_island_pso(target_func: Callable, bounds: Tuple[float, float], dimensions: int, 
                   num_particles: int, max_iter: int, migration_interval: int = 25) -> Tuple:
    
    n_islands = max(1, mp.cpu_count() - 1)
    particles_per_island = num_particles // n_islands
    
    pipes = [mp.Pipe(duplex=False) for _ in range(n_islands)]
    result_queue = mp.Queue()
    processes = []
    
    start_time = time.perf_counter()

    for i in range(n_islands):
        pipe_in = pipes[i][0]
        pipe_out = pipes[(i + 1) % n_islands][1]
        p = mp.Process(
            target=island_worker,
            args=(i, particles_per_island, target_func, bounds, dimensions, 
                  max_iter, pipe_in, pipe_out, result_queue, migration_interval)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        
    exec_time = time.perf_counter() - start_time
    best_overall_score = float('inf')
    best_overall_pos = None
    all_histories = [] 
    
    while not result_queue.empty():
        _, score, pos, hist = result_queue.get()
        all_histories.append(hist)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_pos = pos

    global_history = np.min(all_histories, axis=0) if all_histories else []
    return best_overall_score, best_overall_pos, exec_time, global_history