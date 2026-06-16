import time
import numpy as np
import logging

from particle import Particle
from pso import PSO
from visualizer import Visualizer

logger = logging.getLogger(__name__)

def classic_pso(target_function, bounds, dimensions, n_particles, max_iter, w, c1, c2, v_max, func_title, known_optimum):
   
    logger.info("[CLASSIC PSO] Запуск алгоритму в один потік (без паралелізму)")
    
    t0 = time.perf_counter()
    pso_engine = PSO(w=w, c1=c1, c2=c2)
    
    swarm = [Particle(num_dimensions=dimensions, bounds=bounds) for _ in range(n_particles)]
    
    global_best_score = float('inf')
    global_best_pos = np.zeros(dimensions)
    trajectory_history = []
    convergence_history = []
    
    initial_positions = np.array([p.position.copy() for p in swarm])
    trajectory_history.append(initial_positions)
    
    for p in swarm:
        score = float(target_function(p.position))
        p.fitness_value = score
        p.best_fitness_value = score
        p.p_best_position = p.position.copy()
        
        if score < global_best_score:
            global_best_score = score
            global_best_pos = p.position.copy()
       
    for step in range(max_iter):
        pso_engine.g_best_position = global_best_pos.copy()
        pso_engine.g_best_value = global_best_score
        
        for p in swarm:
            pso_engine.calc_velocity(p)
            p.velocity = np.clip(p.velocity, -v_max, v_max)
            p.update_position()
            p.position = np.clip(p.position, bounds[0], bounds[1])
            
            score = float(target_function(p.position))
            p.fitness_value = score
            
            if score < p.best_fitness_value:
                p.best_fitness_value = score
                p.p_best_position = p.position.copy()
                
            if score < global_best_score:
                global_best_score = score
                global_best_pos = p.position.copy()
                
        current_positions = np.array([p.position.copy() for p in swarm])
        trajectory_history.append(current_positions)
        convergence_history.append(global_best_score)
        
        if (step + 1) % 20 == 0 or step == max_iter - 1:
            logger.info(f"Ітерація {step + 1}/{max_iter} | Послідовний gbest: {global_best_score:.8f}")

    exec_time = time.perf_counter() - t0
    logger.info(f"[SERIAL CLASSIC PSO] Завершено за {exec_time:.3f} s. Рекорд: {global_best_score:.8f}")
    
    #Visualizer.animate_swarm_2d(trajectory_history, bounds=bounds, max_iter=max_iter, func_name=func_title, optimum_pos=known_optimum)
    #Visualizer.animate_swarm_3d(trajectory_history, bounds=bounds, max_iter=max_iter, obj_func=target_function, func_name=func_title, optimum_pos=known_optimum)
    
    return global_best_score, global_best_pos, exec_time, trajectory_history, convergence_history