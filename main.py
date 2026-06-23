import logging
import multiprocessing as mp
import os

from src.functions import rastrigin_function, sphere_function
from src.visualizer import Visualizer
from src.optimizers.classic_pso import classic_pso
from src.optimizers.optimizer_island import run_island_pso

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    search_bounds = (-5.12, 5.12)
    target_function = rastrigin_function
    dimensions = 2 
    n_particles = 100
    max_iter = 50 
    
    os.makedirs("results/plots", exist_ok=True)
    
    MODE = "Classic" 
    
    if MODE == "Classic":
        logger.info("запуск Classic PSO...")
        score, pos, t_exec, trajectory, hist = classic_pso(
            target_function, search_bounds, dimensions, n_particles, max_iter, 
            0.7, 1.5, 1.5, 1.5, "Demo Classic", (0.0, 0.0)
        )
        logger.info(f"Знайдено мінімум: {score:.6f} за {t_exec:.4f} сек")
        
        # Visualizer.create_gif(trajectory, ...)
        Visualizer.animate_swarm_2d(
        trajectory=trajectory,
        bounds=search_bounds,
        max_iter=max_iter,
        func_name="Classic PSO (2D)",
        optimum_pos=(0.0, 0.0),
        out_gif="results/animations/pso_classic_2d.gif",
        fps=10
        )
    
    elif MODE == "Island":
        logger.info("запуск Island PSO...")
        score, pos, t_exec, hist = run_island_pso(
            target_function, search_bounds, dimensions, n_particles, max_iter, migration_interval=10
        )
        logger.info(f"Знайдено мінімум: {score:.6f} за {t_exec:.4f} сек")


if __name__ == "__main__":
    mp.freeze_support()
    main()