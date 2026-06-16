import logging
import multiprocessing
import numpy as np
from functions import rastrigin_function,sphere_function
from visualizer import Visualizer
from optimizer_island import run_island_pso
from classic_pso import classic_pso

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    search_bounds = (-5.12, 5.12)
    target_function = rastrigin_function
    dimensions = 2
    n_particles = 30
    max_iter = 200
    
    MODE = "Clasic" 
    
    if MODE == "Clasic":
        logger.info("Запуск Clasic PSO (30 спроб для графіка)...")
        scores, times = [], []
        all_histories = []
        
        for _ in range(30):
            s, _, t, _, hist = classic_pso(
                target_function, search_bounds, dimensions, n_particles, max_iter, 
                0.7, 1.5, 1.5, 1.5, "Rastrigin", (0.0, 0.0)
            )
            scores.append(s)
            times.append(t)
            all_histories.append(hist)
            
        avg_history = np.mean(all_histories, axis=0)
        
        Visualizer.plot_convergence(avg_history, "Rastrigin Classic (Baseline)", out_file="conv_classic.png")
        
        logger.info(f"Classic PSO -> Середній фітнес: {np.mean(scores):.6f}, Середній час: {np.mean(times):.4f} с")
        
    elif MODE == "Island":
        intervals = [10, 25, 50, 100]
        results_data = []
        for interval in intervals:
            scores, times = [], []
            for _ in range(30):
                s, _, t, hist = run_island_pso(target_function, search_bounds, dimensions, n_particles, max_iter, migration_interval=interval)
                scores.append(s)
                times.append(t)
            
            results_data.append({"interval": interval, "mean_score": np.mean(scores), "std_dev": np.std(scores), "mean_time": np.mean(times)})
            Visualizer.plot_convergence(hist, f"Rastrigin (Int={interval})", out_file=f"conv_{interval}.png")

        Visualizer.plot_convergence(avg_history, "Rastrigin Classic (Baseline)", out_file="conv_classic.png")
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()