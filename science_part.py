import logging
import multiprocessing as mp
import numpy as np
import csv
import os
from typing import Dict, List, Any
from src.functions import rastrigin_function
from src.optimizers.classic_pso import classic_pso
from src.optimizers.optimizer_island import run_island_pso

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][SCIENCE] %(message)s')
logger = logging.getLogger(__name__)

class PSOScienceRunner:
    def __init__(self):
        self.bounds = (-5.12, 5.12)
        self.target_func = rastrigin_function
        self.dimensions = 20
        self.max_iter = 200
        self.n_runs = 10  
        
        os.makedirs("results/data", exist_ok=True)
        
        self.csv_file = "results/data/pso_experiments_raw.csv"
        self.report_file = "results/data/pso_statistical_summary.csv"        
        
        self._init_storage()

    def _init_storage(self) -> None:
        with open(self.csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Architecture", "Total_Agents", "Cores_Used", 
                "Migration_Interval", "Run_ID", "Final_Fitness", "Execution_Time_Sec"
            ])

    def log_raw_run(self, arch: str, agents: int, cores: int, interval: Any, run_id: int, score: float, t_exec: float) -> None:
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([arch, agents, cores, interval, run_id, score, t_exec])

    def compile_metrics(self, scores: List[float], times: List[float], baseline_time: float = None) -> Dict[str, float]:
        scores_arr = np.array(scores)
        times_arr = np.array(times)
        mean_time = float(np.mean(times_arr))        
        speedup = float(baseline_time / mean_time) if baseline_time else 1.0
        
        return {
            "best": float(np.min(scores_arr)),
            "worst": float(np.max(scores_arr)),
            "mean": float(np.mean(scores_arr)),
            "std_dev": float(np.std(scores_arr)),
            "mean_time": mean_time,
            "speedup": speedup
        }

    def execute_macro_loop(self) -> None:
        classic_experiments_summary = []
        island_experiments_summary = []
        baseline_reference_time = None

        logger.info(" Classic PSO")
        agent_grid_classic = [30, 60, 100, 120]
        
        for n_agents in agent_grid_classic:
            logger.info(f"тестування PSO: розмір {n_agents} частинок")
            scores, times = [], []
            
            for run in range(self.n_runs):
                np.random.seed(1000 + run) 
                score, _, exec_time, _, _ = classic_pso(
                    self.target_func, self.bounds, self.dimensions, n_particles=n_agents, max_iter=self.max_iter,
                    w=0.7, c1=1.5, c2=1.5, v_max=1.5, func_title="Science Sweep", known_optimum=(0.0,0.0)
                )
                
                scores.append(score)
                times.append(exec_time)
                self.log_raw_run("Classic Serial", n_agents, 1, "N/A", run, score, exec_time)
            
            stats = self.compile_metrics(scores, times)
            classic_experiments_summary.append({"agents": n_agents, "stats": stats})
            
            if n_agents == 120:
                baseline_reference_time = stats["mean_time"]

        core_variants = [2, 4, 8]            
        migration_intervals = [10, 25, 100]   
        fixed_total_particles = 120

        for cores in core_variants:
            for interval in migration_intervals:
                logger.info(f" Island PSO: ядер = {cores}, інтервал міграції = {interval}")
                scores, times = [], []
                
                for run in range(self.n_runs):
                    np.random.seed(5000 + run)
                    
                    score, _, exec_time, _ = run_island_pso(
                        self.target_func, self.bounds, self.dimensions, 
                        num_particles=fixed_total_particles, max_iter=self.max_iter, migration_interval=interval
                    )
                    
                    scores.append(score)
                    times.append(exec_time)
                    self.log_raw_run("Island Parallel", fixed_total_particles, cores, interval, run, score, exec_time)
                
                stats = self.compile_metrics(scores, times, baseline_time=baseline_reference_time)
                island_experiments_summary.append({
                    "cores": cores,
                    "interval": interval,
                    "stats": stats
                })

        self.write_final_report(classic_experiments_summary, island_experiments_summary, fixed_total_particles)
        logger.info(f"дані збережено в: '{self.csv_file}', звіт: '{self.report_file}'")
    
    def write_final_report(self, classic_data: List[Dict], island_data: List[Dict], fixed_n: int) -> None:
        with open(self.report_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Algorithm_Type", "Configuration", "Migration_Interval", 
                "Best_Fitness_Min", "Mean_Fitness", "Std_Dev", "Time_Sec", "Speedup"
            ])
            
            for item in classic_data:
                st = item["stats"]
                writer.writerow([
                    "Classic Serial", f"{item['agents']} agents", "N/A",
                    f"{st['best']:.6e}", f"{st['mean']:.6f}", f"{st['std_dev']:.4f}", 
                    f"{st['mean_time']:.4f}", "1.00"
                ])
                
            for item in island_data:
                st = item["stats"]
                writer.writerow([
                    "Island Parallel", f"{item['cores']} cores", item['interval'],
                    f"{st['best']:.6e}", f"{st['mean']:.6f}", f"{st['std_dev']:.5f}", 
                    f"{st['mean_time']:.4f}", f"{st['speedup']:.2f}"
                ])

if __name__ == "__main__":
    mp.freeze_support()
    runner = PSOScienceRunner()
    runner.execute_macro_loop()