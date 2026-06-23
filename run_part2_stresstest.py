import logging
import multiprocessing as mp
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from src.functions import griewank_function
from src.optimizers.classic_pso import classic_pso
from src.optimizers.optimizer_island import run_island_pso

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][PART 2 - STRESS TEST] %(message)s')
logger = logging.getLogger(__name__)

class Part2StressTestRunner:
    def __init__(self):
        self.bounds = (-600.0, 600.0)
        self.target_func = griewank_function 
        self.dimensions = 50
        self.max_iter = 300
        self.n_runs = 5 
        self.total_particles = 400
        os.makedirs("results/data", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)

        # self.dimensions = 100       
        #self.max_iter = 300         
        #self.n_runs = 3             #
        #self.total_particles = 800 
        
        self.csv_file_raw = "results/data/part2_experiments_raw.csv"
        self.csv_file_summary = "results/data/part2_statistical_summary.csv"
        self.plot_file = "results/plots/part2_griewank_comparison.png"
        
        self._init_storage()

    def _init_storage(self) -> None:
        with open(self.csv_file_raw, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Algorithm", "Total_Agents", "Cores_Used", 
                "Migration_Interval", "Run_ID", "Final_Fitness", "Execution_Time_Sec"
            ])

    def log_raw_run(self, algo: str, agents: int, cores: int, interval: Any, run_id: int, score: float, t_exec: float) -> None:
        with open(self.csv_file_raw, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([algo, agents, cores, interval, run_id, score, t_exec])

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

    def write_final_report(self, classic_data: Dict, island_data: Dict) -> None:
        with open(self.csv_file_summary, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Algorithm_Type", "Configuration", "Migration_Interval", 
                "Best_Fitness_Min", "Mean_Fitness", "Std_Dev", "Time_Sec", "Speedup"
            ])
            
            st = classic_data["stats"]
            writer.writerow([
                "Classic Serial", f"{classic_data['agents']} agents", "N/A",
                f"{st['best']:.6e}", f"{st['mean']:.6f}", f"{st['std_dev']:.4f}", 
                f"{st['mean_time']:.4f}", "1.00"
            ])
            
            st = island_data["stats"]
            writer.writerow([
                "Island Parallel", f"{island_data['cores']} cores", island_data['interval'],
                f"{st['best']:.6e}", f"{st['mean']:.6f}", f"{st['std_dev']:.5f}", 
                f"{st['mean_time']:.4f}", f"{st['speedup']:.2f}"
            ])

    def plot_best_comparison(self, classic_hist: np.ndarray, island_hist: np.ndarray, island_info: str):
        plt.figure(figsize=(10, 6))
        plt.plot(classic_hist, label=f"Classic PSO (1 Core, {self.total_particles} agents)", linewidth=2, color='#1f77b4')
        plt.plot(island_hist, label=f"Island PSO ({island_info}, {self.total_particles} agents total)", linewidth=2, color='#ff7f0e', linestyle='--')
        
        plt.title(f"Griewank Function Optimization (D={self.dimensions})", fontsize=14, fontweight='bold')
        plt.xlabel("Ітерації", fontsize=12)
        plt.ylabel("Глобальний мінімум (Log Scale)", fontsize=12)
        plt.yscale('log')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        plt.savefig(self.plot_file, dpi=300)
        logger.info(f"Графік збережено у {self.plot_file}")
        plt.close()

    def execute_part2(self):
        logger.info(f" Griewank, D={self.dimensions}, Агентів={self.total_particles}")
        
        logger.info("-> Запуск Classic PSO...")
        scores, times, classic_histories = [], [], []
        
        for run in range(self.n_runs):
            np.random.seed(100 + run)
            score, _, t_exec, _, hist = classic_pso(
                self.target_func, self.bounds, self.dimensions, n_particles=self.total_particles, max_iter=self.max_iter,
                w=0.7, c1=1.5, c2=1.5, v_max=50.0, func_title="Griewank", known_optimum=(0.0, 0.0)
            )
            scores.append(score)
            times.append(t_exec)
            classic_histories.append(hist)
            self.log_raw_run("Classic Serial", self.total_particles, 1, "N/A", run, score, t_exec)
            
        classic_stats = self.compile_metrics(scores, times)
        baseline_reference_time = classic_stats["mean_time"]
        classic_data = {"agents": self.total_particles, "stats": classic_stats}
        logger.info(f"[Classic] Час: {baseline_reference_time:.2f}с")

        best_cores = min(4, max(1, mp.cpu_count() - 1))
        best_interval = 20 #50
        logger.info(f" Island PSO ({best_cores} ядер, інтервал {best_interval})...")
        
        scores, times, island_histories = [], [], []
        
        for run in range(self.n_runs):
            np.random.seed(500 + run)
            score, _, t_exec, hist = run_island_pso(
                self.target_func, self.bounds, self.dimensions, 
                num_particles=self.total_particles, max_iter=self.max_iter, migration_interval=best_interval
            )
            scores.append(score)
            times.append(t_exec)
            island_histories.append(hist)
            self.log_raw_run("Island Parallel", self.total_particles, best_cores, best_interval, run, score, t_exec)

        island_stats = self.compile_metrics(scores, times, baseline_time=baseline_reference_time)
        island_data = {"cores": best_cores, "interval": best_interval, "stats": island_stats}
        logger.info(f"[Island] Час: {island_stats['mean_time']:.2f}с | Прискорення: {island_stats['speedup']:.2f}x")

        self.write_final_report(classic_data, island_data)
        avg_classic_hist = np.mean(classic_histories, axis=0)
        avg_island_hist = np.mean(island_histories, axis=0)
        self.plot_best_comparison(avg_classic_hist, avg_island_hist, f"{best_cores} Cores, Int={best_interval}")
        
if __name__ == "__main__":
    mp.freeze_support()
    runner = Part2StressTestRunner()
    runner.execute_part2()