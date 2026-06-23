"""
visualizer.py
Handles 2D and 3D rendering of the swarm trajectory, 
as well as static academic convergence plots.
Strict Academic Style with Agent Path Legend.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Callable

class Visualizer:
    @staticmethod
    def animate_swarm_2d(
        trajectory: list[np.ndarray],
        bounds: tuple[float, float],
        max_iter: int,
        func_name: str,
        optimum_pos: tuple[float, float] | None,
        *,
        out_gif: str = "results/animations/pso_swarm_2d.gif",
        fps: int = 15,
    ) -> None:
        
        out_dir = os.path.dirname(out_gif)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        low, high = bounds
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.set_facecolor('white') 
        ax.grid(True, linestyle='-', color='lightgrey', zorder=0)
        
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel(r"$x$", fontsize=12, labelpad=10)
        ax.set_ylabel(r"$y$", fontsize=12, labelpad=10)
        
        if optimum_pos is not None:
            ax.plot(optimum_pos[0], optimum_pos[1], marker='o', markersize=14, markerfacecolor='none', markeredgecolor='darkred', label="Global Optimum", zorder=20)
            
        step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, 
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgrey', alpha=0.9), zorder=20)

        num_particles = trajectory[0].shape[0]
        cmap = plt.get_cmap('tab20') 
        
        trails = [ax.plot([], [], c=cmap(i % 20), alpha=0.6, linewidth=1.5, label=f"Path Agent {i+1}", zorder=5)[0] for i in range(num_particles)]
        scat = ax.scatter([], [], c="black", s=20, edgecolors='none', label="Current Position", zorder=10)
        
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=2, facecolor='white', edgecolor='lightgrey', framealpha=1.0)
        ax.set_title(func_name, fontsize=14, pad=15)
        plt.tight_layout() 

        def init():
            scat.set_offsets(np.zeros((0, 2)))
            step_text.set_text('')
            for trail in trails:
                trail.set_data([], [])
            return (scat, step_text, *trails)

        total_logs = len(trajectory)
        frame_indices = np.linspace(0, total_logs - 1, max_iter, dtype=int)

        def update(frame_number: int):
            target_idx = frame_indices[frame_number]
            pos = trajectory[target_idx]
            scat.set_offsets(pos)
            
            if target_idx > 0:
                history_up_to = np.array(trajectory[:target_idx+1])
                for i in range(num_particles):
                    p_history = history_up_to[:, i, :]
                    trails[i].set_data(p_history[:, 0], p_history[:, 1])

            step_text.set_text(f"Step: {frame_number + 1} / {max_iter}")
            return (scat, step_text, *trails)

        ani = animation.FuncAnimation(fig, update, frames=max_iter, init_func=init, interval=80, blit=False)

        print(f"[2D] Saving swarm animation to {out_gif}...")
        ani.save(out_gif, writer=animation.PillowWriter(fps=fps))
        plt.close(fig) 

    @staticmethod
    def animate_swarm_3d(
        trajectory: list[np.ndarray],
        bounds: tuple[float, float],
        max_iter: int,
        obj_func: Callable,
        func_name: str,
        optimum_pos: tuple[float, float] | None,
        *,
        out_gif: str = "results/animations/pso_swarm_3d.gif",
        fps: int = 15,
    ) -> None:
        from mpl_toolkits.mplot3d import Axes3D
        
        out_dir = os.path.dirname(out_gif)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        low, high = bounds

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=35, azim=45)
        
        ax.set_facecolor('white')

        x_mesh = np.linspace(low, high, 50)
        y_mesh = np.linspace(low, high, 50)
        X, Y = np.meshgrid(x_mesh, y_mesh)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = obj_func(np.array([X[i, j], Y[i, j]]))

        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.5, edgecolor='none', zorder=0)
        z_min, z_max = np.min(Z), np.max(Z)
        
        if optimum_pos is not None:
            opt_z = obj_func(np.array(optimum_pos))
            ax.scatter(optimum_pos[0], optimum_pos[1], opt_z, s=150, c='red', marker='o', edgecolors='darkred', label="Global Optimum", zorder=20)
            
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_zlim(z_min, z_max + (z_max - z_min)*0.1) 
        
        ax.set_xlabel(r"$x$", fontsize=12, labelpad=10)
        ax.set_ylabel(r"$y$", fontsize=12, labelpad=10)
        ax.set_zlabel(r"$f(x, y)$", fontsize=12, labelpad=10)
        ax.set_title(func_name, fontsize=14, pad=20)

        step_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgrey'))

        num_particles = trajectory[0].shape[0]
        cmap = plt.get_cmap('tab20')
        
        trails = [ax.plot([], [], [], c=cmap(i % 20), alpha=0.7, linewidth=1.5, label=f"Path Agent {i+1}", zorder=5)[0] for i in range(num_particles)]
        scat_points = [ax.plot([], [], [], marker='o', c='black', markersize=6, ls='', label="Current Position" if _ == 0 else "")[0] for _ in range(num_particles)]
        
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8, ncol=2, facecolor='white', edgecolor='lightgrey', framealpha=1.0)
        plt.tight_layout()

        def init():
            step_text.set_text('')
            for trail in trails:
                trail.set_data([], [])
                trail.set_3d_properties([])
            for pt in scat_points:
                pt.set_data([], [])
                pt.set_3d_properties([])
            return (step_text, *trails, *scat_points)

        total_logs = len(trajectory)
        frame_indices = np.linspace(0, total_logs - 1, max_iter, dtype=int)

        def update(frame_number: int):
            target_idx = frame_indices[frame_number]
            pos_data = trajectory[target_idx]
            z_data = np.array([obj_func(p) for p in pos_data]) 
            
            for i in range(num_particles):
                scat_points[i].set_data([pos_data[i, 0]], [pos_data[i, 1]])
                scat_points[i].set_3d_properties([z_data[i]])

            if target_idx > 0:
                history_up_to = np.array(trajectory[:target_idx+1])
                for i in range(num_particles):
                    p_hist_x = history_up_to[:, i, 0]
                    p_hist_y = history_up_to[:, i, 1]
                    p_hist_z = np.array([obj_func(history_up_to[k, i]) for k in range(len(history_up_to))])
                    
                    trails[i].set_data(p_hist_x, p_hist_y)
                    trails[i].set_3d_properties(p_hist_z)

            step_text.set_text(f"Step: {frame_number + 1} / {max_iter}")
            return (step_text, *trails, *scat_points)

        ani = animation.FuncAnimation(fig, update, frames=max_iter, init_func=init, interval=80, blit=False)

        print(f"[3D] Saving swarm animation to {out_gif}...")
        ani.save(out_gif, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)

    @staticmethod
    def plot_convergence(history, func_name, out_file="results/plots/convergence_curve.png"):
        """будує графік збіжності"""
        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(len(history)), history, color='royalblue', linewidth=2.5)
        ax.set_yscale('log')
        ax.set_title(f"Convergence: {func_name}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Global best fitness (Log)")
        ax.grid(True, linestyle='--')
        fig.savefig(out_file, dpi=300)
        plt.close(fig)

    @staticmethod
    def print_experiment_table(results: list[dict]):
        """таблиця в терміналі"""
        print(f"\n{'='*65}")
        print(f"{'Інтервал':<10} | {'сер. фітнес':<15} | {'Std Dev':<10} | {'Сер. час (с)':<10}")
        print("-" * 65)
        for res in results:
            print(f"{res['interval']:<10} | {res['mean_score']:<15.6f} | {res['std_dev']:<10.4f} | {res['mean_time']:<10.4f}")
        print("=" * 65 + "\n")