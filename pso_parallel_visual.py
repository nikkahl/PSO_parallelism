"""
Parallel Particle Swarm Optimization (PSO) - 2D & 3D Sphere Function (x1, x2 notation).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Callable, Iterable
import multiprocessing

import numpy as np


# ----------------------------------------------------------------------------
# 1. Global Objective
# ----------------------------------------------------------------------------
def objective_function(x: float | np.ndarray) -> float:
    """
    Класична проста функція Сфери: f(x1, x2) = x1^2 + x2^2
    Має один ідеальний мінімум рівно в точці (0, 0).
    """
    arr = np.asarray(x, dtype=float)
    return float(np.sum(arr**2))


# ----------------------------------------------------------------------------
# 2. Asynchronous Async PSO Worker
# ----------------------------------------------------------------------------
def _async_worker_task(
    p_idx: int,
    particle: Particle,
    obj_func: Callable[[float | np.ndarray], float],
    bounds: tuple[float, float],
    max_iter: int,
    w: float,
    c1: float,
    c2: float,
    v_max: float,
    dimensions: int,
    n_particles: int,
    shared_gbest_pos,
    shared_gbest_score,
    shared_positions,
    shared_history,
    shared_trajectory,
    lock,
    seed: int
) -> Particle:
    
    rng = np.random.default_rng(seed)
    low, high = bounds
    
    pos = np.copy(particle.position)
    vel = np.copy(particle.velocity)
    pbest_pos = np.copy(particle.pbest_position)
    pbest_score = particle.pbest_score
    
    for _ in range(max_iter):
        current_score = float(obj_func(pos))
            
        if current_score < pbest_score:
            pbest_score = current_score
            pbest_pos = np.copy(pos)
            
        with lock:
            if current_score < shared_gbest_score.value:
                shared_gbest_score.value = current_score
                for d in range(dimensions):
                    shared_gbest_pos[d] = pos[d]
            
            current_gbest_pos = np.array(shared_gbest_pos[:])
            
            start_idx = p_idx * dimensions
            for d in range(dimensions):
                shared_positions[start_idx + d] = pos[d]
                
            shared_history.append(shared_gbest_score.value)
            shared_trajectory.append(np.array(shared_positions[:]).reshape(n_particles, dimensions))
            
        r1 = rng.random(size=(dimensions,))
        r2 = rng.random(size=(dimensions,))
        
        cognitive = c1 * r1 * (pbest_pos - pos)
        social = c2 * r2 * (current_gbest_pos - pos)
        
        vel = w * vel + cognitive + social
        vel = np.clip(vel, -v_max, v_max)
        
        pos = pos + vel
        pos = np.clip(pos, low, high)
        
    particle.position = pos
    particle.velocity = vel
    particle.pbest_position = pbest_pos
    particle.score = current_score
    particle.pbest_score = pbest_score
    return particle


# ----------------------------------------------------------------------------
# 3. Particle dataclass
# ----------------------------------------------------------------------------
@dataclass(slots=True)
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    pbest_position: np.ndarray
    score: float
    pbest_score: float

    @classmethod
    def create(cls, bounds: tuple[float, float], dimensions: int, rng: np.random.Generator) -> "Particle":
        low, high = bounds
        position = rng.uniform(low, high, size=(dimensions,))
        velocity = rng.uniform(-3.0, 3.0, size=(dimensions,))
        return cls(
            position=position,
            velocity=velocity,
            pbest_position=np.copy(position),
            score=float("inf"),
            pbest_score=float("inf"),
        )


# ----------------------------------------------------------------------------
# 4. Asynchronous Swarm Optimizer
# ----------------------------------------------------------------------------
class SwarmOptimizer:
    def __init__(
        self,
        obj_func: Callable[[float | np.ndarray], float],
        n_particles: int,
        bounds: tuple[float, float],
        max_iter: int,
        w: float,
        c1: float,
        c2: float,
        v_max: float,
        *,
        dimensions: int = 2,
        random_seed: int = 42,
        n_workers: int | None = None,
    ) -> None:
        self.obj_func = obj_func
        self.n_particles = int(n_particles)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.max_iter = int(max_iter)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.v_max = float(v_max)
        self.dimensions = int(dimensions)
        self.random_seed = int(random_seed)
        self.n_workers = n_workers

        rng = np.random.default_rng(self.random_seed)
        self.particles: list[Particle] = [
            Particle.create(self.bounds, self.dimensions, rng) for _ in range(self.n_particles)
        ]

        self.gbest_position = np.copy(self.particles[0].position)
        self.gbest_score = float("inf")

        self.history: list[float] = []
        self.iter_seconds: list[float] = []
        self.trajectory: list[np.ndarray] = []

        self._positions = np.stack([p.position for p in self.particles], axis=0)

    def optimize(self) -> tuple[np.ndarray, float]:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_workers = int(self.n_workers or (os.cpu_count() or 1))
        n_workers = max(1, min(n_workers, self.n_particles))

        manager = multiprocessing.Manager()
        shared_gbest_score = manager.Value('d', self.gbest_score)
        shared_gbest_pos = manager.Array('d', self.gbest_position.tolist())
        shared_positions = manager.Array('d', self._positions.flatten().tolist())
        shared_history = manager.list()
        shared_trajectory = manager.list()
        lock = manager.Lock()

        t0 = time.perf_counter()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i, p in enumerate(self.particles):
                seed = self.random_seed + i + 1
                futures.append(
                    executor.submit(
                        _async_worker_task,
                        i, p, self.obj_func, self.bounds, self.max_iter,
                        self.w, self.c1, self.c2, self.v_max, self.dimensions, self.n_particles,
                        shared_gbest_pos, shared_gbest_score, shared_positions,
                        shared_history, shared_trajectory, lock, seed
                    )
                )
            
            self.particles = [future.result() for future in as_completed(futures)]

        self.gbest_score = shared_gbest_score.value
        self.gbest_position = np.array(shared_gbest_pos[:])
        self.history = list(shared_history)
        self.trajectory = list(shared_trajectory)
        
        return np.copy(self.gbest_position), float(self.gbest_score)


# ----------------------------------------------------------------------------
# 5. Visualizer (2D + 3D)
# ----------------------------------------------------------------------------
class Visualizer:
    @staticmethod
    def animate_swarm_2d(
        trajectory: list[np.ndarray],
        bounds: tuple[float, float],
        max_iter: int,
        *,
        out_gif: str = "pso_swarm_2d.gif",
        fps: int = 15,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        low, high = bounds

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.grid(True, linewidth=1.5)
        ax.set_xlabel("X1 Coordinate", fontsize=12)
        ax.set_ylabel("X2 Coordinate", fontsize=12)
        
        # Точка оптимуму в центрі (тепер просто червона точка)
        ax.plot(0, 0, marker='o', markersize=10, markerfacecolor="red", markeredgecolor="black", label="Minimum (0, 0)", zorder=5)
        
        step_text = ax.text(0.03, 0.96, '', transform=ax.transAxes, fontsize=16, 
                            fontweight='bold', verticalalignment='top', 
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8), zorder=20)

        num_particles = trajectory[0].shape[0]
        cmap = plt.get_cmap('tab20') 
        
        trails = [ax.plot([], [], c=cmap(i % 20), alpha=0.6, linewidth=1.5)[0] for i in range(num_particles)]
        scat = ax.scatter([], [], c="black", s=20, zorder=10)
        
        ax.legend(loc="lower right", framealpha=1.0)

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
            ax.set_title("2D View: Agents converging to the center", fontsize=14, pad=15)
            
            return (scat, step_text, *trails)

        ani = animation.FuncAnimation(
            fig, update, frames=max_iter, init_func=init, interval=80, blit=True
        )

        print(f"[2D] Saving swarm animation to {out_gif}...")
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_gif, writer=writer)
        print(f"[2D] Successfully saved: {out_gif}")
        plt.close(fig) 

    @staticmethod
    def animate_swarm_3d(
        trajectory: list[np.ndarray],
        bounds: tuple[float, float],
        max_iter: int,
        obj_func: Callable,
        *,
        out_gif: str = "pso_swarm_3d.gif",
        fps: int = 15,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d import Axes3D

        low, high = bounds

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=35, azim=45)

        x_mesh = np.linspace(low, high, 50)
        y_mesh = np.linspace(low, high, 50)
        X, Y = np.meshgrid(x_mesh, y_mesh)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = obj_func(np.array([X[i, j], Y[i, j]]))

        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.5, edgecolor='none')
        
        # Точка оптимуму на дні чаші (тепер червона точка)
        ax.scatter(0, 0, 0, s=80, c='red', marker='o', edgecolors='black', label="Minimum (0, 0, 0)", zorder=10)

        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_zlim(0, np.max(Z))
        ax.set_xlabel("X1 (Pos)", fontsize=10, labelpad=10)
        ax.set_ylabel("X2 (Pos)", fontsize=10, labelpad=10)
        ax.set_zlabel("Z (Fitness)", fontsize=10, labelpad=10)

        step_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=14, 
                              fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        num_particles = trajectory[0].shape[0]
        cmap = plt.get_cmap('tab20')
        
        trails = [ax.plot([], [], [], c=cmap(i % 20), alpha=0.7, linewidth=1.5)[0] for i in range(num_particles)]
        scat_points = [ax.plot([], [], [], marker='o', c='black', markersize=6)[0] for _ in range(num_particles)]
        
        ax.legend(loc="upper right")

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
            ax.set_title("3D View: Swarm falling into the Bowl", fontsize=14, pad=20)
            
            return (step_text, *trails, *scat_points)

        ani = animation.FuncAnimation(
            fig, update, frames=max_iter, init_func=init, interval=80, blit=False
        )

        print(f"[3D] Saving swarm animation to {out_gif}...")
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_gif, writer=writer)
        print(f"[3D] Successfully saved: {out_gif}")
        plt.close(fig) 


# ----------------------------------------------------------------------------
# 6. Main Execution
# ----------------------------------------------------------------------------
def main() -> None:
    # Межі карти від -100 до 100
    bounds = (-100.0, 100.0)
    n_particles = 20 
    max_iter = 100
    
    # PSO коефіцієнти
    w, c1, c2 = 0.7, 1.5, 1.5
    v_max = 6.0 

    t0 = time.perf_counter()
    optimizer = SwarmOptimizer(
        objective_function,
        n_particles=n_particles,
        bounds=bounds,
        max_iter=max_iter,
        w=w,
        c1=c1,
        c2=c2,
        v_max=v_max, 
        dimensions=2,
        n_workers=None,
    )

    print(f"Starting Asynchronous PSO with {n_particles} parallel workers...")
    best_pos, best_score = optimizer.optimize()
    total_s = time.perf_counter() - t0
    
    print("Best solution found:")
    print(f"  x* = {best_pos.tolist()}")
    print(f"  Best Function Value = {best_score:.4f}")
    print(f"  total_time = {total_s:.3f} s\n")

    print("Generating visualizations. Please wait...")
    
    Visualizer.animate_swarm_2d(
        optimizer.trajectory,
        bounds,
        max_iter=max_iter,
        out_gif="pso_swarm_2d.gif"
    )
    
    Visualizer.animate_swarm_3d(
        optimizer.trajectory,
        bounds,
        max_iter=max_iter,
        obj_func=objective_function,
        out_gif="pso_swarm_3d.gif"
    )
    
    print("\nAll done! Check your folder for 'pso_swarm_2d.gif' and 'pso_swarm_3d.gif'.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()