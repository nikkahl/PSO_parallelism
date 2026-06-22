"""
functions.py
Library of objective functions for PSO evaluation.
"""
import numpy as np

def sphere_function(x: float | np.ndarray) -> float:
    """
    Classic Sphere function f(x) = sum(x^2).
    Minimum at (0, 0, ...) with a value of 0.
    """
    arr = np.asarray(x, dtype=float)
    return float(np.sum(arr**2))

def rastrigin_function(x: float | np.ndarray) -> float:
    """
    Rastrigin function: Complex multimodal surface with many local minima.
    Minimum at (0, 0, ...) with a value of 0.
    """
    arr = np.asarray(x, dtype=float)
    A = 10.0
    return float(A * len(arr) + np.sum(arr**2 - A * np.cos(2 * np.pi * arr)))

def basic_fucntion() -> float:
    pass 