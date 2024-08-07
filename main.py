import numpy as np
import cupy as cp
import time

# Задаем размер матрицы и вектора
n = 20000  # можно увеличить размер для более длительного тестирования

# Генерация произвольной матрицы K и вектора F
np.random.seed(0)  # для воспроизводимости
start = time.time()
K_global = np.random.rand(n, n)
F = np.random.rand(n)
print("Generation time:", time.time() - start)

# Проверка решения с использованием numpy
start_time = time.time()
temperatures_numpy = np.linalg.solve(K_global, F)
print("Numpy solve time:", time.time() - start_time)

# Проверка времени копирования данных
copy_start_time = time.time()
K_global_gpu = cp.asarray(K_global)
F_gpu = cp.asarray(F)
copy_time = time.time() - copy_start_time
print("Data copy time to GPU:", copy_time)

# Проверка решения с использованием cupy, без учета времени копирования данных
solve_start_time = time.time()
temperatures_cupy = cp.linalg.solve(K_global_gpu, F_gpu)
solve_time = time.time() - solve_start_time
print("CuPy solve time (excluding data copy):", solve_time)
