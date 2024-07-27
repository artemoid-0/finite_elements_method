import numpy as np

def apply_boundary_conditions(K, F, fixed_nodes, fixed_temperatures):
    """
    Применяет граничные условия к системе уравнений для теплопередачи.

    Parameters:
    K (np.ndarray): Глобальная матрица проводимости.
    F (np.ndarray): Вектор правых частей.
    fixed_nodes (list of int): Список индексов фиксированных узлов.
    fixed_temperatures (list of float): Температуры фиксированных узлов.

    Returns:
    np.ndarray, np.ndarray: Измененные матрица проводимости и вектор правых частей.
    """
    for i, node in enumerate(fixed_nodes):
        K[node, :] = 0
        K[:, node] = 0
        K[node, node] = 1
        F[node] = fixed_temperatures[i]
    return K, F

if __name__ == '__main__':
    # Пример использования функции
    K = np.array([[2, -1], [-1, 2]], dtype=float)
    F = np.array([0, 0], dtype=float)
    fixed_nodes = [0]
    fixed_temperatures = [100.0]
    K_bc, F_bc = apply_boundary_conditions(K, F, fixed_nodes, fixed_temperatures)
    print(K_bc)
    print(F_bc)
