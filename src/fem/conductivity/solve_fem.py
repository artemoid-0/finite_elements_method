import numpy as np
from src.fem.conductivity.conductivity_matrix import element_conductivity_matrix, element_conductivity_matrix, element_conductivity_matrix, assemble_global_conductivity_matrix
from src.fem.conductivity.boundary_conditions import apply_boundary_conditions

def solve_fem_heat_transfer(node_coords, elements, k, fixed_nodes, fixed_temperatures, heat_sources):
    """
    Решает задачу теплопередачи методом конечных элементов.

    Parameters:
    node_coords (np.ndarray): Координаты узлов (Nx2).
    elements (list of list of int): Список элементов, каждый из которых задан как список индексов узлов.
    k (float): Коэффициент теплопроводности материала.
    fixed_nodes (list of int): Список индексов фиксированных узлов.
    fixed_temperatures (list of float): Список температур для фиксированных узлов.
    heat_sources (np.ndarray): Вектор тепловых потоков (N).

    Returns:
    np.ndarray: Вектор температур (N).
    """
    K_global = assemble_global_conductivity_matrix(elements, node_coords, k)
    F = heat_sources.copy()

    # Применение граничных условий
    K_global, F = apply_boundary_conditions(K_global, F, fixed_nodes, fixed_temperatures)

    # Решение системы уравнений
    temperatures = np.linalg.solve(K_global, F)
    return temperatures

# Пример использования функции
if __name__ == "__main__":
    k = 1.0  # Коэффициент теплопроводности
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elements = [[0, 1, 2], [0, 2, 3]]
    fixed_nodes = [0, 3]
    fixed_temperatures = [100, 50]  # Пример граничных температур
    heat_sources = np.zeros(len(node_coords))  # Вектор тепловых потоков

    temperatures = solve_fem_heat_transfer(node_coords, elements, k, fixed_nodes, fixed_temperatures, heat_sources)
    print(temperatures)
