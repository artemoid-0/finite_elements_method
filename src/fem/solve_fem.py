import numpy as np
from stiffness_matrix import assemble_global_stiffness_matrix
from boundary_conditions import apply_boundary_conditions

def solve_fem(node_coords, elements, E, nu, fixed_nodes, forces):
    """
    Решает задачу методом конечных элементов.

    Parameters:
    node_coords (np.ndarray): Координаты узлов (Nx2).
    elements (list of list of int): Список элементов, каждый из которых задан как список индексов узлов.
    E (float): Модуль Юнга материала.
    nu (float): Коэффициент Пуассона материала.
    fixed_nodes (list of int): Список индексов фиксированных узлов.
    forces (np.ndarray): Вектор внешних сил (2N).

    Returns:
    np.ndarray: Вектор перемещений (2N).
    """
    K_global = assemble_global_stiffness_matrix(elements, node_coords, E, nu)
    F = forces.copy()

    # Применение граничных условий
    K_global, F = apply_boundary_conditions(K_global, F, fixed_nodes)

    # Решение системы уравнений
    displacements = np.linalg.solve(K_global, F)
    return displacements

# Пример использования функции
if __name__ == "__main__":
    E = 210e9  # Модуль Юнга
    nu = 0.3   # Коэффициент Пуассона
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elements = [[0, 1, 2], [0, 2, 3]]
    fixed_nodes = [0, 3]
    forces = np.array([0, 0, 0, 0, 0, -1000, 0, 0])  # Пример внешних сил

    displacements = solve_fem(node_coords, elements, E, nu, fixed_nodes, forces)
    print(displacements)