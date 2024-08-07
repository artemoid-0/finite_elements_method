import numpy as np
from src.fem.mass.mass_matrix import assemble_global_mass_matrix
from src.fem.mass.boundary_conditions import apply_boundary_conditions_mass

def solve_fem_mass(node_coords, elements, rho, fixed_nodes, external_forces):
    """
    Решает задачу конечных элементов для динамики с матрицей массы.

    Parameters:
    node_coords (np.ndarray): Координаты узлов.
    elements (list of list of int): Список элементов.
    rho (float): Плотность материала.
    fixed_nodes (list of int): Список индексов фиксированных узлов.
    external_forces (np.ndarray): Вектор внешних сил.

    Returns:
    np.ndarray: Вектор перемещений узлов.
    """
    M_global = assemble_global_mass_matrix(elements, node_coords, rho)
    M_global, external_forces = apply_boundary_conditions_mass(M_global, external_forces, fixed_nodes)

    try:
        print("Глобальная матрица массы перед решением системы уравнений:")
        print(M_global)
        print("Вектор внешних сил перед решением системы уравнений:")
        print(external_forces)
        displacements = np.linalg.solve(M_global, external_forces)
        print("Перемещения узлов:")
        print(displacements)
        return displacements
    except np.linalg.LinAlgError:
        print("Ошибка: Сингулярная матрица")
        return None


# Пример использования функции
if __name__ == "__main__":
    rho = 1.0  # Плотность материала
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elements = [[0, 1, 2], [0, 2, 3]]
    fixed_nodes = [0, 3]
    external_forces = np.zeros(2 * len(node_coords))  # Вектор внешних сил

    displacements = solve_fem_mass(node_coords, elements, rho, fixed_nodes, external_forces)
    print(displacements)
