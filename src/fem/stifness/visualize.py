import numpy as np
import matplotlib.pyplot as plt
from src.fem.mesh import create_regular_triangular_mesh_in_rectangle, create_triangular_mesh, create_adaptive_mesh, plot_mesh, plot_elements
from stiffness_matrix import assemble_global_stiffness_matrix
from boundary_conditions import apply_boundary_conditions
from solve_fem import solve_fem

def visualize_results(node_coords, elements, displacements, scale=1.0):
    """
    Визуализирует результаты МКЭ.

    Parameters:
    node_coords (np.ndarray): Координаты узлов (Nx2).
    elements (list of list of int): Список элементов, каждый из которых задан как список индексов узлов.
    displacements (np.ndarray): Вектор перемещений (2N).
    scale (float): Масштаб для отображения деформаций.
    """
    deformed_coords = node_coords + scale * displacements.reshape(-1, 2)

    fig, ax = plt.subplots()
    for element in elements:
        x = node_coords[element, 0]
        y = node_coords[element, 1]
        ax.fill(x, y, edgecolor='black', fill=False)

        x_def = deformed_coords[element, 0]
        y_def = deformed_coords[element, 1]
        ax.fill(x_def, y_def, edgecolor='red', fill=False)

    ax.set_aspect('equal')
    plt.show()


def main():
    np.random.seed(0)

    # Пример для прямоугольной сетки
    node_coords, elements = create_regular_triangular_mesh_in_rectangle(0, 1, 0, 1, 5, 5)

    # Визуализация исходной сетки
    plot_mesh(node_coords, elements, "Rectangular Mesh")
    plot_elements(node_coords, elements, "Rectangular Mesh Elements")

    # Параметры материала
    E = 210e9  # Модуль Юнга
    nu = 0.3  # Коэффициент Пуассона

    # Применение сил и граничных условий
    fixed_nodes = [0, 5]
    forces = np.zeros(2 * len(node_coords))
    forces[12] = -5e9  # Пример внешних сил

    # Решение задачи методом конечных элементов
    displacements = solve_fem(node_coords, elements, E, nu, fixed_nodes, forces)

    print(displacements)

    # Визуализация результатов
    visualize_results(node_coords, elements, displacements, scale=1.0)

if __name__ == "__main__":
    main()
