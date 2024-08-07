import matplotlib.pyplot as plt
import numpy as np
from src.fem.mesh import create_regular_triangular_mesh_in_rectangle, create_triangular_mesh, create_adaptive_mesh
from src.fem.mass.mass_matrix import assemble_global_mass_matrix
from src.fem.mass.boundary_conditions import apply_boundary_conditions_mass
from src.fem.mass.solve_fem import solve_fem_mass

def visualize_mass(node_coords, elements, displacements):
    """
    Визуализирует результаты задачи с матрицей массы.

    Parameters:
    node_coords (np.ndarray): Координаты узлов (Nx2).
    elements (list of list of int): Список элементов, каждый из которых задан как список индексов узлов.
    displacements (np.ndarray): Вектор перемещений (2N).
    """
    x = node_coords[:, 0] + displacements[::2]
    y = node_coords[:, 1] + displacements[1::2]

    plt.triplot(node_coords[:, 0], node_coords[:, 1], elements, color='lightgrey')
    plt.plot(x, y, 'ro')
    plt.triplot(x, y, elements)
    plt.title("Displacement Visualization")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def main():
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elements = [[0, 1, 2], [0, 2, 3]]
    fixed_nodes = [0, 3]
    external_forces = np.zeros(2 * len(node_coords))
    external_forces[4] = 100  # Пример внешней силы на узле 2 в направлении X

    rho = 1.0  # Плотность материала
    displacements = solve_fem_mass(node_coords, elements, rho, fixed_nodes, external_forces)

    if displacements is not None:
        visualize_mass(node_coords, elements, displacements)

if __name__ == "__main__":
    main()
