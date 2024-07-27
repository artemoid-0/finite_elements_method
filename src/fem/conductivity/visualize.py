import matplotlib.pyplot as plt
import numpy as np
from src.fem.mesh import create_rectangular_mesh, create_triangular_mesh, create_adaptive_mesh
from src.fem.conductivity.conductivity_matrix import assemble_global_conductivity_matrix
from src.fem.conductivity.boundary_conditions import apply_boundary_conditions
from src.fem.conductivity.solve_fem import solve_fem_heat_transfer

def visualize_heat_transfer(node_coords, elements, temperatures):
    """
    Визуализирует результаты задачи теплопередачи.

    Parameters:
    node_coords (np.ndarray): Координаты узлов (Nx2).
    elements (list of list of int): Список элементов, каждый из которых задан как список индексов узлов.
    temperatures (np.ndarray): Вектор температур (N).
    """
    plt.tricontourf(node_coords[:, 0], node_coords[:, 1], np.array(elements), temperatures, levels=14, cmap='RdYlBu')
    plt.colorbar()
    plt.title("Temperature Distribution")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def main():
    np.random.seed(0)

    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elements = [[0, 1, 2], [0, 2, 3]]
    fixed_nodes = [0, 3]
    fixed_temperatures = [100, 50]  # Пример граничных температур
    heat_sources = np.zeros(len(node_coords))  # Вектор тепловых потоков

    k = 1.0  # Коэффициент теплопроводности
    temperatures = solve_fem_heat_transfer(node_coords, elements, k, fixed_nodes, fixed_temperatures, heat_sources)
    visualize_heat_transfer(node_coords, elements, temperatures)


if __name__ == "__main__":
    main()
