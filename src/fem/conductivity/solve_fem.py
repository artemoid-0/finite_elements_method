import numpy as np
import cupy as cp
from datetime import datetime
from src.fem.conductivity.conductivity_matrix import element_conductivity_matrix, element_conductivity_matrix, element_conductivity_matrix, assemble_global_conductivity_matrix
from src.fem.conductivity.boundary_conditions import apply_boundary_conditions

def solve_fem_heat_transfer(node_coords, elements, k, fixed_nodes, fixed_temperatures, heat_sources):
    """
    Solves a finite element heat transfer problem.

    Parameters:
    node_coords (np.ndarray): Node coordinates (Nx2).
    elements (list of list of int): List of elements, each specified as a list of node indices.
    k (float): Thermal conductivity of the material.
    fixed_nodes (list of int): List of indices of fixed nodes.
    fixed_temperatures (list of float): List of temperatures for fixed nodes.
    heat_sources (np.ndarray): Vector of heat flows (N).

    Returns:
    np.ndarray: Vector of temperatures (N).
    """

    start_time = datetime.now()
    K_global = assemble_global_conductivity_matrix(elements, node_coords, k)
    print('Time taken to assemble global conductivity matrix: ', datetime.now() - start_time)
    F = heat_sources.copy()

    # Application of boundary conditions
    start_time = datetime.now()
    K_global, F = apply_boundary_conditions(K_global, F, fixed_nodes, fixed_temperatures)
    print('Time taken to apply boundary conditions: ', datetime.now() - start_time)

    # Solution of a system of equations
    start_time = datetime.now()
    temperatures = np.linalg.solve(K_global, F)
    # print("Converting:", datetime.now())
    # K_global_cp = cp.asarray(K_global)
    # F_cp = cp.asarray(F)
    # print("Converting completed:", datetime.now())
    # temperatures = (cp.linalg.solve(cp.asarray(K_global), cp.asarray(F)))
    # print("Converting:", datetime.now())
    # temperatures = cp.asnumpy(temperatures)
    # print("Converting completed:", datetime.now())
    print('Time taken to solve a system of equations: ', datetime.now() - start_time)
    return temperatures


if __name__ == "__main__":
    node_coords = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [0, 3], [1, 3], [2, 3], [3, 3]
    ])

    elements = [
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
        [4, 5, 9], [4, 9, 8], [5, 6, 10], [5, 10, 9], [6, 7, 11], [6, 11, 10],
        [8, 9, 13], [8, 13, 12], [9, 10, 14], [9, 14, 13], [10, 11, 15], [10, 15, 14]
    ]

    k = 1.0

    # Boundary conditions: we record temperatures only at corner nodes and one internal node
    fixed_nodes = [0, 3, 12, 15]
    fixed_temperatures = [100.0, 100.0, 0.0, 0.0]

    # Heat sources: setting uniform heat distribution
    heat_sources = np.zeros(len(node_coords))
    heat_sources[5] = 75.0  # Example of an internal heat source

    temperatures = solve_fem_heat_transfer(node_coords, elements, k, fixed_nodes, fixed_temperatures, heat_sources)

    print("Temperatures in nodes:\n", temperatures)
