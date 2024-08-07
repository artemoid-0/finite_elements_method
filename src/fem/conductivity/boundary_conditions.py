import numpy as np

from src.fem.conductivity.conductivity_matrix import assemble_global_conductivity_matrix


def apply_boundary_conditions(K, F, fixed_nodes, fixed_temperatures):
    """
    Applies boundary conditions to a system of equations for heat transfer.

    Parameters:
    K (np.ndarray): Global conductivity matrix.
    F (np.ndarray): Right-hand side vector.
    fixed_nodes (list of int): List of fixed node indices.
    fixed_temperatures (list of float): Fixed node temperatures.

    Returns:
    np.ndarray, np.ndarray: Modified conductivity matrix and right-hand side vector.
    """
    for i, node in enumerate(fixed_nodes):
        K[node, :] = 0
        K[:, node] = 0
        K[node, node] = 1
        F[node] = fixed_temperatures[i]
    return K, F


if __name__ == '__main__':
    # Example of coordinates of nodes and elements
    node_coords = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2]
    ])
    elements = [
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]
    ]

    # Coefficient of thermal conductivity
    k = 1.0

    # Assembling the global conductivity matrix
    K_global = assemble_global_conductivity_matrix(elements, node_coords, k)

    # Right hand side vector
    F = np.zeros(len(node_coords))

    # Application of boundary conditions
    fixed_nodes = [0, 2, 6, 8]
    fixed_temperatures = [100.0, 100.0, 50.0, 50.0]
    K_global_bc, F_bc = apply_boundary_conditions(K_global, F, fixed_nodes, fixed_temperatures)

    print("Global conductivity matrix with boundary conditions:\n", K_global_bc)
    print("Vector of right-hand sides with boundary conditions:\n", F_bc)
