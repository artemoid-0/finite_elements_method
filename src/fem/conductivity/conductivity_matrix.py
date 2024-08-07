import numpy as np

def element_conductivity_matrix(k, coords):
    """
    Calculates the elemental conductivity matrix for a triangular element.

    Parameters:
    k (float): Thermal conductivity of the material.
    coords (np.ndarray): Element node coordinates (3x2).

    Returns:
    np.ndarray: Elemental conductivity matrix (3x3).
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    # Calculating the area of an element
    A = 0.5 * np.abs(np.linalg.det(np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ])))

    # Matrix B for heat transfer
    B = np.array([
        [y2 - y3, y3 - y1, y1 - y2],
        [x3 - x2, x1 - x3, x2 - x1]
    ]) / (2 * A)

    # Elemental conductivity matrix
    ke = (k * A) * (B.T @ B)

    return ke

def assemble_global_conductivity_matrix(elements, node_coords, k):
    """
    Builds a global conductivity matrix from element matrices.

    Parameters:
    elements (list of list of int): List of elements, each specified as a list of node indices.
    node_coords (np.ndarray): Node coordinates (Nx2).
    k (float): Thermal conductivity of the material.

    Returns:
    np.ndarray: Global conductivity matrix (N x N).
    """
    N = len(node_coords)
    K_global = np.zeros((N, N))

    for element in elements:
        coords = node_coords[element]
        ke = element_conductivity_matrix(k, coords)

        for i in range(3):
            for j in range(3):
                K_global[element[i], element[j]] += ke[i, j]

    return K_global


if __name__ == '__main__':
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elements = [[0, 1, 2], [0, 2, 3]]
    k = 1.0
    K_global = assemble_global_conductivity_matrix(elements, node_coords, k)
    print(K_global)
