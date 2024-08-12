import numpy as np


def element_stiffness_matrix(E, nu, coords):
    """
    Calculates the elemental stiffness matrix for a triangular element.

    Parameters:
    E (float): Young's modulus of the material.
    nu (float): Poisson's ratio of the material.
    coords (np.ndarray): Element node coordinates (3x2).

    Returns:
    np.ndarray: Elemental stiffness matrix (6x6).
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    # Calculating the area of ​​an element
    A = 0.5 * np.linalg.det(np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ]))

    # Matrix B
    B = np.array([
        [y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
        [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
        [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]
    ]) / (2 * A)

    # Matrix D (plane stress-strain state)
    D = (E / (1 - nu ** 2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])

    # Elemental stiffness matrix
    ke = A * np.dot(np.dot(B.T, D), B)

    return ke


def assemble_global_stiffness_matrix(elements, node_coords, E, nu):
    """
    Builds a global stiffness matrix from element matrices.

    Parameters:
    elements (list of list of int): List of elements, each specified as a list of node indices.
    node_coords (np.ndarray): Node coordinates (Nx2).
    E (float): Young's modulus of the material.
    nu (float): Poisson's ratio of the material.

    Returns:
    np.ndarray: Global stiffness matrix (2N x 2N).
    """
    N = len(node_coords)
    K_global = np.zeros((2 * N, 2 * N))

    for element in elements:
        coords = node_coords[element]
        ke = element_stiffness_matrix(E, nu, coords)

        for i in range(3):
            for j in range(3):
                K_global[2 * element[i]:2 * element[i] + 2, 2 * element[j]:2 * element[j] + 2] += ke[2 * i:2 * i + 2, 2 * j:2 * j + 2]

    return K_global


if __name__ == '__main__':
    # Example of using the function
    E = 210e9  # Young's modulus for steel, in Pa
    nu = 0.3  # Poisson's ratio
    elements = [[0, 1, 2]]  # One element connecting nodes 0, 1 and 2
    node_coords = np.array([[0, 0], [1, 0], [0, 1]])  # Node coordinates

    K_global = assemble_global_stiffness_matrix(elements, node_coords, E, nu)
    print(K_global)
