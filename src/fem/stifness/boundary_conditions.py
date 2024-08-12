import numpy as np

def apply_boundary_conditions(K, F, fixed_nodes):
    """
    Applies boundary conditions to the system of equations.

    Parameters:
    K (np.ndarray): Global stiffness matrix.
    F (np.ndarray): Right-hand side vector.
    fixed_nodes (list of int): List of fixed node indices.

    Returns:
    np.ndarray, np.ndarray: Modified stiffness matrix and right-hand side vector.
    """
    for node in fixed_nodes:
        dof = [2 * node, 2 * node + 1]
        for d in dof:
            K[d, :] = 0
            K[:, d] = 0
            K[d, d] = 1
            F[d] = 0
    return K, F
