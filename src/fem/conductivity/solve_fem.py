from datetime import datetime

import numpy as np
import scipy as sp

from src.fem.conductivity.boundary_conditions import apply_boundary_conditions
from src.fem.conductivity.conductivity_matrix import assemble_global_conductivity_matrix


def solve_fem_heat_transfer(node_coords, elements, k, fixed_nodes, fixed_temperatures, heat_sources,
                            solver_method='solve'):
    """
    Solves a finite element heat transfer problem.

    Parameters:
    node_coords (np.ndarray): Node coordinates (Nx2).
    elements (list of list of int): List of elements, each specified as a list of node indices.
    k (float): Thermal conductivity of the material.
    fixed_nodes (list of int): List of indices of fixed nodes.
    fixed_temperatures (list of float): List of temperatures for fixed nodes.
    heat_sources (np.ndarray): Vector of heat flows (N).
    solver_method (str): Method to solve the system of equations.

    Returns:
    np.ndarray: Vector of temperatures (N).
    """

    start_time = datetime.now()
    K_global = assemble_global_conductivity_matrix(elements, node_coords, k)
    print('Time taken to assemble global conductivity matrix: ', datetime.now() - start_time)

    F = np.array(heat_sources).flatten()

    start_time = datetime.now()
    K_global, F = apply_boundary_conditions(K_global, F, fixed_nodes, fixed_temperatures)
    print('Time taken to apply boundary conditions: ', datetime.now() - start_time)

    density = np.count_nonzero(K_global) / K_global.size
    print(f"Matrix density: {density}")

    # Solution of a system of equations
    start_time = datetime.now()
    if solver_method == 'solve':
        temperatures = np.linalg.solve(K_global, F)
    elif solver_method == 'spsolve':
        temperatures = sp.sparse.linalg.spsolve(K_global, F)
    elif solver_method == 'lsqr':
        temperatures, istop, itn, r1norm = sp.sparse.linalg.lsqr(K_global, F)[:4]
        print(f"Residual norm (r1norm) for LSQR: {r1norm}")
    elif solver_method == 'cg':
        temperatures, info = sp.sparse.linalg.cg(K_global, F)
        if info != 0:
            print(f"Residual norm for CG: {np.linalg.norm(K_global @ temperatures - F)}")
    elif solver_method == 'bicg':
        temperatures, info = sp.sparse.linalg.bicg(K_global, F)
        if info != 0:
            print(f"Residual norm for BiCG: {np.linalg.norm(K_global @ temperatures - F)}")
    elif solver_method == 'bicgstab':
        temperatures, info = sp.sparse.linalg.bicgstab(K_global, F)
        if info != 0:
            print(f"Residual norm for BiCGStab: {np.linalg.norm(K_global @ temperatures - F)}")
    elif solver_method == 'gmres':
        temperatures, info = sp.sparse.linalg.gmres(K_global, F)
        if info != 0:
            print(f"Residual norm for GMRES: {np.linalg.norm(K_global @ temperatures - F)}")
    elif solver_method == 'minres':
        temperatures, info = sp.sparse.linalg.minres(K_global, F)
        if info != 0:
            print(f"Residual norm for MINRES: {np.linalg.norm(K_global @ temperatures - F)}")
    else:
        raise ValueError(f"Unknown solver method: {solver_method}")

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
