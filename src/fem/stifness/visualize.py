import numpy as np
import matplotlib.pyplot as plt
from src.fem.mesh import create_regular_triangular_mesh_in_rectangle, plot_mesh, plot_elements
from stiffness_matrix import assemble_global_stiffness_matrix
from boundary_conditions import apply_boundary_conditions
from solve_fem import solve_fem

def visualize_results(node_coords, elements, displacements, scale=1.0):
    """
    Visualizes the results of the FEM.

    Parameters:
    node_coords (np.ndarray): Node coordinates (Nx2).
    elements (list of list of int): List of elements, each specified as a list of node indices.
    displacements (np.ndarray): Displacement vector (2N).
    scale (float): Scale for displaying deformations.
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

    # Example for a rectangular grid
    node_coords, elements = create_regular_triangular_mesh_in_rectangle(0, 1, 0, 1, 5, 5)

    # Visualization of the original mesh
    plot_mesh(node_coords, elements, "Rectangular Mesh")
    plot_elements(node_coords, elements, "Rectangular Mesh Elements")

    # Material parameters
    E = 210e9  # Young's modulus
    nu = 0.3  # Poisson's ratio

    # Application of forces and boundary conditions
    fixed_nodes = [0, 5]
    forces = np.zeros(2 * len(node_coords))
    forces[12] = -5e9  # Example of external forces

    # Solving the problem using the finite element method
    displacements = solve_fem(node_coords, elements, E, nu, fixed_nodes, forces)

    print(displacements)

    # Visualization of results
    visualize_results(node_coords, elements, displacements, scale=1.0)


if __name__ == "__main__":
    main()
