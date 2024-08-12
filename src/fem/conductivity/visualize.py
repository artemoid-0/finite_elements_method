from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from src.fem.mesh import create_regular_triangular_mesh_in_rectangle, create_random_triangular_mesh_in_rectangle, create_random_triangular_mesh_in_polygon, \
    plot_mesh, plot_elements, create_adaptive_triangular_mesh_in_polygon, refinement_criteria
from src.fem.conductivity.solve_fem import solve_fem_heat_transfer


import matplotlib.pyplot as plt
import numpy as np

def visualize_heat_transfer(node_coords, elements, temperatures, title="None"):
    """
    Visualizes the results of a heat transfer problem.

    Parameters:
    node_coords (np.ndarray): Node coordinates (Nx2).
    elements (list of list of int): List of elements, each specified as a list of node indices.
    temperatures (np.ndarray): Vector of temperatures (N).
    title (str): Title for the graph. Default is "Temperature Distribution".
    """
    plt.tricontourf(node_coords[:, 0], node_coords[:, 1], np.array(elements), temperatures, levels=14, cmap='coolwarm')
    plt.colorbar()
    plt.title(title)  # Use the passed title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def main():
    np.random.seed(0)

    # Define the vertices of the polygon (for simplicity, we use a square area)
    polygon_vertices = np.array([
        [0, 0], [3, 0], [3, 3], [0, 3]
    ])
    initial_num_points = 10000
    solver_method = 'spsolve'

    start_time = datetime.now()
    node_coords, elements = create_adaptive_triangular_mesh_in_polygon(polygon_vertices, initial_num_points,
                                                                       refinement_criteria)
    print("Initial number of nodes:", initial_num_points)
    print("Total number of nodes:", len(node_coords))

    print("Mesh creation time:", datetime.now() - start_time)
    k = 1.0

    # Setting fixed nodes and their temperatures
    fixed_nodes = list(range(50))
    fixed_temperatures = np.random.uniform(0, 100, len(fixed_nodes))

    # Initialization of heat sources
    heat_sources = np.zeros(len(node_coords))

    # Definition of heat source clusters
    clusters = [
        {"center": [0.5, 0.5], "radius": 0.5, "total_heat": 2000},
        {"center": [2.5, 0.5], "radius": 0.5, "total_heat": 2500},
        {"center": [1.5, 2.5], "radius": 0.5, "total_heat": 3000}
    ]

    # Assign heat to nodes within each cluster
    for cluster in clusters:
        nodes_in_cluster = [i for i, coord in enumerate(node_coords) if
                            np.linalg.norm(coord - cluster["center"]) < cluster["radius"]]
        heat_per_node = cluster["total_heat"] / len(nodes_in_cluster)
        for i in nodes_in_cluster:
            heat_sources[i] = heat_per_node

    temperatures = solve_fem_heat_transfer(node_coords, elements, k, fixed_nodes, fixed_temperatures, heat_sources,
                                               solver_method)

    print("Temperatures in nodes:\n", temperatures)

    start_time = datetime.now()
    visualize_heat_transfer(node_coords, elements, temperatures, title=solver_method)
    print("Heat transfer visualization time:", datetime.now() - start_time)


if __name__ == "__main__":
    main()
