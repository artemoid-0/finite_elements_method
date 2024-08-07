import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt
from matplotlib.path import Path


def create_regular_triangular_mesh_in_rectangle(x_min, x_max, y_min, y_max, nx, ny):
    """
    Creates a regular triangular mesh in rectangular area.

    Parameters:
    x_min (float): Minimum value along the x-axis.
    x_max (float): Maximum value along the x-axis.
    y_min (float): Minimum value along the y-axis.
    y_max (float): Maximum value along the y-axis.
    nx (int): Number of nodes along the x-axis.
    ny (int): Number of nodes along the y-axis.

    Returns:
    tuple: Grid nodes and elements.
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)
    nodes = np.vstack([xx.ravel(), yy.ravel()]).T
    elements = []

    for i in range(ny - 1):
        for j in range(nx - 1):
            n1 = i * nx + j
            n2 = n1 + 1
            n3 = n1 + nx
            n4 = n3 + 1
            elements.append([n1, n2, n4])
            elements.append([n1, n4, n3])

    return nodes, np.array(elements)


def create_random_triangular_mesh_in_rectangle(x_min, x_max, y_min, y_max, num_points, seed=None):
    """
    Creates a triangular mesh using Delaunay triangulation in a rectangular area.

    Parameters:
    x_min (float): Minimum value along the x-axis.
    x_max (float): Maximum value along the x-axis.
    y_min (float): Minimum value along the y-axis.
    y_max (float): Maximum value along the y-axis.
    num_points (int): Number of random points in the region.
    seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
    tuple: Mesh nodes and elements.
    """
    if seed is not None:
        np.random.seed(seed)

    points = np.random.rand(num_points, 2)
    points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
    points[:, 1] = points[:, 1] * (y_max - y_min) + y_min

    # Creating Delaunay triangulation
    tri = scipy.spatial.Delaunay(points)

    # Mesh nodes are the coordinates of the points
    nodes = points

    # Mesh elements are the indices of the points that form the triangles
    elements = tri.simplices

    return nodes, elements


def create_random_triangular_mesh_in_polygon(vertices, num_points):
    """
    Creates a triangular mesh using Delaunay triangulation in the area bounded by a given polygon

    Parameters:
    vertices (list of tuple of float): The coordinates of the polygon's vertices.
    num_points (int): The number of random points inside the polygon.

    Returns:
    tuple: The mesh nodes and elements.
    """
    # Create a polygon based on vertices
    poly = np.array(vertices)
    poly_path = scipy.spatial.Delaunay(poly)

    # Generate random points inside a given polygon
    points = []
    while len(points) < num_points:
        random_point = np.random.rand(1, 2)
        random_point[:, 0] = random_point[:, 0] * (poly[:, 0].max() - poly[:, 0].min()) + poly[:, 0].min()
        random_point[:, 1] = random_point[:, 1] * (poly[:, 1].max() - poly[:, 1].min()) + poly[:, 1].min()
        if poly_path.find_simplex(random_point) >= 0:
            points.append(random_point[0])

    points = np.array(points)

    # Add polygon vertices to points
    points = np.vstack([points, poly])

    # Perform triangulation
    tri = scipy.spatial.Delaunay(points)
    nodes = points
    elements = tri.simplices

    return nodes, elements


def create_adaptive_triangular_mesh_in_polygon(polygon_vertices, initial_num_points, refinement_criteria):
    """
    Creates an adaptive triangular mesh from a given polygon using Delaunay triangulation and a refinement criterion.

    Parameters:
    polygon_vertices (np.ndarray): Polygon vertex coordinates (Mx2).
    initial_num_points (int): Initial number of random points in the region.
    refinement_criteria (callable): Function to determine whether an element needs to be refined.

    Returns:
    tuple: Grid nodes and elements.
    """
    # Generate initial random points inside the polygon
    points = np.random.rand(initial_num_points, 2)
    min_x, min_y = polygon_vertices.min(axis=0)
    max_x, max_y = polygon_vertices.max(axis=0)
    points[:, 0] = points[:, 0] * (max_x - min_x) + min_x
    points[:, 1] = points[:, 1] * (max_y - min_y) + min_y

    # We leave only those points that are inside the polygon
    path = Path(polygon_vertices)
    points = points[path.contains_points(points)]

    # Adding polygon vertices to points
    points = np.vstack((points, polygon_vertices))

    # Performing Delaunay triangulation
    tri = scipy.spatial.Delaunay(points)

    def refine(points, simplices):
        new_points = []
        for simplex in simplices:
            vertices = points[simplex]
            centroid = vertices.mean(axis=0)
            if refinement_criteria(vertices):
                new_points.append(centroid)
        if new_points:
            points = np.vstack([points, new_points])
            tri = scipy.spatial.Delaunay(points)
            return points, tri.simplices
        else:
            return points, simplices

    points, elements = refine(points, tri.simplices)
    while len(points) < initial_num_points * 2:
        points, elements = refine(points, elements)

    return points, elements


# def refinement_criteria(vertices):
#     """
#     Takes the coordinates of the vertices of an element and returns True if the element needs to be refined
#     This criterion is based on the length of the side of the element
#     """
#     max_edge_length = 0.2
#     for i in range(len(vertices)):
#         for j in range(i+1, len(vertices)):
#             edge_length = np.linalg.norm(vertices[i] - vertices[j])
#             if edge_length > max_edge_length:
#                 return True
#     return False


def refinement_criteria(vertices):
    return np.linalg.norm(vertices[0] - vertices[1]) > 0.01  # Increased refinement criteria


def calculate_element_areas(nodes, elements):
    """
    Calculates the areas of elements.

    Parameters:
    nodes (np.ndarray): Node coordinates.
    elements (np.ndarray): Grid elements.

    Returns:
    np.ndarray: Element areas.
    """
    areas = []
    for element in elements:
        x1, y1 = nodes[element[0]]
        x2, y2 = nodes[element[1]]
        x3, y3 = nodes[element[2]]
        area = 0.5 * np.abs(np.linalg.det(np.array([
            [1, x1, y1],
            [1, x2, y2],
            [1, x3, y3]
        ])))
        areas.append(area)
    return np.array(areas)


def plot_mesh(nodes, elements, title="Mesh Visualization"):
    """
    Visualizes a mesh and numbers the nodes.

    Parameters:
    nodes (np.ndarray): Node coordinates.
    elements (np.ndarray): Grid elements.
    title (str): Plot title.
    """
    plt.figure(figsize=(10, 10))
    for element in elements:
        polygon = plt.Polygon(nodes[element], edgecolor='k', facecolor='none')
        plt.gca().add_patch(polygon)

        # Adding element node numbers
        for node_index in element:
            node = nodes[node_index]
            plt.text(node[0] + 0.01, node[1] + 0.01, str(node_index), fontsize=12,
                     ha='center', va='center', color='blue',
                     bbox=dict(facecolor='white', edgecolor='none', pad=1))

    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Узлы
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot_elements(nodes, elements, title="Element Visualization"):
    """
    Visualizes mesh elements by connecting points into elements.

    Parameters:
    nodes (np.ndarray): Node coordinates.
    elements (np.ndarray): Grid elements.
    title (str): Plot title.
    """
    plt.figure(figsize=(10, 10))

    for i, element in enumerate(elements):
        # Uniting points into elements
        for j in range(len(element)):
            start_node = nodes[element[j]]
            end_node = nodes[element[(j + 1) % len(element)]]
            plt.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'k-')

        # Find the center of the element and add the ordinal number
        element_center = np.mean(nodes[element], axis=0)
        plt.text(element_center[0], element_center[1], str(i), color='blue', fontsize=12, ha='center', va='center')

    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Nodes
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':

    np.random.seed(4)

    # Coordinates of the vertices of a pentagon (non-standard shape)
    vertices = np.array([(0, 0), (2, 1), (1, 3), (0.5, 2), (1, 1)])

    initial_num_points = 15

    nodes, elements = create_adaptive_triangular_mesh_in_polygon(vertices, initial_num_points,
                                                                 refinement_criteria)

    plot_mesh(nodes, elements, title="Adaptive Triangular Mesh in Polygon")
