import matplotlib.pyplot as plt
import numpy as np


def plot_mesh(nodes, elements, title="Mesh Visualization"):
    """
    Визуализирует сетку.

    Parameters:
    nodes (np.ndarray): Координаты узлов.
    elements (np.ndarray): Элементы сетки.
    title (str): Заголовок графика.
    """
    plt.figure(figsize=(10, 10))
    for element in elements:
        polygon = plt.Polygon(nodes[element], edgecolor='k', facecolor='none')
        plt.gca().add_patch(polygon)
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Узлы
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot_elements(nodes, elements, title="Element Visualization"):
    """
    Визуализирует элементы сетки, соединяя точки в элементы.

    Parameters:
    nodes (np.ndarray): Координаты узлов.
    elements (np.ndarray): Элементы сетки.
    title (str): Заголовок графика.
    """
    plt.figure(figsize=(10, 10))
    for element in elements:
        for i in range(len(element)):
            start_node = nodes[element[i]]
            end_node = nodes[element[(i + 1) % len(element)]]
            plt.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'k-')
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Узлы
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def visualize_results(node_coords, elements, displacements, scale=1.0):
    """
    Визуализирует результаты МКЭ.

    Parameters:
    node_coords (np.ndarray): Координаты узлов (Nx2).
    elements (list of list of int): Список элементов, каждый из которых задан как список индексов узлов.
    displacements (np.ndarray): Вектор перемещений (2N).
    scale (float): Масштаб для отображения деформаций.
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
    from fem.mesh import create_rectangular_mesh, create_triangular_mesh, create_adaptive_mesh

    np.random.seed(0)

    # Пример для прямоугольной сетки
    nodes, elements = create_rectangular_mesh(0, 1, 0, 1, 5, 5)
    plot_mesh(nodes, elements, "Rectangular Mesh")
    plot_elements(nodes, elements, "Rectangular Mesh Elements")

    # Пример для треугольной сетки
    nodes, elements = create_triangular_mesh(0, 1, 0, 1, 30)
    plot_mesh(nodes, elements, "Triangular Mesh")
    plot_elements(nodes, elements, "Triangular Mesh Elements")

    # Пример для адаптивной треугольной сетки
    def refinement_criteria(vertices):
        return np.linalg.norm(vertices[1] - vertices[0]) > 0.2

    nodes, elements = create_adaptive_mesh(0, 1, 0, 1, 10, refinement_criteria)
    plot_mesh(nodes, elements, "Adaptive Triangular Mesh")
    plot_elements(nodes, elements, "Adaptive Triangular Mesh Elements")


if __name__ == "__main__":
    # main()
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elements = [[0, 1, 2], [0, 2, 3]]
    displacements = np.array([0, 0, 0, 0, 0.01, -0.02, 0, 0])  # Пример перемещений

    visualize_results(node_coords, elements, displacements, scale=1.0)
