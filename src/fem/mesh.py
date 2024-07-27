import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt


def create_rectangular_mesh(x_min, x_max, y_min, y_max, nx, ny):
    """
    Создает прямоугольную сетку.

    Parameters:
    x_min (float): Минимальное значение по оси x.
    x_max (float): Максимальное значение по оси x.
    y_min (float): Минимальное значение по оси y.
    y_max (float): Максимальное значение по оси y.
    nx (int): Количество узлов по оси x.
    ny (int): Количество узлов по оси y.

    Returns:
    tuple: Узлы сетки и элементы.
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


def create_triangular_mesh(x_min, x_max, y_min, y_max, num_points):
    """
    Создает треугольную сетку с использованием Delaunay триангуляции.

    Parameters:
    x_min (float): Минимальное значение по оси x.
    x_max (float): Максимальное значение по оси x.
    y_min (float): Минимальное значение по оси y.
    y_max (float): Максимальное значение по оси y.
    num_points (int): Количество случайных точек в области.

    Returns:
    tuple: Узлы сетки и элементы.
    """
    points = np.random.rand(num_points, 2)
    points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
    points[:, 1] = points[:, 1] * (y_max - y_min) + y_min
    tri = scipy.spatial.Delaunay(points)
    nodes = points
    elements = tri.simplices

    return nodes, elements


def create_adaptive_mesh(x_min, x_max, y_min, y_max, initial_num_points, refinement_criteria):
    """
    Создает адаптивную треугольную сетку с использованием Delaunay триангуляции и критерия уточнения.

    Parameters:
    x_min (float): Минимальное значение по оси x.
    x_max (float): Максимальное значение по оси x.
    y_min (float): Минимальное значение по оси y.
    y_max (float): Максимальное значение по оси y.
    initial_num_points (int): Начальное количество случайных точек в области.
    refinement_criteria (callable): Функция для определения необходимости уточнения элемента.

    Returns:
    tuple: Узлы сетки и элементы.
    """
    points = np.random.rand(initial_num_points, 2)
    points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
    points[:, 1] = points[:, 1] * (y_max - y_min) + y_min
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
    while len(points) < initial_num_points * 2:  # Условие для завершения уточнения
        points, elements = refine(points, elements)

    return points, elements


def calculate_element_areas(nodes, elements):
    """
    Вычисляет площади элементов.

    Parameters:
    nodes (np.ndarray): Координаты узлов.
    elements (np.ndarray): Элементы сетки.

    Returns:
    np.ndarray: Площади элементов.
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