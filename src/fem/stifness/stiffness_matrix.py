import numpy as np


def element_stiffness_matrix(E, nu, coords):
    """
    Вычисляет элементную матрицу жесткости для треугольного элемента.

    Parameters:
    E (float): Модуль Юнга материала.
    nu (float): Коэффициент Пуассона материала.
    coords (np.ndarray): Координаты узлов элемента (3x2).

    Returns:
    np.ndarray: Элементная матрица жесткости (6x6).
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    # Вычисление площади элемента
    A = 0.5 * np.linalg.det(np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ]))

    # Матрица B
    B = np.array([
        [y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
        [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
        [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]
    ]) / (2 * A)

    # Матрица D (плоское напряженно-деформированное состояние)
    D = (E / (1 - nu ** 2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])

    # Элементная матрица жесткости
    ke = A * np.dot(np.dot(B.T, D), B)

    return ke


def assemble_global_stiffness_matrix(elements, node_coords, E, nu):
    """
    Составляет глобальную матрицу жесткости из элементных матриц.

    Parameters:
    elements (list of list of int): Список элементов, каждый из которых задан как список индексов узлов.
    node_coords (np.ndarray): Координаты узлов (Nx2).
    E (float): Модуль Юнга материала.
    nu (float): Коэффициент Пуассона материала.

    Returns:
    np.ndarray: Глобальная матрица жесткости (2N x 2N).
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
    # Пример использования функции
    E = 210e9  # Модуль Юнга для стали, в Па
    nu = 0.3  # Коэффициент Пуассона
    elements = [[0, 1, 2]]  # Один элемент, соединяющий узлы 0, 1 и 2
    node_coords = np.array([[0, 0], [1, 0], [0, 1]])  # Координаты узлов

    K_global = assemble_global_stiffness_matrix(elements, node_coords, E, nu)
    print(K_global)
