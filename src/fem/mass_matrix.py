import numpy as np


def element_mass_matrix(rho, coords):
    """
    Вычисляет элементную матрицу массы для треугольного элемента.

    Parameters:
    rho (float): Плотность материала.
    coords (np.ndarray): Координаты узлов элемента (3x2).

    Returns:
    np.ndarray: Элементная матрица массы (3x3).
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    # Вычисление площади элемента
    A = 0.5 * np.abs(np.linalg.det(np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ])))

    # Матрица массы для треугольного элемента
    me = (rho * A / 12) * np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ])

    return me


def assemble_global_mass_matrix(elements, node_coords, rho):
    """
    Составляет глобальную матрицу массы из элементных матриц.

    Parameters:
    elements (list of list of int): Список элементов, каждый из которых задан как список индексов узлов.
    node_coords (np.ndarray): Координаты узлов (Nx2).
    rho (float): Плотность материала.

    Returns:
    np.ndarray: Глобальная матрица массы (2N x 2N).
    """
    N = len(node_coords)
    M_global = np.zeros((2 * N, 2 * N))

    for element in elements:
        coords = node_coords[element]
        me = element_mass_matrix(rho, coords)

        for i in range(3):
            for j in range(3):
                M_global[2 * element[i]:2 * element[i] + 2, 2 * element[j]:2 * element[j] + 2] += me[i, j]

    return M_global


if __name__ == '__main__':
    # Пример использования функции
    coords = np.array([[0, 0], [1, 0], [0, 1]])
    rho = 1.0
    elements = [[0, 1, 2]]
    node_coords = np.array([[0, 0], [1, 0], [0, 1]])
    M_global = assemble_global_mass_matrix(elements, node_coords, rho)
    print(M_global)
