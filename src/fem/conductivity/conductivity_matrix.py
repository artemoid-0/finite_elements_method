import numpy as np

def element_conductivity_matrix(k, coords):
    """
    Вычисляет элементную матрицу проводимости для треугольного элемента.

    Parameters:
    k (float): Коэффициент теплопроводности материала.
    coords (np.ndarray): Координаты узлов элемента (3x2).

    Returns:
    np.ndarray: Элементная матрица проводимости (3x3).
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

    # Матрица B для теплопередачи
    B = np.array([
        [y2 - y3, y3 - y1, y1 - y2],
        [x3 - x2, x1 - x3, x2 - x1]
    ]) / (2 * A)

    # Элементная матрица проводимости
    ke = (k * A) * (B.T @ B)

    return ke

def assemble_global_conductivity_matrix(elements, node_coords, k):
    """
    Составляет глобальную матрицу проводимости из элементных матриц.

    Parameters:
    elements (list of list of int): Список элементов, каждый из которых задан как список индексов узлов.
    node_coords (np.ndarray): Координаты узлов (Nx2).
    k (float): Коэффициент теплопроводности материала.

    Returns:
    np.ndarray: Глобальная матрица проводимости (N x N).
    """
    N = len(node_coords)
    K_global = np.zeros((N, N))

    for element in elements:
        coords = node_coords[element]
        ke = element_conductivity_matrix(k, coords)

        for i in range(3):
            for j in range(3):
                K_global[element[i], element[j]] += ke[i, j]

    return K_global

if __name__ == '__main__':
    # Пример использования функций
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elements = [[0, 1, 2], [0, 2, 3]]
    k = 1.0
    K_global = assemble_global_conductivity_matrix(elements, node_coords, k)
    print(K_global)
