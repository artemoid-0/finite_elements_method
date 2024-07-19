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

if __name__ == '__main__':
    # Пример использования функции
    coords = np.array([[0, 0], [1, 0], [0, 1]])
    k = 1.0
    ke = element_conductivity_matrix(k, coords)
    print(ke)
