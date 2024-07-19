import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


def create_mesh(nx, ny, length, height):
    """
    Создает сетку узлов в двумерной области.

    Parameters:
    nx (int): Количество узлов по оси x.
    ny (int): Количество узлов по оси y.
    length (float): Длина области по оси x.
    height (float): Высота области по оси y.

    Returns:
    cp.ndarray: Массив координат узлов (xv, yv).
    """
    dx = length / (nx - 1)
    dy = height / (ny - 1)
    x = cp.linspace(0, length, nx)
    y = cp.linspace(0, height, ny)
    xv, yv = cp.meshgrid(x, y)
    return xv, yv


def visualize_mesh(xv, yv):
    """
    Визуализирует сетку узлов с соединением точек в элементы.

    Parameters:
    xv (cp.ndarray): Массив координат узлов по оси x.
    yv (cp.ndarray): Массив координат узлов по оси y.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(xv, yv, marker='o', color='b', label='Nodes')

    nx, ny = xv.shape

    for i in range(nx - 1):
        for j in range(ny - 1):
            # Координаты четырех узлов элемента
            x_coords = [xv[i, j], xv[i + 1, j], xv[i + 1, j + 1], xv[i, j + 1], xv[i, j]]
            y_coords = [yv[i, j], yv[i + 1, j], yv[i + 1, j + 1], yv[i, j + 1], yv[i, j]]

            # Рисуем линии, соединяющие узлы элемента
            plt.plot(x_coords, y_coords, 'r-')

    plt.title('Mesh Grid Visualization with Elements')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()


# Пример использования функций
if __name__ == "__main__":
    length = 1.0
    height = 1.0
    nx = 10
    ny = 10

    # Создание сетки
    xv, yv = create_mesh(nx, ny, length, height)
    xv = cp.asnumpy(xv)
    yv = cp.asnumpy(yv)

    # Визуализация сетки с элементами
    visualize_mesh(xv, yv)
