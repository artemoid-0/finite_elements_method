import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from datetime import datetime
import csv


# Генерация полной системы
def generate_large_system_scipy(size, seed):
    np.random.seed(seed)
    A = np.random.rand(size, size)
    B = np.random.rand(size)
    return A, B


# Генерация разреженной системы
def generate_large_sparse_system_scipy(size, density, seed):
    np.random.seed(seed)
    A = sps.random(size, size, density=density, format='csr')
    B = np.random.rand(size)
    return A, B


# Тестирование различных функций SciPy
def test_scipy_spsolve(size, density, seed):
    A, B = generate_large_sparse_system_scipy(size, density, seed)
    start_time = datetime.now()
    X = spla.spsolve(A, B)
    end_time = datetime.now()
    return end_time - start_time, "spsolve"


def test_scipy_lsqr(size, density, seed):
    A, B = generate_large_sparse_system_scipy(size, density, seed)
    start_time = datetime.now()
    X = spla.lsqr(A, B)
    end_time = datetime.now()
    return end_time - start_time, "lsqr"


def test_scipy_cg(size, density, seed):
    A, B = generate_large_sparse_system_scipy(size, density, seed)
    start_time = datetime.now()
    X, _ = spla.cg(A, B)
    end_time = datetime.now()
    return end_time - start_time, "cg"


def test_scipy_bicg(size, density, seed):
    A, B = generate_large_sparse_system_scipy(size, density, seed)
    start_time = datetime.now()
    X, _ = spla.bicg(A, B)
    end_time = datetime.now()
    return end_time - start_time, "bicg"


def test_scipy_bicgstab(size, density, seed):
    A, B = generate_large_sparse_system_scipy(size, density, seed)
    start_time = datetime.now()
    X, _ = spla.bicgstab(A, B)
    end_time = datetime.now()
    return end_time - start_time, "bicgstab"


def test_scipy_gmres(size, density, seed):
    A, B = generate_large_sparse_system_scipy(size, density, seed)
    start_time = datetime.now()
    X, _ = spla.gmres(A, B)
    end_time = datetime.now()
    return end_time - start_time, "gmres"


def test_scipy_minres(size, density, seed):
    A, B = generate_large_sparse_system_scipy(size, density, seed)
    start_time = datetime.now()
    X, _ = spla.minres(A, B)
    end_time = datetime.now()
    return end_time - start_time, "minres"


def test_scipy_solve(size, seed):
    A, B = generate_large_system_scipy(size, seed)
    start_time = datetime.now()
    X = sp.linalg.solve(A, B)
    end_time = datetime.now()
    return end_time - start_time, "solve"


# Основной блок выполнения
if __name__ == "__main__":
    size = 30000  # Задайте нужный размер
    density = 0.01  # Плотность разреженной матрицы
    seed = 1  # Текущий seed для воспроизводимости

    # Определяем функции, которые требуют аргумент density
    sparse_test_functions = [
        test_scipy_spsolve,
        test_scipy_lsqr,
        test_scipy_cg,
        test_scipy_bicg,
        test_scipy_bicgstab,
        # test_scipy_gmres,
        test_scipy_minres
    ]

    dense_test_functions = [
        test_scipy_solve
    ]

    # Запуск тестов и запись результатов
    with open("results/scipy_test_results.csv", "a", newline='') as csvfile:
        fieldnames = ['Matrix Size', 'Seed', 'Function', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Запись заголовка только если файл пуст
        if csvfile.tell() == 0:
            writer.writeheader()

        # Запуск тестов для разреженных матриц
        for test_func in sparse_test_functions:
            time_sp, method = test_func(size, density, seed)
            writer.writerow({
                'Matrix Size': size,
                'Seed': seed,
                'Function': method,
                'Time': time_sp
            })
            print(f"{method} time: {time_sp}")

        # Запуск тестов для плотных матриц
        for test_func in dense_test_functions:
            time_sp, method = test_func(size, seed)
            writer.writerow({
                'Matrix Size': size,
                'Seed': seed,
                'Function': method,
                'Time': time_sp
            })
            print(f"{method} time: {time_sp}")
