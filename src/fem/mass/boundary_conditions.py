import numpy as np

def apply_boundary_conditions_mass(M, F, fixed_nodes):
    """
    Применяет граничные условия к системе уравнений для задачи с матрицей массы.

    Parameters:
    M (np.ndarray): Глобальная матрица массы.
    F (np.ndarray): Вектор внешних сил.
    fixed_nodes (list of int): Список индексов фиксированных узлов.

    Returns:
    np.ndarray, np.ndarray: Измененные матрица массы и вектор внешних сил.
    """
    for node in fixed_nodes:
        for i in range(2):
            row = 2 * node + i
            M[row, :] = 0
            M[:, row] = 0
            M[row, row] = 1
            F[row] = 0

    print("Глобальная матрица массы после применения граничных условий:")
    print(M)
    print("Вектор внешних сил после применения граничных условий:")
    print(F)

    return M, F

