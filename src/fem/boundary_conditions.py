import numpy as np

def apply_boundary_conditions(K, F, fixed_nodes):
    """
    Применяет граничные условия к системе уравнений.

    Parameters:
    K (np.ndarray): Глобальная матрица жесткости.
    F (np.ndarray): Вектор правых частей.
    fixed_nodes (list of int): Список индексов фиксированных узлов.

    Returns:
    np.ndarray, np.ndarray: Измененные матрица жесткости и вектор правых частей.
    """
    for node in fixed_nodes:
        dof = [2 * node, 2 * node + 1]
        for d in dof:
            K[d, :] = 0
            K[:, d] = 0
            K[d, d] = 1
            F[d] = 0
    return K, F
