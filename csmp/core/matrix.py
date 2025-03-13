import numpy as np


def measurement_matrix(n: int, m: int) -> np.ndarray:
    """
    Генерация измерительной матрицы с размерами M x N.

    :param n: Размерность исходного сигнала.
    :param m: Количество измерений (m < n).
    :return: Матрица измерений.
    """
    return np.random.randn(m, n) / np.sqrt(m)