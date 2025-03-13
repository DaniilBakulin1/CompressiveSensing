import numpy as np


def compressive_sensing(data: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Сжатие сигнала с помощью измерительной матрицы.

    :param data: Исходный сигнал.
    :param matrix: Матрица измерений.

    :return: Сжатый сигнал.
    """
    return matrix @ data