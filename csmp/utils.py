import numpy as np


def generate_basic_signal(size, sparsity):
    """
    Генерация случайного разреженного сигнала.

    Args:
        size (int): Размер исходного сигнала.
        sparsity (int): Количество ненулевых элементов.

    Returns:
        np.ndarray: Разреженный сигнал.
    """
    signal = np.zeros(size)
    non_zero_indices = np.random.choice(size, sparsity, replace=False)
    signal[non_zero_indices] = np.random.randn(sparsity)
    return signal


def generate_measurement_matrix(N, M):
    """
    Генерация измерительной матрицы с размерами M x N.

    Args:
        N (int): Размерность исходного сигнала.
        M (int): Количество измерений (M < N).

    Returns:
        np.ndarray: Измерительная матрица.
    """
    return np.random.randn(M, N) / np.sqrt(M)  # Нормированная случайная матрица