import numpy as np


def basic_signal(size, sparsity):
    """
    Генерация случайного разреженного сигнала для внутренних тестов.

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
