import numpy as np

from .utils import generate_measurement_matrix


def compress(data, M, epsilon=None):
    """
    Сжатие сигнала с помощью измерительной матрицы.

    Args:
        data (np.ndarray): Исходный сигнал.
        M (int): Количество измерений (M < N).
        epsilon (float, optional): Пороговое значение для зануления элементов. Если None, не используется.

    Returns:
        np.ndarray: Сжатый сигнал,
        np.ndarray: Матрица измерений.
    """
    # Применение порога epsilon для разреживания данных
    if epsilon is not None:
        data = data.copy()
        data[np.abs(data) < epsilon] = 0

    measurement_matrix = generate_measurement_matrix(len(data), M)

    return measurement_matrix @ data, measurement_matrix


def match_pursuit(data, matrix, threshold=0.01, max_iterations=1000):
    """
    Реализация алгоритма Matching Pursuit для восстановления сигнала.

    Args:
        data (np.ndarray): Сжатый сигнал.
        matrix (np.ndarray): Матрица измерений.
        threshold (float): Порог ошибки для остановки.
        max_iterations (int): Максимальное количество итераций.

    Returns:
        np.ndarray: Коэффициенты разложения сигнала.
    """
    if not isinstance(data, np.ndarray) or not isinstance(matrix, np.ndarray):
        raise TypeError("Ожидались входные данные типа numpy.ndarray")
    if data.ndim != 1:
        raise ValueError("data должен быть одномерным массивом")
    if matrix.shape[0] != data.shape[0]:
        raise ValueError("Число строк в словаре должно совпадать с размером данных")

    # Инициализация
    residual = data.copy()
    recovered_signal = np.zeros(matrix.shape[1])
    selected_indices = []

    for _ in range(max_iterations):
        correlations = np.dot(matrix.T, residual)
        best_index = np.argmax(np.abs(correlations))
        selected_indices.append(best_index)

        step_size = correlations[best_index] / np.dot(matrix[:, best_index], matrix[:, best_index])
        recovered_signal[best_index] += step_size

        residual -= step_size * matrix[:, best_index]

        # Условие остановки
        if np.linalg.norm(residual) < threshold:
            break

    return recovered_signal


def orthogonal_match_pursuit(data, matrix, threshold=0.01, max_iterations=1000):
    """
    Реализация алгоритма Orthogonal Matching Pursuit для восстановления сигнала.

    Args:
        data (np.ndarray): Сжатый сигнал.
        matrix (np.ndarray): Матрица измерений.
        threshold (float): Порог ошибки для остановки.
        max_iterations (int): Максимальное количество итераций.

    Returns:
        np.ndarray: Коэффициенты разложения сигнала.
    """
    if not isinstance(data, np.ndarray) or not isinstance(matrix, np.ndarray):
        raise TypeError("Ожидались входные данные типа numpy.ndarray")
    if data.ndim != 1:
        raise ValueError("data должен быть одномерным массивом")
    if matrix.shape[0] != data.shape[0]:
        raise ValueError("Число строк в словаре должно совпадать с размером данных")

    # Инициализация
    residual = data.copy()
    indices = []
    recovered_signal = np.zeros(matrix.shape[1])

    for _ in range(max_iterations):
        projections = np.dot(matrix.T, residual)
        best_index = np.argmax(np.abs(projections))

        if best_index in indices:
            continue

        indices.append(best_index)

        selected_columns = matrix[:, indices]
        pseudo_inverse = np.linalg.pinv(selected_columns)
        signal_subset = np.dot(pseudo_inverse, data)
        residual = data - np.dot(selected_columns, signal_subset)

        # Условие остановки
        if np.linalg.norm(residual) < threshold:
            break

    for i, index in enumerate(indices):
        recovered_signal[index] = signal_subset[i]

    return recovered_signal