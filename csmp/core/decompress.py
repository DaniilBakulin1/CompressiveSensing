import numpy as np


def match_pursuit(
    data: np.ndarray,
    matrix: np.ndarray,
    threshold: float = 0.01,
    max_iterations: int = 1000
) -> np.ndarray:
    """
    Реализация алгоритма Matching Pursuit для восстановления сигнала.

    :param data: Сжатый сигнал.
    :param matrix: Матрица измерений.
    :param threshold: Порог ошибки для остановки.
    :param max_iterations: Максимальное количество итераций.
    :return: Коэффициенты разложения сигнала.
    """
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


def orthogonal_match_pursuit(
    data: np.ndarray,
    matrix: np.ndarray,
    threshold: float = 0.01,
    max_iterations: int = 1000
) -> np.ndarray:
    """
    Реализация алгоритма Orthogonal Matching Pursuit для восстановления сигнала.

    :param data: Сжатый сигнал.
    :param matrix: Матрица измерений.
    :param threshold: Порог ошибки для остановки.
    :param max_iterations: Максимальное количество итераций.
    :return: Коэффициенты разложения сигнала.
    """
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