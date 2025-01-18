import numpy as np


def compress_signal(signal, measurement_matrix):
    """
    Сжатие сигнала с помощью измерительной матрицы.

    Args:
        signal (np.ndarray): Исходный сигнал.
        measurement_matrix (np.ndarray): Измерительная матрица.

    Returns:
        np.ndarray: Сжатый сигнал.
    """
    return measurement_matrix @ signal


def match_pursuit(data, dictionary, threshold=0.1, max_iterations=100, epsilon=None):
    """
    Реализация алгоритма Matching Pursuit для восстановления сигнала с опцией разреживания.

    Args:
        data (np.ndarray): Сжатый сигнал.
        dictionary (np.ndarray): Словарь атомов (матрица, где каждый столбец — атом).
        threshold (float): Порог ошибки для остановки.
        max_iterations (int): Максимальное количество итераций.
        epsilon (float, optional): Пороговое значение для зануления элементов. Если None, не используется.

    Returns:
        np.ndarray: Коэффициенты разложения сигнала по словарю.
        list: Индексы выбранных атомов.
    """
    if not isinstance(data, np.ndarray) or not isinstance(dictionary, np.ndarray):
        raise TypeError("Ожидались входные данные типа numpy.ndarray")
    if data.ndim != 1:
        raise ValueError("data должен быть одномерным массивом")
    if dictionary.shape[0] != data.shape[0]:
        raise ValueError("Число строк в словаре должно совпадать с размером данных")

    # Применение порога epsilon для разреживания данных
    if epsilon is not None:
        data = data.copy()
        data[np.abs(data) < epsilon] = 0

    # Инициализация
    residual = data.copy()  # Остаток
    coefficients = np.zeros(dictionary.shape[1])  # Коэффициенты
    selected_atoms = []  # Индексы выбранных атомов

    for _ in range(max_iterations):
        # Выбор атома, наиболее коррелирующего с остатком
        correlations = dictionary.T @ residual
        best_atom = np.argmax(np.abs(correlations))
        selected_atoms.append(best_atom)

        # Обновление коэффициентов
        atom = dictionary[:, best_atom]
        atom_coeff = np.dot(atom, residual) / np.dot(atom, atom)
        coefficients[best_atom] += atom_coeff

        # Обновление остатка
        residual -= atom_coeff * atom

        # Условие остановки
        if np.linalg.norm(residual) < threshold:
            break

    return coefficients, selected_atoms


def sparse_reconstruction(coefficients, dictionary):
    """
    Восстановление сигнала из разреженного представления.

    Args:
        coefficients (np.ndarray): Коэффициенты разложения.
        dictionary (np.ndarray): Словарь атомов.

    Returns:
        np.ndarray: Восстановленный сигнал.
    """
    if not isinstance(coefficients, np.ndarray) or not isinstance(dictionary, np.ndarray):
        raise TypeError("Ожидались входные данные типа numpy.ndarray")

    if coefficients.shape[0] != dictionary.shape[1]:
        raise ValueError("Размер коэффициентов должен совпадать с числом столбцов словаря")

    return dictionary @ coefficients