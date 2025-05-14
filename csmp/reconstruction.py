from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ReconstructionAlgorithm(ABC):
    """
    Абстрактный класс для алгоритмов восстановления сигнала.
    """

    @abstractmethod
    def reconstruct(self,
                    sensing_matrix: np.ndarray,
                    compressed_signal: np.ndarray,
                    **kwargs) -> np.ndarray:
        """
        Восстановление разреженного представления сигнала.

        Args:
            sensing_matrix: Матрица базиса.
            compressed_signal: Сжатый сигнал y.
            **kwargs: Дополнительные параметры.

        Returns:
            Восстановленное разреженное представление сигнала.
        """
        pass


class MP(ReconstructionAlgorithm):
    """
    Реализация алгоритма Matching Pursuit для восстановления сигнала.
    """

    def reconstruct(self,
                    sensing_matrix: np.ndarray,
                    compressed_signal: np.ndarray,
                    max_iter: int = 100,
                    epsilon: float = 1e-6,
                    sparsity: Optional[int] = None) -> np.ndarray:
        """
        Восстановление разреженного представления сигнала методом MP.

        Args:
            sensing_matrix: Матрица базиса.
            compressed_signal: Сжатый сигнал y.
            max_iter: Максимальное количество итераций.
            epsilon: Порог ошибки для остановки алгоритма.
            sparsity: Заданная разреженность K (если None, используется epsilon).

        Returns:
            Восстановленное разреженное представление сигнала.
        """
        M, N = sensing_matrix.shape

        # Проверяем, содержит ли матрица базиса комплексные числа
        is_complex = np.iscomplexobj(sensing_matrix)

        # Инициализация разреженного представления
        s = np.zeros(N, dtype=complex if is_complex else float)

        # Инициализация остатка с правильным типом данных
        r = np.copy(compressed_signal).astype(complex if is_complex else float)

        iterations = 0
        while iterations < max_iter:
            # Вычисление корреляции остатка со столбцами матрицы базиса
            h = np.dot(sensing_matrix.conj().T if is_complex else sensing_matrix.T, r)

            # Нахождение индекса с максимальной корреляцией
            k = np.argmax(np.abs(h))

            # Обновление коэффициента разложения
            s[k] += h[k]

            # Обновление остатка
            r -= h[k] * sensing_matrix[:, k]

            # Проверка критериев остановки
            iterations += 1

            # Остановка по разреженности
            if sparsity is not None and np.count_nonzero(s) >= sparsity:
                break

            # Остановка по ошибке
            if np.linalg.norm(r) < epsilon:
                break

        return s


class OMP(ReconstructionAlgorithm):
    """
    Реализация для алгоритма Orthogonal Matching Pursuit.
    """

    def reconstruct(self,
                    sensing_matrix: np.ndarray,
                    compressed_signal: np.ndarray,
                    max_iter: int = 100,
                    epsilon: float = 1e-6,
                    sparsity: Optional[int] = None) -> np.ndarray:
        """
        Восстановление разреженного представления сигнала методом OMP.

        Args:
            sensing_matrix: Матрица базиса.
            compressed_signal: Сжатый сигнал y.
            max_iter: Максимальное количество итераций.
            epsilon: Порог ошибки для остановки алгоритма.
            sparsity: Заданная разреженность K (если None, используется epsilon).

        Returns:
            Восстановленное разреженное представление сигнала.
        """
        M, N = sensing_matrix.shape

        # Проверяем, содержит ли матрица базиса комплексные числа
        is_complex = np.iscomplexobj(sensing_matrix)

        # Инициализация разреженного представления
        s = np.zeros(N, dtype=complex if is_complex else float)

        # Инициализация остатка с правильным типом данных
        r = np.copy(compressed_signal).astype(complex if is_complex else float)

        # Множество индексов
        selected_indices = []

        iterations = 0
        while iterations < max_iter:
            # Вычисление корреляции остатка со столбцами матрицы базиса
            h = np.dot(sensing_matrix.conj().T if is_complex else sensing_matrix.T, r)

            # Нахождение индекса с максимальной корреляцией
            k = np.argmax(np.abs(h))

            # Добавление индекса в множество выбранных
            if k not in selected_indices:
                selected_indices.append(k)

            # Решение задачи наименьших квадратов для текущего набора выбранных столбцов
            selected_columns = sensing_matrix[:, selected_indices]

            # Вычисление псевдообратной матрицы и решение МНК
            # y = Φ * s_selected, хотим найти s_selected
            s_selected, _, _, _ = np.linalg.lstsq(selected_columns, compressed_signal, rcond=None)

            # Обновление остатка
            r = compressed_signal - np.dot(selected_columns, s_selected)

            # Проверка критериев остановки
            iterations += 1

            # Остановка по разреженности
            if sparsity is not None and len(selected_indices) >= sparsity:
                break

            # Остановка по ошибке
            if np.linalg.norm(r) < epsilon:
                break

        # Формирование итогового разреженного вектора
        for i, idx in enumerate(selected_indices):
            s[idx] = s_selected[i]

        return s