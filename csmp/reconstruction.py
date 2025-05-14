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
            sensing_matrix: Матрица измерений (Phi * Psi).
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
                    max_iter: int = 1000,
                    epsilon: float = 1e-6,
                    sparsity: Optional[int] = None) -> np.ndarray:
        """
        Восстановление разреженного представления сигнала методом MP.

        Args:
            sensing_matrix: Матрица измерений (Phi * Psi).
            compressed_signal: Сжатый сигнал y.
            max_iter: Максимальное количество итераций.
            epsilon: Порог ошибки для остановки алгоритма.
            sparsity: Заданная разреженность K (если None, используется epsilon).

        Returns:
            Восстановленное разреженное представление сигнала.
        """
        M, N = sensing_matrix.shape

        # Инициализация разреженного представления и остатка
        s = np.zeros(N)
        r = np.copy(compressed_signal)

        iterations = 0
        while iterations < max_iter:
            # Вычисление корреляции остатка со столбцами матрицы измерений
            h = np.dot(sensing_matrix.T, r)

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
        Заглушка для метода OMP.

        Args:
            sensing_matrix: Матрица измерений (Phi * Psi).
            compressed_signal: Сжатый сигнал y.
            max_iter: Максимальное количество итераций.
            epsilon: Порог ошибки для остановки алгоритма.
            sparsity: Заданная разреженность K (если None, используется epsilon).

        Returns:
            Восстановленное разреженное представление сигнала.
        """
        raise NotImplementedError("Метод OMP еще не реализован")