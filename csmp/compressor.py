from typing import Callable, Optional

import numpy as np

from csmp import _compress
from csmp import _measurement_matrix
from csmp import _match_pursuit
from csmp import _orthogonal_match_pursuit

CompressFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
MatrixFunc = Callable[[int, int], np.ndarray]
RecoveryFunc = Callable[[np.ndarray, np.ndarray, float, int], np.ndarray]


class Compressor:
    def __init__(
            self,
            compress_func: Optional[CompressFunc] = None,
            matrix_func: Optional[MatrixFunc] = None,
            recovery_func: Optional[RecoveryFunc] = None,
    ):
        self.compress_func = compress_func or _compress
        self.matrix_func = matrix_func or _measurement_matrix
        self.recovery_func = recovery_func or _orthogonal_match_pursuit
        self._matrix: Optional[np.ndarray] = None

    def compress(
            self,
            data: np.ndarray,
            compress: int,
            epsilon: float = None
    ) -> np.ndarray:
        """
        Сжатие сигнала

        :param data: Исходный сигнал.
        :param compress: Количество измерений.
        :param epsilon: Пороговое значение для зануления элементов. Если None, не используется.
        :return: Сжатый сигнал.
        """
        # Применение порога epsilon для разреживания данных
        if epsilon is not None:
            if epsilon < 0:
                raise ValueError("epsilon cannot be negative")

            data = data.copy()
            data[np.abs(data) < epsilon] = 0

        self._matrix = self.matrix_func(len(data), compress)
        compressed_data = self.compress_func(data, self._matrix)
        return compressed_data

    def decompress(
            self,
            compressed_data: np.ndarray,
            threshold: float = 0.001,
            max_iterations: int = 10000
    ) -> np.ndarray:
        """
        Восстановление сигнала

        :param compressed_data: Сжатый сигнал
        :param threshold: Порог ошибки для остановки.
        :param max_iterations: Максимальное количество итераций.
        :return: Восстановленный сигнал
        """
        if self._matrix is None:
            raise RuntimeError("Compressor.decompress called before compress")

        recovered_data = self.recovery_func(compressed_data, self._matrix, threshold, max_iterations)
        return recovered_data

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
