import numpy as np

from typing import Callable, Optional, TypeVar
from csmp.core.compress import compressive_sensing
from csmp.core.decompress import match_pursuit
from csmp.core.decompress import orthogonal_match_pursuit
from csmp.core.matrix import measurement_matrix

CompressFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
MatrixFunc = Callable[[int, int], np.ndarray]
RecoveryFunc = Callable[[np.ndarray, np.ndarray, float, int], np.ndarray]

T = TypeVar('T')


class Compressor:
    def __init__(
            self,
            compress_func: Optional[CompressFunc] = None,
            matrix_func: Optional[MatrixFunc] = None,
            recovery_func: Optional[RecoveryFunc] = None,
    ):
        self.compress_func = compress_func or compressive_sensing
        self.matrix_func = matrix_func or measurement_matrix
        self.recovery_func = recovery_func or orthogonal_match_pursuit
        self._matrix: Optional[np.ndarray] = None
        self._data: Optional[np.ndarray] = None
        self._recovered_data: Optional[np.ndarray] = None

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
        self._data = data.copy()
        _data = self._data.copy()

        # Применение порога epsilon для разреживания данных
        if epsilon is not None:
            if epsilon < 0:
                raise ValueError("epsilon cannot be negative")

            _data[np.abs(_data) < epsilon] = 0

        self._matrix = self.matrix_func(len(_data), compress)
        compressed_data = self.compress_func(_data, self._matrix)
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

        self._recovered_data = self.recovery_func(compressed_data, self._matrix, threshold, max_iterations)
        return self._recovered_data

    def metric(
            self,
            func: Callable[[np.ndarray, np.ndarray], T],
    ) -> T:
        """
        Общая функция для вывода метрик
        :param func:
        :return:
        """
        if self._data is None:
            raise RuntimeError("Compressor.metric called before compress")

        if self._recovered_data is None:
            raise RuntimeError("Compressor.metric called before decompress")

        return func(self._data, self._recovered_data)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
