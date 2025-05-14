from abc import ABC, abstractmethod
from scipy.fftpack import dct
from scipy.fftpack import idct

import numpy as np


class Basis(ABC):
    """
    Абстрактный класс для базисов, в которых сигнал может быть разреженным или сжимаемым.
    """

    @abstractmethod
    def forward(self, signal: np.ndarray) -> np.ndarray:
        """
        Прямое преобразование сигнала в выбранный базис.

        Args:
            signal: Входной сигнал в виде numpy массива.

        Returns:
            Представление сигнала в данном базисе.
        """
        pass

    @abstractmethod
    def backward(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Обратное преобразование из базиса в сигнал.

        Args:
            coefficients: Коэффициенты разложения в базисе.

        Returns:
            Восстановленный сигнал.
        """
        pass

    @abstractmethod
    def get_matrix(self, signal_length: int) -> np.ndarray:
        """
        Получение матрицы базиса для сигнала заданной длины.

        Args:
            signal_length: Длина сигнала.

        Returns:
            Матрица базиса размера (signal_length, signal_length).
        """
        pass


class DCTBasis(Basis):
    """
    Реализация дискретного косинусного преобразования (ДКП) как базиса.
    """

    def __init__(self):
        """
        Инициализация базиса ДКП.
        """
        self.cached_matrices = {}

    def forward(self, signal: np.ndarray) -> np.ndarray:
        """
        Прямое ДКП преобразование.

        Args:
            signal: Входной сигнал.

        Returns:
            ДКП коэффициенты.
        """
        return dct(signal, type=3, norm='ortho')

    def backward(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Обратное ДКП преобразование.

        Args:
            coefficients: ДКП коэффициенты.

        Returns:
            Восстановленный сигнал.
        """
        return idct(coefficients, type=3, norm='ortho')

    def get_matrix(self, signal_length: int) -> np.ndarray:
        """
        Построение матрицы ДКП для сигнала заданной длины.

        Args:
            signal_length: Длина сигнала.

        Returns:
            Матрица ДКП размера (signal_length, signal_length).
        """
        # Используем кэширование для избежания повторных вычислений
        if signal_length in self.cached_matrices:
            return self.cached_matrices[signal_length]

        N = signal_length
        matrix = np.zeros((N, N))

        matrix[0, :] = 1 / np.sqrt(N)
        for k in range(1, N):
            for n in range(N):
                matrix[k, n] = np.sqrt(2 / N) * np.cos(np.pi * k * (2 * n + 1) / (2 * N))

        self.cached_matrices[signal_length] = matrix
        return matrix


class DFTBasis(Basis):
    """
    Реализация дискретного преобразования Фурье (ДПФ) как базиса.
    """

    def __init__(self):
        """
        Инициализация базиса ДПФ.
        """
        self.cached_matrices = {}

    def forward(self, signal: np.ndarray) -> np.ndarray:
        """
        Прямое ДПФ преобразование.

        Args:
            signal: Входной сигнал.

        Returns:
            ДПФ коэффициенты.
        """
        return np.fft.fft(signal) / np.sqrt(len(signal))

    def backward(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Обратное ДПФ преобразование.

        Args:
            coefficients: ДПФ коэффициенты.

        Returns:
            Восстановленный сигнал.
        """
        return np.real(np.fft.ifft(coefficients) * np.sqrt(len(coefficients)))

    def get_matrix(self, signal_length: int) -> np.ndarray:
        """
        Построение матрицы ДПФ для сигнала заданной длины.

        Args:
            signal_length: Длина сигнала.

        Returns:
            Матрица ДПФ размера (signal_length, signal_length).
        """
        # Используем кэширование для избежания повторных вычислений
        if signal_length in self.cached_matrices:
            return self.cached_matrices[signal_length]

        N = signal_length
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)

        self.cached_matrices[signal_length] = M
        return M