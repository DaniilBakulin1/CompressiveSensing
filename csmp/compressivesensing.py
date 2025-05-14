from typing import Dict

import numpy as np

from csmp.basis import Basis
from csmp.matrix import SamplingMatrix
from csmp.reconstruction import ReconstructionAlgorithm


class CompressiveSensing:
    """
    Основной класс для работы с Compressive Sensing.
    """

    def __init__(self, basis: Basis):
        """
        Инициализация CS с выбранным базисом.

        Args:
            basis: Базис для представления сигнала.
        """
        self.basis = basis
        self.sampling_matrix = None
        self.sampling_indices = None

    def compress(self,
                 signal: np.ndarray,
                 compression_ratio: float,
                 sampling_method: str = 'random_rows') -> np.ndarray:
        """
        Сжатие сигнала с помощью CS.

        Args:
            signal: Входной сигнал.
            compression_ratio: Коэффициент сжатия (0 < ratio < 1).
            sampling_method: Метод выбора отсчетов ('random_rows', 'gaussian', 'bernoulli').

        Returns:
            Сжатый сигнал.
        """
        N = len(signal)
        M = int(N * compression_ratio)

        # Создание матрицы выбора отсчетов
        if sampling_method == 'random_rows':
            self.sampling_matrix, self.sampling_indices = SamplingMatrix.random_rows(M, N)
        elif sampling_method == 'gaussian':
            self.sampling_matrix = SamplingMatrix.gaussian(M, N)
        elif sampling_method == 'bernoulli':
            self.sampling_matrix = SamplingMatrix.bernoulli(M, N)
        else:
            raise ValueError(f"Неизвестный метод выбора отсчетов: {sampling_method}")

        # Сжатие сигнала
        if sampling_method == 'random_rows':
            # Если используем выбор случайных строк, просто выбираем отсчеты сигнала
            return signal[self.sampling_indices]
        else:
            # Иначе используем матрицу выбора отсчетов
            return np.dot(self.sampling_matrix, signal)

    def reconstruct(self,
                    compressed_signal: np.ndarray,
                    signal_length: int,
                    algorithm: ReconstructionAlgorithm,
                    **kwargs) -> np.ndarray:
        """
        Восстановление сигнала по сжатому представлению.

        Args:
            compressed_signal: Сжатый сигнал.
            signal_length: Длина исходного сигнала.
            algorithm: Алгоритм восстановления.
            **kwargs: Дополнительные параметры для алгоритма.

        Returns:
            Восстановленный сигнал.
        """
        # Получение матрицы базиса
        basis_matrix = self.basis.get_matrix(signal_length)

        # Формирование матрицы измерений A = Phi * Psi
        if self.sampling_indices is not None:
            # Если используем выбор случайных строк, берем соответствующие строки из матрицы базиса
            sensing_matrix = basis_matrix[self.sampling_indices, :]
        else:
            # Иначе используем произведение матриц
            sensing_matrix = np.dot(self.sampling_matrix, basis_matrix)

        # Восстановление разреженного представления сигнала
        sparse_coeffs = algorithm.reconstruct(sensing_matrix, compressed_signal, **kwargs)

        # Восстановление исходного сигнала из разреженного представления
        reconstructed_signal = self.basis.backward(sparse_coeffs)

        return reconstructed_signal

    @staticmethod
    def evaluate(original_signal: np.ndarray,
                 reconstructed_signal: np.ndarray) -> Dict[str, float]:
        """
        Оценка качества восстановления сигнала.

        Args:
            original_signal: Исходный сигнал.
            reconstructed_signal: Восстановленный сигнал.

        Returns:
            Словарь с метриками оценки качества.
        """
        # Среднеквадратичная ошибка (MSE)
        mse = np.mean((original_signal - reconstructed_signal) ** 2)

        # Среднеквадратичная ошибка (MAE)
        mae = np.mean(np.abs(original_signal - reconstructed_signal))

        # Отношение сигнал/шум (SNR)
        signal_power = np.mean(original_signal ** 2)
        noise_power = mse
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        return {
            'mse': mse,
            'mae': mae,
            'snr': snr,
        }
