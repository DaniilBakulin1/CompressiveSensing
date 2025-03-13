import numpy as np

from csmp import Compressor


def calculate_mse(compressor: Compressor) -> float:
    """
    Вычисляет Mean Square Error (MSE).
    :param compressor:
    :return:
    """
    return compressor.metric(lambda x, y: np.mean((x - y) ** 2))


def calculate_mae(compressor: Compressor) -> float:
    """
    Вычисляет Mean Absolute Error (MAE).
    :param compressor:
    :return:
    """
    return compressor.metric(lambda x, y: np.mean(np.abs(x - y)))


def calculate_snr(compressor: Compressor) -> float:
    """
    Вычисляет Signal to Noise Ratio (SNR).
    :param compressor:
    :return:
    """
    def snr_lambda(x: np.ndarray, y: np.ndarray) -> float:
        signal_power = np.mean(x ** 2)
        noise_power = np.mean((x - y) ** 2)
        return 10 * np.log10(signal_power / noise_power)
    return compressor.metric(snr_lambda)