from typing import Any
import numpy as np
from numpy import floating


def calculate_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    signal_power = np.sum(np.abs(original) ** 2)
    noise_power = np.sum(np.abs(original - reconstructed) ** 2)
    return 10 * np.log10(signal_power / noise_power)


def calculate_mse(original: np.ndarray, reconstructed: np.ndarray) -> floating[Any]:
    return np.mean((original - reconstructed) ** 2)


def calculate_mae(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return np.mean(np.abs(original - reconstructed))