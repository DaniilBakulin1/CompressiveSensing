from typing import Tuple

import numpy as np


def compress_signal(x: np.ndarray, m: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    n = len(x)
    perm = np.random.choice(n, m, replace=False)
    y = x[perm]
    Theta = np.exp(-2j * np.pi * np.outer(perm, np.arange(n)) / n) / np.sqrt(n)
    return y, Theta, perm
