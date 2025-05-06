from typing import Tuple

import numpy as np


def generate_signal(n: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, n, endpoint=False)
    x = np.cos(2 * np.pi * 97 * t) + np.cos(2 * np.pi * 777 * t)
    return t, x