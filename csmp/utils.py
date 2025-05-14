import numpy as np


def generate_test_signal(
        signal_type: str = 'sinusoid',
        length: int = 4096,
        **kwargs
) -> np.ndarray:
    """
    Генерация тестового сигнала.

    Args:
        signal_type: Тип сигнала ('sinusoid', 'sparse', 'step', 'chirp').
        length: Длина сигнала.
        **kwargs: Дополнительные параметры для генерации сигнала.

    Returns:
        Сгенерированный сигнал.
    """
    x = np.linspace(0, 1, length, endpoint=False)

    if signal_type == 'sinusoid':
        # Синусоида с несколькими частотами
        freq1 = kwargs.get('freq1', 0.05)
        freq2 = kwargs.get('freq2', 0.15)
        freq3 = kwargs.get('freq3', None)

        signal = (
                np.cos(2 * np.pi * freq1 * x) +
                np.cos(2 * np.pi * freq2 * x) +
                (np.cos(2 * np.pi * freq3 * x) if freq3 is not None else 0)
        )

    elif signal_type == 'sparse':
        # Разреженный сигнал с K ненулевыми элементами
        K = kwargs.get('K', 10)
        signal = np.zeros(length)
        indices = np.random.choice(length, K, replace=False)
        values = np.random.uniform(-1, 1, K)
        signal[indices] = values

    elif signal_type == 'step':
        # Ступенчатый сигнал
        step_positions = kwargs.get('step_positions', [int(length / 4), int(3 * length / 4)])
        signal = np.zeros(length)

        current_value = 0
        for pos in step_positions:
            current_value = 1 - current_value
            signal[pos:] = current_value

    elif signal_type == 'chirp':
        # Линейный частотно-модулированный сигнал (chirp)
        f0 = kwargs.get('f0', 0.01)
        f1 = kwargs.get('f1', 0.4)
        t = np.linspace(0, 1, length)
        signal = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t))

    else:
        raise ValueError(f"Неизвестный тип сигнала: {signal_type}")

    return signal