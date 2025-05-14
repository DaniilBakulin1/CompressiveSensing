import numpy as np


class SamplingMatrix:
    """
    Класс для создания матрицы выбора отсчетов.
    """

    @staticmethod
    def random_rows(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Создание матрицы выбора отсчетов со случайными строками.

        Args:
            rows: Количество строк (измерений M).
            cols: Количество столбцов (длина сигнала N).

        Returns:
            Матрица выбора отсчетов.
        """
        # Создаем матрицу выбора случайных строк
        indices = np.sort(np.random.choice(cols, rows, replace=False))
        sampling_matrix = np.zeros((rows, cols))

        for i, idx in enumerate(indices):
            sampling_matrix[i, idx] = 1

        return sampling_matrix, indices

    @staticmethod
    def gaussian(rows: int, cols: int) -> np.ndarray:
        """
        Создание матрицы выбора отсчетов с элементами из нормального распределения.

        Args:
            rows: Количество строк (измерений M).
            cols: Количество столбцов (длина сигнала N).

        Returns:
            Матрица выбора отсчетов.
        """
        return np.random.normal(0, 1 / cols, (rows, cols))

    @staticmethod
    def bernoulli(rows: int, cols: int) -> np.ndarray:
        """
        Создание матрицы выбора отсчетов с элементами из распределения Бернулли.

        Args:
            rows: Количество строк (измерений M).
            cols: Количество столбцов (длина сигнала N).

        Returns:
            Матрица выбора отсчетов.
        """
        values = np.random.choice([-1 / np.sqrt(rows), 1 / np.sqrt(rows)], size=(rows, cols))
        return values