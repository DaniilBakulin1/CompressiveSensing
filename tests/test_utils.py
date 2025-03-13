def test_create_measurement_matrix():
    import numpy as np
    import csmp

    n, m = 100, 50
    matrix = csmp.measurement_matrix(n, m)
    assert matrix.shape == (m, n), "Размер матрицы неверный"
    assert np.all(matrix >= -1) and np.all(matrix <= 1), "Элементы матрицы должны быть в диапазоне [-1, 1]"