def test_create_measurement_matrix():
    import numpy as np
    from csmp import generate_measurement_matrix

    n, m = 100, 50
    matrix = generate_measurement_matrix(n, m)
    assert matrix.shape == (n, m), "Размер матрицы неверный"
    assert np.all(matrix >= -1) and np.all(matrix <= 1), "Элементы матрицы должны быть в диапазоне [-1, 1]"