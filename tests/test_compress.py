def test_compress_signal():
    import numpy as np
    from csmp import compress_signal, generate_measurement_matrix

    signal = np.random.randn(100)
    matrix = generate_measurement_matrix(100, 50)
    compressed = compress_signal(signal, matrix)

    assert compressed.shape == (50,), "Сжатый сигнал имеет неправильный размер"
    assert np.allclose(compressed, matrix.T @ signal), "Сжатый сигнал должен быть результатом произведения матрицы на сигнал"


def test_sparse_reconstruction():
    import numpy as np
    from csmp import sparse_reconstruction

    dictionary = np.eye(5)  # Ортонормированный словарь
    coefficients = np.array([1, 0, 0, 2, 0])
    reconstructed = sparse_reconstruction(coefficients, dictionary)

    assert np.allclose(reconstructed, np.array([1, 0, 0, 2, 0])), "Восстановленный сигнал неверный"


def test_match_pursuit():
    import numpy as np
    from csmp import match_pursuit

    dictionary = np.eye(5)  # Ортонормированный словарь
    signal = np.array([1, 0, 0, 2, 0])
    compressed = dictionary.T @ signal

    coefficients, _ = match_pursuit(compressed, dictionary, threshold=0.01, epsilon=0.1)
    reconstructed = dictionary @ coefficients

    # Проверка восстановления
    assert np.allclose(reconstructed, signal), "Сигнал не восстановлен точно"

    # Проверка разреживания
    assert coefficients[1] == 0 and coefficients[4] == 0, "Коэффициенты должны быть разреженными"