from csmp.compressor import Compressor

def test_compress_signal():
    import csmp

    compressor = Compressor()
    signal = csmp.basic_signal(1000, 300)
    compressed_signal = compressor.compress(signal, 500)

    assert len(compressed_signal) == 500, "Сжатый сигнал имеет неправильный размер"


def test_match_pursuit():
    import csmp
    import numpy as np

    compressor = Compressor()
    signal = csmp.basic_signal(1000, 300)
    compressed_signal = compressor.compress(signal, 600)
    reconstructed = compressor.decompress(compressed_signal, threshold=0.1, max_iterations=300)  # Восстановление сигнала

    # Вычисление ошибки восстановления
    error = np.abs(signal - reconstructed)
    mean_error = np.mean(error)

    print(f"Средняя ошибка восстановления: {mean_error}")

    assert mean_error < 0.1, f"Средняя ошибка восстановления {mean_error} превышает допустимый порог 0.1"


def test_omp_speed():
    import time
    import csmp

    # Замер времени для MP
    compressor = Compressor(recovery_func=csmp._match_pursuit)
    signal = csmp.basic_signal(10000, 300)
    compressed = compressor.compress(signal, 6000)

    start_time = time.time()
    reconstructed_mp = compressor.decompress(compressed)
    mp_time = time.time() - start_time

    # Замер времени для OMP
    compressor = Compressor(recovery_func=csmp._orthogonal_match_pursuit)
    signal = csmp.basic_signal(10000, 300)
    compressed = compressor.compress(signal, 6000)

    start_time = time.time()
    reconstructed_omp = compressor.decompress(compressed)
    omp_time = time.time() - start_time

    # Вывод времени выполнения
    print(f"Время выполнения MP: {mp_time:.4f} секунд")
    print(f"Время выполнения OMP: {omp_time:.4f} секунд")

    assert omp_time < mp_time, "OMP не быстрее MP"