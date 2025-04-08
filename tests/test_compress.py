from csmp.compressor import Compressor

def test_compress_signal():
    import csmp

    compressor = Compressor()
    signal = csmp.basic_signal(1000, 300)
    compressed_signal = compressor.compress(signal, 500)

    assert len(compressed_signal) == 500, "Сжатый сигнал имеет неправильный размер"


def test_match_pursuit():
    import csmp

    compressor = Compressor()
    signal = csmp.basic_signal(10000, 9000)
    compressed_signal = compressor.compress(signal, 10)
    reconstructed = compressor.decompress(compressed_signal)  # Восстановление сигнала

    # Вычисление ошибки восстановления
    mse = csmp.calculate_mae(compressor)
    mae = csmp.calculate_mae(compressor)
    snr = csmp.calculate_snr(compressor)

    print(f"\nMSE: {mse}")
    print(f"MAE: {mae}")
    print(f"SNR: {snr}")

    assert mse < 0.1, f"Средняя ошибка восстановления {mse} превышает допустимый порог 0.1"


def test_decompress_speed():
    import time
    import csmp

    signal = csmp.basic_signal(1000, 900)

    # Замер времени для MP
    mp_compressor = Compressor(recovery_func=csmp.match_pursuit)
    mp_compressed = mp_compressor.compress(signal, 500)

    start_time = time.time()
    reconstructed_mp = mp_compressor.decompress(mp_compressed, threshold=0.01, max_iterations=1000)
    mp_time = time.time() - start_time
    mp_mse = csmp.calculate_mse(mp_compressor)

    # Замер времени для OMP
    omp_compressor = Compressor(recovery_func=csmp.orthogonal_match_pursuit)
    omp_compressed = omp_compressor.compress(signal, 500)

    start_time = time.time()
    reconstructed_omp = omp_compressor.decompress(omp_compressed, threshold=0.01, max_iterations=1000)
    omp_time = time.time() - start_time
    omp_mse = csmp.calculate_mse(omp_compressor)

    # Замер времени для ROMP
    romp_compressor = Compressor(recovery_func=csmp.regularized_orthogonal_match_pursuit)
    romp_compressed = romp_compressor.compress(signal, 500)

    start_time = time.time()
    reconstructed_romp = romp_compressor.decompress(romp_compressed, threshold=0.01, max_iterations=1000)
    romp_time = time.time() - start_time
    romp_mse = csmp.calculate_mse(romp_compressor)

    # Вывод времени выполнения
    print(f"\nВремя выполнения MP: {mp_time:.4f} секунд")
    print(f"MSE MP: {mp_mse}")
    print(f"\nВремя выполнения OMP: {omp_time:.4f} секунд")
    print(f"MSE OMP: {omp_mse}")
    print(f"\nВремя выполнения ROMP: {romp_time:.4f} секунд")
    print(f"MSE ROMP: {romp_mse}")

    assert omp_time > mp_time, "MP не быстрее OMP"