import time
import csmp


def test_compression_size():
    _, x = csmp.generate_signal(1000)
    y, Theta, _ = csmp.compress_signal(x, 500)
    assert len(y) == 500, "Сжатый сигнал имеет неправильный размер"


def test_mp_reconstruction_accuracy():
    _, x = csmp.generate_signal(1000)
    y, Theta, _ = csmp.compress_signal(x, 300)
    s_hat = csmp.orthogonal_matching_pursuit(Theta, y, K=2)
    x_hat = csmp.reconstruct_signal(s_hat)

    snr = csmp.calculate_snr(x, x_hat)
    mse = csmp.calculate_mse(x, x_hat)
    mae = csmp.calculate_mae(x, x_hat)

    print(f"SNR: {snr:.2f} dB")
    print(f"MSE: {mse:.4e}")
    print(f"MAE: {mae:.4e}")

    assert snr > 5, "SNR слишком низкий"
    assert mse < 1.0, "MSE слишком высокий"


def test_algorithm_speed_comparison():
    _, x = csmp.generate_signal(1000)
    y, Theta, _ = csmp.compress_signal(x, 300)

    start = time.time()
    csmp.matching_pursuit(Theta, y, K=10)
    mp_time = time.time() - start

    start = time.time()
    csmp.orthogonal_matching_pursuit(Theta, y, K=10)
    omp_time = time.time() - start

    print(f"MP time: {mp_time:.4f}s")
    print(f"OMP time: {omp_time:.4f}s")

    assert mp_time < omp_time, "MP должен быть быстрее или сравним по скорости с OMP"