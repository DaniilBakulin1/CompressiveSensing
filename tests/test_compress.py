import time
import csmp
from csmp import CompressiveSensing, OMP, MP, DFTBasis, DCTBasis


def test_compression_size():
    original_signal = csmp.generate_test_signal(
        signal_type='sinusoid',
        length=1000,
        freq1=97,
        freq2=777,
    )
    cs = CompressiveSensing(basis=DFTBasis())

    # Сжатие сигнала
    compressed_signal = cs.compress(
        signal=original_signal,
        compression_ratio=0.3,  # сжатие до 30% от исходной длины
        sampling_method='random_rows'
    )

    assert len(compressed_signal) == 300, "Сжатый сигнал имеет неправильный размер"


def test_mp_reconstruction_accuracy():
    original_signal = csmp.generate_test_signal(
        signal_type='sinusoid',
        length=1000,
        freq1=97,
        freq2=777,
    )
    cs = CompressiveSensing(basis=DFTBasis())

    # Сжатие сигнала
    compressed_signal = cs.compress(
        signal=original_signal,
        compression_ratio=0.3,  # сжатие до 30% от исходной длины
        sampling_method='random_rows'
    )

    mp_algorithm = OMP()
    reconstructed_signal = cs.reconstruct(
        compressed_signal=compressed_signal,
        signal_length=len(original_signal),
        algorithm=mp_algorithm,
        max_iter=10000,
        sparsity=100
    )

    metrics = cs.evaluate(original_signal, reconstructed_signal)

    print(f"SNR: {metrics['snr']} dB")
    print(f"MSE: {metrics['mse']}")
    print(f"MAE: {metrics['mae']}")

    assert metrics['snr'] > 5, "SNR слишком низкий"
    assert metrics['mse'] < 1.0, "MSE слишком высокий"


def test_algorithm_speed_comparison():
    original_signal = csmp.generate_test_signal(
        signal_type='sinusoid',
        length=1000,
        freq1=97,
        freq2=777,
    )
    cs = CompressiveSensing(basis=DFTBasis())

    # Сжатие сигнала
    compressed_signal = cs.compress(
        signal=original_signal,
        compression_ratio=0.1,  # сжатие до 10% от исходной длины
        sampling_method='random_rows'
    )

    start = time.time()
    mp_algorithm = MP()
    cs.reconstruct(
        compressed_signal=compressed_signal,
        signal_length=len(original_signal),
        algorithm=mp_algorithm,
        max_iter=10000,
        sparsity=100
    )
    mp_time = time.time() - start

    start = time.time()
    omp_algorithm = OMP()
    cs.reconstruct(
        compressed_signal=compressed_signal,
        signal_length=len(original_signal),
        algorithm=omp_algorithm,
        max_iter=10000,
        sparsity=100
    )
    omp_time = time.time() - start

    print("\n")
    print(f"MP time: {mp_time:.4f}s")
    print(f"OMP time: {omp_time:.4f}s")

    # Сжатие сигнала
    compressed_signal = cs.compress(
        signal=original_signal,
        compression_ratio=0.3,  # сжатие до 30% от исходной длины
        sampling_method='random_rows'
    )

    start = time.time()
    mp_algorithm = MP()
    cs.reconstruct(
        compressed_signal=compressed_signal,
        signal_length=len(original_signal),
        algorithm=mp_algorithm,
        max_iter=10000,
        sparsity=100
    )
    mp_time = time.time() - start

    start = time.time()
    omp_algorithm = OMP()
    cs.reconstruct(
        compressed_signal=compressed_signal,
        signal_length=len(original_signal),
        algorithm=omp_algorithm,
        max_iter=10000,
        sparsity=100
    )
    omp_time = time.time() - start

    print("\n")
    print(f"MP time: {mp_time:.4f}s")
    print(f"OMP time: {omp_time:.4f}s")

    # Сжатие сигнала
    compressed_signal = cs.compress(
        signal=original_signal,
        compression_ratio=0.5,  # сжатие до 50% от исходной длины
        sampling_method='random_rows'
    )

    start = time.time()
    mp_algorithm = MP()
    cs.reconstruct(
        compressed_signal=compressed_signal,
        signal_length=len(original_signal),
        algorithm=mp_algorithm,
        max_iter=10000,
        sparsity=100
    )
    mp_time = time.time() - start

    start = time.time()
    omp_algorithm = OMP()
    cs.reconstruct(
        compressed_signal=compressed_signal,
        signal_length=len(original_signal),
        algorithm=omp_algorithm,
        max_iter=10000,
        sparsity=100
    )
    omp_time = time.time() - start

    print("\n")
    print(f"MP time: {mp_time:.4f}s")
    print(f"OMP time: {omp_time:.4f}s")