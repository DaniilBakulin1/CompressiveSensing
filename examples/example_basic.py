import csmp
from csmp import CompressiveSensing, DCTBasis, MP, OMP, DFTBasis
import matplotlib.pyplot as plt

def main():
    original_signal = csmp.generate_test_signal(
        signal_type='sinusoid',
        length=256,
        freq1=97,
        freq2=777,
    )

    # Создание экземпляра CS с ДКП базисом
    cs = CompressiveSensing(basis=DFTBasis())

    # Сжатие сигнала
    compressed_signal = cs.compress(
        signal=original_signal,
        compression_ratio=0.3,  # сжатие до 30% от исходной длины
        sampling_method='random_rows'
    )

    # Восстановление сигнала с помощью MP
    mp_algorithm = OMP()
    reconstructed_signal = cs.reconstruct(
        compressed_signal=compressed_signal,
        signal_length=len(original_signal),
        algorithm=mp_algorithm,
        max_iter=1000,
        epsilon=1e-6
    )

    # Оценка качества восстановления
    metrics = cs.evaluate(original_signal, reconstructed_signal)
    print(metrics)

    plt.figure(figsize=(18, 5))
    plt.plot(original_signal)
    plt.plot(reconstructed_signal)
    plt.show()

if __name__ == '__main__':
    main()