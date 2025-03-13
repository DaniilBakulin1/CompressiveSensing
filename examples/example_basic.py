import csmp
from csmp import Compressor
import matplotlib.pyplot as plt

def main():
    signal = csmp.basic_signal(100, 60)

    compressor = Compressor()
    compressed = compressor.compress(signal, 90)
    recovered_signal = compressor.decompress(compressed)

    mse = csmp.calculate_mse(compressor)
    print("MSE:", mse)

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Исходный сигнал', color='blue')
    plt.plot(recovered_signal, label='Восстановленный сигнал', color='red', linestyle='dashed')
    plt.title('Исходный и восстановленный сигналы')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()