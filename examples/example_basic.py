import csmp
from csmp.compressor import Compressor


def main():
    signal = csmp.basic_signal(1000, 400)
    print(signal)

    compressor = Compressor()
    compressed = compressor.compress(signal, 500)
    print(compressed)

    recovered_signal = compressor.decompress(compressed)
    print(recovered_signal)


if __name__ == '__main__':
    main()