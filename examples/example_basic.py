import csmp.compress


def main():
    signal = csmp.generate_basic_signal(10, 4)
    print(signal)

    compressed, matrix = csmp.compress(signal, 5)
    print(compressed)

    recovered_signal = csmp.orthogonal_match_pursuit(compressed, matrix)
    print(recovered_signal)


if __name__ == '__main__':
    main()