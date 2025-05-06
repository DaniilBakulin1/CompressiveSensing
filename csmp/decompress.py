import numpy as np
from numpy.fft import ifft


def matching_pursuit(Theta: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    n = Theta.shape[1]
    residual = y.copy()
    s_hat = np.zeros(n, dtype=complex)
    for _ in range(K):
        corr = np.abs(Theta.conj().T @ residual)
        idx = np.argmax(corr)
        phi = Theta[:, idx]
        alpha = np.vdot(phi, residual)
        s_hat[idx] += alpha
        residual = residual - alpha * phi
    return s_hat


def orthogonal_matching_pursuit(Theta: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    n = Theta.shape[1]
    residual = y.copy()
    selected_indices = []
    s_hat = np.zeros(n, dtype=complex)

    for _ in range(K):
        h = Theta.conj().T @ residual
        k = np.argmax(np.abs(h))
        selected_indices.append(k)

        Psi_s = Theta[:, selected_indices]
        gamma_s, *_ = np.linalg.lstsq(Psi_s, y, rcond=None)

        residual = y - Psi_s @ gamma_s

    for i, idx in enumerate(selected_indices):
        s_hat[idx] = gamma_s[i]

    return s_hat


def reconstruct_signal(s_hat: np.ndarray) -> np.ndarray:
    return np.real(ifft(s_hat) * np.sqrt(len(s_hat)))