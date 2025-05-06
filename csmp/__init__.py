from csmp.utils import generate_signal
from csmp.compress import compress_signal
from csmp.decompress import matching_pursuit, orthogonal_matching_pursuit, reconstruct_signal
from csmp.metrics import calculate_mse, calculate_mae, calculate_snr

__all__ = [
    "generate_signal",
    "compress_signal",
    "matching_pursuit",
    "orthogonal_matching_pursuit",
    "reconstruct_signal",
    "calculate_mse",
    "calculate_mae",
    "calculate_snr",
]

__version__ = "0.1.0-alpha3"
__author__ = "xephosbot"
__email__ = "xephosbot@gmail.com"
__description__ = "Library for compressive sensing matching pursuit"