from csmp.compressor import Compressor
from csmp.utils import basic_signal
from csmp.core.compress import compressive_sensing
from csmp.core.decompress import match_pursuit, orthogonal_match_pursuit, regularized_orthogonal_match_pursuit
from csmp.core.matrix import measurement_matrix
from csmp.core.metrics import calculate_mse, calculate_mae, calculate_snr

__all__ = [
    "Compressor",
    "basic_signal",
    "compressive_sensing",
    "match_pursuit",
    "orthogonal_match_pursuit",
    "regularized_orthogonal_match_pursuit",
    "measurement_matrix",
    "calculate_mse",
    "calculate_mae",
    "calculate_snr",
]

__version__ = "0.1.0-alpha2"
__author__ = "xephosbot"
__email__ = "xephosbot@gmail.com"
__description__ = "Library for compressive sensing matching pursuit"