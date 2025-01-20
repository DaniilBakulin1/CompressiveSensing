from .compress import compress, match_pursuit, orthogonal_match_pursuit
from .utils import generate_basic_signal, generate_measurement_matrix

__all__ = [
    "compress",
    "match_pursuit",
    "orthogonal_match_pursuit",
    "generate_basic_signal",
    "generate_measurement_matrix",
]
__version__ = "0.1.0-alpha2"
__author__ = "xephosbot"
__email__ = "xephosbot@gmail.com"
__description__ = "Library for compressive sensing matching pursuit"