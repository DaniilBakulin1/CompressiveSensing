from .compress import _compress, _match_pursuit, _orthogonal_match_pursuit
from .compressor import Compressor
from .matrix import _measurement_matrix
from .utils import basic_signal

__all__ = [
    "Compressor",
    "_compress",
    "_match_pursuit",
    "_orthogonal_match_pursuit",
    "basic_signal",
    "_measurement_matrix",
]

__version__ = "0.1.0-alpha2"
__author__ = "xephosbot"
__email__ = "xephosbot@gmail.com"
__description__ = "Library for compressive sensing matching pursuit"