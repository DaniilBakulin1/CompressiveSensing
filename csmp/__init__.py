from csmp.utils import generate_test_signal
from csmp.compressivesensing import CompressiveSensing
from csmp.reconstruction import ReconstructionAlgorithm, MP, OMP
from csmp.basis import Basis, DCTBasis, DFTBasis

__all__ = [
    "generate_test_signal",
    "CompressiveSensing",
    "ReconstructionAlgorithm",
    "MP",
    "OMP",
    "Basis",
    "DCTBasis",
    "DFTBasis",
]

__version__ = "0.1.0-alpha3"
__author__ = "xephosbot"
__email__ = "xephosbot@gmail.com"
__description__ = "Library for compressive sensing matching pursuit"