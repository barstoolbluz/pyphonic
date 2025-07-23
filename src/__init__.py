"""
Pyphonic - Mathematically rigorous DSP coefficient generation.
"""

from .fir_filter_gen import FIRFilterGenerator
from .sinc_table_gen import SincTableGenerator
from .polyphase_gen import PolyphaseGenerator
from .verification import verify_filter_response, compare_with_libsox

__version__ = "0.1.0"
__all__ = [
    "FIRFilterGenerator",
    "SincTableGenerator", 
    "PolyphaseGenerator",
    "verify_filter_response",
    "compare_with_libsox"
]