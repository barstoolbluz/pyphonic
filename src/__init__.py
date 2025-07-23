"""
Pyphonic - Mathematically rigorous DSP coefficient generation.
"""

from .fir_filter_gen import FIRFilterGenerator
from .sinc_table_gen import generate_sinc_table, generate_polyphase_table
from .polyphase_gen import PolyphaseGenerator
from .polyphase_gen_fixed import PolyphaseGenerator as PolyphaseGeneratorFixed
from .verification import verify_filter_response, compare_with_libsox

__version__ = "0.1.0"
__all__ = [
    "FIRFilterGenerator",
    "generate_sinc_table",
    "generate_polyphase_table", 
    "PolyphaseGenerator",
    "PolyphaseGeneratorFixed",
    "verify_filter_response",
    "compare_with_libsox"
]