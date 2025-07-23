#!/usr/bin/env python3
"""
Wrapper to make sinc_table_gen compatible with polyphase_gen_fixed.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from .sinc_table_gen import generate_sinc_table


class SincTableGenerator:
    """
    Wrapper class for sinc table generation to work with polyphase generator.
    """
    
    def __init__(
        self,
        zero_crossings: int,
        oversample_factor: int,
        cutoff: float = 0.5,
        window_type: str = 'kaiser',
        window_param: Optional[float] = None
    ):
        self.zero_crossings = zero_crossings
        self.oversample_factor = oversample_factor
        self.cutoff = cutoff
        self.window_type = window_type
        self.window_param = window_param if window_param is not None else 8.6
        
    def generate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate sinc table."""
        return generate_sinc_table(
            N_taps=self.zero_crossings * 2,  # Total taps
            oversample=self.oversample_factor,
            cutoff=self.cutoff,
            win_type=self.window_type,
            beta=self.window_param
        )