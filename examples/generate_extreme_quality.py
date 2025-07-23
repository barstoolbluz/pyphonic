#!/usr/bin/env python3
"""
Example: Generate extreme quality coefficients exceeding libsoxr.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from polyphase_gen import PolyphaseGenerator
from verification import verify_filter_response
import numpy as np
import logging


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    print("Generating extreme quality polyphase filter bank...")
    print("Target: -192 dB stopband (32-bit integer noise floor)")
    print()
    
    # Generate extreme quality filter
    generator = PolyphaseGenerator(
        taps_per_phase=256,      # High tap count
        phase_count=2048,        # High oversampling
        cutoff=0.45,            # Conservative cutoff
        window='kaiser',
        window_param=20.0,      # Very high beta for steep rolloff
        stopband_db=192.0       # Target: 32-bit integer limit
    )
    
    # Generate and save
    table, metadata = generator.generate()
    generator.save('../output/extreme_quality')
    
    # Verify specifications
    print("\nVerification Results:")
    print("-" * 50)
    
    verification = metadata['verification']
    print(f"Odd kernel length: {verification.get('odd_kernel', False)}")
    print(f"Symmetry error: {verification.get('symmetry_error', 0):.2e}")
    print(f"DC normalization error: {verification.get('dc_error', 0):.2e}")
    
    if 'measured_stopband_db' in verification:
        print(f"Measured stopband: {verification['measured_stopband_db']:.1f} dB")
        print(f"Target stopband: {metadata['target_stopband_db']:.1f} dB")
        
        if verification['measured_stopband_db'] >= metadata['target_stopband_db']:
            print("✓ Exceeds target stopband!")
        else:
            print("✗ Does not meet target stopband")
    
    if 'passband_ripple_db' in verification:
        print(f"Passband ripple: ±{verification['passband_ripple_db']/2:.2e} dB")
    
    print(f"\nTotal coefficients: {table.size:,}")
    print(f"Memory usage: {table.nbytes / 1024 / 1024:.1f} MB")
    print(f"Generation time: {metadata['generation_time']:.2f} seconds")
    
    # Compare with typical specs
    print("\nComparison with typical resamplers:")
    print("-" * 50)
    print("Specification         | Typical | Ours")
    print("---------------------|---------|--------")
    print(f"Stopband (dB)        | -120    | -{verification.get('measured_stopband_db', 0):.0f}")
    print(f"Passband ripple (dB) | ±0.001  | ±{verification.get('passband_ripple_db', 0)/2:.6f}")
    print(f"Taps per phase       | 32-64   | {metadata['taps_per_phase']}")
    print(f"Phase count          | 64-256  | {metadata['phase_count']}")


if __name__ == '__main__':
    main()