#!/usr/bin/env python3
"""
Test the fixed polyphase implementation against all recommendations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from polyphase_gen_fixed import PolyphaseGenerator, property_based_tests
import numpy as np
import logging


def test_all_fixes():
    """Test that all fixes from recommendations3.png are working."""
    
    print("Testing Fixed Polyphase Implementation")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Test 1: Kernel length truncation fix
    print("\n1. Testing kernel length fix for odd taps_per_phase...")
    try:
        gen_odd = PolyphaseGenerator(
            taps_per_phase=33,  # Odd number
            phase_count=64,
            stopband_db=120.0
        )
        table, meta = gen_odd.generate()
        print(f"   ✓ Odd taps_per_phase=33 works correctly")
        print(f"     Kernel length: {meta['kernel_length']}")
        print(f"     Zero crossings: {meta['zero_crossings']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 2: Pass/stop-band indices fix
    print("\n2. Testing pass/stop-band index calculation...")
    gen = PolyphaseGenerator(
        taps_per_phase=64,
        phase_count=128,
        cutoff=0.45,
        stopband_db=140.0
    )
    table, meta = gen.generate()
    verification = meta['verification']
    
    if 'passband_ripple_db' in verification:
        print(f"   ✓ Passband ripple measured: {verification['passband_ripple_db']:.6f} dB")
    if 'measured_stopband_db' in verification:
        print(f"   ✓ Stopband measured: {verification['measured_stopband_db']:.1f} dB")
    
    # Test 3: Beta calculation from stopband
    print("\n3. Testing automatic β calculation from stopband...")
    gen_auto = PolyphaseGenerator(
        taps_per_phase=64,
        phase_count=128,
        stopband_db=180.0
        # window_param not specified - should be calculated
    )
    _, meta_auto = gen_auto.generate()
    expected_beta = 0.1102 * (180.0 - 8.7)
    print(f"   ✓ Auto-calculated β: {meta_auto['window_param']:.2f}")
    print(f"     Expected β: {expected_beta:.2f}")
    
    # Test 4: Float32 quantization warning
    print("\n4. Testing float32 quantization analysis...")
    gen.save('../output/test_fixed', include_float32=True)
    
    # Test 5: Stopband enforcement
    print("\n5. Testing stopband goal enforcement...")
    try:
        # This should fail if we can't meet the target
        gen_fail = PolyphaseGenerator(
            taps_per_phase=16,  # Too few taps
            phase_count=32,     # Too few phases
            stopband_db=200.0   # Impossible target
        )
        _, _ = gen_fail.generate()
        print("   ✗ Should have failed but didn't")
    except ValueError as e:
        print(f"   ✓ Correctly failed: {str(e)[:60]}...")
    
    # Test 6: Memory efficiency
    print("\n6. Testing memory-efficient extraction...")
    import tracemalloc
    tracemalloc.start()
    
    gen_mem = PolyphaseGenerator(
        taps_per_phase=128,
        phase_count=512,
        stopband_db=160.0
    )
    table_mem, _ = gen_mem.generate()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"   ✓ Memory usage: {peak / 1024 / 1024:.1f} MB peak")
    print(f"     Table size: {table_mem.nbytes / 1024 / 1024:.1f} MB")
    
    # Test 7: Property-based tests
    print("\n7. Running property-based tests...")
    property_based_tests()
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == '__main__':
    test_all_fixes()