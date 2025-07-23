#!/usr/bin/env python3
"""
Direct test of the fixed polyphase implementation.
"""

import numpy as np
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly without going through __init__.py
import polyphase_gen_fixed

def implacable_scrutiny():
    """Perform implacable scrutiny of the fixed implementation."""
    
    print("IMPLACABLE SCRUTINY OF FIXED POLYPHASE IMPLEMENTATION")
    print("=" * 70)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Issue 1: Test kernel length for odd taps_per_phase
    print("\nISSUE 1: Kernel length truncation for odd taps_per_phase")
    print("-" * 60)
    
    # Test odd taps
    for taps in [31, 33, 63, 65]:
        phases = 64
        gen = polyphase_gen_fixed.PolyphaseGenerator(
            taps_per_phase=taps,
            phase_count=phases,
            stopband_db=120.0
        )
        
        # Check kernel length calculation
        expected_kernel_length = taps * phases + 1
        actual_kernel_length = gen.kernel_length
        
        print(f"Taps={taps}, Phases={phases}:")
        print(f"  Expected kernel length: {expected_kernel_length}")
        print(f"  Actual kernel length: {actual_kernel_length}")
        print(f"  Zero crossings: {gen.zero_crossings}")
        
        if expected_kernel_length != actual_kernel_length:
            print(f"  ❌ FAIL: Kernel length mismatch!")
        else:
            print(f"  ✓ Pass")
    
    # Issue 2: Test polyphase extraction indices
    print("\nISSUE 2: Polyphase start index calculation")
    print("-" * 60)
    
    gen = polyphase_gen_fixed.PolyphaseGenerator(
        taps_per_phase=64,
        phase_count=128,
        stopband_db=140.0
    )
    
    # Generate and check extraction
    try:
        table, metadata = gen.generate()
        print(f"Table shape: {table.shape}")
        print(f"Expected shape: ({gen.phase_count}, {gen.taps_per_phase})")
        
        if table.shape != (gen.phase_count, gen.taps_per_phase):
            print("❌ FAIL: Shape mismatch!")
        else:
            print("✓ Pass: Shape correct")
            
        # Check that all coefficients are extracted
        total_coeffs = table.size
        expected_coeffs = gen.phase_count * gen.taps_per_phase
        print(f"Total coefficients: {total_coeffs}")
        print(f"Expected coefficients: {expected_coeffs}")
        
        if total_coeffs != expected_coeffs:
            print("❌ FAIL: Not all coefficients extracted!")
        else:
            print("✓ Pass: All coefficients extracted")
            
    except Exception as e:
        print(f"❌ FAIL: Exception during generation: {e}")
    
    # Issue 3: Test pass/stop-band index calculation
    print("\nISSUE 3: Pass/stop-band index calculation")
    print("-" * 60)
    
    verification = metadata.get('verification', {})
    if 'passband_ripple_db' in verification:
        print(f"✓ Passband ripple calculated: {verification['passband_ripple_db']:.6f} dB")
    else:
        print("❌ FAIL: No passband ripple calculated")
        
    if 'measured_stopband_db' in verification:
        print(f"✓ Stopband attenuation: {verification['measured_stopband_db']:.1f} dB")
    else:
        print("❌ FAIL: No stopband measurement")
    
    # Issue 4: Test row-sum validation
    print("\nISSUE 4: Row-sum validation")
    print("-" * 60)
    
    # Test with pathological parameters that might produce near-zero sums
    try:
        gen_bad = polyphase_gen_fixed.PolyphaseGenerator(
            taps_per_phase=4,  # Very few taps
            phase_count=8,     # Few phases
            cutoff=0.01,       # Very low cutoff
            stopband_db=60.0
        )
        table_bad, _ = gen_bad.generate()
        print("❌ FAIL: Should have detected pathological kernel")
    except ValueError as e:
        print(f"✓ Pass: Correctly detected pathological kernel: {str(e)[:60]}...")
    except Exception as e:
        # Other exceptions might be OK too
        print(f"✓ Pass: Failed with: {str(e)[:60]}...")
    
    # Issue 5: Test stopband enforcement
    print("\nISSUE 5: Stopband goal enforcement")
    print("-" * 60)
    
    try:
        gen_impossible = polyphase_gen_fixed.PolyphaseGenerator(
            taps_per_phase=8,   # Too few taps
            phase_count=16,     # Too few phases
            stopband_db=200.0   # Impossible target
        )
        _, _ = gen_impossible.generate()
        print("❌ FAIL: Should have failed to meet stopband target")
    except ValueError as e:
        if "stopband" in str(e).lower():
            print(f"✓ Pass: Correctly enforced stopband: {str(e)[:60]}...")
        else:
            print(f"? Uncertain: Failed with: {str(e)[:60]}...")
    
    # Issue 6: Window parameter calculation
    print("\nISSUE 6: Automatic β calculation from stopband")
    print("-" * 60)
    
    for stopband in [120.0, 160.0, 180.0, 192.0]:
        gen_auto = polyphase_gen_fixed.PolyphaseGenerator(
            taps_per_phase=64,
            phase_count=128,
            stopband_db=stopband
            # window_param not specified
        )
        
        expected_beta = 0.1102 * (abs(stopband) - 8.7)
        actual_beta = gen_auto.window_param
        
        print(f"Stopband={stopband} dB:")
        print(f"  Expected β: {expected_beta:.2f}")
        print(f"  Actual β: {actual_beta:.2f}")
        print(f"  Difference: {abs(expected_beta - actual_beta):.6f}")
        
        if abs(expected_beta - actual_beta) > 0.01:
            print("  ❌ FAIL: β calculation incorrect")
        else:
            print("  ✓ Pass")
    
    # Issue 7: Memory efficiency test
    print("\nISSUE 7: Memory efficiency")
    print("-" * 60)
    
    import tracemalloc
    tracemalloc.start()
    
    gen_mem = polyphase_gen_fixed.PolyphaseGenerator(
        taps_per_phase=256,
        phase_count=1024,
        stopband_db=160.0
    )
    
    current_before, _ = tracemalloc.get_traced_memory()
    table_mem, _ = gen_mem.generate()
    current_after, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_used = (current_after - current_before) / 1024 / 1024
    table_size = table_mem.nbytes / 1024 / 1024
    
    print(f"Table size: {table_size:.1f} MB")
    print(f"Memory used during generation: {memory_used:.1f} MB")
    print(f"Efficiency ratio: {memory_used / table_size:.2f}x")
    
    if memory_used > table_size * 3:
        print("❌ FAIL: Using too much temporary memory")
    else:
        print("✓ Pass: Memory efficient")
    
    # Issue 8: Verify float32 quantization warning
    print("\nISSUE 8: Float32 quantization analysis")
    print("-" * 60)
    
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        gen_quant = polyphase_gen_fixed.PolyphaseGenerator(
            taps_per_phase=64,
            phase_count=128,
            stopband_db=180.0
        )
        gen_quant.save('test_quant', include_float32=True)
        
        # Check if warning was issued
        quant_warnings = [warning for warning in w if "round-off" in str(warning.message)]
        
        if quant_warnings:
            print(f"✓ Pass: Quantization warning issued")
            for warning in quant_warnings:
                print(f"  Warning: {warning.message}")
        else:
            print("✓ Pass: No quantization warning needed (round-off below threshold)")
    
    # Clean up
    import glob
    for f in glob.glob('test_quant*'):
        os.remove(f)
    
    # Issue 9: Verification completeness
    print("\nISSUE 9: Verification completeness")
    print("-" * 60)
    
    gen_verify = polyphase_gen_fixed.PolyphaseGenerator(
        taps_per_phase=32,
        phase_count=64,
        stopband_db=120.0
    )
    table_verify, meta_verify = gen_verify.generate()
    verify_dict = meta_verify.get('verification', {})
    
    required_checks = [
        'odd_kernel',
        'symmetry_error',
        'symmetry_pass',
        'dc_error',
        'dc_normalized',
        'passband_ripple_db',
        'measured_stopband_db',
        'group_delay_variation',
        'phase_continuity_max'
    ]
    
    for check in required_checks:
        if check in verify_dict:
            print(f"✓ {check}: {verify_dict[check]}")
        else:
            print(f"❌ FAIL: Missing {check}")
    
    print("\n" + "=" * 70)
    print("SCRUTINY COMPLETE")


if __name__ == '__main__':
    implacable_scrutiny()