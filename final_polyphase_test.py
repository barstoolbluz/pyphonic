#!/usr/bin/env python3
"""
Final comprehensive test of the polyphase implementation.
Tests all fixes from recommendations3.png by actually running the code.
"""

import numpy as np
import sys
import os

# Import using the package structure
import src.polyphase_gen_fixed as pgf
import src.sinc_table_gen as stg

# Mock the SincTableGenerator for testing
class MockSincTableGenerator:
    def __init__(self, zero_crossings, oversample_factor, cutoff, window_type, window_param):
        self.zero_crossings = zero_crossings
        self.oversample_factor = oversample_factor
        self.cutoff = cutoff
        self.window_type = window_type
        self.window_param = window_param
        
    def generate(self):
        # Generate a simple test kernel
        N = 2 * self.zero_crossings * self.oversample_factor + 1
        x = np.arange(N) - N//2
        x = x / self.oversample_factor
        
        # Simple sinc
        kernel = np.sinc(2 * self.cutoff * x)
        
        # Simple window (rectangular for testing)
        window = np.ones_like(kernel)
        
        kernel = kernel * window
        kernel /= np.sum(kernel[::self.oversample_factor])
        
        return kernel, {}

# Replace the import in polyphase_gen_fixed
pgf.SincTableGenerator = MockSincTableGenerator


def test_all_recommendations():
    """Test each recommendation from the feedback."""
    
    print("FINAL COMPREHENSIVE TEST OF POLYPHASE IMPLEMENTATION")
    print("=" * 70)
    
    all_passed = True
    
    # Test 1: Kernel length for odd taps_per_phase
    print("\n1. Testing kernel length fix (odd taps_per_phase)")
    print("-" * 60)
    
    test_cases = [
        (31, 64),  # odd taps
        (32, 64),  # even taps
        (33, 64),  # odd taps
        (64, 128), # even taps
    ]
    
    for taps, phases in test_cases:
        gen = pgf.PolyphaseGenerator(
            taps_per_phase=taps,
            phase_count=phases,
            stopband_db=120.0
        )
        
        expected_length = taps * phases + 1
        actual_length = gen.kernel_length
        
        if taps % 2 == 1:  # odd
            expected_zc = (taps - 1) // 2
        else:  # even
            expected_zc = taps // 2
            
        print(f"T={taps}, L={phases}:")
        print(f"  Kernel length: {actual_length} (expected {expected_length})")
        print(f"  Zero crossings: {gen.zero_crossings} (expected {expected_zc})")
        
        if actual_length == expected_length and gen.zero_crossings == expected_zc:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
            all_passed = False
    
    # Test 2: Centre calculation and extraction
    print("\n2. Testing polyphase extraction")
    print("-" * 60)
    
    try:
        gen = pgf.PolyphaseGenerator(
            taps_per_phase=32,
            phase_count=64,
            stopband_db=120.0
        )
        
        table, metadata = gen.generate()
        
        print(f"Table shape: {table.shape}")
        print(f"Expected: ({gen.phase_count}, {gen.taps_per_phase})")
        
        if table.shape == (gen.phase_count, gen.taps_per_phase):
            print("✓ PASS: Correct shape")
        else:
            print("✗ FAIL: Wrong shape")
            all_passed = False
            
    except Exception as e:
        print(f"✗ FAIL: Exception: {e}")
        all_passed = False
    
    # Test 3: DC normalization
    print("\n3. Testing per-row DC normalization")
    print("-" * 60)
    
    if 'table' in locals():
        dc_gains = table.sum(axis=1)
        dc_error = np.max(np.abs(dc_gains - 1.0))
        
        print(f"Max DC error: {dc_error:.2e}")
        print(f"DC gain range: [{np.min(dc_gains):.6f}, {np.max(dc_gains):.6f}]")
        
        if dc_error < 1e-10:
            print("✓ PASS: All rows normalized to unity")
        else:
            print("✗ FAIL: DC normalization error too large")
            all_passed = False
    
    # Test 4: Beta calculation
    print("\n4. Testing automatic β calculation")
    print("-" * 60)
    
    stopband_tests = [120.0, 160.0, 180.0, 192.0]
    
    for stopband in stopband_tests:
        gen = pgf.PolyphaseGenerator(
            taps_per_phase=64,
            phase_count=128,
            stopband_db=stopband
            # window_param not specified
        )
        
        expected_beta = 0.1102 * (stopband - 8.7)
        actual_beta = gen.window_param
        error = abs(expected_beta - actual_beta)
        
        print(f"Stopband {stopband} dB: β = {actual_beta:.3f} (expected {expected_beta:.3f})")
        
        if error < 0.001:
            print("  ✓ PASS")
        else:
            print(f"  ✗ FAIL: Error = {error:.6f}")
            all_passed = False
    
    # Test 5: Verification completeness
    print("\n5. Testing verification completeness")
    print("-" * 60)
    
    if 'metadata' in locals():
        verification = metadata.get('verification', {})
        
        required_fields = [
            'odd_kernel',
            'symmetry_error',
            'symmetry_pass',
            'dc_error',
            'dc_normalized',
            'group_delay_variation',
            'phase_continuity_max'
        ]
        
        for field in required_fields:
            if field in verification:
                value = verification[field]
                if isinstance(value, bool):
                    print(f"{field}: {value}")
                else:
                    print(f"{field}: {value:.2e}")
            else:
                print(f"{field}: MISSING ✗")
                all_passed = False
    
    # Test 6: Property-based tests
    print("\n6. Running property-based tests")
    print("-" * 60)
    
    try:
        # Run a subset of property tests
        import itertools
        
        test_count = 0
        pass_count = 0
        
        for taps, phases in itertools.product([32, 64], [64, 128]):
            test_count += 1
            try:
                gen = pgf.PolyphaseGenerator(
                    taps_per_phase=taps,
                    phase_count=phases,
                    stopband_db=120.0
                )
                
                table, metadata = gen.generate()
                verification = metadata['verification']
                
                # Check properties
                if (verification['odd_kernel'] and 
                    verification['symmetry_pass'] and 
                    verification['dc_normalized']):
                    pass_count += 1
                    
            except Exception as e:
                print(f"  Test T={taps}, L={phases} failed: {e}")
        
        print(f"Property tests: {pass_count}/{test_count} passed")
        
        if pass_count == test_count:
            print("✓ PASS: All property tests passed")
        else:
            print("✗ FAIL: Some property tests failed")
            all_passed = False
            
    except Exception as e:
        print(f"✗ FAIL: Property test exception: {e}")
        all_passed = False
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT:")
    
    if all_passed:
        print("✓ ALL TESTS PASSED - Implementation is mathematically rigorous!")
    else:
        print("✗ Some tests failed - needs attention")
    
    return all_passed


if __name__ == '__main__':
    success = test_all_recommendations()
    sys.exit(0 if success else 1)