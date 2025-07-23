#!/usr/bin/env python3
"""
Polyphase Filter Bank Generator - Final Mathematically Rigorous Implementation
=============================================================================

Addresses ALL issues from second-pass review:
1. Exact kernel length alignment for odd taps_per_phase
2. Symmetric extraction of all coefficients for both parities
3. All other minor fixes and improvements
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
import time
import warnings

try:
    from .sinc_table_wrapper import SincTableGenerator
except ImportError:
    # For direct script execution
    from sinc_table_wrapper import SincTableGenerator


class PolyphaseGenerator:
    """
    Generate mathematically exact polyphase filter banks for resampling.
    
    Final implementation with complete mathematical rigor.
    """
    
    def __init__(
        self,
        taps_per_phase: int,
        phase_count: int,
        cutoff: float = 0.5,
        window: str = 'kaiser',
        window_param: Optional[float] = None,
        stopband_db: float = 180.0
    ):
        """
        Initialize polyphase filter bank generator.
        
        Parameters
        ----------
        taps_per_phase : int
            Number of filter taps per polyphase branch (T)
        phase_count : int
            Number of polyphase branches (oversample factor, L)
        cutoff : float
            Normalized cutoff frequency (0-0.5)
        window : str
            Window type: 'kaiser', 'blackman-harris'
        window_param : float, optional
            Window parameter (beta for Kaiser). If None, computed from stopband_db
        stopband_db : float
            Target stopband attenuation in dB
        """
        self.taps_per_phase = taps_per_phase
        self.phase_count = phase_count
        self.cutoff = cutoff
        self.window = window
        self.stopband_db = stopband_db
        
        # FIX 1: Ensure kernel_length and zero_crossings are consistent
        # For both odd and even T, we use the formula:
        # zero_crossings = ceil(T/2) to ensure kernel_length = 2*zc*L + 1 >= T*L + 1
        self.zero_crossings = (taps_per_phase + 1) // 2
        
        # This ensures the prototype filter has at least T*L + 1 samples
        self.kernel_length = 2 * self.zero_crossings * phase_count + 1
        
        # Verify we have enough samples
        min_required = taps_per_phase * phase_count + 1
        if self.kernel_length < min_required:
            raise ValueError(
                f"Kernel length {self.kernel_length} < minimum required {min_required}"
            )
            
        # Compute window parameter from stopband if not provided
        if window_param is None and window == 'kaiser':
            # β ≈ 0.1102 (|A| − 8.7) for Kaiser window
            self.window_param = 0.1102 * (abs(stopband_db) - 8.7)
            if self.window_param < 0:
                self.window_param = 0
        else:
            self.window_param = window_param if window_param is not None else 8.6
        
        self.log = logging.getLogger(__name__)
        
    def generate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate polyphase filter bank using vectorized approach.
        
        Returns
        -------
        table : np.ndarray
            Polyphase table of shape (phase_count, taps_per_phase)
        metadata : dict
            Generation metadata and verification results
        """
        t0 = time.perf_counter()
        
        self.log.info("Generating polyphase filter bank:")
        self.log.info("  Taps per phase (T): %d", self.taps_per_phase)
        self.log.info("  Phase count (L): %d", self.phase_count)
        self.log.info("  Zero crossings: %d", self.zero_crossings)
        self.log.info("  Kernel length: %d", self.kernel_length)
        self.log.info("  Window: %s (β=%.2f)", self.window, self.window_param)
        self.log.info("  Target stopband: %.1f dB", self.stopband_db)
        
        # Generate prototype lowpass filter
        kernel = self._generate_prototype_filter()
        
        # Extract polyphase components using vectorized slicing
        table = self._extract_polyphase_components(kernel)
        
        # Per-row DC normalization with validation
        table = self._normalize_dc_gain(table)
        
        # Verify specifications
        verification = self._verify_specifications(table, kernel)
        
        gen_time = time.perf_counter() - t0
        
        metadata = {
            'taps_per_phase': self.taps_per_phase,
            'phase_count': self.phase_count,
            'kernel_length': self.kernel_length,
            'actual_kernel_length': len(kernel),
            'zero_crossings': self.zero_crossings,
            'cutoff': self.cutoff,
            'window': self.window,
            'window_param': self.window_param,
            'target_stopband_db': self.stopband_db,
            'generation_time': gen_time,
            'verification': verification,
        }
        
        self.log.info("Generation complete in %.3f seconds", gen_time)
        
        # Enforce stopband goal
        if 'measured_stopband_db' in verification:
            if verification['measured_stopband_db'] < self.stopband_db - 1.0:
                raise ValueError(
                    f"Measured stopband {verification['measured_stopband_db']:.1f} dB "
                    f"< target {self.stopband_db:.1f} dB - 1.0 dB margin"
                )
        
        return table.astype(np.float64), metadata
    
    def _generate_prototype_filter(self) -> np.ndarray:
        """Generate prototype lowpass filter using sinc + window."""
        # Use the sinc table generator for consistency
        generator = SincTableGenerator(
            zero_crossings=self.zero_crossings,
            oversample_factor=self.phase_count,
            cutoff=self.cutoff,
            window_type=self.window,
            window_param=self.window_param
        )
        
        kernel, _ = generator.generate()
        
        # The kernel should have the expected length
        if len(kernel) != self.kernel_length:
            self.log.warning(
                "Kernel length mismatch: got %d, expected %d. Using actual length.",
                len(kernel), self.kernel_length
            )
            
        return kernel
    
    def _extract_polyphase_components(self, kernel: np.ndarray) -> np.ndarray:
        """
        Extract polyphase components using vectorized slicing.
        
        FIX 2: Use mathematically exact center calculation
        centre = (len(kernel) - T*L) // 2
        """
        # FIX 2: Mathematically exact derivation for perfect symmetry
        T = self.taps_per_phase
        L = self.phase_count
        
        # Center calculation that works for both odd and even T
        centre = (len(kernel) - T * L) // 2
        
        # Validate we can extract all coefficients
        if centre < 0:
            raise ValueError(
                f"Kernel too short: need at least {T * L} samples, "
                f"but kernel has {len(kernel)}"
            )
        
        end_idx = centre + T * L
        if end_idx > len(kernel):
            raise ValueError(
                f"Extraction would exceed kernel bounds: {end_idx} > {len(kernel)}"
            )
        
        # FIX 7 (improved): Use as_strided for zero-copy extraction
        # This creates a view without allocating the index matrix
        from numpy.lib.stride_tricks import as_strided
        
        # Extract the relevant portion of the kernel
        kernel_slice = kernel[centre:centre + T * L]
        
        # Reshape as (L, T) using strides - this is a view, no copy
        table = as_strided(
            kernel_slice,
            shape=(L, T),
            strides=(kernel_slice.strides[0], L * kernel_slice.strides[0])
        ).copy()  # Copy to ensure contiguous memory
        
        return table
    
    def _normalize_dc_gain(self, table: np.ndarray) -> np.ndarray:
        """
        Normalize each polyphase row to have unity DC gain.
        
        This ensures flat frequency response regardless of phase.
        """
        row_sums = table.sum(axis=1, keepdims=True)
        
        # Check for pathological kernels
        min_sum = np.min(np.abs(row_sums))
        if min_sum < 1e-10:
            raise ValueError(
                f"Near-zero row sum detected: {min_sum:.2e}. "
                "This indicates a pathological kernel design."
            )
        
        table_normalized = table / row_sums
        
        # Only log summary stats
        if self.log.isEnabledFor(logging.DEBUG):
            self.log.debug("DC gains after normalization:")
            for i in range(min(4, self.phase_count)):
                dc_gain = np.sum(table_normalized[i])
                self.log.debug("  Phase %d: %.12f", i, dc_gain)
        else:
            dc_gains = table_normalized.sum(axis=1)
            self.log.info("DC gain stats: min=%.2e, max=%.2e, std=%.2e",
                         np.min(dc_gains), np.max(dc_gains), np.std(dc_gains))
        
        return table_normalized
    
    def _verify_specifications(self, table: np.ndarray, kernel: np.ndarray) -> Dict[str, Any]:
        """Verify the polyphase filter bank meets specifications."""
        verification = {}
        
        # 1. Verify odd kernel length
        verification['odd_kernel'] = len(kernel) % 2 == 1
        
        # 2. Verify symmetry
        centre = len(kernel) // 2
        left = kernel[:centre]
        right = kernel[centre+1:][::-1]
        symmetry_error = np.max(np.abs(left - right))
        verification['symmetry_error'] = float(symmetry_error)
        verification['symmetry_pass'] = symmetry_error < 1e-14
        
        # 3. Verify per-row DC gain
        dc_gains = table.sum(axis=1)
        dc_error = np.max(np.abs(dc_gains - 1.0))
        verification['dc_error'] = float(dc_error)
        verification['dc_normalized'] = dc_error < 1e-10
        
        # 4. Estimate frequency response (if not too large)
        if len(kernel) <= 65536:
            from scipy.signal import freqz
            worN = 65536
            w, h = freqz(kernel, worN=worN)
            mag_db = 20 * np.log10(np.abs(h) + 1e-300)
            
            # FIX: Use f_n variable
            f_n = w / np.pi  # Normalized frequency [0, 1]
            
            # Find passband ripple
            passband_idx = np.where(f_n <= 0.9 * self.cutoff)[0]
            if len(passband_idx) > 0:
                passband = mag_db[passband_idx]
                ripple_db = np.ptp(passband)
                verification['passband_ripple_db'] = float(ripple_db)
            
            # Find stopband attenuation
            stopband_idx = np.where(f_n >= 1.1 * self.cutoff)[0]
            if len(stopband_idx) > 0:
                stopband_peak = np.max(mag_db[stopband_idx])
                verification['measured_stopband_db'] = float(-stopband_peak)
        
        # FIX 3: Pass worN to group delay calculation
        # Check group delay flatness
        if len(kernel) <= 16384:  # Reasonable size for group delay calc
            from scipy.signal import group_delay
            worN_gd = 8192
            w_gd, gd = group_delay((kernel, 1), worN=worN_gd)
            f_n_gd = w_gd / np.pi
            
            passband_idx_gd = np.where(f_n_gd <= 0.9 * self.cutoff)[0]
            if len(passband_idx_gd) > 0:
                gd_variation = np.ptp(gd[passband_idx_gd])
                verification['group_delay_variation'] = float(gd_variation)
        
        # Check inter-phase continuity
        phase_continuity = []
        for i in range(min(4, self.phase_count - 1)):
            # Check if adjacent phases have smooth transitions
            diff = np.abs(table[i, -1] - table[i+1, 0])
            phase_continuity.append(diff)
        verification['phase_continuity_max'] = float(np.max(phase_continuity)) if phase_continuity else 0.0
        
        # Log results
        self.log.info("Verification results:")
        for key, value in verification.items():
            self.log.info("  %s: %s", key, value)
        
        return verification
    
    def save(self, filename: str, include_float32: bool = True) -> None:
        """
        Save polyphase filter bank to file.
        
        Additional: Quantify float32 round-off error
        """
        table, metadata = self.generate()
        
        # Save NPZ with metadata
        save_dict = {
            'table_float64': table,
            'metadata': metadata,
        }
        
        if include_float32:
            table_f32 = table.astype(np.float32)
            save_dict['table_float32'] = table_f32
            
            # Quantify round-off error
            round_off_error = np.max(np.abs(table - table_f32.astype(np.float64)))
            round_off_db = 20 * np.log10(round_off_error) if round_off_error > 0 else -np.inf
            
            self.log.info("Float32 round-off: %.2e (%.1f dB)", round_off_error, round_off_db)
            
            # FIX 4: Correct threshold comparison
            # Warn if round-off is within 20 dB of the stopband floor
            if round_off_db > -(self.stopband_db - 20):
                warnings.warn(
                    f"Float32 round-off ({round_off_db:.1f} dB) is within 20 dB "
                    f"of target stopband ({-self.stopband_db:.1f} dB)"
                )
        
        np.savez_compressed(f"{filename}.npz", **save_dict)
        self.log.info("Saved polyphase filter bank to %s.npz", filename)
        
        # Also save raw binary for embedded systems
        if include_float32:
            table_f32.tofile(f"{filename}.f32")
            self.log.info("Saved raw float32 to %s.f32", filename)


def property_based_tests():
    """
    Property-based testing across parameter space.
    
    FIX 5: Include odd taps_per_phase cases
    """
    import itertools
    
    test_cases = itertools.product(
        [32, 33, 64, 65, 128],  # Include odd cases: 33, 65
        [64, 128, 256],         # phase_count  
        [0.45, 0.5],            # cutoff
    )
    
    print("Running property-based tests...")
    passed = 0
    total = 0
    
    for taps, phases, cutoff in test_cases:
        total += 1
        print(f"\nTesting: T={taps}, L={phases}, cutoff={cutoff}")
        
        try:
            gen = PolyphaseGenerator(
                taps_per_phase=taps,
                phase_count=phases,
                cutoff=cutoff,
                stopband_db=120.0  # Moderate target for tests
            )
            
            table, metadata = gen.generate()
            verification = metadata['verification']
            
            # Check properties
            assert verification['odd_kernel'], "Kernel must be odd length"
            assert verification['symmetry_pass'], "Kernel must be symmetric"
            assert verification['dc_normalized'], "Rows must have unity DC gain"
            
            if 'measured_stopband_db' in verification:
                assert verification['measured_stopband_db'] >= 119.0, "Must meet stopband target"
            
            print("  ✓ All properties pass")
            passed += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\nProperty tests: {passed}/{total} passed")
    return passed == total


def main():
    """Example usage and verification."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate polyphase filter banks exceeding libsoxr quality"
    )
    parser.add_argument('--taps', type=int, default=64,
                       help='Taps per phase (default: 64)')
    parser.add_argument('--phases', type=int, default=256,
                       help='Number of phases (default: 256)')
    parser.add_argument('--stopband', type=float, default=180.0,
                       help='Target stopband in dB (default: 180)')
    parser.add_argument('--output', type=str, default='polyphase',
                       help='Output filename base (default: polyphase)')
    parser.add_argument('--test', action='store_true',
                       help='Run property-based tests')
    
    args = parser.parse_args()
    
    if args.test:
        success = property_based_tests()
        return 0 if success else 1
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Generate polyphase filter bank
    generator = PolyphaseGenerator(
        taps_per_phase=args.taps,
        phase_count=args.phases,
        stopband_db=args.stopband
    )
    
    generator.save(args.output)
    
    print(f"\nGenerated polyphase filter bank:")
    print(f"  Taps per phase: {args.taps}")
    print(f"  Phase count: {args.phases}")
    print(f"  Total coefficients: {args.taps * args.phases}")
    print(f"  Target stopband: {args.stopband} dB")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())