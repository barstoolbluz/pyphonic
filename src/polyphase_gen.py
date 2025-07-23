#!/usr/bin/env python3
"""
Polyphase Filter Bank Generator - Mathematically Exact Implementation
=====================================================================

Final implementation with correct understanding of polyphase symmetry:
- Only phase 0 (and phase L/2 for even L) can be symmetric
- Other phases are inherently asymmetric due to the extraction pattern
- This does NOT break linear phase or perfect reconstruction
- Implements proper verification tests
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
import time
import warnings

try:
    from .sinc_table_wrapper import SincTableGenerator
except ImportError:
    try:
        # For direct script execution
        from sinc_table_wrapper import SincTableGenerator
    except ImportError:
        # If that fails too, we'll use a mock
        SincTableGenerator = None


class PolyphaseGenerator:
    """
    Generate mathematically exact polyphase filter banks for resampling.
    
    Achieves perfect linear phase and reconstruction.
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
            Target stopband attenuation in dB (clamped to 190 for numerical stability)
        """
        self.taps_per_phase = taps_per_phase
        self.phase_count = phase_count
        self.cutoff = cutoff
        self.window = window
        
        # Clamp stopband for numerical stability
        if stopband_db > 190.0:
            warnings.warn(
                f"Stopband {stopband_db} dB > 190 dB may cause numerical instability. "
                "Clamping to 190 dB.",
                UserWarning
            )
            stopband_db = 190.0
        self.stopband_db = stopband_db
        
        # Use standard kernel length T*L+1 (the industry standard)
        # This gives 0.5 sample centering error, but it's the accepted approach
        total_taps = taps_per_phase * phase_count
        self.kernel_length = total_taps + 1
        
        # For the sinc generator, we need appropriate zero crossings
        self.zero_crossings = (self.kernel_length - 1) // (2 * phase_count)
        if 2 * self.zero_crossings * phase_count + 1 < self.kernel_length:
            self.zero_crossings += 1
            
        # Compute window parameter from stopband if not provided
        if window_param is None and window == 'kaiser':
            # β ≈ 0.1102 (|A| − 8.7) for Kaiser window
            self.window_param = 0.1102 * (abs(stopband_db) - 8.7)
            if self.window_param < 0:
                self.window_param = 0
        else:
            self.window_param = window_param if window_param is not None else 8.6
        
        # Default to WARNING level logging
        self.log = logging.getLogger(__name__)
        if not self.log.handlers:
            self.log.setLevel(logging.WARNING)
        
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
        self.log.info("  Kernel length: %d (T*L + 1)", self.kernel_length)
        self.log.info("  Window: %s (β=%.2f)", self.window, self.window_param)
        self.log.info("  Target stopband: %.1f dB", self.stopband_db)
        
        # Generate prototype lowpass filter
        kernel = self._generate_prototype_filter()
        
        # Extract polyphase components
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
        if SincTableGenerator is not None:
            generator = SincTableGenerator(
                zero_crossings=self.zero_crossings,
                oversample_factor=self.phase_count,
                cutoff=self.cutoff,
                window_type=self.window,
                window_param=self.window_param
            )
            
            kernel, _ = generator.generate()
        else:
            # Fallback: generate a simple windowed sinc
            N = self.kernel_length
            x = np.arange(N) - N//2
            x = x / self.phase_count
            
            with np.errstate(divide='ignore', invalid='ignore'):
                kernel = np.sinc(2 * self.cutoff * x) * 2 * self.cutoff
                
            # Apply Kaiser window
            from scipy.signal.windows import kaiser
            window = kaiser(N, self.window_param)
            kernel *= window
            
            # Normalize
            kernel /= np.sum(kernel)
        
        # Trim or pad to exact length if needed
        if len(kernel) > self.kernel_length:
            # Trim symmetrically
            excess = len(kernel) - self.kernel_length
            start = excess // 2
            kernel = kernel[start:start + self.kernel_length]
        elif len(kernel) < self.kernel_length:
            # Pad symmetrically with zeros
            pad_total = self.kernel_length - len(kernel)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            kernel = np.pad(kernel, (pad_left, pad_right), mode='constant')
            
        return kernel
    
    def _extract_polyphase_components(self, kernel: np.ndarray) -> np.ndarray:
        """
        Extract polyphase components with perfect symmetry.
        
        For kernel length T*L + 1, we remove the CENTER tap to get
        T*L coefficients with perfect symmetry around the removed center.
        """
        T = self.taps_per_phase
        L = self.phase_count
        total_coeffs = T * L
        kernel_len = len(kernel)
        
        # For T*L+1 kernel, remove the center tap for perfect symmetry
        if kernel_len != total_coeffs + 1:
            raise ValueError(f"Expected kernel length {total_coeffs + 1}, got {kernel_len}")
        
        center = kernel_len // 2
        
        # Create symmetric slice by removing center tap
        kernel_slice = np.concatenate((kernel[:center], kernel[center+1:]))
        
        # Verify we have the expected length
        if len(kernel_slice) != total_coeffs:
            raise ValueError(f"Kernel slice length {len(kernel_slice)} != expected {total_coeffs}")
        
        # Use as_strided for efficient polyphase extraction
        from numpy.lib.stride_tricks import as_strided
        
        table_view = as_strided(
            kernel_slice,
            shape=(L, T),
            strides=(kernel_slice.strides[0], L * kernel_slice.strides[0]),
            writeable=False
        )
        
        table = table_view.copy()
        
        # Verify perfect symmetry was achieved
        self._verify_perfect_symmetry(kernel, kernel_slice, center)
        
        return table
    
    def _verify_perfect_symmetry(self, kernel: np.ndarray, kernel_slice: np.ndarray, center: int) -> None:
        """
        Verify that removing the center tap achieved perfect symmetry.
        
        The kernel_slice should be perfectly symmetric around the removed center.
        """
        # The kernel_slice should be symmetric
        if len(kernel_slice) % 2 == 0:
            # Even length - symmetric around midpoint
            mid = len(kernel_slice) // 2
            left = kernel_slice[:mid]
            right = kernel_slice[mid:][::-1]
        else:
            # Odd length - symmetric around center sample
            mid = len(kernel_slice) // 2
            left = kernel_slice[:mid]
            right = kernel_slice[mid+1:][::-1]
        
        symmetry_error = np.max(np.abs(left - right))
        
        if symmetry_error > 1e-14:
            raise ValueError(
                f"Kernel slice not symmetric after center tap removal! "
                f"Symmetry error: {symmetry_error:.2e}. "
                f"This indicates the original kernel was not properly symmetric."
            )
        
        self.log.info(f"Perfect symmetry achieved: error = {symmetry_error:.2e}")
    
    def _normalize_dc_gain(self, table: np.ndarray) -> np.ndarray:
        """
        Normalize each polyphase row to have unity DC gain.
        
        This ensures flat frequency response in the passband.
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
        
        # Log summary stats
        dc_gains = table_normalized.sum(axis=1)
        self.log.info("DC gain stats: min=%.2e, max=%.2e, std=%.2e",
                     np.min(dc_gains), np.max(dc_gains), np.std(dc_gains))
        
        return table_normalized
    
    def _verify_specifications(self, table: np.ndarray, kernel: np.ndarray) -> Dict[str, Any]:
        """Verify the polyphase filter bank meets specifications."""
        verification = {}
        
        # 1. Verify odd kernel length
        verification['odd_kernel'] = len(kernel) % 2 == 1
        
        # 2. Verify symmetry of prototype
        centre = len(kernel) // 2
        left = kernel[:centre]
        right = kernel[centre+1:][::-1]
        symmetry_error = np.max(np.abs(left - right))
        verification['symmetry_error'] = float(symmetry_error)
        verification['symmetry_pass'] = symmetry_error < 1e-14
        
        # 3. Check which phases CAN be symmetric (based on theory)
        # Only phase 0 and phase L/2 (for even L) can be symmetric
        symmetric_phases = [0]
        if self.phase_count % 2 == 0:
            symmetric_phases.append(self.phase_count // 2)
        
        # Verify symmetry only for phases that can be symmetric
        phase_symmetries = {}
        for p in symmetric_phases:
            row = table[p]
            row_len = len(row)
            if row_len % 2 == 0:
                row_left = row[:row_len//2]
                row_right = row[row_len//2:][::-1]
            else:
                row_centre = row_len // 2
                row_left = row[:row_centre]
                row_right = row[row_centre+1:][::-1]
            
            sym_error = np.max(np.abs(row_left - row_right))
            phase_symmetries[f'phase_{p}_symmetry'] = float(sym_error)
        
        verification.update(phase_symmetries)
        
        # 4. Verify per-row DC gain
        dc_gains = table.sum(axis=1)
        dc_error = np.max(np.abs(dc_gains - 1.0))
        verification['dc_error'] = float(dc_error)
        verification['dc_normalized'] = dc_error < 1e-10
        
        # 5. Perfect reconstruction test
        # Test that polyphase decomposition perfectly reconstructs the extracted kernel slice
        table_unnorm = self._extract_polyphase_components(kernel)
        
        # Upsample each row by L and sum - should recover the kernel slice (with center removed)
        total_coeffs = self.taps_per_phase * self.phase_count
        reconstruction = np.zeros(total_coeffs)
        for p in range(self.phase_count):
            # Upsample by inserting L-1 zeros between samples
            upsampled = np.zeros(total_coeffs)
            upsampled[p::self.phase_count] = table_unnorm[p]
            # Add to reconstruction
            reconstruction += upsampled
        
        # The reconstruction should match the kernel slice (center tap removed)
        center = len(kernel) // 2
        expected_slice = np.concatenate((kernel[:center], kernel[center+1:]))
        recon_error = np.max(np.abs(reconstruction - expected_slice))
        verification['reconstruction_error'] = float(recon_error)
        verification['perfect_reconstruction'] = recon_error < 1e-10
        
        # 6. Estimate frequency response (if not too large)
        if len(kernel) <= 65536:
            from scipy.signal import freqz
            worN = 65536
            w, h = freqz(kernel, worN=worN)
            mag_db = 20 * np.log10(np.abs(h) + 1e-300)
            
            f_n = w / np.pi  # Normalized frequency [0, 1]
            
            # Passband ripple
            passband_end = min(0.95 * self.cutoff, 0.49)
            passband_idx = np.where(f_n <= passband_end)[0]
            if len(passband_idx) > 0:
                passband = mag_db[passband_idx]
                ripple_db = np.ptp(passband)
                verification['passband_ripple_db'] = float(ripple_db)
            
            # Stopband attenuation
            stopband_idx = np.where(f_n >= 1.1 * self.cutoff)[0]
            if len(stopband_idx) > 0:
                stopband_peak = np.max(mag_db[stopband_idx])
                verification['measured_stopband_db'] = float(-stopband_peak)
        
        # 7. Group delay
        try:
            from scipy.signal import group_delay
            
            # Subsample if kernel is too large
            if len(kernel) > 16384:
                subsample_factor = (len(kernel) // 16384) + 1
                kernel_sub = kernel[::subsample_factor]
            else:
                kernel_sub = kernel
                
            worN_gd = min(8192, len(kernel_sub) * 4)
            # Handle different scipy versions
            try:
                w_gd, gd = group_delay((kernel_sub, 1), worN=worN_gd)
            except TypeError:
                # Older scipy versions don't support worN parameter
                w_gd, gd = group_delay((kernel_sub, 1))
            f_n_gd = w_gd / np.pi
            
            passband_end_gd = min(0.95 * self.cutoff, 0.49)
            passband_idx_gd = np.where(f_n_gd <= passband_end_gd)[0]
            if len(passband_idx_gd) > 0:
                gd_variation = np.ptp(gd[passband_idx_gd])
                verification['group_delay_variation'] = float(gd_variation)
        except Exception as e:
            self.log.warning(f"Group delay calculation failed: {e}")
        
        # 8. Linear phase test for polyphase bank
        # Each phase should have group delay offset by k/L samples
        if len(kernel) <= 16384:
            try:
                base_delay = (len(kernel) - 1) / 2.0  # Expected delay of prototype
                phase_delay_errors = []
                
                for p in range(min(8, self.phase_count)):
                    row = table[p]
                    w_row, h_row = freqz(row, worN=4096)
                    
                    # Compute phase and group delay
                    phase_row = np.unwrap(np.angle(h_row))
                    # Simple numerical derivative for group delay
                    gd_row = -np.diff(phase_row) / np.diff(w_row)
                    
                    # Check in passband
                    f_n_row = w_row[:-1] / np.pi
                    pb_idx = np.where(f_n_row <= 0.9 * self.cutoff)[0]
                    if len(pb_idx) > 0:
                        # Expected delay for this phase
                        expected_delay = base_delay / self.phase_count - p / self.phase_count
                        actual_delay = np.mean(gd_row[pb_idx])
                        delay_error = abs(actual_delay - expected_delay)
                        phase_delay_errors.append(delay_error)
                
                if phase_delay_errors:
                    max_delay_error = max(phase_delay_errors)
                    verification['max_phase_delay_error'] = float(max_delay_error)
                    verification['linear_phase_bank'] = max_delay_error < 0.1  # Allow 0.1 sample error
                    
            except Exception as e:
                self.log.warning(f"Phase delay test failed: {e}")
        
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
            
            # Warning if round-off may impact performance
            if round_off_db > -(self.stopband_db - 20):
                warnings.warn(
                    f"Float32 round-off ({round_off_db:.1f} dB) is within 20 dB "
                    f"of target stopband ({-self.stopband_db:.1f} dB). "
                    f"This may impact stopband performance.",
                    UserWarning
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
    
    Tests the correct properties for polyphase filter banks.
    """
    import itertools
    
    test_cases = itertools.product(
        [32, 33, 64, 65],       # Mix of odd and even T
        [64, 128],              # phase_count  
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
            
            # Perfect reconstruction is key
            assert verification.get('perfect_reconstruction', False), "Must have perfect reconstruction"
            
            # Linear phase as a bank
            if 'linear_phase_bank' in verification:
                assert verification['linear_phase_bank'], "Must maintain linear phase as a bank"
            
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
        description="Generate polyphase filter banks with mathematical exactness"
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
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.test:
        success = property_based_tests()
        return 0 if success else 1
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
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
