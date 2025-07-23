#!/usr/bin/env python3
"""
Sinc Interpolation Table Generator - Fully Vectorized Implementation
===================================================================

Based on the concrete recipe for proper vectorization:
- Generates only right half and mirrors for perfect symmetry
- Uses np.sinc for stable near-zero computation
- Fully vectorized with no Python loops
- Produces polyphase table format (oversample, N_taps)
"""

import argparse
import logging
import sys
import time
import numpy as np
from scipy.special import i0 as bessel_i0
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def build_window(M: int, win_type: str = "kaiser", beta: float = 8.6) -> np.ndarray:
    """
    Return an M-point full window.
    
    Parameters:
    -----------
    M : int
        Full window length
    win_type : str
        Window type: 'kaiser', 'hann', 'hamming', 'blackman-harris'
    beta : float
        Kaiser window beta parameter (only for kaiser)
        
    Returns:
    --------
    np.ndarray
        Full window
    """
    n = np.arange(M, dtype=np.float64)
    
    if win_type == "hann":
        # Raised-cosine: 0.5 - 0.5*cos(2πn/(M-1))
        w = 0.5 - 0.5 * np.cos(2 * np.pi * n / (M - 1))
    elif win_type == "hamming":
        w = 0.54 - 0.46 * np.cos(2 * np.pi * n / (M - 1))
    elif win_type == "blackman-harris":
        # 4-term Blackman-Harris
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        x = 2 * np.pi * n / (M - 1)
        w = a0 - a1*np.cos(x) + a2*np.cos(2*x) - a3*np.cos(3*x)
    elif win_type == "kaiser":
        # Kaiser-Bessel window
        w = bessel_i0(beta * np.sqrt(1 - (2*n/(M-1) - 1)**2)) / bessel_i0(beta)
    elif win_type == "rectangular":
        w = np.ones_like(n)
    else:
        raise ValueError(f"Unsupported window type: {win_type}")
    
    return w


def kaiser_beta_to_attenuation(beta: float) -> float:
    """
    Convert Kaiser β to stopband attenuation using classical approximation.
    
    Note: This is an approximation accurate to ±0.5 dB. For strict spec compliance,
    use Newton iterations to invert the exact Kaiser attenuation equation.
    """
    if beta > 8.7:
        return 2.285 * (beta + 0.1)
    elif beta > 0:
        return -20 * np.log10(0.5842 * (beta / 2.09) ** 0.4 + 0.07886 * (beta / 2.09))
    else:
        return 21.0


def generate_sinc_table(
    N_taps: int,
    oversample: int,
    cutoff: float = 0.5,
    win_type: str = "kaiser",
    beta: float = 8.6,
    log: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Vectorized, even-symmetric sinc-table generator.
    
    Parameters:
    -----------
    N_taps : int
        Number of zero crossings on each side (NOT the number of non-zero lobes).
        For textbook 2·N·R + 1 length, this excludes outermost ±N_taps positions
        where sinc is zero. Current implementation uses even length 2·N·R, giving
        half-integer group delay of N·R - 0.5 samples.
    oversample : int
        Oversampling factor (R)
    cutoff : float
        Normalized cutoff frequency (0-0.5, default 0.5)
    win_type : str
        Window type
    beta : float
        Kaiser window beta parameter
    log : Logger
        Optional logger
        
    Returns:
    --------
    table : np.ndarray
        Shape (N_taps * oversample * 2,) - full symmetric table
    metadata : dict
        Generation metadata
    """
    if log is None:
        log = logging.getLogger(__name__)
    
    # Edge-case guards
    assert oversample >= 1, "Oversample factor must be >= 1"
    assert N_taps >= 1, "N_taps (zero crossings) must be >= 1"
    assert 0 < cutoff <= 0.5, "Cutoff must be normalized to Nyquist (0 < cutoff <= 0.5)"
    
    t0 = time.perf_counter()
    
    # Total table length
    table_length = N_taps * oversample * 2
    
    log.info("Generating sinc table (vectorized):")
    log.info("  Zero crossings: ±%d", N_taps)
    log.info("  Oversample factor: %d", oversample)
    log.info("  Table size: %d samples", table_length)
    log.info("  Cutoff: %.6f (normalized)", cutoff)
    log.info("  Window: %s", win_type)
    
    # 1. Coordinates for the *right* half (center to +end)
    # For a table of length N, we generate N//2 + 1 points
    half_len = table_length // 2 + 1  # Include center sample
    x = np.arange(half_len, dtype=np.float64) / oversample
    
    # 2. Core sinc (np.sinc is sin(πx)/(πx))
    # np.sinc handles x=0 case internally with proper limit
    sinc_core = 2 * cutoff * np.sinc(2 * cutoff * x)
    
    # 3. Window (same length as x)
    # Build full window and extract the right half including center
    window_params = {'beta': beta} if win_type == 'kaiser' else {}
    full_window = build_window(table_length, win_type, beta)
    
    # Extract right half using clear symmetric approach
    if table_length % 2 == 0:
        # Even length: use flip for clarity and avoid extra copy
        half_len = table_length // 2 + 1  # Include center
        w = np.flip(full_window[:half_len])
    else:
        # Odd length: extract from center (inclusive) to end
        w = full_window[table_length//2:]
    
    # Verify we have the correct length to match x
    assert len(w) == len(x), f"Window length {len(w)} != sinc length {len(x)}"
    
    kernel_right = sinc_core * w
    
    # 4. Mirror to get full kernel (perfect symmetry guaranteed)
    # For even length tables, we exclude the center when mirroring
    if table_length % 2 == 0:
        kernel = np.concatenate((kernel_right[-2::-1], kernel_right[:-1]))
    else:
        kernel = np.concatenate((kernel_right[-2::-1], kernel_right))
    
    # 5. Normalize for unity DC gain
    # For interpolation, we sum taps at integer positions (excludes outermost zeros)
    center = table_length // 2
    norm_indices = center + np.arange(-N_taps, N_taps) * oversample
    valid = (norm_indices >= 0) & (norm_indices < table_length)
    dc_sum = np.sum(kernel[norm_indices[valid]])
    
    # Tighter threshold: worst-case rounding for 4k samples ~1e-12 in float64
    if abs(dc_sum) > 1e-12:
        kernel /= dc_sum
        log.info("Normalized by factor %.15e", dc_sum)
    else:
        log.warning("DC sum %.2e below normalization threshold 1e-12", abs(dc_sum))
    
    gen_time = time.perf_counter() - t0
    log.info("Table generated in %.3f seconds (fully vectorized)", gen_time)
    
    # Verify perfect symmetry
    center = len(kernel) // 2
    left = kernel[:center]
    right = kernel[center:][::-1]
    symmetry_error = np.max(np.abs(left - right))
    log.info("Symmetry error: %.2e (should be ~machine epsilon)", symmetry_error)
    
    # Build metadata
    metadata = {
        'zero_crossings': N_taps,
        'oversample_factor': oversample,
        'cutoff': cutoff,
        'window_type': win_type,
        'window_params': window_params,
        'table_size': table_length,
        'generation_time': gen_time,
        'normalization_factor': dc_sum if abs(dc_sum) > 1e-12 else 1.0,
        'symmetry_error': float(symmetry_error),
        'symmetry_guaranteed': True,
        'generator_version': 'vectorized',
    }
    
    # Add window-specific metadata
    if win_type == 'kaiser':
        metadata['kaiser_beta'] = beta
        metadata['theoretical_stopband_db'] = kaiser_beta_to_attenuation(beta)
    
    return kernel.astype(np.float64), metadata


def generate_polyphase_table(
    N_taps: int,
    oversample: int,
    cutoff: float = 0.5,
    win_type: str = "kaiser",
    beta: float = 8.6,
    normalize_phases: bool = False,
    log: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate polyphase sinc table in (oversample, N_taps) format.
    Each row is a phase for a specific fractional delay.
    """
    if log is None:
        log = logging.getLogger(__name__)
        
    # Generate the full table
    kernel, metadata = generate_sinc_table(N_taps, oversample, cutoff, win_type, beta, log)
    
    # The polyphase decomposition requires careful indexing
    # For fractional delay d/oversample, we need samples at positions:
    # ..., -2*oversample+d, -oversample+d, d, oversample+d, 2*oversample+d, ...
    
    # Create polyphase table using vectorized extraction
    center = len(kernel) // 2
    
    # Ensure symmetric indexing works correctly
    assert N_taps % 2 == 0, f"N_taps must be even for symmetric polyphase extraction, got {N_taps}"
    
    # Vectorized approach: use broadcasting to create all indices at once
    # Note: For very large oversample factors (>10^4), consider chunking to reduce memory
    taps_range = np.arange(N_taps).reshape(1, -1)  # (1, N_taps)
    phases_range = np.arange(oversample).reshape(-1, 1)  # (oversample, 1)
    
    # Compute all kernel indices using broadcasting (centered around middle)
    kernel_indices = center + (taps_range - N_taps//2) * oversample + phases_range
    
    # Create bounds mask
    valid_mask = (kernel_indices >= 0) & (kernel_indices < len(kernel))
    
    # Initialize table
    table = np.zeros((oversample, N_taps), dtype=np.float64)
    
    # Extract values where valid (vectorized)
    table[valid_mask] = kernel[kernel_indices[valid_mask]]
    
    # Optional: normalize each phase to unity DC gain
    if normalize_phases:
        row_sums = np.sum(table, axis=1, keepdims=True)
        # Avoid division by zero
        valid_rows = np.abs(row_sums.flatten()) > 1e-15
        table[valid_rows] = table[valid_rows] / row_sums[valid_rows]
        log.info("Applied per-phase DC normalization")
    
    # Each row should sum to approximately 1 for DC
    # But only phase 0 will sum to exactly 1 without normalization
    log.info("Polyphase table DC gains:")
    for p in range(min(4, oversample)):
        dc_gain = np.sum(table[p])
        log.info("  Phase %d: %.6f", p, dc_gain)
    
    metadata['format'] = 'polyphase'
    metadata['shape'] = table.shape
    metadata['phases_normalized'] = normalize_phases
    
    # Respect precision - don't force float32 conversion here
    return table, metadata


def verify_sinc_table(
    table: np.ndarray,
    metadata: Dict[str, Any],
    log: logging.Logger
) -> bool:
    """Run comprehensive automated verification tests for production quality."""
    
    log.info("Running comprehensive verification suite...")
    all_pass = True
    
    # For 1D table format
    if table.ndim == 1:
        # Test 1: Perfect symmetry
        center = len(table) // 2
        left = table[:center]
        right = table[center:][::-1]
        
        symmetry_error = np.max(np.abs(left - right))
        symmetry_pass = symmetry_error < 1e-15  # Machine precision limit
        
        log.info("Test 1 - Symmetry: %s (error: %.2e, threshold: 1e-15)", 
                 "PASS" if symmetry_pass else "FAIL", symmetry_error)
        all_pass &= symmetry_pass
        
        # Test 2: Unity DC gain (tightened threshold)
        zc = metadata['zero_crossings']
        osf = metadata['oversample_factor']
        
        norm_indices = center + np.arange(-zc, zc) * osf
        valid = (norm_indices >= 0) & (norm_indices < len(table))
        dc_sum = np.sum(table[norm_indices[valid]])
        
        dc_error = abs(dc_sum - 1.0)
        dc_pass = dc_error < 1e-14  # Tighter for production
        
        log.info("Test 2 - DC gain: %s (sum: %.15e, error: %.2e)", 
                 "PASS" if dc_pass else "FAIL", dc_sum, dc_error)
        all_pass &= dc_pass
        
        # Test 3: Peak near center (allowing for even-length offset)
        peak_idx = np.argmax(np.abs(table))
        # For even-length tables, peak may be at center±0.5
        peak_pass = abs(peak_idx - center) <= 1
        
        log.info("Test 3 - Peak position: %s (at index %d, center %d, offset %d)", 
                 "PASS" if peak_pass else "FAIL", peak_idx, center, peak_idx - center)
        all_pass &= peak_pass
        
        # Test 4: FFT-based stopband verification (NEW)
        if 'window_type' in metadata and metadata['window_type'] == 'kaiser':
            try:
                from scipy.signal import freqz
                
                # Compute frequency response
                w, h = freqz(table, worN=8192)
                mag_db = 20 * np.log10(np.abs(h) + 1e-300)
                
                # Find stopband (beyond cutoff + transition)
                cutoff = metadata.get('cutoff', 0.5)
                # Better transition width estimate for windowed sinc
                # Main lobe width ≈ 4π/N for Kaiser window
                N_taps = metadata['zero_crossings'] 
                transition_est = 4.0 / (N_taps * 2)  # Normalized frequency
                
                freqs_norm = w / np.pi
                stopband_mask = freqs_norm > (cutoff + transition_est)
                
                if stopband_mask.any():
                    actual_stopband = -np.max(mag_db[stopband_mask])
                    
                    # Use theoretical Kaiser prediction as pass criterion
                    if 'kaiser_beta' in metadata:
                        theoretical_stopband = kaiser_beta_to_attenuation(metadata['kaiser_beta'])
                        # Allow reasonable tolerance for practical implementation
                        stopband_pass = actual_stopband >= (theoretical_stopband * 0.8)  # 80% of theory
                        
                        log.info("Test 4 - Stopband: %s (actual: %.1f dB, theory: %.1f dB, need: %.1f dB)", 
                                "PASS" if stopband_pass else "FAIL", 
                                actual_stopband, theoretical_stopband, theoretical_stopband * 0.8)
                        all_pass &= stopband_pass
                    else:
                        # Fallback for non-Kaiser windows
                        reasonable_stopband = actual_stopband > 40.0  
                        log.info("Test 4 - Stopband: %s (actual: %.1f dB, minimum: 40 dB)", 
                                "PASS" if reasonable_stopband else "FAIL", actual_stopband)
                        all_pass &= reasonable_stopband
            except ImportError:
                log.warning("scipy.signal not available, skipping stopband test")
        
        # Test 5: Energy conservation (simplified check)
        # Just verify energy is reasonable (not infinite or zero)
        energy = np.sum(np.abs(table)**2)
        energy_pass = 0.1 < energy < 1e6  # Reasonable bounds
        
        log.info("Test 5 - Energy bounds: %s (energy: %.6e, bounds: [0.1, 1e6])",
                "PASS" if energy_pass else "FAIL", energy)
        all_pass &= energy_pass
    
    else:  # Polyphase format
        log.info("Polyphase table shape: %s", table.shape)
        oversample, n_taps = table.shape
        
        # Test 1: Phase 0 DC gain (relaxed for polyphase)
        phase0_sum = np.sum(table[0])
        dc_error = abs(phase0_sum - 1.0)
        dc_pass = dc_error < 0.002  # 0.2% tolerance for polyphase
        
        log.info("Test 1 - Phase 0 DC gain: %s (sum: %.6f, error: %.2e, tol: 0.2%%)", 
                 "PASS" if dc_pass else "FAIL", phase0_sum, dc_error)
        all_pass &= dc_pass
        
        # Test 2: Per-phase DC error (reasonable tolerance for fractional delays)
        all_dc_gains = np.sum(table, axis=1)
        max_dc_error = np.max(np.abs(all_dc_gains - 1.0))
        all_dc_pass = max_dc_error < 0.02  # 2% tolerance for non-zero phases
        
        log.info("Test 2 - All phases DC: %s (max error: %.4f, worst phase: %d, tol: 2%%)",
                "PASS" if all_dc_pass else "FAIL", 
                max_dc_error, np.argmax(np.abs(all_dc_gains - 1.0)))
        all_pass &= all_dc_pass
        
        # Test 3: Phase continuity - adjacent phases should be similar
        if oversample > 1:
            phase_diffs = np.max(np.abs(np.diff(table, axis=0)), axis=1)
            max_phase_diff = np.max(phase_diffs)
            continuity_pass = max_phase_diff < np.max(np.abs(table)) * 0.5  # 50% of peak
            
            log.info("Test 3 - Phase continuity: %s (max diff: %.6e)",
                    "PASS" if continuity_pass else "FAIL", max_phase_diff)
            all_pass &= continuity_pass
    
    if all_pass:
        log.info("🎯 All verification tests PASSED - filter meets production quality")
    else:
        log.error("❌ Some verification tests FAILED - review filter design")
    
    return all_pass


def save_sinc_table(
    table: np.ndarray,
    metadata: Dict[str, Any],
    basename: str,
    save_format: str,
    save_float64: bool,
    log: logging.Logger
) -> None:
    """Save sinc table with metadata."""
    
    if save_float64:
        npy64_path = f"{basename}_float64.npy"
        np.save(npy64_path, table.astype(np.float64))
        log.info("Saved float64 table to %s", npy64_path)
    
    if save_format in ['npy', 'both']:
        npy_path = f"{basename}.npy"
        np.save(npy_path, table.astype(np.float32))
        log.info("Saved float32 table to %s", npy_path)
    
    if save_format in ['npz', 'both']:
        npz_path = f"{basename}.npz"
        
        save_metadata = metadata.copy()
        save_metadata['command_line'] = ' '.join(sys.argv)
        save_metadata['numpy_version'] = np.__version__
        
        np.savez(npz_path,
                table=table.astype(np.float32),
                table_float64=table.astype(np.float64) if save_float64 else None,
                metadata=save_metadata)
        
        log.info("Saved table and metadata to %s", npz_path)


def main():
    parser = argparse.ArgumentParser(
        description="Fully vectorized sinc table generator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--zeros', '-z', type=int, required=True,
                       help='Number of zero crossings on each side (NOT non-zero lobes). '
                            'Creates even-length table with half-integer group delay.')
    parser.add_argument('--oversample', '-o', type=int, default=512,
                       help='Oversample factor (default: 512)')
    parser.add_argument('--cutoff', '-c', type=float, default=0.5,
                       help='Normalized cutoff frequency (0-0.5, default: 0.5)')
    parser.add_argument('--window', '-w', 
                       choices=['kaiser', 'blackman-harris', 'hann', 'hamming', 'rectangular'],
                       default='kaiser',
                       help='Window function type (default: kaiser)')
    parser.add_argument('--beta', '-b', type=float, default=8.6,
                       help='Kaiser window β parameter (default: 8.6, approx ±0.5 dB)')
    parser.add_argument('--polyphase', '-p', action='store_true',
                       help='Generate polyphase format (oversample, N_taps)')
    parser.add_argument('--normalize-phases', action='store_true',
                       help='Normalize each polyphase row to unity DC gain (stricter than default)')
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float32',
                       help='Output precision (default: float32). Use float64 for mastering/reference applications.')
    
    # Output options
    parser.add_argument('--format', '-f', choices=['npy', 'npz', 'both'], default='npz')
    parser.add_argument('--basename', type=str)
    parser.add_argument('--float64', action='store_true')
    parser.add_argument('--verify', '-v', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    log = logging.getLogger('sinc_vectorized')
    
    # Generate table
    if args.polyphase:
        table, metadata = generate_polyphase_table(
            args.zeros, args.oversample, args.cutoff, 
            args.window, args.beta, args.normalize_phases, log
        )
    else:
        table, metadata = generate_sinc_table(
            args.zeros, args.oversample, args.cutoff,
            args.window, args.beta, log
        )
    
    # Apply precision setting
    if args.precision == 'float64':
        table = table.astype(np.float64)
        args.float64 = True  # For save_sinc_table compatibility
    else:
        table = table.astype(np.float32)
        args.float64 = False
    
    # Verify if requested
    if args.verify:
        if not verify_sinc_table(table, metadata, log):
            log.error("Verification FAILED!")
            sys.exit(1)
        else:
            log.info("All verification tests PASSED")
    
    # Generate output filename
    if args.basename:
        basename = args.basename
    else:
        parts = [f"sinc_{args.zeros}z"]
        if args.oversample != 512:
            parts.append(f"{args.oversample}x")
        if abs(args.cutoff - 0.5) > 0.001:
            parts.append(f"{int(args.cutoff*1000)}c")
        parts.append(args.window)
        if args.window == 'kaiser':
            parts.append(f"b{args.beta:.1f}".replace('.', '_'))
        if args.polyphase:
            parts.append('poly')
        parts.append('vec')
        basename = '_'.join(parts)
    
    # Save
    save_sinc_table(table, metadata, basename, args.format, args.float64, log)
    
    print(f"\nGenerated sinc table: {basename}")
    print(f"  Format: {'polyphase' if args.polyphase else 'linear'}")
    print(f"  Size: {table.shape if table.ndim > 1 else len(table)} samples")
    print(f"  Zero crossings: ±{args.zeros}")
    print(f"  Oversample: {args.oversample}x")
    if 'symmetry_error' in metadata:
        print(f"  Symmetry error: {metadata['symmetry_error']:.2e}")


if __name__ == '__main__':
    main()
