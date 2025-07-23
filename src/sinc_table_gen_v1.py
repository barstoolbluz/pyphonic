#!/usr/bin/env python3
"""
Sinc Interpolation Table Generator with Mathematical Rigor
==========================================================

Generates windowed sinc interpolation tables for variable-rate resampling
with the same mathematical precision as the FIR filter generator.

Key features:
- Supports Kaiser, Blackman-Harris, and other windows
- Generates high-resolution tables for fractional delay interpolation
- Saves as NPY (single table) or NPZ (with metadata)
- Full verification and analysis capabilities
- Handles extreme specifications (256+ zero crossings)

Example usage:
--------------
# Basic sinc table for 16 zero crossings:
python3 generate_sinc_tables.py --zeros 16 --oversample 512

# High quality with Kaiser window:
python3 generate_sinc_tables.py --zeros 32 --oversample 1024 --window kaiser --beta 12.0

# Extreme quality for obsessive audiophiles:
python3 generate_sinc_tables.py --zeros 64 --oversample 2048 --window kaiser --beta 20.0

# With specific cutoff for band-limiting:
python3 generate_sinc_tables.py --zeros 32 --cutoff 0.45 --oversample 512

# Analyze existing table:
python3 generate_sinc_tables.py --analyze sinc_32z_512x_kaiser.npz --plot
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.signal import freqz, windows
from scipy.special import i0 as bessel_i0


def kaiser_window_exact(n: int, length: int, beta: float) -> float:
    """
    Compute Kaiser window value with extreme precision.
    Matches the rigor of the FIR generator.
    """
    if length <= 1:
        return 1.0
    
    # Map n to [-1, 1]
    x = 2.0 * n / (length - 1) - 1.0
    
    if abs(x) > 1.0:
        return 0.0
    
    # Use scipy's bessel function for consistency
    arg = beta * np.sqrt(1.0 - x * x)
    return bessel_i0(arg) / bessel_i0(beta)


def generate_windowed_sinc(
    zero_crossings: int,
    oversample_factor: int,
    cutoff: float,
    window_type: str,
    window_params: Dict[str, Any],
    log: logging.Logger
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate high-resolution windowed sinc table for interpolation.
    
    Parameters:
    -----------
    zero_crossings : int
        Number of zero crossings on each side of center (total = 2 * zero_crossings)
    oversample_factor : int
        Table resolution - higher = better interpolation accuracy
    cutoff : float
        Normalized cutoff frequency (0-1, default 0.5 for Nyquist)
    window_type : str
        Window function: 'kaiser', 'blackman-harris', 'hann', etc.
    window_params : dict
        Window-specific parameters (e.g., {'beta': 8.6} for Kaiser)
    
    Returns:
    --------
    table : np.ndarray
        Sinc interpolation table of size (2 * zero_crossings * oversample_factor)
    metadata : dict
        Table generation parameters and statistics
    """
    
    # Calculate table size
    table_size = 2 * zero_crossings * oversample_factor
    
    log.info("Generating sinc table:")
    log.info("  Zero crossings: %d (±%d)", 2 * zero_crossings, zero_crossings)
    log.info("  Oversample factor: %d", oversample_factor)
    log.info("  Table size: %d samples", table_size)
    log.info("  Cutoff: %.6f (normalized)", cutoff)
    log.info("  Window: %s", window_type)
    
    # Allocate table
    table = np.zeros(table_size, dtype=np.float64)
    
    # Generate sinc values with extreme precision
    t0 = time.perf_counter()
    
    for i in range(table_size):
        # Position in terms of zero crossings
        # i=0 corresponds to -zero_crossings
        # i=table_size-1 corresponds to +zero_crossings
        x = (i - table_size/2.0) / oversample_factor
        
        # Sinc function with cutoff
        if abs(x) < 1e-15:  # Use tighter tolerance than FIR generator
            sinc_val = 2.0 * cutoff
        else:
            arg = 2.0 * np.pi * cutoff * x
            sinc_val = np.sin(arg) / (np.pi * x)
        
        # Apply window
        if window_type == 'kaiser':
            beta = window_params.get('beta', 8.6)
            # Map to window position [0, 1]
            window_pos = i / (table_size - 1)
            window_val = kaiser_window_exact(i, table_size, beta)
        
        elif window_type == 'blackman-harris':
            # 4-term Blackman-Harris
            window_pos = i / (table_size - 1)
            a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
            window_val = (a0 
                         - a1 * np.cos(2*np.pi*window_pos)
                         + a2 * np.cos(4*np.pi*window_pos)
                         - a3 * np.cos(6*np.pi*window_pos))
        
        elif window_type == 'hann':
            window_pos = i / (table_size - 1)
            window_val = 0.5 - 0.5 * np.cos(2*np.pi*window_pos)
        
        elif window_type == 'hamming':
            window_pos = i / (table_size - 1)
            window_val = 0.54 - 0.46 * np.cos(2*np.pi*window_pos)
        
        else:  # rectangular
            window_val = 1.0
        
        table[i] = sinc_val * window_val
    
    # Normalize for unity gain at DC
    # For interpolation, we want the sum of all samples used in one
    # interpolation to equal 1.0
    # Since we use 2*zero_crossings samples, spaced by oversample_factor:
    stride = oversample_factor
    norm_sum = 0.0
    
    # Sum contributions at integer positions
    center = table_size // 2
    for k in range(-zero_crossings, zero_crossings):
        idx = center + k * oversample_factor
        if 0 <= idx < table_size:
            norm_sum += table[idx]
    
    if abs(norm_sum) > 1e-10:
        table /= norm_sum
        log.info("Normalized by factor %.12f", norm_sum)
    
    gen_time = time.perf_counter() - t0
    log.info("Table generated in %.3f seconds", gen_time)
    
    # Compute metadata
    metadata = {
        'zero_crossings': zero_crossings,
        'oversample_factor': oversample_factor,
        'cutoff': cutoff,
        'window_type': window_type,
        'window_params': window_params,
        'table_size': table_size,
        'generation_time': gen_time,
        'normalization_factor': norm_sum if abs(norm_sum) > 1e-10 else 1.0,
    }
    
    # Add window-specific metadata
    if window_type == 'kaiser':
        # Theoretical stopband attenuation for Kaiser window
        beta = window_params.get('beta', 8.6)
        if beta > 8.7:
            atten_db = 8.7 + 0.1102 * (beta - 8.7) * 20
        elif beta > 0:
            atten_db = 20 * (0.5842 * (beta/2.09) ** 0.4 + 0.07886 * (beta/2.09))
        else:
            atten_db = 21.0
        metadata['theoretical_stopband_db'] = atten_db
    
    return table, metadata


def analyze_sinc_table(
    table: np.ndarray,
    metadata: Dict[str, Any],
    log: logging.Logger,
    show_plot: bool = False
) -> None:
    """Analyze sinc interpolation table characteristics."""
    
    print("\n" + "="*60)
    print("SINC TABLE ANALYSIS")
    print("="*60)
    
    print("\nTable Configuration")
    print("-" * 20)
    print(f"Zero crossings    : ±{metadata['zero_crossings']}")
    print(f"Oversample factor : {metadata['oversample_factor']}")
    print(f"Table size        : {metadata['table_size']:,} samples")
    print(f"Memory usage      : {table.nbytes / 1024:.1f} KB")
    print(f"Cutoff frequency  : {metadata['cutoff']:.6f} (normalized)")
    print(f"Window type       : {metadata['window_type']}")
    
    if metadata['window_type'] == 'kaiser':
        beta = metadata['window_params'].get('beta', 8.6)
        print(f"Kaiser β          : {beta:.3f}")
        if 'theoretical_stopband_db' in metadata:
            print(f"Theoretical atten : {metadata['theoretical_stopband_db']:.1f} dB")
    
    # Analyze table properties
    print("\nTable Statistics")
    print("-" * 20)
    print(f"Maximum value     : {table.max():.12f}")
    print(f"Minimum value     : {table.min():.12f}")
    print(f"RMS value         : {np.sqrt(np.mean(table**2)):.12f}")
    print(f"Peak position     : {table.argmax()} (sample)")
    
    # Check symmetry
    center = len(table) // 2
    if len(table) % 2 == 0:
        # Even length - check symmetry around center
        left = table[:center]
        right = table[center:][::-1]
        symmetry_error = np.max(np.abs(left - right))
    else:
        # Odd length - check symmetry around center point
        left = table[:center]
        right = table[center+1:][::-1]
        symmetry_error = np.max(np.abs(left - right))
    
    print(f"Symmetry error    : {symmetry_error:.2e}")
    
    # Frequency response analysis
    print("\nFrequency Response")
    print("-" * 20)
    
    # For frequency analysis, create an equivalent FIR filter
    # by taking samples at integer positions
    zc = metadata['zero_crossings']
    osf = metadata['oversample_factor']
    
    # Extract FIR taps at integer positions
    center_idx = len(table) // 2
    fir_taps = []
    for k in range(-zc, zc):
        idx = center_idx + k * osf
        if 0 <= idx < len(table):
            fir_taps.append(table[idx])
    
    fir_taps = np.array(fir_taps)
    
    # Compute frequency response
    w, h = freqz(fir_taps, worN=8192)
    w_norm = w / np.pi  # Normalize to [0, 1]
    mag_db = 20 * np.log10(np.abs(h) + 1e-300)
    
    # Find key frequencies
    cutoff_idx = np.argmin(np.abs(w_norm - metadata['cutoff']))
    cutoff_db = mag_db[cutoff_idx]
    
    # Find -3dB point
    idx_3db = np.argmin(np.abs(mag_db + 3))
    f_3db = w_norm[idx_3db]
    
    # Find stopband attenuation
    # Look beyond cutoff + transition
    transition_band = 0.1  # Assume 10% transition
    stopband_start = metadata['cutoff'] + transition_band
    if stopband_start < 1.0:
        sb_idx = w_norm > stopband_start
        if sb_idx.any():
            stopband_peak = mag_db[sb_idx].max()
        else:
            stopband_peak = -200
    else:
        stopband_peak = -200
    
    print(f"DC gain           : {mag_db[0]:.3f} dB")
    print(f"Gain at cutoff    : {cutoff_db:.3f} dB")
    print(f"-3 dB frequency   : {f_3db:.6f} (normalized)")
    print(f"Stopband peak     : {stopband_peak:.1f} dB")
    
    # Interpolation accuracy analysis
    print("\nInterpolation Quality")
    print("-" * 20)
    
    # Test interpolation of DC (should be perfect)
    dc_sum = 0.0
    for k in range(-zc, zc):
        idx = center_idx + k * osf
        if 0 <= idx < len(table):
            dc_sum += table[idx]
    
    print(f"DC interpolation  : {dc_sum:.12f} (ideal: 1.0)")
    print(f"DC error          : {abs(dc_sum - 1.0):.2e}")
    
    # Resolution analysis
    time_resolution = 1.0 / osf
    print(f"Time resolution   : {time_resolution:.6f} samples")
    print(f"                  : {1000*time_resolution:.3f} ms @ 48kHz")
    
    print("\n" + "="*60)
    
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Full table
            ax = axes[0, 0]
            x = np.linspace(-zc, zc, len(table))
            ax.plot(x, table)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Position (zero crossings)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Windowed Sinc Table ({metadata["window_type"]})')
            
            # Plot 2: Center zoom
            ax = axes[0, 1]
            zoom_range = int(2 * osf)  # ±2 zero crossings
            center_start = center_idx - zoom_range
            center_end = center_idx + zoom_range
            x_zoom = np.linspace(-2, 2, 2*zoom_range)
            ax.plot(x_zoom, table[center_start:center_end], 'b-', label='Table values')
            ax.plot(x_zoom[::osf], table[center_start:center_end:osf], 'ro', 
                   markersize=6, label='Integer positions')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Position (zero crossings)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Center Region Detail')
            ax.legend()
            
            # Plot 3: Frequency response
            ax = axes[1, 0]
            ax.semilogx(w_norm[1:], mag_db[1:])
            ax.axvline(metadata['cutoff'], color='r', linestyle='--', 
                      label=f'Cutoff ({metadata["cutoff"]:.3f})')
            ax.axhline(-3, color='g', linestyle=':', alpha=0.5, label='-3 dB')
            ax.grid(True, which='both', alpha=0.3)
            ax.set_xlabel('Normalized Frequency')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title('Frequency Response')
            ax.set_ylim(bottom=-120, top=5)
            ax.legend()
            
            # Plot 4: Phase response
            ax = axes[1, 1]
            phase = np.unwrap(np.angle(h))
            ax.plot(w_norm, phase)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Normalized Frequency')
            ax.set_ylabel('Phase (radians)')
            ax.set_title('Phase Response')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            log.warning("matplotlib not available, skipping plots")


def save_sinc_table(
    table: np.ndarray,
    metadata: Dict[str, Any],
    basename: str,
    save_format: str,
    log: logging.Logger
) -> None:
    """Save sinc table in requested format(s)."""
    
    if save_format in ['npy', 'both']:
        # Save as NPY (just the table)
        npy_path = f"{basename}.npy"
        np.save(npy_path, table.astype(np.float32))
        log.info("Saved table to %s", npy_path)
    
    if save_format in ['npz', 'both']:
        # Save as NPZ (table + metadata)
        npz_path = f"{basename}.npz"
        
        # Convert metadata to saveable format
        save_metadata = metadata.copy()
        save_metadata['command_line'] = ' '.join(sys.argv)
        save_metadata['numpy_version'] = np.__version__
        
        np.savez(npz_path,
                table=table.astype(np.float32),
                metadata=save_metadata)
        
        log.info("Saved table and metadata to %s", npz_path)


def load_and_analyze(npz_path: Path, show_plot: bool, log: logging.Logger) -> None:
    """Load and analyze a previously saved sinc table."""
    
    log.info("Loading sinc table from %s", npz_path)
    
    with np.load(npz_path, allow_pickle=True) as data:
        table = data['table']
        metadata = data['metadata'].item()
    
    log.info("Loaded table: %d samples, %.1f KB", 
             len(table), table.nbytes / 1024)
    
    analyze_sinc_table(table, metadata, log, show_plot)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--zeros', '-z', type=int,
        help='Number of zero crossings on each side (e.g., 16 = ±16)'
    )
    mode_group.add_argument(
        '--analyze', '-a', type=Path,
        help='Analyze existing .npz file instead of generating'
    )
    
    # Generation parameters
    gen_group = parser.add_argument_group('Generation parameters')
    gen_group.add_argument(
        '--oversample', '-o', type=int, default=512,
        help='Oversample factor for table resolution (default: 512)'
    )
    gen_group.add_argument(
        '--cutoff', '-c', type=float, default=0.5,
        help='Normalized cutoff frequency (0-1, default: 0.5 = Nyquist)'
    )
    gen_group.add_argument(
        '--window', '-w', 
        choices=['kaiser', 'blackman-harris', 'hann', 'hamming', 'rectangular'],
        default='kaiser',
        help='Window function type (default: kaiser)'
    )
    gen_group.add_argument(
        '--beta', '-b', type=float, default=8.6,
        help='Kaiser window β parameter (default: 8.6, range: 0-50)'
    )
    
    # Output options
    out_group = parser.add_argument_group('Output options')
    out_group.add_argument(
        '--format', '-f', choices=['npy', 'npz', 'both'], default='npz',
        help='Output format: npy (table only), npz (table+metadata), both'
    )
    out_group.add_argument(
        '--basename', type=str,
        help='Output filename base (default: auto-generated)'
    )
    out_group.add_argument(
        '--plot', '-p', action='store_true',
        help='Show analysis plots (requires matplotlib)'
    )
    
    # Misc options
    parser.add_argument(
        '--debug', '-d', action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    log = logging.getLogger('sinc')
    
    # Analyze mode
    if args.analyze:
        load_and_analyze(args.analyze, args.plot, log)
        return
    
    # Validate generation parameters
    if args.zeros < 1:
        parser.error("Zero crossings must be >= 1")
    
    if args.oversample < 1:
        parser.error("Oversample factor must be >= 1")
    
    if not 0 < args.cutoff <= 1:
        parser.error("Cutoff must be in range (0, 1]")
    
    if args.window == 'kaiser' and not 0 <= args.beta <= 50:
        parser.error("Kaiser beta must be in range [0, 50]")
    
    # Setup window parameters
    window_params = {}
    if args.window == 'kaiser':
        window_params['beta'] = args.beta
    
    # Generate table
    table, metadata = generate_windowed_sinc(
        args.zeros,
        args.oversample,
        args.cutoff,
        args.window,
        window_params,
        log
    )
    
    # Analyze
    analyze_sinc_table(table, metadata, log, args.plot)
    
    # Generate output filename
    if args.basename:
        basename = args.basename
    else:
        # Auto-generate descriptive name
        parts = [f"sinc_{args.zeros}z"]
        
        if args.oversample != 512:
            parts.append(f"{args.oversample}x")
            
        if abs(args.cutoff - 0.5) > 0.001:
            parts.append(f"{int(args.cutoff*1000)}c")
            
        parts.append(args.window)
        
        if args.window == 'kaiser':
            parts.append(f"b{args.beta:.1f}".replace('.', '_'))
            
        basename = '_'.join(parts)
    
    # Save
    save_sinc_table(table, metadata, basename, args.format, log)
    
    # Print usage example
    print("\nUsage example for C/Rust:")
    print("-" * 30)
    print(f"// Load table from '{basename}.npz'")
    print(f"// Table size: {len(table)} samples")
    print(f"// Zero crossings: ±{args.zeros}")
    print(f"// Oversample factor: {args.oversample}")
    print(f"//")
    print(f"// For fractional delay 'frac' (0-1):")
    print(f"//   table_pos = (frac + {args.zeros}) * {args.oversample}")
    print(f"//   Interpolate between table[floor(table_pos)] and table[ceil(table_pos)]")


if __name__ == '__main__':
    main()