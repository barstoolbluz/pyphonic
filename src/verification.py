#!/usr/bin/env python3
"""
Verification tools for filter coefficients.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional


def verify_filter_response(
    coefficients: np.ndarray,
    sample_rate: float = 48000,
    target_stopband_db: float = 180.0,
    target_passband_ripple_db: float = 0.00001,
    plot: bool = False
) -> Dict[str, Any]:
    """
    Verify filter meets specifications.
    
    Parameters
    ----------
    coefficients : np.ndarray
        Filter coefficients
    sample_rate : float
        Sample rate in Hz
    target_stopband_db : float
        Target stopband attenuation in dB
    target_passband_ripple_db : float
        Target passband ripple in dB
    plot : bool
        Whether to plot frequency response
        
    Returns
    -------
    dict
        Verification results
    """
    # Compute frequency response
    w, h = signal.freqz(coefficients, worN=65536)
    freq = w * sample_rate / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h) + 1e-300)
    phase = np.unwrap(np.angle(h))
    
    # Find -3dB point
    idx_3db = np.argmin(np.abs(mag_db + 3))
    f_3db = freq[idx_3db]
    
    # Measure passband ripple (up to 90% of -3dB point)
    passband_end = int(0.9 * idx_3db)
    passband = mag_db[:passband_end]
    ripple_db = np.ptp(passband) if len(passband) > 0 else 0.0
    
    # Measure stopband (from 110% of -3dB point)
    stopband_start = int(1.1 * idx_3db)
    stopband = mag_db[stopband_start:] if stopband_start < len(mag_db) else []
    stopband_peak_db = np.max(stopband) if len(stopband) > 0 else -np.inf
    
    # Measure group delay
    _, gd = signal.group_delay((coefficients, 1), w=w)
    gd_variation = np.ptp(gd[:passband_end]) if passband_end > 0 else 0.0
    
    results = {
        'f_3db': f_3db,
        'passband_ripple_db': ripple_db,
        'stopband_atten_db': -stopband_peak_db,
        'group_delay_var': gd_variation,
        'meets_stopband': -stopband_peak_db >= target_stopband_db,
        'meets_passband': ripple_db <= target_passband_ripple_db,
    }
    
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Magnitude response
        ax1.plot(freq/1000, mag_db)
        ax1.axhline(-target_stopband_db, color='r', linestyle='--', label=f'Target: -{target_stopband_db} dB')
        ax1.axvline(f_3db/1000, color='g', linestyle='--', label=f'-3dB: {f_3db:.1f} Hz')
        ax1.set_xlabel('Frequency (kHz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Frequency Response')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(-200, 5)
        
        # Phase response
        ax2.plot(freq/1000, phase)
        ax2.set_xlabel('Frequency (kHz)')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_title('Phase Response')
        ax2.grid(True, alpha=0.3)
        
        # Group delay
        ax3.plot(freq/1000, gd)
        ax3.set_xlabel('Frequency (kHz)')
        ax3.set_ylabel('Group Delay (samples)')
        ax3.set_title(f'Group Delay (variation: {gd_variation:.2f} samples)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results


def compare_with_libsox(
    our_coeffs: np.ndarray,
    libsox_coeffs: Optional[np.ndarray] = None,
    plot: bool = True
) -> Dict[str, Any]:
    """
    Compare our coefficients with libsoxr reference.
    
    Parameters
    ----------
    our_coeffs : np.ndarray
        Our filter coefficients
    libsox_coeffs : np.ndarray, optional
        libsoxr coefficients (if available)
    plot : bool
        Whether to plot comparison
        
    Returns
    -------
    dict
        Comparison results
    """
    results = {}
    
    # Analyze our filter
    our_response = verify_filter_response(our_coeffs, plot=False)
    results['our_specs'] = our_response
    
    if libsox_coeffs is not None:
        # Analyze libsox filter
        sox_response = verify_filter_response(libsox_coeffs, plot=False)
        results['libsox_specs'] = sox_response
        
        # Compare
        results['stopband_improvement_db'] = (
            our_response['stopband_atten_db'] - sox_response['stopband_atten_db']
        )
        results['ripple_improvement_factor'] = (
            sox_response['passband_ripple_db'] / our_response['passband_ripple_db']
            if our_response['passband_ripple_db'] > 0 else np.inf
        )
        
        if plot:
            # Plot both responses
            w_our, h_our = signal.freqz(our_coeffs, worN=8192)
            w_sox, h_sox = signal.freqz(libsox_coeffs, worN=8192)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(w_our/np.pi, 20*np.log10(np.abs(h_our)), 
                   label='Our implementation', linewidth=2)
            ax.plot(w_sox/np.pi, 20*np.log10(np.abs(h_sox)), 
                   label='libsoxr', linewidth=1, alpha=0.7)
            ax.set_xlabel('Normalized Frequency')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title('Filter Comparison')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-200, 5)
            plt.show()
    
    return results