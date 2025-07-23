#!/usr/bin/env python3
"""
Exact-spec FIR Up/Downsampler Designer – Kaiser & Parks-McClellan
=================================================================

Supports both upsampling and downsampling with mathematically rigorous design
using either Kaiser windows (near-optimal) or Parks-McClellan (minimax optimal).
Features RAM-adaptive verification and integrated analysis capabilities.

Key features:
- Handles extreme specifications (ripple to 0.000001 dB, attenuation to 400+ dB)
- Quantization analysis shows effective resolution relative to largest tap
- Complete alias-band checking for downsampling
- Minimax-correct ripple measurement for verification
- Quantization report measures noise relative to max-tap amplitude

CLI examples
------------
# Basic upsampling 32× with Kaiser window:
python3 fir_enhanced.py \
    --rate-in 96000 --upsample 32 --cutoff 48000 \
    --transition 1.5625

# High-precision upsampling with specific tap count:
python3 fir_enhanced.py \
    --rate-in 96000 --upsample 32 --cutoff 48000 \
    --taps-pow2 28 --transition 1.5625 --pb-ripple 0.001 --sb-atten 150

# Downsampling 8× with Parks-McClellan for minimax optimality:
python3 fir_enhanced.py \
    --rate-in 384000 --downsample 8 --cutoff 22000 \
    --transition 2000 --pb-ripple 0.001 --sb-atten 150 \
    --method parks-mcclellan

# Extreme precision: 0.000001 dB ripple, 400 dB attenuation:
python3 fir_enhanced.py \
    --rate-in 192000 --downsample 4 --cutoff 40000 \
    --transition 500 --pb-ripple 0.000001 --sb-atten 400 \
    --method parks-mcclellan --force

# Ultra-narrow transition (0.1 Hz!) for surgical filtering:
python3 fir_enhanced.py \
    --rate-in 48000 --upsample 1 --cutoff 1000 \
    --transition 0.1 --pb-ripple 0.01 --sb-atten 120 --force

# Perfect flat passband (0 dB ripple):
python3 fir_enhanced.py \
    --rate-in 96000 --downsample 2 --cutoff 40000 \
    --pb-ripple 0 --sb-atten 140 --method parks-mcclellan

# Analyze existing filter with plot:
python3 fir_enhanced.py --analyze u32_96000to3072000Hz_2^26.npz --plot

# Just analyze without redesigning:
python3 fir_enhanced.py --analyze my_filter.npz --verify-factor 32
"""

from __future__ import annotations
import argparse, logging, math, os, sys, time, warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.signal import firwin, kaiserord, freqz, remez

try:
    import psutil
except ImportError:
    psutil = None


# ───────────────────────── Data structures ────────────────────────── #

@dataclass
class FilterSpec:
    """Complete specification of a filter design."""
    mode: str  # 'upsample' or 'downsample'
    factor: int
    fs_in: float
    fs_out: float
    cutoff: float
    transition: float
    pb_ripple: float
    sb_atten: float
    forced_k: Optional[int]
    design_method: str = 'kaiser'  # 'kaiser' or 'parks-mcclellan'
    no_pow2: bool = False  # Allow non-power-of-2 lengths
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FilterSpec':
        return cls(**d)


# ───────────────────────── helpers ────────────────────────── #

def bytes_free() -> int:
    if psutil:
        return psutil.virtual_memory().available
    if hasattr(os, "sysconf"):
        pages = os.sysconf("SC_AVPHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return pages * size
    return 0

def fmt_time(sec: float) -> str:
    return f"{sec/60:.1f} min" if sec >= 60 else f"{sec:.1f} s"

def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def human_size(bytes_: float) -> str:
    if bytes_ < 1024:
        return f"{bytes_:.1f} B"
    elif bytes_ < 1024**2:
        return f"{bytes_/1024:.1f} KB"
    elif bytes_ < 1024**3:
        return f"{bytes_/1024**2:.1f} MB"
    else:
        return f"{bytes_/1024**3:.2f} GB"

def fmt_pow2_tag(k: Optional[int]) -> str:
    """Return ' (2^k)' or ' (non-power-of-2)'."""
    return f" (2^{k})" if k is not None else " (non-power-of-2)"


# ─────────────────────── design routine ─────────────────────── #

def design_kernel(spec: FilterSpec, force_ram: bool, log: logging.Logger) -> Tuple[np.ndarray, float, Optional[int], float]:
    """Design FIR kernel for either upsampling or downsampling."""
    
    if spec.mode == 'upsample':
        if spec.factor <= 0:
            sys.exit("Upsample factor must be positive")
        fs_out = spec.fs_in * spec.factor
        design_fs = fs_out  # Design at output rate
    else:  # downsample
        if spec.factor <= 0:
            sys.exit("Downsample factor must be positive")
        fs_out = spec.fs_in / spec.factor
        design_fs = spec.fs_in  # Design at input rate
    
    nyq_design = design_fs / 2
    
    # For downsampling, complete alias-safety check
    if spec.mode == 'downsample':
        nyq_out = fs_out / 2
        if spec.cutoff + spec.transition >= nyq_out:
            log.warning("Cutoff %.1f Hz + transition %.1f Hz exceeds output Nyquist %.1f Hz; clamping cutoff", 
                       spec.cutoff, spec.transition, nyq_out)
            cutoff = nyq_out - spec.transition - 1e-9  # leave ε margin
            spec.cutoff = cutoff  # ← keep spec consistent
        else:
            cutoff = spec.cutoff
    else:
        cutoff = spec.cutoff
    
    if not 0 < cutoff < nyq_design:
        sys.exit(f"Cutoff must lie between 0 and Nyquist ({nyq_design} Hz)")

    Δf = spec.transition or spec.fs_in / 128
    if Δf <= 0:
        sys.exit("Transition bandwidth must be positive")
    
    if cutoff - Δf <= 0:
        sys.exit(f"Transition width {Δf} Hz too large for cutoff {cutoff} Hz "
                 "(cutoff-Δf must be positive)")
    
    # Allow extremely narrow transitions but warn
    if Δf < 0.1:
        log.warning("ULTRA-NARROW transition: %.6f Hz (%.3f mHz)", Δf, Δf * 1000)
        log.warning("This may require extreme filter lengths and verification time")
    elif Δf < 1.0:
        log.warning("Extremely narrow transition: %.6f Hz", Δf)
    
    if Δf >= nyq_design - cutoff:
        sys.exit(f"Transition width {Δf} Hz too large (cutoff={cutoff}, Nyquist={nyq_design})")

    # ── derive/validate length ──
    if spec.design_method == 'kaiser':
        if spec.forced_k is None:
            width_norm = 2 * Δf / design_fs
            order, beta = kaiserord(spec.sb_atten, width_norm)
            N_theory = order + 1  # taps = order + 1
            
            # Round up to power of 2 unless --no-pow2 specified
            if hasattr(spec, 'no_pow2') and spec.no_pow2:
                N = N_theory
                k = None  # Not a power of 2
            else:
                N = next_pow2(N_theory)
                k = int(math.log2(N))
        else:
            k = spec.forced_k
            N = 1 << k
            width_norm = 2 * Δf / design_fs
            need_order, beta = kaiserord(spec.sb_atten, width_norm)
            need = need_order + 1
            if N < need:
                sys.exit(f"2^{k} taps cannot meet spec (needs ≥ {need})")

        beta = float(beta)
        theor_As = 2.285 * (N - 1) * (2*Δf/design_fs) + 7.95
        log.info("Kaiser theory: β=%.3f  ⇒  worst-stop ≈ %.1f dB", beta, theor_As)
        
        # For ultra-narrow transitions, provide context
        if Δf < 0.1:
            relative_bw = Δf / design_fs
            log.info("Ultra-narrow transition: %.3f mHz = %.2e × Fs", 
                    Δf * 1000, relative_bw)
            log.info("This is equivalent to %.2f ppm of sample rate", 
                    relative_bw * 1e6)
    
    else:  # parks-mcclellan
        if spec.forced_k is None:
            # Estimate initial length for Parks-McClellan
            # For extreme specs, we need to be more careful
            
            # Convert dB to linear, handling extreme values
            if spec.pb_ripple < 0.0001:  # Less than -80 dB
                dp = spec.pb_ripple / 8.686  # Approximate for very small values
            else:
                dp = 10**(spec.pb_ripple/20) - 1
            
            if spec.sb_atten > 300:  # Beyond -300 dB
                ds = 10**(-300/20)  # Cap at reasonable value
            else:
                ds = 10**(-spec.sb_atten/20)
            
            # Use Hermann approximation for extreme specs
            # N ≈ (-10*log10(δp*δs) - 13) / (14.6 * Δf/fs)
            # But for extreme values, use Kaiser estimate as starting point
            if spec.pb_ripple < 0.001 or spec.sb_atten > 200 or Δf < 1.0:
                # Use Kaiser estimate for extreme specs
                order, beta = kaiserord(spec.sb_atten, 2 * Δf / design_fs)
                N_est = order + 1  # taps = order + 1
                # Add extra margin for Parks-McClellan with narrow transitions
                margin = 2.0 if Δf < 0.1 else 1.5
                if spec.no_pow2:
                    N = int(N_est * margin)
                    k = None
                else:
                    N = next_pow2(int(N_est * margin))
                    k = int(math.log2(N))
            else:
                # Standard Bellanger/Hermann formula
                try:
                    product = dp * ds
                    if product > 0:
                        N_est = (-10*np.log10(product) - 13) / (14.6 * Δf/design_fs)
                    else:
                        # Fallback for numerical issues
                        N_est = (spec.sb_atten + 10) / (14.6 * Δf/design_fs)
                    
                    if spec.no_pow2:
                        N = int(N_est * 1.2)
                        k = None
                    else:
                        N = next_pow2(int(N_est * 1.2))
                        k = int(math.log2(N))
                except:
                    # Ultimate fallback
                    if spec.no_pow2:
                        N = int(spec.sb_atten * design_fs / (14.6 * Δf))
                        k = None
                    else:
                        N = next_pow2(int(spec.sb_atten * design_fs / (14.6 * Δf)))
                        k = int(math.log2(N))
                # Still need beta for potential fallback
                order, beta = kaiserord(spec.sb_atten, 2 * Δf / design_fs)
        else:
            k = spec.forced_k
            N = 1 << k
            # Need beta for potential Kaiser fallback
            _, beta = kaiserord(spec.sb_atten, 2 * Δf / design_fs)
        
        log.info("Parks-McClellan initial estimate: N=%d%s", N, fmt_pow2_tag(k))

    # ── RAM guards ──
    free = bytes_free()
    
    # More permissive limits for extreme requirements
    if k is not None and k > 30:
        log.warning("EXTREME: Requesting 2^%d = %d taps (%.1f billion)", 
                   k, 1<<k, (1<<k)/1e9)
        if not force_ram:
            sys.exit(f"Filter requires 2^{k} taps. Use --force to attempt anyway.")
    elif k is not None and free <= 64 << 30 and k > 27 and not force_ram:
        sys.exit("Machine reports ≤64 GiB free – hard limit 2^27 taps. "
                 "Use --force to override.")
    else:
        hard_k = 25 if free < 64 << 30 else 27 if free < 128 << 30 else 30
        if k is not None and k > hard_k and not force_ram:
            sys.exit(f"tap exponent {k} exceeds conservative limit {hard_k} "
                     f"(free={free/2**30:.1f} GiB). Use --force if you are sure.")

    log.info("Designing %d taps%s for %s by %d× using %s method…", 
             N, fmt_pow2_tag(k), spec.mode, spec.factor, spec.design_method)
    t0 = time.perf_counter()
    
    if spec.design_method == 'kaiser':
        # Design filter with Kaiser window
        taps = firwin(N, cutoff / nyq_design, window=('kaiser', beta), scale=True)
    else:  # parks-mcclellan
        # Design filter with Parks-McClellan
        # Need to ensure N is odd for type I filter
        if N % 2 == 0:
            N += 1
            log.info("Adjusted to N=%d (odd) for Parks-McClellan type I", N)
        
        # After the bump N is not a pure power-of-two → no meaningful k
        if (N & (N-1)) == 0:  # power of two?
            k = int(math.log2(N))
        else:
            k = None
        
        # Setup bands for remez
        # Note: remez requires first band edge to be exactly 0
        bands = [0, cutoff - Δf, cutoff + Δf, nyq_design]
        desired = [1, 0]
        
        # Convert dB specs to linear weights
        # δp > 0 (peak linear deviation)
        dp = 10**(spec.pb_ripple/20) - 1
        if dp <= 1e-30:  # covers ripple == 0 dB
            dp = np.finfo(float).eps
        ds = 10**(-spec.sb_atten/20)      # Works for extreme attenuation
        
        # Avoid overflow - scale so max(weight)=1
        # Note: absolute scale irrelevant for remez; only weight ratios matter
        w0, w1 = 1.0/dp, 1.0/ds
        scale = max(w0, w1)
        weights = [w0/scale, w1/scale]
        
        # Log extreme values in scientific notation
        if dp < 1e-10 or ds < 1e-10 or scale > 1e15:
            log.info("Parks-McClellan specs: dp=%.2e, ds=%.2e, weights=[%.2e, %.2e] (scaled)", 
                     dp, ds, weights[0], weights[1])
        
        log.info("Parks-McClellan: dp=%.2e, ds=%.2e, weights=[%.2e, %.2e]", 
                 dp, ds, weights[0], weights[1])
        
        try:
            taps = remez(N, bands, desired, weight=weights, fs=design_fs)
            log.info("Parks-McClellan succeeded")
        except Exception as e:
            log.error("Parks-McClellan failed: %s", e)
            log.info("Falling back to Kaiser design")
            # kaiserord already computed beta above, just use it
            taps = firwin(N, cutoff / nyq_design, window=('kaiser', beta), scale=True)
    
    log.info("FIR generated in %s", fmt_time(time.perf_counter() - t0))
    return taps, fs_out, k, Δf


# ─────────────── rigorous full-FFT verification ────────────── #

def verify_kernel(taps: np.ndarray, spec: FilterSpec,
                  over_init: int, allow_shrink: bool,
                  log: logging.Logger) -> bool:
    """
    Choose the densest full-grid FFT that fits safely in RAM.
    Verifies the filter meets spec using minimax ripple measurement.
    Returns True if the filter meets spec.
    """
    N = len(taps)
    free = bytes_free()

    if over_init < 1:
        sys.exit("--verify-factor must be ≥ 1")

    # Use appropriate sample rate for verification
    if spec.mode == 'upsample':
        verify_fs = spec.fs_out
    else:  # downsample
        verify_fs = spec.fs_in

    over = over_init
    while True:
        M = max(next_pow2(N * over), 1 << 20)
        # bytes: input float64 + output complex128
        need = M * 8 + (M // 2 + 1) * 16
        
        # For extreme filters, be more permissive
        if N > (1 << 28):  # > 256M taps
            fits = need <= 0.9 * free  # Allow up to 90% for extreme cases
        else:
            fits = need <= 0.6 * free
            
        res  = verify_fs / M
        
        # For extremely narrow transitions, use graduated resolution requirements
        if spec.transition < 0.1:
            # For ultra-narrow transitions, even 1 point is valuable
            res_ok = res <= spec.transition * 2  # Just 0.5 points minimum
        elif spec.transition < 1.0:
            # For very narrow transitions, relax to 2 points
            res_ok = res <= spec.transition / 2
        else:
            # Normal requirement: 4 points across transition
            res_ok = res <= spec.transition / 4

        if fits and res_ok:
            break
        if not allow_shrink:
            msg = ("Verification grid needs "
                   f"{need/2**30:.1f} GiB (> 60 % free) or "
                   f"resolution {res:.2f} Hz (> Δf/4). "
                   "Refusing because --no-auto-shrink was given.")
            sys.exit(msg)
        if over == 1:
            msg = ("Even the coarsest grid (over=1) would need "
                   f"{need/2**30:.1f} GiB or resolution {res:.2f} Hz "
                   "which exceeds Δf/4. Reduce taps or use a bigger machine.")
            sys.exit(msg)
        over //= 2   # try next smaller grid

    if over != over_init:
        log.info("Down-scaled oversample factor %d → %d "
                 "(grid=%.1f Mi samples, %.2f Hz bin).",
                 over_init, over, M / 2**20, res)

    log.info("Verifying on %d-pt FFT (≈ %.1f GiB scratch)…",
             M, need / 2**30)
    t0 = time.perf_counter()
    H = np.fft.rfft(taps, n=M)
    freqs = np.fft.rfftfreq(M, 1 / verify_fs)
    mags = 20 * np.log10(np.abs(H) + 1e-300)
    
    pb = freqs < (spec.cutoff - spec.transition)
    sb = freqs > (spec.cutoff + spec.transition)
    
    # For downsampling, check ALL image bands that could alias
    image_violations = []
    if spec.mode == 'downsample':
        # Check all image bands up to input Nyquist
        # Each image k maps: [k*fs_out - (cutoff+trans), k*fs_out - (cutoff-trans)]
        # Note: if cutoff was clamped during design, these bands still correctly
        # identify frequencies that alias into the passband after decimation
        for k in range(1, int(spec.factor) + 1):
            band_lo = k * spec.fs_out - (spec.cutoff + spec.transition)
            band_hi = k * spec.fs_out - (spec.cutoff - spec.transition)
            
            # Skip if band is entirely above input Nyquist
            if band_lo >= verify_fs / 2:
                break
                
            # Clip band to valid frequency range
            band_lo = max(band_lo, 0)
            band_hi = min(band_hi, verify_fs / 2)
            
            # Find frequencies in this image band
            ib = (freqs >= band_lo) & (freqs <= band_hi)
            
            if ib.any():
                ib_max = mags[ib].max()
                ib_freq = freqs[ib][mags[ib].argmax()]
                alias_target = ib_freq - k * spec.fs_out
                
                log.info("Image band %d: worst %.2f dB @ %.1f Hz (aliases to %.1f Hz)",
                         k, ib_max, ib_freq, abs(alias_target))
                
                if -ib_max < spec.sb_atten - 0.01:
                    image_violations.append((k, ib_max, ib_freq))
    
    # Guard against empty bands
    if not pb.any():
        log.warning("No passband frequencies found (cutoff - transition <= 0)")
        ripple = 0
        pb_ref = 0
    else:
        # Compute reference level as minimax center: (max + min) / 2
        # This is the textbook-correct reference for Chebyshev equiripple filters
        pb_max = mags[pb].max()
        pb_min = mags[pb].min()
        pb_ref = (pb_max + pb_min) / 2
        
        # Ripple is the maximum deviation from this center reference
        ripple = max(pb_max - pb_ref, pb_ref - pb_min)
        
        # Also report peak-to-peak for comparison
        ripple_pp = pb_max - pb_min
        log.debug("Passband: max %.6f dB, min %.6f dB, center %.6f dB", 
                  pb_max, pb_min, pb_ref)
        log.debug("Ripple: %.6f dB (P-P: %.6f dB)", ripple, ripple_pp)
    
    if not sb.any():
        log.warning("No stopband frequencies found")
        sb_max = -200
        sb_freq = 0
    else:
        sb_max = mags[sb].max()
        sb_freq = freqs[sb][mags[sb].argmax()]

    log.info("Ripple %.6f dB (ref %.3f dB), worst stop %.2f dB @ %.1f Hz",
             ripple, pb_ref, sb_max, sb_freq)
    
    # Check specifications with adaptive tolerance
    ripple_tol = max(0.001, 0.1 * spec.pb_ripple)  # Scale tolerance with spec
    ripple_ok = ripple <= spec.pb_ripple + ripple_tol
    stop_ok = -sb_max >= spec.sb_atten - 0.01
    
    # For downsampling, check all image bands
    image_ok = True
    if spec.mode == 'downsample' and image_violations:
        image_ok = False
        for k, ib_max, ib_freq in image_violations:
            log.warning("Image band %d violates spec: %.2f dB @ %.1f Hz (need < %.2f dB)", 
                       k, ib_max, ib_freq, -spec.sb_atten)
    
    ok = ripple_ok and stop_ok and image_ok
    
    log.info("Spec %s in %s", "PASS" if ok else "FAIL",
             fmt_time(time.perf_counter() - t0))
    
    return ok


# ─────────────────────── Analysis functions ──────────────────────── #

def analyze_quantization_effects(taps: np.ndarray, spec: FilterSpec, log: logging.Logger):
    """Analyze the effects of quantizing filter coefficients."""
    # Create output helper
    output_lines = []
    def output(msg=""):
        print(msg)
        output_lines.append(msg)
    
    output("\nQuantization Analysis")
    output("-" * 25)
    
    # Original float64 taps
    taps_f64 = taps.astype(np.float64)
    
    # Quantized versions
    taps_f32 = taps.astype(np.float32)
    
    # For integer quantization, we need to scale
    max_tap = np.abs(taps).max()
    
    # 32-bit Q31 format (SoX native format)
    # Q31 uses range [-1, 1) mapped to [-2^31, 2^31-1]
    # Note: +1.0 cannot be represented exactly in Q31
    scale_q31 = 2**31 - 1
    
    # Use nextafter to ensure we never round to exactly ±1.0
    q31_max = np.nextafter(1.0, 0.0)  # Largest value < 1.0
    q31_min = np.nextafter(-1.0, 0.0)  # Smallest value > -1.0
    taps_clipped = np.clip(taps, q31_min, q31_max)
    
    # Convert to integer in int64 to avoid overflow
    taps_q31_int64 = np.round(taps_clipped * scale_q31).astype(np.int64)
    taps_q31 = taps_q31_int64 / scale_q31  # Keep as float64 after scaling back
    
    # 24-bit signed integer quantization
    scale_24 = 2**23 - 1
    taps_24bit_int64 = np.round(taps * scale_24).astype(np.int64)
    taps_24bit_clipped = np.clip(taps_24bit_int64, -2**23, 2**23 - 1)
    taps_24bit = taps_24bit_clipped / scale_24
    
    # 16-bit signed integer quantization  
    scale_16 = 2**15 - 1
    taps_16bit = np.round(taps * scale_16).astype(np.int16) / scale_16
    
    # Compute quantization noise
    noise_f32 = taps_f64 - taps_f32
    noise_q31 = taps_f64 - taps_q31
    noise_24bit = taps_f64 - taps_24bit
    noise_16bit = taps_f64 - taps_16bit
    
    # RMS errors
    rms_f32 = np.sqrt(np.mean(noise_f32**2))
    rms_q31 = np.sqrt(np.mean(noise_q31**2))
    rms_24bit = np.sqrt(np.mean(noise_24bit**2))
    rms_16bit = np.sqrt(np.mean(noise_16bit**2))
    
    output(f"Quantization noise (RMS):")
    output(f"  float32: {rms_f32:.2e} ({20*np.log10(rms_f32/max_tap):.1f} dB)")
    output(f"  Q31 (SoX): {rms_q31:.2e} ({20*np.log10(rms_q31/max_tap):.1f} dB)")
    output(f"  24-bit:  {rms_24bit:.2e} ({20*np.log10(rms_24bit/max_tap):.1f} dB)")
    output(f"  16-bit:  {rms_16bit:.2e} ({20*np.log10(rms_16bit/max_tap):.1f} dB)")
    
    # Define theoretical limits for each format (amplitude domain, dBFS)
    # These are the best-possible noise floors for each format
    ideal_floor_f32 = -149.0  # IEEE 754 single precision (amplitude limit, ~-298 dB power)
    ideal_floor_q31 = -186.6  # 32-bit Q31: 6.02 * 31 = 186.6 dB (amplitude)
    ideal_floor_24 = -144.5   # 24-bit: 6.02 * 24 = 144.5 dB (amplitude)
    ideal_floor_16 = -96.3    # 16-bit: 6.02 * 16 = 96.3 dB (amplitude)
    
    # Frequency domain impact
    output("\nFrequency domain impact:")
    output("(Note: SoX uses Q31 format, shown separately below)")
    
    # Use appropriate sample rate
    if spec.mode == 'upsample':
        fs = spec.fs_out
    else:
        fs = spec.fs_in
    
    # Compute frequency responses
    w = np.linspace(0, fs/2, 4096)
    _, H_f64 = freqz(taps_f64, worN=w, fs=fs)
    _, H_f32 = freqz(taps_f32, worN=w, fs=fs)
    _, H_q31 = freqz(taps_q31, worN=w, fs=fs)
    _, H_24bit = freqz(taps_24bit, worN=w, fs=fs)
    _, H_16bit = freqz(taps_16bit, worN=w, fs=fs)
    
    # Convert to dB
    mag_f64 = 20 * np.log10(np.abs(H_f64) + 1e-300)
    mag_f32 = 20 * np.log10(np.abs(H_f32) + 1e-300)
    mag_q31 = 20 * np.log10(np.abs(H_q31) + 1e-300)
    mag_24bit = 20 * np.log10(np.abs(H_24bit) + 1e-300)
    mag_16bit = 20 * np.log10(np.abs(H_16bit) + 1e-300)
    
    # Find passband and stopband regions
    pb_idx = w < (spec.cutoff - spec.transition)
    sb_idx = w > (spec.cutoff + spec.transition)
    
    if pb_idx.any():
        # Passband error relative to float64
        pb_err_f32 = np.abs(mag_f64[pb_idx] - mag_f32[pb_idx]).max()
        pb_err_q31 = np.abs(mag_f64[pb_idx] - mag_q31[pb_idx]).max()
        pb_err_24bit = np.abs(mag_f64[pb_idx] - mag_24bit[pb_idx]).max()
        pb_err_16bit = np.abs(mag_f64[pb_idx] - mag_16bit[pb_idx]).max()
        
        output(f"\n  Max passband error (vs float64):")
        output(f"    float32:   {pb_err_f32:.6f} dB")
        output(f"    Q31 (SoX): {pb_err_q31:.6f} dB")
        output(f"    24-bit:    {pb_err_24bit:.6f} dB") 
        output(f"    16-bit:    {pb_err_16bit:.6f} dB")
    
    if sb_idx.any():
        # Actual stopband floors
        sb_floor_f32 = mag_f32[sb_idx].max()
        sb_floor_q31 = mag_q31[sb_idx].max()
        sb_floor_24bit = mag_24bit[sb_idx].max()
        sb_floor_16bit = mag_16bit[sb_idx].max()
        
        # Excess above theoretical format limits
        excess_f32 = sb_floor_f32 - ideal_floor_f32
        excess_q31 = sb_floor_q31 - ideal_floor_q31
        excess_24 = sb_floor_24bit - ideal_floor_24
        excess_16 = sb_floor_16bit - ideal_floor_16
        
        output(f"\n  Stopband floor (absolute):")
        output(f"    float32:   {sb_floor_f32:.1f} dB")
        output(f"    Q31 (SoX): {sb_floor_q31:.1f} dB")
        output(f"    24-bit:    {sb_floor_24bit:.1f} dB")
        output(f"    16-bit:    {sb_floor_16bit:.1f} dB")
        
        output(f"\n  Stopband excess above format's theoretical limit:")
        output(f"    float32:   {excess_f32:+6.1f} dB (ideal floor {ideal_floor_f32:.1f} dB)")
        output(f"    Q31 (SoX): {excess_q31:+6.1f} dB (ideal floor {ideal_floor_q31:.1f} dB)")
        output(f"    24-bit:    {excess_24:+6.1f} dB (ideal floor {ideal_floor_24:.1f} dB)")
        output(f"    16-bit:    {excess_16:+6.1f} dB (ideal floor {ideal_floor_16:.1f} dB)")
    
    # Effective bits - corrected formula using max_tap as reference
    # Return infinity for unmeasurably low noise
    def calc_eff_bits(rms, max_tap):
        if rms <= 0 or not np.isfinite(rms):
            return float('inf')  # Below measurement floor
        eff_bits = -20*np.log10(rms/max_tap) / 6.02
        return min(42.0, max(0, eff_bits))  # Cap at 42 for finite values
    
    eff_bits_f32 = calc_eff_bits(rms_f32, max_tap)
    eff_bits_q31 = calc_eff_bits(rms_q31, max_tap)
    eff_bits_24 = calc_eff_bits(rms_24bit, max_tap)
    eff_bits_16 = calc_eff_bits(rms_16bit, max_tap)
    
    output(f"\nEffective resolution:")
    output(f"  float32:   ~{eff_bits_f32:.1f} bits" if np.isfinite(eff_bits_f32) else "  float32:   ∞ bits (below measurement floor)")
    output(f"  Q31 (SoX): ~{eff_bits_q31:.1f} bits" if np.isfinite(eff_bits_q31) else "  Q31 (SoX): ∞ bits (below measurement floor)") 
    output(f"  24-bit:    ~{eff_bits_24:.1f} bits" if np.isfinite(eff_bits_24) else "  24-bit:    ∞ bits (below measurement floor)")
    output(f"  16-bit:    ~{eff_bits_16:.1f} bits" if np.isfinite(eff_bits_16) else "  16-bit:    ∞ bits (below measurement floor)")
    
    output(f"\n** For SoX usage: Check the 'Q31 (SoX)' row above **")
    
    # Write all output to log
    for line in output_lines:
        log.info(line.rstrip())


def print_filter_stats(taps: np.ndarray, spec: FilterSpec, log: logging.Logger):
    """Print comprehensive filter statistics and theory."""
    N = len(taps)
    k = int(np.log2(N)) if (N & (N-1)) == 0 else None
    
    # Determine effective parameters based on mode
    if spec.mode == 'upsample':
        design_fs = spec.fs_out
        nyq = spec.fs_out / 2
    else:  # downsample
        design_fs = spec.fs_in
        nyq = spec.fs_in / 2
    
    delta_w = spec.transition / nyq
    
    # Note: We're using scipy's kaiserord which dynamically computes beta,
    # not using the approximation formula. This shows what the approximation would give:
    A_approx = 2.285 * (N - 1) * delta_w + 7.95
    beta_approx = 0.1102 * (A_approx - 8.7) if A_approx > 50 else \
           (0.5842 * (A_approx - 21) ** 0.4 + 0.07886 * (A_approx - 21)) if A_approx > 21 else 0.0
    
    # Get actual beta used (recalculate with kaiserord)
    _, beta_actual = kaiserord(spec.sb_atten, 2 * spec.transition / design_fs)
    first_lobe_theory = -(spec.sb_atten - 13)

    print("\n" + "="*60)
    print("FILTER ANALYSIS REPORT")
    print("="*60)
    
    print("\nFilter Configuration")
    print("-" * 20)
    print(f"Mode           : {spec.mode.upper()} by {spec.factor}×")
    print(f"Input Fs       : {spec.fs_in:,} Hz")
    print(f"Output Fs      : {spec.fs_out:,} Hz")
    print(f"Cutoff         : {spec.cutoff} Hz")
    print(f"Transition     : {spec.transition} Hz")
    print(f"PB ripple spec : ±{spec.pb_ripple} dB")
    print(f"SB atten spec  : {spec.sb_atten} dB")
    
    print("\nFilter Design Theory")
    print("-" * 20)
    print(f"Taps           : {N:,} ({'2^'+str(k) if k is not None else 'not power of 2'})")
    print(f"Normalized Δf  : {delta_w:.7f}")
    print(f"Kaiser β (actual) : {beta_actual:.3f}")
    print(f"Kaiser β (approx formula): {beta_approx:.3f}")
    print(f"Theoretical stop-band (approx): {A_approx:.1f} dB")
    print(f"First lobe theory: {first_lobe_theory:.1f} dB")
    print(f"Design method: {spec.design_method}")
    if spec.design_method == 'kaiser':
        print(f"  Kaiser β (actual): {beta_actual:.3f}")
        print(f"  Kaiser β (approx formula): {beta_approx:.3f}")
        print(f"  (Using scipy's optimal beta from kaiserord)")
    else:
        print(f"  Parks-McClellan (Remez exchange) - minimax optimal")
    
    print("\nComputational Requirements")
    print("-" * 25)
    print(f"Kernel RAM (mono f64): {human_size(N*8)}")
    print(f"Kernel RAM (stereo f64): {human_size(N*16)}")
    
    if spec.mode == 'upsample':
        # For upsampling, we process at input rate but produce output rate
        macs_per_s = N * spec.fs_in
        latency_samples_in = N // (2 * spec.factor)
        latency_samples_out = N // 2
        latency_sec = latency_samples_in / spec.fs_in
    else:  # downsample
        # For downsampling, we process at input rate
        macs_per_s = N * spec.fs_in
        latency_samples_in = N // 2
        latency_samples_out = N // (2 * spec.factor)
        latency_sec = latency_samples_in / spec.fs_in
    
    print(f"Computational load: {macs_per_s/1e12:.2f} TMAC/s")
    print(f"Latency: {latency_samples_in} input samples = {latency_samples_out} output samples = {latency_sec*1000:.3f} ms")
    
    # DC gain
    dc_gain = np.sum(taps)
    dc_gain_db = 20 * np.log10(abs(dc_gain) + 1e-300)
    print(f"DC gain: {dc_gain:.12f} ({dc_gain_db:.9f} dB)")
    
    # Probe key frequencies
    print("\nFrequency Response Probes")
    print("-" * 25)
    probe_freqs = [0, spec.cutoff - spec.transition/2, spec.cutoff, 
                   spec.cutoff + spec.transition, spec.cutoff + 2*spec.transition]
    
    # For downsampling, add image band probes
    if spec.mode == 'downsample':
        for k in range(1, min(4, int(spec.factor) + 1)):  # Show first 3 image bands
            image_center = k * spec.fs_out - spec.cutoff
            if 0 < image_center < design_fs / 2:
                probe_freqs.append(image_center)
    
    # Ensure probes are within valid range
    max_freq = design_fs / 2
    probe_freqs = sorted(set(f for f in probe_freqs if 0 <= f < max_freq))
    probe_freqs.append(max_freq * 0.999)  # Near Nyquist
    
    w = 2 * np.pi * np.array(probe_freqs) / design_fs
    _, H = freqz(taps, worN=w)
    mag_db = 20 * np.log10(np.abs(H) + 1e-300)
    
    for f, m in zip(probe_freqs, mag_db):
        label = ""
        if abs(f) < 1e-6:
            label = " (DC)"
        elif abs(f - spec.cutoff) < 1e-6:
            label = " (cutoff, expect -6.02 dB)"
        elif abs(f - max_freq * 0.999) < 1e-6:
            label = " (near Nyquist)"
        elif spec.mode == 'downsample':
            # Check if this is an image band center
            for k in range(1, int(spec.factor) + 1):
                if abs(f - (k * spec.fs_out - spec.cutoff)) < 1e-6:
                    label = f" (image {k} of cutoff, aliases to {spec.cutoff} Hz)"
                    break
        print(f"  {f:10.2f} Hz: {m:9.4f} dB{label}")
    
    # SoX command examples
    print("\nSoX Command Examples")
    print("-" * 20)
    if spec.mode == 'upsample':
        print(f'sox "input.wav" -r {int(spec.fs_out)} -b 32 -t wav - \\')
        print(f'    upsample {spec.factor} \\')
        print(f'    fir <kernel.txt> \\')
        print(f'    -b 24 "output_{int(spec.fs_out/1000)}k.wav"')
    else:  # downsample
        print(f'sox "input.wav" -r {int(spec.fs_out)} -b 32 -t wav - \\')
        print(f'    rate -v -s {int(spec.fs_out)} \\')
        print(f'    fir <kernel.txt>')
        print("\nNote: For downsampling, apply filter BEFORE rate conversion")
    
    print("\n" + "="*60)


def analyze_saved_filter(npz_path: Path, verify_factor: int = 16, 
                        allow_shrink: bool = True, show_plot: bool = False,
                        log: logging.Logger = None) -> bool:
    """Analyze a previously saved filter from .npz file."""
    if log is None:
        log = logging.getLogger("analyze")
    
    log.info("Loading filter from %s...", npz_path)
    data = np.load(npz_path)
    
    # Extract taps and spec
    taps = data['taps']
    spec_dict = data['spec'].item()  # Convert 0-d array to dict
    spec = FilterSpec.from_dict(spec_dict)
    
    # Print comprehensive stats
    print_filter_stats(taps, spec, log)
    analyze_quantization_effects(taps, spec, log)
    
    # Run verification
    ok = verify_kernel(taps, spec, verify_factor, allow_shrink, log)
    
    if show_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed; skipping plot")
        else:
            # Use appropriate sample rate for plot
            if spec.mode == 'upsample':
                plot_fs = spec.fs_out
            else:
                plot_fs = spec.fs_in
            
            # Compute frequency response
            w, H = freqz(taps, worN=8192, fs=plot_fs)
            mag_db = 20 * np.log10(np.abs(H) + 1e-300)
            
            plt.figure(figsize=(10, 6))
            plt.semilogx(w[1:], mag_db[1:])
            plt.axvline(spec.cutoff, color='r', linestyle='--', label='Cutoff')
            plt.axvline(spec.cutoff - spec.transition, color='g', linestyle='--', label='Passband edge')
            plt.axvline(spec.cutoff + spec.transition, color='g', linestyle='--', label='Stopband edge')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.title(f"FIR magnitude response ({spec.mode} {spec.factor}×)")
            plt.grid(True, which="both", ls=":")
            plt.legend()
            plt.ylim(-max(180, spec.sb_atten + 20), 5)
            plt.show()
    
    return ok


# ─────────────────────────── CLI ──────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Exact-spec Kaiser FIR designer for integer-factor up/down-sampling.\n"
            "\n"
            "Example 1 – 32× upsample brick-wall:\n"
            "  python3 fir_enhanced.py \\\n"
            "      --rate-in 96000   --upsample 32 \\\n"
            "      --cutoff 48000    --transition 1.5625\n"
            "\n"
            "Example 2 – 8× downsample with tight specs:\n"
            "  python3 fir_enhanced.py \\\n"
            "      --rate-in 384000  --downsample 8 \\\n"
            "      --cutoff 20000    --transition 2000 \\\n"
            "      --pb-ripple 0.001 --sb-atten 150\n"
            "\n"
            "Example 3 – Analyze saved filter:\n"
            "  python3 fir_enhanced.py --analyze filter.npz --plot\n"
        ),
    )

    # ─── Mode selection ───
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--upsample", "-L", type=int,
                           help="Integer up-sample factor L. E.g., --upsample 4 means "
                                "multiply sample rate by 4.")
    mode_group.add_argument("--downsample", "-M", type=int,
                           help="Integer down-sample factor M. E.g., --downsample 8 means "
                                "divide sample rate by 8.")
    mode_group.add_argument("--analyze", type=Path,
                           help="Analyze existing .npz filter file instead of designing. "
                                "Shows comprehensive stats and verifies specifications.")

    # ─── Spec (audio quality) ───
    g = p.add_argument_group("Spec")
    g.add_argument("--pb-ripple", type=float, default=0.05,
                   help="Pass-band ripple tolerance δp (dB, **half** peak-to-peak). "
                        "E.g., 0.05 means ±0.05 dB deviation from 0 dB. "
                        "Use 0 for perfectly flat passband.")
    g.add_argument("--sb-atten", type=float, default=120,
                   help="Minimum stop-band attenuation As (dB). "
                        "E.g., 120 means stopband is at least 120 dB below passband. "
                        "Extreme values up to 400 dB are supported.")

    # ─── Sampling parameters ───
    g = p.add_argument_group("Sampling")
    g.add_argument("--rate-in", type=float, default=44_100,
                   help="Input sample-rate Fs (Hz). Common values: 44100, 48000, "
                        "96000, 192000, 384000.")
    g.add_argument("--cutoff", type=float,
                   help="Brick-wall cut-off frequency (Hz). For anti-aliasing, "
                        "this should be ≤ output_nyquist - transition.")
    g.add_argument("--transition", type=float,
                   help="Transition width Δf (Hz). Narrower = longer filter. "
                        "Defaults to Fs_in / 128. Can be as low as 0.05 Hz for surgical filtering.")

    # ─── Design method ───
    g = p.add_argument_group("Design method")
    g.add_argument("--method", choices=['kaiser', 'parks-mcclellan'], default='kaiser',
                   help="Filter design method. Kaiser is near-optimal with smooth response. "
                        "Parks-McClellan guarantees minimax optimality (smallest possible "
                        "maximum error) but may have more ripples.")

    # ─── Length control ───
    g = p.add_argument_group("Length control")
    g.add_argument("--taps-pow2", type=int,
                   help="Force tap count to 2^k (e.g., --taps-pow2 20 = 1M taps). "
                        "If omitted, automatically finds smallest filter that meets specs.")
    g.add_argument("--no-pow2", action="store_true",
                   help="Allow non-power-of-2 tap counts for minimum length. "
                        "Saves RAM but may be slightly slower to compute. "
                        "Parks-McClellan designs may take longer with non-power-of-2 lengths.")
    g.add_argument("--force", action="store_true",
                   help="Override RAM safety limits. Required for extreme designs "
                        "(>2^27 taps on limited RAM, or specifications requiring 2^28+ taps).")

    # ─── Verification knobs ───
    g = p.add_argument_group("Verification")
    g.add_argument("--verify-factor", type=int, default=16,
                   help="FFT oversample factor for verification (default: 16). "
                        "Higher values give more accurate verification but need more RAM/time. "
                        "Lower to 4-8 for quick checks, raise to 32-64 for publication-quality validation.")
    g.add_argument("--no-auto-shrink", action="store_true",
                   help="Fail immediately if verification FFT would exceed RAM, "
                        "instead of automatically reducing resolution.")
    g.add_argument("--skip-verification", action="store_true",
                   help="Skip verification entirely (DANGEROUS!). Filter is saved but NOT "
                        "verified to meet specifications. Use only for iterative design. "
                        "Always verify before production use with --analyze.")

    # ─── Output/Analysis ───
    g = p.add_argument_group("Output/Analysis")
    g.add_argument("--basename",
                   help="Custom filename stem for outputs. Default auto-generates "
                        "descriptive names like 'u32_96000to3072000Hz_2^26'.")
    g.add_argument("--plot", action="store_true",
                   help="Show magnitude response plot (requires matplotlib). "
                        "Displays cutoff, transition bands, and achieved response.")
    g.add_argument("--no-analysis", action="store_true",
                   help="Skip the detailed analysis report after design. "
                        "Useful for batch processing or when you only need the filter coefficients.")

    # ─── Misc ───
    g = p.add_argument_group("Misc")
    g.add_argument("--debug", action="store_true",
                   help="Enable DEBUG-level logging for extra detail.")
    g.add_argument("--log-file", type=str,
                   help="Write all output to specified log file in addition to console. "
                        "Useful for documenting design decisions and verification results.")

    a = p.parse_args()

    # Set up logging with optional file output
    log_handlers = [logging.StreamHandler(sys.stderr)]
    
    # Choose format based on whether we're logging to file
    if a.log_file:
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        
        # Create log file with timestamp if not specified
        if a.log_file == 'auto':
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"fir_design_{timestamp}.log"
        else:
            log_filename = a.log_file
        
        file_handler = logging.FileHandler(log_filename, mode='w')
        log_handlers.append(file_handler)
    else:
        log_format = "%(levelname)s %(message)s"
    
    logging.basicConfig(
        handlers=log_handlers,
        level=logging.DEBUG if a.debug else logging.INFO,
        format=log_format
    )
    log = logging.getLogger("fir")
    
    if a.log_file:
        log.info("Logging to file: %s", log_filename if 'log_filename' in locals() else a.log_file)

    # ── Validate design mode ──
    if a.analyze:
        ok = analyze_saved_filter(a.analyze, a.verify_factor, 
                                 not a.no_auto_shrink, a.plot, log)
        sys.exit(0 if ok else 1)

    # ── Validate design mode ──
    if not a.upsample and not a.downsample:
        p.error("Must specify either --upsample or --downsample (or --analyze)")
    
    if not a.cutoff:
        p.error("--cutoff is required for filter design")
    
    if a.taps_pow2 and a.no_pow2:
        p.error("Cannot specify both --taps-pow2 and --no-pow2")

    # ── Create filter spec ──
    if a.upsample:
        mode = 'upsample'
        factor = a.upsample
        fs_out = a.rate_in * factor
    else:  # downsample
        mode = 'downsample'
        factor = a.downsample
        fs_out = a.rate_in / factor
    
    spec = FilterSpec(
        mode=mode,
        factor=factor,
        fs_in=a.rate_in,
        fs_out=fs_out,
        cutoff=a.cutoff,
        transition=a.transition or a.rate_in / 128,
        pb_ripple=a.pb_ripple,
        sb_atten=a.sb_atten,
        forced_k=a.taps_pow2,
        design_method=a.method,
        no_pow2=a.no_pow2
    )

    # ── design (auto-iterate if length not forced) ──
    attempt = 0
    k = spec.forced_k
    while True:
        if attempt == 0:
            spec.forced_k = k
        else:
            # For auto-retry, increment the power-of-two exponent
            if k is not None:
                spec.forced_k = k + 1
            else:
                # If k was None (odd Parks-McClellan), compute next power of 2
                # Use the actual N from design, not len(taps) which might be different
                current_n = len(taps)
                next_k = int(np.ceil(np.log2(current_n)))
                if (1 << next_k) == current_n:
                    next_k += 1  # If already power of 2, go to next
                spec.forced_k = next_k
        
        taps, fs_out, k, Δf = design_kernel(spec, a.force, log)
        spec.transition = Δf  # Update with actual value

        if a.skip_verification:
            log.warning("VERIFICATION SKIPPED! Filter has NOT been verified to meet specs.")
            log.warning("This filter should be verified with --analyze before production use.")
            # Print red banner to stdout for CI visibility
            print("\n" + "="*60)
            print("⚠️  WARNING: VERIFICATION SKIPPED!")
            print("This filter has NOT been verified to meet specifications.")
            print("Run with --analyze before production use!")
            print("="*60 + "\n")
            ok = True  # Assume OK but mark as unverified
        else:
            ok = verify_kernel(taps, spec, a.verify_factor,
                              not a.no_auto_shrink, log)
        
        if ok or a.taps_pow2 is not None:
            break
        attempt += 1
        log.info("Increasing to next size and retrying …")

    # ── Run analysis unless disabled ──
    if not a.no_analysis:
        print_filter_stats(taps, spec, log)
        analyze_quantization_effects(taps, spec, log)

    # ── Save outputs ──
    if k is not None:
        label = f"2^{k}"
    else:
        label = f"{len(taps)}tap"
    
    if mode == 'upsample':
        stem = a.basename or f"u{factor}_{int(a.rate_in)}to{int(fs_out)}Hz_{label}"
    else:
        stem = a.basename or f"d{factor}_{int(a.rate_in)}to{int(fs_out)}Hz_{label}"
    
    # Save text format
    txt = Path(stem + ".txt")
    np.savetxt(txt, taps, fmt="%.18e")
    
    # Save binary formats
    np.save(stem + ".npy", taps.astype(np.float32))
    
    # Save comprehensive .npz with metadata
    metadata = {
        'spec': spec.to_dict(),
        'command_line': ' '.join(sys.argv),
        'verified': not a.skip_verification
    }
    np.savez(stem + ".npz", 
             taps=taps.astype(np.float32),
             **metadata)
    
    log.info("Saved %s.txt, %s.npy, and %s.npz", stem, stem, stem)
    
    if a.skip_verification:
        log.warning("Files marked as UNVERIFIED - verify with: python3 %s --analyze %s.npz", 
                   sys.argv[0], stem)
    
    # Save analysis log if requested
    if a.log_file:
        log.info("\nComplete analysis has been saved to: %s", 
                 log_filename if 'log_filename' in locals() else a.log_file)

    # ── Display example commands ──
    print("\n=== Example Usage Commands ===")
    if mode == 'upsample':
        print(f"sox \"input.wav\" -r {int(fs_out)} -b 32 -t wav - \\")
        print(f"    upsample {factor} \\")
        print(f"    fir {txt.name} \\")
        print(f"    -b 24 \"output_{int(fs_out/1000)}k.wav\"")
    else:
        print(f"sox \"input.wav\" -b 32 -t wav - \\")
        print(f"    fir {txt.name} \\")
        print(f"    rate -v -s {int(fs_out)} \\") 
        print(f"    -b 24 \"output_{int(fs_out/1000)}k.wav\"")
        print("\nNote: For downsampling, filter MUST be applied before rate conversion!")
    
    print(f"\nTo re-analyze this filter later:")
    print(f"  python3 {Path(__file__).name} --analyze {stem}.npz --plot")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Fatal: %s", e)
        logging.debug("Traceback:", exc_info=True)
        sys.exit(1)
