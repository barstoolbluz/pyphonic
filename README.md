# Pyphonic

Mathematically rigorous DSP coefficient generation for audio processing.

## Overview

Pyphonic provides reference implementations for generating high-precision filter coefficients used in audio resampling and filtering applications. All implementations prioritize mathematical correctness and serve as golden references for production implementations.

## Features

- **FIR Filter Design**: Parks-McClellan, windowed sinc, and Kaiser optimal filters with extreme precision support
- **Sinc Interpolation Tables**: Fully vectorized generation for arbitrary sample rate conversion  
- **Polyphase Filter Banks**: Mathematically exact center-tap removal for perfect symmetry
- **Extreme Precision**: Support for up to 2^30 taps with -400 dB stopband attenuation
- **Comprehensive Verification**: FFT-based analysis and comparison with theoretical limits

## Installation

```bash
pip install -r requirements.txt
```

## Filter Implementations

### 1. FIR Filter Generator (`src/fir_filter_gen.py`)

The most sophisticated implementation supporting both Kaiser windows and Parks-McClellan (Remez exchange) design with extreme specifications.

#### Key Features
- Handles extreme specifications (ripple to 0.000001 dB, attenuation to 400+ dB)
- RAM-adaptive verification with alias-band checking for downsampling
- Quantization analysis for fixed-point implementations
- Comprehensive frequency response analysis

#### Basic Usage

```python
from src.fir_filter_gen import FilterSpec, design_kernel

# Basic 32x upsampling filter
spec = FilterSpec(
    mode='upsample',
    factor=32,
    fs_in=96000,
    fs_out=96000 * 32,
    cutoff=48000,
    transition=1500,
    pb_ripple=0.05,
    sb_atten=120.0
)

taps, fs_out, k, delta_f = design_kernel(spec, force_ram=False, log=logger)
```

#### Command Line Examples

```bash
# Basic upsampling 32× with Kaiser window
python3 src/fir_filter_gen.py \
    --rate-in 96000 --upsample 32 --cutoff 48000 \
    --transition 1500 --pb-ripple 0.05 --sb-atten 120

# High-precision upsampling with specific tap count
python3 src/fir_filter_gen.py \
    --rate-in 96000 --upsample 32 --cutoff 48000 \
    --taps-pow2 20 --transition 1500 --pb-ripple 0.001 --sb-atten 150

# Downsampling 8× with Parks-McClellan for minimax optimality
python3 src/fir_filter_gen.py \
    --rate-in 384000 --downsample 8 --cutoff 22000 \
    --transition 2000 --pb-ripple 0.001 --sb-atten 150 \
    --method parks-mcclellan

# Extreme precision: 0.000001 dB ripple, 400 dB attenuation
python3 src/fir_filter_gen.py \
    --rate-in 192000 --downsample 4 --cutoff 40000 \
    --transition 500 --pb-ripple 0.000001 --sb-atten 400 \
    --method parks-mcclellan --force

# Ultra-narrow transition (0.1 Hz!) for surgical filtering
python3 src/fir_filter_gen.py \
    --rate-in 48000 --upsample 1 --cutoff 1000 \
    --transition 0.1 --pb-ripple 0.01 --sb-atten 120 --force

# Analyze existing filter with detailed report and plot
python3 src/fir_filter_gen.py --analyze my_filter.npz --plot --verify-factor 32
```

#### Advanced Features

**Quantization Analysis**: Automatic analysis of float32, Q31, 24-bit, and 16-bit quantization effects:
```bash
# Includes comprehensive quantization report
python3 src/fir_filter_gen.py --rate-in 48000 --upsample 4 --cutoff 20000 --transition 1000
```

**Memory Management**: Automatic RAM monitoring and adaptive verification:
```bash
# For extreme designs requiring >100GB RAM
python3 src/fir_filter_gen.py --rate-in 192000 --downsample 8 --cutoff 20000 \
    --transition 10 --sb-atten 300 --force --verify-factor 64
```

### 2. Sinc Interpolation Table Generator (`src/sinc_table_gen.py`)

Production-grade sinc table generation with perfect symmetry guarantees, comprehensive verification, and optimized performance.

#### Key Features
- **Perfect symmetry**: Generates only right half and mirrors for exact symmetry (< 1e-15 error)
- **Numerical stability**: Uses `np.sinc()` for stable near-zero computation
- **Dual formats**: Linear and polyphase table outputs with optional per-phase normalization
- **Multiple windows**: Kaiser, Blackman-Harris, Hann, Hamming, rectangular
- **Production verification**: FFT-based stopband tests, DC gain validation, energy conservation checks
- **Optimized extraction**: Fully vectorized polyphase generation (~100× faster than loops)
- **Precision control**: Explicit float32/float64 output selection

#### Basic Usage

```python
from src.sinc_table_gen import generate_sinc_table, generate_polyphase_table

# Generate linear sinc table
kernel, metadata = generate_sinc_table(
    N_taps=32,           # Zero crossings on each side (NOT non-zero lobes)
    oversample=512,      # Oversampling factor
    cutoff=0.5,         # Normalized cutoff (0-0.5)
    win_type='kaiser',   # Window type
    beta=8.6            # Kaiser beta parameter
)

# Generate polyphase format table with optional normalization
table, metadata = generate_polyphase_table(
    N_taps=16,              # Must be even for symmetric extraction
    oversample=256,
    cutoff=0.45,
    win_type='kaiser',
    beta=12.0,
    normalize_phases=True   # Unity DC gain per phase
)
```

#### Command Line Examples

```bash
# Basic sinc table with Kaiser window
python3 src/sinc_table_gen.py --zeros 32 --oversample 512 \
    --window kaiser --beta 8.6 --verify

# High-quality polyphase table with per-phase normalization
python3 src/sinc_table_gen.py --zeros 16 --oversample 256 \
    --cutoff 0.45 --window kaiser --beta 12.0 --polyphase \
    --normalize-phases --precision float64 --verify

# Extreme precision table with Blackman-Harris window
python3 src/sinc_table_gen.py --zeros 64 --oversample 1024 \
    --window blackman-harris --precision float64 --verify

# Custom cutoff frequency for specific applications
python3 src/sinc_table_gen.py --zeros 24 --oversample 128 \
    --cutoff 0.4 --window kaiser --beta 14.0 --basename custom_sinc

# Generate both formats with full verification
python3 src/sinc_table_gen.py --zeros 32 --oversample 512 \
    --format both --precision float64 --verify --debug
```

#### Production Features

**Enhanced Verification Suite**:
```bash
# Comprehensive production validation
python3 src/sinc_table_gen.py --zeros 16 --oversample 64 --verify
# Tests include:
# - Symmetry: < 1e-15 error
# - DC gain: < 1e-14 error  
# - Stopband: Validated against Kaiser theory
# - Peak position: Correct within ±1 sample
# - Energy conservation: Within reasonable bounds
```

**Per-Phase DC Normalization** (for high-SNR applications):
```bash
# Ensure every polyphase row has exactly unity DC gain
python3 src/sinc_table_gen.py --zeros 8 --oversample 32 --polyphase \
    --normalize-phases --verify
# Result: All phases DC gain = 1.000000 (exact)
```

**Precision Control**:
```bash
# Float64 for mastering/reference applications
python3 src/sinc_table_gen.py --zeros 32 --oversample 512 \
    --precision float64 --polyphase

# Float32 for embedded/real-time systems (default)
python3 src/sinc_table_gen.py --zeros 16 --oversample 256 \
    --precision float32
```

#### Multiple Window Functions

```bash
# Kaiser window (most common)
python3 src/sinc_table_gen.py --zeros 32 --oversample 512 --window kaiser --beta 8.6

# Blackman-Harris (excellent stopband)
python3 src/sinc_table_gen.py --zeros 32 --oversample 512 --window blackman-harris

# Hann window (simple raised cosine)
python3 src/sinc_table_gen.py --zeros 24 --oversample 256 --window hann

# Custom rectangular (no windowing)
python3 src/sinc_table_gen.py --zeros 16 --oversample 128 --window rectangular
```

### 3. Polyphase Filter Bank Generator (`src/polyphase_gen_exact.py`)

The mathematically exact polyphase implementation using proper center-tap removal for perfect symmetry.

#### Key Features
- **Perfect symmetry**: Uses center-tap removal instead of contiguous extraction
- **Zero reconstruction error**: Mathematically exact polyphase decomposition
- **Industry standard**: T×L+1 kernel length with proper extraction
- **Comprehensive verification**: Tests symmetry, reconstruction, and linear phase

#### Basic Usage

```python
from src.polyphase_gen_exact import PolyphaseGenerator

# Create high-quality polyphase filter bank
generator = PolyphaseGenerator(
    taps_per_phase=32,    # T: taps per phase
    phase_count=64,       # L: number of phases
    cutoff=0.5,          # Normalized cutoff
    window='kaiser',      # Window type
    stopband_db=180.0    # Target stopband attenuation
)

# Generate the filter bank
table, metadata = generator.generate()

# Save with comprehensive metadata
generator.save('polyphase_32x64_180db')
```

#### Advanced Usage

```python
# Extreme precision polyphase bank
generator = PolyphaseGenerator(
    taps_per_phase=64,
    phase_count=256,
    cutoff=0.45,
    window='kaiser',
    window_param=15.0,    # Custom Kaiser beta
    stopband_db=192.0
)

table, metadata = generator.generate()

# Verification results are in metadata['verification']
verification = metadata['verification']
print(f"Perfect reconstruction: {verification['perfect_reconstruction']}")
print(f"Reconstruction error: {verification['reconstruction_error']:.2e}")
print(f"Kernel symmetric: {verification['symmetry_pass']}")
```

#### Command Line Examples

```bash
# Basic polyphase filter bank
python3 src/polyphase_gen_exact.py --taps 32 --phases 64 --stopband 180 --verbose

# High-quality bank for professional resampling
python3 src/polyphase_gen_exact.py --taps 64 --phases 256 --stopband 192 --verbose

# Run property-based tests across parameter space
python3 src/polyphase_gen_exact.py --test

# Generate and save with custom output name
python3 src/polyphase_gen_exact.py --taps 48 --phases 128 --stopband 160 \
    --output professional_resampler --verbose
```

#### Mathematical Verification

The implementation includes comprehensive verification:

```python
# All verification results are automatically included
verification = metadata['verification']

# Key mathematical properties
assert verification['perfect_reconstruction']  # Reconstruction error = 0.0
assert verification['symmetry_pass']          # Kernel perfectly symmetric  
assert verification['dc_normalized']          # Unity DC gain per row
assert verification['odd_kernel']             # Odd-length kernel for linear phase

# Performance metrics
print(f"Measured stopband: {verification['measured_stopband_db']:.1f} dB")
print(f"Passband ripple: {verification['passband_ripple_db']:.6f} dB")
```

## Quality Targets

| Specification | FIR Filter | Sinc Table | Polyphase |
|--------------|------------|------------|-----------|
| Passband ripple | ±1e-6 dB | ±1e-5 dB | ±1e-5 dB |
| Stopband attenuation | -400 dB | -200 dB | -192 dB |
| Phase linearity | Perfect | Perfect | Perfect |
| Reconstruction error | N/A | < 1e-14 | 0.0 (exact) |
| Coefficient precision | Float64 | Float64 | Float64 |
| Max filter length | 2^30 taps | 2^20 samples | 2^20 coeffs |

## Mathematical Correctness

### Center-Tap Removal (Polyphase)

The polyphase implementation uses the mathematically correct approach:

```python
# CORRECT: Remove center tap for perfect symmetry
center = len(kernel) // 2
kernel_slice = np.concatenate((kernel[:center], kernel[center+1:]))

# INCORRECT: Contiguous extraction (creates 0.5 sample offset)
# kernel_slice = kernel[0:T*L]  # DON'T DO THIS
```

### Perfect Reconstruction Verification

```python
# Verify polyphase decomposition reconstructs original
reconstruction = np.zeros(total_coeffs)
for p in range(phase_count):
    upsampled = np.zeros(total_coeffs)
    upsampled[p::phase_count] = table_unnorm[p]
    reconstruction += upsampled

# Should be exactly zero for perfect reconstruction
error = np.max(np.abs(reconstruction - expected_slice))
assert error < 1e-15  # Machine precision limit
```

### Symmetry Guarantees (Sinc)

```python
# Sinc tables guarantee perfect symmetry by construction
center = len(kernel) // 2
left = kernel[:center]
right = kernel[center:][::-1]
symmetry_error = np.max(np.abs(left - right))
assert symmetry_error < 1e-15  # Perfect within machine precision
```

## Integration Examples

### SoX Integration
```bash
# Use generated FIR coefficients with SoX
sox input.wav -r 192000 -t wav - upsample 4 fir my_filter.txt output.wav
```

### Production Pipeline
```python
# Generate coefficient sets for different quality levels
qualities = [
    ('broadcast', {'sb_atten': 120, 'pb_ripple': 0.01}),
    ('mastering', {'sb_atten': 180, 'pb_ripple': 0.001}), 
    ('reference', {'sb_atten': 250, 'pb_ripple': 0.0001})
]

for name, params in qualities:
    # Generate FIR coefficients
    fir_cmd = f"python3 src/fir_filter_gen.py --rate-in 48000 --upsample 4 " \
              f"--cutoff 20000 --sb-atten {params['sb_atten']} " \
              f"--pb-ripple {params['pb_ripple']} --basename {name}_4x"
    
    # Generate polyphase bank  
    poly_cmd = f"python3 src/polyphase_gen_exact.py --taps 32 --phases 64 " \
               f"--stopband {params['sb_atten']} --output {name}_polyphase"
```

## License

MIT

## Contributing

This repository serves as the mathematical reference for audio DSP implementations. All contributions must maintain numerical precision and include comprehensive verification tests.
