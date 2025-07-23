# Pyphonic

Mathematically rigorous DSP coefficient generation for audio processing.

## Overview

Pyphonic provides reference implementations for generating high-precision filter coefficients used in audio resampling and filtering applications. All implementations prioritize mathematical correctness and serve as golden references for production implementations.

## Features

- **FIR Filter Design**: Parks-McClellan, windowed sinc, and Kaiser optimal filters
- **Sinc Interpolation Tables**: For arbitrary sample rate conversion
- **Polyphase Filter Banks**: Efficient structures for rational and irrational rate conversion
- **Extreme Precision**: Support for up to 2^24 taps with -192 dB stopband attenuation
- **Comprehensive Verification**: FFT-based analysis and comparison with theoretical limits

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
# Generate high-quality sinc interpolation table
from pyphonic import SincTableGenerator

generator = SincTableGenerator(
    zero_crossings=32,
    oversample_factor=512,
    window='kaiser',
    beta=14.0
)

table, metadata = generator.generate()
generator.save('output/sinc_32x512_kaiser.npz')
```

## Scripts

- `fir_filter_gen.py` - FIR filter coefficient generation
- `sinc_table_gen.py` - Sinc interpolation table generation
- `polyphase_gen.py` - Polyphase filterbank generation
- `verify_coefficients.py` - Verification and analysis tools

## Quality Targets

| Specification | Target | Achieved |
|--------------|--------|----------|
| Passband ripple | ±1e-5 dB | ✓ |
| Stopband attenuation | -192 dB | ✓ |
| Phase linearity | < 1e-6 samples | ✓ |
| Coefficient precision | Float64 | ✓ |

## License

MIT

## Contributing

This repository serves as the mathematical reference for audio DSP implementations. All contributions must maintain numerical precision and include verification tests.