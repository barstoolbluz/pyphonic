"""Setup script for pyphonic."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyphonic",
    version="0.1.0",
    author="Your Name",
    description="Mathematically rigorous DSP coefficient generation for audio processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyphonic",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.6.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
)