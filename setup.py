"""
Setup script for naim_simple package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="naim-simple",
    version="0.1.0",
    author="Emilio Gordillo Esparragoza",
    description="A simplified scikit-learn-like wrapper for the NAIM model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/naim-simple",
    license="MIT",
    license_files=("LICENSE.txt",),
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4.0",
        "click>=8.0.0",
        "pydantic>=1.8.0",
    ],
    entry_points={
        "console_scripts": [
            "naim-train=naim_simple.cli:cli",
            "naim-predict=naim_simple.cli:cli",
        ],
    },
)