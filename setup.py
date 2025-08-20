#!/usr/bin/env python3
"""Setup script for ddp_kbit package."""

from setuptools import setup, find_packages

setup(
    name="ddp-kbit",
    version="0.1.0",
    description="Distributed Deep Learning System with K-Bit compression",
    author="KBIT Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch==2.4.0",
        "torchvision==0.19.0",
        "pyspark==3.5.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "matplotlib=3.4.3",
        "seaborn==0.13.0",
        "scikit-learn==1.4.2",
        "tqdm==4.66.1",
        "kafka-python==2.0.2",
        "scipy==1.13.0",
        "Pillow==10.1.0",
        "PyYAML==6.0.1",
        "protobuf==5.27.0",
        "pymongo==4.13.2",
        "pyarrow==13.0.0",
        "joblib==1.3.2",
        "h5py==3.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ddp-kbit=ddp_kbit.main:main",
            "ddp_kbit=ddp_kbit.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)