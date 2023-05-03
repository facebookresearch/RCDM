# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from setuptools import setup

setup(
    name="guided-diffusion-rcdm",
    py_modules=["guided_diffusion_rcdm"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "numpy", "torchvision", "faiss-gpu"],
)
