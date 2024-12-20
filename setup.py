# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import find_packages, setup

setup(
    name="uco3d",
    version="1.0",
    packages=find_packages(exclude=["tests", "dataset_download", "examples"]),
    install_requires=[
        "sqlalchemy>=2.0",
        "pandas",
        "tqdm",
        "torchvision",
        "torch",
        "matplotlib",
        "plyfile",
        "h5py",
    ],
)
