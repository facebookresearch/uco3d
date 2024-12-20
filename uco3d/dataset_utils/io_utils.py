# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

import h5py
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from .data_types import PointCloud


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))

    return transpose_normalize_image(im)


def load_mask(path: str) -> np.ndarray:
    with Image.open(path) as pil_im:
        mask = np.array(pil_im)

    return transpose_normalize_image(mask)


def load_depth(path: str, scale_adjustment: float) -> np.ndarray:
    if path.lower().endswith(".exr"):
        # NOTE: environment variable OPENCV_IO_ENABLE_OPENEXR must be set to 1 before
        # importing cv2. You will have to accept these vulnerabilities by using OpenEXR:
        # https://github.com/opencv/opencv/issues/21326
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        import cv2

        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = load_16bit_png_depth(path)
    elif path.lower().endswith(".npy"):
        d = np.load(path)
    else:
        raise ValueError('unsupported depth file name "%s"' % path)

    assert len(d.shape) == 2

    d = d * scale_adjustment

    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel


def load_16bit_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def load_h5_depth(path: str, frame_num: int) -> np.ndarray:
    with h5py.File(path, "r") as h5file:
        depth_map = h5file[str(frame_num)][:].astype(np.float32)

    return depth_map


def load_1bit_png_mask(file: str) -> np.ndarray:
    with Image.open(file) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def load_depth_mask(path: str) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth mask file name "%s"' % path)
    m = load_1bit_png_mask(path)
    return m[None]  # fake feature channel


def load_point_cloud(path: str):
    ply_data = PlyData.read(path)
    xyz = torch.tensor(
        ply_data.elements[0].data[["x", "y", "z"]].tolist(),
        dtype=torch.float32,
    )
    rgb = (
        torch.tensor(
            ply_data.elements[0].data[["red", "green", "blue"]].tolist(),
            dtype=torch.float32,
        )
        / 255.0
    )
    return PointCloud(xyz, rgb)


def transpose_normalize_image(image: np.ndarray) -> np.ndarray:
    im = np.atleast_3d(image).transpose((2, 0, 1))
    return im.astype(np.float32) / 255.0
