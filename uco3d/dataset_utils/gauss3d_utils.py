# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import dataclasses
import json
import math
import os

from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import DictConfig
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from . import gauss3d_convert

from .data_types import GaussianSplats


def load_compressed_gaussians(
    compressed_dir: str,
    load_higher_order_harms: bool = True,
) -> GaussianSplats:
    """
    Load compressed Gaussian splats from a directory.
    """
    with open(os.path.join(compressed_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    splats = {}
    n = meta["means"]["shape"][0]
    assert tuple(meta["sh0"]["shape"]) == (n, 1, 3)
    # adjust the sh0 shape to match our convention
    meta["sh0"]["shape"] = (n, 3)
    for param_name, param_meta in meta.items():
        if not load_higher_order_harms and param_name == "shN":
            continue
        decompress_fn = _get_decompress_fn(param_name)
        param_val = decompress_fn(compressed_dir, param_name, param_meta)
        if param_name == "means":
            param_val = _inverse_log_transform(param_val)
        tgt_shape = meta[param_name]["shape"]
        assert (
            tgt_shape[0] == n
        ), f"# gaussians mismatch for {param_name} ({n} vs {tgt_shape[0]})"
        splats[param_name] = param_val.reshape(tgt_shape)
    return GaussianSplats(**splats)


def truncate_bg_gaussians(splats: GaussianSplats) -> GaussianSplats:
    """
    Remove background splats from the Gaussian splats.
    """
    assert splats.fg_mask is not None, "fg_mask is not present in splats"
    splats_dict = dataclasses.asdict(copy.copy(splats))
    fg_mask = splats.fg_mask.reshape(-1) > 0.5
    for k, v in splats_dict.items():
        if torch.is_tensor(v):
            splats_dict[k] = v[fg_mask]
    return GaussianSplats(**splats_dict)


@torch.no_grad()
def save_gsplat_ply(splats: GaussianSplats, path: str):
    """
    Save gsplats to a ply file following the standard gsplat convention.

    The result ply file can be visualized in standard 3D viewers.
    E.g. in https://antimatter15.com/splat/.
    """
    splats = dataclasses.asdict(splats)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    splats = splats.copy()
    xyz = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = splats["sh0"].detach().flatten(start_dim=1).contiguous().cpu().numpy()
    n = xyz.shape[0]
    if splats.get("shN", None) is None:  # add dummy 0 degree harmonics
        splats["shN"] = torch.zeros(f_dc.shape[0], 1, 3).float()
    else:
        splats["shN"] = splats["shN"].reshape(n, -1, 3)
    f_rest = (
        splats["shN"]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    opacities = splats["opacities"].detach().reshape(-1, 1).cpu().numpy()
    scale = splats["scales"].detach().cpu().numpy()
    rotation = splats["quats"].detach().cpu().numpy()
    dtype_full = [
        (attribute, "f4") for attribute in _construct_list_of_attributes(splats)
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation),
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def transform_gaussian_splats(
    splats_dataclass: GaussianSplats,
    R: torch.Tensor,
    T: torch.Tensor,
    s: torch.Tensor,
) -> GaussianSplats:
    """
    Apply a transformation to the Gaussian splats.

    The convention for the rotation `R`, translation `T`, and scale `s` is:
    ```
    x_transformed = (x @ R + T) * s
    ```

    Args:
        splats (GaussianSplats): Gaussian splats
        R (torch.Tensor): Tensor of shape (3, 3) containing the rotation matrix
        T (torch.Tensor): A Tensor of shape (3,) contaning the translation vector
        s (torch.Tensor): A scalar Tensor containing the scale vector
    """

    # start = time.time()

    splats = dataclasses.asdict(splats_dataclass)

    if splats.get("shN", None) is not None:
        sh_degree = math.log2(splats["shN"].shape[-2] + 1) - 1
        assert sh_degree.is_integer()
        sh_degree = int(sh_degree)
    else:
        sh_degree = 0

    N = splats["means"].shape[0]
    splat_data = gauss3d_convert.SplatData(
        xyz=splats["means"].numpy(),
        features_dc=splats["sh0"].numpy()[..., None],
        features_rest=(
            splats["shN"].numpy().transpose(0, 2, 1) if sh_degree > 0 else None
        ),
        opacity=splats["opacities"].numpy(),
        scaling=splats["scales"].numpy(),
        rotation=splats["quats"].numpy(),
        active_sh_degree=sh_degree,
    )

    R_ = R.t().numpy()
    T_ = (T * s).numpy()
    s_ = (s).numpy()

    rot_eul = Rotation.from_matrix(R_).as_euler("xyz", degrees=True)
    transform_conf = DictConfig(
        dict(
            transform=dict(
                position=T_.tolist(),
                rotation=rot_eul.tolist(),
                scale=s_.tolist(),
            ),
            unity_transform=False,
            rotate_sh=sh_degree > 0,
            max_sh_degree=sh_degree,
        )
    )
    splat_data_t = gauss3d_convert.transform_data(transform_conf, splat_data)

    splats = splats.copy()
    splats.update(
        {
            "means": torch.from_numpy(splat_data_t.xyz).float(),
            "sh0": torch.from_numpy(splat_data_t.features_dc).float()[..., 0],
            "opacities": torch.from_numpy(splat_data_t.opacity).float(),
            "scales": torch.from_numpy(splat_data_t.scaling).float(),
            "quats": torch.from_numpy(splat_data_t.rotation).float(),
        }
    )
    if sh_degree > 0:
        splats["shN"] = (
            torch.from_numpy(splat_data.features_rest)
            .reshape(N, 3, -1)
            .permute(0, 2, 1)
        )

    # print(f"Applying alignment took {time.time() - start:.5f}s")

    return GaussianSplats(**splats)


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to sh0.

    Args:
        sh0 (torch.Tensor): Tensor of shape (..., 3) containing the RGB values

    Returns:
        torch.Tensor: Tensor of shape (..., 3) containing the sh0 values
    """
    return (rgb - 0.5) / 0.2820947917738781


def _construct_list_of_attributes(splats):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(splats["sh0"].shape[-1]):
        l.append("f_dc_{}".format(i))
    for i in range(splats["shN"].shape[-2:].numel()):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(splats["scales"].shape[-1]):
        l.append("scale_{}".format(i))
    for i in range(splats["quats"].shape[-1]):
        l.append("rot_{}".format(i))
    return l


def _get_decompress_fn(param_name: str):
    decompress_fn_map = _get_decompress_fn_map()
    if param_name in decompress_fn_map:
        return decompress_fn_map[param_name]
    raise NotImplementedError(
        f"Decompression function for {param_name} is not implemented"
    )


def _get_decompress_fn_map():
    decompress_fn_map = {
        "means": _decompress_png_16bit,
        "scales": _decompress_png,
        "quats": _decompress_png,
        "opacities": _decompress_png,
        "sh0": _decompress_png,
        "shN": _decompress_kmeans,
        "fg_mask": _decompress_npz,
    }
    return decompress_fn_map


def _decompress_png_16bit(
    compress_dir: str, param_name: str, meta: Dict[str, Any], resize=False
) -> torch.Tensor:
    """Decompress parameters from PNG files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """

    if not np.all(meta["shape"]):
        return meta

    img_l = imageio.imread(os.path.join(compress_dir, f"{param_name}_l.png"))
    img_u = imageio.imread(os.path.join(compress_dir, f"{param_name}_u.png"))
    img_u = img_u.astype(np.uint16)

    img = (img_u << 8) + img_l

    img_norm = img / (2**16 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    if resize:
        grid = grid.reshape(meta["shape"])
    grid = grid.to(dtype=getattr(torch, meta["dtype"]))
    return grid


def _decompress_png(
    compress_dir: str, param_name: str, meta: Dict[str, Any], resize=False
) -> torch.Tensor:
    """Decompress parameters from PNG file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return params

    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    if resize:
        grid = grid.reshape(meta["shape"])
    grid = grid.to(dtype=getattr(torch, meta["dtype"]))
    return grid


def _decompress_kmeans(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> torch.Tensor:
    """Decompress parameters from K-means compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return params

    npz_dict = np.load(os.path.join(compress_dir, f"{param_name}.npz"))
    centroids_quant = npz_dict["centroids"]
    labels = torch.tensor(npz_dict["labels"].astype(np.int64))

    centroids_norm = centroids_quant / (2 ** meta["quantization"] - 1)
    centroids_norm = torch.tensor(centroids_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    centroids = centroids_norm * (maxs - mins) + mins

    params = centroids[labels]
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))

    return params


def _decompress_npz(
    compress_dir: str,
    param_name: str,
    meta: Dict[str, Any],
    resize: bool = False,
    **kwargs,
):
    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return params
    npz_dict = np.load(os.path.join(compress_dir, f"{param_name}.npz"))
    array = torch.tensor(npz_dict["arr"])
    if resize:
        array = array.reshape(meta["shape"])
    array = array.to(dtype=getattr(torch, meta["dtype"]))
    return array


def _inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
