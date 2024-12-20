# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 Ending Hsiao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
from dataclasses import dataclass

import numpy as np
from omegaconf import OmegaConf

from scipy.spatial.transform import Rotation

from .sh_utils import SHRotator


def get_config(a):
    conf = OmegaConf.load(a.config)
    if a.input is not None:
        conf.input = a.input
    assert os.path.isfile(conf.input), f"File not exist: {conf.input}"

    if a.output is not None:
        conf.output = a.output

    return conf


@dataclass
class SplatData:
    xyz: np.ndarray
    features_dc: np.ndarray
    features_rest: np.ndarray
    opacity: np.ndarray
    scaling: np.ndarray
    rotation: np.ndarray
    active_sh_degree: int


def get_shs(data: SplatData):
    return np.concatenate([data.features_dc, data.features_rest], axis=-1)


def transform_xyz(T, R, S, xyz):
    return xyz @ (R @ S).T + T


def batch_compose_rs(R2, S2, r1, s1):
    w, x, y, z = r1.T  # (4, n)
    R1 = Rotation.from_quat(np.stack([x, y, z, w], axis=-1)).as_matrix()
    S1 = np.eye(3) * s1[..., np.newaxis]

    R2S2 = R2 @ S2
    R1S1 = np.einsum("bij,bjk->bik", R1, S1)
    RS = np.einsum("ij,bjk->bik", R2S2, R1S1)
    return RS


def batch_decompose_rs(RS):
    sx = np.linalg.norm(RS[..., 0], axis=-1)
    sy = np.linalg.norm(RS[..., 1], axis=-1)
    sz = np.linalg.norm(RS[..., 2], axis=-1)

    RS[..., 0] /= sx[..., np.newaxis]
    RS[..., 1] /= sy[..., np.newaxis]
    RS[..., 2] /= sz[..., np.newaxis]
    x, y, z, w = Rotation.from_matrix(RS).as_quat().T
    r = np.stack([w, x, y, z], axis=-1)
    s = np.stack([sx, sy, sz], axis=-1)
    return r, s


def batch_rotate_sh(R, shs_in, max_sh_degree=3):
    # shs_in: (n, 3, deg)
    # SH is in yzx order so here shift the order of rot mat
    # rot_fn = SHRotator(R, deg=max_sh_degree)  # original
    rot_fn = SHRotator(R.T, deg=max_sh_degree)  # bugfixed
    shs_out = np.stack(
        [
            rot_fn(shs_in[..., 0, :]),
            rot_fn(shs_in[..., 1, :]),
            rot_fn(shs_in[..., 2, :]),
        ],
        axis=-2,
    )
    return shs_out


def transform_data(conf, data: SplatData):

    position = np.asarray(conf.transform.position, dtype=np.float32)
    rotation = np.asarray(conf.transform.rotation, dtype=np.float32)
    scale = np.asarray(conf.transform.scale, dtype=np.float32)

    if conf.unity_transform:
        x, y, z = rotation
        q = Rotation.from_euler("zxy", [z, x, y], degrees=True).as_quat()
        q = Rotation.from_quat([-q[0], -q[2], -q[1], q[3]])
        q_shift_r = np.pi / 4
        q_shift_r = Rotation.from_quat([np.sin(q_shift_r), 0.0, 0.0, np.cos(q_shift_r)])
        q_shift_l = Rotation.from_euler("xyz", [90, 180, 0], degrees=True)
        q = q_shift_l * q * q_shift_r
        rotation = q.as_euler("xyz", degrees=True)
        position[0] = -position[0]
        position[1] = -position[1]
        position[2] = -position[2]

    # object to world
    S = np.eye(3) * scale
    R = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
    T = np.array(position, dtype=np.float32)
    data.xyz = transform_xyz(T, R, S, data.xyz)
    r, s = data.rotation, np.exp(data.scaling)
    RS = batch_compose_rs(R, S, r, s)
    r, s = batch_decompose_rs(RS)
    data.rotation, data.scaling = r, np.log(s)
    if conf.rotate_sh:
        shs_out = batch_rotate_sh(R, get_shs(data), conf.max_sh_degree)
        data.features_dc = shs_out[..., :, :1]
        data.features_rest = shs_out[..., :, 1:]
    return data
