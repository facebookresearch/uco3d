# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import math
from typing import Tuple

import torch

from .data_types import Cameras, GaussianSplats
from .utils import opencv_cameras_projection_from_uco3d

GSPLAT_AVAILABLE = True
try:
    from gsplat import rasterization
except ImportError:
    GSPLAT_AVAILABLE = False


def render_splats(
    cameras: Cameras,
    splats: GaussianSplats,
    render_size: Tuple[int, int],
    device: str = "cuda:0",
    near_plane: float = 0.01,
    **kwargs,
):
    """
    Render a set of Gaussian splats using the gsplat library.

    Args:
        cameras: Rendering cameras.
        splats: A GaussianSplats object containing the splats to render.
        render_size: A tuple of integers (width, height) specifying the size of the rendered image.
        device: The device to use for rendering. Default is "cuda:0".
        near_plane: The near plane distance for rendering. Default is 1.0.

    Returns:
        render_colors: A tensor of shape [C, H, W, 3] containing the rendered colors.
        render_alphas: A tensor of shape [C, H, W] containing the rendered alphas.
        info: A dictionary containing additional information about the rendering.
    """
    n = cameras.R.shape[0]
    image_size_tensor = torch.tensor(render_size)[None].repeat(n, 1)
    Rcv, tvec, camera_matrix = opencv_cameras_projection_from_uco3d(
        cameras,
        image_size=image_size_tensor,
    )
    viewmats = torch.eye(4)[None].repeat(n, 1, 1)
    viewmats[:, :3, :4] = torch.cat(
        [Rcv, tvec[..., None]],
        dim=-1,
    )
    return render_splats_opencv(
        viewmats=viewmats,
        camera_matrix=camera_matrix,
        splats=splats,
        render_size=render_size,
        device=device,
        near_plane=near_plane,
        camera_matrix_in_ndc=False,
        **kwargs,
    )


def render_splats_opencv(
    viewmats: torch.Tensor,
    camera_matrix: torch.Tensor,
    splats: GaussianSplats,
    render_size: Tuple[int, int],
    device: str = "cuda:0",
    near_plane: float = 0.01,
    camera_matrix_in_ndc: bool = False,
    **kwargs,
):
    """
    Render a set of Gaussian splats using the gsplat library.

    Args:
        viewmats: A tensor of shape [C, 4, 4] containing the view matrices for each camera.
        camera_matrix: A tensor of shape [C, 3, 3] containing the camera matrices for each camera.

        splats: A GaussianSplats object containing the splats to render.
        render_size: A tuple of integers (width, height) specifying the size of the rendered image.
        device: The device to use for rendering. Default is "cuda:0".
        near_plane: The near plane distance for rendering. Default is 1.0.
        camera_matrix_in_ndc: A boolean indicating whether the camera matrix is in ndc coordinates.
            Default is False.
            If True, The intrinsic matrix is in ndc coordinates,
            i.e. the image is in the range [-1, 1].

    Returns:
        render_colors: A tensor of shape [C, H, W, 3] containing the rendered colors.
        render_alphas: A tensor of shape [C, H, W] containing the rendered alphas.
        info: A dictionary containing additional information about the rendering.
    """

    if not GSPLAT_AVAILABLE:
        raise RuntimeError(
            "Please install gsplat by running"
            + " `pip install git+https://github.com/nerfstudio-project/gsplat.git`"
        )

    height, width = render_size
    n_cams = viewmats.shape[0]
    device = torch.device(device)

    # move splats to the device
    splats = dataclasses.asdict(splats)
    for k, v in splats.items():
        if torch.is_tensor(v):
            splats[k] = v.to(device, dtype=torch.float32)

    # parse splats
    N = splats["means"].shape[0]
    means = splats["means"]  # [N, 3]
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"].flatten())  # [N,]

    colors = torch.cat(
        [
            splats["sh0"][:, None],
            (
                splats["shN"]
                if splats["shN"] is not None
                else torch.zeros([N, 1, 3], device=device)
            ),
        ],
        1,
    )  # [N, K, 3]

    sh_degree = math.log2(colors.shape[1]) - 1
    assert sh_degree.is_integer()
    sh_degree = int(sh_degree)

    if camera_matrix_in_ndc:
        # convert ndc camera matrix to pixel camera matrix
        camera_matrix_pix = torch.tensor(
            [
                [0.5 * width, 0, 0.5 * width],
                [0, 0.5 * height, 0.5 * height],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )[None] @ camera_matrix.to(device, dtype=torch.float32)
    else:
        camera_matrix_pix = camera_matrix.to(device, dtype=torch.float32)

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats.to(device=device),  # [C, 4, 4]
        Ks=camera_matrix_pix,  # [C, 3, 3]
        width=width,
        height=height,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        sh_degree=sh_degree,
        near_plane=near_plane,
        backgrounds=torch.ones(n_cams, 3, dtype=torch.float32, device=device),
        **kwargs,
    )

    return render_colors, render_alphas, info
