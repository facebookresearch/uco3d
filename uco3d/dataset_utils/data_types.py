# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import warnings
from dataclasses import dataclass, fields
from typing import List, Optional, Sequence, Tuple, Union

import torch

try:
    from pytorch3d.renderer import PerspectiveCameras

    _NO_PYTORCH3D = False
except ImportError:
    _NO_PYTORCH3D = True

logger = logging.getLogger(__name__)


_FocalLengthType = Union[
    float, Sequence[Tuple[float]], Sequence[Tuple[float, float]], torch.Tensor
]
_PrincipalPointType = Union[Tuple[float, float], torch.Tensor]


@dataclass
class Cameras:
    """
    A class to represent a batch of cameras.
    Follows the same conventions as pytorch3d's PerspectiveCameras class, with additional
    fields for colmap distortion coefficients and image size.

    To convert to the corresponding pytorch3d class, use the to_pytorch3d_cameras
    method.
    """

    R: torch.Tensor = torch.eye(3)[None]
    T: torch.Tensor = torch.zeros(1, 3)
    focal_length: _FocalLengthType = torch.ones(1, 2)
    principal_point: _PrincipalPointType = torch.zeros(1, 2)
    colmap_distortion_coeffs: torch.Tensor = torch.zeros(1, 12)
    device: torch.device = torch.device("cpu")
    in_ndc: bool = True
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None

    def __post_init__(self):
        for field in [
            "R",
            "T",
            "focal_length",
            "principal_point",
            "colmap_distortion_coeffs",
            "image_size",
        ]:
            if torch.is_tensor(getattr(self, field)):
                assert getattr(self, field).device == self.device

    def __len__(self):
        return self.R.shape[0]

    def to(self, *args, **kwargs):
        def _to_tensor(x):
            return x.to(*args, **kwargs) if torch.is_tensor(x) else x

        R = self.R.to(*args, **kwargs)
        return Cameras(
            R=R,
            T=self.T.to(*args, **kwargs),
            focal_length=_to_tensor(self.focal_length),
            principal_point=_to_tensor(self.principal_point),
            colmap_distortion_coeffs=_to_tensor(self.colmap_distortion_coeffs),
            device=R.device,
            in_ndc=self.in_ndc,
            image_size=_to_tensor(self.image_size),
        )

    def transform_points_to_camera_coords(
        self,
        world_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert points from world coordinates to camera coordinates.

        Args:
            world_points: Tensor of shape (B, N, 3) giving
                the world coordinates of the points.

        Returns:
            Tensor of shape (B, N, 3) giving the
                camera coordinates of the points.
        """
        assert world_points.ndim == 3
        assert world_points.shape[-1] == 3
        assert self.R.ndim == 3
        assert self.T.ndim == 2
        assert self.R.shape[1] == 3
        assert self.R.shape[2] == 3
        assert self.T.shape[1] == 3
        return world_points @ self.R + self.T[:, None]

    def transform_points_to_world_coords(
        self,
        camera_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert points from camera coordinates to world coordinates.

        Args:
            camera_points: Tensor of shape (B, N, 3) giving
                the world coordinates of the points.

        Returns:
            Tensor of shape (B, N, 3) giving the
                world coordinates of the points.
        """
        assert camera_points.ndim == 3
        assert camera_points.shape[-1] == 3
        assert self.R.ndim == 3
        assert self.T.ndim == 2
        assert self.R.shape[1] == 3
        assert self.R.shape[2] == 3
        assert self.T.shape[1] == 3
        return (camera_points - self.T[:, None]) @ self.R.permute(0, 2, 1)

    def transform_points(
        self,
        world_points: torch.Tensor,
        eps: float | None = 1e-5,
    ) -> torch.Tensor:
        """
        Transform points from world coordinates
        to PyTorch3D NDC image coordinates.

        Args:
            world_points: Tensor of shape (B, N, 3) giving the
                world coordinates of the points.
            eps: A small value to avoid division by zero.

        Returns:
            Tensor of shape (B, N, 2) giving the ndc image coordinates
                of the points.
        """
        camera_points = self.transform_points_to_camera_coords(world_points)
        depth = camera_points[..., 2:3]
        if eps is not None:
            depth = torch.clamp(depth, min=eps)
        projected_points = camera_points[..., :2] / depth
        focal_length_tensor = self._handle_focal_length(like=projected_points)[:, None]
        principal_point_tensor = self._handle_principal_point(like=projected_points)[
            :, None
        ]
        return projected_points * focal_length_tensor + principal_point_tensor

    def transform_points_screen(
        self,
        world_points: torch.Tensor,
        eps: float | None = 1e-5,
    ):
        """
        Transform points from world coordinates to screen coordinates.

        Args:
            world_points: Tensor of shape (B, N, 3) giving the
                world coordinates of the points.
            eps: A small value to avoid division by zero.

        Returns:
            Tensor of shape (B, N, 2) giving the screen coordinates
                of the points.
        """
        assert self.in_ndc, "Camera must be in NDC space"
        # We require the image size, which is necessary for the transform
        if self.image_size is None:
            raise ValueError(
                "For NDC to screen conversion, image_size=(height, width)"
                " needs to be specified."
            )
        ndc_points = self.transform_points(world_points, eps=eps)
        if not torch.is_tensor(self.image_size):
            image_size = torch.tensor(self.image_size, device=self.device)
        image_size = self.image_size.view(-1, 2)  # of shape (1 or B)x2
        height, width = image_size.unbind(1)
        assert height.shape[0] == len(self)
        assert width.shape[0] == len(self)
        # For non square images, we scale the points such that smallest side
        # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
        # This convention is consistent with the PyTorch3D renderer
        scale = image_size.min(dim=1).values / 2.0
        screen_points = ndc_points * scale[:, None, None]
        screen_points[..., 0] = screen_points[..., 0] - width[:, None] / 2.0
        screen_points[..., 1] = screen_points[..., 1] - height[:, None] / 2.0
        screen_points *= -1.0
        return screen_points

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = False,
    ):
        """
        Unproject points from image coordinates to world coordinates.

        Args:
            xy_depth: Tensor of shape (B, N, 3) giving the
                image NDC coordinates of the points with depth concatenated
                as the last dimension.
            world_coordinates: If True, return unprojected points in
                world coordinates. Otherwise, return camera coordinates.

        Returns:
            Tensor of shape (B, N, 3) giving the
                world or camera coordinates of the points.
        """
        assert xy_depth.ndim == 3
        assert xy_depth.shape[-1] == 3
        assert len(xy_depth) == len(self)
        focal_length_tensor = self._handle_focal_length(like=xy_depth)[:, None]
        principal_point_tensor = self._handle_principal_point(like=xy_depth)[:, None]
        xy, depth = xy_depth[..., :2], xy_depth[..., 2:]
        xy_camera = (xy - principal_point_tensor) / focal_length_tensor
        camera_points = torch.cat([depth * xy_camera, depth], dim=-1)
        if world_coordinates:
            return self.transform_points_to_world_coords(camera_points)
        return camera_points

    def unproject_screen_points(
        self,
        xy_screen_depth: torch.Tensor,
        world_coordinates: bool = False,
    ):
        """
        Unproject points from screen coordinates to world coordinates.

        Args:
            xy_screen_depth: Tensor of shape (B, N, 3) giving the
                image screen coordinates of the points with depth concatenated
                as the last dimension.
            world_coordinates: If True, return unprojected points in
                world coordinates. Otherwise, return camera coordinates.

        Returns:
            Tensor of shape (B, N, 3) giving the
                world or camera coordinates of the points.
        """
        assert xy_screen_depth.ndim == 3
        assert xy_screen_depth.shape[-1] == 3
        assert len(xy_screen_depth) == len(self)
        if not torch.is_tensor(self.image_size):
            image_size = torch.tensor(self.image_size, device=self.device)
        image_size = self.image_size.view(-1, 2)  # of shape (1 or B)x2
        height, width = image_size.unbind(1)
        assert height.shape[0] == len(self)
        assert width.shape[0] == len(self)
        scale = image_size.min(dim=1).values / 2.0
        xy_ndc = xy_screen_depth[..., :2].clone()
        xy_ndc *= -1.0
        xy_ndc[..., 0] = xy_ndc[..., 0] + width[:, None] / 2.0
        xy_ndc[..., 1] = xy_ndc[..., 1] + height[:, None] / 2.0
        xy_ndc /= scale[:, None, None]
        return self.unproject_points(
            torch.cat([xy_ndc, xy_screen_depth[..., 2:3]], dim=-1),
            world_coordinates=world_coordinates,
        )

    def to_pytorch3d_cameras(self):
        if _NO_PYTORCH3D:
            raise ImportError(
                "PyTorch3D is not installed to convert to PyTorch3D cameras."
            )
        if (self.colmap_distortion_coeffs != 0.0).any():
            warnings.warn(
                "Converting Cameras with non-trivial undistortion coefficients"
                " to PyTorch3D PerspectiveCameras."
                " However, PyTorch3D does not support distortion coefficients."
            )
        return PerspectiveCameras(
            R=self.R,
            T=self.T,
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            in_ndc=self.in_ndc,
            device=self.device,
            image_size=self.image_size,
        )

    def _handle_focal_length(self, like: torch.Tensor) -> torch.Tensor:
        return (
            self.focal_length.to(like.device)
            if torch.is_tensor(self.focal_length)
            else (
                like.new_tensor(
                    self.focal_length
                    if isinstance(self.focal_length, tuple)
                    else (self.focal_length, self.focal_length)
                )
            )
        )

    def _handle_principal_point(self, like: torch.Tensor) -> torch.Tensor:
        return (
            self.principal_point.to(like.device)
            if torch.is_tensor(self.principal_point)
            else like.new_tensor(self.principal_point)
        )


# TODO: support gsplats
@dataclass
class PointCloud:
    """
    A class to represent a point cloud with xyz and rgb attributes.
    The rgb values are range between [0, 1].
    """

    xyz: torch.Tensor
    rgb: torch.Tensor

    def to(self, *args, **kwargs):
        return PointCloud(
            xyz=self.xyz.to(*args, **kwargs),
            rgb=self.rgb.to(*args, **kwargs),
        )


@dataclass
class GaussianSplats:
    """
    A class to represent Gaussian splats for a single scene.

    Args:
        means: Tensor of shape (N, 3) giving the means of the Gaussians.
        sh0: Tensor of shape (N, 3) giving the DC spherical harmonics coefficients.
        shN: Optional Tensor of shape (N, L, 3) giving the rest of the spherical harmonics coefficients.
        opacities: Tensor of shape (N, 1) giving the opacities of the Gaussians.
        scales: Tensor of shape (N, 3) giving the scales of the Gaussians.
        quats: Tensor of shape (N, 4) giving the quaternions of the Gaussians.
        fg_mask: Optional Tensor of shape (N, 1) giving the foreground mask.
    """

    means: torch.Tensor
    sh0: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    shN: Optional[torch.Tensor] = None
    fg_mask: Optional[torch.Tensor] = None

    def __len__(self):
        return self.means.shape[0]

    @classmethod
    def empty(self, device: torch.device = torch.device("cpu")):
        return GaussianSplats(
            means=torch.empty(0, 3, device=device),
            sh0=torch.empty(0, 3, device=device),
            opacities=torch.empty(0, 1, device=device),
            scales=torch.empty(0, 3, device=device),
            quats=torch.empty(0, 4, device=device),
        )


def join_uco3d_cameras_as_batch(cameras_list: Sequence[Cameras]) -> Cameras:
    """
    Create a batched cameras object by concatenating a list of input
    cameras objects. All the tensor attributes will be joined along
    the batch dimension.

    Args:
        cameras_list: List of camera classes all of the same type and
            on the same device. Each represents one or more cameras.
    Returns:
        cameras: single batched cameras object of the same
            type as all the objects in the input list.
    """
    # Get the type and fields to join from the first camera in the batch
    c0 = cameras_list[0]
    field_list = fields(c0)

    if not all(isinstance(c, Cameras) for c in cameras_list):
        raise ValueError("cameras in cameras_list must inherit from CamerasBase")

    if not all(type(c) is type(c0) for c in cameras_list[1:]):
        raise ValueError("All cameras must be of the same type")

    if not all(c.device == c0.device for c in cameras_list[1:]):
        raise ValueError("All cameras in the batch must be on the same device")

    # Concat the fields to make a batched tensor
    kwargs = {}
    kwargs["device"] = c0.device

    for field in field_list:
        if field.name == "device":
            continue
        field_not_none = [(getattr(c, field.name) is not None) for c in cameras_list]
        if not any(field_not_none):
            continue
        if not all(field_not_none):
            raise ValueError(f"Attribute {field.name} is inconsistently present")

        attrs_list = [getattr(c, field.name) for c in cameras_list]
        if field.name == "in_ndc":
            # Only needs to be set once
            if not all(a == attrs_list[0] for a in attrs_list):
                raise ValueError(
                    f"Attribute {field.name} is not constant across inputs"
                )

            kwargs[field.name] = attrs_list[0]
        elif isinstance(attrs_list[0], torch.Tensor):
            # In the init, all inputs will be converted to
            # batched tensors before set as attributes
            # Join as a tensor along the batch dimension
            kwargs[field.name] = torch.cat(attrs_list, dim=0)
        else:
            raise ValueError(f"Field {field.name} type is not supported for batching")

    return c0.__class__(**kwargs)
