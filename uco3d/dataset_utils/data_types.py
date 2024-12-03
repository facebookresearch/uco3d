# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, fields
from typing import Optional, Tuple, Union, List, Sequence
import torch
import logging
import copy

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

    R: torch.Tensor = torch.eye(3)
    T: torch.Tensor = torch.zeros(3)
    focal_length: _FocalLengthType = 1.0
    principal_point: _PrincipalPointType = (0.0, 0.0)
    colmap_distortion_coeffs: torch.Tensor = torch.zeros(12)
    device: str = "cpu"
    in_ndc: bool = True
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None

    def to(self, *args, **kwargs):
        def _to_tensor(x):
            return x.to(*args, **kwargs) if torch.is_tensor(x) else x

        return Cameras(
            R=self.R.to(*args, **kwargs),
            T=self.T.to(*args, **kwargs),
            focal_length=_to_tensor(self.focal_length),
            principal_point=_to_tensor(self.principal_point),
            colmap_distortion_coeffs=_to_tensor(self.colmap_distortion_coeffs),
            device=self.device,
            in_ndc=self.in_ndc,
            image_size=_to_tensor(self.image_size),
        )

    def project_points(self, world_points: torch.Tensor, eps=1e-5) -> torch.Tensor:
        # Apply the projection formula: XR + T
        world_points = world_points @ self.R + self.T
        # Divide by the Z component
        return world_points[..., :2] / world_points[..., 2:3].clip(eps)

    def transform_points(self, world_points: torch.Tensor) -> torch.Tensor:
        projected_points = self.project_points(world_points)

        focal_length_tensor = projected_points.new_tensor(
            self.focal_length
            if isinstance(self.focal_length, tuple)
            else (self.focal_length, self.focal_length)
        )
        principal_point_tensor = projected_points.new_tensor(self.principal_point)
        return projected_points * focal_length_tensor + principal_point_tensor

    def to_pytorch3d_cameras(self):
        if _NO_PYTORCH3D:
            raise ImportError(
                "pytorch3d is not installed to convert to PyTorch3D cameras"
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
    shN: Optional[torch.Tensor]
    opacities: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    fg_mask: Optional[torch.Tensor] = None


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
