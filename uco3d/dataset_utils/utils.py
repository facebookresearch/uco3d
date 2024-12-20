# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import time
import warnings
from dataclasses import dataclass, field

from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import torch

from .data_types import Cameras

logger = logging.getLogger(__name__)

UCO3D_DATASET_ROOT_ENV_VAR = "UCO3D_DATASET_ROOT"


def get_dataset_root(assert_exists: bool = False) -> str:
    """
    Returns the root directory of the UCO3D dataset.
    If the environment variable stored in UCO3D_DATASET_ROOT_ENV_VAR is set,
    it will be used. Otherwise, None is returned.

    Args:
        assert_exists: If True, the function will raise an error if the
            dataset root does not exist.
    """
    dataset_root = os.getenv(UCO3D_DATASET_ROOT_ENV_VAR, None)
    if assert_exists:
        if dataset_root is None:
            raise ValueError(
                f"Environment variable {UCO3D_DATASET_ROOT_ENV_VAR} is not set."
            )
        if not os.path.exists(dataset_root):
            raise ValueError(
                f"Environment variable {UCO3D_DATASET_ROOT_ENV_VAR} points"
                f" to a non-existing path {dataset_root}."
            )
    return dataset_root


def get_bbox_from_mask(
    mask: np.ndarray, thr: float, decrease_quant: float = 0.05
) -> Tuple[int, int, int, int]:
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    if thr <= 0.0:
        warnings.warn(
            f"Empty masks_for_bbox (thr={thr}) => using full image.", stacklevel=1
        )

    x0, x1 = get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0


def crop_around_box(
    tensor: torch.Tensor, bbox: torch.Tensor, impath: str = ""
) -> torch.Tensor:
    # bbox is xyxy, where the upper bound is corrected with +1
    bbox = clamp_box_to_image_bounds_and_round(
        bbox,
        image_size_hw=tuple(tensor.shape[-2:]),
    )
    tensor = tensor[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]
    assert all(c > 0 for c in tensor.shape), f"squashed image {impath}"
    return tensor


def clamp_box_to_image_bounds_and_round(
    bbox_xyxy: torch.Tensor,
    image_size_hw: Tuple[int, int],
) -> torch.LongTensor:
    bbox_xyxy = bbox_xyxy.clone()
    bbox_xyxy[[0, 2]] = torch.clamp(bbox_xyxy[[0, 2]], 0, image_size_hw[-1])
    bbox_xyxy[[1, 3]] = torch.clamp(bbox_xyxy[[1, 3]], 0, image_size_hw[-2])
    if not isinstance(bbox_xyxy, torch.LongTensor):
        bbox_xyxy = bbox_xyxy.round().long()
    return bbox_xyxy  # pyre-ignore [7]


T = TypeVar("T", bound=torch.Tensor)


def bbox_xyxy_to_xywh(xyxy: T) -> T:
    wh = xyxy[2:] - xyxy[:2]
    xywh = torch.cat([xyxy[:2], wh])
    return xywh  # pyre-ignore


def get_clamp_bbox(
    bbox: torch.Tensor,
    box_crop_context: float = 0.0,
    image_path: str = "",
) -> torch.Tensor:
    # box_crop_context: rate of expansion for bbox
    # returns possibly expanded bbox xyxy as float

    bbox = bbox.clone()  # do not edit bbox in place

    # increase box size
    if box_crop_context > 0.0:
        c = box_crop_context
        bbox = bbox.float()
        bbox[0] -= bbox[2] * c / 2
        bbox[1] -= bbox[3] * c / 2
        bbox[2] += bbox[2] * c
        bbox[3] += bbox[3] * c

    if (bbox[2:] <= 1.0).any():
        raise ValueError(
            f"squashed image {image_path}!! The bounding box contains no pixels."
        )

    bbox[2:] = torch.clamp(bbox[2:], 2)  # set min height, width to 2 along both axes
    bbox_xyxy = bbox_xywh_to_xyxy(bbox, clamp_size=2)

    return bbox_xyxy


def rescale_bbox(
    bbox: torch.Tensor,
    orig_res: Union[Tuple[int, int], torch.LongTensor],
    new_res: Union[Tuple[int, int], torch.LongTensor],
) -> torch.Tensor:
    assert bbox is not None
    assert np.prod(orig_res) > 1e-8
    # average ratio of dimensions
    # pyre-ignore
    rel_size = (new_res[0] / orig_res[0] + new_res[1] / orig_res[1]) / 2.0
    return bbox * rel_size


def bbox_xywh_to_xyxy(
    xywh: torch.Tensor, clamp_size: Optional[int] = None
) -> torch.Tensor:
    xyxy = xywh.clone()
    if clamp_size is not None:
        xyxy[2:] = torch.clamp(xyxy[2:], clamp_size)
    xyxy[2:] += xyxy[:2]
    return xyxy


def get_1d_bounds(arr: np.ndarray) -> Tuple[int, int]:
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1] + 1


def resize_image(
    image: Union[np.ndarray, torch.Tensor],
    image_height: Optional[int],
    image_width: Optional[int],
    mode: str = "bilinear",
) -> Tuple[torch.Tensor, float, torch.Tensor]:

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if image_height is None or image_width is None:
        # skip the resizing
        return image, 1.0, torch.ones_like(image[:1])
    # takes numpy array or tensor, returns pytorch tensor
    minscale = min(
        image_height / image.shape[-2],
        image_width / image.shape[-1],
    )
    imre = torch.nn.functional.interpolate(
        image[None],
        scale_factor=minscale,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        recompute_scale_factor=True,
    )[0]
    imre_ = torch.zeros(image.shape[0], image_height, image_width)
    imre_[:, : imre.shape[1], : imre.shape[2]] = imre
    mask = torch.zeros(1, image_height, image_width)
    mask[:, : imre.shape[1], : imre.shape[2]] = 1.0
    return imre_, minscale, mask


def safe_as_tensor(data, dtype):
    return torch.tensor(data, dtype=dtype) if data is not None else None


def _convert_ndc_to_pixels(
    focal_length: torch.Tensor,
    principal_point: torch.Tensor,
    image_size_wh: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    return focal_length_px, principal_point_px


def _convert_pixels_to_ndc(
    focal_length_px: torch.Tensor,
    principal_point_px: torch.Tensor,
    image_size_wh: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point = (half_image_size - principal_point_px) / rescale
    focal_length = focal_length_px / rescale
    return focal_length, principal_point


def adjust_camera_to_bbox_crop_(
    camera: Cameras,
    image_size_wh: torch.Tensor,
    clamp_bbox_xywh: torch.Tensor,
) -> None:
    if len(camera) != 1:
        raise ValueError("Adjusting currently works with singleton cameras camera only")

    focal_length_px, principal_point_px = _convert_ndc_to_pixels(
        camera.focal_length[0],
        camera.principal_point[0],
        image_size_wh,
    )
    principal_point_px_cropped = principal_point_px - clamp_bbox_xywh[:2]

    focal_length, principal_point_cropped = _convert_pixels_to_ndc(
        focal_length_px,
        principal_point_px_cropped,
        clamp_bbox_xywh[2:],
    )

    camera.focal_length = focal_length[None]
    camera.principal_point = principal_point_cropped[None]


def adjust_camera_to_image_scale_(
    camera: Cameras,
    original_size_wh: torch.Tensor,
    new_size_wh: torch.LongTensor,
) -> Cameras:
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(
        camera.focal_length[0],
        camera.principal_point[0],
        original_size_wh,
    )

    # now scale and convert from pixels to NDC
    image_size_wh_output = new_size_wh.float()
    scale = (image_size_wh_output / original_size_wh).min(dim=-1, keepdim=True).values
    focal_length_px_scaled = focal_length_px * scale
    principal_point_px_scaled = principal_point_px * scale

    focal_length_scaled, principal_point_scaled = _convert_pixels_to_ndc(
        focal_length_px_scaled,
        principal_point_px_scaled,
        image_size_wh_output,
    )
    camera.focal_length = focal_length_scaled[None]
    camera.principal_point = principal_point_scaled[None]  # pyre-ignore
    camera.image_size = new_size_wh.flip(dims=[0])[None]


def undistort_frame_data_opencv(frame_data):
    time_0 = time.time()
    R_opencv, T_opencv, mtx_th = opencv_cameras_projection_from_uco3d(
        frame_data.camera, frame_data.image_size_hw[None]
    )
    mtx = mtx_th[0].data.numpy()  # TODO: check they are same
    expected_size_wh = tuple(frame_data.image_size_hw.tolist()[::-1])
    distortion = np.asarray(frame_data.camera.colmap_distortion_coeffs)
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, distortion, expected_size_wh, 0.0, expected_size_wh
    )
    new_uco3d_cameras = uco3d_cameras_from_opencv_projection(
        R=R_opencv,
        tvec=T_opencv,
        camera_matrix=torch.tensor(new_mtx)[None],
        image_size=frame_data.image_size_hw[None],
    )
    time_1 = time.time()
    logger.debug(f"Time for camera undistortion = {time_1-time_0:.5f}")
    x, y, w, h = roi
    if (frame_data.image_rgb is None) and (frame_data.fg_probability is None):
        return (
            None,
            None,
            new_uco3d_cameras,
        )
    time_2 = time.time()
    if frame_data.image_rgb is not None:
        height, width = frame_data.image_rgb.shape[-2:]
    elif frame_data.fg_probability is not None:
        height, width = frame_data.fg_probability.shape[-2:]
    else:
        raise ValueError("Either image or mask must be present")
    size = (width, height)
    [undistort_map_x, undistort_map_y] = cv2.initUndistortRectifyMap(
        mtx, distortion, None, new_mtx, size, cv2.CV_16SC2
    )
    logger.debug(f"Time for getting the undistort map {time_2-time_1:.5f}")
    undistorted_mask = None
    undistorted_image = None
    if frame_data.fg_probability is not None:
        undistorted_mask = cv2.remap(
            np.asarray(frame_data.fg_probability).transpose(1, 2, 0),
            undistort_map_x,
            undistort_map_y,
            cv2.INTER_LINEAR,
        )
        undistorted_mask = torch.from_numpy(undistorted_mask).unsqueeze(0)
    if frame_data.image_rgb is not None:
        undistorted_image = cv2.remap(
            np.asarray(frame_data.image_rgb).transpose(1, 2, 0),
            undistort_map_x,
            undistort_map_y,
            cv2.INTER_LINEAR,
        )
        undistorted_image = torch.from_numpy(undistorted_image.transpose((2, 0, 1)))
    time_3 = time.time()
    logger.debug(f"Time for actually undistorting is {time_3-time_2:.5f}")
    return (
        undistorted_image,
        undistorted_mask,
        new_uco3d_cameras,
    )


def uco3d_cameras_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
) -> Cameras:
    """
    Converts the opencv camera parameters to the uco3d camera parameters.
    Args:
        R: Rotation matrix of shape (N, 3, 3).
        tvec: Translation vector of shape (N, 3).
        camera_matrix: Camera matrix of shape (N, 3, 3).
        image_size: Image size of shape (N, 2).
    Returns:
        Cameras: uco3d camera parameters.
    """
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    return Cameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=R.device,
    )


def opencv_cameras_projection_from_uco3d(
    cameras: Cameras,
    image_size: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts the uco3d camera parameters to the opencv camera parameters.
    Args:
        cameras: uco3d camera parameters.
        image_size: Image size of shape (N, 2).
    Returns:
        R: Rotation matrix of shape (N, 3, 3).
        T: Translation vector of shape (N, 3).
        camera_matrix: Intrinsic camera matrix of shape (N, 3, 3).
    """
    R_pytorch3d = cameras.R.clone()
    T_pytorch3d = cameras.T.clone()
    focal_pytorch3d = cameras.focal_length
    p0_pytorch3d = cameras.principal_point
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R, tvec, camera_matrix


K_type = TypeVar("K")
T_type = TypeVar("T")


@dataclass
class LruCacheWithCleanup(Generic[K_type, T_type]):
    """Requires Python 3.6+ since it assumes insertion-ordered dict."""

    create_fn: Callable[[K_type], T_type]
    cleanup_fn: Callable[[T_type], None] = field(default=lambda key: None)
    max_size: int | None = None
    _cache: dict[K_type, T_type] = field(init=False, default_factory=lambda: {})

    def __post_init__(self) -> None:
        self._n_hits = 0
        self._n_misses = 0

    def __getitem__(self, key: K_type) -> T_type:
        if key in self._cache:
            # logger.debug(f"Cache hit: {key}")
            value = self._cache.pop(key)
            self._cache[key] = value  # update the order
            self._n_hits += 1
            return value
        self._n_misses += 1

        # inserting a new element
        if self.max_size and len(self._cache) >= self.max_size:
            # need to clean up the oldest element
            oldest_key = next(iter(self._cache))
            oldest_value = self._cache.pop(oldest_key)
            # logger.debug(f"Releasing an object for {oldest_key}")
            self.cleanup_fn(oldest_value)

        assert self.max_size is None or len(self._cache) < self.max_size
        # logger.debug(f"Creating an object for {key}")
        value = self.create_fn(key)
        self._cache[key] = value
        return value

    def cleanup_all(self) -> None:
        for value in self._cache.values():
            self.cleanup_fn(value)

        self._cache = {}
