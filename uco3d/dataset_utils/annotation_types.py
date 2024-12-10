# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional, Tuple, TypeVar


_X = TypeVar("_X")

TF3 = Tuple[float, float, float]


@dataclass
class ImageAnnotation:
    # path to jpg file, relative w.r.t. dataset_root
    path: str
    # H x W
    size: Tuple[int, int]  # TODO: rename size_hw?


@dataclass
class DepthAnnotation:
    # path to png file, relative w.r.t. dataset_root, storing `depth / scale_adjustment`
    path: str
    # a factor to convert png values to actual depth: `depth = png * scale_adjustment`
    scale_adjustment: float
    # path to png file, relative w.r.t. dataset_root, storing binary `depth` mask
    mask_path: Optional[str]


@dataclass
class MaskAnnotation:
    # path to png file storing (Prob(fg | pixel) * 255)
    path: str
    # (soft) number of pixels in the mask; sum(Prob(fg | pixel))
    mass: Optional[float] = None
    # tight bounding box around the foreground mask
    bounding_box_xywh: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ViewpointAnnotation:
    # In right-multiply (PyTorch3D) format. X_cam = X_world @ R + T
    R: Tuple[TF3, TF3, TF3]
    T: TF3

    focal_length: Tuple[float, float]
    principal_point: Tuple[float, float]
    colmap_distortion_coeffs: Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    intrinsics_format: str = "ndc_norm_image_bounds"


@dataclass
class PointCloudAnnotation:
    # path to ply file with points only, relative w.r.t. dataset_root
    path: str
    n_points: Optional[int] = None


@dataclass
class GaussianSplatsAnnotation:
    # path to folder with stored compressed gaussian splats
    dir: str
    n_gaussians: Optional[int] = None


@dataclass
class VideoAnnotation:
    # path to the original video file, relative w.r.t. dataset_root
    path: str
    # length of the video in seconds
    length: Optional[float] = None


@dataclass
class ReconstructionQualityAnnotation:
    # camera-quality SVM classifier score
    viewpoint: Optional[float] = None
    # gaussian splats SVM classifier score
    gaussian_splats: Optional[float] = None
    # gaussian splats PSNR evaluated on held-out images
    gaussian_splats_psnr: Optional[float] = None
    # gaussian splats SSIM evaluated on held-out images
    gaussian_splats_ssim: Optional[float] = None
    # gaussian splats LPIPS evaluated on held-out images
    gaussian_splats_lpips: Optional[float] = None
    # number of scene cameras reconstructed by SfM
    #   ... (does not need to be the same as the number of cameras in the annotations)
    sfm_n_registered_cameras: Optional[int] = None
    # the mean length of the tracks in the SfM reconstruction
    sfm_mean_track_length: Optional[float] = None
    # the mean reprojection error of the final SfM solution
    sfm_bundle_adjustment_final_cost: Optional[float] = None


@dataclass
class AlignmentAnnotation:
    # In right-multiply (PyTorch3D) format. X_transformed = scale * (X @ R + T)
    R: tuple[TF3, TF3, TF3] | None = None
    T: TF3 | None = None
    scale: float = 1.0


@dataclass
class CaptionAnnotation:
    text: str | None = None
    clip_score: float | None = None
