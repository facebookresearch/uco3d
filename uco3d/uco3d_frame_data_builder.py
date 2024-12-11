# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import functools
import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import cv2
import numpy as np
import torch

from .dataset_utils.data_types import Cameras, GaussianSplats, PointCloud
from .dataset_utils.frame_data import UCO3DFrameData

from .dataset_utils.gauss3d_utils import (
    load_compressed_gaussians,
    transform_gaussian_splats,
    truncate_bg_gaussians,
)

from .dataset_utils.io_utils import (
    load_depth,
    load_depth_mask,
    load_image,
    load_mask,
    load_point_cloud,
    transpose_normalize_image,
    load_h5_depth,
)

from .dataset_utils.orm_types import UCO3DFrameAnnotation, UCO3DSequenceAnnotation

from .dataset_utils.utils import (
    get_bbox_from_mask,
    get_dataset_root,
    LruCacheWithCleanup,
    safe_as_tensor,
    UCO3D_DATASET_ROOT_ENV_VAR,
    undistort_frame_data_opencv,
)


logger = logging.getLogger(__name__)


POINT_CLOUD_TYPE = Literal["sparse", "dense", "dense_segmented"]


@dataclass
class UCO3DFrameDataBuilder:
    """
    A class to build a FrameData object, load and process the binary data (crop and
    resize). Beware that modifications of frame data are done in-place.

    Args:
        dataset_root: The root folder of the dataset; all paths in frame / sequence
            annotations are defined w.r.t. this root. Has to be set if any of the
            load_* flags below is true.
        load_images: Enable loading the frame RGB data.
        load_depths: Enable loading the frame depth maps.
        load_depth_masks: Enable loading the frame depth map masks denoting the
            depth values used for evaluation (the points consistent across views).
        load_masks: Enable loading frame foreground masks.
        load_point_clouds: Enable loading sequence-level point clouds.
        max_points: Cap on the number of loaded points in the point cloud;
            if reached, they are randomly sampled without replacement.
        load_gaussian_splats: Enable loading sequence-level 3D gaussian splats.
        gaussian_splats_truncate_background: Whether to truncate the background
            means of the gaussian splats.
        gaussian_splats_load_higher_order_harms: Whether to load higher order
            harmonics for the gaussian splats.
        mask_images: Whether to mask the images with the loaded foreground masks;
            0 value is used for background.
        mask_depths: Whether to mask the depth maps with the loaded foreground
            masks; 0 value is used for background.
        image_height: The height of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        image_width: The width of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        box_crop: Enable cropping of the image around the bounding box inferred
            from the foreground region of the loaded segmentation mask; masks
            and depth maps are cropped accordingly; cameras are corrected.
        box_crop_mask_thr: The threshold used to separate pixels into foreground
            and background based on the foreground_probability mask; if no value
            is greater than this threshold, the loader lowers it and repeats.
        box_crop_context: The amount of additional padding added to each
            dimension of the cropping bounding box, relative to box size.
        apply_alignment: Whether to apply the alignment transform mapping the
            cameras, point clouds, and gaussians to the canonical object-centric
            coordinate frame.
        use_cache: Whether to cache the video capture objects, point clouds and gaussians.
        video_capture_cache_size: Size of the cache for video capture objects.
        point_cloud_cache_size: Size of the cache for point clouds.
        gaussian_splat_cache_size: Size of the cache for gaussian splats.
        path_manager: Optionally a PathManager for interpreting paths in a special way.
    """

    dataset_root: Optional[str] = get_dataset_root()
    load_images: bool = True
    load_depths: bool = True
    load_depth_masks: bool = True
    load_masks: bool = True
    load_point_clouds: bool = False
    load_segmented_point_clouds: bool = False
    load_sparse_point_clouds: bool = False
    max_points: int = 0
    load_gaussian_splats: bool = False
    gaussian_splats_truncate_background: bool = False
    gaussian_splats_load_higher_order_harms: bool = True
    undistort_loaded_blobs: bool = True
    load_frames_from_videos: bool = True
    image_height: Optional[int] = 800
    image_width: Optional[int] = 800
    box_crop: bool = True
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 0.3
    apply_alignment: bool = True
    mask_images: bool = False
    mask_depths: bool = False
    video_capture_cache_size: int = 16
    depth_video_fps: int = 20
    point_cloud_cache_size: int = 32
    gaussian_splat_cache_size: int = 32
    use_cache: bool = True
    path_manager: Any = None

    def __post_init__(self) -> None:
        if (self.image_width is None) != (self.image_height is None):
            raise ValueError(
                "Both image_height and image_width have to be set or unset."
            )
        if self.dataset_root is None and any(
            [
                self.load_images,
                self.load_depths,
                self.load_masks,
                self.load_point_clouds,
                self.load_segmented_point_clouds,
                self.load_sparse_point_clouds,
                self.load_gaussian_splats,
            ]
        ):
            raise ValueError(
                "dataset_root has to be set if any of the load_* flags is true."
                f"Either set the {UCO3D_DATASET_ROOT_ENV_VAR} env variable or"
                " set the UCO3DFrameDataBuilder.dataset_root to a correct path."
            )

        if self.use_cache:
            self._video_capture_cache = LruCacheWithCleanup[str, cv2.VideoCapture](
                create_fn=lambda path: cv2.VideoCapture(path),
                cleanup_fn=lambda capture: capture.release(),
                max_size=self.video_capture_cache_size,
            )
            self._point_cloud_cache = LruCacheWithCleanup[str, PointCloud](
                create_fn=load_point_cloud,
                cleanup_fn=lambda _: None,
                max_size=self.point_cloud_cache_size,
            )
            self._gaussian_splat_cache = LruCacheWithCleanup[str, GaussianSplats](
                create_fn=functools.partial(
                    load_compressed_gaussians,
                    load_higher_order_harms=self.gaussian_splats_load_higher_order_harms,
                ),
                cleanup_fn=lambda _: None,
                max_size=self.point_cloud_cache_size,
            )

    def build(
        self,
        frame_annotation: UCO3DFrameAnnotation,
        sequence_annotation: UCO3DSequenceAnnotation,
        *,
        load_blobs: bool = True,
    ) -> UCO3DFrameData:

        super_category = sequence_annotation.super_category
        category = sequence_annotation.category
        frame_data = UCO3DFrameData(
            frame_number=safe_as_tensor(frame_annotation.frame_number, torch.long),
            frame_timestamp=safe_as_tensor(
                frame_annotation.frame_timestamp, torch.float
            ),
            sequence_name=frame_annotation.sequence_name,
            sequence_category=category,
            sequence_super_category=super_category,
            camera_quality_score=safe_as_tensor(
                sequence_annotation.reconstruction_quality.viewpoint,
                torch.float,
            ),
            gaussian_splats_quality_score=safe_as_tensor(
                sequence_annotation.reconstruction_quality.gaussian_splats,
                torch.float,
            ),
            sequence_caption=sequence_annotation.caption.text,
            sequence_short_caption=sequence_annotation.short_caption.text,
        )

        if frame_annotation.viewpoint is not None:
            frame_data.camera = self._get_uco3d_camera(frame_annotation)

        mask_annotation = frame_annotation.mask
        if mask_annotation is not None:
            fg_mask_np: Optional[np.ndarray] = None
            if load_blobs and self.load_masks:
                mask_image_load_start_time = time.time()
                if self.load_frames_from_videos:
                    mask_path = sequence_annotation.mask_video.path
                    mask_video_path_loc = self._local_path(mask_path)
                    fg_mask_np = self._frame_from_video(
                        mask_video_path_loc,
                        frame_annotation.frame_timestamp,
                    )
                    if fg_mask_np is None:
                        raise ValueError(
                            f"Cannot load mask frame from {mask_video_path_loc}"
                        )
                    fg_mask_np = fg_mask_np[:1]  # OpenCV converts grayscale to RGB
                else:
                    fg_mask_np, mask_path = self._load_fg_probability(frame_annotation)
                frame_data.mask_path = mask_path

                logger.debug(
                    f"Mask frame load time {time.time()-mask_image_load_start_time}"
                )
                frame_data.fg_probability = safe_as_tensor(fg_mask_np, torch.float)

            bbox_xywh = mask_annotation.bounding_box_xywh
            if bbox_xywh is None and fg_mask_np is not None:
                bbox_xywh = get_bbox_from_mask(fg_mask_np, self.box_crop_mask_thr)

            frame_data.bbox_xywh = safe_as_tensor(bbox_xywh, torch.float)

        if frame_annotation.image is not None:
            image_size_hw = safe_as_tensor(frame_annotation.image.size, torch.long)
            frame_data.image_size_hw = image_size_hw  # original image size
            # image size after crop/resize
            frame_data.effective_image_size_hw = image_size_hw
            if (
                frame_annotation.image.path is not None
                and self.dataset_root is not None
            ):
                frame_data.image_path = os.path.join(
                    self.dataset_root,
                    frame_annotation.image.path,
                )

            if load_blobs and self.load_images:
                assert frame_data.image_path is not None
                rgb_image_load_start_time = time.time()
                if self.load_frames_from_videos:
                    image_np = self._frame_from_video(
                        self._local_path(sequence_annotation.video.path),
                        frame_annotation.frame_timestamp,
                    )
                else:
                    image_np = load_image(self._local_path(frame_data.image_path))
                assert image_np is not None
                logger.debug(
                    f"rgb frame load time {time.time()-rgb_image_load_start_time}"
                )
                frame_data.image_rgb = self._postprocess_image(
                    image_np, frame_annotation.image.size, frame_data.fg_probability
                )

        # Undistort masks and rgb images, and adjust camera
        if self.undistort_loaded_blobs:
            (
                frame_data.image_rgb,
                frame_data.fg_probability,
                frame_data.camera,
            ) = undistort_frame_data_opencv(frame_data)

        # Load depth map from depth_video.mkv
        if load_blobs and self.load_depths:
            if self.load_frames_from_videos:
                (
                    depth_map,
                    depth_path,
                    depth_mask,
                ) = self._load_mask_depth_from_video(
                    frame_annotation,
                    sequence_annotation,
                    frame_data.fg_probability,
                )
            else:
                (
                    depth_map,
                    depth_path,
                    depth_mask,
                ) = self._load_mask_depth_from_file(
                    frame_annotation,
                    frame_data.fg_probability,
                )

            # apply the scale adjustment
            depth_map = depth_map * frame_annotation._depth_scale_adjustment

            # remove infinite values
            depth_map[~torch.isfinite(depth_map)] = 0.0

            # finally assign to frame_data
            frame_data.depth_scale_adjustment = frame_annotation._depth_scale_adjustment
            frame_data.depth_path = depth_path
            frame_data.depth_map = depth_map
            frame_data.depth_mask = depth_mask

        # load all possible types of point clouds
        if load_blobs:
            for pcl_type_str in ["", "sparse_", "segmented_"]:
                do_load_pcl = getattr(self, f"load_{pcl_type_str}point_clouds")
                if not do_load_pcl:
                    continue
                pcl_annot = getattr(
                    sequence_annotation,
                    f"{pcl_type_str}point_cloud",
                    None,
                )
                assert pcl_annot is not None
                pcl_path = os.path.join(self.dataset_root, pcl_annot.path)
                point_cloud = self._load_point_cloud(pcl_path)
                setattr(
                    frame_data, f"sequence_{pcl_type_str}point_cloud_path", pcl_path
                )
                setattr(frame_data, f"sequence_{pcl_type_str}point_cloud", point_cloud)

        # warnings.warn("Test gaussian splat loading!")
        if load_blobs and self.load_gaussian_splats:
            gaussians_dir = os.path.join(
                self.dataset_root,
                sequence_annotation.gaussian_splats.dir,
            )
            if self.use_cache:
                sequence_gaussians = copy.deepcopy(
                    self._gaussian_splat_cache[gaussians_dir]
                )  # make sure we do not overwrite the cache
            else:
                gaussians_dir_local = self._local_path(gaussians_dir)
                sequence_gaussians = load_compressed_gaussians(
                    gaussians_dir_local,
                    load_higher_order_harms=self.gaussian_splats_load_higher_order_harms,
                )
            if self.gaussian_splats_truncate_background:
                if sequence_gaussians.fg_mask is None:
                    warnings.warn(
                        f"No Gaussian foreground mask found for truncation"
                        f" {gaussians_dir}! Skipping background cropping."
                    )
                else:
                    sequence_gaussians = truncate_bg_gaussians(sequence_gaussians)
            frame_data.sequence_gaussian_splats = sequence_gaussians

        if self.box_crop:
            frame_data.crop_by_metadata_bbox_(self.box_crop_context)

        if self.image_height is not None and self.image_width is not None:
            new_size = (self.image_height, self.image_width)
            frame_data.resize_frame_(
                new_size_hw=torch.tensor(new_size, dtype=torch.long),  # pyre-ignore
            )

        if self.apply_alignment:
            self._apply_alignment_transform_(sequence_annotation, frame_data)

        return frame_data

    def _apply_alignment_transform_(self, sequence_annotation, frame_data):
        assert sequence_annotation.alignment is not None
        assert sequence_annotation.alignment.R
        assert sequence_annotation.alignment.T
        R = torch.tensor(sequence_annotation.alignment.R, dtype=torch.float32)
        T = torch.tensor(sequence_annotation.alignment.T, dtype=torch.float32)
        s = torch.tensor(sequence_annotation.alignment.scale, dtype=torch.float32)

        # camera_before = copy.deepcopy(frame_data.camera)

        # align the camera using the align transform
        frame_data.camera.R = R.transpose(-1, -2) @ frame_data.camera.R
        frame_data.camera.T = s * (frame_data.camera.T - T @ frame_data.camera.R)

        # align point clouds
        for pcl_type_str in ["", "sparse_", "segmented_"]:
            pcl_field = f"sequence_{pcl_type_str}point_cloud"
            assert hasattr(frame_data, pcl_field), f"Missing {pcl_field}!"
            pcl = getattr(frame_data, pcl_field, None)
            if pcl is None:
                continue
            # points_before = torch.nn.functional.normalize(
            #     pcl.xyz @ camera_before.R + camera_before.T
            # )
            pcl_align = copy.copy(pcl)
            pcl_align.xyz = (pcl.xyz @ R + T) * s
            setattr(frame_data, pcl_field, pcl_align)
            # points_after = torch.nn.functional.normalize(
            #     pcl.xyz @ frame_data.camera.R + frame_data.camera.T
            # )
            # print((points_after - points_before).abs().max())
            # assert torch.allclose(points_after, points_before, atol=1e-2)

        if frame_data.sequence_gaussian_splats is not None:
            frame_data.sequence_gaussian_splats = transform_gaussian_splats(
                frame_data.sequence_gaussian_splats, R, T, s
            )

        if frame_data.depth_map is not None:
            # dont forget to rescale the depth map as well
            frame_data.depth_map = frame_data.depth_map * s

    def _load_point_cloud(self, path: str) -> PointCloud:
        path_local = self._local_path(path)
        if not os.path.exists(path_local):
            raise FileNotFoundError(
                f"PointCloud file {path} at {path_local} does not exist."
            )
        if self.use_cache:
            return copy.deepcopy(
                self._point_cloud_cache[path]
            )  # deepcopy to prevent cache overwrite
        return load_point_cloud(path_local)

    # TODO: NOT USED SINCE WE LOAD FRAMES FROM VIDEOS -> consider removing
    def _load_mask_depth_from_file(
        self,
        frame_annotation: UCO3DFrameAnnotation,
        fg_mask: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        entry_depth = frame_annotation.depth
        dataset_root = self.dataset_root
        assert dataset_root is not None
        assert entry_depth is not None and entry_depth.path is not None
        path = os.path.join(dataset_root, entry_depth.path)
        depth_map = load_depth(self._local_path(path), entry_depth.scale_adjustment)

        if self.mask_depths:
            assert fg_mask is not None
            depth_map *= fg_mask

        mask_path = entry_depth.mask_path
        if self.load_depth_masks and mask_path is not None:
            mask_path = os.path.join(dataset_root, mask_path)
            depth_mask = load_depth_mask(self._local_path(mask_path))
        else:
            depth_mask = (depth_map > 0.0).astype(np.float32)

        return torch.from_numpy(depth_map), path, torch.from_numpy(depth_mask)

    def _load_mask_depth_from_video(
        self,
        frame_annotation: UCO3DFrameAnnotation,
        sequence_annotation: UCO3DSequenceAnnotation,
        fg_mask: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        time_0 = time.time()
        depth_h5_path = os.path.join(
            self.dataset_root,
            sequence_annotation.depth_video.path,
        )
        depth_h5_path_local = self._local_path(depth_h5_path)
        if not os.path.isfile(depth_h5_path_local):
            raise FileNotFoundError(f"Depth video {depth_h5_path} does not exist.")
        
        print("!!! REPLACE FRAME NUM PARSING WITH A MORE PRINCIPLED SOLUTION !!!")
        h5_frame_num = int(''.join(
            filter(str.isdigit, os.path.split(frame_annotation.image.path)[-1]))
        )
        assert h5_frame_num==(frame_annotation.frame_number + 1)
        depth_map = load_h5_depth(depth_h5_path_local, h5_frame_num)[None]
        if self.mask_depths:
            assert fg_mask is not None
            depth_map *= fg_mask
        depth_mask = (depth_map > 0.0).float()
        logger.debug(f"Depth H5 {depth_h5_path} time for reading is {time.time()-time_0}.")
        return depth_map, "", depth_mask
    
    def _frame_from_video(
        self, video_path: str | None, timestamp_sec: float
    ) -> np.ndarray | None:
        logger.debug(f"Current video is {video_path}.")
        start_time = time.time()
        assert self.dataset_root is not None
        if timestamp_sec < 0:
            raise ValueError(
                f"Cannot get a frame at a negative timestamp {timestamp_sec} s"
                + f" from {video_path}."
            )
        full_video_path = os.path.join(self.dataset_root, video_path)
        path = self._local_path(full_video_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video {path} does not exist.")
        time_1 = time.time()
        logger.debug(
            f"Video {video_path} time elapsed till before creating capture"
            f" object is {time_1-start_time}"
        )
        capture = (
            # For cache we need to use the non-local full_video_path
            # because local paths can conflict
            self._video_capture_cache[full_video_path]
            if self.use_cache
            else cv2.VideoCapture(path)
        )
        time_2 = time.time()
        logger.debug(
            f"Video {video_path} Time for creating capture object is {time_2-time_1}."
        )
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        time_3 = time.time()
        logger.debug(f"Video {video_path} Time for capture set is {time_3-time_2}.")
        ret, image = capture.read()
        logger.debug(f"Video {video_path} Time for reading is {time.time()-time_3}.")
        if not ret:
            logger.warning(f"Failed to get frame from {video_path} at {timestamp_sec}.")
            return None
        image = image[..., ::-1]
        return transpose_normalize_image(image)

    def _load_fg_probability(
        self,
        entry: UCO3DFrameAnnotation,
    ) -> Tuple[np.ndarray, str]:
        assert self.dataset_root is not None and entry.mask is not None
        full_path = os.path.join(self.dataset_root, entry.mask.path)
        fg_probability = load_mask(self._local_path(full_path))
        if fg_probability.shape[-2:] != entry.image.size:
            raise ValueError(
                f"bad mask size: {fg_probability.shape[-2:]} vs {entry.image.size}!"
            )
        return fg_probability, full_path

    def _local_path(self, path: str) -> str:
        if self.path_manager is None:
            return path
        return self.path_manager.get_local_path(path)

    def _exists_in_dataset_root(self, relpath) -> bool:
        if not self.dataset_root:
            return False
        full_path = os.path.join(self.dataset_root, relpath)
        if self.path_manager is None:
            return os.path.exists(full_path)
        else:
            return self.path_manager.exists(full_path)

    def _postprocess_image(
        self,
        image_np: np.ndarray,
        image_size: Tuple[int, int],
        fg_probability: Optional[torch.Tensor],
    ) -> torch.Tensor:
        image_rgb = safe_as_tensor(image_np, torch.float)
        if image_rgb.shape[-2:] != image_size:
            raise ValueError(f"bad image size: {image_rgb.shape[-2:]} vs {image_size}!")
        if self.mask_images:
            assert fg_probability is not None
            image_rgb *= fg_probability
        return image_rgb

    def _get_uco3d_camera(
        self,
        entry: UCO3DFrameAnnotation,
    ) -> Cameras:
        entry_viewpoint = entry.viewpoint
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(
            entry_viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)
        colmap_distortion_coeffs = torch.tensor(
            entry_viewpoint.colmap_distortion_coeffs, dtype=torch.float
        )
        return Cameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            colmap_distortion_coeffs=colmap_distortion_coeffs[None],
            R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        )
