# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

from uco3d.dataset_utils.utils import get_dataset_root

from uco3d.uco3d_dataset import UCO3DDataset
from uco3d.uco3d_frame_data_builder import UCO3DFrameDataBuilder


def get_all_load_dataset(
    dataset_kwargs={},
    frame_data_builder_kwargs={},
):
    dataset_root = get_dataset_root(assert_exists=True)
    print("!!! REMOVE THIS !!!")
    setlists_file = os.path.join(
        dataset_root,
        "set_lists_allcat_val1100.sqlite",
    )
    frame_data_builder_kwargs = {
        **dict(
            apply_alignment=True,
            load_images=True,
            load_depths=True,
            load_masks=True,
            load_depth_masks=True,
            load_gaussian_splats=True,
            gaussian_splats_truncate_background=True,
            load_point_clouds=True,
            load_segmented_point_clouds=True,
            load_sparse_point_clouds=True,
            box_crop=True,
            load_frames_from_videos=True,
            image_height=800,
            image_width=800,
            undistort_loaded_blobs=True,
        ),
        **frame_data_builder_kwargs,
    }
    frame_data_builder = UCO3DFrameDataBuilder(**frame_data_builder_kwargs)
    dataset_kwargs = {
        **dict(
            subset_lists_file=setlists_file,
            subsets=["train"],
            frame_data_builder=frame_data_builder,
        ),
        **dataset_kwargs,
    }
    dataset = UCO3DDataset(**dataset_kwargs)
    return dataset
