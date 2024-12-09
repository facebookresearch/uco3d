# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
import os

import numpy as np

import torch
from PIL import Image

from uco3d.dataset_utils.utils import get_dataset_root

from uco3d.uco3d_dataset import UCO3DDataset
from uco3d.uco3d_frame_data_builder import UCO3DFrameDataBuilder


def get_all_load_dataset(
    dataset_kwargs={},
    frame_data_builder_kwargs={},
):
    """
    Get a UCO3D dataset with all data loading flags set to True.
    Make sure to set the environment variable for UCO3D_DATASET_ROOT
    to the root of the dataset.
    """
    dataset_root = get_dataset_root(assert_exists=True)
    setlists_file = os.path.join(dataset_root, "set_lists_small.sqlite")
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


def load_whole_sequence(
    dataset,
    seq_name: str,
    max_frames: int,
    num_workers: int = 10,
):
    seq_idx = dataset.sequence_indices_in_order(seq_name)
    seq_idx = list(seq_idx)
    if max_frames > 0 and len(seq_idx) > max_frames:
        sel = (
            torch.linspace(
                0,
                len(seq_idx) - 1,
                max_frames,
            )
            .round()
            .long()
        )
        seq_idx = [seq_idx[i] for i in sel]
    seq_dataset = torch.utils.data.Subset(
        dataset,
        seq_idx,
    )
    dataloader = torch.utils.data.DataLoader(
        seq_dataset,
        batch_size=len(seq_dataset),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.frame_data_type.collate,
    )
    frame_data = next(iter(dataloader))
    return frame_data


def fig_to_np_array(fig):
    """
    Convert a matplotlib figure to a numpy array.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    # Parse a numpy array from the image
    img = Image.open(buf)
    img_array = np.array(img)
    return img_array
