# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os

from typing import Optional

import torch
from uco3d.dataset_utils.frame_data import UCO3DFrameData
from uco3d.uco3d_dataset import UCO3DDataset, UCO3DFrameDataBuilder

from .dataset_utils.utils import get_dataset_root


logger = logging.getLogger(__name__)


def get_all_load_dataset(
    dataset_kwargs={},
    frame_data_builder_kwargs={},
) -> UCO3DDataset:
    """
    Get a UCO3D dataset with all data loading flags set to True.
    Make sure to set the environment variable for UCO3D_DATASET_ROOT
    to the root of the dataset.

    Args:
        dataset_kwargs: Additional keyword arguments to pass to the
            UCO3DDataset constructor.
        frame_data_builder_kwargs: Additional keyword arguments to pass to the
            UCO3DFrameDataBuilder constructor.
    Returns:
        A UCO3DDataset object.
    """
    dataset_root = get_dataset_root(assert_exists=True)
    subset_lists_file = os.path.join(
        dataset_root,
        "set_lists",
        "set_lists_small.sqlite",
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
            image_height=512,
            image_width=512,
            undistort_loaded_blobs=True,
        ),
        **frame_data_builder_kwargs,
    }
    frame_data_builder = UCO3DFrameDataBuilder(**frame_data_builder_kwargs)
    dataset_kwargs = {
        **dict(
            subset_lists_file=subset_lists_file,
            subsets=["train"],
            frame_data_builder=frame_data_builder,
        ),
        **dataset_kwargs,
    }
    dataset = UCO3DDataset(**dataset_kwargs)
    return dataset


def load_whole_sequence(
    dataset: UCO3DDataset,
    seq_name: str,
    max_frames: int,
    num_workers: int = 10,
    random_frames: bool = False,
) -> UCO3DFrameData:
    """
    Load a whole sequence from a UCO3D dataset into a single
    FrameData object.

    Args:
        dataset: The UCO3D dataset to load from.
        seq_name: The name of the sequence to load.
        max_frames: The maximum number of frames to load.
        num_workers: The number of workers to use for data loading.
        random_frames: If True, randomly select frames from the sequence.
            Otherwise, select frames in order.
    Returns:
        A FrameData object containing the loaded data.
    """
    seq_idx = dataset.sequence_indices_in_order(seq_name)
    seq_idx = list(seq_idx)
    if max_frames > 0 and len(seq_idx) > max_frames:
        if random_frames:
            sel = torch.randperm(len(seq_idx))[:max_frames]
        else:
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