# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import random
import unittest

import torch
from torch.utils.data import DataLoader
from uco3d.data_utils import get_all_load_dataset
from uco3d.dataset_utils.io_utils import load_depth
from uco3d.dataset_utils.scene_batch_sampler import SceneBatchSampler
from uco3d.dataset_utils.utils import resize_image


# To resolve memory leaks giving received 0 items from anecdata
# Reference link https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")


class TestDataloader(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.dataset = get_all_load_dataset()

    def test_iterate_dataset(self):
        dataset = self.dataset
        load_idx = [random.randint(0, len(dataset)) for _ in range(10)]
        for i in load_idx:
            _ = dataset[i]

    def test_iterate_dataloader(self):
        dataset = self.dataset
        dataloader = DataLoader(
            dataset,
            num_workers=4,
            collate_fn=dataset.frame_data_type.collate,
            shuffle=True,
        )
        for i, batch in enumerate(dataloader):
            if i > 10:
                break

    def test_iterate_scene_batch_sampler(self):
        dataset = self.dataset
        scene_batch_sampler = SceneBatchSampler(
            dataset=dataset,
            batch_size=16,
            num_batches=10,
            images_per_seq_options=[8],
        )
        dataloader = DataLoader(
            dataset,
            num_workers=4,
            batch_sampler=scene_batch_sampler,
            collate_fn=dataset.frame_data_type.collate,
        )
        for _ in dataloader:
            pass

    def _test_depth_map_from_video(self):
        depth_map_root = "/fsx-repligen/shared/datasets/uCO3D/temp_depth_check"

        if not os.path.exists(depth_map_root):
            print("Skipping test_depth_map_from_video - depth_map_root does not exist")
            return

        dataset_depth = get_all_load_dataset(
            frame_data_builder_kwargs=dict(
                apply_alignment=False,
                load_images=True,
                load_depths=True,
                load_masks=False,
                load_depth_masks=False,
                load_gaussian_splats=True,
                gaussian_splats_truncate_background=False,
                load_point_clouds=False,
                load_segmented_point_clouds=False,
                load_sparse_point_clouds=False,
                box_crop=False,  # !!!
                load_frames_from_videos=True,
                image_height=800,
                image_width=800,
                undistort_loaded_blobs=True,
            )
        )
        load_idx = [random.randint(0, len(dataset_depth)) for _ in range(100)]
        for i in load_idx:
            frame_data = dataset_depth[i]
            depth_file_path = os.path.join(
                depth_map_root,
                frame_data.sequence_name,
                os.path.splitext(os.path.split(frame_data.image_path)[-1])[0] + ".npy",
            )
            depth_frame = load_depth(
                depth_file_path,
                frame_data.depth_scale_adjustment,
            )
            depth_frame = torch.from_numpy(depth_frame)
            depth_frame_image, _, _ = resize_image(
                depth_frame,
                image_height=dataset_depth.frame_data_builder.image_height,
                image_width=dataset_depth.frame_data_builder.image_width,
                mode="nearest",
            )
            depth_frame_video = frame_data.depth_map
            assert depth_frame_image.shape == depth_frame_video.shape
            # df = (depth_frame_image-depth_frame_video).abs()
            # print(float(df.max()), float(df.mean()))
            torch.allclose(depth_frame_image, depth_frame_video)


if __name__ == "__main__":
    unittest.main()
