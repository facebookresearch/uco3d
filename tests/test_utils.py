# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import unittest

import torch
from uco3d.dataset_utils.data_types import Cameras, PointCloud
from uco3d.dataset_utils.frame_data import UCO3DFrameData


class TestUtils(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Tests moving to a CUDA device")
    def test_frame_data_to(
        self,
    ):
        rand_frame_data = UCO3DFrameData(
            frame_number=torch.randint(10, [1]),
            sequence_name="test",
            sequence_category="test",
            frame_timestamp=torch.tensor([0.0]),
            image_size_hw=torch.tensor([800, 800]),
            effective_image_size_hw=torch.tensor([800, 800]),
            image_path="test",
            image_rgb=torch.rand(3, 800, 800),
            # masks out padding added due to cropping the square bit
            mask_crop=torch.rand(1, 800, 800),
            depth_path="test",
            depth_map=torch.rand(1, 800, 800),
            depth_mask=torch.rand(1, 800, 800),
            mask_path="test",
            fg_probability=torch.rand(1, 800, 800),
            bbox_xywh=torch.rand(4),
            crop_bbox_xywh=torch.rand(4),
            camera=Cameras(
                R=torch.rand(3, 3),
                T=torch.rand(3),
                focal_length=torch.rand(2),
                principal_point=torch.rand(2),
                colmap_distortion_coeffs=torch.rand(4),
                device=torch.device("cpu"),
                in_ndc=True,
                image_size=torch.tensor([800, 800]),
            ),
            sequence_point_cloud_path="test",
            sequence_point_cloud=PointCloud(
                xyz=torch.rand(100, 3),
                rgb=torch.rand(100, 3),
            ),
            sequence_point_cloud_idx=torch.randint(100, [1]),
            frame_type="test",
            meta=dict(),
        )

        for device_str in ("cpu", "cuda:0"):
            device = torch.device(device_str)

            def _test_on_device(frame_data_on_device):
                for field in dataclasses.fields(frame_data_on_device):
                    value = getattr(frame_data_on_device, field.name)
                    if torch.is_tensor(value):
                        assert value.device == device
                    elif isinstance(value, Cameras):
                        _test_on_device(value)
                    elif isinstance(value, PointCloud):
                        _test_on_device(value)

            rand_frame_data_on_device = rand_frame_data.to(device)
            _test_on_device(rand_frame_data_on_device)


if __name__ == "__main__":
    unittest.main()
