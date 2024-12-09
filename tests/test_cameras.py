# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import unittest

import torch

from testing_utils import get_all_load_dataset


class TestCameras(unittest.TestCase):
    def test_pytorch3d_camera_compatibility(self):
        try:
            import pytorch3d  # noqa
        except ImportError:
            print("Skipping test_pytorch3d_camera because pytorch3d is not installed.")
            return

        dataset = get_all_load_dataset(
            frame_data_builder_kwargs=dict(
                apply_alignment=False,
                load_gaussian_splats=False,
            )
        )

        load_idx = [random.randint(0, len(dataset)) for _ in range(3)]
        for i in load_idx:
            frame_data = dataset[i]
            xyz = frame_data.sequence_point_cloud.xyz
            camera = frame_data.camera
            camera_pt3d = frame_data.camera.to_pytorch3d_cameras()

            # cut off points that project too close to camera
            depth = (xyz[None] @ camera.R + camera.T)[0][..., 2]
            xyz = xyz[depth > 1e-1]

            # project points with pytorch3d and ours

            # project to camera coords
            y_cam = camera.transform_points_camera_coords(xyz[None])
            y_cam_pt3d = camera_pt3d.get_world_to_view_transform().transform_points(
                xyz[None]
            )
            assert torch.allclose(y_cam, y_cam_pt3d, atol=1e-4)

            # project to ndc
            y_ndc = camera.transform_points(xyz[None], eps=1e-4)
            y_ndc_pt3d = camera_pt3d.transform_points(xyz[None], eps=1e-4)
            assert torch.allclose(y_ndc_pt3d[..., :2], y_ndc, atol=1e-5)

            # project to screen coords
            y_screen = camera.transform_points_screen(xyz[None], eps=1e-4)
            y_screen_pt3d = camera_pt3d.transform_points_screen(
                xyz[None],
                with_xyflip=True,
                eps=1e-4,
            )
            # error is in pixels, hence the higher atol
            assert torch.allclose(y_screen_pt3d[..., :2], y_screen, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
