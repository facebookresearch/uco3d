# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import unittest

import torch

from uco3d import Cameras

from uco3d.data_utils import get_all_load_dataset

NO_PYTORCH3D = False
try:
    import pytorch3d  # noqa
    from pytorch3d.renderer import PerspectiveCameras
    from pytorch3d.transforms import random_rotations

except ImportError:
    NO_PYTORCH3D = True


class TestCameras(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        torch.manual_seed(42)

    def test_pytorch3d_real_data_camera_compatibility(self):
        if NO_PYTORCH3D:
            print(
                "Skipping test_pytorch3d_camera_compatibility because pytorch3d is not installed."
            )
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
            self._test_camera_compatibility_one(camera, camera_pt3d, xyz)

    def test_pytorch3d_random_camera_compatibility(self):
        if NO_PYTORCH3D:
            print(
                "Skipping test_pytorch3d_random_camera_compatibility because pytorch3d is not installed."
            )
            return

        n_pts = 100
        for n_cams in [1, 5]:
            for try_num in range(10):
                # print(f"Testing {n_cams} cameras, try {try_num}")
                R = random_rotations(n_cams)
                T = torch.randn(n_cams, 3)
                T[:, 2] += 3.0  # move the camera away from the origin
                focal_length = torch.rand(n_cams, 2) + 0.5
                principal_point = (torch.rand(n_cams, 2) - 0.5) * 0.1
                image_size = torch.randint(100, 1000, (n_cams, 2))

                if try_num == 0:
                    principal_point *= 0.0  # test principal point at origin

                camera = Cameras(
                    R=R,
                    T=T,
                    focal_length=focal_length,
                    principal_point=principal_point,
                    image_size=image_size,
                )
                camera_pt3d = PerspectiveCameras(
                    R=R,
                    T=T,
                    focal_length=focal_length,
                    principal_point=principal_point,
                    image_size=image_size,
                )

                # get xyz points and cut off points that project too close to camera
                ok = torch.zeros((1,), dtype=torch.bool)
                while ok.long().sum() < n_pts / 3:
                    xyz = torch.randn(n_pts, 3)
                    depth = (xyz[None] @ camera.R + camera.T[:, None])[..., 2]
                    ok = depth.min(dim=0).values > 0.1
                xyz = xyz[ok]

                self._test_camera_compatibility_one(camera, camera_pt3d, xyz)

    def _test_camera_compatibility_one(
        self, camera, camera_pt3d, xyz, depth_min: float = 0.1
    ):
        # remove points that are too close to the camera:
        depth = (xyz[None] @ camera.R + camera.T[:, None])[..., 2]
        ok = depth.min(dim=0).values > depth_min
        xyz = xyz[ok]
        assert xyz.shape[0] > 0

        # project to camera coords
        y_cam = camera.transform_points_to_camera_coords(xyz[None])
        y_cam_pt3d = camera_pt3d.get_world_to_view_transform().transform_points(
            xyz[None]
        )
        assert torch.allclose(y_cam, y_cam_pt3d, atol=1e-4)

        # project to ndc
        y_ndc = camera.transform_points(xyz[None], eps=1e-3)
        y_ndc_pt3d = camera_pt3d.transform_points(xyz[None], eps=1e-3)
        df = torch.abs(y_ndc_pt3d[..., :2] - y_ndc).abs().max()
        assert torch.allclose(y_ndc_pt3d[..., :2], y_ndc, atol=1e-4), df

        # project to screen coords
        y_screen = camera.transform_points_screen(xyz[None], eps=1e-4)
        y_screen_pt3d = camera_pt3d.transform_points_screen(
            xyz[None],
            with_xyflip=True,
            eps=1e-4,
        )
        # error is in pixels, hence the higher atol
        assert torch.allclose(y_screen_pt3d[..., :2], y_screen, atol=2e-2)

        # test unprojection
        depth = (xyz[None] @ camera.R + camera.T[:, None])[..., 2]
        y_ndc_depth = torch.cat([y_ndc, depth[..., None]], dim=-1)
        y_screen_depth = torch.cat([y_screen, depth[..., None]], dim=-1)
        for world_coordinates in [True, False]:
            # unproject ndc points
            xyz_unproj = camera.unproject_points(
                y_ndc_depth, world_coordinates=world_coordinates
            )
            xyz_unproj_pt3d = camera_pt3d.unproject_points(
                y_ndc_depth, world_coordinates=world_coordinates
            )
            assert torch.allclose(xyz_unproj, xyz_unproj_pt3d, atol=1e-4)
            if world_coordinates:
                assert torch.allclose(xyz_unproj, xyz, atol=1e-4)

            # unproject screen points
            xyz_unproj = camera.unproject_screen_points(
                y_screen_depth, world_coordinates=world_coordinates
            )
            if world_coordinates:
                assert torch.allclose(xyz, xyz_unproj, atol=1e-4)
            else:
                assert torch.allclose(y_cam, xyz_unproj, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
