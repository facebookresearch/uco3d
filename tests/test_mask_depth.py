# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
import os
import random
import unittest

import torch
import torchvision

from testing_utils import VISUALIZATION_DIR

from uco3d import GaussianSplats, render_splats
from uco3d.data_utils import get_all_load_dataset, load_whole_sequence
from uco3d.dataset_utils.gauss3d_utils import rgb_to_sh0, save_gsplat_ply


class TestMaskDepth(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_visualize_masks(self):
        """
        Visualize segmentation masks.
        """

        for box_crop in [True, False]:
            dataset = get_all_load_dataset(
                frame_data_builder_kwargs=dict(
                    load_depths=False,
                    load_point_clouds=False,
                    load_segmented_point_clouds=False,
                    load_sparse_point_clouds=False,
                    load_gaussian_splats=False,
                    box_crop=box_crop,
                    box_crop_context=0.1,
                )
            )
            seq_names = list(dataset.sequence_names())[:3]
            for seq_name in seq_names:
                self._test_visualize_masks_one(
                    dataset, seq_name, "_boxcrop" if box_crop else ""
                )

    def _test_visualize_masks_one(
        self,
        dataset,
        seq_name: str,
        postfix: str = "",
        max_frames_display: int = 200,
    ):
        frame_data = load_whole_sequence(
            dataset,
            seq_name,
            max_frames_display,
        )

        masks = frame_data.fg_probability
        ims = frame_data.image_rgb
        frames = torch.cat(
            [
                ims.mean(dim=1, keepdim=True),
                torch.zeros_like(ims[:, :1]),
                masks,
            ],
            dim=1,
        ).clamp(0, 1)

        frames = (frames * 255).round().to(torch.uint8).permute(0, 2, 3, 1)
        outdir = VISUALIZATION_DIR
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"mask_video_{seq_name}{postfix}.mp4")
        print(f"test_visualize_masks: Writing {outfile}.")
        torchvision.io.write_video(
            outfile,
            frames,
            fps=20,
            video_codec="h264",
            options={"-crf": "18", "-b": "2000k", "-pix_fmt": "yuv420p"},
        )

    def test_render_unprojected_depth(self):
        """
        Unproject monocular depth and render back into the views
        using Gaussian Splatting.
        """

        try:
            import gsplat  # noqa
        except ImportError:
            print(
                "Skipping test_visualize_gaussian_render"
                " because gsplat is not installed."
            )
            return

        for box_crop in [False, True]:
            dataset = get_all_load_dataset(
                frame_data_builder_kwargs=dict(
                    load_depths=True,
                    load_point_clouds=False,
                    load_segmented_point_clouds=False,
                    load_sparse_point_clouds=False,
                    load_gaussian_splats=False,
                    box_crop=box_crop,
                    box_crop_context=0.3,
                    apply_alignment=True,
                )
            )
            seq_names = list(dataset.sequence_names())[:3]
            for seq_name in seq_names:
                self._test_render_unprojected_depth_one(
                    dataset,
                    seq_name,
                    "_boxcrop" if box_crop else "",
                )

    def _test_render_unprojected_depth_one(
        sef,
        dataset,
        seq_name: str,
        postfix: str = "",
        max_gaussians: int = 250000,
    ):
        frame_data = load_whole_sequence(
            dataset,
            seq_name,
            max_frames=8,
        )

        # unproject depth from frame_data
        depth_map = frame_data.depth_map
        n_ims = depth_map.shape[0]
        xy_screen_grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, depth_map.shape[2]) + 0.5,
                torch.arange(0, depth_map.shape[3]) + 0.5,
            )
        ).flip(0)
        xy_screen = xy_screen_grid[None].repeat(n_ims, 1, 1, 1)
        xy_screen_depth = torch.cat([xy_screen, depth_map], dim=1)
        xy_screen_depth_flat = xy_screen_depth.permute(0, 2, 3, 1).reshape(n_ims, -1, 3)
        xyz = frame_data.camera.unproject_screen_points(
            xy_screen_depth_flat, world_coordinates=True
        ).reshape(-1, 3)

        # obtain the colored point cloud
        rgb = frame_data.image_rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        mask = frame_data.fg_probability.permute(0, 2, 3, 1).reshape(-1)
        ok = mask > 0.5
        xyz, rgb = xyz[ok], rgb[ok]
        n_pts = xyz.shape[0]
        if n_pts > max_gaussians:
            sel = torch.randperm(n_pts)[:max_gaussians]
            xyz, rgb = xyz[sel], rgb[sel]
        n_pts = xyz.shape[0]

        # make the corresponding gaussian point cloud
        splats = GaussianSplats(
            means=xyz,
            sh0=rgb_to_sh0(rgb),
            opacities=torch.ones_like(rgb[:, 0]),
            scales=torch.log(torch.ones_like(rgb) * 0.04),
            quats=torch.cat(
                [
                    torch.zeros(n_pts, 3),
                    torch.ones(n_pts, 1),
                ],
                dim=1,
            ),
        )

        # get the target viewpoints
        frame_data_target = load_whole_sequence(
            dataset,
            seq_name,
            max_frames=8,
            random_frames=True,
        )

        # render as gaussians
        rgb_render, mask_render, _ = render_splats(
            cameras=frame_data_target.camera,
            splats=splats,
            render_size=list(frame_data_target.image_rgb.shape[2:]),
        )

        frame = rgb_render.permute(0, 3, 1, 2).clamp(0, 1).cpu()
        frame = torch.concatenate(
            [
                frame,
                frame_data_target.image_rgb,
                (frame - frame_data_target.image_rgb).abs(),
            ],
            dim=3,
        )

        outdir = VISUALIZATION_DIR
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"gauss_unproj_renders_{seq_name}{postfix}.png")
        print(f"test_render_unprojected_depth: Writing {outfile}.")
        torchvision.utils.save_image(
            frame,
            outfile,
            nrow=int(round(math.sqrt(frame.shape[0] / 3))),
        )

        outfile = os.path.join(outdir, f"gauss_unproj_{seq_name}{postfix}.ply")
        print(f"test_render_unprojected_depth: Writing {outfile}.")
        save_gsplat_ply(splats, outfile)


if __name__ == "__main__":
    unittest.main()
