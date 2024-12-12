# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import math
import os
import random
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from testing_utils import fig_to_np_array, VISUALIZATION_DIR

from tqdm import tqdm
from uco3d.data_utils import get_all_load_dataset, load_whole_sequence
from uco3d.dataset_utils.gauss3d_rendering import render_splats
from uco3d.dataset_utils.gauss3d_utils import save_gsplat_ply


class TestGaussiansPCL(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_alignment(self):
        """
        Check that the point cloud reprojection is the same
        when aligned and not aligned.
        """
        dataset_aligned = get_all_load_dataset(
            frame_data_builder_kwargs=dict(
                apply_alignment=True,
                use_cache=False,
            )
        )
        dataset_aligned_cached = get_all_load_dataset(
            frame_data_builder_kwargs=dict(
                apply_alignment=True,
                use_cache=True,
            )
        )
        dataset_not_aligned = get_all_load_dataset(
            frame_data_builder_kwargs=dict(
                apply_alignment=False,
                use_cache=False,
            )
        )
        load_idx = [random.randint(0, len(dataset_aligned)) for _ in range(10)]
        for i in load_idx:
            frame_data_aligned = dataset_aligned[i]
            frame_data_not_aligned = dataset_not_aligned[i]
            frame_data_aligned_cached = dataset_aligned_cached[i]

            for pcl_type in [
                "sequence_point_cloud",
                "sequence_segmented_point_cloud",
                "sequence_sparse_point_cloud",
            ]:

                def _get_pcl_proj_rays(frame_data):
                    xyz = getattr(frame_data, pcl_type).xyz
                    camera = frame_data.camera
                    xyz_cam = camera.transform_points_to_camera_coords(xyz[None])[0]
                    rays = torch.nn.functional.normalize(xyz_cam, dim=-1)
                    return rays

                rays_aligned = _get_pcl_proj_rays(frame_data_aligned)
                rays_aligned_cached = _get_pcl_proj_rays(frame_data_aligned_cached)
                rays_not_aligned = _get_pcl_proj_rays(frame_data_not_aligned)
                assert torch.allclose(rays_aligned, rays_not_aligned, atol=1e-4)
                assert torch.allclose(rays_not_aligned, rays_aligned_cached, atol=1e-4)

    def test_visualize_gaussian_alignment(self):
        """
        Compare the rendering of the gaussian splats
        when aligned and not aligned.
        """
        try:
            import gsplat  # noqa
        except ImportError:
            print(
                "Skipping test_visualize_gaussian_render"
                " because gsplat is not installed."
            )
            return
        dataset_aligned, dataset_not_aligned = [
            get_all_load_dataset(
                frame_data_builder_kwargs=dict(
                    apply_alignment=apply_alignment,
                    load_depths=False,
                    load_masks=False,
                    load_gaussian_splats=True,
                    gaussian_splats_truncate_background=False,
                    gaussian_splats_load_higher_order_harms=True,
                    load_sparse_point_clouds=False,
                    load_point_clouds=False,
                    load_segmented_point_clouds=False,
                    box_crop=True,
                    box_crop_context=0.4,
                )
            )
            for apply_alignment in [True, False]
        ]
        seq_names = list(dataset_not_aligned.sequence_names())[:3]
        for seq_name in seq_names:
            self._test_visualize_gaussian_alignment_one(
                dataset_aligned,
                dataset_not_aligned,
                seq_name,
            )

    def _test_visualize_gaussian_alignment_one(
        self,
        dataset_aligned,
        dataset_not_aligned,
        seq_name: str,
        max_frames_render: int = 6,
    ):
        render_colors = []
        for dataset in [dataset_aligned, dataset_not_aligned]:
            frame_data = load_whole_sequence(
                dataset,
                seq_name,
                max_frames_render,
            )

            # rendering the sequence
            # print(
            #     "test_visualize_gaussian_rotation:"
            #     + f" Rendering gaussians for sequence {seq_name}."
            # )
            cameras = frame_data.camera
            gaussian_splats = frame_data.sequence_gaussian_splats[0]
            assert gaussian_splats is not None
            render_colors_now, _, _ = render_splats(
                cameras=cameras,
                splats=gaussian_splats,
                render_size=(512, 512),
                near_plane=0.01,
            )
            render_colors.append(render_colors_now.cpu())

        frame = torch.cat(render_colors, dim=1)
        frame = frame.clamp(0, 1).permute(0, 3, 1, 2)
        outdir = VISUALIZATION_DIR
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"aligned_vs_not_aligned_{seq_name}.png")
        print(f"test_visualize_gaussian_rotation: Writing {outfile}.")
        torchvision.utils.save_image(frame, outfile)

    def test_visualize_gaussian_render(self):
        """
        Visualise the rendering of the gaussian splats.
        """
        try:
            import gsplat  # noqa
        except ImportError:
            print(
                "Skipping test_visualize_gaussian_render"
                " because gsplat is not installed."
            )
            return
        dataset = get_all_load_dataset(
            frame_data_builder_kwargs=dict(
                apply_alignment=True,
                load_depths=False,
                load_masks=False,
                load_gaussian_splats=True,
                gaussian_splats_truncate_background=False,
                load_sparse_point_clouds=False,
                load_point_clouds=False,
                load_segmented_point_clouds=False,
                box_crop=True,
                box_crop_context=0.5,
            )
        )
        seq_names = list(dataset.sequence_names())[:3]
        for seq_name in seq_names:
            self._test_visualize_gaussian_render_one(dataset, seq_name)

    def _test_visualize_gaussian_render_one(
        self,
        dataset,
        seq_name: str,
        max_frames_render: int = 16,
    ):
        frame_data = load_whole_sequence(
            dataset,
            seq_name,
            max_frames_render,
        )

        # rendering the sequence
        # print(
        #     "test_visualize_gaussian_render:"
        #     + f" Rendering gaussians for sequence {seq_name}."
        # )
        cameras = frame_data.camera
        gaussian_splats = frame_data.sequence_gaussian_splats[0]
        assert gaussian_splats is not None
        im = frame_data.image_rgb.permute(0, 2, 3, 1)
        render_colors, render_alphas, info = render_splats(
            cameras=cameras,
            splats=gaussian_splats,
            render_size=(im.shape[1], im.shape[2]),
            near_plane=0.01,
        )

        frames = torch.cat(
            [
                render_colors.cpu(),
                im.cpu(),
                (render_colors.cpu() - im.cpu()).abs().cpu(),
            ],
            dim=2,
        ).clamp(0, 1)

        if True:  # save images
            frame = frames.permute(0, 3, 1, 2)
            outdir = VISUALIZATION_DIR
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, f"gauss_renders_{seq_name}.png")
            print(f"test_visualize_gaussian_render: Writing {outfile}.")
            torchvision.utils.save_image(
                frame,
                outfile,
                nrow=int(round(math.sqrt(frame.shape[0] / 3))),
            )

        else:  # save video
            frames = (frames * 255).round().to(torch.uint8)
            outdir = VISUALIZATION_DIR
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, f"gauss_renders_{seq_name}.mp4")
            print(f"test_visualize_gaussian_render: Writing {outfile}.")
            torchvision.io.write_video(
                outfile,
                frames,
                fps=20,
                video_codec="h264",
                options={"-crf": "18", "-b": "2000k", "-pix_fmt": "yuv420p"},
            )

    def test_visualize_pcl_reprojection(self):
        """
        Visualise the reprojection of the point clouds on the image.
        """
        dataset = get_all_load_dataset(
            frame_data_builder_kwargs=dict(
                apply_alignment=True,
                load_gaussian_splats=False,
                box_crop=True,
                box_crop_context=0.3,
            )
        )
        seq_names = list(dataset.sequence_names())[:3]
        for seq_name in seq_names:
            self._test_visualize_pcl_reprojection_one(dataset, seq_name)

    def _test_visualize_pcl_reprojection_one(
        self,
        dataset,
        seq_name: str,
        max_pts_plot: int = 100,
        max_frames_plot: int = 12,
    ):
        seq_idx = dataset.sequence_indices_in_order(seq_name)
        seq_idx = list(seq_idx)
        if max_frames_plot > 0 and len(seq_idx) > max_frames_plot:
            sel = (
                torch.linspace(
                    0,
                    len(seq_idx) - 1,
                    max_frames_plot,
                )
                .round()
                .long()
            )
            seq_idx = [seq_idx[i] for i in sel]
        frames = []
        pcl_sel = {}
        pcl_rgb_sel = {}
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for frame_idx, dataset_idx in enumerate(tqdm(seq_idx)):
            # get frame_data and camera
            frame_data = dataset[dataset_idx]
            camera = frame_data.camera
            # create the plot
            for pcli, pcl_type in enumerate(
                [
                    "sequence_point_cloud",
                    "sequence_segmented_point_cloud",
                    "sequence_sparse_point_cloud",
                ]
            ):
                ax_now = ax[pcli]
                ax_now.clear()
                # obtain the point cloud
                pcl = getattr(frame_data, pcl_type)
                if frame_idx == 0:
                    xyz_now = pcl.xyz
                    if xyz_now.shape[0] > max_pts_plot:
                        xyz_now = xyz_now[
                            torch.randperm(xyz_now.shape[0])[:max_pts_plot]
                        ]
                    pcl_rgb_sel[pcl_type] = np.random.rand(len(xyz_now), 3)
                    pcl_sel[pcl_type] = xyz_now
                xyz = pcl_sel[pcl_type]
                y = camera.transform_points_screen(xyz[None])[0]
                im = frame_data.image_rgb.permute(1, 2, 0).clamp(0, 1).numpy()
                ax_now.imshow(im)
                ax_now.scatter(
                    y[:, 0].numpy(),
                    y[:, 1].numpy(),
                    c=pcl_rgb_sel[pcl_type],
                    s=20.0,
                    marker="x",
                )
                ax_now.set_xlim(0, im.shape[1])
                ax_now.set_ylim(im.shape[0], 0)
                ax_now.set_xticks([])
                ax_now.set_yticks([])
                ax_now.set_title(pcl_type)
            plt.tight_layout()
            frame = torch.from_numpy(fig_to_np_array(fig))[..., :3]
            frames.append(frame)

        plt.close(fig)
        frames = torch.stack(frames)

        outdir = VISUALIZATION_DIR
        os.makedirs(outdir, exist_ok=True)
        if True:  # save images
            outfile = os.path.join(outdir, f"pcl_reprojections_{seq_name}.png")
            print(f"test_visualize_pcl_reprojection: Writing {outfile}.")
            torchvision.utils.save_image(
                frames.float().permute(0, 3, 1, 2) / 255,
                outfile,
                nrow=int(round(math.sqrt(frames.shape[0] / 3))),
            )

        else:  # save video
            outfile = os.path.join(outdir, f"pcl_reprojections_{seq_name}.mp4")
            print(f"test_visualize_pcl_reprojection: Writing {outfile}.")
            torchvision.io.write_video(
                outfile,
                frames,
                fps=10,
            )

    def test_store_gaussians(self):
        outdir = VISUALIZATION_DIR
        os.makedirs(outdir, exist_ok=True)
        dataset = get_all_load_dataset(
            frame_data_builder_kwargs=dict(
                apply_alignment=True,
                load_gaussian_splats=True,
                gaussian_splats_truncate_background=False,
            )
        )
        forked_random = random.Random(42)
        load_idx = [forked_random.randint(0, len(dataset)) for _ in range(3)]
        for i in load_idx:
            entry = dataset[i]
            outfile = os.path.join(
                outdir,
                f"test_store_gaussians_{entry.sequence_name}.ply",
            )
            # truncate points outside a given spherical boundary:
            if entry.sequence_gaussian_splats.fg_mask is None:
                fg_mask = torch.ones(entry.sequence_gaussian_splats.means.shape[0], dtype=bool)
            else:
                fg_mask = entry.sequence_gaussian_splats.fg_mask
            centroid = entry.sequence_gaussian_splats.means[fg_mask].mean(dim=0, keepdim=True)
            ok = (entry.sequence_gaussian_splats.means - centroid).norm(dim=1) < 4.5
            dct = dataclasses.asdict(entry.sequence_gaussian_splats)
            splats_truncated = type(entry.sequence_gaussian_splats)(
                **{k: v[ok] for k, v in dct.items() if v is not None}
            )
            # store splats
            save_gsplat_ply(splats_truncated, outfile)


if __name__ == "__main__":
    unittest.main()
