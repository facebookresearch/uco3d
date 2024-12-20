# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import dataclasses
import os

import numpy as np
import torch
import torchvision

from uco3d import (
    GaussianSplats,
    get_all_load_dataset,
    render_splats_opencv,
)


def main():
    argparser = argparse.ArgumentParser(
        description=(
            "Render turn-table videos of Gaussian Splat"
            + "reconstructions from the uCO3D dataset."
        )
    )
    argparser.add_argument(
        "--output_root",
        type=str,
        help="Folder for outputting turn-table videos.",
        default=os.path.join(
            os.path.dirname(__file__),
            "rendered_gaussian_turntables",
        ),
    )
    argparser.add_argument(
        "--num_scenes",
        type=int,
        help="The number of visualised scenes.",
        default=10,
    )
    args = argparser.parse_args()

    # create output root folder
    outroot = str(args.output_root)
    os.makedirs(outroot, exist_ok=True)

    # obtain the dataset
    dataset = get_all_load_dataset(
        frame_data_builder_kwargs=dict(
            load_gaussian_splats=True,
            gaussian_splats_truncate_background=False,
            apply_alignment=True,
            # --- turn off all other data loading options ---
            load_images=False,
            load_depths=False,
            load_masks=False,
            load_depth_masks=False,
            load_point_clouds=False,
            load_segmented_point_clouds=False,
            load_sparse_point_clouds=False,
            # -----------------------------------------------
        ),
    )
    # sort the sequences based on the reconstruction quality score
    seq_annots = dataset.sequence_annotations()
    sequence_name_to_score = {
        sa.sequence_name: sa.reconstruction_quality.gaussian_splats for sa in seq_annots
    }
    sequence_name_to_score = dict(
        sorted(
            sequence_name_to_score.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # iterate over sequences and render a 360 video of each
    for seqi, seq_name in enumerate(sequence_name_to_score):
        if seqi >= int(args.num_scenes):
            break
        print(f"Rendering {seq_name}: {sequence_name_to_score[seq_name]}")
        outfile = os.path.join(outroot, seq_name + ".mp4")
        dataset_idx = next(dataset.sequence_indices_in_order(seq_name))
        frame_data = dataset[dataset_idx]
        assert seq_name == frame_data.sequence_name
        print(f"Rendering gaussians of sequence {seq_name}.")
        _render_gaussians(frame_data, outfile)
        print(f"Wrote video {os.path.abspath(outfile)}.")


def _render_gaussians(
    frame_data,
    outfile: str,
    n_frames: int = 23 * 4,
    fps: int = 23,
    truncate_gaussians_outside_sphere_thr: float = 4.5,
):
    # truncate gaussians outside a spherical boundary
    splats_truncated = _truncate_gaussians_outside_sphere(
        frame_data.sequence_gaussian_splats,
        truncate_gaussians_outside_sphere_thr,
    )

    # generate a circular camera path
    camera_matrix, viewmats = _generate_circular_path(n_frames=n_frames)
    camera_matrices = camera_matrix[None].repeat(n_frames, 1, 1)

    # render the splats
    renders, _, _ = render_splats_opencv(
        viewmats,
        camera_matrices,
        splats_truncated,
        [512, 512],
        near_plane=1.0,
        camera_matrix_in_ndc=True,
    )

    # finally write the visualisation
    torchvision.io.write_video(
        outfile,
        (renders.clamp(0, 1).cpu() * 255).round().to(torch.uint8),
        fps=fps,
    )


def _truncate_gaussians_outside_sphere(
    splats: GaussianSplats,
    thr: float,
) -> GaussianSplats:
    if splats.fg_mask is None:
        fg_mask = torch.ones_like(splats.means[:, 0], dtype=torch.bool)
    else:
        fg_mask = splats.fg_mask
    centroid = splats.means[fg_mask].mean(dim=0, keepdim=True)
    ok = (splats.means - centroid).norm(dim=1) < thr
    dct = dataclasses.asdict(splats)
    splats_truncated = GaussianSplats(
        **{k: v[ok] for k, v in dct.items() if v is not None}
    )
    return splats_truncated


def _viewmatrix(
    lookdir: np.ndarray, up: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """Construct lookat view matrix."""

    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x)

    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    m = np.concatenate([m, np.array([[0, 0, 0, 1]])], axis=0)
    return m


def _generate_circular_path(
    n_frames: int = 120,
    focal_ndc: float = 2.0,
    height: float = 7.0,
    radius: float = 10.0,
    up=np.array([0, -1, 0]),
    cam_tgt=np.zeros(3),
):
    """Calculates a circular path for rendering."""
    render_poses = []
    for theta in np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False):
        position = np.array([np.cos(theta) * radius, height, np.sin(theta) * radius])
        lookdir = cam_tgt - position
        render_poses.append(_viewmatrix(lookdir, up, position))
    render_poses = np.stack(render_poses, axis=0)
    K = np.array(
        [
            [focal_ndc, 0, 0],
            [0, focal_ndc, 0],
            [0, 0, 1],
        ]
    )
    return (
        torch.from_numpy(K).float(),
        torch.from_numpy(render_poses).float().inverse(),
    )


if __name__ == "__main__":
    main()
