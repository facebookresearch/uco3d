# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import math
import glob
import dataclasses
import numpy as np
import tempfile
from PIL import Image
from typing import Tuple

# To resolve memory leaks giving received 0 items from anecdata
# Reference link https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")

from uco3d import UCO3DDataset, UCO3DFrameDataBuilder, Cameras, GaussianSplats
from uco3d.dataset_utils.scene_batch_sampler import SceneBatchSampler
from uco3d.dataset_utils.utils import load_depth, resize_image

from gsplat import rasterization, DefaultStrategy


def render_gaussians(
    frame_data,
    outfile: str,
    n_frames: int = 60,
    fps: int = 20,
):
    # truncate points outside a given spherical boundary:
    if frame_data.sequence_gaussian_splats.fg_mask is None:
        fg_mask = torch.ones_like(
            frame_data.sequence_gaussian_splats.means[:, 0], dtype=torch.bool
        )
    else:
        fg_mask = frame_data.sequence_gaussian_splats.fg_mask
    centroid = frame_data.sequence_gaussian_splats.means[fg_mask].mean(dim=0, keepdim=True)
    
    ok = (frame_data.sequence_gaussian_splats.means - centroid).norm(dim=1) < 4.5
    dct = dataclasses.asdict(frame_data.sequence_gaussian_splats)
    splats_truncated = type(frame_data.sequence_gaussian_splats)(
        **{k: v[ok] for k, v in dct.items() if v is not None}
    )
    
    camera_matrix, viewmats = generate_circular_path(n_frames=n_frames)
    camera_matrices = camera_matrix[None].repeat(n_frames, 1, 1)
    renders, _, _ = render_splats(
        viewmats,
        camera_matrices,
        splats_truncated,
        [512, 512],
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for ri, render in renders:
            framefile = os.path.join(tmpdir, f"{ri:04d}.png")
            Image.fromarray(render).save(framefile)
        
    
    raise NotImplementedError("Finish this function!")
   
    
def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x)
    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    m = np.concatenate([m, np.array([[0, 0, 0, 1]])], axis=0)
    return m
    
    
def generate_circular_path(
    n_frames: int = 120,
    focal: float = 2.6,
    height: float = 5.0,
    radius: float = 5.0,
    up = np.array([0, -1, 0]),
    cam_tgt = np.zeros(3),
):
    """Calculates a forward facing spiral path for rendering."""
    # Generate poses for spiral path.
    render_poses = []
    for theta in np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False):
        position = np.array(
            [np.cos(theta) * radius, height, -np.sin(theta) * radius]
        )
        lookdir = cam_tgt - position
        render_poses.append(viewmatrix(lookdir, up, position))
    render_poses = np.stack(render_poses, axis=0)
    K = np.array(
        [
            [focal, 0, 0.5],
            [0, focal, 0.5],
            [0, 0, 1],
        ]
    )
    return (
        torch.from_numpy(K).float(),
        torch.from_numpy(render_poses).float(),
    )


@torch.no_grad()
def render_splats(
    viewmats: np.ndarray,
    camera_matrix: np.ndarray,
    splats: GaussianSplats,
    render_size: Tuple[int, int],
    antialiased: bool = False,
    packed: bool = False,
    absgrad: bool = False,
    sparse_grad: bool = False,
    device: str = "cpu",  # "cuda:0",
    near_plane: float = 1.0,
    **kwargs,
):
    
    height, width = render_size
    n_cams = viewmats.shape[0]
    device = torch.device(device)

    strategy = DefaultStrategy()
    
    # w2c = torch.eye(4)[None].repeat(len(R), 1, 1)
    # w2c[:, :3, :4] = torch.cat(
    #     [R.cpu(), tvec.cpu()[..., None]],
    #     dim=-1,
    # )
    # w2c = w2c.to(device, dtype=torch.float32)
    
    # move splats to the device
    splats = dataclasses.asdict(splats)
    for k, v in splats.items():
        if torch.is_tensor(v):
            splats[k] = v.to(device, dtype=torch.float32)
    
    # parse splats
    N = splats["means"].shape[0]
    means = splats["means"]  # [N, 3]
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"].flatten())  # [N,]
    colors = torch.cat(
        [
            splats["sh0"][:, None],
            splats.get("shN", torch.empty([N, 0, 3], device=device)),
        ], 
        1,
    )  # [N, K, 3]

    rasterize_mode = "antialiased" if antialiased else "classic"

    sh_degree = math.log2(colors.shape[1]) - 1
    assert sh_degree.is_integer()
    sh_degree = int(sh_degree)

    # convert ndc camera matrix to pixel camera matrix
    camera_matrix_pix = (
        camera_matrix.to(device, dtype=torch.float32)
        @ torch.tensor(
            [
                [width, 0, 1],
                [0, height, 1],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )[None]
    )
    
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,  # [C, 4, 4]
        Ks=camera_matrix_pix,  # [C, 3, 3]
        width=width,
        height=height,
        packed=packed,
        absgrad=(
            absgrad
            if isinstance(strategy, DefaultStrategy)
            else False
        ),
        sparse_grad=sparse_grad,
        rasterize_mode=rasterize_mode,
        distributed=False,
        sh_degree=sh_degree,
        near_plane=near_plane,
        backgrounds=torch.ones(n_cams, 3, dtype=torch.float32, device=device),
        **kwargs,
    )
    
    return render_colors.clamp(0, 1), render_alphas, info


def get_dataset() -> UCO3DDataset:
    dataset_root = os.getenv(
        "UCO3D_DATASET_ROOT",
        "/fsx-repligen/shared/datasets/uCO3D/batch_reconstruction/dataset_export/",
    )
    metadata_file = os.path.join(dataset_root, "metadata_vgg_1128_test15.sqlite")
    setlists_file = os.path.join(
        dataset_root,
        "set_lists_allcat_val1100.sqlite",
    )
    frame_data_builder_kwargs = dict(
        dataset_root=dataset_root,
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
    )
    frame_data_builder = UCO3DFrameDataBuilder(**frame_data_builder_kwargs)
    dataset_kwargs = dict(
        dataset_root=dataset_root,
        sqlite_metadata_file=metadata_file,
        subset_lists_file=setlists_file,
        subsets=["train"],
        frame_data_builder=frame_data_builder,
    )
    dataset = UCO3DDataset(**dataset_kwargs)
    return dataset


# TODO: erase this function
def _copy_dataset_to_target_dir(
    dataset,
    export_dataset_root: str,
):
    import shutil
    from tqdm import tqdm
    fls_to_copy = []
    for seq_name in dataset.sequence_names():
        for modality in [
            "depth_videos",
            "gaussian_splats",
            "mask_videos",
            "point_clouds",
            "rgb_videos",
            "segmented_point_clouds",
            "sparse_point_clouds",
        ]:
            fls_to_copy.extend(
                glob.glob(
                    os.path.join(
                        dataset.dataset_root,
                        modality,
                        seq_name,
                        "*",
                    )
                )
            )
            
    fls_to_copy.append(os.path.join(dataset.sqlite_metadata_file))
    fls_to_copy.append(os.path.join(dataset.subset_lists_file))
            
    for fl in tqdm(fls_to_copy):
        tgt_file = fl.replace(
            dataset.dataset_root,
            export_dataset_root,
        )
        print(f"{fl}\n    -> {tgt_file}")
        os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
        if os.path.isdir(fl):
            shutil.copytree(fl, tgt_file)
        else:
            shutil.copy(fl, tgt_file)


def main():
    outroot = os.path.join(os.path.dirname(__file__), "render_rotating_gaussians")
    os.makedirs(outroot, exist_ok=True)
    
    dataset = get_dataset()
    
    # _copy_dataset_to_target_dir(
    #     dataset,
    #     export_dataset_root="/fsx-repligen/dnovotny/datasets/uco3d_sample/",
    # )
    
    seq_annots = dataset.sequence_annotations()
    sequence_name_to_score = dict(
        zip(
            seq_annots["sequence_name"],
            seq_annots["_reconstruction_quality_gaussian_splats"],
        )
    )
    
    # sort sequence_name_to_score by score descendingly
    sequence_name_to_score = dict(
        sorted(
            sequence_name_to_score.items(),
            key=lambda item: item[1], 
            reverse=True,
        )
    )
    
    for seq_name in sequence_name_to_score:
        print(f"{seq_name}: {sequence_name_to_score[seq_name]}")
        outfile = os.path.join(outroot, seq_name + "")
        dataset_idx = next(dataset.sequence_indices_in_order(seq_name))
        frame_data = dataset[dataset_idx]

        _copy_scene_to_tempdir(frame_data, outfile)


        render_gaussians(frame_data, outfile)


if __name__=="__main__":
    try:
        import gsplat
    except ImportError:
        raise ImportError("Please install gsplat by running `pip install gsplat`")
    main()