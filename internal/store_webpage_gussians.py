# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import os
import torch
import torchvision
from tqdm import tqdm
from uco3d.data_utils import get_all_load_dataset, get_dataset_root
from uco3d.dataset_utils.gauss3d_utils import (
    save_gsplat_ply,
    transform_gaussian_splats,
)


def _crop_mask_to_center_offsets(crop_mask):
    max_x = crop_mask.min(dim=1).indices.max()
    max_y = crop_mask.min(dim=0).indices.max()
    if max_x == 0:
        max_x = crop_mask.shape[1]
    if max_y == 0:
        max_y = crop_mask.shape[0]
    # else:
    #     raise ValueError("Crop mask is not a rectangle")
    #     offs_top = (crop_mask.shape[0] - max_y)//2
    #     offs_bot = crop_mask.shape[0]-max_y-offs_top
    #     assert offs_top+offs_bot+max_y == crop_mask.shape[0]
    #     offs_left = offs_right = 0
    # elif max_y==0:
    #     offs_left = (crop_mask.shape[1] - max_x)//2
    #     offs_right = crop_mask.shape[1]-max_x-offs_left
    #     assert offs_left+offs_right+max_x == crop_mask.shape[1]
    #     offs_bot = offs_top = 0
    #     height = crop_mask.shape[0]
    #     width = crop_mask.shape[1] - max_x
    # else:
    #     raise ValueError("Crop mask is not a rectangle")
    offs_top = (crop_mask.shape[0] - max_y) // 2
    offs_bot = crop_mask.shape[0] - max_y - offs_top
    offs_left = (crop_mask.shape[1] - max_x) // 2
    offs_right = crop_mask.shape[1] - max_x - offs_left
    assert offs_top + offs_bot + max_y == crop_mask.shape[0]
    assert offs_left + offs_right + max_x == crop_mask.shape[1]
    height = max_y
    width = max_x
    return offs_top, height, offs_left, width


outdir = "/fsx-repligen/dnovotny/datasets/uCO3D/webpage_gaussians/"
os.makedirs(outdir, exist_ok=True)
dataset = get_all_load_dataset(
    frame_data_builder_kwargs=dict(
        apply_alignment=True,
        load_gaussian_splats=True,
        gaussian_splats_truncate_background=False,
        box_crop=True,
        box_crop_context=0.3,
        image_width=512,
        image_height=512,
        # image_width=None,
        # image_height=None,
    ),
    dataset_kwargs=dict(
        subset_lists_file=os.path.join(
            get_dataset_root(assert_exists=True),
            "set_lists",
            "set_lists_static-categories-accurate-reconstruction.sqlite",
        ),
        subsets=["val"],
    ),  # this will load the whole dataset without any setlists
)


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
seq_names = list(sequence_name_to_score.keys())[:50]


for seq_name in tqdm(seq_names):
    i = next(dataset.sequence_indices_in_order(seq_name))
    entry = dataset[i]

    outfile_thumb = os.path.join(
        outdir,
        f"{entry.sequence_name}.png",
    )
    # store thumbnail
    print(os.path.abspath(outfile_thumb))

    im = entry.image_rgb
    crop_mask = entry.mask_crop[0]
    # get the boundaries of the crop mask
    offs_top, height, offs_left, width = _crop_mask_to_center_offsets(crop_mask)
    im_center = torch.ones_like(im)
    im_center[
        :,
        offs_top : offs_top + height,
        offs_left : offs_left + width,
    ] = entry.image_rgb[
        :,
        :height,
        :width,
    ]

    torchvision.utils.save_image(
        im_center,
        outfile_thumb,
    )

    # store caption
    outfile_caption = os.path.join(
        outdir,
        f"{entry.sequence_name}.caption",
    )
    with open(outfile_caption, "w") as f:
        f.write(entry.sequence_short_caption)

    outfile = os.path.join(
        outdir,
        f"{entry.sequence_name}.ply",
    )

    if entry.sequence_gaussian_splats.means.shape[0] <= 0:
        continue

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

    if splats_truncated.means.shape[0] <= 0:
        continue

    # flip upside down
    splats_truncated = transform_gaussian_splats(
        splats_truncated,
        T=torch.tensor([0.0, 0.0, 0.0]),
        R=torch.tensor(
            [
                [1.0, 0, 0],
                [0, -1.0, 0],
                [0, 0, -1.0],
            ],
        ),
        s=torch.tensor((1.0,)),
    )

    # store splats
    print(os.path.abspath(outfile))
    save_gsplat_ply(splats_truncated, outfile)
