# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import os
import torch
from tqdm import tqdm
from uco3d.data_utils import get_all_load_dataset
from uco3d.dataset_utils.gauss3d_utils import save_gsplat_ply

outdir = "./webpage_gaussians/"
os.makedirs(outdir, exist_ok=True)
dataset = get_all_load_dataset(
    frame_data_builder_kwargs=dict(
        apply_alignment=True,
        load_gaussian_splats=True,
    )
)
seq_names = dataset.sequence_names()
for seq_name in tqdm(seq_names[1:]):
    i = next(dataset.sequence_indices_in_order(seq_name))
    entry = dataset[i]
    outfile = os.path.join(
        outdir,
        f"{entry.sequence_name}.ply",
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
    print(os.path.abspath(outfile))
    save_gsplat_ply(splats_truncated, outfile)