# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import random
import unittest
import dataclasses


from testing_utils import get_all_load_dataset
from uco3d.dataset_utils.gauss3d_utils import save_gsplat_ply


class TestGaussians(unittest.TestCase):
    def setUp(self):
        self.dataset = get_all_load_dataset(
            frame_data_builder_kwargs={"gaussian_splats_truncate_background": False},
        )

    def test_store_gaussians(self):
        outdir = os.path.join(os.path.dirname(__file__), "test_outputs")
        os.makedirs(outdir, exist_ok=True)
        dataset = self.dataset        
        forked_random = random.Random(42)
        load_idx = [forked_random.randint(0, len(dataset)) for _ in range(3)]
        for i in load_idx:
            entry = dataset[i]
            outfile = os.path.join(
                outdir,
                f"test_store_gaussians_{entry.sequence_name}.ply",
            )
            print(outfile)
            # truncate points outside a given spherical boundary:
            centroid = entry.sequence_gaussian_splats.means[
                entry.sequence_gaussian_splats.fg_mask
            ].mean(dim=0, keepdim=True)
            ok = (entry.sequence_gaussian_splats.means - centroid).norm(dim=1) < 4.5
            dct = dataclasses.asdict(entry.sequence_gaussian_splats)
            splats_truncated = type(entry.sequence_gaussian_splats)(
                **{k: v[ok] for k, v in dct.items()}
            )
            # store splats
            save_gsplat_ply(splats_truncated, outfile)

if __name__ == "__main__":
    unittest.main()
