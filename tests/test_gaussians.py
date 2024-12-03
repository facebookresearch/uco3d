# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import random
import unittest


from testing_utils import get_all_load_dataset
from uco3d.dataset_utils.gauss3d_utils import save_gsplat_ply


class TestGaussians(unittest.TestCase):
    def setUp(self):
        self.dataset = get_all_load_dataset()

    def test_store_gaussians(self):
        outdir = os.path.join(os.path.dirname(__file__), "test_outputs")
        os.makedirs(outdir, exist_ok=True)
        dataset = self.dataset
        load_idx = [random.randint(0, len(dataset)) for _ in range(3)]
        for i in load_idx:
            entry = dataset[i]
            outfile = os.path.join(outdir, f"{i:03d}.ply")
            print(outfile)
            save_gsplat_ply(entry.sequence_gaussian_splats, outfile)

if __name__ == "__main__":
    unittest.main()
