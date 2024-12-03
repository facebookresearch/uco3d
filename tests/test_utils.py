# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import warnings
import argparse
import torch
import random
import unittest

# To resolve memory leaks giving received 0 items from anecdata
# Reference link https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")

from torch.utils.data import DataLoader
import logging

# logging.basicConfig(level=logging.DEBUG)


from uco3d.uco3d_dataset import UCO3DDataset
from uco3d.uco3d_frame_data_builder import UCO3DFrameDataBuilder
from uco3d.dataset_utils.scene_batch_sampler import SceneBatchSampler
from uco3d.dataset_utils.utils import load_depth, load_depth_mask, resize_image

from .utils import get_all_load_dataset


class TestDataloader(unittest.TestCase):
    def setUp(self):
        self.dataset = get_all_load_dataset(
            self.dataset_root,
            self.metadata_file,
            self.setlists_file,
        )
        self.dataset_root = self.dataset.dataset_root
        self.setlists_file = self.dataset.setlists_file
        self.metadata_file = self.dataset.metadata_file


if __name__ == "__main__":
    unittest.main()
