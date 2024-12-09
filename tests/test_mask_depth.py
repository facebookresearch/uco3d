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

from testing_utils import get_all_load_dataset, load_whole_sequence, VISUALIZATION_DIR

from tqdm import tqdm

from uco3d.dataset_utils.utils import load_point_cloud


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


if __name__ == "__main__":
    unittest.main()
