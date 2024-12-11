import glob
import logging
import os
from typing import List

import numpy as np

from video_utils import make_video_mosaic


def main():
    video_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "rendered_gaussian_turntables",
    )
    video_list = glob.glob(os.path.join(video_dir, "*.mp4"))
    outpath = "./grid.mp4"
    make_video_mosaic(
        video_list,
        outpath,
        max_frames=60,
        one_vid_size=256,
        W=6,
        worker_pool_size=8,
        always_square=True,
    )


if __name__ == "__main__":
    main()
