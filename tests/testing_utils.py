# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
import os

import numpy as np

from PIL import Image

VISUALIZATION_DIR = os.path.join(
    os.path.dirname(__file__),
    "test_outputs",
)


def fig_to_np_array(fig):
    """
    Convert a matplotlib figure to a numpy array.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    # Parse a numpy array from the image
    img = Image.open(buf)
    img_array = np.array(img)
    return img_array
