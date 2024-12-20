import glob
import os
import glob
import random
import math

from video_utils import make_video_mosaic


def main():
    # video_dir = os.path.join(
    #     os.path.dirname(__file__),
    #     "..",
    #     "examples",
    #     "rendered_gaussian_turntables",
    # )

    video_dir = (
        "/fsx-repligen/dnovotny/visuals/uco3d_gauss_turntables_thr3p5_clipscore_uptgt"
    )
    bad_scenes = ["1-3955-88962"]
    video_list = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    video_list = [v for v in video_list if os.path.basename(v) not in bad_scenes]

    random.seed(42)
    random.shuffle(video_list)

    import pdb

    pdb.set_trace()

    for sz in [4, 8, 16, 20, 24, 30]:
        video_list_now = video_list[: (sz * sz * 2)]
        aspect = 2
        out_video_path = (
            os.path.join(
                os.path.dirname(video_dir),
                os.path.split(video_dir)[-1],
            )
            + f"_sz{sz}_aspect{aspect}.mp4"
        )
        best_rows, best_cols = _reshape_array(video_list_now, aspect)
        W = best_rows
        make_video_mosaic(
            video_list_now,
            out_video_path,
            max_frames=4 * 23,
            fps=23,
            one_vid_size=128 if sz <= 8 else 64,
            # one_vid_size=16,
            W=W,
            worker_pool_size=8,
            always_square=True,
        )
        print(out_video_path)


def _reshape_array(arr, aspect):
    """
    Reshape a 1D array into a 2D array with an aspect ratio as close to 2.0 as possible.

    Parameters:
    arr (list): The input 1D array.

    Returns:
    list: A 2D array with the desired aspect ratio.
    """

    # Calculate the total number of elements in the array
    n = len(arr)

    # Initialize variables for the best dimensions found so far
    best_diff = float("inf")
    best_rows = 0
    best_cols = 0

    # Iterate over all possible numbers of rows
    for rows in range(1, n + 1):
        # Check if the number of columns would be an integer
        if n % rows == 0:
            cols = n // rows

            # Calculate the difference between the current aspect ratio and 2.0
            diff = abs(rows / cols - float(aspect))

            # Update the best dimensions if the current ones are better
            if diff < best_diff:
                best_diff = diff
                best_rows = rows
                best_cols = cols

    return best_rows, best_cols


if __name__ == "__main__":
    main()
