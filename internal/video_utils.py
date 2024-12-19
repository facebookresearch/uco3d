import glob
import itertools
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, TypeVar

import av
import numpy as np
from PIL import Image

try:
    from pytorch3d.implicitron.tools.video_writer import VideoWriter
except ImportError:
    warnings.warn("No pytorch3d.")
from tqdm.auto import tqdm


logger: logging.Logger = logging.getLogger(__name__)

_EPS = np.finfo(float).eps * 4.0
Image_ANTIALIAS = getattr(Image, "ANTIALIAS", Image.LANCZOS)


def get_ffm_map(stereo_left: bool, input_index=0):
    stream_index = 1 if stereo_left else 0
    return f"{input_index}:v:{stream_index}"


def get_ffm_stereo_crop_opts(stereo_left: bool):
    if stereo_left:
        return "crop=w=1/2*in_w:h=in_h"
    else:
        return "crop=w=1/2*in_w:h=in_h:x=1/2*in_w"


def resize_and_pad(im, desired_size, fix_width=False, bg_fill=1.0):
    # If fix_width = False:
    #   get no padding, just resize the image to have the desired width
    #   the height will be whatever fits
    # If fix_width = True
    #   output will be a square, with black padding if needed.
    from PIL import Image

    old_size = im.size  # old_size[0] is in (width, height) format

    if fix_width:
        ratio = float(desired_size) / old_size[0]
        new_size = tuple([int(x * ratio) for x in old_size])
        new_im = im.resize([desired_size, new_size[1]], Image_ANTIALIAS)
    else:
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image_ANTIALIAS)
        new_im = Image.new(
            "RGB",
            (desired_size, desired_size),
            color=tuple([int(bg_fill * 255)] * 3),
        )
        new_im.paste(
            im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
        )
    return new_im


def stack_4d_to_3d_array(arr, w=None):
    """
    Given an array of N images arr of shape [N, C, H, W]
    make them into a image by gluing them into
    rows then columns.
    returns an array of shape [C, enough * H, w * W]
    """
    N, C, H, W = arr.shape
    if w is None:
        w = int(math.ceil(math.sqrt(N)))
    h = int(math.ceil(N / w))
    n_add = w * h - N
    arr = np.concatenate((arr, np.tile(arr[:1] * 0.0, (n_add, 1, 1, 1))), axis=0)
    # arr now has shape [w*h, C, H, W]
    arr = arr.reshape(h, w, C, H, W)
    arr = arr.transpose(2, 0, 3, 1, 4)  # shape [C, h, H, w, W]
    arr = arr.reshape(C, h * H, w * W)
    return arr


def make_video_mosaic(
    video_paths,
    outpath,
    max_frames=50,
    one_vid_size=200,
    W=None,
    n_sel_frames=10,
    worker_pool_size=1,
    always_square=False,
    bg_fill: float = 1.0,
    rm_rows=False,
    fps=20,
):
    """
    Write a single video containing a mosaic of input videos.

    Args:
        video_paths: list of source paths or the string "None" for blank
        outpath: desination path for new video
        max_frames: number of frames of each video to take.
                    If a video has fewer frames, the last is repeated.
        one_vid_size: common size (width and height)
                        taken by the square of each individual video
        W: number of videos per row
        n_sel_frames: if not None, also return this many frames per video
        worker_pool_size: number of subprocesses to use to extract frames.
        always_square: make squares even if W=1
    """
    if not len(video_paths):
        raise ValueError("no videos specified")

    pool = None
    if worker_pool_size > 1:
        pool = Pool(worker_pool_size)
    not_square = not always_square

    all_frames, sel_frames = _mosaic_extraction(
        video_paths,
        W=W,
        one_vid_size=one_vid_size,
        n_sel_frames=n_sel_frames,
        max_frames=max_frames,
        pool=pool,
        not_square=not_square,
        bg_fill=bg_fill,
        rm_rows=rm_rows,
    )

    vw = VideoWriter(out_path=outpath, fps=fps)
    for fri in tqdm(range(max_frames), total=max_frames, leave=None):
        frames = [f[min(fri, len(f) - 1)] for f in all_frames]
        if W == 1:
            frame = np.concatenate(frames, axis=1)
        else:
            frame_tiles = np.stack(frames)
            frame = stack_4d_to_3d_array(frame_tiles, w=W)
        vw.write_frame(frame.astype(np.uint8).transpose(1, 2, 0))
    vw.get_video()

    if pool is not None:
        pool.terminate()

    return sel_frames


def _mosaic_worker(
    video,
    *,
    max_frames,
    one_vid_size,
    W,
    n_sel_frames,
    not_square,
    bg_fill,
    rm_rows,
):
    if video == "None":
        return np.full((max_frames, 3, one_vid_size, one_vid_size), 128, np.uint8), None
    frames = video_to_frames_in_memory(video, max_frames=max_frames)
    resized_frames = []
    for frame in frames[:max_frames]:
        if not_square:
            frame_ = resize_and_pad(
                frame, one_vid_size, fix_width=True, bg_fill=bg_fill
            )
        else:
            frame_ = resize_and_pad(frame, one_vid_size, bg_fill=bg_fill)
        frame_arr = np.array(frame_).transpose(2, 0, 1)
        if rm_rows:
            frame_arr = np.delete(frame_arr, rm_rows, axis=1)
        resized_frames.append(frame_arr)
    resized_frames = np.stack(resized_frames)

    sel_frames_ = None
    if n_sel_frames is not None:
        sel = np.round(np.linspace(0, max_frames - 1, n_sel_frames)).astype(int)
        sel_frames_ = stack_4d_to_3d_array(resized_frames[sel])
        sel_frames_ = Image.fromarray(sel_frames_.astype(np.uint8).transpose(1, 2, 0))

    return resized_frames, sel_frames_


def _mosaic_extraction(
    videos,
    *,
    max_frames,
    one_vid_size,
    W,
    n_sel_frames,
    pool,
    not_square,
    bg_fill,
    rm_rows,
):
    """
    Data extraction part of make_video_mosaic.
    This is only a separate function because it's simpler for multiprocessing
    to only be called in a small function.
    """

    f = partial(
        _mosaic_worker,
        W=W,
        max_frames=max_frames,
        n_sel_frames=n_sel_frames,
        one_vid_size=one_vid_size,
        not_square=not_square,
        bg_fill=bg_fill,
        rm_rows=rm_rows,
    )

    for video in videos:
        if video != "None":
            assert os.path.isfile(str(video)), str(video)

    if pool is None:
        worker_return = [f(v) for v in tqdm(videos, leave=None)]
    else:
        # This is the tqdm version of "worker_return = p.map(f, videos)"
        worker_return = list(tqdm(pool.imap(f, videos), total=len(videos), leave=None))

    resized_frames = [i[0] for i in worker_return]
    if n_sel_frames is None:
        return resized_frames, None
    return resized_frames, [i[1] for i in worker_return]


def video_to_frames_in_memory(video, **kwargs):
    with tempfile.TemporaryDirectory() as frame_dir:
        frames = video_to_frames(video, frame_dir, refresh=True, **kwargs)
        frames = [Image.open(x) for x in frames]
    return frames


def video_to_frames(
    video_list,
    frame_dir,
    *,
    refresh=False,
    trim_range_seconds: Tuple[Optional[float], Optional[float]] = (None, None),
    max_frames=None,
    video_splitting_fps=None,
    gamma_correction=None,
    stereo_left=False,
    stereo_channel_encoding=True,
    frame_name_template="frame%06d",
    frame_name_postfix=".jpg",
    high_quality=True,
    frame_number_offset: int = 0,
    output_max_frame_size: Optional[int] = None,
    convert_frame_numbers_to_global_ms: bool = False,
) -> List[str]:
    if not isinstance(video_list, list):
        video_list = [video_list]

    # If both max_frames and video_splitting_fps are set, raise an exception.
    if max_frames and video_splitting_fps:
        raise ValueError("You can't set both `max_frames` and `video_splitting_fps`")

    # If none is set, raise an exception.
    if not (max_frames or video_splitting_fps):
        raise ValueError(
            "You need to set at least one: `max_frames` or `video_splitting_fps`"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_args = _get_input_args_with_trim(video_list, *trim_range_seconds)
        cmd = ["ffmpeg", *itertools.chain(*input_args)]

        if max_frames is not None:
            vlen = sum(get_video_len(video) for video in video_list)
            fps = max(max_frames - 2, 1) / vlen
            cmd.extend(["-r", str(fps)])
        elif video_splitting_fps is not None:
            cmd.extend(["-r", str(video_splitting_fps)])

        video_filter_args = _get_video_filter_args(
            video_list[0],
            crop_from_stereo=not stereo_channel_encoding,
            gamma_correction=gamma_correction,
            stereo_left=stereo_left,
            output_max_frame_size=output_max_frame_size,
        )
        filter_type, map_args, new_video_filter_args = _get_filter_map_args(
            len(input_args), stereo_left
        )
        video_filter_args = new_video_filter_args + video_filter_args

        cmd.extend(map_args)
        if len(video_filter_args) > 0:
            cmd.extend([filter_type, ",".join(video_filter_args)])

        if not stereo_channel_encoding:
            cmd.extend(["-vf", get_ffm_stereo_crop_opts(stereo_left)])

        if high_quality:
            cmd.extend(["-qscale:v", "2"])

        if frame_number_offset > 0:
            cmd.extend(["-start_number", str(frame_number_offset)])

        cmd.append(os.path.join(tmpdir, f"{frame_name_template}{frame_name_postfix}"))

        logger.debug("ffmpeg command:\n" + " ".join(cmd))

        run_cmd(cmd, quiet=True)
        # run_cmd_ignore_output(cmd)

        frames_tmp = sorted(glob.glob(os.path.join(tmpdir, f"*{frame_name_postfix}")))

        frames = []
        for i, f in enumerate(frames_tmp, start=max(frame_number_offset, 1)):
            basename = os.path.basename(f)
            if convert_frame_numbers_to_global_ms:
                assert (
                    os.path.splitext(basename)[0] == frame_name_template % i
                ), f"Mismatch in the order: #{i} is named {basename}"
                basename = (
                    frame_name_template % round(i * 1000 / video_splitting_fps)
                ) + frame_name_postfix

            f_tgt = os.path.join(os.path.normpath(frame_dir), basename)
            shutil.move(f, f_tgt)
            frames.append(f_tgt)

    return frames


def video_to_thumbnail(
    video,
    video_out,
    *,
    trim_range_seconds: Tuple[Optional[float], Optional[float]] = (None, None),
    max_size=300,
    stereo_left=False,
    gamma_correction=None,
):
    if not isinstance(video, list):
        video = [video]

    video_filter_args = _get_video_filter_args(
        video[0],  # TODO: check that videos are compatible, if multiple
        stereo_left=stereo_left,
        gamma_correction=gamma_correction,
        output_max_frame_size=max_size,
    )
    assert len(video_filter_args) > 0, "It should at least set resolution"

    input_args = _get_input_args_with_trim(video, *trim_range_seconds)
    filter_type, map_args, new_video_filter_args = _get_filter_map_args(
        len(input_args), stereo_left
    )
    video_filter_args = new_video_filter_args + video_filter_args

    cmd = [
        "ffmpeg",
        *itertools.chain(*input_args),
        *map_args,
        filter_type,
        ",".join(video_filter_args),
        video_out,
    ]

    logger.debug(" ".join(cmd))
    run_cmd(cmd, quiet=True)


def get_video_scale(video):
    out = run_cmd(
        [
            "ffmpeg",
            "-i",
            video,
            "-map",
            get_ffm_map(False),
            "-c",
            "copy",
            "-f",
            "null",
            "-",
        ],
        get_output=True,
        quiet=True,
    )

    sz = None
    for row in out.strip().splitlines():
        re_match = re.search("\d{3,5}x\d{3,5}", row)
        if re_match is not None:
            sz = [int(x) for x in re_match.group().split("x")]
            break

    if sz is None:
        raise BaseException("Cant parse the size of video frames %s!" % video)

    return [sz[1], sz[0]]


def frames_to_video(frames, video, fps=20, max_sz=None):
    fsz = Image.open(frames[0]).size
    if max_sz is not None:
        resize = float(max_sz / max(fsz))
    else:
        resize = None
    vw = VideoWriter(out_path=video, fps=fps)
    for f in frames:
        vw.write_frame(f, resize=resize)
    vw.get_video()


def get_video_len_av(video, return_fps=False) -> Tuple[Optional[float], bool]:
    """
    Length of video in seconds if it can easily be found by libav.
    An example of a video where this fails is
    NO_PROJECT/41/1937/5756/1591886855931-27022413-video-VID_20200611_104645.mkv .

    Also returns a bool which is True if we discover video has been edited.
    """
    try:
        try:
            vid = av.open(str(video))
        except UnicodeDecodeError:
            vid = av.open(str(video), metadata_encoding="latin1")
    except Exception:
        return None, False

    with vid:
        if not vid.streams.video:
            return None, False
        stream = vid.streams.video[0]
        if stream.duration is None:
            return None, False
        duration_sec = float(stream.time_base * stream.duration)
        if not return_fps:
            ret2 = vid.metadata.get("title", "").startswith("clideo")
        else:
            ret2 = stream.frames / duration_sec
        return round(duration_sec, 2), ret2


def get_video_len_ffmpeg(video) -> float:
    try:
        out = run_cmd(
            [
                "ffmpeg",
                "-i",
                video,
                "-map",
                get_ffm_map(False),
                "-c",
                "copy",
                "-f",
                "null",
                "-",
            ],
            get_output=True,
            quiet=True,
        )
    except RuntimeError:
        print("ffmpeg failed!")
        return None
    out = out.strip().splitlines()
    lenrow = out[-2]
    duration_str = lenrow[lenrow.find("time=") :].split(" ")[0][5:]
    a, b, c = duration_str.split(":")
    duration_sec = int(a) * 3600 + int(b) * 60 + float(c)
    return duration_sec


def get_video_len(video):
    """
    Length of video in seconds.
    """
    duration_av, _ = get_video_len_av(video)
    if duration_av is not None:
        return duration_av

    return get_video_len_ffmpeg(video)


def get_video_len_for_check(video):
    """
    Returns tuple:
        - Length of video in seconds
        - Whether we noticed a video editor
    """
    duration_av, edited = get_video_len_av(video)
    if duration_av is not None:
        return duration_av, edited

    return get_video_len_ffmpeg(video), edited


def _get_filter_map_args(num_videos, stereo_left) -> Tuple[str, List[str], List[str]]:
    filter_type = "-vf"
    map_args = ["-map", get_ffm_map(stereo_left)]
    new_video_filter_args = []
    if num_videos > 1:
        filter_type = "-filter_complex"
        input_streams = "".join(
            f"[{get_ffm_map(stereo_left, input_index=i)}]" for i in range(num_videos)
        )
        new_video_filter_args = [f"{input_streams}concat=n={num_videos}:v=1:a=0"]
        map_args = []

    return filter_type, map_args, new_video_filter_args


def _get_video_segments(local_videos):
    curr_length = 0.0
    yield curr_length
    for local_video_path in local_videos:
        vlen, fps = get_video_len_av(local_video_path, return_fps=True)
        if vlen is None:
            logger.warning(f"{local_video_path} not a video!")
            return
        curr_length += vlen + 1 / fps  # adding the gap between last and first frame
        yield curr_length


def _get_input_args_with_trim(
    videos: List[str], start: Optional[float], end: Optional[float]
) -> List[List[str]]:
    if start is not None and end is not None and start >= end:
        raise ValueError("Asking to trim to an empty range!")

    if start is not None and start < 0.0:
        start = None

    segments, segments_next = itertools.tee(_get_video_segments(videos))
    next(segments_next, None)
    args_by_segment = []
    for seg_start, seg_end, video in zip(segments, segments_next, videos):
        assert seg_start <= seg_end
        logger.debug(
            f"start={start}, end={end}, seg_start={seg_start}, seg_end={seg_end}"
        )

        if (
            start is not None
            and seg_end <= start
            or end is not None
            and seg_start >= end
        ):
            continue

        # if we get here, the segment is at least partially chosen

        seg_args = []
        if start is not None and seg_start < start < seg_end:
            seg_args += ["-ss", str(start - seg_start)]

        if end is not None and seg_start < end < seg_end:
            seg_args += ["-to", str(end - seg_start)]

        seg_args += ["-i", video]
        args_by_segment.append(seg_args)

    return args_by_segment


def _get_trim_args(start: Optional[float], end: Optional[float]) -> List[str]:
    if start is None:
        return []

    start_arg = ["-ss", str(start)]
    if end is None:
        return start_arg

    if start >= end:
        raise ValueError("Asking to trim to an empty range!")

    return start_arg + ["-to", str(end)]
    # another way to achieve this is to use filters (below), however
    # it is slower and stores all frames before `start` unless
    # `setpts=PTS-STARTPTS` is specified
    # video_filter_args.append(f"trim={start}:{end}")


def _get_video_filter_args(
    video_path,
    *,
    crop_from_stereo=False,
    gamma_correction=None,
    stereo_left=False,
    output_max_frame_size: Optional[int] = None,
) -> List[str]:
    video_filter_args = []
    if crop_from_stereo:
        video_filter_args.append(get_ffm_stereo_crop_opts(stereo_left))

    if output_max_frame_size is not None:
        vsz = get_video_scale(video_path)
        rescale = output_max_frame_size / max(vsz)
        sz_thumb = [2 * round(s * rescale / 2) for s in vsz]  # enforce size%2=0
        video_filter_args.append(f"scale={sz_thumb[1]}:{sz_thumb[0]}")

    value = lookup_pattern_in_config_dict(
        gamma_correction, os.path.basename(video_path), logger, "gamma_correction"
    )
    if value is not None:
        video_filter_args.append(f"eq=gamma={value}")

    return video_filter_args


def check_ffmpeg():
    try:
        subprocess.call(
            ["ffmpeg", "-h"],
            stdout=open(os.devnull, "wb"),
            stderr=open(os.devnull, "wb"),
        )
    except Exception:
        return False
    return True


def run_cmd_ignore_output(args):
    subprocess.check_call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_cmd(args, *, get_output=False, quiet=False, logger_print=None, **kwargs):
    if quiet:
        print_fn = lambda *args: None
    elif logger_print is not None:
        print_fn = logger_print
    else:
        print_fn = lambda *args: print(*args, end="")

    # This is approximately the same as the ! magic in ipython.
    p = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
    )

    if get_output:
        out_buff = ""

    # This should be safe because we're not piping stdin to the process.
    # It gets tricky if we are, because the process can be waiting for
    # input while we're waiting for output.
    while True:
        # Wait for some output, read it and print it.
        output = p.stdout.read1(1024).decode("utf-8")
        if not quiet:
            print_fn(output)
        if get_output:
            out_buff += output
        if p.poll() is not None:
            output = p.stdout.read().decode("utf-8")
            if not quiet:
                print_fn(output)
            if get_output:
                out_buff += output
            break

    if p.returncode != 0:
        raise RuntimeError("Exited with error code:", p.returncode)

    if get_output:
        return out_buff


T = TypeVar("T")


def lookup_pattern_in_config_dict(
    config_dict: Optional[Dict[str, T]],
    name: str,
    logger: Optional[logging.Logger] = None,
    flavour: str = "",
) -> Optional[T]:
    for key, value in (config_dict or {}).items():
        # From Python 3.7 dictionary iteration order
        # is guaranteed to be in order of insertion.
        # Yaml file is parsed top to bottom,
        # if you put more specific key higher it will work correctly.
        if key in name:
            if logger is not None:
                logger.debug(
                    f"Key {key} found in name {name}. Setting {flavour} to {value}"
                )
            return value

    return None
