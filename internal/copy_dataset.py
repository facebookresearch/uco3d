# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import glob
import os

from uco3d import UCO3DDataset, UCO3DFrameDataBuilder


def main():
    # dataset = get_dataset()

    export_dataset_root = "/fsx-repligen/dnovotny/datasets/uco3d_sample_v2_exported/"
    # _copy_dataset_to_target_dir(
    #     dataset,
    #     export_dataset_root=export_dataset_root,
    # )

    # send to aws
    cmd = f"aws s3 sync {export_dataset_root} s3://genai-transfer/dnovotny/datasets/uco3d_sample_v2/"
    os.system(cmd)


def get_dataset() -> UCO3DDataset:
    # dataset_root = os.getenv(  # AWS
    #     "UCO3D_DATASET_ROOT",
    #     "/fsx-repligen/shared/datasets/uCO3D/batch_reconstruction/dataset_export/",
    # )
    dataset_root = os.getenv(  # MAST
        "UCO3D_DATASET_ROOT",
        "/home/dnovotny/data/uco3d_sample_v2/",
    )
    metadata_file = os.path.join(dataset_root, "metadata.sqlite")
    setlists_file = os.path.join(
        dataset_root,
        "set_lists",
        "set_lists_small.sqlite",
    )
    frame_data_builder_kwargs = dict(
        dataset_root=dataset_root,
        apply_alignment=True,
        load_images=True,
        load_depths=True,
        load_masks=True,
        load_depth_masks=True,
        load_gaussian_splats=True,
        gaussian_splats_truncate_background=False,
        load_point_clouds=False,
        load_segmented_point_clouds=False,
        load_sparse_point_clouds=False,
        box_crop=True,
        load_frames_from_videos=True,
        image_height=800,
        image_width=800,
        undistort_loaded_blobs=True,
    )
    frame_data_builder = UCO3DFrameDataBuilder(**frame_data_builder_kwargs)
    dataset_kwargs = dict(
        sqlite_metadata_file=metadata_file,
        subset_lists_file=setlists_file,
        subsets=["train", "val", "test"],
        frame_data_builder=frame_data_builder,
    )
    dataset = UCO3DDataset(**dataset_kwargs)
    return dataset


# TODO: erase this function
def _copy_dataset_to_target_dir(
    dataset,
    export_dataset_root: str,
):
    import shutil
    from tqdm import tqdm

    fls_to_copy = []
    for seq_annotation in dataset.sequence_annotations():
        glob_stmt = os.path.join(
            dataset.frame_data_builder.dataset_root,
            os.path.dirname(seq_annotation.video.path),
            "*",
        )
        fls_to_copy_now = glob.glob(glob_stmt)
        fls_to_copy.extend(fls_to_copy_now)

    fls_to_copy.append(os.path.join(dataset.sqlite_metadata_file))
    fls_to_copy.append(os.path.join(dataset.subset_lists_file))
    fls_to_copy = sorted(list(set(fls_to_copy)))

    for fl in tqdm(fls_to_copy):
        tgt_file = fl.replace(
            dataset.dataset_root,
            export_dataset_root,
        )
        print(f"{fl}\n    -> {tgt_file}")
        os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
        if os.path.isdir(fl):
            shutil.copytree(fl, tgt_file, symlinks=False, copy_function=shutil.copy2)
        else:
            shutil.copy(fl, tgt_file, follow_symlinks=True)


if __name__ == "__main__":
    main()
