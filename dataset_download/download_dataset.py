# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

from argparse import ArgumentParser
from download_dataset_impl import download_dataset


DEFAULT_LINK_LIST_FILE = os.path.join(os.path.dirname(__file__), "links.json")
DEFAULT_CATEGORY_TO_ARCHIVES_FILE = os.path.join(
    os.path.dirname(__file__),
    "category_to_archives_file.json",
)


def build_arg_parser(
    dataset_name: str,
    default_link_list_file: str,
    default_category_to_archives_file: str,
) -> ArgumentParser:
    parser = ArgumentParser(description=f"Download the {dataset_name} dataset.")
    parser.add_argument(
        "--download_folder",
        type=str,
        required=True,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        "--n_download_workers",
        type=int,
        default=4,
        help="The number of parallel workers for downloading the dataset files.",
    )
    parser.add_argument(
        "--n_extract_workers",
        type=int,
        default=4,
        help="The number of parallel workers for extracting the dataset files.",
    )
    parser.add_argument(
        "--download_super_categories",
        type=lambda x: [x_.strip() for x_ in x.split(",")],
        default=None,
        help=f"A comma-separated list of {dataset_name} sub categories to download."
        + " If a super-category is specified, all its categories will be downloaded."
        + " Example: 'vegetables_and_legumes,stationery' will download only"
        + " the vegetables&legumes and stationery super-categories",
    )
    parser.add_argument(
        "--download_modalities",
        type=lambda x: [x_.strip() for x_ in x.split(",")],
        default=None,
        help=f"A comma-separated list of {dataset_name} modalities to download."
        + " Example: 'rgb_videos,point_clouds' will download only rgb videos and point clouds",
    )
    parser.add_argument(
        "--link_list_file",
        type=str,
        default=default_link_list_file,
        help=(f"The file with html links to the {dataset_name} dataset files."),
    )
    parser.add_argument(
        "--category_to_archives_file",
        type=str,
        default=default_category_to_archives_file,
        help=(f"The file with per-category zip files and their sha256 checksums."),
    )
    parser.add_argument(
        "--checksum_check",
        action="store_true",
        default=False,
        help="Check the SHA256 checksum of each downloaded file before extraction.",
    )
    parser.add_argument(
        "--no_checksum_check",
        action="store_false",
        dest="checksum_check",
        default=True,
        help="Does not check the SHA256 checksum of each downloaded file before extraction.",
    )
    parser.set_defaults(checksum_check=True)
    parser.add_argument(
        "--clear_archives_after_unpacking",
        action="store_true",
        default=False,
        help="Delete the unnecessary downloaded archive files after unpacking.",
    )
    parser.add_argument(
        "--redownload_existing_archives",
        action="store_true",
        default=False,
        help="Redownload the already-downloaded archives.",
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser(
        "uCO3D",
        DEFAULT_LINK_LIST_FILE,
        DEFAULT_CATEGORY_TO_ARCHIVES_FILE,
    )
    args = parser.parse_args()
    download_dataset(
        str(args.category_to_archives_file),
        str(args.link_list_file),
        str(args.download_folder),
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        download_super_categories=args.download_super_categories,
        download_modalities=args.download_modalities,
        checksum_check=bool(args.checksum_check),
        clear_archives_after_unpacking=bool(args.clear_archives_after_unpacking),
        skip_downloaded_archives=not bool(args.redownload_existing_archives),
    )
