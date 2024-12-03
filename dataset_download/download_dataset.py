# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

from download_dataset_impl import build_arg_parser, download_dataset


DEFAULT_LINK_LIST_FILE = os.path.join(os.path.dirname(__file__), "links.json")
DEFAULT_SHA256S_FILE = os.path.join(os.path.dirname(__file__), "links/co3d_sha256.json")
DEFAULT_CATEGORY_LIST_FILE = os.path.join(
    os.path.dirname(__file__), "links/category_zip_map.json"
)
DEFAULT_SUB_CATEGORY_LIST_FILE = os.path.join(
    os.path.dirname(__file__), "links/sub_category_zip_map.json"
)

if __name__ == "__main__":
    # parser = build_arg_parser("CO3D", DEFAULT_LINK_LIST_FILE, DEFAULT_SHA256S_FILE)
    parser = build_arg_parser(
        "uCO3D",
        DEFAULT_CATEGORY_LIST_FILE,
        DEFAULT_SUB_CATEGORY_LIST_FILE,
        DEFAULT_SHA256S_FILE,
    )
    args = parser.parse_args()
    print("Checksum check is ", args.checksum_check)
    args.checksum_check = False
    download_dataset(
        # str(args.link_list_file),
        str(args.category_map_file),
        str(args.sub_category_map_file),
        str(args.download_folder),
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        download_categories=args.download_categories,
        download_sub_categories=args.download_sub_categories,
        download_modalities=args.download_modalities,
        checksum_check=bool(args.checksum_check),
        clear_archives_after_unpacking=bool(args.clear_archives_after_unpacking),
        sha256s_file=str(args.sha256_file),
        skip_downloaded_archives=not bool(args.redownload_existing_archives),
    )
