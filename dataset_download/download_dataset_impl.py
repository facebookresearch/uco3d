# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import boto3
import os
import shutil
import requests
import functools
import json
import warnings

from typing import List, Optional
from multiprocessing import Pool
from tqdm import tqdm

from check_checksum import check_uco3d_sha256


def download_dataset(
    link_list_file: str,
    download_folder: str,
    n_download_workers: int = 4,
    n_extract_workers: int = 4,
    download_super_categories: Optional[List[str]] = None,
    download_categories: Optional[List[str]] = None,
    download_modalities: Optional[List[str]] = None,
    checksum_check: bool = False,
    clear_archives_after_unpacking: bool = False,
    skip_downloaded_archives: bool = True,
    sha256s_file: Optional[str] = None,
):
    """
    Downloads and unpacks the dataset in UCO3D format.

    Note: The script will make a folder `<download_folder>/_in_progress`, which
        stores files whose download is in progress. The folder can be safely deleted
        the download is finished.

    Args:
        link_list_file: A text file with the list of zip file download links.
        download_folder: A local target folder for downloading the
            the dataset files.
        n_download_workers: The number of parallel workers
            for downloading the dataset files.
        n_extract_workers: The number of parallel workers
            for extracting the dataset files.
        download_super_categories: A list of super categories to download.
            If `None`, downloads all.
        download_categories: A list of categories to download.
            If `None`, downloads all.
        download_modalities: A list of modalities to download.
            If `None`, downloads all.
        checksum_check: Enable validation of the downloaded file's checksum before
            extraction.
        clear_archives_after_unpacking: Delete the unnecessary downloaded archive files
            after unpacking.
        skip_downloaded_archives: Skip re-downloading already downloaded archives.
    """

    if checksum_check and not sha256s_file:
        raise ValueError(
            "checksum_check is requested but ground-truth SHA256 file not provided!"
        )

    if not os.path.isfile(link_list_file):
        raise ValueError(
            "Please specify `link_list_file` with a valid path to a json"
            " with zip file download links."
        )
    
    if not os.path.isdir(download_folder):
        raise ValueError(
            "Please specify `download_folder` with a valid path to a target folder"
            + " for downloading the dataset."
            + f" {download_folder} does not exist."
        )

    # read the links file
    with open(link_list_file, "r") as f:
        links = json.load(f)

    # extract plausible modalities, categories, super categories
    uco3d_modalities = set()
    uco3d_categories = set()
    uco3d_super_categories = set()
    for modality, modality_links in links.items():
        uco3d_modalities.add(modality)
        if modality=="metadata":
            continue
        for super_category, super_category_links in modality_links.items():
            uco3d_super_categories.add(super_category)
            for category, _ in super_category_links.items():
                uco3d_categories.add(category)

    # check if the requested categories, super_categories, or modalities are valid
    for sel_name, download_sel, possible in zip(
        ("super_category", "category", "modality"),
        (download_super_categories, download_categories, download_modalities),
        (uco3d_super_categories, uco3d_categories, uco3d_modalities),
    ):
        if download_sel is not None:
            for sel in download_sel:
                if sel not in possible:
                    raise ValueError(
                        f"Invalid choice for '{sel_name}': {sel}. "
                        + f"Possible choices are: {str(possible)}."
                    )
    
    def _is_for_download(modality: str, super_category: str, category: str) -> bool:
        if download_modalities is not None and modality not in download_modalities:
            return False
        if (
            download_super_categories is not None 
            and super_category in download_super_categories
        ):
            return True
        if download_categories is not None and category not in download_categories:
            return False
        return True
    
    # determine links to files we want to download
    data_links = []
    def _add_to_data_links(link: str):
        data_links.append((f"part_{len(data_links):06d}.zip", link))
    for modality, modality_links in links.items():
        if modality=="metadata":
            assert isinstance(modality_links, str)
            _add_to_data_links(modality_links)
            continue
        for super_category, super_category_links in modality_links.items():
            for category, category_links in super_category_links.items():
                if _is_for_download(modality, super_category, category):
                    for l in category_links:
                        _add_to_data_links(l)

    # multiprocessing pool
    with Pool(processes=n_download_workers) as download_pool:
        print(f"Downloading {len(data_links)} dataset files ...")
        download_ok = {}
        for link_name, ok in tqdm(
            download_pool.imap(
                functools.partial(
                    _download_file,
                    download_folder,
                    checksum_check,
                    sha256s_file,
                    skip_downloaded_archives,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            download_ok[link_name] = ok

        if not all(download_ok.values()):
            not_ok_links = [n for n, ok in download_ok.items() if not ok]
            not_ok_links_str = "\n".join(not_ok_links)
            raise AssertionError(
                "The SHA256 checksums did not match for some of the downloaded files:\n"
                + not_ok_links_str
                + "\n"
                + "This is most likely due to a network failure."
                + " Please restart the download script."
            )

    print(
        f"Extracting {len(data_links)} dataset files ..."
    )
    with Pool(processes=n_extract_workers) as extract_pool:
        for _ in tqdm(
            extract_pool.imap(
                functools.partial(
                    _unpack_file,
                    download_folder,
                    clear_archives_after_unpacking,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            pass

    print("Done")


def _unpack_file(
    download_folder: str,
    clear_archive: bool,
    link: str,
):
    link_name, _ = link
    local_fl = os.path.join(download_folder, link_name)
    print(f"Unpacking dataset file {local_fl} ({link_name}) to {download_folder}.")
    shutil.unpack_archive(local_fl, download_folder)
    if clear_archive:
        os.remove(local_fl)


def _download_file(
    download_folder: str,
    checksum_check: bool,
    sha256s_file: Optional[str],
    skip_downloaded_files: bool,
    link: str,
):
    link_name, url = link
    local_fl_final = os.path.join(download_folder, link_name)

    if skip_downloaded_files and os.path.isfile(local_fl_final):
        print(f"Skipping {local_fl_final}, already downloaded!")
        return link_name, True

    in_progress_folder = os.path.join(download_folder, "_in_progress")
    os.makedirs(in_progress_folder, exist_ok=True)
    local_fl = os.path.join(in_progress_folder, link_name)

    print(f"Downloading dataset file {link_name} ({url}) to {local_fl}.")
    _download_with_progress_bar(url, local_fl, link_name)
    if checksum_check:
        print(f"Checking SHA256 for {local_fl}.")
        try:
            check_uco3d_sha256(
                local_fl,
                sha256s_file=sha256s_file,
            )
        except AssertionError:
            warnings.warn(
                f"Checksums for {local_fl} did not match!"
                + " This is likely due to a network failure,"
                + " please restart the download script."
            )
            return link_name, False

    os.rename(local_fl, local_fl_final)
    return link_name, True


def _download_with_progress_bar(url: str, fname: str, filename: str):
    # taken from https://stackoverflow.com/a/62113293/986477
    resp = requests.get(url, stream=True)
    print(url)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for datai, data in enumerate(resp.iter_content(chunk_size=1024)):
            size = file.write(data)
            bar.update(size)
            if datai % max((max(total // 1024, 1) // 20), 1) == 0:
                print(
                    f"{filename}: Downloaded {100.0*(float(bar.n)/max(total, 1)):3.1f}%."
                )
                print(bar)
