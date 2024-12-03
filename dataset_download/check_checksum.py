# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import glob
import argparse
import hashlib
import json

from typing import Optional
from multiprocessing import Pool
from tqdm import tqdm


BLOCKSIZE = 65536


def get_expected_sha256s(sha256s_file: str):
    with open(sha256s_file, "r") as f:
        expected_sha256s = json.load(f)
    return expected_sha256s


def check_uco3d_sha256(
    path: str,
    sha256s_file: str,
    expected_sha256s: Optional[dict] = None,
    single_sequence_subset: bool = False,
    do_assertion: bool = True,
):
    zipname = os.path.split(path)[-1]
    if expected_sha256s is None:
        expected_sha256s = get_expected_sha256s(
            sha256s_file=sha256s_file,
            single_sequence_subset=single_sequence_subset,
        )
    extracted_hash = sha256_file(path)
    if do_assertion:
        assert (
            extracted_hash == expected_sha256s[zipname]
        ), f"{zipname}: ({extracted_hash} != {expected_sha256s[zipname]})"
    else:
        return extracted_hash == expected_sha256s[zipname]


def sha256_file(path: str):
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        file_buffer = f.read(BLOCKSIZE)
        while len(file_buffer) > 0:
            sha256_hash.update(file_buffer)
            file_buffer = f.read(BLOCKSIZE)
    digest_ = sha256_hash.hexdigest()
    return digest_