import os
import sys
import sqlite3
import subprocess
import contextlib
import json
import copy
import tempfile
import pandas as pd
import random
import shutil
from uco3d import get_all_load_dataset
from pathlib import Path


DATASET_DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset_download")
TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "tests")
sys.path.insert(0, DATASET_DOWNLOAD_DIR)
sys.path.insert(0, TEST_DIR)


@contextlib.contextmanager
#temporarily change to a different working directory
def temporary_working_directory(path):
    _oldCWD = os.getcwd()
    os.chdir(os.path.abspath(path))

    try:
        yield
    finally:
        os.chdir(_oldCWD)


def _make_temporary_setlists(download_folder):
    metadata_file = os.path.join(download_folder, "metadata.sqlite")
    frame_annots = pd.read_sql_table("frame_annots", f"sqlite:///{metadata_file}")
    # add 10 sequences from frame_annots to train
    downloaded_seqs = list(_get_downloaded_seqs(download_folder))
    random.shuffle(downloaded_seqs)
    train_seqs = downloaded_seqs[:10]
    val_seqs = downloaded_seqs[10:13]
    
    target_setlist_file = os.path.join(
        download_folder,
        "set_lists",
        "set_lists_3categories-debug.sqlite",
    )
    os.makedirs(os.path.dirname(target_setlist_file), exist_ok=True)
    print(f"writing {target_setlist_file}")
    con = sqlite3.connect(target_setlist_file)
    setlists = pd.DataFrame()
    for set_ in ["train", "val"]:
        set_seqs = train_seqs if set_=="train" else val_seqs
        setlists_now = frame_annots[["frame_number","sequence_name"]]
        setlists_now = setlists_now[setlists_now["sequence_name"].isin(set_seqs)]
        setlists_now["subset"] = set_
        setlists = pd.concat([setlists, setlists_now])
    setlists.to_sql("set_lists", con, if_exists='replace', index=False)
    con.close()


def _get_downloaded_seqs(download_folder):
    print("determining downloaded sequences")
    downloaded_seqs = []
    for root, dirs, files in os.walk(download_folder):
        # Calculate the depth by counting the number of path separators
        depth = root.replace(download_folder, '').count(os.sep)
        if depth == 2:
            downloaded_seqs.append(os.path.basename(root))
    downloaded_seqs = set(downloaded_seqs)
    return downloaded_seqs


def _restrict_metadata_to_downloaded_files(big_metadata_file, download_folder):
    downloaded_seqs = _get_downloaded_seqs(download_folder)
    for tab_name in ["sequence_annots", "frame_annots"]:
        print(f"loading {tab_name}")
        annots = pd.read_sql_table(tab_name, f"sqlite:///{big_metadata_file}")
        print(f"filtering {tab_name}")
        annots = annots[annots["sequence_name"].isin(downloaded_seqs)]
        if tab_name=="sequence_annots":
            sequence_annots = annots
            assert len(annots) <= len(downloaded_seqs)
        elif tab_name=="frame_annots":
            frame_annots = annots
        else:
            raise ValueError(f"unknown table name {tab_name}")
    
    # Connect to the SQLite database (it will be created if it doesn't exist)
    with tempfile.TemporaryDirectory() as temp_dir:    
        target_db_file = os.path.join(temp_dir, "temp_metadata.sqlite")
        print(f"writing {target_db_file}")
        # assert not os.path.isfile(target_db_file)
        con = sqlite3.connect(target_db_file)
        # Write the dataframes to the database
        sequence_annots.to_sql('sequence_annots', con, if_exists='replace', index=False)
        frame_annots.to_sql('frame_annots', con, if_exists='replace', index=False)
        # Close the connection
        con.close()
        target_db_file_final = os.path.join(download_folder, "metadata.sqlite")    
        shutil.move(target_db_file, target_db_file_final)
        
def _test_iterate_dataset(
    download_folder,
    download_modalities = None,
):
    
    modality_to_load_switch = {
        "rgb_videos": "load_images",
        "depth_maps": "load_depths",
        "mask_videos": "load_masks",
        "gaussian_splats": "load_gaussian_splats",
        "point_clouds": "load_point_clouds",
        "segmented_point_clouds": "load_segmented_point_clouds",
        "sparse_point_clouds": "load_sparse_point_clouds",
    }
    dataset = get_all_load_dataset(
        dataset_kwargs={},
        frame_data_builder_kwargs={
            "dataset_root": download_folder,
            **{
                load_switch: (
                    True if download_modalities is None
                    else modality in download_modalities
                ) for modality, load_switch in modality_to_load_switch.items()
            }
        },
    )
    load_idx = [random.randint(0, len(dataset)) for i in range(20)]
    for i in load_idx:
        fd = dataset[i]  # load some data from it
        print(fd)


def _run_one_download(
    zipfiles_folder,
    download_folder,
    link_list_file,
    sha256_file,
    download_categories = None,
    download_super_categories = None,
    download_modalities = None,
    n_download_workers: int = 8,
    n_extract_workers: int = 8,
    big_metadata_file: str = None,
):
    
    # adjust the mapping file so it contains full paths to the archives
    with open(link_list_file, "r") as f:
        links = json.load(f)
    links_full_path = copy.deepcopy(links)
    for mod, mod_links in links.items():
        if mod=="metadata":
            assert isinstance(mod_links, str)
            links_full_path[mod] = os.path.join(zipfiles_folder, mod_links)
            continue
        for sc, sc_links in mod_links.items():
            for cat, cat_links in sc_links.items():
                links_full_path[mod][sc][cat] = [
                    os.path.join(zipfiles_folder, link) for link in cat_links
                ]
                assert all(
                    os.path.isfile(l) for l in links_full_path[mod][sc][cat]
                )
                
    if True:
        with tempfile.TemporaryDirectory() as temp_dir:

            link_list_file_temp = os.path.join(temp_dir, "links.json")
            with open(link_list_file_temp, "w") as f:
                json.dump(links_full_path, f)

            cmd = [
                "python",
                "download_dataset.py",
                "--download_folder",
                download_folder,
                "--link_list_file",
                link_list_file_temp,
                "--sha256_file",
                sha256_file,
                "--checksum_check",
                "--clear_archives_after_unpacking",
                "--redownload_existing_archives",
                "--n_download_workers",
                str(int(n_download_workers)),
                "--n_extract_workers",
                str(int(n_extract_workers)),
            ]
            
            def _add_to_cmd_if_not_none(arg_name, arg_value):
                if arg_value is not None:
                    cmd.extend([f"--{arg_name}", ",".join(arg_value)])
            
            _add_to_cmd_if_not_none(
                "download_categories",
                download_categories,
            )
            _add_to_cmd_if_not_none(
                "download_super_categories",
                download_super_categories,
            )
            _add_to_cmd_if_not_none(
                "download_modalities",
                download_modalities,
            )
            
            print(" ".join(cmd))
            
            os.makedirs(download_folder, exist_ok=True)
            
            with temporary_working_directory(DATASET_DOWNLOAD_DIR):
                # Code executed within this block will have the new directory 
                # as the current working directory
                print("Current directory:", os.getcwd())
                subprocess.check_call(cmd)
 
    _restrict_metadata_to_downloaded_files(big_metadata_file, download_folder)
 
    _make_temporary_setlists(download_folder)
 
    _test_iterate_dataset(
        download_folder,
        download_modalities = download_modalities,
    )
    
    if download_modalities is None:
        # run the testing suite
        with temporary_working_directory(TEST_DIR):
            cmd = ["python", "run.py"]
            print(" ".join(cmd))
            # update env to contain the new download
            env = os.environ.copy()
            env["UCO3D_DATASET_ROOT"] = download_folder
            subprocess.check_call(cmd, env=env)
 
 
 # download everything
# _run_one_download(
#     zipfiles_folder = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_zip/compressed/",
#     download_folder = "/fsx-repligen/dnovotny/datasets/uCO3D/extract_test/v1/",
#     link_list_file = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_zip/file_mapping.json",
#     sha256_file = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_zip/sha256_hashes.json",
#     download_categories = None,
#     download_super_categories = None,
#     download_modalities = None,
#     # big_metadata_file="/fsx-repligen/shared/datasets/uCO3D/dataset_export/metadata.sqlite",
#     big_metadata_file="/fsx-repligen/dnovotny/datasets/uCO3D/extract_test/v1/metadata.sqlite",
# )

# download some modalities
_run_one_download(
    zipfiles_folder = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_zip/compressed/",
    download_folder = "/fsx-repligen/dnovotny/datasets/uCO3D/extract_test/v1_modalities/",
    link_list_file = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_zip/file_mapping.json",
    sha256_file = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_zip/sha256_hashes.json",
    download_categories = None,
    download_super_categories = None,
    download_modalities = ["rgb_videos", "sparse_point_clouds", "point_clouds"],
    # big_metadata_file="/fsx-repligen/shared/datasets/uCO3D/dataset_export/metadata.sqlite",
    big_metadata_file="/fsx-repligen/dnovotny/datasets/uCO3D/extract_test/v1_modalities/metadata.sqlite",
)

# download some modalities and categories
# _run_one_download(
#     zipfiles_folder = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_zip/compressed/",
#     download_folder = "/fsx-repligen/dnovotny/datasets/uCO3D/extract_test/v0_modalities_categories/",
#     link_list_file = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_zip/file_mapping.json",
#     sha256_file = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_zip/sha256_hashes.json",
#     download_categories = None,
#     download_super_categories = ["safety_and_security_items"],
#     download_modalities = ["rgb_videos", "sparse_point_clouds", "point_clouds"],
#     big_metadata_file="/fsx-repligen/shared/datasets/uCO3D/dataset_export/metadata.sqlite",
# )