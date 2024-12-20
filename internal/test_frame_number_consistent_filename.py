import pandas as pd
import os
from tqdm import tqdm

# metadata_path = "/fsx-repligen/shared/datasets/uCO3D/dataset_export/"
metadata_path = "/fsx-repligen/shared/datasets/uCO3D/dataset_export_tool/temp_database_1214/metadata_1734152765.9553552.sqlite"
table = pd.read_sql_table("frame_annots", f"sqlite:///{metadata_path}")


def _path_to_frame_number(path):
    return int("".join(filter(str.isdigit, os.path.split(path)[-1])))


image_paths = table["_image_path"]
depth_paths = table["_depth_path"]
mask_paths = table["_mask_path"]
frame_numbers = table["frame_number"]
for row in tqdm(range(len(image_paths)), total=len(image_paths)):
    frame_num_path = _path_to_frame_number(image_paths[row])
    frame_num = frame_numbers[row]
    assert frame_num_path == (frame_num + 1)
