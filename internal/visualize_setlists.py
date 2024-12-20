import math
import torch
import warnings
import torchvision
import pandas as pd
import random
import os
from tqdm import tqdm


def _make_set_image_grid(canonical_render_dir, seqs, sz: int = 512):
    grid = []
    seqs = random.sample(seqs, min(256, len(seqs)))
    for seq in tqdm(seqs):
        seqgrid = os.path.join(canonical_render_dir, seq + ".png")
        if not os.path.exists(seqgrid):
            warnings.warn(f"missing {seqgrid}")
            continue
        seqgrid = torchvision.io.read_image(seqgrid)
        seqgrid = torch.nn.functional.interpolate(
            seqgrid[None] / 256, size=(sz, sz), mode="bilinear"
        )[0]
        grid.append(seqgrid)
    grid = torch.stack(grid)
    return grid


def _visualize_setlists(
    outdir,
    canonical_render_dir,
    set_lists_dir,
    set_list_name,
):
    set_list_path = os.path.join(set_lists_dir, set_list_name + ".sqlite")
    print(f"loading table {set_list_path}")
    table = pd.read_sql_table("set_lists", f"sqlite:///{set_list_path}")
    table_val = table[table["subset"] == "val"]
    val_seqs = table_val["sequence_name"].unique().tolist()
    print("visualizing")
    set_image = _make_set_image_grid(canonical_render_dir, val_seqs)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"set_images_val_{set_list_name}.png")
    print(outfile)
    torchvision.utils.save_image(
        set_image,
        outfile,
        nrow=int(math.ceil(math.sqrt(set_image.shape[0]))),
    )


outdir = "/fsx-repligen/dnovotny/visuals/uco3d_setlists/"
canonical_render_dir = (
    "/fsx-repligen/dnovotny/datasets/uCO3D/canonical_renders/v1_segmented=False"
)
set_lists_dir = "/fsx-repligen/shared/datasets/uCO3D/dataset_export/set_lists/"
set_list_names = [
    "set_lists_3categories-debug",
    "set_lists_static-categories-accurate-reconstruction",
    "set_lists_all-categories",
    "set_lists_dynamic-categories",
    "set_lists_static-categories",
]

for set_list_name in set_list_names:
    _visualize_setlists(
        outdir=outdir,
        canonical_render_dir=canonical_render_dir,
        set_lists_dir=set_lists_dir,
        set_list_name=set_list_name,
    )
