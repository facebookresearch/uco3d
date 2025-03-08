<center>
<img src="./uco3d_logo.png" width="600" />
</center>

# uCO3D: UnCommon Objects in 3D

[[paper](https://arxiv.org/abs/2501.07574)] [[project page](https://uco3d.github.io)]

This repository contains download scripts and tooling for working with the **UnCommon Objects in 3D (uCO3D)** dataset.

**uCO3D** contains ~170,000 turn-table videos capturing objects from the LVIS taxonomy of object categories.

The dataset is described in our paper ["UnCommon Objects in 3D"](https://arxiv.org/abs/2501.07574).

<center>
<img src="./uco3d_grid.gif" width="600" alt="grid"/>
</center>

## Updates
- Now the uCO3D dataset is also available on [hugging face](https://huggingface.co/datasets/facebook/uco3d)!


## Main features
- **170,000 videos** scanning diverse objects from all directions.
- Objects come from the LVIS taxonomy of **~1000 categories**, grouped into 50 super-categories.
- Unlike CO3Dv2, **uCO3D releases full original videos** instead of frames.
- Each video is annotated with object segmentation, camera poses, and **3 types of point clouds**.
- The dataset newly contains a **3D Gaussian Splat reconstruction for each video**.
- Each scene contains a long and short caption obtained with a large video-language model.
- Significantly improved annotation quality and size w.r.t. CO3Dv2.

# Download & Install

The full dataset (processed version) takes **~19.3 TB of space**. We distribute it in chunks up to 20 GB. We provide an automated way of downloading and decompressing the data.

First, run the install script that will also take care of dependencies:

```bash
git clone git@github.com:facebookresearch/uco3d.git
cd uco3d
pip install -e .
```

Then run the download script (make sure to change `<DESTINATION_FOLDER>`):

```bash
python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --checksum_check
```

## Download a small subset of the dataset

We allow users to download a small 52-video subset of uCO3D for preview and debugging purposes.
The videos represent a random sample of categories.
The small subset takes **~9.6 GB**, i.e. more than 1000x smaller than the full dataset.

To download the small subset, run the following:
```bash
python dataset_download/download_dataset.py --download_small_subset --download_folder <SMALL_SUBSET_DESTINATION_FOLDER>
```

## Download specific subsets of the dataset

As detailed here, we allow users to download only specific subsets of the dataset (e.g. only Gaussian Splats and RGB videos of specific object categories). This allows to greatly decrease the amount of required space.
Note that this functionality is not supported for [downloading the small subset](#download-a-small-subset-of-the-dataset) - the small subset will always be downloaded in full.

### Downloading specific modalities

Setting `--download_modalities` to a comma-separated list of specific modality names will download only a subset of available modalities.
For instance
```bash
python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --download_modalities "rgb_videos,point_clouds"
```
will only download rgb videos and point clouds.
Execute `python dataset_download/download_dataset.py -h` for the list of all downloadable modalities.

#### Dataset size per modality

The following table contains the size of all videos for a given modality:

```
----------------------------------
Modality                 Size (TB)     
----------------------------------
rgb_videos                    7.59
mask_videos                   0.16
depth_maps                    9.69
gaussian_splats               1.18
point_clouds                  0.57
segmented_point_clouds        0.04
sparse_point_clouds           0.04
----------------------------------
Total                        19.27
```

### Downloading specific super-categories

Setting `--download_super_categories` will instruct the script to download only a subset of the available categories.
For instance
```bash
python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --download_super_categories "vegetables_and_legumes,stationery"
```
will download only the vegetables&legumes and stationery super-categories.

Note that `--download_modalities` can be mixed with `--download_super_categories` to enable choosing any possible subset of the dataset.

Run `python dataset_download/download_dataset.py -h` for the full list of options.


# API Quick Start and Examples

1) [Download the dataset and install the `uco3d` package](#download--install)

2) Setup the dataset root environment var
    ```bash
    export UCO3D_DATASET_ROOT=<DESTINATION_FOLDER>
    ```
pointing to the root folder with the uCO3D dataset.

3) Create the dataset object and fetch its data:
    ```python
    from uco3d import UCO3DDataset, UCO3DFrameDataBuilder
    from uco3d.dataset_utils.utils import get_dataset_root
    import os
    # Get the dataset root folder and check that
    # all required metadata files exist.
    dataset_root = get_dataset_root(assert_exists=True)
    # Get the "small" subset list containing a small subset
    # of the uCO3D categories. For loading the whole dataset
    # use "set_lists_all-categories.sqlite".
    subset_lists_file = os.path.join(
        dataset_root,
        "set_lists", 
        "set_lists_3categories-debug.sqlite",
    )
    dataset = UCO3DDataset(
        subset_lists_file=subset_lists_file,
        subsets=["train"],
        frame_data_builder=UCO3DFrameDataBuilder(
            apply_alignment=True,
            load_images=True,
            load_depths=False,
            load_masks=True,
            load_depth_masks=True,
            load_gaussian_splats=True,
            gaussian_splats_truncate_background=True,
            load_point_clouds=True,
            load_segmented_point_clouds=True,
            load_sparse_point_clouds=True,
            box_crop=True,
            box_crop_context=0.4,
            load_frames_from_videos=True,
            image_height=800,
            image_width=800,
            undistort_loaded_blobs=True,
        )
    )
    # query the dataset object to obtain a single video frame of a sequence
    frame_data = dataset[100]
    # obtain the RGB image of the frame
    image_rgb = frame_data.image_rgb
    # obtain the 3D gaussian splats reconstructing the whole scene
    gaussian_splats = frame_data.sequence_gaussian_splats
    # render the scene gaussian splats into the camera of the loaded frame
    # NOTE: This requires the 'gsplat' library. You can install it with:
    #        > pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
    from uco3d import render_splats
    render_colors, render_alphas, render_info = render_splats(
        cameras=frame_data.camera,
        splats=gaussian_splats
        render_size=[512, 512]
    )
    ```

## Examples
The [examples](./examples/) folder contains python scripts with examples using the dataset.

## Tests
The [tests](./tests/) folder runs various tests checking the correctness of the implementation, and also visualizing various loadable modalities, such as point clouds or 3D Gaussian Splats.

To run tests execute the following:
```bash
cd tests
python run.py
```

#### Visualisation tests

[The 3D Gaussian Splat and Pointcloud tests](./tests/test_gaussians_pcl.py) contain many scripts for loading and visualizing all point-cloud and 3D Gaussian Splat data which the dataset contains. Make sure to explore these to familiarize yourself with the dataset interface.


# Dataset format

The dataset is organized in the filesystem as follows:

```
├── metadata.sqlite
├── set_lists
│   ├── set_lists_3categories-debug.sqlite
│   ├── set_lists_all-categories.sqlite
│   ├── set_lists_<subset_lists_name_2>.sqlite
│   ├── ...
├── <super_category_1>
│   ├── <category_1>
│   │   ├── <sequence_name_1>
│   │   │   ├── depth_maps.h5
│   │   │   ├── gaussian_splats
│   │   │   ├── mask_video.mkv
│   │   │   ├── rgb_video.mp4
│   │   │   ├── point_cloud.ply
│   │   │   ├── segmented_point_cloud.ply
│   │   │   └── sparse_point_cloud.ply
│   │   ├── <sequence_name_2>
│   │   │   ├── depth_maps.h5
│   │   │   ├── gaussian_splats
│   │   │   ├── mask_video.mkv
│   │   │   ├── rgb_video.mp4
│   │   │   ├── point_cloud.ply
│   │   │   ├── segmented_point_cloud.ply
│   │   │   └── sparse_point_cloud.ply
│   │   ├── ...
│   │   ├── <sequence_name_S>
│   ├── ...
│   ├── <category_C>
├── ...
├── <super_category_S>
```

#### Data stored as videos
Note that, differently from CO3Dv2, the frame-level data such as images or depth maps is solely released in form of **videos** or **h5 files** to save space. The provided `UCO3DFrameDataBuilder` dataset object then seeks rgb/depth/mask frames from the loaded videos on-the-fly.

## Per-sequence data
Each sequence-specific folder `<super_category>/<category>/<sequence_name>` contains the following files:
- **`rgb_video.mp4`** : The original crowd-sourced video capturing the object from the visual category `<category>` and super-category `<super_category>`.
- **`mask_video.mkv`** : Segmentation video of the same length as `rgb_video.mp4` containing the video-segmentation of the foreground object. The latter was obtained using `LangSAM` in combination with a video segmentation refiner based on `XMem`.
- **`depth_maps.h5`** : `hdf5` file containing a depth map for each of the 200 frames sampled equidistantly from the input video. We first run `DepthAnythingV2` and align the result depth map's scale with the scene sparse point cloud from `sparse_point_cloud.ply`. Hence, the depth maps have a consistent scale within each scene, although they do not achieve strict pixel-wise consistency across multiple views. We are working on improving this and should provide more consistent depth maps in the future.
- **`gaussian_splats`** : 3D Gaussian Splat reconstruction of the scene obtained with the `gsplat` library (v1.3.0). The splats are compressed using the standard `gsplat` compression method which sorts the gaussians using [Self-Organizing Gaussian Grids](https://arxiv.org/pdf/2312.13299) followed by `png` compression.
- **`point_cloud.ply`** : A dense colored 3D pointcloud reconstructing the scene. Obtained using [VGGSfM](https://github.com/facebookresearch/vggsfm).
- **`segmented_point_cloud.ply`** : Same as `point_cloud.ply` but restricted only to points covering the foreground object.
- **`sparse_point_cloud.ply`** : Sparse geometrically-accurate scene pointcloud used to reconstruct the scene cameras. Obtained using [VGGSfM](https://github.com/facebookresearch/vggsfm).


## Metadata database
The `$UCO3D_DATASET_ROOT/metadata.sqlite` file contains a database of all frame-level and video-level metadata such as paths to individual RGB/mask videos, or camera poses for each frame. We opted for an SQL database since it provides fast access times without the need to store all metadata in memory (loading all metadata to memory usually takes minutes to hours for the whole dataset), and is widely supported.

### `PyTorch3D` camera convention

The provided camera annotations follow the [`PyTorch3D` convention](https://pytorch3d.org/docs/cameras) and are represented in the PyTorch3D NDC space. Note that `PyTorch3D` is only an optional dependency which enables extra functionalities and tests within the codebase.

#### Converting uCO3D `Cameras` to `PyTorch3D` `PerspectiveCameras`

Note that, if `PyTorch3D` is installed, the `Cameras` objects loaded using the `UCO3DDataset` object can be converted to the corresponding `PyTorch3D` `PerspectiveCameras` object using the `Cameras.to_pytorch3d_cameras` function.

#### Converting uCO3D `Cameras` to OpenCV cameras

We also provide a conversion to the OpenCV (`cv2`) camera format:
```python
from uco3d import UCO3DDataset, UCO3DFrameDataBuilder
# import the camera conversion function:
from uco3d import opencv_cameras_projection_from_uco3d
# instantiate the dataset
dataset = UCO3DDataset(
    ...
)
# query the dataset object to obtain a single video frame of a sequence
frame_data = dataset[100]
R, tvec, camera_matrix = opencv_cameras_projection_from_uco3d(
    frame_data.camera,
    image_size=frame_data.image_size_hw[None],
)  # R, tvec, camera_matrix follow OpenCV's camera definition
```

### 3D Gaussian Splat convention

uCO3D also contains 3D Gaussian Splat (3DGS) reconstructions in each folder. Here, our Gaussian Splat reconstructions were obtained using [`gsplat`](https://github.com/nerfstudio-project/gsplat) (v1.3.0). `gsplat` is an optional dependency that allows fast rendering of the provided 3DGS reconstructions.

#### Installing `gsplat`

The easiest way to install the supported version of `gsplat` is to use `pip+git`:
```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
```

#### Rendering with `gsplat`

Note that we also provide functions for rendering the loaded splats:
```python
from uco3d import UCO3DDataset, UCO3DFrameDataBuilder
from uco3d import render_splats
# instantiate the dataset
dataset = UCO3DDataset(
    ...
)
# query the dataset object to obtain a single video frame of a sequence
frame_data = dataset[100]
# render the scene gaussian splats into the camera of the loaded frame
render_colors, render_alphas, render_info = render_splats(
    cameras=frame_data.camera,
    splats=frame_data.sequence_gaussian_splats,
    render_size=[512, 512]
)
```

## Subset lists

The subset lists files:
```bash
$UCO3D_DATASET_ROOT/set_lists/set_lists_<SETLIST_NAME>.sqlite
```
definine dataset splits. Specifically, each file contains a list of frames (identified with their `sequence_name` and `frame_number`) in the "train" and "val" subsets of the dataset.

In order to select a specific subset of the dataset, one passes the correct subset list path, and the subset name to the constructore of `UCO3DDataset`.

For instance
```python
dataset = UCO3DDataset(
    subset_lists_file="<UCO3D_DATASET_ROOT>/set_lists/set_lists_all-categories.sqlite",
    subsets=["train"],
    frame_data_builder=...,
)
```
will load the "train" subset of the `set_lists_all-categories.sqlite` subset list which contains the whole uCO3D dataset.

### Available subset lists

The folder `<UCO3D_DATASET_ROOT>/set_lists/` provides the following subset lists:
- **set_lists_3categories-debug.sqlite** - A small split for debugging purposes. Very fast to load allowing fast iteration on the code using the dataset.
- **set_lists_all-categories.sqlite** - Contains the whole dataset.
- **set_lists_static-categories.sqlite** - Contains the videos of all rigid categories of uCO3D.
- **set_lists_static-categories-accurate-reconstruction.sqlite** - Contains the videos of all rigid categories of uCO3D with high-quality reconstructions.
- **set_lists_dynamic-categories.sqlite** - Contains the videos of all flexible categories (e.g. animals) of uCO3D.

### Creating custom subset lists

Subset lists are stored as `sqlite` tables. The easiest way is to use `pandas` to select a subset of the main `<UCO3D_DATASET_ROOT>/metadata.sqlite` table. The following example makes a subset list from or 100 and 30 training and validation samples respectively, by taking the first 130 sequences from the metadata:
```python
import pandas as pd, sqlite3, os

# read the main metadata table (takes long time)
metadata_file = os.path.join(UCO3D_DATASET_ROOT, "metadata.sqlite")
frame_annots = pd.read_sql_table("frame_annots", f"sqlite:///{metadata_file}")

# pick dataset sequence names by taking unique sequence names from frame annotations
seqs = frame_annots["sequence_name"].unique()

# choose the list of train/val sequences
train_seqs, val_seqs = downloaded_seqs[:100], downloaded_seqs[100:130]

# training setlist
setlists_train = frame_annots[frame_annots.isin(train_seqs)][["frame_number","sequence_name"]]
setlists_train["subset"] = "train"
# validation setlist
setlists_val = frame_annots[frame_annots.isin(val_seqs)][["frame_number","sequence_name"]]
setlists_val["subset"] = "val"
# concatenate train and val
setlists = pd.concat([setlists_train, setlists_val])

# the new setlists will be stored as the set_lists_130.sqlite file in the original root
setlist_file = os.path.join(UCO3D_DATASET_ROOT, "set_lists", "set_lists_130.sqlite")
# store the new table
with sqlite3.connect(setlist_file) as con:
    setlists.to_sql("set_lists", con, if_exists='replace', index=False)
```

# License

The data are released under the [CC BY 4.0 license](LICENSE).

# Third-Party Code

This project uses code from other sources, which are licensed under their respective licenses:
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [gsplat-convert](https://github.com/Ending2015a/gsplat-convert)

# Reference

If you use our dataset, please use the following citation:
```bibtex
@inproceedings{liu24uco3d,
	Author = {Liu, Xingchen and Tayal, Piyush and Wang, Jianyuan and Zarzar, Jesus and Monnier, Tom and Tertikas, Konstantinos and Duan, Jiali and Toisoul, Antoine and Zhang, Jason Y. and Neverova, Natalia and Vedaldi, Andrea and Shapovalov, Roman and Novotny, David},
	Booktitle = {arXiv},
	Title = {UnCommon Objects in 3D},
	Year = {2025},
}
```
