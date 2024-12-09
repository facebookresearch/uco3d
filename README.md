<center>
<img src="./uco3d_logo.png" width="400" />
</center>

# uCO3D Dataset

uCO3D is a dataset of around 170,000 turn-tabe videos capturing objects from
the LVIS taxonomy of object categories.

This repository contains download scripts and classes to access the data.

```
       ===> TODO: Dataset GIF <===
```

## Download & Install

The full dataset (processed version) takes **XXX GB of space**. We distribute it in chunks up to 20 GB.
We provide an automated way of downloading and decompressing the data.

First, run the install script that will take care of dependencies:

```bash
git clone git@github.com:facebookresearch/uco3d.git
cd uco3d
pip install -e .
```

Then run the download script (make sure to change `<DESTINATION_FOLDER>`):

```bash
python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --checksum_check
```

### Download subsets of the dataset

#### Downloading specific modalities

Setting `--download_modalities` to a comma-separated list of specific modality names will download only a subset of available modalities.
For instance
```bash
python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --download_modalities "rgb_videos,point_clouds"
```
will only download rgb videos and point clouds.
Execute `python dataset_download/download_dataset.py -h` for the list of all downloadable modalities.

#### Downloading specific categories or super-categories

Setting `--download_super_categories` and `--download_categories` will instruct the script to download only a subset of the available categories.
For instance
```bash
python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --download_super_categories "vegetables_and_legumes,stationery" --download_categories "kettle"
```
will download only the vegetables&legumes and stationery super-categories, together with the kettle category. Note that if a super-category is selected for download, it will download all of its categories regardless of the `--download_categories` content.

Note that `--download_modalities` can be mixed with `--download_categories` and  `--download_super_categories` to enable choosing any possible subset of the dataset.

Run `python dataset_download/download_dataset.py -h` for the full list of options.


## API Quick Start and Examples

1) [Download the dataset and install the package](#download--install)

2) Setup the dataset root environment var
```bash
export UCO3D_DATASET_ROOT=<DESTINATION_FOLDER>
```
pointing to the root folder with the uCO3D dataset.

3) Create the dataset object and fetch its data:
```python
from uco3d import UCO3DDataset, UCO3DFrameDataBuilder
# Get the dataset root folder and check that
# all required metadata files exist.
dataset_root = get_dataset_root(assert_exists=True)
# Get the "small" subset list containing a small subset
# of the uCO3D categories. For loading the whole dataset
# use "set_lists_all.sqlite".
subset_lists_file = os.path.join(dataset_root, "set_lists_small.sqlite")
dataset = UCO3DDataset(
    subset_lists_file=subset_lists_file,
    subsets=["train"],
    frame_data_builder=UCO3DFrameDataBuilder(
        apply_alignment=True,
        load_images=True,
        load_depths=True,
        load_masks=True,
        load_depth_masks=True,
        load_gaussian_splats=True,
        gaussian_splats_truncate_background=True,
        load_point_clouds=True,
        load_segmented_point_clouds=True,
        load_sparse_point_clouds=True,
        box_crop=True,
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
# NOTE: This requires the 'gsplat' library. You can install
#       it with:
#        > pip install git+https://github.com/nerfstudio-project/gsplat.git
from uco3d import render_splats
render_colors, render_alphas, render_info = render_splats(
    cameras=frame_data.camera,
    splats=gaussian_splats
    render_size=[512, 512]
)
```

### Examples
The [examples](./examples/) folder contains python scripts with examples using the dataset.

### Tests
The [tests](./tests/) folder runs various tests checking the correctness of the implementation, and also visualizing various loadable modalities, such as point clouds or 3D Gaussian Splats.

Specifically, [the 3D Gaussian Splat and Pointcloud tests](./tests/test_gaussians_pcl.py) contain many scripts for loading and visualizing all point-cloud and 3D Gaussian Splat data which the dataset contains.

## License

The data are released under the [CC BY 4.0 license](LICENSE).

## Dataset format

The dataset is organized in the filesystem as follows:



## Reference
