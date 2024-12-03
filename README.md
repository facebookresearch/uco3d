# 🔁 uCO3D Dataset

uCO3D is a dataset of around 170,000 turn-tabe videos capturing objects from 
the LVIS taxonomy of object categories.

This repository contains download scripts and classes to access the data.

## Download

The full dataset (processed version) takes **XXX GB of space**. We distribute it in chunks up to 20 GB.
The links to all dataset files are present in this repository 
in [dataset_download/links/links.json](links/links.json).
We provide an automated way of downloading and decompressing the data.

First, run the install script that will take care of dependencies:

```bash
pip install -e .
```

Then run the script (make sure to change `<DESTINATION_FOLDER>`):

```bash
python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --checksum_check
```

### Download subsets of the dataset

#### Downloading specific modalities

Setting `--download_modalities` to a comma-separated list of specific dataset modalities will download only a subset of available modalities.
For instance
```bash
python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --download_modalities "rgb_videos,point_clouds"
```
will only download rgb videos and point clouds.

#### Downloading specific categories or super-categories

Setting `--download_super_categories` and `--download_categories` will instruct the script to download only a subset of the available categories.
For instance
```bash
python dataset_download/download_dataset.py --download_folder <DESTINATION_FOLDER> --download_super_categories "vegetables_and_legumes,stationery" --download_categories "kettle"
```
will download only the vegetables&legumes and stationery super-categories, together with the kettle category. Note that if a super-category is selected for download, it will download all of its categories regardless of the `--download_categories` switch.

Note that `--download_modalities` can be mixed with `--download_categories` and  `--download_super_categories` to enable choosing any possible subset of the dataset.

Run `python dataset_download/download_dataset.py -h` for the full list of options.


## API Quick Start and Tutorials

Make sure the setup is done and the dataset is downloaded as per above.

We provide `ReplayDataset` class to access the data and are working on the tutorials on using it with Implicitron.
For now, please check the [unit test](tests/test_replay_dataset.py).

## License

The data are released under the [CC BY 4.0 license](LICENSE).