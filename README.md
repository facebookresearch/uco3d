# 🔁 uCO3D Dataset

uCO3D is a dataset of around 186000 videos of diverse objects.
Each scene is about 5 minutes long and filmed with 12 cameras, static and dynamic.

This repository contains download scripts and classes to access the data.
Examples of usage and format description are coming soon.

## Download

The full dataset (processed version) takes **244 GB of space**. We distribute it in chunks up to 20 GB.
The links to all dataset files are present in this repository in [links/links.json](links/links.json).
We provide an automated way of downloading and decompressing the data.

First, run the install script that will take care of dependencies:

```
pip install -e .
```

Then run the script (make sure to change `<DESTINATION_FOLDER>`):

```
replay_dataset/download_dataset.py --download_folder <DESTINATION_FOLDER> --checksum_check
```

The script has multiple parameters, e.g. `--download_categories audio,videos,masks` will download all modalities (the default behaviour).
You can select only a subset of those, e.g. you can skip `audio` files.
Metadata will be always downloaded.
Another flag, `--clear_archives_after_unpacking`, will remove the redundant archives.
Run `replay_dataset/download_dataset.py -h` for the full list of options.


## API Quick Start and Tutorials

Make sure the setup is done and the dataset is downloaded as per above.

We provide `ReplayDataset` class to access the data and are working on the tutorials on using it with Implicitron.
For now, please check the [unit test](tests/test_replay_dataset.py).

## Dataset Format

TBD

## Reference

TBD

## License

The data are released under the [CC BY-NC 4.0 license](LICENSE).

