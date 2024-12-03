import torch
import pickle
from tqdm import tqdm

# To resolve memory leaks giving received 0 items from anecdata
# Reference link https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
import sys

sys.path.append("/fsx-repligen/piyusht1/projects/uCO3D/uco3d/")
# sys.path.append("../../uco3d/")
from uco3d.uco3d_dataset import UCO3DDataset, UCO3DFrameDataBuilder
from uco3d.dataset_utils.scene_batch_sampler import (
    SceneBatchSampler,
)

DATASET_ROOT = "/fsx-repligen/shared/datasets/uCO3D/batch_reconstruction/dataset_export"
# METADATA_FILE = os.path.join(DATASET_ROOT, "metadata_1766.sqlite")
METADATA_FILE = "/fsx-repligen/shared/datasets/uCO3D/tmp_romansh_export/metadata_1729602322.8402915.sqlite"
NO_BLOBS_KWARGS = {
    "load_images": False,
    "load_depths": False,
    "load_masks": False,
    "load_depth_masks": False,
    "box_crop": False,
    "image_height": 800,
}
SCENE_BATCH_SAMPLER_KWARGS = {
    "image_height": 800,
    "image_width": 800,
    "load_depths": False,
    # "load_frames_from_videos": False,
}
# frame_data_builder_args = NO_BLOBS_KWARGS
frame_data_builder_args = SCENE_BATCH_SAMPLER_KWARGS

frame_data_builder = UCO3DFrameDataBuilder(
    # dataset_root="/fsx-repligen/piyusht1/projects/uCO3D_SfM/export_reconstruction_2000/dataset_export",
    dataset_root=DATASET_ROOT,
    **frame_data_builder_args,
)

dataset = UCO3DDataset(
    # dataset_root="/fsx-repligen/piyusht1/projects/uCO3D_SfM/export_reconstruction_2000/dataset_export",
    dataset_root="/fsx-repligen/shared/datasets/uCO3D/batch_reconstruction/dataset_export",
    sqlite_metadata_file=METADATA_FILE,
    remove_empty_masks=False,
    frame_data_builder=frame_data_builder,
)

images_per_seq_options = [8]
scene_batch_sampler = SceneBatchSampler(
    dataset=dataset,
    batch_size=48,
    num_batches=1,
    images_per_seq_options=images_per_seq_options,
)


def my_collate_fn(batch):
    return [batch]


num_workers = 0
dataloader = DataLoader(
    dataset,
    batch_sampler=scene_batch_sampler,
    num_workers=num_workers,
    collate_fn=dataset.frame_data_type.collate,
)

## Example to get the batches
for i, data in enumerate(dataloader):
    print(f"Batch Number {i}")

sequences = dataset.sequence_names()
seq = sequences[1343]
print("Number of sequences:", len(sequences))
print("Sequence name:", seq)

# sequence_frames_in_order returns an iterator over (timestamp, frame_id, global_index)
# frame_id is unique within a sequence; global_index is unique within the dataset
# to get length, we convert it to a list
seq_frame_ids = list(dataset.sequence_frames_in_order(seq))
print("Number of frames in the sequence:", len(seq_frame_ids))
ts, frame_id, idx = seq_frame_ids[42]

import multiprocessing, time


def process_item(i_seq):
    i, seq = i_seq[0], i_seq[1]
    print("Starting", i)
    frame = dataset[seq, list(dataset.sequence_frames_in_order(seq))[-1][1]]
    print("Done with ", i)


def worker(queue):
    while True:
        task = queue.get()
        if task is None:
            break
        # Process the task
        process_item(task)
        # queue.put(result)


if __name__ == "__main__":
    all_sequences_pickle_file = (
        "/fsx-repligen/shared/datasets/uCO3D/export_pickles/all_videos_119k.pickle"
    )
    with open(all_sequences_pickle_file, "wb") as f:
        pickle.dump(sequences, f)
    for seq in tqdm(sequences):
        frame = dataset[seq, list(dataset.sequence_frames_in_order(seq))[-1][1]]
