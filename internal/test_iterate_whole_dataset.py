import argparse
import functools
import getpass
import multiprocessing as mp
import os
import sys
import traceback
import json
import glob
import sqlite3
import os


import torch
from typing import List
from tqdm import tqdm
from uco3d.data_utils import get_all_load_dataset


# To resolve memory leaks giving received 0 items from anecdata
# Reference link https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")


def iterate_dataset_worker(
    rank: int,
    world_size: int,
    log_dir: str,
    dataset_root: str,
    num_workers: int,
    num_frames_per_batch: int = 16,
    specific_dataset_idx: List[int] = None,
):
    
    if specific_dataset_idx is not None:
        assert world_size <= 1
        assert rank==0
    
    print("loading dataset ...")
    dataset = get_all_load_dataset(
        frame_data_builder_kwargs=dict(
            dataset_root=dataset_root,
        ),
        dataset_kwargs=dict(
            # subset_lists_file=None,
            # subsets=None,
            subset_lists_file=os.path.join(
                dataset_root,
                "set_lists",
                "set_lists_all-categories.sqlite",
                # "set_lists_3categories-debug.sqlite",
            ),
            subsets=["train", "val"],
        )  # this will load the whole dataset without any setlists
    )
    print("done loading dataset.")
    assert not dataset.is_filtered(), "Dataset is filtered, this script is for full dataset only"
    all_idx = torch.arange(len(dataset))
    idx_chunk = torch.chunk(all_idx, world_size)
    idx_this_worker = idx_chunk[rank]
    
    if True:
        
        # dataset = torch.utils.data.Subset(dataset, idx_this_worker)
        if specific_dataset_idx is not None:
            batch_sampler = [specific_dataset_idx]
        else:
            batch_sampler = [
                b.tolist() for b in torch.split(
                    idx_this_worker,
                    num_frames_per_batch,
                )
            ]
            
        # resume from checkpoint if needed
        batch_idx_start = _load_worker_checkpoint(log_dir, rank)
        if batch_idx_start is None:
            batch_idx_start = 0    
        batch_nums = list(range(len(batch_sampler)))[batch_idx_start:]
        batch_sampler = batch_sampler[batch_idx_start:]
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=lambda x: x,
            shuffle=False,
            batch_sampler=batch_sampler,
        )
        
        assert len(batch_nums) == len(batch_sampler)
        dataloader_iter = iter(dataloader)
        for iter_idx, batch_idx in tqdm(
            enumerate(batch_nums),
            desc=f"worker {rank+1} / {world_size}",
            total=len(batch_nums),
        ):
            # _ = next(dataloader_iter)
            batch_indices = batch_sampler[iter_idx]
            try:
                _ = next(dataloader_iter)
            except Exception as e:
                exc_file = os.path.join(log_dir, f"{batch_indices[0]}_exc.txt")
                print(f"Fail for idx {batch_idx}")
                print(traceback.format_exc())
                with open(exc_file, "w") as f:
                    f.write(traceback.format_exc())
                
                # get sequence names and paths to the problematic scenes
                try:
                    batch_meta = [
                        dataset.meta[batch_index] for batch_index in batch_indices
                    ]
                    batch_sequences = [
                        (
                            bm.sequence_super_category,
                            bm.sequence_category,
                            bm.sequence_name,
                        ) for bm in batch_meta
                    ]
                except Exception as e:
                    print(f"Failed to get batch_sequences for batch {batch_idx}")
                    print(traceback.format_exc())
                    batch_sequences = None
                batch_file = os.path.join(log_dir, f"{batch_indices[0]}_batch.json")
                with open(batch_file, "w") as f:
                    json.dump({
                        "batch_indices": batch_indices,
                        "batch_sequence_names": batch_sequences,
                    }, f)
                    
            if iter_idx % 100 == 0 and iter_idx > 0:
                _store_worker_checkpoint(log_dir, rank, batch_idx)
    else:
        idx_this_worker = idx_this_worker.tolist()
        for idx in tqdm(idx_this_worker, desc=f"worker {rank} / {world_size}"):
            try:
                _ = dataset[idx]
            except Exception as e:
                exc_file = os.path.join(log_dir, f"{idx:015d}.txt")
                print(f"Fail for idx {idx}")
                print(traceback.format_exc())
                with open(exc_file, "w") as f:
                    f.write(traceback.format_exc())


def _analyze_logs(log_dir):
    exc_files = sorted(glob.glob(os.path.join(log_dir, "*_exc.txt")))
    bad_gauss_splats = []
    missing_segmented_pcls = []
    
    for exc_file in tqdm(exc_files):
        with open(exc_file, "r") as f:
            exc_string = f.read()
        batch_file = exc_file.replace("_exc.txt", "_batch.json")
        with open(batch_file, "r") as f:
            batch_indices = json.load(f)
        
        exc_lines = exc_string.split()
        
        if exc_lines[-1].endswith("/gaussian_splats/meta.json'"):
            start = exc_lines[-1].rfind("/fsx-repligen/shared/")
            splats_folder = exc_lines[-1][start:-1]
            bad_gauss_splats.append(splats_folder)
            continue
          
        if exc_lines[-4].endswith("segmented_point_cloud.ply"):
            missing_segmented_pcls.append(exc_lines[-4])
            continue
          
        # print("\n\n\n\n----------\n\n\n\n")
        # print(exc_string)
        # print(batch_indices)
        # import pdb; pdb.set_trace()
        # pass

    for missing_segmented_pcl in missing_segmented_pcls:
        print(missing_segmented_pcl)

    # bad_gauss_splats = sorted(list(set(bad_gauss_splats)))
    # for splats_folder in bad_gauss_splats:
    #     print(splats_folder)
    # import pdb; pdb.set_trace()


def _get_worker_checkpoint_file(log_dir, rank):
    return os.path.join(log_dir, f"worker_{rank}_checkpoint.txt")


def _store_worker_checkpoint(log_dir, rank, batch_idx):
    checkpoint_file = _get_worker_checkpoint_file(log_dir, rank)
    print(f"Storing checkpoint for worker {rank} at batch {batch_idx}: {checkpoint_file}")
    with open(checkpoint_file, "w") as f:
        f.write(f"{batch_idx}")    


def _load_worker_checkpoint(log_dir, rank):
    checkpoint_file = _get_worker_checkpoint_file(log_dir, rank)
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            batch_idx = int(f.read())
        print(f"Loading checkpoint for worker {rank} at batch {batch_idx}: {checkpoint_file}")
        return batch_idx
    else:
        return None


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--dataset_root", type=str, required=True)
    argparse.add_argument("--log_dir", type=str, required=True)
    argparse.add_argument("--world_size", type=int, default=4)
    argparse.add_argument("--num_workers", type=int, default=4)
    argparse.add_argument("--run_locally", action="store_true")
    argparse.add_argument("--analyze_logs", action="store_true")
    args = argparse.parse_args()

    if bool(args.analyze_logs):    
        _analyze_logs(str(args.log_dir))
        sys.exit(0)

    os.makedirs(str(args.log_dir), exist_ok=True)

    # broken depth h5:
    # [10577174, 10577175, 10577176, 10577177, 10577178, 10577179, 10577180, 10577181, 10577182, 10577183, 10577184, 10577185, 10577186, 10577187, 10577188, 10577189]

    if bool(args.run_locally):
        
        world_size = int(args.world_size)
        if world_size <= 0:
            iterate_dataset_worker(
                0,
                1,
                log_dir=str(args.log_dir),
                dataset_root=str(args.dataset_root),
                num_workers=int(args.num_workers),
                # specific_dataset_idx=[25894583, 25894584, 25894585, 25894586, 25894587, 25894588, 25894589, 25894590, 25894591, 25894592, 25894593, 25894594, 25894595, 25894596, 25894597, 25894598],
                # specific_dataset_idx=[10576774, 10576775, 10576776, 10576777, 10576778, 10576779, 10576780, 10576781, 10576782, 10576783, 10576784, 10576785, 10576786, 10576787, 10576788, 10576789],
                # specific_dataset_idx=[10577174, 10577175, 10577176, 10577177, 10577178, 10577179, 10577180, 10577181, 10577182, 10577183, 10577184, 10577185, 10577186, 10577187, 10577188, 10577189],  # weird depth
                # specific_dataset_idx=[13548035, 13548036, 13548037, 13548038, 13548039, 13548040, 13548041, 13548042, 13548043, 13548044, 13548045, 13548046, 13548047, 13548048, 13548049, 13548050],  # bad crc
                # specific_dataset_idx=[16606624, 16606625, 16606626, 16606627, 16606628, 16606629, 16606630, 16606631, 16606632, 16606633, 16606634, 16606635, 16606636, 16606637, 16606638, 16606639],  # bad crc
                # specific_dataset_idx=[17683231, 17683232, 17683233, 17683234, 17683235, 17683236, 17683237, 17683238, 17683239, 17683240, 17683241, 17683242, 17683243, 17683244, 17683245, 17683246],  # bad crc
                # specific_dataset_idx=[22040411, 22040412, 22040413, 22040414, 22040415, 22040416, 22040417, 22040418, 22040419, 22040420, 22040421, 22040422, 22040423, 22040424, 22040425, 22040426],  # bad crc
                # [24957512, 24957513, 24957514, 24957515, 24957516, 24957517, 24957518, 24957519, 24957520, 24957521, 24957522, 24957523, 24957524, 24957525, 24957526, 24957527]  # cannot get image from dataloader
                # [26304631, 26304632, 26304633, 26304634, 26304635, 26304636, 26304637, 26304638, 26304639, 26304640, 26304641, 26304642, 26304643, 26304644, 26304645, 26304646]  # bad gaussian shape
            )
        else:
            with mp.get_context("spawn").Pool(processes=world_size) as pool:
                worker = functools.partial(
                    iterate_dataset_worker,
                    world_size=world_size,
                    log_dir=str(args.log_dir),
                    dataset_root=str(args.dataset_root),
                    num_workers=int(args.num_workers),
                )
                pool.map(worker, list(range(world_size)))

    else:
        from griddle.submitit_jobs import submitit_jobs
        
        username = getpass.getuser()
        user_slurm_log_dir = f"/fsx-repligen/{username}/slurm_jobs_uco3d/"
        os.makedirs(user_slurm_log_dir, exist_ok=True)
        root_job_name = "iterate_uco3d"
        debug = False

        kwargs_list = [
            {
                "rank": i,
                "world_size": int(args.world_size),
                "log_dir": str(args.log_dir),
                "dataset_root": str(args.dataset_root),
                "num_workers": int(args.num_workers),
            }
            for i in range(int(args.world_size))
        ]

        submitit_jobs(
            iterate_dataset_worker,
            kwargs_list,
            root_job_name=root_job_name,
            slurm_dir=user_slurm_log_dir,
            slurm_gpus_per_task=2,
            slurm_cpus_per_gpu=int(args.num_workers)+1,
            slurm_ntasks_per_node=1,
            nodes=1,
            mem_per_cpu=16,
            slurm_time=3600,
            slurm_partition="learn",
            slurm_account="repligen",
            slurm_qos="low",
            debug=debug,
            disable_job_state_monitor=False,
            slurm_array_parallelism=32,
        )




# RUNS:
# python ./test_iterate_whole_dataset.py --run_locally --world_size 0 --log_dir="$HOME/data/uco3d_iterate_log_241213/" --dataset_root="$HOME/data//"
# python ./test_iterate_whole_dataset.py --run_locally --world_size 0 --log_dir="/fsx-repligen/dnovotny/datasets/uCO3D/uco3d_iterate_log_241213_debug/" --dataset_root="/fsx-repligen/shared/datasets/uCO3D/dataset_export/" --num_workers 4
# python ./test_iterate_whole_dataset.py --world_size 32 --log_dir="/fsx-repligen/dnovotny/datasets/uCO3D/uco3d_iterate_log_241213_2/" --dataset_root="/fsx-repligen/shared/datasets/uCO3D/dataset_export/" --num_workers 16
# python ./test_iterate_whole_dataset.py --world_size 32 --log_dir="/fsx-repligen/dnovotny/datasets/uCO3D/uco3d_iterate_log_241215/" --dataset_root="/fsx-repligen/shared/datasets/uCO3D/dataset_export/" --num_workers 16
# python ./test_iterate_whole_dataset.py --world_size 0 --log_dir="/fsx-repligen/dnovotny/datasets/uCO3D/uco3d_iterate_log_241215_debug/" --dataset_root="/fsx-repligen/shared/datasets/uCO3D/dataset_export/" --num_workers 16 --run_locally



# ... 33144799 iterations in total ~ 300 days @ 0.75 sec per iteration
# ... => submit 500 jobs
# python ./test_iterate_whole_dataset.py --world_size 500 --log_dir="/fsx-repligen/dnovotny/datasets/uCO3D/uco3d_iterate_log_241213/" --dataset_root="/fsx-repligen/shared/datasets/uCO3D/dataset_export/"
