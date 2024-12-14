import argparse
import functools
import getpass
import multiprocessing as mp
import os
import sys
import traceback
import json

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
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=lambda x: x,
            shuffle=False,
            batch_sampler=batch_sampler,
        )
        
        batch_nums = list(range(len(dataloader)))
        assert len(batch_nums) == len(batch_sampler)
        dataloader_iter = iter(dataloader)
        for batch_idx in tqdm(batch_nums, desc=f"worker {rank+1} / {world_size}"):
            # _ = next(dataloader_iter)
            try:
                _ = next(dataloader_iter)
            except Exception as e:
                batch_indices = batch_sampler[batch_idx]
                exc_file = os.path.join(log_dir, f"{batch_indices[0]}_exc.txt")
                print(f"Fail for idx {batch_idx}")
                print(traceback.format_exc())
                with open(exc_file, "w") as f:
                    f.write(traceback.format_exc())
                batch_file = os.path.join(log_dir, f"{batch_indices[0]}_batch.json")
                with open(batch_file, "w") as f:
                    json.dump(batch_indices, f)
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


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--dataset_root", type=str, required=True)
    argparse.add_argument("--log_dir", type=str, required=True)
    argparse.add_argument("--world_size", type=int, default=4)
    argparse.add_argument("--num_workers", type=int, default=4)
    argparse.add_argument("--run_locally", action="store_true")
    args = argparse.parse_args()

    os.makedirs(str(args.log_dir), exist_ok=True)

    if bool(args.run_locally):
        world_size = int(args.world_size)
        if world_size <= 0:
            iterate_dataset_worker(
                0,
                1,
                log_dir=str(args.log_dir),
                dataset_root=str(args.dataset_root),
                num_workers=int(args.num_workers),
                specific_dataset_idx=[25894583, 25894584, 25894585, 25894586, 25894587, 25894588, 25894589, 25894590, 25894591, 25894592, 25894593, 25894594, 25894595, 25894596, 25894597, 25894598],
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



# ... 33144799 iterations in total ~ 300 days @ 0.75 sec per iteration
# ... => submit 500 jobs
# python ./test_iterate_whole_dataset.py --world_size 500 --log_dir="/fsx-repligen/dnovotny/datasets/uCO3D/uco3d_iterate_log_241213/" --dataset_root="/fsx-repligen/shared/datasets/uCO3D/dataset_export/"
