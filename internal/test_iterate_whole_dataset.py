import argparse
import functools
import getpass
import multiprocessing as mp
import os
import sys
import traceback

import torch
from tqdm import tqdm
from uco3d.data_utils import get_all_load_dataset


def iterate_dataset_worker(
    rank: int,
    world_size: int,
    log_dir: str,
):
    dataset = get_all_load_dataset(
        dataset_kwargs=dict(
            subset_lists_file=None,
            subsets=None,
        )  # this will load the whole dataset without any setlists
    )

    all_idx = torch.arange(len(dataset))
    idx_chunk = torch.chunk(all_idx, world_size)
    idx_this_worker = idx_chunk[rank].tolist()
    for idx in tqdm(idx_this_worker, desc=f"worker {rank} / {world_size}"):
        try:
            _ = dataset[idx]
            if idx % 100 == 0:
                1 / 0
        except Exception as e:
            # exc_type, exc_value, exc_traceback = sys.exc_info()
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
    argparse.add_argument("--run_locally", action="store_true")
    args = argparse.parse_args()

    os.makedirs(str(args.log_dir), exist_ok=True)

    os.environ["UCO3D_DATASET_ROOT"] = str(args.dataset_root)

    if bool(args.run_locally):
        world_size = int(args.world_size)
        if world_size <= 0:
            iterate_dataset_worker(0, 1, log_dir=str(args.log_dir))
        else:
            with mp.get_context("spawn").Pool(processes=world_size) as pool:
                worker = functools.partial(
                    iterate_dataset_worker,
                    world_size=world_size,
                    log_dir=str(args.log_dir),
                )
                pool.map(worker, list(range(world_size)))

    else:
        from griddle import submitit_jobs

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
            }
            for i in range(int(args.world_size))
        ]

        submitit_jobs(
            iterate_dataset_worker,
            kwargs_list,
            root_job_name=root_job_name,
            slurm_dir=user_slurm_log_dir,
            slurm_gpus_per_task=0,
            slurm_cpus_per_gpu=2,
            slurm_ntasks_per_node=1,
            nodes=1,
            slurm_time=5000,
            slurm_partition="learn",
            slurm_account="repligen",
            slurm_qos="repligen",
            debug=debug,
            disable_job_state_monitor=False,
        )


# RUNS:
# python ./test_iterate_whole_dataset.py --run_locally --world_size 0 --log_dir="$HOME/data/uco3d_iterate_log/" --dataset_root="$HOME/data/uco3d_sample/"
