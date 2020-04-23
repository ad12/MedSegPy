"""Benchmark reading data from h5py.

Determine if it is faster to slice data while loading or 


@author: Arjun Desai, arjundd@stanford.edu
"""
import datetime
import functools
import logging
import os
import random
import time
from typing import Tuple

import h5py
import numpy as np
import pandas as pd

from fvcore.common.file_io import PathManager

def make_block(
    block_size: Tuple[int, ...], 
    total_size: Tuple[int, ...], 
    rand: random.Random = None,
) -> Tuple[slice, ...]:
    if not rand:
        rand = random.Random()
    slices = []
    for idx, (b_len, tot_len) in enumerate(zip(block_size, total_size)):
        if b_len == tot_len:
            slices.append(slice(0,b_len))
            continue
        start = rand.randint(0, tot_len - b_len - 1)
        slices.append(slice(start, start + b_len))
    return tuple(slices)


def slice_during_load(file_name: str, window: Tuple[slice, ...]) -> float:
    start = time.perf_counter()
    with h5py.File(file_name, "r") as f:
        data = f["volume"][window]

    time_elapsed = time.perf_counter() - start
    return time_elapsed


def slice_after_load(file_name: str, window: Tuple[slice, ...]) -> float:
    start = time.perf_counter()
    with h5py.File(file_name, "r") as f:
        data = f["volume"][:]
    data = data[window]
    time_elapsed = time.perf_counter() - start
    return time_elapsed


log_file = "./{}.log".format(os.path.splitext(os.path.basename(__file__))[0])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
fh.setFormatter(plain_formatter)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(plain_formatter)
logger.addHandler(fh)
logger.addHandler(sh)


dirpaths = ["h5_files_3d/train", "h5_files_3d/val", "h5_files_3d/test"]
filenames = [x for dp in dirpaths for x in os.listdir(dp)]
filepaths = [os.path.join(dp, x) for dp in dirpaths for x in os.listdir(dp)]

total_size = (384, 384, 160)
block_sizes = (
    (10, 10, 10), 
    (50, 50, 50), 
    (100, 100, 100), 
    (200, 200, 160), 
    (384, 384, 1), 
    (384, 384, 50),
    (384, 384, 100),
    (384, 384, 160),
)
seeds = (7001, 8001, 9001)

count = 0
total_exps = len(seeds) * len(block_sizes)
for seed in seeds:
    rand = random.Random(seed)
    for block_size in block_sizes:
        params = {"seed": seed, "block_size": block_size, "total_size": total_size}
        logger.info("=========== Experiment {}/{} ===========".format(count, total_exps))
        logger.info("Params: " + "\t".join(["{}: {}".format(k, v) for k, v in params.items()]))
        rt_h5 = []
        rt_np = []
        exp_start_time = time.perf_counter()
        for f_idx, file_name in enumerate(filepaths):
            block = make_block(block_size, total_size, rand)

            # Randomly pick which method to do first to avoid minimize cache differences.
            if rand.random() > 0.5:
                rt_h5.append(slice_during_load(file_name, block))
                rt_np.append(slice_after_load(file_name, block))
            else:
                rt_np.append(slice_after_load(file_name, block))
                rt_h5.append(slice_during_load(file_name, block))

            total_seconds_per_img = (time.perf_counter() - exp_start_time) / (f_idx + 1)
            eta = datetime.timedelta(
                seconds=int(total_seconds_per_img * (len(filepaths) - f_idx - 1))
            )
            logger.info("{}/{} ({}) h5: {:0.4f}s - numpy:{:0.4f}s - ETA: {}".format(
                f_idx, len(filepaths), filenames[f_idx], rt_h5[-1], rt_np[-1], str(eta),
            ))

        # Write dataframe.
        df = pd.DataFrame([filenames, rt_h5, rt_np], index=["Files", "Method 1 (H5)", "Method 2 (Numpy)"]).T
        for k, v in params.items():
            df[k] = [v]*len(filenames)

        # Write experiment to csv file.
        if count == 0:
            use_header = True
            mode = "w"
        else:
            use_header = False
            mode = "a"
        df.to_csv("./benchmark_h5py.csv", mode=mode, header=use_header)
        count += 1


