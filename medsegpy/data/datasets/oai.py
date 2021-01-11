"""OAI iMorphics Dataset"""
import logging
import os
import re

import h5py

from medsegpy.data.catalog import DatasetCatalog, MetadataCatalog
from medsegpy.utils.cluster import Cluster

logger = logging.getLogger(__name__)

_DATA_CATALOG = {}
_TEST_SET_METADATA_PIK = ""

_DATA_CATALOG = {
    "oai_2d_train": "oai_data/h5_files_2d/train",
    "oai_2d_val": "oai_data/h5_files_2d/valid",
    "oai_2d_test": "oai_data/h5_files_2d/test",
    "oai_2d_whitened_train": "oai_data/h5_files_whitened_2d/train",
    "oai_2d_whitened_val": "oai_data/h5_files_whitened_2d/valid",
    "oai_2d_whitened_test": "oai_data/h5_files_whitened_2d/test",
    "oai_3d_train": "oai_data/h5_files_3d/train",
    "oai_3d_val": "oai_data/h5_files_3d/val",
    "oai_3d_test": "oai_data/h5_files_3d/test",
    "oai_3d_whitened_train": "oai_data/h5_files_whitened_3d/train",
    "oai_3d_whitened_val": "oai_data/h5_files_whitened_3d/val",
    "oai_3d_whitened_test": "oai_data/h5_files_whitened_3d/test",
    "oai_3d_sf_whitened_train": "oai_data/h5_files_whitened_3d/train.h5",
    "oai_3d_sf_whitened_val": "oai_data/h5_files_whitened_3d/val.h5",
    "oai_3d_sf_whitened_test": "oai_data/h5_files_whitened_3d/test.h5",
}


OAI_CATEGORIES = [
    {"color": [220, 20, 60], "id": 0, "name": "femoral cartilage", "abbrev": "fc"},
    {"color": [119, 11, 32], "id": (1, 2), "name": "tibial cartilage", "abbrev": "tc"},
    {"color": [0, 0, 142], "id": 3, "name": "patellar cartilage", "abbrev": "pc"},
    {"color": [0, 0, 230], "id": (4, 5), "name": "meniscus", "abbrev": "men"},
]


def load_oai_2d_from_dir(scan_root, dataset_name=None):
    # sample scan name: "9311328_V01-Aug04_072.im"
    # format: %7d_V%02d-Aug%02d_%03d
    FNAME_REGEX = "([\d]+)_V([\d]+)-Aug([\d]+)_([\d]+)"

    files = sorted(os.listdir(scan_root))
    filepaths = [os.path.join(scan_root, f) for f in files if f.endswith(".im")]

    dataset_dicts = []
    for fp in filepaths:
        _, pid, time_point, _, slice_id, _ = tuple(re.split(FNAME_REGEX, fp))
        pid = int(pid)
        time_point = int(time_point)
        dataset_dicts.append(
            {
                "file_name": fp,
                "sem_seg_file": "{}.seg".format(os.path.splitext(fp)[0]),
                "scan_id": "{:07d}_V{:02d}".format(pid, time_point),
                "subject_id": pid,
                "time_point": time_point,
                "slice_id": int(slice_id),
                "scan_num_slices": 160,
            }
        )

    num_scans = len({d["scan_id"] for d in dataset_dicts})
    num_subjects = len({d["subject_id"] for d in dataset_dicts})
    if dataset_name:
        logger.info("Loaded {} from {}".format(dataset_name, scan_root))
    logger.info(
        "Loaded {} scans from {} subjects ({} slices)".format(
            num_scans, num_subjects, len(dataset_dicts)
        )
    )

    return dataset_dicts


def load_oai_3d_from_dir(scan_root, dataset_name=None):
    # sample scan name: "9311328_V01-Aug04_072.h5"
    FNAME_REGEX = "([\d]+)_V([\d]+)"

    files = sorted(os.listdir(scan_root))
    filepaths = [os.path.join(scan_root, f) for f in files]
    dataset_dicts = []
    for fp in filepaths:
        _, pid, time_point, _ = tuple(re.split(FNAME_REGEX, fp))
        pid = int(pid)
        time_point = int(time_point)
        dataset_dicts.append(
            {
                "file_name": fp,
                "sem_seg_file": fp,
                "scan_id": "{:07d}_V{:02d}".format(pid, time_point),
                "subject_id": pid,
                "time_point": time_point,
                "image_size": (384, 384, 160),
            }
        )

    num_scans = len(dataset_dicts)
    num_subjects = len({d["subject_id"] for d in dataset_dicts})
    if dataset_name:
        logger.info("Loaded {} from {}".format(dataset_name, scan_root))
    logger.info("Loaded {} scans from {} subjects".format(num_scans, num_subjects))

    return dataset_dicts


def load_oai_3d_sf_from_dir(scan_root, dataset_name=None):
    """
    Expected file structure:
          keys=> image_files (e.g. '0000001_V00');
          subkeys=> image type (e.g. ['seg', 'volume']);
    """
    FNAME_REGEX = "([\d]+)_V([\d]+)"
    f = h5py.File(scan_root, "r")
    keys = [key for key in f.keys()]
    f.close()
    dataset_dicts = []
    for key in keys:
        _, pid, time_point, _ = tuple(re.split(FNAME_REGEX, key))
        pid = int(pid)
        time_point = int(time_point)
        dataset_dicts.append(
            {
                "file_name": key,
                "sem_seg_file": key,
                "scan_id": "{:07d}_V{:02d}".format(pid, time_point),
                "subject_id": pid,
                "time_point": time_point,
                "image_size": (384, 384, 160),
                "singlefile_path": scan_root,
            }
        )

    num_scans = len(dataset_dicts)
    num_subjects = len({d["subject_id"] for d in dataset_dicts})
    if dataset_name:
        logger.info("Loaded {} from {}".format(dataset_name, scan_root))
    logger.info("Loaded {} scans from {} subjects".format(num_scans, num_subjects))

    return dataset_dicts


def register_oai(name, scan_root):
    load_func = None
    if name.startswith("oai_2d"):
        load_func = load_oai_2d_from_dir
    elif name.startswith("oai_3d_sf"):
        load_func = load_oai_3d_sf_from_dir
    else:
        load_func = load_oai_3d_from_dir

    DatasetCatalog.register(name, lambda: load_func(scan_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging

    MetadataCatalog.get(name).set(
        scan_root=scan_root,
        spacing=(0.3125, 0.3125, 0.7),
        test_set_metadata_pik=_TEST_SET_METADATA_PIK,
        category_ids=[x["id"] for x in OAI_CATEGORIES],
        category_abbreviations=[x["abbrev"] for x in OAI_CATEGORIES],
        categories=[x["name"] for x in OAI_CATEGORIES],
        category_colors=[x["color"] for x in OAI_CATEGORIES],
        category_id_to_contiguous_id={x["id"]: idx for idx, x in enumerate(OAI_CATEGORIES)},
        evaluator_type="SemSegEvaluator",
    )


def register_all_oai():
    for dataset_name, scan_root in _DATA_CATALOG.items():
        if not os.path.isabs(scan_root):
            scan_root = os.path.join(Cluster.working_cluster().data_dir, scan_root)
        register_oai(dataset_name, scan_root)
