"""Abdominal CT Dataset."""
import logging
import os
import re

from medsegpy.data.catalog import DatasetCatalog, MetadataCatalog
from medsegpy.utils.cluster import Cluster, CLUSTER

logger = logging.getLogger(__name__)

_DATA_CATALOG = {}

if CLUSTER in (Cluster.ROMA, Cluster.VIGATA):
    _DATA_CATALOG = {
        "abCT_v0.0.1_train": "/bmrNAS/people/akshay/dl/abct_data/train",
        "abCT_v0.0.1_val": "/bmrNAS/people/akshay/dl/abct_data/valid",
        "abCT_v0.0.1_test": "/bmrNAS/people/akshay/dl/abct_data/test",
    }
else:
    pass

ABCT_CATEGORIES = [
    {"color": [220, 20, 60], "id": 0, "name": "background", "abbrev": "bg"},
    {"color": [119, 11, 32], "id": 1, "name": "cat2", "abbrev": "cat2"},
    {"color": [0, 0, 142], "id": 2, "name": "cat3", "abbrev": "cat3"},
    {"color": [0, 0, 230], "id": 3, "name": "cat4", "abbrev": "cat4"},
    {"color": [250, 170, 30], "id": 4, "name": "cat5", "abbrev": "cat5"},
]


def load_abct(scan_root, dataset_name=None):
    # sample scan name: "9311328_V01-Aug04_072.h5"
    FNAME_REGEX = "([\d]+)_V([\d]+)-Aug([\d]+)_([\d]+)"

    files = sorted(os.listdir(scan_root))
    filepaths = [os.path.join(scan_root, f) for f in files if f.endswith(".im")]

    dataset_dicts = []
    for fp in filepaths:
        _, pid, study_id, _, slice_id, _ = tuple(re.split(FNAME_REGEX, fp))
        pid = int(pid)
        study_id = int(study_id)
        dataset_dicts.append(
            {
                "file_name": fp,
                "sem_seg_file": "{}.seg".format(os.path.splitext(fp)[0]),
                "scan_id": "{:07d}_V{:02d}".format(pid, study_id),
                "subject_id": pid,
                "study_id": study_id,
            }
        )

    num_scans = len(dataset_dicts)
    num_subjects = len({d["subject_id"] for d in dataset_dicts})
    if dataset_name:
        logger.info("Loaded {} from {}".format(dataset_name, scan_root))
    logger.info(
        "Loaded {} scans from {} subjects".format(num_scans, num_subjects)
    )

    return dataset_dicts


def register_abct(name, scan_root):
    DatasetCatalog.register(name, lambda: load_abct(scan_root, name))
    MetadataCatalog.get(name).set(
        scan_root=scan_root,
        category_ids=[x["id"] for x in ABCT_CATEGORIES],
        category_abbreviations=[x["abbrev"] for x in ABCT_CATEGORIES],
        categories=[x["name"] for x in ABCT_CATEGORIES],
        categories_colors=[x["color"] for x in ABCT_CATEGORIES],
        category_id_to_contiguous_id={
            x["id"]: idx for idx, x in enumerate(ABCT_CATEGORIES)
        },
        evaluator_type="SemSegEvaluator",
    )


def register_all_abct():
    for dataset_name, scan_root in _DATA_CATALOG.items():
        register_abct(dataset_name, scan_root)
