"""OAI iMorphics Dataset"""
import logging
import os
import re

from medsegpy.data.catalog import DatasetCatalog, MetadataCatalog
from medsegpy.utils.cluster import Cluster, CLUSTER

logger = logging.getLogger(__name__)

_DATA_CATALOG = {}
_TEST_SET_METADATA_PIK = ""

if CLUSTER in (Cluster.ROMA, Cluster.VIGATA):
    _DATA_CATALOG = {
        "oai_2d_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/train",
        "oai_2d_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/valid",
        "oai_2d_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/test",

        "oai_2d_whitened_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_2d/train",  # noqa
        "oai_2d_whitened_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_2d/valid",  # noqa
        "oai_2d_whitened_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_2d/test",  # noqa

        "oai_3d_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_3d/train",
        "oai_3d_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_3d/val",
        "oai_3d_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_3d/test",

        "oai_3d_whitened_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_3d/train",
        "oai_3d_whitened_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_3d/val",
        "oai_3d_whitened_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_3d/test",
    }
    _TEST_SET_METADATA_PIK = (
        "/bmrNAS/people/arjun/msk_seg_networks/oai_metadata/oai_data.dat"
    )
elif CLUSTER == CLUSTER.NERO:
    _DATA_CATALOG = {
        "oai_2d_train": "/share/pi/bah/data/oai_data/h5_files_2d/train",
        "oai_2d_val": "/share/pi/bah/data/oai_data/h5_files_2d/val",
        "oai_2d_test": "/share/pi/bah/data/oai_data/h5_files_2d/test",
    }
else:
    pass

OAI_CATEGORIES = [
    {
        "color": [220, 20, 60],
        "id": 0,
        "name": "femoral cartilage",
        "abbrev": "fc",
    },
    {
        "color": [119, 11, 32],
        "id": (1, 2),
        "name": "tibial cartilage",
        "abbrev": "tc",
    },
    {
        "color": [0, 0, 142],
        "id": 3,
        "name": "patellar cartilage",
        "abbrev": "pc",
    },
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
                "total_num_slices": 160,
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
    logger.info(
        "Loaded {} scans from {} subjects".format(num_scans, num_subjects)
    )

    return dataset_dicts


def register_oai(name, scan_root):
    load_func = (
        load_oai_3d_from_dir
        if name.startswith("oai_3d")
        else load_oai_2d_from_dir
    )
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
        category_id_to_contiguous_id={
            x["id"]: idx for idx, x in enumerate(OAI_CATEGORIES)
        },
        evaluator_type="SemSegEvaluator",
    )


def register_all_oai():
    for dataset_name, scan_root in _DATA_CATALOG.items():
        register_oai(dataset_name, scan_root)
