"""Stanford abdominal CT Dataset."""
import json
import logging
import os
import re

from medsegpy.data.catalog import DatasetCatalog, MetadataCatalog
from medsegpy.utils.cluster import Cluster

logger = logging.getLogger(__name__)

_TEST_SET_METADATA_PIK = ""

_DATA_CATALOG = {
    "abCT_seg_train": "abct_data/json_files/train.json",
    "abCT_seg_50_train": "abct_data/json_files/50_perc_train/50_perc.json",
    "abCT_seg_25_train": "abct_data/json_files/25_perc_train/25_perc.json",
    "abCT_seg_10_train": "abct_data/json_files/10_perc_train/10_perc.json",
    "abCT_seg_5_train": "abct_data/json_files/5_perc_train/5_perc.json",
    "abCT_375_perc_train": "abct_data/json_files/percentage_jsons/375_perc.json",
    "abCT_650_perc_train": "abct_data/json_files/percentage_jsons/650_perc.json",
    "abCT_925_perc_train": "abct_data/json_files/percentage_jsons/925_perc.json",
    "abCT_1200_perc_train": "abct_data/json_files/percentage_jsons/1200_perc.json",
    "abCT_seg_val": "abct_data/valid",
    "abCT_seg_test": "abct_data/test",
}


ABCT_INPAINTING_CATEGORIES = [
    {"color": [0, 0, 0], "id": 0, "name": "soft", "abbrev": "soft"},
    {"color": [0, 0, 0], "id": 1, "name": "bone", "abbrev": "bone"},
    {"color": [0, 0, 0], "id": 2, "name": "custom", "abbrev": "custom"},
]


ABCT_CATEGORIES = [
    {"color": [0, 214, 4], "id": 0, "name": "background", "abbrev": "bg"},
    {"color": [255, 102, 0], "id": 1, "name": "muscle", "abbrev": "muscle"},
    {"color": [0, 102, 255], "id": 2, "name": "intramuscular fat", "abbrev": "imat"},
    {"color": [204, 0, 255], "id": 3, "name": "visceral fat", "abbrev": "vat"},
    {"color": [250, 170, 30], "id": 4, "name": "subcutaneous fat", "abbrev": "sat"},
]


def load_abct_from_dir(scan_root, dataset_name=None):
    files = sorted(os.listdir(scan_root))
    filepaths = [os.path.join(scan_root, f) for f in files if f.endswith(".im")]

    return load_abct(
        filepaths=filepaths, txt_file_or_scan_root=scan_root, dataset_name=dataset_name
    )


def load_abct_from_json(json_file, dataset_name=None):
    FNAME_REGEX = "([\d]+)_V([\d]+)-Aug([\d]+)_([\d]+)"
    with open(json_file, "r") as fp:
        json_dict = json.load(fp)
    filepaths = []
    split_type = dataset_name.split("_")[-1]
    for img_dict in json_dict["images"]:
        _, pid, _, _, _, _ = tuple(re.split(FNAME_REGEX, img_dict["file_name"]))
        if int(pid) < 700:
            data_dir = os.path.join(Cluster.working_cluster().data_dir, "abct", split_type)
        else:
            data_dir = os.path.join(Cluster.working_cluster().data_dir, "abct/unlabeled")
        filepaths.append(os.path.join(data_dir, img_dict["file_name"]))

    return load_abct(
        filepaths=filepaths, txt_file_or_scan_root=json_file, dataset_name=dataset_name
    )


def load_abct(filepaths, txt_file_or_scan_root, dataset_name=None):
    # sample scan name: "9311328_V01-Aug04_072.im"
    FNAME_REGEX = "([\d]+)_V([\d]+)-Aug([\d]+)_([\d]+)"

    dataset_dicts = []
    for fp in filepaths:
        # Check if image file exists
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"File {fp} not found")

        # Get attributes for image file
        _, pid, study_id, _, slice_id, _ = tuple(re.split(FNAME_REGEX, fp))
        pid = int(pid)
        study_id = int(study_id)
        if study_id == 0 or study_id == 2:
            scan_spacing = (0.703125, 0.703125)
        elif study_id == 1:
            scan_spacing = (0.80078125, 0.80078125)
        else:
            raise ValueError("Got study_id {}".format(study_id))

        cur_data_dict = {
            "file_name": fp,
            "scan_id": "{:07d}_V{:02d}".format(pid, study_id),
            "subject_id": pid,
            "study_id": study_id,
            "scan_spacing": scan_spacing,
        }

        # Determine if mask file exists
        sem_seg_file_path = "{}.seg".format(os.path.splitext(fp)[0])
        if os.path.isfile(sem_seg_file_path):
            cur_data_dict["sem_seg_file"] = sem_seg_file_path

        dataset_dicts.append(cur_data_dict)

    num_scans = len(dataset_dicts)
    num_subjects = len({d["subject_id"] for d in dataset_dicts})
    if dataset_name:
        logger.info("Loaded {} from {}".format(dataset_name, txt_file_or_scan_root))
    logger.info("Loaded {} scans from {} subjects".format(num_scans, num_subjects))

    return dataset_dicts


def register_abct(name, txt_file_or_scan_root):
    if txt_file_or_scan_root.endswith(".json"):
        DatasetCatalog.register(name, lambda: load_abct_from_json(txt_file_or_scan_root, name))
    else:
        DatasetCatalog.register(name, lambda: load_abct_from_dir(txt_file_or_scan_root, name))

    MetadataCatalog.get(name).set(
        scan_root="dummy_root",
        category_ids={
            "segmentation": [x["id"] for x in ABCT_CATEGORIES],
            "inpainting": [x["id"] for x in ABCT_INPAINTING_CATEGORIES],
        },
        category_abbreviations={
            "segmentation": [x["abbrev"] for x in ABCT_CATEGORIES],
            "inpainting": [x["abbrev"] for x in ABCT_INPAINTING_CATEGORIES],
        },
        categories={
            "segmentation": [x["name"] for x in ABCT_CATEGORIES],
            "inpainting": [x["name"] for x in ABCT_INPAINTING_CATEGORIES],
        },
        category_colors={
            "segmentation": [x["color"] for x in ABCT_CATEGORIES],
            "inpainting": [x["color"] for x in ABCT_INPAINTING_CATEGORIES],
        },
        category_id_to_contiguous_id={
            "segmentation": {x["id"]: idx for idx, x in enumerate(ABCT_CATEGORIES)},
            "inpainting": {x["id"]: idx for idx, x in enumerate(ABCT_INPAINTING_CATEGORIES)},
        },
        evaluator_type={"segmentation": "CTEvaluator", "inpainting": "SemSegEvaluator"},
    )


def register_all_abct():
    for dataset_name, txt_file_or_scan_root in _DATA_CATALOG.items():
        if not os.path.isabs(txt_file_or_scan_root):
            txt_file_or_scan_root = os.path.join(
                Cluster.working_cluster().data_dir, txt_file_or_scan_root
            )
        register_abct(dataset_name, txt_file_or_scan_root)
