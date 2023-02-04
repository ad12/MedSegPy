import json
import logging
import os
import re

from medsegpy.data.catalog import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

BASE_ANNOT_DIR = "/bmrNAS/people/arjun/data/qdess_knee_2020/annotations/v0.0.1"
BASE_IMAGE_DIR = "/dataNAS/people/arjun/data/qdess_knee_2020/image_files_total"
BASE_UNUSED_DIR = "/dataNAS/people/arjun/data/qdess_knee_2020/image_files_total"
BASE_UNLABELED_DIR = (
    "/dataNAS/people/arjun/data/qdess_knee_2020/unlabeled_outputs_dcm_fullanon_oct2021"
)

UNLABELED_SCANS = [
    "MTR_002",
    "MTR_003",
    "MTR_004",
    "MTR_007",
    "MTR_009",
    "MTR_011",
    "MTR_012",
    "MTR_014",
    "MTR_017",
    "MTR_021",
    "MTR_022",
    "MTR_024",
    "MTR_027",
    "MTR_029",
    "MTR_031",
    "MTR_032",
    "MTR_035",
    "MTR_036",
    "MTR_039",
    "MTR_041",
    "MTR_054",
    "MTR_055",
    "MTR_058",
    "MTR_059",
    "MTR_063",
    "MTR_064",
    "MTR_067",
    "MTR_068",
    "MTR_072",
    "MTR_073",
    "MTR_075",
    "MTR_081",
    "MTR_082",
    "MTR_085",
    "MTR_086",
    "MTR_089",
    "MTR_090",
    "MTR_091",
    "MTR_092",
    "MTR_102",
    "MTR_106",
    "MTR_108",
    "MTR_111",
    "MTR_114",
    "MTR_117",
    "MTR_119",
    "MTR_121",
    "MTR_123",
    "MTR_125",
    "MTR_128",
    "MTR_131",
    "MTR_132",
    "MTR_136",
    "MTR_137",
    "MTR_138",
    "MTR_153",
    "MTR_155",
    "MTR_157",
    "MTR_160",
    "MTR_162",
    "MTR_165",
    "MTR_166",
    "MTR_168",
    "MTR_169",
    "MTR_170",
    "MTR_181",
    "MTR_185",
    "MTR_195",
    "MTR_197",
    "MTR_231",
    "MTR_234",
    "MTR_242",
    "MTR_246",
    "MTR_247",
    "MTR_249",
]

UNUSED_SCANS = [
    "MTR_025",
    "MTR_042",
    "MTR_050",
    "MTR_060",
    "MTR_070",
    "MTR_076",
    "MTR_078",
    "MTR_088",
    "MTR_134",
    "MTR_141",
    "MTR_143",
    "MTR_147",
    "MTR_171",
    "MTR_180",
    "MTR_187",
    "MTR_202",
    "MTR_217",
    "MTR_226",
    "MTR_238",
    "MTR_239",
    "MTR_250",
]

_DATA_CATALOG = {
    "qdess_seg_train": os.path.join(BASE_ANNOT_DIR, "annotations/v0.0.1/train.json"),
    "qdess_seg_val": os.path.join(BASE_ANNOT_DIR, "annotations/v0.0.1/val.json"),
    "qdess_seg_test": os.path.join(BASE_ANNOT_DIR, "annotations/v0.0.1/test.json"),
    "qdess_seg_50_train": "qdess_data/50_perc_train/50_perc.json",
    "qdess_seg_25_train": "qdess_data/25_perc_train/25_perc.json",
    "qdess_seg_10_train": "qdess_data/10_perc_train/10_perc_split.json",
    "qdess_seg_5_train": "qdess_data/5_perc_train/5_perc_split.json",
    "qdess_125_perc_train": "qdess_data/percentage_jsons/125_perc.json",
    "qdess_150_perc_train": "qdess_data/percentage_jsons/150_perc.json",
    "qdess_175_perc_train": "qdess_data/percentage_jsons/175_perc.json",
    "qdess_200_perc_train": "qdess_data/percentage_jsons/200_perc.json",
    "qdess_unlabeled_all_train": "qdess_data/unlabeled_images/all_images.json",
    "qdess_seg_unused_train": "qdess_data/unlabeled_images/unused_orig_images.json",
}


QDESS_INPAINTING_CATEGORIES = [
    {"id": 0, "name": "inpainting_dummy_category", "abbrev": "inpainted", "color": [0, 0, 0]}
]

QDESS_SEGMENTATION_CATEGORIES = [
    {"id": 0, "name": "patellar cartilage", "abbrev": "pc", "color": [220, 20, 60]},
    {"id": 1, "name": "femoral cartilage", "abbrev": "fc", "color": [250, 170, 30]},
    {"id": (2, 3), "name": "tibial cartilage", "abbrev": "tc", "color": [0, 142, 0]},
    {"id": (4, 5), "name": "meniscus", "abbrev": "men", "color": [203, 92, 255]},
]


def load_2d_from_filepaths(filepaths: list, source_path: str, dataset_name: str = None):
    """
    Creates a list of dictionaries, where each dictionary contains information
    about each 2D image and corresponding ground truth segmentation in
    a particular directory (scan_root).

    The function assumes all 2D images are stored in the following format:

    %7d_V%02d-Aug%02d_%03d
    e.g. "9311328_V01-Aug04_072.im"

    Args:
        scan_root: The path of the directory containing the 2D images and
                    corresponding ground truth segmentations.
        total_num_slices: The total number of slices for this dataset.
        dataset_name: The name of the dataset.
    Returns:
        dataset_dicts: A list of dictionaries, described above in the
                        docstring.
    """
    FNAME_REGEX = "([\d]+)_V([\d]+)-Aug([\d]+)_([\d]+)"

    # Read appropriate JSON file for qDESS 2020 dataset
    split_type = dataset_name.split("_")[-1]
    annot_json_path = os.path.join(BASE_ANNOT_DIR, split_type + ".json")
    with open(annot_json_path, "r") as fp:
        annot_json = json.load(fp)
    volume_list = annot_json["images"]

    dataset_dicts = []
    for fp in filepaths:
        _, pid, time_point, _, slice_id, _ = tuple(re.split(FNAME_REGEX, fp))
        pid = int(pid)
        time_point = int(time_point)
        scan_id = f"{pid:07d}_V{time_point:02d}"

        # Iterate through image list of annotation JSON file to determine
        # number of slices in scan
        found_scan = False
        num_slices_in_scan = -1
        for volume in volume_list:
            if volume["msp_id"] == scan_id:
                found_scan = True
                num_slices_in_scan = volume["matrix_shape"][2]
        assert found_scan, f"Could not find scan with msp_id = {scan_id}"

        dataset_dicts.append(
            {
                "file_name": fp,
                "sem_seg_file": "{}.seg".format(os.path.splitext(fp)[0]),
                "scan_id": scan_id,
                "subject_id": pid,
                "time_point": time_point,
                "slice_id": int(slice_id),
                "scan_num_slices": num_slices_in_scan,
            }
        )

    num_scans = len({d["scan_id"] for d in dataset_dicts})
    num_subjects = len({d["subject_id"] for d in dataset_dicts})
    if dataset_name:
        logger.info("Loaded {} from {}".format(dataset_name, source_path))
    logger.info(
        "Loaded {} scans from {} subjects ({} slices)".format(
            num_scans, num_subjects, len(dataset_dicts)
        )
    )
    return dataset_dicts


def load_2d_from_dir(scan_root: str, dataset_name: str = None):
    files = sorted(os.listdir(scan_root))
    filepaths = [os.path.join(scan_root, f) for f in files if f.endswith(".im")]

    return load_2d_from_filepaths(filepaths, scan_root, dataset_name=dataset_name)


def load_2d_from_json(json_path: str, dataset_name: str = None):
    with open(json_path, "r") as fp:
        data_info = json.load(fp)
    base_dir = BASE_IMAGE_DIR
    scan_list = data_info["scans"]
    all_files = sorted(os.listdir(base_dir))
    all_images = [os.path.join(base_dir, f) for f in all_files if f.endswith(".im")]

    filepaths = []
    for img_path in all_images:
        img_name = img_path.split("/")[-1]
        scan_id = img_name.split("-")[0]
        if scan_id in scan_list:
            filepaths.append(img_path)

    return load_2d_from_filepaths(filepaths, json_path, dataset_name=dataset_name)


def load_qdess(json_file, dataset_name=None):
    dataset_names = [dataset_name]
    json_files = [json_file]

    dataset_dicts = []
    for json_path, data_name in zip(json_files, dataset_names):  # noqa: B007
        with open(json_path, "r") as f:
            data = json.load(f)

        images = data["images"]

        for img in images:
            if img["scan_id"] in UNLABELED_SCANS:
                base_dir = BASE_UNLABELED_DIR
            elif img["scan_id"] in UNUSED_SCANS:
                base_dir = BASE_UNUSED_DIR
            else:
                base_dir = BASE_IMAGE_DIR
            fp = os.path.join(base_dir, img["file_name"])
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"File {fp} not found")
            dataset_dicts.append(
                {
                    "file_name": fp,
                    "sem_seg_file": fp,
                    "scan_id": img["scan_id"],
                    "subject_id": img["subject_id"],
                    "scan_spacing": img["voxel_spacing"],
                    "image_size": img["matrix_shape"],
                }
            )

    num_scans = len(dataset_dicts)
    num_subjects = len({d["subject_id"] for d in dataset_dicts})
    if dataset_name:
        logger.info("Loaded {} from {}".format(dataset_name, json_file))
    logger.info("Loaded {} scans from {} subjects".format(num_scans, num_subjects))
    return dataset_dicts


def register_qdess_dataset(scan_root: str, dataset_name: str):
    """
    Registers the qDESS knee MRI 2020 dataset and adds metadata for the
    dataset.

    The function will choose the correct list of categories based on
    "dataset_name". If the task is inpainting, the names of the datasets used
    for inpainting must include the word "inpainting".

    Args:
        scan_root: The path of the directory containing the 2D images and
                    corresponding ground truth segmentations.
        dataset_name: The name of the dataset.

    Returns:
        N/A
    """
    if scan_root.endswith(".json"):
        load_func = load_qdess
    else:
        load_func = load_2d_from_dir

    DatasetCatalog.register(dataset_name, lambda: load_func(scan_root, dataset_name))

    # Add Metadata about the qDESS MRI dataset
    MetadataCatalog.get(dataset_name).set(
        scan_root="dummy_root",
        category_ids={
            "segmentation": [x["id"] for x in QDESS_SEGMENTATION_CATEGORIES],
            "inpainting": [x["id"] for x in QDESS_INPAINTING_CATEGORIES],
        },
        category_abbreviations={
            "segmentation": [x["abbrev"] for x in QDESS_SEGMENTATION_CATEGORIES],
            "inpainting": [x["abbrev"] for x in QDESS_INPAINTING_CATEGORIES],
        },
        categories={
            "segmentation": [x["name"] for x in QDESS_SEGMENTATION_CATEGORIES],
            "inpainting": [x["name"] for x in QDESS_INPAINTING_CATEGORIES],
        },
        category_colors={
            "segmentation": [x["color"] for x in QDESS_SEGMENTATION_CATEGORIES],
            "inpainting": [x["color"] for x in QDESS_INPAINTING_CATEGORIES],
        },
        category_id_to_contiguous_id={
            "segmentation": {x["id"]: idx for idx, x in enumerate(QDESS_SEGMENTATION_CATEGORIES)},
            "inpainting": {x["id"]: idx for idx, x in enumerate(QDESS_INPAINTING_CATEGORIES)},
        },
        evaluator_type={"segmentation": "QDESSEvaluator", "inpainting": "SemSegEvaluator"},
    )


def register_all_qdess_datasets():
    """
    Registers all qDESS MRI datasets listed in _DATA_CATALOG.
    """
    for dataset_name, scan_root in _DATA_CATALOG.items():
        register_qdess_dataset(scan_root=scan_root, dataset_name=dataset_name)
