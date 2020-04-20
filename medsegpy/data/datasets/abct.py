from medsegpy.data.catalog import MetadataCatalog
from medsegpy.utils.cluster import Cluster, CLUSTER

_DATA_CATALOG = {}

if CLUSTER in (Cluster.ROMA, Cluster.VIGATA):
    DATA_CATALOG = {
        "oai_2d_train": "/share/pi/bah/data/oai_data/h5_files_2d/train",
        "oai_2d_val": "/share/pi/bah/data/oai_data/h5_files_2d/val",
        "oai_2d_test": "/share/pi/bah/data/oai_data/h5_files_2d/test",
    }
else:
    pass

ABCT_CATEGORIES = [
    {"color": [220, 20, 60], "id": 0, "name": "cat1", "abbrev": "cat1"},
    {"color": [119, 11, 32], "id": 1, "name": "cat2", "abbrev": "cat2"},
    {"color": [0, 0, 142], "id": 2, "name": "cat3", "abbrev": "cat3"},
    {"color": [0, 0, 230], "id": 3, "name": "cat4", "abbrev": "cat4"},
    {"color": [250, 170, 30], "id": 4, "name": "cat5", "abbrev": "cat5"},
]


def register_abct():
    for dataset_name, dir_path in _DATA_CATALOG.items():
        MetadataCatalog.get(dataset_name).set(
            scan_root=dir_path,
            category_ids=[x["id"] for x in ABCT_CATEGORIES],
            category_abbreviations=[x["abbrev"] for x in ABCT_CATEGORIES],
            categories=[x["name"] for x in ABCT_CATEGORIES],
            categories_colors=[x["color"] for x in ABCT_CATEGORIES],
            evaluator_type="SemSegEvaluator",
        )
