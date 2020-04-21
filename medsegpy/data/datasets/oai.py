from medsegpy.data.catalog import MetadataCatalog
from medsegpy.utils.cluster import Cluster, CLUSTER

_DATA_CATALOG = {}
_TEST_SET_METADATA_PIK = ""

if CLUSTER in (Cluster.ROMA, Cluster.VIGATA):
    _DATA_CATALOG = {
        "oai_2d_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/train",
        "oai_2d_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/valid",
        "oai_2d_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/test",

        "oai_2d_whitened_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_2d/train",
        "oai_2d_whitened_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_2d/valid",
        "oai_2d_whitened_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_2d/test",

        "oai_3d_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_3d/train",
        "oai_3d_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_3d/val",
        "oai_3d_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_3d/test",
    }
    _TEST_SET_METADATA_PIK = "/bmrNAS/people/arjun/msk_seg_networks/oai_metadata/oai_data.dat"
elif CLUSTER == CLUSTER.NERO:
    _DATA_CATALOG = {
        "oai_2d_train": "/share/pi/bah/data/oai_data/h5_files_2d/train",
        "oai_2d_val": "/share/pi/bah/data/oai_data/h5_files_2d/val",
        "oai_2d_test": "/share/pi/bah/data/oai_data/h5_files_2d/test",
    }
else:
    pass

OAI_CATEGORIES = [
    {"color": [220, 20, 60], "id": 0, "name": "femoral cartilage", "abbrev": "fc"},
    {"color": [119, 11, 32], "id": (1, 2), "name": "tibial cartilage", "abbrev": "tc"},
    {"color": [0, 0, 142], "id": 3, "name": "patellar cartilage", "abbrev": "pc"},
    {"color": [0, 0, 230], "id": (4, 5), "name": "meniscus", "abbrev": "men"},
]


def register_oai():
    for dataset_name, dir_path in _DATA_CATALOG.items():
        MetadataCatalog.get(dataset_name).set(
            scan_root=dir_path,
            voxel_spacing=(0.3125, 0.3125, 0.7),
            test_set_metadata_pik=_TEST_SET_METADATA_PIK,
            category_ids=[x["id"] for x in OAI_CATEGORIES],
            category_abbreviations=[x["abbrev"] for x in OAI_CATEGORIES],
            categories=[x["name"] for x in OAI_CATEGORIES],
            category_colors=[x["color"] for x in OAI_CATEGORIES],
            category_id_to_contiguous_id={x["id"]: idx for idx, x in enumerate(OAI_CATEGORIES)},
            evaluator_type="SemSegEvaluator",
        )
