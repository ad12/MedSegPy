import os

from fvcore.common.file_io import PathManager

from medsegpy.data import DatasetCatalog, MetadataCatalog
from medsegpy.data.datasets.oai import (
    _DATA_CATALOG,
    OAI_CATEGORIES,
    load_oai_2d_from_dir,
    load_oai_3d_from_dir,
)
from medsegpy.utils.io_utils import GeneralPathHandler

LOCAL_FOLDER = os.path.dirname(os.path.abspath(__file__))


class MockDataPathHandler(GeneralPathHandler):
    PREFIX = "mock_data://"

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]
        return os.path.join(LOCAL_FOLDER, "mock_data", name)


class ModelImagesPathHandler(GeneralPathHandler):
    PREFIX = "model://"

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]
        return os.path.join(LOCAL_FOLDER, "model", name)


def _register_handlers():
    PathManager.register_handler(MockDataPathHandler())
    PathManager.register_handler(ModelImagesPathHandler())


def load_oai_2d_mini(scan_root, dataset_name=None):
    """Takes 2 subjects (in sorted order) for full dataset"""
    dataset_dicts = load_oai_2d_from_dir(scan_root, dataset_name)
    subject_ids = sorted({x["subject_id"] for x in dataset_dicts})[:2]
    dataset_dicts = [x for x in dataset_dicts if x["subject_id"] in subject_ids]
    return dataset_dicts


def load_oai_3d_mini(scan_root, dataset_name=None):
    """Takes 2 subjects (in sorted order) for full dataset"""
    dataset_dicts = load_oai_3d_from_dir(scan_root, dataset_name)
    subject_ids = sorted({x["subject_id"] for x in dataset_dicts})[:2]
    dataset_dicts = [x for x in dataset_dicts if x["subject_id"] in subject_ids]
    return dataset_dicts


def register_oai_mini(name, scan_root):
    name = "mini_{}".format(name)
    if "oai_3d" in name:
        DatasetCatalog.register(name, lambda: load_oai_3d_mini(scan_root, name))
    else:
        DatasetCatalog.register(name, lambda: load_oai_2d_mini(scan_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging

    MetadataCatalog.get(name).set(
        scan_root=scan_root,
        voxel_spacing=(0.3125, 0.3125, 0.7),
        category_ids=[x["id"] for x in OAI_CATEGORIES],
        category_abbreviations=[x["abbrev"] for x in OAI_CATEGORIES],
        categories=[x["name"] for x in OAI_CATEGORIES],
        category_colors=[x["color"] for x in OAI_CATEGORIES],
        category_id_to_contiguous_id={
            x["id"]: idx for idx, x in enumerate(OAI_CATEGORIES)
        },
        evaluator_type="SemSegEvaluator",
    )


_register_handlers()

for dataset_name, scan_root in _DATA_CATALOG.items():
    register_oai_mini(dataset_name, scan_root)
