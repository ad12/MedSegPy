"""Test model output reproducability.

These tests check that Monte Carlo dropout during inference produces 
reproducible results.
"""

import unittest
import numpy as np
import os
import h5py
import shutil
from fvcore.common.file_io import PathManager

from medsegpy.config import UNetConfig
from medsegpy.modeling.meta_arch import build_model
from medsegpy.modeling.model import Model
from medsegpy.data import DefaultDataLoader

class TestMCDropout(unittest.TestCase):
    IMG_SIZE = (512, 512, 1)
    NUM_CLASSES = 4
    FILE_PATH = "mock_data://temp_data/scan.h5"

    @classmethod
    def setUpClass(cls):
        img = np.random.rand(*cls.IMG_SIZE).astype(np.float32)
        seg = (np.random.rand(*cls.IMG_SIZE, cls.NUM_CLASSES) >= 0.5).astype(np.uint8)

        file_path = PathManager.get_local_path(cls.FILE_PATH)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with h5py.File(file_path, "w") as f:
            f.create_dataset("data", data=img)
            f.create_dataset("volume", data=img)
            f.create_dataset("seg", data=seg)

    @classmethod
    def tearDownClass(cls):
        file_path = PathManager.get_local_path(cls.FILE_PATH)
        shutil.rmtree(os.path.dirname(file_path))

    def get_dataset_dicts(self):
        file_path = PathManager.get_local_path(self.FILE_PATH)
        return [
            {
                "file_name": file_path,
                "sem_seg_file": file_path,
                "scan_id": os.path.splitext(os.path.basename(file_path))[0],
                "image_size": self.IMG_SIZE,
            }
        ]

    def test_inference_with_mc_dropout(self):
        cfg = UNetConfig()
        cfg.MC_DROPOUT = True
        cfg.MC_DROPOUT_T = 10
        cfg.IMG_SIZE = self.IMG_SIZE
        model = build_model(cfg)

        with h5py.File(PathManager.get_local_path(self.FILE_PATH), "r") as f:
            volume = f["volume"][:]
            mask = f["seg"][:]
        dataset_dicts = self.get_dataset_dicts()
        data_loader = DefaultDataLoader(cfg, dataset_dicts, is_test=True, shuffle=False)

        # Feed same data to inference generator twice
        kwargs = dict()
        kwargs["mc_dropout"] = data_loader._cfg.MC_DROPOUT
        kwargs["mc_dropout_T"] = data_loader._cfg.MC_DROPOUT_T
        x1, y1, preds1 = Model.inference_generator_static(model, data_loader, **kwargs)
        x2, y2, preds2 = Model.inference_generator_static(model, data_loader, **kwargs)

        # Outputs should be the same
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
        assert np.array_equal(preds1["preds"], preds2["preds"])
        assert np.array_equal(preds1["preds_mc_dropout"], preds2["preds_mc_dropout"])


if "__name__" == "__main__":
    unittest.main()
