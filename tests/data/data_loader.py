import os
import shutil
import unittest

import h5py
import numpy as np
from fvcore.common.file_io import PathManager

from medsegpy.config import Config
from medsegpy.data import N5dDataLoader, PatchDataLoader

from .. import utils  # noqa


class TestPatchDataLoader(unittest.TestCase):
    IMG_SIZE = (10, 20, 30)
    NUM_CLASSES = 4
    FILE_PATH = "mock_data://temp_data/scan1.h5"

    @classmethod
    def setUpClass(cls):
        img = np.random.rand(*cls.IMG_SIZE).astype(np.float32)
        seg = (np.random.rand(*cls.IMG_SIZE, cls.NUM_CLASSES) >= 0.5).astype(
            np.uint8
        )

        file_path = PathManager.get_local_path(cls.FILE_PATH)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with h5py.File(file_path, "w") as f:
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

    def test_slicing(self):
        """Test selecting slices.

        Currently only verified for last dimension.
        """
        with h5py.File(PathManager.get_local_path(self.FILE_PATH), "r") as f:
            volume = f["volume"][:]
            mask = f["seg"][:]

        cfg = Config("")
        cfg.IMG_SIZE = (10, 20, 1)
        cfg.CATEGORIES = (0, 1, 3)
        dataset_dicts = self.get_dataset_dicts()

        # Simple slice
        data_loader = PatchDataLoader(
            cfg, dataset_dicts, is_test=False, shuffle=False
        )
        assert len(data_loader) == 30
        for i in range(30):
            img, seg = data_loader[i]
            img, seg = np.squeeze(img, axis=(0, 3)), np.squeeze(seg, axis=0)
            assert np.all(img == volume[:, :, i])
            assert np.all(seg == mask[:, :, i, cfg.CATEGORIES])


class TestN5dDataLoader(unittest.TestCase):
    IMG_SIZE = (10, 20, 30)
    NUM_CLASSES = 4
    FILE_PATH = "mock_data://temp_data/scan1.h5"

    @classmethod
    def setUpClass(cls):
        img = np.random.rand(*cls.IMG_SIZE).astype(np.float32)
        seg = (np.random.rand(*cls.IMG_SIZE, cls.NUM_CLASSES) >= 0.5).astype(
            np.uint8
        )

        file_path = PathManager.get_local_path(cls.FILE_PATH)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with h5py.File(file_path, "w") as f:
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

    def test_simple(self):
        """Test simple 2.5d configuration.
        """
        with h5py.File(PathManager.get_local_path(self.FILE_PATH), "r") as f:
            volume = f["volume"][:]
            mask = f["seg"][:]

        num_slices = 3
        window = num_slices // 2
        cfg = Config("")
        cfg.IMG_SIZE = (10, 20, num_slices)
        cfg.CATEGORIES = (0, 1, 3)
        dataset_dicts = self.get_dataset_dicts()

        # No padding
        # First and last slices will be skipped b/c no padding is used
        data_loader = N5dDataLoader(
            cfg, dataset_dicts, is_test=False, shuffle=False
        )
        assert len(data_loader) == 30 - 2 * (num_slices // 2)
        for idx, i in enumerate(range(1, 29)):
            img, seg = data_loader[idx]
            img, seg = np.squeeze(img, axis=0), np.squeeze(seg, axis=0)
            assert np.all(img == volume[:, :, i - window : i + window + 1])
            assert np.all(seg == mask[:, :, i, cfg.CATEGORIES])

    def test_padding(self):
        with h5py.File(PathManager.get_local_path(self.FILE_PATH), "r") as f:
            volume = f["volume"][:]
            mask = f["seg"][:]

        num_slices = 3
        window = num_slices // 2
        cfg = Config("")
        cfg.IMG_SIZE = (10, 20, num_slices)
        cfg.CATEGORIES = (0, 1, 3)
        dataset_dicts = self.get_dataset_dicts()

        pad_size = ((0, 0), (0, 0), (window, window))
        pad_mode = "edge"
        cfg.IMG_PAD_SIZE = (0, 0, window)
        cfg.IMG_PAD_MODE = pad_mode
        data_loader = N5dDataLoader(
            cfg, dataset_dicts, is_test=False, shuffle=False
        )
        _s_volume = np.pad(volume, pad_size, pad_mode)
        _s_mask = np.pad(mask, pad_size + ((0, 0),), pad_mode)
        assert len(data_loader) == 30
        for idx, i in enumerate(range(window, 30 + window)):
            img, seg = data_loader[idx]
            img, seg = np.squeeze(img, axis=0), np.squeeze(seg, axis=0)
            assert np.all(img == _s_volume[:, :, i - window : i + window + 1])
            assert np.all(seg == _s_mask[:, :, i, cfg.CATEGORIES])


if __name__ == "__main__":
    unittest.main()
