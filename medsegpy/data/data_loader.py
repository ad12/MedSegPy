raise NotImplementedError("This module is not ready for use.")

import math
from abc import ABC, abstractmethod
from collections import defaultdict
import random
import logging
import time
from typing import Any, Dict, List, Sequence, Tuple

from keras import utils as k_utils
import numpy as np
import h5py

from medsegpy.config import Config
from .im_gens import GeneratorState, OAIGenerator
from .sem_seg_utils import collect_mask, add_background_labels

logger = logging.getLogger(__name__)

# Keys to use when sorting dataset during inference.
_SORT_KEYS = ("scan_id", "slice_id", "block_id")


def get_data_loader(
    data_loader_name,
    **kwargs
) -> "DataLoader":
    """Get data loader based on config `TAG` value"""
    for generator in [DefaultDataLoader]:
        try:
            gen = generator(config, state, **kwargs)
            if gen:
                return gen
        except ValueError:
            continue

    raise ValueError('No data loader found for tag `{}`'.format(generator))


class DataLoader(k_utils.Sequence, ABC):
    """Data loader following :class:`keras.utils.Sequence` API.

    To avoid changing the order of the base list, we shuffle a list of indices
    and query based on the index.
    Data loaders in medsegpy also have the ability to yield inference results
    per scan.
    """
    ALIASES = []

    def __init__(
        self,
        cfg: Config,
        dataset_dicts: List[Dict],
        is_test: bool = False,
        shuffle: bool = True,
        drop_last: bool = False,
        batch_size: int = 1,
    ):
        """
        Args:
            cfg (Config): A config object.
            dataset_dicts (List[Dict]): List of data in medsegpy dataset format.
            is_test (:obj:`bool`, optional): If `True`, configures loader as a
                testing/inference loader. This is typically used when running
                evaluation.
            shuffle (bool, optional): If `True`, shuffle data every epoch.
            drop_last (:obj:`bool`, optional): Drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If
                `False` and the size of the dataset is not divisible by batch
                size, then the last batch will be smaller. This can affect
                loss calculations.
            batch_size (:obj:`int`, optional): Batch size.
        """
        self._cfg = cfg
        self._dataset_dicts = dataset_dicts
        self.shuffle = shuffle
        seed = cfg.SEED
        self._random = random.Random(seed) if seed else random.Random()
        self.drop_last = drop_last
        self._batch_size = batch_size
        self._category_idxs = cfg.CATEGORIES
        self._is_test = is_test
        self._idxs = list(range(0, self._num_elements()))
        if shuffle:
            self._random.shuffle(self._idxs)

    def on_epoch_end(self):
        if self.shuffle:
            self._random.shuffle(self._idxs)

    def __len__(self):
        """Number of batches.

        By default, each element in the dataset dict is independent.
        """
        _num_elements = self._num_elements()
        if self.drop_last:
            return _num_elements // self._batch_size
        else:
            return math.ceil(_num_elements / self._batch_size)

    def _num_elements(self):
        """Number of elements in the data loader."""
        return len(self._dataset_dicts)

    @abstractmethod
    def inference(self, model) -> Tuple[str, Any, Any, Any, float]:
        """Yields x_in, y_true, y_pred, scan_id, time_elapsed in order.

        Args:
            model: A model to run inference on.
        """
        yield None, None, None, None, None


class DefaultDataLoader(DataLoader):
    """The default data loader functionality in medsegy.

    This class takes a dataset dict in the MedSegPy Dataset format and maps it
    to a format that can be used by the model for semantic segmentation.

    This is the default data loader.

    1. Read the input matrix from "file_name"
    2. Read the ground truth mask matrix from "sem_seg_file_name"
    3. If needed:
        a. Add binary labels for background
    """
    _GENERATOR_TYPE = OAIGenerator

    def __init__(
        self,
        cfg: Config,
        dataset_dicts: List[Dict],
        state: GeneratorState = GeneratorState.TRAINING,
        shuffle: bool = True,
        drop_last: bool = False,
        batch_size: int = 1,
    ):
        super().__init__(
            cfg, dataset_dicts, state, shuffle, drop_last, batch_size
        )

        self._image_size = cfg.IMG_SIZE
        self._include_background = cfg.INCLUDE_BACKGROUND
        self._num_neighboring_slices = cfg.num_neighboring_slices()
        self._num_classes = cfg.get_num_classes()

    def _load_input(self, image_file, sem_seg_file):
        with h5py.File(image_file) as f:
            image = f["data"][:]
        if image.shape[-1] != 1:
            image = image[..., np.newaxis]

        with h5py.File(sem_seg_file) as f:
            mask = f["data"][:]

        cat_idxs = self._category_idxs
        mask = collect_mask(mask, index=cat_idxs)
        if self._include_background:
            mask = add_background_labels(mask)

        return image, mask

    def _load_batch(self, idxs: Sequence[int]):
        img_size = self._image_size
        dataset_dicts = self._dataset_dicts

        total_classes = self._num_classes
        mask_size = img_size[:-1] + (total_classes,)

        images = []
        masks = []
        for file_idx in idxs:
            dataset_dict = dataset_dicts[file_idx]
            file_name = dataset_dict["file_name"]
            sem_seg_file_name = dataset_dict["sem_seg_file_name"]

            image, mask = self._load_input(file_name, sem_seg_file_name)

            assert image.shape == img_size, (
                "Image shape mismatch. Expected {}, got {}".format(
                    img_size, image.shape,
                )
            )
            assert mask.shape == mask_size, (
                "Mask shape mismatch. Expected {}, got {}".format(
                    mask_size, mask.shape
                )
            )

            images.append(image)
            masks.append(mask)

        return np.stack(images, axis=0), np.stack(masks, axis=0)

    def __getitem__(self, idx):
        """
        Args:
            idx: Batch index.

        Returns:
            ndarray, ndarray: images NxHxWx(...)x1, masks NxHxWx(...)x1
        """
        batch_size = self._batch_size
        start = idx * batch_size
        stop = min(idx * (batch_size + 1), self._num_elements())

        return self._load_batch(self._idxs[start:stop])

    def _reformat_data(self, vols: Sequence[np.ndarray]):
        return tuple(vols)

    def inference(self, model):
        scan_to_idx_mapping = defaultdict(list)
        for idx, d in enumerate(self._dataset_dicts):
            scan_to_idx_mapping[d["scan_id"]].append(idx)

        scan_ids = sorted(list(scan_to_idx_mapping.keys()))

        for scan_id in scan_ids:
            x, y = self._load_batch(scan_to_idx_mapping[scan_id])
            start = time.perf_counter()
            y_pred = model.predict(x, batch_size=self._batch_size)
            time_elapsed = time.perf_counter() - start
            x, y, y_pred = self._reformat_data((x, y, y_pred))
            yield x, y, y_pred, scan_id, time_elapsed
