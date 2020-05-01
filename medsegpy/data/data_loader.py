import logging
import math
import random
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
from fvcore.common.registry import Registry
from keras import utils as k_utils

from medsegpy.config import Config
from medsegpy.modeling import Model

from .sem_seg_utils import add_background_labels, collect_mask
from .transforms import apply_transform_gens, build_preprocessing

logger = logging.getLogger(__name__)


DATA_LOADER_REGISTRY = Registry("DATA_LOADER")
"""
Registry for data loaders, which can be used with `model.fit_generator()` and
`model.predict_generator()`. The evaluator type should be registered with
dataset_dicts, cfg, and other extra parameters.

The registered object will be called with
`obj(dataset_dicts, cfg, **kwargs)`.
The call should return a :class:`DataLoader` object.
"""

_LEGACY_DATA_LOADER_MAP = {
    ("oai_aug", "oai", "oai_2d", "oai_aug_2d"): "DefaultDataLoader"
}

LEGACY_DATA_LOADER_NAMES = {
    x: v for k, v in _LEGACY_DATA_LOADER_MAP.items() for x in k
}


def build_data_loader(
    cfg: Config, dataset_dicts: List[Dict], **kwargs
) -> "DataLoader":
    """Get data loader based on config `TAG` or name, value.
    """
    name = cfg.TAG
    try:
        data_loader_cls = DATA_LOADER_REGISTRY.get(name)
    except KeyError:
        prev_name = name
        if name in LEGACY_DATA_LOADER_NAMES:
            name = LEGACY_DATA_LOADER_NAMES[name]
            if prev_name != name:
                warnings.warn(
                    "TAG {} is deprecated. Use {} instead".format(
                        prev_name, name
                    )
                )

        data_loader_cls = DATA_LOADER_REGISTRY.get(name)

    return data_loader_cls(cfg, dataset_dicts, **kwargs)


class DataLoader(k_utils.Sequence, ABC):
    """Data loader following :class:`keras.utils.Sequence` API.

    Data loaders load data per batch in the following way:
    1. Collate inputs and outputs
    2. Optionally apply preprocessing

    To avoid changing the order of the base list, we shuffle a list of indices
    and query based on the index.

    Data loaders in medsegpy also have the ability to yield inference results
    per scan (see :meth:`inference`).
    """

    ALIASES = []

    def __init__(
        self,
        cfg: Config,
        dataset_dicts: List[Dict],
        is_test: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
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
        if not self._is_test and self.drop_last:
            return _num_elements // self._batch_size
        else:
            return math.ceil(_num_elements / self._batch_size)

    def _num_elements(self):
        """Number of elements in the data loader."""
        return len(self._dataset_dicts)

    def num_scans(self):
        return len({x["scan_id"] for x in self._dataset_dicts})

    @abstractmethod
    def inference(self, model, **kwargs):
        """Yields dictionaries of inputs, outputs per scan.

        In medical settings, data is often processed per scan, not necessarily
        per example. This distinction is critical. For example, a 2D
        segmentation network may take in 2D slices of a scan as input.
        However, during inference, it is standard to compute metrics on the full
        scan, not individual slices.

        This method should yield scan-specific inputs and outputs as
        dictionaries. The following keys should be in the `input` and `output`
        dictionaries for each scan at minimum.

        Input keys:
            * "scan_id" (str): the scan identifier
            * "x" (ndarray): the raw (unprocessed) input. Shape HxWx...
                If the network takes multiple inputs, each input should
                correspond to a unique key that will be handled by your
                specified evaluator.
            * "scan_XXX" (optional) scan-related parameters that will simplify
                evaluation. e.g. "scan_spacing". MedSegPy evaluators will
                default to scan specific information, if provided. For example,
                if "scan_spacing" is specified, the value specified will
                override the default spacing for the dataset.
            * "subject_id" (optional): the subject identifier for the scan.
                Useful for grouping results by subject.

        Output keys:
            * "time_elapsed" (required): Amount of time required for inference
                on scan.
                This quantity typically includes data loading time as well.
            * "sem_seg_gt_mask" (ndarray): Ground truth binary mask for semantic
                segmentation. Shape HxWx...xC.
                Required for semantic segmentation inference.
            * "sem_seg_pred" (ndarray): Prediction probabilities for semantic
                segmentation. Shape HxWx...xC.
                Required for semantic segmentation inference.
            * "class_gts" (Sequence): classification-related ground truth by
                category id.
            * "class_preds" (Sequence): classification-related predictions
                (probabilities).

        All output keys except "time_elapsed" are optional and task specific.

        Args:
            model: A model to run inference on.
            kwargs: Keyword arguments to `model.predict_generator()`

        Yields:
            dict, dict: Dictionaries of inputs and outputs corresponding to a
                single scan.
        """
        yield {}, {}


@DATA_LOADER_REGISTRY.register()
class DefaultDataLoader(DataLoader):
    """The default data loader functionality in medsegy.

    This class takes a dataset dict in the MedSegPy Dataset format and maps it
    to a format that can be used by the model for semantic segmentation.

    This is the default data loader.

    1. Read the input matrix from "file_name"
    2. Read the ground truth mask matrix from "sem_seg_file_name"
    3. If needed:
        a. Add binary labels for background
    4. Apply :class:`MedTransform` transforms to input and masks.
    5. If training, return input (preprocessed), output.
       If testing, return input (preprocessed), output, input (raw).
       The testing structure is useful for tracking the original input
       without any preprocessing. This return structure does not conflict with
       existing Keras model functionality.
    """

    def __init__(
        self,
        cfg: Config,
        dataset_dicts: List[Dict],
        is_test: bool = False,
        shuffle: bool = True,
        drop_last: bool = False,
        batch_size: int = 1,
    ):
        super().__init__(
            cfg, dataset_dicts, is_test, shuffle, drop_last, batch_size
        )

        self._image_size = cfg.IMG_SIZE
        self._include_background = cfg.INCLUDE_BACKGROUND
        self._num_neighboring_slices = cfg.num_neighboring_slices()
        self._num_classes = cfg.get_num_classes()
        self._transform_gen = build_preprocessing(cfg)
        self._load_masks = True

    def _load_input(self, image_file, sem_seg_file):
        with h5py.File(image_file, "r") as f:
            image = f["data"][:]
        if image.shape[-1] != 1:
            image = image[..., np.newaxis]

        if sem_seg_file:
            with h5py.File(sem_seg_file, "r") as f:
                mask = f["data"][:]

            cat_idxs = self._category_idxs
            mask = collect_mask(mask, index=cat_idxs)
            if self._include_background:
                mask = add_background_labels(mask)
        else:
            mask = None

        return image, mask

    def _load_batch(self, idxs: Sequence[int]):
        dataset_dicts = self._dataset_dicts

        images = []
        masks = []
        for file_idx in idxs:
            dataset_dict = dataset_dicts[file_idx]
            file_name = dataset_dict["file_name"]
            sem_seg_file_name = dataset_dict["sem_seg_file"]

            image, mask = self._load_input(file_name, sem_seg_file_name)

            images.append(image)
            masks.append(mask)

        return np.stack(images, axis=0), np.stack(masks, axis=0)

    def _preprocess(self, inputs: np.ndarray, outputs: np.ndarray):
        img, transforms = apply_transform_gens(self._transform_gen, inputs)
        outputs = transforms.apply_segmentation(outputs)
        return img, outputs

    def __getitem__(self, idx):
        """
        Args:
            idx: Batch index.

        Returns:
            ndarray, ndarray: images NxHxWx(...)x1, masks NxHxWx(...)x1
        """
        batch_size = self._batch_size
        start = idx * batch_size
        stop = min((idx + 1) * batch_size, self._num_elements())

        inputs, outputs = self._load_batch(self._idxs[start:stop])
        inputs_preprocessed, outputs = self._preprocess(inputs, outputs)

        if self._is_test:
            return inputs_preprocessed, outputs, inputs
        else:
            return inputs_preprocessed, outputs

    def _restructure_data(self, vols: Sequence[np.ndarray]):
        """By default the batch dimension is moved to be the third dimension.

        Args:
            vols (ndarrays): Shapes of NxHxWx...

        Returns:
            vols (ndarrays): Shapes of HxWxNx...
        """
        new_vols = []
        for v in vols:
            axes = (1, 2, 0)
            if v.ndim > 3:
                axes = axes + tuple(i for i in range(3, v.ndim))
            new_vols.append(v.transpose(axes))
        vols = (np.squeeze(v) for v in new_vols)
        return tuple(vols)

    def inference(self, model: Model, **kwargs):
        scan_to_dict_mapping = defaultdict(list)
        for d in self._dataset_dicts:
            scan_to_dict_mapping[d["scan_id"]].append(d)

        scan_ids = sorted(scan_to_dict_mapping.keys())
        dataset_dicts = self._dataset_dicts

        for scan_id in scan_ids:
            self._dataset_dicts = scan_to_dict_mapping[scan_id]

            start = time.perf_counter()
            x, y, preds = model.inference_generator(self, **kwargs)
            time_elapsed = time.perf_counter() - start

            x, y, preds = self._restructure_data((x, y, preds))

            input = {"x": x, "scan_id": scan_id}
            scan_params = {
                k: v for k, v in self._dataset_dicts[0].items()
                if isinstance(k, str) and k.startswith("scan")
            }
            input.update(scan_params)

            output = {
                "y_pred": preds,
                "y_true": y,
                "time_elapsed": time_elapsed,
            }

            yield input, output

        self._dataset_dicts = dataset_dicts
