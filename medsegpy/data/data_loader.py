import os
import logging
import math
import random
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Sequence

import h5py
import keras.backend as K
import numpy as np
from fvcore.common.registry import Registry
from keras import utils as k_utils
from tqdm import tqdm

from medsegpy.config import Config
from medsegpy.modeling import Model
from medsegpy.utils import env

from .data_utils import add_background_labels, collect_mask, compute_patches
from .transforms import apply_transform_gens, build_preprocessing

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)

DATA_LOADER_REGISTRY = Registry("DATA_LOADER")
DATA_LOADER_REGISTRY.__doc__ = """
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

        This method does the following:
            1. Loads dataset dicts corresponding to a scan
            2. Structures data from these dicts
            3. Runs predictions on the structured data
            4. Restructures inputs. Images/volumes are restructured to HxWx...
                Segmentation masks and predictions are restructured to
                HxWx...xC.
            5. Yield input, output dictionaries for the scan. Yielding continues
                until all scans have been processed.

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
            * "y_true" (ndarray): Ground truth binary mask for semantic
                segmentation. Shape HxWx...xC.
                Required for semantic segmentation inference.
            * "y_pred" (ndarray): Prediction probabilities for semantic
                segmentation. Shape HxWx...xC.
                Required for semantic segmentation inference.

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

    This class takes a dataset dict in the MedSegPy 2D Dataset format and maps
    it to a format that can be used by the model for semantic segmentation.

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

        self._include_background = cfg.INCLUDE_BACKGROUND
        self._num_classes = cfg.get_num_classes()
        self._transform_gen = build_preprocessing(cfg)
        self._cached_data = None

    def _load_input(self, dataset_dict):
        image_file = dataset_dict["file_name"]
        sem_seg_file = dataset_dict.get("sem_seg_file", None)

        if self._cached_data is not None:
            image, mask = self._cached_data[(image_file, sem_seg_file)]
        else:
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
        """
        TODO: run test to determine if casting inputs/outputs is required.
        """
        dataset_dicts = self._dataset_dicts

        images = []
        masks = []
        for file_idx in idxs:
            dataset_dict = dataset_dicts[file_idx]
            image, mask = self._load_input(dataset_dict)

            images.append(image)
            masks.append(mask)

        return (
            np.stack(images, axis=0).astype(K.floatx()),
            np.stack(masks, axis=0).astype(K.floatx()),
        )

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

        TODO: Change signature to specify if it is a segmentation volume or
        image volume. Downstream data loaders need to distinguish between the
        two (i.e. 2.5D networks).

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

        workers = kwargs.pop("workers", self._cfg.NUM_WORKERS)
        use_multiprocessing = kwargs.pop("use_multiprocessing", workers > 1)
        for scan_id in scan_ids:
            self._dataset_dicts = scan_to_dict_mapping[scan_id]

            start = time.perf_counter()
            if not isinstance(model, Model):
                if not env.is_tf2():
                    raise ValueError("model must be a medsegpy.modeling.model.Model for TF1.0")
                x, y, preds = Model.inference_generator_static(
                    model,
                    self,
                    workers=workers,
                    use_multiprocessing=use_multiprocessing,
                    **kwargs
                )
            else:
                x, y, preds = model.inference_generator(
                    self,
                    workers=workers,
                    use_multiprocessing=use_multiprocessing,
                    **kwargs
                )
            time_elapsed = time.perf_counter() - start

            x, y, preds = self._restructure_data((x, y, preds))

            input = {"x": x, "scan_id": scan_id}
            scan_params = {
                k: v
                for k, v in self._dataset_dicts[0].items()
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


_SUPPORTED_PADDING_MODES = (
    "constant",
    "edge",
    "reflect",
    "symmetric",
    "warp",
    "empty",
)


@DATA_LOADER_REGISTRY.register()
class PatchDataLoader(DefaultDataLoader):
    """
    This data loader pre-computes patch locations and padding based on
    patch size (`cfg.IMG_SIZE`), pad type (`cfg.IMG_PAD_MODE`), pad size
    (`cfg.IMG_PAD_SIZE`), and stride (`cfg.IMG_STRIDE`) parameters specified
    in the config.

    Assumptions:
        * all dataset dictionaries have the same image dimensions
        * "image_size" in dataset dict
    """

    def __init__(
        self,
        cfg: Config,
        dataset_dicts: List[Dict],
        is_test: bool = False,
        shuffle: bool = True,
        drop_last: bool = False,
        batch_size: int = 1,
        use_singlefile: bool = False
    ):
        # Create patch elements from dataset dict.
        # TODO: change pad/patching based on test/train
        self._use_singlefile = use_singlefile
        expected_img_dim = len(dataset_dicts[0]["image_size"])
        img_dim = len(cfg.IMG_SIZE)
        self._add_dim = False
        if img_dim > expected_img_dim:
            assert img_dim - expected_img_dim == 1
            patch_size = cfg.IMG_SIZE[:-1]
            self._add_dim = True
        elif len(cfg.IMG_SIZE) == expected_img_dim:
            patch_size = cfg.IMG_SIZE
        else:
            extra_dims = (1,) * (expected_img_dim - img_dim)
            patch_size = tuple(cfg.IMG_SIZE) + extra_dims

        self._patch_size = patch_size
        self._pad_mode = cfg.IMG_PAD_MODE
        if self._pad_mode not in _SUPPORTED_PADDING_MODES:
            raise ValueError(
                "pad mode {} not supported".format(cfg.IMG_PAD_MODE)
            )
        pad_size = cfg.IMG_PAD_SIZE if cfg.IMG_PAD_SIZE else None
        stride = cfg.IMG_STRIDE if cfg.IMG_STRIDE else (1,) * len(patch_size)

        dd_patched = []
        for dd in dataset_dicts:
            patches = compute_patches(
                dd["image_size"], self._patch_size, pad_size, stride
            )
            if len(patches) == 0:
                logger.warn(f"Dropping {dd['scan_id']} - no patches found.")
            for patch, pad in patches:
                dataset_dict = dd.copy()
                dataset_dict.update({"_patch": patch, "_pad": pad})
                dd_patched.append(dataset_dict)

        super().__init__(
            cfg, dd_patched, is_test, shuffle, drop_last, batch_size
        )

        self._preload_data = cfg.PRELOAD_DATA
        self._cached_data = None
        self._f = None

        if self._use_singlefile:
            self._singlefile_fp = dataset_dicts[0]["singlefile_path"]

        if self._preload_data:
            if threading.current_thread() is not threading.main_thread():
                raise ValueError(
                    "Data pre-loading can only be done on the main thread."
                )
            logger.info("Pre-loading data...")
            self._cached_data = self._load_all_data(
                dataset_dicts, cfg.NUM_WORKERS
            )

    def __del__(self):
        if hasattr(self, "_f") and self._f is not None:
            self._f.close()

    def __getitem__(self, idx):
        """
        Args:
            idx: Batch index.

        Returns:
            ndarray, ndarray: images NxHxWx(...)x1, masks NxHxWx(...)x1
        """
        if self._use_singlefile and self._f is None:
            self._f = h5py.File(self._singlefile_fp, "r")
        batch_size = self._batch_size
        start = idx * batch_size
        stop = min((idx + 1) * batch_size, self._num_elements())

        inputs, outputs = self._load_batch(self._idxs[start:stop])
        inputs_preprocessed, outputs = self._preprocess(inputs, outputs)

        if self._is_test:
            return inputs_preprocessed, outputs, inputs
        else:
            return inputs_preprocessed, outputs

    def _load_all_data(self, dataset_dicts, num_workers: int = 1) -> Dict:
        """
        We assume that that the tuple `("file_name", "sem_seg_file")`
        is sufficient for determining the uniqueness of each base dataset
        dictionary.
        """

        def _load(dataset_dict):
            image, mask = self._load_patch(dataset_dict, skip_patch=True)
            if set(np.unique(mask)) == {0, 1}:
                mask = mask.astype(np.bool)
            return {"image": image, "mask": mask}

        cache = [_load(dd) for dd in tqdm(dataset_dicts)]
        cache = {
            (dd["file_name"], dd["sem_seg_file"]): x
            for dd, x in zip(dataset_dicts, cache)
        }
        return cache

    def _load_patch(self, dataset_dict, skip_patch: bool = False, img_key=None, seg_key=None):
        image_file = dataset_dict["file_name"]
        sem_seg_file = dataset_dict.get("sem_seg_file", None)
        patch = Ellipsis if skip_patch else dataset_dict["_patch"]

        mask = None

        is_img_seg_file_same = image_file == sem_seg_file
        if seg_key is None:
            seg_key = "seg" if is_img_seg_file_same else "data"
        if img_key is None:
            img_key = "volume" if is_img_seg_file_same else "data"

        # Load data from one h5 file if self._use_singlefile
        if not self._use_singlefile:
            f = h5py.File(image_file, "r")
            image = f[img_key][patch]  # HxWxDx...
            if sem_seg_file and is_img_seg_file_same:
                mask = f[seg_key][patch]  # HxWxDx...xC
            else:
                s = h5py.File(sem_seg_file, "r") 
                mask = s[seg_key][patch]
                s.close()
            f.close()
        else:
            image = self._f[image_file][img_key][patch]
            if sem_seg_file:
                mask = self._f[image_file][seg_key][patch]

        if mask is not None:
            cat_idxs = self._category_idxs
            mask = collect_mask(mask, index=cat_idxs)
            if self._include_background:
                mask = add_background_labels(mask)

        return image, mask

    def _load_input(self, dataset_dict):
        if self._cached_data is not None:
            patch = dataset_dict["_patch"]
            image_file = dataset_dict["file_name"]
            sem_seg_file = dataset_dict.get("sem_seg_file", None)
            data = self._cached_data[(image_file, sem_seg_file)]
            image, mask = data["image"], data["mask"]
            image, mask = image[patch], mask[patch]
        else:
            image, mask = self._load_patch(dataset_dict)

        pad = dataset_dict["_pad"]
        if pad is not None:
            image = np.pad(image, pad, self._pad_mode)
            if mask is not None:
                mask = np.pad(mask, tuple(pad) + ((0, 0),), self._pad_mode)

        if self._add_dim:
            image = image[..., np.newaxis]
            # mask = mask[..., np.newaxis, :]

        return image, mask

    def _restructure_data(self, vols_patched: Sequence[np.ndarray]):
        """By default the batch dimension is moved to be the third dimension.

        This method assumes that `self._dataset_dicts` is limited to dataset
        dictionaries for only one scan. It also assumes that the order of
        each patch in `vols_patches` is ordered based on the dataset dictionary.

        Args:
            vols_patched (ndarrays): Each has shape of NxP1xP2x...

        Returns:
            vols (ndarrays): Shapes of HxWxDx...
        """
        assert self._is_test

        image_size = self._dataset_dicts[0]["image_size"]
        coords = [dd["_patch"] for dd in self._dataset_dicts]

        num_patches = vols_patched[0].shape[0]
        assert len(coords) == num_patches, "{} patches, {} coords".format(
            num_patches, len(coords)
        )
        # num_vols = len(vols_patched)
        # TODO: fix in case that v.shape[-1] is not actually a channel dimension
        new_vols = [
            np.zeros(tuple(image_size) + (v.shape[-1],)) for v in vols_patched
        ]  # VxNxHxWx...

        for idx, c in enumerate(coords):
            for vol_id in range(len(new_vols)):
                # Hacky solution to handle extra axis dimension, if exists.
                x = vols_patched[vol_id][idx]
                if x.ndim == new_vols[vol_id][c].ndim - 1:
                    x = x[..., np.newaxis, :]
                new_vols[vol_id][c] = x

        return tuple(new_vols)


@DATA_LOADER_REGISTRY.register()
class N5dDataLoader(PatchDataLoader):
    """n.5D data loader.

    Use this for 2.5D, 3.5D, etc. implementations.
    Currently only last dimension is supported as the channel dimension.
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
        expected_img_dim = len(dataset_dicts[0]["image_size"])
        img_dim = len(cfg.IMG_SIZE)
        if img_dim != expected_img_dim:
            raise ValueError(
                "Data has {} dimensions. cfg.IMG_SIZE is {} dimensions".format(
                    expected_img_dim, img_dim
                )
            )
        if cfg.IMG_SIZE[-1] % 2 != 1:
            raise ValueError("channel dimension must be odd")

        super().__init__(
            cfg, dataset_dicts, is_test, shuffle, drop_last, batch_size
        )

    def _load_input(self, dataset_dict):
        image, mask = super()._load_input(dataset_dict)
        dim = mask.shape[-2]
        mask = mask[..., dim // 2, :]
        return image, mask


@DATA_LOADER_REGISTRY.register()
class S25dDataLoader(DefaultDataLoader):
    """Special case of 2.5D data loader compatible with 2D MedSegPy data format.

    Each dataset dict should represent a slice and must have the additional
    keys:
    - "slice_id" (int): Slice id (1-indexed) that the dataset corresponds to.
    - "scan_num_slices" (int): Number of total slices in the scan that the
        dataset dict is derived from

    Padding is automatically applied to ensure all slices are considered.

    This is a temporary solution until the slow loading speeds of the
    :class:`N5dDataLoader` are properly debugged.
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
        self._window = cfg.IMG_SIZE[-1]
        assert len(cfg.IMG_SIZE) == 3
        assert self._window % 2 == 1
        self._pad_mode = cfg.IMG_PAD_MODE

        # Create a mapping from scan_id to list of dataset dicts in order of
        # slice.
        # TODO: remove copying dictionaries if runtime speed issues found
        mapping = defaultdict(list)
        sorted_dataset_dicts = sorted(
            dataset_dicts, key=lambda d: (d["scan_id"], d["slice_id"])
        )
        for dd in sorted_dataset_dicts:
            mapping[dd["scan_id"]].append(dd)
        for scan_id, dds in mapping.items():
            slice_order = [dd["slice_id"] for dd in dds]
            assert sorted(slice_order) == slice_order, (
                "Error in sorting dataset dictionaries "
                "for scan {} by slice id".format(scan_id)
            )

        self._scan_to_dicts = mapping

        super().__init__(
            cfg, dataset_dicts, is_test, shuffle, drop_last, batch_size
        )

    def _load_input(self, dataset_dict):
        """Find dataset dicts corresponding to flanking/neighboring slices and
        load.
        """
        slice_id = dataset_dict["slice_id"]  # 1-indexed
        scan_id = dataset_dict["scan_id"]
        total_num_slices = dataset_dict["scan_num_slices"]  # 1-indexed

        num_flank_slices = self._window // 2
        l_pad = r_pad = 0
        if total_num_slices - slice_id < num_flank_slices:
            # Right pad the volume.
            r_pad = num_flank_slices - (total_num_slices - slice_id)
        if slice_id - num_flank_slices <= 0:
            # Left pad the volume.
            l_pad = num_flank_slices - slice_id + 1
        pad = ((0, 0), (0, 0), (l_pad, r_pad)) if l_pad or r_pad else None

        # Load images for neighboring slices.
        idx = slice_id - 1
        start = max(0, idx - num_flank_slices)
        end = min(total_num_slices, idx + 1 + num_flank_slices)
        dataset_dicts = self._scan_to_dicts[scan_id][start:end]

        images = []
        for dd in dataset_dicts:
            image_file = dd["file_name"]
            with h5py.File(image_file, "r") as f:
                image = f["data"][:]
            if image.shape[-1] == 1:
                image = np.squeeze(image)
            images.append(image)
        image = np.stack(images, axis=-1)
        if pad is not None:
            image = np.pad(image, pad, self._pad_mode)

        # Load segmentation only for center slice
        sem_seg_file = dataset_dict.get("sem_seg_file", None)
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

    def _restructure_data(self, vols: Sequence[np.ndarray]):
        """By default the batch dimension is moved to be the third dimension.

        This method assumes that `self._dataset_dicts` is limited to dataset
        dictionaries for only one scan. It also assumes that the order of
        each patch in `vols_patches` is ordered based on the dataset dictionary.

        Args:
            vols_patched (ndarrays): Each has shape of NxP1xP2x...

        Returns:
            vols (ndarrays): Shapes of HxWxDx...
        """
        x, y, preds = vols
        x = x[..., self._window // 2]
        assert x.ndim == 3, "NxHxW"
        return super()._restructure_data((x, y, preds))
