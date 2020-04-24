"""Transforms specific to medical image data.

Many transform operations are typically written with the assumption of 2D data.
However, medical images typically have >2 dimensions.

Some transforms implemented in this file are meant to overload transforms in the
`fvcore.transforms` module.
"""
from abc import ABC
from typing import Sequence, Tuple

from fvcore.transforms.transform import Transform
import numpy as np

from medsegpy.config import Config

__all__ = [
    "build_preprocessing",
    "CropTransform",
    "MedTransform",
    "ZeroMeanNormalization"
]


class MedTransform(Transform, ABC):
    """
    Base class for implementations of __deterministic__ transfomations for
    _medical_ image and other data structures. Like the `fvcore.transforms`
    module, there should be a higher-level policy that generates (likely with
    random variations) these transform ops.

    By default, all transforms only handle image and segmentation data types.
    Coordinates, bounding boxes, and polygons are not supported by default.
    However, these methods can be overloaded if generalized methods are
    written for these data types.

    Medical images are seldom in the uint8 format and are not always
    normalized between [0, 1] in the floating point format. Transforms should
    not expect that data is normalized in this format.

    Note, each method may choose to modify the input data in-place for
    efficiency.
    """

    def apply_coords(self, coords: np.ndarray):
        raise NotImplementedError(
            "apply_coords not implemented for MedTransform"
        )

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "apply_box not implemented for MedTransform"
        )

    def apply_polygons(self, polygons: list) -> list:
        raise NotImplementedError(
            "apply_polygon not implemented for MedTransform"
        )


class CropTransform(MedTransform):
    """Crop nD volumes.

    Volumes will be cropped as ``vol[..., z:z+d, y:y+h, x:x+w]``.

    Args:
        coords0 (:obj:`int(s)`): Initial coordinates. In order x,y,z,...
        crop_size (:obj:`int(s)`): Crop size. In order w,h,d,...
    """

    def __init__(
        self,
        coords0: Sequence[int, ...],
        crop_size: Sequence[int, ...],
    ):
        assert len(coords0) == len(crop_size)
        super().__init__()
        window = [slice(c, c+s) for c, s in zip(coords0, crop_size)]
        window.insert(0, Ellipsis)
        self._set_attributes({"window": window})

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """Crop the image(s).

        Args:
            img (ndarray): of shape >=`len(self.window)`

        Returns:
            ndarray: cropped image(s).
        """

        return img[self.window]


class ZeroMeanNormalization(MedTransform):
    """Zero mean unit variance normalization.

    Volumes are dynamically normalized to be zero-mean and unit variance.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray):
        """Crop the image(s).

        Args:
            img (ndarray): of shape NxCxDxHxW, or NxDxHxW or DxHxW.

        Returns:
            ndarray: zero-mean, unit variance image.
        """
        mean = img.mean(axis=(-1, -2, -3), keepdims=True)
        std = img.mean(axis=(-1, -2, -3), keepdims=True)
        return (img - mean) / std

    def apply_segmentation(self, segmentation: np.ndarray):
        """Segmentation should not be normalized."""
        return segmentation


class Windowing(MedTransform):
    """Clip image magnitude between values `lower`, `upper`.

    If multiple lower/upper bound pairs are provided, the output will be stacked
    along the last dimension.
    """
    def __init__(self, bounds: Sequence[Tuple[int, int]]):
        """
        Args:
            bounds: `[(10,20)]` or `[(10,20), (40, 100)]` etc.
        """
        self._set_attributes(locals())
        super().__init__()

    def apply_image(self, img: np.ndarray):
        """Crop the image(s).

        Args:
            img (ndarray): of shape NxCxDxHxW, or NxDxHxW or DxHxW.

        Returns:
            ndarray: zero-mean, unit variance image.
        """
        imgs = []
        bounds = self.bounds
        for l, u in bounds:
            imgs.append(np.clip(img, a_min=l, a_max=u))

        if len(imgs) == 1:
            return imgs[0]
        elif img.shape[-1] == 1:
            return np.concatenate(imgs, axis=-1)
        else:
            return np.stack(imgs, axis=-1)

    def apply_segmentation(self, segmentation: np.ndarray):
        """Segmentation should not be windowed."""
        return segmentation


def build_preprocessing(cfg: Config):
    transforms = []
    for pp in cfg.PREPROCESSING:
        if pp == "Windowing":
            transforms.append(Windowing(cfg.PREPROCESSING_WINDOWS))

    return transforms