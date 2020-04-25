"""Transforms specific to medical image data.

Many transform operations are typically written with the assumption of 2D data.
However, medical images typically have >2 dimensions.

Some transforms implemented in this file are meant to overload transforms in the
`fvcore.transforms` module.
"""
from abc import ABC, abstractmethod
import inspect
from typing import Sequence, Tuple, Callable, TypeVar

import numpy as np

from medsegpy.config import Config

__all__ = [
    "build_preprocessing",
    "MedTransform",
    "TransformList",
    "CropTransform",
    "ZeroMeanNormalization"
]


class MedTransform(ABC):
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
    def _set_attributes(self, params: list = None):
        """
        Set attributes from the input list of parameters.

        Args:
            params (list): list of parameters.
        """

        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img: np.ndarray):
        """
        Apply the transform on an image.

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: image after apply the transformation.
        """
        pass

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
            or bool dtype.

        Returns:
            ndarray: segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    @classmethod
    def register_type(cls, data_type: str, func: Callable):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.

        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.

        Examples:

        .. code-block:: python

            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type("voxel", func)

            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        assert callable(func), (
            "You can only register a callable to a MedTransform. "
            "Got {} instead.".format(
                func
            )
        )
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(
                str(argspec)
            )
        )
        setattr(cls, "apply_" + data_type, func)


_T = TypeVar("_T")


# pyre-ignore-all-errors
class TransformList:
    """
    Maintain a list of transform operations which will be applied in sequence.
    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms: list):
        """
        Args:
            transforms (list[Transform]): list of transforms to perform.
        """
        super().__init__()
        for t in transforms:
            assert isinstance(t, MedTransform), t
        self.transforms = transforms

    def _apply(self, x: _T, meth: str) -> _T:
        """
        Apply the transforms on the input.
        Args:
            x: input to apply the transform operations.
            meth (str): meth.
        Returns:
            x: after apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattr__(self, name: str):
        """
        Args:
            name (str): name of the attribute.
        """
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        raise AttributeError(
            "TransformList object has no attribute {}".format(name)
        )

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = (
            other.transforms if isinstance(other, TransformList) else [other]
        )
        return TransformList(self.transforms + others)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = (
            other.transforms if isinstance(other, TransformList) else [other]
        )
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = (
            other.transforms if isinstance(other, TransformList) else [other]
        )
        return TransformList(others + self.transforms)

    def __len__(self) -> int:
        """
        Returns:
            Number of transforms contained in the TransformList.
        """
        return len(self.transforms)


class CropTransform(MedTransform):
    """Crop nD volumes.

    Volumes will be cropped as ``vol[..., z:z+d, y:y+h, x:x+w]``.

    Args:
        coords0 (:obj:`int(s)`): Initial coordinates. In order x,y,z,...
        crop_size (:obj:`int(s)`): Crop size. In order w,h,d,...
    """

    def __init__(
        self, coords0: Sequence[int], crop_size: Sequence[int],
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
