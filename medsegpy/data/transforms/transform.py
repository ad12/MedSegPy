"""Transforms specific to medical image data.

Many transform operations are typically written with the assumption of 2D data.
However, medical images typically have >2 dimensions.

Some transforms implemented in this file are meant to overload transforms in the
`fvcore.transforms` module.
"""
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Sequence, Tuple, TypeVar

import numpy as np

__all__ = [
    "MedTransform",
    "TransformList",
    "CropTransform",
    "ZeroMeanNormalization",
    "FillRegionsWithValue",
    "Swap2DPatches",
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

    def _set_attributes(self, params: dict = None):
        """
        Set attributes from the input list of parameters.

        Args:
            params (list): list of parameters.
        """

        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def state_dict(self, ignore) -> Dict[str, Any]:
        return dict(self.__dict__)

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
        assert callable(
            func
        ), "You can only register a callable to a MedTransform. " "Got {} instead.".format(func)
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(str(argspec))
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
        raise AttributeError("TransformList object has no attribute {}".format(name))

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(self.transforms + others)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
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

    def __init__(self, coords0: Sequence[int], crop_size: Sequence[int]):
        assert len(coords0) == len(crop_size)
        super().__init__()
        window = [slice(c, c + s) for c, s in zip(coords0, crop_size)] + [slice()]
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

    Args:
        axis (int|Tuple[int]): Axis or axes along which the statistics
            (means and standard deviations) are computed. The default is
            to compute the statistics over the spatial dimensions only
            (i.e. batch and channel dims are ignored).
    """

    def __init__(self, axis=None):
        super().__init__()
        self._set_attributes({"axis": axis})

    def apply_image(self, img: np.ndarray):
        """Crop the image(s).

        Args:
            img (ndarray): of shape `CxHxWx...`

        Returns:
            ndarray: zero-mean, unit variance image.
        """
        axis = self.axis
        if axis is None:
            axis = tuple(range(1, img.ndim))
        mean = img.mean(axis=axis, keepdims=True)
        std = img.mean(axis=axis, keepdims=True)
        return (img - mean) / std

    def apply_segmentation(self, segmentation: np.ndarray):
        """Segmentation should not be normalized."""
        return segmentation


class AffineNormalization(MedTransform):
    """Affine normalization by ``scale`` and ``bias``.

    Image ``x`` will be normalized as ``x = (x - bias) / scale``.
    If ``bias`` or ``scale`` are 1D vectors, they will be applied along
    the channel dimension.

    Args:
        scale (float | array-like): Scaling parameter.
            If this is 1D, it will be applied along the channel dimension.
        bias (float | array-like): Bias parameter.
            If this is 1D, it will be applied along the channel dimension.
    """

    def __init__(self, scale=1.0, bias=0.0):
        super().__init__()
        self._set_attributes({"scale": scale, "bias": bias})

    def _broadcast_param(self, param, ndim):
        param = np.asarray(param)
        if param.ndim == 1:
            param = param.reshape((param.shape[0],) + (1,) * (ndim - 2))
        return param

    def apply_image(self, img: np.ndarray):
        """Crop the image(s).

        Args:
            img (ndarray): of shape `NxCx...`

        Returns:
            ndarray: zero-mean, unit variance image.
        """
        scale = self._broadcast_param(self.scale, img.ndim)
        bias = self._broadcast_param(self.bias, img.ndim)
        return (img - bias) / scale

    def apply_segmentation(self, segmentation: np.ndarray):
        """Segmentation should not be normalized."""
        return segmentation


class FlipTransform(MedTransform):
    """Flip image along axis or axes.

    Input images and segmentations will be flipped along ``axis``.
    In most cases, these axes should correspond to spatial dimensions.
    If a channel dimension is specified, the segmentation channels will
    also flip.

    Args:
        axis (int|Tuple[int]): Axis or axes along which to flip.
            Defaults to no flipping.
    """

    def __init__(self, axis=None) -> None:
        super().__init__()
        self._set_attributes({"axis": axis})

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """Crop the image(s).

        Args:
            img (ndarray): of shape >=`len(self.window)`

        Returns:
            ndarray: cropped image(s).
        """
        axis = self.axis
        if not axis:
            return img
        return np.flip(img, axis=self.axis)


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
        for lower, upper in bounds:
            imgs.append(np.clip(img, a_min=lower, a_max=upper))

        if len(imgs) == 1:
            return imgs[0]
        elif img.shape[-1] == 1:
            return np.concatenate(imgs, axis=-1)
        else:
            return np.stack(imgs, axis=-1)

    def apply_segmentation(self, segmentation: np.ndarray):
        """Segmentation should not be windowed."""
        return segmentation


class FillRegionsWithValue(MedTransform):
    """Fills rectangular regions of an image with a constant value.

    This transform was included to help with the implementation of Coarse
    Dropout, located in "medsegpy/data/transforms/transform_gen.py".

    If a stack of images is provided as input to this transform, the
    transform will be applied independently for each image. As a result,
    the regions chosen for one image may be different from the regions
    chosen for another image.

    The implementation is adapted from a similar function in:
    https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py.
    """

    def __init__(self, hole_mask: np.ndarray, fill_value: float = 0.0):
        """
        Args:
            hole_mask: A mask, the same size as the image, indicating which
                pixels should be filled with `fill_value`.
            fill_value: The value used to fill in each hole.
        """
        self._set_attributes(locals())
        super().__init__()

    def apply_image(self, img: np.ndarray):
        """Applies the transform to an image array, which may consist of a
        stack of images.

        If the images have > 1 channel (the last dimension of the input
        image array is > 1), the same holes will be filled in for
        all channels.

        Args:
            img: A N x H x W x C array, containing N images of height H,
                    width W, and consisting of C channels.

        Returns:
            img_filled: A N x H x W x C array, containing the transformed
                image.
        """
        assert img.shape == self.hole_mask.shape, (
            f"Shape of 'hole_mask' ({self.hole_mask.shape}) must match shape "
            f"of 'image' ({img.shape})"
        )
        img_filled = img.copy()
        img_filled[np.array(self.hole_mask, dtype=bool)] = self.fill_value
        return img_filled

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Segmentation should not be modified."""
        return segmentation


class Swap2DPatches(MedTransform):
    """Swaps pairs of patches in an image.
    """

    def __init__(self, patch_pairs):
        """
        Args:
            patch_pairs: A list of dictionaries, where each dictionary has
                            the following structure:

                            {"patch_1": (x1, y1, x2, y2),
                             "patch_2": (x1, y1, x2, y2)}

                            (x1, y1) is the coordinate of the top-left
                            corner of the patch, and (x2, y2) is the
                            coordinate of the bottom-right corner of the
                            patch.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        """Applies the transform to an image array, which may consist of a
        stack of images.

        If the input image array consists of a stack of images (i.e. the
        first dimension is > 1), the appropriate pair of patches will be
        swapped based on the index of the image in the input image array.
        For example, the patches to be swapped for the first image
        (index 0 in the first dimension of the input image array) are
        located index 0 of "patch_pairs".

        If the images have > 1 channel (the last dimension of the input
        image array is > 1), the same pairs of patches will be swapped
        for all channels.

        Args:
            img: A N x H x W x C array, containing N images of height H,
                    width W, and consisting of C channels.

        Returns:
            img_modified: A N x H x W x C array containing the modified
                            image.
        """
        assert img.shape[0] == len(
            self.patch_pairs
        ), "Number of images does not equal the number of patch pairs!"
        img_modified = img.copy()
        for i, patch_pair in enumerate(self.patch_pairs):
            tl = patch_pair[0]
            br = patch_pair[1]
            for num_pair in range(tl.shape[-1]):
                # Get coordinates of both patches
                x1_p1, y1_p1 = tl[:, 0, num_pair]
                x2_p1, y2_p1 = br[:, 0, num_pair]
                x1_p2, y1_p2 = tl[:, 1, num_pair]
                x2_p2, y2_p2 = br[:, 1, num_pair]

                # Swap patches
                tmp_img_patch = np.copy(img_modified[i, y1_p1:y2_p1, x1_p1:x2_p1, :])
                img_modified[i, y1_p1:y2_p1, x1_p1:x2_p1, :] = np.copy(
                    img_modified[i, y1_p2:y2_p2, x1_p2:x2_p2, :]
                )
                img_modified[i, y1_p2:y2_p2, x1_p2:x2_p2, :] = tmp_img_patch

        return img_modified

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Segmentation should not be modified."""
        return segmentation


class NoOpTransform(MedTransform):
    """
    A transform that does nothing.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def inverse(self) -> MedTransform:
        return self

    def __getattr__(self, name: str):
        if name.startswith("apply_"):
            return lambda x: x
        raise AttributeError("NoOpTransform object has no attribute {}".format(name))
