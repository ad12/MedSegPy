# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File: transformer.py
from typing import Sequence, Tuple, Union

import inspect
import numpy as np
import pprint
from abc import ABCMeta, abstractmethod
from fvcore.transforms.transform import (
    Transform,
    TransformList,
)

from .transform import MedTransform, CropTransform

__all__ = [
    "RandomCrop",
    "TransformGen",
    "apply_transform_gens",
]


def check_dtype(img: np.ndarray):
    assert isinstance(img, np.ndarray), (
        "[TransformGen] Needs an numpy array, but got a {}!".format(
            type(img)
        )
    )
    assert not isinstance(img.dtype, np.integer) or img.dtype == np.uint8, (
        "[TransformGen] Got image of type {}, "
        "use uint8 or floating points instead!".format(
            img.dtype
        )
    )
    assert img.ndim > 2, img.ndim


class TransformGen(metaclass=ABCMeta):
    """
    TransformGen takes an array of type float as input.

    It creates a :class:`Transform` based on the given image, sometimes with
    randomness. The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class is that the image itself is sufficient to
    instantiate a transform. When this assumption is not true, you need to
    create the transforms by your own.

    A list of `TransformGen` can be applied with :func:`apply_transform_gens`.
    """

    def _init(self, params = None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low = 1.0, high = None, size = None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyTransformGen(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(
                        name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


class RandomCrop(TransformGen):
    """Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size: Tuple[float]):
        """
        Args:
            crop_type (str): One of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): The relative ratio or absolute pixels of
                width, height, (...)
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._init(locals())

    def get_transform(self, img: np.ndarray):
        """
        Args:
            img: a `(...,)H,W` image.
        """
        cdim = len(self.crop_size)
        image_size = img.shape[-cdim:][::-1]
        crop_size = self.get_crop_size(image_size)
        assert all([
            dim >= crop_dim for dim, crop_dim in zip(image_size, crop_size)
        ]), (
            "Shape computation in {} has bugs.".format(self)
        )

        # Format: x,y,z,... and w,h,d,...
        image_size = image_size[::-1]
        crop_size = crop_size[::-1]
        coords0 = [np.random.randint(img_dim - crop_dim + 1)
                   for img_dim, crop_dim in zip(image_size, crop_size)]
        return CropTransform(coords0, crop_size)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): width, height, (...)

        Returns:
            crop_size (tuple): width, height, (...) in absolute pixels
        """
        if self.crop_type == "relative":
            crop_size = self.crop_size
            return tuple([
                int(dim * crop_dim + 0.5)
                for dim, crop_dim in zip(image_size, crop_size)
            ])
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            cdim = len(len(crop_size))
            crop_size = crop_size + np.random.rand(cdim) * (1 - crop_size)
            return tuple([
                int(dim * crop_dim + 0.5)
                for dim, crop_dim in zip(image_size, crop_size)
            ])
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


def apply_transform_gens(
    transform_gens: Sequence[Union[TransformGen, MedTransform]],
    img: np.ndarray,
) -> Tuple[np.ndarray, TransformList]:
    """
    Apply a list of :class:`TransformGen` on the input image, and
    returns the transformed image and a list of transforms.

    We cannot simply create and return all transforms without
    applying it to the image, because a subsequent transform may
    need the output of the previous one.

    Args:
        transform_gens (list): list of :class:`TransformGen` instance to
            be applied.
        img (ndarray): uint8 or floating point images with 1 or 3 channels.

    Returns:
        ndarray: the transformed image
        TransformList: contain the transforms that's used.
    """
    for g in transform_gens:
        assert isinstance(g, TransformGen), g

    check_dtype(img)

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(img) if isinstance(g, TransformGen) else g
        assert isinstance(
            tfm, Transform
        ), "TransformGen {} must return an instance of Transform! Got {} instead".format(
            g, tfm)
        img = tfm.apply_image(img)
        tfms.append(tfm)
    return img, TransformList(tfms)
