# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File: transformer.py
import logging
import importlib
import inspect
import pprint
import tqdm
from abc import ABCMeta, abstractmethod
from medsegpy.config import Config
from medsegpy.data.data_utils import generate_poisson_disc_mask
from typing import Sequence, Tuple, Union

import numpy as np

from .transform import (
    CropTransform,
    MedTransform,
    FillRegionsWithValue,
    Swap2DPatches,
    TransformList,
)

__all__ = ["RandomCrop", "TransformGen", "apply_transform_gens", "build_preprocessing"]


# Create custom handler for tqdm
# Copied from "https://stackoverflow.com/questions/38543506/
# change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit"
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())


def check_dtype(img: np.ndarray):
    assert isinstance(img, np.ndarray), "[TransformGen] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or img.dtype == np.uint8, (
        "[TransformGen] Got image of type {}, "
        "use uint8 or floating points instead!".format(img.dtype)
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

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
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
                    "Default __repr__ only works if attributes match "
                    "the constructor.".format(name)
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
    """Randomly crop a subimage out of an image."""

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
        assert all(
            dim >= crop_dim for dim, crop_dim in zip(image_size, crop_size)
        ), "Shape computation in {} has bugs.".format(self)

        # Format: x,y,z,... and w,h,d,...
        image_size = image_size[::-1]
        crop_size = crop_size[::-1]
        coords0 = [
            np.random.randint(img_dim - crop_dim + 1)
            for img_dim, crop_dim in zip(image_size, crop_size)
        ]
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
            return tuple(int(dim * crop_dim + 0.5) for dim, crop_dim in zip(image_size, crop_size))
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            cdim = len(len(crop_size))
            crop_size = crop_size + np.random.rand(cdim) * (1 - crop_size)
            return tuple(int(dim * crop_dim + 0.5) for dim, crop_dim in zip(image_size, crop_size))
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class CoarseDropout(TransformGen):
    """Randomly select rectangular regions in images and fill them with a
    constant value.

    The rectangular regions are called "holes" in this implementation. All
    rectangular regions will be 2-D for now.

    Ideally, the constant value chosen should be the mean of the image
    intensity distribution, such that this modification does not significantly
    alter the intensity distribution. Significant differences in the input
    intensity distribution between the training set and test set could
    negatively impact the performance of a machine learning model.

    This implementation is adapted from the implementation of coarse dropout
    in:
    https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py.
    """

    def __init__(
        self,
        max_holes: int,
        max_height: int,
        max_width: int,
        min_holes: int = None,
        min_height: int = None,
        min_width: int = None,
        img_shape: Sequence[int] = (512, 512),
        max_perc_area_to_remove: float = 0.25,
        fill_value: float = 0.0,
        sampling_pattern: str = "uniform",
        num_precompute: int = 100,
    ):
        """
        Args:
            max_holes: The maximum number of holes to drop out.
            max_height: The maximum height of each hole.
            max_width: The maximum width of each hole.
            min_holes: The minimum number of holes to drop out.
                            If None, the maximum number of holes will
                        be dropped out.
            min_height: The minimum height of each hole.
                            If None, each hole will have maximum height.
            min_width: The minimum width of each hole.
                            If None, each hole will have maximum width.
            img_shape: The height and width of each image.
            max_perc_area_to_remove: The maximum fraction of the image that
                                    will be removed and filled with a constant
                                    value. This is useful to prevent too
                                    much of the image from being modified.
            fill_value: The value used to fill in each hole.
            sampling_pattern: The type of sampling pattern to use when
                                selecting random patches.
                Possible values: "uniform", "poisson"
            num_precompute: The number of masks to precompute.
        """
        super().__init__()
        if min_holes is None:
            min_holes = max_holes
        if min_height is None:
            min_height = max_height
        if min_width is None:
            min_width = max_width
        self._init(locals())

        # Precompute masks
        self.precomputed_masks = []
        if sampling_pattern == "poisson":
            assert min_height == max_height == min_width == max_width, (
                "Only square patches are allowed if sampling pattern is " "'poisson'"
            )

            hole_size = min_width
            # Check max_perc_area_to_remove
            #   - Assuming a hexagonal packing of circles will output the
            #       most number of samples when using Poisson disc sampling.
            #   - From https://mathworld.wolfram.com/CirclePacking.html:
            #       Max packing density when using hexagonal packing is
            #       pi / (2 * sqrt(3))
            max_pos_area = (img_shape[0] * img_shape[1]) * (np.pi / (2 * np.sqrt(3)))
            max_num_patches = max_pos_area // ((np.pi / 2) * (hole_size ** 2))
            max_pos_perc_area = np.round(
                (max_num_patches * (hole_size ** 2)) / (img_shape[0] * img_shape[1]), decimals=3
            )
            if max_perc_area_to_remove >= max_pos_perc_area:
                raise ValueError(
                    f"Value of 'max_perc_area_to_remove' "
                    f"(= {max_perc_area_to_remove}) is too large. "
                    f"An overestimate of the maximum possible % area that can "
                    f"be corrupted given {hole_size} x {hole_size} patches "
                    f"is: {max_pos_perc_area}. Make sure "
                    f"'max_perc_area_to_remove' is less than this value."
                )

            # If `sampling_pattern` is "poisson", precompute
            # `num_precompute` masks
            num_samples = ((img_shape[0] * img_shape[1]) * max_perc_area_to_remove) // (
                hole_size ** 2
            )
            logger.info("Precomputing masks...")
            for _ in tqdm.tqdm(range(num_precompute)):
                _, patch_mask = generate_poisson_disc_mask(
                    (img_shape[0], img_shape[1]),
                    min_distance=hole_size * np.sqrt(2),
                    num_samples=num_samples,
                    patch_size=hole_size,
                    k=10,
                )
                self.precomputed_masks.append(patch_mask)
            logger.info("Finished precomputing masks!")
        else:
            logger.info("Precomputing masks...")
            img_height = img_shape[0]
            img_width = img_shape[1]
            max_area_to_remove = int((img_height * img_width) * max_perc_area_to_remove)
            for _ in tqdm.tqdm(range(num_precompute)):
                patch_mask = np.zeros(img_shape)
                cur_num_holes = 0
                if max_holes < 0:
                    num_holes = np.inf
                else:
                    num_holes = np.random.randint(min_holes, max_holes + 1)
                while (
                    cur_num_holes <= num_holes and np.count_nonzero(patch_mask) < max_area_to_remove
                ):
                    hole_width = np.random.randint(min_width, max_width + 1)
                    hole_height = np.random.randint(min_height, max_height + 1)
                    tl_x = np.random.randint(img_width - hole_width)
                    tl_y = np.random.randint(img_height - hole_height)
                    br_x = tl_x + hole_width
                    br_y = tl_y + hole_height
                    hole_area = (br_x - tl_x) * (br_y - tl_y)
                    if hole_area > max_area_to_remove and np.count_nonzero(patch_mask) == 0:
                        continue
                    patch_mask[tl_y:br_y, tl_x:br_x] = 1
                    cur_num_holes += 1
                self.precomputed_masks.append(patch_mask)
            logger.info("Finished precomputing masks!")

    def get_transform(self, img: np.ndarray) -> MedTransform:
        """
        Determines the holes for 2D images. The location, number, and size
        of these holes are chosen at random, according to the max/min values
        provided by the user.

        The holes will be chosen one by one until either the chosen number of
        holes have been removed or "max_area_to_remove" pixels have been
        removed.

        If multiple 2D images are provided, the holes will be chosen
        independently for each image. Therefore, the chosen holes for one
        2D image may be different from the chosen holes for another 2D image.

        Args:
            img: A N x H x W x C image, containing N images with height H,
                    width W, and consisting of C channels.
        Returns:
            An instance of the MedTransform "FillRegionsWithValue",
                initialized with the appropriate parameters based on the
                input image.
        """
        num_images = img.shape[0]
        hole_mask = np.zeros(img.shape)

        for i in range(num_images):
            # Randomly select one of the precomputed masks
            rand_idx = np.random.randint(self.num_precompute)
            mask = self.precomputed_masks[rand_idx]

            # Randomly rotate by either 0, 90, 180, or 270 degrees
            # counter-clockwise
            mask_rot = np.rot90(mask, k=np.random.randint(4))

            # Get coordinates of patches
            hole_mask[i, :, :, :] = mask_rot[np.newaxis, :, :, np.newaxis]

        return FillRegionsWithValue(hole_mask=hole_mask, fill_value=self.fill_value)


class SwapPatches(TransformGen):
    """Randomly select pairs of non-overlapping patches that will
    be swapped in an iterative manner.

    The size for both patches in a pair will be chosen at random,
    and both patches in a pair will have the same size.

    Furthermore, the two patches in a pair will not overlap.
    """

    def __init__(
        self,
        max_height: int,
        min_height: int = None,
        max_width: int = None,
        min_width: int = None,
        max_iterations: int = 30,
        min_iterations: int = 30,
        is_square: bool = True,
        img_shape: Sequence[int] = (512, 512),
        max_perc_area_to_modify: float = 0.25,
        sampling_pattern: str = "uniform",
        num_precompute: int = 100,
    ):
        """
        Args:
            max_height: The maximum height of each patch. If "is_square" is
                        True, this will also be the maximum width of each
                        patch.
            min_height: The minimum height of each patch. If "is_square" is
                        True, this will also be the minimum width of each
                        patch.
            max_width: The maximum width of each patch. This is only used
                        if "is_square" is False.
            min_width: The minimum width of each patch. This is only used
                        if "is_square" is False.
            max_iterations: The maximum number of pairs of patches to swap. If
                            "max_perc_area_to_modify" is < 1, then the number
                            of iterations may be reduced to ensure only
                            the specified percentage of the area is modified.
            min_iterations: The minimum number of pairs of patches to swap.
            is_square: If True, all patches will be square. Otherwise, all
                        patches will be rectangular, with the possibility of
                        having non-equal heights and widths.
            img_shape: The height and width of each image.
            max_perc_area_to_modify: The maximum fraction of the image that
                                        will be modified. This is useful to
                                        ensure not too much of the image
                                        is changed.
            sampling_pattern: The type of sampling pattern to use when
                                selecting random patches.
                Possible values: "uniform", "poisson"
            num_precompute: The number of masks to precompute.
        """
        super().__init__()
        if sampling_pattern not in ["uniform", "poisson"]:
            raise ValueError(
                f"Invalid value for 'sampling_pattern' ("
                f"Got '{sampling_pattern}'). Must "
                f"be either 'uniform' or 'poisson'."
            )
        if min_height is None:
            min_height = max_height
        if not is_square:
            if max_width is None:
                raise ValueError("Value of 'max_width' must not be None if patch is " "not square")
            if min_width is None:
                min_width = max_width
        self._init(locals())

        self.precomputed_masks = []
        if sampling_pattern == "poisson":
            assert max_height % 2 == 0, f"'max_height' (= {max_height}) must be even"
            assert min_height == max_height, (
                f"If sampling_pattern is 'poisson', min_height (= {min_height}) "
                f"must equal max_height (= {max_height})"
            )
            assert is_square, "Only square patches are allowed if sampling pattern is " "'poisson'"

            patch_size = max_height
            # Check max_perc_area_to_modify
            # -- Assuming a hexagonal packing of circles will output the
            #       most number of samples when using Poisson disc sampling.
            # -- From https://mathworld.wolfram.com/CirclePacking.html:
            #       Max packing density when using hexagonal packing is
            #       pi / (2 * sqrt(3))
            max_pos_area = (img_shape[0] * img_shape[1]) * (np.pi / (2 * np.sqrt(3)))
            max_num_patches = max_pos_area // ((np.pi / 2) * (patch_size ** 2))
            max_pos_perc_area = np.round(
                (max_num_patches * (patch_size ** 2)) / (img_shape[0] * img_shape[1]), decimals=3
            )
            if max_perc_area_to_modify >= max_pos_perc_area:
                raise ValueError(
                    f"Value of 'max_perc_area_to_modify' "
                    f"(= {max_perc_area_to_modify}) is too large. "
                    f"An overestimate of the maximum possible % area that can "
                    f"be corrupted given {patch_size} x {patch_size} patches "
                    f"is: {max_pos_perc_area}. Make sure "
                    f"'max_perc_area_to_modify' is less than this value."
                )

            # If `sampling_pattern` is "poisson", precompute
            # `num_precompute` masks
            num_samples = ((img_shape[0] * img_shape[1]) * max_perc_area_to_modify) // (
                patch_size ** 2
            )
            assert num_samples >= 2, f"Number of samples (= {num_samples}) must be >= 2"
            # Ensure number of samples is even
            if num_samples % 2:
                num_samples -= 1

            logger.info("Precomputing masks...")
            for _ in tqdm.tqdm(range(num_precompute)):
                pd_mask, _ = generate_poisson_disc_mask(
                    (img_shape[0], img_shape[1]),
                    min_distance=patch_size * np.sqrt(2),
                    num_samples=num_samples,
                    patch_size=patch_size,
                    k=10,
                )
                self.precomputed_masks.append(pd_mask)
            logger.info("Finished precomputing masks!")
        else:
            img_height = img_shape[0]
            img_width = img_shape[1]
            area_check = np.zeros((img_height, img_width))
            max_area_to_modify = int((img_height * img_width) * max_perc_area_to_modify)

            logger.info("Precomputing masks...")
            for _ in tqdm.tqdm(range(num_precompute)):
                patch_coords = []
                area_check[:] = 0
                if max_iterations < 0:
                    num_iterations = np.inf
                else:
                    num_iterations = np.random.randint(min_iterations, max_iterations + 1)
                while (len(patch_coords) / 4) <= num_iterations and np.count_nonzero(
                    area_check
                ) < max_area_to_modify:
                    patch_height = np.random.randint(min_height, max_height + 1)
                    if is_square:
                        patch_width = patch_height
                    else:
                        patch_width = np.random.randint(min_width, max_width + 1)
                    # Get coordinates of first patch
                    tl_x_1 = np.random.randint(img_width - patch_width)
                    tl_y_1 = np.random.randint(img_height - patch_height)
                    br_x_1 = tl_x_1 + patch_width
                    br_y_1 = tl_y_1 + patch_height
                    patch_area = (br_x_1 - tl_x_1) * (br_y_1 - tl_y_1)
                    if patch_area > (max_area_to_modify / 2) and not patch_coords:
                        continue
                    patch_coords.append([tl_x_1, tl_y_1])
                    patch_coords.append([br_x_1, br_y_1])

                    # Get coordinates of second patch, ensuring the second
                    # patch will not overlap with the first patch
                    tl_y_2 = np.random.randint(img_height - patch_height)
                    if tl_y_2 <= tl_y_1 - patch_height or tl_y_2 >= br_y_1:
                        tl_x_2 = np.random.randint(img_width - patch_width)
                    else:
                        possible_columns = list(range(tl_x_1 - patch_width + 1)) + list(
                            range(br_x_1, img_width - patch_width)
                        )
                        tl_x_2 = np.random.choice(np.array(possible_columns))
                    br_x_2 = tl_x_2 + patch_width
                    br_y_2 = tl_y_2 + patch_height
                    patch_coords.append([tl_x_2, tl_y_2])
                    patch_coords.append([br_x_2, br_y_2])

                    # Record the areas of image that were modified by the current
                    # pair of patches
                    area_check[tl_y_1:br_y_1, tl_x_1:br_x_1] = 1
                    area_check[tl_y_2:br_y_2, tl_x_2:br_x_2] = 1
                coord_matrix = np.array(patch_coords).T
                self.precomputed_masks.append(coord_matrix)
            logger.info("Finished precomputing masks!")

    def get_transform(self, img: np.ndarray) -> MedTransform:
        """
        Determines the pairs of patches that will be swapped in the image.

        Args:
            img: A N x H x W x C image, containing N images with height H,
                    width W, and consisting of C channels.
        Returns:
            An instance of the MedTransform "Swap2DPatches",
                initialized with the appropriate parameters based on the
                input image.
        """
        num_images = img.shape[0]
        img_height = img.shape[1]
        img_width = img.shape[2]
        img_center = np.array([[(img_width - 1) / 2], [(img_height - 1) / 2]]).astype("float64")
        patch_pairs = [[] for _ in range(num_images)]

        if self.sampling_pattern == "uniform":
            for i in range(num_images):
                # Randomly select one of the precomputed list of patch pairs
                rand_idx = np.random.randint(self.num_precompute)
                coord_matrix = self.precomputed_masks[rand_idx]
                coord_matrix = coord_matrix.astype("float64")

                # Create rotation matrix to randomly rotate coordinates by 0,
                # 90, 180, or 270 degrees counter-clockwise
                num_rotate = np.random.randint(4)
                rot_rad = (90 * num_rotate) * (np.pi / 180)
                rot_matrix = np.array(
                    [[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]]
                ).astype("float64")

                # Move all points such that the origin is the center of the image
                coord_matrix -= img_center

                # Apply rotation matrix
                rot_coord = np.dot(rot_matrix, coord_matrix)

                # Reset origin to top left corner of the image
                rot_coord += img_center

                # Set all points to be integers
                rot_coord = np.rint(rot_coord).astype(int)

                # Get top-left and bottom-right coordinates
                rot_coord = np.reshape(rot_coord, (2, rot_coord.shape[1] // 2, 2))
                tl = np.min(rot_coord, axis=2)
                br = np.max(rot_coord, axis=2)

                # Reshape tl and br to (2 x 2 x num_pairs)
                tl = np.reshape(tl, (2, tl.shape[1] // 2, 2))
                tl = np.transpose(tl, (0, 2, 1))
                br = np.reshape(br, (2, br.shape[1] // 2, 2))
                br = np.transpose(br, (0, 2, 1))

                patch_pairs[i].extend([tl, br])

        else:
            for i in range(num_images):
                # Randomly select one of the precomputed masks
                rand_idx = np.random.randint(self.num_precompute)
                pd_mask = self.precomputed_masks[rand_idx]

                # Randomly rotate by either 0, 90, 180, or 270 degrees
                # counter-clockwise
                num_rotate = np.random.randint(4)
                pd_mask_rot = np.rot90(pd_mask, k=num_rotate)

                # Get sample locations
                qy, qx = np.where(pd_mask_rot)
                all_locs = [(y, x) for y, x in zip(qy, qx)]
                np.random.shuffle(all_locs)

                # Get pairs of patches
                tl = np.zeros((2, 2, len(all_locs) // 2)).astype(int)
                br = np.zeros((2, 2, len(all_locs) // 2)).astype(int)
                half_patch = self.max_height // 2
                for pair_idx in range(len(all_locs) // 2):
                    p1_qy = all_locs[pair_idx * 2][0]
                    p1_qx = all_locs[pair_idx * 2][1]
                    tl[:, 0, pair_idx] = [p1_qx - half_patch, p1_qy - half_patch]
                    br[:, 0, pair_idx] = [p1_qx + half_patch, p1_qy + half_patch]

                    p2_qy = all_locs[pair_idx * 2 + 1][0]
                    p2_qx = all_locs[pair_idx * 2 + 1][1]
                    tl[:, 1, pair_idx] = [p2_qx - half_patch, p2_qy - half_patch]
                    br[:, 1, pair_idx] = [p2_qx + half_patch, p2_qy + half_patch]
                patch_pairs[i].extend([tl, br])

        return Swap2DPatches(patch_pairs=patch_pairs)


def apply_transform_gens(
    transform_gens: Sequence[Union[TransformGen, MedTransform]], img: np.ndarray
) -> Tuple[np.ndarray, TransformList]:
    """
    Apply a list of :class:`TransformGen` or :class:`MedTransform` on the
    input images, and returns the transformed images and a list of transforms.

    We cannot simply create and return all transforms without
    applying it to the images, because a subsequent transform may
    need the output of the previous one.

    Args:
        transform_gens: A list of :class:`TransformGen` or
                        :class:`MedTransform` instances to be applied.
        img: A stack of images of type uint8 or floating point with 1 or 3
                channels. The shape of img will be (N x H x W x C). Therefore,
                each of the N images will have height, H, and width, W,
                with C channels.

    Returns:
        transformed_imgs (np.ndarray): The transformed images.
        transform_list (TransformList): A TransformList that contains the
                                        transforms that are used.
    """
    for g in transform_gens:
        assert isinstance(g, (TransformGen, MedTransform)), g

    check_dtype(img)

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(img) if isinstance(g, TransformGen) else g
        assert isinstance(tfm, MedTransform), (
            "TransformGen {} must return an instance of MedTransform! "
            "Got {} instead".format(g, tfm)
        )
        img = tfm.apply_image(img)
        tfms.append(tfm)
    return img, TransformList(tfms)


def build_preprocessing(cfg: Config):
    transforms = []
    assert len(cfg.PREPROCESSING) == len(cfg.PREPROCESSING_ARGS), (
        "cfg.PREPROCESSING and cfg.PREPROCESSING_ARGS must have the same " "length"
    )
    for i, pp in enumerate(cfg.PREPROCESSING):
        transform_gen_module = importlib.import_module(__name__)
        transform_module = importlib.import_module("medsegpy.data.transforms.transform")
        try:
            transform_class = getattr(transform_gen_module, pp)
        except AttributeError:
            try:
                transform_class = getattr(transform_module, pp)
            except AttributeError:
                raise ValueError("{} is not a valid transform!".format(pp))
        assert inspect.isclass(transform_class), "{} is not a class!".format(pp)
        assert issubclass(transform_class, TransformGen) or issubclass(
            transform_class, MedTransform
        ), "{} is not a valid transform!".format(pp)
        transforms.append(transform_class(**cfg.PREPROCESSING_ARGS[i]))

    return transforms
