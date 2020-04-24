from typing import Tuple

import numpy as np

from detectron2.data.transforms import TransformGen
from fvcore.transforms.transform import CropTransform

from .transform import CropTransform


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
