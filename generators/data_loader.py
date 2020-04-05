import math
from abc import ABC, abstractmethod
import random
import logging

from keras import utils as k_utils
import numpy as np

from config import Config
from .im_gens import GeneratorState, Generator, OAIGenerator

logger = logging.getLogger("msk_seg.{}".format(__name__))


class DataLoader(k_utils.Sequence, ABC):
    """DataLoader following :class:`keras.utils.Sequence` API.

    When using `fit_generator`, pass `shuffle=False` into `fit_generator`.
    Shuffling is handled by the :class:`DataLoader`.
    """
    # These should be overwritten by subclasses.
    _GENERATOR_TYPE = Generator

    def __init__(self, config: Config, state: GeneratorState,
                 shuffle: bool = None, drop_last: bool = False,
                 generator: Generator = None):
        """
        Args:
            config (Config): A config object.
            state (GeneratorState): Current state.
            shuffle (bool, optional): If `True`, shuffle data every epoch. If
                `None`, defaults to `True` if
                `state == GeneratorState.TRAINING`, otherwise `False`.
            drop_last (bool, optional): Drop the last incomplete batch, if the
                dataset size is not divisible by the batch size. If `False` and
                the size of the dataset is not divisible by batch size, then
                the last batch will be smaller.
        """
        assert issubclass(self._GENERATOR_TYPE, Generator), (
            "specify generator type in subclass"
        )
        supported_tags = self.supported_tags()
        assert supported_tags and supported_tags[0] != ""

        self.config = config
        self._state = state
        self.shuffle = shuffle if shuffle is not None \
            else state == GeneratorState.TRAINING
        self.drop_last = drop_last

        if generator:
            assert isinstance(generator, self._GENERATOR_TYPE)
        self._generator = self._GENERATOR_TYPE(config) if not generator \
            else generator

        params = self._generator._img_generator_base_info(self._state)
        self._batch_size = params["batch_size"]

    def __len__(self):
        """Number of batches."""
        num_examples = self._generator.num_examples(self._state)
        return num_examples // self._batch_size if self.drop_last else math.ceil(num_examples / self._batch_size)

    def summary(self) -> str:
        s = ""
        s += "State: {}\n".format(self._state)
        s += "Batch Size: {}\n".format(self._batch_size)
        s += "Num Examples: {}\n".format(
            self._generator.num_examples(self._state)
        )
        s += "Num Batches: {}\n".format(len(self))
        s += "Shuffle: {}\n".format(self.shuffle)
        s += "Drop Last: {}\n".format(self.drop_last)
        return s

    @classmethod
    def supported_tags(cls):
        return cls._GENERATOR_TYPE.SUPPORTED_TAGS


class OAIDataLoader(DataLoader):
    _GENERATOR_TYPE = OAIGenerator

    def __init__(self, config: Config, state: GeneratorState,
                 shuffle: bool = None, drop_last: bool = False,
                 generator: Generator = None):
        super().__init__(
            config, 
            state, 
            shuffle=shuffle, 
            drop_last=drop_last,
            generator=generator,
        )

        self._files, _, self._max_slice_num = self._generator._calc_generator_info(
            self._state)
        if self.shuffle:
            random.shuffle(self._files)
        else:
            self._files = self._generator.sort_files(self._files)
        self._image_size = config.IMG_SIZE
        self._tissues = config.TISSUES
        self._include_background = config.INCLUDE_BACKGROUND
        self._num_neighboring_slices = config.num_neighboring_slices()
        self._num_classes = config.get_num_classes()

    def __getitem__(self, idx):
        """
        Args:
            idx: Batch index.

        Returns:
            ndarray, ndarray: images (N,X,Y,1), masks (N,X,Y,#classes)
        """
        img_size = self._image_size
        tissues = self._tissues
        include_background = self._include_background
        num_neighboring_slices = self._num_neighboring_slices

        batch_size = self._batch_size
        max_slice_num = self._max_slice_num

        total_classes = self._num_classes
        mask_size = img_size[:-1] + (total_classes,)

        files = self._files
        generator: OAIGenerator = self._generator

        start = idx * batch_size
        stop = min(idx * (batch_size + 1), len(files))
        images = []
        masks = []
        for file_idx in range(start, stop):
            filepath = files[file_idx]
            im, seg = generator._load_input_helper(
                filepath=filepath,
                tissues=tissues,
                num_neighboring_slices=num_neighboring_slices,
                max_slice_num=max_slice_num,
                include_background=include_background
            )

            assert im.shape == img_size, "Input shape mismatch. Expected %s, got %s" % (
                img_size, im.shape)
            assert seg.shape == mask_size, "Ouput shape mismatch. Expected %s, got %s" % (
                mask_size, seg.shape)

            images.append(im)
            masks.append(seg)

        return np.stack(images, axis=0), np.stack(masks, axis=0)

    def on_epoch_end(self):
        """Shuffle (if applicable) on epoch end."""
        if self.shuffle:
            random.shuffle(self._files)


def get_data_loader(
    config: Config,
    state: GeneratorState,
    **kwargs
) -> DataLoader:
    """Get data loader based on config `TAG` value"""
    for generator in [OAIDataLoader]:
        try:
            gen = generator(config, state, **kwargs)
            if gen:
                return gen
        except ValueError:
            continue

    raise ValueError('No data loader found for tag `{}`'.format(generator))
