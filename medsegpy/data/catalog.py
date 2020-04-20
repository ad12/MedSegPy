"""Metadata catalogs for different datasets.

Metadata stores information like directory paths, mapping from class ids to
name, etc.

Adopted from Facebook's detectron2.
https://github.com/facebookresearch/detectron2
"""
import copy
import logging
import types

from medsegpy.utils.logger import log_first_n

__all__ = ["MetadataCatalog"]


class Metadata(types.SimpleNamespace):
    """
    A class that supports simple attribute setter/getter.
    It is intended for storing metadata of a dataset and make it accessible globally.

    Examples:

    .. code-block:: python

        # somewhere when you load the data:
        MetadataCatalog.get("mydataset").thing_classes = ["person", "dog"]

        # somewhere when you print statistics or visualize:
        classes = MetadataCatalog.get("mydataset").thing_classes
    """

    # the name of the dataset
    # set default to N/A so that `self.name` in the errors will not trigger
    # getattr again
    name: str = "N/A"

    _RENAMED = {}

    def __getattr__(self, key):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(
                    key,
                    self._RENAMED[key],
                ),
                n=10,
            )
            return getattr(self, self._RENAMED[key])

        raise AttributeError(
            "Attribute '{}' does not exist in the metadata of '{}'. "
            "Available keys are {}.".format(
                key,
                self.name,
                str(self.__dict__.keys()),
            )
        )

    def __setattr__(self, key, val):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(
                    key,
                    self._RENAMED[key]
                ),
                n=10,
            )
            setattr(self, self._RENAMED[key], val)

        # Ensure that metadata of the same name stays consistent
        try:
            oldval = getattr(self, key)
            assert oldval == val, (
                "Attribute '{}' in the metadata of '{}' cannot be set "
                "to a different value!\n{} != {}".format(
                    key, self.name, oldval, val,
                )
            )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self):
        """
        Returns all the metadata as a dict.
        Note that modifications to the returned dict will not reflect on the
        Metadata object.
        """
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        """
        Set multiple metadata with kwargs.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def get(self, key, default=None):
        """
        Access an attribute and return its value if exists.
        Otherwise return default.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class MetadataCatalog:
    """
    MetadataCatalog provides access to "Metadata" of a given dataset.

    The metadata associated with a certain name is a singleton: once created,
    the metadata will stay alive and will be returned by future calls to
    `get(name)`.

    It's like global variables, so don't abuse it.
    It's meant for storing knowledge that's constant and shared across the
    execution of the program, e.g.: the class names in OAI iMorphics.
    """

    _NAME_TO_META = {}

    @staticmethod
    def get(name):
        """
        Args:
            name (str): name of a dataset (e.g. oai_2d_train).

        Returns:
            Metadata: The :class:`Metadata` instance associated with this name,
            or create an empty one if none is available.
        """
        assert len(name)
        if name in MetadataCatalog._NAME_TO_META:
            ret = MetadataCatalog._NAME_TO_META[name]
            return ret
        else:
            m = MetadataCatalog._NAME_TO_META[name] = Metadata(name=name)
            return m

    @staticmethod
    def convert_path_to_dataset(path):
        catalog = {m.get("scan_root", None): name
                   for name, m in MetadataCatalog._NAME_TO_META.items()}
        catalog.pop(None)
        return catalog[path]

