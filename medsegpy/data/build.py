"""Build dataset dictionaries."""
import itertools
import logging
from typing import List, Hashable, Dict, Sequence, Union

from medsegpy.config import Config

from .catalog import DatasetCatalog
from .data_loader import build_data_loader, DefaultDataLoader


def filter_dataset(
    dataset_dicts: List[Dict],
    by: Hashable,
    accepted_elements,
    include_missing: bool = False
):
    """Filter by common dataset fields.

    Args:
        dataset_dicts (List[Dict]): data in MedSegPy Dataset format.
        by (Hashable): Field to filter by.
        accepted_elements (Sequence): Acceptable elements.
        include_missing (bool, optional): If `True`, include elements without
            `by` field in dictionary representation.

    Returns:
        List[Dict]: Filtered dataset dictionaries.
    """
    num_before = len(dataset_dicts)
    dataset_dicts = [
        x for x in dataset_dicts
        if include_missing or (by in x and x[by] in accepted_elements)
    ]
    num_after = len(dataset_dicts)

    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} elements with filter '{}'. {} elements left.".format(
            num_before - num_after, by, num_after,
        )
    )
    return dataset_dicts


def get_sem_seg_dataset_dicts(
    dataset_names: Sequence[str], filter_empty: bool = True,
):
    """Load and prepare dataset dicts for semantic segmentation.

    Args:
        dataset_names (Sequence[str])": A list of dataset names.
        filter_empty (bool, optional): Filter datasets without field
            `'sem_seg_file'`.
    """
    assert len(dataset_names)
    dataset_dicts = [
        DatasetCatalog.get(dataset_name) for dataset_name in dataset_names
    ]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    logger = logging.getLogger(__name__)
    if filter_empty:
        num_before = len(dataset_dicts)
        dataset_dicts = [
            x for x in dataset_dicts if "sem_seg_file" in x
        ]
        num_after = len(dataset_dicts)
        logger.info(
            "Removed {} elements without annotations. {} elements left.".format(
                num_before - num_after, num_after,
            )
        )

    return dataset_dicts


def build_loader(
    cfg: Config,
    dataset_names: Union[str, Sequence[str]],
    batch_size: int,
    is_test: bool,
    shuffle: bool,
    drop_last: bool,
    **kwargs,
):
    if isinstance(dataset_names, str):
        dataset_names = (dataset_names,)
    kwargs["batch_size"] = batch_size
    kwargs["is_test"] = is_test
    kwargs["shuffle"] = shuffle
    kwargs["drop_last"] = drop_last

    dataset_dicts = get_sem_seg_dataset_dicts(dataset_names, filter_empty=True)
    return build_data_loader(
        cfg,
        dataset_dicts,
        **kwargs
    )

