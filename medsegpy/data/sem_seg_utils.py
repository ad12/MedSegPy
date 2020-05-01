from typing import Sequence, Union

import numpy as np


def collect_mask(
    mask: np.ndarray, index: Sequence[Union[int, Sequence[int], int]]
):
    """Collect masks by index.

    Collated indices will be summed. For example, `index=(1,(3,4))` will return
    `np.stack(mask[...,1], mask[...,3]+mask[...,4])`.

    Args:
        mask (ndarray): A (...)xC array.
        index (Sequence[int]): The index/indices to select in mask.
            If sub-indices are collated, they will be summed.
    """
    if isinstance(index, int):
        index = (index,)
    if not any(isinstance(idx, Sequence) for idx in index):
        mask = mask[..., index]
    else:
        o_seg = []
        for idx in index:
            c_seg = mask[..., idx]
            if isinstance(idx, Sequence):
                c_seg = np.sum(c_seg, axis=-1)
            o_seg.append(c_seg)
        mask = np.stack(o_seg, axis=-1)

    # TODO: Remove dependency on legacy OAI data structure
    # OAI masks are stored in shape (...)xHxWx1xC.
    if mask.shape[-2] == 1:
        mask = np.squeeze(mask, axis=-2)
    return mask


def add_background_labels(mask: np.ndarray, background_last=False):
    """Generate background labels based on segmentations.

    Args:
        seg_mask (ndarray): A binary ndarray with last dimension corresponding
            to different classes. i.e. if 3 classes, segs.shape[-1] = 3
        background_last (:obj:`bool`, optional): If `True`, background will
            be concatenated to the end of the mask.

    Returns:
        ndarray: Binary masks such that first index is backround
    :return:
    """
    all_tissues = np.sum(mask, axis=-1, dtype=np.bool)
    background = np.asarray(~all_tissues, dtype=mask.dtype)
    background = background[..., np.newaxis]
    if background_last:
        return np.concatenate([mask, background], axis=-1)
    else:
        return np.concatenate([background, mask], axis=-1)
