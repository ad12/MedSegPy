import itertools
import warnings
from typing import Sequence, Union

import numpy as np


def collect_mask(mask: np.ndarray, index: Sequence[Union[int, Sequence[int], int]]):
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


def compute_patches(
    img_size: Sequence[int],
    patch_size: Sequence[int] = None,
    pad_size: Union[int, Sequence[int]] = None,
    strides: Union[int, Sequence[int]] = None,
):
    """Compute start/stop indices for patches and corresponding pad_size.

    Padding is left aligned.

    Terminology:
        - `Dx`: size of dimension x.
        - `Px`: patch size desired for dimension x.
        - `px`: pad_size size for dimension x.
        - `block`: sequence of ranges where start/stop values can exceed image
            dimensions (i.e. start < 0, stop > Dx)
        - `patch`: clipped blocks. ranges will be within image dimensions.
            Should be used with appropriate pad_size.

    Args:
        img_size (Sequence[int]): Size of image to patch `(D1, D2, ...)`.
        patch_size (Sequence[int]): Size of each patch `(P1, P2, ...).
            Use `None` to indicate dimension should not be patched.
            The `pad_size` for this dimension should either be `None` or 0.
        pad_size (Sequence[int]): Padding for each dimension `(p1, p2, ...)`.
            Effective padding will be symmetric `2*p1`, `2*p2`, ...
        strides (int(s), optional): Strides to use for different dimensions.
            If integer, same stride will be used across all dimensions.
            Defaults to `patch_size` (i.e. disjoint patches).

    Returns:
        Tuple: Sequence of `(patch, pad)` tuples. `pad` will be `None` if no
            padding is needed for that patch.
    """
    assert len(img_size) == len(patch_size)

    if pad_size is None:
        pad_size = [None] * len(img_size)
    if isinstance(pad_size, int):
        pad_size = [pad_size] * len(img_size)
    else:
        if len(pad_size) != len(img_size):
            raise ValueError("pad_size sequence must have same length as img_size")
        pad_size = [
            patch_size[i] - img_size[i] if p is None and img_size[i] < patch_size[i] else p
            for i, p in enumerate(pad_size)
        ]
        pad_size = [p if isinstance(p, int) else 0 for p in pad_size]

    is_pad_lt_patch = all(
        (p == 0) or (P is None and p == 0) or p < P for p, P in zip(pad_size, patch_size)
    )
    if not is_pad_lt_patch:
        raise ValueError(
            "Padding {} must be less than patch {} in all dims".format(pad_size, patch_size)
        )

    if strides is None:
        strides = list(patch_size)
    elif isinstance(strides, int):
        strides = [strides] * len(patch_size)

    if any(P is not None and P < s for s, P in zip(strides, patch_size)):
        warnings.warn(
            "Found stride dimension ({}) larger than "
            "patch dimension ({}). "
            "Data along this dimension will be skipped over.".format(strides, patch_size)
        )

    start_idxs = [
        list(range(-px, Dx + px - Px + 1, stride)) if Px is not None and Px != -1 else [None]
        for Dx, Px, px, stride in zip(img_size, patch_size, pad_size, strides)
    ]
    start_idxs = list(itertools.product(*start_idxs))
    stop_idxs = [
        [
            i + patch_size[idx] if patch_size[idx] is not None else None
            for idx, i in enumerate(start)
        ]
        for start in start_idxs
    ]
    assert len(start_idxs) == len(stop_idxs), ("start and stop indices should be 1-to-1",)

    blocks = [[(s, e) for s, e in zip(start, stop)] for start, stop in zip(start_idxs, stop_idxs)]

    patches = []
    padding = []
    for block in blocks:
        patches.append(
            tuple(
                slice(max(0, s), min(e, Dx)) if s is not None else slice(None)
                for (s, e), Dx in zip(block, img_size)
            )
        )
        pad = tuple(
            (max(0, -s), max(0, e - Dx)) if s is not None else (0, 0)
            for (s, e), Dx, px in zip(block, img_size, pad_size)
        )
        valid_pad = any(p != (0, 0) for p in pad)
        padding.append(pad if valid_pad else None)

    return tuple(zip(patches, padding))


def generate_poisson_disc_mask(
        img_shape: Sequence[int],
        min_distance: float,
        num_samples: int,
        patch_size: float = 0.0,
        k: float = 30,
        seed: int = None
):
    """Generate Poisson-disc sampling mask

    Adapted from Arjun's adaption of the function in sigpy:
    "sigpy.mri.poisson".

    Args:
        img_shape (Sequence[int]): length-2 image shape.
        min_distance (float): the minimum distance between samples.
        num_samples (int): The number of samples to select.
        patch_size (float): The patch size.
        k (float): maximum number of samples to reject.
        seed (int): Random seed.

    Returns:
        array: Poisson-disc sampling mask.

    References:
        Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
        SIGGRAPH sketches. 2007.
    """
    y, x = np.mgrid[:img_shape[-2], :img_shape[-1]]
    x = np.maximum(abs(x - img_shape[-1] / 2), 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2), 0)
    y /= y.max()
    r = np.sqrt(x ** 2 + y ** 2)

    # Quick checks
    assert int(num_samples) == num_samples, \
        f"Number of required samples must be an integer. " \
        f"(Got num_samples = {num_samples})."
    num_samples = int(num_samples)

    rounded_min_dist = np.round(min_distance, decimals=3)

    R = np.zeros_like(r)
    R[:] = np.round(rounded_min_dist, decimals=3)
    mask = np.zeros(1)
    num_iterations = 0
    while np.sum(mask[:]) != num_samples:
        mask, patch_mask = _poisson(img_shape[-1],
                                    img_shape[-2],
                                    k,
                                    R,
                                    num_samples=num_samples,
                                    patch_size=patch_size,
                                    seed=seed)
        num_iterations += 1
        if num_iterations > 10:
            raise ValueError("Cannot find enough samples. Please make sure "
                             "the number of samples is not too large and the "
                             "minimum distance between samples is not too "
                             "large as well.")

    mask = mask.reshape(img_shape).astype(int)
    patch_mask = patch_mask.reshape(img_shape).astype(int)
    return mask, patch_mask


def _poisson(nx, ny, K, R, num_samples=None, patch_size=0.0, seed=None):
    mask = np.zeros((ny, nx))
    patch_mask = np.zeros((ny, nx))
    f = ny / nx

    half_patch = patch_size / 2

    if seed is not None:
        rand_state = np.random.RandomState(int(seed))
    else:
        rand_state = np.random

    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = rand_state.randint(0, nx)
    pys[0] = rand_state.randint(0, ny)
    m = 1

    if num_samples:
        max_limit = num_samples
    else:
        max_limit = nx * ny

    while 0 < m < nx * ny and np.sum(mask[:]) < max_limit:
        i = rand_state.randint(0, m)
        px = pxs[i]
        py = pys[i]
        rad = R[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < K:
            # Generate point randomly from R to 2R
            rd = rad * (rand_state.random() * 3 + 1) ** 0.5
            t = 2 * np.pi * rand_state.random()
            qx = int(px + rd * np.cos(t))
            qy = int(py + rd * f * np.sin(t))

            # Reject if outside grid, patch will not fit,
            # or close to other points
            if (0 <= qx < nx and
                    0 <= qy < ny and
                    (qx - half_patch) >= 0 and
                    (qx + half_patch) < nx and
                    (qy - half_patch) >= 0 and
                    (qy + half_patch) < ny):
                startx = max(int(qx - rad), 0)
                endx = min(int(qx + rad + 1), nx)
                starty = max(int(qy - rad * f), 0)
                endy = min(int(qy + rad * f + 1), ny)

                done = True
                for x in range(startx, endx):
                    for y in range(starty, endy):
                        if (mask[y, x] == 1
                                and (((qx - x) / R[y, x]) ** 2 +
                                     ((qy - y) / (R[y, x] * f)) ** 2 < 1)):
                            done = False
                            break
                    if not done:
                        break
            k += 1

        # Add point if done else remove active
        if done:
            pxs[m] = qx
            pys[m] = qy
            mask[int(qy), int(qx)] = 1
            if patch_size > 0:
                tl = (int(int(qx) - half_patch), int(int(qy) - half_patch))
                br = (int(int(qx) + half_patch), int(int(qy) + half_patch))
                patch_mask[tl[1]:br[1], tl[0]:br[0]] = 1
            m += 1
        else:
            pxs[i] = pxs[m - 1]
            pys[i] = pys[m - 1]
            m -= 1

    return mask, patch_mask
