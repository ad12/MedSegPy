import os

import cv2
import h5py
import numpy as np
import SimpleITK as sitk
import seaborn as sns

import sys

from io_utils import check_dir


class MultiClassOverlay():
    """
    Class to simplify overlaying images and labels
    """
    def __init__(self, num_classes,
                 color_palette=sns.color_palette('pastel'),
                 background_label=0,
                 opacity=0.7,
                 dirpath = ''):
        """
        Constructor
        :param num_classes: Number of classes

        Optional:
        :param color_palette: list of RGB tuples to use for color. Default seaborn.color_palette('pastel').
        :param background_label: Label to exclude for background. Default: 0.
                                 To include background, set to None.
        :param opacity: How transparent overlay should be (0-1). Default: 0.7
        """
        effective_num_classes = num_classes-1 if background_label is not None else num_classes
        if len(color_palette < effective_num_classes):
            raise ValueError('Must provide at least %d colors' % effective_num_classes)

        if opacity < 0 or opacity > 1:
            raise ValueError('opacity must be between 0-1')

        self.num_classes = num_classes
        self.background_label = background_label
        self.opacity = opacity
        self.dirpath = check_dir(dirpath)

        # set colormap
        color_palette = color_palette
        colormap = dict()
        cp_ind = 0
        for i in range(num_classes):
            if i == background_label:
                continue
            cp_ind += 1
            colormap[i] = color_palette[cp_ind]
        self.colormap = colormap

    def im_overlay(self, dirpath, volume: np.ndarray, logits: np.ndarray):
        dirpath = check_dir(dirpath)
        assert volume.ndim == 3, "Volume must be 3D array with shape [Y, X, Z]"
        assert logits.ndim == 4, "Labels must be 4D array with shape [Y, X, Z, classes]"

        # labels are argmax(logits) in the class dimension
        labels = np.argmax(logits, axis=-1)
        labels_colored = self.__apply_colormap(labels)

        for z in range(volume.shape[-1]):
            x_im = volume[..., z]
            label_overlay = labels_colored[..., z, :]

            slice_name = '%03d.png' % (z+1)

            filepath = os.path.join(dirpath, slice_name)
            self.__im_overlay(x_im, label_overlay, filepath)

    def __apply_colormap(self, labels: np.ndarray):
        colormap = self.colormap
        background_label = self.background_label

        labels_colored = np.zeros(labels.shape + (3,))

        for c in np.unique(labels):
            if c == background_label:
                continue

            labels_colored[labels == c, :] = colormap[c]
            print('h')

        return labels_colored

    def __im_overlay(self, x, c_label, filepath=None):
        x_rgb = np.stack([x, x, x], axis=-1).astype(np.uint8)
        overlap_img = cv2.addWeighted(x_rgb, 1, c_label, self.opacity, 0)

        if filepath:
            cv2.imwrite(filepath, overlap_img)


def write_im_overlay(dir_path, xs, im_overlay):
    """
    Overlap input (xs) with mask (im_overlap) and save to directory
    :param dir_path: path to directory to save images
    :param xs: inputs
    :param im_overlay: overlay images
    """
    dir_path = check_dir(dir_path)
    num_slices = xs.shape[0]
    for i in range(num_slices):
        x = scale_img(np.squeeze(xs[i, ...]))
        x = np.stack([x, x, x], axis=-1).astype(np.uint8)
        im = im_overlay[i]

        slice_name = '%03d.png' % i

        overlap_img = cv2.addWeighted(x, 1, im, OPACITY, 0)
        cv2.imwrite(os.path.join(dir_path, slice_name), overlap_img)


def write_sep_im_overlay(dir_path, xs, y_true, y_pred):
    """
    Overlap input (xs) with mask (im_overlap) and save to directory
    :param dir_path: path to directory to save images
    :param xs: inputs
    :param im_overlay: overlay images
    """
    correct_dir_path = check_dir(os.path.join(dir_path, 'true_pos'))
    error_dir_path = check_dir(os.path.join(dir_path, 'error'))
    num_slices = xs.shape[0]
    for i in range(num_slices):
        x = scale_img(np.squeeze(xs[i, ...]))
        x = np.stack([x, x, x], axis=-1).astype(np.uint8)
        im_correct, im_error = generate_sep_ovlp_image(y_true[i, ...], y_pred[i, ...])

        slice_name = '%03d.png' % i

        overlap_img_correct = cv2.addWeighted(x, 1, im_correct, 1.0, 0)
        cv2.imwrite(os.path.join(correct_dir_path, slice_name), overlap_img_correct)

        overlap_img_error = cv2.addWeighted(x, 1, im_error, 1.0, 0)
        cv2.imwrite(os.path.join(error_dir_path, slice_name), overlap_img_error)


def generate_sep_ovlp_image(y_true, y_pred):
    """
    TODO: write comment
    :param y_true: numpy array of ground truth labels
    :param y_pred: numpy array of predicted labels
    :return: a BGR image
    """
    assert (y_true.shape == y_pred.shape)
    assert len(y_true.shape) == 2, "shape should be 2d, but is " + y_true.shape

    y_true = y_true.astype(np.bool)
    y_pred = y_pred.astype(np.bool)

    TP = y_true * y_pred
    FN = y_true * (~y_pred)
    FP = (~y_true) * y_pred

    # BGR format
    img_corr = np.stack([np.zeros(TP.shape), TP, np.zeros(TP.shape)], axis=-1).astype(np.uint8) * 255
    img_err = np.stack([FP, np.zeros(TP.shape), FN], axis=-1).astype(np.uint8) * 255

    return (img_corr, img_err)


def write_ovlp_masks(dir_path, y_true, y_pred):
    """
    Overlap ground truth with prediction and save to directory
    Red - false negative
    Green - true positive
    Blue - false negative
    :param dir_path: path to directory
    :param y_true: numpy array of ground truth labels
    :param y_pred: numpy array of predicted labels
    """
    dir_path = check_dir(dir_path)
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    assert (y_true.shape == y_pred.shape)
    ims = []
    num_slices = y_true.shape[0]
    for i in range(num_slices):
        slice_true = y_true[i, :, :]
        slice_pred = y_pred[i, :, :]
        img = generate_ovlp_image(slice_true, slice_pred)
        ims.append(img)

        slice_name = '%03d.png' % i
        cv2.imwrite(os.path.join(dir_path, slice_name), img)

    return ims


def write_mask(dir_path, y_true):
    """
    Save ground truth mask to directory
    :param dir_path: path to directory
    :param y_true: numpy array of ground truth labels
    """
    dir_path = check_dir(dir_path)
    y_true = np.squeeze(y_true) * 255
    num_slices = y_true.shape[0]
    for i in range(num_slices):
        slice_name = '%03d.png' % i
        cv2.imwrite(os.path.join(dir_path, slice_name), y_true[i, :, :])


def write_prob_map(dir_path, y_probs):
    """
    Write probablity map for prediction as image (colormap jet)
    :param dir_path: path to directory
    :param y_probs: numpy array of prediction probabilities
    """
    dir_path = check_dir(dir_path)
    y_probs = np.squeeze(y_probs)
    num_slices = y_probs.shape[0]
    for i in range(num_slices):
        slice_name = '%03d.png' % i
        im = y_probs[i, :, :] * 255
        im = im[..., np.newaxis].astype(np.uint8)
        imC = cv2.applyColorMap(im, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(dir_path, slice_name), imC)


def scale_img(im, scale=255):
    """
    Scale image from 0-scale
    :param im: input image
    :param scale: max value
    :return:
    """
    im = im.astype(np.float32)
    im = im - np.min(im)
    im = im / np.max(im)
    im *= scale

    return im


def generate_ovlp_image(y_true, y_pred):
    """
    Overlap ground truth and predicted labels
    :param y_true: numpy array of ground truth labels
    :param y_pred: numpy array of predicted labels
    :return: a BGR image
    """
    assert (y_true.shape == y_pred.shape)
    assert len(y_true.shape) == 2, "shape should be 2d, but is " + y_true.shape

    y_true = y_true.astype(np.bool)
    y_pred = y_pred.astype(np.bool)

    TP = y_true * y_pred
    FN = y_true * (~y_pred)
    FP = (~y_true) * y_pred

    # BGR format
    img = np.stack([FP, TP, FN], axis=-1).astype(np.uint8)
    img *= 255
    return img


OPACITY = 0.7


def save_ims(filepath):
    im_path = '%s.im' % filepath
    with h5py.File(im_path, 'r') as f:
        im = f['data'][:]

    seg_path = '%s.seg' % filepath
    with h5py.File(seg_path, 'r') as f:
        seg = f['data'][:].astype('float32')
        seg = seg[..., 0, 0]
    filepath = check_dir(filepath)
    # save ims
    cv2.imwrite(os.path.join(filepath, 'im.png'), scale_img(im))

    # save segs
    cv2.imwrite(os.path.join(filepath, 'seg.png'), scale_img(seg))

#
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     num_classes = 4
#     a = np.argmax(np.random.rand(100, 100, num_classes), axis=-1)
#     plt.imshow(a, cmap='gray')
#     plt.show()
#
#     mco = MultiClassOverlay(num_classes=num_classes)
#     l = mco.arr_to_label_img(a)
#     print(l.shape)
#     plt.imshow(np.squeeze(l), cmap='gray')
#     plt.show()
#
