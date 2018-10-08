import ast
import configparser
import os
import pickle
import re

import cv2
import h5py
import numpy as np


def check_dir(dir_path):
    """
    If directory does not exist, make directory
    :param dir_path: path to directory
    :return: path to directory
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


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

        overlap_img = cv2.addWeighted(x, 1, im, 0.3, 0)
        cv2.imwrite(os.path.join(dir_path, slice_name), overlap_img)


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
        imC = cv2.applyColorMap(im, cv2.COLORMAP_JET)
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


def save_pik(data, filepath):
    """
    Save data using pickle
    :param data: data to save
    :param filepath: a string
    :return:
    """
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pik(filepath):
    """
    Load data using pickle
    :param filepath: filepath to load from
    :return: data saved using save_pik
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_config(a_dict, filepath):
    """
    Save information in a dictionary
    :param a_dict: a dictionary of information to save
    :param filepath: a string
    :return:
    """
    config = configparser.ConfigParser(a_dict)

    with open(filepath, 'w+') as configfile:
        config.write(configfile)


def load_config(filepath):
    """
    Read in information saved using save_config
    :param filepath: a string
    :return: a dictionary of Config params
    """
    config = configparser.ConfigParser()
    config.read(filepath)

    return config['DEFAULT']


def convert_data_type(var_string, data_type):
    """
    Convert string to relevant data type
    :param var_string: variable as a string (e.g.: '[0]', '1', '2.0', 'hellow')
    :param data_type: the type of the data
    :return: string converted to data_type
    """
    if (data_type is str):
        return var_string

    if (data_type is float):
        return float(var_string)

    if (data_type is int):
        return int(var_string)

    if (data_type is bool):
        return ast.literal_eval(var_string)

    if (data_type is list):
        return ast.literal_eval(var_string)

    if (data_type is tuple):
        return ast.literal_eval(var_string)


def get_weights(base_folder):
    """
    Gets the best weights file inside the base_folder
    :param base_folder: dirpath where weights are stored
    :return: h5 file

    Assumes that only the best weights are stored, so searching for the epoch should be enough
    """
    files = os.listdir(base_folder)
    max_epoch = -1
    best_file = ''
    for file in files:
        file_fullpath = os.path.join(base_folder, file)
        # Ensure the file is an h5 file
        if not (os.path.isfile(file_fullpath) and file_fullpath.endswith('.h5') and 'weights' in file):
            continue

        # Get file with max epochs
        train_info = file.split('.')[1]
        epoch = int(train_info.split('-')[0])

        if (epoch > max_epoch):
            max_epoch = epoch
            best_file = file_fullpath

    return best_file


def save_optimizer(optimizer, dirpath):
    """
    Serialize a model and add the config of the optimizer
    :param optimizer: a Keras optimizer
    :param dirpath: path to directory
    :return:
    """
    if optimizer is None:
        return

    config = dict()
    config['optimizer'] = optimizer.get_config()

    filepath = os.path.join(dirpath, 'optimizer.dat')
    # Save optimizer state
    save_pik(config, filepath)


def load_optimizer(dirpath):
    """
    Return model and optimizer in previous state
    :param dirpath: path to directory storing optimizer
    :return: optimizer
    """
    from keras import optimizers
    filepath = os.path.join(dirpath, 'optimizer.dat')
    model_dict = load_pik(filepath)
    optimizer_params = dict([(k, v) for k, v in model_dict.get('optimizer').items()])
    optimizer = optimizers.get(optimizer_params)

    return optimizer


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


def parse_results_file(filepath):
    # returns mean
    with open(filepath) as search:
        for line in search:
            line = line.rstrip()  # remove '\n' at end of line
            if 'MEAN' not in line.upper() or 'DSC' not in line.upper():
                continue

            vals = re.findall("\d+\.\d+", line)
            return float(vals[0])


def calc_cv(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    cv = np.std([np.sum(y_true), np.sum(y_pred)]) / np.mean([np.sum(y_true), np.sum(y_pred)])
    return cv


if __name__ == '__main__':
    save_ims('./test_data/9968924_V01-Aug00_056')
