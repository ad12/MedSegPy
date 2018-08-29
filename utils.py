import os
import numpy as np
import cv2
import pickle
import configparser
import ast

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path

def write_im_overlay(dir_path, xs, im_overlay):
    num_slices = xs.shape[0]
    for i in range(num_slices):
        x = xs[i, ...]
        im = im_overlay[i]

        slice_name = '%03d.png' % i

        overlap_img = cv2.addWeighted(x, 1, im_overlay, 0.3, 0)
        cv2.imwrite(os.path.join(dir_path, slice_name), overlap_img)

def write_ovlp_masks(dir_path, y_true, y_pred):
    dir_path = check_dir(dir_path)
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    assert(y_true.shape == y_pred.shape)
    ims = []
    num_slices = y_true.shape[0]
    for i in range(num_slices):
        slice_true = y_true[i, :, :]
        slice_pred = y_pred[i, :, :]
        img = generate_ovlp_image(slice_true, slice_pred)
        ims.append(img)

        slice_name = '%03d.png' % i
        cv2.imwrite(os.path.join(dir_path, slice_name), img)

def write_mask(dir_path, y_true):
    dir_path = check_dir(dir_path)
    y_true = np.squeeze(y_true) * 255
    num_slices = y_true.shape[0]
    for i in range(num_slices):
        slice_name = '%03d.png' % i
        cv2.imwrite(os.path.join(dir_path, slice_name), y_true[i,:,:])

def write_prob_map(dir_path, y_probs):
    dir_path = check_dir(dir_path)
    y_probs = np.squeeze(y_probs)
    num_slices = y_probs.shape[0]
    for i in range(num_slices):
        slice_name = '%03d.png' % i
        imC = cv2.applyColorMap(y_probs[i, :, :], cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(dir_path, slice_name), imC)


def generate_ovlp_image(y_true, y_pred):
    assert(y_true.shape == y_pred.shape)
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
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pik(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_config(a_dict, filepath):
    config = configparser.ConfigParser(a_dict)

    with open(filepath, 'w+') as configfile:
        config.write(configfile)

def load_config(filepath):
    config = configparser.ConfigParser()
    config.read(filepath)

    return config['DEFAULT']

def convert_data_type(var_string, data_type):
    if (data_type is str):
        return var_string

    if (data_type is float):
        return float(var_string)

    if (data_type is int):
        return int(var_string)

    if (data_type is bool):
        return bool(data_type)

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
        if not(os.path.isfile(file_fullpath) and file_fullpath.endswith('.h5') and 'weights' in file):
            continue

        # Get file with max epochs
        train_info = file.split('.')[1]
        epoch = int(train_info.split('-')[0])

        if (epoch > max_epoch):
            max_epoch = epoch
            best_file = file_fullpath

    return best_file

def save_optimizer(optimizer, dirpath):
    """Serialize a model and add the config of the optimizer
    """
    if optimizer is None:
        return

    config = dict()
    config['optimizer'] = optimizer.get_config()

    filepath = os.path.join(dirpath, 'optimizer.dat')
    # Save optimizer state
    save_pik(config, filepath)

def load_optimizer(dirpath):
    """ Return model and optimizer in previous state
    """
    from keras import optimizers
    filepath = os.path.join(dirpath, 'optimizer.dat')
    model_dict = load_pik(filepath)
    optimizer_params = dict([(k,v) for k,v in model_dict.get('optimizer').items()])
    optimizer = optimizers.get(optimizer_params)

    return optimizer

if __name__ == '__main__':
    from config import DeeplabV3Config

    config = DeeplabV3Config(create_dirs=False)
    config.load_config('test_data/config.ini')

    print('')
