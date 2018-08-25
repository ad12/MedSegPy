import os
import numpy as np
import cv2
import pickle
import configparser


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path

def write_ovlp_masks(dir_path, y_true, y_pred):
    dir_path = check_dir(dir_path)
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    assert(y_true.shape == y_pred.shape)
    
    num_slices = y_true.shape[0]
    for i in range(num_slices):
        slice_true = y_true[i, :, :]
        slice_pred = y_pred[i, :, :]
        img = generate_ovlp_image(slice_true, slice_pred)
        
        slice_name = '%03d.png' % i
        cv2.imwrite(os.path.join(dir_path, slice_name), img)

def write_mask(dir_path, y_true):
    dir_path = check_dir(dir_path)
    y_true = np.squeeze(y_true) * 255
    num_slices = y_true.shape[0]
    for i in range(num_slices):
        slice_name = '%03d.png' % i
        cv2.imwrite(os.path.join(dir_path, slice_name), y_true[i,:,:])        

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
        # Ensure the file is an h5 file
        if not(os.path.isfile(file) and file.endswith('.h5')):
            continue

        # Get file with max epochs
        train_info = file.split('.')[1]
        epoch = int(train_info.split('-')[0])

        if (epoch > max_epoch):
            max_epoch = epoch
            best_file = file

    return os.path.join(base_folder, best_file)

if __name__ == '__main__':
    from config import DeeplabV3Config

    config = DeeplabV3Config(create_dirs=False)
    config.load_config('test_data/config.ini')
