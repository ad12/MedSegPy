import os
import pickle
import re

import h5py


def parse_results_file(filepath):
    # returns mean
    with open(filepath) as search:
        for line in search:
            line = line.rstrip()  # remove '\n' at end of line
            if 'MEAN' not in line.upper() or 'DSC' not in line.upper():
                continue

            vals = re.findall("\d+\.\d+", line)
            return float(vals[0])


def load_h5(filepath):
    """Load data in H5DF format
    :param filepath: path to h5 file
    :return: dictionary of data values stored using save_h5
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError('%s does not exist' % filepath)

    data = dict()
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = f.get(key).value

    return data


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


def check_dir(dir_path):
    """
    If directory does not exist, make directory
    :param dir_path: path to directory
    :return: path to directory
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path
