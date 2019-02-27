import ast
import os

from utils.im_utils import save_ims


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


if __name__ == '__main__':
    save_ims('./test_data/9968924_V01-Aug00_056')
