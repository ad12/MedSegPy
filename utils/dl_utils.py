import os


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

    if not best_file:
        raise FileNotFoundError('No weights file found in %s' % base_folder)

    return best_file