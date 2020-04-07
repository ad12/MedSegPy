import os
import subprocess


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


def get_available_gpus(num_gpus: int=None):
    """Get gpu ids for gpus that are >95% free.

    Tensorflow does not support checking free memory on gpus.
    This is a crude method that relies on `nvidia-smi` to
    determine which gpus are occupied and which are free.
    
    Args:
        num_gpus: Number of requested gpus. If not specified,
            ids of all available gpu(s) are returned.
    Returns:
        List[int]: List of gpu ids that are free. Length
            will equal `num_gpus`, if specified.
    """
    # Built-in tensorflow gpu id.
    assert isinstance(num_gpus, (type(None), int))
    if num_gpus == 0:
        return [-1]

    num_requested_gpus = num_gpus
    num_gpus = len(subprocess.check_output("nvidia-smi --list-gpus", shell=True).decode().split("\n")) - 1

    out_str = subprocess.check_output("nvidia-smi | grep MiB", shell=True).decode()
    mem_str = [x for x in out_str.split() if "MiB" in x]
    # First 2 * num_gpu elements correspond to memory for gpus
    # Order: (occupied-0, total-0, occupied-1, total-1, ...)
    mems = [float(x[:-3]) for x in mem_str]
    gpu_percent_occupied_mem = [mems[2*gpu_id] / mems[2*gpu_id+1] for gpu_id in range(num_gpus)]

    available_gpus = [gpu_id for gpu_id, mem in enumerate(gpu_percent_occupied_mem) if mem < 0.05]
    if num_requested_gpus and num_requested_gpus > len(available_gpus):
        raise ValueError("Requested {} gpus, only {} are free".format(num_requested_gpus, len(available_gpus)))

    return available_gpus[:num_requested_gpus] if num_requested_gpus else available_gpus
