import os
import subprocess


def get_weights(experiment_dir):
    """Gets the weights file corresponding to lowest validation loss.

    Assumes that only the best weights are stored, so searching for the epoch
    should be enough.
    TODO: remove this assumption.

    Args:
        experiment_dir (str): Experiment directory where weights are stored.

    Returns:
        str: Path to weights h5 file.
    """
    files = os.listdir(experiment_dir)
    max_epoch = -1
    best_file = ''
    for file in files:
        file_fullpath = os.path.join(experiment_dir, file)
        # Ensure the file is an h5 file
        if not (os.path.isfile(file_fullpath) and file_fullpath.endswith('.h5') and 'weights' in file):
            continue

        # Get file with max epochs
        train_info = file.split('.')[1]
        epoch = int(train_info.split('-')[0])

        if epoch > max_epoch:
            max_epoch = epoch
            best_file = file_fullpath

    if not best_file:
        raise FileNotFoundError('No weights file found in %s' % experiment_dir)

    return best_file


def _check_results_file(base_path):
    """Recursively check for results.txt file.
    """
    if (base_path is None) or (not os.path.isdir(base_path)) or (base_path == ''):
        return []

    results_filepath = os.path.join(base_path, 'results.txt')

    results_paths = []
    if os.path.isfile(results_filepath):
        results_paths.append(results_filepath)

    files = os.listdir(base_path)
    for file in files:
        possible_dir = os.path.join(base_path, file)
        if os.path.isdir(possible_dir):
            subdir_results_files = _check_results_file(possible_dir)
            results_paths.extend(subdir_results_files)

    return results_paths


def get_valid_subdirs(root_dir: str, exist_ok: bool = False):
    """Recursively search for experiments that are ready to be tested.

    Different experiments live in different folders. Based on training protocol,
    we assume that an valid experiment has completed training if its folder
    contains files "config.ini" and "pik_data.dat".

    To avoid recomputing experiments with results, `exist_ok=False` by default.

    Args:
        root_dir (str): Root folder to search.
        exist_ok (:obj:`bool`, optional): If `True`, recompute results for
            experiments.

    Return:
        List[str]: Experiment directories to test.
    """
    no_results = not exist_ok
    if (root_dir is None) or (not os.path.isdir(root_dir)) or (root_dir == []):
        return []

    subdirs = []
    config_path = os.path.join(root_dir, 'config.ini')
    pik_data_path = os.path.join(root_dir, 'pik_data.dat')
    test_results_dirpath = os.path.join(root_dir, 'test_results')
    results_file_exists = len(_check_results_file(test_results_dirpath)) > 0

    # 1. Check if you are a valid subdirectory - must contain a pik data path
    if os.path.isfile(config_path) and os.path.isfile(pik_data_path):
        if (no_results and (not results_file_exists)) or ((not no_results)):
            subdirs.append(root_dir)

    files = os.listdir(root_dir)
    # 2. Recursively search through other subdirectories
    for file in files:
        possible_dir = os.path.join(root_dir, file)
        if os.path.isdir(possible_dir):
            rec_subdirs = get_valid_subdirs(possible_dir, no_results)
            subdirs.extend(rec_subdirs)

    return subdirs


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


def num_gpus():
    if "CUDA_VISIBLE_DEVICES" not in os.environ \
        or not os.environ["CUDA_VISIBLE_DEVICES"]:
        return 0

    return len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
