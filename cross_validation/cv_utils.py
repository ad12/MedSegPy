import os
import re
import sys
import warnings

DATASET_NAME = 'oai_imorphics'  # Change based on dataset used for analysis. Do not use any digits

# DO NOT CHANGE CONSTANTS BELOW
K_REGEX_PATTERN = '_cv-k[0-9]+'
K_BIN_FILENAME_BASE = DATASET_NAME + '_cv-k%d'
K_BIN_SAVE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

sys.path.append('../')
from utils import io_utils


class CrossValidationWrapper():
    def __init__(self, k_or_filepath):
        if type(k_or_filepath) not in [int, str]:
            raise ValueError('`k_or_filepath` must be either the k value (int) or the filepath (str)')

        if type(k_or_filepath) is int:
            k = k_or_filepath
            filepath = get_cross_validation_file(k)
            if not filepath:
                raise ValueError('No file found for k=%d' % k)
        else:
            filepath = k_or_filepath

        if not os.path.isfile(filepath):
            raise FileNotFoundError('File %s not found' % filepath)

        self._filepath = filepath
        self._k = get_k_from_file(filepath)

    @property
    def filepath(self):
        return self._filepath

    @property
    def k(self):
        return self._k


def get_cross_validation_file(k):
    save_directory = K_BIN_SAVE_DIRECTORY
    base_name = K_BIN_FILENAME_BASE % k
    for f in os.listdir(save_directory):
        if base_name in f:
            return os.path.join(save_directory, f)

    return None


def get_k_from_file(filepath: str) -> int:
    filename = os.path.basename(filepath)

    matches = re.findall(K_REGEX_PATTERN, filename)
    if len(matches) > 1:
        warnings.warn('Multiple matches found - using match at 0th index')
    match = matches[0]

    return int(re.findall('[0-9]+', match)[0])


def load_cross_validation(k_or_filepath):
    if type(k_or_filepath) not in [int, str]:
        raise ValueError('`k_or_filepath` must be either the k value (int) or the filepath (str)')

    if type(k_or_filepath) is int:
        k = k_or_filepath
        filepath = get_cross_validation_file(k)
        if not filepath:
            raise ValueError('No file found for k=%d' % k)
    else:
        filepath = k_or_filepath

    if not os.path.isfile(filepath):
        raise FileNotFoundError('File %s not found' % filepath)

    print('Loading %d-fold cross-validation data from %s...' % (get_k_from_file(filepath), filepath))
    return io_utils.load_pik(filepath)


def get_cv_experiments(k, num_valid_bins=1, num_test_bins=1):
    num_holdout = num_valid_bins + num_test_bins
    if num_holdout > k:
        raise ValueError('Number of holdout bins (validation + test) must be < k')

    # inference sets cannot overlap in cross-validation
    if k % num_test_bins != 0:
        raise ValueError("There can be no overlap in test bins across different cross-validation trials")

    if num_holdout / k > 0.4:
        warnings.warn('%0.1f holdout - validation: %0.1f, test: %0.1f' % (num_holdout / k * 100,
                                                                          num_valid_bins / k * 100,
                                                                          num_test_bins / k * 100))
    num_train = k - num_holdout

    test_bin_start_ind = 0
    exps_bin_division = []
    for i in range(int(k / num_test_bins)):
        valid_bin_start_ind = test_bin_start_ind + num_test_bins
        train_bin_start_ind = valid_bin_start_ind + num_valid_bins

        test_bins = [ind % k for ind in range(test_bin_start_ind, valid_bin_start_ind)]
        valid_bins = [ind % k for ind in range(valid_bin_start_ind, train_bin_start_ind)]
        train_bins = [ind % k for ind in range(train_bin_start_ind, train_bin_start_ind + num_train)]

        assert len(set(train_bins) & set(valid_bins)) == 0, "Training and Validation bins must be mutually exclusive"
        assert len(set(train_bins) & set(test_bins)) == 0, "Training and Test bins must be mutually exclusive"
        assert len(set(valid_bins) & set(test_bins)) == 0, "Validation and Test bins must be mutually exclusive"

        exps_bin_division.append((train_bins, valid_bins, test_bins))

        test_bin_start_ind += num_test_bins

    # Check to make sure all test bins are mutually exclusive
    temp = []
    for d in exps_bin_division:
        temp.append(d[-1])

    for i in range(len(temp)):
        for j in range(i + 1, len(temp)):
            assert len(set(temp[i]) & set(temp[j])) == 0, "Test bins %d and %d not mutually exclusive - %d overlap" % (
            i, j, len(set(temp[i]) & set(temp[j])))

    return exps_bin_division


def get_fnames(bins_files, bin_inds):
    train_inds, valid_inds, test_inds = bin_inds

    train_files = [bins_files[x] for x in train_inds]
    train_files = [filepath for x in train_files for filepath in x]

    valid_files = [bins_files[x] for x in valid_inds]
    valid_files = [filepath for x in valid_files for filepath in x]

    test_files = [bins_files[x] for x in test_inds]
    test_files = [filepath for x in test_files for filepath in x]

    return train_files, valid_files, test_files


if __name__ == '__main__':
    print(get_cv_experiments(6, num_test_bins=2))
