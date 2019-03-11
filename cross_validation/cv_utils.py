import os
import sys
import warnings

K_BIN_FILENAME_BASE = 'oai_cv-k%d'  # Do not change unless
K_BIN_SAVE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

sys.path.append('../')
from utils import io_utils


def get_cross_validation_file(k):
    save_directory = K_BIN_SAVE_DIRECTORY
    base_name = K_BIN_FILENAME_BASE % k
    for f in os.listdir(save_directory):
        if base_name in f:
            return os.path.join(save_directory, f)

    return None


def load_cross_validation(k):
    filepath = get_cross_validation_file(k)

    print('Loading %d-fold cross-validation data from %s...' % (k, filepath))
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
