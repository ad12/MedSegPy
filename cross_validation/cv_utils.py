import os
import warnings

K_BIN_SAVE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'oai_data-k%d.cv')

import utils


def load_cross_validation(k):
    return utils.load_pik(K_BIN_SAVE_PATH % k)


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
    for i in range(k / num_test_bins):
        test_bin_start_ind = test_bin_start_ind + i * num_test_bins
        valid_bin_start_ind = test_bin_start_ind + num_test_bins
        train_bin_start_ind = valid_bin_start_ind + num_valid_bins

        test_bins = [ind % k for ind in range(test_bin_start_ind, valid_bin_start_ind)]
        valid_bins = [ind % k for ind in range(valid_bin_start_ind, train_bin_start_ind)]
        train_bins = [ind % k for ind in range(train_bin_start_ind, train_bin_start_ind + num_train)]

        assert len(set(train_bins) & set(valid_bins)) == 0, "Training and Validation bins must be mutually exclusive"
        assert len(set(train_bins) & set(test_bins)) == 0, "Training and Test bins must be mutually exclusive"
        assert len(set(valid_bins) & set(test_bins)) == 0, "Validation and Test bins must be mutually exclusive"

        exps_bin_division.append((train_bins, valid_bins, test_bins))

    # Check to make sure all test bins are mutually exclusive
    temp = []
    for d in exps_bin_division:
        temp.append(d[-1])

    for i in range(len(temp)):
        for j in range(i + 1, len(temp)):
            assert len(set(temp[i]) & set(temp[j])) == 0, "Test bins %d and %d not mutually exclusive" % (i, j)

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
