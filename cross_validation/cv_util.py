import os
import random
import re
import sys
import typing
import warnings
from time import strftime, localtime
from typing import Union

sys.path.append('../')

sys.path.append('../')
from utils import io_utils

# Change based on dataset used for analysis. Do not use any digits
DATASET_NAME = 'oai_imorphics'

# =======DO NOT CHANGE CONSTANTS BELOW=======
K_REGEX_PATTERN = '_cv-k[0-9]+'
K_BIN_FILENAME_BASE = DATASET_NAME + '_cv-k%d'
K_BIN_SAVE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


class CrossValidationProcessor():
    """A processor class for handling computation for cross-validation experiments"""

    __k_NUM_VALID_BINS_KEY = 'num_valid_bins'
    __k_NUM_TEST_BINS_KEY = 'num_test_bins'

    def __init__(self, k_or_filepath: Union[int, str], **kwargs):
        """
        :param k_or_filepath: an int describing the number of bins (k)
                              or a filepath to an existing cross-validation file
        :param \**kwargs: See below

        :keyword num_valid_bins: Number of bins allocated for validation. Must be used with keyword num_test_bins.
        :keyword num_test_bins: Number of bins allocated for testing. Must be used with keyword num_valid_bins.
        """
        if type(k_or_filepath) not in [int, str]:
            raise ValueError('`k_or_filepath` must be either the k value (int) or the filepath (str)')

        if type(k_or_filepath) is int:
            k = k_or_filepath
            filepaths = self.__get_cross_validation_files(k)

            if not filepaths:
                raise ValueError('No file found for k=%d. Create file with `create_cv_bins.py`' % k)

            if len(filepaths) != 1:
                raise ValueError('Multiple files corresponding to k=%d. Please specify explicit filepath.')

            filepath = filepaths[0]
        else:
            filepath = k_or_filepath

        if not os.path.isfile(filepath):
            raise FileNotFoundError('File %s not found' % filepath)

        k_from_filename = self.__get_k_from_file(filepath)
        cv_data = io_utils.load_pik(filepath)
        assert len(cv_data) == k_from_filename, "Corrupted file: mismatch bins"

        self._filepath = filepath
        self._k = k_from_filename
        self._bin_files = cv_data

        # These fields should be appropriately populated by instance methods
        self.num_valid_bins = 0
        self.num_test_bins = 0
        self.bins_split = None

        # Handle kwargs
        if self.__k_NUM_VALID_BINS_KEY in kwargs and self.__k_NUM_TEST_BINS_KEY in kwargs:
            self.init_cv_experiments(kwargs.get('num_valid_bins'), kwargs.get('num_test_bins'))

        if (self.__k_NUM_VALID_BINS_KEY not in kwargs and self.__k_NUM_TEST_BINS_KEY in kwargs) or \
                (self.__k_NUM_VALID_BINS_KEY in kwargs and self.__k_NUM_TEST_BINS_KEY not in kwargs):
            raise ValueError('%s and %s must be specified together' % (self.__k_NUM_VALID_BINS_KEY,
                                                                       self.__k_NUM_TEST_BINS_KEY))

    def init_cv_experiments(self, num_valid_bins=1, num_test_bins=1):
        """
        Organize k bins into training, validation, testing bins for k/num_test_bins experiments

        :param num_valid_bins: Number of bins for validation
        :param num_test_bins: Number of bins for testing. Must be perfect divisor of k (k % num_test_bins = 0)

        """
        k = self.k

        # number of holdout bins cannot exceed k
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

            assert len(
                set(train_bins) & set(valid_bins)) == 0, "Training and Validation bins must be mutually exclusive"
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
                assert len(
                    set(temp[i]) & set(temp[j])) == 0, "Test bins %d and %d not mutually exclusive - %d overlap" % (
                    i, j, len(set(temp[i]) & set(temp[j])))

        self.num_valid_bins = num_valid_bins
        self.num_test_bins = num_test_bins
        self.bins_split = exps_bin_division

    def run(self):
        """
        Yields training, validation, testing files and train, validation, testing bin indexes as single tuple
        """
        bins_split = self.bins_split
        bins_files = self.bin_files

        if not bins_split:
            raise RuntimeError('Cross-validation experiment is not initialized. Call init_cv_experiments')

        for bin_inds in bins_split:
            assert len(bin_inds) == 3, "Expected 3 sets of bin indices - train, valid, test"
            train_bins, valid_bins, test_bins = tuple(bin_inds)
            train_files, valid_files, test_files = self.get_fnames((train_bins, valid_bins, test_bins))

            yield train_files, valid_files, test_files, train_bins, valid_bins, test_bins

    def get_fnames(self, bin_inds):
        """
        Get filepaths for training, validation, and testing
        :param bin_inds: A tuple of indexes for training, validation, and testing respectively

        :return: A tuple of lists. Each list consists of filepaths for training, validation, testing respectively
        """
        bin_files = self.bin_files
        train_inds, valid_inds, test_inds = bin_inds

        train_files = [bin_files[x] for x in train_inds]
        train_files = [filepath for x in train_files for filepath in x]

        valid_files = [bin_files[x] for x in valid_inds]
        valid_files = [filepath for x in valid_files for filepath in x]

        test_files = [bin_files[x] for x in test_inds]
        test_files = [filepath for x in test_files for filepath in x]

        return train_files, valid_files, test_files

    @staticmethod
    def __get_cross_validation_files(k: int) -> tuple:
        """
        Get list of cross-validation files for given k value
        :param k: The k value (k-fold cross validation)

        :return: A tuple of filepaths to .cv files
        """
        save_directory = K_BIN_SAVE_DIRECTORY
        base_name = K_BIN_FILENAME_BASE % k

        files = []
        for f in os.listdir(save_directory):
            if base_name in f:
                files.append(os.path.join(save_directory, f))

        return tuple(files)

    @staticmethod
    def __get_k_from_file(filepath: str) -> int:
        """
        Get k value (k-fold) from filepath
        :param filepath: A filepath to a cross-validation (.cv) file

        :return: An int
        """
        filename = os.path.basename(filepath)

        matches = re.findall(K_REGEX_PATTERN, filename)
        if len(matches) > 1:
            warnings.warn('Multiple matches found - using match at 0th index')
        match = matches[0]

        return int(re.findall('[0-9]+', match)[0])

    @property
    def filepath(self):
        return self._filepath

    @property
    def k(self):
        return self._k

    @property
    def bin_files(self):
        return self._bin_files


class CrossValidationFileGenerator():
    """A generator class for creating file for cross validation data"""

    def __init__(self, k_bins: int, data_paths: typing.Collection[str], dataset_tag: str, overwrite: bool = False):
        """
        :param k_bins: Number of bins to create for k-fold cross-validation
        :param data_paths: Collection of paths where data is stored. Invalid filepaths will be ignored
        :param dataset_tag: An identifier for the data used for cross validation
        :param overwrite: Whether existing file should be overwritten (optional)
        """
        self.k_bins = k_bins
        self.data_paths = data_paths

        fname = dataset_tag + '_cv-k%d-%s.cv' % (k_bins, strftime("%Y-%m-%d-%H-%M-%S", localtime()))
        self.save_path = os.path.join(K_BIN_SAVE_DIRECTORY, fname)
        tmp_path = os.path.join(K_BIN_SAVE_DIRECTORY, '%s_cv-k%d.cv' % ('tmp', k_bins))

        save_path = self.save_path

        # Overwrite file if exists
        if os.path.isfile(save_path):
            if not overwrite:
                raise FileExistsError('File %s exists.' % save_path)

        bins = self.generate_bins()

        # save data to filepath
        io_utils.save_pik(bins, tmp_path)

        try:
            # Verify list
            self.verify_bins(tmp_path, k_bins)
        except Exception as e:
            os.remove(tmp_path)
            raise e
        
        os.remove(tmp_path)
        io_utils.save_pik(bins, self.save_path)

    def verify_bins(self, filepath, expected_k):
        cv_processor = CrossValidationProcessor(filepath)

        # verify number of bins are expected length
        assert cv_processor.k == expected_k, "Expected %d bins, got %d bins" % (expected_k, cv_processor.k)
        k_bins = cv_processor.k
        bins = cv_processor.bin_files

        bin_to_pid_dict = dict()
        bin_to_scanid_dict = dict()

        for bin_id in range(k_bins):
            bin = bins[bin_id]
            pids = []
            scan_ids = []
            for filepath in bin:
                file_info = self.__get_file_info(os.path.basename(filepath), os.path.dirname(filepath))
                pids.append(file_info['pid'])
                scan_ids.append(file_info['scanid'])

            bin_to_pid_dict[bin_id] = list(set(pids))
            bin_to_scanid_dict[bin_id] = list(set(scan_ids))

        max_num_pids = max([len(bin_to_pid_dict[bin_id]) for bin_id in range(k_bins)])
        min_num_pids = min([len(bin_to_pid_dict[bin_id]) for bin_id in range(k_bins)])
        assert max_num_pids - min_num_pids <= 1, "Difference in number of subjects between bins should be <= 1"

        # max_num_scan_ids = max([len(bin_to_scanid_dict[bin_id]) for bin_id in range(k)])
        # min_num_scan_ids = min([len(bin_to_scanid_dict[bin_id]) for bin_id in range(k)])
        # assert max_num_scan_ids - min_num_scan_ids <= 2  # each pid has 2 scans

    def generate_bins(self):
        # get pid dictionary
        pids_dict = self.__parse_pids()
        pids = list(pids_dict.keys())

        # Shuffle pids in random order
        random.shuffle(pids)

        # Allocate each pid to a bin
        bins_list = self.__get_bins_list(len(pids), self.k_bins)
        pid_bin_map = dict()
        for i in range(len(pids)):
            pid_bin_map[pids[i]] = bins_list[i]

        bins = []  # stores filepaths for scans/scan slices from patients with pid corresponding to this bin
        for i in range(self.k_bins):
            bins.append([])
        for dp in self.data_paths:
            for fname in os.listdir(dp):
                if fname.endswith('.im'):
                    im_info = self.__get_file_info(fname, dp)

                    # Get bin that this pid should be in
                    bin_id = pid_bin_map[im_info['pid']]
                    filepath = os.path.join(dp, im_info['fname'])

                    bins[bin_id].append(filepath)

        # Check that bins are mutually exclusive
        for i in range(len(bins)):
            for j in range(i + 1, len(bins)):
                if len(set(bins[i]) & set(bins[j])) != 0:
                    overlap = list(set(bins[i]) & set(bins[j]))
                    overlap.sort()
                    for fp in overlap:
                        print(fp)
                    print(pid_bin_map)
                    raise ValueError('Bins %d and %d not exclusive' % (i, j))

            # Check for duplicates
        self.check_duplicates(bins)

        return bins

    def __parse_pids(self):
        """Get patient identifiers"""
        pids = dict()
        for dp in self.data_paths:
            for fname in os.listdir(dp):
                if fname.endswith('.im'):
                    im_info = self.__get_file_info(fname, dp)
                    curr_pid = im_info['pid']
                    if curr_pid in pids.keys():
                        assert dp == pids[curr_pid], "dirpath mismatch. Expected: %s, got %s" % (dp, pids[curr_pid])
                    else:
                        pids[curr_pid] = dp

        pids_dict = pids.copy()

        return pids_dict

    def __get_file_info(self, fname, dirpath):
        fname, ext = os.path.splitext(fname)
        f_data = fname.split('-')
        scan_id = f_data[0]
        pid_timepoint_split = scan_id.split('_')
        pid = pid_timepoint_split[0]
        f_aug_slice = f_data[1].split('_')
        data = {'pid': pid,
                'timepoint': int(pid_timepoint_split[1][1:]),
                'aug': int(f_aug_slice[0][3:]),
                'slice': int(f_aug_slice[1]),
                'fname': fname,
                'impath': os.path.join(dirpath, '%s.%s' % (fname, 'im')),
                'segpath': os.path.join(dirpath, '%s.%s' % (fname, 'seg')),
                'scanid': scan_id}
        assert data['pid'] == fname[:7], str(data)

        return data

    def __get_bins_list(self, num_pids, k):
        num_bins = [int(num_pids / k)] * k
        remainder_num_elements = num_pids % k

        # randomly pick which bins take on additional elements
        remainder_bin_list = random.sample(range(k), remainder_num_elements)
        for bin_id in remainder_bin_list:
            num_bins[bin_id] += 1

        assert max(num_bins) - min(num_bins) <= 1
        assert len(num_bins) == k

        # Each pid should get a bin
        bin_ids = [[bin_id] * num_bins[bin_id] for bin_id in range(k)]

        # flatten list
        bin_ids = [x for id_list in bin_ids for x in id_list]

        assert len(bin_ids) == num_pids

        return bin_ids

    @staticmethod
    def check_duplicates(x_list: list):
        for i in range(len(x_list)):
            if len(x_list[i]) != len(set(x_list[i])):
                raise ValueError('Duplicates in list %d' % i)

# if __name__ == '__main__':
#     pass
#     # print(get_cv_experiments(6, num_test_bins=2))
