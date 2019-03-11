"""
Create K bins for K-fold cross-validation

Data is stored in Pickle format
"""

import os
import random
import sys
from time import strftime, localtime

sys.path.append('../')
from utils import io_utils
from cross_validation import cv_utils

DATA_PATHS = ['/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug/',
              '/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid/',
              '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test']
K_BIN_FILENAME_BASE = 'oai_cv-k%d'  # Do not change unless
K_BIN_SAVE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), K_BIN_FILENAME_BASE + '-%s.cv')


def get_file_info(fname, dirpath):
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


def get_bins_list(num_pids, k):
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


def check_duplicates(x_list: list):
    for i in range(len(x_list)):
        if len(x_list[i]) != len(set(x_list[i])):
            raise ValueError('Duplicates in list %d' % i)


def verify_bins(k):
    bins = cv_utils.load_cross_validation(k)
    assert len(bins) == k

    bin_to_pid_dict = dict()
    bin_to_scanid_dict = dict()

    for bin_id in range(k):
        bin = bins[bin_id]
        pids = []
        scan_ids = []
        for filepath in bin:
            file_info = get_file_info(os.path.basename(filepath), os.path.dirname(filepath))
            pids.append(file_info['pid'])
            scan_ids.append(file_info['scanid'])

        bin_to_pid_dict[bin_id] = list(set(pids))
        bin_to_scanid_dict[bin_id] = list(set(scan_ids))

    max_num_pids = max([len(bin_to_pid_dict[bin_id]) for bin_id in range(k)])
    min_num_pids = min([len(bin_to_pid_dict[bin_id]) for bin_id in range(k)])
    assert max_num_pids - min_num_pids <= 1

    max_num_scan_ids = max([len(bin_to_scanid_dict[bin_id]) for bin_id in range(k)])
    min_num_scan_ids = min([len(bin_to_scanid_dict[bin_id]) for bin_id in range(k)])
    assert max_num_scan_ids - min_num_scan_ids <= 2  # each pid has 2 scans


if __name__ == '__main__':
    k = int(sys.argv[1])
    save_path = K_BIN_SAVE_PATH % (k, strftime("%Y-%m-%d-%H-%M-%S", localtime()))
    base_name = K_BIN_FILENAME_BASE % k

    save_directory = os.path.dirname(save_path)
    for f in os.listdir(save_directory):
        if base_name in f:
            raise FileExistsError(
                'Cross-validation with %d bins already exists (%s). To overwrite, manually delete previous file' % (
                k, os.path.join(save_directory, f)))

    # Get all patient ids (pids)
    pids = dict()
    for dp in DATA_PATHS:
        for fname in os.listdir(dp):
            if fname.endswith('.im'):
                im_info = get_file_info(fname, dp)
                curr_pid = im_info['pid']
                if curr_pid in pids.keys():
                    assert dp == pids[curr_pid], "dirpath mismatch. Expected: %s, got %s" % (dp, pids[curr_pid])
                else:
                    pids[curr_pid] = dp

    pids_dict = pids.copy()
    pids = list(pids.keys())

    # Shuffle pids in random order
    random.shuffle(pids)

    # Allocate each pid to a bin
    bins_list = get_bins_list(len(pids), k)
    pid_bin_map = dict()
    for i in range(len(pids)):
        pid_bin_map[pids[i]] = bins_list[i]

    bins = []  # stores filepaths for scans/scan slices from patients with pid corresponding to this bin
    for i in range(k):
        bins.append([])
    for dp in DATA_PATHS:
        for fname in os.listdir(dp):
            if fname.endswith('.im'):
                im_info = get_file_info(fname, dp)

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
    check_duplicates(bins)

    # save data to filepath
    io_utils.save_pik(bins, save_path)

    try:
        # Verify list
        verify_bins(k)
    except Exception as e:
        os.remove(save_path)
        raise e
