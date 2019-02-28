"""
Create K bins for K-fold cross-validation

Data is stored in Pickle format
"""

import os
import random
import sys

sys.path.append('../')
from utils import io_utils

DATA_PATHS = ['/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug/',
              '/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid/',
              '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test']
K_BIN_SAVE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'oai_data-k%d.cv')


def get_file_info(fname, dirpath):
    fname, ext = os.path.splitext(fname)
    f_data = fname.split('-')
    scan_id = f_data[0].split('_')
    f_aug_slice = f_data[1].split('_')
    data = {'pid': scan_id[0],
            'timepoint': int(scan_id[1][1:]),
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


if __name__ == '__main__':
    k = int(sys.argv[1])
    save_path = K_BIN_SAVE_PATH % k
    if os.path.isfile(save_path):
        raise FileExistsError(
            'Cross-validation with %d bins already exists. To overwrite, manually delete previous file' % k)

    # Get all patient ids (pids)
    pids = []
    for dp in DATA_PATHS:
        for fname in os.listdir(dp):
            if fname.endswith('.im'):
                im_info = get_file_info(fname, dp)
                pids.append(im_info['pid'])
    pids = list(set(pids))

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
    for i in range(len(bins)):
        if len(bins[i]) != len(set(bins[i])):
            raise ValueError('Duplicates in bin %d' % i)

    # save data to filepath
    io_utils.save_pik(bins, save_path)
