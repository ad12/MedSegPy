import sys

sys.path.insert(0, '../')

import os
import argparse

import utils
import oai_test as tst


def find_best_test_dir(base_folder):
    subdirs = tst.get_valid_subdirs(base_folder, no_results=False)
    max_dsc_details = (0, '')

    for subdir in subdirs:
        base_results = os.path.join(subdir, 'test_results')
        results_files = tst.check_results_file(base_results)
        assert not ((results_files is None) or (len(results_files) == 0)), "Checking results file failed - %s" % subdir
        for results_file in results_files:
            mean = utils.parse_results_file(results_file)
            potential_data = (mean, results_file)
            print(potential_data)
            if mean > max_dsc_details[0]:
                max_dsc_details = potential_data
    print('\nMAX')
    print(max_dsc_details)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find best results')
    parser.add_argument('-d', nargs=1, type=str, help='base directory to start search')

    args = parser.parse_args()
    find_best_test_dir(args.d[0])
