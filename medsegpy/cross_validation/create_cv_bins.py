"""
Create K bins for K-fold cross-validation

Data is stored in Pickle format
"""

import argparse
from .cv_util import CrossValidationFileGenerator, DATASET_NAME

DATA_PATHS = [defaults.TRAIN_PATH, defaults.VALID_PATH, defaults.TEST_PATH]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate bins for k-fold cross-validation'
    )

    parser.add_argument('-k', '--k_bins', metavar='K', nargs=1, type=int,
                        help='Number of bins to create for k-fold cross-validation')

    parser.add_argument('--filepaths', metavar='F', nargs='?',
                        default=DATA_PATHS,
                        help='Filepaths to data')

    parser.add_argument('--tag', metavar='T', nargs='?',
                        default=DATASET_NAME,
                        help='Tag uniquely identifying dataset used')

    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite existing file with same name',
                        dest='overwrite')

    args = parser.parse_args()

    k_bins = args.k_bins[0]
    filepaths = args.filepaths
    tag = args.tag
    overwrite = args.overwrite

    cv_generator = CrossValidationFileGenerator(k_bins, filepaths, tag, overwrite=overwrite)
