"""
Create K bins for K-fold cross-validation

Data is stored in Pickle format
"""

import argparse

from cv_util import CrossValidationFileGenerator

DATA_PATHS = ['/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug/',
              '/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid/',
              '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate bins for k-fold cross-validation')

    parser.add_argument('-k', '--k_bins', metavar='K', nargs=1,
                        help='Number of bins to create for k-fold cross-validation')

    parser.add_argument('--filepaths', metavar='F', nargs='?',
                        default=DATA_PATHS,
                        help='Filepaths to data')

    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite existing file with same name',
                        dest='overwrite')

    args = parser.parse_args()

    k_bins = args.k_bins
    filepaths = args.filepaths
    overwrite = args.overwrite

    cv_generator = CrossValidationFileGenerator(k_bins, filepaths, overwrite=overwrite)
