"""
Use 3d models to run inference on full dataset, but ground truth should be downsampled masks
"""
import logging
import os
import sys
import numpy as np

sys.path.append('../')
from medsegpy.data import im_gens
from medsegpy import config
from medsegpy.utils import dl_utils
from medsegpy.modeling import get_model
from medsegpy.utils import MetricWrapper
from medsegpy.oai_test import interp_slice, get_stats_string, TEST_SET_MD
from medsegpy.scan_metadata import ScanMetadata
import h5py
from medsegpy.utils import utils, im_utils, io_utils
import time
import scipy.io as sio
import argparse
import keras.backend as K

logger = logging.getLogger(__name__)

DOWNSAMPLED_TEST_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'

DEFAULT_VOXEL_SPACING = (0.3125, 0.3125, 1.4)  # mm

GPU = '1'


class InterpolationTest():
    def __init__(self, dirpath, voxel_spacing, **kwargs):
        self.config_dict = dict()
        self.save_h5_data = False
        self.weights_path = None
        self.zoom_spline_order = 1

        if 'config_dict' in kwargs:
            config_dict = kwargs.get('config_dict')
            if type(config_dict) is not dict:
                raise TypeError('config_dict must be a dict')

            self.config_dict.update(config_dict)

        if 'save_h5_data' in kwargs:
            save_h5_data = kwargs.get('save_h5_data')
            if type(save_h5_data) is not bool:
                raise TypeError('save_h5_data must be bool')

            self.save_h5_data = save_h5_data

        if 'weights_path' in kwargs:
            weights_path = kwargs.get('weights_path')
            if type(weights_path) is not str:
                raise TypeError('weights_path must be a string to the weights path')
            self.weights_path = weights_path

        if 'TEST_RESULTS_FOLDER_NAME' in self.config_dict:
            self.config_dict['TEST_RESULTS_FOLDER_NAME'] = '%s-downsampled' % self.config_dict[
                'TEST_RESULTS_FOLDER_NAME']
        else:
            self.config_dict['TEST_RESULTS_FOLDER_NAME'] = 'test_results-downsampled'

        self.dirpath = dirpath
        self.voxel_spacing = voxel_spacing

        test_set_md = dict()
        for k in TEST_SET_MD.keys():
            test_set_md[k] = ScanMetadata(TEST_SET_MD[k])
        self.test_set_md = test_set_md

        # start timer
        self.start_time = time.time()

        # initialize config from dirpath (should correspond to config for low resolution data)
        self.hr_config = self.__init_hr_config__(self.dirpath)

        # run analysis
        self.test_downsample_hr()

    def __init_hr_config__(self, dirpath: str):
        weights_path = self.weights_path
        if not weights_path:
            weights_path = dl_utils.get_weights(dirpath)

        logger.info('Weights selected: %s' % weights_path)

        config_filepath = os.path.join(dirpath, 'config.ini')
        logger.info('Config: %s' % config_filepath)
        c = self.load_config(config_filepath)

        c.load_config(config_filepath)
        c.TEST_WEIGHT_PATH = weights_path

        config_dict = self.config_dict
        if config_dict:
            for key in config_dict.keys():
                val = config_dict[key]
                c.set_attr(key, val)

        c.change_to_test()

        return c

    def test_downsample_hr(self):
        """
        Downsample hr inference by factor of 2 to get in same dimensions
        :return:
        """
        c_hr = self.hr_config
        test_result_path = c_hr.TEST_RESULT_PATH

        model = get_model(c_hr)
        model.load_weights(c_hr.TEST_WEIGHT_PATH)
        test_gen = im_gens.get_generator(c_hr)
        mw = MetricWrapper()

        pids_str = ''
        x_interp = []
        y_interp = []
        x_total = []
        y_total = []

        img_cnt = 0

        # Iterate through the files to be segmented
        for x_test, y_test, recon, fname in test_gen.img_generator_test(model):
            x_test = np.transpose(np.squeeze(x_test[8:-8, ...]), [1,2,0])
            recon = np.transpose(np.squeeze(recon[8:-8, ...]), [1, 2, 0])
            y_test = np.transpose(np.squeeze(y_test[8:-8, ...]), [1, 2, 0])

            labels = (recon > 0.5)

            y_pred = np.array(recon)
            y_pred = (y_pred[..., 0::2] + y_pred[..., 1::2]) / 2

            # downsample labels using OR mask
            labels = np.logical_or(labels[..., 0::2], labels[..., 1::2])
            y_test = np.logical_or(y_test[..., 0::2], y_test[..., 1::2])
            labels = labels.astype(np.float32)
            y_test = y_test.astype(np.float32)

            pids_str += self.analysis(x_test=x_test, y_test=y_test, recon=y_pred, labels=labels,
                                      mw=mw, voxel_spacing=self.voxel_spacing,
                                      img_cnt=img_cnt, fname=fname, test_set_md=self.test_set_md,
                                      x_interp=x_interp, y_interp=y_interp, x_total=x_total, y_total=y_total,
                                      save_file=1, save_h5_data=self.save_h5_data, test_result_path=test_result_path)

            img_cnt += 1
        model = None
        K.clear_session()

        stats_string = get_stats_string(mw, skipped_count=0, testing_time=(time.time() - self.start_time))

        # Print some summary statistics
        logger.info('--' * 20)
        logger.info(stats_string)
        logger.info('--' * 20)

        # Write details to test file
        with open(os.path.join(test_result_path, 'results.txt'), 'w+') as f:
            f.write('Results generated on %s\n' % time.strftime('%X %x %Z'))
            f.write('Weights Loaded: %s\n' % os.path.basename(self.hr_config.TEST_WEIGHT_PATH))
            f.write('Voxel Spacing: %s' % str(self.voxel_spacing))
            f.write('--' * 20)
            f.write('\n')
            f.write(pids_str)
            f.write('--' * 20)
            f.write('\n')
            f.write(stats_string)

        # Save metrics in dat format using pickle
        results_dat = os.path.join(test_result_path, 'metrics.dat')
        io_utils.save_pik(mw.metrics, results_dat)

        x_interp = np.asarray(x_interp)
        y_interp = np.asarray(y_interp)
        x_total = np.asarray(x_total)
        y_total = np.asarray(y_total)

        sio.savemat(os.path.join(test_result_path, 'total_interp_data.mat'), {'xs': x_interp,
                                                                              'ys': y_interp,
                                                                              'xt': x_total,
                                                                              'yt': y_total})

        mw.summary()

    def load_config(self, config_filepath):
        # get config
        cp_save_tag = config.get_cp_save_tag(config_filepath)
        return config.get_config(cp_save_tag, create_dirs=False)

    def analysis(self, x_test, y_test, recon, labels, mw: MetricWrapper, voxel_spacing,
                 img_cnt, fname, test_set_md,
                 x_interp, y_interp, x_total, y_total,
                 save_file, save_h5_data, test_result_path):
        num_slices = y_test.shape[-1]
        mw.compute_metrics(np.transpose(np.squeeze(y_test), axes=[1, 2, 0]),
                           np.transpose(np.squeeze(labels), axes=[1, 2, 0]),
                           voxel_spacing=voxel_spacing)

        print_str = 'Scan #%03d (name = %s, %d slices) = DSC: %0.3f, VOE: %0.3f, CV: %0.3f, ASSD (mm): %0.3f' % (
            img_cnt, fname,
            num_slices,
            mw.metrics['dsc'][-1],
            mw.metrics['voe'][-1],
            mw.metrics['cv'][-1],
            mw.metrics['assd'][-1])
        logger.info(print_str)

        if fname in test_set_md.keys():
            slice_dir = test_set_md[fname].slice_dir

            # interpolate region of interest
            xs, ys, xt, yt = interp_slice(y_test, labels, orientation=slice_dir)
            x_interp.append(xs)
            y_interp.append(ys)
            x_total.append(xt)
            y_total.append(yt)

        if save_file == 1:
            if save_h5_data:
                save_name = '%s/%s.pred' % (test_result_path, fname)
                with h5py.File(save_name, 'w') as h5f:
                    h5f.create_dataset('recon', data=recon)
                    h5f.create_dataset('gt', data=y_test)
            
            x_write = x_test

            y_test = np.transpose(y_test, [2, 0, 1])
            labels = np.transpose(labels, [2, 0, 1])
            recon = np.transpose(recon, [2, 0, 1])
            x_write = np.transpose(x_write, [2, 0, 1])

            # Save mask overlap
            ovlps = im_utils.write_ovlp_masks(os.path.join(test_result_path, 'ovlp', fname), y_test, labels)
            im_utils.write_mask(os.path.join(test_result_path, 'gt', fname), y_test)
            im_utils.write_prob_map(os.path.join(test_result_path, 'prob_map', fname), recon)
            #im_utils.write_im_overlay(os.path.join(test_result_path, 'im_ovlp', fname), x_write, ovlps)
            im_utils.write_mask(os.path.join(test_result_path, 'labels', fname), labels)
            # im_utils.write_sep_im_overlay(os.path.join(test_result_path, 'im_ovlp_sep', fname), x_write,
            #                               np.squeeze(y_test), np.squeeze(labels))

        return print_str + '\n'


def add_testing_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dirpath', metavar='dp', type=str, nargs=1,
                        required=True,
                        help='path to config to test')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use. default=0')
    parser.add_argument('--cpu', action='store_const', default=False, const=True,
                        help='use cpu. will overridie `-g` gpu flag')

    parser.add_argument('--batch_size', default=72, type=int, nargs='?')
    parser.add_argument('--save_h5_data', action='store_const', const=True, default=False,
                        help='save h5 data')
    parser.add_argument('--tag', default=None, nargs='?', type=str,
                        help='change tag for inference')
    parser.add_argument('--img_size', default=None, nargs='?')
    parser.add_argument('--voxel_spacing', default=str(DEFAULT_VOXEL_SPACING), nargs='?', type=str,
                        help='set voxel spacing (y, x, z)')


def create_config_dict(vargin):
    config_dict = {'TEST_BATCH_SIZE': vargin['batch_size']}
    if vargin['tag']:
        config_dict['TAG'] = vargin['tag']
        config_dict['TEST_RESULTS_FOLDER_NAME'] = 'test_results_%s' % vargin['tag']

    return config_dict


if __name__ == '__main__':
    raise DeprecationWarning("This file is deprecated. Use nn_test.")
    base_parser = argparse.ArgumentParser(description='Run inference (testing)')
    add_testing_arguments(base_parser)

    # Parse input arguments
    args = base_parser.parse_args()
    vargin = vars(args)

    config_filepath = vargin['dirpath'][0]
    if not os.path.isdir(config_filepath):
        raise NotADirectoryError('Directory %s does not exist.' % config_filepath)

    gpu = args.gpu
    cpu = args.cpu

    if not cpu:
        logger.info('Using GPU %s' % gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    else:

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    voxel_spacing = utils.convert_data_type(vargin['voxel_spacing'], tuple)

    m = InterpolationTest(dirpath=config_filepath, voxel_spacing=voxel_spacing,
                          config_dict=create_config_dict(vargin), save_h5_data=vargin['save_h5_data'])
