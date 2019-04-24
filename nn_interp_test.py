"""
Compare interpolating masks on downsampled scans
"""
import os,sys
from copy import deepcopy
import numpy as np

sys.path.append('../')
from generators import im_gens
import config
from utils import dl_utils, io_utils
from models.models import get_model
from utils.metric_utils import MetricWrapper
from config import Config
from scipy import ndimage
from oai_test import interp_slice, get_stats_string, TEST_SET_MD
from scan_metadata import ScanMetadata
import h5py
from utils import im_utils, utils
import time
import scipy.io as sio
import argparse
import keras.backend as K
from scipy.misc import imresize
from keras.utils import plot_model

# EXP_PATH = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/deeplabv3_2d/2018-11-30-05-49-49/fine_tune/'
HR_TEST_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_3d/test'

#
# WEIGHTS_PATH = dl_utils.get_weights(EXP_PATH)
# CONFIG_PATH = os.path.join(EXP_PATH, 'config.ini')
#
DEFAULT_VOXEL_SPACING = (0.3125, 0.3125, 0.7)  # mm
# INTERPOLATION_RESULTS_PATH = io_utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/interpolation')
# INTERPOLATION_EXP = ''

GPU = '1'


class InterpolationTest():
    def __init__(self, dirpath, voxel_spacing, **kwargs):
        self.config_dict = dict()
        self.save_h5_data = False
        self.weights_path = None
        self.hr_test_path = HR_TEST_PATH
        self.zoom_spline_order = 3

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
            self.config_dict['TEST_RESULTS_FOLDER_NAME'] = '%s-interpolate' % self.config_dict['TEST_RESULTS_FOLDER_NAME']
        else:
            self.config_dict['TEST_RESULTS_FOLDER_NAME'] = 'test_results-interpolated'

        self.dirpath = dirpath
        self.voxel_spacing = voxel_spacing

        test_set_md = dict()
        for k in TEST_SET_MD.keys():
            test_set_md[k] = ScanMetadata(TEST_SET_MD[k])
        self.test_set_md = test_set_md

        # start timer
        self.start_time = time.time()

        # initialize config from dirpath (should correspond to config for low resolution data)
        self.lr_config = self.__init_lr_config__(self.dirpath)

        # use config to generate pixel-wise probability maps for low-resolution volumes
        self.lr_prob_maps = self.get_lr_prob_maps(self.lr_config)

        # run analysis
        self.interp_lr_hr()

    def __init_lr_config__(self, dirpath: str):
        weights_path = self.weights_path
        if not weights_path:
            weights_path = dl_utils.get_weights(dirpath)

        print('Weights selected: %s' % weights_path)

        config_filepath = os.path.join(dirpath, 'config.ini')
        print('Config: %s' % config_filepath)
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

    def get_lr_prob_maps(self, c: config.Config):
        # Load model
        model = get_model(c)
        plot_model(model, os.path.join(c.TEST_RESULT_PATH, 'model.png'), show_shapes=True)
        model.load_weights(c.TEST_WEIGHT_PATH)

        test_gen = im_gens.get_generator(c)

        y_pred_dict = {}
        print(c.TEST_WEIGHT_PATH)
        # Iterate through the files to be segmented
        for x_test, y_test, recon, fname in test_gen.img_generator_test(model):
            if c.INCLUDE_BACKGROUND:
                recon = recon[..., 1]
                recon = recon[..., np.newaxis]

            recon = np.transpose(np.squeeze(recon), [1, 2, 0])
            y_pred_dict[fname] = recon
        
        model = None
        K.clear_session()
        return y_pred_dict

    def or_mask(self, y_test):
        return ((y_test[..., 0:y_test.shape[-1]:2] + y_test[..., 1:y_test.shape[-1]:2]).astype(np.bool)).astype(np.float32)

    def interp_lr_hr(self):
        """
        Interpolate low resolution data to high resolution (self.interp_dimensions) and run inference
        :return:
        """
        test_result_path = self.lr_config.TEST_RESULT_PATH

        # Get config and make new config for hr data
        c_hr = deepcopy(self.lr_config)
        c_hr.TEST_PATH = self.hr_test_path

        # get probability masks from inference on downsampled masks
        y_pred_prob_maps = self.lr_prob_maps

        test_gen = im_gens.get_generator(c_hr)
        mw = MetricWrapper()

        pids_str = ''
        x_interp = []
        y_interp = []
        x_total = []
        y_total = []

        img_cnt = 0

        # Iterate through the files to be segmented
        for x_test, y_test, _, fname in test_gen.img_generator_test():
            y_test = np.transpose(np.squeeze(y_test), [1, 2, 0])

            y_test = y_test[..., 8:-8]
            #y_test = self.or_mask(y_test)

            # interpolate y_pred using ndimage.zoom function
            y_pred = y_pred_prob_maps[fname]
            # y_pred_orig = y_pred

            # #import pdb; pdb.set_trace()
            # zoom_factor = np.asarray(y_test.shape) / np.asarray(y_pred.shape)
            # assert (zoom_factor >= 1).all, "zoom_factor is %s. All values should be >= 1" % str(tuple(zoom_factor))
            #
            # y_pred = ndimage.zoom(y_pred, zoom_factor, order=self.zoom_spline_order)
            # assert y_pred.shape == y_test.shape, "Shape mismatch: y_pred: %s. y_test: %s" % (str(y_pred.shape),
            #                                                                                  str(y_test.shape))
            # y_pred = np.clip(y_pred, 0, 1)
            # assert (y_pred >= 0).all() and (y_pred <= 1).all(), "Error with interpolation - all values must be between [0,1]"

            y_pred_new = np.zeros([y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]*2])
            for i in range(y_pred.shape[2]):
                y_pred[..., 2*i] = y_pred[..., i]
                y_pred[..., 2*(i+1)] = y_pred[..., i]
            y_pred = y_pred_new

            labels = (y_pred > 0.5).astype(np.float32)

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
        print('--' * 20)
        print(stats_string)
        print('--' * 20)

        # Write details to test file
        with open(os.path.join(test_result_path, 'results.txt'), 'w+') as f:
            f.write('Results generated on %s\n' % time.strftime('%X %x %Z'))
            f.write('Weights Loaded: %s\n' % os.path.basename(self.lr_config.TEST_WEIGHT_PATH))
            f.write('High-Resolution test path: %s' % c_hr.TEST_PATH)
            f.write('Interpolation Method: ndimage.zoom (order %d)' % self.zoom_spline_order)
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
        print(print_str)

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

            # in case of 2.5D, we want to only select center slice
            x_write = x_test[..., x_test.shape[-1] // 2]

            y_test = np.transpose(y_test, [2, 0, 1])
            labels = np.transpose(labels, [2, 0, 1])
            recon = np.transpose(recon, [2, 0, 1])
            x_write = np.transpose(x_write, [2, 0, 1])

            # Save mask overlap
            ovlps = im_utils.write_ovlp_masks(os.path.join(test_result_path, 'ovlp', fname), y_test, labels)
            im_utils.write_mask(os.path.join(test_result_path, 'gt', fname), y_test)
            im_utils.write_mask(os.path.join(test_result_path, 'labels', fname), labels)
            im_utils.write_prob_map(os.path.join(test_result_path, 'prob_map', fname), recon)
           # im_utils.write_im_overlay(os.path.join(test_result_path, 'im_ovlp', fname), x_write, ovlps)
            # im_utils.write_sep_im_overlay(os.path.join(test_result_path, 'im_ovlp_sep', fname), x_write,
            #                               np.squeeze(y_test), np.squeeze(labels))

        return print_str


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
        print('Using GPU %s' % gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    else:

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    voxel_spacing = utils.convert_data_type(vargin['voxel_spacing'], tuple)

    m = InterpolationTest(dirpath=config_filepath, voxel_spacing=voxel_spacing,
                          config_dict=create_config_dict(vargin), save_h5_data=vargin['save_h5_data'])
