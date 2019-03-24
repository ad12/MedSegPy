"""
Compare interpolating masks on downsampled scans
"""
import os
from copy import deepcopy
import numpy as np

from generators import im_gens
import config
from utils import dl_utils, io_utils
from models.models import get_model
from utils.metric_utils import MetricWrapper

EXP_PATH = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/deeplabv3_2d/2018-11-30-05-49-49/fine_tune/'
HR_TEST_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_3d/test'

WEIGHTS_PATH = dl_utils.get_weights(EXP_PATH)
CONFIG_PATH = os.path.join(EXP_PATH, 'config.ini')

VOXEL_SPACING = (0.3125, 0.3125, 0.7)  # mm
INTERPOLATION_RESULTS_PATH = io_utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/interpolation')
INTERPOLATION_EXP = ''

def load_config():
    # get config
    cp_save_tag = config.get_cp_save_tag(CONFIG_PATH)
    return config.get_config(cp_save_tag, create_dirs=False)


def get_downsampled_masks(c: config.Config):
    # Load model
    model = get_model(c)
    model.load_weights(WEIGHTS_PATH)

    test_batch_size = c.TEST_BATCH_SIZE
    test_gen = im_gens.get_generator(c)

    y_pred_dict = {}
    # # Iterature through the files to be segmented
    for x_test, y_test, fname, num_slices in test_gen.img_generator_test():
        recon = model.predict(x_test, batch_size=test_batch_size)
        if c.INCLUDE_BACKGROUND:
            y_test = y_test[..., 1]
            recon = recon[..., 1]
            y_test = y_test[..., np.newaxis]
            recon = recon[..., np.newaxis]

        y_pred_dict[fname] = recon

    return y_pred_dict


def inference_hr():
    c = load_config()
    c_hr = deepcopy(c)
    c_hr.TEST_PATH = HR_TEST_PATH

    # Load model
    model = get_model(c_hr)
    model.load_weights(WEIGHTS_PATH)

    test_batch_size = c.TEST_BATCH_SIZE
    test_gen = im_gens.get_generator(c_hr)
    mw = MetricWrapper()

    y_pred_dict = {}
    # # Iterature through the files to be segmented
    for x_test, y_test, fname, num_slices in test_gen.img_generator_test():
        recon = model.predict(x_test, batch_size=test_batch_size)
        if c.INCLUDE_BACKGROUND:
            y_test = y_test[..., 1]
            recon = recon[..., 1]
            y_test = y_test[..., np.newaxis]
            recon = recon[..., np.newaxis]

        labels = (recon > 0.5).astype(np.float32)

        mw.compute_metrics(np.transpose(np.squeeze(y_test), axes=[1, 2, 0]),
                           np.transpose(np.squeeze(labels), axes=[1, 2, 0]),
                           voxel_spacing=VOXEL_SPACING)

    mw.summary()
    io_utils.save_pik(mw.metrics, os.path.join(INTERPOLATION_RESULTS_PATH, '0-inference-direct.dat'))



def interp_lr_hr():
    # Get config and make new config for hr data
    c = load_config()
    c_hr = deepcopy(c)
    c_hr.TEST_PATH = HR_TEST_PATH

    # get probability masks from inference on downsampled masks
    y_pred_dict = get_downsampled_masks(c)

    test_gen = im_gens.get_generator(c_hr)
    mw = MetricWrapper()

    # # Iterature through the files to be segmented
    for x_test, y_test, fname, num_slices in test_gen.img_generator_test():
        # interpolate y_pred
        y_pred = y_pred_dict[fname]

        # add to metrics
        mw.compute_metrics(np.transpose(np.squeeze(y_test), axes=[1, 2, 0]),
                           np.transpose(np.squeeze(y_pred), axes=[1, 2, 0]),
                           voxel_spacing=VOXEL_SPACING)

    mw.summary()
    io_utils.save_pik(mw.metrics, os.path.join(INTERPOLATION_RESULTS_PATH, INTERPOLATION_EXP))


if __name__ == '__main__':
    inference_hr()
