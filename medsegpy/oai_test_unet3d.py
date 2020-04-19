import logging
import os
import time
from time import strftime

import h5py
import matplotlib
import numpy as np
import scipy.io as sio

from medsegpy.utils import im_utils, io_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from medsegpy import oai_test
from medsegpy.utils import dice_score_coefficient, volumetric_overlap_error
from medsegpy.scan_metadata import ScanMetadata

logger = logging.getLogger(__name__)

UNET_3D_TEST_PATH = '/bmrNAS/people/arjun/msk_seg_networks/volume_limited/unet_3d_asc/test'
UNET_3D_TEST_RESULT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/volume_limited/unet_3d_asc/test_results'
TEST_BATCH_SIZE = 64
NUM_SLICES = 64

TEST_SET_METADATA_PIK = '/bmrNAS/people/arjun/msk_seg_networks/oai_data_test/oai_test_data.dat'
TEST_SET_MD = io_utils.load_pik(TEST_SET_METADATA_PIK)

save_file = 1


def load_pid_data(dirpath=UNET_3D_TEST_PATH):
    files = os.listdir(dirpath)
    scans = dict()

    for file in files:
        if not os.path.isfile(os.path.join(dirpath, file)):
            continue

        str_split = file.rsplit('_', 1)
        scan_id = str_split[0]

        if scan_id in scans.keys():
            continue

        im1 = io_utils.load_h5(os.path.join(dirpath, '%s_1.im' % scan_id))['data'][:]
        im2 = io_utils.load_h5(os.path.join(dirpath, '%s_2.im' % scan_id))['data'][:]

        seg1 = io_utils.load_h5(os.path.join(dirpath, '%s_1.seg' % scan_id))['data'][:]
        seg2 = io_utils.load_h5(os.path.join(dirpath, '%s_2.seg' % scan_id))['data'][:]
        seg_ind = 0
        if scan_id == '9908796_V01':
            seg_ind = 4
        seg1 = seg1[..., seg_ind]
        seg2 = seg2[..., seg_ind]

        pred1 = io_utils.load_h5(os.path.join(dirpath, '%s_1.pred' % scan_id))['pred'][:]
        pred2 = io_utils.load_h5(os.path.join(dirpath, '%s_2.pred' % scan_id))['pred'][:]

        im = np.concatenate((im1, im2), axis=-1)
        seg = np.concatenate((seg1, seg2), axis=-1)
        pred = np.concatenate((pred1, pred2), axis=-1)

        # make number of slices first dimension
        im = np.transpose(im, [2, 0, 1])
        seg = np.transpose(seg, [2, 0, 1])
        pred = np.transpose(pred, [2, 0, 1])

        im = im[..., np.newaxis]
        seg = seg[..., np.newaxis]
        pred = pred[..., np.newaxis]

        scans[scan_id] = (im, seg, pred)

    return scans


def test_model():
    """
    Test model
    :param config: a Config object
    :param save_file: save data (default = 0)
    """

    test_set_md = dict()
    for k in TEST_SET_MD.keys():
        test_set_md[k] = ScanMetadata(TEST_SET_MD[k])

    # Load config data
    test_path = UNET_3D_TEST_PATH
    test_result_path = io_utils.check_dir(UNET_3D_TEST_RESULT_PATH)
    test_batch_size = TEST_BATCH_SIZE

    scans_data = load_pid_data(test_path)
    scans_keys = list(scans_data.keys())
    scans_keys.sort()

    img_cnt = 0

    dice_losses = np.array([])
    voes = np.array([])
    cv_values = np.array([])

    start = time.time()
    skipped_count = 0

    # Read the files that will be segmented
    # logger.info('INFO: Test size: %d, batch size: %d, # subjects: %d' % (len(test_files), test_batch_size, len(scans_data)))
    logger.info('Save path: %s' % (test_result_path))
    logger.info('Test path: %s' % test_path)

    pids_str = ''

    x_interp = []
    y_interp = []
    x_total = []
    y_total = []

    # # Iterature through the files to be segmented
    for fname in scans_keys:
        x_test, y_test, y_pred = scans_data[fname]
        num_slices = NUM_SLICES
        # Perform the actual segmentation using pre-loaded model
        # Threshold at 0.5

        labels = (y_pred > 0.5).astype(np.float32)

        # Calculate real time dice coeff for analysis
        dl = dice_score_coefficient(y_test, labels)
        voe = volumetric_overlap_error(y_test, labels)
        cv = io_utils.calc_cv(y_test, labels)

        dice_losses = np.append(dice_losses, dl)
        voes = np.append(voes, voe)
        cv_values = np.append(cv_values, cv)

        logger.info_str = 'DSC, VOE, CV for image #%d (name = %s, %d slices) = %0.3f, %0.3f, %0.3f' % (
            img_cnt, fname, num_slices, dl, voe, cv)
        pids_str = pids_str + logger.info_str + '\n'
        logger.info(logger.info_str)

        slice_dir = test_set_md[fname].slice_dir

        # interpolate region of interest
        xs, ys, xt, yt = oai_test.interp_slice(y_test, labels, orientation=slice_dir)
        x_interp.append(xs)
        y_interp.append(ys)
        x_total.append(xt)
        y_total.append(yt)

        if save_file == 1:
            save_name = '%s/%s_recon.pred' % (test_result_path, fname)
            with h5py.File(save_name, 'w') as h5f:
                h5f.create_dataset('recon', data=labels)

            # in case of 2.5D, we want to only select center slice
            x_write = x_test[..., x_test.shape[-1] // 2]

            # Save mask overlap
            ovlps = im_utils.write_ovlp_masks(os.path.join(test_result_path, 'ovlp', fname), y_test, labels)
            im_utils.write_mask(os.path.join(test_result_path, 'gt', fname), y_test)
            im_utils.write_prob_map(os.path.join(test_result_path, 'prob_map', fname), labels)
            im_utils.write_im_overlay(os.path.join(test_result_path, 'im_ovlp', fname), x_write, ovlps)

        img_cnt += 1

    end = time.time()

    stats_string = oai_test.get_stats_string(dice_losses, voes, cv_values, skipped_count, end - start)
    # Log some summary statistics
    logger.info('--' * 20)
    logger.info(stats_string)
    logger.info('--' * 20)

    # Write details to test file
    with open(os.path.join(test_result_path, 'results.txt'), 'w+') as f:
        f.write('Results generated on %s \n' % strftime('%X %x %Z'))
        f.write('--' * 20)
        f.write('\n')
        f.write(pids_str)
        f.write('--' * 20)
        f.write('\n')
        f.write(stats_string)

    # Save metrics in dat format using pickle
    results_dat = os.path.join(test_result_path, 'metrics.dat')
    metrics = {'dsc': dice_losses,
               'voe': voes,
               'cvs': cv_values}
    io_utils.save_pik(metrics, results_dat)

    x_interp = np.asarray(x_interp)
    y_interp = np.asarray(y_interp)
    x_total = np.asarray(x_total)
    y_total = np.asarray(y_total)

    sio.savemat(os.path.join(test_result_path, 'total_interp_data.mat'), {'xs': x_interp,
                                                                          'ys': y_interp,
                                                                          'xt': x_total,
                                                                          'yt': y_total})

    x_interp_mean = np.mean(x_interp, 0)
    y_interp_mean = np.mean(y_interp, 0)
    y_interp_sem = np.std(y_interp, 0) / np.sqrt(y_interp.shape[0])

    plt.clf()
    plt.plot(x_interp_mean, y_interp_mean, 'b-')
    plt.fill_between(x_interp_mean, y_interp_mean - y_interp_sem, y_interp_mean + y_interp_sem, alpha=0.35)
    plt.xlabel('FOV (%)')
    plt.ylabel('Dice')
    plt.savefig(os.path.join(test_result_path, 'interp_slices.png'))


if __name__ == '__main__':
    test_model()
    # A1 = utils.load_h5('data_visualization/9905863_V00_1.im')['data'][:]
    # A2 = utils.load_h5('data_visualization/9905863_V00_1.seg')['data'][:]
    # A3 = utils.load_h5('data_visualization/9905863_V00_1.pred')['pred'][:]
    #
    # B1 = utils.load_h5('data_visualization/9905863_V00_2.im')['data'][:]
    # B2 = utils.load_h5('data_visualization/9905863_V00_2.seg')['data'][:]
    # B3 = utils.load_h5('data_visualization/9905863_V00_2.pred')['pred'][:]
    #
    # logger.info('hello')
