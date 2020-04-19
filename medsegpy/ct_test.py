import argparse
import ast
import logging
import os
import time

import h5py
import keras.backend as K
import numpy as np

os.environ["MSK_SEG_NETWORKS_PROJECT"] = "abCT"

from medsegpy import config as MCONFIG
from medsegpy.data.im_gens import CTGenerator
from medsegpy.modeling import get_model
from medsegpy.oai_test import get_stats_string
from medsegpy.utils import ct_utils, dl_utils, io_utils
from medsegpy.utils.logger import setup_logger
from medsegpy.utils.metric_utils import MetricsManager, SegMetric
from medsegpy.utils.im_utils import MultiClassOverlay

logger = logging.getLogger(__name__)


def add_testing_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dirpath', metavar='dp', type=str, nargs=1,
                        help='path to config to test')
    parser.add_argument("--num_gpus",
                        default=1,
                        type=int,
                        help="number of gpus to use. defaults to 1")
    parser.add_argument("--windows",
                        metavar="W", type=str, nargs="*",
                        help="windowing function")


def test_model(config, windows=None, save_file=0):
    """Test model

    Args:
        config
        windows
        save_file
    """
    test_gen = CTGenerator(config, windows)

    # Load config data
    test_result_path = config.TEST_RESULT_PATH

    K.set_image_data_format('channels_last')

    # Load weights into Deeplabv3 model
    model = get_model(config)
    model.load_weights(config.TEST_WEIGHT_PATH)

    img_cnt = 1

    start = time.time()

    test_gen.summary()
    logger.info('Save path: {}'.format(test_result_path))
    io_utils.check_dir(test_result_path)

    class_names = ["class {}".format(x) for x in config.TISSUES]

    metrics_manager = MetricsManager(
        metrics=[SegMetric.DSC],
        class_names=class_names,
    )

    # image writer
    mc_overlay = MultiClassOverlay(config.get_num_classes())

    # # Iterature through the files to be segmented
    for x_orig, x_test, y_test, recon, fname, seg_time in test_gen.img_generator_test(model):
        # Perform the actual segmentation using pre-loaded model
        # Threshold at 0.5
        if config.LOSS[1] == "sigmoid":
            # sigmoid activation function used
            labels = (recon > 0.5).astype(np.float32)
        else:
            # else, softmax used
            labels = np.zeros(recon.shape)
            l_argmax = np.argmax(recon, axis=-1)
            for c in range(labels.shape[-1]):
                labels[l_argmax == c, c] = 1
            labels = labels.astype(np.float32)

        # background is always excluded from analysis
        if config.INCLUDE_BACKGROUND:
            y_test = y_test[..., 1:]
            recon = recon[..., 1:]
            labels = labels[..., 1:]
            if y_test.ndim == 3:
                y_test = y_test[..., np.newaxis]
                recon = recon[..., np.newaxis]
                labels = labels[..., np.newaxis]

        num_slices = x_test.shape[0]

        summary = metrics_manager.analyze(
            fname,
            np.transpose(y_test, axes=[1, 2, 0, 3]),
            np.transpose(labels, axes=[1, 2, 0, 3]),
            voxel_spacing=(1., 1., 1.),
            runtime=seg_time
        )

        logger_info_str = "Scan #{:03d} (name = {}, {} slices, {:.2f}s) = {}".format(
            img_cnt, fname, num_slices, seg_time, summary
        )
        logger.info(logger_info_str)

        if save_file == 1:
            save_name = '%s/%s.pred' % (test_result_path, fname)
            with h5py.File(save_name, 'w') as h5f:
                h5f.create_dataset('recon', data=recon)
                h5f.create_dataset('gt', data=y_test)

            # in case of 2.5D, we want to only select center slice
            x_write = x_test[..., x_test.shape[-1] // 2]

            # Save mask overlap
            # TODO (arjundd): fix writing masks to files
            # x_write_o = np.transpose(x_write, (1, 2, 0))
            # recon_oo = np.transpose(recon_o, (1, 2, 0, 3))
            # mc_overlay.im_overlay(os.path.join(test_result_path, 'im_ovlp', fname), x_write_o, recon_oo)
            # ovlps = im_utils.write_ovlp_masks(os.path.join(test_result_path, 'ovlp', fname), y_test[...,0], labels[...,0])
            # im_utils.write_mask(os.path.join(test_result_path, 'gt', fname), y_test)
            # im_utils.write_mask(os.path.join(test_result_path, 'labels', fname), labels)
            # im_utils.write_prob_map(os.path.join(test_result_path, 'prob_map', fname), recon)
            # im_utils.write_im_overlay(os.path.join(test_result_path, 'im_ovlp', fname), x_write, ovlps)
            # im_utils.write_sep_im_overlay(os.path.join(test_result_path, 'im_ovlp_sep', fname), x_write,
            #                               np.squeeze(y_test), np.squeeze(labels))

        img_cnt += 1

    end = time.time()

    stats_string = get_stats_string(metrics_manager, end - start)
    logger.info('--' * 20)
    logger.info(stats_string)
    logger.info('--' * 20)

    test_results_summary_path = os.path.join(test_result_path, "results.txt")
    # Write details to test file
    with open(test_results_summary_path, "w+") as f:
        f.write("Results generated on {}\n".format(time.strftime("%X %x %Z")))
        f.write(
            "Weights Loaded: {}\n".format(
                os.path.basename(config.TEST_WEIGHT_PATH)
            )
        )
        f.write(stats_string)

    # Save metrics in dat format using pickle
    results_dat = os.path.join(test_result_path, 'metrics.dat')
    io_utils.save_pik(metrics_manager.data, results_dat)


def test_dir(dirpath, windows=None, config = None, vals_dict = None, best_weight_path = None):
    """
    Run testing experiment
    By default, save all data
    :param dirpath: path to directory storing model config
    :param config: a Config object
    :param vals_dict: vals_dict: a dictionary of config parameters to change (default = None)
                      e.g. {'INITIAL_LEARNING_RATE': 1e-6, 'USE_STEP_DECAY': True}
    :param best_weight_path: path to best weights (default = None)
                                if None, automatically search dirpath for the best weight path
    """
    # Create config, if not provided.
    config_filepath = os.path.join(dirpath, 'config.ini')
    if not config:
        config = MCONFIG.get_config(MCONFIG.get_cp_save_tag(config_filepath),
                                    create_dirs=False)

    # Get best weight path
    if best_weight_path is None:
        best_weight_path = dl_utils.get_weights(dirpath)

    config.load_config(config_filepath)
    config.TEST_WEIGHT_PATH = best_weight_path

    # Initialize logger.
    setup_logger(config.OUTPUT_DIR)
    logger.info('OUTPUT_DIR: %s' % config.OUTPUT_DIR)
    logger.info('Config: %s' % config_filepath)
    logger.info('Best weights: %s' % best_weight_path)

    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.change_to_test()

    test_model(
        config,
        windows=windows,
        save_file=1
    )

    K.clear_session()


if __name__ == '__main__':
    logger = logging.getLogger("medsegpy.ct_test.{}".format(__name__))
    base_parser = argparse.ArgumentParser(description='Run inference (testing)')
    add_testing_arguments(base_parser)

    # Parse input arguments
    args = base_parser.parse_args()
    vargin = vars(args)

    config_filepath = vargin["dirpath"][0]
    if not os.path.isdir(config_filepath):
        raise NotADirectoryError(
            "Directory {} does not exist".format(config_filepath)
        )

    num_gpus = args.num_gpus

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if num_gpus > 0:
        gpu_ids = dl_utils.get_available_gpus(num_gpus)
        gpu_ids_tf_str = ",".join([str(g_id) for g_id in gpu_ids])
        logger.info("Using {} GPU(s): {}".format(num_gpus, gpu_ids_tf_str))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_tf_str
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    test_dir(
        config_filepath,
        windows=ct_utils.parse_windows(args.windows) if args.windows else None
    )
