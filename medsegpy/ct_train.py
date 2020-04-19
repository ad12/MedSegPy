import copy
import logging

import keras.backend as K
import numpy as np

from medsegpy.ct_test import test_dir
from medsegpy.data.im_gens import CTGenerator
from medsegpy.nn_train import NNTrain
from medsegpy.utils import ct_utils

CLASS_WEIGHTS = np.asarray([100, 1])
SAVE_BEST_WEIGHTS = True
FREEZE_LAYERS = None

logger = logging.getLogger(__name__)


class CTTrain(NNTrain):
    __DESCRIPTION__ = 'Train networks for ct segmentation'

    _ARG_KEY_WINDOWS = "windows"

    @staticmethod
    def _add_classes_parser(parser):
        parser.add_argument("--classes", type=int, nargs="+",
                            required=False,
                            default=[],
                            help="tissue indices to segment")

    def _parse_classes(self):
        return self.args["classes"]

    def _add_default_args(self, parser):
        super()._add_default_args(parser)
        parser.add_argument('--{}'.format(self._ARG_KEY_WINDOWS),
                            metavar='W', type=str, nargs='*',
                            dest=self._ARG_KEY_WINDOWS,
                            help='(min, max) windows for clipping data')

    def _train_cross_validation(self, config):
        raise NotImplementedError(
            "Cross validation not supported for CT training")

    def _build_data_loaders(self, config):
        window_keys = self.get_arg(self._ARG_KEY_WINDOWS)
        windows = ct_utils.parse_windows(window_keys) if window_keys else None
        generator = CTGenerator(config, windows)

        return generator, generator

    def _test(self, config):
        logger.info("Beginning testing...")
        config = copy.deepcopy(config)  # will be modified below.
        dirpath = config.OUTPUT_DIR

        window_keys = self.get_arg(self._ARG_KEY_WINDOWS)
        windows = ct_utils.parse_windows(window_keys) if window_keys else None
        test_dir(
            dirpath,
            windows=windows
        )

        K.clear_session()


if __name__ == '__main__':
    logger = logging.getLogger("medsegpy.ct_train.{}".format(__name__))
    nn_train = CTTrain()
    nn_train.parse()
    nn_train.run()
