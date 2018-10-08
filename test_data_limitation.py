import collections
import random
import unittest

import parse_pids
import utils
from config import DeeplabV3Config, SegnetConfig, UNetConfig
from im_generator import calc_generator_info


class DataLimitationTest(unittest.TestCase):
    def setUp(self):
        print("Testing: ", self._testMethodName)

    def test_max_pids(self):
        """Test that files are the same for pids=None and pids are size 60"""

        configs = [DeeplabV3Config(), SegnetConfig(), UNetConfig()]

        for config in configs:
            # config when PIDS are none
            gen_train_files, _ = calc_generator_info(config.TRAIN_PATH, config.TRAIN_BATCH_SIZE,
                                                     learn_files=[],
                                                     pids=config.PIDS,
                                                     augment_data=config.AUGMENT_DATA)

            # config when PIDS is list of all subjects
            pids = utils.load_pik(parse_pids.PID_TXT_PATH)
            num_pids = len(pids)
            pids_sampled = random.sample(pids, num_pids)
            config.PIDS = pids_sampled
            pids_train_files, _ = calc_generator_info(config.TRAIN_PATH, config.TRAIN_BATCH_SIZE,
                                                      learn_files=[],
                                                      pids=config.PIDS,
                                                      augment_data=config.AUGMENT_DATA)
            assert (len(pids_train_files) == len(gen_train_files))
            assert (collections.Counter(gen_train_files) == collections.Counter(pids_train_files))


if __name__ == '__main__':
    unittest.main()
