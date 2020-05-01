import unittest

from medsegpy.config import UNetConfig
from medsegpy.data import build_loader
from medsegpy.data.im_gens import GeneratorState, get_generator


class TestDefaultDataLoader(unittest.TestCase):
    def test_same_data(self):
        """Test that the data loaded by DefaultDataLoader and OAIGenerator
        is the same for OAI Data.
        """
        cfg = UNetConfig()
        cfg.TRAIN_DATASET = "oai_2d_train"
        cfg.VAL_DATASET = "oai_2d_val"
        cfg.TEST_DATASET = "oai_2d_test"
        cfg.SEED = 1
        cfg.CATEGORIES = [0]
        cfg.IMG_SIZE = (384, 384, 1)

        gen = get_generator(cfg)
        train_steps, val_steps = gen.num_steps()

        train_loader = build_loader(
            cfg,
            dataset_names=cfg.TRAIN_DATASET,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            drop_last=True,
            is_test=False,
            shuffle=True,
        )
        val_loader = build_loader(
            cfg,
            dataset_names=cfg.VAL_DATASET,
            batch_size=cfg.VALID_BATCH_SIZE,
            drop_last=True,
            is_test=False,
            shuffle=False,
        )
        test_loader = build_loader(
            cfg,
            dataset_names=cfg.TEST_DATASET,
            batch_size=cfg.TEST_BATCH_SIZE,
            drop_last=False,
            is_test=True,
            shuffle=False,
        )

        # Number of steps per epoch should be the same
        assert len(train_loader) == train_steps
        assert len(val_loader) == val_steps

        assert test_loader.num_scans() == gen.num_scans(GeneratorState.TESTING)


if __name__ == "__main__":
    unittest.main()
