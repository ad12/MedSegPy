import unittest
from medsegpy.config import Config


class TestConfig(unittest.TestCase):
    def test_merge_from_list(self):
        cfg = Config("")

        with self.assertRaises(KeyError):
            cfg.merge_from_list(["lr_scheduler_name", "stepdecay"])

        with self.assertRaises(ValueError):
            cfg.merge_from_list(["LR_SCHEDULER_NAME", "stepdecay", "N_EPOCHS"])

        with self.assertRaises(ValueError):
            cfg.merge_from_list(["CP_SAVE_TAG", "deeplabv3_2d"])

        cfg.merge_from_list(["LR_SCHEDULER_NAME", "StepDecay"])
        assert cfg.LR_SCHEDULER_NAME == "StepDecay"

        # Ignore deprecated values.
        cfg.merge_from_list(["TRAIN_PATH", "./"])
        # Throw error on renamed values.
        with self.assertRaises(KeyError):
            cfg.merge_from_list(["CP_SAVE_PATH", "./"])


if __name__ == "__main__":
    unittest.main()
