import unittest
from fvcore.common.file_io import PathManager
from medsegpy.config import Config, DeeplabV3Config

# Required to set up paths.
from tests import utils


class TestConfig(unittest.TestCase):
    def test_merge_from_list(self):
        cfg = Config("")

        with self.assertRaises(KeyError):
            cfg.merge_from_list(["lr_scheduler_name", "stepdecay"])

        with self.assertRaises(ValueError):
            cfg.merge_from_list(["LR_SCHEDULER_NAME", "stepdecay", "N_EPOCHS"])

        with self.assertRaises(ValueError):
            cfg.merge_from_list(["MODEL_NAME", "deeplabv3_2d"])

        cfg.merge_from_list(["LR_SCHEDULER_NAME", "StepDecay"])
        assert cfg.LR_SCHEDULER_NAME == "StepDecay"

        # Ignore deprecated values.
        cfg.merge_from_list(["TRAIN_PATH", "./"])
        # Throw error on renamed values.
        with self.assertRaises(KeyError):
            cfg.merge_from_list(["CP_SAVE_PATH", "./"])

    def test_merge_from_file(self):
        cfg1 = DeeplabV3Config()
        cfg1.merge_from_file(
            PathManager.get_local_path("mock_data://configs/deeplabv3_2d.ini")
        )

        cfg2 = DeeplabV3Config()
        cfg2.merge_from_file(
            PathManager.get_local_path("mock_data://configs/deeplabv3_2d.yaml")
        )

        members = [
            attr for attr in dir(cfg1)
            if not callable(getattr(cfg1, attr))
               and not attr.startswith("__")
               and not (hasattr(type(cfg1), attr) and isinstance(getattr(type(cfg1), attr), property))
        ]
        cfg1_dict = {k: getattr(cfg1, k) for k in members}

        members = [
            attr for attr in dir(cfg2)
            if not callable(getattr(cfg2, attr))
               and not attr.startswith("__")
               and not (hasattr(type(cfg2), attr) and isinstance(getattr(type(cfg2), attr), property))
        ]
        cfg2_dict = {k: getattr(cfg2, k) for k in members}
        for k in (cfg1_dict.keys() | cfg2_dict.keys()):
            assert cfg1_dict[k] == cfg2_dict[k]


if __name__ == "__main__":
    unittest.main()
