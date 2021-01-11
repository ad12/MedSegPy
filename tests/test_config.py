import unittest

from fvcore.common.file_io import PathManager

from medsegpy.config import Config, DeeplabV3Config

# Import utils to register handlers
from . import utils  # noqa


class TestConfig(unittest.TestCase):
    def _compare_cfgs(self, cfg1: Config, cfg2: Config, skip_keys=()):
        members = [
            attr
            for attr in dir(cfg1)
            if not callable(getattr(cfg1, attr))
            and not attr.startswith("__")
            and not (hasattr(type(cfg1), attr) and isinstance(getattr(type(cfg1), attr), property))
        ]
        cfg1_dict = {k: getattr(cfg1, k) for k in members}

        members = [
            attr
            for attr in dir(cfg2)
            if not callable(getattr(cfg2, attr))
            and not attr.startswith("__")
            and not (hasattr(type(cfg2), attr) and isinstance(getattr(type(cfg2), attr), property))
        ]
        cfg2_dict = {k: getattr(cfg2, k) for k in members}
        for k in cfg1_dict.keys() | cfg2_dict.keys():
            if k in skip_keys:
                continue
            assert cfg1_dict[k] == cfg2_dict[k]

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

    def test_types_merge_from_file(self):
        cfg1 = DeeplabV3Config()
        cfg1.merge_from_file(PathManager.get_local_path("mock_data://configs/deeplabv3_2d.ini"))

        cfg2 = DeeplabV3Config()
        cfg2.merge_from_file(PathManager.get_local_path("mock_data://configs/deeplabv3_2d.yaml"))

        self._compare_cfgs(cfg1, cfg2)

    def test_merge_from_base_file(self):
        base_cfg = DeeplabV3Config()
        child_cfg = DeeplabV3Config()
        grandchild_cfg = DeeplabV3Config()

        base_cfg.merge_from_file(PathManager.get_local_path("mock_data://configs/deeplabv3_2d.ini"))
        child_cfg.merge_from_file(
            PathManager.get_local_path("mock_data://configs/child_configs/child1.yaml")
        )
        grandchild_cfg.merge_from_file(
            PathManager.get_local_path("mock_data://configs/child_configs/grandchild1.yaml")
        )

        # All values should be the same except N_EPOCHS.
        epochs_key = "N_EPOCHS"
        skip_keys = (epochs_key,)
        self._compare_cfgs(base_cfg, child_cfg, skip_keys)
        assert getattr(base_cfg, epochs_key) == 1
        assert getattr(child_cfg, epochs_key) == 12

        dropout_key = "DROPOUT_RATE"
        skip_keys = (epochs_key, dropout_key)
        self._compare_cfgs(base_cfg, grandchild_cfg, skip_keys)
        self._compare_cfgs(child_cfg, grandchild_cfg, skip_keys)
        assert getattr(grandchild_cfg, epochs_key) == 100
        assert getattr(base_cfg, dropout_key) == 0.1
        assert getattr(child_cfg, dropout_key) == 0.1
        assert getattr(grandchild_cfg, dropout_key) == 0.3


if __name__ == "__main__":
    unittest.main()
