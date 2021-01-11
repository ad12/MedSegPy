import unittest

from medsegpy.config import DeeplabV3_2_5DConfig, DeeplabV3Config
from medsegpy.modeling.meta_arch import build
from medsegpy.utils import env

if not env.is_tf2():
    _TF2 = False
    from medsegpy.modeling.build import get_model
else:
    _TF2 = True
    get_model = None


class TestDeeplabV3Plus(unittest.TestCase):
    """Test building 2D/2.5D UNet using the builder."""

    def _compare_layers(cls, m1, m2):
        for l1, l2 in zip(m1.layers, m2.layers):
            assert type(l1) == type(l2)  # noqa

            l1_cfg = l1.get_config()
            l2_cfg = l2.get_config()
            l2_cfg.pop("name"), l1_cfg.pop("name")

            assert l1_cfg == l2_cfg, "{}: {}\n{}: {}".format(
                l1.name, l1.get_config(), l2.name, l2.get_config()
            )

    def test_same_deeplabv3_2d(self):
        """
        2D DeeplabV3+ should be same between builder and function construct.
        """
        if _TF2:
            return
        cfg = DeeplabV3Config()
        cfg.CATEGORIES = [0]
        m1 = get_model(cfg)
        m2 = build.build_model(cfg)

        self._compare_layers(m1, m2)

    def test_same_deeplabv3_2_5d(self):
        """
        2.5D DeeplabV3+ should be same between builder and function construct.
        """
        if _TF2:
            return
        cfg1 = DeeplabV3_2_5DConfig()
        cfg1.IMG_SIZE = (288, 288, 3)
        cfg1.CATEGORIES = [0]
        cfg2 = DeeplabV3Config()
        cfg2.IMG_SIZE = (288, 288, 3)
        cfg2.CATEGORIES = [0]
        m1 = get_model(cfg1)
        m2 = build.build_model(cfg2)

        self._compare_layers(m1, m2)
