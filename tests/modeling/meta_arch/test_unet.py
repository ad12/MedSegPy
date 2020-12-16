import unittest

from keras import backend as K
from keras.layers import Activation, Concatenate, Conv3D, Conv3DTranspose

from medsegpy.config import UNet2_5DConfig, UNet3DConfig, UNetConfig
from medsegpy.modeling.meta_arch import build
from medsegpy.utils import env

if not env.is_tf2():
    _TF2 = False
    from medsegpy.modeling.build import get_model
else:
    _TF2 = True
    get_model = None


class TestUNet2D(unittest.TestCase):
    """Test building 2D/2.5D UNet using the builder."""

    @classmethod
    def _compare_layers(cls, m1, m2):
        for l1, l2 in zip(m1.layers, m2.layers):
            assert type(l1) == type(l2)  # noqa

            l1_cfg = l1.get_config()
            l2_cfg = l2.get_config()
            l2_cfg.pop("name"), l1_cfg.pop("name")

            if isinstance(l2, Concatenate):
                channel_axis = (
                    -1 if K.image_data_format() == "channels_last" else 1
                )  # noqa
                mapping = {channel_axis: 3}
                l2_cfg["axis"] = mapping[l2_cfg["axis"]]

            assert l1_cfg == l2_cfg, "{}: {}\n{}: {}".format(
                l1.name, l1.get_config(), l2.name, l2.get_config()
            )

    def test_same_unet2d(self):
        """2D U-Net should be same between builder and function construct."""
        if _TF2: return
        cfg = UNetConfig()
        cfg.CATEGORIES = [0]
        m1 = get_model(cfg)
        m2 = build.build_model(cfg)

        self._compare_layers(m1, m2)

    def test_same_unet_2_5d(self):
        """2.5D U-Net should be same between builder and function construct.

        The builder takes in the 2D U-Net config and constructs a 2.5 config.
        This motivates deprecating UNet2_5DConfig.
        """
        if _TF2: return
        cfg1 = UNet2_5DConfig()
        cfg1.IMG_SIZE = (288, 288, 3)
        cfg1.CATEGORIES = [0]
        cfg2 = UNetConfig()
        cfg2.IMG_SIZE = (288, 288, 3)
        cfg2.CATEGORIES = [0]
        m1 = get_model(cfg1)
        m2 = build.build_model(cfg2)

        self._compare_layers(m1, m2)


class TestUNet3D(unittest.TestCase):
    """Test building 3D UNet using the builder."""

    def test_same_unet_3d(self):
        """3D U-Net should be same between builder and function construct."""
        if _TF2: return
        cfg = UNet3DConfig()
        cfg.CATEGORIES = [0]
        m1 = get_model(cfg)
        m2 = build.build_model(cfg)

        l1_idx, l2_idx = 0, 0

        while True:
            reached_l1 = l1_idx == len(m1.layers)
            reached_l2 = l2_idx == len(m2.layers)
            if reached_l1 and reached_l2:
                break

            assert (
                not reached_l1 and not reached_l2
            ), "l1_idx: {}, l2_idx: {}".format(l1_idx, l2_idx)

            l1 = m1.layers[l1_idx]
            l1_cfg = l1.get_config()
            l2 = m2.layers[l2_idx]
            l2_cfg = l2.get_config()

            l1_cfg.pop("name"), l2_cfg.pop("name")

            if isinstance(l2, Concatenate):
                channel_axis = (
                    -1 if K.image_data_format() == "channels_last" else 1
                )  # noqa
                mapping = {channel_axis: 4}
                l2_cfg["axis"] = mapping[l2_cfg["axis"]]
            elif isinstance(l1, Conv3D) and not isinstance(l1, Conv3DTranspose):
                # Handle conv+activation case.
                l1_cfg.pop("activation")
                l1_activation = m1.layers[l1_idx + 1]
                assert isinstance(
                    l1_activation, Activation
                ), "Expected Activation - l1_idx: {}: {}\n{}".format(
                    l1_idx,
                    type(l1_activation),
                    m1.layers[l1_idx - 1 : l1_idx + 1],
                )
                l1_activation = l1_activation.get_config()["activation"]
                l2_activation = l2_cfg.pop("activation")
                assert (
                    l1_activation == l2_activation
                ), "l1_idx: {} - {}\nl2_cfg: {} - {}".format(
                    l1_idx, l1_cfg, l2_idx, l2_cfg
                )
                l1_idx += 1

            assert l1_cfg == l2_cfg, "{}: {}\n{}: {}".format(
                l1.name, l1_cfg, l2.name, l2_cfg
            )

            l1_idx += 1
            l2_idx += 1


if __name__ == "__main__":
    unittest.main()
