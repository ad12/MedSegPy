import warnings
from abc import ABC, abstractmethod

from fvcore.common.registry import Registry

from medsegpy.config import Config

from ..model import Model

META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`. The resulting object
should be duck typed with `build_model(input_tensor)`.
"""

_MODEL_MAP = {
    ("unet_2d", "unet_2_5d"): "UNet2D",
    ("unet_3d",): "UNet3D",
    ("deeplabv3_2d", "deeplabv3_2_5d", "deeplabv3+"): "DeeplabV3Plus",
}

LEGACY_MODEL_NAMES = {x: v for k, v in _MODEL_MAP.items() for x in k}


def build_model(cfg, input_tensor=None) -> Model:
    """
    Build the whole model architecture, defined by ``cfg.MODEL_NAME``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.MODEL_NAME
    try:
        META_ARCH_REGISTRY.get(name)
    except KeyError:
        # Legacy code used different tags for building models.
        prev_name = name
        if name in LEGACY_MODEL_NAMES:
            name = LEGACY_MODEL_NAMES[name]
            if prev_name != name:
                warnings.warn("MODEL_NAME {} is deprecated. Use {} instead".format(prev_name, name))

    builder = META_ARCH_REGISTRY.get(name)(cfg)
    model = builder.build_model(input_tensor)
    assert isinstance(model, Model), (
        "ModelBuilder.build_model should output model of type " "medsegpy.modeling.Model"
    )
    return model


class ModelBuilder(ABC):
    def __init__(self, cfg: Config):
        self._cfg = cfg

    @abstractmethod
    def build_model(self, input_tensor=None) -> Model:
        """Build model."""
        pass
