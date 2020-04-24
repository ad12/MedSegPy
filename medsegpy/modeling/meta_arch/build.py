from fvcore.common.registry import Registry
from ..model import Model

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`. The resulting object
should be duck typed with `build_model(input_tensor)`.
"""

_MODEL_MAP = {
    ("unet_2d", "unet_2_5d"): "UNet2D",
    ("unet_3d",): "UNet3D",
}


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
        for k, v in _MODEL_MAP.items():
            if name in k:
                name = v
                break

    builder = META_ARCH_REGISTRY.get(name)(cfg)
    return builder.build_model(input_tensor)
