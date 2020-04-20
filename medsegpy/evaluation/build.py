from fvcore.common.registry import Registry

from medsegpy.config import Config
from medsegpy.data import MetadataCatalog

EVALUATOR_REGISTRY = Registry("EVALUATOR")
"""
Registry for evaluators, which process input/output pairs to compute metric
scores and save data to disk. The evaluator type should be registered with
it's metadata in :class:`MetadataCatalog`.

The registered object will be called with `obj(dataset_name, cfg, output_dir, save_raw_data)`.
The call should return a `nn.Module` object.
"""


def build_evaluator(
    dataset_name: str,
    cfg: Config,
    output_dir: str=None,
    save_raw_data: bool = False,
):
    name = MetadataCatalog.get(dataset_name).evaluator_type
    evaluator_cls = EVALUATOR_REGISTRY.get(name)

    return evaluator_cls(dataset_name, cfg, output_dir, save_raw_data)
