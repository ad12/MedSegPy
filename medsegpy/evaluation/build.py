from typing import Sequence

from fvcore.common.registry import Registry

from medsegpy.config import Config
from medsegpy.data import MetadataCatalog

EVALUATOR_REGISTRY = Registry("EVALUATOR")
"""
Registry for evaluators, which process input/output pairs to compute metric
scores and save data to disk. The evaluator type should be registered with
it's metadata in :class:`MetadataCatalog`.

The registered object will be called with
`obj(dataset_name, cfg, output_dir, save_raw_data)`.
The call should return a :class:`DatasetEvaluator` object.
"""


def build_evaluator(
    dataset_name: str, cfg: Config, output_folder: str = None, save_raw_data: bool = False
):
    name = MetadataCatalog.get(dataset_name).evaluator_type
    if isinstance(name, str):
        name = [name]
    elif isinstance(name, dict):
        primary_task = cfg.PRIMARY_TASK
        assert primary_task in name, (
            f"Primary Task (= {primary_task}) is not a key in "
            f"evaluator dictionary for dataset '{dataset_name}'"
        )
        name = [name[primary_task]]
    assert isinstance(name, Sequence)

    evaluators = []
    for n in name:
        evaluator_cls = EVALUATOR_REGISTRY.get(n)
        evaluators.append(evaluator_cls(dataset_name, cfg, output_folder, save_raw_data))

    return evaluators
