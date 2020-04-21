"""Medical semantic segmentation training script.

This script reads a given config file and runs the training or evaluation.
It is a standardized entry point for medsegpy.

This script is designed to support training of many models. As a result,
it may not be suitable for your specific project, which may consist of
new dataset loaders, evaluators, etc.

We recommend using medsegpy as a library and use this file as an example of how
to use the library. If your project requires new dataset loaders, evaluators, or
other customizations, we suggest writing your own script.
"""
import logging

from medsegpy import config
from medsegpy.engine.defaults import default_argument_parser, default_setup
from medsegpy.engine.trainer import DefaultTrainer
from medsegpy.utils import dl_utils

logger = logging.getLogger(__name__)

FREEZE_LAYERS = None


def setup(args):
    """
    Create configs and perform basic setups.
    """
    config_file = args.config_file
    model_name = config.get_cp_save_tag(config_file)
    cfg = config.get_config(model_name, create_dirs=False)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    default_setup(cfg, args)
    return cfg


def main(args, trainer_cls: type = DefaultTrainer):
    assert issubclass(trainer_cls, DefaultTrainer)
    cfg = setup(args)
    if args.eval_only:
        model = trainer_cls.build_model(cfg)
        return trainer_cls.test(cfg, model, dl_utils.get_weights(cfg.OUTPUT_DIR))

    trainer = trainer_cls(cfg)
    return trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger("medsegpy.nn_train.{}".format(__name__))
    args = default_argument_parser().parse_args()
    main(args)
