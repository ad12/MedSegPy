"""Medical semantic segmentation training script.

This script reads a given config file and runs the training or evaluation.
It is a standardized entry point for medsegpy.

This script is designed to support training of many models. As a result,
it may not be suitable for your specific project, which may consist of
new dataset loaders, evaluators, etc.

We recommend using medsegpy as a library and use this file as an example of how
to use the library. If your project requires new dataset loaders, evaluators, or
other customizations, we suggest writing your own script. See `ct_train.py` for
an example of how to extend to different use cases.
"""
from medsegpy import config
from medsegpy.engine.defaults import default_argument_parser, default_setup
from medsegpy.engine.trainer import DefaultTrainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    config_file = args.config_file
    model_name = config.get_model_name(config_file)
    cfg = config.get_config(model_name, create_dirs=False)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    default_setup(cfg, args)
    return cfg


def main(args, trainer_cls: type = DefaultTrainer):
    assert issubclass(trainer_cls, DefaultTrainer)
    cfg = setup(args)
    if args.eval_only:
        if not cfg.TEST_DATASET:
            raise ValueError("TEST_DATASET not specified")
        model = trainer_cls.build_model(cfg)
        return trainer_cls.test(cfg, model)

    trainer = trainer_cls(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
