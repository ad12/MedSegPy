"""This script is to verify that we are able to train using the data loader.

TODO:
Once this is verified and functionality similarity between generators and
data loaders in MedSegPy are verified, delete.
"""
from medsegpy import config
from medsegpy.engine.defaults import default_argument_parser, default_setup
from medsegpy.engine.trainer import DefaultTrainerDataLoader


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


def main(args, trainer_cls: type = DefaultTrainerDataLoader):
    assert issubclass(trainer_cls, DefaultTrainerDataLoader)
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
