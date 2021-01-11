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
import os
import warnings

from medsegpy import config
from medsegpy.engine.defaults import default_argument_parser, default_setup
from medsegpy.engine.trainer import DefaultTrainer
from medsegpy.modeling import Model, model_from_json
from medsegpy.utils import env

try:
    import wandb
except:
    pass


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

    # TODO: Add suppport for resume.
    if env.supports_wandb():
        exp_name = cfg.EXP_NAME
        if not exp_name:
            warnings.warn("EXP_NAME not specified. Defaulting to basename...")
            exp_name = os.path.basename(cfg.OUTPUT_DIR)
        wandb.init(
            project="tech-considerations",
            name=exp_name,
            config=cfg,
            sync_tensorboard=False,
            job_type="training",
            dir=cfg.OUTPUT_DIR,
            entity="arjundd",
            notes=cfg.DESCRIPTION,
        )

    return cfg


def main(args, trainer_cls: type = DefaultTrainer):
    assert issubclass(trainer_cls, DefaultTrainer)
    cfg = setup(args)
    if args.eval_only:
        if not cfg.TEST_DATASET:
            raise ValueError("TEST_DATASET not specified")

        # Try to build model from json file.
        # If fails, build straight from config.
        model = None
        model_json_file = os.path.join(cfg.OUTPUT_DIR, "model.json")
        if os.path.isfile(model_json_file):
            try:
                with open(model_json_file) as f:
                    json_str = f.read()
                model = model_from_json(json_str, custom_objects={"Model": Model})
            except Exception as e:
                print(e)
        if model is None:
            model = trainer_cls.build_model(cfg)
        if env.is_tf2():
            model.run_eagerly = not args.non_eagerly

        return trainer_cls.test(cfg, model)

    # Configure Keras to run non-eagerly instead of disabling eager mode in tensorflow.
    # Allows larger batch size for some reason
    extra_args = {"run_eagerly": not args.non_eagerly} if env.is_tf2() else {}
    trainer = trainer_cls(cfg, **extra_args)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
