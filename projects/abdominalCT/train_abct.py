"""Sample extension of train_net.py for the abdominal CT dataset."""
import os
from typing import Sequence

from medsegpy import config
from medsegpy.engine.defaults import default_argument_parser, default_setup
from medsegpy.engine.trainer import DefaultTrainer
from medsegpy.modeling import Model, model_from_json


def parse_windows(windows):
    """Parse windows provided by the user.

    These windows can either be strings corresponding to popular windowing
    thresholds for CT or tuples of (upper, lower) bounds.
    """
    windowing = {
        "soft": (400, 50),
        "bone": (1800, 400),
        "liver": (150, 30),
        "spine": (250, 50),
        "custom": (500, 50),
    }
    vals = []
    for w in windows:
        if isinstance(w, Sequence) and len(w) == 2:
            assert_msg = "Expected tuple of (lower, upper) bound"
            assert len(w) == 2, assert_msg
            assert isinstance(w[0], (float, int)), assert_msg
            assert isinstance(w[1], (float, int)), assert_msg
            assert w[0] < w[1], assert_msg
            vals.append(w)
            continue

        if w not in windowing:
            raise KeyError("Window {} not found".format(w))
        window_width = windowing[w][0]
        window_level = windowing[w][1]
        upper = window_level + window_width / 2
        lower = window_level - window_width / 2

        vals.append((lower, upper))

    return tuple(vals)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    config_file = args.config_file
    model_name = config.get_model_name(config_file)
    cfg = config.get_config(model_name, create_dirs=False)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Convert preprocessing windows from string representation to integer
    # values.
    if cfg.PREPROCESSING_WINDOWS:
        cfg.PREPROCESSING = ("Windowing",)
        windows = cfg.PREPROCESSING_WINDOWS
        assert cfg.IMG_SIZE[-1] == len(
            windows
        ), "Expected {} channels for {} windows".format(len(windows), windows)
        cfg.PREPROCESSING_WINDOWS = parse_windows(windows)

    default_setup(cfg, args)
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
                model = model_from_json(
                    json_str, custom_objects={"Model": Model}
                )
            except Exception as e:
                print(e)
        if model is None:
            model = trainer_cls.build_model(cfg)

        return trainer_cls.test(cfg, model)

    trainer = trainer_cls(cfg)
    return trainer.train()


if __name__ == "__main__":
    basename = os.path.splitext(os.path.basename(__file__))[0]
    args = default_argument_parser().parse_args()
    main(args)
