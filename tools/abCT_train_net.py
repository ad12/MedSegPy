"""Sample extension of train_net.py for the abdominal CT dataset."""
import os

from medsegpy.data.im_gens import CTGenerator
from medsegpy.engine.defaults import default_argument_parser
from medsegpy.engine.trainer import DefaultTrainer
from train_net import main


def parse_windows(windows):
    windowing = {
        "soft": (400, 50),
        "bone": (1800, 400),
        "liver": (150, 30),
        "spine": (250, 50),
        "custom": (500, 50)
    }
    vals = []
    for w in windows:
        if w not in windowing:
            raise KeyError("Window {} not found".format(w))
        window_width = windowing[w][0]
        window_level = windowing[w][1]
        upper = window_level + window_width / 2
        lower = window_level - window_width / 2

        vals.append((lower, upper))

    return vals


class AbCTTrainer(DefaultTrainer):
    _ARG_KEY_WINDOWS = "windows"

    def _build_data_loaders(self, cfg):
        windows = cfg.PREPROCESSING_WINDOWS
        windows = parse_windows(windows) if windows else None
        generator = CTGenerator(cfg, windows)

        return generator, generator

    def build_test_data_loader(self, cfg):
        windows = cfg.PREPROCESSING_WINDOWS
        windows = parse_windows(windows) if windows else None
        return CTGenerator(cfg, windows)


if __name__ == '__main__':
    basename = os.path.splitext(os.path.basename(__file__))[0]
    args = default_argument_parser().parse_args()
    main(args, AbCTTrainer)

