import os

os.environ["MSK_SEG_NETWORKS_PROJECT"] = "abCT"

import numpy as np
import h5py
from medsegpy.data import im_gens
from medsegpy.nn_train import NNTrain

from medsegpy.config import Config

CLASS_WEIGHTS = np.asarray([100, 1])
SAVE_BEST_WEIGHTS = True
FREEZE_LAYERS = None


class CTGenerator(im_gens.OAIGenerator):
    """
        Generator to be used with files where training/testing data is written as 2D slices
        Filename: PATIENTID_VISIT_AUGMENTATION-NUMBER_SLICE-NUMBER
        Filename Format: '%07d_V%02d_Aug%02d_%03d' (e.g. '0000001_V00_Aug00_001.h5
    """
    SUPPORTED_TAGS = ['abct', 'ct']
    __EXPECTED_IMG_SIZE_DIMS__ = 2

    def __init__(self, config: Config, windows = None):
        if windows and config.num_neighboring_slices() != len(windows):
            raise ValueError(
                "Expected {} windows".format(config.num_neighboring_slices()))
        self.windows = windows
        super().__init__(config)

    def _load_inputs(self, data_path: str, file: str):
        im, seg = self._load_inputs_basic(data_path, file)
        im = self._preprocess(im, self.windows[0] if self.windows else None)
        return im, seg

    def _load_inputs_basic(self, data_path: str, file: str):
        im_path = '%s/%s.im' % (data_path, file)
        with h5py.File(im_path, 'r') as f:
            im = f['data'][:]
            if len(im.shape) == 2:
                im = im[..., np.newaxis]

        seg_path = '%s/%s.seg' % (data_path, file)
        with h5py.File(seg_path, 'r') as f:
            seg = f['data'][:].astype('float32')
        seg = np.expand_dims(seg, axis=2)  # legacy purposes

        assert len(im.shape) == 3
        assert len(seg.shape) == 4 and seg.shape[-2] == 1

        return im, seg

    def _preprocess(self, im, window):
        # Apply windowing.
        if window:
            im = np.clip(im, window[0], window[1])

        # Preprocess by max normalizing.
        im -= np.min(im)
        im /= np.max(im)

        return im

    def __load_neighboring_slices__(self, num_slices, filepath, max_slice):
        """Stacks 2D CT slices from single patient clipped at different window levels.

        Overloads traditional 2.5D networks that look at neighboring slices.
        """
        data_path, filename = os.path.dirname(filepath), os.path.basename(
            filepath)
        im, seg = self._load_inputs_basic(data_path, filename)
        h, w = im.shape[:2]

        ims = []
        for window in self.windows:
            ims.append(np.squeeze(self._preprocess(np.copy(im), window)))
        im = np.stack(ims, axis=-1)

        assert im.shape == (h, w, self.config.num_neighboring_slices())
        return im, seg


class CTTrain(NNTrain):
    __DESCRIPTION__ = 'Train networks for ct segmentation'

    _ARG_KEY_WINDOWS = "windows"

    @staticmethod
    def _add_classes_parser(parser):
        parser.add_argument("--classes", type=int, nargs="+",
                            required=True,
                            help="tissue indices to segment")

    def _add_default_args(self, parser):
        super()._add_default_args(parser)
        parser.add_argument('--{}'.format(self._ARG_KEY_WINDOWS),
                            metavar='W', type=str, nargs='*',
                            dest=self._ARG_KEY_WINDOWS,
                            help='(min, max) windows for clipping data')

    def parse_windows(self, windows):
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

    def _train_cross_validation(self, config):
        raise NotImplementedError(
            "Cross validation not supported for CT training")

    def _build_data_loaders(self, config):
        window_keys = self.get_arg(self._ARG_KEY_WINDOWS)
        windows = self.parse_windows(window_keys) if window_keys else None
        generator = CTGenerator(config, windows)

        return generator, generator


if __name__ == '__main__':
    nn_train = CTTrain()
    nn_train.parse()
    nn_train.run()
