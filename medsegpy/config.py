import ast
import configparser
import copy
import logging
import os
import warnings
import yaml
from itertools import groupby
from typing import Any, Dict, Tuple

from fvcore.common.file_io import PathManager

from medsegpy.cross_validation import cv_util
from medsegpy.losses import DICE_LOSS, get_training_loss_from_str
from medsegpy.utils import utils as utils

logger = logging.getLogger(__name__)

# Keys that have been deprecated.
DEPRECATED_KEYS = [
    "NUM_CLASSES",
    "TRAIN_FILES_CV",
    "VALID_FILES_CV",
    "TEST_FILES_CV",
    "USE_STEP_DECAY",
    "PIK_SAVE_PATH_DIR",
    "PIK_SAVE_PATH",
    "TF_LOG_DIR",
    "TRAIN_PATH",
    "VALID_PATH",
    "TEST_PATH",
    "PLOT_MODEL_PATH",
    "FINE_TUNE",
    "LEARN_FILES",
    "DEBUG",
    "TEST_RESULT_PATH",
    "TEST_RESULTS_FOLDER_NAME",
    "DATE_TIME_STR",
]

RENAMED_KEYS = {
    "CP_SAVE_PATH": "OUTPUT_DIR",
    "CP_SAVE_TAG": "MODEL_NAME",
    "INIT_WEIGHT_PATH": "INIT_WEIGHTS",
    "TISSUES": "CATEGORIES",
}

BASE_KEY = "_BASE_"


class Config(object):
    """A config object that is 1-to-1 with supported models.

    Each subclass of :class:`Config` corresponds to a specific model
    architecture.
    """

    VERSION = 7

    # Model name specific to config. Cannot be changed.
    MODEL_NAME = ""

    # Experiment name and description of the config.
    EXP_NAME = ""
    DESCRIPTION = ""

    # Loss function in form (id, output_mode)
    LOSS = DICE_LOSS
    CLASS_WEIGHTS = None
    # Class name for robust loss computation
    ROBUST_LOSS_NAME = ""
    ROBUST_LOSS_STEP_SIZE = 1e-1

    # PIDS to include, None = all pids
    PIDS = None

    # Training and validation image size
    IMG_SIZE = (288, 288, 1)
    # Patch-based image loading params used with PatchDataLoader
    IMG_PAD_SIZE = ()
    IMG_PAD_MODE = "edge"
    IMG_STRIDE = ()

    # Training parameters
    N_EPOCHS = 100
    AUGMENT_DATA = False

    # Step Decay params
    INITIAL_LEARNING_RATE = 1e-4
    LR_SCHEDULER_NAME = ""
    MIN_LEARNING_RATE = 1e-8
    DROP_FACTOR = 0.7
    DROP_RATE = 1.0
    LR_MIN_DELTA = 1e-4
    LR_PATIENCE = 0
    LR_COOLDOWN = 0
    NUM_GRAD_STEPS = 1

    # ADAM optimizer decay
    ADAM_DECAY = 0.0
    USE_AMSGRAD = False

    # Early stopping criterion
    USE_EARLY_STOPPING = False
    EARLY_STOPPING_MIN_DELTA = 0.0
    EARLY_STOPPING_PATIENCE = 0
    EARLY_STOPPING_CRITERION = "val_loss"

    # Batch sizes
    TRAIN_BATCH_SIZE = 12
    VALID_BATCH_SIZE = 35
    TEST_BATCH_SIZE = 72

    # Categories
    CATEGORIES = []
    INCLUDE_BACKGROUND = False

    # File Types
    FILE_TYPES = ["im"]

    # Transfer Learning
    INIT_WEIGHTS = ""
    FREEZE_LAYERS = ()

    # Dataset names
    TRAIN_DATASET = ""
    VAL_DATASET = ""
    TEST_DATASET = ""

    # Cross-Validation-Parameters
    USE_CROSS_VALIDATION = False
    CV_FILE = ""
    CV_K = 0
    CV_TRAIN_BINS = []
    CV_VALID_BINS = []
    CV_TEST_BINS = []
    __CV_TRAIN_FILES__ = None
    __CV_VALID_FILES__ = None
    __CV_TEST_FILES__ = None

    # Training Model Paths
    OUTPUT_DIR = ""

    # DataLoader tag.
    TAG = "DefaultDataLoader"
    PRELOAD_DATA = False

    # Weights kernel initializer.
    KERNEL_INITIALIZER = "he_normal"

    # System params
    NUM_WORKERS = 1
    SEED = None

    # Evaluation params
    TEST_WEIGHT_PATH = ""
    TEST_METRICS = ["DSC", "VOE", "ASSD", "CV"]

    # Extra parameters related to different parameters.
    PREPROCESSING = ()
    PREPROCESSING_WINDOWS = ()

    def __init__(self, cp_save_tag, state="training", create_dirs=True):
        if state not in ["testing", "training"]:
            raise ValueError("state must either be 'training' or 'testing'")

        self.MODEL_NAME = cp_save_tag
        self.STATE = state

    def init_cross_validation(
        self,
        train_files,
        valid_files,
        test_files,
        train_bins,
        valid_bins,
        test_bins,
        cv_k,
        cv_file,
        output_dir,
    ):
        """Initialize config for cross validation.

        Returns:
            Config: A deep copy of the config. This copy is initialized for
                cross validation.
        """
        assert (
            self.STATE == "training"
        ), "Initializing cross-validation must be done in training state"

        config = copy.deepcopy(self)
        config.USE_CROSS_VALIDATION = True
        config.CV_TRAIN_BINS = train_bins
        config.CV_VALID_BINS = valid_bins
        config.CV_TEST_BINS = test_bins
        config.CV_K = cv_k
        config.CV_FILE = cv_file

        config.__CV_TRAIN_FILES__ = train_files
        config.__CV_VALID_FILES__ = valid_files
        config.__CV_TEST_FILES__ = test_files

        os.makedirs(output_dir, exist_ok=True)
        config.OUTPUT_DIR = output_dir

        return config

    def save_config(self):
        """Save params of config to ini file."""
        filepath = os.path.join(self.OUTPUT_DIR, "config.yaml")
        config_vars = self.to_dict()

        # Save config
        with PathManager.open(filepath, "w") as configfile:
            yaml.dump(config_vars, configfile)

        logger.info("Full config saved to {}".format(os.path.abspath(filepath)))

    def to_dict(self):
        """Returns a dictionary representation of the config.

        This representation only contains public instance variable fields.
        Private and protected instance variables are excluded. Methods are excluded.

        Returns:
            dict: The config as a dictionary.
        """
        members = [
            attr
            for attr in dir(self)
            if not attr.startswith("__")
            and not callable(getattr(self, attr))
            and not (hasattr(type(self), attr) and isinstance(getattr(type(self), attr), property))
        ]
        return {m_var: getattr(self, m_var) for m_var in members}

    def _parse_special_attributes(self, full_key: str, value: Any) -> Tuple[str, Any]:
        """Special parsing values for attributes.

        Used when loading config from a file or from list.

        Args:
            full_key (str): Upper case attribute representation.
            value (Any): Corresponding value.
        """
        # Avoid circular dependencies.
        from medsegpy.data import MetadataCatalog  # noqa
        from medsegpy.data.data_loader import LEGACY_DATA_LOADER_NAMES  # noqa
        from medsegpy.modeling.meta_arch.build import LEGACY_MODEL_NAMES  # noqa

        prev_key, prev_val = full_key, value

        if full_key in ("TRAIN_PATH", "VALID_PATH", "TEST_PATH") and value:
            mapping = {
                "TRAIN_PATH": "TRAIN_DATASET",
                "VALID_PATH": "VAL_DATASET",
                "TEST_PATH": "TEST_DATASET",
            }
            value = MetadataCatalog.convert_path_to_dataset(value)
            full_key = mapping[full_key]
            logger.info("Converting {} -> {}: {} -> {}".format(prev_key, full_key, prev_val, value))
        elif full_key == "TAG" and value in LEGACY_DATA_LOADER_NAMES:
            value = LEGACY_DATA_LOADER_NAMES[value]
            logger.info("Converting {}: {} -> {}".format(full_key, prev_val, value))
        elif full_key == "MODEL_NAME" and value in LEGACY_MODEL_NAMES:
            value = LEGACY_MODEL_NAMES[value]
            logger.info("Converting {}: {} -> {}".format(full_key, prev_val, value))
        elif full_key == "LOSS" and isinstance(value, str):
            try:
                value = get_training_loss_from_str(value)
            except ValueError:
                pass
        elif full_key == "OUTPUT_DIR":
            value = PathManager.get_local_path(value)

        return full_key, value

    def merge_from_file(self, cfg_filename: str, new_allowed=False):
        """Load a ini or yaml config file and merge it with this object.

        "MODEL_NAME" must be specified in the file.

        Args:
            cfg_filename (str): File path to yaml or ini file.
            new_allowed (bool, optional): If `True`, new keys, not already defined,
                are allowed.
        """
        # Avoid circular dependencies.
        from medsegpy.modeling.meta_arch.build import LEGACY_MODEL_NAMES

        vars_dict = self._load_dict_from_file(cfg_filename)

        # Load information from base files first.
        if BASE_KEY in vars_dict:
            vars_dict = self.parse_base_config(cfg_filename)

        # TODO: Handle cp save tag as a protected key.
        model_name = (
            vars_dict["MODEL_NAME"] if "MODEL_NAME" in vars_dict else vars_dict["CP_SAVE_TAG"]
        )
        if model_name in LEGACY_MODEL_NAMES:
            model_name = LEGACY_MODEL_NAMES[model_name]
        if model_name != self.MODEL_NAME:
            raise ValueError(
                "Wrong config. Expected {}. Got {}".format(self.MODEL_NAME, model_name)
            )

        for full_key, value in vars_dict.items():
            full_key = str(full_key).upper()

            if full_key in RENAMED_KEYS:
                new_name = RENAMED_KEYS[full_key]
                logger.warning("Key {} has been renamed to {}".format(full_key, new_name))
                full_key = new_name

            full_key, value = self._parse_special_attributes(full_key, value)

            if full_key in DEPRECATED_KEYS:
                logger.warning("Key {} is deprecated, not loading".format(full_key))
                continue

            if not new_allowed and not hasattr(self, full_key):
                raise ValueError("Key {} does not exist.".format(full_key))

            is_new = not hasattr(self, full_key)

            if is_new or full_key == "TRAIN_DATASET":
                value = self._decode_cfg_value(value, "auto")
            else:
                value = self._decode_cfg_value(value, type(self.__getattribute__(full_key)))
                value = _check_and_coerce_cfg_value_type(
                    value, self.__getattribute__(full_key), full_key
                )

            # Loading config
            self.__setattr__(full_key, value)

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list (e.g. from command line).

        For example, cfg_list = ['FOO_BAR', 0.5, 'BAR_FOO', (0,3,4)]
        """
        _error_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; " "it must be a list of pairs".format(cfg_list),
        )

        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if full_key == "MODEL_NAME":
                raise ValueError("Cannot change key MODEL_NAME")
            if self.key_is_deprecated(full_key):
                continue

            if self.key_is_renamed(full_key):
                self.raise_key_rename_error(full_key)

            _error_with_logging(
                hasattr(self, full_key),
                "Non-existent key: {}".format(full_key),
                error_type=KeyError,
            )
            value = self._decode_cfg_value(v, type(self.__getattribute__(full_key)))
            value = _check_and_coerce_cfg_value_type(
                value, self.__getattribute__(full_key), full_key
            )
            self.__setattr__(full_key, value)

    @classmethod
    def _decode_cfg_value(cls, value, data_type):
        """
        Decodes a raw config value (e.g., from a yaml config files or command
        line argument) into a Python object.

        If the value is a dict, it will be interpreted as a new CfgNode.
        If the value is a str, it will be evaluated as literals.
        Otherwise it is returned as-is.
        """
        # Configs parsed from raw yaml will contain dictionary keys that need to
        # be converted to CfgNode objects
        """
        Convert string to relevant data type
        :param var_string: variable as a string (e.g.: '[0]', '1', 'hellow')
        :param data_type: the type of the data
        :return: string converted to data_type
        """
        if isinstance(value, dict):
            return {k: cls._decode_cfg_value(v, "auto") for k, v in value.items()}
        elif not isinstance(value, str):
            return value

        if data_type is str:
            return str(value)
        elif data_type is float:
            return float(value)
        elif data_type is int:
            return int(value)
        else:
            try:
                return ast.literal_eval(value)
            except (SyntaxError, ValueError) as e:
                if data_type == "auto":
                    return value
                else:
                    raise e

    def key_is_deprecated(self, full_key):
        """Test if a key is deprecated."""
        if full_key in DEPRECATED_KEYS:
            logger.warning("Deprecated config key (ignoring): {}".format(full_key))
            return True
        return False

    def key_is_renamed(self, full_key):
        """Test if a key is renamed."""
        return full_key in RENAMED_KEYS

    def add_new_key(self, key, value) -> None:
        """Add a new key to the config schema.

        Args:
            key (str): Key name
            value (Any): Value of the key to set.
        """
        if key in self.__dict__:
            raise KeyError("Key {} already exists in config.".format(key))
        value = self._decode_cfg_value(value, "auto")
        self.__setattr__(key, value)

    def raise_key_rename_error(self, full_key):
        new_key = RENAMED_KEYS[full_key]
        if isinstance(new_key, tuple):
            msg = " Note: " + new_key[1]
            new_key = new_key[0]
        else:
            msg = ""
        raise KeyError(
            "Key {} was renamed to {}; please update your config.{}".format(full_key, new_key, msg)
        )

    @classmethod
    def parse_base_config(cls, cfg_filename: str) -> Dict:
        local_dir = os.path.dirname(PathManager.get_local_path(cfg_filename))
        vars_dict = cls._load_dict_from_file(cfg_filename)
        base_cfg_filename = vars_dict.pop(BASE_KEY, None)
        if base_cfg_filename:
            if base_cfg_filename.startswith("~"):
                base_cfg_filename = os.path.expanduser(base_cfg_filename)
            elif any(map(base_cfg_filename.startswith, ["/", "https://", "http://"])):
                raise ValueError("Remote configs not currently supported.")
            else:
                # the path to base cfg is relative to the config file itself.
                base_cfg_filename = os.path.join(local_dir, base_cfg_filename)
            base_dict = cls.parse_base_config(base_cfg_filename)
            base_dict.update(vars_dict)
            vars_dict = base_dict

        assert BASE_KEY not in vars_dict, "{} should be popped off!".format(BASE_KEY)
        return vars_dict

    @classmethod
    def _load_dict_from_file(cls, cfg_filename):
        filename = PathManager.get_local_path(cfg_filename)
        if filename.endswith(".ini"):
            cfg = configparser.ConfigParser()
            if not os.path.isfile(filename):
                raise FileNotFoundError("Config file {} not found".format(filename))
            cfg.read(filename)
            vars_dict = cfg["DEFAULT"]
            vars_dict = {k.upper(): v for k, v in vars_dict.items()}
        elif filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(filename, "r") as f:
                vars_dict = yaml.load(f)
        else:
            raise ValueError("file {} not supported".format(filename))

        return vars_dict

    def set_attr(self, attr, val):
        """
        Wrapper method to set attributes of config
        :param attr: a string attr
        :param val: value to set attribute to

        :raises ValueError: if attr is not a string
                            if attr does not exist for the config
                            if type of val is different from the default type
        """
        if type(attr) is not str:
            raise ValueError("attr must be of type str")

        if not hasattr(self, attr):
            raise ValueError("The attribute %s does not exist" % attr)
        curr_val = self.__getattribute__(attr)

        if type(val) is str and type(curr_val) is not str:
            val = utils.convert_data_type(var_string=val, original=curr_val)

        if curr_val is not None and (type(val) != type(curr_val)):
            raise ValueError(
                "%s is of type %s. Expected %s" % (attr, str(type(val)), str(type(curr_val)))
            )

        self.__setattr__(attr, val)

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError(f"Key {key} does not exist")
        return getattr(self, key)

    def get(self, key, default=None):
        keys = key.split(".")
        try:
            val = self
            for k in keys:
                val = val[k]
        except KeyError:
            return default

        return val

    def change_to_test(self):
        """
        Initialize testing state
        """
        self.STATE = "testing"

        # if cross validation is enabled, load testing cross validation bin
        if self.USE_CROSS_VALIDATION:
            assert self.CV_FILE, "No cross-validation file found in config"
            cv_processor = cv_util.CrossValidationProcessor(self.CV_FILE)
            bins = (self.CV_TRAIN_BINS, self.CV_VALID_BINS, self.CV_TEST_BINS)

            train_files, valid_files, test_files = cv_processor.get_fnames(bins)

            self.__CV_TRAIN_FILES__ = train_files
            self.__CV_VALID_FILES__ = valid_files
            self.__CV_TEST_FILES__ = test_files

    def summary(self, additional_vars=None):
        """
        Print config summary
        :param additional_vars: additional list of variables to print
        :return:
        """

        summary_vals = ["MODEL_NAME", "TAG", ""]
        summary_vals.extend(
            [
                "TRAIN_DATASET",
                "VAL_DATASET",
                "TEST_DATASET",
                "",
                "CATEGORIES",
                "",
                "IMG_SIZE",
                "",
                "N_EPOCHS",
                "AUGMENT_DATA",
                "LOSS",
                "CLASS_WEIGHTS",
                "ROBUST_LOSS_NAME"
                if self.ROBUST_LOSS_NAME
                else "" "ROBUST_LOSS_STEP_SIZE"
                if self.ROBUST_LOSS_NAME
                else "" "",
                "USE_CROSS_VALIDATION",
                "CV_K" if self.USE_CROSS_VALIDATION else "",
                "CV_FILE" if self.USE_CROSS_VALIDATION else "",
                "CV_TRAIN_BINS" if self.USE_CROSS_VALIDATION else "",
                "CV_VALID_BINS" if self.USE_CROSS_VALIDATION else "",
                "CV_TEST_BINS" if self.USE_CROSS_VALIDATION else "",
                "" "TRAIN_BATCH_SIZE",
                "VALID_BATCH_SIZE",
                "TEST_BATCH_SIZE",
                "",
                "NUM_GRAD_STEPS",
                "",
                "INITIAL_LEARNING_RATE",
                "LR_SCHEDULER_NAME",
                "DROP_FACTOR" if self.LR_SCHEDULER_NAME else "",
                "DROP_RATE" if self.LR_SCHEDULER_NAME else "",
                "MIN_LEARNING_RATE" if self.LR_SCHEDULER_NAME else "",
                "LR_MIN_DELTA" if self.LR_SCHEDULER_NAME else "",
                "LR_PATIENCE" if self.LR_SCHEDULER_NAME else "",
                "LR_COOLDOWN" if self.LR_SCHEDULER_NAME else "",
                "" "USE_EARLY_STOPPING",
                "EARLY_STOPPING_MIN_DELTA" if self.USE_EARLY_STOPPING else "",
                "EARLY_STOPPING_PATIENCE" if self.USE_EARLY_STOPPING else "",
                "EARLY_STOPPING_CRITERION" if self.USE_EARLY_STOPPING else "",
                "",
                "KERNEL_INITIALIZER",
                "SEED" if self.SEED else "",
                "" "INIT_WEIGHTS",
                "",
                "TEST_WEIGHT_PATH",
                "TEST_METRICS",
                "" "NUM_WORKERS",
                "OUTPUT_DIR",
                "",
            ]
        )
        if additional_vars:
            summary_vals.extend(additional_vars)

        # Remove consecutive elements in summary vals that are the same
        summary_vals = [x[0] for x in groupby(summary_vals)]

        logger.info("")
        logger.info("==" * 40)
        logger.info("Config Summary")
        logger.info("==" * 40)

        for attr in summary_vals:
            if attr == "":
                logger.info("")
                continue
            logger.info(attr + ": " + str(self.__getattribute__(attr)))

        logger.info("==" * 40)
        logger.info("")

    def get_num_classes(self):
        if self.INCLUDE_BACKGROUND:
            return len(self.CATEGORIES) + 1

        return len(self.CATEGORIES)

    def num_neighboring_slices(self):
        return None

    @property
    def testing(self):
        return self.STATE == "testing"

    @property
    def training(self):
        return self.STATE == "training"


class DeeplabV3Config(Config):
    """
    Configuration for 2D Deeplabv3+ architecture
    (https://arxiv.org/abs/1802.02611).
    """

    MODEL_NAME = "DeeplabV3Plus"

    OS = 16
    DIL_RATES = (2, 4, 6)
    AT_DIVISOR = 2
    DROPOUT_RATE = 0.1

    def __init__(self, state="training", create_dirs=True):
        super().__init__(self.MODEL_NAME, state, create_dirs=create_dirs)

    def summary(self, additional_vars=None):
        summary_attrs = ["OS", "DIL_RATES", "DROPOUT_RATE"]
        super().summary(summary_attrs)


class SegnetConfig(Config):
    """
    Configuration for 2D Segnet architecture (https://arxiv.org/abs/1505.07293)
    """

    MODEL_NAME = "segnet_2d"

    TRAIN_BATCH_SIZE = 15

    DEPTH = 6
    NUM_CONV_LAYERS = [2, 2, 3, 3, 3, 3]
    NUM_FILTERS = [64, 128, 256, 256, 512, 512]

    SINGLE_BN = False
    CONV_ACT_BN = False
    USE_BOTTLENECK = False

    INITIAL_LEARNING_RATE = 1e-3

    def __init__(self, state="training", create_dirs=True):
        super().__init__(self.MODEL_NAME, state, create_dirs=create_dirs)

    def summary(self, additional_vars=None):
        summary_attrs = ["DEPTH", "NUM_CONV_LAYERS", "NUM_FILTERS"]
        super().summary(summary_attrs)


class UNetConfig(Config):
    """
    Configuration for 2D U-Net architecture (https://arxiv.org/abs/1505.04597)
    """

    MODEL_NAME = "UNet2D"

    INIT_UNET_2D = False

    INITIAL_LEARNING_RATE = 2e-2
    DROP_FACTOR = 0.8 ** (1 / 5)
    DROP_RATE = 1.0
    TRAIN_BATCH_SIZE = 35

    DEPTH = 6
    NUM_FILTERS = None

    # Boolean arguments for attention and deep supervision
    USE_ATTENTION = False
    USE_DEEP_SUPERVISION = False

    def __init__(self, state="training", create_dirs=True):
        super().__init__(self.MODEL_NAME, state, create_dirs=create_dirs)

    def summary(self, additional_vars=None):
        summary_vars = ["DEPTH", "NUM_FILTERS", ""]
        if additional_vars:
            summary_vars.extend(additional_vars)
        super().summary(summary_vars)


class ResidualUNet(Config):
    """
    Configuration for 2D Residual U-Net architecture
    """

    MODEL_NAME = "res_unet"

    DEPTH = 6
    NUM_FILTERS = None

    DROPOUT_RATE = 0.0
    LAYER_ORDER = ["relu", "bn", "dropout", "conv"]

    USE_SE_BLOCK = False
    SE_RATIO = 8

    def __init__(self, state="training", create_dirs=True):
        super().__init__(self.MODEL_NAME, state, create_dirs=create_dirs)

    def summary(self, additional_vars=None):
        summary_attrs = [
            "DEPTH",
            "NUM_FILTERS",
            "DROPOUT_RATE",
            "",
            "LAYER_ORDER",
            "",
            "USE_SE_BLOCK",
            "SE_RATIO",
        ]
        super().summary(summary_attrs)

    def num_neighboring_slices(self):
        return self.IMG_SIZE[-1] if self.IMG_SIZE[-1] != 1 else None


class UNet2_5DConfig(UNetConfig):
    """
    Configuration for 3D U-Net architecture
    """

    MODEL_NAME = "UNet2D"

    IMG_SIZE = (288, 288, 7)

    N_EPOCHS = 20
    AUGMENT_DATA = False
    INITIAL_LEARNING_RATE = 1e-2

    DROP_RATE = 1.0
    DROP_FACTOR = 0.8

    def __init__(self, state="training", create_dirs=True):
        warnings.warn("UNet2_5DConfig is deprecated. Use UNet2DConfig instead.", DeprecationWarning)
        super().__init__(state, create_dirs)

    def num_neighboring_slices(self):
        return self.IMG_SIZE[2]


class UNet3DConfig(UNetConfig):
    MODEL_NAME = "UNet3D"

    IMG_SIZE = (288, 288, 4, 1)

    N_EPOCHS = 20
    INITIAL_LEARNING_RATE = 1e-2

    DROP_RATE = 1.0
    DROP_FACTOR = 0.8

    TAG = "oai_3d"

    SLICE_SUBSET = None  # 1 indexed inclusive - i.e. (5, 64) means slices [5, 64]

    NUM_FILTERS = [32, 64, 128, 256, 512, 1024]

    def num_neighboring_slices(self):
        return self.IMG_SIZE[2]

    def summary(self, additional_vars=None):
        summary_attrs = ["SLICE_SUBSET"]
        super().summary(summary_attrs)


class DeeplabV3_2_5DConfig(DeeplabV3Config):
    """2.5D DeeplabV3+."""

    IMG_SIZE = (288, 288, 3)

    def __init__(self, state="training", create_dirs=True):
        warnings.warn("UNet2_5DConfig is deprecated. Use UNet2DConfig instead.", DeprecationWarning)
        super().__init__(state, create_dirs)

    def num_neighboring_slices(self):
        return self.IMG_SIZE[2]


class AnisotropicUNetConfig(Config):
    """2D Anisotropic U-Net."""

    MODEL_NAME = "anisotropic_unet"

    IMG_SIZE = (288, 72, 1)

    INITIAL_LEARNING_RATE = 2e-2
    DROP_FACTOR = 0.85
    DROP_RATE = 1.0
    TRAIN_BATCH_SIZE = 60

    DEPTH = 6
    NUM_FILTERS = None

    KERNEL_SIZE = (7, 3)

    def __init__(self, state="training", create_dirs=True):
        super().__init__(self.MODEL_NAME, state, create_dirs=create_dirs)

    def summary(self, additional_vars=None):
        summary_attrs = ["DEPTH", "NUM_FILTERS", "KERNEL_SIZE"]
        super().summary(summary_attrs)


class RefineNetConfig(Config):
    """Configuration for RefineNet architecture as suggested by paper below
    http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi
    -Path_Refinement_CVPR_2017_paper.pdf
    """

    MODEL_NAME = "refinenet"

    INITIAL_LEARNING_RATE = 1e-3

    def __init__(self, state="training", create_dirs=True):
        super().__init__(self.MODEL_NAME, state, create_dirs=create_dirs)


class FCDenseNetConfig(Config):
    MODEL_NAME = "FCDenseNet"

    DEPTH = 5
    NUM_LAYERS = (5,)
    NUM_FILTERS = (12,)
    NUM_FILTERS_HEAD_CONV = 48
    DROPOUT = 0.2

    def __init__(self, state="training", create_dirs=True):
        super().__init__(self.MODEL_NAME, state, create_dirs=create_dirs)


def _check_and_coerce_cfg_value_type(replacement, original, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # TODO: Convert all to have non-None values by default.
    if isinstance(original, type(None)):
        return replacement

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(original_type, replacement_type, original, replacement, full_key)
    )


def _assert_with_logging(cond, msg):
    if not cond:
        logger.debug(msg)
    assert cond, msg


def _error_with_logging(cond, msg, error_type=ValueError):
    if not cond:
        logger.error(msg)
        raise error_type(msg)


SUPPORTED_CONFIGS = [
    UNetConfig,
    SegnetConfig,
    DeeplabV3Config,
    ResidualUNet,
    AnisotropicUNetConfig,
    RefineNetConfig,
    UNet3DConfig,
    UNet2_5DConfig,
    DeeplabV3_2_5DConfig,
    FCDenseNetConfig,
]


def get_config(config_cp_save_tag: str, create_dirs: bool = True):
    """Get config using config_cp_save_tag

    Args:
        config_cp_save_tag: config cp_save_tag
        create_dirs: if directory should be created

    Return:
        Config: A config.
    """
    from medsegpy.modeling.meta_arch.build import LEGACY_MODEL_NAMES

    if config_cp_save_tag in LEGACY_MODEL_NAMES:
        config_cp_save_tag = LEGACY_MODEL_NAMES[config_cp_save_tag]

    configs = SUPPORTED_CONFIGS
    for config in configs:
        if config.MODEL_NAME == config_cp_save_tag:
            c = config(create_dirs=create_dirs)
            return c

    raise ValueError("config %s not found" % config_cp_save_tag)


def get_model_name(cfg_filename: str):
    """Get "MODEL_NAME" attribute from config file.

    Args:
        cfg_filename: filepath to INI or YAML file where config is stored

    Returns:
        str: The model name.
    """
    vars_dict = Config.parse_base_config(cfg_filename)
    return vars_dict["MODEL_NAME"] if "MODEL_NAME" in vars_dict else vars_dict["CP_SAVE_TAG"]
