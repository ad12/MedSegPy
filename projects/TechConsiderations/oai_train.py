import logging
import os
import pickle

import keras.callbacks as kc
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard as tfb
from keras.optimizers import Adam
from keras.utils import plot_model

import medsegpy.utils.dl_utils
from medsegpy import glob_constants
from medsegpy.data import im_gens
from medsegpy.modeling import get_model
from medsegpy.modeling.losses import (
    WEIGHTED_CROSS_ENTROPY_LOSS,
    dice_loss,
    focal_loss,
    get_training_loss,
)
from medsegpy.utils import dl_utils, io_utils
from medsegpy.utils.logger import setup_logger

logger = logging.getLogger(__name__)

CLASS_WEIGHTS = np.asarray([100, 1])
SAVE_BEST_WEIGHTS = True
FREEZE_LAYERS = None


def train_model(config, optimizer=None, model=None, class_weights=None):
    """
    Train model
    :param config: a Config object
    :param optimizer: a Keras optimizer (default = None)
    """
    raise DeprecationWarning(
        "oai_train.train_model is deprecated. "
        "Use nn_train.DefaultTrainer._train_model() instead."
    )

    # Load data from config
    output_dir = config.OUTPUT_DIR
    cp_save_tag = config.CP_SAVE_TAG
    n_epochs = config.N_EPOCHS
    pik_save_path = config.PIK_SAVE_PATH
    loss = config.LOSS
    num_workers = config.NUM_WORKERS

    # Initialize logger.
    setup_logger(output_dir)
    logger.info("OUTPUT_DIR: {}".format(output_dir))

    # Initialize global params.
    glob_constants.SEED = config.SEED

    if model is None:
        model = get_model(config)

    # plot model
    plot_model(
        model, to_file=os.path.join(output_dir, "model.png"), show_shapes=True
    )

    # Fine tune - initialize with weights
    if config.FINE_TUNE:
        logger.info("loading weights")
        model.load_weights(config.INIT_WEIGHT_PATH)
        if FREEZE_LAYERS:
            if len(FREEZE_LAYERS) == 1:
                fl = range(FREEZE_LAYERS[0], len(model.layers))
            else:
                fl = range(FREEZE_LAYERS[0], FREEZE_LAYERS[1])
            logger.info("freezing layers %s" % fl)
            for i in fl:
                model.layers[i].trainable = False

    # Replicate model on multiple gpus - note this does not solve issue of having too large of a model
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if num_gpus > 1:
        logger.info("Running multi gpu model")
        model = medsegpy.utils.dl_utils.ModelMGPU(model, gpus=num_gpus)

    # If no optimizer is provided, default to Adam
    if optimizer is None:
        optimizer = Adam(
            lr=config.INITIAL_LEARNING_RATE,
            beta_1=0.99,
            beta_2=0.995,
            epsilon=1e-8,
            decay=config.ADAM_DECAY,
            amsgrad=config.USE_AMSGRAD,
        )

    # Load loss function
    # if weighted cross entropy, load weights
    if loss == WEIGHTED_CROSS_ENTROPY_LOSS and class_weights is None:
        # logger.info('calculating freq')
        # freq_file = CLASS_FREQ_DAT_WEIGHTS_AUG if config.AUGMENT_DATA else CLASS_FREQ_DAT_WEIGHTS_NO_AUG
        # logger.info('Weighting with file: %s' % freq_file)
        # class_freqs = utils.load_pik(freq_file)
        # class_weights = get_class_weights(class_freqs)
        # class_weights = np.reshape(class_weights, (1, 2))
        logger.info(class_weights)

    loss_func = get_training_loss(loss, weights=class_weights)
    lr_metric = get_lr_metric(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=[lr_metric, dice_loss, focal_loss()],
    )

    if config.FINE_TUNE and FREEZE_LAYERS:
        model.summary()

    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format("channels_last")

    # model callbacks
    cp_cb = ModelCheckpoint(
        os.path.join(
            output_dir, cp_save_tag + "_weights.{epoch:03d}-{val_loss:.4f}.h5"
        ),
        save_best_only=SAVE_BEST_WEIGHTS,
    )
    tfb_cb = tfb(config.TF_LOG_DIR, write_grads=False, write_images=False)
    hist_cb = LossHistory()

    callbacks_list = [tfb_cb, cp_cb, hist_cb]

    # Step decay for learning rate
    if config.USE_STEP_DECAY:
        lr_cb = lrs(
            step_decay_wrapper(
                config.INITIAL_LEARNING_RATE,
                config.MIN_LEARNING_RATE,
                config.DROP_FACTOR,
                config.DROP_RATE,
            )
        )
        callbacks_list.append(lr_cb)

    # use early stopping
    if config.USE_EARLY_STOPPING:
        es_cb = EarlyStopping(
            monitor=config.EARLY_STOPPING_CRITERION,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            patience=config.EARLY_STOPPING_PATIENCE,
        )
        callbacks_list.append(es_cb)

    generator = im_gens.get_generator(config)
    generator.summary()

    train_nbatches, valid_nbatches = generator.num_steps()

    train_gen = generator.img_generator(state=im_gens.GeneratorState.TRAINING)
    val_gen = generator.img_generator(state=im_gens.GeneratorState.VALIDATION)

    # Start training
    model.fit_generator(
        train_gen,
        train_nbatches,
        epochs=n_epochs,
        validation_data=val_gen,
        validation_steps=valid_nbatches,
        callbacks=callbacks_list,
        workers=num_workers,
        use_multiprocessing=num_workers > 1,
        max_queue_size=train_nbatches,
        verbose=1,
    )

    # Save optimizer state
    io_utils.save_optimizer(model.optimizer, config.CP_SAVE_PATH)

    # Save files to write as output
    data = [hist_cb.epoch, hist_cb.losses, hist_cb.val_losses]
    with open(pik_save_path, "wb") as f:
        pickle.dump(data, f)

    model_json = model.to_json()
    model_json_save_path = os.path.join(config.CP_SAVE_PATH, "model.json")
    with open(model_json_save_path, "w") as json_file:
        json_file.write(model_json)

    # # Save model
    # model.save(filepath=os.path.join(config.CP_SAVE_PATH, 'model.h5'), overwrite=True)


def get_class_weights(freqs):
    # weight by median and scale to 1
    weights = np.median(freqs) / freqs
    weights = weights / np.min(weights)

    return weights


def get_lr_metric(optimizer):
    """
    Wrapper for learning rate tensorflow metric
    :param optimizer: a Keras optimizer
    :return: a Tensorflow callback
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def step_decay_wrapper(
    initial_lr=1e-4, min_lr=1e-8, drop_factor=0.8, drop_rate=1.0
):
    """
    Wrapper for learning rate step decay
    :param initial_lr: initial learning rate (default = 1e-4)
    :param min_lr: minimum learning rate (default = None)
    :param drop_factor: factor to drop (default = 0.8)
    :param drop_rate: rate of learning rate drop (default = 1.0 epochs)
    :return: a Tensorflow callback
    """
    initial_lr = initial_lr
    drop_factor = drop_factor
    drop_rate = drop_rate
    min_lr = min_lr

    def step_decay(epoch):
        import math

        lrate = initial_lr * math.pow(
            drop_factor, math.floor((1 + epoch) / drop_rate)
        )

        if lrate < min_lr:
            lrate = min_lr

        return lrate

    return step_decay


class LossHistory(kc.Callback):
    """
    A Keras callback to log training history
    """

    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.losses = []
        # self.lr = []
        self.epoch = []

    def on_epoch_end(self, batch, logs={}):
        self.val_losses.append(logs.get("val_loss"))
        self.losses.append(logs.get("loss"))
        # self.lr.append(step_decay(len(self.losses)))
        self.epoch.append(len(self.losses))


def fine_tune(dirpath, config, vals_dict=None, class_weights=None):
    # # If a fine-tune directory already exits, skip this directory
    # if (os.path.isdir(os.path.join(dirpath, 'fine_tune'))):
    #     logger.info('Skipping %s - fine_tune folder exists' % dirpath)

    # Initialize for fine tuning
    config.merge_from_file(os.path.join(dirpath, "config.ini"))

    # Get best weight path
    best_weight_path = dl_utils.get_weights(dirpath)
    logger.info("Best weight path: %s" % best_weight_path)

    config.init_fine_tune(best_weight_path)

    # only load command line arguments that are not the default
    temp_config = type(config)(create_dirs=False)
    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            val_default = getattr(temp_config, key)
            if val != val_default:
                config.set_attr(key, val)

    config.save_config()
    config.summary()

    train_model(config, class_weights=class_weights)

    K.clear_session()


def train(config, vals_dict=None, class_weights=CLASS_WEIGHTS):
    """
    Train config after applying vals_dict
    :param config: a Config object
    :param vals_dict: a dictionary of config parameters to change (default = None)
                      e.g. {'INITIAL_LEARNING_RATE': 1e-6, 'USE_STEP_DECAY': True}
    """

    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.save_config()
    config.summary()

    train_model(config, class_weights=class_weights)

    K.clear_session()


EXP_DIR_MAP = {
    "arch": "architecture_limited",
    "aug": "augment_limited",
    "best": "best_network",
    "data": "data_limit",
    "loss": "loss_limit",
    "vol": "volume_limited",
    "control": "control_exps",
}

if __name__ == "__main__":
    raise DeprecationWarning("This file is deprecated. Use nn_train")

    base_parser = argparse.ArgumentParser(description="Train OAI dataset")
    arg_subparser = base_parser.add_subparsers(
        help="supported configs for different architectures", dest="config"
    )
    subparsers = MCONFIG.init_cmd_line_parser(arg_subparser)

    for s_parser in subparsers:
        s_parser.add_argument(
            "-g",
            "--gpu",
            metavar="G",
            type=str,
            nargs="?",
            default="0",
            help="gpu id to use. default=0",
        )
        s_parser.add_argument(
            "-s",
            "--seed",
            metavar="S",
            type=int,
            nargs="?",
            default=None,
            help="python seed to initialize filter weights. default=None",
        )
        s_parser.add_argument(
            "-k",
            "--k_fold_cross_validation",
            metavar="K",
            default=None,
            nargs="?",
            help="Use k-fold cross-validation for training. Argument is k (int) or filepath (str)",
        )
        s_parser.add_argument(
            "--ho_test",
            metavar="T",
            type=int,
            default=1,
            nargs="?",
            help="Number of hold-out test bins",
        )
        s_parser.add_argument(
            "--ho_valid",
            metavar="V",
            type=int,
            default=1,
            nargs="?",
            help="Number of hold-out validation bins",
        )
        s_parser.add_argument(
            "--class_weights",
            type=tuple,
            nargs="?",
            default=CLASS_WEIGHTS,
            help="weight classes in order",
        )
        s_parser.add_argument(
            "--experiment",
            type=str,
            nargs="?",
            default="",
            help="experiment to run",
        )
        s_parser.add_argument(
            "--fine_tune_path",
            type=str,
            default="",
            nargs="?",
            help="directory to fine tune.",
        )
        s_parser.add_argument(
            "--freeze_layers",
            type=str,
            default=None,
            nargs="?",
            help="range of layers to freeze. eg. `(0,100)`, `(5, 45)`, `(5,)`",
        )

        s_parser.add_argument(
            "--save_all_weights",
            default=False,
            action="store_const",
            const=True,
            help="store weights for each epoch. Default: False",
        )

        # add support for specifying tissues
        mri_utils.init_cmd_line(s_parser)

    # Parse input arguments
    args = base_parser.parse_args()
    vargin = vars(args)

    experiment_type = args.experiment
    fine_tune_dirpath = args.fine_tune_path

    if not fine_tune_dirpath and not experiment_type:
        raise ValueError("--experiment must be specified if not fine-tuning")

    experiment_filepath = (
        EXP_DIR_MAP[experiment_type]
        if experiment_type in EXP_DIR_MAP.keys()
        else experiment_type
    )
    MCONFIG.SAVE_PATH_PREFIX = os.path.join(
        "/bmrNAS/people/arjun/msk_seg_networks", experiment_filepath
    )

    # MCONFIG.SAVE_PATH_PREFIX = os.path.join('./sample_data/cmd_line', experiment_filepath)

    gpu = args.gpu
    glob_constants.SEED = args.seed
    k_fold_cross_validation = args.k_fold_cross_validation

    logger.info(glob_constants.SEED)

    logger.info("Using GPU %s" % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    SAVE_BEST_WEIGHTS = not args.save_all_weights

    c = MCONFIG.get_config(
        config_cp_save_tag=vargin["config"], create_dirs=not fine_tune_dirpath
    )
    config_dict = c.parse_cmd_line(vargin)

    # parse tissues
    config_dict["CATEGORIES"] = mri_utils.parse_tissues(vargin)

    if fine_tune_dirpath:
        # parse freeze layers
        freeze_layer = vargin["freeze_layers"]
        FREEZE_LAYERS = (
            utils.convert_data_type(vargin["freeze_layers"], tuple)
            if freeze_layer
            else None
        )
        fine_tune(fine_tune_dirpath, c, config_dict)
        exit(0)

    if k_fold_cross_validation:
        if k_fold_cross_validation.isdigit():
            k_fold_cross_validation = int(k_fold_cross_validation)

        ho_valid = args.ho_valid
        ho_test = args.ho_test

        # Initialize CrossValidation wrapper
        cv_wrapper = cv_util.CrossValidationProcessor(
            k_fold_cross_validation,
            num_valid_bins=ho_valid,
            num_test_bins=ho_test,
        )

        logger.info(
            "Loading %d-fold cross-validation data from %s..."
            % (cv_wrapper.k, cv_wrapper.filepath)
        )

        cv_file = cv_wrapper.filepath
        cv_k = cv_wrapper.k

        cv_exp_id = 1

        base_save_path = c.CP_SAVE_PATH
        for (
            tr_f,
            val_f,
            test_f,
            tr_bins,
            val_bins,
            test_bins,
        ) in cv_wrapper.run():
            c.init_cross_validation(
                train_files=tr_f,
                valid_files=val_f,
                test_files=test_f,
                train_bins=tr_bins,
                valid_bins=val_bins,
                test_bins=test_bins,
                cv_k=cv_k,
                cv_file=cv_file,
                output_dir=os.path.join(
                    base_save_path, "cv-exp-%03d" % cv_exp_id
                ),
            )
            cv_exp_id += 1

            train(c, config_dict)
    else:
        train(c, config_dict)
