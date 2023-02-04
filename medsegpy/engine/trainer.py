import copy
import logging
import os
import pickle
import warnings
from typing import Tuple

import tensorflow as tf
from keras import callbacks as kc
from keras.utils import plot_model

from medsegpy import config, solver
from medsegpy.data import build_loader, im_gens
from medsegpy.engine.callbacks import LossHistory, WandBLogger, lr_callback
from medsegpy.evaluation import build_evaluator, inference_on_dataset
from medsegpy.losses import build_loss, dice_loss
from medsegpy.modeling.meta_arch import build_model
from medsegpy.utils import dl_utils, env, io_utils
from medsegpy.modeling.ssl_utils import (
    SelfSupervisedInfo,
    load_specific_weights
)

try:
    _SUPPORTS_DISTRIBUTED = True
    from tensorflow.distribute import MirroredStrategy
except ModuleNotFoundError:
    _SUPPORTS_DISTRIBUTED = False
    MirroredStrategy = None

logger = logging.getLogger(__name__)


class DefaultTrainer(object):
    """Default trainer for medical semantic segmentation."""

    def __init__(self, cfg: config.Config, run_eagerly=None, strategy=None):
        """
        Args:
            cfg (Config): An experiment config.
            run_eagerly (bool, optional): If `True`, runs eagerly.
                Only available in tensorflow>=2.0.
            strategy (tf.distribute.Strategy, optional): The strategy to use for training.
                Only available if `tf.distribute` package is available (tensorflow>=1.14).
        """
        self._cfg = cfg
        self._loss_history = None
        self._default_strategy = (
            tf.distribute.get_strategy() if _SUPPORTS_DISTRIBUTED else dl_utils.NoOpStrategy()
        )
        num_gpus = dl_utils.num_gpus()

        if not env.is_tf2() and run_eagerly is not None:
            warnings.warn("`run_eagerly` can only be specified in Tensorflow >2.0. " "Ignoring...")
            run_eagerly = None
        self._run_eagerly = run_eagerly

        if strategy is None:
            strategy = self._default_strategy
            if _SUPPORTS_DISTRIBUTED and num_gpus > 1:
                logger.info("Running multi gpu model")
                strategy = MirroredStrategy()
        self.strategy = strategy

        with self.strategy.scope():
            # Prepare for self-supervised learning, if needed
            SelfSupervisedInfo.clear()
            SelfSupervisedInfo.init_self_supervised(cfg)

            model = self.build_model(cfg)
            if cfg.PRETRAINED_WEIGHTS_PATH or cfg.INIT_WEIGHTS:
                self._init_model(model)

        plot_model(model, to_file=os.path.join(cfg.OUTPUT_DIR, "model.png"), show_shapes=True)
        model.summary(line_length=120, print_fn=lambda x: logger.info(x))
        model_json = model.to_json()
        model_json_save_path = os.path.join(cfg.OUTPUT_DIR, "model.json")
        with open(model_json_save_path, "w") as json_file:
            json_file.write(model_json)

        # Replicate model on multiple gpus when tensorflow.distribute module not available.
        # Note this does not solve issue of having too large of a model
        if not _SUPPORTS_DISTRIBUTED and num_gpus > 1:
            logger.info("Running multi gpu model")
            model = dl_utils.ModelMGPU(model, gpus=num_gpus)

        self._train_loader, self._val_loader = self._build_data_loaders(cfg)
        self._model = model

    def train(self):
        """Train model specified by config.

        Do not call this under a strategy scope. Instead, set `self.strategy`.
        """
        cfg = self._cfg

        with self.strategy.scope():
            self._train_model()
            # After training, remove preloaded data
            self._train_loader.clear_cached_data()
            self._val_loader.clear_cached_data()

        if cfg.TEST_DATASET:
            # Specialized strategies are not currently supported for testing.
            if _SUPPORTS_DISTRIBUTED and not isinstance(
                self.strategy, (dl_utils.NoOpStrategy, type(self._default_strategy))
            ):
                logger.error(
                    f"Strategy '{type(self.strategy).__name__}' not currently "
                    f"supported for testing. "
                    f"Please run testing separately on a single gpu."
                )
                return {}
            return self.test(cfg, self._model)
        else:
            return {}

    def _init_model(self, model):
        """Initialize model with weights and apply any freezing necessary."""
        cfg = self._cfg
        if cfg.PRETRAINED_WEIGHTS_PATH:
            load_specific_weights(model,
                                  cfg,
                                  debug=True)
        else:
            if os.path.isdir(cfg.INIT_WEIGHTS):
                weight_file = dl_utils.get_weights(cfg.INIT_WEIGHTS)
            else:
                weight_file = cfg.INIT_WEIGHTS
            logger.info("Loading weights from {}".format(weight_file))
            model.load_weights(weight_file)
        frozen_layers = cfg.FREEZE_LAYERS
        if frozen_layers:
            fl = range(frozen_layers[0], frozen_layers[1])
            logger.info("Freezing layers [{}, {})".format(fl.start, fl.stop))
            for i in fl:
                model.layers[i].trainable = False

    def build_callbacks(self):
        cfg = self._cfg
        output_dir = cfg.OUTPUT_DIR
        callbacks = []

        if cfg.LR_SCHEDULER_NAME:
            callbacks.append(solver.build_lr_scheduler(cfg))
        if cfg.USE_EARLY_STOPPING:
            callbacks.append(
                kc.EarlyStopping(
                    monitor=cfg.EARLY_STOPPING_CRITERION,
                    min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
                    patience=cfg.EARLY_STOPPING_PATIENCE,
                )
            )

        self._loss_history = LossHistory()

        tb_kwargs = dict(update_freq="batch") if env.is_tf2() else {}
        callbacks.extend(
            [
                kc.ModelCheckpoint(
                    os.path.join(output_dir, "weights.{epoch:03d}-{val_loss:.4f}.h5"),
                    save_best_only=True,
                    save_weights_only=True,
                ),
                kc.TensorBoard(output_dir, write_grads=False, write_images=False, **tb_kwargs),
                WandBLogger() if env.supports_wandb() else None,
                kc.CSVLogger(os.path.join(output_dir, "metrics.log")),
                self._loss_history,
            ]
        )
        callbacks = [x for x in callbacks if x is not None]

        return callbacks

    def build_loss(self):
        """Builds loss function used with ``model.compile(loss=...)``.
        """
        return build_loss(self._cfg)

    def _train_model(self):
        """Train model.

        If multi-gpu training and distributed training is supported (tensorflow>=1.15),
        call this function with the appropriate strategy scope::

            with self.strategy.scope():
                self._train_model()
        """
        cfg = self._cfg
        n_epochs = cfg.N_EPOCHS
        num_workers = cfg.NUM_WORKERS
        output_dir = cfg.OUTPUT_DIR

        model = self._model

        # TODO: Add more options for metrics.
        optimizer = solver.build_optimizer(cfg)
        loss_func = self.build_loss()
        metrics = [lr_callback(optimizer), dice_loss]

        callbacks = self.build_callbacks()
        if isinstance(loss_func, kc.Callback):
            callbacks.insert(0, loss_func)
            metrics.append(loss_func.criterion)

        model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
        if env.is_tf2():
            run_eagerly = tf.executing_eagerly() if self._run_eagerly is None else self._run_eagerly
            model.run_eagerly = run_eagerly

        train_loader, val_loader = self._train_loader, self._val_loader
        use_multiprocessing = num_workers > 1

        # Start training
        model.fit_generator(
            train_loader,
            epochs=n_epochs,
            validation_data=val_loader,
            callbacks=callbacks,
            workers=num_workers,
            use_multiprocessing=use_multiprocessing,
            verbose=1,
            shuffle=False,
        )

        # Save optimizer state
        io_utils.save_optimizer(model.optimizer, output_dir)

        # Save files to write as output
        # TODO: refactor to save dataframe.
        hist_cb = self._loss_history
        data = [hist_cb.epoch, hist_cb.losses, hist_cb.val_losses]
        pik_data_path = os.path.join(output_dir, "pik_data.dat")
        with open(pik_data_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def test(cls, cfg: config.Config, model):
        logger.info("Beginning testing...")
        cfg = copy.deepcopy(cfg)  # will be modified below.
        cfg.change_to_test()

        weights = cfg.TEST_WEIGHT_PATH
        if not cfg.TEST_WEIGHT_PATH:
            weights = dl_utils.get_weights(cfg.OUTPUT_DIR)
            logger.info("Best weights: {}".format(weights))
        model.load_weights(weights)

        test_dataset = cfg.TEST_DATASET
        test_gen = cls.build_test_data_loader(cfg)
        evaluator = build_evaluator(test_dataset, cfg, save_raw_data=False)
        test_results = inference_on_dataset(model, test_gen, evaluator)
        # After testing, remove pre-loaded data
        test_gen.clear_cached_data()
        return test_results

    @classmethod
    def build_model(cls, cfg):
        try:
            return build_model(cfg)
        except KeyError:
            # TODO (TF2.X)
            if env.is_tf2():
                raise ValueError(
                    "`get_model` not currently supported for tf2. "
                    "We are working on backwards compatibility"
                )
            from medsegpy.modeling import get_model

            return get_model(cfg)

    def _build_data_loaders(self, cfg) -> Tuple[im_gens.Generator, im_gens.Generator]:
        """Builds train and val data loaders."""
        train_loader = build_loader(
            cfg,
            dataset_names=cfg.TRAIN_DATASET,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            drop_last=True,
            is_test=False,
            shuffle=True,
        )
        val_loader = build_loader(
            cfg,
            dataset_names=cfg.VAL_DATASET,
            batch_size=cfg.VALID_BATCH_SIZE,
            drop_last=True,
            is_test=False,
            shuffle=False,
        )
        return train_loader, val_loader

    @classmethod
    def build_test_data_loader(cls, cfg):
        return build_loader(
            cfg,
            dataset_names=cfg.TEST_DATASET,
            batch_size=cfg.TEST_BATCH_SIZE,
            drop_last=False,
            is_test=True,
            shuffle=False,
        )
