import copy
import logging
import os
import pickle
from typing import Tuple

from keras import callbacks as kc
from keras.utils import plot_model

from medsegpy import config, solver
from medsegpy.data import build_loader, im_gens
from medsegpy.engine.callbacks import LossHistory, lr_callback
from medsegpy.evaluation import build_evaluator, inference_on_dataset
from medsegpy.losses import dice_loss, get_training_loss
from medsegpy.modeling import get_model
from medsegpy.modeling.meta_arch import build_model
from medsegpy.utils import dl_utils, io_utils

logger = logging.getLogger(__name__)


class DefaultTrainer(object):
    """Default trainer for medical semantic segmentation."""

    def __init__(self, cfg: config.Config):
        self._cfg = cfg
        self._loss_history = None

        model = get_model(cfg)
        plot_model(
            model,
            to_file=os.path.join(cfg.OUTPUT_DIR, "model.png"),
            show_shapes=True,
        )
        model.summary(print_fn=lambda x: logger.info(x))
        model_json = model.to_json()
        model_json_save_path = os.path.join(cfg.OUTPUT_DIR, "model.json")
        with open(model_json_save_path, "w") as json_file:
            json_file.write(model_json)

        if cfg.INIT_WEIGHTS:
            self._init_model(model)

        # Replicate model on multiple gpus.
        # Note this does not solve issue of having too large of a model
        num_gpus = dl_utils.num_gpus()
        if num_gpus > 1:
            logger.info("Running multi gpu model")
            model = dl_utils.ModelMGPU(model, gpus=num_gpus)

        self._train_loader, self._val_loader = self._build_data_loaders(cfg)
        self._model = model

    def train(self):
        """Train model specified by config.
        """
        cfg = self._cfg

        self._train_model()

        if cfg.TEST_DATASET:
            return self.test(cfg, self._model)
        else:
            return {}

    def _init_model(self, model):
        """Initialize model with weights and apply any freezing necessary."""
        cfg = self._cfg
        logger.info("Loading weights from {}".format(cfg.INIT_WEIGHTS))
        model.load_weights(cfg.INIT_WEIGHTS)
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

        callbacks.extend(
            [
                kc.ModelCheckpoint(
                    os.path.join(
                        output_dir, "weights.{epoch:03d}-{val_loss:.4f}.h5"
                    ),
                    save_best_only=True,
                    save_weights_only=True,
                ),
                kc.TensorBoard(
                    output_dir, write_grads=False, write_images=False
                ),
                kc.CSVLogger(os.path.join(output_dir, "metrics.log")),
                self._loss_history,
            ]
        )

        return callbacks

    def _train_model(self):
        """Train model."""
        cfg = self._cfg
        n_epochs = cfg.N_EPOCHS
        loss = cfg.LOSS
        class_weights = cfg.CLASS_WEIGHTS
        num_workers = cfg.NUM_WORKERS
        output_dir = cfg.OUTPUT_DIR

        model = self._model

        # TODO: Add more options for metrics.
        optimizer = solver.build_optimizer(cfg)
        loss_func = get_training_loss(loss, weights=class_weights)
        model.compile(
            optimizer=optimizer,
            loss=loss_func,
            metrics=[lr_callback(optimizer), dice_loss],
        )
        callbacks = self.build_callbacks()

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
        evaluator = build_evaluator(test_dataset, cfg, save_raw_data=True)
        return inference_on_dataset(model, test_gen, evaluator)

    @classmethod
    def build_model(cls, cfg):
        try:
            return build_model(cfg)
        except KeyError:
            return get_model(cfg)

    def _build_data_loaders(
        self, cfg
    ) -> Tuple[im_gens.Generator, im_gens.Generator]:
        """Builds train and val data loaders.
        """
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
