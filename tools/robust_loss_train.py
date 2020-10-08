import os
import warnings
import time
import logging
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from keras import backend as K
from medsegpy.config import UNetConfig, UNet3DConfig, DeeplabV3Config
from medsegpy.data import build_loader, DatasetCatalog, DefaultDataLoader, PatchDataLoader
from medsegpy.modeling.meta_arch import build_model
from medsegpy.losses import (
    DICE_LOSS,
    dice_loss,
    focal_loss,
    get_training_loss,
    MULTI_CLASS_DICE_LOSS,
)
from medsegpy.evaluation.evaluator import DatasetEvaluator
from medsegpy.evaluation import build_evaluator, inference_on_dataset, SemSegEvaluator
from medsegpy.utils.logger import setup_logger

from medsegpy.losses import NaiveAdaRobLossComputer
from medsegpy.engine import WandBLogger
import wandb
# setup_logger()

#wb.init(project="benchmark_unet3d", magic=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # run on gpu1

logger = setup_logger()
logger.info("start test")

cfg2d = UNetConfig()
cfg2d.TRAIN_DATASET = "oai_2d_train"
cfg2d.VAL_DATASET = "oai_2d_val"
cfg2d.TEST_DATASET = "oai_2d_test"
cfg2d.CATEGORIES = (0, (1, 2), 3, (4, 5))
cfg2d.IMG_SIZE = (384, 384, 1)
cfg2d.NUM_WORKERS = 24

cfg = cfg2d
exp_name = ""
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

def test_lms_vs_patch():
    loss_func = get_training_loss(DICE_LOSS, reduce="class")
    
    # loss_computer = NaiveAdaRobLossComputer(loss_func, 4, 0.1)
    # loss = loss_computer
    # callbacks = [loss_computer, WandBLogger()]

    loss = loss_func
    callbacks = [WandBLogger()]

    cfg2d.IMG_SIZE = (384, 384, 1)

    model = build_model(cfg2d)
    model.compile(
         optimizer='adam',
         loss=loss,
         metrics=[dice_loss],
    )
    model.run_eagerly = True

    train_dataloader = build_loader(
        cfg2d,
        cfg2d.TRAIN_DATASET,
        batch_size=16,
        is_test=False,
        shuffle=True,
        drop_last=True,
    )

    val_dataloader = build_loader(
        cfg2d,
        cfg2d.VAL_DATASET,
        batch_size=12,
        is_test=False,
        shuffle=True,
        drop_last=True,
    )

    test_dataloader = build_loader(
        cfg2d,
        cfg2d.TEST_DATASET,
        batch_size=8,
        is_test=True,
        shuffle=False,
        drop_last=False,
    )
    
    start = time.perf_counter()
    model.fit_generator(
        train_dataloader,
        epochs=1,
        validation_data=val_dataloader,
        workers=cfg2d.NUM_WORKERS,
        use_multiprocessing=True,
        verbose=1,
        shuffle=False,
        callbacks=callbacks,
        # TODO: Remove steps after debugging
        steps_per_epoch=20,
        validation_steps=10,
        #callbacks=[lms, WandbCallback()]
    )
    time_elapsed = time.perf_counter() - start
    # logger.info("LMS training time: {}".format(time_elapsed)) 
    #K.get_session().close()

    # model.load_weights('/home/swathii/MedSegPy/benchmarking/unet3d-weights-basic-4.h5')
    cfg2d.TEST_METRICS = ["DSC"]
    evaluator = SemSegEvaluator(cfg2d.TEST_DATASET, cfg2d, save_raw_data=False)
    results = inference_on_dataset(model, test_dataloader, evaluator) 
    # print(results)
    # f = open("unet3d-results-basic-4.txt","w")
    # f.write( str(results) )
    # f.close()
     
test_lms_vs_patch()