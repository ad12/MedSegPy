"""Benchmark dataloading: 
    - single vs. multiple file loading time
    - 2D vs. 3D loading time
    - single vs. multiple file training time across
        - batch size
        - patch size
        - gradient acc steps
@author: Swathi Iyer, swathii@stanford.edu
"""
import os
import time
import logging
import wandb as wb
import tensorflow as tf
from medsegpy.config import UNetConfig, UNet3DConfig
from medsegpy.data import build_loader, DatasetCatalog, DefaultDataLoader, PatchDataLoader
from medsegpy.modeling import get_model
from medsegpy.losses import (
    DICE_LOSS,
    dice_loss,
    focal_loss,
    get_training_loss,
)

#wb.init(project="benchmark_dataloaders")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Specify gpu here

cfg2d = UNetConfig()
cfg2d.TRAIN_DATASET = "oai_2d_train"
cfg2d.VAL_DATASET = "oai_2d_val"
cfg2d.TEST_DATASET = "oai_2d_test"
cfg2d.CATEGORIES = (0, (1, 2), 3, (4, 5))
cfg2d.IMG_SIZE = (384, 384, 1)


cfg3d = UNet3DConfig()
cfg3d.TAG = "PatchDataLoader"
cfg3d.TRAIN_DATASET = "oai_3d_sf_whitened_train"
cfg3d.VAL_DATASET = "oai_3d_sf_whitened_val"
cfg3d.TEST_DATASET = "oai_3d_sf_whitened_test"
cfg3d.CATEGORIES = (0, (1, 2), 3, (4, 5))
cfg3d.IMG_SIZE = (384, 384, 4, 1)


def test_loading():
    singlefile_dataloader = build_loader(
        cfg3d,
        cfg3d.TRAIN_DATASET,
        batch_size=1,
        is_test=False,
        shuffle=True,
        drop_last=True,
        use_singlefile=True
    )

    cfg3d.TRAIN_DATASET = "oai_3d_whitened_train"
    original_dataloader = build_loader(
        cfg3d,
        cfg3d.TRAIN_DATASET,
        batch_size=1,
        is_test=False,
        shuffle=True,
        drop_last=True,
        use_singlefile=False
    )
     
    logger.info("Test Original...")
    average = 0
    for i in range(5):
        start_time = time.perf_counter()
        img, seg = original_dataloader[i]
        time_elapsed = time.perf_counter() - start_time
        logger.info("load time: {}".format(time_elapsed))
        average += time_elapsed
    
    average /= 5
    logger.info("Original average loading time: {}".format(average))
    
    logger.info("Test Singlefile...")
    average = 0
    for i in range(5):
        start_time = time.perf_counter()
        img, seg = singlefile_dataloader[i]
        time_elapsed = time.perf_counter() - start_time
        logger.info("load time: {}".format(time_elapsed))
        average += time_elapsed

    average /= 5
    logger.info("Singlefile average loading time: {}".format(average))


def test_2d_v_3d():
    dataloader_2d = build_loader(
        cfg2d,
        cfg2d.TRAIN_DATASET,
        batch_size=1,
        is_test=False,
        shuffle=True,
        drop_last=True,
    )

    dataloader_3d = build_loader(
        cfg3d,
        cfg3d.TRAIN_DATASET,
        batch_size=1,
        is_test=False,
        shuffle=True,
        drop_last=True,
        use_singlefile=True
    )

    logger.info("Test 2D Average Loading Time...")
    average = 0
    for i in range(30):
        start_time = time.perf_counter()
        img, seg = dataloader_2d[i]
        time_elapsed = time.perf_counter() - start_time
        logger.info("load time: {}".format(time_elapsed))
        average += time_elapsed
    
    average /= 30
    logger.info("2D average loading time: {}".format(average))

    logger.info("Test 3D Average Loading Time...")
    average = 0
    for i in range(30):
        start_time = time.perf_counter()
        img, seg = dataloader_3d[i]
        time_elapsed = time.perf_counter() - start_time
        logger.info("load time: {}".format(time_elapsed))
        average += time_elapsed
    
    average /= 30
    logger.info("3D average loading time (singlefile): {}".format(average))


def test_batchsize():
    logger.info("Starting batchsize test, 2D vs 3D...")
    for batchsize in range(1, 7):
        dataloader_2d = build_loader(
            cfg2d,
            cfg2d.TRAIN_DATASET,
            batch_size=batchsize,
            is_test=False,
            shuffle=True,
            drop_last=True,
        )

        dataloader_3d = build_loader(
            cfg3d,
            cfg3d.TRAIN_DATASET,
            batch_size=batchsize,
            is_test=False,
            shuffle=True,
            drop_last=True,
            use_singlefile=True
        )

        average = 0
        for i in range(20):
            start_time = time.perf_counter()
            img, seg = dataloader_2d[i]
            time_elapsed = time.perf_counter() - start_time
            average += time_elapsed
        
        average /= 20
        logger.info("Average 2D loading time: {}, batchsize = {}".format(average, batchsize))

        average = 0
        for i in range(20):
            start_time = time.perf_counter()
            img, seg = dataloader_3d[i]
            time_elapsed = time.perf_counter() - start_time
            average += time_elapsed
        
        average /= 20
        logger.info("Average 3D loading time: {}, batchsize = {}".format(average, batchsize))

def test_training_filetype():
    logger.info("Start training filetype test...")
    loss_func = get_training_loss(DICE_LOSS)
    
    logger.info("Training 3D Singlefile...")
  
    for patch_size, batch_size, grad_acc in \
            [(4, 4, 1), (4, 4, 4), (16, 1, 1), (16, 1, 4)]:

        cfg3d.IMG_SIZE = (384, 384, patch_size, 1)
        cfg3d.NUM_GRAD_STEPS = grad_acc

        model = get_model(cfg3d)
        model.compile(
             optimizer='adam',
             loss=loss_func,
             metrics=[dice_loss]
        )

        singlefile_dataloader = build_loader(
            cfg3d,
            cfg3d.TRAIN_DATASET,
            batch_size=batch_size,
            is_test=False,
            shuffle=True,
            drop_last=True,
            use_singlefile=True
        )
        
        start = time.perf_counter()
        model.fit_generator(
            singlefile_dataloader,
            epochs=1,
            #validation_data=original_dataloader,
            workers=4,
            use_multiprocessing=True,
            verbose=1,
            shuffle=False,
        )
        time_elapsed = time.perf_counter() - start
        logger.info("Singlefile training time: {}; patch_size={}, batch_size={}, grad_acc={}".format(time_elapsed, patch_size, batch_size, grad_acc)) 


    for patch_size, batch_size, grad_acc in \
        [(4, 4, 1), (4, 4, 4), (16, 1, 1), (16, 1, 4)]:

        logger.info("Training 3D Multifile...")
        cfg3d.IMG_SIZE = (384, 384, patch_size, 1)
        cfg3d.NUM_GRAD_STEPS = grad_acc
        model = get_model(cfg3d)
        model.compile(
                 optimizer='adam',
                 loss=loss_func,
                 metrics=[dice_loss]
        )
        multifile_dataloader = build_loader(
            cfg3d,
            cfg3d.TRAIN_DATASET,
            batch_size=batch_size,
            is_test=False,
            shuffle=True,
            drop_last=True,
            use_singlefile=False
        )
        
        start = time.perf_counter()
        model.fit_generator(
            multifile_dataloader,
            epochs=1,
            #validation_data=original_dataloader,
            workers=4,
            use_multiprocessing=True,
            verbose=1,
            shuffle=False,
        )
        time_elapsed = time.perf_counter() - start
        logger.info("Multifile training time: {}; patch_size={}, batch_size={}, grad_acc={}".format(time_elapsed, patch_size, batch_size, grad_acc))
     

logger.info("Start DataLoader Tests...")
test_loading()
test_2d_v_3d()
test_batchsize()
test_training_filetype()
