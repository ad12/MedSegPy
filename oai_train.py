# Author: Arjun Desai, arjun.desai@duke.edu, 2018 June

from __future__ import print_function, division

import argparse
import pickle
import os
import random
import numpy as np

from keras.optimizers import Adam
from keras import backend as K

from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import ReduceLROnPlateau as rlrp
from keras.callbacks import TensorBoard as tfb
import keras.callbacks as kc

import config as MCONFIG
from config import DeeplabV3Config, SegnetConfig, EnsembleUDSConfig, UNetConfig, UNetMultiContrastConfig
from im_generator import calc_generator_info, img_generator, img_generator_oai, get_class_freq
from losses import get_training_loss, WEIGHTED_CROSS_ENTROPY_LOSS

from weight_classes import CLASS_FREQ_DAT_PATH

from models import get_model

import utils
import parse_pids


def train_model(config, optimizer=None):
    """
    Train model
    :param config: a Config object
    :param optimizer: a Keras optimizer (default = None)
    """

    # Load data from config
    train_path = config.TRAIN_PATH
    valid_path = config.VALID_PATH
    train_batch_size = config.TRAIN_BATCH_SIZE
    valid_batch_size = config.VALID_BATCH_SIZE
    cp_save_path = config.CP_SAVE_PATH
    cp_save_tag = config.CP_SAVE_TAG
    n_epochs = config.N_EPOCHS
    file_types = config.FILE_TYPES
    pik_save_path = config.PIK_SAVE_PATH
    tag = config.TAG
    learn_files = config.LEARN_FILES
    loss = config.LOSS
    layers_to_freeze = []


    # Get model based on config
    img_size = config.IMG_SIZE
    model = get_model(config)

    # Fine tune - initialize with weights
    if (config.FINE_TUNE):
        print('loading weights')
        model.load_weights(config.INIT_WEIGHT_PATH, by_name=True)

    # If no optimizer is provided, default to Adam
    if optimizer is None:
        optimizer = Adam(lr=config.INITIAL_LEARNING_RATE, beta_1=0.99, beta_2=0.995, epsilon=1e-8, decay=config.ADAM_DECAY, amsgrad=config.USE_AMSGRAD)

    # Load loss function
    class_weights = None
    # if weighted cross entropy, load weights
    if loss == WEIGHTED_CROSS_ENTROPY_LOSS and class_weights is None:
        print('calculating freq')
        class_freqs = utils.load_pik(CLASS_FREQ_DAT_PATH)
        class_weights = get_class_weights(class_freqs)
        class_weights = np.reshape(class_weights, (1,2))
        print(class_weights)

    loss_func = get_training_loss(loss, weights=class_weights)
    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer,
                  loss=loss_func, metrics=[lr_metric])

    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')
    train_files, train_nbatches = calc_generator_info(train_path, train_batch_size, learn_files=learn_files, pids=config.PIDS, augment_data=config.AUGMENT_DATA)
    valid_files, valid_nbatches = calc_generator_info(valid_path, valid_batch_size)

    print('INFO: Train size: %d, batch size: %d' % (len(train_files), train_batch_size))
    print('INFO: Valid size: %d, batch size: %d' % (len(valid_files), valid_batch_size))
    print('INFO: Image size: %s' % (img_size,))
    print('INFO: Image types included in training: %s' % (file_types,))
    print('INFO: Number of frozen layers: %s' % len(layers_to_freeze))

    # model callbacks
    cp_cb = ModelCheckpoint(
        os.path.join(cp_save_path, cp_save_tag + '_weights.{epoch:03d}-{val_loss:.4f}.h5'),
        save_best_only=True)
    tfb_cb = tfb(config.TF_LOG_DIR,
                 write_grads=False,
                 write_images=False)
    hist_cb = LossHistory()

    callbacks_list = [tfb_cb, cp_cb, hist_cb]

    # Step decay for learning rate
    if (config.USE_STEP_DECAY):
        lr_cb = lrs(step_decay_wrapper(config.INITIAL_LEARNING_RATE, config.MIN_LEARNING_RATE, config.DROP_FACTOR,
                                       config.DROP_RATE))
        callbacks_list.append(lr_cb)

    # Determine training generator based on version of config
    if (config.VERSION > 1):
        train_gen = img_generator_oai(train_path,
                                      train_batch_size,
                                      img_size,
                                      config.TISSUES,
                                      shuffle_epoch=True,
                                      pids=config.PIDS,
                                      augment_data=config.AUGMENT_DATA)
        val_gen = img_generator_oai(valid_path,
                                    valid_batch_size,
                                    img_size,
                                    config.TISSUES,
                                    tag=tag,
                                    shuffle_epoch=False,
                                    augment_data=False)
    else:
        train_gen = img_generator(train_path, train_batch_size, img_size, tag, config.TISSUES, pids=config.PIDS)
        val_gen = img_generator(valid_path, valid_batch_size, img_size, tag, config.TISSUES)


    # Start training
    model.fit_generator(
        train_gen,
        train_nbatches,
        epochs=n_epochs,
        validation_data=val_gen,
        validation_steps=valid_nbatches,
        callbacks=callbacks_list,
        verbose=1)

    # Save optimizer state
    utils.save_optimizer(model.optimizer, config.CP_SAVE_PATH)

    # Save files to write as output
    data = [hist_cb.epoch, hist_cb.losses, hist_cb.val_losses]
    with open(pik_save_path, "wb") as f:
        pickle.dump(data, f)

    # Save model
    model.save(filepath=os.path.join(config.CP_SAVE_PATH, 'model.h5'), overwrite=True)

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


def step_decay_wrapper(initial_lr=1e-4, min_lr=1e-8, drop_factor=0.8, drop_rate=1.0):
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
        lrate = initial_lr * math.pow(drop_factor, math.floor((1 + epoch) / drop_rate))
        
        if (lrate < min_lr):
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
        #self.lr = []
        self.epoch = []

    def on_epoch_end(self, batch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.losses.append(logs.get('loss'))
       # self.lr.append(step_decay(len(self.losses)))
        self.epoch.append(len(self.losses))


def fine_tune(dirpath, config, vals_dict=None):
    # If a fine-tune directory already exits, skip this directory
    if (os.path.isdir(os.path.join(dirpath, 'fine_tune'))):
        print('Skipping %s - fine_tune folder exists' % dirpath)

    # Initialize for fine tuning
    config.load_config(os.path.join(dirpath, 'config.ini'))

    # Get best weight path
    best_weight_path = utils.get_weights(dirpath)
    print('Best weight path: %s' % best_weight_path)

    config.init_fine_tune(best_weight_path)
    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.summary()
    config.save_config()

    train_model(config)

    K.clear_session()


def train_debug():
    print('')
    print('DEBUGGING.....')
    config = DeeplabV3Config()
    config.DEBUG = True
    config.N_EPOCHS = 1
    config.OS = 16
    config.DIL_RATES = (1, 1, 1)

    config.save_config()

    train_model(config)

    K.clear_session()


def data_limitation_train(pid_counts=[5, 15, 30, 60]):
    """
    Train data limited networks
    :return:
    """
    print('Data limitation......')
    import math
    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_data_limit/oai_data'
    pids = utils.load_pik(parse_pids.PID_TXT_PATH)
    num_pids = len(pids)

    for pid_count in pid_counts:
        MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_data_limit/oai_data/%03d' % pid_count

        if (pid_count > num_pids):
            pid_count = num_pids

        # Randomly subsample pids
        pids_sampled = random.sample(pids, pid_count)
        s_ratio = math.ceil(num_pids / pid_count)

        config = SegnetConfig()
        #config.DIL_RATES = (1, 9, 18)
        config.N_EPOCHS = math.ceil(20 * num_pids / pid_count)
        config.DROP_FACTOR = config.DROP_FACTOR ** (1/s_ratio)
        config.INITIAL_LEARNING_RATE = 1e-3
        config.USE_STEP_DECAY = False
        config.PIDS = pids_sampled if pid_count != num_pids else None
        print('# Subjects: %d' % pid_count)
        config.save_config()
        config.summary()

        train_model(config)
        
        print('Epochs: %d' % config.N_EPOCHS)
        K.clear_session()

    # must exit because config constant has been overwritten
    exit()


def unet_2d_multi_contrast_train():
    """
    Train multi contrast network
    """
    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/akshay/dl/oai_data/segnet_2d/'

    # By default, loads weights from original 2D unet
    config = UNetMultiContrastConfig()

    # By default, loads weights from original 2D unet
    # To not load these weights by default, uncomment line below
    #config.INIT_UNET_2D = False

    # Adjust hyperparameters
    config.N_EPOCHS = 25
    config.DROP_FACTOR = 0.8
    config.DROP_RATE = 1.0

    train_model(config)

    # need to exit because overwritting config parameters
    exit()

def train(config, vals_dict=None):
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
    
    train_model(config)

    K.clear_session()

# Use these for fine tuning
DEEPLAB_TEST_PATHS_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d'
DEEPLAB_TEST_PATHS = ['2018-08-26-20-01-32', # OS=16, DIL_RATES=(6, 12, 18)
        '2018-08-27-02-49-06', # OS=16, DIL_RATES=(1, 9, 18)
                      '2018-08-27-15-48-56', # OS=16, DIL_RATES=(3, 6, 9)
                     ]

DATA_LIMIT_PATHS_PREFIX = os.path.join('/bmrNAS/people/arjun/msk_data_limit/oai_data', '%03d', 'unet_2d')
DATA_LIMIT_NUM_DATE_DICT = {5:'2018-08-26-20-19-31',
                            15:'2018-08-27-03-43-46',
                            30:'2018-08-27-11-18-07',
                            60:'2018-08-27-18-29-19'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OAI dataset')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use')
    args = parser.parse_args()
    gpu = args.gpu

    print('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

    # train with weighted cross entropy
    #train(DeeplabV3Config(), {'OS': 16, 'DIL_RATES': (2,4,6), 'DROPOUT_RATE': 0.0})

    data_limitation_train(pid_counts=[60])
    #fine tune
    #fine_tune('/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-09-13-07-11-03/', DeeplabV3Config(), vals_dict={'INITIAL_LEARNING_RATE': 1e-6, 'USE_STEP_DECAY': False, 'N_EPOCHS': 20})
    #train(DeeplabV3Config(), {'OS': 16, 'DIL_RATES': (2, 4, 6)})

    #train(SegnetConfig(), {'INITIAL_LEARNING_RATE': 1e-3, 'FINE_TUNE': False, 'TRAIN_BATCH_SIZE': 15})
    #train(SegnetConfig(), {'INITIAL_LEARNING_RATE': 1e-3, 'CONV_ACT_BN': True, 'TRAIN_BATCH_SIZE': 15})

    #train(SegnetConfig(), {'INITIAL_LEARNING_RATE': 1e-3, 'DEPTH': 7, 'NUM_CONV_LAYERS': [3, 3, 3, 3, 3, 3, 3], 'NUM_FILTERS': [16, 32, 64, 128, 256, 512, 1024], 'TRAIN_BATCH_SIZE': 35})
    #fine_tune('/bmrNAS/people/arjun/msk_seg_networks/oai_data/segnet_2d/2018-09-01-22-39-39', SegnetConfig(), vals_dict = {'INITIAL_LEARNING_RATE': 1e-5, 'USE_STEP_DECAY': True, 'DROP_FACTOR': 0.7, 'DROP_RATE': 8.0, 'N_EPOCHS': 20})
