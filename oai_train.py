# Author: Arjun Desai, arjun.desai@duke.edu, 2018 June

from __future__ import print_function, division

import argparse
import pickle
import os
import random

from keras.optimizers import Adam
from keras import backend as K

from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import ReduceLROnPlateau as rlrp
from keras.callbacks import TensorBoard as tfb
import keras.callbacks as kc

import config as MCONFIG
from config import DeeplabV3Config, SegnetConfig, EnsembleUDSConfig, UNetConfig, UNetMultiContrastConfig
from im_generator import calc_generator_info, img_generator, img_generator_oai
from losses import dice_loss

from models import get_model

import utils
import parse_pids

def train_model(config, optimizer=None):
    
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
    layers_to_freeze = []


    # Create Deeplab model
    img_size = config.IMG_SIZE
    model = get_model(config)

    if (config.FINE_TUNE):
        print('loading weights')
        model.load_weights(config.INIT_WEIGHT_PATH, by_name=True)

    if optimizer is None:
        optimizer = Adam(lr=config.INITIAL_LEARNING_RATE, beta_1=0.99, beta_2=0.995, epsilon=1e-8, decay=config.ADAM_DECAY)

    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer,
                  loss=dice_loss, metrics=[lr_metric])

    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')
    print(type(config.PIDS))
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


    # Start the training
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

    return hist_cb

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

# learning rate schedule

def step_decay_wrapper(initial_lr, min_lr, drop_factor, drop_rate):
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


# Print and asve the training history
class LossHistory(kc.Callback):
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

def train_deeplab(OS, dilation_rates):
    config = DeeplabV3Config()
    config.OS = OS
    config.DIL_RATES = dilation_rates
    
    config.N_EPOCHS = 25
    config.save_config()
    #config.TRAIN_BATCH_SIZE = 5
    train_model(config)

    K.clear_session()


def fine_tune(dirpath, config):
    # If a fine-tune directory already exits, skip this directory
    if (os.path.isdir(os.path.join(dirpath, 'fine_tune'))):
        print('Skipping %s - fine_tune folder exists' % dirpath)

    # Initialize for fine tuning
    config.load_config(os.path.join(dirpath, 'config.ini'))

    # Get best weight path
    best_weight_path = utils.get_weights(dirpath)
    print('Best weight path: %s' % best_weight_path)
    config.init_fine_tune(best_weight_path)

    config.N_EPOCHS = 10
    config.INITIAL_LEARNING_RATE = 4e-7
    config.DROP_RATE = 1.0
    config.DROP_FACTOR = 0.5
    config.MIN_LEARNING_RATE=1e-9

    config.summary()

    train_model(config)

    K.clear_session()


def fine_tune_deeplab(base_path):
    files = os.listdir(base_path)
    f_subdirs = []

    for file in files:
        possible_dir = os.path.join(base_path, file)
        if (os.path.isdir(possible_dir) and
                os.path.isfile(os.path.join(possible_dir,'config.ini')) and
                not os.path.isdir(os.path.join(possible_dir, 'fine_tune'))):
            f_subdirs.append(possible_dir)

    for subdir in f_subdirs:
        # Initialize config
        config = DeeplabV3Config(create_dirs=False)
        config.load_config(os.path.join(base_path, 'config.ini'))
        best_weight_path = utils.get_weights(subdir)
        config.init_fine_tune(best_weight_path)

        config.N_EPOCHS = 10
        config.INITIAL_LEARNING_RATE = 1e-6
        config.DROP_RATE = 2.0

        train_model(config)


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


def data_limitation_train():
    print('Data limitation......')
    import math
    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_data_limit/oai_data'
    pids = utils.load_pik(parse_pids.PID_TXT_PATH)
    num_pids = len(pids)

    # run network training
    pid_counts = [1]
    pid_counts.extend(list(range(5,num_pids+1,5)))
    pid_counts = [30, 60]

    for pid_count in pid_counts:
        MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_data_limit/oai_data/%03d' % pid_count

        if (pid_count > num_pids):
            pid_count = num_pids

        # Randomly subsample pids
        pids_sampled = random.sample(pids, pid_count)
        s_ratio = math.ceil(num_pids / pid_count)

        config = UNetConfig()
        config.N_EPOCHS = math.ceil(10 * num_pids / pid_count)
        config.DROP_FACTOR = config.DROP_FACTOR ** (1/s_ratio)
       
        config.PIDS = pids_sampled

        config.save_config()

        train_model(config)
        
        print('Epochs: %d' % config.N_EPOCHS)
        K.clear_session()

    # must exit because config constant has been overwritten
    exit()


def unet_2d_multi_contrast_train():
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

    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.save_config()
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

    #train_deeplab(16, (6, 12, 18))
    #train_deeplab(16, (1, 9, 18))
    #train_deeplab(16, (3, 6, 9))
    #train_deeplab(16, (2, 4, 6))
    #train_deeplab(16, (2, 3, 8))

    #Fine tune deeplab
    #for mdir in DEEPLAB_TEST_PATHS:
     #   filepath = os.path.join(DEEPLAB_TEST_PATHS_PREFIX, mdir)
      #  config = DeeplabV3Config(create_dirs=False)
       # fine_tune(filepath, config)

    #train_debug()

    # No augmentation
    train(DeeplabV3Config(), {'OS': 16, 'DIL_RATES': (1, 9, 18), 'AUGMENT_DATA': False, 'N_EPOCHS': 75})

    # No step decay
    train(DeeplabV3Config(), {'OS': 16, 'DIL_RATES': (1, 9, 18), 'USE_STEP_DECAY': False, 'INITIAL_LEARNING_RATE': 5e-3})

    # Train with lr, etc from original setup
    train(DeeplabV3Config(), {'OS': 16, 'DIL_RATES': (1, 9, 18), 'INITIAL_LEARNING_RATE': 2e-3, 'DROP_FACTOR': 0.5,
                              'DROP_RATE': 2.0})


