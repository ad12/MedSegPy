# Author: Arjun Desai, arjun.desai@duke.edu, 2018 June

from __future__ import print_function, division

import pickle
import os

from keras.optimizers import Adam
from keras import backend as K

from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import ReduceLROnPlateau as rlrp
from keras.callbacks import TensorBoard as tfb
import keras.callbacks as kc

from config import DeeplabV3Config, SegnetConfig, EnsembleUDSConfig
from im_generator import calc_generator_info, img_generator
from losses import dice_loss

from models import get_model

import utils


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
        optimizer = Adam(lr=config.INITIAL_LEARNING_RATE, beta_1=0.99, beta_2=0.995, epsilon=1e-8, decay=0)

    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer,
                  loss=dice_loss, metrics=[lr_metric])

    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')
    train_files, train_nbatches = calc_generator_info(train_path, train_batch_size, learn_files)
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
    lr_cb = lrs(step_decay_wrapper(config.INITIAL_LEARNING_RATE, config.MIN_LEARNING_RATE, config.DROP_FACTOR, config.DROP_RATE))
    hist_cb = LossHistory()

    callbacks_list = [tfb_cb, cp_cb, lr_cb, hist_cb]
    
    print('Starting training')
    
    if (config.DEBUG):
        config.N_EPOCHS = 1
        train_nbatches = 5

    # Start the training
    model.fit_generator(
        img_generator(train_path, train_batch_size, img_size, tag, config.TISSUES),
        train_nbatches,
        epochs=n_epochs,
        validation_data=img_generator(valid_path, valid_batch_size, img_size, tag, config.TISSUES),
        validation_steps=valid_nbatches,
        callbacks=callbacks_list,
        verbose=1)

    # Save files to write as output
    data = [hist_cb.epoch, hist_cb.losses, hist_cb.val_losses]
    with open(pik_save_path, "wb") as f:
        pickle.dump(data, f)

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
    train_model(config)
    
    config.save_config()

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

        optimizer = config.model_from_dict_w_opt(utils.load_pik(os.path.join(subdir, 'config.dat')))
        config.N_EPOCHS = 10
        config.INITIAL_LEARNING_RATE = 1e-6

        train_model(config, optimizer)



def train_debug():
    print('')
    print('DEBUGGING.....')
    config = DeeplabV3Config()
    config.DEBUG = True
    config.N_EPOCHS = 1
    config.OS = 8
    config.DIL_RATES = (1, 1, 1)
    train_model(config)

    config.save_config()

    K.clear_session()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    
    #train_deeplab(16, (1, 9, 18))
    #train_deeplab(16, (2, 4, 6))
    #train_deeplab(16, (3, 6, 9))

    train_deeplab(8, (1, 9, 18))
    train_deeplab(8, (2, 4, 6))
    train_deeplab(8, (3, 6, 9))
    #train_deeplab(8, (2, 6, 12))

    #train_debug()

