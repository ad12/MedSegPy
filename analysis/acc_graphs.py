import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np
import scipy.io as sio


import utils


# Architecture result paths
ARCH_UNET = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/unet_2d/original_akshaysc'
ARCH_SEGNET = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/segnet_2d/2018-09-26-19-08-34' #VERIFY
ARCH_DEEPLAB = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-09-27-07-52-25' # VERIFY

ARCHS = {'filename': 'architecture.png',
         'keys': ['unet', 'segnet', 'deeplabv3+'],
         'unet': ARCH_UNET, 'segnet': ARCH_SEGNET, 'deeplabv3+': ARCH_DEEPLAB}

# Loss function result paths
LOSS_DSC = ARCH_UNET
LOSS_WCE = ''
LOSS_BCE = ''

LOSSES = {'filename': 'losses.png',
          'keys': ['dsc', 'wce', 'bce'],
          'dsc': LOSS_DSC, 'wce': LOSS_WCE, 'bce': LOSS_BCE}

# Augmentation result paths
AUG_YES = ARCH_UNET
AUG_NO = ''

AUGS = {'filename': 'augment.png',
        'keys': ['aug', 'no aug'],
        'aug': AUG_YES, 'no aug': AUG_NO}

# Volume result paths (2D/2.5D/3D)
VOLUME_2D = ARCH_UNET
VOLUME_2_5D = ''
VOLUME_3D = ''

VOLUMES = {'filename': 'volume.png',
           'keys': ['2d', '2.5d', '3d'],
           '2d': VOLUME_2D, '2.5d': VOLUME_2_5D, '3d': VOLUME_3D}

SAVE_PATH = '/bmrNAS/people/arjun/msk_seg_networks/analysis/exp_graphs'
utils.check_dir(SAVE_PATH)

EXP_DICTS = [ARCHS]
COLORS = ['#0000FF', '#00FF00', '#FF0000', '#CC4F1B', '#6600CC']


def graph_acc(exp_dict):
    data_keys = exp_dict['keys']
    filename = exp_dict['filename']

    plt.clf()

    c = 0
    for data_key in data_keys:
        mat_filepath = os.path.join(exp_dict[data_key], 'test_results', 'total_interp_data.mat')
        mat_data = sio.loadmat(mat_filepath)
        xs = mat_data['xs']
        ys = mat_data['ys']

        x_interp_mean = np.mean(xs, 0)
        y_interp_mean = np.mean(ys, 0)
        y_interp_sem = np.std(ys, 0) / np.sqrt(ys.shape[0])

        plt.plot(x_interp_mean, y_interp_mean, 'k', color=COLORS[c])
        plt.fill_between(x_interp_mean, y_interp_mean - y_interp_sem, y_interp_mean + y_interp_sem, alpha=0.35)

        c += 1

    plt.xlabel('FOV (%)')
    plt.ylabel('Dice')
    plt.legend(data_keys)
    plt.savefig(os.path.join(SAVE_PATH, filename))


if __name__ == '__main__':
    for exp_dict in EXP_DICTS:
        graph_acc(exp_dict)