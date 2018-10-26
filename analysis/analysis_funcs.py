"""
Functions used for analysis
@author: Arjun Desai, arjundd@stanford.edu
"""

import sys
sys.path.insert(0, '../')

import matplotlib
#%matplotlib inline
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np
import scipy.io as sio

import seaborn as sns

import utils


cpal = sns.color_palette("pastel", 8)
SAVE_PATH = '/bmrNAS/people/arjun/msk_seg_networks/analysis/exp_graphs'


def graph_slice_exp(exp_dict, show_plot=False):
    """
    Compute %FOV vs Dice Accuracy using data saved in 'total_interp_data.mat'" for multiple test result files
    
    @param exp_dict: A dictionary with the following fields
                        - 'filename': image filename to save graph (e.g. "architecture.png")
                        - 'keys': list of strings corresponding to keys to process in dictionary
                                    These keys should be keys in exp_dict where values are path to the test_results folder
                                    e.g. 'U-Net': ['/unet/test_results/'], 
                                         'No aug': ['/no_aug/seed1/test_results', '/no_aug/seed2/test_results'])
    """
    data_keys = exp_dict['keys']
    filename = exp_dict['filename']

    plt.clf()
    
    legend_keys = []
    c = 0
    for data_key in data_keys:
        data_dirpaths = exp_dict[data_key]
        if type(data_dirpaths) is str:
            data_dirpaths = [data_dirpaths]
            
        xs = []
        ys = []

        for data_dirpath in data_dirpaths:
            if len(data_dirpath) == 0:
                continue

            mat_filepath = os.path.join(data_dirpath, 'total_interp_data.mat')
            mat_data = sio.loadmat(mat_filepath)
            xs.append(mat_data['xs'])
            ys.append(mat_data['ys'])
        
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        
        x_interp_mean = np.mean(xs, 0)
        y_interp_mean = np.mean(ys, 0)
        y_interp_sem = np.std(ys, 0) / np.sqrt(ys.shape[0])

        plt.plot(x_interp_mean, y_interp_mean, 'k', color=cpal[c])
        plt.fill_between(x_interp_mean, y_interp_mean - y_interp_sem, y_interp_mean + y_interp_sem, alpha=0.35, edgecolor=cpal[c], facecolor=cpal[c])
        
        legend_keys.append(data_key)
        c += 1

    plt.ylim([0.6, 1])
    plt.xlabel('FOV (%)')
    plt.ylabel('Dice')
    plt.legend(legend_keys)
    plt.savefig(os.path.join(SAVE_PATH, filename))# Architecture experiment
    
    if show_plot:
        plt.show()