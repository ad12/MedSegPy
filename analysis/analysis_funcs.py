"""
Functions used for analysis
@author: Arjun Desai, arjundd@stanford.edu
"""

import sys

sys.path.insert(0, '../')

# %matplotlib inline
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np
import scipy.io as sio

import seaborn as sns

import utils

from scipy import optimize as sop
from matplotlib.ticker import ScalarFormatter

cpal = sns.color_palette("pastel", 8)
SAVE_PATH = utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/analysis/exp_graphs')


def graph_slice_exp(exp_dict, show_plot=False, ax=None, title=''):
    """
    Compute %FOV vs Dice Accuracy using data saved in 'total_interp_data.mat'" for multiple test result files
    
    @param exp_dict: A dictionary with the following fields
                        - 'filename': image filename to save graph (e.g. "architecture")
                        - 'keys': list of strings corresponding to keys to process in dictionary
                                    These keys should be keys in exp_dict where values are path to the test_results folder
                                    e.g. 'U-Net': ['/unet/test_results/'], 
                                         'No aug': ['/no_aug/seed1/test_results', '/no_aug/seed2/test_results'])
    """
    data_keys = exp_dict['keys']
    filename = exp_dict['filename']

    if ax is None:
        plt.figure()
        ax = plt.gca()

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

        ax.plot(x_interp_mean, y_interp_mean, 'k', color=cpal[c])
        ax.fill_between(x_interp_mean, y_interp_mean - y_interp_sem, y_interp_mean + y_interp_sem, alpha=0.35,
                        edgecolor=cpal[c], facecolor=cpal[c])

        legend_keys.append(data_key)
        c += 1

    ax.set_ylim([0.6, 1])
    ax.set_xlabel('FOV (%)', labelpad=0)
    ax.set_ylabel('DSC')
    ax.set_title(title)
    # plt.legend(legend_keys)
    # txt = fig.text(0.49, -0.04, 'FOV (%)', fontsize=13)
    lgd = ax.legend(legend_keys, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=3)
    plt.savefig(os.path.join(SAVE_PATH, '%s.png' % filename), format='png', dpi=1000, bbox_inches='tight')

    if show_plot:
        plt.show()


def graph_data_limitation(data, filename):
    fig, ax_array = plt.subplots(1, len(list(data.keys())), figsize=(len(list(data.keys())) * 6, 3))

    i = 0
    for k in data.keys():
        ylabel = k.upper()

        if ylabel.endswith('S'):
            ylabel = ylabel[:-1]

        ax = ax_array[i]

        print('=====================')
        print('        %s          ' % ylabel)
        print('=====================')
        results = get_data_limitation(data[k], k)
        c = 0
        for model in results.keys():
            xs, ys, SEs, x_sim, y_sim, r2 = results[model]
            ax.semilogx(xs, ys, 'o', color=cpal[c], label='%s' % model)
            ax.errorbar(xs, ys, yerr=SEs, ecolor=cpal[c], fmt='none')

            print('r2, r - %s : %0.4f, %0.4f' % (model, r2, np.sqrt(r2)))

            ax.semilogx(x_sim, y_sim, 'k--', color=cpal[c])
            ax.set_xticks([5, 10, 20, 40, 60])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            c += 1
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlabel('# Patients', fontsize=13)
        i += 1

    ax_center = ax_array[len(ax_array) // 2]
    # txt = fig.text(0.49, -0.04, '#Patients', fontsize=13)
    lgd = ax_center.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.25),
                           fancybox=True, shadow=True, ncol=3)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(os.path.join(SAVE_PATH, '%s.png' % filename), format='png', dpi=1000, bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def get_data_limitation(multi_data, metric_id):
    data_keys = multi_data['keys']
    num_patients = [5, 15, 30, 60]
    c = 0

    results_dict = {}
    for k in data_keys:
        data = multi_data[k]

        num_patients_data = {}
        for num_p in num_patients:
            num_patients_data[num_p] = np.asarray([])

        for i in range(len(data[0])):
            num_p = num_patients[i]
            for j in range(len(data)):
                test_results_folder = data[j][i]
                metrics_filepath = os.path.join(test_results_folder, 'metrics.dat')
                metrics = utils.load_pik(metrics_filepath)

                num_patients_data[num_p] = np.append(num_patients_data[num_p], metrics[metric_id].flatten())

        xs = []
        ys = []
        SEs = []
        for num_p in num_patients:
            xs.append(num_p)
            ys.append(np.mean(num_patients_data[num_p]))
            SEs.append(np.std(num_patients_data[num_p]) / np.sqrt(num_patients_data[num_p].shape[0]))

        x_sim, y_sim, r2 = fit_power_law(xs, ys)

        results_dict[k] = (xs, ys, SEs, x_sim, y_sim, r2)

    return results_dict


__EPSILON__ = 1e-8


def fit_power_law(xs, ys):
    def func(x, a, b):
        exp = x ** b
        return a * exp

    x = np.asarray(xs)
    y = np.asarray(ys)

    popt, _ = sop.curve_fit(func, x, y, p0=[1, 1], maxfev=1000)

    residuals = y - func(x, popt[0], popt[1])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (ss_res / (ss_tot + __EPSILON__))

    # Simulate on data
    x_sim = np.linspace(np.min(x), np.max(x), 100)
    y_sim = func(x_sim, popt[0], popt[1])

    return x_sim, y_sim, r_squared
