"""
Functions used for analysis
@author: Arjun Desai, arjundd@stanford.edu
"""

import matplotlib.pyplot as plt

fsize = 12
params = {'legend.fontsize': fsize * 0.925,
          'axes.labelsize': fsize,
          'axes.titlesize': fsize * 1.25,
          'xtick.labelsize': fsize * 0.925,
          'ytick.labelsize': fsize * 0.925}

import os, sys
import numpy as np
import scipy.io as sio

import seaborn as sns

from scipy import optimize as sop
from matplotlib.ticker import ScalarFormatter
import pandas as pd

sys.path.append('../')
from medsegpy.utils import io_utils

# Define some custom color palettes
american_palette = ['#ffeaa7', '#00cec9', '#0984e3', '#6c5ce7', '#b2bec3']  # yellow too pale
color_brewer_1 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99']  # not good
color_brewer_2 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']  # not good

cpal = sns.color_palette("pastel", 8)
# cpal = sns.color_palette(american_palette)

SAVE_PATH = io_utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/analysis/exp_graphs')


def graph_slice_exp(exp_dict, show_plot=False, ax=None, title='', ylim=[0.6, 1], show_error=True, legend_loc='side',
                    working_dir=SAVE_PATH, **kwargs):
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
    filename = exp_dict['filename'] if 'filename' in exp_dict.keys() else None

    if ax is None:
        plt.figure()
        ax = plt.gca()

    legend_keys = []
    c = 0
    
    cpal = kwargs.get('cpal') if 'cpal' in kwargs else sns.color_palette("muted", len(data_keys))
    
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
        y_interp_sem = np.std(ys, 0)

        ax.plot(x_interp_mean, y_interp_mean, 'k', color=cpal[c])
        if show_error:
            ax.fill_between(x_interp_mean, y_interp_mean - y_interp_sem, y_interp_mean + y_interp_sem, alpha=0.35,
                            edgecolor=cpal[c], facecolor=cpal[c])

        legend_keys.append(data_key)
        c += 1

    ax.set_ylim(ylim)
    ax.set_xlabel('FOV (%)', labelpad=0)
    ax.set_ylabel('DSC')
    ax.set_title(title)
    ax.autoscale_view()
    # plt.legend(legend_keys)
    # txt = fig.text(0.49, -0.04, 'FOV (%)', fontsize=13)
    if legend_loc:
        if legend_loc == 'side':
            lgd = ax.legend(legend_keys, loc='center left', bbox_to_anchor=(1.0, 0.5),
                            fancybox=True, shadow=True, ncol=1)
        else:
            lgd = ax.legend(legend_keys, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                            fancybox=True, shadow=True, ncol=len(exp_dict['keys']))
    
    if filename:
        plt.savefig(os.path.join(working_dir, '%s.png' % filename), format='png', dpi=1000, bbox_inches='tight')

    if show_plot:
        plt.show()
    
    return ax


def graph_data_limitation(data, filename, decay_exp_fit=False):
    cpal = sns.color_palette("muted", 3)
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
        asymtote = 1.0 if k == 'dsc' else 0.0

        results = get_data_limitation(data[k], k, decay_exp_fit=decay_exp_fit, asymtote=asymtote)
        c = 0
        for model in results.keys():
            xs, ys, SEs, x_sim, y_sim, r2, a, b = results[model]
            ax.semilogx(xs, ys, 'o', color=cpal[c], label='%s' % model)
            ax.errorbar(xs, ys, yerr=SEs, ecolor=cpal[c], fmt='none', capsize=5)

            print('r2, r - %s : %0.4f, %0.4f' % (model, r2, np.sqrt(r2)))

            ax.semilogx(x_sim, y_sim, 'k--', color=cpal[c])
            ax.set_xticks([5, 10, 20, 40, 60])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            c += 1
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlabel('# Patients', fontsize=13)
        i += 1

    ax_center = ax_array[len(ax_array) // 2]
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # txt = fig.text(0.49, -0.04, '#Patients', fontsize=13)
    lgd = ax_center.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5),
                           fancybox=True, shadow=True, ncol=3)
    if filename:
        plt.savefig(os.path.join(SAVE_PATH, '%s.png' % filename), format='png', dpi=1000, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')


def get_data_limitation(multi_data, metric_id, decay_exp_fit=False, asymtote=0):
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
                metrics = io_utils.load_pik(metrics_filepath)

                num_patients_data[num_p] = np.append(num_patients_data[num_p], metrics[metric_id].flatten())

        xs = []
        ys = []
        SEs = []
        for num_p in num_patients:
            xs.append(num_p)
            ys.append(np.mean(num_patients_data[num_p]))
            SEs.append(np.std(num_patients_data[num_p]))

        if decay_exp_fit:
            x_sim, y_sim, r2, a, b = fit_decay_exp(xs, ys, asymtote)
        else:
            x_sim, y_sim, r2, a, b = fit_power_law(xs, ys)

        results_dict[k] = (xs, ys, SEs, x_sim, y_sim, r2, a, b)

    return results_dict


__EPSILON__ = 1e-8


def fcn_exp(base_paths, exp_names, dirname):
    if type(base_paths) is str:
        base_paths = [base_paths]

    if type(exp_names) is str:
        exp_names = [exp_names]

    test_set_name = ['V0 (288x288x72)', 'V1 (320x320x80)', 'V2 (352x352x80)', 'V3 (384x384x80)']
    test_folders = ['test_results', 'test_results_midcrop1', 'test_results_midcrop2', 'test_results_nocrop']

    exp_means = []
    exp_stds = []
    for i in range(len(base_paths)):
        base_path = base_paths[i]
        test_folder_paths = []
        for tfolder in test_folders:
            test_folder_paths.append(os.path.join(base_path, tfolder))

        # get dice accuracy metric
        metrics = stats.get_metrics(test_folder_paths)
        dsc = metrics['DSC']

        bar_width = 0.35
        opacity = 0.8
        sub_means = []
        sub_stds = []
        for ind in range(len(test_set_name)):
            vals = np.asarray(dsc[ind])
            sub_means.append(np.mean(vals))
            std = np.std(vals) if len(vals) > 1 else None
            sub_stds.append(std)

        exp_means.append(sub_means)
        exp_stds.append(sub_stds)

        # Do kruskal dunn analysis
        print('==' * 30)
        print(exp_names[i])
        print('==' * 30)
        stats.kruskal_dunn_analysis(test_folder_paths, test_set_name, dirname)
        print('==' * 30)

    exp_means = pd.DataFrame(exp_means, index=exp_names, columns=test_set_name)
    exp_stds = pd.DataFrame(exp_stds, index=exp_names, columns=test_set_name)

    # Display bar graph
    display_bar_graph(exp_means, exp_stds, os.path.join(SAVE_PATH, '%s.png' % dirname), ylabel='DSC', legend_loc='best',
                      ncol=2)

def fit_power_law(xs, ys):
    def func(x, a, b):
        exp = x ** b
        return a * exp

    def res_func(x, a, b):
        # y = a*x^b ---> log(y) = b*log(x) + log(a)
        return b * np.log(x) + np.log(a)

    x = np.asarray(xs)
    y = np.asarray(ys)

    popt, _ = sop.curve_fit(func, x, y, p0=[1, 1], maxfev=1000)
    print(popt)
    log_y = np.log(y)
    residuals = log_y - res_func(x, popt[0], popt[1])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)

    r_squared = 1 - (ss_res / (ss_tot + __EPSILON__))

    # Simulate on data
    x_sim = np.linspace(np.min(x), np.max(x), 100)
    y_sim = func(x_sim, popt[0], popt[1])

    return x_sim, y_sim, r_squared, popt[0], popt[1]


def fit_decay_exp(xs, ys, asymtote=1.0):
    def func(x, a, b):
        # y = asymtote + A*exp(bx)
        exp = np.exp(b * x)
        return asymtote + a * exp

    def res_func(x, a, b):
        # y = asymtote + A*exp(bx) --> log(y-asymtote) = log(A) + bx
        return np.log(a) + b * x

    x = np.asarray(xs)
    y = np.asarray(ys)
    p00 = 1 if asymtote == 0 else -1
    popt, _ = sop.curve_fit(func, x, y, p0=[p00, -1], maxfev=1000)
    log_y = np.log(y - asymtote)
    residuals = log_y - (res_func(x, popt[0], popt[1]) - asymtote)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)

    r_squared = 1 - (ss_res / (ss_tot + __EPSILON__))

    # Simulate on data
    x_sim = np.linspace(np.min(x), np.max(x), 100)
    y_sim = func(x_sim, popt[0], popt[1])

    return x_sim, y_sim, r_squared, popt[0], popt[1]


def print_metrics_summary(dir_paths):
    print('')
    for dp in dir_paths:
        print(dp)
        print('--' * 40)
        test_file = os.path.join(dp, 'results.txt')
        with open(test_file) as search:
            for line in search:
                line = line.rstrip()  # remove '\n' at end of line
                if 'MEAN +/- STD' in line.upper() or 'RMS +/- STD' in line.upper():
                    print(line)
        print('')
