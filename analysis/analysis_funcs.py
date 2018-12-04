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
import pandas as pd

# Define some custom color palettes
american_palette = ['#ffeaa7', '#00cec9', '#0984e3', '#6c5ce7', '#b2bec3'] # yellow too pale
color_brewer_1 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99'] # not good
color_brewer_2 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3'] # not good

cpal = sns.color_palette("muted", 8)
#cpal = sns.color_palette(american_palette)

SAVE_PATH = utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/analysis/exp_graphs')

import stats

def graph_slice_exp(exp_dict, show_plot=False, ax=None, title='', ylim=[0.6, 1]):
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
        y_interp_sem = np.std(ys, 0)

        ax.plot(x_interp_mean, y_interp_mean, 'k', color=cpal[c])
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
    lgd = ax.legend(legend_keys, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=len(exp_dict['keys']))
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
            SEs.append(np.std(num_patients_data[num_p]))

        x_sim, y_sim, r2 = fit_power_law(xs, ys)

        results_dict[k] = (xs, ys, SEs, x_sim, y_sim, r2)

    return results_dict


__EPSILON__ = 1e-8


def fcn_exp(base_paths, exp_names, dirname):
    
    if type(base_paths) is str:
        base_paths = [base_paths]
        
    if type(exp_names) is str:
        exp_names = [exp_names]
    
    test_set_name = ['original (V0)', 'midcrop1 (V1)', 'midcrop2 (V2)', 'nocrop (V3)']
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
        print('=='*30)
        print(exp_names[i])
        print('=='*30)
        stats.kruskal_dunn_analysis(test_folder_paths, test_set_name, dirname)
        print('=='*30)
    
    exp_means = pd.DataFrame(exp_means, index=exp_names, columns=test_set_name)
    exp_stds = pd.DataFrame(exp_stds, index=exp_names, columns=test_set_name)
    
    # Display bar graph
    display_bar_graph(exp_means, exp_stds)
    
def display_bar_graph(df_mean, df_error, exp_filepath=None, legend_loc='bottom', sig_markers=[]):
    line_width = 1
    
    assert df_mean.shape == df_error.shape, "Both dataframes must be same shape"
    
    x_labels = df_mean.index.tolist()
    n_groups = len(x_labels)
    x_index = np.arange(0, n_groups*2, 2)
    
    columns = df_mean.columns.tolist()
    
    fig, ax = plt.subplots()
    bar_width = 0.25
    opacity = 0.9
    
    df_mean_arr = np.asarray(df_mean)
    df_error_arr = np.asarray(df_error)
    
    p = []
    e = []
    errs = []
    for ind in range(len(columns)):
        sub_means = df_mean_arr[..., ind]
        sub_errors = df_error_arr[..., ind]
        
        p.append(ax.bar(x_index + (bar_width)*ind, sub_means, bar_width,
                        alpha=opacity,
                        color=cpal[ind],
                        label=columns[ind],
                        edgecolor='gray',
                        linewidth=line_width,
                        bottom=0))
        
        e.append(ax.errorbar(x_index + (bar_width)*ind, sub_means,
                             yerr=[np.zeros(sub_errors.shape), sub_errors],
                             ecolor='gray', 
                             elinewidth=line_width, 
                             capsize=5, 
                             capthick=line_width, 
                             linewidth=0))
        errs.append(sub_errors)
        
    for eb in e:
        BarCapSizer(eb.lines[1], 0.1)
    
    delta = (len(columns) - 1)*bar_width/2
    plt.xticks(x_index + delta, x_labels)
    
    if legend_loc == 'bottom':
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                           fancybox=True, shadow=True, ncol=len(columns))
    else:
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, fancybox=True)
    
    #display_sig_markers(p, e, [(0,1,'*')], ax)
    
    if exp_filepath is not None:
        plt.savefig(exp_filepath, format='png',
                    dpi=1000,
                    bbox_inches='tight')
    else:
        plt.show()

        
def display_sig_markers(p, errs, sig_markers, ax):
    def draw_sig_marker(rect1, rect2, marker):
        x1, y1, width1 = rect1.get_x(), rect1.get_height(), rect1.get_width()
        x2, y2, width2 = rect2.get_x(), rect2.get_height(), rect2.get_width()
        
        print(rect1.get_xerr())
        
        cx1 = x1 + width1/2
        cx2 = x2 + width2/2
        
        y = 1.1*max(y1, y2)
        props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':20,'shrinkB':20,'linewidth':10}
        plt.plot([cx1, cx2], [y, y], 'k-', lw=2)
        
    flat_list = tuple(item for sublist in p for item in sublist)

    for i1, i2, marker in sig_markers:
        rect1 = flat_list[i1]
        rect2 = flat_list[i2]
        draw_sig_marker(rect1, rect2, marker)
        
    if len(sig_markers) == 0:
        return
    
class BarCapSizer():
    def __init__(self, caps, size=1):
        self.size=size
        self.caps = caps
        self.ax = self.caps[0].axes
        self.resize()

    def resize(self):
        ppd=72./self.ax.figure.dpi
        trans = self.ax.transData.transform
        s =  ((trans((self.size,1))-trans((0,0)))*ppd)[0]
        for i,cap in enumerate(self.caps):
            cap.set_markersize(s)
            
    
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
    
