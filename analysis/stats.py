import sys

sys.path.insert(0, '../')
from analysis.analysis_utils import get_metrics, get_metrics_v2

import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

from utils import io_utils

from analysis.analysis_funcs import cpal
from analysis import analysis_funcs as af

from scipy import optimize as sop

ALPHA = 0.05
SAVE_PATH = io_utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/analysis/exp_graphs')


def compare_metrics_v2(dirpaths, exp_names, save_directory: str, metrics=('dsc', 'voe', 'cv', 'assd')):
    """
    Plot and save bar graph comparing specified metrics for specified experiments

    :param dirpaths: iterable of directory paths were test_result information is stored
    :param exp_names: names of experiments corresponding to directory paths
    :param save_directory: directory to save bar graph
    :param metrics: metrics to plot
    """
    # n_groups = len(x_labels)
    # x_index = np.arange(0, n_groups * 2, 2)

    exp_filepath = os.path.join(SAVE_PATH, save_directory, 'bar.png')

    metrics_dict = get_metrics_v2(dirpaths)

    exp_means = []
    exp_stds = []
    for ind in range(len(exp_names)):
        sub_means = []
        sub_stds = []
        for metric in metrics:
            exp_vals = metrics_dict[metric]
            vals = np.asarray(exp_vals[ind])
            sub_means.append(np.mean(vals))
            std = np.std(vals) if len(vals) > 1 else None
            sub_stds.append(std)

        exp_means.append(sub_means)
        exp_stds.append(sub_stds)

    exp_means = pd.DataFrame(exp_means, index=exp_names, columns=metrics).T
    exp_stds = pd.DataFrame(exp_stds, index=exp_names, columns=metrics).T

    # Display bar graph
    af.display_bar_graph(exp_means, exp_stds, exp_filepath=exp_filepath, legend_loc='best', bar_width=0.35)


def compare_metrics(dirpaths, names, dirname):
    raise DeprecationWarning('`compare_metrics` is deprecated. Use `compare_metrics_v2` to compare metrics')
    x_labels = ('dsc', 'voe', 'cv', 'assd')
    n_groups = len(x_labels)
    x_index = np.arange(0, n_groups * 2, 2)

    exp_names = names
    exp_filepath = os.path.join(SAVE_PATH, dirname, 'bar.png')

    metrics_dict = get_metrics(dirpaths)

    # Create figure
    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.8

    for ind in range(len(exp_names)):
        sub_means = []
        sub_stds = []
        for metric in x_labels:
            exp_vals = metrics_dict[metric]
            vals = np.asarray(exp_vals[ind])
            sub_means.append(np.mean(vals))
            std = np.std(vals) if len(vals) > 1 else None
            sub_stds.append(std)

        rects = plt.bar(x_index + (bar_width) * ind, sub_means, bar_width,
                        alpha=opacity,
                        color=cpal[ind],
                        label=exp_names[ind],
                        yerr=sub_stds)

    delta = (len(names) - 1) * bar_width / 2
    plt.xticks(x_index + delta, x_labels)
    plt.legend()

    plt.savefig(exp_filepath, format='png',
                dpi=1000,
                bbox_inches='tight',
                transparent=True)


def kruskal_dunn_analysis(dirpaths, names, metrics=('dsc', 'voe', 'cv', 'assd')):
    assert len(dirpaths) == len(names), '%d vs %d' % (len(dirpaths), len(names))

    metrics = get_metrics_v2(dirpaths)

    metrics_results = dict()
    for k in metrics.keys():
        # vals = np.transpose(np.stack(metrics[k]))
        # df = pd.DataFrame(data=vals, columns=names)
        metrics_results[k] = kruskal_dunn(metrics[k], names)

    for k in metrics:
        print_results(metrics_results[k], k)
        print('')


def kruskal_dunn(data, names):
    assert len(data) == len(names)

    results = dict()
    f, p = stats.kruskal(*data)

    results['f'] = f
    results['p'] = p
    results['dunn'] = None

    if p > ALPHA:
        return results

    # if significant, find where we have significance
    dunn_results = sp.posthoc_dunn(data, p_adjust='bonferroni')

    df = pd.DataFrame(dunn_results, columns=names, index=names)

    results['dunn'] = df
    results['dunn-h'] = pd.DataFrame(dunn_results <= ALPHA, columns=names, index=names)

    return results


def print_results(data, metric):
    print('===================')
    print(metric)
    print('===================')
    print('F-value: %0.4f' % data['f'])
    print('p-value: %0.4f' % data['p'])

    def highlight_significant(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        strings, black otherwise.
        """
        bg_color = 'yellow' if abs(val) < ALPHA else ''
        return 'background-color: %s' % bg_color

    if data['dunn'] is not None:
        print('Dunn: ')
        s = data['dunn'].style.applymap(highlight_significant)
        display(s)


def fit(x, y, func, p0):
    popt, _ = sop.curve_fit(func, x, y, p0=p0, maxfev=3000)

    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    print(ss_res)
    print(ss_tot)

    r_squared = 1 - (ss_res / (ss_tot + 1e-8))

    return popt, r_squared


if __name__ == '__main__':
    # Base unet - best performing network
    BASE_UNET = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/unet_2d/original_akshaysc/test_results'
    ARCH_UNET = BASE_UNET
    ARCH_SEGNET = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/segnet_2d/2018-09-01-22-39-39/fine_tune/test_results'
    ARCH_DEEPLAB = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-09-26-19-07-53/fine_tune/test_results/16_2-4-6'
    names = ['U-Net', 'SegNet', 'DLV3+']
    dirpaths = [ARCH_UNET, ARCH_SEGNET, ARCH_DEEPLAB]

    kruskal_dunn_analysis(dirpaths, names)
