import sys

sys.path.insert(0, '../')

# %matplotlib inline
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

import utils

ALPHA = 0.05
SAVE_PATH = utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/analysis/exp_graphs')


def parse_results_file(dirpath):
    filepath = os.path.join(dirpath, 'results.txt')
    scans = dict()
    # returns mean
    with open(filepath) as search:
        for line in search:
            line = line.rstrip()  # remove '\n' at end of line
            if 'NAME' not in line.upper():
                continue

            name_extra = re.findall("name\s{1}=\s{1}([\S\s]+)", line)
            scan_id = name_extra[0].split(',')[0]

            vals = re.findall("\d+\.\d+", line)

            scans[scan_id] = [float(x) for x in vals]

    scans_keys_sorted = list(scans.keys())
    scans_keys_sorted.sort()

    dsc = []
    voe = []
    cv = []

    for sk in scans_keys_sorted:
        vals = scans[sk]
        dsc.append(vals[0])
        voe.append(vals[1])
        cv.append(vals[2])

    return np.asarray(dsc), np.asarray(voe), np.asarray(cv)


def kruskal_dunn_analysis(dirpaths, names, dirname):
    # assert len(dirpaths) >= 3, "ANOVA should be used with 3 or more populations"
    save_dirpath = os.path.join(SAVE_PATH, dirname)
    assert len(dirpaths) == len(names), '%d vs %d' % (len(dirpaths), len(names))

    metrics = {'DSC': [], 'VOE': [], 'CV': []}

    for dp in dirpaths:
        if type(dp) is list:
            c = 0
            for d_p in dp:
                if c == 0:
                    dsc, voe, cv = parse_results_file(d_p)
                else:
                    dsc1, voe1, cv1 = parse_results_file(d_p)
                    dsc = np.add(dsc, dsc1)
                    voe = np.add(voe, voe1)
                    cv = np.add(cv, cv1)
                c += 1
            print("asd: " + str(dsc.shape))
            dsc = dsc / 3
            voe = voe / 3
            cv = cv / 3
        else:
            dsc, voe, cv = parse_results_file(dp)

        metrics['DSC'].append(dsc)
        metrics['VOE'].append(voe)
        metrics['CV'].append(cv)

    metrics_results = dict()
    for k in metrics.keys():
        vals = np.transpose(np.stack(metrics[k]))
        df = pd.DataFrame(data=vals, columns=names)
        # print(df.values.shape)
        plt.figure()
        ax = plt.gca()
        bxplt = df.boxplot(column=names, ax=ax)
        # sns.boxplot(column=names, ax=ax)
        ax.set_title(k)
        utils.check_dir(save_dirpath)
        plt.savefig(os.path.join(save_dirpath, '%s.png' % k), format='png', dpi=1000, bbox_inches='tight')

        metrics_results[k] = kruskal_dunn(metrics[k], names)

    for k in ['DSC', 'VOE', 'CV']:
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

    if data['dunn'] is not None:
        print('Dunn: ')
        print(data['dunn'])


if __name__ == '__main__':
    # Base unet - best performing network
    BASE_UNET = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/unet_2d/original_akshaysc/test_results'
    ARCH_UNET = BASE_UNET
    ARCH_SEGNET = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/segnet_2d/2018-09-01-22-39-39/fine_tune/test_results'
    ARCH_DEEPLAB = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-09-26-19-07-53/fine_tune/test_results/16_2-4-6'
    names = ['U-Net', 'SegNet', 'DLV3+']
    dirpaths = [ARCH_UNET, ARCH_SEGNET, ARCH_DEEPLAB]

    kruskal_dunn_analysis(dirpaths, names)
