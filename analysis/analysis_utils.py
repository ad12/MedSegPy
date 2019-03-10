import os, sys
import re

import deprecated
import numpy as np

sys.path.insert(0, '../')
from utils import io_utils


def parse_results_file(dirpath):
    raise DeprecationWarning('`parse_results_file` is deprecated. Use `load_metrics` to load relevant metrics')

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


def load_metrics(dirpath):
    filepath = os.path.join(dirpath, 'metrics.dat')
    metrics = io_utils.load_pik(filepath)
    for k in metrics.keys():
        metrics[k] = np.asarray(metrics[k])

    return metrics


def get_metrics(dirpaths):
    raise DeprecationWarning('`get_metrics` is deprecated. Use `get_metrics_v2` to load relevant metrics')
    metrics = {'dsc': [], 'voe': [], 'cv': []}

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
            dsc = dsc / len(dp)
            voe = voe / len(dp)
            cv = cv / len(dp)
        else:
            dsc, voe, cv = parse_results_file(dp)

        metrics['dsc'].append(dsc)
        metrics['voe'].append(voe)
        metrics['cv'].append(cv)

    return metrics


def get_metrics_v2(dirpaths):
    metrics = {'dsc': [], 'voe': [], 'cv': [], 'assd': [], 'precision': [], 'recall': []}

    for dp in dirpaths:
        exp_metrics = dict()
        if type(dp) is list:
            c = 0
            for d_p in dp:
                if c == 0:
                    exp_metrics = load_metrics(d_p)
                else:
                    em = load_metrics(d_p)
                    for k in em.keys():
                        exp_metrics[k] += em[k]
                c += 1
            for k in exp_metrics.keys():
                exp_metrics[k] = exp_metrics[k] / len(dp)
        else:
            exp_metrics = load_metrics(dp)

        for k in metrics.keys():
            metrics[k].append(exp_metrics[k])

    return metrics