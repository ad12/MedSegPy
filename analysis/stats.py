import sys
sys.path.insert(0, '../')

import os
import re
import numpy as np


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


def anova(dirpaths, names, analysis_filepath):
    assert len(dirpaths) >= 3, "ANOVA should be used with 3 or more populations"
    assert len(dirpaths) == len(names)

    metrics = {'DSC': [], 'VOE': [], 'CV': []}

    for dp in dirpaths:
        dsc, voe, cv = parse_results_file(dp)
        metrics['DSC'].append(dsc)
        metrics['VOE'].append(voe)
        metrics['CV'].append(cv)

    for k in metrics.keys():
        a = np.stack(metrics[k])
        metrics[k] = a
        print(a.shape)





if __name__ == '__main__':
    parse_results_file('../test_data')