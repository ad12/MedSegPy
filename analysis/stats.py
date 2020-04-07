import numpy as np

from medsegpy.utils import io_utils

from analysis_utils import get_metrics

from pystats.pystats import stats as pyss

from scipy import optimize as sop

SAVE_PATH = io_utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/analysis/exp_graphs')


def kruskal_dunn_analysis(dirpaths, names, metrics=('dsc', 'voe', 'cv', 'assd'), suppress_output=False):
    assert len(dirpaths) == len(names), '%d vs %d' % (len(dirpaths), len(names))

    metrics_data = get_metrics(dirpaths)

    metrics_results = dict()
    for k in metrics:
        metrics_results[k] = pyss.kruskal_wallis(metrics_data[k], posthoc_test='dunn', names=names, p_adjust='bonferroni')
    
    pmats = []
    for k in metrics:
        if not suppress_output:
            print('%s:' % k)
            display(metrics_results[k]['dunn'])
            print('')
            
        pmats.append(metrics_results[k]['dunn'])
    
    return pmats

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

