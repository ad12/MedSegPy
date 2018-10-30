from abc import ABC

import numpy as np


class ScanMetadata():
    def __init__(self, scan_id, slice_dir=None, kl_grade=None):
        self.scan_id = scan_id
        self.slice_dir = slice_dir
        self.kl_grade = kl_grade

        self.cv = None
        self.dsc = None
        self.voe = None


class Metrics():
    def __init__(self, name, vals_arr=None):
        self.name = name
        self.vals_arr = vals_arr

        assert(np.sum(np.isfinite(vals_arr)) == len(vals_arr))

        self.mean = np.mean(vals_arr)
        self.std = np.std(vals_arr)
        self.median = np.median(vals_arr)
        self.SEM = self.std / np.sqrt(len(vals_arr))

    def to_string(self):
        return '%'
