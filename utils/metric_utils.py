from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr

from medpy.metric import dc, assd, recall, precision, sensitivity, specificity, positive_predictive_value
from typing import Collection


def cv(y_pred, y_true):
    """
    Coefficient of Variation
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    cv = np.std([np.sum(y_true), np.sum(y_pred)]) / np.mean([np.sum(y_true), np.sum(y_pred)])
    return cv


def dice_score_coefficient(y_true, y_pred):
    """
    Dice Score Coefficient
    :param y_true: binary ground truth
    :param y_pred: binary prediction
    :return:
    """

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    ovlp = np.sum(y_true * y_pred)

    mu = 1e-07
    dice = (2.0 * ovlp + mu) / (np.sum(y_true) + np.sum(y_pred) + mu)

    return dice


def volumetric_overlap_error(y_pred, y_true):
    """
    Volumetric overlap error
    :param y_true:
    :param y_pred:
    :return:
    """

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_true_bool = np.asarray(y_true, dtype=np.bool)
    y_pred_bool = np.asarray(y_pred, dtype=np.bool)
    TP = np.sum(y_true_bool * y_pred_bool, axis=-1)
    FP = np.sum(~y_true_bool * y_pred_bool, axis=-1)
    FN = np.sum(y_true_bool * ~y_pred_bool, axis=-1)

    mu = 1e-07

    voe = 1 - (TP + mu) / (TP + FP + FN + mu)

    return voe


class MetricsManager():
    def __init__(self, class_names: Collection[str]):
        self.scan_names = []
        self.class_names = class_names

        # Initialize segmentation metrics
        class_seg_metrics = {}
        for n in class_names:
            class_seg_metrics[n] = SegMetricsProcessor()

    def __analyze(self, scan_name: str,  y_true: np.ndarray, y_pred: np.ndarray, voxel_spacing: tuple):
        pass


class SegMetric(Enum):
    DSC = 1, 'Dice Score Coefficient', dc
    VOE = 2, 'Volumetric Overlap Error', volumetric_overlap_error
    CV = 3, 'Coefficient of Variation', cv
    ASSD = 4, 'Average Symmetric Surface Distance', assd, True
    PRECISION = 5, 'Precision', precision
    RECALL = 6, 'Recall', recall
    SENSITIVITY = 7, 'Sensitivity', sensitivity
    SPECIFICITY = 8, 'Specificity', specificity
    PPV = 9, "Positive Predictive Value", positive_predictive_value

    def __new__(cls, keycode, full_name, func, use_voxel_spacing=False):
        obj = object.__new__(cls)
        obj._value_ = keycode
        obj.full_name = full_name
        obj.func = func
        obj.use_voxel_spacing = use_voxel_spacing
        return obj

    def compute(self, y_pred, y_true, voxel_spacing):
        if self.use_voxel_spacing:
            return self.func(y_pred, y_true, voxel_spacing)
        else:
            return self.func(y_pred, y_true)


class SegMetricsProcessor():
    # Default is to capitalize all metric names. If another name is, please specify here
    __METRICS_DISPLAY_NAMES = {SegMetric.DSC: SegMetric.DSC.name,
                               SegMetric.VOE: SegMetric.VOE.name,
                               SegMetric.CV: SegMetric.CV.name,
                               SegMetric.ASSD: 'ASSD (mm)',
                               SegMetric.PRECISION: 'Precision',
                               SegMetric.RECALL: 'Recall',
                               SegMetric.SENSITIVITY: 'Sensitivity',
                               SegMetric.SPECIFICITY: 'Specificity',
                               SegMetric.PPV: SegMetric.PPV.name}

    def __init__(self, metrics, class_names):
        """Constructor

        :param metrics: Metrics to analyze. Default is all supported metrics
        :param class_names: Class names, in order of channels in input
        """
        self.metrics = metrics
        self.class_names = class_names

        self.__scan_ids = []
        self.__scan_seg_data = dict()
        self.__data_xr = None

    def compute_metrics(self, scan_id, y_true: np.ndarray, y_pred: np.ndarray, voxel_spacing: tuple):
        """
        Compute segmentation metrics for volume
        :param scan_id
        :param y_true:
        :param y_pred:
        :param voxel_spacing: Voxel spacing (in mm)
        :return:
        """
        assert type(y_true) is np.ndarray and type(y_pred) is np.ndarray, "y_true and y_pred must be numpy arrays"
        assert y_true.shape == y_pred.shape, "Shape mismatch: y_true and y_pred must have the same shape"
        assert y_true.ndim == 3 or y_true.ndim == 4, "Arrays must be (Y,X,Z) or (Y, X, Z, #classes)"

        if y_true.ndim == 3:
            y_true = y_true[..., np.newaxis]
            y_pred = y_pred[..., np.newaxis]

        assert y_true.shape[-1] == len(self.class_names), "Expected %d classes. Got %d" % (len(self.class_names),
                                                                                           y_true.shape[-1])
        num_classes = len(self.class_names)

        metrics_data = []
        metrics_names = []
        for m in self.metrics:
            metrics_names.append(m.name)
            metrics_data.append(m.func(y_true[..., c], y_pred[..., c], voxel_spacing) for c in range(num_classes))

        metrics_data = pd.DataFrame(metrics_data, index=metrics_names, columns=self.class_names)
        metrics_data = metrics_data.T

        if scan_id in self.__scan_ids:
            raise ValueError('Scan id already exists, use different id')

        self.__scan_ids.append(scan_id)
        self.__scan_seg_data[scan_id] = metrics_data

    def rms(self, metric):
        data = np.asarray(self.metrics[metric])
        return np.sqrt(np.mean(data ** 2))

    def mean(self, metric):
        data = np.asarray(self.metrics[metric])
        return float(np.mean(data))

    def std(self, metric):
        data = np.asarray(self.metrics[metric])
        return float(np.std(data))

    def median(self, metric):
        data = np.asarray(self.metrics[metric])
        return float(np.median(data))

    def summary(self):
        s = 'Format: Mean +/- Std, Median'
        s += 'DSC: %0.4f +/- %0.3f, %0.4f\n' % (self.mean('dsc'),
                                                self.std('dsc'),
                                                self.median('dsc'))

        s += 'VOE: %0.4f +/- %0.3f, %0.4f\n' % (self.mean('voe'),
                                                self.std('voe'),
                                                self.median('voe'))

        s += 'CV (RMS):  %0.4f +/- %0.3f, %0.4f\n' % (self.rms('cv'),
                                                      self.std('cv'),
                                                      self.median('cv'))

        s += 'ASSD (mm): %0.4f +/- %0.3f, %0.4f\n' % (self.mean('assd'),
                                                      self.std('assd'),
                                                      self.median('assd'))

        s += 'Precision: %0.4f +/- %0.3f, %0.4f\n' % (self.mean('precision'),
                                                      self.std('precision'),
                                                      self.median('precision'))

        s += 'Recall: %0.4f +/- %0.3f, %0.4f\n' % (self.mean('recall'),
                                                   self.std('recall'),
                                                   self.median('recall'))
        return s

    @property
    def scan_id_data(self):
        return self.__scan_seg_data

    @property
    def data_xr(self):
        ds = xr.concat([xr.DataArray(self.__scan_seg_data[scan_id]) for scan_id in self.__scan_ids])
        return ds
