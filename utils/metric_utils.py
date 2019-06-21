import numpy as np
import scipy.stats as spstats
import pandas as pd
import tabulate

from medpy.metric import dc, assd, recall, precision, sensitivity, specificity, positive_predictive_value
from typing import Collection
from enum import Enum

def cv(y_true, y_pred):
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


def volumetric_overlap_error(y_true, y_pred):
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


class SegMetric(Enum):
    DSC = 1, 'Dice Score Coefficient', dc, False
    VOE = 2, 'Volumetric Overlap Error', volumetric_overlap_error, False
    CV = 3, 'Coefficient of Variation', cv, False
    ASSD = 4, 'Average Symmetric Surface Distance', assd, True
    PRECISION = 5, 'Precision', precision, False
    RECALL = 6, 'Recall', recall, False
    SENSITIVITY = 7, 'Sensitivity', sensitivity, False
    SPECIFICITY = 8, 'Specificity', specificity, False
    PPV = 9, "Positive Predictive Value", positive_predictive_value, False

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


class MetricOperation(Enum):
    MEAN = 1, lambda x, **kwargs: np.mean(x, **kwargs)
    MEDIAN = 2, lambda x, **kwargs: np.median(x, **kwargs)
    RMS = 3, lambda x, **kwargs: np.sqrt(np.mean(x ** 2, **kwargs))

    def __new__(cls, keycode, func):
        obj = object.__new__(cls)
        obj._value_ = keycode
        obj.func = func
        return obj

    def compute(self, x:np.ndarray, **kwargs):
        return self.func(np.asarray(x), **kwargs)


class MetricError(Enum):
    STANDARD_DEVIATION = 1, lambda x, **kwargs: np.std(x, **kwargs)
    STANDARD_ERROR = 2, lambda x, **kwargs: __sem__(x, **kwargs)

    def __new__(cls, keycode, func):
        obj = object.__new__(cls)
        obj._value_ = keycode
        obj.func = func
        return obj

    def compute(self, x:np.ndarray, **kwargs):
        return self.func(np.asarray(x), **kwargs)


def __sem__(x, **kwargs):
    args = {'axis': 0, 'ddof': 0}
    args.update(**kwargs)
    return spstats.sem(x, **args)


class MetricsManager():
    __METRICS = [SegMetric.DSC, SegMetric.VOE, SegMetric.ASSD, SegMetric.CV]

    def __init__(self, class_names: Collection[str], metrics=__METRICS):
        self.scan_names = []
        self.class_names = class_names

        self.__seg_metrics_processor = SegMetricsProcessor(metrics, class_names)
        self.runtimes = []

    def analyze(self, scan_name: str,  y_true: np.ndarray, y_pred: np.ndarray, voxel_spacing: tuple,
                runtime: float=np.nan):
        self.scan_names.append(scan_name)
        summary = self.__seg_metrics_processor.compute_metrics(scan_name, y_true, y_pred, voxel_spacing)

        self.runtimes.append(runtime)

        return summary

    @property
    def data(self):
        return {'scan_ids': self.scan_names,
                'runtimes': self.runtimes,
                'seg_metrics': self.__seg_metrics_processor.data}

    @property
    def seg_metrics_processor(self):
        return self.__seg_metrics_processor


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

    __DEFAULT_METRICS_OPERATIONS = {SegMetric.CV: MetricOperation.RMS}

    def __init__(self, metrics, class_names,
                 metrics_to_operations = __DEFAULT_METRICS_OPERATIONS,
                 error_metric = MetricError.STANDARD_DEVIATION):
        """Constructor

        :param metrics: Metrics to analyze. Default is all supported metrics
        :param class_names: Class names, in order of channels in input
        """
        self.metrics = metrics
        self.class_names = class_names

        self.__scan_ids = []
        self.__scan_seg_data = dict()
        self.__data = dict()
        self.__is_data_stale = False

        # Default operations (mean, RMS, Median, etc) and error (std. dev., SEM, etc.) to use per metric
        self.metrics_to_operations = metrics_to_operations
        self.error_metric = error_metric

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
            metrics_names.append(self.__METRICS_DISPLAY_NAMES[m])
            metrics_data.append([m.compute(y_true[..., c], y_pred[..., c], voxel_spacing) for c in range(num_classes)])

        metrics_data = pd.DataFrame(metrics_data, index=metrics_names, columns=self.class_names)

        if scan_id in self.__scan_ids:
            raise ValueError('Scan id already exists, use different id')

        self.__scan_ids.append(scan_id)
        self.__scan_seg_data[scan_id] = metrics_data
        self.__is_data_stale = True

        return self.scan_summary(scan_id)

    def scan_summary(self, scan_id):
        scan_data = self.__scan_seg_data[scan_id]
        avg_data = scan_data.mean(axis=1)

        metrics = avg_data.index.tolist()

        summary_str_format = '%s: %0.3f, ' * len(metrics)
        summary_str_format = summary_str_format[:-2]

        data = []
        for name in avg_data.index.tolist():
            data.extend([name, avg_data[name]])
        
        return summary_str_format % (tuple(data))

    def summary(self):
        data = self.data
        arr = []
        names = []
        for metric_name in data.keys():
            names.append(metric_name)
            arr.append(np.asarray(data[metric_name].mean()).flatten())

        df = pd.DataFrame(arr, index=names, columns=self.class_names)
        return tabulate.tabulate(df, headers=self.class_names) + "\n"

    @property
    def scan_id_data(self):
        return self.__scan_seg_data

    @property
    def data(self):
        if self.__is_data_stale:
            self.__refresh_data()

        return self.__data

    def __refresh_data(self):
        # create array with dimensions subjects, classes, metrics
        arr = np.stack([np.asarray(self.__scan_seg_data[scan_id]) for scan_id in self.__scan_ids], axis=-1)
        arr = arr.transpose((2, 1, 0))

        data = dict()
        for ind, m in enumerate(self.metrics):
            data[self.__METRICS_DISPLAY_NAMES[m]] = pd.DataFrame(arr[..., ind],
                                                                 index=self.__scan_ids,
                                                                 columns=self.class_names)

        self.__data = data


if __name__ == '__main__':
    a = np.asarray([[1,2,3,4], [5,7,9,11]])
    print(__sem__(a, axis=0))
    print(np.std(a, axis=0) / np.sqrt(2))

    df = pd.DataFrame(a, index=['aplha', 'b'])
    print(df)
    print(df.mean(axis=1)['b'])
