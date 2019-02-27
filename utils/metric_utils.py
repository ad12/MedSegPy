import numpy as np
from medpy.metric import dc, assd, recall, precision


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


class MetricWrapper():

    def __init__(self):
        self.metrics = {'dsc': [], 'voe': [], 'cv': [], 'assd': [], 'precision': [], 'recall': []}

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, voxel_spacing: tuple):
        assert type(y_true) is np.ndarray and type(y_pred) is np.ndarray, "y_true and y_pred must be numpy arrays"
        assert y_true.shape == y_pred.shape, "Shape mismatch: y_true and y_pred must have the same shape"
        assert len(y_true.shape) == len(
            voxel_spacing), "Shape mismatch: voxel spacing must be specified for each dimension"

        self.metrics['dsc'].append(dc(y_pred, y_true))
        self.metrics['voe'].append(volumetric_overlap_error(y_true, y_pred))
        self.metrics['cv'].append(cv(y_true, y_pred))
        self.metrics['assd'].append(assd(y_pred, y_true, voxelspacing=voxel_spacing))
        self.metrics['precision'].append(precision(y_pred, y_true))
        self.metrics['recall'].append(recall(y_pred, y_true))

    def rms(self, metric):
        data = np.asarray(self.metrics[metric])
        return np.sqrt(np.mean(data ** 2))

    def mean(self, metric):
        data = np.asarray(self.metrics[metric])
        return np.mean(data)

    def std(self, metric):
        data = np.asarray(self.metrics[metric])
        return np.std(data)

    def median(self, metric):
        data = np.asarray(self.metrics[metric])
        return np.median(data)

    def summary(self):
        s = ''
        s += 'DSC - Mean +/- Std, Median = %0.4f +/- %0.3f, %0.4f\n' % (self.mean('dsc'),
                                                                        self.std('dsc'),
                                                                        self.median('dsc'))

        s += 'VOE - Mean +/- Std, Median = %0.4f +/- %0.3f, %0.4f\n' % (self.mean('voe'),
                                                                        self.std('voe'),
                                                                        self.median('voe'))

        s += 'CV - RMS +/- Std, Median = %0.4f +/- %0.3f, %0.4f\n' % (self.rms('cv'),
                                                                      self.std('cv'),
                                                                      self.median('cv'))

        s += 'ASSD - Mean +/- Std, Median = %0.4f +/- %0.3f, %0.4f\n' % (self.mean('assd'),
                                                                         self.std('assd'),
                                                                         self.median('assd'))

        s += 'Precision - Mean +/- Std, Median = %0.4f +/- %0.3f, %0.4f\n' % (self.mean('precision'),
                                                                              self.std('precision'),
                                                                              self.median('precision'))

        s += 'Recall - Mean +/- Std, Median = %0.4f +/- %0.3f, %0.4f\n' % (self.mean('recall'),
                                                                           self.std('recall'),
                                                                           self.median('recall'))
        return s
