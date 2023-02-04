"""Stanford qDESS dataset"""
import dosma as dm
import h5py
import json
import numpy as np
import os
import pandas as pd

from dosma.scan_sequences.mri import QDess
from dosma.tissues import FemoralCartilage, TibialCartilage, PatellarCartilage, Meniscus
from medsegpy.data.datasets.qdess_mri import QDESS_SEGMENTATION_CATEGORIES
from medsegpy.evaluation import EVALUATOR_REGISTRY, SemSegEvaluator
from medsegpy.evaluation.metrics import Metric, Reductions, flatten_non_category_dims


class T2Relaxation(Metric):
    """Compute T2 relaxation time for each tissue.
    """
    _METADATA_PATH = "/dataNAS/people/arjun/data/skm-tea/v1-release/all_metadata.csv"
    _BASE_SCAN_DIR = "/dataNAS/people/arjun/data/skm-tea/v1-release/image_files"
    _TEST_ANNOTATION_PATH = "/bmrNAS/people/arjun/data/qdess_knee_2020/annotations/v0.0.1/test.json"

    def __init__(self, method, units=""):
        super().__init__(units)
        assert method in ["pred", "base"]
        self._method = method

        # Get specific image metadata for all scans in the test set
        with open(self._TEST_ANNOTATION_PATH, 'r') as fp:
            test_metadata = json.load(fp)
        self.scan_to_metadata = {
            img_dict["scan_id"]: img_dict
            for img_dict in test_metadata["images"]
        }

        # Get additional metadata for all scans
        self.additional_metadata = pd.read_csv(self._METADATA_PATH)

    def _compute_t2_relaxation(self, y, scan_id, category_dim: int = None):
        # Convert mask to binary
        y_bin = y.astype(np.bool)

        # Get scan metadata
        scan_metadata = self.scan_to_metadata[scan_id]

        # Build affine matrix
        affine = dm.to_affine(scan_metadata["orientation"],
                              scan_metadata["voxel_spacing"])

        # Make qDESS object
        scan_path = os.path.join(self._BASE_SCAN_DIR, f"{scan_id}.h5")
        with h5py.File(scan_path, 'r') as fp:
            e1 = dm.MedicalVolume(fp["echo1"][()], affine)
            e2 = dm.MedicalVolume(fp["echo2"][()], affine)
        qdess = QDess([e1, e2])

        # Determine if left or right side corresponds to the medial direction
        scan_additional_metadata = self.additional_metadata[
            self.additional_metadata["MTR_ID"] == scan_id
        ]
        is_left_medial = (scan_additional_metadata["MedialDirection"] == "L").item()

        # Make tissue objects
        fc = FemoralCartilage(medial_to_lateral=is_left_medial)
        tc = TibialCartilage(medial_to_lateral=is_left_medial)
        pc = PatellarCartilage(medial_to_lateral=is_left_medial)
        men = Meniscus(medial_to_lateral=is_left_medial)

        # Set mask for each tissue
        category_indices = {
            x["abbrev"]: idx
            for idx, x in enumerate(QDESS_SEGMENTATION_CATEGORIES)
        }
        fc.set_mask(dm.MedicalVolume(y_bin[..., category_indices["fc"]], affine),
                    use_largest_cc=True)
        tc.set_mask(dm.MedicalVolume(y_bin[..., category_indices["tc"]], affine),
                    use_largest_ccs=True)
        pc.set_mask(dm.MedicalVolume(y_bin[..., category_indices["pc"]], affine),
                    use_largest_cc=True)
        men.set_mask(dm.MedicalVolume(y_bin[..., category_indices["men"]], affine),
                     use_largest_ccs=True)

        # Calculate T2 relaxation time for femoral cartilage
        fc_t2_map = qdess.generate_t2_map(
            tissue=fc,
            suppress_fat=True,
            suppress_fluid=True,
            gl_area=float(scan_additional_metadata["SpoilerGradientArea"]),
            tg=float(scan_additional_metadata["SpoilerGradientTime"]),
            tr=float(scan_additional_metadata["RepetitionTime"]),
            te=float(scan_additional_metadata["EchoTime1"]),
            alpha=float(scan_additional_metadata["FlipAngle"]),
            t1=1200,
            nan_bounds=(0, 100)
        )

        # Calculate T2 relaxation time for tibial cartilage
        tc_t2_map = qdess.generate_t2_map(
            tissue=tc,
            suppress_fat=True,
            suppress_fluid=True,
            gl_area=float(scan_additional_metadata["SpoilerGradientArea"]),
            tg=float(scan_additional_metadata["SpoilerGradientTime"]),
            tr=float(scan_additional_metadata["RepetitionTime"]),
            te=float(scan_additional_metadata["EchoTime1"]),
            alpha=float(scan_additional_metadata["FlipAngle"]),
            t1=1200,
            nan_bounds=(0, 100)
        )

        # Calculate T2 relaxation time for patellar cartilage
        pc_t2_map = qdess.generate_t2_map(
            tissue=pc,
            suppress_fat=True,
            suppress_fluid=True,
            gl_area=float(scan_additional_metadata["SpoilerGradientArea"]),
            tg=float(scan_additional_metadata["SpoilerGradientTime"]),
            tr=float(scan_additional_metadata["RepetitionTime"]),
            te=float(scan_additional_metadata["EchoTime1"]),
            alpha=float(scan_additional_metadata["FlipAngle"]),
            t1=1200,
            nan_bounds=(0, 100)
        )

        # Calculate T2 relaxation time for meniscus
        men_t2_map = qdess.generate_t2_map(
            tissue=men,
            suppress_fat=True,
            suppress_fluid=True,
            gl_area=float(scan_additional_metadata["SpoilerGradientArea"]),
            tg=float(scan_additional_metadata["SpoilerGradientTime"]),
            tr=float(scan_additional_metadata["RepetitionTime"]),
            te=float(scan_additional_metadata["EchoTime1"]),
            alpha=float(scan_additional_metadata["FlipAngle"]),
            t1=1200,
            nan_bounds=(0, 100)
        )

        # Return mean T2 relaxation time for each tissue
        mean_t2_vals = [0.0, 0.0, 0.0, 0.0]
        mean_t2_vals[category_indices["fc"]] = np.mean(
            fc_t2_map.volumetric_map.volume[y_bin[..., category_indices["fc"]]]
        )
        mean_t2_vals[category_indices["tc"]] = np.mean(
            tc_t2_map.volumetric_map.volume[y_bin[..., category_indices["tc"]]]
        )
        mean_t2_vals[category_indices["pc"]] = np.mean(
            pc_t2_map.volumetric_map.volume[y_bin[..., category_indices["pc"]]]
        )
        mean_t2_vals[category_indices["men"]] = np.mean(
            men_t2_map.volumetric_map.volume[y_bin[..., category_indices["men"]]]
        )

        return np.array(mean_t2_vals)

    def __call__(self, y_pred, y_true, scan_id, category_dim: int = None):
        # Ensure category_dim is -1
        assert category_dim == -1

        # Compute T2 relaxation time
        if self._method == "pred":
            t2_time = self._compute_t2_relaxation(y_pred, scan_id, category_dim)
        else:
            t2_time = self._compute_t2_relaxation(y_true, scan_id, category_dim)

        return (t2_time,)

    def name(self):
        return f"T2 {self._method}"


class TissueVolume(Metric):
    def __init__(self, method, units=""):
        super().__init__(units)
        assert method in ["pred", "base"]
        self._method = method

    def _compute_volume(self, y, pixel_volume, category_dim):
        y = flatten_non_category_dims(y.astype(np.bool), category_dim)
        return pixel_volume * np.count_nonzero(y, -1)

    def __call__(self, y_pred, y_true, spacing=None, category_dim: int = None):
        # Ensure category_dim is -1
        assert category_dim == -1

        # Get volume of each pixel
        pixel_volume = np.prod(spacing) if spacing else 1

        # Compute volume of tissue
        if self._method == "pred":
            tissue_volume = self._compute_volume(y_pred, pixel_volume, category_dim)
        else:
            tissue_volume = self._compute_volume(y_true, pixel_volume, category_dim)

        return (tissue_volume,)

    def name(self):
        return f"Volume {self._method}"


class T2RelaxationDiff(Metric):
    """Compare T2 Relaxation times using predicted segmentation
    and ground truth segmentation
    """

    def __init__(self, method, units=""):
        super().__init__(units)
        assert method in ["absolute", "percent"]
        self._method = method

        self._t2_pred = T2Relaxation(method="pred")
        self._t2_true = T2Relaxation(method="base")

    def __call__(self, y_pred, y_true, scan_id, category_dim: int = None):
        # Ensure category_dim is -1
        assert category_dim == -1

        # Calculate T2 relaxation times
        t2_time_pred = self._t2_pred(y_pred, y_true, scan_id, category_dim)
        t2_time_true = self._t2_true(y_pred, y_true, scan_id, category_dim)

        # Determine difference
        vals = t2_time_pred[0] - t2_time_true[0]
        if self._method == "percent":
            vals = (vals / t2_time_true[0]) * 100

        return (vals,)

    def name(self):
        symbol = "Abs" if self._method == "absolute" else "%"
        return f"T2 {symbol}Diff"


class TissueVolumeDiff(Metric):
    def __init__(self, method, units=""):
        super().__init__(units)
        assert method in ["absolute", "percent"]
        self._method = method

        self._volume_pred = TissueVolume(method="pred", units="(mm^3)")
        self._volume_true = TissueVolume(method="base", units="(mm^3)")

    def __call__(self, y_pred, y_true, spacing=None, category_dim: int = None):
        # Ensure category_dim is -1
        assert category_dim == -1

        # Calculate tissue volumes
        volume_pred = self._volume_pred(y_pred, y_true, spacing, category_dim)
        volume_true = self._volume_true(y_pred, y_true, spacing, category_dim)

        # Determine difference
        vals = volume_pred[0] - volume_true[0]
        if self._method == "percent":
            vals = (vals / volume_true[0]) * 100

        return (vals,)

    def name(self):
        symbol = "Abs" if self._method == "absolute" else "%"
        return f"Volume {symbol}Diff"


@EVALUATOR_REGISTRY.register()
class QDESSEvaluator(SemSegEvaluator):
    """Evaluate semantic segmentation on the qDESS dataset.
    """
    def _get_metrics(self):
        metrics = list(super()._get_metrics())
        paired_metrics = [
            T2Relaxation(method="pred"),
            T2Relaxation(method="base"),
            TissueVolume(method="pred", units="(mm^3)"),
            TissueVolume(method="base", units="(mm^3)")
        ]
        metrics.extend([
            T2RelaxationDiff(method="absolute"),
            T2RelaxationDiff(method="percent"),
            TissueVolumeDiff(method="absolute", units="(mm^3)"),
            TissueVolumeDiff(method="percent")
        ])
        metrics.extend(paired_metrics)

        self._metric_pairs = [
            (p.name(), b.name())
            for p, b in zip(paired_metrics[::2], paired_metrics[1::2])
        ]
        return metrics

    def reset(self):
        super().reset()
        for pred_key, base_key in self._metric_pairs:
            self._metrics_manager.register_pairs(
                pred_key,
                base_key,
                (Reductions.RMS_CV, Reductions.RMSE_CV,)
            )

    def process(self, inputs, outputs):
        if not self.spacing:
            assert all("scan_spacing" in x for x in inputs)
        super().process(inputs, outputs)
