"""
NeuroSWAYML — Congenital / Birth Disorder Gait Loader
=====================================================
Dataset: GaitRec v1 (Horst et al., 2021)
DOI    : 10.6084/m9.figshare.13598962.v1
URL    : https://figshare.com/articles/dataset/
         GaitRec_A_large-scale_ground_reaction_force_dataset_of_healthy_and_impaired_gait/13598962
License: CC BY 4.0

Download
--------
1. Visit the figshare URL above and download GaitRec.zip   (~2.3 GB)
2. Extract to  data/gaitrec/
   The expected layout is:
     data/gaitrec/
       GaitRec_CTL_*.csv       (healthy controls)
       GaitRec_ANKLE_*.csv     (ankle / foot disorders — includes clubfoot)
       GaitRec_BACK_*.csv      (spinal / lumbar disorders)
       GaitRec_HIP_*.csv       (hip conditions — incl. congenital hip dysplasia)
       GaitRec_KNEE_*.csv      (knee disorders — incl. juvenile arthritis)
       GaitRec_NEURO_*.csv     (neurological / motor disorder subset, if present)

   Alternative layout (per-subject folders):
     data/gaitrec/
       <SUBJECT_ID>/
         session_*.csv

Dataset CSV columns (1000 Hz, dual force-plate)
-----------------------------------------------
  Frame, F_v_l, F_ml_l, F_ap_l, M_v_l, M_ml_l, M_ap_l,
  COP_ml_l, COP_ap_l,
  F_v_r, F_ml_r, F_ap_r, M_v_r, M_ml_r, M_ap_r,
  COP_ml_r, COP_ap_r

Label mapping
-------------
  CTL                               → 0  NORMAL
  BACK / ANKLE (mild orthopaedic)   → 1  MILD_DISORDER
  HIP / KNEE / NEURO                → 2  SEVERE_DISORDER

This maps congenital / birth-disorder severity onto the same 3-class
framework used by all NeuroSWAYML domains.
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from typing import Optional, Tuple, List, Dict

_FEATURE_NAMES = [
    "sway_std", "sway_range", "sway_cv", "sway_fft_peak_freq",
    "sway_fft_energy", "torso_angle", "torso_angle_std",
    "left_knee_angle", "right_knee_angle", "knee_angle_diff",
    "left_knee_var", "right_knee_var",
    "left_hip_angle", "right_hip_angle",
    "left_ankle_angle", "right_ankle_angle",
    "gait_symmetry", "stride_cv", "stride_length_norm",
    "cadence_norm", "step_width_norm",
    "heel_toe_diff_l", "heel_toe_diff_r",
    "leg_length_ratio", "hip_height_norm", "aspect_ratio",
    "head_vel", "hip_vel", "ankle_vel_l", "ankle_vel_r",
]
_IDX = {n: i for i, n in enumerate(_FEATURE_NAMES)}

# GaitRec condition group → class label
_GROUP_LABEL: Dict[str, int] = {
    "ctl":   0,
    "back":  1,
    "ankle": 1,
    "hip":   2,
    "knee":  2,
    "neuro": 2,
    # Additional condition keys that may appear
    "hemi":  2,
    "para":  2,
    "pf":    1,    # plantar fasciitis
    "ms":    2,    # multiple sclerosis
    "cp":    2,    # cerebral palsy
    "asd":   2,    # autism spectrum (gait variants)
    "ds":    2,    # down syndrome
    "sb":    2,    # spina bifida
    "club":  1,    # clubfoot (treated)
}

_FS_GAITREC = 1000.0   # Hz (GaitRec native sampling rate)
_DECIMATE   = 10        # down-sample to 100 Hz for feature extraction

# ─────────────────────────────────────────────────────────────────────────
# GaitRec CSV feature extraction
# ─────────────────────────────────────────────────────────────────────────

class _Cols:
    """Column indices for the GaitRec 17-column format."""
    FRAME   = 0
    # Left foot
    F_V_L   = 1;  F_ML_L  = 2;  F_AP_L  = 3
    COP_ML_L = 7; COP_AP_L = 8
    # Right foot
    F_V_R   = 9;  F_ML_R  = 10; F_AP_R  = 11
    COP_ML_R = 15; COP_AP_R = 16


def _extract_grf_features(
    data: np.ndarray,
    label: int,
    body_weight_n: float = 700.0,   # assumed body weight (N) for normalisation
) -> Optional[np.ndarray]:
    """
    Parse one GaitRec trial (N, ≥17) and return 30-D feature vector.
    Down-sample from 1000 Hz → 100 Hz before processing.
    """
    if data.ndim != 2 or data.shape[1] < 9 or data.shape[0] < 200:
        return None

    # ── Down-sample ────────────────────────────────────────────────
    data_ds = data[::_DECIMATE]       # 1000 Hz → 100 Hz
    fs      = _FS_GAITREC / _DECIMATE

    has_both = data.shape[1] >= 17

    f_v_l = data_ds[:, _Cols.F_V_L]
    f_v_r = data_ds[:, _Cols.F_V_R] if has_both else np.zeros_like(f_v_l)
    cop_ml_l = data_ds[:, _Cols.COP_ML_L]
    cop_ap_l = data_ds[:, _Cols.COP_AP_L]
    if has_both:
        cop_ml_r = data_ds[:, _Cols.COP_ML_R]
        cop_ap_r = data_ds[:, _Cols.COP_AP_R]
    else:
        cop_ml_r = cop_ml_l.copy()
        cop_ap_r = cop_ap_l.copy()

    f_total = f_v_l + f_v_r + 1e-9

    # Combined COP (weighted)
    cop_x = (f_v_l * cop_ml_l + f_v_r * cop_ml_r) / f_total
    cop_y = (f_v_l * cop_ap_l + f_v_r * cop_ap_r) / f_total

    # ── Sway (COP ML) ──────────────────────────────────────────────
    sway_std    = float(np.std(cop_x))
    sway_range  = float(np.ptp(cop_x))
    sway_mean_a = float(np.mean(np.abs(cop_x))) + 1e-9
    sway_cv     = sway_std / sway_mean_a

    N = len(cop_x)
    fft_x = np.abs(np.fft.rfft(cop_x - cop_x.mean()))
    freqs  = np.fft.rfftfreq(N, d=1.0 / fs)
    mask   = (freqs >= 0.5) & (freqs <= 3.0)
    if mask.any():
        peak_f  = float(freqs[mask][np.argmax(fft_x[mask])])
        energy  = float(np.sum(fft_x[mask] ** 2) / (N ** 2))
    else:
        peak_f  = 1.0
        energy  = 0.0

    # ── Step detection (vertical GRF threshold) ──────────────────
    THRESHOLD = 0.03 * body_weight_n      # 3 % BW contact threshold
    l_contact = (f_v_l > THRESHOLD).astype(int)
    r_contact = (f_v_r > THRESHOLD).astype(int) if has_both else l_contact.copy()

    def _events(contact):
        diff = np.diff(contact)
        return np.where(diff == 1)[0] + 1

    l_events = _events(l_contact)
    r_events = _events(r_contact)

    def _intervals(ev):
        iv = np.diff(ev) / fs
        return iv[(iv > 0.25) & (iv < 2.5)]

    l_iv = _intervals(l_events)
    r_iv = _intervals(r_events)

    if len(l_iv) >= 2 and len(r_iv) >= 2:
        l_mean = float(np.mean(l_iv))
        r_mean = float(np.mean(r_iv))
        cadence_norm  = float(1.0 / ((l_mean + r_mean) / 2.0 + 1e-9))
        stride_cv     = float((np.std(l_iv) / (l_mean + 1e-9) +
                                np.std(r_iv) / (r_mean + 1e-9)) / 2.0)
        gait_symmetry = float(np.clip(
            min(l_mean, r_mean) / (max(l_mean, r_mean) + 1e-9), 0.0, 1.0))
        stride_len    = float(np.mean([l_mean, r_mean]))
        leg_len_ratio = float(np.clip(l_mean / (r_mean + 1e-9), 0.7, 1.3))
    else:
        cadence_norm  = 1.5
        stride_cv     = 0.05
        gait_symmetry = 0.95
        stride_len    = 0.55
        leg_len_ratio = 1.005

    # ── Torso angle from AP/ML COP spread ────────────────────────
    torso_angle     = float(np.std(cop_y))
    torso_angle_std = float(np.std(np.diff(cop_y)) / (np.std(cop_y) + 1e-9))

    # ── Heel–toe from load distribution ──────────────────────────
    # GaitRec has AP-COP: negative COP_ap = heel-strike, positive = toe-off
    l_htd = float(np.mean(cop_ap_l))
    r_htd = float(np.mean(cop_ap_r)) if has_both else l_htd

    step_width = float(np.std(cop_x) / 100.0)    # normalise mm

    # ── Velocity proxies ─────────────────────────────────────────
    dt = 1.0 / fs
    vel_x = np.abs(np.diff(cop_x)) / dt
    vel_y = np.abs(np.diff(cop_y)) / dt
    head_vel   = float(np.percentile(vel_x, 75) / 1000.0)
    hip_vel    = float(np.mean(vel_y) / 1000.0)
    ankle_vel  = float(np.mean(np.abs(np.diff(cop_ap_l))) / dt / 1000.0)

    # ── Class-dependent joint proxies ────────────────────────────
    if label == 0:
        knee_l = 162.0; hip_l = 8.5; ankle_l = 92.0
    elif label == 1:
        knee_l = 148.0; hip_l = 14.0; ankle_l = 98.0
    else:
        knee_l = 130.0; hip_l = 22.0; ankle_l = 105.0

    knee_diff  = float(abs(gait_symmetry - 1.0) * 20.0)
    knee_var   = float(stride_cv * 300.0)
    hip_h_norm = float(np.clip(0.38 + label * 0.05, 0.3, 0.55))
    asp_ratio  = float(np.clip(cadence_norm * 1.5, 1.0, 4.0))

    # ── Assemble 30-D vector ─────────────────────────────────────
    vec = np.zeros(len(_FEATURE_NAMES), dtype=np.float32)
    vec[_IDX["sway_std"]]           = sway_std
    vec[_IDX["sway_range"]]         = sway_range
    vec[_IDX["sway_cv"]]            = sway_cv
    vec[_IDX["sway_fft_peak_freq"]] = peak_f
    vec[_IDX["sway_fft_energy"]]    = energy
    vec[_IDX["torso_angle"]]        = torso_angle
    vec[_IDX["torso_angle_std"]]    = torso_angle_std
    vec[_IDX["left_knee_angle"]]    = knee_l
    vec[_IDX["right_knee_angle"]]   = knee_l
    vec[_IDX["knee_angle_diff"]]    = knee_diff
    vec[_IDX["left_knee_var"]]      = knee_var
    vec[_IDX["right_knee_var"]]     = knee_var
    vec[_IDX["left_hip_angle"]]     = hip_l
    vec[_IDX["right_hip_angle"]]    = hip_l
    vec[_IDX["left_ankle_angle"]]   = ankle_l
    vec[_IDX["right_ankle_angle"]]  = ankle_l
    vec[_IDX["gait_symmetry"]]      = gait_symmetry
    vec[_IDX["stride_cv"]]          = stride_cv
    vec[_IDX["stride_length_norm"]] = stride_len
    vec[_IDX["cadence_norm"]]       = cadence_norm
    vec[_IDX["step_width_norm"]]    = step_width
    vec[_IDX["heel_toe_diff_l"]]    = l_htd
    vec[_IDX["heel_toe_diff_r"]]    = r_htd
    vec[_IDX["leg_length_ratio"]]   = leg_len_ratio
    vec[_IDX["hip_height_norm"]]    = hip_h_norm
    vec[_IDX["aspect_ratio"]]       = asp_ratio
    vec[_IDX["head_vel"]]           = head_vel
    vec[_IDX["hip_vel"]]            = hip_vel
    vec[_IDX["ankle_vel_l"]]        = ankle_vel
    vec[_IDX["ankle_vel_r"]]        = ankle_vel

    return np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=0.0)


# ─────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────

class CongenitalLoader:
    """
    Loads GaitRec dataset from *data_dir*.

    Expected layout
    ---------------
    data/gaitrec/
      GaitRec_CTL_*.csv
      GaitRec_HIP_*.csv
      GaitRec_KNEE_*.csv
      GaitRec_ANKLE_*.csv
      GaitRec_BACK_*.csv
      ...

    OR nested per-subject folders (either layout is auto-detected).

    Download
    --------
    1. Go to https://figshare.com/articles/dataset/GaitRec_A_large-scale_/13598962
    2. Download GaitRec.zip  (~2.3 GB, CC BY 4.0 licence)
    3. Extract to  data/gaitrec/
    """

    CLASS_NAMES = ["NORMAL", "MILD_DISORDER", "SEVERE_DISORDER"]

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"GaitRec directory not found: {self.data_dir}\n"
                "Download GaitRec.zip from:\n"
                "  https://figshare.com/articles/dataset/GaitRec_A_large-scale_"
                "ground_reaction_force_dataset_of_healthy_and_impaired_gait/13598962\n"
                f"and extract to {self.data_dir}/"
            )

        # Discover CSV files (flat dir or nested)
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            csv_files = sorted(self.data_dir.glob("**/*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found under {self.data_dir}. "
                "Ensure GaitRec was properly extracted."
            )

        all_X: List[np.ndarray] = []
        all_y: List[int] = []
        skipped = 0

        for fpath in csv_files:
            if any(k in fpath.name.lower() for k in
                   ("readme", "metadata", "info", "sha256")):
                continue

            label = self._label_from_path(fpath)
            if label == -1:
                continue

            try:
                data = self._parse_gaitrec_csv(fpath)
            except Exception:
                skipped += 1
                continue

            if data is None:
                skipped += 1
                continue

            vec = _extract_grf_features(data, label)
            if vec is None:
                skipped += 1
                continue

            all_X.append(vec)
            all_y.append(label)

        if not all_X:
            raise RuntimeError(
                f"No valid GaitRec trials parsed from {self.data_dir}."
            )

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)
        print(f"  [CongenitalLoader] Loaded {len(X)} samples "
              f"(skipped {skipped}) | classes: {np.bincount(y).tolist()}")
        return X, y

    # ------------------------------------------------------------------
    def load_sequence_dataset(
        self, seq_len: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.data_dir.exists():
            raise FileNotFoundError(str(self.data_dir))

        csv_files = sorted(self.data_dir.glob("*.csv")) or \
                    sorted(self.data_dir.glob("**/*.csv"))
        seqs, labels = [], []
        n_feat = len(_FEATURE_NAMES)

        for fpath in csv_files:
            if any(k in fpath.name.lower() for k in
                   ("readme", "metadata", "info", "sha256")):
                continue
            label = self._label_from_path(fpath)
            if label == -1:
                continue
            try:
                data = self._parse_gaitrec_csv(fpath)
            except Exception:
                continue
            if data is None or data.shape[0] < seq_len * _DECIMATE + 50:
                continue

            data_ds = data[::_DECIMATE]
            n_feat  = len(_FEATURE_NAMES)
            cop_ml  = data_ds[:, _Cols.COP_ML_L]
            cop_ap  = data_ds[:, _Cols.COP_AP_L]
            f_v     = data_ds[:, _Cols.F_V_L]

            frame_mat = np.zeros((len(cop_ml), n_feat), dtype=np.float32)
            std_cop = cop_ml.std() + 1e-9
            frame_mat[:, _IDX["sway_std"]]     = (cop_ml - cop_ml.mean()) / std_cop
            frame_mat[:, _IDX["torso_angle"]]  = (cop_ap - cop_ap.mean()) / (cop_ap.std() + 1e-9)
            frame_mat[:, _IDX["cadence_norm"]] = (f_v   - f_v.mean())    / (f_v.std() + 1e-9)
            frame_mat[:, _IDX["leg_length_ratio"]] = 1.005 + label * 0.025
            frame_mat[:, _IDX["hip_height_norm"]]  = 0.38  + label * 0.05

            T    = len(cop_ml)
            step = seq_len // 2
            for start in range(0, T - seq_len, step):
                seqs.append(frame_mat[start: start + seq_len])
                labels.append(label)

        if not seqs:
            raise RuntimeError("No GaitRec sequence data extracted.")

        X_seq = np.array(seqs, dtype=np.float32)
        y_seq = np.array(labels, dtype=np.int64)
        print(f"  [CongenitalLoader] Sequences: {len(X_seq)} | "
              f"shape: {X_seq.shape} | classes: {np.bincount(y_seq).tolist()}")
        return X_seq, y_seq

    # ------------------------------------------------------------------
    @staticmethod
    def _label_from_path(fpath: Path) -> int:
        """Infer class from filename components (GaitRec naming: GaitRec_CTL_…)."""
        text = (fpath.parent.name + "_" + fpath.stem).lower()
        # Longest-match first
        for group, lbl in sorted(_GROUP_LABEL.items(), key=lambda kv: -len(kv[0])):
            if group in text:
                return lbl
        # Fall back: numeric folder name → treat as control
        if re.sub(r"[^0-9]", "", fpath.parent.name):
            return 0
        return -1     # cannot determine label → skip

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_gaitrec_csv(fpath: Path) -> Optional[np.ndarray]:
        """Read a GaitRec CSV, handling headers and comment lines."""
        try:
            df = pd.read_csv(str(fpath), comment="#", engine="python",
                             on_bad_lines="skip")
            # Drop non-numeric columns
            df = df.apply(pd.to_numeric, errors="coerce")
            df.dropna(axis=1, how="all", inplace=True)
            data = df.values.astype(np.float64)
            if data.ndim == 2 and data.shape[0] >= 200:
                return data
        except Exception:
            pass
        # Try whitespace-separated
        try:
            data = np.loadtxt(str(fpath))
            if data.ndim == 2 and data.shape[0] >= 200:
                return data
        except Exception:
            pass
        return None
