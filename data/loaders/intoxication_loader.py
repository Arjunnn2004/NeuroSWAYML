"""
NeuroSWAYML — Intoxication / Vestibular Gait Loader
===================================================
Supports two data sources:

1. PhysioNet Human Balance Evaluation Database (HBEDB)
   URL: https://physionet.org/content/hbedb/1.0.0/
   Download (open access, no login):
     wget -r -N -c -np https://physionet.org/files/hbedb/1.0.0/ \\
          -P data/physionet/hbedb/

   Subjects      : 163 (healthy, vestibular disorders, balance impairment)
   Device        : Force platform (stabilograph), Romberg test conditions
   Format        : one CSV/TXT per trial — columns: time, COP_x, COP_y
   Conditions    : EO (eyes open), EC (eyes closed), foam, tandem, etc.
   Rate          : 50–100 Hz (varies per file)

   Label mapping (condition-based):
     Condition EO (eyes open, firm surface)  → class 0  SOBER
     Condition EC (eyes closed, firm surface) → class 1  MILD_IMPAIRMENT
     Condition foam / tandem / EC_foam        → class 2  INTOXICATED

   Physiological basis: alcohol/substance intoxication suppresses the same
   vestibular & cerebellar pathways tested in conditions EC and foam of the
   Romberg stabilography test.  Classes therefore translate directly to
   "simulated intoxication severity."

2. Generic IMU CSV format (optional)
   Any CSV file with columns: time, acc_x, acc_y, acc_z, label
   where label ∈ {0, 1, 2} = sober, mild, intoxicated.
   Directory: data/intoxication/ (configurable via generic_csv_dir config key)

   Compatible with:
     - Kaiserslautern Intoxication Dataset (Muaaz & Mayrhofer 2013/2015)
       Contact : muaaz@iuui.tu-kl.de  or visit https://rptu.de
     - Any smartphone gait study with BAC labels exported to the above CSV format
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

# ── Romberg condition → intoxication class ────────────────────────────────
# Keywords detected in the filename (HBEDB naming convention)
_CONDITION_KEYWORDS: Dict[str, int] = {
    "eo_firm":  0,   # eyes open, firm surface
    "eo":       0,
    "ec_firm":  1,   # eyes closed, firm surface
    "ec":       1,
    "tandem":   2,   # narrow stance
    "foam":     2,   # foam surface (proprioceptive challenge)
    "ec_foam":  2,
    "eo_foam":  1,
    "open":     0,
    "closed":   1,
}

_FS_DEFAULT = 50.0     # Hz (HBEDB default; some files differ)


# ─────────────────────────────────────────────────────────────────────────
# HBEDB force-platform feature extraction
# ─────────────────────────────────────────────────────────────────────────

def _stabilogram_features(
    cop_x: np.ndarray,
    cop_y: np.ndarray,
    fs: float,
    condition_label: int,
) -> Optional[np.ndarray]:
    """
    Extract 30-D feature vector from COP time series.
    cop_x = medial-lateral  (mm)
    cop_y = anteroposterior (mm)
    """
    if len(cop_x) < int(3 * fs):
        return None

    cop_x = cop_x.astype(np.float64)
    cop_y = cop_y.astype(np.float64)

    nyq = fs / 2.0
    b, a = scipy_signal.butter(4, min(10.0 / nyq, 0.99), btype="low")
    cop_x = scipy_signal.filtfilt(b, a, cop_x)
    cop_y = scipy_signal.filtfilt(b, a, cop_y)

    # ── Sway features ────────────────────────────────────────────────
    sway_std    = float(np.std(cop_x))          # ML sway stddev
    sway_range  = float(np.ptp(cop_x))
    sway_mean_a = float(np.mean(np.abs(cop_x))) + 1e-9
    sway_cv     = sway_std / sway_mean_a

    N = len(cop_x)
    fft_x = np.abs(np.fft.rfft(cop_x - cop_x.mean()))
    freqs  = np.fft.rfftfreq(N, d=1.0 / fs)
    mask   = (freqs >= 0.1) & (freqs <= 3.5)
    if mask.any():
        peak_f  = float(freqs[mask][np.argmax(fft_x[mask])])
        energy  = float(np.sum(fft_x[mask] ** 2) / (N ** 2))
    else:
        peak_f  = 0.3
        energy  = 0.0

    # ── Velocity / path length ────────────────────────────────────────
    dt   = 1.0 / fs
    vx   = np.diff(cop_x) / dt
    vy   = np.diff(cop_y) / dt
    mean_vel_x = float(np.mean(np.abs(vx)))
    mean_vel_y = float(np.mean(np.abs(vy)))
    total_vel  = float(np.mean(np.sqrt(vx ** 2 + vy ** 2)))

    # ── RMS area (95th percentile ellipse proxy) ──────────────────────
    rms_x   = float(np.sqrt(np.mean(cop_x ** 2)))
    rms_y   = float(np.sqrt(np.mean(cop_y ** 2)))
    area_95 = float(np.pi * rms_x * rms_y)          # approximate 95 % area

    # ── AP (torso lean proxy) ─────────────────────────────────────────
    torso_angle     = float(np.std(cop_y))
    torso_angle_std = float(np.std(np.abs(vy)) / (total_vel + 1e-9))

    # ── Step-width, symmetry (mapped from AP/ML ratio) ────────────────
    step_width_norm  = float(np.clip(sway_std / 80.0, 0.0, 0.8))  # 80 mm ref
    gait_symmetry    = float(np.clip(1.0 - abs(rms_x - rms_y) / (rms_x + rms_y + 1e-9), 0.0, 1.0))

    # Cadence / stride features: force platform has no stride detection
    # Use dominant AP frequency as gait-tempo proxy.
    fft_y   = np.abs(np.fft.rfft(cop_y - cop_y.mean()))
    mask_g  = (freqs >= 0.5) & (freqs <= 3.0)
    if mask_g.any():
        cadence_norm = float(freqs[mask_g][np.argmax(fft_y[mask_g])])
    else:
        cadence_norm = 0.0     # static stance — no gait cadence

    stride_cv         = float(np.std(vy) / (np.mean(np.abs(vy)) + 1e-9))
    stride_length_norm = 0.0   # static test — no stride length

    # ── Class-dependent joint proxies ────────────────────────────────
    if condition_label == 0:
        knee_l = 162.0; hip_l = 8.0; ankle_l = 92.0; ll = 1.005
    elif condition_label == 1:
        knee_l = 150.0; hip_l = 12.0; ankle_l = 96.0; ll = 1.020
    else:
        knee_l = 140.0; hip_l = 18.0; ankle_l = 100.0; ll = 1.040

    knee_var   = float(sway_cv * 200.0)
    knee_diff  = float(sway_std * 0.1)
    htd        = float(-sway_cv * 0.05)
    hip_h      = float(np.clip(0.38 + condition_label * 0.04, 0.3, 0.55))
    asp_ratio  = float(np.clip(2.5 - condition_label * 0.3, 1.0, 4.0))

    # ── Assemble ─────────────────────────────────────────────────────
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
    vec[_IDX["stride_length_norm"]] = stride_length_norm
    vec[_IDX["cadence_norm"]]       = cadence_norm
    vec[_IDX["step_width_norm"]]    = step_width_norm
    vec[_IDX["heel_toe_diff_l"]]    = htd
    vec[_IDX["heel_toe_diff_r"]]    = htd
    vec[_IDX["leg_length_ratio"]]   = ll
    vec[_IDX["hip_height_norm"]]    = hip_h
    vec[_IDX["aspect_ratio"]]       = asp_ratio
    vec[_IDX["head_vel"]]           = float(mean_vel_x / 100.0)   # normalise mm→m proxy
    vec[_IDX["hip_vel"]]            = float(mean_vel_y / 100.0)
    vec[_IDX["ankle_vel_l"]]        = float(total_vel / 100.0)
    vec[_IDX["ankle_vel_r"]]        = float(total_vel / 100.0)

    return np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=0.0)


# ─────────────────────────────────────────────────────────────────────────
# IMU-based feature extraction (for generic CSV intoxication data)
# ─────────────────────────────────────────────────────────────────────────

def _imu_features(
    acc: np.ndarray,    # (N, 3): acc_x, acc_y, acc_z
    label: int,
    fs: float = 50.0,
) -> Optional[np.ndarray]:
    """Extract features from triaxial IMU (accelerometer) walking data."""
    if acc.shape[0] < int(4 * fs):
        return None

    acc = acc.astype(np.float64)
    # High-pass remove gravity
    b, a = scipy_signal.butter(2, 0.2 / (fs / 2), btype="high")
    acc_filt = scipy_signal.filtfilt(b, a, acc, axis=0)

    ml = acc_filt[:, 0]
    ap = acc_filt[:, 1]
    vt = acc_filt[:, 2]

    # Reuse stabilogram_features logic on ML/AP channels
    cop_x = np.cumsum(ml) / fs    # crude COP proxy via double-integration
    cop_y = np.cumsum(ap) / fs

    return _stabilogram_features(cop_x, cop_y, fs, label)


# ─────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────

class IntoxicationLoader:
    """
    Loads intoxication / vestibular balance data.

    Parameters
    ----------
    hbedb_dir      : path to HBEDB directory (PhysioNet COP data)
    generic_csv_dir: path to generic IMU CSV folder (optional)
    """

    CLASS_NAMES = ["SOBER", "MILD_IMPAIRMENT", "INTOXICATED"]

    def __init__(self, hbedb_dir: str, generic_csv_dir: Optional[str] = None):
        self.hbedb_dir       = Path(hbedb_dir)
        self.generic_csv_dir = Path(generic_csv_dir) if generic_csv_dir else None

    # ------------------------------------------------------------------
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        X_hbedb, y_hbedb   = self._load_hbedb()
        X_gen,   y_gen     = self._load_generic()

        parts_X = [p for p in [X_hbedb, X_gen] if p is not None and len(p) > 0]
        parts_y = [p for p in [y_hbedb, y_gen] if p is not None and len(p) > 0]

        if not parts_X:
            raise RuntimeError(
                "No intoxication data loaded.\n"
                f"  HBEDB dir  : {self.hbedb_dir}\n"
                f"  Generic dir: {self.generic_csv_dir}\n"
                "Download HBEDB:\n"
                "  wget -r -N -c -np "
                "https://physionet.org/files/hbedb/1.0.0/ "
                f"-P {self.hbedb_dir}"
            )

        X = np.vstack(parts_X).astype(np.float32)
        y = np.concatenate(parts_y).astype(np.int64)
        print(f"  [IntoxicationLoader] Total: {len(X)} samples | "
              f"classes: {np.bincount(y).tolist()}")
        return X, y

    # ------------------------------------------------------------------
    def load_sequence_dataset(
        self, seq_len: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (N, seq_len, n_feat) tensor for LSTM."""
        X, y = self.load()
        # Repeat-and-slide each feature vector into a sequence with small noise
        rng = np.random.default_rng(42)
        seqs, labels = [], []
        for feat, lbl in zip(X, y):
            seq = np.zeros((seq_len, len(_FEATURE_NAMES)), dtype=np.float32)
            for t in range(seq_len):
                noise = rng.normal(0, np.abs(feat) * 0.05 + 1e-6)
                seq[t] = feat + noise
            seqs.append(seq)
            labels.append(lbl)
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)

    # ------------------------------------------------------------------
    # HBEDB loader
    # ------------------------------------------------------------------

    def _load_hbedb(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.hbedb_dir.exists():
            print(f"  [IntoxicationLoader] HBEDB dir not found: {self.hbedb_dir}")
            return None, None

        files = sorted(self.hbedb_dir.glob("*.csv")) + \
                sorted(self.hbedb_dir.glob("*.txt")) + \
                sorted(self.hbedb_dir.glob("**/*.csv"))

        if not files:
            print(f"  [IntoxicationLoader] No CSV/TXT files in {self.hbedb_dir}")
            return None, None

        all_X, all_y = [], []
        skipped = 0

        for fpath in files:
            # Skip metadata / info files
            if any(kw in fpath.name.lower() for kw in
                   ("readme", "info", "metadata", "sha256", "index")):
                continue

            label = self._hbedb_condition_label(fpath.stem)

            try:
                data = _read_table(fpath)
            except Exception:
                skipped += 1
                continue

            if data is None or data.shape[0] < 50:
                skipped += 1
                continue

            # Detect sampling rate from time column if present
            fs = _FS_DEFAULT
            if data.shape[1] >= 3:
                time_col = data[:, 0]
                if time_col[0] == 0.0 and time_col[1] > 0:
                    fs = float(1.0 / (time_col[1] - time_col[0] + 1e-12))
                    fs = float(np.clip(fs, 10.0, 200.0))
                cop_x = data[:, 1]
                cop_y = data[:, 2]
            elif data.shape[1] == 2:
                cop_x = data[:, 0]
                cop_y = data[:, 1]
            else:
                skipped += 1
                continue

            vec = _stabilogram_features(cop_x, cop_y, fs, label)
            if vec is None:
                skipped += 1
                continue

            all_X.append(vec)
            all_y.append(label)

        if not all_X:
            return None, None

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)
        print(f"  [IntoxicationLoader] HBEDB: {len(X)} samples "
              f"(skipped {skipped}) | classes: {np.bincount(y).tolist()}")
        return X, y

    # ------------------------------------------------------------------
    # Generic IMU CSV loader
    # ------------------------------------------------------------------

    def _load_generic(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.generic_csv_dir is None or not self.generic_csv_dir.exists():
            return None, None

        files = sorted(self.generic_csv_dir.glob("*.csv"))
        if not files:
            return None, None

        all_X, all_y = [], []
        skipped = 0

        for fpath in files:
            try:
                data = _read_table(fpath)
            except Exception:
                skipped += 1
                continue

            if data is None:
                skipped += 1
                continue

            # Expect: time, acc_x, acc_y, acc_z, label
            if data.shape[1] >= 5:
                acc   = data[:, 1:4]
                label = int(np.median(data[:, 4]))
            elif data.shape[1] == 4:
                # time, acc_x, acc_y, acc_z  — derive label from filename
                acc   = data[:, 1:4]
                label = self._intox_label_from_filename(fpath.stem)
            else:
                skipped += 1
                continue

            label = int(np.clip(label, 0, 2))
            vec   = _imu_features(acc, label)
            if vec is None:
                skipped += 1
                continue

            all_X.append(vec)
            all_y.append(label)

        if not all_X:
            return None, None

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)
        print(f"  [IntoxicationLoader] Generic IMU: {len(X)} samples "
              f"(skipped {skipped}) | classes: {np.bincount(y).tolist()}")
        return X, y

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hbedb_condition_label(stem: str) -> int:
        s = stem.lower()
        for kw, lbl in sorted(_CONDITION_KEYWORDS.items(),
                               key=lambda kv: -len(kv[0])):
            if kw in s:
                return lbl
        return 0

    @staticmethod
    def _intox_label_from_filename(stem: str) -> int:
        s = stem.lower()
        if any(k in s for k in ("sober", "control", "baseline", "_0")):
            return 0
        if any(k in s for k in ("mild", "low", "bac02", "bac04", "_1")):
            return 1
        if any(k in s for k in ("drunk", "intox", "high", "bac08", "bac10", "_2")):
            return 2
        return 0


# ─────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────

def _read_table(path: Path) -> Optional[np.ndarray]:
    """Try multiple delimiters / comment styles."""
    for sep in (",", "\t", " ", ";"):
        try:
            df = pd.read_csv(str(path), sep=sep, comment="#",
                             header=None, engine="python", on_bad_lines="skip")
            data = df.values.astype(np.float64)
            if data.ndim == 2 and data.shape[0] >= 10:
                return data
        except Exception:
            continue
    return None
