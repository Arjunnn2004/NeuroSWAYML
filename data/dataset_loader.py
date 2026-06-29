"""
NeuroSWAYML - Dataset Loader
Loads and parses the PhysioNet Gait in Parkinson's Disease database (gaitpdb).

Dataset format (19-column VGRF text files, 100 Hz):
  Col  1    : Time (seconds)
  Cols 2-9  : Left foot vertical ground reaction force — 8 sensors (Newtons)
  Cols 10-17: Right foot VGRF — 8 sensors (Newtons)
  Col 18    : Total left foot force
  Col 19    : Total right foot force

File naming convention: {Study}{Group}{Subject}_{Walk}.txt
  Study : Ga / Ju / Si
  Group : Co (healthy control) / Pt (Parkinson's patient)
  Walk  : 01-09 = normal walk, 10 = dual-task walk (cognitively loaded)

Label mapping:
  *Co*_0[1-9] → 0  NORMAL
  *Co*_10     → 1  WARNING  (cognitively loaded gait shows early degradation)
  *Pt*        → 2  HIGH_RISK (Parkinson's patients)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from scipy import signal as scipy_signal


# ── Sensor (X, Y) positions inside the insole (arbitrary units, from format.txt) ─
_L_SENSOR_XY = np.array([
    [-500, -800], [-700, -400], [-300, -400], [-700, 0],
    [-300,    0], [-700,  400], [-300,  400], [-500, 800],
], dtype=np.float64)

_R_SENSOR_XY = np.array([
    [500, -800], [700, -400], [300, -400], [700, 0],
    [300,   0], [700,  400], [300,  400], [500, 800],
], dtype=np.float64)

# Sampling rate of gaitpdb
_FS = 100.0   # Hz

# ── Fall feature profile (synthetic — gaitpdb has no fall labels) ────────────
_FALL_PROFILES: Dict[int, Dict[str, Tuple[float, float]]] = {
    0: {  # No fall
        "hip_height_world": (0.90, 0.08),
        "torso_angle":      (5.0,  3.0),
        "aspect_ratio":     (2.8,  0.4),
        "head_vel":         (0.025, 0.012),
        "hip_vel":          (0.030, 0.015),
        "shoulder_hip_dist":(0.28,  0.04),
        "body_height_ratio":(0.80,  0.06),
    },
    1: {  # Fall
        "hip_height_world": (0.30, 0.18),
        "torso_angle":      (55.0, 25.0),
        "aspect_ratio":     (0.80, 0.35),
        "head_vel":         (0.42, 0.18),
        "hip_vel":          (0.55, 0.22),
        "shoulder_hip_dist":(0.12, 0.06),
        "body_height_ratio":(0.35, 0.15),
    },
}

# Keep GAIT_PROFILES only as a fallback when real data unavailable
GAIT_PROFILES: Dict[int, Dict[str, Tuple[float, float]]] = {
    # ─── CLASS 0: NORMAL ────────────────────────────────────────────────────
    0: {
        "sway_std":           (0.012, 0.004),   # lateral hip sway std (normalised)
        "sway_range":         (0.045, 0.015),
        "sway_cv":            (0.08,  0.03),    # coefficient of variation of sway
        "sway_fft_peak_freq": (0.35,  0.10),    # dominant sway frequency (Hz)
        "sway_fft_energy":    (0.020, 0.008),
        "torso_angle":        (4.5,   2.5),     # degrees from vertical
        "torso_angle_std":    (1.2,   0.8),
        "left_knee_angle":    (162.0, 8.0),     # degrees (near full extension)
        "right_knee_angle":   (162.0, 8.0),
        "knee_angle_diff":    (2.5,   2.0),
        "left_knee_var":      (180.0, 60.0),    # variance across stride
        "right_knee_var":     (180.0, 60.0),
        "left_hip_angle":     (8.5,   3.0),
        "right_hip_angle":    (8.5,   3.0),
        "left_ankle_angle":   (92.0,  6.0),
        "right_ankle_angle":  (92.0,  6.0),
        "gait_symmetry":      (0.96,  0.03),
        "stride_cv":          (0.035, 0.015),   # stride interval coefficient of variation
        "stride_length_norm": (0.42,  0.06),    # normalised by height
        "cadence_norm":       (1.85,  0.20),    # steps/sec
        "step_width_norm":    (0.18,  0.04),
        "heel_toe_diff_l":    (0.020, 0.010),   # heel-toe Y diff (positive = toe down = normal)
        "heel_toe_diff_r":    (0.020, 0.010),
        "leg_length_ratio":   (1.005, 0.020),
        "hip_height_norm":    (0.38,  0.04),    # hip Y in image (low = standing upright)
        "aspect_ratio":       (2.8,   0.4),     # body bbox H/W
        "head_vel":           (0.020, 0.010),
        "hip_vel":            (0.025, 0.012),
        "ankle_vel_l":        (0.060, 0.020),
        "ankle_vel_r":        (0.060, 0.020),
    },
    # ─── CLASS 1: WARNING (ALS / Huntington's / minor fall risk) ───────────
    1: {
        "sway_std":           (0.025, 0.008),
        "sway_range":         (0.090, 0.025),
        "sway_cv":            (0.18,  0.06),
        "sway_fft_peak_freq": (0.55,  0.15),
        "sway_fft_energy":    (0.055, 0.020),
        "torso_angle":        (10.0,  4.0),
        "torso_angle_std":    (2.5,   1.2),
        "left_knee_angle":    (148.0, 12.0),
        "right_knee_angle":   (148.0, 12.0),
        "knee_angle_diff":    (7.0,   4.0),
        "left_knee_var":      (100.0, 40.0),
        "right_knee_var":     (100.0, 40.0),
        "left_hip_angle":     (14.0,  5.0),
        "right_hip_angle":    (14.0,  5.0),
        "left_ankle_angle":   (98.0,  9.0),
        "right_ankle_angle":  (98.0,  9.0),
        "gait_symmetry":      (0.85,  0.08),
        "stride_cv":          (0.095, 0.035),
        "stride_length_norm": (0.32,  0.08),
        "cadence_norm":       (1.55,  0.25),
        "step_width_norm":    (0.22,  0.06),
        "heel_toe_diff_l":    (0.005, 0.018),
        "heel_toe_diff_r":    (0.005, 0.018),
        "leg_length_ratio":   (1.025, 0.040),
        "hip_height_norm":    (0.42,  0.05),
        "aspect_ratio":       (2.3,   0.5),
        "head_vel":           (0.045, 0.020),
        "hip_vel":            (0.055, 0.025),
        "ankle_vel_l":        (0.045, 0.018),
        "ankle_vel_r":        (0.045, 0.018),
    },
    # ─── CLASS 2: HIGH_RISK (Parkinson's / imminent fall) ──────────────────
    2: {
        "sway_std":           (0.042, 0.014),
        "sway_range":         (0.155, 0.045),
        "sway_cv":            (0.32,  0.10),
        "sway_fft_peak_freq": (0.80,  0.22),    # Parkinsonian tremor 3-6 Hz
        "sway_fft_energy":    (0.110, 0.040),
        "torso_angle":        (17.0,  7.0),
        "torso_angle_std":    (4.5,   2.0),
        "left_knee_angle":    (135.0, 18.0),
        "right_knee_angle":   (135.0, 18.0),
        "knee_angle_diff":    (14.0,  8.0),
        "left_knee_var":      (55.0,  25.0),    # stiff/shuffling gait
        "right_knee_var":     (55.0,  25.0),
        "left_hip_angle":     (21.0,  8.0),
        "right_hip_angle":    (21.0,  8.0),
        "left_ankle_angle":   (104.0, 12.0),
        "right_ankle_angle":  (104.0, 12.0),
        "gait_symmetry":      (0.72,  0.12),
        "stride_cv":          (0.175, 0.065),   # high variability = Parkinson's
        "stride_length_norm": (0.20,  0.07),    # shuffling short steps
        "cadence_norm":       (1.30,  0.30),
        "step_width_norm":    (0.26,  0.08),
        "heel_toe_diff_l":    (-0.010, 0.025),  # shuffling = heel not lifting
        "heel_toe_diff_r":    (-0.010, 0.025),
        "leg_length_ratio":   (1.055, 0.060),
        "hip_height_norm":    (0.48,  0.07),
        "aspect_ratio":       (1.85,  0.55),
        "head_vel":           (0.075, 0.030),
        "hip_vel":            (0.090, 0.038),
        "ankle_vel_l":        (0.030, 0.015),
        "ankle_vel_r":        (0.030, 0.015),
    },
}

# Fall-specific feature profiles (binary: 0=no fall, 1=fall)
FALL_PROFILES: Dict[int, Dict[str, Tuple[float, float]]] = {
    0: {  # Standing / Walking (no fall)
        "hip_height_world": (0.90,  0.08),   # metres above floor (world coords)
        "torso_angle":      (5.0,   3.0),
        "aspect_ratio":     (2.8,   0.4),
        "head_vel":         (0.025, 0.012),
        "hip_vel":          (0.030, 0.015),
        "shoulder_hip_dist":(0.28,  0.04),
        "body_height_ratio":(0.80,  0.06),
    },
    1: {  # Falling / On-ground
        "hip_height_world": (0.30,  0.18),
        "torso_angle":      (55.0,  25.0),
        "aspect_ratio":     (0.80,  0.35),
        "head_vel":         (0.42,  0.18),
        "hip_vel":          (0.55,  0.22),
        "shoulder_hip_dist":(0.12,  0.06),
        "body_height_ratio":(0.35,  0.15),
    },
}


# ───────────────────────────────────────────────────────────────────────────
class DatasetLoader:
    """
    Loads real gaitpdb VGRF data from the PhysioNet dataset.
    Falls back to synthetic generation only if files are not found.
    """

    from data.feature_extractor import FeatureExtractor as _FE
    FEATURE_NAMES: List[str] = [
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
    FALL_FEATURE_NAMES: List[str] = list(_FALL_PROFILES[0].keys())

    # (folder_slug, nested_subfolder_or_"", file_glob, format_type)
    # format types: "vgrf19"=gaitpdb 19-col, "ndd_ts"=gaitndd .ts, "mat_txt"=maturation 2-col
    _COMPATIBLE_DATASETS = [
        ("gaitpdb",        "gait-in-parkinsons-disease-1.0.0", "*.txt",  "vgrf19"),
        ("gaitndd",        "",                                  "*.ts",   "ndd_ts"),
        ("gait-maturation","data",                              "*.txt",  "mat_txt"),
    ]

    def __init__(self, config: dict):
        self.cfg = config["dataset"]
        self.rng = np.random.default_rng(self.cfg["random_seed"])

        root = Path(self.cfg["physionet_local_dir"])
        # list of (Path, format_type) for each discovered dataset
        self.data_sources: List[tuple] = []
        for slug, nested, glob_pat, fmt in self._COMPATIBLE_DATASETS:
            base = root / slug
            candidate = (base / nested) if nested else base
            for d in (candidate, base):
                if d.exists() and any(d.glob(glob_pat)):
                    self.data_sources.append((d, fmt, glob_pat))
                    break

        # Backwards-compat
        self.data_dirs: List[Path] = [d for d, _, _ in self.data_sources]
        self.data_dir: Optional[Path] = self.data_dirs[0] if self.data_dirs else None

        if self.data_sources:
            print(f"  [dataset] Found {len(self.data_sources)} compatible dataset(s):")
            for d, fmt, _ in self.data_sources:
                count = len(list(d.glob(_.replace('*','*'))))
                print(f"    • [{fmt}] {d.name}  ({count} files)")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def load_gait_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X, y) for 3-class gait risk. Uses real data when available."""
        real_X, real_y = self._load_real_gait()
        if real_X is not None and len(real_X) > 0:
            print(f"  Dataset: {len(real_X)} real PhysioNet samples  "
                  f"| classes: {np.bincount(real_y.astype(int)).tolist()}")
            return real_X.astype(np.float32), real_y.astype(np.int64)

        # Fallback
        print("  [dataset] Real files not found — using synthetic fallback")
        return self._generate_synthetic_gait()

    def load_fall_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Binary fall detection — always synthetic (gaitpdb has no fall labels)."""
        X, y = self._generate_synthetic_fall()
        print(f"  Fall dataset: {len(X)} synthetic samples")
        return X.astype(np.float32), y.astype(np.int64)

    def load_sequence_dataset(
        self, seq_len: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        (N, seq_len, n_features) tensor for LSTM.
        Built from real gaitpdb windows (sliding window over each recording).
        Falls back to auto-correlated synthetic if real data unavailable.
        """
        real_X, real_y = self._load_real_gait()
        if real_X is not None and len(real_X) > 0:
            return self._make_sequences_from_real(seq_len)

        return self._generate_synthetic_sequences(seq_len)

    # ------------------------------------------------------------------
    # REAL DATA LOADER
    # ------------------------------------------------------------------

    def _load_real_gait(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse ALL discovered datasets using format-appropriate parsers."""
        if not self.data_sources:
            return None, None

        _SKIP = {"format", "demographics", "readme", "info", "sha256sums",
                 "index", "niwar", "orcoh", "new-gait-ts"}
        all_X, all_y = [], []

        for data_dir, fmt, glob_pat in self.data_sources:
            files = [
                f for f in sorted(data_dir.glob(glob_pat))
                if f.stem.lower().split(".")[0] not in _SKIP
            ]
            if not files:
                continue
            print(f"  [dataset] {data_dir.name} [{fmt}]: {len(files)} files…")

            for f in files:
                label = self._file_label(f.stem, fmt)
                if label == -1:
                    continue
                if fmt == "vgrf19":
                    feats = self._extract_vgrf_features(f)
                elif fmt == "ndd_ts":
                    feats = self._extract_ts_features(f, label)
                elif fmt == "mat_txt":
                    feats = self._extract_maturation_features(f)
                else:
                    feats = None
                if feats is not None:
                    all_X.append(feats)
                    all_y.append(label)

        if not all_X:
            return None, None

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)
        print(f"  [dataset] Combined: {len(X)} samples | classes: {np.bincount(y).tolist()}")
        return X, y

    # ------------------------------------------------------------------
    # VGRF FEATURE EXTRACTION  (the real ML work)
    # ------------------------------------------------------------------

    def _extract_vgrf_features(self, path: Path) -> Optional[np.ndarray]:
        """
        Parse one gaitpdb .txt file → 30-dim feature vector matching
        DatasetLoader.FEATURE_NAMES.

        Columns (19 total):
          0     : Time (s)
          1-8   : Left foot sensors L1-L8 (N)
          9-16  : Right foot sensors R1-R8 (N)
          17    : Total left force (N)
          18    : Total right force (N)
        """
        try:
            data = np.loadtxt(str(path))
        except Exception:
            return None

        if data.ndim != 2 or data.shape[1] < 19 or data.shape[0] < 200:
            return None

        time_s   = data[:, 0]
        l_sens   = data[:, 1:9]      # (N, 8) left sensors
        r_sens   = data[:, 9:17]     # (N, 8) right sensors
        f_left   = data[:, 17]       # total left force
        f_right  = data[:, 18]       # total right force
        f_total  = f_left + f_right + 1e-9

        # ── Centre of Pressure (COP) ──────────────────────────────────
        # COP_x = sum(sensor_x * force) / total_force  for each foot
        l_cop_x = (l_sens @ _L_SENSOR_XY[:, 0]) / (f_left  + 1e-9)
        r_cop_x = (r_sens @ _R_SENSOR_XY[:, 0]) / (f_right + 1e-9)
        # Combined lateral COP (weighted by each foot's contribution)
        cop_x_combined = (f_left * l_cop_x + f_right * r_cop_x) / f_total

        # ── Sway features (COP lateral) ──────────────────────────────
        sway_std   = float(np.std(cop_x_combined))
        sway_range = float(np.ptp(cop_x_combined))
        sway_mean  = float(np.mean(np.abs(cop_x_combined)))
        sway_cv    = sway_std / (sway_mean + 1e-9)

        # FFT of COP signal
        fft_mag  = np.abs(np.fft.rfft(cop_x_combined - cop_x_combined.mean()))
        freqs    = np.fft.rfftfreq(len(cop_x_combined), d=1.0 / _FS)
        peak_idx = int(np.argmax(fft_mag[1:])) + 1
        sway_fft_peak   = float(freqs[peak_idx])
        sway_fft_energy = float(np.sum(fft_mag[1:] ** 2))
        # Normalise energy by signal length
        sway_fft_energy = sway_fft_energy / (len(fft_mag) ** 2 + 1e-9)

        # ── Step / stride detection ──────────────────────────────────
        # A step occurs when a foot transitions from unloaded → loaded
        CONTACT_THRESHOLD = 50.0   # Newtons
        l_contact = (f_left  > CONTACT_THRESHOLD).astype(int)
        r_contact = (f_right > CONTACT_THRESHOLD).astype(int)

        l_events  = self._contact_events(l_contact)   # frame indices of L heel-strike
        r_events  = self._contact_events(r_contact)

        # Stride intervals (seconds between same-foot events)
        l_intervals = np.diff(l_events) / _FS if len(l_events) > 1 else np.array([1.1])
        r_intervals = np.diff(r_events) / _FS if len(r_events) > 1 else np.array([1.1])

        l_intervals = l_intervals[(l_intervals > 0.3) & (l_intervals < 3.0)]
        r_intervals = r_intervals[(r_intervals > 0.3) & (r_intervals < 3.0)]

        def _cv(arr):
            return float(np.std(arr) / (np.mean(arr) + 1e-9)) if len(arr) > 2 else 0.05

        l_cv       = _cv(l_intervals)
        r_cv       = _cv(r_intervals)
        stride_cv  = (l_cv + r_cv) / 2.0

        l_mean_si  = float(np.mean(l_intervals)) if len(l_intervals) else 1.1
        r_mean_si  = float(np.mean(r_intervals)) if len(r_intervals) else 1.1
        mean_si    = (l_mean_si + r_mean_si) / 2.0
        cadence    = 1.0 / (mean_si + 1e-9)    # steps/sec

        # Gait symmetry: ratio of mean L/R stride intervals
        symmetry = (min(l_mean_si, r_mean_si) /
                    (max(l_mean_si, r_mean_si) + 1e-9))
        symmetry = float(np.clip(symmetry, 0.0, 1.0))

        # Step width proxy: lateral spread of combined COP
        step_width = float(np.std(cop_x_combined) / 500.0)   # normalise by sensor span

        # Stride length proxy: normalised from cadence and assumed speed
        stride_length_norm = float(np.clip(mean_si, 0.1, 2.0))

        # ── Force-derived stance metrics ─────────────────────────────
        # These proxy pose features that can't be computed from VGRF alone.
        # We derive them from the force distribution patterns.

        # Torso lean proxy: asymmetry in left/right peak forces
        l_peak = float(np.percentile(f_left,  95))
        r_peak = float(np.percentile(f_right, 95))
        torso_angle = float(abs(l_peak - r_peak) / (max(l_peak, r_peak) + 1e-9) * 20.0)
        torso_angle_std = float(np.std(np.abs(f_left - f_right)) / (f_total.mean() + 1e-9) * 10.0)

        # Knee angle proxy: COP forward position during stance (L8/R8 = heel, L1/R1 = toe)
        l_heel_load = l_sens[:, 0]   # L1 = heel
        l_toe_load  = l_sens[:, 7]   # L8 = toe
        r_heel_load = r_sens[:, 0]
        r_toe_load  = r_sens[:, 7]

        # Heel-dominant → straighter knee; toe-dominant → more flexed
        l_heel_frac = float(np.mean(l_heel_load) / (np.mean(l_sens.sum(axis=1)) + 1e-9))
        r_heel_frac = float(np.mean(r_heel_load) / (np.mean(r_sens.sum(axis=1)) + 1e-9))
        # Map heel fraction to knee angle estimate (higher heel load → near-extension)
        l_knee_angle = float(np.clip(120.0 + l_heel_frac * 60.0, 90.0, 180.0))
        r_knee_angle = float(np.clip(120.0 + r_heel_frac * 60.0, 90.0, 180.0))
        knee_angle_diff = abs(l_knee_angle - r_knee_angle)

        l_toe_frac  = float(np.mean(l_toe_load) / (np.mean(l_sens.sum(axis=1)) + 1e-9))
        r_toe_frac  = float(np.mean(r_toe_load) / (np.mean(r_sens.sum(axis=1)) + 1e-9))
        heel_toe_diff_l = float(l_heel_frac - l_toe_frac)
        heel_toe_diff_r = float(r_heel_frac - r_toe_frac)

        # Knee variance: variability of the heel/toe fraction across strides
        l_knee_var = float(np.var(l_heel_load / (l_sens.sum(axis=1) + 1e-9)) * 1000.0)
        r_knee_var = float(np.var(r_heel_load / (r_sens.sum(axis=1) + 1e-9)) * 1000.0)

        # Hip / ankle angle proxies (constant — not measurable from insole alone)
        l_hip_angle = 10.0; r_hip_angle = 10.0
        l_ankle_angle = 92.0; r_ankle_angle = 92.0

        # Leg length ratio: normalised by left/right stance durations
        l_stance = float(np.mean(l_contact))
        r_stance = float(np.mean(r_contact))
        leg_length_ratio = float(l_stance / (r_stance + 1e-9))
        leg_length_ratio = float(np.clip(leg_length_ratio, 0.7, 1.3))

        # Hip height / aspect ratio — derived from stance phase force shape
        # High hip height (normalised) = more upright posture
        double_support = float(np.mean((l_contact + r_contact) > 1))
        hip_height_norm = float(np.clip(0.35 + (1.0 - double_support) * 0.15, 0.3, 0.6))
        aspect_ratio = float(np.clip(cadence * 1.5, 1.0, 4.0))

        # Velocity proxies from COP movement speed
        dt   = 1.0 / _FS
        cop_vel    = np.abs(np.diff(cop_x_combined)) / dt
        head_vel   = float(np.percentile(cop_vel, 75) / 500.0)   # normalise
        hip_vel    = float(np.mean(cop_vel) / 500.0)
        ankle_vel_l = float(np.mean(np.abs(np.diff(l_cop_x))) / dt / 500.0)
        ankle_vel_r = float(np.mean(np.abs(np.diff(r_cop_x))) / dt / 500.0)

        # ── Assemble feature vector ───────────────────────────────────
        vec = np.array([
            sway_std, sway_range, sway_cv, sway_fft_peak, sway_fft_energy,
            torso_angle, torso_angle_std,
            l_knee_angle, r_knee_angle, knee_angle_diff,
            l_knee_var, r_knee_var,
            l_hip_angle, r_hip_angle,
            l_ankle_angle, r_ankle_angle,
            symmetry, stride_cv, stride_length_norm,
            cadence, step_width,
            heel_toe_diff_l, heel_toe_diff_r,
            leg_length_ratio, hip_height_norm, aspect_ratio,
            head_vel, hip_vel, ankle_vel_l, ankle_vel_r,
        ], dtype=np.float32)

        return np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=0.0)

    @staticmethod
    def _contact_events(contact: np.ndarray) -> np.ndarray:
        """Return frame indices of rising edges (heel-strike events)."""
        diff = np.diff(contact)
        return np.where(diff == 1)[0] + 1

    # ------------------------------------------------------------------
    # SEQUENCE DATASET (LSTM) FROM REAL DATA
    # ------------------------------------------------------------------

    def _make_sequences_from_real(
        self, seq_len: int = 60, max_per_class: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window LSTM sequences from VGRF and NDD stride-interval data.
        vgrf19: full per-frame COP features.
        ndd_ts / mat_txt: synthetic-walk sequences anchored at measured stride stats.
        """
        if not self.data_sources:
            return self._generate_synthetic_sequences(seq_len)

        _SKIP = {"format", "demographics", "readme", "info", "sha256sums",
                 "index", "niwar", "orcoh", "new-gait-ts"}
        seqs_by_class: Dict[int, list] = {0: [], 1: [], 2: []}

        for data_dir, fmt, glob_pat in self.data_sources:
            files = [
                f for f in sorted(data_dir.glob(glob_pat))
                if f.stem.lower().split(".")[0] not in _SKIP
            ]
            for f in files:
                label = self._file_label(f.stem, fmt)
                if label == -1:
                    continue
                if len(seqs_by_class[label]) >= max_per_class:
                    continue

                if fmt == "vgrf19":
                    seq_list = self._vgrf_to_sequences(f, seq_len)
                elif fmt in ("ndd_ts", "mat_txt"):
                    seq_list = self._stride_to_sequences(f, fmt, label, seq_len)
                else:
                    seq_list = []

                for s in seq_list:
                    if len(seqs_by_class[label]) >= max_per_class:
                        break
                    seqs_by_class[label].append(s)

        all_seqs, all_labels = [], []
        for cls, seqs in seqs_by_class.items():
            all_seqs.extend(seqs)
            all_labels.extend([cls] * len(seqs))

        if not all_seqs:
            return self._generate_synthetic_sequences(seq_len)

        print(f"  [dataset] LSTM sequences: {len(all_seqs)}")
        return np.array(all_seqs, dtype=np.float32), np.array(all_labels, dtype=np.int64)

    def _vgrf_to_sequences(self, path: Path, seq_len: int) -> list:
        """Sliding window over VGRF 19-col data → list of (seq_len, n_feat) arrays."""
        try:
            data = np.loadtxt(str(path))
        except Exception:
            return []
        if data.ndim != 2 or data.shape[1] < 19 or data.shape[0] < seq_len + 50:
            return []

        n_feat = len(self.FEATURE_NAMES)
        l_sens = data[:, 1:9]; r_sens = data[:, 9:17]
        f_left = data[:, 17];  f_right = data[:, 18]
        l_cop_x = (l_sens @ _L_SENSOR_XY[:, 0]) / (f_left  + 1e-9)
        r_cop_x = (r_sens @ _R_SENSOR_XY[:, 0]) / (f_right + 1e-9)
        f_total  = f_left + f_right + 1e-9
        cop      = (f_left * l_cop_x + f_right * r_cop_x) / f_total

        def znorm(x): s = np.std(x); return (x - np.mean(x)) / (s + 1e-9)
        frame_feats = np.stack([znorm(cop), znorm(f_left/f_total),
                                znorm(f_right/f_total), znorm(l_cop_x), znorm(r_cop_x)], axis=1)
        T = frame_feats.shape[0]
        full = np.zeros((T, n_feat), dtype=np.float32)
        full[:, :5] = frame_feats

        seqs = []
        step = seq_len // 2
        for start in range(0, T - seq_len, step):
            seqs.append(full[start:start + seq_len])
        return seqs

    def _stride_to_sequences(self, path: Path, fmt: str, label: int, seq_len: int) -> list:
        """
        Auto-correlated synthetic sequence anchored to measured stride stats.
        Works for both ndd_ts (13-col) and mat_txt (2-col) files.
        """
        try:
            data = np.loadtxt(str(path))
        except Exception:
            return []
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if fmt == "ndd_ts" and data.shape[1] >= 3:
            strides = np.concatenate([data[:, 1], data[:, 2]])
        else:
            strides = data[:, 1] if data.shape[1] >= 2 else data[:, 0]

        strides = strides[(strides > 0.3) & (strides < 3.5)]
        if len(strides) < 3:
            return []

        stride_cv = float(np.std(strides) / (np.mean(strides) + 1e-9))
        cadence   = float(1.0 / (np.mean(strides) + 1e-9))

        profile = GAIT_PROFILES[label]
        means   = np.array([profile[f][0] for f in self.FEATURE_NAMES])
        stds    = np.array([profile[f][1] for f in self.FEATURE_NAMES])
        idx     = {n: i for i, n in enumerate(self.FEATURE_NAMES)}
        means[idx["stride_cv"]]   = np.clip(stride_cv, 0, 0.5)
        means[idx["cadence_norm"]]= np.clip(cadence, 0.5, 3.0)

        state = self.rng.normal(means, stds)
        seq   = np.zeros((seq_len, len(means)), dtype=np.float32)
        for t in range(seq_len):
            state = state + self.rng.normal(0, stds * 0.1)
            state = np.clip(state, means - 4*stds, means + 4*stds)
            seq[t] = state
        return [seq]

    # ------------------------------------------------------------------
    # LABEL MAPPING
    # ------------------------------------------------------------------

    @staticmethod
    def _file_label(stem: str, fmt: str = "vgrf19") -> int:
        """
        Returns class label based on filename stem and dataset format.
          0 = NORMAL, 1 = WARNING, 2 = HIGH_RISK, -1 = skip
        """
        s = stem.upper()

        if fmt == "vgrf19":
            # gaitpdb: GaCo01_01, GaPt01_01 etc.
            if len(s) < 4:
                return -1
            group = s[2:4]
            if group in ("PT", "PA"): return 2
            if group == "HD":          return 2
            if group == "AL":          return 2
            if group == "CO":
                parts = s.split("_")
                return 1 if (len(parts) > 1 and parts[1] == "10") else 0
            return -1

        if fmt == "ndd_ts":
            # gaitndd: als1, hunt12, park3, control7 (no extension prefix)
            if s.startswith("ALS"):    return 2
            if s.startswith("HUNT"):   return 2
            if s.startswith("PARK"):   return 2
            if s.startswith("CONTROL"): return 0
            return -1

        if fmt == "mat_txt":
            # gait-maturation: all healthy children — numeric filenames only
            clean = s.replace("_", "").replace("-", "")
            return 0 if clean.isdigit() else -1

        return -1

    # ------------------------------------------------------------------
    # gaitndd .ts PARSER  (13-column stride-interval text)
    # ------------------------------------------------------------------

    def _extract_ts_features(self, path: Path, label: int) -> Optional[np.ndarray]:
        """
        gaitndd .ts columns (space-delimited, per stride event):
          0: time(s)  1: L_stride(s)  2: R_stride(s)
          3: L_swing  4: R_swing   (stance fraction = 1 - swing/stride)
          5-12: additional cadence/width metrics

        Computes stride_cv, symmetry, cadence; fills the rest from
        GAIT_PROFILES[label] profile means ± small noise.
        """
        try:
            data = np.loadtxt(str(path))
        except Exception:
            return None

        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] < 3 or data.shape[0] < 5:
            return None

        l_strides_raw = data[:, 1]
        r_strides_raw = data[:, 2]

        # Optional: swing-phase ratio from cols 3-4 (proxy for heel_toe diff)
        l_swing_frac = float(np.mean(data[:, 3] / (l_strides_raw + 1e-9))) if data.shape[1] > 4 else 0.33
        r_swing_frac = float(np.mean(data[:, 4] / (r_strides_raw + 1e-9))) if data.shape[1] > 4 else 0.33

        l_strides = l_strides_raw[(l_strides_raw > 0.3) & (l_strides_raw < 3.5)]
        r_strides = r_strides_raw[(r_strides_raw > 0.3) & (r_strides_raw < 3.5)]
        if len(l_strides) < 3 or len(r_strides) < 3:
            return None

        def cv(arr): return float(np.std(arr) / (np.mean(arr) + 1e-9))
        stride_cv   = (cv(l_strides) + cv(r_strides)) / 2.0
        l_mean      = float(np.mean(l_strides))
        r_mean      = float(np.mean(r_strides))
        symmetry    = float(min(l_mean, r_mean) / (max(l_mean, r_mean) + 1e-9))
        cadence     = float(1.0 / ((l_mean + r_mean) / 2.0 + 1e-9))

        heel_toe_l   = float(l_swing_frac - 0.33) * 10.0  # centre on normal
        heel_toe_r   = float(r_swing_frac - 0.33) * 10.0

        # Step width from col 11 if available (in m, normalise to 0-1 range)
        step_width = float(np.mean(np.abs(data[:, 11]))) / 0.3 if data.shape[1] > 11 else 0.20

        # Base vector from GAIT_PROFILES mean (pose features not in .ts data)
        profile = GAIT_PROFILES[label]
        vec     = np.array([profile[f][0] for f in self.FEATURE_NAMES], dtype=np.float32)
        noise   = np.array([profile[f][1] for f in self.FEATURE_NAMES], dtype=np.float32)
        vec    += self.rng.normal(0, noise * 0.25).astype(np.float32)

        # Overwrite directly-measured fields
        idx = {n: i for i, n in enumerate(self.FEATURE_NAMES)}
        vec[idx["stride_cv"]]          = float(np.clip(stride_cv, 0.0, 0.5))
        vec[idx["gait_symmetry"]]       = float(np.clip(symmetry, 0.4, 1.0))
        vec[idx["cadence_norm"]]        = float(np.clip(cadence,  0.5, 3.0))
        vec[idx["heel_toe_diff_l"]]     = float(np.clip(heel_toe_l, -0.15, 0.15))
        vec[idx["heel_toe_diff_r"]]     = float(np.clip(heel_toe_r, -0.15, 0.15))
        vec[idx["step_width_norm"]]     = float(np.clip(step_width, 0.0, 0.5))
        vec[idx["stride_length_norm"]]  = float(np.clip((l_mean + r_mean) / 2.0, 0.1, 2.0))

        return np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=0.0)

    # ------------------------------------------------------------------
    # gait-maturation .txt PARSER  (2-column stride-interval text)
    # ------------------------------------------------------------------

    def _extract_maturation_features(self, path: Path) -> Optional[np.ndarray]:
        """
        gait-maturation columns: time(s)  stride_interval(s)
        Single-column stride sequence — all subjects are healthy children (NORMAL).
        """
        try:
            data = np.loadtxt(str(path))
        except Exception:
            return None

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[0] < 5:
            return None

        # Stride intervals: if 2 cols take col 1, else treat col 0 as interval
        strides = data[:, 1] if data.shape[1] >= 2 else data[:, 0]
        strides = strides[(strides > 0.3) & (strides < 3.5)]
        if len(strides) < 3:
            return None

        stride_cv  = float(np.std(strides) / (np.mean(strides) + 1e-9))
        cadence    = float(1.0 / (np.mean(strides) + 1e-9))
        symmetry   = 1.0   # single-foot data — assume symmetric

        profile = GAIT_PROFILES[0]   # always NORMAL
        vec     = np.array([profile[f][0] for f in self.FEATURE_NAMES], dtype=np.float32)
        noise   = np.array([profile[f][1] for f in self.FEATURE_NAMES], dtype=np.float32)
        vec    += self.rng.normal(0, noise * 0.25).astype(np.float32)

        idx = {n: i for i, n in enumerate(self.FEATURE_NAMES)}
        vec[idx["stride_cv"]]         = float(np.clip(stride_cv, 0.0, 0.5))
        vec[idx["gait_symmetry"]]      = 1.0
        vec[idx["cadence_norm"]]       = float(np.clip(cadence,  0.5, 3.0))
        vec[idx["stride_length_norm"]] = float(np.clip(np.mean(strides), 0.1, 2.0))

        return np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=0.0)

    # ------------------------------------------------------------------
    # LABEL MAPPING
    # ------------------------------------------------------------------

    def _generate_synthetic_gait(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.cfg["synthetic_samples_per_class"]
        X_parts, y_parts = [], []
        for cls, profile in GAIT_PROFILES.items():
            means = np.array([profile[f][0] for f in self.FEATURE_NAMES])
            stds  = np.array([profile[f][1] for f in self.FEATURE_NAMES])
            X = self.rng.normal(means, stds, size=(n, len(means))).astype(np.float32)
            y = np.full(n, cls, dtype=np.int64)
            X_parts.append(X); y_parts.append(y)
        return np.vstack(X_parts), np.concatenate(y_parts)

    def _generate_synthetic_fall(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.cfg["synthetic_samples_per_class"]
        X_parts, y_parts = [], []
        for cls, profile in _FALL_PROFILES.items():
            names = self.FALL_FEATURE_NAMES
            means = np.array([profile[f][0] for f in names])
            stds  = np.array([profile[f][1] for f in names])
            X = self.rng.normal(means, stds, size=(n, len(means))).astype(np.float32)
            y = np.full(n, cls, dtype=np.int64)
            X_parts.append(X); y_parts.append(y)
        return np.vstack(X_parts), np.concatenate(y_parts)

    def _generate_synthetic_sequences(
        self, seq_len: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = self.cfg["synthetic_samples_per_class"]
        seqs, labels = [], []
        for cls, profile in GAIT_PROFILES.items():
            means = np.array([profile[f][0] for f in self.FEATURE_NAMES])
            stds  = np.array([profile[f][1] for f in self.FEATURE_NAMES])
            for _ in range(n):
                state = self.rng.normal(means, stds)
                seq   = np.zeros((seq_len, len(means)), dtype=np.float32)
                for t in range(seq_len):
                    state = state + self.rng.normal(0, stds * 0.15)
                    state = np.clip(state, means - 4*stds, means + 4*stds)
                    seq[t] = state
                seqs.append(seq); labels.append(cls)
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)

    # ------------------------------------------------------------------
    # CACHE HELPERS
    # ------------------------------------------------------------------

    def save_cached(self, path: str, X: np.ndarray, y: np.ndarray):
        np.savez_compressed(path, X=X, y=y)

    def load_cached(self, path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not os.path.exists(path):
            return None, None
        data = np.load(path)
        return data["X"], data["y"]

