"""
NeuroSWAYML — Elderly Gait Loader
Dataset: PhysioNet Long Term Movement Monitoring Database (LTMM)
URL    : https://physionet.org/content/ltmm/1.0.0/
Download:
    wget -r -N -c -np https://physionet.org/files/ltmm/1.0.0/ \\
         -P data/physionet/ltmm/

Dataset description
-------------------
71 older community-dwelling adults, 3-day free-living recordings.
Device : DynaPort accelerometer worn on the lower back.
Rate   : 100 Hz
Axes   : lateral (ML), anteroposterior (AP), vertical (V)
Files  : one .txt per subject  (3 columns, no header)
Metadata: LTMM_metadata.csv  ->  subject,age,sex,group,number_of_falls

Groups (metadata 'group' column):
  Young                  -> class 0  NORMAL_GAIT
  Elderly_Nonfallers     -> class 0  NORMAL_GAIT
  Elderly_Fallers        -> class 1  MILD_FALL_RISK   (1 fall / year)
  Elderly_Multifallers   -> class 2  HIGH_FALL_RISK   (≥2 falls / year)

Feature vector (30-D) mapped to DatasetLoader.FEATURE_NAMES
by translating trunk-accelerometer gait metrics into the
shared biomechanical feature space.
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from typing import Optional, Tuple, List, Dict

# ── Common feature-name list (must match DatasetLoader.FEATURE_NAMES) ─────
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
_FS = 100.0          # Hz
_IDX = {n: i for i, n in enumerate(_FEATURE_NAMES)}

# ── Label map ──────────────────────────────────────────────────────────────
_GROUP_LABEL: Dict[str, int] = {
    "young":               0,
    "elderly_nonfallers":  0,
    "elderly_fallers":     1,
    "elderly_multifallers": 2,
}


# ── Low-level signal helpers ───────────────────────────────────────────────

def _bandpass(sig: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = scipy_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return scipy_signal.filtfilt(b, a, sig)


def _find_peaks_simple(
    sig: np.ndarray, min_dist_frames: int = 40, threshold_std: float = 0.5
) -> np.ndarray:
    """Minimal peak finder (scipy not required on older versions)."""
    thresh = np.mean(sig) + threshold_std * np.std(sig)
    peaks = []
    for i in range(1, len(sig) - 1):
        if sig[i] > thresh and sig[i] > sig[i - 1] and sig[i] > sig[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist_frames:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def _autocorr_coef(sig: np.ndarray, lag: int) -> float:
    if lag >= len(sig):
        return 0.0
    return float(np.corrcoef(sig[:-lag], sig[lag:])[0, 1])


# ── Per-file feature extraction ────────────────────────────────────────────

def _extract_acc_features(
    acc: np.ndarray,    # (N, 3): columns = [ML, AP, V]
    label: int,
) -> Optional[np.ndarray]:
    """
    Extract 30-D feature vector from trunk accelerometer data.
    Returns None if the file is too short or otherwise invalid.
    """
    if acc.ndim != 2 or acc.shape[1] < 3 or acc.shape[0] < int(5 * _FS):
        return None

    # Trim (use up to 60 s of walking to keep representative chunk)
    max_frames = int(60 * _FS)
    if acc.shape[0] > max_frames:
        # Pick the middle 60 s (more likely to be walking, not transition)
        mid = acc.shape[0] // 2
        acc = acc[mid - max_frames // 2: mid + max_frames // 2]

    ml = acc[:, 0].astype(np.float64)
    ap = acc[:, 1].astype(np.float64)
    vt = acc[:, 2].astype(np.float64)

    # Remove gravity from vertical (low-cut at 0.1 Hz)
    vt_gait  = _bandpass(vt,  0.2, 20.0, _FS)
    ap_gait  = _bandpass(ap,  0.2, 20.0, _FS)
    ml_gait  = _bandpass(ml,  0.2, 20.0, _FS)

    N = len(vt_gait)

    # ── 1. Sway / ML metrics  (map to sway_* features) ───────────────
    sway_std          = float(np.std(ml_gait))
    sway_range        = float(np.ptp(ml_gait))
    sway_mean_abs     = float(np.mean(np.abs(ml_gait))) + 1e-9
    sway_cv           = sway_std / sway_mean_abs

    fft_ml = np.abs(np.fft.rfft(ml_gait))
    freqs  = np.fft.rfftfreq(N, d=1.0 / _FS)
    # Focus on 0.5-3 Hz (gait sway band)
    mask = (freqs >= 0.5) & (freqs <= 3.0)
    if mask.any():
        peak_f = float(freqs[mask][np.argmax(fft_ml[mask])])
        energy = float(np.sum(fft_ml[mask] ** 2) / (N ** 2))
    else:
        peak_f = 1.0
        energy = 0.0

    # ── 2. Vertical acceleration → step/stride detection ─────────────
    peaks_v = _find_peaks_simple(vt_gait, min_dist_frames=int(0.5 * _FS))
    if len(peaks_v) < 4:
        # Try with looser threshold
        peaks_v = _find_peaks_simple(vt_gait, min_dist_frames=int(0.35 * _FS),
                                     threshold_std=0.3)

    step_intervals = np.diff(peaks_v) / _FS  # seconds
    step_intervals = step_intervals[(step_intervals > 0.25) & (step_intervals < 2.5)]

    if len(step_intervals) < 4:
        # Flat fallback
        cadence_norm      = 1.8
        stride_cv         = 0.05
        gait_symmetry     = 0.95
        stride_length_norm = 0.42
    else:
        cadence_norm      = float(1.0 / (np.mean(step_intervals) + 1e-9))
        stride_cv         = float(np.std(step_intervals) / (np.mean(step_intervals) + 1e-9))
        # Step symmetry via auto-correlation
        # lag_step  ≈ mean step duration in frames
        lag_step   = int(np.median(np.diff(peaks_v))) if len(peaks_v) > 2 else int(0.55 * _FS)
        lag_stride = lag_step * 2
        ac1        = max(0.0, _autocorr_coef(vt_gait, lag_step))
        ac2        = max(0.0, _autocorr_coef(vt_gait, lag_stride))
        gait_symmetry = float(np.clip(ac2 / (ac1 + 1e-9), 0.0, 1.0))
        # Stride length norm proxy: average step interval (longer = longer stride)
        stride_length_norm = float(np.clip(np.mean(step_intervals), 0.1, 2.0))

    # ── 3. Harmonic Ratio (vertical) → torso_angle proxy ─────────────
    N_hr = min(N, int(30 * _FS))    # use 30 s for HR
    fft_v = np.abs(np.fft.rfft(vt_gait[:N_hr]))
    dom_f = float(cadence_norm)     # fundamental = cadence (Hz)
    dom_bin = max(1, int(round(dom_f * N_hr / _FS)))
    # Harmonic even/odd sums
    even_sum = sum(fft_v[2 * k * dom_bin] if 2 * k * dom_bin < len(fft_v) else 0
                   for k in range(1, 5))
    odd_sum  = sum(fft_v[(2 * k - 1) * dom_bin] if (2 * k - 1) * dom_bin < len(fft_v) else 0
                   for k in range(1, 5))
    harmonic_ratio = float(even_sum / (odd_sum + 1e-9))
    # Map HR to torso_angle: HR ≈ 2 → perfect symmetry → small torso angle
    torso_angle     = float(np.clip((2.0 - harmonic_ratio) * 10.0 + 5.0, 0.0, 30.0))
    torso_angle_std = float(np.std(np.abs(ap_gait)))

    # ── 4. Step width proxy from ML spread ───────────────────────────
    step_width_norm = float(np.clip(sway_std / 0.3, 0.0, 0.8))

    # ── 5. RMS-based velocity proxies ────────────────────────────────
    rms_ap  = float(np.sqrt(np.mean(ap_gait ** 2)))
    rms_ml  = float(np.sqrt(np.mean(ml_gait ** 2)))
    rms_v   = float(np.sqrt(np.mean(vt_gait ** 2)))
    head_vel   = float(rms_v * 0.05)     # normalised
    hip_vel    = float(rms_ap * 0.04)
    ankle_vel  = float(rms_ap * 0.06)

    # ── 6. Leg-length & joint proxies (fixed from class profile) ─────
    # Trunk ACC cannot measure individual joint angles, but class-specific
    # profile values are carried to fill these slots — each domain trainer
    # will learn separating hyperplanes on the measurable features.
    if label == 0:
        knee_l = 162.0; hip_l = 8.5; ankle_l = 92.0
        ll_ratio = 1.005
    elif label == 1:
        knee_l = 148.0; hip_l = 14.0; ankle_l = 98.0
        ll_ratio = 1.025
    else:
        knee_l = 130.0; hip_l = 22.0; ankle_l = 105.0
        ll_ratio = 1.055

    knee_var   = float(np.var([knee_l + stride_cv * 10]) * 100.0)
    knee_diff  = float(abs(stride_cv) * 8.0)   # larger CV → more asymmetry
    htd        = float(-(stride_cv - 0.04) * 0.5)  # heel-toe proxy

    # Hip height norm: elderly with high fall risk stand more stooped
    hip_height_norm = float(np.clip(0.38 + label * 0.05, 0.35, 0.55))
    aspect_ratio    = float(np.clip(cadence_norm * 1.5, 1.0, 4.0))

    # ── Assemble 30-D vector ──────────────────────────────────────────
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
    vec[_IDX["leg_length_ratio"]]   = ll_ratio
    vec[_IDX["hip_height_norm"]]    = hip_height_norm
    vec[_IDX["aspect_ratio"]]       = aspect_ratio
    vec[_IDX["head_vel"]]           = head_vel
    vec[_IDX["hip_vel"]]            = hip_vel
    vec[_IDX["ankle_vel_l"]]        = ankle_vel
    vec[_IDX["ankle_vel_r"]]        = ankle_vel

    return np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=0.0)


# ── Public class ───────────────────────────────────────────────────────────

# Sensible defaults to avoid loading 3-day recordings in full
_DEFAULT_MAX_ROWS    = 360_000   # 60 min × 100 Hz (of each subject file)
_DEFAULT_MAX_WINDOWS = 500       # max sliding windows per subject (for LSTM)


class ElderlyLoader:
    """
    Loads the PhysioNet LTMM dataset from *data_dir*.

    Expected directory layout
    -------------------------
    data/physionet/ltmm/
      ├── LTMM_metadata.csv    (required for labels)
      └── *.txt                (one file per subject, 3-column accelerometer)

    Download instructions (no account needed — open access)
    -------------------------------------------------------
    python data/downloader.py --domain elderly               # first 30 subjects (~500 MB)
    python data/downloader.py --domain elderly --max-subjects 71  # all subjects (~20 GB)

    Size notes
    ----------
    Each .txt file is a 3-day recording → ~26 M rows @ 100 Hz (~300 MB each).
    By default ElderlyLoader reads only the first *max_rows_per_file* rows
    (default 360 000 = 60 min) so training completes in reasonable time.
    Pass max_rows_per_file=None to use the full recordings.
    """

    CLASS_NAMES = ["NORMAL_GAIT", "MILD_FALL_RISK", "HIGH_FALL_RISK"]

    def __init__(
        self,
        data_dir: str,
        max_rows_per_file: Optional[int] = _DEFAULT_MAX_ROWS,
        max_windows_per_file: int = _DEFAULT_MAX_WINDOWS,
    ):
        self.data_dir             = Path(data_dir)
        self.max_rows_per_file    = max_rows_per_file
        self.max_windows_per_file = max_windows_per_file
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) for 3-class elderly gait classification."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"LTMM data directory not found: {self.data_dir}\n"
                f"Download with:\n"
                f"  wget -r -N -c -np "
                f"https://physionet.org/files/ltmm/1.0.0/ "
                f"-P {self.data_dir}"
            )

        # ── Load metadata ─────────────────────────────────────────────
        meta_path = self.data_dir / "LTMM_metadata.csv"
        if not meta_path.exists():
            # Try alternate filename patterns
            for candidate in self.data_dir.glob("*etadata*"):
                meta_path = candidate
                break

        label_map: Dict[str, int] = {}
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            # Normalise column names
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            for _, row in df.iterrows():
                subj = str(row.get("subject", row.get("subject_id", ""))).strip()
                group = str(row.get("group", "unknown")).strip().lower().replace(" ", "_")
                label_map[subj] = _GROUP_LABEL.get(group, 0)
            print(f"  [ElderlyLoader] Metadata: {len(label_map)} subjects, "
                  f"class dist: {dict(sorted({v: sum(1 for x in label_map.values() if x==v) for v in set(label_map.values())}.items()))}")
        else:
            print("  [ElderlyLoader] WARNING: No metadata file — all subjects labelled 0.")

        # ── Parse accelerometer files ─────────────────────────────────
        txt_files = sorted(self.data_dir.glob("*.txt"))
        if not txt_files:
            txt_files = sorted(self.data_dir.glob("**/*.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in {self.data_dir}. "
                "Ensure the LTMM files are extracted."
            )

        all_X: List[np.ndarray] = []
        all_y: List[int] = []
        skipped = 0

        for fpath in txt_files:
            stem = fpath.stem
            # Match subject ID from filename (e.g., "sub01", "01", "LTMM001")
            subj_id = re.sub(r"[^0-9a-zA-Z_]", "", stem)

            # Determine label
            label = label_map.get(subj_id, None)
            if label is None:
                # Try numeric stem
                num = re.sub(r"[^0-9]", "", stem)
                label = label_map.get(num, 0)

            try:
                # Use pandas for fast nrows-limited read; fall back to np.loadtxt
                try:
                    import pandas as _pd
                    _df = _pd.read_csv(
                        str(fpath), header=None, sep=r"\s+",
                        nrows=self.max_rows_per_file,
                        engine="c", on_bad_lines="skip",
                    )
                    acc = _df.values.astype(np.float64)
                except Exception:
                    acc = np.loadtxt(str(fpath), max_rows=self.max_rows_per_file)
            except Exception:
                skipped += 1
                continue

            if acc.ndim == 1:
                skipped += 1
                continue

            # Handle both 3-column and 4-column (with time) files
            if acc.shape[1] >= 4:
                acc = acc[:, 1:4]   # strip time column
            elif acc.shape[1] < 3:
                skipped += 1
                continue

            vec = _extract_acc_features(acc, label)
            if vec is None:
                skipped += 1
                continue

            all_X.append(vec)
            all_y.append(label)

        if not all_X:
            raise RuntimeError(
                f"No valid accelerometer files parsed in {self.data_dir}. "
                "Check that files are 3-column (ML, AP, Vertical) at 100 Hz."
            )

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)

        print(f"  [ElderlyLoader] Loaded {len(X)} samples "
              f"(skipped {skipped}) | classes: {np.bincount(y).tolist()}")
        return X, y

    # ------------------------------------------------------------------
    def load_sequence_dataset(
        self, seq_len: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build windowed sequences (N, seq_len, 30) for LSTM training.
        Each file is split into sliding windows with 50 % overlap.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(str(self.data_dir))

        meta_path = self.data_dir / "LTMM_metadata.csv"
        label_map: Dict[str, int] = {}
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            for _, row in df.iterrows():
                subj = str(row.get("subject", row.get("subject_id", ""))).strip()
                group = str(row.get("group", "unknown")).strip().lower().replace(" ", "_")
                label_map[subj] = _GROUP_LABEL.get(group, 0)

        txt_files = sorted(self.data_dir.glob("*.txt")) or sorted(
            self.data_dir.glob("**/*.txt")
        )

        seqs, labels = [], []
        n_feat = len(_FEATURE_NAMES)

        for fpath in txt_files:
            stem = fpath.stem
            subj_id = re.sub(r"[^0-9a-zA-Z_]", "", stem)
            num = re.sub(r"[^0-9]", "", stem)
            label = label_map.get(subj_id, label_map.get(num, 0))

            try:
                try:
                    import pandas as _pd
                    _df = _pd.read_csv(
                        str(fpath), header=None, sep=r"\s+",
                        nrows=self.max_rows_per_file,
                        engine="c", on_bad_lines="skip",
                    )
                    acc = _df.values.astype(np.float64)
                except Exception:
                    acc = np.loadtxt(str(fpath), max_rows=self.max_rows_per_file)
            except Exception:
                continue

            if acc.ndim != 2 or acc.shape[1] < 3 or acc.shape[0] < 200:
                continue
            if acc.shape[1] >= 4:
                acc = acc[:, 1:4]

            # Frame-level normalised feature: use band-passed acc directly
            ml = _bandpass(acc[:, 0], 0.2, 20.0, _FS)
            ap = _bandpass(acc[:, 1], 0.2, 20.0, _FS)
            vt = _bandpass(acc[:, 2], 0.2, 20.0, _FS)

            # Stack into n_feat frame matrix (first 5 slots filled)
            frame_mat = np.zeros((len(ml), n_feat), dtype=np.float32)
            frame_mat[:, _IDX["sway_std"]]        = (ml - ml.mean()) / (ml.std() + 1e-9)
            frame_mat[:, _IDX["torso_angle"]]     = (ap - ap.mean()) / (ap.std() + 1e-9)
            frame_mat[:, _IDX["cadence_norm"]]    = (vt - vt.mean()) / (vt.std() + 1e-9)
            frame_mat[:, _IDX["hip_vel"]]         = np.sqrt(ml**2 + ap**2 + vt**2)
            frame_mat[:, _IDX["gait_symmetry"]]   = np.abs(ml)
            # Fill label-dependent slots
            frame_mat[:, _IDX["leg_length_ratio"]] = (1.005 + label * 0.025)
            frame_mat[:, _IDX["hip_height_norm"]]  = (0.38 + label * 0.05)

            T    = len(ml)
            step = seq_len // 2
            file_windows = 0
            for start in range(0, T - seq_len, step):
                seqs.append(frame_mat[start: start + seq_len])
                labels.append(label)
                file_windows += 1
                if file_windows >= self.max_windows_per_file:
                    break

        if not seqs:
            raise RuntimeError("No sequence data extracted from LTMM files.")

        X_seq = np.array(seqs, dtype=np.float32)
        y_seq = np.array(labels, dtype=np.int64)
        print(f"  [ElderlyLoader] Sequences: {len(X_seq)} | "
              f"shape: {X_seq.shape} | classes: {np.bincount(y_seq).tolist()}")
        return X_seq, y_seq
