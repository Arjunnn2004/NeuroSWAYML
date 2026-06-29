"""
NeuroSWAYML — URFD Video Loader  (Elderly Fall Risk Domain)
============================================================
Dataset : University of Rzeszów Fall Detection (URFD)
URL     : http://fenix.ur.edu.pl/~mkepski/ds/uf.html
Licence : Free for research use

Size    : ~240 MB total across 70 individual zip files
  fall-01-cam0-rgb.zip … fall-30-cam0-rgb.zip  — 30 fall sequences
  adl-01-cam0-rgb.zip  … adl-40-cam0-rgb.zip   — 40 ADL sequences
  (each zip contains PNG frames for one sequence, ~2–6 MB each)

WHY this is the right dataset for this project
-----------------------------------------------
The live app uses MediaPipe Pose → FeatureExtractor → 30-D vector.
ALL other gait datasets (LTMM, HBEDB, GaitRec) measure different
signals (accelerometer / force-platform / GRF) that require lossy
signal mapping into the feature space.

URFD gives us VIDEO frames of real people, so we run the EXACT same
MediaPipe + FeatureExtractor pipeline used at inference time, producing
features with ZERO domain gap.

Label mapping
-------------
  adl-*  sequences  → class 0  NORMAL_GAIT
  fall-* first 60%  → class 1  MILD_FALL_RISK   (pre-fall / losing balance)
  fall-* last  40%  → class 2  HIGH_FALL_RISK   (active fall)

Processing pipeline
-------------------
  Video frames → MediaPipe Pose → FeatureExtractor → 30-D vector per frame
  Results are cached to data/urfd/features_cache.npz so re-processing
  only happens when the cache is missing or --force-reprocess is set.

Download
--------
  python data/downloader.py --domain elderly          # auto-download
  python data/downloader.py --domain elderly --check  # check status

Manual (if auto-download fails)
--------------------------------
  1. Go to http://fenix.ur.edu.pl/~mkepski/ds/uf.html
  2. Download every  fall-NN-cam0-rgb.zip  and  adl-NN-cam0-rgb.zip  listed
  3. Extract all zips into  data/urfd/
  4. Run  python training/train_elderly.py
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

# ── Constants ──────────────────────────────────────────────────────────────
_BASE_URL       = "http://fenix.ur.edu.pl/~mkepski/ds/data"
_N_FALLS        = 30   # fall-01 … fall-30
_N_ADLS         = 40   # adl-01  … adl-40

_PRE_FALL_FRAC  = 0.60   # first 60% of fall sequence → MILD_FALL_RISK
_FALL_FRAC      = 0.40   # last  40% → HIGH_FALL_RISK
_MIN_FRAMES_VEC = 15     # skip sequences shorter than this

_CACHE_FILE     = "features_cache.npz"
_IMG_EXTS       = {".png", ".jpg", ".jpeg", ".bmp"}

# MediaPipe pose landmarker model (new Tasks API, 0.10+)
_MP_MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_MP_MODEL_NAME  = "pose_landmarker_lite.task"


def _sequence_zip_list() -> List[Tuple[str, str]]:
    """
    Returns [(filename, url), ...] for every cam0-rgb sequence zip:
      fall-01-cam0-rgb.zip … fall-30-cam0-rgb.zip
      adl-01-cam0-rgb.zip  … adl-40-cam0-rgb.zip
    """
    items: List[Tuple[str, str]] = []
    for i in range(1, _N_FALLS + 1):
        name = f"fall-{i:02d}-cam0-rgb.zip"
        items.append((name, f"{_BASE_URL}/{name}"))
    for i in range(1, _N_ADLS + 1):
        name = f"adl-{i:02d}-cam0-rgb.zip"
        items.append((name, f"{_BASE_URL}/{name}"))
    return items


class URFDLoader:
    """
    Loads the URFD dataset, processes frames through MediaPipe + FeatureExtractor,
    and returns (X, y) arrays that exactly match the live inference feature space.

    Parameters
    ----------
    data_dir       : root URFD directory  (default: data/urfd)
    use_cam0_only  : use only cam0 view to avoid label duplication  (default True)
    force_reprocess: ignore existing cache and re-run MediaPipe     (default False)
    max_sequences  : cap number of sequences per class (None = all)
    """

    CLASS_NAMES = ["NORMAL_GAIT", "MILD_FALL_RISK", "HIGH_FALL_RISK"]

    def __init__(
        self,
        data_dir: str = "data/urfd",
        use_cam0_only: bool = True,
        force_reprocess: bool = False,
        max_sequences: Optional[int] = None,
    ):
        self.data_dir        = Path(data_dir)
        self.use_cam0_only   = use_cam0_only
        self.force_reprocess = force_reprocess
        self.max_sequences   = max_sequences
        self._cache_path     = self.data_dir / _CACHE_FILE

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def download(self, verbose: bool = True) -> bool:
        """
        Download all per-sequence cam0-rgb zip files from URFD.
        30 fall zips + 40 ADL zips = 70 files.
        Skips files already on disk.  Returns True if all succeeded.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        items = _sequence_zip_list()
        total  = len(items)
        failed = 0
        done   = 0
        for name, url in items:
            dest = self.data_dir / name
            if dest.exists():
                done += 1
                if verbose:
                    print(f"  [{done:2d}/{total}] ✓ {name}", end="\r", flush=True)
                continue
            ok = self._download_one(url, dest, done, total, verbose)
            if not ok:
                failed += 1
            done += 1
        if verbose:
            print()
            if failed:
                print(f"  [URFDLoader] {done-failed}/{total} zips downloaded; {failed} failed.")
            else:
                print(f"  [URFDLoader] All {total} zips ready.")
        return failed == 0

    def extract(self, verbose: bool = True):
        """Extract every downloaded per-sequence zip into data_dir."""
        zips = list(self.data_dir.glob("*-cam0-rgb.zip"))
        if not zips:
            return
        for i, fpath in enumerate(sorted(zips), 1):
            seq_name = fpath.stem            # e.g. fall-01-cam0-rgb
            seq_dir  = self.data_dir / seq_name
            if seq_dir.exists() and any(seq_dir.glob("*.png")):
                continue                      # already extracted
            if verbose:
                print(f"  Extracting {fpath.name} …", end="\r", flush=True)
            with zipfile.ZipFile(fpath, "r") as zf:
                zf.extractall(self.data_dir)
        if verbose:
            n = len(list(self.data_dir.glob("*-cam0-rgb")))
            print(f"  Extracted {n} sequence dirs → {self.data_dir}                  ")

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (X, y):
          X: (N, 30) float32  — MediaPipe+FeatureExtractor features
          y: (N,)   int64     — 0/1/2 labels
        Uses cache if available.
        """
        if not self.force_reprocess and self._cache_path.exists():
            print(f"  [URFDLoader] Loading cached features from {self._cache_path}")
            data = np.load(self._cache_path)
            X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
            print(f"  [URFDLoader] Cached: {X.shape}  classes: {np.bincount(y).tolist()}")
            return X, y

        return self._process_videos()

    def load_sequence_dataset(
        self, seq_len: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (N, seq_len, 30) tensor for LSTM training.
        Uses consecutive frames from each sequence as natural temporal windows.
        """
        seqs_list, labels_list = self._collect_sequences(seq_len)

        if not seqs_list:
            raise RuntimeError(
                "No sequences built. Run load() first to process videos."
            )

        X_seq = np.array(seqs_list, dtype=np.float32)
        y_seq = np.array(labels_list, dtype=np.int64)
        print(f"  [URFDLoader] Sequences: {len(X_seq)}  shape: {X_seq.shape}  "
              f"classes: {np.bincount(y_seq).tolist()}")
        return X_seq, y_seq

    def is_ready(self) -> bool:
        """Returns True if data is downloaded and processable."""
        if self._cache_path.exists():
            return True
        seq_dirs = self._find_sequence_dirs()
        return len(seq_dirs) > 0

    def status(self) -> str:
        if self._cache_path.exists():
            data = np.load(self._cache_path)
            n = len(data["y"])
            return f"✓ Cache ready ({n} samples)"
        seq_dirs = self._find_sequence_dirs()
        if seq_dirs:
            return f"✓ {len(seq_dirs)} sequence dirs found (not yet processed)"
        zips = list(self.data_dir.glob("*.zip"))
        if zips:
            return f"✓ {len(zips)} zip(s) downloaded (not yet extracted)"
        return "✗ Not downloaded"

    # ------------------------------------------------------------------
    # CORE PROCESSING — MediaPipe + FeatureExtractor
    # ------------------------------------------------------------------

    def _ensure_pose_model(self) -> str:
        """Download pose_landmarker_lite.task if not already present."""
        model_path = self.data_dir / _MP_MODEL_NAME
        if not model_path.exists():
            print(f"  Downloading MediaPipe pose model (~3 MB) …", end="", flush=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_MP_MODEL_URL, str(model_path))
            print(" done.")
        return str(model_path)

    def _make_landmarker(self, model_path: str):
        """Create a MediaPipe Tasks PoseLandmarker (IMAGE mode, stateless per-frame)."""
        from mediapipe.tasks import python as mp_tp
        from mediapipe.tasks.python import vision as mp_vision

        base_opts = mp_tp.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        return mp_vision.PoseLandmarker.create_from_options(opts)

    def _process_videos(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run MediaPipe on every frame of every URFD sequence."""
        try:
            import cv2
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(
                f"Required package missing: {e}\n"
                "Install with:  pip install opencv-python mediapipe"
            ) from e

        # Import FeatureExtractor from project root
        _project_root = str(Path(__file__).parent.parent.parent)
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from data.feature_extractor import FeatureExtractor

        seq_dirs = self._find_sequence_dirs()
        if not seq_dirs:
            raise FileNotFoundError(
                f"No URFD sequence directories found in {self.data_dir}\n"
                "Run:  python data/downloader.py --domain elderly"
            )

        model_path = self._ensure_pose_model()

        print(f"  [URFDLoader] Found {len(seq_dirs)} sequence directories.")
        print(f"  [URFDLoader] Processing with MediaPipe Pose + FeatureExtractor …")
        print(f"               (this takes ~5–15 min first time; cached afterwards)")

        all_X: List[np.ndarray] = []
        all_y: List[int]        = []
        skipped = 0

        fall_dirs = [d for d in seq_dirs if d.name.lower().startswith("fall")]
        adl_dirs  = [d for d in seq_dirs if d.name.lower().startswith("adl")]

        if self.max_sequences:
            fall_dirs = fall_dirs[:self.max_sequences]
            adl_dirs  = adl_dirs[:self.max_sequences]

        total = len(fall_dirs) + len(adl_dirs)
        done  = 0

        with self._make_landmarker(model_path) as landmarker:
            # ── Process ADL sequences (all frames → class 0) ──────────────
            for seq_dir in adl_dirs:
                frames_X = self._process_sequence_dir(seq_dir, landmarker, FeatureExtractor)
                if frames_X is not None and len(frames_X) >= _MIN_FRAMES_VEC:
                    all_X.extend(frames_X)
                    all_y.extend([0] * len(frames_X))
                else:
                    skipped += 1
                done += 1
                print(f"  [{done:3d}/{total}] {seq_dir.name}  (ADL → class 0)",
                      end="\r", flush=True)

            # ── Process fall sequences (split → class 1 + class 2) ────────
            for seq_dir in fall_dirs:
                frames_all = self._process_sequence_dir(seq_dir, landmarker, FeatureExtractor)
                if frames_all is None or len(frames_all) < _MIN_FRAMES_VEC:
                    skipped += 1
                    done += 1
                    continue

                n     = len(frames_all)
                split = int(n * _PRE_FALL_FRAC)

                all_X.extend(frames_all[:split])
                all_y.extend([1] * split)
                all_X.extend(frames_all[split:])
                all_y.extend([2] * (n - split))

                done += 1
                print(f"  [{done:3d}/{total}] {seq_dir.name}  "
                      f"(fall: {split} mild + {n-split} high)",
                      end="\r", flush=True)

        print()

        if not all_X:
            raise RuntimeError(
                "MediaPipe found no valid pose in any URFD frame.\n"
                "Ensure the sequence folders contain .png/.jpg frame images."
            )

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)

        np.savez_compressed(str(self._cache_path), X=X, y=y)
        print(f"  [URFDLoader] Saved cache → {self._cache_path}")
        print(f"  [URFDLoader] Result: {X.shape}  "
              f"classes: {np.bincount(y).tolist()}  skipped: {skipped}")
        return X, y

    def _process_sequence_dir(
        self,
        seq_dir: Path,
        landmarker,
        FeatureExtractor,
    ) -> Optional[List[np.ndarray]]:
        """Run MediaPipe + FeatureExtractor on all frames in one sequence dir."""
        import cv2
        import mediapipe as mp

        frame_files = sorted([
            f for f in seq_dir.iterdir()
            if f.suffix.lower() in _IMG_EXTS
        ])
        if not frame_files:
            return None

        feat_ex = FeatureExtractor(fps=30.0)
        results_list: List[np.ndarray] = []

        for img_path in frame_files:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_img)

            if not result.pose_landmarks:
                continue

            # Tasks API: pose_landmarks[person_idx] is a list of NormalizedLandmark
            lm_2d = result.pose_landmarks[0]
            lm_3d = (result.pose_world_landmarks[0]
                     if result.pose_world_landmarks else None)

            feat = feat_ex.extract(lm_2d, lm_3d, w, h)
            results_list.append(feat.copy())

        return results_list if results_list else None

    # ------------------------------------------------------------------
    # SEQUENCE DATASET (for LSTM)
    # ------------------------------------------------------------------

    def _collect_sequences(
        self, seq_len: int
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load per-sequence raw feature arrays from cache or processing,
        then build overlapping windows of length seq_len.
        """
        # Try loading per-sequence cache
        seq_cache = self.data_dir / "seq_cache.npz"
        if not self.force_reprocess and seq_cache.exists():
            data = np.load(seq_cache, allow_pickle=True)
            all_seqs = data["seqs"]     # object array of variable-length arrays
            all_lbls = data["labels"]
        else:
            all_seqs, all_lbls = self._build_per_sequence_arrays()
            np.savez_compressed(str(seq_cache), seqs=all_seqs, labels=all_lbls)

        windows, win_labels = [], []
        step = seq_len // 2

        for seq_feat, base_label in zip(all_seqs, all_lbls):
            seq_feat = np.array(seq_feat, dtype=np.float32)
            T = len(seq_feat)
            if T < seq_len:
                continue
            for start in range(0, T - seq_len, step):
                windows.append(seq_feat[start: start + seq_len])
                win_labels.append(int(base_label))

        return windows, win_labels

    def _build_per_sequence_arrays(self):
        """Return object array of per-sequence feature matrices."""
        try:
            import cv2
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(f"Required package missing: {e}") from e

        _project_root = str(Path(__file__).parent.parent.parent)
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from data.feature_extractor import FeatureExtractor

        model_path = self._ensure_pose_model()
        seq_dirs   = self._find_sequence_dirs()

        fall_dirs = [d for d in seq_dirs if d.name.lower().startswith("fall")]
        adl_dirs  = [d for d in seq_dirs if d.name.lower().startswith("adl")]
        if self.max_sequences:
            fall_dirs = fall_dirs[:self.max_sequences]
            adl_dirs  = adl_dirs[:self.max_sequences]

        seqs, labels = [], []

        with self._make_landmarker(model_path) as landmarker:
            for seq_dir in adl_dirs:
                feats = self._process_sequence_dir(seq_dir, landmarker, FeatureExtractor)
                if feats and len(feats) >= _MIN_FRAMES_VEC:
                    seqs.append(feats)
                    labels.append(0)

            for seq_dir in fall_dirs:
                feats = self._process_sequence_dir(seq_dir, landmarker, FeatureExtractor)
                if feats and len(feats) >= _MIN_FRAMES_VEC:
                    n     = len(feats)
                    split = int(n * _PRE_FALL_FRAC)
                    seqs.append(feats[:split] if split >= _MIN_FRAMES_VEC else feats)
                    labels.append(1)
                    if n - split >= _MIN_FRAMES_VEC:
                        seqs.append(feats[split:])
                        labels.append(2)

        return np.array(seqs, dtype=object), np.array(labels, dtype=np.int64)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _find_sequence_dirs(self) -> List[Path]:
        """
        Find all URFD sequence directories.
        Handles various extract layouts:
          data/urfd/fall-01-cam0-rgb/
          data/urfd/falls/fall-01-cam0-rgb/
          data/urfd/urfd-falls/fall-01-cam0-rgb/
        """
        candidates: List[Path] = []

        for d in self.data_dir.rglob("*"):
            if not d.is_dir():
                continue
            name = d.name.lower()
            if name.startswith(("fall-", "adl-")):
                # skip cam1 if use_cam0_only
                if self.use_cam0_only and "cam1" in name:
                    continue
                candidates.append(d)

        return sorted(candidates)

    def _download_one(self, url: str, dest: Path, done: int, total: int, verbose: bool) -> bool:
        """Download a single zip with counter prefix."""
        try:
            if verbose:
                print(f"  [{done+1:2d}/{total}] Downloading {dest.name} …", end="\r", flush=True)
            urllib.request.urlretrieve(url, str(dest))
            return True
        except urllib.error.URLError as e:
            if verbose:
                print(f"  [{done+1:2d}/{total}] FAILED {dest.name}: {e}  ")
            return False
        except Exception as e:
            if verbose:
                print(f"  [{done+1:2d}/{total}] ERROR  {dest.name}: {e}  ")
            return False
