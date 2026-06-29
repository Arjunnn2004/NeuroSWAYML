"""
NeuroSWAYML — Elderly Gait / Fall-Risk Training  (URFD Video Dataset)
======================================================================
Dataset: University of Rzeszów Fall Detection (URFD)
URL    : http://fenix.ur.edu.pl/~mkepski/ds/uf.html
  30 fall sequences + 40 ADL sequences, multiple camera views.
  Real-person videos → MediaPipe Pose → FeatureExtractor (30-D) →
  feature vectors IDENTICAL to live inference.

Labels:
  adl-* sequences (normal activity)       → 0  NORMAL_GAIT
  fall-* first 60% of frames              → 1  MILD_FALL_RISK  (pre-fall)
  fall-* last  40% of frames              → 2  HIGH_FALL_RISK  (active fall)

Download (~240 MB, one-time):
    python data/downloader.py --domain elderly

Usage:
    python training/train_elderly.py
    python training/train_elderly.py --data-dir data/urfd
    python training/train_elderly.py --force-reprocess   # re-run MediaPipe
    python training/train_elderly.py --dry-run           # show dataset info
"""

import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.loaders.urfd_loader import URFDLoader
from models.domain_classifier import DomainModel
from models.gait_classifier   import GaitClassifier
from models.lstm_model        import LSTMModel
from models.autoencoder       import Autoencoder
from models.ensemble          import EnsembleModel

from sklearn.model_selection import train_test_split


CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "..", "config_ml.json")
MODELS_ROOT  = os.path.join(os.path.dirname(__file__), "..", "saved_models")
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "urfd")


# ─────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def split_data(X, y, cfg):
    seed   = cfg["dataset"]["random_seed"]
    t_size = cfg["dataset"]["test_split"]
    v_size = cfg["dataset"]["val_split"]

    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=t_size, stratify=y, random_state=seed
    )
    val_frac = v_size / (1.0 - t_size)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=seed
    )
    return X_tr, X_v, X_te, y_tr, y_v, y_te


# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train elderly gait / fall-risk models (URFD video dataset)"
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA,
                        help="Path to local URFD directory (default: data/urfd)")
    parser.add_argument("--force-reprocess", action="store_true",
                        help="Re-run MediaPipe even if features cache exists")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Cap number of sequences per class (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load dataset and print info without training")
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 60)
    print("  NeuroSWAYML — Elderly Gait / Fall-Risk Training")
    print("  Dataset: URFD (University of Rzeszów Fall Detection)")
    print("=" * 60)

    cfg = load_config()

    # ── 1. Load dataset ───────────────────────────────────────────────
    print(f"\n[1/5] Loading URFD dataset from {args.data_dir} …")

    loader = URFDLoader(
        data_dir=args.data_dir,
        force_reprocess=args.force_reprocess,
        max_sequences=args.max_sequences,
    )

    if not loader.is_ready():
        print("\n[NOTICE] URFD data not found. Attempting auto-download (~240 MB) …")
        ok = loader.download(verbose=True)
        if ok:
            loader.extract(verbose=True)
        else:
            print("\n[ERROR] Auto-download failed.")
            print("  → Manual download:")
            print("    1. Visit http://fenix.ur.edu.pl/~mkepski/ds/uf.html")
            print("    2. Download all fall-NN-cam0-rgb.zip and adl-NN-cam0-rgb.zip listed there")
            print(f"    3. Extract all zips to {args.data_dir}/")
            print("    4. Re-run this script.")
            sys.exit(1)

    X, y = loader.load()

    if len(X) == 0:
        print("\n[ERROR] No features extracted. Check URFD folder structure.")
        sys.exit(1)

    print(f"  Loaded: {X.shape}  classes: {np.bincount(y).tolist()}")
    print(f"  Class names: {loader.CLASS_NAMES}")

    if args.dry_run:
        print("\n[DRY RUN] Dataset loaded successfully. Exiting without training.")
        print(f"\n  Status: {loader.status()}")
        return

    seq_len = cfg["inference"]["sequence_length"]
    print(f"\n  Generating sequence dataset (seq_len={seq_len}) …")
    X_seq, y_seq = loader.load_sequence_dataset(seq_len=seq_len)
    print(f"  Sequence dataset: {X_seq.shape}")

    # ── 2. Split ──────────────────────────────────────────────────────
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, cfg)
    Xs_tr, Xs_v, Xs_te, ys_tr, ys_v, ys_te = split_data(X_seq, y_seq, cfg)

    # ── 3. Instantiate DomainModel ────────────────────────────────────
    print("\n[2/5] Building elderly domain model stack …")
    domain = DomainModel("elderly", cfg, MODELS_ROOT)

    # ── 4. Train GaitClassifier ───────────────────────────────────────
    print("\n[3/5] Training GaitClassifier (RF + XGBoost) …")
    gc_metrics = domain.gait_clf.train(X_tr, y_tr, X_v, y_v)

    # ── 5. Train LSTM ─────────────────────────────────────────────────
    print("\n[4/5] Training LSTM Sequence Model …")
    lstm_metrics = domain.lstm_model.train(Xs_tr, ys_tr, Xs_v, ys_v)

    # ── 6. Calibrate Autoencoder on class-0 (normal gait) ────────────
    print("\n[5/5] Calibrating Autoencoder on NORMAL_GAIT samples …")
    normal = X_tr[y_tr == 0]
    print(f"  {len(normal)} normal-class samples for AE calibration")
    domain.ae.calibrate(normal, verbose=True)

    ae_scores_n = np.array([domain.ae.anomaly_score(x) for x in X_te[y_te == 0]])
    ae_scores_a = np.array([domain.ae.anomaly_score(x) for x in X_te[y_te != 0]])
    ae_sep = float(np.mean(ae_scores_a) - np.mean(ae_scores_n)) if len(ae_scores_a) else 0.0
    ae_metrics = {
        "threshold":          domain.ae.threshold,
        "normal_score_mean":  float(np.mean(ae_scores_n)),
        "abnormal_score_mean": float(np.mean(ae_scores_a)),
        "separation":         ae_sep,
    }
    print(f"  AE separation: {ae_sep:.4f}")

    # ── 7. Train Fall Classifier ──────────────────────────────────────
    # Reuse binary fall labels: class-0 = no fall risk, class 1+2 = fall risk
    y_fall_tr = (y_tr > 0).astype(int)
    y_fall_v  = (y_v  > 0).astype(int)
    fall_metrics = domain.ensemble.train_fall_classifier(
        X_tr, y_fall_tr, X_v, y_fall_v
    )

    # ── 8. Save everything ────────────────────────────────────────────
    print(f"\n  Saving elderly models → {domain.models_dir}")
    domain.save_all(gc_metrics, lstm_metrics, ae_metrics, fall_metrics)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Elderly domain training complete in {elapsed:.1f}s")
    print(f"  GaitClassifier: {gc_metrics}")
    print(f"  LSTM:           {lstm_metrics}")
    print(f"  {domain.readiness}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
