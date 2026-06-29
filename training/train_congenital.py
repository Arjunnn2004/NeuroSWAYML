"""
NeuroSWAYML — Congenital / Birth Disorder Gait Training
========================================================
Dataset: GaitRec (Horst et al., 2021)
Source:  figshare  DOI: 10.6084/m9.figshare.13598962.v1   (CC-BY 4.0)
  2,084 subjects (normal controls + 7 pathology groups)
  17-column bilateral GRF data @ 1000 Hz
  Groups mapped:
    CTL              → 0 (NORMAL)
    BACK, ANKLE      → 1 (MILD_DISORDER)
    HIP, KNEE, NEURO, CP, DS, SB → 2 (SEVERE_DISORDER)

Manual download required (≈2.3 GB):
    See instructions: python data/downloader.py --domain congen

Usage:
    python training/train_congenital.py
    python training/train_congenital.py --data-dir data/gaitrec
    python training/train_congenital.py --dry-run
"""

import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.loaders.congenital_loader import CongenitalLoader
from models.domain_classifier       import DomainModel

from sklearn.model_selection import train_test_split


CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "..", "config_ml.json")
MODELS_ROOT  = os.path.join(os.path.dirname(__file__), "..", "saved_models")
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "gaitrec")


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


def main():
    parser = argparse.ArgumentParser(
        description="Train congenital/birth-disorder gait models (GaitRec dataset)"
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA,
                        help="Path to local GaitRec directory")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Load dataset and print info without training")
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 60)
    print("  NeuroSWAYML — Congenital / Birth Disorder Gait Training")
    print("  Dataset: GaitRec (figshare)")
    print("=" * 60)

    cfg = load_config()

    # ── 1. Load dataset ───────────────────────────────────────────────
    print(f"\n[1/5] Loading GaitRec dataset from {args.data_dir} …")
    loader = CongenitalLoader(args.data_dir)
    X, y   = loader.load()

    if len(X) == 0:
        print("\n[ERROR] No data loaded.")
        print("  → Download GaitRec v1 manually from figshare:")
        print("    python data/downloader.py --domain congen")
        print("    Then re-run this script.")
        sys.exit(1)

    print(f"  Loaded: {X.shape}  classes: {np.bincount(y).tolist()}")
    print(f"  Class names: {loader.CLASS_NAMES}")

    if args.dry_run:
        print("\n[DRY RUN] Dataset loaded successfully. Exiting without training.")
        return

    seq_len = cfg["inference"]["sequence_length"]
    print(f"\n  Generating sequence dataset (seq_len={seq_len}) …")
    X_seq, y_seq = loader.load_sequence_dataset(seq_len=seq_len)
    print(f"  Sequence dataset: {X_seq.shape}")

    # ── 2. Split ──────────────────────────────────────────────────────
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, cfg)
    Xs_tr, Xs_v, Xs_te, ys_tr, ys_v, ys_te = split_data(X_seq, y_seq, cfg)

    # ── 3. Instantiate DomainModel ────────────────────────────────────
    print("\n[2/5] Building congenital domain model stack …")
    domain = DomainModel("congenital", cfg, MODELS_ROOT)

    # ── 4. Train GaitClassifier ───────────────────────────────────────
    print("\n[3/5] Training GaitClassifier (RF + XGBoost) …")
    gc_metrics = domain.gait_clf.train(X_tr, y_tr, X_v, y_v)

    # ── 5. Train LSTM ─────────────────────────────────────────────────
    print("\n[4/5] Training LSTM Sequence Model …")
    lstm_metrics = domain.lstm_model.train(Xs_tr, ys_tr, Xs_v, ys_v)

    # ── 6. Calibrate Autoencoder on class-0 (normal gait) ────────────
    print("\n[5/5] Calibrating Autoencoder on NORMAL control samples …")
    normal = X_tr[y_tr == 0]
    print(f"  {len(normal)} control samples for AE calibration")
    domain.ae.calibrate(normal, verbose=True)

    ae_scores_n = np.array([domain.ae.anomaly_score(x) for x in X_te[y_te == 0]])
    ae_scores_a = np.array([domain.ae.anomaly_score(x) for x in X_te[y_te != 0]])
    ae_sep = float(np.mean(ae_scores_a) - np.mean(ae_scores_n)) if len(ae_scores_a) else 0.0
    ae_metrics = {
        "threshold":           domain.ae.threshold,
        "normal_score_mean":   float(np.mean(ae_scores_n)),
        "abnormal_score_mean": float(np.mean(ae_scores_a)),
        "separation":          ae_sep,
    }
    print(f"  AE separation: {ae_sep:.4f}")

    # ── 7. Train Fall Classifier ──────────────────────────────────────
    # Severe disorder (class-2) has highest fall risk in congenital gait
    y_fall_tr = (y_tr == 2).astype(int)
    y_fall_v  = (y_v  == 2).astype(int)
    fall_metrics = domain.ensemble.train_fall_classifier(
        X_tr, y_fall_tr, X_v, y_fall_v
    )

    # ── 8. Save everything ────────────────────────────────────────────
    print(f"\n  Saving congenital models → {domain.models_dir}")
    domain.save_all(gc_metrics, lstm_metrics, ae_metrics, fall_metrics)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Congenital domain training complete in {elapsed:.1f}s")
    print(f"  GaitClassifier: {gc_metrics}")
    print(f"  LSTM:           {lstm_metrics}")
    print(f"  {domain.readiness}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
