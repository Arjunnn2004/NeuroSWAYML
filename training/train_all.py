"""
NeuroSWAYML - Full Multi-Domain Training Pipeline
==================================================
Trains models for ALL four analysis domains:
  1. Neurodegenerative  (Parkinson's / ALS / HD)   → saved_models/neuro/
  2. Elderly Gait       (URFD video dataset)              → saved_models/elderly/
  3. Intoxication/Ataxia (HBEDB / PhysioNet)        → saved_models/intoxication/
  4. Congenital Disorder (GaitRec / figshare)        → saved_models/congenital/

Usage:
    python training/train_all.py                    # train all domains
    python training/train_all.py --domain neuro     # train one domain only
    python training/train_all.py --domain elderly
    python training/train_all.py --domain intox
    python training/train_all.py --domain congen
    python training/train_all.py --skip-missing     # skip domains with no data

Outputs per domain (saved to saved_models/<domain>/):
    gait_classifier.pkl
    lstm_model.pt
    autoencoder.pkl
    fall_classifier.pkl
    training_report.json
"""

import sys
import os
import json
import time
import argparse
import subprocess
import numpy as np

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset_loader import DatasetLoader
from models.gait_classifier import GaitClassifier
from models.lstm_model import LSTMModel
from models.autoencoder import Autoencoder
from models.ensemble import EnsembleModel
from models.domain_classifier import DomainModel
from data.feature_extractor import FeatureExtractor

from sklearn.model_selection import train_test_split


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config_ml.json")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def split(X, y, cfg):
    seed = cfg["dataset"]["random_seed"]
    test  = cfg["dataset"]["test_split"]
    val   = cfg["dataset"]["val_split"]
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test,
                                                    stratify=y, random_state=seed)
    val_of_tv = val / (1.0 - test)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=val_of_tv,
                                                       stratify=y_tv, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ───────────────────────────────────────────────────────────────────────────

def train_neuro(cfg: dict, skip_missing: bool = False) -> dict:
    """Train neurodegenerative domain from PhysioNet gaitpdb / gaitndd."""
    print("\n" + "═" * 60)
    print("  [DOMAIN 1/4] Neurodegenerative  (PD / ALS / HD)")
    print("═" * 60)

    domain = DomainModel("neurodegenerative", cfg, MODELS_DIR)

    loader = DatasetLoader(cfg)
    try:
        X_gait, y_gait = loader.load_gait_dataset()
    except Exception as e:
        if skip_missing:
            print(f"  [SKIP] Could not load neuro data: {e}")
            return {}
        raise

    print(f"  Gait dataset: {X_gait.shape}  classes: {np.bincount(y_gait).tolist()}")

    X_fall, y_fall = loader.load_fall_dataset()
    print(f"  Fall dataset: {X_fall.shape}  classes: {np.bincount(y_fall).tolist()}")

    seq_len = cfg["inference"]["sequence_length"]
    X_seq, y_seq = loader.load_sequence_dataset(seq_len=seq_len)
    print(f"  Sequence dataset: {X_seq.shape}")

    Xtr, Xv, Xte, ytr, yv, yte = split(X_gait, y_gait, cfg)
    Xst, Xsv, _, yst, ysv, _   = split(X_seq, y_seq, cfg)

    gc_metrics   = domain.gait_clf.train(Xtr, ytr, Xv, yv)
    lstm_metrics = domain.lstm_model.train(Xst, yst, Xsv, ysv)

    domain.ae.calibrate(Xtr[ytr == 0], verbose=True)
    ae_sc_n = np.array([domain.ae.anomaly_score(x) for x in Xte[yte == 0]])
    ae_sc_a = np.array([domain.ae.anomaly_score(x) for x in Xte[yte != 0]])
    ae_sep  = float(np.mean(ae_sc_a) - np.mean(ae_sc_n)) if len(ae_sc_a) else 0.0
    ae_metrics = {
        "threshold": domain.ae.threshold,
        "normal_score_mean": float(np.mean(ae_sc_n)),
        "abnormal_score_mean": float(np.mean(ae_sc_a)),
        "separation": ae_sep,
    }

    Xff, Xfv, _, yff, yfv, _ = split(X_fall, y_fall, cfg)
    fall_metrics = domain.ensemble.train_fall_classifier(Xff, yff, Xfv, yfv)

    domain.save_all(gc_metrics, lstm_metrics, ae_metrics, fall_metrics)

    # ── Backward-compat: also copy to flat saved_models/ ─────────────
    import shutil, pathlib
    flat_dir = pathlib.Path(MODELS_DIR)
    sub_dir  = flat_dir / "neuro"
    for fname in ["gait_classifier.pkl", "lstm_model.pt",
                  "autoencoder.pkl", "fall_classifier.pkl"]:
        src = sub_dir / fname
        if src.exists():
            shutil.copy2(src, flat_dir / fname)
    print(f"  [Neuro] Also copied models to {flat_dir} (backward compat.)")

    print(f"  {domain.readiness}")
    return {"gait": gc_metrics, "lstm": lstm_metrics, "ae": ae_metrics, "fall": fall_metrics}


def _run_domain_script(script_name: str, skip_missing: bool) -> int:
    """Launch a per-domain training script as a sub-process."""
    script = os.path.join(os.path.dirname(__file__), script_name)
    extra  = ["--dry-run"] if skip_missing else []
    ret = subprocess.run([sys.executable, script] + extra, cwd=os.path.dirname(__file__) + "/..")
    return ret.returncode


# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NeuroSWAYML — Multi-Domain Training")
    parser.add_argument(
        "--domain",
        choices=["neuro", "elderly", "intox", "congen", "all"],
        default="all",
        help="Which domain to train (default: all). "
             "'all' trains neuro + intox + congen. "
             "Use --domain elderly explicitly to train the URFD dataset (~240 MB).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip domains whose dataset directories are absent instead of aborting",
    )
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("  NeuroSWAYML — Multi-Domain Training Pipeline")
    active = args.domain if args.domain != "all" else "neuro + intox + congen"
    print(f"  Domain(s): {active}")
    print("=" * 60)

    cfg = load_config()
    os.makedirs(MODELS_DIR, exist_ok=True)
    report: dict = {"domains": {}, "config": cfg["training"]}

    do_neuro   = args.domain in ("neuro",  "all")
    do_elderly = args.domain == "elderly"   # excluded from 'all' by default; ~240 MB URFD
    do_intox   = args.domain in ("intox",   "all")
    do_congen  = args.domain in ("congen",  "all")

    # ── Neuro (handled inline — uses DatasetLoader) ───────────────────
    if do_neuro:
        try:
            r = train_neuro(cfg, skip_missing=args.skip_missing)
            report["domains"]["neurodegenerative"] = r
        except Exception as e:
            print(f"\n[ERROR] Neuro training failed: {e}")
            if not args.skip_missing:
                raise

    # ── Elderly (delegated to train_elderly.py) ───────────────────────
    if do_elderly:
        print("\n" + "═" * 60)
        print("  [DOMAIN 2/4] Elderly Gait / Fall-Risk  (LTMM)")
        print("═" * 60)
        rc = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "train_elderly.py")],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        ).returncode
        if rc != 0 and not args.skip_missing:
            print("[ERROR] Elderly training failed.")
            sys.exit(rc)
        report["domains"]["elderly"] = "trained" if rc == 0 else "failed/skipped"

    # ── Intoxication (delegated to train_intoxication.py) ────────────
    if do_intox:
        print("\n" + "═" * 60)
        print("  [DOMAIN 3/4] Intoxication / Ataxia  (HBEDB)")
        print("═" * 60)
        rc = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "train_intoxication.py")],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        ).returncode
        if rc != 0 and not args.skip_missing:
            print("[ERROR] Intoxication training failed.")
            sys.exit(rc)
        report["domains"]["intoxication"] = "trained" if rc == 0 else "failed/skipped"

    # ── Congenital (delegated to train_congenital.py) ─────────────────
    if do_congen:
        print("\n" + "═" * 60)
        print("  [DOMAIN 4/4] Congenital / Birth Disorder  (GaitRec)")
        print("═" * 60)
        rc = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "train_congenital.py")],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        ).returncode
        if rc != 0 and not args.skip_missing:
            print("[ERROR] Congenital training failed.")
            sys.exit(rc)
        report["domains"]["congenital"] = "trained" if rc == 0 else "failed/skipped"

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    report["elapsed_seconds"] = round(elapsed, 1)
    print(f"\n{'='*60}")
    print(f"  All requested domains trained in {elapsed:.1f}s")
    print(f"  Models saved under: {os.path.abspath(MODELS_DIR)}")
    for dom, status in report["domains"].items():
        print(f"    {dom:20s}: {status}")
    print(f"{'='*60}\n")

    report_path = os.path.join(MODELS_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Master training report → {report_path}")


if __name__ == "__main__":
    main()
