"""
NeuroSWAYML — Domain Classifier
================================
A thin adapter that gives every analysis domain its own named
GaitClassifier + LSTMModel + Autoencoder + EnsembleModel stack,
stored in an isolated subdirectory of saved_models/.

All domain classifiers share the exact same architecture
(RF + XGB + LSTM + AE ensemble) — only class names and
model weights differ.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from models.gait_classifier import GaitClassifier
from models.lstm_model       import LSTMModel
from models.autoencoder      import Autoencoder
from models.ensemble         import EnsembleModel, CLASS_COLORS


# ── Domain metadata ────────────────────────────────────────────────────────
DOMAIN_META: Dict[str, Dict] = {
    "neurodegenerative": {
        "display_name": "Neuro (PD / ALS / HD)",
        "class_names":  ["NORMAL", "WARNING", "HIGH_RISK"],
        "class_colors": {0: (0, 220, 0), 1: (0, 165, 255), 2: (0, 0, 255)},
        "models_subdir": "neuro",
        "key_hint": "1",
    },
    "elderly": {
        "display_name": "Elderly Gait & Fall Risk",
        "class_names":  ["NORMAL_GAIT", "MILD_FALL_RISK", "HIGH_FALL_RISK"],
        "class_colors": {0: (0, 220, 0), 1: (0, 165, 255), 2: (0, 0, 180)},
        "models_subdir": "elderly",
        "key_hint": "2",
    },
    "intoxication": {
        "display_name": "Intoxication / Ataxia",
        "class_names":  ["SOBER", "MILD_IMPAIRMENT", "INTOXICATED"],
        "class_colors": {0: (0, 200, 0), 1: (0, 140, 255), 2: (0, 60, 255)},
        "models_subdir": "intoxication",
        "key_hint": "3",
    },
    "congenital": {
        "display_name": "Congenital / Birth Disorder",
        "class_names":  ["NORMAL", "MILD_DISORDER", "SEVERE_DISORDER"],
        "class_colors": {0: (0, 220, 0), 1: (0, 165, 255), 2: (180, 0, 180)},
        "models_subdir": "congenital",
        "key_hint": "4",
    },
}

DOMAIN_KEYS   = list(DOMAIN_META.keys())
DOMAIN_ALIASES = {
    "neuro": "neurodegenerative",
    "neurodegenerate": "neurodegenerative",
    "old": "elderly",
    "elder": "elderly",
    "intox": "intoxication",
    "drunk": "intoxication",
    "alcohol": "intoxication",
    "birth": "congenital",
    "cp": "congenital",
    "congen": "congenital",
}


def resolve_domain(name: str) -> str:
    """Return canonical domain key, or raises ValueError."""
    key = name.lower().strip()
    if key in DOMAIN_META:
        return key
    if key in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[key]
    raise ValueError(
        f"Unknown domain '{name}'. Valid names: {list(DOMAIN_META.keys())}"
    )


# ── DomainModel class ──────────────────────────────────────────────────────

class DomainModel:
    """
    Encapsulates the full model stack for one analysis domain.

    Usage
    -----
    dm = DomainModel("elderly", config, models_root)
    dm.load()                        # load .pkl / .pt from disk
    risk = dm.ensemble.predict_risk(feat, seq_buf)
    fall = dm.ensemble.predict_fall(fall_feat)
    """

    def __init__(
        self,
        domain_name: str,
        config: dict,
        models_root: str,
        n_features: int = 30,
        seq_len: int = 60,
    ):
        self.domain     = resolve_domain(domain_name)
        self.meta       = DOMAIN_META[self.domain]
        self.n_features = n_features
        self.seq_len    = seq_len

        self.models_dir = Path(models_root) / self.meta["models_subdir"]
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # ── Build model objects (same architecture for all domains) ────
        self.gait_clf   = GaitClassifier(config)
        self.lstm_model = LSTMModel(config, n_features=n_features, seq_len=seq_len)
        self.ae         = Autoencoder(config, n_features=n_features)
        self.ensemble   = EnsembleModel(self.gait_clf, self.lstm_model, self.ae, config)

        self.is_loaded       = False
        self.class_names     = self.meta["class_names"]
        self.class_colors    = self.meta["class_colors"]
        self.display_name    = self.meta["display_name"]

    # ------------------------------------------------------------------
    def load(self, verbose: bool = True) -> bool:
        """Load saved model files. Returns True if at least gait_clf loaded."""
        d    = self.models_dir
        ok   = False
        msgs = []

        try:
            self.gait_clf.load(str(d / "gait_classifier.pkl"))
            ok = True
        except Exception as e:
            msgs.append(f"    GaitCLF: {e}")

        try:
            self.lstm_model.load(str(d / "lstm_model.pt"))
        except Exception as e:
            msgs.append(f"    LSTM: {e}")

        try:
            self.ae.load(str(d / "autoencoder.pkl"))
        except Exception as e:
            msgs.append(f"    AE: {e}")

        try:
            self.ensemble.load_fall_clf(str(d / "fall_classifier.pkl"))
        except Exception as e:
            msgs.append(f"    FallCLF: {e}")

        if verbose:
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] [{self.meta['key_hint']}] {self.display_name}")
            for m in msgs:
                print(m)

        self.is_loaded = ok
        return ok

    # ------------------------------------------------------------------
    def save_all(self, gait_metrics: dict, lstm_metrics: dict,
                 ae_metrics: dict, fall_metrics: dict):
        """Save all models + training report."""
        import json, time
        d = self.models_dir
        d.mkdir(parents=True, exist_ok=True)

        self.gait_clf.save(str(d / "gait_classifier.pkl"))
        self.ae.save(str(d / "autoencoder.pkl"))
        self.ensemble.save_fall_clf(str(d / "fall_classifier.pkl"))
        try:
            self.lstm_model.save(str(d / "lstm_model.pt"))
        except Exception as e:
            print(f"  [Warning] LSTM save failed: {e}")

        report = {
            "domain":       self.domain,
            "class_names":  self.class_names,
            "gait":         gait_metrics,
            "lstm":         lstm_metrics,
            "autoencoder":  ae_metrics,
            "fall":         fall_metrics,
            "saved_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        report_path = d / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  [DomainModel] Training report → {report_path}")

    # ------------------------------------------------------------------
    def predict_risk(
        self,
        feat_vec:  np.ndarray,
        seq_buf:   Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Call the ensemble and remap class labels/colors to this domain.
        Returns same dict structure as EnsembleModel.predict_risk().
        """
        result = self.ensemble.predict_risk(feat_vec, seq_buf)

        # Remap generic labels to domain-specific ones
        cls = result["risk_class"]
        result["class_label"] = self.class_names[cls]
        result["color"]       = self.class_colors[cls]
        result["domain"]      = self.domain
        result["domain_display"] = self.display_name
        return result

    def predict_fall(self, fall_feat: np.ndarray) -> float:
        return self.ensemble.predict_fall(fall_feat)

    # ------------------------------------------------------------------
    def recalibrate(self):
        """Reset AE personal calibration (call from MLAnalyzer.recalibrate)."""
        self.ae.is_calibrated = False

    @property
    def readiness(self) -> str:
        return self.ensemble.readiness_summary()
