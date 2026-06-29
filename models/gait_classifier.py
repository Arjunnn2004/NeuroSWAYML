"""
NeuroSWAYML - Gait Risk Classifier
Random Forest + XGBoost soft-vote ensemble for 3-class gait risk prediction.
Classes: 0=NORMAL, 1=WARNING, 2=HIGH_RISK
"""

import os
import numpy as np
import joblib
from typing import Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class GaitClassifier:
    """
    Soft-vote ensemble:
      • RandomForest   (weight configurable)
      • XGBoost        (weight configurable, falls back to GradientBoosting)

    All predictions are probability vectors [P(normal), P(warning), P(high_risk)].
    """

    CLASS_NAMES = ["NORMAL", "WARNING", "HIGH_RISK"]

    def __init__(self, config: dict):
        tr = config["training"]
        self.rf_weight  = config["ensemble"]["rf_weight"]
        self.xgb_weight = config["ensemble"]["xgb_weight"]

        # ── Random Forest pipeline ────────────────────────────────────
        rf_base = RandomForestClassifier(
            n_estimators=tr["rf_n_estimators"],
            max_depth=tr["rf_max_depth"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.rf_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     CalibratedClassifierCV(rf_base, method="isotonic", cv=3)),
        ])

        # ── XGBoost / GradientBoosting pipeline ──────────────────────
        if XGB_AVAILABLE:
            xgb_base = XGBClassifier(
                n_estimators=tr["xgb_n_estimators"],
                max_depth=tr["xgb_max_depth"],
                learning_rate=tr["xgb_learning_rate"],
                use_label_encoder=False,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
            )
        else:
            print("  [GaitClassifier] XGBoost not found — using GradientBoosting")
            xgb_base = GradientBoostingClassifier(
                n_estimators=tr["xgb_n_estimators"],
                max_depth=tr["xgb_max_depth"],
                learning_rate=tr["xgb_learning_rate"],
                random_state=42,
            )

        self.xgb_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb",   CalibratedClassifierCV(xgb_base, method="isotonic", cv=3)),
        ])

        self.is_trained = False

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        """Train both classifiers and return metrics dict."""
        print("  [GaitClassifier] Training RandomForest…")
        self.rf_pipeline.fit(X_train, y_train)

        print("  [GaitClassifier] Training XGBoost/GBM…")
        self.xgb_pipeline.fit(X_train, y_train)

        self.is_trained = True

        metrics = {}
        if X_val is not None and y_val is not None:
            rf_acc  = self.rf_pipeline.score(X_val, y_val)
            xgb_acc = self.xgb_pipeline.score(X_val, y_val)
            ens_probs = self.predict_proba(X_val)
            ens_preds = np.argmax(ens_probs, axis=1)
            ens_acc   = float(np.mean(ens_preds == y_val))

            print(f"\n  [GaitClassifier] Val accuracy → RF: {rf_acc:.4f}  "
                  f"XGB: {xgb_acc:.4f}  Ensemble: {ens_acc:.4f}")
            print(classification_report(y_val, ens_preds,
                                        labels=list(range(len(self.CLASS_NAMES))),
                                        target_names=self.CLASS_NAMES, zero_division=0))

            metrics = {
                "rf_val_acc":  rf_acc,
                "xgb_val_acc": xgb_acc,
                "ens_val_acc": ens_acc,
                "confusion_matrix": confusion_matrix(y_val, ens_preds).tolist(),
            }
        return metrics

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability matrix (N, 3).
        Weighted soft vote between RF and XGB.
        """
        if not self.is_trained:
            raise RuntimeError("GaitClassifier not trained yet")
        rf_p  = self.rf_pipeline.predict_proba(X)
        xgb_p = self.xgb_pipeline.predict_proba(X)
        total = self.rf_weight + self.xgb_weight
        return (self.rf_weight * rf_p + self.xgb_weight * xgb_p) / total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns class indices (N,)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_single(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Single-sample inference.
        Returns (class_idx, prob_vector).
        """
        proba = self.predict_proba(x.reshape(1, -1))[0]
        return int(np.argmax(proba)), proba

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "rf_pipeline":  self.rf_pipeline,
            "xgb_pipeline": self.xgb_pipeline,
            "rf_weight":    self.rf_weight,
            "xgb_weight":   self.xgb_weight,
            "is_trained":   self.is_trained,
        }, path)
        print(f"  [GaitClassifier] Saved → {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        data = joblib.load(path)
        self.rf_pipeline  = data["rf_pipeline"]
        self.xgb_pipeline = data["xgb_pipeline"]
        self.rf_weight    = data["rf_weight"]
        self.xgb_weight   = data["xgb_weight"]
        self.is_trained   = data["is_trained"]
        print(f"  [GaitClassifier] Loaded <- {path}")
