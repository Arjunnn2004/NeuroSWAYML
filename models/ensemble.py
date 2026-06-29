"""
NeuroSWAYML - Ensemble Model
Combines GaitClassifier (RF+XGB), LSTMModel, and Autoencoder into a single
weighted decision. Also includes a standalone binary fall classifier.

Output:
  risk_score  ∈ [0, 1]   — continuous risk level for the UI
  risk_class  ∈ {0,1,2}  — NORMAL / WARNING / HIGH_RISK
  fall_prob   ∈ [0, 1]   — probability of an active fall event
  class_label str        — human-readable label
"""

import os
import numpy as np
import joblib
from typing import Optional, Tuple, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


CLASS_NAMES = ["NORMAL", "WARNING", "HIGH_RISK"]
CLASS_COLORS = {0: (0, 220, 0), 1: (0, 165, 255), 2: (0, 0, 255)}


class EnsembleModel:
    """
    Weighted soft-vote ensemble of all NeuroSWAYML models.

    Call flow (real-time):
      proba3 = ensemble.predict_risk(feat_vec, seq_buf)
      fall_p = ensemble.predict_fall(fall_feat_vec)
    """

    def __init__(
        self,
        gait_clf,       # GaitClassifier instance
        lstm_model,     # LSTMModel instance
        autoencoder,    # Autoencoder instance
        config: dict,
    ):
        self.gait_clf   = gait_clf
        self.lstm       = lstm_model
        self.ae         = autoencoder
        self.cfg        = config

        w = config["ensemble"]
        self.rf_w    = w["rf_weight"]
        self.xgb_w   = w["xgb_weight"]
        self.lstm_w  = w["lstm_weight"]
        self.ae_w    = w["ae_weight"]

        self.warn_thresh  = w["warning_threshold"]
        self.crit_thresh  = w["critical_threshold"]

        # Separate lightweight fall classifier (trained in train_all.py)
        self._fall_clf    = None
        self._fall_scaler = None
        self.fall_trained = False

    # ------------------------------------------------------------------
    # RISK PREDICTION
    # ------------------------------------------------------------------

    def predict_risk(
        self,
        feat_vec: np.ndarray,        # (n_features,)
        seq_buf: Optional[np.ndarray] = None,  # (seq_len, n_features) or None
    ) -> Dict[str, Any]:
        """
        Returns dict:
          risk_score, risk_class, class_label, color,
          proba_normal, proba_warning, proba_high_risk,
          model_votes (per-model contributions)
        """
        proba = np.zeros(3, dtype=np.float32)
        votes: Dict[str, np.ndarray] = {}
        total_weight = 0.0

        # ── GaitClassifier (RF + XGB internal ensemble) ───────────────
        if self.gait_clf.is_trained:
            gait_p = self.gait_clf.predict_proba(feat_vec.reshape(1, -1))[0]
            w = self.rf_w + self.xgb_w   # GaitClassifier already blends RF+XGB
            proba       += w * gait_p
            total_weight += w
            votes["gait_clf"] = gait_p

        # ── LSTM ─────────────────────────────────────────────────────
        if self.lstm.is_trained and seq_buf is not None and len(seq_buf) == self.lstm.seq_len:
            lstm_p = self.lstm.predict_proba(seq_buf[np.newaxis])[0]
            proba       += self.lstm_w * lstm_p
            total_weight += self.lstm_w
            votes["lstm"] = lstm_p

        # ── Autoencoder (personalised anomaly) ───────────────────────
        if self.ae.is_calibrated:
            ae_p = self.ae.predict_proba_from_anomaly(feat_vec)
            proba       += self.ae_w * ae_p
            total_weight += self.ae_w
            votes["autoencoder"] = ae_p

        # Normalise
        if total_weight > 0:
            proba /= total_weight
        else:
            proba = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Continuous risk score (0-1)
        risk_score = float(proba[1] * 0.5 + proba[2] * 1.0)
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # Class
        risk_class = int(np.argmax(proba))
        # Override via continuous score thresholds (more granular)
        if risk_score >= self.crit_thresh:
            risk_class = 2
        elif risk_score >= self.warn_thresh:
            risk_class = max(risk_class, 1)

        return {
            "risk_score":      risk_score,
            "risk_class":      risk_class,
            "class_label":     CLASS_NAMES[risk_class],
            "color":           CLASS_COLORS[risk_class],
            "proba_normal":    float(proba[0]),
            "proba_warning":   float(proba[1]),
            "proba_high_risk": float(proba[2]),
            "model_votes":     {k: v.tolist() for k, v in votes.items()},
        }

    # ------------------------------------------------------------------
    # FALL PREDICTION (binary)
    # ------------------------------------------------------------------

    def train_fall_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        """Train the lightweight binary fall classifier."""
        print("  [Ensemble] Training fall classifier…")

        if XGB_AVAILABLE:
            clf = XGBClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, tree_method="hist",
                use_label_encoder=False, eval_metric="logloss",
                random_state=42,
            )
        else:
            clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                          class_weight="balanced", random_state=42)

        self._fall_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    clf),
        ])
        self._fall_clf.fit(X_train, y_train)
        self.fall_trained = True

        metrics = {}
        if X_val is not None and y_val is not None:
            acc = self._fall_clf.score(X_val, y_val)
            print(f"  [Ensemble] Fall classifier val accuracy: {acc:.4f}")
            metrics["fall_val_acc"] = acc
        return metrics

    def predict_fall(self, fall_feat_vec: np.ndarray) -> float:
        """Returns probability of fall (0-1)."""
        if not self.fall_trained or self._fall_clf is None:
            return 0.0
        proba = self._fall_clf.predict_proba(fall_feat_vec.reshape(1, -1))[0]
        return float(proba[1] if len(proba) > 1 else proba[0])

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self.gait_clf.is_trained or self.fall_trained

    def readiness_summary(self) -> str:
        parts = []
        parts.append(f"GaitCLF={'OK' if self.gait_clf.is_trained else 'NOT TRAINED'}")
        parts.append(f"LSTM={'OK' if self.lstm.is_trained else 'NOT TRAINED'}")
        parts.append(f"AE={'calibrated' if self.ae.is_calibrated else 'not calibrated'}")
        parts.append(f"FallCLF={'OK' if self.fall_trained else 'NOT TRAINED'}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------

    def save_fall_clf(self, path: str):
        if not self.fall_trained:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"fall_clf": self._fall_clf}, path)
        print(f"  [Ensemble] Fall classifier saved → {path}")

    def load_fall_clf(self, path: str):
        if not os.path.exists(path):
            print(f"  [Ensemble] No fall classifier at {path} — skipping")
            return
        data = joblib.load(path)
        self._fall_clf = data["fall_clf"]
        self.fall_trained = True
        print(f"  [Ensemble] Fall classifier loaded <- {path}")
