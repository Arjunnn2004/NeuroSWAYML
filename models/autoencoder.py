"""
NeuroSWAYML - Personalization Autoencoder
Learns THIS PATIENT's normal gait pattern during a calibration phase.
Anomaly score = reconstruction error → flags deviations from personal baseline.

Architecture (PyTorch MLP):
  Encoder: n_features → 32 → 16 → 8
  Decoder: 8 → 16 → 32 → n_features

Falls back to sklearn MLPRegressor if PyTorch is unavailable.
"""

import os
import numpy as np
import joblib
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AE_AVAILABLE = True
except ImportError:
    SKLEARN_AE_AVAILABLE = False


# ───────────────────────────────────────────────────────────────────────────

class _AENet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_size: int, hidden_sizes: list):
        super().__init__()
        # Encoder
        enc_layers = []
        in_sz = input_size
        for h in hidden_sizes:
            enc_layers += [nn.Linear(in_sz, h), nn.ReLU(), nn.BatchNorm1d(h)]
            in_sz = h
        # Decoder (reversed)
        dec_layers = []
        dec_dims = hidden_sizes[:-1][::-1] + [input_size]
        for h in dec_dims:
            dec_layers += [nn.Linear(in_sz, h), nn.ReLU()]
            in_sz = h
        dec_layers[-1] = nn.Identity()   # remove last ReLU on output layer

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        return self.encoder(x)


# ───────────────────────────────────────────────────────────────────────────

class Autoencoder:
    """
    Personalization autoencoder.

    Usage:
      1. Collect 90 frames of the patient's normal gait → call calibrate(X_normal)
      2. Each frame: score = anomaly_score(feature_vec)
         score in [0, 1] — high = abnormal for THIS patient
    """

    def __init__(self, config: dict, n_features: int = 30):
        self.n_features = n_features
        tr = config["training"]
        self.hidden_sizes   = tr["ae_hidden_sizes"]
        self.epochs         = tr["ae_epochs"]
        self.lr             = tr["ae_lr"]
        self.percentile     = tr["ae_anomaly_percentile"]
        self.ae_weight      = config["ensemble"]["ae_weight"]
        self.device_str     = config["inference"].get("device", "cpu")

        self.threshold      = None   # set after calibration
        self.is_calibrated  = False
        self._scaler        = None

        if TORCH_AVAILABLE:
            self._device = torch.device(
                self.device_str if torch.cuda.is_available() else "cpu"
            )
            self.net = _AENet(n_features, self.hidden_sizes).to(self._device)
        else:
            self._device = None
            self.net     = None
            print("  [Autoencoder] PyTorch unavailable — using sklearn fallback")

    # ------------------------------------------------------------------
    def calibrate(
        self,
        X_normal: np.ndarray,
        verbose: bool = True,
    ) -> float:
        """
        Train the autoencoder on normal-gait frames for THIS patient.
        Returns the anomaly threshold.
        """
        if verbose:
            print(f"  [Autoencoder] Calibrating on {len(X_normal)} normal frames…")

        # Normalise features
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler().fit(X_normal)
        X_norm = self._scaler.transform(X_normal).astype(np.float32)

        if TORCH_AVAILABLE:
            self.threshold = self._train_torch(X_norm, verbose)
        elif SKLEARN_AE_AVAILABLE:
            self.threshold = self._train_sklearn(X_norm, verbose)
        else:
            self.threshold = 0.05
            self.is_calibrated = False
            return self.threshold

        self.is_calibrated = True
        if verbose:
            print(f"  [Autoencoder] Anomaly threshold set to {self.threshold:.5f}")
        return self.threshold

    # ------------------------------------------------------------------
    def anomaly_score(self, x: np.ndarray) -> float:
        """
        Returns float in [0, 1].
        0 = perfectly normal for this patient.
        1 = maximally abnormal.
        """
        if not self.is_calibrated:
            return 0.0

        x_s = self._scaler.transform(x.reshape(1, -1)).astype(np.float32)
        err = self._reconstruction_error(x_s)
        # Sigmoid-normalise around threshold
        ratio = err / (self.threshold + 1e-9)
        score = float(1.0 / (1.0 + np.exp(-5.0 * (ratio - 1.0))))   # S-curve
        return float(np.clip(score, 0.0, 1.0))

    def predict_proba_from_anomaly(self, x: np.ndarray) -> np.ndarray:
        """Returns (3,) probability vector usable by Ensemble."""
        s = self.anomaly_score(x)
        # Map anomaly score to class distribution
        normal_p  = max(0.0, 1.0 - 2 * s)
        warn_p    = 2 * s * (1 - s)
        risk_p    = s ** 2
        total     = normal_p + warn_p + risk_p + 1e-9
        return np.array([normal_p, warn_p, risk_p], dtype=np.float32) / total

    # ------------------------------------------------------------------
    def _train_torch(self, X_norm: np.ndarray, verbose: bool) -> float:
        X_t = torch.FloatTensor(X_norm).to(self._device)
        ds  = TensorDataset(X_t)
        loader = DataLoader(ds, batch_size=min(32, len(X_norm)), shuffle=True)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            for (Xb,) in loader:
                optimizer.zero_grad()
                recon = self.net(Xb)
                loss  = criterion(recon, Xb)
                loss.backward()
                optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == self.epochs):
                print(f"    AE epoch {epoch:3d}/{self.epochs} | loss={loss.item():.6f}")

        # Compute threshold from training reconstruction errors
        self.net.eval()
        with torch.no_grad():
            recon = self.net(X_t)
            errors = torch.mean((recon - X_t) ** 2, dim=1).cpu().numpy()
        return float(np.percentile(errors, self.percentile))

    def _train_sklearn(self, X_norm: np.ndarray, verbose: bool) -> float:
        enc_dims = tuple(self.hidden_sizes)
        ae = MLPRegressor(
            hidden_layer_sizes=enc_dims,
            activation="relu",
            max_iter=self.epochs,
            learning_rate_init=self.lr,
            random_state=42,
            verbose=False,
        )
        ae.fit(X_norm, X_norm)
        self._sklearn_ae = ae
        errors = np.mean((ae.predict(X_norm) - X_norm) ** 2, axis=1)
        return float(np.percentile(errors, self.percentile))

    def _reconstruction_error(self, x_s: np.ndarray) -> float:
        if TORCH_AVAILABLE and self.net is not None:
            self.net.eval()
            with torch.no_grad():
                x_t   = torch.FloatTensor(x_s).to(self._device)
                recon = self.net(x_t)
                err   = torch.mean((recon - x_t) ** 2).item()
            return float(err)
        elif hasattr(self, "_sklearn_ae"):
            recon = self._sklearn_ae.predict(x_s)
            return float(np.mean((recon - x_s) ** 2))
        return 0.0

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "n_features":    self.n_features,
            "hidden_sizes":  self.hidden_sizes,
            "threshold":     self.threshold,
            "is_calibrated": self.is_calibrated,
            "scaler":        self._scaler,
        }
        if TORCH_AVAILABLE and self.net is not None:
            state["net_state_dict"] = {k: v.cpu() for k, v in self.net.state_dict().items()}
        elif hasattr(self, "_sklearn_ae"):
            state["sklearn_ae"] = self._sklearn_ae

        joblib.dump(state, path)
        print(f"  [Autoencoder] Saved → {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Autoencoder model not found: {path}")
        state = joblib.load(path)
        self.n_features    = state["n_features"]
        self.hidden_sizes  = state["hidden_sizes"]
        self.threshold     = state["threshold"]
        self.is_calibrated = state["is_calibrated"]
        self._scaler       = state["scaler"]
        if TORCH_AVAILABLE and "net_state_dict" in state:
            self.net = _AENet(self.n_features, self.hidden_sizes).to(self._device)
            self.net.load_state_dict({k: v.to(self._device) for k, v in state["net_state_dict"].items()})
            self.net.eval()
        elif "sklearn_ae" in state:
            self._sklearn_ae = state["sklearn_ae"]
        print(f"  [Autoencoder] Loaded <- {path}")
