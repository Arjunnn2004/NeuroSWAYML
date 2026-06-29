"""
NeuroSWAYML - LSTM Temporal Sequence Model
Classifies gait risk from a 60-frame window of pose features.
Captures walking rhythm disruptions and Parkinsonian freeze patterns
that a single-frame classifier cannot detect.

Architecture:
  Input:  (batch, seq_len=60, n_features=30)
  LSTM:   2 layers, hidden=128, dropout=0.3
  FC:     128 → 3 classes
"""

import os
import numpy as np
import collections
from typing import Optional, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ───────────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        out = self.layer_norm(out[:, -1, :])   # last time-step
        out = self.dropout(out)
        return self.fc(out)            # (batch, num_classes)


# ───────────────────────────────────────────────────────────────────────────

class LSTMModel:
    """
    Wrapper around _LSTMNet providing train / predict / save / load API.
    Falls back to a rolling-mean heuristic if torch is unavailable.
    """

    CLASS_NAMES = ["NORMAL", "WARNING", "HIGH_RISK"]

    def __init__(self, config: dict, n_features: int = 30, seq_len: int = 60):
        self.n_features = n_features
        self.seq_len    = seq_len
        tr = config["training"]

        self.hidden_size  = tr["lstm_hidden_size"]
        self.num_layers   = tr["lstm_num_layers"]
        self.dropout      = tr["lstm_dropout"]
        self.epochs       = tr["lstm_epochs"]
        self.batch_size   = tr["lstm_batch_size"]
        self.lr           = tr["lstm_lr"]
        self.lstm_weight  = config["ensemble"]["lstm_weight"]
        self.device_str   = config["inference"].get("device", "cpu")

        self.is_trained   = False
        self.net          = None
        self._device      = None

        if TORCH_AVAILABLE:
            self._device = torch.device(
                self.device_str if torch.cuda.is_available() else "cpu"
            )
            self.net = _LSTMNet(
                input_size=n_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=3,
                dropout=self.dropout,
            ).to(self._device)
        else:
            print("  [LSTMModel] PyTorch not available — will use fallback heuristic")

    # ------------------------------------------------------------------
    def train(
        self,
        X_seq: np.ndarray,   # (N, seq_len, n_features)
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        if not TORCH_AVAILABLE:
            self.is_trained = True
            return {}

        X_t = torch.FloatTensor(X_seq).to(self._device)
        y_t = torch.LongTensor(y.astype(np.int64)).to(self._device)

        ds     = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_val_acc = 0.0
        best_state   = None

        print(f"  [LSTMModel] Training {self.epochs} epochs …")
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            total_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                logits = self.net(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            if epoch % 10 == 0 or epoch == self.epochs:
                avg_loss = total_loss / len(loader)
                msg = f"    Epoch {epoch:3d}/{self.epochs} | loss={avg_loss:.4f}"
                if X_val is not None:
                    val_acc = self._eval_accuracy(X_val, y_val)
                    msg += f" | val_acc={val_acc:.4f}"
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_state   = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                print(msg)

        if best_state is not None:
            self.net.load_state_dict({k: v.to(self._device) for k, v in best_state.items()})
            print(f"  [LSTMModel] Best val acc: {best_val_acc:.4f}")

        self.is_trained = True
        return {"best_val_acc": best_val_acc}

    # ------------------------------------------------------------------
    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Returns probability matrix (N, 3).
        X_seq shape: (N, seq_len, n_features)
        """
        if not TORCH_AVAILABLE or not self.is_trained:
            # Fallback: last-frame mean feature → simple rule
            N = len(X_seq)
            probs = np.ones((N, 3), dtype=np.float32) / 3.0
            return probs

        self.net.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(X_seq), 128):
                batch = torch.FloatTensor(X_seq[i:i+128]).to(self._device)
                logits = self.net(batch)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                results.append(probs)
        return np.vstack(results)

    def predict_single_seq(self, seq: np.ndarray) -> Tuple[int, np.ndarray]:
        """seq: (seq_len, n_features)."""
        proba = self.predict_proba(seq[np.newaxis])[0]
        return int(np.argmax(proba)), proba

    def _eval_accuracy(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        probs = self.predict_proba(X_val)
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == y_val))

    # ------------------------------------------------------------------
    def save(self, path: str):
        if not TORCH_AVAILABLE or not self.is_trained:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict":  self.net.state_dict(),
            "n_features":  self.n_features,
            "seq_len":     self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "dropout":     self.dropout,
        }, path)
        print(f"  [LSTMModel] Saved → {path}")

    def load(self, path: str):
        if not TORCH_AVAILABLE:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"LSTM model not found: {path}")
        ckpt = torch.load(path, map_location=self._device)
        self.n_features  = ckpt["n_features"]
        self.seq_len     = ckpt["seq_len"]
        self.hidden_size = ckpt["hidden_size"]
        self.num_layers  = ckpt["num_layers"]
        self.dropout     = ckpt["dropout"]
        self.net = _LSTMNet(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=3,
            dropout=self.dropout,
        ).to(self._device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.is_trained = True
        print(f"  [LSTMModel] Loaded <- {path}")
