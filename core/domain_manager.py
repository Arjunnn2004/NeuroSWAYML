"""
NeuroSWAYML — Domain Manager
==============================
Central hub that owns all 4 domain model stacks
and handles hot-switching between them at runtime.

Usage
-----
dm = DomainManager(config)
dm.load_all_domains(models_root)

dm.set_active("elderly")
result = dm.predict_risk(feat_vec, seq_buf)   # uses elderly domain
info   = dm.active_info()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from models.domain_classifier import DomainModel, DOMAIN_META, DOMAIN_KEYS, resolve_domain


# ───────────────────────────────────────────────────────────────────────────

class DomainManager:
    """
    Manages all 4 DomainModel instances.

    Keys:
      'neurodegenerative'  → Press 1
      'elderly'            → Press 2
      'intoxication'       → Press 3
      'congenital'         → Press 4
    """

    # Keyboard ord → canonical domain name
    KEY_MAP: Dict[int, str] = {
        ord("1"): "neurodegenerative",
        ord("2"): "elderly",
        ord("3"): "intoxication",
        ord("4"): "congenital",
    }

    def __init__(self, config: dict):
        self.cfg         = config
        self.n_features  = 30
        self.seq_len     = config["inference"]["sequence_length"]

        # Domain models (created lazily on load)
        self._domains: Dict[str, DomainModel] = {}
        self._active_name: str = "elderly"

    # ------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------

    def load_all_domains(self, models_root: str, verbose: bool = True):
        """
        Load (or gracefully skip) every domain's saved model set.
        models_root layout:
          models_root/neuro/          gait_classifier.pkl …
          models_root/elderly/        …
          models_root/intoxication/   …
          models_root/congenital/     …
        """
        if verbose:
            print(f"\n[DomainManager] Loading elderly domain models from {models_root}")
        dm = DomainModel("elderly", self.cfg, models_root,
                         n_features=self.n_features, seq_len=self.seq_len)
        dm.load(verbose=verbose)
        self._domains["elderly"] = dm
        self._active_name = "elderly"
        if verbose:
            print(f"  Active domain: elderly  status: {dm.readiness}")

    # ------------------------------------------------------------------
    # DOMAIN SWITCHING
    # ------------------------------------------------------------------

    def set_active(self, name: str) -> bool:
        """Switch to domain by canonical name or alias. Returns True on success."""
        try:
            canonical = resolve_domain(name)
        except ValueError as e:
            print(f"  [DomainManager] {e}")
            return False

        if canonical not in self._domains:
            print(f"  [DomainManager] Domain '{canonical}' not loaded yet")
            return False

        prev = self._active_name
        self._active_name = canonical
        if prev != canonical:
            dm = self._domains[canonical]
            print(f"  [DomainManager] Switched: {prev} → {canonical}  "
                  f"({dm.display_name})")
            # Reset calibration so AE adapts to the person in the new domain context
            dm.recalibrate()
        return True

    def handle_key(self, key_ord: int) -> bool:
        """
        Handle a keypress by ord value.
        Returns True if domain was switched, False otherwise.
        """
        if key_ord in self.KEY_MAP:
            return self.set_active(self.KEY_MAP[key_ord])
        return False

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    @property
    def active(self) -> DomainModel:
        return self._domains[self._active_name]

    def predict_risk(
        self,
        feat_vec: np.ndarray,
        seq_buf:  Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run risk prediction on the currently active domain model."""
        return self.active.predict_risk(feat_vec, seq_buf)

    def predict_fall(self, fall_feat: np.ndarray) -> float:
        return self.active.predict_fall(fall_feat)

    def recalibrate_active(self):
        self.active.recalibrate()

    # ------------------------------------------------------------------
    # STATUS / UI HELPERS
    # ------------------------------------------------------------------

    def active_info(self) -> Dict[str, Any]:
        dm = self.active
        return {
            "domain":         self._active_name,
            "display_name":   dm.display_name,
            "class_names":    dm.class_names,
            "class_colors":   dm.class_colors,
            "is_loaded":      dm.is_loaded,
            "key_hint":       dm.meta["key_hint"],
            "readiness":      dm.readiness,
        }

    def domain_menu_lines(self) -> list[str]:
        """Return list of lines for the in-frame domain selector overlay."""
        lines = ["DOMAINS (press key to switch):"]
        for key_ord, domain in self.KEY_MAP.items():
            dm     = self._domains.get(domain)
            active = "►" if domain == self._active_name else " "
            loaded = "✓" if (dm and dm.is_loaded) else "○"
            name   = DOMAIN_META[domain]["display_name"]
            lines.append(f"  {active} [{chr(key_ord)}] {loaded} {name}")
        return lines

    def all_status(self) -> str:
        parts = []
        for domain in DOMAIN_KEYS:
            dm = self._domains.get(domain)
            if dm:
                tag = "active" if domain == self._active_name else "     "
                parts.append(f"  [{tag}] {dm.display_name}: {dm.readiness}")
        return "\n".join(parts)
