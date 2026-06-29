"""
NeuroSWAYML — ML Analyzer  (Multi-Domain Edition)
===================================================
Real-time inference pipeline with 4 switchable analysis domains:

  1 — Neurodegenerative   (PD / ALS / HD)
  2 — Elderly Gait        (Fall Risk)
  3 — Intoxication / Ataxia
  4 — Congenital / Birth Disorder

Call flow per frame:
  PoseResult → FeatureExtractor → DomainManager.predict_risk() → annotated dict

Other responsibilities:
  • Calibration phase (AE personal baseline)
  • Sequence buffer for LSTM
  • Structured log output
  • On-screen panel with domain badge + live sensor data
"""

import os
import cv2
import json
import time
import collections
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

from data.feature_extractor  import FeatureExtractor
from core.domain_manager     import DomainManager
from core.pose_engine        import PoseResult
from core.balance_tests      import BalanceTestAnalyzer

import mediapipe as mp


# MediaPipe pose skeleton connections (formerly mp.solutions.pose.POSE_CONNECTIONS)
_POSE_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
])


# ───────────────────────────────────────────────────────────────────────────

class MLAnalyzer:
    """
    Stateful ML analyzer — domain-switching edition.

    Lifecycle:
      analyzer = MLAnalyzer(config)
      analyzer.load_models()                    # load all domain models
      result = analyzer.process(pose_result)    # call each frame
      analyzer.set_domain("elderly")            # switch domain at runtime
      analyzer.save_log()                       # persist session log
    """

    CALIBRATION_MESSAGE = "CALIBRATING — Please walk normally…"

    def __init__(self, config: dict):
        self.cfg     = config
        self.seq_len = config["inference"]["sequence_length"]
        self.calib_n = config["inference"]["calibration_frames"]
        self.fps     = 30.0

        # Feature extractor
        self.feat_ex    = FeatureExtractor(fps=self.fps)
        self.n_features = len(FeatureExtractor.FEATURE_NAMES)
        self.balance_tests = BalanceTestAnalyzer(fps=self.fps)

        # Sequence buffer (reset on domain switch)
        self._seq_buf  = collections.deque(maxlen=self.seq_len)

        # Calibration state
        self._calib_buf    = []
        self.is_calibrated = False

        # ── Domain manager owns all 4 model stacks ────────────────────
        self.domain_mgr = DomainManager(config)

        # Models directory
        self._models_dir = os.path.join(
            os.path.dirname(__file__), "..", "saved_models"
        )

        # Session state
        self._log_events   = []
        self._frame_count  = 0
        self._fall_frames  = 0
        self._last_result  : Optional[Dict[str, Any]] = None

        # Output dirs
        out = config["output"]
        if out["save_logs"]:
            os.makedirs(out["log_dir"], exist_ok=True)
        if out["save_fall_frames"]:
            os.makedirs(out["fall_frames_dir"], exist_ok=True)

    # ------------------------------------------------------------------
    # MODEL MANAGEMENT
    # ------------------------------------------------------------------

    def load_models(self):
        """Load the elderly domain models from saved_models/elderly/."""
        print(f"  [MLAnalyzer] Loading models from {self._models_dir}")
        self.domain_mgr.load_all_domains(self._models_dir, verbose=True)

        if self.domain_mgr.active.ae.is_calibrated:
            self.is_calibrated = True

    # ------------------------------------------------------------------
    # DOMAIN SWITCHING
    # ------------------------------------------------------------------

    def set_domain(self, name: str) -> bool:
        """Set the active analysis domain."""
        ok = self.domain_mgr.set_active(name)
        if ok:
            self._calib_buf.clear()
            self._seq_buf.clear()
            self._fall_frames  = 0
            self.is_calibrated = self.domain_mgr.active.ae.is_calibrated
        return ok

    # ------------------------------------------------------------------
    # MAIN PROCESSING
    # ------------------------------------------------------------------

    def process(self, pose_res: PoseResult) -> Dict[str, Any]:
        """Process one PoseResult. Returns annotated frame + full result dict."""
        self._frame_count += 1
        frame = pose_res.frame.copy()
        h, w  = frame.shape[:2]

        domain_info = self.domain_mgr.active_info()

        result: Dict[str, Any] = {
            "frame_idx":      pose_res.frame_idx,
            "timestamp":      pose_res.timestamp,
            "pose_detected":  False,
            "calibrating":    not self.is_calibrated,
            "risk_score":     0.0,
            "risk_class":     0,
            "class_label":    domain_info["class_names"][0],
            "color":          (0, 220, 0),
            "fall_prob":      0.0,
            "fall_detected":  False,
            "domain":         domain_info["domain"],
            "domain_display": domain_info["display_name"],
            "balance_tests":  {},
        }

        if pose_res.landmarks_2d is None:
            result["balance_tests"] = self.balance_tests.update(None, pose_res.timestamp)
            self._draw_no_pose(frame)
            result["annotated_frame"] = frame
            return result

        result["pose_detected"] = True
        lm_2d = pose_res.landmarks_2d
        lm_3d = pose_res.landmarks_3d  # already a list or None

        # ── Feature extraction ────────────────────────────────────────
        feat = self.feat_ex.extract(lm_2d, lm_3d, w, h)

        # ── App-compatible sensor readouts ────────────────────────────
        _fn = FeatureExtractor.FEATURE_NAMES
        result["sway_idx"]      = float(feat[_fn.index("sway_std")]        * 100)
        result["leg_ratio"]     = float(feat[_fn.index("leg_length_ratio")])
        result["heel_toe_l"]    = float(feat[_fn.index("heel_toe_diff_l")] * 100)
        result["heel_toe_r"]    = float(feat[_fn.index("heel_toe_diff_r")] * 100)
        result["torso_angle_v"] = float(feat[_fn.index("torso_angle")])
        result["gait_sym"]      = float(feat[_fn.index("gait_symmetry")])
        result["stride_cv_v"]   = float(feat[_fn.index("stride_cv")])
        result["cadence_v"]     = float(feat[_fn.index("cadence_norm")])
        result["knee_diff"]     = float(feat[_fn.index("knee_angle_diff")])

        # Threshold-based rule alerts (independent of ML domain)
        _issues: list = []
        if result["sway_idx"] > 2.5:
            _issues.append("INTOXICATION / ATAXIA (Excessive Sway)")
        if result["leg_ratio"] > 1.05:
            _issues.append("ASYMMETRY: Left Leg Longer")
        elif result["leg_ratio"] < 0.95:
            _issues.append("ASYMMETRY: Right Leg Longer")
        if result["heel_toe_l"] > 8.0 or result["heel_toe_r"] > 8.0:
            _issues.append("TOE-DOMINANT GAIT (Tip-Toeing)")
        elif result["heel_toe_l"] < -3.0 or result["heel_toe_r"] < -3.0:
            _issues.append("HEEL-DOMINANT GAIT")
        if result["stride_cv_v"] > 0.15:
            _issues.append("IRREGULAR STRIDE (High CV)")
        if result["torso_angle_v"] > 15.0:
            _issues.append("FORWARD LEAN / STOOPED POSTURE")
        result["threshold_issues"] = _issues
        result["balance_tests"] = self.balance_tests.update(lm_2d, pose_res.timestamp)

        # ── Calibration phase ─────────────────────────────────────────
        if not self.is_calibrated:
            self._calib_buf.append(feat.copy())
            progress = len(self._calib_buf) / self.calib_n
            if len(self._calib_buf) >= self.calib_n:
                self._finish_calibration()
            result["calibration_progress"] = progress
            self._draw_calibration(frame, progress)
            self._draw_skeleton(frame, pose_res.landmarks_2d)
            result["annotated_frame"] = frame
            return result

        # ── Sequence buffer ───────────────────────────────────────────
        self._seq_buf.append(feat.copy())
        seq_buf = (np.array(self._seq_buf, dtype=np.float32)
                   if len(self._seq_buf) == self.seq_len else None)

        # ── Domain risk prediction ────────────────────────────────────
        risk_result = self.domain_mgr.predict_risk(feat, seq_buf)
        result.update(risk_result)

        # ── Fall detection ────────────────────────────────────────────
        fall_prob     = self.domain_mgr.predict_fall(feat)
        fall_detected = fall_prob > 0.60
        result["fall_prob"]     = fall_prob
        result["fall_detected"] = fall_detected

        if fall_detected:
            self._fall_frames += 1
            if self._fall_frames >= 3:
                self._log_fall(result, frame)
                if self.cfg["output"]["save_fall_frames"]:
                    self._save_fall_frame(frame)
        else:
            self._fall_frames = max(0, self._fall_frames - 1)

        # ── Draw UI ───────────────────────────────────────────────────
        self._draw_skeleton(frame, pose_res.landmarks_2d)
        self._draw_panel(frame, result, domain_info)

        self._last_result = result
        result["annotated_frame"] = frame
        return result

    # ------------------------------------------------------------------
    # CALIBRATION
    # ------------------------------------------------------------------

    def _finish_calibration(self):
        X_normal = np.array(self._calib_buf, dtype=np.float32)
        print(f"  [MLAnalyzer] Calibrating AE on {len(X_normal)} normal frames "
              f"({self.domain_mgr.active.display_name})")
        self.domain_mgr.active.ae.calibrate(X_normal, verbose=False)
        self.is_calibrated = True
        self._calib_buf.clear()
        print("  [MLAnalyzer] Autoencoder personalised to this patient")

    def recalibrate(self):
        """Restart personal baseline calibration for the active domain."""
        self.is_calibrated = False
        self._calib_buf.clear()
        self.feat_ex.reset()
        self.balance_tests.reset()
        self._seq_buf.clear()
        self.domain_mgr.recalibrate_active()
        print("  [MLAnalyzer] Recalibration started")

    # ------------------------------------------------------------------
    # DRAWING
    # ------------------------------------------------------------------

    def _draw_skeleton(self, frame, landmarks):
        """Draw pose skeleton using cv2 (no mp.solutions dependency)."""
        if landmarks is None:
            return
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for i, j in _POSE_CONNECTIONS:
            if i < len(pts) and j < len(pts):
                cv2.line(frame, pts[i], pts[j], (0, 200, 100), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 4, (0, 180, 80), 1)

    def _draw_calibration(self, frame, progress: float):
        h, w  = frame.shape[:2]
        bar_w = int(w * 0.6)
        bar_x = (w - bar_w) // 2
        bar_y = h - 80
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (50, 50, 50), -1)
        fill = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 20), (0, 200, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (200, 200, 200), 1)
        cv2.putText(frame, self.CALIBRATION_MESSAGE,
                    (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"{int(progress * 100)}%",
                    (bar_x + bar_w // 2 - 20, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_no_pose(self, frame):
        cv2.putText(frame, "No pose detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def _draw_domain_badge(self, frame, domain_info: dict):
        """Small badge in top-right corner showing active domain."""
        h, w  = frame.shape[:2]
        label = f"[{domain_info['key_hint']}] {domain_info['display_name']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bx = w - tw - 16
        by = 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx - 4, by), (bx + tw + 4, by + th + 6), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, label, (bx, by + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 180), 1)
        cv2.putText(frame, "M=menu  1-4=switch", (bx - 20, by + th + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    def _draw_domain_menu(self, frame):
        """Semi-transparent domain-selector overlay (press M to toggle)."""
        h, w   = frame.shape[:2]
        lines  = self.domain_mgr.domain_menu_lines()
        lh     = 22
        box_h  = len(lines) * lh + 20
        box_w  = 340
        bx     = w // 2 - box_w // 2
        by     = h // 2 - box_h // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, 200, 150), 1)
        for i, line in enumerate(lines):
            color = (0, 200, 150) if i == 0 else (220, 220, 220)
            cv2.putText(frame, line, (bx + 10, by + 18 + i * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    def _draw_panel(self, frame: np.ndarray, result: dict, domain_info: dict):
        h, w    = frame.shape[:2]
        alpha   = self.cfg["visualization"]["panel_alpha"]
        color   = result["color"]
        rs      = result["risk_score"]
        label   = result["class_label"]
        fp      = result["fall_prob"]
        cn      = domain_info["class_names"]

        PANEL_W = 440
        PANEL_H = 650

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (PANEL_W, PANEL_H), (10, 10, 10), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (0, 0), (PANEL_W, PANEL_H), color, 2)

        y = 26
        cv2.putText(frame, "NeuroSWAYML — Elderly Fall Risk", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
        y += 30
        cv2.putText(frame, f"RISK:  {label}", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        y += 30

        # Risk score bar
        bar_w = PANEL_W - 50
        bar_filled = int(bar_w * rs)
        cv2.rectangle(frame, (10, y), (10 + bar_w, y + 14), (50, 50, 50), -1)
        bar_color = (0, 255, 0) if rs < 0.4 else ((0, 165, 255) if rs < 0.65 else (0, 0, 255))
        cv2.rectangle(frame, (10, y), (10 + bar_filled, y + 14), bar_color, -1)
        cv2.putText(frame, f"{rs:.2f}", (bar_w // 2, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 28

        pn  = result.get("proba_normal",    0.0)
        pw  = result.get("proba_warning",   0.0)
        ph  = result.get("proba_high_risk", 0.0)
        cv2.putText(frame,
                    f"{cn[0][:5]}:{pn:.2f}  {cn[1][:5]}:{pw:.2f}  {cn[2][:5]}:{ph:.2f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
        y += 24

        fall_col = (0, 0, 255) if result["fall_detected"] else (200, 200, 200)
        fall_txt = "FALL DETECTED!" if result["fall_detected"] else f"Fall prob: {fp:.2f}"
        cv2.putText(frame, fall_txt, (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, fall_col, 2)
        y += 28

        cv2.line(frame, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
        y += 14
        issues = result.get("threshold_issues", [])
        if issues:
            for issue in issues:
                cv2.putText(frame, f"[!] {issue}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 100, 255), 1)
                y += 20
        else:
            cv2.putText(frame, "No threshold alerts", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 200, 0), 1)
            y += 20

        cv2.line(frame, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
        y += 14
        balance = result.get("balance_tests", {})
        if balance:
            four = balance.get("four_stage", {})
            flamingo = balance.get("flamingo", {})
            cv2.putText(frame, "--- BALANCE TESTS ---", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 150, 150), 1)
            y += 18
            stance_col = (0, 220, 0) if balance.get("stable") else (0, 165, 255)
            cv2.putText(frame,
                        f"Stance: {balance.get('stance_label', '-')}  "
                        f"{'STABLE' if balance.get('stable') else 'UNSTABLE'}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, stance_col, 1)
            y += 18
            cv2.putText(frame,
                        f"Four-stage: {four.get('passed_count', 0)}/"
                        f"{four.get('total_count', 4)} passed",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (200, 200, 200), 1)
            y += 18
            for stage in four.get("stages", [])[:4]:
                mark = "OK" if stage.get("passed") else ("*" if stage.get("active") else "--")
                cv2.putText(frame,
                            f"{mark} {stage.get('name', '-')[:12]} "
                            f"{stage.get('best_seconds', 0):.1f}/"
                            f"{stage.get('target_seconds', 10):.0f}s",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (0, 220, 0) if stage.get("passed") else (190, 190, 190), 1)
                y += 16
            cv2.putText(frame,
                        f"Flamingo: {flamingo.get('best_seconds', 0):.1f}/"
                        f"{flamingo.get('target_seconds', 60):.0f}s  "
                        f"losses:{flamingo.get('losses', 0)}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41,
                        (0, 220, 0) if flamingo.get("passed") else (200, 200, 200), 1)
            y += 20

        cv2.line(frame, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
        y += 14
        cv2.putText(frame, "--- LIVE SENSOR DATA ---", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 150, 150), 1)
        y += 18

        sway   = result.get("sway_idx",    0.0)
        lratio = result.get("leg_ratio",   1.0)
        htl    = result.get("heel_toe_l",  0.0)
        htr    = result.get("heel_toe_r",  0.0)
        tor    = result.get("torso_angle_v", 0.0)
        sym    = result.get("gait_sym",    1.0)
        scv    = result.get("stride_cv_v", 0.0)
        cad    = result.get("cadence_v",   0.0)
        kdiff  = result.get("knee_diff",   0.0)

        sensor_lines = [
            (f"Sway Idx:    {sway:.2f}  (Thresh: 2.5)",
             (0, 100, 255) if sway > 2.5 else (200, 200, 200)),
            (f"Leg Ratio:   {lratio:.3f} (Ideal: 1.0)",
             (0, 100, 255) if lratio > 1.05 or lratio < 0.95 else (200, 200, 200)),
            (f"Heel-Toe L: {htl:.1f}  R: {htr:.1f}",
             (0, 100, 255) if htl > 8.0 or htl < -3.0 else (200, 200, 200)),
            (f"Torso Angle: {tor:.1f} deg",
             (0, 165, 255) if tor > 15.0 else (200, 200, 200)),
            (f"Gait Sym:    {sym:.3f}  (Ideal: 1.0)",
             (0, 165, 255) if sym < 0.85 else (200, 200, 200)),
            (f"Stride CV:   {scv:.3f}  (Thresh: 0.15)",
             (0, 165, 255) if scv > 0.15 else (200, 200, 200)),
            (f"Cadence:     {cad:.2f} steps/s",
             (200, 200, 200)),
            (f"Knee Diff:   {kdiff:.1f} deg",
             (0, 165, 255) if kdiff > 10.0 else (200, 200, 200)),
        ]
        for txt, col in sensor_lines:
            if y + 18 > PANEL_H - 5:
                break
            cv2.putText(frame, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, col, 1)
            y += 18

        if self.cfg["visualization"]["show_confidence_bar"]:
            conf = float(max(pn, pw, ph))
            cv2.putText(frame, f"Conf: {conf:.2f}", (10, PANEL_H - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        if self.cfg["visualization"]["show_fps"]:
            cv2.putText(frame, f"FPS: {result.get('fps', 0):.1f}",
                        (w - 100, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if result["fall_detected"]:
            alert_overlay = frame.copy()
            cv2.rectangle(alert_overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(alert_overlay, 0.15, frame, 0.85, 0, frame)
            cv2.putText(frame, "!! FALL DETECTED !!",
                        (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------

    def _log_fall(self, result: dict, frame: np.ndarray):
        event = {
            "timestamp":  datetime.now().isoformat(),
            "frame_idx":  result["frame_idx"],
            "fall_prob":  result["fall_prob"],
            "risk_score": result["risk_score"],
            "risk_class": result["risk_class"],
            "domain":     result.get("domain", "unknown"),
        }
        self._log_events.append(event)

    def _save_fall_frame(self, frame: np.ndarray):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.cfg["output"]["fall_frames_dir"], f"fall_{ts}.jpg")
        cv2.imwrite(path, frame)

    def save_log(self):
        if not self.cfg["output"]["save_logs"] or not self._log_events:
            return
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.cfg["output"]["log_dir"], f"ml_session_{ts}.json")
        with open(log_path, "w") as f:
            json.dump({
                "session_start":  ts,
                "total_frames":   self._frame_count,
                "active_domain":  self.domain_mgr._active_name,
                "fall_events":    self._log_events,
            }, f, indent=2)
        print(f"  [MLAnalyzer] Log saved → {log_path}")

