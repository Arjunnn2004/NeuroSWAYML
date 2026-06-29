"""
Threshold-based balance tests using MediaPipe pose landmarks only.

Implements:
  - Four-Stage Balance Test: side-by-side, semi-tandem, tandem, single-leg
  - Flamingo Balance Test: one-leg stance duration and balance-loss count
"""

import collections
from typing import Any, Dict, Optional

import numpy as np


_NOSE = 0
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_HIP = 23
_R_HIP = 24
_L_ANKLE = 27
_R_ANKLE = 28
_L_HEEL = 29
_R_HEEL = 30
_L_TOE = 31
_R_TOE = 32


_FOUR_STAGE_ORDER = ("side_by_side", "semi_tandem", "tandem", "single_leg")
_FOUR_STAGE_LABELS = {
    "side_by_side": "Side-by-side",
    "semi_tandem": "Semi-tandem",
    "tandem": "Tandem",
    "single_leg": "Single-leg",
}


def _visibility(lm, idx: int) -> float:
    return float(getattr(lm[idx], "visibility", 1.0))


class BalanceTestAnalyzer:
    """Stateful balance-test scorer driven by normalized MediaPipe landmarks."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.target_four_stage_sec = 10.0
        self.target_flamingo_sec = 60.0
        self.reset()

    def reset(self):
        self._last_ts: Optional[float] = None
        self._active_stage: Optional[str] = None
        self._stage_hold = {name: 0.0 for name in _FOUR_STAGE_ORDER}
        self._stage_best = {name: 0.0 for name in _FOUR_STAGE_ORDER}
        self._stage_passed = {name: False for name in _FOUR_STAGE_ORDER}
        self._sway_buf = collections.deque(maxlen=45)
        self._foot_buf = collections.deque(maxlen=20)

        self._flamingo_active = False
        self._flamingo_side: Optional[str] = None
        self._flamingo_hold = 0.0
        self._flamingo_best = 0.0
        self._flamingo_losses = 0
        self._flamingo_no_pose_sec = 0.0
        self._last_loss_ts = -999.0

    def update(self, landmarks, timestamp: float) -> Dict[str, Any]:
        dt = self._frame_dt(timestamp)

        if landmarks is None or len(landmarks) < 33 or not self._has_reliable_lower_body(landmarks):
            self._handle_missing_pose(dt)
            return self._snapshot(
                stance="no_pose",
                stable=False,
                sway_norm=0.0,
                torso_angle=0.0,
                foot_motion=0.0,
                support_side=None,
                reason="No pose detected",
            )

        metrics = self._measure(landmarks)
        stance = metrics["stance"]
        stable = metrics["stable"]

        if stance != self._active_stage:
            self._active_stage = stance
            self._sway_buf.clear()
            self._foot_buf.clear()

        self._sway_buf.append(metrics["hip_mid_x"])
        self._foot_buf.append(metrics["foot_mid"])
        sway_norm = self._sway_norm(metrics["shoulder_width"])
        foot_motion = self._foot_motion(metrics["shoulder_width"])
        stable = stable and sway_norm <= metrics["sway_limit"] and foot_motion <= metrics["foot_motion_limit"]

        self._update_four_stage(stance, stable, dt)
        self._update_flamingo(stance, stable, metrics, sway_norm, timestamp, dt)

        return self._snapshot(
            stance=stance,
            stable=stable,
            sway_norm=sway_norm,
            torso_angle=metrics["torso_angle"],
            foot_motion=foot_motion,
            support_side=metrics["support_side"],
            reason=metrics["reason"] if not stable else "Stable hold",
        )

    def _frame_dt(self, timestamp: float) -> float:
        if self._last_ts is None:
            self._last_ts = timestamp
            return 1.0 / self.fps
        dt = float(np.clip(timestamp - self._last_ts, 1.0 / 90.0, 0.25))
        self._last_ts = timestamp
        return dt

    def _has_reliable_lower_body(self, lm) -> bool:
        required = (_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP, _L_ANKLE, _R_ANKLE)
        return all(_visibility(lm, idx) >= 0.45 for idx in required)

    def _measure(self, lm) -> Dict[str, Any]:
        def p(idx):
            return np.array([lm[idx].x, lm[idx].y], dtype=np.float64)

        nose = p(_NOSE)
        l_sh = p(_L_SHOULDER)
        r_sh = p(_R_SHOULDER)
        l_hip = p(_L_HIP)
        r_hip = p(_R_HIP)
        l_ank = p(_L_ANKLE)
        r_ank = p(_R_ANKLE)
        l_heel = p(_L_HEEL)
        r_heel = p(_R_HEEL)
        l_toe = p(_L_TOE)
        r_toe = p(_R_TOE)

        shoulder_width = max(float(abs(l_sh[0] - r_sh[0])), 0.06)
        left_foot = np.mean([l_ank, l_heel, l_toe], axis=0)
        right_foot = np.mean([r_ank, r_heel, r_toe], axis=0)
        foot_x_gap = float(abs(left_foot[0] - right_foot[0]))
        foot_y_gap = float(abs(left_foot[1] - right_foot[1]))
        body_height = max(float(abs(((l_ank[1] + r_ank[1]) / 2.0) - nose[1])), 0.35)

        hip_mid = (l_hip + r_hip) / 2.0
        sh_mid = (l_sh + r_sh) / 2.0
        torso_angle = float(np.degrees(np.arctan2(abs(hip_mid[0] - sh_mid[0]), abs(hip_mid[1] - sh_mid[1]) + 1e-9)))

        left_foot_y = float(max(l_ank[1], l_heel[1], l_toe[1]))
        right_foot_y = float(max(r_ank[1], r_heel[1], r_toe[1]))
        left_raised = right_foot_y - left_foot_y > 0.08 * body_height
        right_raised = left_foot_y - right_foot_y > 0.08 * body_height

        support_side = None
        if left_raised and not right_raised:
            stance = "single_leg"
            support_side = "right"
            foot_mid = right_foot
        elif right_raised and not left_raised:
            stance = "single_leg"
            support_side = "left"
            foot_mid = left_foot
        else:
            x_gap_ratio = foot_x_gap / shoulder_width
            y_gap_ratio = foot_y_gap / body_height
            if y_gap_ratio >= 0.09 and x_gap_ratio <= 0.75:
                stance = "tandem"
            elif y_gap_ratio >= 0.04 and x_gap_ratio <= 1.05:
                stance = "semi_tandem"
            elif x_gap_ratio >= 0.55:
                stance = "side_by_side"
            elif x_gap_ratio >= 0.25:
                stance = "semi_tandem"
            else:
                stance = "tandem"
            foot_mid = (left_foot + right_foot) / 2.0

        sway_limit = {
            "side_by_side": 0.38,
            "semi_tandem": 0.30,
            "tandem": 0.24,
            "single_leg": 0.28,
        }[stance]
        torso_limit = 18.0 if stance != "single_leg" else 22.0
        stable = torso_angle <= torso_limit
        reason = "Torso lean over threshold" if not stable else ""
        foot_motion_limit = 0.28 if stance == "single_leg" else 0.22

        return {
            "stance": stance,
            "stable": stable,
            "reason": reason,
            "sway_limit": sway_limit,
            "shoulder_width": shoulder_width,
            "hip_mid_x": float(hip_mid[0]),
            "foot_mid": foot_mid.astype(np.float64),
            "torso_angle": torso_angle,
            "support_side": support_side,
            "foot_motion_limit": foot_motion_limit,
        }

    def _sway_norm(self, shoulder_width: float) -> float:
        if len(self._sway_buf) < 6:
            return 0.0
        return float((max(self._sway_buf) - min(self._sway_buf)) / (shoulder_width + 1e-9))

    def _foot_motion(self, shoulder_width: float) -> float:
        if len(self._foot_buf) < 6:
            return 0.0
        arr = np.array(self._foot_buf)
        span = np.max(np.linalg.norm(arr - arr[0], axis=1))
        return float(span / (shoulder_width + 1e-9))

    def _update_four_stage(self, stance: str, stable: bool, dt: float):
        for stage in _FOUR_STAGE_ORDER:
            if stage == stance and stable:
                self._stage_hold[stage] += dt
            else:
                self._stage_hold[stage] = 0.0

            self._stage_best[stage] = max(self._stage_best[stage], self._stage_hold[stage])
            self._stage_passed[stage] = self._stage_best[stage] >= self.target_four_stage_sec

    def _update_flamingo(
        self,
        stance: str,
        stable: bool,
        metrics: Dict[str, Any],
        sway_norm: float,
        timestamp: float,
        dt: float,
    ):
        if stance == "single_leg":
            if self._flamingo_active and metrics["support_side"] != self._flamingo_side:
                self._register_flamingo_loss(timestamp)

            if not self._flamingo_active or metrics["support_side"] != self._flamingo_side:
                self._flamingo_active = True
                self._flamingo_side = metrics["support_side"]
                self._flamingo_hold = 0.0
                self._flamingo_no_pose_sec = 0.0

            if stable:
                self._flamingo_hold += dt
                self._flamingo_best = max(self._flamingo_best, self._flamingo_hold)
            else:
                self._register_flamingo_loss(timestamp)
                self._flamingo_hold = 0.0
        elif self._flamingo_active:
            self._register_flamingo_loss(timestamp)
            self._flamingo_hold = 0.0
            self._flamingo_active = False

        if sway_norm > 0.36 and self._flamingo_active:
            self._register_flamingo_loss(timestamp)
            self._flamingo_hold = 0.0

    def _handle_missing_pose(self, dt: float):
        if self._flamingo_active:
            self._flamingo_no_pose_sec += dt
            if self._flamingo_no_pose_sec > 0.5:
                self._flamingo_losses += 1
                self._flamingo_hold = 0.0
                self._flamingo_active = False
        for stage in _FOUR_STAGE_ORDER:
            self._stage_hold[stage] = 0.0

    def _register_flamingo_loss(self, timestamp: float):
        if timestamp - self._last_loss_ts < 0.8:
            return
        if self._flamingo_hold >= 0.8:
            self._flamingo_losses += 1
        self._last_loss_ts = timestamp

    def _snapshot(
        self,
        stance: str,
        stable: bool,
        sway_norm: float,
        torso_angle: float,
        foot_motion: float,
        support_side: Optional[str],
        reason: str,
    ) -> Dict[str, Any]:
        four_stages = []
        for stage in _FOUR_STAGE_ORDER:
            four_stages.append({
                "key": stage,
                "name": _FOUR_STAGE_LABELS[stage],
                "target_seconds": self.target_four_stage_sec,
                "hold_seconds": round(self._stage_hold[stage], 1),
                "best_seconds": round(self._stage_best[stage], 1),
                "passed": self._stage_passed[stage],
                "active": stance == stage,
            })

        passed_count = sum(1 for stage in _FOUR_STAGE_ORDER if self._stage_passed[stage])
        four_status = "PASS" if passed_count == len(_FOUR_STAGE_ORDER) else "IN_PROGRESS"
        if passed_count < 3 and any(self._stage_best[s] >= 1.0 for s in _FOUR_STAGE_ORDER):
            four_status = "FALL_RISK_SCREEN"

        flamingo_passed = self._flamingo_best >= self.target_flamingo_sec and self._flamingo_losses <= 3
        flamingo_status = "PASS" if flamingo_passed else ("ACTIVE" if self._flamingo_active else "READY")

        return {
            "stance": stance,
            "stance_label": _FOUR_STAGE_LABELS.get(stance, "No pose"),
            "stable": stable,
            "reason": reason,
            "sway_norm": round(sway_norm, 3),
            "torso_angle": round(torso_angle, 1),
            "foot_motion": round(foot_motion, 3),
            "four_stage": {
                "status": four_status,
                "passed_count": passed_count,
                "total_count": len(_FOUR_STAGE_ORDER),
                "stages": four_stages,
            },
            "flamingo": {
                "status": flamingo_status,
                "support_side": support_side or self._flamingo_side,
                "target_seconds": self.target_flamingo_sec,
                "hold_seconds": round(self._flamingo_hold, 1),
                "best_seconds": round(self._flamingo_best, 1),
                "losses": self._flamingo_losses,
                "passed": flamingo_passed,
            },
        }
