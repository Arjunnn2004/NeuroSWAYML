"""
NeuroSWAYML - Feature Extractor
Converts raw MediaPipe pose landmark results into the consistent
numerical feature vector consumed by the ML classifiers.

Feature vector order must exactly match DatasetLoader.FEATURE_NAMES.
"""

import math
import numpy as np
import collections
from typing import Optional, List, Dict, Any

# MediaPipe landmark indices (same as mp.solutions.pose.PoseLandmark)
_NOSE         = 0
_L_SHOULDER   = 11; _R_SHOULDER = 12
_L_ELBOW      = 13; _R_ELBOW    = 14
_L_WRIST      = 15; _R_WRIST    = 16
_L_HIP        = 23; _R_HIP      = 24
_L_KNEE       = 25; _R_KNEE     = 26
_L_ANKLE      = 27; _R_ANKLE    = 28
_L_HEEL       = 29; _R_HEEL     = 30
_L_TOE        = 31; _R_TOE      = 32

_SWAY_BUF     = 90   # frames for sway history
_VEL_BUF      = 6    # frames for velocity (low-pass)
_ANGLE_BUF    = 20   # frames for angle smoothing
_STRIDE_BUF   = 30   # frames for stride stats


class FeatureExtractor:
    """
    Stateful feature extractor.
    Call `extract(lm_2d, lm_world, image_w, image_h)` each frame.
    Returns a flat np.ndarray matching DatasetLoader.FEATURE_NAMES.

    Also produces a separate `extract_fall_features()` vector matching
    DatasetLoader.FALL_FEATURE_NAMES.
    """

    FEATURE_NAMES = [
        "sway_std", "sway_range", "sway_cv", "sway_fft_peak_freq",
        "sway_fft_energy", "torso_angle", "torso_angle_std",
        "left_knee_angle", "right_knee_angle", "knee_angle_diff",
        "left_knee_var", "right_knee_var",
        "left_hip_angle", "right_hip_angle",
        "left_ankle_angle", "right_ankle_angle",
        "gait_symmetry", "stride_cv", "stride_length_norm",
        "cadence_norm", "step_width_norm",
        "heel_toe_diff_l", "heel_toe_diff_r",
        "leg_length_ratio", "hip_height_norm", "aspect_ratio",
        "head_vel", "hip_vel", "ankle_vel_l", "ankle_vel_r",
    ]

    FALL_FEATURE_NAMES = [
        "hip_height_world", "torso_angle", "aspect_ratio",
        "head_vel", "hip_vel", "shoulder_hip_dist", "body_height_ratio",
    ]

    # ------------------------------------------------------------------
    def __init__(self, fps: float = 30.0):
        self.fps = fps

        # Rolling buffers
        self._sway_buf        = collections.deque(maxlen=_SWAY_BUF)
        self._torso_buf       = collections.deque(maxlen=_ANGLE_BUF)
        self._stride_buf      = collections.deque(maxlen=_STRIDE_BUF)
        self._l_knee_buf      = collections.deque(maxlen=_ANGLE_BUF)
        self._r_knee_buf      = collections.deque(maxlen=_ANGLE_BUF)

        # Velocity tracking (3D world coords if available)
        self._prev_head_pos   = None
        self._prev_hip_pos    = None
        self._prev_l_ankle    = None
        self._prev_r_ankle    = None

        self._head_vel_buf    = collections.deque(maxlen=_VEL_BUF)
        self._hip_vel_buf     = collections.deque(maxlen=_VEL_BUF)
        self._l_ank_vel_buf   = collections.deque(maxlen=_VEL_BUF)
        self._r_ank_vel_buf   = collections.deque(maxlen=_VEL_BUF)

        # Stride / cadence
        self._prev_l_ankle_x  = None
        self._prev_r_ankle_x  = None
        self._stride_event_ts : List[float] = []
        self._frame_count     = 0

    # ------------------------------------------------------------------
    # MAIN EXTRACTION
    # ------------------------------------------------------------------

    def extract(
        self,
        lm,           # results.pose_landmarks.landmark  (2D normalised)
        lm_world,     # results.pose_world_landmarks.landmark (3D metres) or None
        image_w: int,
        image_h: int,
    ) -> np.ndarray:
        """
        Returns feature vector of length len(FEATURE_NAMES).
        Any unavailable metric defaults to 0.
        """
        self._frame_count += 1

        # ── Convenience accessors ─────────────────────────────────────
        def p2(idx):
            """2D normalised landmark (x, y)."""
            lmk = lm[idx]
            return np.array([lmk.x, lmk.y], dtype=np.float64)

        def p3(idx):
            """3D world landmark (x, y, z) in metres, if available."""
            if lm_world is None:
                return None
            w = lm_world[idx]
            return np.array([w.x, w.y, w.z], dtype=np.float64)

        # ── 1. SWAY ANALYSIS ─────────────────────────────────────────
        mid_hip_x = (lm[_L_HIP].x + lm[_R_HIP].x) / 2.0
        self._sway_buf.append(mid_hip_x)

        sway_arr   = np.array(self._sway_buf)
        sway_std   = float(np.std(sway_arr))  if len(sway_arr) > 5  else 0.0
        sway_range = float(np.ptp(sway_arr))  if len(sway_arr) > 5  else 0.0
        sway_cv    = sway_std / (np.mean(np.abs(sway_arr)) + 1e-9)

        # FFT on sway buffer (detect Parkinsonian tremor frequency)
        sway_fft_peak = 0.0
        sway_fft_energy = 0.0
        if len(sway_arr) >= 32:
            fft_mag  = np.abs(np.fft.rfft(sway_arr - sway_arr.mean()))
            freqs    = np.fft.rfftfreq(len(sway_arr), d=1.0 / self.fps)
            peak_idx = int(np.argmax(fft_mag[1:])) + 1
            sway_fft_peak   = float(freqs[peak_idx])
            sway_fft_energy = float(np.sum(fft_mag[1:] ** 2))

        # ── 2. TORSO ANGLE ───────────────────────────────────────────
        l_sh = p2(_L_SHOULDER); r_sh = p2(_R_SHOULDER)
        l_hp = p2(_L_HIP);      r_hp = p2(_R_HIP)
        sh_mid = (l_sh + r_sh) / 2
        hp_mid = (l_hp + r_hp) / 2
        dx = hp_mid[0] - sh_mid[0]
        dy = hp_mid[1] - sh_mid[1]
        torso_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))
        self._torso_buf.append(torso_angle)
        torso_angle_std = float(np.std(self._torso_buf)) if len(self._torso_buf) > 3 else 0.0

        # ── 3. JOINT ANGLES (2D, degrees) ───────────────────────────
        def angle_2d(a, b, c):
            v1 = a - b; v2 = c - b
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            return math.degrees(math.acos(np.clip(cos_a, -1, 1)))

        l_knee_ang  = angle_2d(p2(_L_HIP),    p2(_L_KNEE),  p2(_L_ANKLE))
        r_knee_ang  = angle_2d(p2(_R_HIP),    p2(_R_KNEE),  p2(_R_ANKLE))
        l_hip_ang   = angle_2d(p2(_L_SHOULDER), p2(_L_HIP), p2(_L_KNEE))
        r_hip_ang   = angle_2d(p2(_R_SHOULDER), p2(_R_HIP), p2(_R_KNEE))
        l_ankle_ang = angle_2d(p2(_L_KNEE),   p2(_L_ANKLE), p2(_L_TOE))
        r_ankle_ang = angle_2d(p2(_R_KNEE),   p2(_R_ANKLE), p2(_R_TOE))

        self._l_knee_buf.append(l_knee_ang)
        self._r_knee_buf.append(r_knee_ang)
        l_knee_var = float(np.var(self._l_knee_buf)) if len(self._l_knee_buf) > 5 else 0.0
        r_knee_var = float(np.var(self._r_knee_buf)) if len(self._r_knee_buf) > 5 else 0.0

        knee_angle_diff = abs(l_knee_ang - r_knee_ang)

        # ── 4. GAIT SYMMETRY ─────────────────────────────────────────
        lk_mean = np.mean(self._l_knee_buf) if self._l_knee_buf else l_knee_ang
        rk_mean = np.mean(self._r_knee_buf) if self._r_knee_buf else r_knee_ang
        gait_symmetry = (
            min(lk_mean, rk_mean) / (max(lk_mean, rk_mean) + 1e-9)
            if max(lk_mean, rk_mean) > 0 else 1.0
        )
        gait_symmetry = float(np.clip(gait_symmetry, 0.0, 1.0))

        # ── 5. STRIDE / CADENCE ──────────────────────────────────────
        l_ank_x = lm[_L_ANKLE].x
        r_ank_x = lm[_R_ANKLE].x
        stride_len = float(abs(l_ank_x - r_ank_x))

        # Simple peak-based step detection for cadence
        if self._prev_l_ankle_x is not None:
            l_delta = abs(l_ank_x - self._prev_l_ankle_x)
            if l_delta > 0.015:   # threshold for step event
                t = self._frame_count / self.fps
                self._stride_event_ts.append(t)
                if len(self._stride_event_ts) > 20:
                    self._stride_event_ts.pop(0)

        self._prev_l_ankle_x = l_ank_x
        self._prev_r_ankle_x = r_ank_x
        self._stride_buf.append(stride_len)

        stride_arr = np.array(self._stride_buf)
        stride_cv  = (float(np.std(stride_arr) / (np.mean(stride_arr) + 1e-9))
                      if len(stride_arr) > 5 else 0.0)
        stride_length_norm = float(np.mean(stride_arr)) if stride_arr.size else 0.0

        cadence_norm = 0.0
        if len(self._stride_event_ts) >= 3:
            intervals = np.diff(self._stride_event_ts[-10:])
            cadence_norm = float(1.0 / (np.mean(intervals) + 1e-9))

        step_width_norm = abs(lm[_L_ANKLE].x - lm[_R_ANKLE].x)

        # ── 6. HEEL-TOE DIFF ─────────────────────────────────────────
        heel_toe_diff_l = float(lm[_L_TOE].y - lm[_L_HEEL].y)
        heel_toe_diff_r = float(lm[_R_TOE].y - lm[_R_HEEL].y)

        # ── 7. LEG LENGTH RATIO ──────────────────────────────────────
        def dist_2d(a, b):
            return float(np.linalg.norm(p2(a) - p2(b)))

        l_leg = dist_2d(_L_HIP, _L_ANKLE) + 1e-9
        r_leg = dist_2d(_R_HIP, _R_ANKLE) + 1e-9
        leg_length_ratio = float(l_leg / r_leg)

        # ── 8. HIP HEIGHT & ASPECT RATIO ─────────────────────────────
        hip_height_norm = float(hp_mid[1])   # Y increases downward; high Y = fallen

        nose_y  = lm[_NOSE].y
        ankle_y = (lm[_L_ANKLE].y + lm[_R_ANKLE].y) / 2
        body_h  = abs(ankle_y - nose_y) + 1e-9
        body_w  = abs(lm[_L_SHOULDER].x - lm[_R_SHOULDER].x) + 1e-9
        aspect_ratio = float(body_h / body_w)

        # ── 9. VELOCITIES ─────────────────────────────────────────────
        def _vel(prev, cur, buf) -> float:
            if prev is None:
                buf.append(0.0)
                return 0.0
            v = float(np.linalg.norm(cur - prev)) * self.fps
            buf.append(v)
            return float(np.mean(buf))

        use_world = lm_world is not None
        head_pos = p3(_NOSE)    if use_world else np.array([lm[_NOSE].x, lm[_NOSE].y, 0.0])
        hip_pos  = (np.array([lm_world[_L_HIP].x + lm_world[_R_HIP].x,
                               lm_world[_L_HIP].y + lm_world[_R_HIP].y,
                               lm_world[_L_HIP].z + lm_world[_R_HIP].z]) / 2
                    if use_world
                    else np.array([hp_mid[0], hp_mid[1], 0.0]))
        l_ank = p3(_L_ANKLE) if use_world else np.array([lm[_L_ANKLE].x, lm[_L_ANKLE].y, 0.0])
        r_ank = p3(_R_ANKLE) if use_world else np.array([lm[_R_ANKLE].x, lm[_R_ANKLE].y, 0.0])

        head_vel   = _vel(self._prev_head_pos,   head_pos, self._head_vel_buf)
        hip_vel    = _vel(self._prev_hip_pos,    hip_pos,  self._hip_vel_buf)
        ankle_vel_l = _vel(self._prev_l_ankle,  l_ank,    self._l_ank_vel_buf)
        ankle_vel_r = _vel(self._prev_r_ankle,  r_ank,    self._r_ank_vel_buf)

        self._prev_head_pos  = head_pos.copy()
        self._prev_hip_pos   = hip_pos.copy()
        self._prev_l_ankle   = l_ank.copy()
        self._prev_r_ankle   = r_ank.copy()

        # ── 10. ASSEMBLE VECTOR ──────────────────────────────────────
        vec = np.array([
            sway_std, sway_range, sway_cv, sway_fft_peak, sway_fft_energy,
            torso_angle, torso_angle_std,
            l_knee_ang, r_knee_ang, knee_angle_diff,
            l_knee_var, r_knee_var,
            l_hip_ang, r_hip_ang,
            l_ankle_ang, r_ankle_ang,
            gait_symmetry, stride_cv, stride_length_norm,
            cadence_norm, step_width_norm,
            heel_toe_diff_l, heel_toe_diff_r,
            leg_length_ratio, hip_height_norm, aspect_ratio,
            head_vel, hip_vel, ankle_vel_l, ankle_vel_r,
        ], dtype=np.float32)

        return np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=0.0)

    # ------------------------------------------------------------------
    # FALL FEATURE EXTRACTION
    # ------------------------------------------------------------------

    def extract_fall_features(
        self,
        lm,
        lm_world,
        image_w: int,
        image_h: int,
    ) -> np.ndarray:
        """
        Returns feature vector of length len(FALL_FEATURE_NAMES).
        """
        use_world = lm_world is not None

        # Hip height world (metres above ground; approximate: negate world Y)
        if use_world:
            hip_y_world = -((lm_world[_L_HIP].y + lm_world[_R_HIP].y) / 2.0)
        else:
            # Fallback: use normalised Y flipped (1 - y gives approx height)
            hip_y_world = 1.0 - ((lm[_L_HIP].y + lm[_R_HIP].y) / 2.0)

        # Torso angle (recompute here independently)
        l_sh_y = (lm[_L_SHOULDER].y + lm[_R_SHOULDER].y) / 2
        l_hp_y = (lm[_L_HIP].y     + lm[_R_HIP].y)      / 2
        l_sh_x = (lm[_L_SHOULDER].x + lm[_R_SHOULDER].x) / 2
        l_hp_x = (lm[_L_HIP].x     + lm[_R_HIP].x)      / 2
        ta = math.degrees(math.atan2(abs(l_hp_x - l_sh_x), abs(l_hp_y - l_sh_y) + 1e-9))

        # Aspect ratio
        nose_y   = lm[_NOSE].y
        ankle_y  = (lm[_L_ANKLE].y + lm[_R_ANKLE].y) / 2
        bh = abs(ankle_y - nose_y) + 1e-9
        bw = abs(lm[_L_SHOULDER].x - lm[_R_SHOULDER].x) + 1e-9
        ar = float(bh / bw)

        # Velocities
        hv  = float(np.mean(self._head_vel_buf))  if self._head_vel_buf  else 0.0
        hipv = float(np.mean(self._hip_vel_buf))  if self._hip_vel_buf   else 0.0

        # Shoulder-hip vertical distance (collapses < 0.12 → fall)
        sh_hip_dist = abs(l_sh_y - l_hp_y)

        # Body height ratio (normalised body height / standing height estimate)
        body_height_ratio = float(bh)  # normalised height in frame

        vec = np.array([
            hip_y_world, ta, ar, hv, hipv, sh_hip_dist, body_height_ratio,
        ], dtype=np.float32)

        return np.nan_to_num(vec, nan=0.0, posinf=3.0, neginf=0.0)

    # ------------------------------------------------------------------
    # SEQUENCE (last N frames as 2-D array for LSTM)
    # ------------------------------------------------------------------

    def get_sequence_buffer(self) -> Optional[np.ndarray]:
        """
        Called externally to retrieve the most recent seq_len feature vectors
        stored by an MLAnalyzer that appends to a separate deque.
        Returns None if buffer not full.  (Buffer is managed by MLAnalyzer.)
        """
        return None  # MLAnalyzer manages its own buffer

    # ------------------------------------------------------------------
    # RESET (per-patient calibration)
    # ------------------------------------------------------------------

    def reset(self):
        for buf in (
            self._sway_buf, self._torso_buf, self._stride_buf,
            self._l_knee_buf, self._r_knee_buf,
            self._head_vel_buf, self._hip_vel_buf,
            self._l_ank_vel_buf, self._r_ank_vel_buf,
        ):
            buf.clear()
        self._prev_head_pos = self._prev_hip_pos = None
        self._prev_l_ankle  = self._prev_r_ankle = None
        self._prev_l_ankle_x = self._prev_r_ankle_x = None
        self._stride_event_ts.clear()
        self._frame_count = 0
