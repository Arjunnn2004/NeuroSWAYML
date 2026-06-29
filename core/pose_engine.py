"""
NeuroSWAYML - Threaded Pose Engine
Runs MediaPipe inference in a background thread so the main loop
never blocks on GPU/CPU pose estimation.

Key improvements over original app.py:
  - model_complexity=0  (Lite model → ~2× faster)
  - pose_world_landmarks enabled (true 3D metric coords)
  - Separate inference thread (producer) & main thread (consumer)
  - Frame dropping on queue full (camera never falls behind)
"""

import cv2
import threading
import queue
import time
import sys
import urllib.request
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, NamedTuple

_MP_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_MP_MODEL_NAME = "pose_landmarker_lite.task"


class PoseResult(NamedTuple):
    frame:          np.ndarray          # original BGR frame
    landmarks_2d:   Optional[object]    # pose_landmarks  (normalised)
    landmarks_3d:   Optional[object]    # pose_world_landmarks (metres)
    timestamp:      float               # time.time() when frame was captured
    frame_idx:      int


# ───────────────────────────────────────────────────────────────────────────

class ThreadedPoseEngine:
    """
    Start with:   engine.start(source)
    Read with:    result = engine.get_result()   (non-blocking, may return None)
    Stop with:    engine.stop()
    """

    def __init__(self, config: dict):
        cfg = config["mediapipe"]
        inf = config["inference"]

        self.model_complexity        = cfg["model_complexity"]
        self.min_detect_confidence   = cfg["min_detection_confidence"]
        self.min_track_confidence    = cfg["min_tracking_confidence"]
        self.use_world               = cfg["use_world_landmarks"]
        self.use_threading           = inf["use_threading"]
        self.q_size                  = inf["frame_queue_size"]

        # State
        self._cap:     Optional[cv2.VideoCapture] = None
        self._thread:  Optional[threading.Thread] = None
        self._result_q: queue.Queue = queue.Queue(maxsize=self.q_size)
        self._running  = False
        self._frame_idx = 0

        # MediaPipe pose (initialised in start() so it lives in the thread)
        self._pose: Optional[object] = None
        self._start_time: float      = 0.0

        # FPS tracking
        self.current_fps   = 0.0
        self._fps_counter  = 0
        self._fps_t0       = time.time()

    # ------------------------------------------------------------------
    def start(self, source: int = 0, width: int = 1280, height: int = 720):
        """Open camera and start inference thread."""
        self._cap = self._open_camera(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise camera latency

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  [PoseEngine] Camera opened: {actual_w}×{actual_h}")

        self._running    = True
        self._start_time = time.time()

        if self.use_threading:
            self._thread = threading.Thread(
                target=self._inference_loop,
                daemon=True,
                name="PoseInferenceThread",
            )
            self._thread.start()
            print("  [PoseEngine] Inference thread started")
        else:
            # Single-thread mode: initialise pose here
            self._pose = self._make_landmarker()
            print("  [PoseEngine] Single-thread mode")

    # ------------------------------------------------------------------
    def get_result(self) -> Optional[PoseResult]:
        """
        Non-blocking fetch of the latest pose result.
        Returns None if nothing is ready yet.
        """
        if self.use_threading:
            try:
                return self._result_q.get_nowait()
            except queue.Empty:
                return None
        else:
            return self._inference_step()

    # ------------------------------------------------------------------
    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        if self._pose:
            self._pose.close()
        print("  [PoseEngine] Stopped")

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _inference_loop(self):
        """Background thread: grab frames → run mediapipe → push to queue."""
        pose = self._make_landmarker()
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self._running = False
                break

            result = self._run_mediapipe(pose, frame)

            # Drop oldest if queue full (real-time priority)
            if self._result_q.full():
                try:
                    self._result_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._result_q.put_nowait(result)
            except queue.Full:
                pass

            # FPS update
            self._fps_counter += 1
            elapsed = time.time() - self._fps_t0
            if elapsed >= 1.0:
                self.current_fps  = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_t0      = time.time()

        pose.close()

    def _inference_step(self) -> Optional[PoseResult]:
        """Single-thread: read one frame and run mediapipe."""
        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            self._running = False
            return None
        result = self._run_mediapipe(self._pose, frame)

        self._fps_counter += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self.current_fps  = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_t0      = time.time()
        return result

    def _run_mediapipe(self, landmarker, frame: np.ndarray) -> PoseResult:
        self._frame_idx += 1
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int((time.time() - self._start_time) * 1000)
        result   = landmarker.detect_for_video(mp_img, ts_ms)

        lm_2d = result.pose_landmarks[0]      if result.pose_landmarks      else None
        lm_3d = (result.pose_world_landmarks[0]
                 if self.use_world and result.pose_world_landmarks else None)

        return PoseResult(
            frame=frame,
            landmarks_2d=lm_2d,
            landmarks_3d=lm_3d,
            timestamp=time.time(),
            frame_idx=self._frame_idx,
        )

    def _ensure_model(self) -> str:
        """Download pose_landmarker_lite.task to saved_models/ if missing."""
        model_dir  = Path(__file__).parent.parent / "saved_models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / _MP_MODEL_NAME
        if not model_path.exists():
            print(f"  [PoseEngine] Downloading pose model (~3 MB)…", end="", flush=True)
            urllib.request.urlretrieve(_MP_MODEL_URL, str(model_path))
            print(" done.")
        return str(model_path)

    def _make_landmarker(self):
        """Create a Tasks-API PoseLandmarker in VIDEO mode (stateful tracking)."""
        from mediapipe.tasks import python as mp_tp
        from mediapipe.tasks.python import vision as mp_vision

        base_opts = mp_tp.BaseOptions(model_asset_path=self._ensure_model())
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_detect_confidence,
            min_tracking_confidence=self.min_track_confidence,
            output_segmentation_masks=False,
        )
        return mp_vision.PoseLandmarker.create_from_options(opts)

    @staticmethod
    def _open_camera(source: int) -> cv2.VideoCapture:
        if sys.platform == "darwin":
            cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")
        return cap
