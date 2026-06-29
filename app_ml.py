"""
NeuroSWAYML — Elderly Fall Risk App
====================================
Real-time fall risk detection using MediaPipe + ML.

Usage:
    python app_ml.py
    python app_ml.py --source 0          # camera index
    python app_ml.py --source video.mp4  # video file
    python app_ml.py --no-thread         # disable inference threading

Keyboard shortcuts:
    Q  — quit
    R  — recalibrate personal baseline
    D  — toggle debug panel
    S  — save session log
"""

import sys
import os
import cv2
import json
import time
import argparse

# ── Path setup ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from core.pose_engine import ThreadedPoseEngine
from core.ml_analyzer import MLAnalyzer


CONFIG_PATH = os.path.join(_HERE, "config_ml.json")
MODELS_DIR  = os.path.join(_HERE, "saved_models")


# ───────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def models_exist() -> bool:
    return os.path.exists(os.path.join(MODELS_DIR, "elderly", "gait_classifier.pkl"))


# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NeuroSWAYML — Elderly Fall Risk")
    parser.add_argument("--source",    type=str, default="0",
                        help="Camera index or video file path")
    parser.add_argument("--no-thread", action="store_true",
                        help="Disable inference threading")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    cfg = load_config()
    if args.no_thread:
        cfg["inference"]["use_threading"] = False

    if not models_exist():
        print("[ERROR] No trained elderly models found.")
        print("  Run:  python training/train_elderly.py")
        sys.exit(1)

    print("\n[NeuroSWAYML] Initialising Elderly Fall Risk analyser…")
    engine   = ThreadedPoseEngine(cfg)
    analyzer = MLAnalyzer(cfg)
    analyzer.load_models()
    analyzer.set_domain("elderly")

    window_title = "NeuroSWAYML — Elderly Fall Risk"
    print(f"\n[NeuroSWAYML] Starting live session (source={source})")
    print("  Keys:  Q=quit   R=recalibrate   D=debug   S=save log")
    engine.start(source=source if isinstance(source, int) else 0)

    last_frame = None

    while engine.is_running:
        pose_res = engine.get_result()

        if pose_res is None:
            if last_frame is not None:
                cv2.imshow(window_title, last_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            time.sleep(0.001)
            continue

        # ── ML inference ──────────────────────────────────────────────
        result = analyzer.process(pose_res)
        result["fps"] = engine.current_fps

        annotated = result.get("annotated_frame", pose_res.frame)
        last_frame = annotated

        # ── Display ───────────────────────────────────────────────────
        cv2.imshow(window_title, annotated)

        # ── Keyboard handler ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        elif key == ord("r"):
            analyzer.recalibrate()
            print("[NeuroSWAYML] Personal recalibration started…")

        elif key == ord("d"):
            cfg["visualization"]["show_feature_debug"] = \
                not cfg["visualization"]["show_feature_debug"]
            state = "ON" if cfg["visualization"]["show_feature_debug"] else "OFF"
            print(f"[NeuroSWAYML] Debug panel: {state}")

        elif key == ord("s"):
            analyzer.save_log()
            print("[NeuroSWAYML] Session log saved.")

    # ── Cleanup ───────────────────────────────────────────────────────
    print("\n[NeuroSWAYML] Shutting down…")
    engine.stop()
    analyzer.save_log()
    cv2.destroyAllWindows()
    print("[NeuroSWAYML] Session ended.")


if __name__ == "__main__":
    main()
