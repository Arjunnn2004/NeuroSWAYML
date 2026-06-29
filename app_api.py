import sys
import os
import cv2
import json
import time
import argparse
import threading
from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from core.pose_engine import ThreadedPoseEngine
from core.ml_analyzer import MLAnalyzer

app = Flask(__name__)
CORS(app)

CONFIG_PATH = os.path.join(_HERE, "config_ml.json")
MODELS_DIR  = os.path.join(_HERE, "saved_models")

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

cfg = load_config()
cfg["inference"]["use_threading"] = False

engine = ThreadedPoseEngine(cfg)
analyzer = MLAnalyzer(cfg)
analyzer.load_models()
analyzer.set_domain("elderly")

# Global state
stream_thread = None
is_streaming = False
last_frame_bytes = None
global_lock = threading.Lock()
analyzer_status = {"running": False}
STREAM_FRAME_INTERVAL = 0.05
STREAM_JPEG_QUALITY = 76
STREAM_MAX_WIDTH = 960

def encode_stream_frame(frame):
    height, width = frame.shape[:2]
    if width > STREAM_MAX_WIDTH:
        scale = STREAM_MAX_WIDTH / width
        frame = cv2.resize(
            frame,
            (STREAM_MAX_WIDTH, int(height * scale)),
            interpolation=cv2.INTER_AREA
        )
    return cv2.imencode(
        '.jpg',
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]
    )

def process_stream(source=0):
    global is_streaming, last_frame_bytes, analyzer_status
    engine.start(source=source)
    is_streaming = True
    
    last_frame_local = None

    while is_streaming and engine.is_running:
        pose_res = engine.get_result()
        if pose_res is None:
            time.sleep(0.01)
            continue
        
        result = analyzer.process(pose_res)
        result["fps"] = engine.current_fps
        annotated = result.get("annotated_frame", pose_res.frame)
        
        with global_lock:
            analyzer_status = {
                "running": True,
                "fps": result["fps"],
                "domain": result.get("domain"),
                "risk_score": result.get("risk_score"),
                "risk_class": result.get("risk_class"),
                "fall_detected": result.get("fall_detected"),
                "fall_prob": result.get("fall_prob"),
                "calibrating": result.get("calibrating"),
                "sway_idx": result.get("sway_idx", 0.0),
                "leg_ratio": result.get("leg_ratio", 1.0),
                "threshold_issues": result.get("threshold_issues", []),
                "balance_tests": result.get("balance_tests", {})
            }

        ret, buffer = encode_stream_frame(annotated)
        if ret:
            with global_lock:
                last_frame_bytes = buffer.tobytes()

    engine.stop()
    analyzer.save_log()
    with global_lock:
        is_streaming = False
        analyzer_status["running"] = False

def generate_frames():
    global last_frame_bytes, is_streaming
    while True:
        if not is_streaming:
            time.sleep(0.1)
            continue
            
        with global_lock:
            frame = last_frame_bytes
            
        if frame is None:
            time.sleep(0.01)
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(STREAM_FRAME_INTERVAL)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    with global_lock:
        status = analyzer_status.copy()
        if not is_streaming:
            status["running"] = False
        return jsonify(status)

@app.route('/api/start')
def start_stream():
    global stream_thread, is_streaming
    if not is_streaming:
        stream_thread = threading.Thread(target=process_stream, args=(0,))
        stream_thread.daemon = True
        stream_thread.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.route('/api/stop')
def stop_stream():
    global is_streaming
    is_streaming = False
    return jsonify({"status": "stopping"})

@app.route('/api/calibrate')
def calibrate():
    analyzer.recalibrate()
    return jsonify({"status": "calibrating"})

@app.route('/api/logs')
def get_logs():
    logs_dir = os.path.join(_HERE, "ml_logs")
    if not os.path.exists(logs_dir):
        return jsonify([])
    
    logs = []
    for f in os.listdir(logs_dir):
        if f.endswith(".json"):
            try:
                with open(os.path.join(logs_dir, f)) as jf:
                    logs.append(json.load(jf))
            except json.JSONDecodeError:
                pass
    return jsonify(logs)

@app.route('/api/images')
def get_images():
    img_dir = os.path.join(_HERE, "ml_fall_detections")
    if not os.path.exists(img_dir):
        return jsonify([])
    
    images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    images.sort(reverse=True)
    return jsonify(images)

@app.route('/api/images/<filename>', methods=['GET'])
def serve_image(filename):
    img_path = os.path.join(_HERE, "ml_fall_detections", filename)
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg', max_age=3600)
    return "Not found", 404

@app.route('/api/images/<filename>', methods=['DELETE'])
def delete_image(filename):
    img_path = os.path.join(_HERE, "ml_fall_detections", filename)
    if os.path.exists(img_path) and ".." not in filename:
        try:
            os.remove(img_path)
            return jsonify({"status": "deleted"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return "Not found", 404

if __name__ == '__main__':
    print("[Flask] Starting NeuroSway API on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
