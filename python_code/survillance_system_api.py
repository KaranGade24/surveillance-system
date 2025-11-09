print("Starting imports...")
import cv2
import os
import json
import time
import socket
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
import signal
import sys
from flask_socketio import SocketIO
import numpy as np

# YOLO import
from ultralytics import YOLO

# Try importing Picamera2 (for Raspberry Pi)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
    print("✅ Picamera2 found! Using Raspberry Pi camera.")
except ImportError:
    print("⚠️ Picamera2 not found. Falling back to OpenCV camera.")
    PICAMERA_AVAILABLE = False

print("Loading YOLO models...")
fire_model = YOLO("fire_detector.pt")
print("✅ YOLO models loaded!")

# -----------------------------
# Flask + SocketIO
# -----------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# Global variables
# -----------------------------
frame_lock = Lock()
global_frame = None
detections_data = []
video_writer = None
current_video_path = None
current_folder = None
frame_width, frame_height = 640, 480  # adjust for performance
cap = None

# -----------------------------
# Helpers
# -----------------------------
def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_output_path():
    """Create folder structure: ~/YOLO_Recordings/YYYY-MM-DD/HH_00-HH_00"""
    base_dir = os.path.expanduser("~/YOLO_Recordings")
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    date_path = os.path.join(base_dir, date_folder)
    os.makedirs(date_path, exist_ok=True)
    hour_folder = f"{now.hour:02d}_00-{(now.hour+1)%24:02d}_00"
    hour_path = os.path.join(date_path, hour_folder)
    os.makedirs(hour_path, exist_ok=True)
    return hour_path

def init_video_writer():
    global video_writer, current_video_path, current_folder
    current_folder = get_output_path()
    filename = f"record_{datetime.now().strftime('%H-%M-%S')}.mp4"
    current_video_path = os.path.join(current_folder, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(current_video_path, fourcc, 10.0, (frame_width, frame_height))
    print(f"🎥 Recording video to: {current_video_path}")

def save_detection_json(data):
    """Append detection data to memory & JSON"""
    global detections_data
    detections_data.append(data)
    log_path = os.path.join(current_folder, "detections.json")
    with open(log_path, "a") as f:
        json.dump(data, f)
        f.write("\n")

def log_detection(label, conf):
    """Emit detection to frontend and save"""
    now = datetime.now()
    data = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch_time": now.timestamp(),
        "object": label,
        "confidence": round(conf, 2),
        "video_file": current_video_path
    }
    socketio.emit("new_detection", data)
    save_detection_json(data)

def cleanup_old(days=7):
    base_dir = os.path.expanduser("~/YOLO_Recordings")
    if not os.path.exists(base_dir):
        return
    now = datetime.now()
    for folder in os.listdir(base_dir):
        try:
            folder_date = datetime.strptime(folder, "%Y-%m-%d")
            if (now - folder_date).days > days:
                import shutil
                shutil.rmtree(os.path.join(base_dir, folder))
        except Exception:
            continue

def handle_exit(sig, frame):
    global cap, video_writer
    print("\n[INFO] Exiting, cleaning up...")
    try:
        if video_writer:
            video_writer.release()
        if cap and hasattr(cap, "isOpened") and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
    sys.exit(0)

# -----------------------------
# Frame processing
# -----------------------------
def process_frame(frame, last_hour):
    global video_writer
    now = datetime.now()
    # New hour => new video file
    if now.hour != last_hour:
        if video_writer:
            video_writer.release()
        init_video_writer()
        last_hour = now.hour

    annotated = frame.copy()

    # Fire detection using YOLO
    results = fire_model.predict(frame, verbose=False, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = "Fire"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if conf > 0.5:
                log_detection(label, conf)

    # Write to video
    if video_writer:
        video_writer.write(cv2.resize(annotated, (frame_width, frame_height)))

    return annotated, last_hour

# -----------------------------
# Camera capture thread
# -----------------------------
def camera_capture():
    global global_frame, cap
    init_video_writer()
    last_hour = datetime.now().hour

    if PICAMERA_AVAILABLE:
        print("📸 Using PiCamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (frame_width, frame_height), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)
        while True:
            frame = picam2.capture_array()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            annotated, last_hour = process_frame(frame, last_hour)
            with frame_lock:
                global_frame = annotated.copy()
            time.sleep(0.05)
    else:
        print("🎦 Using USB/OpenCV camera...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            annotated, last_hour = process_frame(frame, last_hour)
            with frame_lock:
                global_frame = annotated.copy()
            time.sleep(0.05)

# -----------------------------
# Flask endpoints
# -----------------------------
@app.route("/video_feed")
def video_feed():
    def generate_frames():
        global global_frame
        while True:
            with frame_lock:
                if global_frame is None:
                    time.sleep(0.1)
                    continue
                ret, buffer = cv2.imencode(".jpg", global_frame)
                frame_bytes = buffer.tobytes()
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.05)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def get_detections():
    return jsonify(list(reversed(detections_data[-100:])))

@app.route("/recordings")
def list_recordings():
    base_dir = os.path.expanduser("~/YOLO_Recordings")
    tree = []
    if not os.path.exists(base_dir):
        return jsonify(tree)
    for date in sorted(os.listdir(base_dir), reverse=True):
        date_path = os.path.join(base_dir, date)
        if not os.path.isdir(date_path):
            continue
        hours = []
        for hour in sorted(os.listdir(date_path), reverse=True):
            hour_path = os.path.join(date_path, hour)
            if not os.path.isdir(hour_path):
                continue
            files = []
            for f in sorted(os.listdir(hour_path), reverse=True):
                if f.lower().endswith(".mp4"):
                    full = os.path.join(hour_path, f)
                    rel_path = os.path.relpath(full, base_dir).replace("\\", "/")
                    files.append({"name": f, "path": rel_path, "url": f"/recording/{rel_path}"})
            hours.append({"hour": hour, "files": files})
        tree.append({"date": date, "hours": hours})
    return jsonify(tree)

@app.route("/recording/<path:filename>")
def static_recordings(filename):
    base_dir = os.path.expanduser("~/YOLO_Recordings")
    full_path = os.path.join(base_dir, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": "File not found"}), 404
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(full_path, mimetype="video/mp4")
    size = os.path.getsize(full_path)
    byte1, byte2 = 0, None
    m = range_header.replace('bytes=', '').split('-')
    if len(m) == 2:
        if m[0]:
            byte1 = int(m[0])
        if m[1]:
            byte2 = int(m[1])
    length = size - byte1 if byte2 is None else byte2 - byte1 + 1
    with open(full_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)
    resp = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
    resp.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    resp.headers.add('Accept-Ranges', 'bytes')
    return resp

# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    cleanup_old(7)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    cam_thread = Thread(target=camera_capture, daemon=True)
    cam_thread.start()
    ip = get_ip()
    print(f"✅ Flask+SocketIO running at: http://{ip}:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
