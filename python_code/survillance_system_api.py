print("Starting imports...")
import cv2
import os
import json
import time
import socket
import numpy as np
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO
import signal
import sys

print("✅ Imports completed.")

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH, FRAME_HEIGHT = 320, 240      # For streaming
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480     # For video recording
FPS = 10
DETECT_EVERY_N_FRAMES = 3                # YOLO inference every N frames
BASE_RECORD_DIR = os.path.expanduser("~/Recordings")

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
video_writer = None
current_video_path = None
current_folder = None
detections_data = []
cap = None

# -----------------------------
# Load YOLO model
# -----------------------------
print("Loading YOLO fire detection model...")
fire_model = YOLO("fire_detector.pt")
print("✅ YOLO model loaded successfully.")

# -----------------------------
# Helper Functions
# -----------------------------
def get_ip():
    """Get local IP of Raspberry Pi / system."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_output_path():
    """Return path for current date/hour recordings."""
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    date_path = os.path.join(BASE_RECORD_DIR, date_folder)
    os.makedirs(date_path, exist_ok=True)

    hour = now.hour
    hour_folder = f"{hour:02d}_00-{(hour+1)%24:02d}_00"
    hour_path = os.path.join(date_path, hour_folder)
    os.makedirs(hour_path, exist_ok=True)
    return hour_path

def init_video_writer():
    """Initialize high-res video writer."""
    global video_writer, current_video_path, current_folder
    current_folder = get_output_path()
    filename = f"record_{datetime.now().strftime('%H-%M-%S')}.mp4"
    current_video_path = os.path.join(current_folder, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(current_video_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    print(f"🎥 Recording video to: {current_video_path}")

def save_detection(label, conf):
    """Save detection in memory, JSON, and emit via SocketIO."""
    global detections_data
    ts = datetime.now()
    data = {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch_time": ts.timestamp(),
        "object": label,
        "confidence": round(conf, 2),
        "video_file": current_video_path
    }
    detections_data.append(data)
    socketio.emit("new_detection", data)
    log_path = os.path.join(current_folder, "detections.json")
    with open(log_path, "a") as f:
        json.dump(data, f)
        f.write("\n")
    print(f"🔔 Detection: {label} {conf:.2f} at {ts.strftime('%H:%M:%S')}")

def handle_exit(sig, frame):
    """Cleanup on exit."""
    global cap, video_writer
    print("\n[INFO] Exiting... Cleaning up...")
    if video_writer: video_writer.release()
    if cap: cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# -----------------------------
# Camera Capture + Detection
# -----------------------------
def camera_capture():
    global global_frame, video_writer, cap
    print("🎬 Starting camera capture...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

    init_video_writer()
    last_hour = datetime.now().hour
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        stream_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        annotated = frame.copy()

        # -------------------
        # Fire detection every N frames
        # -------------------
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            results = fire_model.predict(frame, stream=True, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = "Fire"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    if conf > 0.5:
                        save_detection(label, conf)

        # -------------------
        # Write video frame
        # -------------------
        if video_writer:
            try:
                video_writer.write(cv2.resize(annotated, (VIDEO_WIDTH, VIDEO_HEIGHT)))
            except cv2.error:
                pass

        # -------------------
        # Update global frame for streaming
        # -------------------
        with frame_lock:
            global_frame = stream_frame.copy()

        # Hourly video rotation
        now = datetime.now()
        if now.hour != last_hour:
            video_writer.release()
            init_video_writer()
            last_hour = now.hour

# -----------------------------
# Flask Streaming Endpoint
# -----------------------------
@app.route("/video_feed")
def video_feed():
    ip = get_ip()
    print(f"🌐 Access live stream at: http://{ip}:5000/video_feed")
    def generate_frames():
        global global_frame
        while True:
            with frame_lock:
                if global_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', global_frame)
                frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------------
# Detections Endpoint
# -----------------------------
@app.route("/detections")
def get_detections():
    return jsonify(list(reversed(detections_data[-100:])))

# -----------------------------
# Recordings List Endpoint
# -----------------------------
@app.route("/recordings")
def list_recordings():
    tree = []
    if not os.path.exists(BASE_RECORD_DIR):
        return jsonify(tree)

    for date in sorted(os.listdir(BASE_RECORD_DIR), reverse=True):
        date_path = os.path.join(BASE_RECORD_DIR, date)
        if not os.path.isdir(date_path): continue

        hours = []
        for hour in sorted(os.listdir(date_path), reverse=True):
            hour_path = os.path.join(date_path, hour)
            if not os.path.isdir(hour_path): continue
            files = []
            for f in sorted(os.listdir(hour_path), reverse=True):
                if f.lower().endswith(".mp4"):
                    rel_path = os.path.relpath(os.path.join(hour_path,f), BASE_RECORD_DIR).replace("\\","/")
                    files.append({"name":f, "path":rel_path, "url": f"/recording/{rel_path}"})
            hours.append({"hour": hour, "files": files})
        tree.append({"date": date, "hours": hours})
    return jsonify(tree)

# -----------------------------
# Serve Recording Files (with Range support)
# -----------------------------
@app.route("/recording/<path:filename>")
def serve_recording(filename):
    full_path = os.path.join(BASE_RECORD_DIR, filename)
    if not os.path.exists(full_path):
        return jsonify({"error":"File not found"}), 404

    range_header = request.headers.get('Range', None)
    size = os.path.getsize(full_path)
    byte1, byte2 = 0, None
    if range_header:
        m = range_header.replace('bytes=', '').split('-')
        byte1 = int(m[0]) if m[0] else 0
        byte2 = int(m[1]) if len(m) > 1 and m[1] else size-1
    length = byte2-byte1+1 if byte2 else size-byte1
    with open(full_path,'rb') as f:
        f.seek(byte1)
        data = f.read(length)
    rv = Response(data, 206 if range_header else 200, mimetype="video/mp4", direct_passthrough=True)
    rv.headers.add('Accept-Ranges','bytes')
    rv.headers.add('Content-Range',f'bytes {byte1}-{byte1+length-1}/{size}')
    return rv

# -----------------------------
# Main Entry
# -----------------------------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    cam_thread = Thread(target=camera_capture, daemon=True)
    cam_thread.start()

    ip = get_ip()
    print(f"✅ Server running. Access live feed at: http://{ip}:5000/video_feed")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
print("✅ Surveillance System API started.")