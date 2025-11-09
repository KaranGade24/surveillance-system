#!/usr/bin/env python3
# survillance_system_api.py
print("Starting imports...")
import cv2
import os
import json
import time
import socket
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import signal
import sys
from flask_socketio import SocketIO

# Try importing Picamera2 (for Raspberry Pi)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
    print("✅ Picamera2 found! Using Raspberry Pi camera.")
except ImportError:
    print("⚠️ Picamera2 not found. Falling back to OpenCV camera.")
    PICAMERA_AVAILABLE = False

print("Imports completed.")
print("Loading YOLO fire detection model...")
fire_model = YOLO("fire_detector.pt")
print("🔥 YOLO model loaded successfully!")

# Flask + SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Globals
frame_lock = Lock()
global_frame = None
detections_data = []
video_writer = None
current_video_path = None
current_folder = None
frame_width, frame_height = 320, 240
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
    base_dir = os.path.expanduser("~/YOLO_Recordings")
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    date_path = os.path.join(base_dir, date_folder)
    os.makedirs(date_path, exist_ok=True)

    hour = int(now.strftime("%H"))
    hour_folder = f"{hour:02d}_00-{(hour+1)%24:02d}_00"
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
    global detections_data
    detections_data.append(data)
    try:
        log_path = os.path.join(current_folder, "detections.json")
        with open(log_path, "a") as f:
            json.dump(data, f)
            f.write("\n")
    except Exception as e:
        print(f"[Error] Failed to write detection JSON: {e}")

def log_detection(label, conf):
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
                full_path = os.path.join(base_dir, folder)
                print(f"🧹 Cleaning old folder: {full_path}")
                shutil.rmtree(full_path)
        except Exception:
            continue

def handle_exit(sig, frame):
    global cap, video_writer
    print("\n[INFO] Program interrupted — cleaning up...")
    try:
        if video_writer:
            video_writer.release()
            print("[INFO] Video writer released.")
        if cap and hasattr(cap, "isOpened") and cap.isOpened():
            cap.release()
            print("[INFO] Camera released.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
    sys.exit(0)

# -----------------------------
# Drawing boxes
# -----------------------------
def draw_detection_boxes(img_bgr, boxes_xyxy, confs, conf_threshold=0.25):
    for i, box in enumerate(boxes_xyxy):
        conf = float(confs[i])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if conf >= 0.5 else (0, 255, 255)
        label = f"Fire {conf:.2f}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - 5), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img_bgr

# -----------------------------
# Camera capture (merged logic)
# -----------------------------
def camera_capture():
    global global_frame, video_writer, cap
    init_video_writer()
    last_hour = datetime.now().hour

    if PICAMERA_AVAILABLE:
        print("📸 Using PiCamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (frame_width, frame_height)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        while True:
            frame = picam2.capture_array()
            if frame is None:
                continue
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            annotated = process_frame(frame, last_hour)
            with frame_lock:
                global_frame = annotated
            time.sleep(0.05)
    else:
        print("🎦 Using USB/OpenCV camera...")
        cap = cv2.VideoCapture(0)
        cap.set(3, frame_width)
        cap.set(4, frame_height)
        if not cap.isOpened():
            print("❌ Error: Cannot open camera.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            annotated = process_frame(frame, last_hour)
            with frame_lock:
                global_frame = annotated
            time.sleep(0.05)

# -----------------------------
# Frame processing
# -----------------------------
def process_frame(frame, last_hour):
    global video_writer
    now = datetime.now()
    if now.hour != last_hour:
        if video_writer:
            video_writer.release()
        init_video_writer()
        last_hour = now.hour

    results = fire_model.predict(frame, verbose=False)
    annotated = frame.copy()

    for r in results:
        if not hasattr(r, "boxes") or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        annotated = draw_detection_boxes(annotated, boxes, confs)
        for conf_val in confs:
            if conf_val >= 0.5:
                log_detection("Fire", float(conf_val))

    if video_writer:
        video_writer.write(cv2.resize(annotated, (frame_width, frame_height)))
    return annotated

# -----------------------------
# Flask Streaming
# -----------------------------
@app.route("/video_feed")
def video_feed():
    def generate_frames():
        global global_frame
        while True:
            with frame_lock:
                if global_frame is None:
                    continue
                ret, buffer = cv2.imencode(".jpg", global_frame)
                frame_bytes = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def get_detections():
    return jsonify(list(reversed(detections_data[-100:])))

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
    print(f"✅ Flask server running at: http://{ip}:5000")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        handle_exit(None, None)
