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
import signal
import sys
from flask_socketio import SocketIO
import numpy as np

# Try importing Picamera2 (for Raspberry Pi)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
    print("✅ Picamera2 found! Using Raspberry Pi camera.")
except ImportError:
    print("⚠️ Picamera2 not found. Falling back to OpenCV camera.")
    PICAMERA_AVAILABLE = False

print("Imports completed.")

# Flask + SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Globals
frame_lock = Lock()
global_frame = None
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
    base_dir = os.path.expanduser("~/Recordings")
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
    # Ensure dimensions are correct before initializing
    if frame_width == 0 or frame_height == 0:
        print("[Error] Frame dimensions are zero, cannot initialize VideoWriter.")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(current_video_path, fourcc, 10.0, (frame_width, frame_height))
    print(f"🎥 Recording video to: {current_video_path}")

def cleanup_old(days=7):
    base_dir = os.path.expanduser("~/Recordings")
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
        # Check if cap is Picamera2 object (no isOpened) or OpenCV (has isOpened)
        if cap and hasattr(cap, "isOpened") and cap.isOpened():
            cap.release()
            print("[INFO] Camera released.")
        elif PICAMERA_AVAILABLE:
            pass
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
    
    # Handle hour change
    if now.hour != last_hour:
        if video_writer:
            video_writer.release()
        init_video_writer()
        last_hour = now.hour

    annotated = frame.copy()

    if video_writer:
        try:
            video_writer.write(cv2.resize(annotated, (frame_width, frame_height)))
        except cv2.error as e:
            print(f"[Error] Failed to write frame to video: {e}")
            
    return annotated, last_hour

# -----------------------------
# Camera capture
# -----------------------------
def camera_capture():
    global global_frame, video_writer, cap
    init_video_writer()
    last_hour = datetime.now().hour

    if PICAMERA_AVAILABLE:
        print("📸 Using PiCamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (frame_width, frame_height), "format": "XBGR8888"})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        while True:
            frame = picam2.capture_array()
            if frame is None:
                time.sleep(0.05)
                continue
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
            print("❌ Error: Cannot open camera.")
            return
            
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            annotated, last_hour = process_frame(frame, last_hour)
            with frame_lock:
                global_frame = annotated.copy()
            time.sleep(0.05)

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
                    time.sleep(0.1)
                    continue
                if not isinstance(global_frame, np.ndarray) or global_frame.size == 0:
                    time.sleep(0.1)
                    continue
                ret, buffer = cv2.imencode(".jpg", global_frame)
                frame_bytes = buffer.tobytes()
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.05)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
        socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        handle_exit(None, None)
