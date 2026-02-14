# START_CLOUDFLARE=true PORT=5000 python3 python_code/survillance_system_api.py

print("Starting imports...")
import cv2
import os
import json
import time
import socket
from datetime import datetime
from threading import Thread, Lock, Event
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
import signal
import sys
from flask_socketio import SocketIO
import numpy as np
import cloudflare

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
frame_width, frame_height = 1280, 720  # Increased frame size
cap = None
# Controls
recording_enabled = True
camera_stop_event = Event()
recordings_base = os.path.expanduser("~/Recordings")
camera_thread = None

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


def start_recording():
    global recording_enabled, video_writer
    if recording_enabled and video_writer is not None:
        return
    recording_enabled = True
    init_video_writer()


def stop_recording():
    global recording_enabled, video_writer
    recording_enabled = False
    if video_writer:
        try:
            video_writer.release()
        except Exception:
            pass
        video_writer = None

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

    if recording_enabled and video_writer:
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
        config = picam2.create_preview_configuration(main={"size": (frame_width, frame_height), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        time.sleep(2)

        while not camera_stop_event.is_set():
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
            
        while not camera_stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            annotated, last_hour = process_frame(frame, last_hour)
            with frame_lock:
                global_frame = annotated.copy()
            time.sleep(0.05)
        # cleanup when stopping
        try:
            if cap and hasattr(cap, "isOpened") and cap.isOpened():
                cap.release()
        except Exception:
            pass

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
# Recordings endpoints
# -----------------------------
def _safe_recording_path(relative_path: str):
    if not relative_path:
        return None
    # Prevent path traversal
    candidate = os.path.normpath(os.path.join(recordings_base, relative_path))
    if not candidate.startswith(os.path.abspath(recordings_base)):
        return None
    return candidate


@app.route("/recordings")
def list_recordings():
    files = []
    base = recordings_base
    if not os.path.exists(base):
        return jsonify([])
    for root, _, filenames in os.walk(base):
        for f in filenames:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, base)
            files.append(rel.replace('\\\\', '/'))
    files.sort(reverse=True)
    return jsonify(files)


def stream_video_file(path):
    capv = cv2.VideoCapture(path)
    if not capv.isOpened():
        yield b""
        return
    try:
        while True:
            ret, frame = capv.read()
            if not ret or frame is None:
                break
            ret2, buffer = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.03)
    finally:
        try:
            capv.release()
        except Exception:
            pass


@app.route("/recordings/stream")
def recordings_stream():
    rel = request.args.get('file')
    safe = _safe_recording_path(rel)
    if not safe or not os.path.exists(safe):
        return jsonify({"error": "file not found"}), 404
    return Response(stream_video_file(safe), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/recordings/download")
def recordings_download():
    rel = request.args.get('file')
    safe = _safe_recording_path(rel)
    if not safe or not os.path.exists(safe):
        return jsonify({"error": "file not found"}), 404
    return send_file(safe, as_attachment=True)


# -----------------------------
# Control endpoints
# -----------------------------
@app.route("/control/start_recording", methods=["GET", "POST"])
def api_start_recording():
    start_recording()
    return jsonify({"status": "recording_started"})


@app.route("/control/stop_recording", methods=["GET", "POST"])
def api_stop_recording():
    stop_recording()
    return jsonify({"status": "recording_stopped"})


@app.route("/control/start_camera", methods=["GET", "POST"])
def api_start_camera():
    global camera_thread, camera_stop_event
    if camera_thread and camera_thread.is_alive():
        return jsonify({"status": "camera_already_running"})
    camera_stop_event.clear()
    camera_thread = Thread(target=camera_capture, daemon=True)
    camera_thread.start()
    return jsonify({"status": "camera_started"})


@app.route("/control/stop_camera", methods=["GET", "POST"])
def api_stop_camera():
    global camera_stop_event
    camera_stop_event.set()
    return jsonify({"status": "camera_stop_requested"})

# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    cleanup_old(7)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Read port and cloudflare settings from environment
    port = int(os.getenv("PORT", "5000"))
    start_cf = os.getenv("START_CLOUDFLARE", "false").lower() in ("1", "true", "yes")

    # Start camera thread
    camera_stop_event.clear()
    camera_thread = Thread(target=camera_capture, daemon=True)
    camera_thread.start()

    # Optionally start cloudflare tunnel in background
    if start_cf:
        try:
            cloudflare.start_cloudflare_background(port=port)
        except Exception as e:
            print(f"[WARN] Failed to start cloudflare background: {e}")

    ip = get_ip()
    print(f"✅ Flask server running at: http://{ip}:{port}")
    try:
        socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        handle_exit(None, None)
