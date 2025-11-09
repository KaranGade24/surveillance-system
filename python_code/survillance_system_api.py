print("Starting imports...")
import cv2
import os
import json
import time
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import signal
import sys
from flask_socketio import SocketIO
# from pycloudflared import open_tunnel

# ✅ Try importing Picamera2 (for Raspberry Pi)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
    print("✅ Picamera2 found! Using Raspberry Pi camera.")
except ImportError:
    print("⚠️ Picamera2 not found. Falling back to OpenCV camera.")
    PICAMERA_AVAILABLE = False

print("Imports completed.")
print("Loading YOLO model...")

# ✅ Load YOLOv8 fire detection model
fire_model = YOLO("fire_detector.pt")
print("🔥 YOLO model loaded successfully!")

# -----------------------------
# 🌐 Flask App Setup
# -----------------------------
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
frame_width, frame_height = 640, 480
cap = None


# -----------------------------
# 🗂️ Helper Functions
# -----------------------------
def get_output_path():
    """Creates timestamped directories for recordings."""
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
    """Initialize new video file writer with timestamp."""
    global video_writer, current_video_path, current_folder
    current_folder = get_output_path()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filename = f"record_{datetime.now().strftime('%H-%M-%S')}.mp4"
    current_video_path = os.path.join(current_folder, filename)

    video_writer = cv2.VideoWriter(current_video_path, fourcc, 10.0, (frame_width, frame_height))
    print(f"🎞️ Recording video to: {current_video_path}")


def save_detection_json(data):
    """Append detection data to JSON log."""
    global detections_data
    detections_data.append(data)
    log_path = os.path.join(current_folder, "detections.json")
    try:
        with open(log_path, "a") as f:
            json.dump(data, f)
            f.write("\n")
    except Exception as e:
        print(f"[Error] Failed to write detection JSON: {e}")


def log_detection(label, conf):
    """Emit detection data to socket and save to JSON."""
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
    """Delete recordings older than specified days."""
    base_dir = os.path.expanduser("~/YOLO_Recordings")
    if not os.path.exists(base_dir):
        return
    now = datetime.now()
    for folder in os.listdir(base_dir):
        try:
            folder_date = datetime.strptime(folder, "%Y-%m-%d")
            if (now - folder_date).days > days:
                full_path = os.path.join(base_dir, folder)
                print(f"🧹 Cleaning old folder: {full_path}")
                import shutil
                shutil.rmtree(full_path)
        except Exception:
            continue


def handle_exit(sig, frame):
    """Handle exit and safely close resources."""
    global cap, video_writer
    print("\n🛑 Program interrupted! Cleaning up...")
    try:
        if video_writer:
            video_writer.release()
            print("[INFO] Video file safely closed.")
        if cap and hasattr(cap, "isOpened") and cap.isOpened():
            cap.release()
            print("[INFO] Camera released.")
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
    sys.exit(0)


# -----------------------------
# 📸 Camera Capture
# -----------------------------
def camera_capture():
    global global_frame, video_writer, cap

    if PICAMERA_AVAILABLE:
        try:
            print("🎥 Starting Picamera2...")
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (frame_width, frame_height)})
            picam2.configure(config)
            picam2.start()
            time.sleep(2)

            init_video_writer()
            last_hour = datetime.now().hour

            while True:
                frame = picam2.capture_array()
                if frame is None:
                    continue

                now = datetime.now()
                if now.hour != last_hour:
                    video_writer.release()
                    init_video_writer()
                    last_hour = now.hour

                results = fire_model.predict(frame, verbose=False)
                annotated = frame.copy()

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

                video_writer.write(annotated)
                with frame_lock:
                    global_frame = annotated
                time.sleep(0.1)

        except Exception as e:
            print(f"❌ Picamera2 error: {e} — switching to OpenCV...")
            start_opencv_camera()

    else:
        start_opencv_camera()


def start_opencv_camera():
    global global_frame, video_writer, cap
    print("🎥 Starting OpenCV camera...")
    cap = cv2.VideoCapture(0)
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    if not cap.isOpened():
        print("❌ Error: Cannot open camera.")
        return

    init_video_writer()
    last_hour = datetime.now().hour

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = datetime.now()
        if now.hour != last_hour:
            video_writer.release()
            init_video_writer()
            last_hour = now.hour

        results = fire_model.predict(frame, verbose=False)
        annotated = frame.copy()

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

        video_writer.write(annotated)
        with frame_lock:
            global_frame = annotated
        time.sleep(0.1)


# -----------------------------
# 🌐 Flask Routes
# -----------------------------
def generate_frames():
    global global_frame
    while True:
        with frame_lock:
            if global_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', global_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/detections")
def get_detections():
    return jsonify(list(reversed(detections_data[-100:])))


# -----------------------------
# 🚀 Main Entry
# -----------------------------
if __name__ == "__main__":
    cleanup_old(7)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    cam_thread = Thread(target=camera_capture, daemon=True)
    cam_thread.start()

    print("✅ Flask app running locally at http://0.0.0.0:5000")
    try:
        print("🚀 Starting Cloudflare Tunnel...")
        try:
            # tunnel = open_tunnel(port=5000)
            print(f"🌍 Public Tunnel URL: ")
        except Exception as e:
            print(f"❌ Cloudflare Tunnel failed: {e}")

        socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        handle_exit(None, None)
    except Exception as e:
        print(f"Failed to start server: {e}")
        handle_exit(None, None)
