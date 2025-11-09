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
# from pycloudflared import open_tunnel

# Try importing Picamera2 (for Raspberry Pi)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
    print("Picamera2 found! Using Raspberry Pi camera.")
except ImportError:
    print("Picamera2 not found. Falling back to OpenCV camera.")
    PICAMERA_AVAILABLE = False

print("Imports completed.")
print("Loading YOLO model...")

# Load YOLOv8 fire detection model
fire_model = YOLO("fire_detector.pt")
print("YOLO model loaded successfully!")

# Flask app
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
# Helpers
# -----------------------------
def get_ip():
    """Get local IP address dynamically."""
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filename = f"record_{datetime.now().strftime('%H-%M-%S')}.mp4"
    current_video_path = os.path.join(current_folder, filename)

    # Ensure writer uses the same size as frame_width/frame_height
    video_writer = cv2.VideoWriter(current_video_path, fourcc, 10.0, (frame_width, frame_height))
    print(f"Recording video to: {current_video_path}")


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
                full_path = os.path.join(base_dir, folder)
                print(f"Cleaning old folder: {full_path}")
                import shutil
                shutil.rmtree(full_path)
        except Exception:
            continue


def handle_exit(sig, frame):
    global cap, video_writer
    print("\nProgram interrupted! Cleaning up...")
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
# Drawing utilities
# -----------------------------
def draw_detection_boxes(img_bgr, boxes_xyxy, confs, class_ids=None, conf_threshold=0.25):
    """
    Draw bounding boxes and labels on img_bgr (OpenCV BGR).
    boxes_xyxy: numpy array of shape (N,4) with x1,y1,x2,y2
    confs: numpy array of shape (N,)
    class_ids: optional numpy array of shape (N,)
    """
    for i, box in enumerate(boxes_xyxy):
        conf = float(confs[i])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        # clamp coords
        x1 = max(0, min(x1, img_bgr.shape[1] - 1))
        y1 = max(0, min(y1, img_bgr.shape[0] - 1))
        x2 = max(0, min(x2, img_bgr.shape[1] - 1))
        y2 = max(0, min(y2, img_bgr.shape[0] - 1))

        # Choose color (red for strong, yellow for low)
        color = (0, 0, 255) if conf >= 0.5 else (0, 255, 255)

        # Draw rectangle
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = "Fire"
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # background rectangle for text
        tx1, ty1 = x1, max(0, y1 - th - 6)
        tx2, ty2 = x1 + tw + 6, ty1 + th + 4
        cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(img_bgr, text, (x1 + 3, ty1 + th + -1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_bgr


# -----------------------------
# Camera capture
# -----------------------------
def camera_capture():
    global global_frame, video_writer, cap

    if PICAMERA_AVAILABLE:
        try:
            print("Starting Picamera2...")
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (frame_width, frame_height)})
            picam2.configure(config)
            picam2.start()
            time.sleep(2)

            init_video_writer()
            last_hour = datetime.now().hour

            while True:
                frame = picam2.capture_array()  # BGR or BGRA (we handle both)
                if frame is None:
                    continue

                # If BGRA -> convert to BGR
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Ensure frame size matches writer size: resize if necessary
                if (frame.shape[1], frame.shape[0]) != (frame_width, frame_height):
                    frame = cv2.resize(frame, (frame_width, frame_height))

                # Convert BGR -> RGB for model
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                now = datetime.now()
                if now.hour != last_hour:
                    if video_writer:
                        video_writer.release()
                    init_video_writer()
                    last_hour = now.hour

                # Run model on RGB image
                # using predict() returns a list of Results
                results = fire_model.predict(rgb, imgsz=(frame_width, frame_height), conf=0.25, verbose=False)

                annotated = frame.copy()  # BGR copy for drawing

                # Collect boxes & confidences properly and draw
                for r in results:
                    if not hasattr(r, "boxes") or len(r.boxes) == 0:
                        continue
                    # r.boxes.xyxy is a tensor (N,4); r.boxes.conf (N,)
                    try:
                        boxes = r.boxes.xyxy.cpu().numpy()  # (N,4)
                        confs = r.boxes.conf.cpu().numpy()
                    except Exception:
                        # fallback: convert via .data if needed
                        boxes = r.boxes.xyxy.numpy()
                        confs = r.boxes.conf.numpy()

                    # Draw boxes (clamped) on annotated frame
                    annotated = draw_detection_boxes(annotated, boxes, confs, conf_threshold=0.25)

                    # Log strong detections (conf >= 0.5)
                    for c_idx, conf_val in enumerate(confs):
                        if conf_val >= 0.5:
                            log_detection("Fire", float(conf_val))

                # Write & update global frame for streaming
                if video_writer:
                    # ensure frame written has same size as writer expects
                    out_frame = cv2.resize(annotated, (frame_width, frame_height))
                    video_writer.write(out_frame)

                with frame_lock:
                    global_frame = annotated

                # short sleep to yield CPU
                time.sleep(0.05)

        except Exception as e:
            print(f"Picamera2 error: {e} — switching to OpenCV...")
            start_opencv_camera()

    else:
        start_opencv_camera()


def start_opencv_camera():
    global global_frame, video_writer, cap
    print("Starting OpenCV camera...")
    cap = cv2.VideoCapture(0)
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    init_video_writer()
    last_hour = datetime.now().hour

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        if (frame.shape[1], frame.shape[0]) != (frame_width, frame_height):
            frame = cv2.resize(frame, (frame_width, frame_height))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = datetime.now()
        if now.hour != last_hour:
            if video_writer:
                video_writer.release()
            init_video_writer()
            last_hour = now.hour

        results = fire_model.predict(rgb, imgsz=(frame_width, frame_height), conf=0.25, verbose=False)

        annotated = frame.copy()

        for r in results:
            if not hasattr(r, "boxes") or len(r.boxes) == 0:
                continue
            try:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
            except Exception:
                boxes = r.boxes.xyxy.numpy()
                confs = r.boxes.conf.numpy()

            annotated = draw_detection_boxes(annotated, boxes, confs, conf_threshold=0.25)

            for c_idx, conf_val in enumerate(confs):
                if conf_val >= 0.5:
                    log_detection("Fire", float(conf_val))

        if video_writer:
            out_frame = cv2.resize(annotated, (frame_width, frame_height))
            video_writer.write(out_frame)

        with frame_lock:
            global_frame = annotated

        time.sleep(0.05)


# -----------------------------
# Flask routes
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
        time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/detections")
def get_detections():
    return jsonify(list(reversed(detections_data[-100:])))


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    cleanup_old(7)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    cam_thread = Thread(target=camera_capture, daemon=True)
    cam_thread.start()

    local_ip = get_ip()
    print(f"Flask app running locally at:")
    print(f"  http://127.0.0.1:5000")
    print(f"  http://{local_ip}:5000")
    print(f"Video Stream: http://{local_ip}:5000/video_feed")

    try:
        print("Starting (Cloudflare) tunnel...")  # kept placeholder
        try:
            # tunnel = open_tunnel(port=5000)
            print("Public Tunnel URL: ")
        except Exception as e:
            print(f"Cloudflare Tunnel failed: {e}")

        socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        handle_exit(None, None)
    except Exception as e:
        print(f"Failed to start server: {e}")
        handle_exit(None, None)
