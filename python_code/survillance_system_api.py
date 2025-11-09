# print("Starting imports...")
# import cv2
# import os
# import json
# import time
# from datetime import datetime
# from threading import Thread, Lock
# from flask import Flask, Response, jsonify,send_from_directory, request
# from flask_cors import CORS
# from ultralytics import YOLO
# import signal
# import sys
# from flask_socketio import SocketIO
# from cloudflare import start_cloudflare_background
# import mimetypes


# print("Imports completed.")
# print("Loading YOLO models...")

# # Load YOLO models
# fire_model = YOLO("fire_detector.pt")
# # weapon_model = YOLO("weapon_detector.pt")
# print("YOLO models loaded successfully!")

# # Flask app
# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Global variables
# frame_lock = Lock()
# global_frame = None
# detections_data = []   # In-memory detection cache (will also be written to JSON)
# video_writer = None
# current_video_path = None
# current_folder = None
# # frame_width, frame_height = 640, 480
# frame_width, frame_height = 320, 240

# cap = None


# # -----------------------------
# # 🗂️  Helper functions
# # -----------------------------
# def get_output_path():
#     """Create and return the correct output folder path (daily/hourly)."""
#     base_dir = "/home/pi/YOLO_Recordings"
#     now = datetime.now()

#     # Daily folder
#     date_folder = now.strftime("%Y-%m-%d")
#     date_path = os.path.join(base_dir, date_folder)
#     os.makedirs(date_path, exist_ok=True)

#     # Hourly folder (e.g., 13_00-14_00)
#     hour = int(now.strftime("%H"))
#     hour_folder = f"{hour:02d}_00-{(hour+1)%24:02d}_00"
#     hour_path = os.path.join(date_path, hour_folder)
#     os.makedirs(hour_path, exist_ok=True)

#     return hour_path


# def init_video_writer():
#     """Initialize a new video writer for the current hour."""
#     global video_writer, current_video_path, current_folder, last_frame_time

#     current_folder = get_output_path()
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     filename = f"record_{datetime.now().strftime('%H-%M-%S')}.mp4"
#     current_video_path = os.path.join(current_folder, filename)
    
    
#     # ✅ Use GStreamer pipeline for MP4 with moov atom at start
#     gst_pipeline = (
#         f"appsrc ! videoconvert ! x264enc tune=zerolatency "
#         f"bitrate=500 speed-preset=superfast ! "
#         f"mp4mux faststart=true ! filesink location={current_video_path}"
#     )
    
    
#     # video_writer = cv2.VideoWriter(current_video_path, fourcc, 5.0, (frame_width, frame_height))  # start with rough guess

#     video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 5.0, (frame_width, frame_height))

#     last_frame_time = time.time()
#     print(f"Recording video to: {current_video_path}")



# def save_detection_json(data):
#     """Append detection data to both memory and file."""
#     global detections_data
#     detections_data.append(data)
    

#     # Save to JSON file in the same hourly folder
#     log_path = os.path.join(current_folder, "detections.json")
#     with open(log_path, "a") as f:
#         json.dump(data, f)
#         f.write("\n")  # line-separated JSON objects


# def log_detection(label, conf):
#     """Create detection record."""
#     now = datetime.now()
#     current_video_start_epoch = datetime.now().timestamp()
#     data = {
#         "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
#         "epoch_time": now.timestamp(),
#         "object": label,
#         "confidence": round(conf, 2),
#         "video_file": current_video_path,
#         "video_start_epoch": current_video_start_epoch
#     }
#     # in log_detection():
#     socketio.emit("new_detection", data)
#     save_detection_json(data)


# def cleanup_old(days=7):
#     """Optional: remove recordings older than N days."""
#     base_dir = "/home/pi/YOLO_Recordings"
#     if not os.path.exists(base_dir):
#         return
#     now = datetime.now()
#     for folder in os.listdir(base_dir):
#         try:
#             folder_date = datetime.strptime(folder, "%Y-%m-%d")
#             if (now - folder_date).days > days:
#                 full_path = os.path.join(base_dir, folder)
#                 print(f"Cleaning old folder: {full_path}")
#                 import shutil
#                 shutil.rmtree(full_path)
#         except Exception:
#             continue


# def handle_exit(sig, frame):
#     global cap, video_writer
#     print("\n[INFO] Program interrupted! Cleaning up...")
#     try:
#         if video_writer:
#             video_writer.release()
#             print("[INFO] Video file safely closed.")
#         if cap and cap.isOpened():
#             cap.release()
#             print("[INFO] Camera released.")
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"[ERROR] Cleanup failed: {e}")
#     sys.exit(0)





# # -----------------------------
# # 📸  Camera thread
# # -----------------------------
# def camera_capture():
#     global global_frame, video_writer, current_video_path, cap
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     cap.set(3, frame_width)
#     cap.set(4, frame_height)

#     if not cap.isOpened():
#         print("❌ Error: Cannot open camera")
#         return

#     print("✅ Camera started, capturing frames...")
#     init_video_writer()
#     last_hour = datetime.now().hour

#     while True:
#         frame_start = time.time()
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         # Check if hour changed — start new video file
#         now = datetime.now()
#         if now.hour != last_hour:
#             video_writer.release()
#             init_video_writer()
#             last_hour = now.hour

#         # Run detections
#         fire_results = fire_model.predict(frame, verbose=False, stream=True)
#         # weapon_results = weapon_model.predict(frame, verbose=False, stream=True)
#         annotated = frame.copy()

#         # Draw and log Fire detections
#         for fire_r in fire_results:
#             for box in fire_r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 label = "Fire"
#                 cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#                 if conf > 0.5:  # log only strong detections
#                     log_detection(label, conf)

#         # Draw and log Weapon detections
#         # for weapon_r in weapon_results:
#         #     for box in weapon_r.boxes:
#         #         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         #         conf = float(box.conf[0])
#         #         label = "Weapon"
#         #         cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         #         cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
#         #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#         #         if conf > 0.5:
#         #             log_detection(label, conf)

#         # Write frame to video
#         video_writer.write(annotated)
#         # adjust FPS dynamically (optional)
#         elapsed = time.time() - frame_start
#         if elapsed > 0:
#             current_fps = 1.0 / elapsed
#         else:
#             current_fps = 5.0
            
#         print(f"FPS: {current_fps:.2f}")

#         # Update shared frame for live stream
#         with frame_lock:
#             global_frame = annotated

#         # Local preview (press 'q' to exit)
#         cv2.imshow("YOLO Live Detection", annotated)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     video_writer.release()
#     cv2.destroyAllWindows()


# # -----------------------------
# # 🌐  Flask routes
# # -----------------------------
# def generate_frames():
#     """Stream MJPEG frames to browser."""
#     global global_frame
#     while True:
#         with frame_lock:
#             if global_frame is None:
#                 continue
#             ret, buffer = cv2.imencode('.jpg', global_frame)
#             frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# @app.route("/video_feed")
# def video_feed():
#     """Live video stream."""
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route("/detections")
# def get_detections():
#     """Return all detections in memory (latest first)."""
#     return jsonify(list(reversed(detections_data[-100:])))




# # -----------------------------
# # 🎥 Serve video by timestamp
# # -----------------------------
# @app.route("/play_video")
# def play_video():
#     """
#     Frontend passes ?timestamp=<epoch_time>
#     Backend finds matching detection & returns the video file.
#     """
#     ts = request.args.get("timestamp")
#     if not ts:
#         return jsonify({"error": "timestamp required"}), 400

#     # Find detection by timestamp
#     target = next((d for d in detections_data if str(d["epoch_time"]) == ts), None)
#     if not target:
#         return jsonify({"error": "No detection found"}), 404

#     # Extract file info
#     path = target["video_file"]
#     if not os.path.exists(path):
#         return jsonify({"error": f"Video not found: {path}"}), 404

#     # Split directory and filename
#     directory, filename = os.path.split(path)
#     mime = mimetypes.guess_type(filename)[0] or "video/mp4"

#     try:
#         return send_from_directory(directory, filename, mimetype=mime, as_attachment=False)
#     except Exception as e:
#         print(f"❌ Error sending file: {e}")
#         return jsonify({"error": f"Unable to send file: {str(e)}"}), 500




# # @app.route("/recordings")
# # def list_recordings():
# #     base_dir = "/home/pi/YOLO_Recordings"
# #     tree = []
# #     if not os.path.exists(base_dir):
# #         return jsonify(tree)
# #     # Example structure: /YYYY-MM-DD/HH_00-HH_00/ -> files
# #     for date in sorted(os.listdir(base_dir), reverse=True):
# #         date_path = os.path.join(base_dir, date)
# #         if not os.path.isdir(date_path): continue
# #         hours = []
# #         for hour in sorted(os.listdir(date_path), reverse=True):
# #             hour_path = os.path.join(date_path, hour)
# #             if not os.path.isdir(hour_path): continue
# #             files = []
# #             for f in sorted(os.listdir(hour_path), reverse=True):
# #                 if f.lower().endswith(".mp4"):
# #                     full = os.path.join(hour_path, f)
# #                     size = os.path.getsize(full)
# #                     mtime = os.path.getmtime(full)
# #                     files.append({
# #                         "name": f,
# #                         "path": full,
# #                         "size": size,
# #                         "mtime": mtime
# #                     })
# #             hours.append({"hour": hour, "files": files})
# #         tree.append({"date": date, "hours": hours})
# #     return jsonify(tree)


# @app.route("/recordings")
# def list_recordings():
#     base_dir = "/home/pi/YOLO_Recordings"
#     tree = []
#     if not os.path.exists(base_dir):
#         return jsonify(tree)

#     for date in sorted(os.listdir(base_dir), reverse=True):
#         date_path = os.path.join(base_dir, date)
#         if not os.path.isdir(date_path):
#             continue

#         hours = []
#         for hour in sorted(os.listdir(date_path), reverse=True):
#             hour_path = os.path.join(date_path, hour)
#             if not os.path.isdir(hour_path):
#                 continue

#             files = []
#             for f in sorted(os.listdir(hour_path), reverse=True):
#                 if f.lower().endswith(".mp4"):
#                     full = os.path.join(hour_path, f)
#                     size = os.path.getsize(full)
#                     mtime = os.path.getmtime(full)

#                     # ✅ URL-safe relative path
#                     rel_path = os.path.relpath(full, base_dir)
#                     rel_path = rel_path.replace("\\", "/")

#                     files.append({
#                         "name": f,
#                         "path": rel_path,
#                         "size": size,
#                         "mtime": mtime,
#                         # Direct playable URL
#                         "url": f"/recording/{rel_path}"
#                     })
#             hours.append({"hour": hour, "files": files})

#         tree.append({"date": date, "hours": hours})

#     return jsonify(tree)


# from flask import Response, request, abort, send_file
# import os

# @app.route("/recording/<path:filename>")
# def static_recordings(filename):
#     """Stream any video file (supports byte-range requests)."""
#     base_dir = "/home/pi/YOLO_Recordings"
#     full_path = os.path.join(base_dir, filename)

#     if not os.path.exists(full_path):
#         return jsonify({"error": "File not found"}), 404

#     range_header = request.headers.get('Range', None)
#     if not range_header:
#         # No Range header — send entire file
#         return send_file(full_path, mimetype="video/mp4")

#     size = os.path.getsize(full_path)
#     byte1, byte2 = 0, None

#     # Parse range header: e.g. "Range: bytes=1000-"
#     m = range_header.replace('bytes=', '').split('-')
#     if len(m) == 2:
#         if m[0]:
#             byte1 = int(m[0])
#         if m[1]:
#             byte2 = int(m[1])

#     length = size - byte1 if byte2 is None else byte2 - byte1 + 1
#     data = None
#     with open(full_path, 'rb') as f:
#         f.seek(byte1)
#         data = f.read(length)

#     resp = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
#     resp.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
#     resp.headers.add('Accept-Ranges', 'bytes')
#     return resp





# # -----------------------------
# # 🚀  Main entry
# # -----------------------------
# if __name__ == "__main__":
#     cleanup_old(7)  # Optional: remove >7-day-old videos

#     # Register Ctrl+C and system exit handlers
#     signal.signal(signal.SIGINT, handle_exit)
#     signal.signal(signal.SIGTERM, handle_exit)

#     # Start camera thread
#     cam_thread = Thread(target=camera_capture, daemon=True)
#     cam_thread.start()

#     print("✅ Flask app running at http://0.0.0.0:5000")
#     try:
#         start_cloudflare_background(port=5000, autorest=True)
#         app.run(host="0.0.0.0", port=5000, debug=False)
#     except KeyboardInterrupt:
#         handle_exit(None, None)




    


print("Starting imports...")
import cv2
import os
import json
import time
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS
from ultralytics import YOLO
import signal
import sys
from flask_socketio import SocketIO
from cloudflare import start_cloudflare_background
import mimetypes

print("Imports completed.")
print("Loading YOLO models...")

# Load YOLO model
fire_model = YOLO("fire_detector.pt")
print("YOLO models loaded successfully!")

# Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
frame_lock = Lock()
global_frame = None
detections_data = []  # In-memory detection cache (will also be written to JSON)
video_writer = None
current_video_path = None
current_folder = None
frame_width, frame_height = 320, 240 # Keep 320x240 for better RPi performance

cap = None


# -----------------------------
# 🗂️  Helper functions
# -----------------------------
def get_output_path():
    """Create and return the correct output folder path (daily/hourly)."""
    base_dir = "/home/pi/YOLO_Recordings"
    now = datetime.now()

    # Daily folder
    date_folder = now.strftime("%Y-%m-%d")
    date_path = os.path.join(base_dir, date_folder)
    os.makedirs(date_path, exist_ok=True)

    # Hourly folder (e.g., 13_00-14_00)
    hour = int(now.strftime("%H"))
    hour_folder = f"{hour:02d}_00-{(hour+1)%24:02d}_00"
    hour_path = os.path.join(date_path, hour_folder)
    os.makedirs(hour_path, exist_ok=True)

    return hour_path


def init_video_writer():
    """Initialize a new video writer for the current hour."""
    global video_writer, current_video_path, current_folder, last_frame_time

    current_folder = get_output_path()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filename = f"record_{datetime.now().strftime('%H-%M-%S')}.mp4"
    current_video_path = os.path.join(current_folder, filename)
    
    
    # ✅ Use GStreamer pipeline for MP4 with moov atom at start
    # This is crucial for web playback of in-progress files
    gst_pipeline = (
        f"appsrc ! videoconvert ! x264enc tune=zerolatency "
        f"bitrate=500 speed-preset=superfast ! "
        f"mp4mux faststart=true ! filesink location={current_video_path}"
    )
    
    # Use 10.0 FPS, adjust if your camera/processing is faster/slower
    video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 10.0, (frame_width, frame_height))

    last_frame_time = time.time()
    print(f"Recording video to: {current_video_path}")



def save_detection_json(data):
    """Append detection data to both memory and file."""
    global detections_data
    detections_data.append(data)
    

    # Save to JSON file in the same hourly folder
    log_path = os.path.join(current_folder, "detections.json")
    try:
        with open(log_path, "a") as f:
            json.dump(data, f)
            f.write("\n")  # line-separated JSON objects
    except Exception as e:
        print(f"[Error] Failed to write detection JSON: {e}")


def log_detection(label, conf):
    """Create detection record."""
    now = datetime.now()
    # Note: Using now.timestamp() for both, as video_start is tricky to pin
    current_time_epoch = now.timestamp()
    data = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch_time": current_time_epoch,
        "object": label,
        "confidence": round(conf, 2),
        "video_file": current_video_path,
        "video_start_epoch": current_time_epoch # Simple reference
    }
    # Emit to websocket
    socketio.emit("new_detection", data)
    save_detection_json(data)


def cleanup_old(days=7):
    """Optional: remove recordings older than N days."""
    base_dir = "/home/pi/YOLO_Recordings"
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
    print("\n[INFO] Program interrupted! Cleaning up...")
    try:
        if video_writer:
            video_writer.release()
            print("[INFO] Video file safely closed.")
        if cap and cap.isOpened():
            cap.release()
            print("[INFO] Camera released.")
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
    sys.exit(0)





# -----------------------------
# 📸  Camera thread
# -----------------------------
def camera_capture():
    global global_frame, video_writer, current_video_path, cap
    
    # --- RASPBERRY PI CAMERA INITIALIZATION ---
    # Try GStreamer pipeline for Pi Camera (libcamerasrc) first
    # This is the modern, high-performance method
    cam_pipeline = (
        f"libcamerasrc ! "
        f"video/x-raw,width={frame_width},height={frame_height},framerate=10/1 ! "
        f"videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(cam_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Error: Cannot open Pi Camera via GStreamer.")
        print("ℹ️ Trying fallback to /dev/video0 (USB Webcam)...")
        # Fallback for USB webcams
        cap = cv2.VideoCapture(0)
        cap.set(3, frame_width)
        cap.set(4, frame_height)

    if not cap.isOpened():
        print("❌ Error: Cannot open any camera source. Exiting thread.")
        return
    # --- END OF CAMERA INIT ---

    print("✅ Camera started, capturing frames...")
    init_video_writer()
    last_hour = datetime.now().hour

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame.")
            time.sleep(0.1) # Avoid busy-looping on error
            continue

        # Check if hour changed — start new video file
        now = datetime.now()
        if now.hour != last_hour:
            print("[INFO] Hour changed, starting new video file.")
            video_writer.release()
            init_video_writer()
            last_hour = now.hour

        # Run detections
        # Note: stream=True is generally for files, use default for single images
        fire_results = fire_model.predict(frame, verbose=False)
        annotated = frame.copy()

        # Draw and log Fire detections
        for fire_r in fire_results:
            for box in fire_r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = "Fire"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if conf > 0.5:  # log only strong detections
                    log_detection(label, conf)

        # Write frame to video
        if video_writer:
            video_writer.write(annotated)
        
        # Update shared frame for live stream
        with frame_lock:
            global_frame = annotated

        # --- HEADLESS OPERATION ---
        # We are on a server, so we do NOT show a local preview window.
        # cv2.imshow("YOLO Live Detection", annotated) <-- REMOVED
        # if cv2.waitKey(1) & 0xFF == ord('q'):     <-- REMOVED
        #     break
        # --- END HEADLESS OPERATION ---

        # Optional: Print FPS for debugging
        elapsed = time.time() - frame_start
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            print(f"FPS: {current_fps:.2f}")


    cap.release()
    if video_writer:
        video_writer.release()


# -----------------------------
# 🌐  Flask routes
# -----------------------------
def generate_frames():
    """Stream MJPEG frames to browser."""
    global global_frame
    while True:
        with frame_lock:
            if global_frame is None:
                # Send a placeholder if no frame is ready
                placeholder = cv2.imencode('.jpg', 
                    cv2.putText(
                        cv2.UMat(
                            cv2.Mat.zeros(frame_height, frame_width, cv2.CV_8UC3)
                        ).get(),
                        'No Signal', 
                        (20, frame_height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                    )
                )[1].tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                time.sleep(1) # Wait 1s if no signal
                continue
            
            # Encode the valid frame
            ret, buffer = cv2.imencode('.jpg', global_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1) # ~10 FPS stream


@app.route("/video_feed")
def video_feed():
    """Live video stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/detections")
def get_detections():
    """Return all detections in memory (latest first)."""
    return jsonify(list(reversed(detections_data[-100:])))




# -----------------------------
# 🎥 Serve video by timestamp
# -----------------------------
@app.route("/play_video")
def play_video():
    """
    Frontend passes ?timestamp=<epoch_time>
    Backend finds matching detection & returns the video file.
    """
    ts = request.args.get("timestamp")
    if not ts:
        return jsonify({"error": "timestamp required"}), 400

    # Find detection by timestamp
    target = next((d for d in detections_data if str(d["epoch_time"]) == ts), None)
    if not target:
        return jsonify({"error": "No detection found"}), 404

    # Extract file info
    path = target["video_file"]
    if not os.path.exists(path):
        return jsonify({"error": f"Video not found: {path}"}), 404

    # Split directory and filename
    directory, filename = os.path.split(path)
    mime = mimetypes.guess_type(filename)[0] or "video/mp4"

    try:
        return send_from_directory(directory, filename, mimetype=mime, as_attachment=False)
    except Exception as e:
        print(f"❌ Error sending file: {e}")
        return jsonify({"error": f"Unable to send file: {str(e)}"}), 500



@app.route("/recordings")
def list_recordings():
    base_dir = "/home/pi/YOLO_Recordings"
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
                    try:
                        size = os.path.getsize(full)
                        mtime = os.path.getmtime(full)
                    except OSError:
                        continue # Skip file if it's gone or unreadable

                    # ✅ URL-safe relative path
                    rel_path = os.path.relpath(full, base_dir)
                    rel_path = rel_path.replace("\\", "/")

                    files.append({
                        "name": f,
                        "path": rel_path,
                        "size": size,
                        "mtime": mtime,
                        # Direct playable URL
                        "url": f"/recording/{rel_path}"
                    })
            if files: # Only add hour if it has files
                hours.append({"hour": hour, "files": files})
        
        if hours: # Only add date if it has hours
            tree.append({"date": date, "hours": hours})

    return jsonify(tree)


@app.route("/recording/<path:filename>")
def static_recordings(filename):
    """Stream any video file (supports byte-range requests)."""
    base_dir = "/home/pi/YOLO_Recordings"
    full_path = os.path.join(base_dir, filename)

    if not os.path.exists(full_path):
        return jsonify({"error": "File not found"}), 404

    range_header = request.headers.get('Range', None)
    if not range_header:
        # No Range header — send entire file
        return send_file(full_path, mimetype="video/mp4")

    size = os.path.getsize(full_path)
    byte1, byte2 = 0, None

    # Parse range header: e.g. "Range: bytes=1000-"
    try:
        m = range_header.replace('bytes=', '').split('-')
        if len(m) >= 1 and m[0]:
            byte1 = int(m[0])
        if len(m) >= 2 and m[1]:
            byte2 = int(m[1])
    except ValueError:
        return jsonify({"error": "Invalid Range header"}), 400

    if byte2 is None:
        byte2 = size - 1
        
    length = byte2 - byte1 + 1
    if length < 0 or byte1 >= size:
        return jsonify({"error": "Invalid range"}), 416 # Range Not Satisfiable

    data = None
    try:
        with open(full_path, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)
    except IOError:
        return jsonify({"error": "Could not read file"}), 500

    resp = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
    resp.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    resp.headers.add('Accept-Ranges', 'bytes')
    resp.headers.add('Content-Length', str(length))
    return resp





# -----------------------------
# 🚀  Main entry
# -----------------------------
if __name__ == "__main__":
    cleanup_old(7)  # Optional: remove >7-day-old videos

    # Register Ctrl+C and system exit handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Start camera thread
    cam_thread = Thread(target=camera_capture, daemon=True)
    cam_thread.start()

    print("✅ Flask app running at http://0.0.0.0:5000")
    try:
        # Start Cloudflare tunnel
        # Make sure you have `pip install flask-cloudflared`
        start_cloudflare_background(port=5000, autorest=True)
        
        # Use socketio.run for proper websocket support
        socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        handle_exit(None, None)
    except Exception as e:
        print(f"Failed to start server: {e}")
        handle_exit(None, None)