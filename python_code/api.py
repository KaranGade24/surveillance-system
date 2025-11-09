# live_stream_yolo.py
print("Starting imports...")
import cv2
from threading import Thread, Lock
from flask import Flask, Response
from flask_cors import CORS
from ultralytics import YOLO
print("completed")
print("Starting loading")

# Load YOLO models
fire_model = YOLO("fire_detector.pt")
weapon_model = YOLO("weapon_detector.pt")
print("YOLO models loaded successfully!")

# Flask app
app = Flask(__name__)
from flask_cors import CORS

# Global variables for camera frame sharing
frame_lock = Lock()
global_frame = None

# Threaded camera capture
def camera_capture():
    global global_frame
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Camera started, reading frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run YOLO detections
        fire_results = fire_model.predict(frame, verbose=False)
        weapon_results = weapon_model.predict(frame, verbose=False)

        annotated = frame.copy()

        # Draw fire boxes
        for fire_r in fire_results:
            for box in fire_r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = f"Fire {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw weapon boxes
        for weapon_r in weapon_results:
            for box in weapon_r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = f"Weapon {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Update the global frame
        with frame_lock:
            global_frame = annotated

        # Show locally
        cv2.imshow("Live YOLO Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Frame generator for Flask streaming
def generate_frames():
    global global_frame
    while True:
        with frame_lock:
            if global_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', global_frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask route for live streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Start camera thread and Flask app
if __name__ == "__main__":
    cam_thread = Thread(target=camera_capture, daemon=True)
    cam_thread.start()
    print("Flask app running at http://0.0.0.0:5000/video_feed")
    app.run(host="0.0.0.0", port=5000, debug=False)
