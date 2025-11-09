from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from threading import Thread
from datetime import datetime
import os
from time import sleep

app = Flask(__name__)

# ===============================
# Camera Setup
# ===============================
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (1280, 720)})
picam2.configure(camera_config)
encoder = H264Encoder()

# ===============================
# Folder Setup
# ===============================
base_dir = "/home/raspberrypi/YOLO_Recordings"
os.makedirs(base_dir, exist_ok=True)

# ===============================
# Start Camera
# ===============================
picam2.start()
sleep(2)

# ===============================
# Global Flags
# ===============================
recording = False
recording_thread = None

# ===============================
# Video Recording Function
# ===============================
def start_recording():
    global recording
    recording = True
    folder = os.path.join(base_dir, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, datetime.now().strftime("record_%H-%M-%S.mp4"))
    print(f"Recording video to: {filename}")

    output = FfmpegOutput(filename)
    picam2.start_recording(encoder, output)

    while recording:
        sleep(1)

    picam2.stop_recording()
    print("Recording stopped.")

# ===============================
# Stream Generator
# ===============================
def generate_frames():
    while True:
        frame = picam2.capture_array()
        if frame is None:
            continue
        from cv2 import imencode, cvtColor, COLOR_BGR2RGB
        frame = cvtColor(frame, COLOR_BGR2RGB)
        _, buffer = imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ===============================
# Flask Routes
# ===============================
@app.route('/')
def index():
    html = """
    <html>
        <head><title>Raspberry Pi Camera Stream</title></head>
        <body style="background-color:#111; color:white; text-align:center;">
            <h2>📹 Raspberry Pi Live Stream</h2>
            <img src="/video_feed" width="720" height="480" /><br><br>
            <a href="/start_recording">Start Recording</a> | 
            <a href="/stop_recording">Stop Recording</a> | 
            <a href="/capture_image">Capture Image</a>
        </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image')
def capture_image():
    filename = os.path.join(base_dir, f"image_{datetime.now().strftime('%H-%M-%S')}.jpg")
    picam2.capture_file(filename)
    return f"✅ Image captured and saved as {filename}"

@app.route('/start_recording')
def start_record():
    global recording_thread
    if not recording_thread or not recording_thread.is_alive():
        recording_thread = Thread(target=start_recording)
        recording_thread.start()
        return "🎥 Recording started."
    return "Already recording."

@app.route('/stop_recording')
def stop_record():
    global recording
    recording = False
    return "🛑 Recording stopped."

# ===============================
# Main
# ===============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
