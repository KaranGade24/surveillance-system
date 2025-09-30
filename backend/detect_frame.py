import sys
from ultralytics import YOLO
import cv2
import numpy as np
import base64

# Load YOLO model once
model = YOLO("yolov8n.pt")

def process_frame(dataUrl):
    # Decode base64 frame
    header, encoded = dataUrl.split(",", 1)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run detection
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    # Encode annotated frame back to base64
    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_img = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_img}"

if __name__ == "__main__":
    frame_data = sys.argv[1]
    output = process_frame(frame_data)
    print(output)  # Node.js receives this as results[0]
